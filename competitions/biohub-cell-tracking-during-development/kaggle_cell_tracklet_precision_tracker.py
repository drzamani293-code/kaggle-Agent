"""
Biohub - Cell Tracking During Development
Milestone 6C: tracklet-level precision tracker.

The conservative tracker (conservative_tracker.py) ran on real Kaggle data
and improved edge_jaccard from 0.00045 to 0.002215 (a 5x gain) by filtering
nodes and edges BEFORE linking. But precision stayed catastrophic
(precision=0.002276, edge_FP=5700 vs edge_TP=13): filtering individual
nodes/edges in isolation still leaves a huge number of individually-
plausible-looking but globally-incoherent short/random edges.

This module moves filtering up one level of abstraction: instead of
judging nodes and edges one at a time, it assembles full temporal
tracklets (connected chains of frame-to-frame links) and judges each
tracklet as a whole. A short, jittery, low-scoring 2-node "tracklet" that
happened to pass the per-edge filters is exactly the kind of thing a
tracklet-level view can catch and reject, in a way a single-edge test
cannot: length, smoothness, and displacement consistency are only
meaningful once you can see more than one step.

Two node-keeping policies are evaluated side by side to answer a specific
question - does keeping only tracklet-participating nodes (policy A) cost
more recall than it's worth, or is dropping the untracked-but-plausible
detections (policy B keeps them) actually the bigger source of noise:

  A. keep_tracklet_nodes_only - both nodes AND edges are restricted to
     kept tracklets.
  B. keep_all_filtered_nodes - nodes stay exactly as conservative_tracker's
     best node filter produced them (unchanged); only edges are
     restricted to kept tracklets.

Detection, node filtering, edge filtering, and tracklet construction are
all fixed/cached once per sample (reusing conservative_tracker.py's best
found config as the baseline) - the tracklet-level filter sweep only
re-filters an already-computed tracklet feature table, and for policy B,
GT node-matching itself is invariant across the WHOLE sweep (the node set
never changes), so it is computed exactly once per sample and reused.

No ML yet - classical detection + conservative filtering + tracklet-level
precision filtering.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

try:
    import blosc2
except ImportError:
    blosc2 = None

try:
    import zstandard
except ImportError:
    zstandard = None

# zarr v3 `data_type` -> numpy dtype code (byte order is applied separately,
# based on the "bytes" codec's "endian" configuration).
DTYPE_CODES = {
    "bool": "b1",
    "int8": "i1", "int16": "i2", "int32": "i4", "int64": "i8",
    "uint8": "u1", "uint16": "u2", "uint32": "u4", "uint64": "u8",
    "float16": "f2", "float32": "f4", "float64": "f8",
}

NODE_PROP_NAMES = ("t", "z", "y", "x")

# Physical voxel scale for this competition's volumes (µm/voxel).
VOXEL_SIZE_UM = {"z": 1.625, "y": 0.40625, "x": 0.40625}
_VOXEL_SCALE = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])


# --------------------------------------------------------------------------- #
# 1. zarr.json metadata parsing (manual offline reader, no zarr/numcodecs)
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_array_metadata(array_dir: Path) -> dict:
    """Parse a zarr v3 array's `zarr.json` into shape/dtype/chunking/codecs."""
    meta_path = array_dir / "zarr.json"
    doc = read_json(meta_path)

    if doc.get("zarr_format") != 3:
        raise ValueError(f"{meta_path}: only zarr v3 is supported, got zarr_format={doc.get('zarr_format')!r}")
    if doc.get("node_type") != "array":
        raise ValueError(f"{meta_path}: not a zarr array (node_type={doc.get('node_type')!r})")

    shape = tuple(int(s) for s in doc["shape"])

    data_type = doc["data_type"]
    if isinstance(data_type, dict):  # some writers nest it as {"name": "uint16"}
        data_type = data_type.get("name")
    code = DTYPE_CODES.get(data_type)
    if code is None:
        raise ValueError(f"{meta_path}: unsupported data_type {data_type!r}")

    codecs = doc.get("codecs", [])
    endian = "little"
    for codec in codecs:
        if codec.get("name") in ("bytes", "endian"):
            endian = codec.get("configuration", {}).get("endian", "little")
            break
    dtype = np.dtype("bool") if code == "b1" else np.dtype(("<" if endian == "little" else ">") + code)

    chunk_grid = doc.get("chunk_grid", {})
    if chunk_grid.get("name", "regular") != "regular":
        raise NotImplementedError(f"{meta_path}: unsupported chunk_grid {chunk_grid.get('name')!r}")
    chunk_shape = tuple(int(c) for c in chunk_grid.get("configuration", {}).get("chunk_shape", shape))

    chunk_key_encoding = doc.get(
        "chunk_key_encoding", {"name": "default", "configuration": {"separator": "/"}}
    )
    fill_value = doc.get("fill_value", 0) or 0

    return {
        "path": array_dir,
        "shape": shape,
        "dtype": dtype,
        "chunk_shape": chunk_shape,
        "codecs": codecs,
        "chunk_key_encoding": chunk_key_encoding,
        "fill_value": fill_value,
    }


# --------------------------------------------------------------------------- #
# 2. Chunk key construction + codec decoding
# --------------------------------------------------------------------------- #
def build_chunk_path(array_dir: Path, index: Sequence[int], chunk_key_encoding: dict) -> Path:
    """Build the on-disk chunk file path for a given chunk index tuple."""
    name = chunk_key_encoding.get("name", "default")
    separator = chunk_key_encoding.get("configuration", {}).get("separator", "/")
    index_strs = [str(i) for i in index]

    if name == "v2":
        key = separator.join(index_strs) if index_strs else "0"
    else:
        key = separator.join(["c", *index_strs]) if index_strs else "c"

    if separator == "/":
        return array_dir.joinpath(*key.split("/"))
    return array_dir / key


def decode_bytes_codecs(raw: bytes, codecs: list[dict], expected_nbytes: int) -> bytes:
    """Reverse the bytes->bytes compressor chain, then hand back raw array bytes.

    `codecs[0]` is the array->bytes codec (must be "bytes"/"endian" - a plain
    passthrough once endianness is known); `codecs[1:]` are bytes->bytes
    compressors applied in encode order, so they're undone in reverse order.
    """
    if not codecs:
        return raw

    array_bytes_codec = codecs[0]
    if array_bytes_codec.get("name") not in ("bytes", "endian"):
        raise NotImplementedError(
            f"unsupported array->bytes codec: {array_bytes_codec.get('name')!r} "
            "(only the plain 'bytes' codec is supported - no sharding)"
        )

    data = raw
    for codec in reversed(codecs[1:]):
        codec_name = codec.get("name")
        if codec_name == "blosc":
            if blosc2 is None:
                raise RuntimeError("chunk uses the 'blosc' codec but the blosc2 package is not installed")
            data = blosc2.decompress(data)
        elif codec_name == "zstd":
            if zstandard is None:
                raise RuntimeError("chunk uses the 'zstd' codec but the zstandard package is not installed")
            data = zstandard.ZstdDecompressor().decompress(data, max_output_size=expected_nbytes)
        else:
            raise NotImplementedError(f"unsupported bytes->bytes codec: {codec_name!r}")

    return data


def decode_chunk_bytes(raw: bytes, codecs: list[dict], dtype: np.dtype, chunk_shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(chunk_shape)) if chunk_shape else 1
    expected_nbytes = count * dtype.itemsize
    decoded = decode_bytes_codecs(raw, codecs, expected_nbytes)
    arr = np.frombuffer(decoded, dtype=dtype, count=count)
    return arr.reshape(chunk_shape)


def read_chunk_at(meta: dict, index: Sequence[int]) -> np.ndarray:
    """Read+decode a single chunk. Missing chunk files are sparse and decode
    to an array filled with the array's fill_value (never raises for that case).
    """
    chunk_path = build_chunk_path(meta["path"], index, meta["chunk_key_encoding"])
    chunk_shape = meta["chunk_shape"]
    if not chunk_path.exists():
        return np.full(chunk_shape, meta["fill_value"], dtype=meta["dtype"])
    raw = chunk_path.read_bytes()
    return decode_chunk_bytes(raw, meta["codecs"], meta["dtype"], chunk_shape)


def read_full_zarr_array(array_dir: Path) -> np.ndarray:
    """Read an entire zarr v3 array (stitching together every chunk)."""
    meta = parse_array_metadata(array_dir)
    shape, chunk_shape = meta["shape"], meta["chunk_shape"]

    if not shape:  # 0-d array
        return read_chunk_at(meta, ())

    n_chunks = [math.ceil(shape[d] / chunk_shape[d]) if shape[d] else 0 for d in range(len(shape))]
    full = np.full(shape, meta["fill_value"], dtype=meta["dtype"])
    for index in itertools.product(*(range(n) for n in n_chunks)):
        chunk = read_chunk_at(meta, index)
        starts = [index[d] * chunk_shape[d] for d in range(len(shape))]
        ends = [min(starts[d] + chunk_shape[d], shape[d]) for d in range(len(shape))]
        dst = tuple(slice(s, e) for s, e in zip(starts, ends))
        src = tuple(slice(0, e - s) for s, e in zip(starts, ends))
        full[dst] = chunk[src]
    return full


def read_image_timepoint(sample_zarr_dir: Path, t: int, level: str = "0") -> np.ndarray:
    """Read one timepoint of an OME-Zarr-style (T, Z, Y, X) image volume,
    fetching only the single chunk that covers timepoint `t`.
    """
    array_dir = sample_zarr_dir / level
    meta = parse_array_metadata(array_dir)
    chunk_shape = meta["chunk_shape"]
    if len(chunk_shape) != 4:
        raise ValueError(f"{array_dir}: expected a 4-D (T,Z,Y,X) array, got chunk_shape={chunk_shape}")

    t_chunk_size = chunk_shape[0]
    chunk_index = (t // t_chunk_size, 0, 0, 0)
    chunk = read_chunk_at(meta, chunk_index)

    local_t = t % t_chunk_size
    z, y, x = meta["shape"][1], meta["shape"][2], meta["shape"][3]
    return np.ascontiguousarray(chunk[local_t, :z, :y, :x])


def get_sample_timepoint_count(sample_zarr_dir: Path, level: str = "0") -> int:
    """Number of timepoints (T) available for a sample - read from metadata
    rather than assumed, so hidden test datasets with a different T are
    handled correctly instead of silently truncated/overrun.
    """
    meta = parse_array_metadata(sample_zarr_dir / level)
    return int(meta["shape"][0])


# --------------------------------------------------------------------------- #
# 3. GEFF (graph exchange file format) reader
# --------------------------------------------------------------------------- #
def _looks_like_geff_root(path: Path) -> bool:
    return (path / "nodes" / "ids" / "zarr.json").exists() and (path / "edges" / "ids" / "zarr.json").exists()


def find_geff_root(sample_zarr_dir: Path) -> Path | None:
    """Locate the GEFF store paired with a sample's image `.zarr` store."""
    candidates = [
        sample_zarr_dir.with_suffix(".geff"),
        sample_zarr_dir / "geff",
        sample_zarr_dir / "tracks",
        sample_zarr_dir / "tracks.geff",
    ]
    for candidate in candidates:
        if _looks_like_geff_root(candidate):
            return candidate

    search_root = sample_zarr_dir.parent
    try:
        for nodes_ids_dir in search_root.rglob("nodes/ids"):
            candidate = nodes_ids_dir.parent.parent
            if _looks_like_geff_root(candidate):
                return candidate
    except OSError:
        pass
    return None


def read_geff(sample_zarr_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the GEFF node/edge graph paired with a sample's image store."""
    node_columns = ["node_id", "t", "z", "y", "x"]
    edge_columns = ["source_id", "target_id"]

    geff_root = find_geff_root(sample_zarr_dir)
    if geff_root is None:
        print(f"[warn] no GEFF store found for {sample_zarr_dir}; returning empty node/edge tables")
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)

    node_ids = read_full_zarr_array(geff_root / "nodes" / "ids")
    props = {
        prop: read_full_zarr_array(geff_root / "nodes" / "props" / prop / "values")
        for prop in NODE_PROP_NAMES
    }
    nodes_df = pd.DataFrame({"node_id": node_ids, **props})[node_columns]

    edges_ids = read_full_zarr_array(geff_root / "edges" / "ids")
    edges_df = pd.DataFrame({"source_id": edges_ids[:, 0], "target_id": edges_ids[:, 1]})[edge_columns]

    return nodes_df, edges_df


# --------------------------------------------------------------------------- #
# 4. Train/test-sample discovery
# --------------------------------------------------------------------------- #
# The real Kaggle mount for this competition's data.
KAGGLE_COMPETITION_INPUT_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"


def get_input_roots() -> list[Path]:
    """Auto-detect candidate input roots.

    Priority order:
    1. ``KAGGLE_INPUT_DIR`` env var - explicit override, e.g. for local testing.
    2. ``KAGGLE_COMPETITION_INPUT_DIR`` - the real Kaggle competition mount.
    3. ``/kaggle/input`` - a broader fallback covering other Kaggle mount layouts.
    4. ``./input`` - a relative fallback for running outside Kaggle.
    """
    candidates = []
    env_root = os.environ.get("KAGGLE_INPUT_DIR")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(KAGGLE_COMPETITION_INPUT_DIR))
    candidates.append(Path("/kaggle/input"))
    candidates.append(Path("input"))

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            roots.append(candidate)
    return roots


def find_zarr_dirs_for_split(split: str, roots: Sequence[Path] | None = None) -> list[Path]:
    """Recursively locate every *.zarr folder belonging to the given split.
    No sample names are hardcoded - unlabeled/hidden datasets are picked up
    the same way as the ones seen during development.
    """
    if roots is None:
        roots = get_input_roots()

    all_zarr_dirs: list[Path] = []
    for root in roots:
        try:
            if not root.exists():
                continue
            for path in root.rglob("*.zarr"):
                if path.is_dir():
                    all_zarr_dirs.append(path)
        except OSError as exc:
            print(f"[warn] could not scan {root}: {exc!r}")

    if not all_zarr_dirs:
        return []

    split_dirs = [p for p in all_zarr_dirs if split.lower() in {part.lower() for part in p.parts}]
    if split_dirs:
        return sorted(set(split_dirs))

    print(
        f"[warn] no .zarr folder had a '{split}' path component; "
        "treating all discovered .zarr folders as this split's data."
    )
    return sorted(set(all_zarr_dirs))


def find_train_zarr_dirs(roots: Sequence[Path] | None = None) -> list[Path]:
    return find_zarr_dirs_for_split("train", roots)


def find_test_zarr_dirs(roots: Sequence[Path] | None = None) -> list[Path]:
    return find_zarr_dirs_for_split("test", roots)


def find_sample_submission_csv(roots: Sequence[Path] | None = None) -> Path | None:
    """Locate a `sample_submission*.csv` file under the input roots, if any -
    used to sanity-check our own column schema against the competition's,
    without hardcoding any path.
    """
    if roots is None:
        roots = get_input_roots()
    for root in roots:
        try:
            if not root.exists():
                continue
            for path in root.rglob("*sample_submission*.csv"):
                if path.is_file():
                    return path
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- #
# 5. Shared primitives (normalize, NMS, physical distance)
# --------------------------------------------------------------------------- #
FIXED_GAUSSIAN_SIGMA_Z = 1.0
FIXED_GAUSSIAN_SIGMA_YX = 1.5
MAX_RAW_CANDIDATES_BEFORE_NMS = 30000
RAW_POOL_SAFETY_CAP_PER_TIMEPOINT = 2000


def normalize_intensity(image: np.ndarray, percentile_low: float, percentile_high: float) -> np.ndarray:
    """Percentile-clip + rescale an image to [0, 1] float32."""
    lo, hi = np.percentile(image, [percentile_low, percentile_high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def _nms_by_physical_distance(candidates_df: pd.DataFrame, min_distance_um: float) -> pd.DataFrame:
    """Greedy non-maximum suppression via KD-tree: candidates must already
    be sorted by score descending. Uses `cKDTree.query_pairs` (one bulk C
    call) rather than a per-point loop - 3-8x faster, identical output.
    """
    if len(candidates_df) <= 1:
        return candidates_df

    scaled_coords = candidates_df[["z", "y", "x"]].to_numpy(dtype=float) * _VOXEL_SCALE
    tree = cKDTree(scaled_coords)
    n = len(candidates_df)
    keep_mask = np.ones(n, dtype=bool)

    pairs = tree.query_pairs(r=min_distance_um, output_type="ndarray")
    if len(pairs):
        order = np.argsort(pairs[:, 0], kind="stable")
        pairs = pairs[order]
        starts = np.searchsorted(pairs[:, 0], np.arange(n))
        ends = np.searchsorted(pairs[:, 0], np.arange(n), side="right")
        for i in range(n):
            if not keep_mask[i]:
                continue
            keep_mask[pairs[starts[i]:ends[i], 1]] = False

    return candidates_df[keep_mask]


def _pairwise_physical_distance_um(coords1_zyx: np.ndarray, coords2_zyx: np.ndarray) -> np.ndarray:
    """Vectorized (N,M) physical-distance cost matrix between two (z,y,x) coordinate arrays."""
    diff = (coords1_zyx[:, None, :] - coords2_zyx[None, :, :]) * _VOXEL_SCALE
    return np.sqrt((diff ** 2).sum(axis=-1))


def physical_distance_um(p1: Sequence[float], p2: Sequence[float]) -> float:
    diff = (np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)) * _VOXEL_SCALE
    return float(np.sqrt((diff ** 2).sum()))


# --------------------------------------------------------------------------- #
# 6. Detector: D_boundary_aware_3d_local_max ONLY (unchanged)
# --------------------------------------------------------------------------- #
def method_d_boundary_aware_3d_local_max(
    image_zyx: np.ndarray, percentile_low: float, percentile_high: float, min_threshold: float, pad: int = 4,
) -> pd.DataFrame:
    """Boundary-aware 3D local maxima (see boundary_detector.py for the full
    rationale/unit tests): manual reflect-padding + mode='nearest' smoothing
    + a direct maximum_filter call, independent of skimage internals.
    """
    normalized = normalize_intensity(image_zyx, percentile_low, percentile_high)
    padded = np.pad(normalized, pad_width=pad, mode="reflect")
    smoothed_padded = gaussian_filter(
        padded, sigma=(FIXED_GAUSSIAN_SIGMA_Z, FIXED_GAUSSIAN_SIGMA_YX, FIXED_GAUSSIAN_SIGMA_YX), mode="nearest"
    )

    footprint = np.ones((3, 3, 3), dtype=bool)
    local_max = ndi.maximum_filter(smoothed_padded, footprint=footprint, mode="nearest")
    peak_mask = (smoothed_padded == local_max) & (smoothed_padded > min_threshold)

    coords_padded = np.argwhere(peak_mask)
    if len(coords_padded) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    coords = coords_padded - pad
    Z, Y, X = image_zyx.shape
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < Z)
        & (coords[:, 1] >= 0) & (coords[:, 1] < Y)
        & (coords[:, 2] >= 0) & (coords[:, 2] < X)
    )
    coords = coords[valid]
    coords_padded = coords_padded[valid]
    if len(coords) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    scores = smoothed_padded[tuple(coords_padded.T)]
    candidates = pd.DataFrame({
        "z": coords[:, 0].astype(float), "y": coords[:, 1].astype(float), "x": coords[:, 2].astype(float),
        "score": scores.astype(float),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if len(candidates) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        candidates = candidates.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]
    return candidates


# --------------------------------------------------------------------------- #
# 7. Raw (NMS-deduped, unfiltered) multi-timepoint detection
# --------------------------------------------------------------------------- #
DEFAULT_DETECTOR_CONFIG = {
    "percentile_low": 1.0,
    "percentile_high": 97.0,
    "threshold": 0.005,
    "min_distance_um": 0.75,
}


def detect_all_raw_nodes_for_sample(
    sample_zarr_dir: Path,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_id_start: int = 0,
    max_timepoints: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Detects D's raw local maxima at every timepoint, deduplicates with
    physical NMS, and assigns globally-increasing node_id values.
    """
    T = get_sample_timepoint_count(sample_zarr_dir)
    if max_timepoints is not None:
        T = min(T, max_timepoints)

    rows = []
    next_node_id = node_id_start
    for t in range(T):
        image = read_image_timepoint(sample_zarr_dir, t)
        raw = method_d_boundary_aware_3d_local_max(
            image, detector_config["percentile_low"], detector_config["percentile_high"], detector_config["threshold"]
        )
        deduped = _nms_by_physical_distance(raw, detector_config["min_distance_um"])
        deduped = deduped.sort_values("score", ascending=False).reset_index(drop=True)
        if len(deduped) > RAW_POOL_SAFETY_CAP_PER_TIMEPOINT:
            deduped = deduped.iloc[:RAW_POOL_SAFETY_CAP_PER_TIMEPOINT]

        for _, r in deduped.iterrows():
            rows.append({
                "node_id": next_node_id, "t": t,
                "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"]), "score": float(r["score"]),
            })
            next_node_id += 1

    nodes_df = pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x", "score"])
    return nodes_df, next_node_id


# --------------------------------------------------------------------------- #
# 8. Node filtering (fixed at conservative_tracker.py's real-data best config)
# --------------------------------------------------------------------------- #
# The real-data best config found by conservative_tracker.py's 24,000-row
# sweep: edge_jaccard=0.002215, edge_FP=5700 (vs simple_tracker's
# edge_jaccard=0.00045, edge_FP=97594). Reused here as the FIXED baseline -
# Milestone 6C sweeps tracklet-level filtering on top of this, not the
# node/edge filter parameters themselves again.
BEST_CONSERVATIVE_NODE_FILTER_CONFIG = {
    "max_nodes_per_timepoint": 75,
    "score_quantile_per_timepoint": 0.90,
    "absolute_score_threshold": 0.05,
}
BEST_CONSERVATIVE_EDGE_FILTER_CONFIG = {
    "max_link_distance_um": 7,
    "require_mutual_nearest_neighbor": True,
    "max_edges_per_frame": 50,
}


def _apply_node_filter_one_timepoint(
    nodes_t_df: pd.DataFrame, max_nodes: int, score_quantile: float, abs_threshold: float,
) -> pd.DataFrame:
    """Absolute threshold -> per-timepoint score quantile -> hard cap."""
    filtered = nodes_t_df[nodes_t_df["score"] >= abs_threshold]
    if len(filtered) == 0:
        return filtered
    q_value = filtered["score"].quantile(score_quantile)
    filtered = filtered[filtered["score"] >= q_value]
    if len(filtered) == 0:
        return filtered
    return filtered.sort_values("score", ascending=False).iloc[:max_nodes]


def apply_node_filter(nodes_df: pd.DataFrame, max_nodes: int, score_quantile: float, abs_threshold: float) -> pd.DataFrame:
    """Applies `_apply_node_filter_one_timepoint` independently per timepoint."""
    if len(nodes_df) == 0:
        return nodes_df
    parts = [
        _apply_node_filter_one_timepoint(nodes_df[nodes_df["t"] == t], max_nodes, score_quantile, abs_threshold)
        for t in nodes_df["t"].unique()
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return nodes_df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# 9. Conservative edge filtering (fixed at the same real-data best config)
# --------------------------------------------------------------------------- #
CANDIDATE_EDGE_COLUMNS = ["t", "source_id", "target_id", "source_score", "target_score", "distance_um", "is_mnn", "edge_score"]


def compute_candidate_edges(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Hungarian assignment + mutual-nearest-neighbor flag between every
    consecutive timepoint pair (see conservative_tracker.py for the full
    rationale) - computed once per sample, reused unfiltered/filtered as
    needed.
    """
    if len(nodes_df) == 0:
        return pd.DataFrame(columns=CANDIDATE_EDGE_COLUMNS)

    rows: list[dict] = []
    timepoints = sorted(nodes_df["t"].unique())

    for t, t_next in zip(timepoints[:-1], timepoints[1:]):
        if t_next != t + 1:
            continue

        cur = nodes_df[nodes_df["t"] == t]
        nxt = nodes_df[nodes_df["t"] == t_next]
        if len(cur) == 0 or len(nxt) == 0:
            continue

        cur_coords = cur[["z", "y", "x"]].to_numpy(dtype=float)
        nxt_coords = nxt[["z", "y", "x"]].to_numpy(dtype=float)
        cost = _pairwise_physical_distance_um(cur_coords, nxt_coords)

        row_ind, col_ind = linear_sum_assignment(cost)
        nearest_nxt_for_cur = cost.argmin(axis=1)
        nearest_cur_for_nxt = cost.argmin(axis=0)

        cur_ids = cur["node_id"].to_numpy()
        cur_scores = cur["score"].to_numpy(dtype=float)
        nxt_ids = nxt["node_id"].to_numpy()
        nxt_scores = nxt["score"].to_numpy(dtype=float)

        for r, c in zip(row_ind, col_ind):
            distance = float(cost[r, c])
            is_mnn = bool(nearest_nxt_for_cur[r] == c and nearest_cur_for_nxt[c] == r)
            source_score = float(cur_scores[r])
            target_score = float(nxt_scores[c])
            edge_score = source_score * target_score / (1.0 + distance)
            rows.append({
                "t": int(t), "source_id": int(cur_ids[r]), "target_id": int(nxt_ids[c]),
                "source_score": source_score, "target_score": target_score,
                "distance_um": distance, "is_mnn": is_mnn, "edge_score": edge_score,
            })

    return pd.DataFrame(rows, columns=CANDIDATE_EDGE_COLUMNS)


def apply_edge_filter(
    candidate_edges_df: pd.DataFrame, max_link_distance_um: float, require_mnn: bool, max_edges_per_frame: int,
) -> pd.DataFrame:
    """Distance cutoff -> optional mutual-nearest-neighbor requirement ->
    top `max_edges_per_frame` edges per timepoint by edge_score. Returns
    the FULL candidate-edge rows (not just source_id/target_id) since
    tracklet-building needs distance_um/edge_score per retained edge.
    """
    if len(candidate_edges_df) == 0:
        return candidate_edges_df

    filtered = candidate_edges_df[candidate_edges_df["distance_um"] <= max_link_distance_um]
    if require_mnn:
        filtered = filtered[filtered["is_mnn"]]
    if len(filtered) == 0:
        return filtered

    filtered = filtered.sort_values(["t", "edge_score"], ascending=[True, False])
    filtered = filtered.groupby("t", sort=False, group_keys=False).head(max_edges_per_frame)
    return filtered.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 10. Tracklet construction + feature computation
# --------------------------------------------------------------------------- #
TRACKLET_COLUMNS = [
    "tracklet_id", "node_ids",
    "length_frames", "mean_node_score", "min_node_score", "mean_edge_score",
    "max_link_distance_um", "mean_link_distance_um",
    "smoothness_error_um", "displacement_consistency", "num_time_gaps",
    "start_t", "end_t",
]


def _tracklet_features(tid: int, chain: list[int], node_lookup: pd.DataFrame, edge_lookup: dict) -> dict:
    """Computes all requested tracklet-level features for one temporal
    chain of node_ids (already time-ordered by construction).
    """
    info = node_lookup.loc[chain]
    ts = info["t"].to_numpy()
    scores = info["score"].to_numpy(dtype=float)
    positions = info[["z", "y", "x"]].to_numpy(dtype=float)

    length_frames = len(chain)
    num_time_gaps = int(np.sum(np.diff(ts) > 1)) if length_frames > 1 else 0

    if length_frames >= 2:
        edge_pairs = list(zip(chain[:-1], chain[1:]))
        distances = np.array([edge_lookup[p][0] for p in edge_pairs], dtype=float)
        edge_scores = np.array([edge_lookup[p][1] for p in edge_pairs], dtype=float)
        mean_edge_score = float(edge_scores.mean())
        max_link_distance_um = float(distances.max())
        mean_link_distance_um = float(distances.mean())

        disp_vectors = (positions[1:] - positions[:-1]) * _VOXEL_SCALE
        if len(disp_vectors) >= 2:
            accel = disp_vectors[1:] - disp_vectors[:-1]
            smoothness_error_um = float(np.linalg.norm(accel, axis=1).mean())
        else:
            smoothness_error_um = 0.0

        net_displacement_um = physical_distance_um(positions[0], positions[-1])
        total_path_length_um = float(np.linalg.norm(disp_vectors, axis=1).sum())
        displacement_consistency = (net_displacement_um / total_path_length_um) if total_path_length_um > 0 else 1.0
    else:
        mean_edge_score = float("nan")
        max_link_distance_um = 0.0
        mean_link_distance_um = 0.0
        smoothness_error_um = 0.0
        displacement_consistency = 1.0

    return {
        "tracklet_id": tid, "node_ids": tuple(chain),
        "length_frames": length_frames,
        "mean_node_score": float(scores.mean()), "min_node_score": float(scores.min()),
        "mean_edge_score": mean_edge_score,
        "max_link_distance_um": max_link_distance_um, "mean_link_distance_um": mean_link_distance_um,
        "smoothness_error_um": smoothness_error_um, "displacement_consistency": displacement_consistency,
        "num_time_gaps": num_time_gaps,
        "start_t": int(ts.min()), "end_t": int(ts.max()),
    }


def build_tracklets(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Converts frame-to-frame edges into connected temporal tracklets.
    Since edges come from a per-frame Hungarian assignment (each row/column
    used at most once) that has only been further subset by filtering,
    every node structurally has at most one outgoing and one incoming
    edge - tracklets are therefore simple, non-branching chains, built by
    following each chain from its start (a node with no incoming edge) to
    its end. Nodes with no edges at all are not part of any tracklet
    (they would fail even the smallest min_tracklet_length=2 sweep value).
    """
    if len(edges_df) == 0:
        return pd.DataFrame(columns=TRACKLET_COLUMNS)

    node_lookup = nodes_df.set_index("node_id")[["t", "z", "y", "x", "score"]]
    succ = dict(zip(edges_df["source_id"].astype(int), edges_df["target_id"].astype(int)))
    has_pred = set(edges_df["target_id"].astype(int))
    all_sources = set(edges_df["source_id"].astype(int))
    edge_lookup = {
        (int(s), int(t)): (float(d), float(e))
        for s, t, d, e in zip(edges_df["source_id"], edges_df["target_id"], edges_df["distance_um"], edges_df["edge_score"])
    }

    chain_starts = [s for s in all_sources if s not in has_pred]

    tracklets = []
    visited: set[int] = set()
    tid = 0
    for start in chain_starts:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = start
        while cur in succ:
            nxt = succ[cur]
            if nxt in visited:  # defensive cycle guard - should not occur with strictly-increasing t
                break
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        tracklets.append(_tracklet_features(tid, chain, node_lookup, edge_lookup))
        tid += 1

    return pd.DataFrame(tracklets, columns=TRACKLET_COLUMNS)


# --------------------------------------------------------------------------- #
# 11. Tracklet-level filtering sweep
# --------------------------------------------------------------------------- #
DEFAULT_MIN_TRACKLET_LENGTH_VALUES = (2, 3, 5, 8, 10, 15, 20)
DEFAULT_MIN_MEAN_NODE_SCORE_VALUES = (0.05, 0.08, 0.10, 0.15, 0.20)
DEFAULT_MIN_MEAN_EDGE_SCORE_QUANTILE_VALUES = (0.50, 0.70, 0.80, 0.90, 0.95)
DEFAULT_MAX_MEAN_LINK_DISTANCE_UM_VALUES = (2, 3, 4, 5, 7)
DEFAULT_MAX_SMOOTHNESS_ERROR_UM_VALUES = (2, 3, 5, 7, 10)
DEFAULT_KEEP_TOP_K_TRACKLETS_VALUES = (1, 2, 3, 5, 10, 20, 50)

NODE_POLICIES = ("A_tracklet_nodes_only", "B_all_filtered_nodes")


def apply_tracklet_filter(
    tracklets_df: pd.DataFrame,
    min_tracklet_length: int,
    min_mean_node_score: float,
    min_mean_edge_score_threshold: float,
    max_mean_link_distance_um: float,
    max_smoothness_error_um: float,
    keep_top_k: int,
) -> pd.DataFrame:
    """Filters an already-built tracklet feature table, then keeps only the
    top `keep_top_k` survivors ranked by mean_edge_score (a composite
    node-confidence x link-distance quality signal, same philosophy as
    conservative_tracker.py's edge_score ranking).
    """
    if len(tracklets_df) == 0:
        return tracklets_df

    filtered = tracklets_df[
        (tracklets_df["length_frames"] >= min_tracklet_length)
        & (tracklets_df["mean_node_score"] >= min_mean_node_score)
        & (tracklets_df["mean_edge_score"] >= min_mean_edge_score_threshold)
        & (tracklets_df["mean_link_distance_um"] <= max_mean_link_distance_um)
        & (tracklets_df["smoothness_error_um"] <= max_smoothness_error_um)
    ]
    if len(filtered) == 0:
        return filtered
    return filtered.sort_values("mean_edge_score", ascending=False).head(keep_top_k)


def _edges_from_kept_tracklets(kept_tracklets_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs source_id/target_id edge rows from each kept tracklet's
    node_ids chain (the exact same consecutive pairs the tracklet was built
    from - no re-derivation needed).
    """
    if len(kept_tracklets_df) == 0:
        return pd.DataFrame(columns=["source_id", "target_id"])
    rows = []
    for node_ids in kept_tracklets_df["node_ids"]:
        for s, t in zip(node_ids[:-1], node_ids[1:]):
            rows.append({"source_id": s, "target_id": t})
    return pd.DataFrame(rows, columns=["source_id", "target_id"])


def reconstruct_policy_a(kept_tracklets_df: pd.DataFrame, node_lookup_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policy A: both nodes and edges restricted to kept tracklets."""
    if len(kept_tracklets_df) == 0:
        return node_lookup_df.iloc[0:0], pd.DataFrame(columns=["source_id", "target_id"])
    kept_node_ids = set(itertools.chain.from_iterable(kept_tracklets_df["node_ids"]))
    pred_nodes = node_lookup_df[node_lookup_df["node_id"].isin(kept_node_ids)].reset_index(drop=True)
    pred_edges = _edges_from_kept_tracklets(kept_tracklets_df)
    return pred_nodes, pred_edges


def reconstruct_policy_b(all_filtered_nodes_df: pd.DataFrame, kept_tracklets_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policy B: nodes stay exactly as the conservative node filter
    produced them (unchanged); only edges are restricted to kept tracklets.
    """
    pred_edges = _edges_from_kept_tracklets(kept_tracklets_df)
    return all_filtered_nodes_df, pred_edges


# --------------------------------------------------------------------------- #
# 12. Local evaluation (embedded copy of local_metric.py's evaluator)
# --------------------------------------------------------------------------- #
def match_nodes_by_timepoint(pred_nodes: pd.DataFrame, gt_nodes: pd.DataFrame, max_distance_um: float = 7.0) -> dict:
    """Match predicted nodes to GT nodes independently per timepoint via the
    Hungarian algorithm on physical (um) distance. See local_metric.py for
    the original/canonical implementation - identical logic, copied here
    to avoid a cross-file import.
    """
    pred_to_gt: dict = {}
    match_rows: list[dict] = []
    unmatched_pred: list = []
    unmatched_gt: list = []

    if len(pred_nodes) == 0 and len(gt_nodes) == 0:
        timepoints: list = []
    else:
        timepoints = sorted(set(pred_nodes["t"].unique()) | set(gt_nodes["t"].unique()))

    for t in timepoints:
        pred_t = pred_nodes[pred_nodes["t"] == t]
        gt_t = gt_nodes[gt_nodes["t"] == t]

        if len(pred_t) == 0 or len(gt_t) == 0:
            unmatched_pred.extend(pred_t["node_id"].tolist())
            unmatched_gt.extend(gt_t["node_id"].tolist())
            continue

        pred_coords = pred_t[["z", "y", "x"]].to_numpy(dtype=float)
        gt_coords = gt_t[["z", "y", "x"]].to_numpy(dtype=float)
        cost = _pairwise_physical_distance_um(pred_coords, gt_coords)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_rows = set()
        matched_cols = set()
        for r, c in zip(row_ind, col_ind):
            distance = float(cost[r, c])
            if distance <= max_distance_um:
                pred_id = pred_t["node_id"].iloc[r]
                gt_id = gt_t["node_id"].iloc[c]
                pred_to_gt[pred_id] = gt_id
                match_rows.append({"t": t, "pred_node_id": pred_id, "gt_node_id": gt_id, "distance_um": distance})
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_pred.extend(pred_t["node_id"].iloc[r] for r in range(len(pred_t)) if r not in matched_rows)
        unmatched_gt.extend(gt_t["node_id"].iloc[c] for c in range(len(gt_t)) if c not in matched_cols)

    matches_df = pd.DataFrame(match_rows, columns=["t", "pred_node_id", "gt_node_id", "distance_um"])
    return {
        "pred_to_gt": pred_to_gt,
        "matches_df": matches_df,
        "unmatched_pred_nodes": unmatched_pred,
        "unmatched_gt_nodes": unmatched_gt,
    }


def compute_edge_jaccard(
    pred_edges: pd.DataFrame, gt_edges: pd.DataFrame, pred_to_gt: dict,
) -> dict:
    """A predicted edge is a true positive if both endpoints map to GT nodes
    and that (source, target) pair is an actual GT edge.
    """
    gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"])) if len(gt_edges) else set()

    tp = 0
    fp = 0
    recovered_gt_edges = set()

    for source_id, target_id in zip(pred_edges["source_id"], pred_edges["target_id"]):
        mapped_source = pred_to_gt.get(source_id)
        mapped_target = pred_to_gt.get(target_id)
        if mapped_source is not None and mapped_target is not None and (mapped_source, mapped_target) in gt_edge_set:
            tp += 1
            recovered_gt_edges.add((mapped_source, mapped_target))
        else:
            fp += 1

    fn = len(gt_edge_set) - len(recovered_gt_edges)
    denom = tp + fp + fn
    edge_jaccard = tp / denom if denom > 0 else 1.0

    return {"edge_TP": tp, "edge_FP": fp, "edge_FN": fn, "edge_jaccard": edge_jaccard}


# --------------------------------------------------------------------------- #
# 13. Full tracklet-level filter sweep (both node policies)
# --------------------------------------------------------------------------- #
EVAL_COLUMNS = [
    "node_policy",
    "min_tracklet_length", "min_mean_node_score", "min_mean_edge_score_quantile",
    "max_mean_link_distance_um", "max_smoothness_error_um", "keep_top_k_tracklets_per_sample",
    "edge_jaccard", "edge_TP", "edge_FP", "edge_FN", "precision", "recall",
    "node_matches", "unmatched_gt_nodes", "unmatched_pred_nodes",
    "avg_pred_nodes_per_timepoint", "avg_edges_per_frame", "avg_tracklet_length", "num_tracklets_kept",
    "runtime_seconds",
]


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    return precision, recall


def run_tracklet_precision_eval(
    n_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_config: dict = BEST_CONSERVATIVE_NODE_FILTER_CONFIG,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    min_tracklet_length_values: Sequence[int] = DEFAULT_MIN_TRACKLET_LENGTH_VALUES,
    min_mean_node_score_values: Sequence[float] = DEFAULT_MIN_MEAN_NODE_SCORE_VALUES,
    min_mean_edge_score_quantile_values: Sequence[float] = DEFAULT_MIN_MEAN_EDGE_SCORE_QUANTILE_VALUES,
    max_mean_link_distance_um_values: Sequence[float] = DEFAULT_MAX_MEAN_LINK_DISTANCE_UM_VALUES,
    max_smoothness_error_um_values: Sequence[float] = DEFAULT_MAX_SMOOTHNESS_ERROR_UM_VALUES,
    keep_top_k_values: Sequence[int] = DEFAULT_KEEP_TOP_K_TRACKLETS_VALUES,
    match_max_distance_um: float = 7.0,
    csv_path: str = "/kaggle/working/tracklet_precision_eval.csv",
    progress_every: int = 200,
) -> pd.DataFrame:
    """Detection, node filtering, edge filtering, and tracklet construction
    are computed ONCE per sample (all fixed at the best conservative
    config). The tracklet-level sweep then only re-filters the already-
    built tracklet feature table. For node_policy B (nodes never change
    across the sweep), GT node-matching is computed ONCE per sample and
    reused for every combo - only edge_TP/FP/FN need recomputation.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=EVAL_COLUMNS)

    samples = train_dirs[:n_samples]
    n_combos = (
        len(min_tracklet_length_values) * len(min_mean_node_score_values) * len(min_mean_edge_score_quantile_values)
        * len(max_mean_link_distance_um_values) * len(max_smoothness_error_um_values) * len(keep_top_k_values)
    )
    n_total = n_combos * len(NODE_POLICIES)
    print(f"Sweeping {n_combos} tracklet-filter combo(s) x {len(NODE_POLICIES)} node policies = {n_total} total row(s).")

    print(f"Detecting + filtering + building tracklets for {len(samples)} sample(s) (fixed conservative baseline)...")
    filtered_nodes_by_sample: dict[Path, pd.DataFrame] = {}
    tracklets_by_sample: dict[Path, pd.DataFrame] = {}
    gt_by_sample: dict[Path, tuple[pd.DataFrame, pd.DataFrame]] = {}
    # Policy B's node set never changes across the sweep - precompute its
    # GT match ONCE per sample and reuse for every combo (no Hungarian
    # needed per combo for policy B at all).
    policy_b_match_by_sample: dict[Path, dict] = {}
    total_timepoints = 0
    total_frame_pairs = 0
    edge_score_quantile_thresholds_by_sample: dict[Path, dict[float, float]] = {}

    for sample_dir in samples:
        t0 = time.time()
        raw_nodes_df, _ = detect_all_raw_nodes_for_sample(sample_dir, detector_config, node_id_start=0)
        filtered_nodes = apply_node_filter(
            raw_nodes_df,
            node_filter_config["max_nodes_per_timepoint"],
            node_filter_config["score_quantile_per_timepoint"],
            node_filter_config["absolute_score_threshold"],
        )
        candidate_edges = compute_candidate_edges(filtered_nodes)
        base_edges = apply_edge_filter(
            candidate_edges,
            edge_filter_config["max_link_distance_um"],
            edge_filter_config["require_mutual_nearest_neighbor"],
            edge_filter_config["max_edges_per_frame"],
        )
        tracklets_df = build_tracklets(filtered_nodes, base_edges)

        filtered_nodes_by_sample[sample_dir] = filtered_nodes[["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
        tracklets_by_sample[sample_dir] = tracklets_df
        gt_nodes, gt_edges = read_geff(sample_dir)
        gt_by_sample[sample_dir] = (gt_nodes, gt_edges)
        policy_b_match_by_sample[sample_dir] = match_nodes_by_timepoint(
            filtered_nodes_by_sample[sample_dir], gt_nodes, match_max_distance_um
        )

        edge_score_quantile_thresholds_by_sample[sample_dir] = {
            q: (float(tracklets_df["mean_edge_score"].quantile(q)) if len(tracklets_df) else float("inf"))
            for q in min_mean_edge_score_quantile_values
        }

        T = get_sample_timepoint_count(sample_dir)
        total_timepoints += T
        total_frame_pairs += max(0, T - 1)
        print(f"  {sample_dir.stem}: {len(filtered_nodes)} filtered node(s), {len(tracklets_df)} tracklet(s) "
              f"built in {time.time() - t0:.1f}s")

    # Precompute per-sample GT edge sets once (avoid rebuilding per combo).
    gt_edge_set_by_sample = {
        sample_dir: (set(zip(edges["source_id"], edges["target_id"])) if len(edges) else set())
        for sample_dir, (_, edges) in gt_by_sample.items()
    }

    rows: list[dict] = []
    combo_idx = 0
    sweep_t0 = time.time()

    for min_tracklet_length in min_tracklet_length_values:
        for min_mean_node_score in min_mean_node_score_values:
            for edge_quantile in min_mean_edge_score_quantile_values:
                for max_mean_link_distance_um in max_mean_link_distance_um_values:
                    for max_smoothness_error_um in max_smoothness_error_um_values:
                        for keep_top_k in keep_top_k_values:
                            combo_idx += 1
                            row_t0 = time.time()

                            kept_by_sample: dict[Path, pd.DataFrame] = {}
                            total_kept_length = 0
                            total_kept_count = 0
                            for sample_dir in samples:
                                threshold = edge_score_quantile_thresholds_by_sample[sample_dir][edge_quantile]
                                kept = apply_tracklet_filter(
                                    tracklets_by_sample[sample_dir],
                                    min_tracklet_length, min_mean_node_score, threshold,
                                    max_mean_link_distance_um, max_smoothness_error_um, keep_top_k,
                                )
                                kept_by_sample[sample_dir] = kept
                                total_kept_length += kept["length_frames"].sum() if len(kept) else 0
                                total_kept_count += len(kept)

                            avg_tracklet_length = (total_kept_length / total_kept_count) if total_kept_count else 0.0

                            for node_policy in NODE_POLICIES:
                                edge_TP = edge_FP = edge_FN = 0
                                node_matches = unmatched_gt = unmatched_pred = 0
                                total_pred_nodes = 0
                                total_edges = 0

                                for sample_dir in samples:
                                    kept = kept_by_sample[sample_dir]
                                    gt_nodes, gt_edges = gt_by_sample[sample_dir]

                                    if node_policy == "A_tracklet_nodes_only":
                                        pred_nodes, pred_edges = reconstruct_policy_a(kept, filtered_nodes_by_sample[sample_dir])
                                        match = match_nodes_by_timepoint(pred_nodes, gt_nodes, match_max_distance_um)
                                    else:
                                        pred_nodes, pred_edges = reconstruct_policy_b(filtered_nodes_by_sample[sample_dir], kept)
                                        match = policy_b_match_by_sample[sample_dir]

                                    edge_result = compute_edge_jaccard(pred_edges, gt_edges, match["pred_to_gt"])
                                    edge_TP += edge_result["edge_TP"]
                                    edge_FP += edge_result["edge_FP"]
                                    edge_FN += edge_result["edge_FN"]
                                    node_matches += len(match["pred_to_gt"])
                                    unmatched_gt += len(match["unmatched_gt_nodes"])
                                    unmatched_pred += len(match["unmatched_pred_nodes"])
                                    total_pred_nodes += len(pred_nodes)
                                    total_edges += len(pred_edges)

                                denom = edge_TP + edge_FP + edge_FN
                                edge_jaccard = edge_TP / denom if denom > 0 else 1.0
                                precision, recall = _precision_recall(edge_TP, edge_FP, edge_FN)

                                rows.append({
                                    "node_policy": node_policy,
                                    "min_tracklet_length": min_tracklet_length,
                                    "min_mean_node_score": min_mean_node_score,
                                    "min_mean_edge_score_quantile": edge_quantile,
                                    "max_mean_link_distance_um": max_mean_link_distance_um,
                                    "max_smoothness_error_um": max_smoothness_error_um,
                                    "keep_top_k_tracklets_per_sample": keep_top_k,
                                    "edge_jaccard": edge_jaccard,
                                    "edge_TP": edge_TP, "edge_FP": edge_FP, "edge_FN": edge_FN,
                                    "precision": precision, "recall": recall,
                                    "node_matches": node_matches,
                                    "unmatched_gt_nodes": unmatched_gt,
                                    "unmatched_pred_nodes": unmatched_pred,
                                    "avg_pred_nodes_per_timepoint": (total_pred_nodes / total_timepoints) if total_timepoints else float("nan"),
                                    "avg_edges_per_frame": (total_edges / total_frame_pairs) if total_frame_pairs else float("nan"),
                                    "avg_tracklet_length": avg_tracklet_length,
                                    "num_tracklets_kept": total_kept_count,
                                    "runtime_seconds": time.time() - row_t0,
                                })

                            if combo_idx % progress_every == 0 or combo_idx == n_combos:
                                elapsed = time.time() - sweep_t0
                                eta = elapsed / combo_idx * (n_combos - combo_idx)
                                print(f"  [{combo_idx}/{n_combos} tracklet-filter combos] "
                                      f"{len(rows)} row(s) so far, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    eval_df = pd.DataFrame(rows, columns=EVAL_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_path, index=False)
    print(f"\nSaved tracklet precision eval sweep ({len(eval_df)} row(s)) to {out_path}")
    return eval_df


def select_best_config(
    eval_df: pd.DataFrame,
    baseline_edge_jaccard: float = 0.002215,
    baseline_precision: float = 0.002276,
    goal_max_edge_fp: int = 1000,
    csv_path: str = "/kaggle/working/tracklet_precision_best_config.csv",
) -> pd.DataFrame:
    """Best config = highest edge_jaccard; ties broken by precision then
    recall descending - "prioritize precision first, then recall" is
    embodied by edge_jaccard maximization since it penalizes both FP and
    FN, and precision is the explicit tie-break.
    """
    if len(eval_df) == 0:
        print("[warn] no eval rows to select a best config from.")
        return pd.DataFrame(columns=EVAL_COLUMNS)

    ranked = eval_df.sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False],
    ).reset_index(drop=True)
    best = ranked.head(1)

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)

    row = best.iloc[0]
    print(f"\nBest config: node_policy={row['node_policy']} "
          f"min_tracklet_length={row['min_tracklet_length']} "
          f"min_mean_node_score={row['min_mean_node_score']} "
          f"min_mean_edge_score_quantile={row['min_mean_edge_score_quantile']} "
          f"max_mean_link_distance_um={row['max_mean_link_distance_um']} "
          f"max_smoothness_error_um={row['max_smoothness_error_um']} "
          f"keep_top_k_tracklets_per_sample={row['keep_top_k_tracklets_per_sample']}")
    print(f"  edge_jaccard={row['edge_jaccard']:.6f} (baseline {baseline_edge_jaccard}) "
          f"precision={row['precision']:.6f} (baseline {baseline_precision}) "
          f"recall={row['recall']:.6f} edge_FP={row['edge_FP']} (goal: < {goal_max_edge_fp})")
    print(f"  goal check: edge_FP<{goal_max_edge_fp} -> {row['edge_FP'] < goal_max_edge_fp}, "
          f"edge_jaccard>{baseline_edge_jaccard} -> {row['edge_jaccard'] > baseline_edge_jaccard}, "
          f"precision>>{baseline_precision} -> {row['precision'] > baseline_precision}")
    print(f"Saved best config to {out_path}")
    return best


# --------------------------------------------------------------------------- #
# 14. Submission generation (uses the best tracklet-level config)
# --------------------------------------------------------------------------- #
SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def build_tracklet_precision_sample_submission_rows(
    sample_zarr_dir: Path,
    detector_config: dict,
    node_filter_config: dict,
    edge_filter_config: dict,
    tracklet_filter_config: dict,
    node_policy: str,
    next_id: int,
    next_node_id: int,
) -> tuple[list[dict], int, int]:
    """Runs the full tracklet-precision pipeline (detect -> node filter ->
    edge filter -> build tracklets -> tracklet filter -> reconstruct by
    node_policy) for one sample.
    """
    dataset_name = sample_zarr_dir.stem
    raw_nodes_df, next_node_id = detect_all_raw_nodes_for_sample(
        sample_zarr_dir, detector_config, node_id_start=next_node_id
    )
    filtered_nodes = apply_node_filter(
        raw_nodes_df,
        node_filter_config["max_nodes_per_timepoint"],
        node_filter_config["score_quantile_per_timepoint"],
        node_filter_config["absolute_score_threshold"],
    )[["node_id", "t", "z", "y", "x", "score"]].reset_index(drop=True)
    candidate_edges = compute_candidate_edges(filtered_nodes)
    base_edges = apply_edge_filter(
        candidate_edges,
        edge_filter_config["max_link_distance_um"],
        edge_filter_config["require_mutual_nearest_neighbor"],
        edge_filter_config["max_edges_per_frame"],
    )
    tracklets_df = build_tracklets(filtered_nodes, base_edges)

    edge_quantile = tracklet_filter_config["min_mean_edge_score_quantile"]
    threshold = float(tracklets_df["mean_edge_score"].quantile(edge_quantile)) if len(tracklets_df) else float("inf")
    kept = apply_tracklet_filter(
        tracklets_df,
        tracklet_filter_config["min_tracklet_length"],
        tracklet_filter_config["min_mean_node_score"],
        threshold,
        tracklet_filter_config["max_mean_link_distance_um"],
        tracklet_filter_config["max_smoothness_error_um"],
        tracklet_filter_config["keep_top_k_tracklets_per_sample"],
    )

    node_lookup = filtered_nodes[["node_id", "t", "z", "y", "x"]]
    if node_policy == "A_tracklet_nodes_only":
        pred_nodes, pred_edges = reconstruct_policy_a(kept, node_lookup)
    else:
        pred_nodes, pred_edges = reconstruct_policy_b(node_lookup, kept)

    rows: list[dict] = []
    for _, r in pred_nodes.iterrows():
        rows.append({
            "id": next_id, "dataset": dataset_name, "row_type": "node",
            "node_id": int(r["node_id"]), "t": int(r["t"]), "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"]),
            "source_id": -1, "target_id": -1,
        })
        next_id += 1

    for _, r in pred_edges.iterrows():
        rows.append({
            "id": next_id, "dataset": dataset_name, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
        })
        next_id += 1

    return rows, next_id, next_node_id


def build_tracklet_precision_submission(
    test_zarr_dirs: Sequence[Path], detector_config: dict, node_filter_config: dict, edge_filter_config: dict,
    tracklet_filter_config: dict, node_policy: str,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    next_id = 0
    next_node_id = 0
    for sample_dir in test_zarr_dirs:
        t0 = time.time()
        rows, next_id, next_node_id = build_tracklet_precision_sample_submission_rows(
            sample_dir, detector_config, node_filter_config, edge_filter_config, tracklet_filter_config, node_policy,
            next_id, next_node_id,
        )
        all_rows.extend(rows)
        print(f"  {sample_dir.stem}: {len(rows)} row(s) generated in {time.time() - t0:.1f}s")
    return pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)


def validate_submission(df: pd.DataFrame, expected_datasets: Sequence[str]) -> str:
    """Structural validation mirroring submission_pipeline.py's checks, plus
    a soft comparison against a real sample_submission.csv's columns if one
    can be found on disk. Raises AssertionError if any hard check fails.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("columns match required schema", list(df.columns) == SUBMISSION_COLUMNS, f"got {list(df.columns)}")

    na_cols = df.columns[df.isna().any()].tolist()
    check("no NaN values anywhere", not na_cols, f"NaN columns: {na_cols}")

    check("id column is consecutive starting at 0", df["id"].tolist() == list(range(len(df))))

    missing = sorted(set(expected_datasets) - set(df["dataset"].unique()))
    check("every expected test dataset is present", not missing, f"missing: {missing}")

    check("row_type values are only 'node'/'edge'", set(df["row_type"].unique()) <= {"node", "edge"})

    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]

    bad_nodes = nodes[(nodes["source_id"] != -1) | (nodes["target_id"] != -1)]
    check("node rows have source_id=target_id=-1", len(bad_nodes) == 0, f"{len(bad_nodes)} bad rows")

    bad_edges = edges[
        (edges["node_id"] != -1) | (edges["t"] != -1) | (edges["z"] != -1)
        | (edges["y"] != -1) | (edges["x"] != -1)
    ]
    check("edge rows have node_id,t,z,y,x=-1", len(bad_edges) == 0, f"{len(bad_edges)} bad rows")

    check("node_id is unique among node rows", nodes["node_id"].is_unique)

    valid_node_ids = set(nodes["node_id"])
    bad_source = edges[~edges["source_id"].isin(valid_node_ids)]
    bad_target = edges[~edges["target_id"].isin(valid_node_ids)]
    check("edge source_id values reference an existing node_id", len(bad_source) == 0, f"{len(bad_source)} bad rows")
    check("edge target_id values reference an existing node_id", len(bad_target) == 0, f"{len(bad_target)} bad rows")

    sample_submission_path = find_sample_submission_csv()
    if sample_submission_path is not None:
        try:
            sample_cols = list(pd.read_csv(sample_submission_path, nrows=0).columns)
            check(
                f"columns match {sample_submission_path.name}",
                sample_cols == SUBMISSION_COLUMNS,
                f"sample_submission columns={sample_cols}",
            )
        except Exception as exc:
            print(f"[warn] could not read {sample_submission_path} for schema comparison: {exc!r}")
    else:
        print("[info] no sample_submission*.csv found on disk to cross-check columns against.")

    lines = ["Validation report:"]
    all_passed = True
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_passed = False
        line = f"  [{mark}] {c['name']}"
        if c["detail"] and not c["passed"]:
            line += f"  -- {c['detail']}"
        lines.append(line)
    report = "\n".join(lines)
    print(report)

    if not all_passed:
        raise AssertionError("Submission validation FAILED. See report above.")
    return report


def generate_and_save_submission(
    detector_config: dict,
    node_filter_config: dict,
    edge_filter_config: dict,
    tracklet_filter_config: dict,
    node_policy: str,
    out_path: str = "/kaggle/working/submission.csv",
) -> pd.DataFrame:
    """Generates predictions for every discovered test .zarr dataset (no
    hardcoded sample names) using the best tracklet-precision config found
    by the sweep, validates the result, and writes it to `out_path`. Never
    calls any Kaggle submission API.
    """
    test_dirs = find_test_zarr_dirs()
    if not test_dirs:
        print("[error] no test .zarr datasets found. Nothing to submit.")
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    print(f"Discovered {len(test_dirs)} test dataset(s): {[p.stem for p in test_dirs]}")
    submission = build_tracklet_precision_submission(
        test_dirs, detector_config, node_filter_config, edge_filter_config, tracklet_filter_config, node_policy
    )
    print(f"\nsubmission shape: {submission.shape}")

    expected_datasets = [p.stem for p in test_dirs]
    validate_submission(submission, expected_datasets)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print(f"\nSaved submission to {out} (exists={out.exists()})")
    return submission


# --------------------------------------------------------------------------- #
# 15. Tests
# --------------------------------------------------------------------------- #
def run_tracklet_precision_tests() -> None:
    """Correctness tests for tracklet construction/features, tracklet-level
    filtering, both node-reconstruction policies, and submission
    generation, using small synthetic fixtures.
    """
    # Test 1: a simple 3-node straight-line tracklet has the expected
    # length, perfect displacement consistency, and zero smoothness error.
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2], "t": [0, 1, 2],
        "z": [0.0, 0.0, 0.0], "y": [0.0, 10.0, 20.0], "x": [0.0, 0.0, 0.0],
        "score": [0.9, 0.8, 0.7],
    })
    edges = pd.DataFrame({
        "t": [0, 1], "source_id": [0, 1], "target_id": [1, 2],
        "source_score": [0.9, 0.8], "target_score": [0.8, 0.7],
        "distance_um": [4.0625, 4.0625], "is_mnn": [True, True],
        "edge_score": [0.9 * 0.8 / 5.0625, 0.8 * 0.7 / 5.0625],
    })
    tracklets = build_tracklets(nodes, edges)
    assert len(tracklets) == 1, f"expected exactly 1 tracklet, got {len(tracklets)}"
    row = tracklets.iloc[0]
    assert row["length_frames"] == 3
    assert row["node_ids"] == (0, 1, 2)
    assert abs(row["displacement_consistency"] - 1.0) < 1e-6, "a straight line should have displacement_consistency == 1.0"
    assert abs(row["smoothness_error_um"]) < 1e-6, "constant-velocity motion should have ~0 smoothness error"
    assert row["num_time_gaps"] == 0

    # Test 2: two separate tracklets (no shared nodes) are built independently.
    nodes2 = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.0, 5.0, 100.0, 105.0], "x": [0.0] * 4,
        "score": [0.9, 0.8, 0.6, 0.5],
    })
    edges2 = pd.DataFrame({
        "t": [0, 0], "source_id": [0, 2], "target_id": [1, 3],
        "source_score": [0.9, 0.6], "target_score": [0.8, 0.5],
        "distance_um": [2.03, 2.03], "is_mnn": [True, True],
        "edge_score": [0.9 * 0.8 / 3.03, 0.6 * 0.5 / 3.03],
    })
    tracklets2 = build_tracklets(nodes2, edges2)
    assert len(tracklets2) == 2, f"expected 2 independent tracklets, got {len(tracklets2)}"

    # Test 3: tracklet-level filtering rejects a short (length 1 chain
    # implied) or low-score tracklet, and keeps a good one.
    filtered = apply_tracklet_filter(
        tracklets2, min_tracklet_length=2, min_mean_node_score=0.55,
        min_mean_edge_score_threshold=0.0, max_mean_link_distance_um=5.0, max_smoothness_error_um=5.0, keep_top_k=10,
    )
    assert len(filtered) == 2, "both length-2 tracklets should pass a lenient filter"
    strict = apply_tracklet_filter(
        tracklets2, min_tracklet_length=2, min_mean_node_score=0.7,
        min_mean_edge_score_threshold=0.0, max_mean_link_distance_um=5.0, max_smoothness_error_um=5.0, keep_top_k=10,
    )
    assert len(strict) == 1, "only the higher-mean-score tracklet should survive min_mean_node_score=0.7"

    # Test 4: keep_top_k_tracklets_per_sample actually caps the count.
    top1 = apply_tracklet_filter(
        tracklets2, min_tracklet_length=2, min_mean_node_score=0.0,
        min_mean_edge_score_threshold=0.0, max_mean_link_distance_um=5.0, max_smoothness_error_um=5.0, keep_top_k=1,
    )
    assert len(top1) == 1

    # Test 5: policy A restricts both nodes and edges to kept tracklets;
    # policy B keeps all nodes but restricts edges the same way.
    node_lookup = nodes2[["node_id", "t", "z", "y", "x"]]
    kept_one = tracklets2[tracklets2["tracklet_id"] == tracklets2.iloc[0]["tracklet_id"]]
    pred_nodes_a, pred_edges_a = reconstruct_policy_a(kept_one, node_lookup)
    pred_nodes_b, pred_edges_b = reconstruct_policy_b(node_lookup, kept_one)
    assert len(pred_nodes_a) == 2, "policy A should only keep the 2 nodes from the 1 kept tracklet"
    assert len(pred_nodes_b) == 4, "policy B should keep ALL 4 filtered nodes regardless of tracklet membership"
    assert len(pred_edges_a) == len(pred_edges_b) == 1, "both policies should produce the same single kept edge"

    # Test 6: full submission schema validity + rejection of a broken submission.
    fake_rows = []
    next_id = 0
    next_node_id = 0
    for dataset_name in ("fake_a", "fake_b"):
        for t, node_id in zip((0, 1), (next_node_id, next_node_id + 1)):
            fake_rows.append({
                "id": next_id, "dataset": dataset_name, "row_type": "node",
                "node_id": node_id, "t": t, "z": 1.0, "y": 2.0, "x": 3.0, "source_id": -1, "target_id": -1,
            })
            next_id += 1
        fake_rows.append({
            "id": next_id, "dataset": dataset_name, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": next_node_id, "target_id": next_node_id + 1,
        })
        next_id += 1
        next_node_id += 2
    fake_submission = pd.DataFrame(fake_rows, columns=SUBMISSION_COLUMNS)
    validate_submission(fake_submission, expected_datasets=["fake_a", "fake_b"])
    broken = fake_submission.copy()
    broken.loc[broken["row_type"] == "edge", "target_id"] = 999999
    raised = False
    try:
        validate_submission(broken, expected_datasets=["fake_a", "fake_b"])
    except AssertionError:
        raised = True
    assert raised, "validate_submission should reject a dangling edge target_id"

    print("All tracklet_precision_tracker tests passed (6/6).")


# --------------------------------------------------------------------------- #
# 16. Top-level driver
# --------------------------------------------------------------------------- #
def run_tracklet_precision_pipeline(
    n_eval_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_config: dict = BEST_CONSERVATIVE_NODE_FILTER_CONFIG,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    baseline_edge_jaccard: float = 0.002215,
    baseline_precision: float = 0.002276,
    submission_path: str = "/kaggle/working/submission.csv",
    generate_submission: bool = True,
) -> dict:
    """Runs the tests, the full tracklet-level filter sweep (+best-config
    selection), and - if `generate_submission` - builds submission.csv for
    every discovered test dataset using the best config found. Never
    calls any Kaggle submission API.
    """
    print("=== Self-test: tracklet construction/filtering + submission-validation unit tests ===")
    run_tracklet_precision_tests()

    print("\n=== Tracklet-precision filter sweep (first %d train sample(s)) ===" % n_eval_samples)
    eval_df = run_tracklet_precision_eval(
        n_samples=n_eval_samples, detector_config=detector_config,
        node_filter_config=node_filter_config, edge_filter_config=edge_filter_config,
    )

    print("\n=== Best config selection ===")
    best_config_df = select_best_config(
        eval_df, baseline_edge_jaccard=baseline_edge_jaccard, baseline_precision=baseline_precision,
    )

    submission_df = pd.DataFrame(columns=SUBMISSION_COLUMNS)
    if generate_submission and len(best_config_df):
        print("\n=== Submission generation (all test datasets, best tracklet-precision config) ===")
        row = best_config_df.iloc[0]
        tracklet_filter_config = {
            "min_tracklet_length": int(row["min_tracklet_length"]),
            "min_mean_node_score": float(row["min_mean_node_score"]),
            "min_mean_edge_score_quantile": float(row["min_mean_edge_score_quantile"]),
            "max_mean_link_distance_um": float(row["max_mean_link_distance_um"]),
            "max_smoothness_error_um": float(row["max_smoothness_error_um"]),
            "keep_top_k_tracklets_per_sample": int(row["keep_top_k_tracklets_per_sample"]),
        }
        node_policy = str(row["node_policy"])
        submission_df = generate_and_save_submission(
            detector_config, node_filter_config, edge_filter_config, tracklet_filter_config, node_policy,
            out_path=submission_path,
        )

    return {"eval_df": eval_df, "best_config_df": best_config_df, "submission_df": submission_df}


run_tracklet_precision_pipeline(n_eval_samples=3)
