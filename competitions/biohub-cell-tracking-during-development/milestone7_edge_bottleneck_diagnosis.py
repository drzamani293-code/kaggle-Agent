"""
Biohub - Cell Tracking During Development
Milestone 7: detector/node-stage recall and edge-quality bottleneck diagnosis.

Milestone 6D widened the tracklet-level filter grid (4,608 rows on real
Kaggle data) and found NO improvement over 6C: best edge_jaccard=0.010667
(identical to 6C), edge_TP never exceeded 4, and even the 72 configs that
recovered recall >= 0.05 only reached edge_jaccard ~0.003761 (TP=9,
FP=2224). Conclusion: widening the tracklet filter alone cannot fix this -
the true edges are being lost somewhere upstream, before tracklet
filtering ever gets a chance to keep or drop them.

This module does NOT build another submission and does NOT run a sweep.
It traces every GT edge through each pipeline stage individually, on the
first 3 training samples, to find exactly where true edges stop being
recoverable:

  1. raw_node_coverage       - are BOTH endpoints even detected by the raw
                                 D_boundary_aware_3d_local_max detector?
  2. filtered_node_coverage  - do both endpoints survive node filtering
                                 (conservative_tracker.py's best config)?
  3. candidate_edge_coverage - does the per-frame Hungarian assignment
                                 actually pair the two matched predictions
                                 together, within max_link_distance_um?
  4. mnn_edge_coverage       - is that pairing also mutual-nearest-neighbor?
  5. capped_edge_coverage    - does that pairing survive the per-frame
                                 top-`max_edges_per_frame` cap by edge_score?

Detection and node/edge-filter configs are fixed at the same real-data
best configs used since Milestone 6B/6C/6D - nothing is swept here. For
each analyzed sample, GT node count, and coverage/loss at every stage are
printed, and every individually lost GT edge is recorded with the exact
stage it was lost at. An aggregate loss-by-stage summary across all
analyzed samples maps the biggest bottleneck onto one of five candidate
next steps:

  A) less aggressive node filtering       (if raw_node_coverage is fine
                                            but filtered_node_coverage drops)
  B) better edge scoring/assignment       (if candidate_edge_coverage or
                                            capped_edge_coverage drops)
  C) non-MNN edge candidates              (if mnn_edge_coverage drops)
  D) detector improvement                 (if raw_node_coverage itself
                                            is already low)
  E) ML detector                          (same signal as D, at a more
                                            severe/systematic level)

Saves /kaggle/working/milestone7_stage_diagnosis.csv (per-sample +
aggregate stage counts/fractions) and /kaggle/working/milestone7_lost_edges.csv
(one row per GT edge that failed to survive to the final stage, with the
exact stage it was lost at). Runtime target: well under 30 minutes (3
samples x fixed-config detection + filtering, no sweep).

No ML yet - this is pure diagnosis.
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


# --------------------------------------------------------------------------- #
# 10. GT node matching (embedded copy of local_metric.py's evaluator)
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


# --------------------------------------------------------------------------- #
# 11. GT edge stage-by-stage recoverability diagnosis
# --------------------------------------------------------------------------- #
NODE_MATCH_MAX_DISTANCE_UM = 7.0

STAGE_NAMES = [
    "gt_total",
    "raw_node_coverage",
    "filtered_node_coverage",
    "candidate_edge_coverage",
    "mnn_edge_coverage",
    "capped_edge_coverage",
]

LOST_EDGES_COLUMNS = [
    "dataset", "gt_source_id", "gt_target_id",
    "raw_node_coverage", "filtered_node_coverage", "candidate_edge_coverage",
    "mnn_edge_coverage", "capped_edge_coverage", "lost_at_stage",
]


def _invert_pred_to_gt(pred_to_gt: dict) -> dict:
    """`pred_node_id -> gt_node_id` becomes `gt_node_id -> pred_node_id`. The
    underlying match is one-to-one within `max_distance_um` (Hungarian per
    timepoint, each row/column used at most once), so inversion is safe.
    """
    return {gt_id: pred_id for pred_id, gt_id in pred_to_gt.items()}


def _edge_pair_set(edges_df: pd.DataFrame) -> set[tuple[int, int]]:
    if len(edges_df) == 0:
        return set()
    return set(zip(edges_df["source_id"].astype(int), edges_df["target_id"].astype(int)))


def _diagnose_edge_bottleneck_core(
    dataset_name: str,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    raw_nodes_df: pd.DataFrame,
    filtered_nodes_df: pd.DataFrame,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Core stage-by-stage diagnosis logic, taking already-computed raw/
    filtered node tables directly (no disk I/O) - this is what makes the
    logic unit-testable on small synthetic fixtures. See
    `diagnose_sample_edge_bottleneck` for the disk-I/O-driving wrapper used
    on real samples.
    """
    gt_edge_pairs = list(zip(gt_edges["source_id"], gt_edges["target_id"]))
    n_gt_nodes = len(gt_nodes)
    n_gt_edges = len(gt_edge_pairs)

    match_raw = match_nodes_by_timepoint(raw_nodes_df, gt_nodes, match_max_distance_um)
    gt_to_raw = _invert_pred_to_gt(match_raw["pred_to_gt"])

    match_filtered = match_nodes_by_timepoint(filtered_nodes_df, gt_nodes, match_max_distance_um)
    gt_to_filtered = _invert_pred_to_gt(match_filtered["pred_to_gt"])

    candidate_edges = compute_candidate_edges(filtered_nodes_df)
    distance_ok_edges = candidate_edges[candidate_edges["distance_um"] <= edge_filter_config["max_link_distance_um"]]
    if edge_filter_config["require_mutual_nearest_neighbor"]:
        mnn_edges = distance_ok_edges[distance_ok_edges["is_mnn"]]
    else:
        mnn_edges = distance_ok_edges
    capped_edges = (
        mnn_edges.sort_values(["t", "edge_score"], ascending=[True, False])
        .groupby("t", sort=False, group_keys=False)
        .head(edge_filter_config["max_edges_per_frame"])
    )

    candidate_pairs = _edge_pair_set(distance_ok_edges)
    mnn_pairs = _edge_pair_set(mnn_edges)
    capped_pairs = _edge_pair_set(capped_edges)

    counts = {name: 0 for name in STAGE_NAMES}
    counts["gt_total"] = n_gt_edges
    lost_rows: list[dict] = []

    for gu, gv in gt_edge_pairs:
        raw_ok = gu in gt_to_raw and gv in gt_to_raw
        filtered_ok = gu in gt_to_filtered and gv in gt_to_filtered
        candidate_ok = mnn_ok = capped_ok = False
        if filtered_ok:
            pu, pv = gt_to_filtered[gu], gt_to_filtered[gv]
            candidate_ok = (pu, pv) in candidate_pairs
            mnn_ok = (pu, pv) in mnn_pairs
            capped_ok = (pu, pv) in capped_pairs

        if raw_ok:
            counts["raw_node_coverage"] += 1
        if filtered_ok:
            counts["filtered_node_coverage"] += 1
        if candidate_ok:
            counts["candidate_edge_coverage"] += 1
        if mnn_ok:
            counts["mnn_edge_coverage"] += 1
        if capped_ok:
            counts["capped_edge_coverage"] += 1

        if not capped_ok:
            if not raw_ok:
                lost_at = "raw_node_coverage"
            elif not filtered_ok:
                lost_at = "filtered_node_coverage"
            elif not candidate_ok:
                lost_at = "candidate_edge_coverage"
            elif not mnn_ok:
                lost_at = "mnn_edge_coverage"
            else:
                lost_at = "capped_edge_coverage"
            lost_rows.append({
                "dataset": dataset_name, "gt_source_id": gu, "gt_target_id": gv,
                "raw_node_coverage": raw_ok, "filtered_node_coverage": filtered_ok,
                "candidate_edge_coverage": candidate_ok, "mnn_edge_coverage": mnn_ok,
                "capped_edge_coverage": capped_ok, "lost_at_stage": lost_at,
            })

    lost_df = pd.DataFrame(lost_rows, columns=LOST_EDGES_COLUMNS)
    return {
        "dataset": dataset_name,
        "gt_node_count": n_gt_nodes,
        "gt_edge_count": n_gt_edges,
        "counts": counts,
        "lost_df": lost_df,
    }


def diagnose_sample_edge_bottleneck(
    sample_zarr_dir: Path,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_config: dict = BEST_CONSERVATIVE_NODE_FILTER_CONFIG,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Reads GT + runs raw detection + node filtering for one sample (fixed
    at the same real-data best configs used since Milestone 6B/6C/6D - not
    swept here), then hands off to `_diagnose_edge_bottleneck_core`.
    """
    dataset_name = sample_zarr_dir.stem
    gt_nodes, gt_edges = read_geff(sample_zarr_dir)

    raw_nodes_df, _ = detect_all_raw_nodes_for_sample(sample_zarr_dir, detector_config, node_id_start=0)
    filtered_nodes_df = apply_node_filter(
        raw_nodes_df,
        node_filter_config["max_nodes_per_timepoint"],
        node_filter_config["score_quantile_per_timepoint"],
        node_filter_config["absolute_score_threshold"],
    )[["node_id", "t", "z", "y", "x", "score"]].reset_index(drop=True)

    return _diagnose_edge_bottleneck_core(
        dataset_name, gt_nodes, gt_edges, raw_nodes_df, filtered_nodes_df,
        edge_filter_config, match_max_distance_um,
    )


# --------------------------------------------------------------------------- #
# 12. Per-sample printing + cross-sample aggregation/recommendation
# --------------------------------------------------------------------------- #
def print_sample_diagnosis(result: dict) -> None:
    counts = result["counts"]
    n_gt_edges = counts["gt_total"]

    def frac(n: int) -> float:
        return (n / n_gt_edges) if n_gt_edges else float("nan")

    print(f"\n--- {result['dataset']} ---")
    print(f"  GT nodes: {result['gt_node_count']}")
    print(f"  GT edges: {n_gt_edges}")
    print(f"  raw detector node coverage:       {counts['raw_node_coverage']} / {n_gt_edges}  ({frac(counts['raw_node_coverage']):.4f})")
    print(f"  filtered node coverage:           {counts['filtered_node_coverage']} / {n_gt_edges}  ({frac(counts['filtered_node_coverage']):.4f})")
    print(f"  candidate edge coverage:          {counts['candidate_edge_coverage']} / {n_gt_edges}  ({frac(counts['candidate_edge_coverage']):.4f})")
    print(f"  mutual-nearest-neighbor coverage: {counts['mnn_edge_coverage']} / {n_gt_edges}  ({frac(counts['mnn_edge_coverage']):.4f})")
    print(f"  capped (max_edges_per_frame) coverage: {counts['capped_edge_coverage']} / {n_gt_edges}  ({frac(counts['capped_edge_coverage']):.4f})")

    lost_at_detector = n_gt_edges - counts["raw_node_coverage"]
    lost_at_node_filter = counts["raw_node_coverage"] - counts["filtered_node_coverage"]
    lost_at_candidate = counts["filtered_node_coverage"] - counts["candidate_edge_coverage"]
    lost_at_mnn = counts["candidate_edge_coverage"] - counts["mnn_edge_coverage"]
    lost_at_cap = counts["mnn_edge_coverage"] - counts["capped_edge_coverage"]
    print(
        f"  lost GT edges at each stage: detector={lost_at_detector}, node_filter={lost_at_node_filter}, "
        f"candidate_construction={lost_at_candidate}, mnn_filter={lost_at_mnn}, per_frame_cap={lost_at_cap}"
    )


STAGE_COLUMNS_FOR_CSV = [c for c in STAGE_NAMES if c != "gt_total"]


def _stage_row(dataset_name: str, gt_node_count: int, counts: dict) -> dict:
    gt_total = counts["gt_total"]
    row = {"dataset": dataset_name, "gt_node_count": gt_node_count, "gt_edge_count": gt_total}
    for name in STAGE_COLUMNS_FOR_CSV:
        row[f"{name}_count"] = counts[name]
        row[f"{name}_frac"] = (counts[name] / gt_total) if gt_total else float("nan")
    return row


def aggregate_and_recommend(results: list[dict]) -> pd.DataFrame:
    """Builds the per-sample + TOTAL stage-diagnosis table and prints the
    aggregate loss-by-stage breakdown with a recommendation for which of
    Milestone 7's five candidate next steps (A-E) addresses the biggest
    single bottleneck.
    """
    rows = [_stage_row(r["dataset"], r["gt_node_count"], r["counts"]) for r in results]

    total_counts = {name: sum(r["counts"][name] for r in results) for name in STAGE_NAMES}
    total_gt_nodes = sum(r["gt_node_count"] for r in results)
    rows.append(_stage_row("TOTAL", total_gt_nodes, total_counts))
    stage_df = pd.DataFrame(rows)

    lost_detector = total_counts["gt_total"] - total_counts["raw_node_coverage"]
    lost_node_filter = total_counts["raw_node_coverage"] - total_counts["filtered_node_coverage"]
    lost_candidate = total_counts["filtered_node_coverage"] - total_counts["candidate_edge_coverage"]
    lost_mnn = total_counts["candidate_edge_coverage"] - total_counts["mnn_edge_coverage"]
    lost_cap = total_counts["mnn_edge_coverage"] - total_counts["capped_edge_coverage"]

    losses = {
        "D_or_E_detector_improvement (raw detector misses GT nodes)": lost_detector,
        "A_less_aggressive_node_filtering (node filter drops detected nodes)": lost_node_filter,
        "B_better_edge_scoring_or_assignment (Hungarian picks the wrong partner)": lost_candidate,
        "C_non_mnn_edge_candidates (mutual-nearest-neighbor rejects a correct pairing)": lost_mnn,
        "B_better_edge_scoring_or_ranking (true edge outranked under the per-frame cap)": lost_cap,
    }

    print("\n=== Aggregate loss by stage (across analyzed samples) ===")
    print(f"  GT edges total:                                        {total_counts['gt_total']}")
    print(f"  lost to detector miss (raw node coverage):             {lost_detector}")
    print(f"  lost to node filtering (too aggressive):               {lost_node_filter}")
    print(f"  lost to candidate-edge assignment (wrong Hungarian pairing): {lost_candidate}")
    print(f"  lost to mutual-nearest-neighbor requirement:           {lost_mnn}")
    print(f"  lost to per-frame top-K cap (true edge outranked):     {lost_cap}")

    ranked = sorted(losses.items(), key=lambda kv: kv[1], reverse=True)
    print("\nFull ranking (most damaging bottleneck first):")
    for name, n in ranked:
        print(f"  {n:6d} lost GT edge(s)  ->  {name}")

    biggest_name, biggest_n = ranked[0]
    print(f"\n=== Recommendation: the biggest single bottleneck is '{biggest_name}' ({biggest_n} lost GT edge(s)) ===")

    return stage_df


# --------------------------------------------------------------------------- #
# 13. Full Milestone 7 driver (first n_samples train samples, no sweep)
# --------------------------------------------------------------------------- #
def run_milestone7_diagnosis(
    n_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_config: dict = BEST_CONSERVATIVE_NODE_FILTER_CONFIG,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    stage_csv_path: str = "/kaggle/working/milestone7_stage_diagnosis.csv",
    lost_edges_csv_path: str = "/kaggle/working/milestone7_lost_edges.csv",
) -> dict:
    """Runs the stage-by-stage GT-edge bottleneck diagnosis on the first
    `n_samples` train samples (fixed configs, no sweep), prints a full
    per-sample + aggregate report, and saves both CSVs. Builds no
    submission. Runtime is bounded by `n_samples` x (raw detection + node
    filtering + one Hungarian pass per frame pair) - the same per-sample
    cost as every prior milestone's detection step, comfortably under 30
    minutes for 3 samples.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return {"stage_df": pd.DataFrame(), "lost_edges_df": pd.DataFrame(columns=LOST_EDGES_COLUMNS)}

    samples = train_dirs[:n_samples]
    print(f"Diagnosing edge-recoverability bottleneck on {len(samples)} train sample(s) (fixed configs, no sweep)...")

    results = []
    for sample_dir in samples:
        t0 = time.time()
        result = diagnose_sample_edge_bottleneck(sample_dir, detector_config, node_filter_config, edge_filter_config)
        results.append(result)
        print_sample_diagnosis(result)
        print(f"  ({time.time() - t0:.1f}s for {sample_dir.stem})")

    stage_df = aggregate_and_recommend(results)
    lost_edges_df = pd.concat([r["lost_df"] for r in results], ignore_index=True) if results else pd.DataFrame(columns=LOST_EDGES_COLUMNS)

    stage_out = Path(stage_csv_path)
    stage_out.parent.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(stage_out, index=False)
    print(f"\nSaved stage diagnosis ({len(stage_df)} row(s)) to {stage_out}")

    lost_out = Path(lost_edges_csv_path)
    lost_out.parent.mkdir(parents=True, exist_ok=True)
    lost_edges_df.to_csv(lost_out, index=False)
    print(f"Saved lost-edge detail ({len(lost_edges_df)} row(s)) to {lost_out}")

    return {"stage_df": stage_df, "lost_edges_df": lost_edges_df, "results": results}


# --------------------------------------------------------------------------- #
# 14. Tests
# --------------------------------------------------------------------------- #
def run_milestone7_tests() -> None:
    """Correctness tests for the stage-by-stage diagnosis logic, using
    small synthetic fixtures (no disk I/O) so the exact loss-attribution
    at each stage can be verified by hand.
    """
    # Test 1: _invert_pred_to_gt correctness (including the empty case).
    assert _invert_pred_to_gt({}) == {}
    assert _invert_pred_to_gt({10: 100, 11: 101}) == {100: 10, 101: 11}

    # Test 2: _edge_pair_set correctness (including the empty-DataFrame case).
    empty_edges = pd.DataFrame(columns=["source_id", "target_id"])
    assert _edge_pair_set(empty_edges) == set()
    some_edges = pd.DataFrame({"source_id": [1, 2], "target_id": [3, 4]})
    assert _edge_pair_set(some_edges) == {(1, 3), (2, 4)}

    # Test 3: a well-separated 2-pair fixture where both GT edges survive
    # every stage (raw detection, node filter, Hungarian assignment, MNN,
    # and the per-frame cap all agree on the same pairing).
    gt_nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.0, 10.0, 200.0, 210.0], "x": [0.0] * 4,
    })
    gt_edges = pd.DataFrame({"source_id": [0, 2], "target_id": [1, 3]})
    raw_nodes_df = pd.DataFrame({
        "node_id": [100, 101, 102, 103], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.1, 10.1, 200.1, 210.1], "x": [0.0] * 4,
        "score": [0.9, 0.9, 0.9, 0.9],
    })
    result = _diagnose_edge_bottleneck_core(
        "clean_fixture", gt_nodes, gt_edges, raw_nodes_df, raw_nodes_df,
        BEST_CONSERVATIVE_EDGE_FILTER_CONFIG, match_max_distance_um=7.0,
    )
    assert result["counts"]["gt_total"] == 2
    assert result["counts"]["raw_node_coverage"] == 2
    assert result["counts"]["filtered_node_coverage"] == 2
    assert result["counts"]["candidate_edge_coverage"] == 2
    assert result["counts"]["mnn_edge_coverage"] == 2
    assert result["counts"]["capped_edge_coverage"] == 2
    assert len(result["lost_df"]) == 0, "a clean well-separated fixture should lose nothing"

    # Test 4: node-filter loss. The raw detector finds all 4 nodes (so
    # raw_node_coverage is full), but the filtered-node table simulates
    # node filtering dropping node 103 (GT node 3's match) - GT edge
    # (2, 3) must be marked lost specifically at filtered_node_coverage,
    # not at raw_node_coverage.
    filtered_nodes_df = raw_nodes_df[raw_nodes_df["node_id"] != 103].reset_index(drop=True)
    result_nf = _diagnose_edge_bottleneck_core(
        "node_filter_loss_fixture", gt_nodes, gt_edges, raw_nodes_df, filtered_nodes_df,
        BEST_CONSERVATIVE_EDGE_FILTER_CONFIG, match_max_distance_um=7.0,
    )
    assert result_nf["counts"]["raw_node_coverage"] == 2, "raw detection still finds both endpoints of both edges"
    assert result_nf["counts"]["filtered_node_coverage"] == 1, "only the (0,1) edge should survive node filtering"
    lost_rows = result_nf["lost_df"]
    assert len(lost_rows) == 1
    lost_row = lost_rows.iloc[0]
    assert lost_row["gt_source_id"] == 2 and lost_row["gt_target_id"] == 3
    assert lost_row["lost_at_stage"] == "filtered_node_coverage"
    assert bool(lost_row["raw_node_coverage"]) is True and bool(lost_row["filtered_node_coverage"]) is False

    # Test 5: per-frame-cap loss. Both GT edges' endpoints are detected and
    # filtered fine, and both pairings are the correct (distance-ok, MNN)
    # Hungarian assignment - but max_edges_per_frame=1 forces only the
    # higher-edge_score pairing to survive the cap, so the other must be
    # marked lost specifically at capped_edge_coverage.
    tight_cap_config = dict(BEST_CONSERVATIVE_EDGE_FILTER_CONFIG)
    tight_cap_config["max_edges_per_frame"] = 1
    result_cap = _diagnose_edge_bottleneck_core(
        "cap_loss_fixture", gt_nodes, gt_edges, raw_nodes_df, raw_nodes_df,
        tight_cap_config, match_max_distance_um=7.0,
    )
    assert result_cap["counts"]["mnn_edge_coverage"] == 2, "both pairings are still MNN before the cap is applied"
    assert result_cap["counts"]["capped_edge_coverage"] == 1, "only 1 of the 2 pairings can survive max_edges_per_frame=1"
    cap_lost_rows = result_cap["lost_df"]
    assert len(cap_lost_rows) == 1
    assert cap_lost_rows.iloc[0]["lost_at_stage"] == "capped_edge_coverage"

    # Test 6: aggregate_and_recommend arithmetic + biggest-bottleneck
    # selection on a hand-built 2-sample fixture.
    fake_results = [
        {
            "dataset": "s0", "gt_node_count": 10, "gt_edge_count": 10,
            "counts": {"gt_total": 10, "raw_node_coverage": 8, "filtered_node_coverage": 3,
                       "candidate_edge_coverage": 3, "mnn_edge_coverage": 3, "capped_edge_coverage": 3},
        },
        {
            "dataset": "s1", "gt_node_count": 10, "gt_edge_count": 10,
            "counts": {"gt_total": 10, "raw_node_coverage": 9, "filtered_node_coverage": 4,
                       "candidate_edge_coverage": 4, "mnn_edge_coverage": 4, "capped_edge_coverage": 4},
        },
    ]
    stage_df = aggregate_and_recommend(fake_results)
    assert len(stage_df) == 3, "2 per-sample rows + 1 TOTAL row"
    total_row = stage_df[stage_df["dataset"] == "TOTAL"].iloc[0]
    assert total_row["gt_edge_count"] == 20
    assert total_row["raw_node_coverage_count"] == 17
    assert total_row["filtered_node_coverage_count"] == 7
    # Biggest loss here is node filtering: (17 - 7) = 10 lost, vs detector's
    # (20 - 17) = 3 lost, and zero loss at every downstream edge-linking stage.
    assert total_row["filtered_node_coverage_count"] < total_row["raw_node_coverage_count"]

    print("All milestone7_edge_bottleneck_diagnosis tests passed (6/6).")


# --------------------------------------------------------------------------- #
# 15. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone7_pipeline(n_samples: int = 3) -> dict:
    """Runs the tests, then the full Milestone 7 stage-by-stage GT-edge
    bottleneck diagnosis on the first `n_samples` train samples. Builds no
    submission, runs no sweep. Never calls any Kaggle submission API.
    """
    print("=== Self-test: stage-by-stage bottleneck diagnosis unit tests ===")
    run_milestone7_tests()

    print("\n=== Milestone 7: GT edge stage-by-stage bottleneck diagnosis ===")
    return run_milestone7_diagnosis(n_samples=n_samples)


if __name__ == "__main__":
    run_milestone7_pipeline(n_samples=3)
