"""
Biohub - Cell Tracking During Development
Milestone 6B: conservative (precision-first) tracking baseline.

The simple tracker (simple_tracker.py) ran on real Kaggle data with a
disastrous edge score: edge_jaccard=0.00045 (edge_TP=44, edge_FP=97594,
edge_FN=125). Node recall was reasonable (~74%, matching the detector's
known behavior), but the tracker linked essentially every one of the
~780 predicted nodes per timepoint - the vast majority of them low-
confidence noise detections - producing a firehose of false-positive
edges that swamped the few genuine ones.

This module adds three layers of conservative filtering BEFORE evaluating
any linking at all:

  1. Node filtering - keep only the highest-confidence detections per
     timepoint (absolute score threshold, per-timepoint score quantile,
     and a hard cap), instead of accepting hundreds of borderline peaks.
  2. Conservative edge filtering - score candidate edges by
     source_score * target_score / (1 + distance_um), optionally require
     mutual-nearest-neighbor consistency (i does not just find j via
     Hungarian - j must ALSO have i as its own nearest neighbor), reject
     anything past max_link_distance_um, and keep only the top edges per
     frame by edge_score.
  3. Tracklet filtering - optionally drop nodes that end up with no
     surviving edges at all (isolated low-confidence detections),
     instead of reporting them as spurious unlinked "tracks".

Detection (the expensive step) is cached once per sample. For a FIXED
node-filter combo, the Hungarian assignment + mutual-nearest-neighbor
check between each consecutive frame pair is also computed once and
reused across every downstream edge-filter x tracklet-filter combo -
the same efficient-caching principle used in every prior calibration
sweep in this project (only the acceptance thresholds/ranking differ
downstream, not the underlying assignment problem).

No ML yet - classical detection + conservative, precision-first linking.
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

# Defensive cap on the per-timepoint candidate pool AFTER NMS-dedup, BEFORE
# the swept node-filter (max_nodes_per_timepoint <= 500) ever sees it -
# just bounds worst-case memory/compute, far above anything the sweep would
# ever select.
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
    call) rather than a per-point loop - see gt_failure_analysis.py for the
    profiling that motivated this (3-8x faster, identical output).
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
# 6. Detector: D_boundary_aware_3d_local_max ONLY (unchanged since Milestone
#    6/6B's conclusion - not revisiting centroid/component detector work)
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
    physical NMS (`detector_config["min_distance_um"]`), and assigns
    globally-increasing node_id values - WITHOUT applying the new,
    aggressive node-filter sweep (max_nodes/score_quantile/abs_threshold)
    yet. node_id stays stable across every node-filter combo tested later,
    since filtering only ever selects a subset of THIS same table.
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
# 8. Node filtering
# --------------------------------------------------------------------------- #
DEFAULT_NODE_FILTER_MAX_NODES_VALUES = (25, 50, 75, 100, 150, 200, 300, 500)
DEFAULT_NODE_FILTER_SCORE_QUANTILE_VALUES = (0.90, 0.95, 0.97, 0.98, 0.99)
DEFAULT_NODE_FILTER_ABS_THRESHOLD_VALUES = (0.05, 0.08, 0.10, 0.15, 0.20)


def _apply_node_filter_one_timepoint(
    nodes_t_df: pd.DataFrame, max_nodes: int, score_quantile: float, abs_threshold: float,
) -> pd.DataFrame:
    """Absolute threshold -> per-timepoint score quantile -> hard cap, all on
    ONE timepoint's already-NMS-deduped candidate pool (already sorted by
    score descending is not assumed - re-sorted here defensively).
    """
    filtered = nodes_t_df[nodes_t_df["score"] >= abs_threshold]
    if len(filtered) == 0:
        return filtered
    q_value = filtered["score"].quantile(score_quantile)
    filtered = filtered[filtered["score"] >= q_value]
    if len(filtered) == 0:
        return filtered
    return filtered.sort_values("score", ascending=False).iloc[:max_nodes]


def apply_node_filter(nodes_df: pd.DataFrame, max_nodes: int, score_quantile: float, abs_threshold: float) -> pd.DataFrame:
    """Applies `_apply_node_filter_one_timepoint` independently per
    timepoint across a full sample's raw node table.
    """
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
# 9. Conservative edge filtering
# --------------------------------------------------------------------------- #
DEFAULT_EDGE_MAX_LINK_DISTANCE_VALUES = (2, 3, 4, 5, 7)
DEFAULT_EDGE_REQUIRE_MNN_VALUES = (True, False)
DEFAULT_EDGE_MAX_EDGES_PER_FRAME_VALUES = (5, 10, 25, 50, 100, 200)

CANDIDATE_EDGE_COLUMNS = ["t", "source_id", "target_id", "source_score", "target_score", "distance_um", "is_mnn", "edge_score"]


def compute_candidate_edges(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """For a FIXED (already node-filtered) sample-wide node table, computes
    the Hungarian assignment between every pair of consecutive timepoints
    ONCE, plus a mutual-nearest-neighbor flag derived from the SAME cost
    matrix (i's nearest nxt-column is j, AND j's nearest cur-row is i -
    both checked directly via argmin over the full cost matrix, independent
    of the Hungarian solution itself). This one table is then reused by
    every downstream edge-filter combo (max_link_distance_um, require_mnn,
    max_edges_per_frame) - only the acceptance/ranking differs, not the
    underlying assignment.
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
    keep only the top `max_edges_per_frame` edges per timepoint, ranked by
    edge_score = source_score * target_score / (1 + distance_um).
    """
    edge_columns = ["source_id", "target_id"]
    if len(candidate_edges_df) == 0:
        return pd.DataFrame(columns=edge_columns)

    filtered = candidate_edges_df[candidate_edges_df["distance_um"] <= max_link_distance_um]
    if require_mnn:
        filtered = filtered[filtered["is_mnn"]]
    if len(filtered) == 0:
        return pd.DataFrame(columns=edge_columns)

    filtered = filtered.sort_values(["t", "edge_score"], ascending=[True, False])
    filtered = filtered.groupby("t", sort=False, group_keys=False).head(max_edges_per_frame)
    return filtered[edge_columns].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 10. Tracklet filtering
# --------------------------------------------------------------------------- #
def apply_tracklet_filter(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, keep_only_linked_nodes: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """If `keep_only_linked_nodes`, drops any node that ends up with no
    surviving edge at all (an isolated low-confidence detection) - edges
    themselves are untouched (an edge, by definition, only ever connects
    two already-linked nodes).
    """
    if not keep_only_linked_nodes:
        return pred_nodes, pred_edges
    if len(pred_edges) == 0:
        return pred_nodes.iloc[0:0], pred_edges
    linked_ids = set(pred_edges["source_id"]) | set(pred_edges["target_id"])
    kept = pred_nodes[pred_nodes["node_id"].isin(linked_ids)]
    return kept.reset_index(drop=True), pred_edges


# --------------------------------------------------------------------------- #
# 11. Local evaluation (embedded copy of local_metric.py's evaluator, so this
#     module stays self-contained/paste-ready with no cross-file import)
# --------------------------------------------------------------------------- #
def match_nodes_by_timepoint(pred_nodes: pd.DataFrame, gt_nodes: pd.DataFrame, max_distance_um: float = 7.0) -> dict:
    """Match predicted nodes to GT nodes independently per timepoint via the
    Hungarian algorithm on physical (um) distance, rejecting matches farther
    than `max_distance_um`. See local_metric.py for the original/canonical
    implementation - identical logic, copied here to avoid a cross-file import.
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
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, gt_nodes: pd.DataFrame, gt_edges: pd.DataFrame,
    max_distance_um: float = 7.0, match: dict | None = None,
) -> dict:
    """A predicted edge is a true positive if both endpoints map to GT nodes
    and that (source, target) pair is an actual GT edge.
    """
    if match is None:
        match = match_nodes_by_timepoint(pred_nodes, gt_nodes, max_distance_um)
    pred_to_gt = match["pred_to_gt"]

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


def evaluate_pred_against_gt(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, gt_nodes: pd.DataFrame, gt_edges: pd.DataFrame,
    max_distance_um: float = 7.0,
) -> dict:
    """Same metrics as local_metric.py's evaluate_sample, but takes GT
    directly instead of re-reading it from disk - this sweep calls this
    function tens of thousands of times, so GT is read from disk exactly
    ONCE per sample up front and reused, rather than once per combo.
    """
    match = match_nodes_by_timepoint(pred_nodes, gt_nodes, max_distance_um)
    edge_result = compute_edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, max_distance_um, match=match)
    return {
        "node_matches": len(match["pred_to_gt"]),
        "unmatched_pred_nodes": len(match["unmatched_pred_nodes"]),
        "unmatched_gt_nodes": len(match["unmatched_gt_nodes"]),
        "edge_TP": edge_result["edge_TP"],
        "edge_FP": edge_result["edge_FP"],
        "edge_FN": edge_result["edge_FN"],
        "edge_jaccard": edge_result["edge_jaccard"],
    }


# --------------------------------------------------------------------------- #
# 12. Full node/edge/tracklet-filter sweep
# --------------------------------------------------------------------------- #
DEFAULT_TRACKLET_KEEP_ONLY_LINKED_VALUES = (False, True)

EVAL_COLUMNS = [
    "max_nodes_per_timepoint", "score_quantile_per_timepoint", "absolute_score_threshold",
    "max_link_distance_um", "require_mutual_nearest_neighbor", "max_edges_per_frame",
    "keep_only_linked_nodes",
    "edge_jaccard", "edge_TP", "edge_FP", "edge_FN",
    "precision", "recall",
    "node_matches", "unmatched_gt_nodes", "unmatched_pred_nodes",
    "avg_pred_nodes_per_timepoint", "avg_edges_per_frame", "runtime_seconds",
]


def run_conservative_tracker_eval(
    n_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_max_nodes_values: Sequence[int] = DEFAULT_NODE_FILTER_MAX_NODES_VALUES,
    node_filter_score_quantile_values: Sequence[float] = DEFAULT_NODE_FILTER_SCORE_QUANTILE_VALUES,
    node_filter_abs_threshold_values: Sequence[float] = DEFAULT_NODE_FILTER_ABS_THRESHOLD_VALUES,
    edge_max_link_distance_values: Sequence[float] = DEFAULT_EDGE_MAX_LINK_DISTANCE_VALUES,
    edge_require_mnn_values: Sequence[bool] = DEFAULT_EDGE_REQUIRE_MNN_VALUES,
    edge_max_edges_per_frame_values: Sequence[int] = DEFAULT_EDGE_MAX_EDGES_PER_FRAME_VALUES,
    tracklet_keep_only_linked_values: Sequence[bool] = DEFAULT_TRACKLET_KEEP_ONLY_LINKED_VALUES,
    match_max_distance_um: float = 7.0,
    csv_path: str = "/kaggle/working/conservative_tracker_eval.csv",
    progress_every: int = 20,
) -> pd.DataFrame:
    """Full grid sweep over node filtering x conservative edge filtering x
    tracklet filtering. Detection runs ONCE per sample (cached); for each
    node-filter combo, the Hungarian assignment + mutual-nearest-neighbor
    check between consecutive frames is computed ONCE (cached) and reused
    across every edge-filter x tracklet-filter combo, since those only
    change acceptance thresholds/ranking, not the underlying assignment.
    GT is read from disk once per sample, not once per combo.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=EVAL_COLUMNS)

    samples = train_dirs[:n_samples]
    n_node_combos = len(node_filter_max_nodes_values) * len(node_filter_score_quantile_values) * len(node_filter_abs_threshold_values)
    n_edge_combos = len(edge_max_link_distance_values) * len(edge_require_mnn_values) * len(edge_max_edges_per_frame_values)
    n_total = n_node_combos * n_edge_combos * len(tracklet_keep_only_linked_values)
    print(f"Sweeping {n_node_combos} node-filter combo(s) x {n_edge_combos} edge-filter combo(s) "
          f"x {len(tracklet_keep_only_linked_values)} tracklet option(s) = {n_total} total row(s).")

    print(f"Detecting raw (NMS-deduped) nodes for {len(samples)} sample(s) (all timepoints)...")
    raw_nodes_by_sample: dict[Path, pd.DataFrame] = {}
    gt_by_sample: dict[Path, tuple[pd.DataFrame, pd.DataFrame]] = {}
    total_timepoints = 0
    total_frame_pairs = 0
    for sample_dir in samples:
        t0 = time.time()
        nodes_df, _ = detect_all_raw_nodes_for_sample(sample_dir, detector_config, node_id_start=0)
        raw_nodes_by_sample[sample_dir] = nodes_df
        gt_by_sample[sample_dir] = read_geff(sample_dir)
        T = get_sample_timepoint_count(sample_dir)
        total_timepoints += T
        total_frame_pairs += max(0, T - 1)
        print(f"  {sample_dir.stem}: {len(nodes_df)} raw node(s) detected in {time.time() - t0:.1f}s")

    rows: list[dict] = []
    node_combo_idx = 0
    sweep_t0 = time.time()

    for max_nodes in node_filter_max_nodes_values:
        for score_quantile in node_filter_score_quantile_values:
            for abs_threshold in node_filter_abs_threshold_values:
                node_combo_idx += 1

                filtered_nodes_by_sample = {
                    sample_dir: apply_node_filter(nodes_df, max_nodes, score_quantile, abs_threshold)
                    for sample_dir, nodes_df in raw_nodes_by_sample.items()
                }
                candidate_edges_by_sample = {
                    sample_dir: compute_candidate_edges(filtered_nodes_by_sample[sample_dir])
                    for sample_dir in samples
                }
                total_pred_nodes = sum(len(df) for df in filtered_nodes_by_sample.values())
                avg_pred_nodes_per_timepoint = (total_pred_nodes / total_timepoints) if total_timepoints else float("nan")

                for max_link_distance_um in edge_max_link_distance_values:
                    for require_mnn in edge_require_mnn_values:
                        for max_edges_per_frame in edge_max_edges_per_frame_values:
                            row_t0 = time.time()
                            edges_by_sample = {
                                sample_dir: apply_edge_filter(
                                    candidate_edges_by_sample[sample_dir], max_link_distance_um, require_mnn, max_edges_per_frame
                                )
                                for sample_dir in samples
                            }
                            total_edges = sum(len(e) for e in edges_by_sample.values())
                            avg_edges_per_frame = (total_edges / total_frame_pairs) if total_frame_pairs else float("nan")

                            for keep_only_linked_nodes in tracklet_keep_only_linked_values:
                                edge_TP = edge_FP = edge_FN = 0
                                node_matches = unmatched_gt = unmatched_pred = 0

                                for sample_dir in samples:
                                    nodes_df = filtered_nodes_by_sample[sample_dir]
                                    edges_df = edges_by_sample[sample_dir]
                                    pred_nodes, pred_edges = apply_tracklet_filter(
                                        nodes_df[["node_id", "t", "z", "y", "x"]], edges_df, keep_only_linked_nodes
                                    )
                                    gt_nodes, gt_edges = gt_by_sample[sample_dir]
                                    result = evaluate_pred_against_gt(pred_nodes, pred_edges, gt_nodes, gt_edges, match_max_distance_um)

                                    edge_TP += result["edge_TP"]
                                    edge_FP += result["edge_FP"]
                                    edge_FN += result["edge_FN"]
                                    node_matches += result["node_matches"]
                                    unmatched_gt += result["unmatched_gt_nodes"]
                                    unmatched_pred += result["unmatched_pred_nodes"]

                                denom = edge_TP + edge_FP + edge_FN
                                edge_jaccard = edge_TP / denom if denom > 0 else 1.0
                                precision = (edge_TP / (edge_TP + edge_FP)) if (edge_TP + edge_FP) > 0 else (1.0 if edge_TP == 0 else 0.0)
                                recall = (edge_TP / (edge_TP + edge_FN)) if (edge_TP + edge_FN) > 0 else (1.0 if edge_TP == 0 else 0.0)

                                rows.append({
                                    "max_nodes_per_timepoint": max_nodes,
                                    "score_quantile_per_timepoint": score_quantile,
                                    "absolute_score_threshold": abs_threshold,
                                    "max_link_distance_um": max_link_distance_um,
                                    "require_mutual_nearest_neighbor": require_mnn,
                                    "max_edges_per_frame": max_edges_per_frame,
                                    "keep_only_linked_nodes": keep_only_linked_nodes,
                                    "edge_jaccard": edge_jaccard,
                                    "edge_TP": edge_TP, "edge_FP": edge_FP, "edge_FN": edge_FN,
                                    "precision": precision, "recall": recall,
                                    "node_matches": node_matches,
                                    "unmatched_gt_nodes": unmatched_gt,
                                    "unmatched_pred_nodes": unmatched_pred,
                                    "avg_pred_nodes_per_timepoint": avg_pred_nodes_per_timepoint,
                                    "avg_edges_per_frame": avg_edges_per_frame,
                                    "runtime_seconds": time.time() - row_t0,
                                })

                if node_combo_idx % progress_every == 0 or node_combo_idx == n_node_combos:
                    elapsed = time.time() - sweep_t0
                    eta = elapsed / node_combo_idx * (n_node_combos - node_combo_idx)
                    print(f"  [{node_combo_idx}/{n_node_combos} node-filter combos] "
                          f"{len(rows)} row(s) so far, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    eval_df = pd.DataFrame(rows, columns=EVAL_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_path, index=False)
    print(f"\nSaved conservative tracker eval sweep ({len(eval_df)} row(s)) to {out_path}")
    return eval_df


def select_best_config(
    eval_df: pd.DataFrame,
    baseline_edge_jaccard: float = 0.00045,
    csv_path: str = "/kaggle/working/conservative_tracker_best_config.csv",
) -> pd.DataFrame:
    """Best config = highest edge_jaccard (the harmonic tradeoff between
    precision and recall - directly answers "did we drastically cut
    edge_FP without destroying recall"); ties broken by precision then
    recall descending, matching the "precision first, then recall"
    priority from the milestone's goal.
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
    print(f"\nBest config: max_nodes_per_timepoint={row['max_nodes_per_timepoint']} "
          f"score_quantile_per_timepoint={row['score_quantile_per_timepoint']} "
          f"absolute_score_threshold={row['absolute_score_threshold']} "
          f"max_link_distance_um={row['max_link_distance_um']} "
          f"require_mutual_nearest_neighbor={row['require_mutual_nearest_neighbor']} "
          f"max_edges_per_frame={row['max_edges_per_frame']} "
          f"keep_only_linked_nodes={row['keep_only_linked_nodes']}")
    print(f"  edge_jaccard={row['edge_jaccard']:.5f} (baseline was {baseline_edge_jaccard}) "
          f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
          f"edge_FP={row['edge_FP']} (goal: drastically reduced)")
    print(f"Saved best config to {out_path}")
    return best


# --------------------------------------------------------------------------- #
# 13. Submission generation (uses the best conservative config)
# --------------------------------------------------------------------------- #
SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def build_conservative_sample_submission_rows(
    sample_zarr_dir: Path,
    detector_config: dict,
    node_filter_config: dict,
    edge_filter_config: dict,
    keep_only_linked_nodes: bool,
    next_id: int,
    next_node_id: int,
) -> tuple[list[dict], int, int]:
    """Runs the full conservative pipeline (detect -> node filter ->
    candidate edges -> edge filter -> tracklet filter) for one sample,
    returning its node/edge rows with `id`/`node_id` threaded through so
    they stay globally consecutive/unique across every dataset in the
    final submission.
    """
    dataset_name = sample_zarr_dir.stem
    raw_nodes_df, next_node_id = detect_all_raw_nodes_for_sample(
        sample_zarr_dir, detector_config, node_id_start=next_node_id
    )
    filtered_nodes_df = apply_node_filter(
        raw_nodes_df,
        node_filter_config["max_nodes_per_timepoint"],
        node_filter_config["score_quantile_per_timepoint"],
        node_filter_config["absolute_score_threshold"],
    )
    candidate_edges_df = compute_candidate_edges(filtered_nodes_df)
    edges_df = apply_edge_filter(
        candidate_edges_df,
        edge_filter_config["max_link_distance_um"],
        edge_filter_config["require_mutual_nearest_neighbor"],
        edge_filter_config["max_edges_per_frame"],
    )
    pred_nodes, pred_edges = apply_tracklet_filter(
        filtered_nodes_df[["node_id", "t", "z", "y", "x"]], edges_df, keep_only_linked_nodes
    )

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


def build_conservative_submission(
    test_zarr_dirs: Sequence[Path], detector_config: dict, node_filter_config: dict, edge_filter_config: dict,
    keep_only_linked_nodes: bool,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    next_id = 0
    next_node_id = 0
    for sample_dir in test_zarr_dirs:
        t0 = time.time()
        rows, next_id, next_node_id = build_conservative_sample_submission_rows(
            sample_dir, detector_config, node_filter_config, edge_filter_config, keep_only_linked_nodes, next_id, next_node_id
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
    keep_only_linked_nodes: bool,
    out_path: str = "/kaggle/working/submission.csv",
) -> pd.DataFrame:
    """Generates predictions for every discovered test .zarr dataset (no
    hardcoded sample names) using the best conservative config found by the
    sweep, validates the result, and writes it to `out_path`. Never calls
    any Kaggle submission API - this only writes a CSV to disk.
    """
    test_dirs = find_test_zarr_dirs()
    if not test_dirs:
        print("[error] no test .zarr datasets found. Nothing to submit.")
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    print(f"Discovered {len(test_dirs)} test dataset(s): {[p.stem for p in test_dirs]}")
    submission = build_conservative_submission(
        test_dirs, detector_config, node_filter_config, edge_filter_config, keep_only_linked_nodes
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
# 14. Tests
# --------------------------------------------------------------------------- #
def run_conservative_tracker_tests() -> None:
    """Correctness tests for node/edge/tracklet filtering and submission
    generation, using small synthetic fixtures - no dependency on real
    Kaggle data being mounted.
    """
    # Test 1: node filter - absolute threshold removes low-score nodes.
    nodes_t = pd.DataFrame({"node_id": [0, 1, 2], "t": [0, 0, 0], "z": [0.0] * 3, "y": [0.0] * 3, "x": [0.0] * 3, "score": [0.9, 0.3, 0.01]})
    filtered = _apply_node_filter_one_timepoint(nodes_t, max_nodes=100, score_quantile=0.0, abs_threshold=0.2)
    assert set(filtered["node_id"]) == {0, 1}, f"expected nodes {{0,1}} to survive abs_threshold=0.2, got {set(filtered['node_id'])}"

    # Test 2: node filter - max_nodes cap keeps only the top-K by score.
    many_nodes = pd.DataFrame({
        "node_id": list(range(10)), "t": [0] * 10, "z": [0.0] * 10, "y": [0.0] * 10, "x": [0.0] * 10,
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    })
    capped = _apply_node_filter_one_timepoint(many_nodes, max_nodes=3, score_quantile=0.0, abs_threshold=0.0)
    assert set(capped["node_id"]) == {0, 1, 2}, f"expected top-3 nodes {{0,1,2}}, got {set(capped['node_id'])}"

    # Test 3: candidate edges + mutual-nearest-neighbor flag on a simple
    # 2-node-per-frame case where both pairs are mutually nearest.
    nodes_2frames = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 0, 1, 1],
        "z": [0.0, 0.0, 0.0, 0.0], "y": [0.0, 100.0, 1.0, 101.0], "x": [0.0, 0.0, 0.0, 0.0],
        "score": [0.9, 0.8, 0.9, 0.8],
    })
    candidates = compute_candidate_edges(nodes_2frames)
    assert len(candidates) == 2
    assert candidates["is_mnn"].all(), "both pairs should be mutually nearest in this well-separated fixture"

    # Test 4: edge filter respects max_link_distance_um, require_mnn, and
    # max_edges_per_frame ranking by edge_score.
    filtered_edges = apply_edge_filter(candidates, max_link_distance_um=5.0, require_mnn=True, max_edges_per_frame=1)
    assert len(filtered_edges) == 1, "max_edges_per_frame=1 should keep exactly 1 edge (only 1 frame pair here)"
    too_strict = apply_edge_filter(candidates, max_link_distance_um=0.01, require_mnn=True, max_edges_per_frame=10)
    assert len(too_strict) == 0, "an unreasonably small max_link_distance_um should reject every edge"

    # Test 5: tracklet filter drops isolated (unlinked) nodes when requested.
    pred_nodes = pd.DataFrame({"node_id": [0, 1, 2, 3], "t": [0, 0, 1, 1], "z": [0.0] * 4, "y": [0.0] * 4, "x": [0.0] * 4})
    pred_edges = pd.DataFrame({"source_id": [0], "target_id": [2]})
    kept_nodes, kept_edges = apply_tracklet_filter(pred_nodes, pred_edges, keep_only_linked_nodes=True)
    assert set(kept_nodes["node_id"]) == {0, 2}, f"expected only linked nodes {{0,2}}, got {set(kept_nodes['node_id'])}"
    all_nodes, all_edges = apply_tracklet_filter(pred_nodes, pred_edges, keep_only_linked_nodes=False)
    assert len(all_nodes) == 4, "keep_only_linked_nodes=False should not remove any node"

    # Test 6: full submission schema validity + global node_id uniqueness,
    # and rejection of a deliberately broken submission.
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
    nodes_only = fake_submission[fake_submission["row_type"] == "node"]
    assert nodes_only["node_id"].is_unique and len(nodes_only) == 4

    broken = fake_submission.copy()
    broken.loc[broken["row_type"] == "edge", "target_id"] = 999999
    raised = False
    try:
        validate_submission(broken, expected_datasets=["fake_a", "fake_b"])
    except AssertionError:
        raised = True
    assert raised, "validate_submission should reject a dangling edge target_id"

    print("All conservative_tracker tests passed (6/6).")


# --------------------------------------------------------------------------- #
# 15. Top-level driver
# --------------------------------------------------------------------------- #
def run_conservative_tracker_pipeline(
    n_eval_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    baseline_edge_jaccard: float = 0.00045,
    submission_path: str = "/kaggle/working/submission.csv",
    generate_submission: bool = True,
) -> dict:
    """Runs the tests, the full node/edge/tracklet-filter sweep (+best-
    config selection), and - if `generate_submission` - builds
    submission.csv for every discovered test dataset using the best
    conservative config found. Never calls any Kaggle submission API.
    """
    print("=== Self-test: node/edge/tracklet filtering + submission-validation unit tests ===")
    run_conservative_tracker_tests()

    print("\n=== Conservative filter sweep (first %d train sample(s)) ===" % n_eval_samples)
    eval_df = run_conservative_tracker_eval(n_samples=n_eval_samples, detector_config=detector_config)

    print("\n=== Best config selection ===")
    best_config_df = select_best_config(eval_df, baseline_edge_jaccard=baseline_edge_jaccard)

    submission_df = pd.DataFrame(columns=SUBMISSION_COLUMNS)
    if generate_submission and len(best_config_df):
        print("\n=== Submission generation (all test datasets, best conservative config) ===")
        row = best_config_df.iloc[0]
        node_filter_config = {
            "max_nodes_per_timepoint": int(row["max_nodes_per_timepoint"]),
            "score_quantile_per_timepoint": float(row["score_quantile_per_timepoint"]),
            "absolute_score_threshold": float(row["absolute_score_threshold"]),
        }
        edge_filter_config = {
            "max_link_distance_um": float(row["max_link_distance_um"]),
            "require_mutual_nearest_neighbor": bool(row["require_mutual_nearest_neighbor"]),
            "max_edges_per_frame": int(row["max_edges_per_frame"]),
        }
        keep_only_linked_nodes = bool(row["keep_only_linked_nodes"])
        submission_df = generate_and_save_submission(
            detector_config, node_filter_config, edge_filter_config, keep_only_linked_nodes, out_path=submission_path,
        )

    return {"eval_df": eval_df, "best_config_df": best_config_df, "submission_df": submission_df}


if __name__ == "__main__":
    run_conservative_tracker_pipeline(n_eval_samples=3)
