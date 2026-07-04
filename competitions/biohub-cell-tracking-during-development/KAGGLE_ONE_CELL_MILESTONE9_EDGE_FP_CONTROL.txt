"""
Biohub - Cell Tracking During Development
Milestone 9: joint node-recovery + stronger edge-quality control.

Milestone 8 swept the node filter alone (60 combos) and found: node
recovery works (filtered_node_coverage improves from 17/169 up to 57/169),
but edge quality collapses (edge_FP grows from ~5700 to 13000-14000 while
edge_TP only reaches 14). Best Milestone 8 edge_jaccard was 0.002117 - not
better than the FAIR pre-tracklet-filter baseline, 6B's edge_jaccard=
0.002215 (TP=13, FP=5700). Node-filter loosening in isolation is a dead
end: the naive per-frame Hungarian assignment used since Milestone 6B
forces EVERY admitted node to receive an edge, so as more (mostly noisy)
nodes get admitted, false edges scale far faster than true ones.

This module attacks that directly with joint node-recovery + a genuinely
different edge-linking mechanism, evaluated over only 5 hand-picked
node-filter configs (not another 60-combo blind sweep):

  1. baseline_6BCD:      max_nodes=75,  quantile=0.90, abs_threshold=0.05
  2. m8_best_tradeoff:   max_nodes=75,  quantile=0.90, abs_threshold=0.03
  3. moderate_100:       max_nodes=100, quantile=0.90, abs_threshold=0.03
  4. moderate_150_085:   max_nodes=150, quantile=0.85, abs_threshold=0.05
  5. stronger_150_080:   max_nodes=150, quantile=0.80, abs_threshold=0.05

Edge-quality mechanism (replaces compute_candidate_edges/apply_edge_filter's
forced full bipartite Hungarian assignment entirely):

  A) Candidate pruning before assignment - a source node only ever
     considers its `k_next_candidates` nearest targets within
     `max_link_distance_um` (never the full next-frame pool).
  B) Combined edge scoring - physical distance x source/target node score
     x local-rank bonus (favors the nearest of the k candidates) x a
     displacement-smoothness bonus computed against a precomputed
     reference incoming-displacement (a lightweight top-1-nearest-neighbor
     "most likely predecessor" chain, kept independent of any
     threshold/assignment decision so it can be built once and reused
     across the whole edge-config sweep). Node score itself is already the
     cheapest available intensity-consistency proxy (the detector's score
     IS derived from local image intensity), so no separate intensity
     feature is bolted on.
  C) Assignment with a null option - candidates are globally sorted by
     edge_score and greedily accepted (respecting per-frame source/target
     exclusivity), but only above `min_edge_score` (a per-sample quantile
     of that combo's raw candidate-score distribution) - unlike Hungarian
     assignment, a node with no sufficiently good candidate is simply left
     unlinked instead of being forced onto its least-bad option.
  D) Tracklet-level FP control - after edge candidates are built for a
     shortlisted set of promising Stage 1 configs, tracklets are
     constructed and filtered (min_tracklet_length, mean-edge-score
     quantile, max smoothness error, keep_top_k_tracklets_per_sample) -
     reusing the exact same tracklet machinery from 6C/6D.

Caching: raw detection + GT are read exactly once per sample (3 samples).
For each of the 5 node-filter configs, node filtering + GT matching + the
reference-displacement chain are computed once and reused across the
whole edge-config grid (5 distance x 3 k = 15 candidate pools per
node-filter config); the raw (unthresholded) candidate-edge table itself
is independent of `min_edge_score`, so it too is built once per (sample,
node-filter, distance, k) and reused across all `min_edge_score_quantile`
values - only greedy assignment + edge_jaccard evaluation is repeated per
`min_edge_score` value.

Staged search:
  Stage 1: for each of the 5 node-filter configs, sweep
    max_link_distance_um x k_next_candidates x min_edge_score_quantile
    (5 x 3 x 4 = 60 edge-configs/node-filter -> 300 total rows). Saved to
    milestone9_edge_filter_eval.csv.
  Stage 2: the union of Stage 1's top 20 rows by edge_jaccard and every
    row with edge_TP > 4 and edge_FP < 2000 is carried forward; for each
    selected (node-filter, edge-config) pair, tracklets are built from its
    accepted edges and swept over min_tracklet_length x
    min_mean_edge_score_quantile x max_smoothness_error_um x
    keep_top_k_tracklets_per_sample (3x3x3x3 = 81 tracklet-filter combos
    per selected config), using node_policy A (tracklet-restricted nodes
    AND edges - the policy that actually controls node-level FP too,
    matching this milestone's "reduce edge_FP" goal). Saved to
    milestone9_tracklet_fp_control_eval.csv.

Final candidates are selected against 3 success criteria and saved to
milestone9_best_candidates.csv:
  1. beat 6B's edge_jaccard=0.002215 while keeping TP >= 13 OR FP < 5700
  2. beat 6C/6D's edge_jaccard=0.010667 (the ultimate, post-tracklet-filter
     target)
  3. a practical candidate with TP > 4 and FP < 1000 or FP < 2000

No final submission is built here. No ML yet.
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


# --------------------------------------------------------------------------- #
# 9. Tracklet construction + feature computation
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
# 10. Tracklet-level filtering + node-reconstruction policies
# --------------------------------------------------------------------------- #
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
# 11. GT node matching (embedded copy of local_metric.py's evaluator)
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
# 12. Local edge_jaccard evaluation (embedded copy of local_metric.py's evaluator)
# --------------------------------------------------------------------------- #
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


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    return precision, recall


# --------------------------------------------------------------------------- #
# 13. Selected node-filter configs (5 hand-picked, NOT another blind sweep)
# --------------------------------------------------------------------------- #
SELECTED_NODE_FILTER_CONFIGS = [
    {"name": "baseline_6BCD", "max_nodes_per_timepoint": 75, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.05},
    {"name": "m8_best_tradeoff", "max_nodes_per_timepoint": 75, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.03},
    {"name": "moderate_100", "max_nodes_per_timepoint": 100, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.03},
    {"name": "moderate_150_085", "max_nodes_per_timepoint": 150, "score_quantile_per_timepoint": 0.85, "absolute_score_threshold": 0.05},
    {"name": "stronger_150_080", "max_nodes_per_timepoint": 150, "score_quantile_per_timepoint": 0.80, "absolute_score_threshold": 0.05},
]

NODE_MATCH_MAX_DISTANCE_UM = 7.0


# --------------------------------------------------------------------------- #
# 14. Reference displacement chain (smoothness prior, independent of any
#     assignment/threshold decision so it can be built once and reused)
# --------------------------------------------------------------------------- #
def build_reference_displacement_by_node(nodes_df: pd.DataFrame) -> dict:
    """For every node, finds its single geometrically-nearest predecessor in
    the previous timepoint (top-1 nearest-neighbor, unconditional - no
    distance gate, no thresholding, no mutual-nearest-neighbor check) and
    records the physical displacement vector. This is used purely as a
    displacement-smoothness PRIOR when scoring real edge candidates (see
    `build_pruned_candidate_edges_for_sample`) - kept deliberately
    independent of `max_link_distance_um`/`k_next_candidates`/
    `min_edge_score` so it is computed exactly ONCE per (sample,
    node-filter config) and reused across the whole edge-config grid.
    """
    ref_disp: dict[int, np.ndarray] = {}
    if len(nodes_df) == 0:
        return ref_disp

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
        cur_ids = cur["node_id"].to_numpy()
        nxt_ids = nxt["node_id"].to_numpy()

        dist = _pairwise_physical_distance_um(cur_coords, nxt_coords)
        nearest_pred_idx = dist.argmin(axis=0)
        for j, target_id in enumerate(nxt_ids):
            i = nearest_pred_idx[j]
            disp = (nxt_coords[j] - cur_coords[i]) * _VOXEL_SCALE
            ref_disp[int(target_id)] = disp

    return ref_disp


# --------------------------------------------------------------------------- #
# 15. Candidate pruning (A) + combined edge scoring (B) - replaces
#     compute_candidate_edges' forced full bipartite Hungarian assignment
# --------------------------------------------------------------------------- #
PRUNED_CANDIDATE_COLUMNS = [
    "t", "source_id", "target_id", "distance_um", "local_rank",
    "source_score", "target_score", "smoothness_um", "edge_score",
]


def build_pruned_candidate_edges_for_sample(
    nodes_df: pd.DataFrame,
    ref_disp: dict,
    max_link_distance_um: float,
    k_next_candidates: int,
) -> pd.DataFrame:
    """Builds the FULL raw (unthresholded) candidate-edge table for one
    sample: each source node considers only its `k_next_candidates`
    nearest targets within `max_link_distance_um` (candidate pruning, A) -
    never the full next-frame pool, so most of the noise a full bipartite
    Hungarian assignment would be forced to pair up is never even
    considered a candidate.

    Combined edge_score (B) = distance/score base term x a local-rank
    bonus (favors the nearest of the k candidates) x a smoothness bonus
    (favors candidates whose new displacement is consistent with the
    source node's own most-likely incoming displacement, when one
    exists). This table does NOT depend on `min_edge_score` - assignment/
    thresholding (C) happens separately in `greedy_assign_with_null`, so
    this table is built once per (sample, node-filter, distance, k) and
    reused across every `min_edge_score_quantile` value.
    """
    if len(nodes_df) == 0:
        return pd.DataFrame(columns=PRUNED_CANDIDATE_COLUMNS)

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
        cur_ids = cur["node_id"].to_numpy()
        cur_scores = cur["score"].to_numpy(dtype=float)
        nxt_ids = nxt["node_id"].to_numpy()
        nxt_scores = nxt["score"].to_numpy(dtype=float)

        dist = _pairwise_physical_distance_um(cur_coords, nxt_coords)

        for i, source_id in enumerate(cur_ids):
            order = np.argsort(dist[i])
            rank = 0
            for j in order:
                distance = float(dist[i, j])
                if distance > max_link_distance_um:
                    break  # ascending order - every remaining candidate is also too far
                rank += 1
                if rank > k_next_candidates:
                    break

                target_id = int(nxt_ids[j])
                source_score = float(cur_scores[i])
                target_score = float(nxt_scores[j])
                base_score = source_score * target_score / (1.0 + distance)
                rank_bonus = 1.0 / rank

                prev_disp = ref_disp.get(int(source_id))
                if prev_disp is not None:
                    new_disp = (nxt_coords[j] - cur_coords[i]) * _VOXEL_SCALE
                    smoothness_um = float(np.linalg.norm(new_disp - prev_disp))
                    smoothness_bonus = 1.0 / (1.0 + smoothness_um)
                else:
                    smoothness_um = float("nan")
                    smoothness_bonus = 1.0

                rows.append({
                    "t": int(t), "source_id": int(source_id), "target_id": target_id,
                    "distance_um": distance, "local_rank": rank,
                    "source_score": source_score, "target_score": target_score,
                    "smoothness_um": smoothness_um,
                    "edge_score": base_score * rank_bonus * smoothness_bonus,
                })

    return pd.DataFrame(rows, columns=PRUNED_CANDIDATE_COLUMNS)


# --------------------------------------------------------------------------- #
# 16. Assignment with a null option (C) - greedy, threshold-gated, allows
#     nodes to stay unlinked instead of forcing every node onto an edge
# --------------------------------------------------------------------------- #
def greedy_assign_with_null(candidates_df: pd.DataFrame, min_edge_score: float) -> pd.DataFrame:
    """Sorts every candidate edge (across ALL frame pairs at once - safe
    because node_id is globally unique per sample, so a node can only ever
    appear as a source/target within its own single frame pair) by
    edge_score descending, then greedily accepts pairs whose source AND
    target are both still unused, stopping consideration entirely below
    `min_edge_score`. Unlike Hungarian assignment, a node with no
    sufficiently good candidate is simply left unlinked.
    """
    columns = list(candidates_df.columns)
    if len(candidates_df) == 0:
        return candidates_df

    eligible = candidates_df[candidates_df["edge_score"] >= min_edge_score]
    if len(eligible) == 0:
        return eligible

    ordered = eligible.sort_values("edge_score", ascending=False)
    used_sources: set = set()
    used_targets: set = set()
    keep_mask = np.zeros(len(ordered), dtype=bool)

    source_ids = ordered["source_id"].to_numpy()
    target_ids = ordered["target_id"].to_numpy()
    for idx in range(len(ordered)):
        s, t = int(source_ids[idx]), int(target_ids[idx])
        if s in used_sources or t in used_targets:
            continue
        used_sources.add(s)
        used_targets.add(t)
        keep_mask[idx] = True

    return ordered[keep_mask][columns].reset_index(drop=True)
# --------------------------------------------------------------------------- #
# 17. Per-sample and per-(sample, node-filter-config) caching
# --------------------------------------------------------------------------- #
def precompute_raw_sample_cache(
    sample_zarr_dir: Path, detector_config: dict = DEFAULT_DETECTOR_CONFIG,
) -> dict:
    """Runs the raw detector and reads GT exactly ONCE per sample - shared
    across all 5 node-filter configs and every downstream edge/tracklet
    combo. The detector itself never re-runs anywhere else in this module.
    """
    dataset_name = sample_zarr_dir.stem
    gt_nodes, gt_edges = read_geff(sample_zarr_dir)
    raw_nodes_df, _ = detect_all_raw_nodes_for_sample(sample_zarr_dir, detector_config, node_id_start=0)
    n_timepoints = get_sample_timepoint_count(sample_zarr_dir)
    return {
        "dataset": dataset_name, "gt_nodes": gt_nodes, "gt_edges": gt_edges,
        "raw_nodes_df": raw_nodes_df, "n_timepoints": n_timepoints,
    }


def precompute_node_filter_cache(
    raw_cache: dict, node_filter_config: dict, match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """For ONE (sample, node-filter-config) pair: applies node filtering,
    matches filtered nodes to GT ONCE (reused across every edge-config
    combo for this node-filter config), and builds the reference
    displacement chain ONCE (reused across every max_link_distance_um /
    k_next_candidates combo).
    """
    filtered_nodes = apply_node_filter(
        raw_cache["raw_nodes_df"],
        node_filter_config["max_nodes_per_timepoint"],
        node_filter_config["score_quantile_per_timepoint"],
        node_filter_config["absolute_score_threshold"],
    )[["node_id", "t", "z", "y", "x", "score"]].reset_index(drop=True)

    match_filtered = match_nodes_by_timepoint(filtered_nodes, raw_cache["gt_nodes"], match_max_distance_um)
    ref_disp = build_reference_displacement_by_node(filtered_nodes)

    return {
        "dataset": raw_cache["dataset"],
        "filtered_nodes": filtered_nodes,
        "match_filtered": match_filtered,
        "ref_disp": ref_disp,
        "gt_edges": raw_cache["gt_edges"],
        "n_timepoints": raw_cache["n_timepoints"],
    }


def build_node_filter_caches(
    raw_caches: list[dict], node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Builds `{node_filter_name: [per_sample_cache, ...]}` for every one of
    the 5 selected node-filter configs, once, up front - reused by both
    Stage 1 and Stage 2 so node filtering/GT-matching/reference-
    displacement are each computed exactly once per (sample, node-filter
    config) for the whole module.
    """
    caches_by_name: dict[str, list[dict]] = {}
    for node_filter_config in node_filter_configs:
        name = node_filter_config["name"]
        caches_by_name[name] = [
            precompute_node_filter_cache(rc, node_filter_config, match_max_distance_um) for rc in raw_caches
        ]
    return caches_by_name


# --------------------------------------------------------------------------- #
# 18. Stage 1: edge candidate/pruning/scoring sweep (per selected node-filter config)
# --------------------------------------------------------------------------- #
DEFAULT_MAX_LINK_DISTANCE_UM_VALUES = (2, 3, 4, 5, 7)
DEFAULT_K_NEXT_CANDIDATES_VALUES = (1, 2, 3)
DEFAULT_MIN_EDGE_SCORE_QUANTILE_VALUES = (0.0, 0.3, 0.5, 0.7)

STAGE1_COLUMNS = [
    "node_filter_name", "max_nodes_per_timepoint", "score_quantile_per_timepoint", "absolute_score_threshold",
    "max_link_distance_um", "k_next_candidates", "min_edge_score_quantile",
    "edge_jaccard", "edge_TP", "edge_FP", "edge_FN", "precision", "recall",
    "n_accepted_edges", "avg_pred_nodes_per_timepoint", "avg_edges_per_frame", "runtime_seconds",
]


def _node_filter_params_by_name(node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS) -> dict:
    return {c["name"]: c for c in node_filter_configs}


def evaluate_edge_config(node_filter_cache: dict, candidates_df: pd.DataFrame, min_edge_score_quantile: float) -> dict:
    """Evaluates ONE (node-filter, distance, k, min_edge_score_quantile)
    combo against one sample's already-built candidate table: resolves
    `min_edge_score_quantile` to an absolute threshold from that sample's
    OWN candidate-score distribution, greedily assigns with the null
    option, then evaluates edge_jaccard/precision/recall against GT (using
    the already-cached GT-node match - no re-matching per combo).
    """
    threshold = (
        float(candidates_df["edge_score"].quantile(min_edge_score_quantile)) if len(candidates_df) else float("inf")
    )
    accepted = greedy_assign_with_null(candidates_df, threshold)

    edge_result = compute_edge_jaccard(
        accepted[["source_id", "target_id"]] if len(accepted) else pd.DataFrame(columns=["source_id", "target_id"]),
        node_filter_cache["gt_edges"], node_filter_cache["match_filtered"]["pred_to_gt"],
    )
    return {
        "edge_TP": edge_result["edge_TP"], "edge_FP": edge_result["edge_FP"], "edge_FN": edge_result["edge_FN"],
        "n_accepted_edges": len(accepted), "accepted_edges": accepted,
    }


def run_milestone9_stage1(
    node_filter_caches_by_name: dict,
    node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    max_link_distance_um_values: Sequence[float] = DEFAULT_MAX_LINK_DISTANCE_UM_VALUES,
    k_next_candidates_values: Sequence[int] = DEFAULT_K_NEXT_CANDIDATES_VALUES,
    min_edge_score_quantile_values: Sequence[float] = DEFAULT_MIN_EDGE_SCORE_QUANTILE_VALUES,
    csv_path: str = "/kaggle/working/milestone9_edge_filter_eval.csv",
    progress_every: int = 20,
) -> pd.DataFrame:
    """5 node-filter configs x (5 distance x 3 k x 4 min_edge_score_quantile
    = 60 edge-configs) = 300 rows. The raw candidate table (distance/k
    dependent only) is built once per (sample, node-filter, distance, k)
    and reused across all 4 min_edge_score_quantile values.
    """
    params_by_name = _node_filter_params_by_name(node_filter_configs)
    n_edge_combos = len(max_link_distance_um_values) * len(k_next_candidates_values) * len(min_edge_score_quantile_values)
    n_total = n_edge_combos * len(node_filter_configs)
    print(f"Stage 1: sweeping {n_edge_combos} edge-config combo(s) x {len(node_filter_configs)} node-filter config(s) = {n_total} total row(s).")

    rows: list[dict] = []
    combo_idx = 0
    sweep_t0 = time.time()

    for node_filter_config in node_filter_configs:
        name = node_filter_config["name"]
        caches = node_filter_caches_by_name[name]
        total_timepoints = sum(c["n_timepoints"] for c in caches)
        total_frame_pairs = sum(max(0, c["n_timepoints"] - 1) for c in caches)
        total_pred_nodes = sum(len(c["filtered_nodes"]) for c in caches)

        for max_dist in max_link_distance_um_values:
            for k in k_next_candidates_values:
                candidate_pools = [
                    build_pruned_candidate_edges_for_sample(c["filtered_nodes"], c["ref_disp"], max_dist, k)
                    for c in caches
                ]
                for q in min_edge_score_quantile_values:
                    combo_idx += 1
                    row_t0 = time.time()

                    edge_TP = edge_FP = edge_FN = 0
                    n_accepted_edges = 0
                    for cache, pool in zip(caches, candidate_pools):
                        result = evaluate_edge_config(cache, pool, q)
                        edge_TP += result["edge_TP"]
                        edge_FP += result["edge_FP"]
                        edge_FN += result["edge_FN"]
                        n_accepted_edges += result["n_accepted_edges"]

                    denom = edge_TP + edge_FP + edge_FN
                    edge_jaccard = edge_TP / denom if denom > 0 else 1.0
                    precision, recall = _precision_recall(edge_TP, edge_FP, edge_FN)

                    rows.append({
                        "node_filter_name": name,
                        "max_nodes_per_timepoint": node_filter_config["max_nodes_per_timepoint"],
                        "score_quantile_per_timepoint": node_filter_config["score_quantile_per_timepoint"],
                        "absolute_score_threshold": node_filter_config["absolute_score_threshold"],
                        "max_link_distance_um": max_dist, "k_next_candidates": k, "min_edge_score_quantile": q,
                        "edge_jaccard": edge_jaccard, "edge_TP": edge_TP, "edge_FP": edge_FP, "edge_FN": edge_FN,
                        "precision": precision, "recall": recall,
                        "n_accepted_edges": n_accepted_edges,
                        "avg_pred_nodes_per_timepoint": (total_pred_nodes / total_timepoints) if total_timepoints else float("nan"),
                        "avg_edges_per_frame": (n_accepted_edges / total_frame_pairs) if total_frame_pairs else float("nan"),
                        "runtime_seconds": time.time() - row_t0,
                    })

                    if combo_idx % progress_every == 0 or combo_idx == n_total:
                        elapsed = time.time() - sweep_t0
                        eta = elapsed / combo_idx * (n_total - combo_idx)
                        print(f"  [{combo_idx}/{n_total} Stage-1 combos] elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    stage1_df = pd.DataFrame(rows, columns=STAGE1_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_df.to_csv(out_path, index=False)
    print(f"\nSaved Stage 1 edge-filter eval ({len(stage1_df)} row(s)) to {out_path}")
    return stage1_df
# --------------------------------------------------------------------------- #
# 19. Stage 1 -> Stage 2 config selection
# --------------------------------------------------------------------------- #
STAGE1_CONFIG_KEYS = ["node_filter_name", "max_link_distance_um", "k_next_candidates", "min_edge_score_quantile"]


def select_stage1_configs_for_stage2(stage1_df: pd.DataFrame, top_n_by_jaccard: int = 20) -> pd.DataFrame:
    """Carries forward the union of Stage 1's top `top_n_by_jaccard` rows by
    edge_jaccard and every row with edge_TP > 4 and edge_FP < 2000,
    deduplicated on the 4 config-identifying columns.
    """
    if len(stage1_df) == 0:
        return stage1_df.iloc[0:0]

    by_jaccard = stage1_df.sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)
    top_by_jaccard = by_jaccard.head(top_n_by_jaccard)
    promising_tp_fp = stage1_df[(stage1_df["edge_TP"] > 4) & (stage1_df["edge_FP"] < 2000)]

    combined = pd.concat([top_by_jaccard, promising_tp_fp], ignore_index=True)
    selected = combined.drop_duplicates(subset=STAGE1_CONFIG_KEYS).reset_index(drop=True)
    print(
        f"Stage 1 -> Stage 2: top {top_n_by_jaccard} by edge_jaccard ({len(top_by_jaccard)} row(s)) "
        f"+ edge_TP>4 & edge_FP<2000 ({len(promising_tp_fp)} row(s)) -> {len(selected)} unique config(s) carried forward."
    )
    return selected


# --------------------------------------------------------------------------- #
# 20. Stage 2: tracklet-level FP control on the shortlisted Stage 1 configs
# --------------------------------------------------------------------------- #
TRACKLET_MIN_LENGTH_VALUES = (3, 4, 5)
TRACKLET_MIN_MEAN_EDGE_SCORE_QUANTILE_VALUES = (0.5, 0.7, 0.9)
TRACKLET_MAX_SMOOTHNESS_ERROR_UM_VALUES = (3, 5, 7)
TRACKLET_KEEP_TOP_K_VALUES = (25, 50, 100)

STAGE2_COLUMNS = STAGE1_COLUMNS[:-1] + [
    "min_tracklet_length", "min_mean_edge_score_quantile", "max_smoothness_error_um", "keep_top_k_tracklets_per_sample",
    "num_tracklets_kept", "runtime_seconds",
]


def run_milestone9_stage2(
    node_filter_caches_by_name: dict,
    stage1_selected_df: pd.DataFrame,
    min_tracklet_length_values: Sequence[int] = TRACKLET_MIN_LENGTH_VALUES,
    min_mean_edge_score_quantile_values: Sequence[float] = TRACKLET_MIN_MEAN_EDGE_SCORE_QUANTILE_VALUES,
    max_smoothness_error_um_values: Sequence[float] = TRACKLET_MAX_SMOOTHNESS_ERROR_UM_VALUES,
    keep_top_k_values: Sequence[int] = TRACKLET_KEEP_TOP_K_VALUES,
    csv_path: str = "/kaggle/working/milestone9_tracklet_fp_control_eval.csv",
    progress_every: int = 50,
) -> pd.DataFrame:
    """For every shortlisted Stage 1 (node-filter, distance, k,
    min_edge_score_quantile) config, rebuilds its accepted edges, builds
    tracklets, and sweeps the tracklet-filter grid (3x3x3x3 = 81 combos),
    using node_policy A (both nodes and edges restricted to kept
    tracklets - this is what actually controls node-level FP too, matching
    the "reduce edge_FP" goal).
    """
    if len(stage1_selected_df) == 0:
        print("[warn] no Stage 1 configs selected - skipping Stage 2.")
        return pd.DataFrame(columns=STAGE2_COLUMNS)

    n_tracklet_combos = (
        len(min_tracklet_length_values) * len(min_mean_edge_score_quantile_values)
        * len(max_smoothness_error_um_values) * len(keep_top_k_values)
    )
    n_total = n_tracklet_combos * len(stage1_selected_df)
    print(
        f"Stage 2: sweeping {n_tracklet_combos} tracklet-filter combo(s) x "
        f"{len(stage1_selected_df)} shortlisted Stage-1 config(s) = {n_total} total row(s)."
    )

    rows: list[dict] = []
    combo_idx = 0
    sweep_t0 = time.time()

    # The raw candidate-edge pool depends only on (node_filter, distance, k)
    # - NOT on min_edge_score_quantile - so multiple selected Stage-1 rows
    # that share the same (name, max_dist, k) but differ only in their
    # quantile reuse the same cached pool instead of rebuilding it.
    candidate_pool_cache: dict[tuple, list[pd.DataFrame]] = {}

    for _, cfg_row in stage1_selected_df.iterrows():
        name = cfg_row["node_filter_name"]
        max_dist = cfg_row["max_link_distance_um"]
        k = int(cfg_row["k_next_candidates"])
        q_edge = cfg_row["min_edge_score_quantile"]
        caches = node_filter_caches_by_name[name]

        total_timepoints = sum(c["n_timepoints"] for c in caches)
        total_frame_pairs = sum(max(0, c["n_timepoints"] - 1) for c in caches)

        pool_key = (name, max_dist, k)
        if pool_key not in candidate_pool_cache:
            candidate_pool_cache[pool_key] = [
                build_pruned_candidate_edges_for_sample(c["filtered_nodes"], c["ref_disp"], max_dist, k) for c in caches
            ]
        candidate_pools = candidate_pool_cache[pool_key]

        per_sample_tracklets = []
        for cache, candidates_df in zip(caches, candidate_pools):
            threshold = float(candidates_df["edge_score"].quantile(q_edge)) if len(candidates_df) else float("inf")
            accepted = greedy_assign_with_null(candidates_df, threshold)
            tracklets_df = build_tracklets(cache["filtered_nodes"], accepted)
            edge_quantile_thresholds = {
                q: (float(tracklets_df["mean_edge_score"].quantile(q)) if len(tracklets_df) else float("inf"))
                for q in min_mean_edge_score_quantile_values
            }
            per_sample_tracklets.append({
                "cache": cache, "tracklets_df": tracklets_df, "edge_quantile_thresholds": edge_quantile_thresholds,
            })

        for min_len in min_tracklet_length_values:
            for q_tracklet in min_mean_edge_score_quantile_values:
                for max_smooth in max_smoothness_error_um_values:
                    for keep_top_k in keep_top_k_values:
                        combo_idx += 1
                        row_t0 = time.time()

                        edge_TP = edge_FP = edge_FN = 0
                        total_pred_nodes = 0
                        total_pred_edges = 0
                        total_kept_tracklets = 0

                        for entry in per_sample_tracklets:
                            cache = entry["cache"]
                            tracklets_df = entry["tracklets_df"]
                            threshold = entry["edge_quantile_thresholds"][q_tracklet]
                            kept = apply_tracklet_filter(
                                tracklets_df, min_len, 0.0, threshold, float("inf"), max_smooth, keep_top_k,
                            )
                            pred_nodes, pred_edges = reconstruct_policy_a(kept, cache["filtered_nodes"][["node_id", "t", "z", "y", "x"]])
                            # Policy A only ever drops nodes from the already-matched filtered
                            # set - restricting the cached filtered-node GT match to the kept
                            # node_ids is exactly equivalent to rematching, with no extra
                            # Hungarian solve needed.
                            kept_node_ids = set(pred_nodes["node_id"]) if len(pred_nodes) else set()
                            restricted_pred_to_gt = {
                                pid: gid for pid, gid in cache["match_filtered"]["pred_to_gt"].items() if pid in kept_node_ids
                            }
                            edge_result = compute_edge_jaccard(pred_edges, cache["gt_edges"], restricted_pred_to_gt)
                            edge_TP += edge_result["edge_TP"]
                            edge_FP += edge_result["edge_FP"]
                            edge_FN += edge_result["edge_FN"]
                            total_pred_nodes += len(pred_nodes)
                            total_pred_edges += len(pred_edges)
                            total_kept_tracklets += len(kept)

                        denom = edge_TP + edge_FP + edge_FN
                        edge_jaccard = edge_TP / denom if denom > 0 else 1.0
                        precision, recall = _precision_recall(edge_TP, edge_FP, edge_FN)

                        rows.append({
                            "node_filter_name": name,
                            "max_nodes_per_timepoint": cfg_row["max_nodes_per_timepoint"],
                            "score_quantile_per_timepoint": cfg_row["score_quantile_per_timepoint"],
                            "absolute_score_threshold": cfg_row["absolute_score_threshold"],
                            "max_link_distance_um": max_dist, "k_next_candidates": k, "min_edge_score_quantile": q_edge,
                            "edge_jaccard": edge_jaccard, "edge_TP": edge_TP, "edge_FP": edge_FP, "edge_FN": edge_FN,
                            "precision": precision, "recall": recall,
                            "n_accepted_edges": total_pred_edges,
                            "avg_pred_nodes_per_timepoint": (total_pred_nodes / total_timepoints) if total_timepoints else float("nan"),
                            "avg_edges_per_frame": (total_pred_edges / total_frame_pairs) if total_frame_pairs else float("nan"),
                            "min_tracklet_length": min_len, "min_mean_edge_score_quantile": q_tracklet,
                            "max_smoothness_error_um": max_smooth, "keep_top_k_tracklets_per_sample": keep_top_k,
                            "num_tracklets_kept": total_kept_tracklets,
                            "runtime_seconds": time.time() - row_t0,
                        })

                        if combo_idx % progress_every == 0 or combo_idx == n_total:
                            elapsed = time.time() - sweep_t0
                            eta = elapsed / combo_idx * (n_total - combo_idx)
                            print(f"  [{combo_idx}/{n_total} Stage-2 combos] elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    stage2_df = pd.DataFrame(rows, columns=STAGE2_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_df.to_csv(out_path, index=False)
    print(f"\nSaved Stage 2 tracklet-FP-control eval ({len(stage2_df)} row(s)) to {out_path}")
    return stage2_df
# --------------------------------------------------------------------------- #
# 21. Combined Stage 1 + Stage 2 reporting
# --------------------------------------------------------------------------- #
_COMMON_METRIC_COLUMNS = ["edge_jaccard", "edge_TP", "edge_FP", "edge_FN", "precision", "recall"]


def _stage1_label(row: pd.Series) -> str:
    return (
        f"[stage1] node_filter={row['node_filter_name']} max_dist={row['max_link_distance_um']} "
        f"k={row['k_next_candidates']} min_edge_q={row['min_edge_score_quantile']}"
    )


def _stage2_label(row: pd.Series) -> str:
    return (
        f"[stage2] node_filter={row['node_filter_name']} max_dist={row['max_link_distance_um']} "
        f"k={row['k_next_candidates']} min_edge_q={row['min_edge_score_quantile']} "
        f"min_len={row['min_tracklet_length']} tracklet_q={row['min_mean_edge_score_quantile']} "
        f"max_smooth={row['max_smoothness_error_um']} keep_top_k={row['keep_top_k_tracklets_per_sample']}"
    )


def build_combined_candidates(stage1_df: pd.DataFrame, stage2_df: pd.DataFrame) -> pd.DataFrame:
    """Combines Stage 1 (edge-config only) and Stage 2 (edge-config +
    tracklet-filter) rows into one table sharing the common evaluation
    metrics, with a `stage` tag and a human-readable `label` describing the
    full config - used for all cross-stage reporting and best-candidate
    selection below.
    """
    parts = []
    if len(stage1_df):
        c1 = stage1_df[_COMMON_METRIC_COLUMNS].copy()
        c1["stage"] = "stage1_edge_config"
        c1["label"] = stage1_df.apply(_stage1_label, axis=1)
        parts.append(c1)
    if len(stage2_df):
        c2 = stage2_df[_COMMON_METRIC_COLUMNS].copy()
        c2["stage"] = "stage2_tracklet_fp_control"
        c2["label"] = stage2_df.apply(_stage2_label, axis=1)
        parts.append(c2)
    if not parts:
        return pd.DataFrame(columns=_COMMON_METRIC_COLUMNS + ["stage", "label"])
    return pd.concat(parts, ignore_index=True)


def _print_candidate_table(df: pd.DataFrame, max_rows: int) -> None:
    if len(df) == 0:
        print("  (no configs match)")
        return
    for _, r in df.head(max_rows).iterrows():
        print(
            f"  {r['label']} | edge_jaccard={r['edge_jaccard']:.6f} TP={r['edge_TP']} FP={r['edge_FP']} "
            f"FN={r['edge_FN']} precision={r['precision']:.6f} recall={r['recall']:.6f}"
        )


def print_summary_reports(combined_df: pd.DataFrame) -> dict:
    """Prints the requested breakdowns: top 30 by edge_jaccard, plus configs
    with edge_FP < 500/1000/2000, edge_TP > 4, edge_TP >= 10 with
    edge_FP < 2000, and recall >= 0.05 with edge_FP < 2000 - across BOTH
    Stage 1 and Stage 2 candidates combined.
    """
    if len(combined_df) == 0:
        print("[warn] no candidates to report on.")
        return {}

    by_jaccard = combined_df.sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)

    fp_lt_500 = by_jaccard[by_jaccard["edge_FP"] < 500]
    fp_lt_1000 = by_jaccard[by_jaccard["edge_FP"] < 1000]
    fp_lt_2000 = by_jaccard[by_jaccard["edge_FP"] < 2000]
    tp_gt_4 = by_jaccard[by_jaccard["edge_TP"] > 4]
    tp_ge_10_fp_lt_2000 = by_jaccard[(by_jaccard["edge_TP"] >= 10) & (by_jaccard["edge_FP"] < 2000)]
    recall_ge_05_fp_lt_2000 = by_jaccard[(by_jaccard["recall"] >= 0.05) & (by_jaccard["edge_FP"] < 2000)]

    print(f"\n=== Top 30 configs by edge_jaccard (of {len(combined_df)} total, both stages) ===")
    _print_candidate_table(by_jaccard, 30)

    print(f"\n=== Top configs with edge_FP < 500 ({len(fp_lt_500)} of {len(combined_df)}) ===")
    _print_candidate_table(fp_lt_500, 15)

    print(f"\n=== Top configs with edge_FP < 1000 ({len(fp_lt_1000)} of {len(combined_df)}) ===")
    _print_candidate_table(fp_lt_1000, 15)

    print(f"\n=== Top configs with edge_FP < 2000 ({len(fp_lt_2000)} of {len(combined_df)}) ===")
    _print_candidate_table(fp_lt_2000, 15)

    print(f"\n=== Top configs with edge_TP > 4 ({len(tp_gt_4)} of {len(combined_df)}) ===")
    _print_candidate_table(tp_gt_4, 15)

    print(f"\n=== Top configs with edge_TP >= 10 and edge_FP < 2000 ({len(tp_ge_10_fp_lt_2000)} of {len(combined_df)}) ===")
    _print_candidate_table(tp_ge_10_fp_lt_2000, 15)

    print(f"\n=== Top configs with recall >= 0.05 and edge_FP < 2000 ({len(recall_ge_05_fp_lt_2000)} of {len(combined_df)}) ===")
    _print_candidate_table(recall_ge_05_fp_lt_2000, 15)

    return {
        "by_edge_jaccard": by_jaccard, "edge_fp_lt_500": fp_lt_500, "edge_fp_lt_1000": fp_lt_1000,
        "edge_fp_lt_2000": fp_lt_2000, "edge_tp_gt_4": tp_gt_4,
        "edge_tp_ge_10_fp_lt_2000": tp_ge_10_fp_lt_2000, "recall_ge_05_fp_lt_2000": recall_ge_05_fp_lt_2000,
    }


# --------------------------------------------------------------------------- #
# 22. Final best-candidate selection against the 3 success criteria +
#     explicit comparison against both baselines
# --------------------------------------------------------------------------- #
BASELINE_6B_EDGE_JACCARD = 0.002215
BASELINE_6B_EDGE_TP = 13
BASELINE_6B_EDGE_FP = 5700
BASELINE_6C6D_EDGE_JACCARD = 0.010667
BASELINE_6C6D_EDGE_TP = 4
BASELINE_6C6D_EDGE_FP = 206


def select_best_candidates(
    combined_df: pd.DataFrame,
    baseline_6B_edge_jaccard: float = BASELINE_6B_EDGE_JACCARD,
    baseline_6B_edge_tp: int = BASELINE_6B_EDGE_TP,
    baseline_6B_edge_fp: int = BASELINE_6B_EDGE_FP,
    baseline_6C6D_edge_jaccard: float = BASELINE_6C6D_EDGE_JACCARD,
    csv_path: str = "/kaggle/working/milestone9_best_candidates.csv",
) -> pd.DataFrame:
    """Flags every Stage 1 + Stage 2 candidate against Milestone 9's 3
    success criteria:
      1. beat 6B's edge_jaccard=0.002215 (the fair pre-tracklet-filter
         stage baseline) while keeping edge_TP >= 13 OR edge_FP < 5700
      2. beat 6C/6D's edge_jaccard=0.010667 (the ultimate,
         post-tracklet-filter target)
      3. a practical candidate: edge_TP > 4 with edge_FP < 1000 or < 2000
    Saves every candidate meeting at least one criterion (sorted by
    edge_jaccard descending) to `csv_path`, and prints an explicit
    before/after comparison against both baselines for the single best
    candidate found.
    """
    if len(combined_df) == 0:
        print("[warn] no candidates to select best from.")
        return combined_df

    df = combined_df.copy()
    df["meets_criterion_1_beat_6B"] = (df["edge_jaccard"] > baseline_6B_edge_jaccard) & (
        (df["edge_TP"] >= baseline_6B_edge_tp) | (df["edge_FP"] < baseline_6B_edge_fp)
    )
    df["meets_criterion_2_beat_6C6D"] = df["edge_jaccard"] > baseline_6C6D_edge_jaccard
    df["meets_criterion_3_practical_fp1000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 1000)
    df["meets_criterion_3_practical_fp2000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 2000)
    df["meets_criterion_3_practical"] = df["meets_criterion_3_practical_fp1000"] | df["meets_criterion_3_practical_fp2000"]

    criterion_cols = ["meets_criterion_1_beat_6B", "meets_criterion_2_beat_6C6D", "meets_criterion_3_practical"]
    df["criteria_met"] = df[criterion_cols].apply(lambda row: ",".join(c for c in criterion_cols if row[c]), axis=1)

    any_criterion = df[criterion_cols].any(axis=1)
    best = df[any_criterion].sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)

    print(f"\n=== Best-candidate selection (of {len(df)} total Stage 1 + Stage 2 candidates) ===")
    print(f"  1) beat 6B (edge_jaccard>{baseline_6B_edge_jaccard}, TP>={baseline_6B_edge_tp} or FP<{baseline_6B_edge_fp}): {int(df['meets_criterion_1_beat_6B'].sum())}")
    print(f"  2) beat 6C/6D (edge_jaccard>{baseline_6C6D_edge_jaccard}): {int(df['meets_criterion_2_beat_6C6D'].sum())}")
    print(f"  3) practical (edge_TP>4, FP<1000: {int(df['meets_criterion_3_practical_fp1000'].sum())}, FP<2000: {int(df['meets_criterion_3_practical_fp2000'].sum())}): {int(df['meets_criterion_3_practical'].sum())}")
    print(f"  -> {len(best)} of {len(df)} candidate(s) meet at least one criterion.")

    print(f"\n  Baseline reminder: 6B (fair stage) edge_jaccard={baseline_6B_edge_jaccard}, TP={baseline_6B_edge_tp}, FP={baseline_6B_edge_fp}")
    print(f"  Baseline reminder: 6C/6D (ultimate target) edge_jaccard={baseline_6C6D_edge_jaccard}, TP={BASELINE_6C6D_EDGE_TP}, FP={BASELINE_6C6D_EDGE_FP}")

    if len(best):
        top = best.iloc[0]
        print(
            f"\n  Top candidate overall: {top['label']}\n"
            f"    edge_jaccard={top['edge_jaccard']:.6f}, edge_TP={top['edge_TP']}, edge_FP={top['edge_FP']}, "
            f"recall={top['recall']:.6f}, criteria_met=[{top['criteria_met']}]"
        )
    else:
        print("\n  No candidate met any of the 3 success criteria.")

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)
    print(f"Saved best candidates ({len(best)} row(s)) to {out_path}")

    return best
# --------------------------------------------------------------------------- #
# 23. Tests
# --------------------------------------------------------------------------- #
def run_milestone9_tests() -> None:
    """Correctness tests for the new edge-quality mechanisms (reference
    displacement, candidate pruning + scoring, greedy null-option
    assignment) and the Stage 1 -> Stage 2 -> best-candidate selection
    logic, using small synthetic fixtures (no disk I/O).
    """
    # Test 1: build_reference_displacement_by_node finds each node's single
    # nearest predecessor and records the correct physical displacement.
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 0, 1],
        "z": [0.0, 0.0, 0.0, 0.0], "y": [0.0, 10.0, 100.0, 108.0], "x": [0.0, 0.0, 0.0, 0.0],
        "score": [0.9, 0.9, 0.9, 0.9],
    })
    ref_disp = build_reference_displacement_by_node(nodes)
    assert set(ref_disp.keys()) == {1, 3}, "only t=1 nodes (which have a t=0 predecessor) get an entry"
    np.testing.assert_allclose(ref_disp[1], [0.0, 10.0 * VOXEL_SIZE_UM["y"], 0.0], atol=1e-9)
    np.testing.assert_allclose(ref_disp[3], [0.0, 8.0 * VOXEL_SIZE_UM["y"], 0.0], atol=1e-9)

    # Test 2: build_pruned_candidate_edges_for_sample respects the distance
    # gate and k_next_candidates limit, and assigns local_rank correctly.
    nodes2 = pd.DataFrame({
        "node_id": [10, 11, 12, 13], "t": [0, 1, 1, 1],
        "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 100.0], "x": [0.0] * 4,
        "score": [0.9, 0.8, 0.8, 0.8],
    })
    # node 10 (t=0) has 3 candidates at t=1: y=1 (closest), y=2, y=100 (far).
    candidates_k1 = build_pruned_candidate_edges_for_sample(nodes2, {}, max_link_distance_um=50.0, k_next_candidates=1)
    assert len(candidates_k1) == 1, "k_next_candidates=1 should keep only the single nearest candidate"
    assert candidates_k1.iloc[0]["target_id"] == 11 and candidates_k1.iloc[0]["local_rank"] == 1

    candidates_k2 = build_pruned_candidate_edges_for_sample(nodes2, {}, max_link_distance_um=50.0, k_next_candidates=2)
    assert len(candidates_k2) == 2, "k_next_candidates=2 should keep the 2 nearest (11 and 12), not 13"
    assert set(candidates_k2["target_id"]) == {11, 12}

    # y differences of 1/2/100 voxels correspond to physical distances of
    # ~0.406/0.813/40.625um (VOXEL_SIZE_UM["y"]=0.40625) - a 0.5um cutoff
    # keeps only node 11.
    candidates_dist_gate = build_pruned_candidate_edges_for_sample(nodes2, {}, max_link_distance_um=0.5, k_next_candidates=3)
    assert set(candidates_dist_gate["target_id"]) == {11}, "the distance gate should exclude 12 (~0.81um away) and 13 (~40.6um away) at a 0.5um cutoff"

    # Test 3: the smoothness bonus actually changes edge_score when a
    # reference displacement disagrees with the new candidate's direction.
    straight_ref = {10: np.array([0.0, 1.0 * VOXEL_SIZE_UM["y"], 0.0])}  # node 10 "arrived" moving +y
    candidates_with_ref = build_pruned_candidate_edges_for_sample(nodes2, straight_ref, max_link_distance_um=50.0, k_next_candidates=1)
    candidates_without_ref = build_pruned_candidate_edges_for_sample(nodes2, {}, max_link_distance_um=50.0, k_next_candidates=1)
    assert not np.isnan(candidates_with_ref.iloc[0]["smoothness_um"])
    assert np.isnan(candidates_without_ref.iloc[0]["smoothness_um"])
    # continuing +y motion into node 11 (also +y) should be a near-perfect
    # smoothness match (~0um), and the resulting edge_score should be at
    # least as high as the no-reference case (whose smoothness_bonus is
    # neutral, 1.0).
    assert candidates_with_ref.iloc[0]["edge_score"] >= candidates_without_ref.iloc[0]["edge_score"] - 1e-9

    # Test 4: greedy_assign_with_null respects the min_edge_score null
    # option (a node with no sufficiently good candidate stays unlinked)
    # and per-frame source/target exclusivity.
    candidates_df = pd.DataFrame({
        "t": [0, 0, 0], "source_id": [0, 1, 1], "target_id": [10, 10, 11],
        "distance_um": [1.0, 1.0, 5.0], "local_rank": [1, 1, 1],
        "source_score": [0.9, 0.9, 0.9], "target_score": [0.9, 0.9, 0.1],
        "smoothness_um": [float("nan")] * 3,
        "edge_score": [0.9, 0.5, 0.01],
    })
    accepted_lenient = greedy_assign_with_null(candidates_df, min_edge_score=0.0)
    assert len(accepted_lenient) == 2, "node 0->10 (best) wins the shared target; node 1 falls back to 1->11"
    assert set(zip(accepted_lenient["source_id"], accepted_lenient["target_id"])) == {(0, 10), (1, 11)}

    accepted_strict = greedy_assign_with_null(candidates_df, min_edge_score=0.4)
    assert len(accepted_strict) == 1, "1->11 (edge_score=0.01) should be rejected by the null option, not forced"
    assert (accepted_strict.iloc[0]["source_id"], accepted_strict.iloc[0]["target_id"]) == (0, 10)

    # Test 5: select_stage1_configs_for_stage2 deduplicates the union of
    # top-N-by-jaccard and TP>4/FP<2000 correctly.
    fixture_stage1 = pd.DataFrame([
        {**{k: v for k, v in zip(STAGE1_CONFIG_KEYS, ["cfgA", 3, 2, 0.3])}, "edge_jaccard": 0.05, "edge_TP": 20, "edge_FP": 300, "edge_FN": 10, "precision": 0.06, "recall": 0.6},
        {**{k: v for k, v in zip(STAGE1_CONFIG_KEYS, ["cfgB", 5, 1, 0.5])}, "edge_jaccard": 0.01, "edge_TP": 6, "edge_FP": 1500, "edge_FN": 100, "precision": 0.004, "recall": 0.05},
        {**{k: v for k, v in zip(STAGE1_CONFIG_KEYS, ["cfgC", 7, 3, 0.7])}, "edge_jaccard": 0.001, "edge_TP": 2, "edge_FP": 5000, "edge_FN": 160, "precision": 0.0004, "recall": 0.012},
    ])
    selected = select_stage1_configs_for_stage2(fixture_stage1, top_n_by_jaccard=1)
    # cfgA is top-1 by jaccard AND meets TP>4/FP<2000; cfgB meets TP>4/FP<2000
    # only; cfgC meets neither - so only cfgA and cfgB should be selected.
    assert set(selected["node_filter_name"]) == {"cfgA", "cfgB"}

    # Test 6: select_best_candidates flags criteria 1-3 correctly on a
    # hand-built combined_df fixture.
    fixture_combined = pd.DataFrame([
        # beats both baselines and is practical
        {"edge_jaccard": 0.02, "edge_TP": 15, "edge_FP": 800, "edge_FN": 50, "precision": 0.018, "recall": 0.23, "stage": "stage2_tracklet_fp_control", "label": "best"},
        # identical to the 6B baseline exactly - should NOT count as "beating" it (strict >)
        {"edge_jaccard": 0.002215, "edge_TP": 13, "edge_FP": 5700, "edge_FN": 156, "precision": 0.002276, "recall": 0.076923, "stage": "stage1_edge_config", "label": "exactly_6B"},
        # meets nothing
        {"edge_jaccard": 0.0009, "edge_TP": 3, "edge_FP": 9000, "edge_FN": 160, "precision": 0.0003, "recall": 0.018, "stage": "stage1_edge_config", "label": "nothing"},
    ])
    best = select_best_candidates(fixture_combined, csv_path="/tmp/milestone9_best_candidates_test.csv")
    assert len(best) == 1, "only the clearly-improved row should meet any criterion"
    assert best.iloc[0]["label"] == "best"
    assert best.iloc[0]["meets_criterion_1_beat_6B"] and best.iloc[0]["meets_criterion_2_beat_6C6D"] and best.iloc[0]["meets_criterion_3_practical"]

    # Test 7: end-to-end sanity on a clean, well-separated 2-edge fixture -
    # candidate pruning + scoring + greedy null-option assignment should
    # recover both edges perfectly (edge_jaccard=1.0, zero FP) when
    # thresholds are lenient.
    gt_nodes_e2e = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.0, 10.0, 300.0, 310.0], "x": [0.0] * 4,
    })
    gt_edges_e2e = pd.DataFrame({"source_id": [0, 2], "target_id": [1, 3]})
    filtered_nodes_e2e = pd.DataFrame({
        "node_id": [100, 101, 102, 103], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.1, 10.1, 300.1, 310.1], "x": [0.0] * 4,
        "score": [0.9, 0.9, 0.9, 0.9],
    })
    match_filtered_e2e = match_nodes_by_timepoint(filtered_nodes_e2e, gt_nodes_e2e, NODE_MATCH_MAX_DISTANCE_UM)
    node_filter_cache_e2e = {
        "dataset": "e2e_fixture", "filtered_nodes": filtered_nodes_e2e, "match_filtered": match_filtered_e2e,
        "ref_disp": {}, "gt_edges": gt_edges_e2e, "n_timepoints": 2,
    }
    candidates_e2e = build_pruned_candidate_edges_for_sample(filtered_nodes_e2e, {}, max_link_distance_um=50.0, k_next_candidates=2)
    result_e2e = evaluate_edge_config(node_filter_cache_e2e, candidates_e2e, min_edge_score_quantile=0.0)
    assert result_e2e["edge_TP"] == 2 and result_e2e["edge_FP"] == 0 and result_e2e["edge_FN"] == 0

    print("All milestone9_edge_fp_control tests passed (7/7).")


# --------------------------------------------------------------------------- #
# 24. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone9_pipeline(
    n_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    top_n_by_jaccard_for_stage2: int = 20,
) -> dict:
    """Runs the tests, then the full Milestone 9 staged search: caches raw
    detection + GT once per sample, builds the 5 node-filter caches, runs
    Stage 1 (edge candidate/pruning/scoring sweep), selects a shortlist for
    Stage 2 (tracklet-level FP control), runs Stage 2, prints the combined
    summary reports, and selects final best candidates against the 3
    success criteria. Builds no submission. Never calls any Kaggle
    submission API.
    """
    print("=== Self-test: candidate pruning/scoring, null-option assignment, and selection-logic unit tests ===")
    run_milestone9_tests()

    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        empty = pd.DataFrame()
        return {"stage1_df": empty, "stage2_df": empty, "best_candidates_df": empty}

    samples = train_dirs[:n_samples]
    print(f"\n=== Caching raw detection + GT for {len(samples)} sample(s) ===")
    raw_caches = [precompute_raw_sample_cache(s, detector_config) for s in samples]
    for rc in raw_caches:
        print(f"  {rc['dataset']}: {len(rc['raw_nodes_df'])} raw node(s), GT edges={len(rc['gt_edges'])}")

    print(f"\n=== Building node-filter caches for {len(node_filter_configs)} selected config(s) ===")
    node_filter_caches_by_name = build_node_filter_caches(raw_caches, node_filter_configs)
    for name, caches in node_filter_caches_by_name.items():
        total_filtered = sum(len(c["filtered_nodes"]) for c in caches)
        print(f"  {name}: {total_filtered} filtered node(s) total across {len(caches)} sample(s)")

    print("\n=== Stage 1: edge candidate/pruning/scoring sweep ===")
    stage1_df = run_milestone9_stage1(node_filter_caches_by_name, node_filter_configs)

    print("\n=== Stage 1 -> Stage 2 config selection ===")
    stage1_selected = select_stage1_configs_for_stage2(stage1_df, top_n_by_jaccard=top_n_by_jaccard_for_stage2)

    print("\n=== Stage 2: tracklet-level FP control ===")
    stage2_df = run_milestone9_stage2(node_filter_caches_by_name, stage1_selected)

    print("\n=== Summary reports (Stage 1 + Stage 2 combined) ===")
    combined_df = build_combined_candidates(stage1_df, stage2_df)
    print_summary_reports(combined_df)

    print("\n=== Final best-candidate selection ===")
    best_candidates_df = select_best_candidates(combined_df)

    return {"stage1_df": stage1_df, "stage2_df": stage2_df, "best_candidates_df": best_candidates_df}


run_milestone9_pipeline(n_samples=3)
