"""
Biohub - Cell Tracking During Development
Milestone 11: classifier-to-tracklet validation and robust OOF aggregation.

Milestone 10's real Kaggle result was the first positive signal after 6C:
best_oracle_ceiling=0.165680 (candidate pool is sufficient - detector/
candidate generation are NOT the bottleneck), GT graph structure shows
division is rare (division_candidate_frac~=0.0013) and edges are always
t->t+1 (frac_edge_time_gap_skip=0.0) - one-step frame-to-frame linking is
structurally appropriate for now. best_classifier_edge_jaccard~=0.013605
nominally beats 6C/6D's 0.010667, using node_filter=moderate_150_085,
classifier=random_forest, per_frame_top_k=1 budget, held out on ONE
specific sample (44b6_0113de3b): TP=2, FP=97.

That result is NOT yet trustworthy as evidence a real submission would
improve, for two structural reasons this module exists to address:
  1. n=3 leave-one-sample-out CV is extremely high-variance - one sample
     looking good proves nothing about the other ~196 unseen samples.
  2. Milestone 10 only ever evaluated independent per-edge top-K ranking
     - it never enforced that a node has at most one outgoing/incoming
     edge, never built a tracklet, and never filtered tracklets by
     quality. A classifier that separates true/false candidates in
     isolation is not yet validated as producing a coherent tracking
     output.

Milestone 11 validates classifier-ranked edges through the FULL real
pipeline - exclusivity, tracklet construction, tracklet filtering - and
reports TRUE aggregate out-of-fold (OOF) metrics (pooled TP/FP/FN summed
across every held-out sample, never a naive mean of per-sample jaccards,
which is unstable when many samples have zero or near-zero GT-positive
candidates - architecture review) across as many samples as the runtime
budget allows, explicitly checking whether any improvement is STABLE
across multiple held-out samples rather than an artifact of one sample.

Design notes incorporating architecture review:

  - Fold structure: FAST mode = leave-one-sample-out over the first 3
    samples (directly comparable to Milestone 10). ROBUST mode = up to
    N samples (fixed-seed selection: first 3 + stratified by dataset-name
    prefix and GT-node-count quantile - cheap to compute since GT-only
    metadata requires no detection), split via GroupKFold(n_splits=3 or
    5) grouped by dataset name, so a sample's candidates are never split
    across train/val. Exactly ONE classifier is trained per FOLD (not per
    sample) and used to score every held-out sample in that fold
    individually, preserving per-sample metrics/per-frame structure.
    Every fold is checked for degenerate positive prevalence (near-zero
    GT-positive candidates in the held-out set) and flagged loudly rather
    than silently producing an undefined per-fold jaccard.

  - Aggregate OOF = pooled TP/FP/FN summed across ALL held-out samples
    (micro-average), computed ONCE per (node-filter, classifier,
    negative_ratio, edge-selection-policy[+params], tracklet-filter[+
    node-policy]) combination - never a mean of per-sample edge_jaccard
    values. Per-sample rows are ALSO saved separately (the "Stable"
    success criterion needs them) but are not what "aggregate OOF"
    reports.

  - Edge-selection-policy -> tracklet-construction shortlisting: sweeping
    the full tracklet-filter grid (4x4x4x5=320 combos x 2 node policies)
    against every one of Part E's ~20-30 edge-selection-policy+parameter
    combos is intractable, so only a SHORTLIST is carried into Part F.
    The shortlist is deliberately NOT just "top-10 by pre-tracklet
    edge_jaccard" - architecture review flagged that greedy_exclusive_
    per_frame (the only policy that enforces real exclusivity) routinely
    scores WORSE at the raw pre-tracklet edge level than policies that
    admit conflicting edges, precisely because it discards more edges;
    ranking promotion by raw edge_jaccard alone would systematically
    exclude the one policy that matters most for a real tracking output.
    The shortlist is therefore the union of: the single best parameter of
    EACH of the 4 policies (regardless of its raw rank), the overall top
    10 by edge_jaccard, the top 3 by recall, and the top 3 by precision -
    deliberately spanning the precision/recall tradeoff rather than a
    tight top-K band clustered at one operating point. Pre-tracklet
    edge_jaccard is treated only as a proxy for a config's RECALL
    CEILING (tracklet filtering is monotone-subtractive - it only ever
    removes edges/nodes), not as a proxy for where its precision lands
    after filtering.

  - Classifier score as the tracklet-builder's edge_score: the existing
    (unmodified) tracklet-construction code computes each tracklet's
    `mean_edge_score` from whatever numeric value is stored in the
    edges table's `edge_score` column; that column is populated with the
    classifier's predicted P(positive) for each surviving edge, so the
    same tracklet-filter machinery works unmodified with a renamed
    `min_mean_classifier_score_quantile` threshold. Per architecture
    review, RandomForest's tie-heavy, low-cardinality probabilities
    (granularity ~1/n_estimators) mean a nominal quantile cutoff can
    select a very different EFFECTIVE fraction of edges than the same
    nominal quantile does on a smooth classifier's scores - so every
    quantile threshold is resolved from that SAME (classifier, held-out
    sample)'s own score distribution (never pooled across classifiers or
    samples), and the effective count of edges actually kept is logged
    alongside the nominal quantile so this artifact is visible rather
    than silently distorting comparisons.

  - Runtime guard: the expensive step is per-sample raw detection - a
    running average cost-per-sample is tracked while caching samples, and
    if the projected total for the remaining planned samples would
    exceed the time budget, the sample list is truncated early with a
    printed explanation, per the milestone's "default to first 6-10
    samples if robust mode may exceed budget" instruction.

No final submission is built (no /kaggle/working/submission.csv is ever
written). Milestone 6C/6D/9/10's files are not modified - all reused
logic (reader/detector/node-filter/GT-matching/candidate-pool/feature-
engineering from Milestone 10, tracklet-construction/filtering/
reconstruction-policy machinery from Milestone 9/recall_recovery_tracker)
is copied in, not imported.
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

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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


# 9. GT node matching (embedded copy of local_metric.py's evaluator)
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


# 10. Local edge_jaccard evaluation (embedded copy of local_metric.py's evaluator)
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


# --------------------------------------------------------------------------- #
# 11. Reference displacement chain (smoothness prior, independent of any
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
# 12. Ceiling candidate-edge pool (built ONCE per sample/node-filter config,
#     tighter (distance, k) grid points are reproduced by filtering this
#     table - see module docstring for the equivalence argument)
# --------------------------------------------------------------------------- #
CEILING_MAX_LINK_DISTANCE_UM = 7.0
CEILING_K_NEXT_CANDIDATES = 5

CANDIDATE_POOL_COLUMNS = [
    "t", "source_id", "target_id", "distance_um", "local_rank",
    "source_score", "target_score", "smoothness_um",
]


def build_ceiling_candidate_pool(nodes_df: pd.DataFrame, ref_disp: dict) -> pd.DataFrame:
    """Builds the FULL candidate-edge table at the loosest (ceiling) gate -
    max_link_distance_um=7, up to k_next_candidates=5 nearest targets per
    source - so every tighter grid point can be reproduced by filtering
    this same table (`distance_um <= d AND local_rank <= k`) instead of
    rebuilding from scratch.

    `local_rank` is assigned by ascending (distance_um, target_id) - the
    target_id tie-break is a DETERMINISTIC secondary sort key, without
    which two candidates at an identical distance could be ranked
    arbitrarily/non-reproducibly, silently breaking the
    filter-instead-of-rebuild equivalence at the k cutoff (flagged in
    architecture review).
    """
    if len(nodes_df) == 0:
        return pd.DataFrame(columns=CANDIDATE_POOL_COLUMNS)

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
            # Deterministic ordering: ascending (distance, target_id).
            order = np.lexsort((nxt_ids, dist[i]))
            rank = 0
            for j in order:
                distance = float(dist[i, j])
                if distance > CEILING_MAX_LINK_DISTANCE_UM:
                    break
                rank += 1
                if rank > CEILING_K_NEXT_CANDIDATES:
                    break

                target_id = int(nxt_ids[j])
                prev_disp = ref_disp.get(int(source_id))
                if prev_disp is not None:
                    new_disp = (nxt_coords[j] - cur_coords[i]) * _VOXEL_SCALE
                    smoothness_um = float(np.linalg.norm(new_disp - prev_disp))
                else:
                    smoothness_um = float("nan")

                rows.append({
                    "t": int(t), "source_id": int(source_id), "target_id": target_id,
                    "distance_um": distance, "local_rank": rank,
                    "source_score": float(cur_scores[i]), "target_score": float(nxt_scores[j]),
                    "smoothness_um": smoothness_um,
                })

    return pd.DataFrame(rows, columns=CANDIDATE_POOL_COLUMNS)


def filter_candidate_pool(pool_df: pd.DataFrame, max_link_distance_um: float, k_next_candidates: int) -> pd.DataFrame:
    """Reproduces what a from-scratch rebuild at the given (distance, k)
    would have produced, by filtering the ceiling pool - see module
    docstring/architecture review for why this is exact, not approximate.
    """
    if len(pool_df) == 0:
        return pool_df
    return pool_df[(pool_df["distance_um"] <= max_link_distance_um) & (pool_df["local_rank"] <= k_next_candidates)]


def label_candidates(candidates_df: pd.DataFrame, pred_to_gt: dict, gt_edge_set: set) -> pd.Series:
    """Labels each candidate (source_id, target_id) positive iff both
    endpoints are GT-matched AND the mapped (gt_source, gt_target) pair is
    a real GT edge. GT-node matching is one-to-one, so each GT edge maps
    to at most one candidate row.
    """
    if len(candidates_df) == 0:
        return pd.Series([], dtype=bool)
    mapped_source = candidates_df["source_id"].map(pred_to_gt)
    mapped_target = candidates_df["target_id"].map(pred_to_gt)
    pairs = list(zip(mapped_source, mapped_target))
    return pd.Series([
        (s is not None) and (t is not None) and ((s, t) in gt_edge_set) for s, t in pairs
    ], index=candidates_df.index)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 13. Tracklet construction (copied from recall_recovery_tracker.py, unmodified)
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

# --------------------------------------------------------------------------- #
# 14. Tracklet filtering + node-reconstruction policies (copied from
#     recall_recovery_tracker.py, unmodified)
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

# --------------------------------------------------------------------------- #
# 15. Part B/C - candidate labeling + feature engineering
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "dataset", "node_filter_name", "t", "source_id", "target_id",
    "distance_um", "dz_um", "dy_um", "dx_um", "displacement_magnitude_um",
    "normalized_distance_score",
    "source_node_score", "target_node_score", "node_score_product", "node_score_min", "node_score_max",
    "local_rank", "rank_score",
    "source_local_density", "target_local_density",
    "source_frame_node_count", "target_frame_node_count",
    "source_competition_count", "target_competition_count",
    "nearest_target_distance_um", "second_nearest_target_distance_um", "nearest_to_second_nearest_ratio",
    "smoothness_um", "smoothness_score",
    "edge_score",
    "ceiling_max_link_distance_um", "ceiling_k_next_candidates",
    "label",
]

# Features used downstream by the classifiers (Part F) - numeric only,
# excludes identifiers/labels/metadata. NaN-tolerant models (HistGB) see
# these as-is; NaN-intolerant models (LogisticRegression) get them
# median-imputed per training fold (see Part F).
CLASSIFIER_FEATURE_COLUMNS = [
    "distance_um", "dz_um", "dy_um", "dx_um", "displacement_magnitude_um",
    "normalized_distance_score",
    "source_node_score", "target_node_score", "node_score_product", "node_score_min", "node_score_max",
    "local_rank", "rank_score",
    "source_local_density", "target_local_density",
    "source_frame_node_count", "target_frame_node_count",
    "source_competition_count", "target_competition_count",
    "nearest_target_distance_um", "second_nearest_target_distance_um", "nearest_to_second_nearest_ratio",
    "smoothness_um", "smoothness_score", "edge_score",
]


def _compute_local_density(nodes_df: pd.DataFrame, radius_um: float) -> dict:
    """For every node, counts OTHER same-frame nodes within `radius_um` (a
    cheap crowding proxy) via one KD-tree radius query per frame.
    """
    density: dict[int, int] = {}
    for t, group in nodes_df.groupby("t"):
        coords = group[["z", "y", "x"]].to_numpy(dtype=float) * _VOXEL_SCALE
        ids = group["node_id"].to_numpy()
        if len(coords) <= 1:
            for nid in ids:
                density[int(nid)] = 0
            continue
        tree = cKDTree(coords)
        neighbor_lists = tree.query_ball_point(coords, r=radius_um)
        for nid, neighbors in zip(ids, neighbor_lists):
            density[int(nid)] = len(neighbors) - 1  # exclude self
    return density


def build_labeled_feature_table(node_filter_cache: dict) -> pd.DataFrame:
    """Builds the full per-candidate feature table (Part C) + label (Part
    B) for one (sample, node-filter-config)'s ceiling candidate pool. All
    "competition"/"density"/"rank" features reflect the ceiling settings
    (the maximum candidate pool ever considered) - Part D's (distance, k)
    sweep re-derives coverage/labels by filtering rows, not by
    recomputing these features.
    """
    pool = node_filter_cache["ceiling_pool"]
    if len(pool) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    filtered_nodes = node_filter_cache["filtered_nodes"]
    node_lookup = filtered_nodes.set_index("node_id")[["z", "y", "x"]]
    node_count_per_frame = filtered_nodes.groupby("t").size()

    df = pool.copy().reset_index(drop=True)

    source_coords = node_lookup.loc[df["source_id"]].to_numpy(dtype=float)
    target_coords = node_lookup.loc[df["target_id"]].to_numpy(dtype=float)
    disp = (target_coords - source_coords) * _VOXEL_SCALE
    df["dz_um"] = disp[:, 0]
    df["dy_um"] = disp[:, 1]
    df["dx_um"] = disp[:, 2]
    df["displacement_magnitude_um"] = np.linalg.norm(disp, axis=1)

    df["normalized_distance_score"] = 1.0 / (1.0 + df["distance_um"])
    df["source_node_score"] = df["source_score"]
    df["target_node_score"] = df["target_score"]
    df["node_score_product"] = df["source_score"] * df["target_score"]
    df["node_score_min"] = df[["source_score", "target_score"]].min(axis=1)
    df["node_score_max"] = df[["source_score", "target_score"]].max(axis=1)

    df["rank_score"] = 1.0 / df["local_rank"]

    density_by_node = _compute_local_density(filtered_nodes, CEILING_MAX_LINK_DISTANCE_UM)
    df["source_local_density"] = df["source_id"].map(density_by_node).fillna(0).astype(int)
    df["target_local_density"] = df["target_id"].map(density_by_node).fillna(0).astype(int)

    df["source_frame_node_count"] = df["t"].map(node_count_per_frame).fillna(0).astype(int)
    df["target_frame_node_count"] = (df["t"] + 1).map(node_count_per_frame).fillna(0).astype(int)

    df["source_competition_count"] = df.groupby("source_id")["target_id"].transform("count")
    df["target_competition_count"] = df.groupby("target_id")["source_id"].transform("count")

    nearest = df.loc[df["local_rank"] == 1].set_index("source_id")["distance_um"]
    second_nearest = df.loc[df["local_rank"] == 2].set_index("source_id")["distance_um"]
    df["nearest_target_distance_um"] = df["source_id"].map(nearest)
    df["second_nearest_target_distance_um"] = df["source_id"].map(second_nearest)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["nearest_target_distance_um"] / df["second_nearest_target_distance_um"]
    df["nearest_to_second_nearest_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    df["smoothness_score"] = 1.0 / (1.0 + df["smoothness_um"])  # NaN stays NaN (no reference available)

    df["edge_score"] = (
        (df["source_score"] * df["target_score"] / (1.0 + df["distance_um"]))
        * df["rank_score"]
        * df["smoothness_score"].fillna(1.0)
    )

    df["ceiling_max_link_distance_um"] = CEILING_MAX_LINK_DISTANCE_UM
    df["ceiling_k_next_candidates"] = CEILING_K_NEXT_CANDIDATES

    df["dataset"] = node_filter_cache["dataset"]
    df["node_filter_name"] = node_filter_cache["node_filter_name"]

    df["label"] = label_candidates(
        df, node_filter_cache["match_filtered"]["pred_to_gt"], node_filter_cache["gt_edge_set"],
    ).to_numpy()

    return df[FEATURE_COLUMNS].reset_index(drop=True)


def build_all_feature_tables(node_filter_caches_by_name: dict) -> pd.DataFrame:
    """Builds and concatenates the labeled feature table for every (sample,
    node-filter-config) pair - the single master table reused by Parts
    D/E/F.
    """
    parts = []
    for name, caches in node_filter_caches_by_name.items():
        for cache in caches:
            parts.append(build_labeled_feature_table(cache))
    if not parts:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.concat(parts, ignore_index=True)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 16. Part A - sample selection (FAST vs ROBUST modes) + runtime guard
# --------------------------------------------------------------------------- #
FAST_MODE_N_SAMPLES = 3
ROBUST_MODE_N_SAMPLES_DEFAULT = 12
ROBUST_MODE_TIME_BUDGET_SECONDS = 75 * 60
DETECTION_PHASE_BUDGET_FRACTION = 0.5  # reserve the other half for classifiers/tracklets


def precompute_raw_sample_cache(sample_zarr_dir: Path, detector_config: dict = DEFAULT_DETECTOR_CONFIG) -> dict:
    """Runs the raw detector and reads GT exactly ONCE per sample - shared
    across every node-filter config and everything downstream.
    """
    dataset_name = sample_zarr_dir.stem
    gt_nodes, gt_edges = read_geff(sample_zarr_dir)
    raw_nodes_df, _ = detect_all_raw_nodes_for_sample(sample_zarr_dir, detector_config, node_id_start=0)
    n_timepoints = get_sample_timepoint_count(sample_zarr_dir)
    return {
        "dataset": dataset_name, "gt_nodes": gt_nodes, "gt_edges": gt_edges,
        "gt_edge_set": set(zip(gt_edges["source_id"], gt_edges["target_id"])),
        "raw_nodes_df": raw_nodes_df, "n_timepoints": n_timepoints,
    }


def _dataset_prefix(sample_zarr_dir: Path) -> str:
    """The part of the dataset name before the first underscore (e.g.
    "44b6" from "44b6_0113de3b") - used to stratify sample selection
    across different source movies, not just node-count.
    """
    stem = sample_zarr_dir.stem
    return stem.split("_")[0] if "_" in stem else stem


def select_train_samples_fast(train_dirs: Sequence[Path] | None = None) -> list[Path]:
    """FAST mode: exactly the first 3 train samples - directly comparable
    to Milestone 10's n=3 LOSO evaluation.
    """
    if train_dirs is None:
        train_dirs = find_train_zarr_dirs()
    return list(train_dirs[:FAST_MODE_N_SAMPLES])


def select_train_samples_robust(
    train_dirs: Sequence[Path] | None = None, n_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0,
) -> list[Path]:
    """ROBUST mode: a fixed-seed, stratified selection of up to `n_samples`
    train samples - always includes the first 3 (for direct comparability
    to FAST mode/Milestone 10), then fills remaining slots by drawing
    across distinct dataset-name prefixes and GT-node-count quartiles
    (a cheap GT-only read - no detection needed for SELECTION itself; the
    expensive detector only ever runs on the samples this function
    actually returns).
    """
    if train_dirs is None:
        train_dirs = find_train_zarr_dirs()
    train_dirs = list(train_dirs)
    if len(train_dirs) <= n_samples:
        return train_dirs

    first_three = train_dirs[:FAST_MODE_N_SAMPLES]
    remaining_pool = [d for d in train_dirs if d not in first_three]

    gt_node_counts = {}
    for d in remaining_pool:
        try:
            gt_nodes, _ = read_geff(d)
            gt_node_counts[d] = len(gt_nodes)
        except Exception:
            gt_node_counts[d] = 0

    counts_series = pd.Series(gt_node_counts)
    try:
        quantile_bins = pd.qcut(counts_series, q=4, labels=False, duplicates="drop")
    except ValueError:
        quantile_bins = pd.Series(0, index=counts_series.index)

    strata: dict[tuple, list[Path]] = {}
    for d in remaining_pool:
        key = (_dataset_prefix(d), int(quantile_bins.get(d, 0)))
        strata.setdefault(key, []).append(d)

    rng = np.random.default_rng(seed)
    strata_keys = sorted(strata.keys())
    for key in strata_keys:
        rng.shuffle(strata[key])

    selected: list[Path] = list(first_three)
    n_remaining_needed = n_samples - len(selected)
    key_idx = 0
    guard = 0
    while n_remaining_needed > 0 and any(strata[k] for k in strata_keys) and guard < 10000:
        key = strata_keys[key_idx % len(strata_keys)]
        if strata[key]:
            selected.append(strata[key].pop())
            n_remaining_needed -= 1
        key_idx += 1
        guard += 1

    return selected


def select_train_samples(mode: str, n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0) -> list[Path]:
    train_dirs = find_train_zarr_dirs()
    if mode == "fast":
        selected = select_train_samples_fast(train_dirs)
    elif mode == "robust":
        selected = select_train_samples_robust(train_dirs, n_samples=n_robust_samples, seed=seed)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'fast' or 'robust'")
    print(f"Selected {len(selected)} train sample(s) for mode={mode!r}: {[d.stem for d in selected]}")
    return selected


def cache_samples_within_budget(
    sample_dirs: Sequence[Path],
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
    detection_phase_fraction: float = DETECTION_PHASE_BUDGET_FRACTION,
) -> list[dict]:
    """Caches raw detection + GT for each sample IN ORDER, tracking a
    running average cost-per-sample; if the projected total for all
    remaining planned samples would exceed the detection-phase budget
    (reserving the rest of `time_budget_seconds` for classifier training/
    evaluation/tracklet construction), the sample list is truncated early
    with a printed explanation - "default to first 6-10 samples if robust
    mode may exceed budget."
    """
    detection_budget = time_budget_seconds * detection_phase_fraction
    caches: list[dict] = []
    elapsed_total = 0.0
    for i, sample_dir in enumerate(sample_dirs):
        t0 = time.time()
        cache = precompute_raw_sample_cache(sample_dir, detector_config)
        dt = time.time() - t0
        elapsed_total += dt
        caches.append(cache)
        avg_per_sample = elapsed_total / (i + 1)
        n_remaining = len(sample_dirs) - (i + 1)
        projected_total = elapsed_total + avg_per_sample * n_remaining
        print(
            f"  [{i + 1}/{len(sample_dirs)}] {cache['dataset']}: cached in {dt:.1f}s "
            f"(avg {avg_per_sample:.1f}s/sample, projected total {projected_total:.1f}s of {detection_budget:.1f}s budget)"
        )
        if projected_total > detection_budget and n_remaining > 0:
            print(
                f"  [runtime guard] projected detection time ({projected_total:.1f}s) would exceed the "
                f"detection-phase budget ({detection_budget:.1f}s) - stopping early with {len(caches)} sample(s) "
                f"cached instead of the originally planned {len(sample_dirs)}."
            )
            break
    return caches

# --------------------------------------------------------------------------- #
# 17. Part B - node-filter configs + per-sample/per-node-filter caching
# --------------------------------------------------------------------------- #
SELECTED_NODE_FILTER_CONFIGS = [
    {"name": "moderate_150_085", "max_nodes_per_timepoint": 150, "score_quantile_per_timepoint": 0.85, "absolute_score_threshold": 0.05},
    {"name": "m8_best_tradeoff", "max_nodes_per_timepoint": 75, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.03},
]

NODE_MATCH_MAX_DISTANCE_UM = 7.0


def precompute_node_filter_cache(
    raw_cache: dict, node_filter_config: dict, match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """For ONE (sample, node-filter-config) pair: applies node filtering,
    matches filtered nodes to GT ONCE, builds the reference-displacement
    chain ONCE, and builds the ceiling candidate-edge pool ONCE - every
    downstream feature/label/tracklet computation reuses this same cached
    pool. Unlike Milestone 10's version, `gt_nodes` is RETAINED (not just
    `gt_edges`/`gt_edge_set`) - Part G's node-level metrics (node_matches,
    unmatched_gt_nodes, unmatched_pred_nodes) need it per held-out sample.
    """
    filtered_nodes = apply_node_filter(
        raw_cache["raw_nodes_df"],
        node_filter_config["max_nodes_per_timepoint"],
        node_filter_config["score_quantile_per_timepoint"],
        node_filter_config["absolute_score_threshold"],
    )[["node_id", "t", "z", "y", "x", "score"]].reset_index(drop=True)

    match_filtered = match_nodes_by_timepoint(filtered_nodes, raw_cache["gt_nodes"], match_max_distance_um)
    ref_disp = build_reference_displacement_by_node(filtered_nodes)
    pool = build_ceiling_candidate_pool(filtered_nodes, ref_disp)

    return {
        "dataset": raw_cache["dataset"],
        "node_filter_name": node_filter_config["name"],
        "gt_nodes": raw_cache["gt_nodes"],
        "filtered_nodes": filtered_nodes,
        "match_filtered": match_filtered,
        "gt_edges": raw_cache["gt_edges"],
        "gt_edge_set": raw_cache["gt_edge_set"],
        "gt_edge_count": len(raw_cache["gt_edges"]),
        "n_timepoints": raw_cache["n_timepoints"],
        "ceiling_pool": pool,
    }


def build_node_filter_caches(
    raw_caches: list[dict], node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Builds `{node_filter_name: [per_sample_cache, ...]}` for every
    selected node-filter config, once, up front.
    """
    caches_by_name: dict[str, list[dict]] = {}
    for node_filter_config in node_filter_configs:
        name = node_filter_config["name"]
        caches_by_name[name] = [
            precompute_node_filter_cache(rc, node_filter_config, match_max_distance_um) for rc in raw_caches
        ]
    return caches_by_name

# --------------------------------------------------------------------------- #
# 18. Part C - validation split (LOSO for FAST mode, GroupKFold for ROBUST)
# --------------------------------------------------------------------------- #
MIN_FOLD_VAL_POSITIVE_COUNT = 3


def build_loso_folds(dataset_names: Sequence[str]) -> list[dict]:
    """FAST mode: leave-one-sample-out - one fold per dataset, val = exactly
    that one dataset, train = every other dataset.
    """
    dataset_names = list(dataset_names)
    return [
        {"fold_id": held_out, "train_datasets": [d for d in dataset_names if d != held_out], "val_datasets": [held_out]}
        for held_out in dataset_names
    ]


def build_group_kfold_folds(dataset_names: Sequence[str], n_splits: int = 3) -> list[dict]:
    """ROBUST mode: GroupKFold grouped by dataset name - a dataset's
    candidates/features never split across train/val within a fold.
    """
    dataset_names = list(dataset_names)
    n_splits = max(2, min(n_splits, len(dataset_names)))
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(dataset_names), 1))
    folds = []
    for i, (train_idx, val_idx) in enumerate(gkf.split(dummy_X, groups=dataset_names)):
        folds.append({
            "fold_id": f"group_fold_{i}",
            "train_datasets": [dataset_names[j] for j in train_idx],
            "val_datasets": [dataset_names[j] for j in val_idx],
        })
    return folds


def build_folds(mode: str, dataset_names: Sequence[str], n_splits: int = 3) -> list[dict]:
    """Dispatches to LOSO (fast) or GroupKFold (robust, n_splits=3 if
    fewer than 10 datasets else 5), then prints the fold structure.
    """
    dataset_names = list(dataset_names)
    if mode == "fast":
        folds = build_loso_folds(dataset_names)
    elif mode == "robust":
        effective_n_splits = 5 if len(dataset_names) >= 10 else 3
        folds = build_group_kfold_folds(dataset_names, n_splits=effective_n_splits)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'fast' or 'robust'")

    print(f"Built {len(folds)} fold(s) for mode={mode!r}:")
    for fold in folds:
        print(f"  fold {fold['fold_id']}: train={len(fold['train_datasets'])} sample(s), val={fold['val_datasets']}")
    return folds


def compute_positive_counts_by_dataset(feature_df: pd.DataFrame) -> dict:
    """Number of positive (label=True) candidate rows per dataset, pooled
    across whichever node-filter config's feature table is passed in -
    used only to flag degenerate folds, not for any metric.
    """
    if feature_df.empty:
        return {}
    return {k: int(v) for k, v in feature_df.groupby("dataset")["label"].sum().items()}


def annotate_fold_degeneracy(
    folds: list[dict], positive_counts_by_dataset: dict, min_positive_count: int = MIN_FOLD_VAL_POSITIVE_COUNT,
) -> list[dict]:
    """Flags any fold whose held-out set has near-zero GT-positive
    candidates - such a fold's per-fold edge_jaccard is unstable/
    near-undefined and must not be silently pooled without a loud warning.
    """
    for fold in folds:
        val_positive_total = sum(positive_counts_by_dataset.get(d, 0) for d in fold["val_datasets"])
        fold["val_positive_count"] = val_positive_total
        fold["degenerate"] = val_positive_total < min_positive_count
        if fold["degenerate"]:
            print(
                f"  [degenerate fold warning] fold {fold['fold_id']}: only {val_positive_total} positive "
                f"candidate(s) in held-out set {fold['val_datasets']} - per-fold jaccard will be unstable."
            )
    return folds

# --------------------------------------------------------------------------- #
# 19. Part D - classifiers: hard-negative sampling + one-classifier-per-fold
#     training + per-held-out-sample scoring
# --------------------------------------------------------------------------- #
NEGATIVE_RATIO_VALUES = (25, 50, 100)
HARD_NEGATIVE_LOCAL_RANK_CEILING = 3

CLASSIFIER_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0),
}
# HistGradientBoostingClassifier accepts NaN features natively; the other
# three need median imputation (computed from the TRAIN fold only, applied
# to both train and validation, to avoid leaking validation-set statistics).
CLASSIFIERS_NEEDING_IMPUTATION = {"logistic_regression", "random_forest", "extra_trees"}


def hard_negative_sample(
    train_df: pd.DataFrame, negative_ratio: int, random_state: int = 0,
) -> pd.DataFrame:
    """All positives + hard negatives (local_rank<=3 - which also covers
    "nearest false candidates" since rank 1 is the nearest -, plus the
    highest node_score_product and highest edge_score negatives) + a
    random top-up from the remaining negatives, capped at
    `negative_ratio x n_positive` total negatives.
    """
    positives = train_df[train_df["label"]]
    negatives = train_df[~train_df["label"]]
    n_positive = max(len(positives), 1)
    cap = int(negative_ratio * n_positive)

    rank_hard = negatives[negatives["local_rank"] <= HARD_NEGATIVE_LOCAL_RANK_CEILING]
    node_score_hard = negatives.nlargest(min(len(negatives), cap), "node_score_product") if cap > 0 else negatives.iloc[0:0]
    edge_score_hard = negatives.nlargest(min(len(negatives), cap), "edge_score") if cap > 0 else negatives.iloc[0:0]
    hard = pd.concat([rank_hard, node_score_hard, edge_score_hard])
    hard = hard[~hard.index.duplicated()]

    rng = np.random.default_rng(random_state)
    if len(hard) >= cap:
        idx = rng.choice(hard.index.to_numpy(), size=cap, replace=False) if cap > 0 else np.array([], dtype=hard.index.dtype)
        sampled_negatives = hard.loc[idx]
    else:
        remaining = negatives.drop(hard.index)
        n_fill = min(len(remaining), cap - len(hard))
        if n_fill > 0:
            fill_idx = rng.choice(remaining.index.to_numpy(), size=n_fill, replace=False)
            sampled_negatives = pd.concat([hard, remaining.loc[fill_idx]])
        else:
            sampled_negatives = hard

    return pd.concat([positives, sampled_negatives]).reset_index(drop=True)


def train_and_score_fold(
    fold: dict, subset: pd.DataFrame, classifier_name: str, negative_ratio: int, random_state: int = 0,
) -> dict | None:
    """Trains exactly ONE classifier on `fold['train_datasets']` (hard-
    negative sampled) and scores EACH of `fold['val_datasets']`
    individually against its complete, unsampled candidate pool - held-out
    validation negatives are never resampled. Returns None if the training
    fold is degenerate (a single class after sampling).
    """
    train_df = subset[subset["dataset"].isin(fold["train_datasets"])]
    if train_df["label"].nunique() < 2:
        return None

    train_sample = hard_negative_sample(train_df, negative_ratio, random_state=random_state)
    if train_sample["label"].nunique() < 2:
        return None

    X_train_raw = train_sample[CLASSIFIER_FEATURE_COLUMNS]
    y_train = train_sample["label"].to_numpy(dtype=int)
    medians = X_train_raw.median()
    X_train_imputed = X_train_raw.fillna(medians)

    needs_imputation = classifier_name in CLASSIFIERS_NEEDING_IMPUTATION
    clf = CLASSIFIER_BUILDERS[classifier_name]()
    clf.fit(X_train_imputed if needs_imputation else X_train_raw, y_train)

    n_train_pos = int(y_train.sum())
    n_train_neg = int(len(y_train) - n_train_pos)

    scored_by_dataset: dict[str, pd.DataFrame] = {}
    for val_dataset in fold["val_datasets"]:
        val_df = subset[subset["dataset"] == val_dataset].copy()
        if val_df.empty:
            continue
        X_val_raw = val_df[CLASSIFIER_FEATURE_COLUMNS]
        X_val = X_val_raw.fillna(medians) if needs_imputation else X_val_raw
        val_df["_score"] = clf.predict_proba(X_val)[:, 1]
        scored_by_dataset[val_dataset] = val_df

    return {
        "classifier": classifier_name, "negative_ratio": negative_ratio,
        "n_train_positive": n_train_pos, "n_train_negative": n_train_neg,
        "scored_by_dataset": scored_by_dataset,
    }

# --------------------------------------------------------------------------- #
# 20. Part E - edge-selection policies (independent ranking vs real
#     source/target exclusivity)
# --------------------------------------------------------------------------- #
GLOBAL_TOP_K_BUDGETS = (100, 200, 500, 1000, 2000)
PER_FRAME_TOP_K_VALUES = (1, 2, 3, 5, 10)
SCORE_QUANTILES = (0.90, 0.95, 0.97, 0.99, 0.995)
GREEDY_SOURCE_CAPACITY_DEFAULT = 1  # divisions are rare (M10 GT diagnosis); 2 is a diagnostic-only option

EDGE_SELECTION_POLICY_PARAM_GRIDS = {
    "global_top_k": GLOBAL_TOP_K_BUDGETS,
    "per_frame_top_k": PER_FRAME_TOP_K_VALUES,
    "per_source_top1_with_score_quantile": SCORE_QUANTILES,
    "greedy_exclusive_per_frame": SCORE_QUANTILES,
}


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    return precision, recall


def _evaluate_edges(selected_df: pd.DataFrame, gt_edge_count: int) -> dict:
    tp = int(selected_df["label"].sum()) if len(selected_df) else 0
    fp = len(selected_df) - tp
    fn = gt_edge_count - tp
    denom = tp + fp + fn
    edge_jaccard = tp / denom if denom > 0 else 1.0
    precision, recall = _precision_recall(tp, fp, fn)
    return {"edge_TP": tp, "edge_FP": fp, "edge_FN": fn, "edge_jaccard": edge_jaccard,
            "precision": precision, "recall": recall, "n_selected": len(selected_df)}


def select_global_top_k(scored_df: pd.DataFrame, budget: int) -> pd.DataFrame:
    return scored_df.sort_values("_score", ascending=False).head(budget)


def select_per_frame_top_k(scored_df: pd.DataFrame, per_frame_k: int) -> pd.DataFrame:
    ordered = scored_df.sort_values(["t", "_score"], ascending=[True, False])
    return ordered.groupby("t", sort=False, group_keys=False).head(per_frame_k)


def select_per_source_top1_with_quantile(scored_df: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, float]:
    """Per source node, keep only its single best (highest-score)
    outgoing candidate, then gate on that candidate's score being at or
    above the given quantile of THIS (classifier, sample)'s own score
    distribution - never a pooled/global threshold (Opus review: RandomForest's
    tie-heavy probabilities make pooled quantile thresholds distort
    comparisons across classifiers/samples).
    """
    threshold = float(scored_df["_score"].quantile(quantile))
    top1 = scored_df.sort_values("_score", ascending=False).groupby("source_id", sort=False).head(1)
    kept = top1[top1["_score"] >= threshold]
    return kept, threshold


def select_greedy_exclusive_per_frame(
    scored_df: pd.DataFrame, quantile: float, source_capacity: int = GREEDY_SOURCE_CAPACITY_DEFAULT,
) -> tuple[pd.DataFrame, float]:
    """Sorts ALL candidates (across every frame transition) by classifier
    score descending, greedily accepts an edge only if its source still has
    spare outgoing capacity (default 1 - divisions are rare) AND its target
    has no accepted incoming edge yet, both scoped to that (t, node) pair
    since edges only ever connect frame t to t+1. This is the only policy
    of the 4 that enforces real one-to-one(-ish) exclusivity, and per Opus
    review it can look WORSE than the other 3 at the raw pre-tracklet
    edge_jaccard precisely because it discards more edges - its value shows
    up after tracklet construction (Part F), not before.
    """
    threshold = float(scored_df["_score"].quantile(quantile))
    candidates = scored_df[scored_df["_score"] >= threshold].sort_values("_score", ascending=False)
    source_used: dict = {}
    target_used: set = set()
    kept_idx = []
    for row in candidates.itertuples():
        key_src = (row.t, row.source_id)
        key_tgt = (row.t, row.target_id)
        if source_used.get(key_src, 0) >= source_capacity:
            continue
        if key_tgt in target_used:
            continue
        kept_idx.append(row.Index)
        source_used[key_src] = source_used.get(key_src, 0) + 1
        target_used.add(key_tgt)
    kept = candidates.loc[kept_idx]
    return kept, threshold


def evaluate_edge_selection_policies(scored_df: pd.DataFrame, gt_edge_count: int) -> list[dict]:
    """Applies all 4 edge-selection policies (with their full param grids)
    to one classifier-scored candidate pool. Each returned row carries both
    the selection metadata (quantile threshold/effective selected-edge
    count) and the resulting edge metrics, plus the actual selected-edges
    dataframe under `_selected_edges` - needed by Part F's tracklet
    construction, and popped out before any row is written to a metrics CSV.
    """
    rows: list[dict] = []
    for budget in GLOBAL_TOP_K_BUDGETS:
        selected = select_global_top_k(scored_df, budget)
        rows.append({"policy": "global_top_k", "param": budget, "score_threshold": None,
                     **_evaluate_edges(selected, gt_edge_count), "_selected_edges": selected})
    for k in PER_FRAME_TOP_K_VALUES:
        selected = select_per_frame_top_k(scored_df, k)
        rows.append({"policy": "per_frame_top_k", "param": k, "score_threshold": None,
                     **_evaluate_edges(selected, gt_edge_count), "_selected_edges": selected})
    for q in SCORE_QUANTILES:
        selected, threshold = select_per_source_top1_with_quantile(scored_df, q)
        rows.append({"policy": "per_source_top1_with_score_quantile", "param": q, "score_threshold": threshold,
                     **_evaluate_edges(selected, gt_edge_count), "_selected_edges": selected})
    for q in SCORE_QUANTILES:
        selected, threshold = select_greedy_exclusive_per_frame(scored_df, q)
        rows.append({"policy": "greedy_exclusive_per_frame", "param": q, "score_threshold": threshold,
                     **_evaluate_edges(selected, gt_edge_count), "_selected_edges": selected})
    return rows

# --------------------------------------------------------------------------- #
# 21. Part F - tracklet construction from classifier-selected edges +
#     tracklet-filter sweep + two node-reconstruction policies
# --------------------------------------------------------------------------- #
TRACKLET_MIN_LENGTH_VALUES = (2, 3, 4, 5)
TRACKLET_MIN_MEAN_CLASSIFIER_SCORE_QUANTILE_VALUES = (0.0, 0.5, 0.7, 0.9)
TRACKLET_MAX_SMOOTHNESS_ERROR_UM_VALUES = (3, 5, 7, 10)
TRACKLET_KEEP_TOP_K_VALUES = (25, 50, 100, 200, 500)
TRACKLET_NODE_POLICIES = ("A_tracklet_nodes_only", "B_all_filtered_nodes_with_classifier_edges")


def enforce_edge_exclusivity(selected_edges_df: pd.DataFrame) -> pd.DataFrame:
    """Re-applies source/target exclusivity (capacity=1, highest-score
    first) to an ARBITRARY pre-selected edge set. `build_tracklets`
    structurally assumes at most one outgoing/incoming edge per node (a
    plain dict silently overwrites duplicates otherwise) - every one of
    Part E's 4 edge-selection policies (including the 3 that do NOT
    natively enforce exclusivity) is therefore pushed through this SAME
    final exclusivity gate before tracklet construction, so the
    across-policy comparison is honest rather than silently corrupted by
    dict-overwrite behavior for the non-exclusive policies. This is
    precisely the "source/target exclusivity" validation step this
    milestone exists to check.
    """
    if len(selected_edges_df) == 0:
        return selected_edges_df
    ordered = selected_edges_df.sort_values("_score", ascending=False)
    source_used: set = set()
    target_used: set = set()
    kept_idx = []
    for row in ordered.itertuples():
        if row.source_id in source_used or row.target_id in target_used:
            continue
        kept_idx.append(row.Index)
        source_used.add(row.source_id)
        target_used.add(row.target_id)
    return ordered.loc[kept_idx]


def build_tracklets_from_selected_edges(selected_edges_df: pd.DataFrame, filtered_nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Renames the classifier's `_score` column to `edge_score` (what the
    unmodified tracklet-construction machinery reads as `mean_edge_score`'s
    input), re-applies source/target exclusivity, then builds tracklets
    with the existing (copied, unmodified) `build_tracklets`.
    """
    if len(selected_edges_df) == 0:
        return pd.DataFrame(columns=TRACKLET_COLUMNS)
    exclusive_edges = enforce_edge_exclusivity(selected_edges_df)
    # Built as a fresh frame (not a rename+select) because the source feature
    # table already has ITS OWN legacy "edge_score" column (M10's hand-crafted
    # heuristic) - a rename would collide into two same-named columns, and
    # selecting "edge_score" would then silently hand back a 2-column
    # DataFrame instead of a Series, corrupting the zip() inside build_tracklets.
    edges_for_tracklets = pd.DataFrame({
        "source_id": exclusive_edges["source_id"].to_numpy(),
        "target_id": exclusive_edges["target_id"].to_numpy(),
        "distance_um": exclusive_edges["distance_um"].to_numpy(),
        "edge_score": exclusive_edges["_score"].to_numpy(),
    })
    return build_tracklets(filtered_nodes_df, edges_for_tracklets)


def evaluate_tracklet_config(
    tracklets_df: pd.DataFrame, node_filter_cache: dict,
    min_tracklet_length: int, min_mean_classifier_score_quantile: float,
    max_smoothness_error_um: float, keep_top_k: int, node_policy: str,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Applies the tracklet filter (with the classifier-score quantile
    resolved from THIS tracklet table's OWN mean_edge_score distribution -
    never pooled across classifiers/samples, per Opus review), reconstructs
    predicted nodes/edges under the given node policy, and computes the
    real edge_jaccard + node-level metrics against GT. For node policy B
    the node set never changes (it's always every filtered node), so the
    node-filter cache's precomputed `match_filtered` is reused rather than
    rerun; for policy A the node subset changes per config, so GT matching
    is recomputed on that subset.
    """
    min_mean_edge_score_threshold = (
        float(tracklets_df["mean_edge_score"].quantile(min_mean_classifier_score_quantile))
        if len(tracklets_df) else 0.0
    )
    kept = apply_tracklet_filter(
        tracklets_df, min_tracklet_length=min_tracklet_length, min_mean_node_score=0.0,
        min_mean_edge_score_threshold=min_mean_edge_score_threshold,
        max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM, max_smoothness_error_um=max_smoothness_error_um,
        keep_top_k=keep_top_k,
    )

    filtered_nodes = node_filter_cache["filtered_nodes"]
    gt_nodes = node_filter_cache["gt_nodes"]
    gt_edges = node_filter_cache["gt_edges"]

    if node_policy == "A_tracklet_nodes_only":
        pred_nodes, pred_edges = reconstruct_policy_a(kept, filtered_nodes)
        match = match_nodes_by_timepoint(pred_nodes, gt_nodes, match_max_distance_um)
    else:
        pred_nodes, pred_edges = reconstruct_policy_b(filtered_nodes, kept)
        match = node_filter_cache["match_filtered"]

    edge_result = compute_edge_jaccard(pred_edges, gt_edges, match["pred_to_gt"])
    precision, recall = _precision_recall(edge_result["edge_TP"], edge_result["edge_FP"], edge_result["edge_FN"])
    n_timepoints = max(node_filter_cache["n_timepoints"], 1)

    return {
        **edge_result, "precision": precision, "recall": recall,
        "node_matches": len(match["pred_to_gt"]), "unmatched_gt_nodes": len(match["unmatched_gt_nodes"]),
        "unmatched_pred_nodes": len(match["unmatched_pred_nodes"]),
        "avg_pred_nodes_per_timepoint": len(pred_nodes) / n_timepoints,
        "avg_edges_per_frame": len(pred_edges) / n_timepoints,
        "num_tracklets_kept": len(kept),
        "min_mean_edge_score_threshold": min_mean_edge_score_threshold,
    }


def run_tracklet_filter_sweep(
    tracklets_df: pd.DataFrame, node_filter_cache: dict,
    min_tracklet_length_values: Sequence[int] = TRACKLET_MIN_LENGTH_VALUES,
    min_mean_classifier_score_quantile_values: Sequence[float] = TRACKLET_MIN_MEAN_CLASSIFIER_SCORE_QUANTILE_VALUES,
    max_smoothness_error_um_values: Sequence[float] = TRACKLET_MAX_SMOOTHNESS_ERROR_UM_VALUES,
    keep_top_k_values: Sequence[int] = TRACKLET_KEEP_TOP_K_VALUES,
    node_policies: Sequence[str] = TRACKLET_NODE_POLICIES,
) -> list[dict]:
    """The full 4x4x4x5=320-combo tracklet-filter grid x 2 node policies -
    only ever run against a SHORTLISTED set of (fold, dataset, edge-
    selection-policy) tracklet tables, never the full pre-shortlist grid.
    """
    rows: list[dict] = []
    for node_policy in node_policies:
        for min_len in min_tracklet_length_values:
            for score_q in min_mean_classifier_score_quantile_values:
                for max_smooth in max_smoothness_error_um_values:
                    for keep_k in keep_top_k_values:
                        result = evaluate_tracklet_config(
                            tracklets_df, node_filter_cache, min_len, score_q, max_smooth, keep_k, node_policy,
                        )
                        rows.append({
                            "node_policy": node_policy, "min_tracklet_length": min_len,
                            "min_mean_classifier_score_quantile": score_q,
                            "max_smoothness_error_um": max_smooth, "keep_top_k_tracklets_per_sample": keep_k,
                            **result,
                        })
    return rows

# --------------------------------------------------------------------------- #
# 22. Orchestration - aggregate OOF, shortlisting, and the main fold/
#     classifier/edge-policy/tracklet sweep loop
# --------------------------------------------------------------------------- #
def aggregate_oof(per_sample_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Pooled TP/FP/FN summed across ALL held-out samples within each
    group (micro-average) - NEVER a mean of per-sample edge_jaccard values,
    which is unstable when many samples have zero or near-zero GT-positive
    candidates. Also reports `n_samples` and `n_samples_with_tp` per group
    - the latter is what the "Stable" success criterion (improvement on
    more than one held-out sample) is checked against.
    """
    if per_sample_df.empty:
        return per_sample_df.copy()
    rows: list[dict] = []
    for keys, g in per_sample_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tp, fp, fn = int(g["edge_TP"].sum()), int(g["edge_FP"].sum()), int(g["edge_FN"].sum())
        denom = tp + fp + fn
        edge_jaccard = tp / denom if denom > 0 else 1.0
        precision, recall = _precision_recall(tp, fp, fn)
        n_samples = int(g["held_out_dataset"].nunique())
        n_samples_with_tp = int((g.groupby("held_out_dataset")["edge_TP"].sum() > 0).sum())
        row = dict(zip(group_cols, keys))
        row.update({
            "edge_TP": tp, "edge_FP": fp, "edge_FN": fn, "edge_jaccard": edge_jaccard,
            "precision": precision, "recall": recall,
            "n_samples": n_samples, "n_samples_with_tp": n_samples_with_tp,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def select_shortlist_for_tracklets(
    oof_edge_policy_df: pd.DataFrame, top_n_jaccard: int = 10, top_n_recall: int = 3, top_n_precision: int = 3,
) -> pd.DataFrame:
    """The union of: the single best-by-jaccard param of EACH edge-
    selection policy (regardless of its raw rank), the overall top
    `top_n_jaccard` by edge_jaccard, the top `top_n_recall` by recall, and
    the top `top_n_precision` by precision. Deliberately NOT just a tight
    top-K band by raw edge_jaccard - per Opus review, `greedy_exclusive_
    per_frame` (the only policy enforcing real exclusivity) routinely
    scores worse pre-tracklet precisely because it discards more edges, so
    ranking by raw edge_jaccard alone would systematically exclude the one
    policy that matters most for a real tracking output.
    """
    if oof_edge_policy_df.empty:
        return oof_edge_policy_df.copy()
    shortlist_idx: set = set()
    for policy in oof_edge_policy_df["policy"].unique():
        policy_rows = oof_edge_policy_df[oof_edge_policy_df["policy"] == policy]
        shortlist_idx.add(policy_rows["edge_jaccard"].idxmax())
    shortlist_idx.update(oof_edge_policy_df.nlargest(top_n_jaccard, "edge_jaccard").index)
    shortlist_idx.update(oof_edge_policy_df.nlargest(top_n_recall, "recall").index)
    shortlist_idx.update(oof_edge_policy_df.nlargest(top_n_precision, "precision").index)
    return oof_edge_policy_df.loc[sorted(shortlist_idx)].reset_index(drop=True)


def run_milestone11_validation(
    mode: str = "fast",
    node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    classifier_builders: dict = CLASSIFIER_BUILDERS,
    negative_ratio_values: Sequence[int] = NEGATIVE_RATIO_VALUES,
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT,
    seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
) -> dict:
    """The full Milestone 11 pipeline: Part A sample selection, Part B
    node-filter caching + feature engineering, Part C fold construction,
    then for every (node_filter, fold, classifier, negative_ratio) - one
    classifier trained per fold, scored per held-out sample individually -
    every Part E edge-selection policy is evaluated, pooled into aggregate
    OOF, shortlisted, and only the shortlisted combos are pushed through
    Part F's tracklet-construction + filter sweep (both node policies).
    """
    t_pipeline_start = time.time()

    selected_dirs = select_train_samples(mode, n_robust_samples=n_robust_samples, seed=seed)
    raw_caches = cache_samples_within_budget(selected_dirs, time_budget_seconds=time_budget_seconds)
    dataset_names = [rc["dataset"] for rc in raw_caches]
    print(f"Cached raw detection for {len(raw_caches)} sample(s): {dataset_names}")

    node_filter_caches_by_name = build_node_filter_caches(raw_caches, node_filter_configs)
    for name, caches in node_filter_caches_by_name.items():
        total_candidates = sum(len(c["ceiling_pool"]) for c in caches)
        print(f"  node_filter={name}: {total_candidates} ceiling candidate edge(s) across {len(caches)} sample(s)")

    feature_df = build_all_feature_tables(node_filter_caches_by_name)
    n_positive = int(feature_df["label"].sum()) if not feature_df.empty else 0
    print(f"Built {len(feature_df)} labeled candidate row(s), {n_positive} positive")

    folds = build_folds(mode, dataset_names)
    positive_counts_by_dataset = compute_positive_counts_by_dataset(feature_df)
    folds = annotate_fold_degeneracy(folds, positive_counts_by_dataset)

    gt_edge_count_by_name_dataset = {
        (name, c["dataset"]): c["gt_edge_count"] for name, caches in node_filter_caches_by_name.items() for c in caches
    }
    node_filter_cache_by_name_dataset = {
        (name, c["dataset"]): c for name, caches in node_filter_caches_by_name.items() for c in caches
    }

    per_sample_edge_policy_rows: list[dict] = []
    shortlist_source_data: dict = {}

    t0 = time.time()
    for node_filter_config in node_filter_configs:
        name = node_filter_config["name"]
        subset = feature_df[feature_df["node_filter_name"] == name]
        if subset.empty:
            continue
        for fold in folds:
            if fold.get("degenerate"):
                continue
            for clf_name in classifier_builders:
                for neg_ratio in negative_ratio_values:
                    fold_result = train_and_score_fold(fold, subset, clf_name, neg_ratio, random_state=seed)
                    if fold_result is None:
                        continue
                    for held_out_dataset, scored_val in fold_result["scored_by_dataset"].items():
                        gt_edge_count = gt_edge_count_by_name_dataset[(name, held_out_dataset)]
                        for r in evaluate_edge_selection_policies(scored_val, gt_edge_count):
                            selected_edges = r.pop("_selected_edges")
                            per_sample_edge_policy_rows.append({
                                "node_filter_name": name, "fold_id": fold["fold_id"], "held_out_dataset": held_out_dataset,
                                "classifier": clf_name, "negative_ratio": neg_ratio,
                                "n_train_positive": fold_result["n_train_positive"],
                                "n_train_negative": fold_result["n_train_negative"],
                                **r,
                            })
                            shortlist_key = (name, clf_name, neg_ratio, r["policy"], r["param"], held_out_dataset)
                            shortlist_source_data[shortlist_key] = {
                                "selected_edges": selected_edges,
                                "node_filter_cache": node_filter_cache_by_name_dataset[(name, held_out_dataset)],
                            }
    print(f"Part D/E: fold x classifier x negative_ratio x edge-policy sweep done in {time.time() - t0:.1f}s, "
          f"{len(per_sample_edge_policy_rows)} row(s)")

    per_sample_edge_policy_df = pd.DataFrame(per_sample_edge_policy_rows)
    oof_group_cols = ["node_filter_name", "classifier", "negative_ratio", "policy", "param"]
    oof_edge_policy_df = aggregate_oof(per_sample_edge_policy_df, oof_group_cols)

    shortlist_df = select_shortlist_for_tracklets(oof_edge_policy_df)
    print(f"Shortlisted {len(shortlist_df)} (node_filter, classifier, negative_ratio, policy, param) combo(s) "
          f"for tracklet construction")

    tracklet_rows: list[dict] = []
    for i, (_, shortlist_row) in enumerate(shortlist_df.iterrows()):
        elapsed_pipeline = time.time() - t_pipeline_start
        if elapsed_pipeline > time_budget_seconds:
            print(f"  [runtime guard] Part F wall-clock budget ({time_budget_seconds:.0f}s) exceeded after "
                  f"{i}/{len(shortlist_df)} shortlisted combo(s) - stopping tracklet sweep early.")
            break

        name, clf_name = shortlist_row["node_filter_name"], shortlist_row["classifier"]
        neg_ratio, policy, param = shortlist_row["negative_ratio"], shortlist_row["policy"], shortlist_row["param"]
        matching_keys = [
            k for k in shortlist_source_data
            if k[0] == name and k[1] == clf_name and k[2] == neg_ratio and k[3] == policy and k[4] == param
        ]
        for key in matching_keys:
            held_out_dataset = key[5]
            src = shortlist_source_data[key]
            tracklets_df = build_tracklets_from_selected_edges(src["selected_edges"], src["node_filter_cache"]["filtered_nodes"])
            for tr in run_tracklet_filter_sweep(tracklets_df, src["node_filter_cache"]):
                tracklet_rows.append({
                    "node_filter_name": name, "held_out_dataset": held_out_dataset, "classifier": clf_name,
                    "negative_ratio": neg_ratio, "edge_selection_policy": policy, "edge_selection_param": param,
                    **tr,
                })

    per_sample_tracklet_df = pd.DataFrame(tracklet_rows)
    tracklet_oof_group_cols = [
        "node_filter_name", "classifier", "negative_ratio", "edge_selection_policy", "edge_selection_param",
        "node_policy", "min_tracklet_length", "min_mean_classifier_score_quantile",
        "max_smoothness_error_um", "keep_top_k_tracklets_per_sample",
    ]
    oof_tracklet_df = aggregate_oof(per_sample_tracklet_df, tracklet_oof_group_cols)

    return {
        "folds": folds,
        "per_sample_edge_policy_df": per_sample_edge_policy_df,
        "oof_edge_policy_df": oof_edge_policy_df,
        "shortlist_df": shortlist_df,
        "per_sample_tracklet_df": per_sample_tracklet_df,
        "oof_tracklet_df": oof_tracklet_df,
    }

# --------------------------------------------------------------------------- #
# 23. Part H - best-candidate selection + decision report
# --------------------------------------------------------------------------- #
BASELINE_6C6D_EDGE_JACCARD = 0.010667
BASELINE_6C6D_TP = 4
BASELINE_6C6D_FP = 206
BASELINE_6C6D_RECALL = 0.023669
MILESTONE10_CLASSIFIER_PROXY_EDGE_JACCARD = 0.013605


def select_best_candidates(
    oof_tracklet_df: pd.DataFrame, per_sample_tracklet_df: pd.DataFrame,
    csv_path: str = "/kaggle/working/milestone11_best_candidates.csv",
) -> pd.DataFrame:
    """Flags every tracklet-validated aggregate-OOF row against Strong
    (edge_jaccard beats 6C/6D), Practical (TP>4&FP<1000, or TP>=10&FP<2000),
    and Stable (the improvement recurs on MORE THAN ONE held-out sample,
    checked against the per-sample rows, not just n_samples_with_tp>0)
    criteria, then keeps rows meeting Strong or Practical.
    """
    if oof_tracklet_df.empty:
        pd.DataFrame().to_csv(csv_path, index=False)
        return oof_tracklet_df.copy()

    metric_cols = {"edge_TP", "edge_FP", "edge_FN", "edge_jaccard", "precision", "recall", "n_samples", "n_samples_with_tp"}
    group_cols = [c for c in oof_tracklet_df.columns if c not in metric_cols]

    # A vectorized groupby+merge, NOT a per-row O(n_groups x n_per_sample_rows)
    # boolean-mask join - at real Kaggle scale (hundreds of samples x a wide
    # tracklet-filter grid) the naive nested-loop join would be far too slow.
    beats_baseline = (per_sample_tracklet_df["edge_jaccard"] > BASELINE_6C6D_EDGE_JACCARD).astype(int)
    beating_counts = (
        per_sample_tracklet_df.assign(_beats_baseline=beats_baseline)
        .groupby(group_cols, dropna=False)["_beats_baseline"].sum()
        .reset_index().rename(columns={"_beats_baseline": "n_samples_beating_baseline"})
    )
    result_df = oof_tracklet_df.merge(beating_counts, on=group_cols, how="left")
    result_df["n_samples_beating_baseline"] = result_df["n_samples_beating_baseline"].fillna(0).astype(int)

    result_df["meets_strong"] = result_df["edge_jaccard"] > BASELINE_6C6D_EDGE_JACCARD
    result_df["meets_practical"] = (
        ((result_df["edge_TP"] > 4) & (result_df["edge_FP"] < 1000))
        | ((result_df["edge_TP"] >= 10) & (result_df["edge_FP"] < 2000))
    )
    result_df["meets_stable"] = result_df["n_samples_beating_baseline"] > 1

    best_candidates_df = result_df[result_df["meets_strong"] | result_df["meets_practical"]].sort_values(
        "edge_jaccard", ascending=False
    )
    best_candidates_df.to_csv(csv_path, index=False)
    return best_candidates_df


def generate_decision_report(
    oof_edge_policy_df: pd.DataFrame, oof_tracklet_df: pd.DataFrame, best_candidates_df: pd.DataFrame,
    json_path: str = "/kaggle/working/milestone11_decision_report.json",
) -> dict:
    """A-E decision logic anchored on whether the tracklet-validated
    aggregate OOF result beats the 6C/6D baseline AND is stable across
    more than one held-out sample - never on a single-sample result, which
    is exactly what Milestone 10's result was and why Milestone 11 exists.
    """
    pre_tracklet_beats_baseline = bool((oof_edge_policy_df["edge_jaccard"] > BASELINE_6C6D_EDGE_JACCARD).any()) if not oof_edge_policy_df.empty else False
    tracklet_beats_baseline = bool((oof_tracklet_df["edge_jaccard"] > BASELINE_6C6D_EDGE_JACCARD).any()) if not oof_tracklet_df.empty else False
    any_strong = bool(best_candidates_df["meets_strong"].any()) if not best_candidates_df.empty else False
    any_practical = bool(best_candidates_df["meets_practical"].any()) if not best_candidates_df.empty else False
    any_stable = bool(best_candidates_df["meets_stable"].any()) if not best_candidates_df.empty else False

    if any_strong and any_stable:
        recommendation = "A"
        rationale = ("Aggregate OOF edge_jaccard beats the 6C/6D baseline AND the improvement recurs on more than "
                     "one held-out sample - build a real classifier-tracklet submission in Milestone 12.")
    elif (any_strong or any_practical) and not any_stable:
        recommendation = "B"
        rationale = ("At least one configuration beats the baseline (strong or practical) but only on a single "
                     "held-out sample/fold - expand training samples (robust mode / more folds) before trusting "
                     "this as a real signal.")
    elif pre_tracklet_beats_baseline and not tracklet_beats_baseline:
        recommendation = "C"
        rationale = ("Pre-tracklet classifier-ranked edges beat the baseline, but tracklet construction/filtering "
                     "erases the gain - improve features/classifier or tracklet-filter design before resubmitting.")
    elif not pre_tracklet_beats_baseline:
        recommendation = "D"
        rationale = ("No edge-selection policy - even before tracklet construction - beats the baseline in "
                     "aggregate OOF, and Milestone 10 already showed the candidate pool/oracle ceiling is NOT the "
                     "bottleneck - the detector's node recall is a more plausible next target than further "
                     "classifier/graph-optimization work.")
    else:
        recommendation = "E"
        rationale = ("Neither ranking-based classifiers nor simple exclusivity policies recover a stable "
                     "improvement - consider division-aware min-cost flow / graph optimization as a structurally "
                     "different approach.")

    decision = {
        "pre_tracklet_beats_6c6d_baseline": pre_tracklet_beats_baseline,
        "tracklet_beats_6c6d_baseline": tracklet_beats_baseline,
        "any_strong": any_strong, "any_practical": any_practical, "any_stable": any_stable,
        "baseline_edge_jaccard": BASELINE_6C6D_EDGE_JACCARD, "baseline_tp": BASELINE_6C6D_TP,
        "baseline_fp": BASELINE_6C6D_FP, "baseline_recall": BASELINE_6C6D_RECALL,
        "milestone10_classifier_proxy_edge_jaccard": MILESTONE10_CLASSIFIER_PROXY_EDGE_JACCARD,
        "recommendation": recommendation, "rationale": rationale,
    }
    with open(json_path, "w") as f:
        json.dump(decision, f, indent=2, default=str)

    print("\n=== Milestone 11 decision report ===")
    for k, v in decision.items():
        print(f"  {k}: {v}")
    return decision


def print_summary_reports(
    oof_edge_policy_df: pd.DataFrame, oof_tracklet_df: pd.DataFrame,
    per_sample_tracklet_df: pd.DataFrame, best_candidates_df: pd.DataFrame, decision: dict,
) -> None:
    print("\n=== Top 30 pre-tracklet edge-selection-policy OOF results (by edge_jaccard) ===")
    if not oof_edge_policy_df.empty:
        for _, r in oof_edge_policy_df.sort_values("edge_jaccard", ascending=False).head(30).iterrows():
            print(
                f"  {r['node_filter_name']}/{r['classifier']}/negratio={r['negative_ratio']}/{r['policy']}={r['param']}: "
                f"TP={r['edge_TP']} FP={r['edge_FP']} FN={r['edge_FN']} jaccard={r['edge_jaccard']:.4f} "
                f"precision={r['precision']:.4f} recall={r['recall']:.4f} n_samples={r['n_samples']}"
            )

    print("\n=== Top 30 tracklet-validated OOF results (by edge_jaccard) ===")
    if not oof_tracklet_df.empty:
        for _, r in oof_tracklet_df.sort_values("edge_jaccard", ascending=False).head(30).iterrows():
            print(
                f"  {r['node_filter_name']}/{r['classifier']}/negratio={r['negative_ratio']}/"
                f"{r['edge_selection_policy']}={r['edge_selection_param']}/{r['node_policy']}/"
                f"len>={r['min_tracklet_length']},scoreq={r['min_mean_classifier_score_quantile']},"
                f"smooth<={r['max_smoothness_error_um']},topk={r['keep_top_k_tracklets_per_sample']}: "
                f"TP={r['edge_TP']} FP={r['edge_FP']} FN={r['edge_FN']} jaccard={r['edge_jaccard']:.4f} "
                f"n_samples={r['n_samples']} n_samples_with_tp={r['n_samples_with_tp']}"
            )

    print(f"\n=== Configs beating 6C/6D baseline (edge_jaccard>{BASELINE_6C6D_EDGE_JACCARD}) ===")
    if not oof_tracklet_df.empty:
        print(f"  {int((oof_tracklet_df['edge_jaccard'] > BASELINE_6C6D_EDGE_JACCARD).sum())} tracklet-validated config(s)")
    if not oof_edge_policy_df.empty:
        print(f"  {int((oof_edge_policy_df['edge_jaccard'] > BASELINE_6C6D_EDGE_JACCARD).sum())} pre-tracklet config(s)")

    print("\n=== Configs meeting practical criteria (TP>4,FP<1000 OR TP>=10,FP<2000) ===")
    if not oof_tracklet_df.empty:
        practical = oof_tracklet_df[
            ((oof_tracklet_df["edge_TP"] > 4) & (oof_tracklet_df["edge_FP"] < 1000))
            | ((oof_tracklet_df["edge_TP"] >= 10) & (oof_tracklet_df["edge_FP"] < 2000))
        ]
        print(f"  {len(practical)} tracklet-validated config(s) meet practical criteria")

    print("\n=== Best tracklet-validated config per held-out sample ===")
    if not per_sample_tracklet_df.empty:
        best_per_sample = per_sample_tracklet_df.loc[per_sample_tracklet_df.groupby("held_out_dataset")["edge_jaccard"].idxmax()]
        for _, r in best_per_sample.iterrows():
            print(
                f"  {r['held_out_dataset']}: {r['node_filter_name']}/{r['classifier']}/{r['edge_selection_policy']}="
                f"{r['edge_selection_param']}/{r['node_policy']} jaccard={r['edge_jaccard']:.4f} "
                f"TP={r['edge_TP']} FP={r['edge_FP']}"
            )

    print(f"\n=== Best candidates (meeting strong or practical criteria): {len(best_candidates_df)} ===")
    print(f"\n=== Final recommendation: {decision['recommendation']} ===")
    print(f"  {decision['rationale']}")

# --------------------------------------------------------------------------- #
# 24. Tests
# --------------------------------------------------------------------------- #
def run_milestone11_tests() -> None:
    """Correctness tests for the NEW Milestone 11 machinery: edge-selection
    policies, greedy exclusivity (generalized + native), fold construction,
    hard-negative sampling v2, tracklet construction from classifier-scored
    edges, aggregate-OOF pooling (vs the naive per-sample mean it must NOT
    equal), shortlist union logic, and a synthetic end-to-end fixture where
    classifier-ranked edges - pushed through real exclusivity + tracklet
    construction + filtering - convincingly beat the 6C/6D baseline. All on
    small in-memory fixtures (no disk I/O).
    """
    # Test 1: global_top_k / per_frame_top_k / per_source_top1_with_quantile
    # selection correctness on a small hand-scored fixture.
    scored = pd.DataFrame({
        "source_id": [1, 1, 2, 2, 3], "target_id": [10, 11, 12, 13, 10], "t": [0, 0, 0, 0, 0],
        "label": [True, False, True, False, False], "_score": [0.9, 0.5, 0.8, 0.3, 0.95],
    })
    top2 = select_global_top_k(scored, budget=2)
    assert set(zip(top2["source_id"], top2["target_id"])) == {(3, 10), (1, 10)}, "top-2 by score should be the 0.95 and 0.9 rows"

    pf1 = select_per_frame_top_k(scored, per_frame_k=1)
    assert len(pf1) == 1 and pf1.iloc[0]["source_id"] == 3, "single frame, top-1/frame keeps only the highest-scored row"

    top1_per_source = scored.sort_values("_score", ascending=False).groupby("source_id", sort=False).head(1)
    threshold_90 = scored["_score"].quantile(0.9)
    kept_q90, thr = select_per_source_top1_with_quantile(scored, 0.9)
    assert thr == threshold_90
    assert (kept_q90["_score"] >= threshold_90).all()
    assert len(kept_q90) == int((top1_per_source["_score"] >= threshold_90).sum())

    # Test 2: greedy exclusive per-frame respects BOTH source capacity (1)
    # and target exclusivity, always preferring the higher-scoring edge.
    scored2 = pd.DataFrame({
        "source_id": [1, 1, 2, 3], "target_id": [10, 11, 10, 12], "t": [0, 0, 0, 0],
        "label": [True, False, False, True], "_score": [0.9, 0.85, 0.95, 0.5],
    })
    kept2, _ = select_greedy_exclusive_per_frame(scored2, quantile=0.0, source_capacity=1)
    assert set(zip(kept2["source_id"], kept2["target_id"])) == {(2, 10), (1, 11), (3, 12)}, (
        "source 2's edge to target 10 (score 0.95) must win over source 1's competing edge to the same "
        "target (score 0.9); source 1 then gets its next-best edge to target 11 (score 0.85)"
    )

    # Test 3: enforce_edge_exclusivity generalizes the same guarantee to an
    # ARBITRARY (non-exclusive) pre-selected edge set, e.g. what
    # per_frame_top_k could hand it (duplicate sources/targets allowed).
    exclusive = enforce_edge_exclusivity(scored2)
    sources = exclusive["source_id"].tolist()
    targets = exclusive["target_id"].tolist()
    assert len(sources) == len(set(sources)), "no source should appear twice after exclusivity is enforced"
    assert len(targets) == len(set(targets)), "no target should appear twice after exclusivity is enforced"
    assert (2, 10) in set(zip(exclusive["source_id"], exclusive["target_id"])), "highest-score edge (0.95) must survive"

    # Test 4: tracklet construction from classifier-scored edges - a clean
    # 3-node chain plus an isolated node not part of any edge.
    nodes4 = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 2, 0],
        "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 5.0], "x": [0.0] * 4, "score": [0.9, 0.8, 0.85, 0.7],
    })
    selected_edges4 = pd.DataFrame({
        "source_id": [0, 1], "target_id": [1, 2], "distance_um": [0.4, 0.4], "_score": [0.9, 0.85],
    })
    tracklets4 = build_tracklets_from_selected_edges(selected_edges4, nodes4)
    assert len(tracklets4) == 1
    assert tracklets4.iloc[0]["node_ids"] == (0, 1, 2)
    assert tracklets4.iloc[0]["length_frames"] == 3

    # Test 5: aggregate_oof is the POOLED sum, which must differ from (and
    # is deliberately never computed as) the naive mean of per-sample
    # edge_jaccard values - the exact instability architecture review
    # flagged for samples with near-zero positives.
    per_sample5 = pd.DataFrame({
        "held_out_dataset": ["s1", "s2"], "node_filter_name": ["nf", "nf"], "classifier": ["c", "c"],
        "negative_ratio": [1, 1], "policy": ["p", "p"], "param": [1, 1],
        "edge_TP": [10, 0], "edge_FP": [10, 0], "edge_FN": [0, 5],
    })
    naive_mean_jaccard = np.mean([10 / 20, 0 / 5])
    oof5 = aggregate_oof(per_sample5, ["node_filter_name", "classifier", "negative_ratio", "policy", "param"])
    assert len(oof5) == 1
    row5 = oof5.iloc[0]
    assert row5["edge_TP"] == 10 and row5["edge_FP"] == 10 and row5["edge_FN"] == 5
    assert abs(row5["edge_jaccard"] - 10 / 25) < 1e-9
    assert abs(row5["edge_jaccard"] - naive_mean_jaccard) > 1e-6, "pooled and naive-mean jaccard must NOT coincide here"
    assert row5["n_samples"] == 2 and row5["n_samples_with_tp"] == 1

    # Test 6: LOSO and GroupKFold fold construction partition every dataset
    # correctly, and degenerate-fold flagging fires only when warranted.
    datasets6 = ["a", "b", "c"]
    loso6 = build_loso_folds(datasets6)
    assert len(loso6) == 3
    for f in loso6:
        assert len(f["val_datasets"]) == 1 and f["val_datasets"][0] not in f["train_datasets"]

    gkf6 = build_group_kfold_folds(datasets6, n_splits=3)
    all_val6 = sorted(sum([f["val_datasets"] for f in gkf6], []))
    assert all_val6 == sorted(datasets6), "every dataset must be held out exactly once across GroupKFold folds"

    annotated6 = annotate_fold_degeneracy(loso6, {"a": 0, "b": 5, "c": 5}, min_positive_count=3)
    degeneracy = {f["fold_id"]: f["degenerate"] for f in annotated6}
    assert degeneracy["a"] is True and degeneracy["b"] is False

    # Test 7: hard-negative sampling v2 always includes all positives, and
    # its hard pool is the union of local_rank<=3, highest node_score_product,
    # and highest edge_score negatives (here rank=9 and rank=10 are boosted
    # on those two scores respectively, despite not being rank<=3).
    n_pos7, cap7 = 2, 4
    local_ranks7 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    node_score7 = [0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.99]
    edge_score7 = [0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.99, 0.4]
    train_df7 = pd.DataFrame({
        "label": [True] * n_pos7 + [False] * 10,
        "local_rank": [0] * n_pos7 + local_ranks7,
        "node_score_product": [0.9] * n_pos7 + node_score7,
        "edge_score": [0.9] * n_pos7 + edge_score7,
    })
    sampled7 = hard_negative_sample(train_df7, negative_ratio=2, random_state=0)
    assert sampled7["label"].sum() == n_pos7
    neg7 = sampled7[~sampled7["label"]]
    assert len(neg7) == cap7
    assert set(neg7["local_rank"].tolist()).issubset({1, 2, 3, 9, 10}), (
        "the 4 sampled negatives must all come from the hard pool (rank<=3, or boosted by "
        "node_score_product/edge_score) since that pool already has 5 members - no random fill needed"
    )

    # Test 8: shortlist selection force-includes the best (even if only)
    # row of EVERY policy, not just a tight top-K/top-recall/top-precision
    # band that could otherwise exclude a structurally-important policy.
    oof8 = pd.DataFrame({
        "policy": ["global_top_k"] * 3 + ["per_frame_top_k"] * 3 + ["per_source_top1_with_score_quantile"] * 1 + ["greedy_exclusive_per_frame"] * 1,
        "param": [100, 200, 500, 1, 2, 3, 0.9, 0.9],
        "edge_jaccard": [0.5, 0.4, 0.3, 0.45, 0.35, 0.25, 0.05, 0.02],
        "recall": [0.1] * 8, "precision": [0.1] * 8,
    })
    shortlist8 = select_shortlist_for_tracklets(oof8, top_n_jaccard=2, top_n_recall=1, top_n_precision=1)
    assert "greedy_exclusive_per_frame" in set(shortlist8["policy"]), "the worst-scoring policy's only row must still be force-included"
    assert "per_source_top1_with_score_quantile" in set(shortlist8["policy"])

    # Test 9: synthetic end-to-end fixture - a cleanly separable true track
    # (nodes 0,1,2) vs. a far-away noise track (nodes 100,101,102), pushed
    # through greedy-exclusive edge selection, tracklet construction, and
    # tracklet-filtered node-policy-B evaluation, should convincingly beat
    # the 6C/6D baseline edge_jaccard.
    filtered_nodes9 = pd.DataFrame({
        "node_id": [0, 1, 2, 100, 101, 102], "t": [0, 1, 2, 0, 1, 2],
        "z": [0.0] * 6, "y": [0.0, 1.0, 2.0, 50.0, 51.0, 52.0], "x": [0.0] * 6,
        "score": [0.9, 0.9, 0.9, 0.5, 0.5, 0.5],
    })
    gt_nodes9 = pd.DataFrame({
        "node_id": [0, 1, 2], "t": [0, 1, 2], "z": [0.0] * 3, "y": [0.0, 1.0, 2.0], "x": [0.0] * 3,
    })
    gt_edges9 = pd.DataFrame({"source_id": [0, 1], "target_id": [1, 2]})
    candidates9 = pd.DataFrame({
        "source_id": [0, 1, 100, 101], "target_id": [1, 2, 101, 102], "t": [0, 1, 0, 1],
        "label": [True, True, False, False], "_score": [0.95, 0.9, 0.05, 0.02], "distance_um": [0.4, 0.4, 0.4, 0.4],
    })
    selected9, _ = select_greedy_exclusive_per_frame(candidates9, quantile=0.0)
    assert len(selected9) == 4, "no source/target conflicts exist in this fixture, so all 4 candidates survive"

    tracklets9 = build_tracklets_from_selected_edges(selected9, filtered_nodes9)
    assert len(tracklets9) == 2

    node_filter_cache9 = {
        "filtered_nodes": filtered_nodes9, "gt_nodes": gt_nodes9, "gt_edges": gt_edges9,
        "match_filtered": match_nodes_by_timepoint(filtered_nodes9, gt_nodes9, NODE_MATCH_MAX_DISTANCE_UM),
        "n_timepoints": 3,
    }
    result9 = evaluate_tracklet_config(
        tracklets9, node_filter_cache9, min_tracklet_length=2, min_mean_classifier_score_quantile=0.0,
        max_smoothness_error_um=999.0, keep_top_k=10, node_policy="B_all_filtered_nodes_with_classifier_edges",
    )
    assert result9["edge_TP"] == 2 and result9["edge_FP"] == 2 and result9["edge_FN"] == 0
    assert abs(result9["edge_jaccard"] - 0.5) < 1e-9
    assert result9["edge_jaccard"] > BASELINE_6C6D_EDGE_JACCARD, (
        "a cleanly separable synthetic case should convincingly beat the 6C/6D baseline after real "
        "exclusivity + tracklet construction + filtering"
    )

    print("All milestone11_classifier_tracklet_validation tests passed (9/9).")

# --------------------------------------------------------------------------- #
# 25. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone11_pipeline(
    mode: str = "fast",
    node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    classifier_builders: dict = CLASSIFIER_BUILDERS,
    negative_ratio_values: Sequence[int] = NEGATIVE_RATIO_VALUES,
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT,
    seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
) -> dict:
    """Runs the unit tests, then the full Milestone 11 classifier-to-
    tracklet validation (Parts A-F via `run_milestone11_validation`), then
    Part H's best-candidate selection + decision report, saving all 5
    required output files. Builds no submission, never calls any Kaggle
    submission API, and never mutates 6C/6D/9/10's files.
    """
    print("=== Self-test: edge-selection, exclusivity, tracklet construction, aggregate-OOF, fold, hard-negative-sampling, and shortlist unit tests ===")
    run_milestone11_tests()

    print(f"\n=== Milestone 11 classifier-to-tracklet validation (mode={mode!r}) ===")
    validation = run_milestone11_validation(
        mode=mode, node_filter_configs=node_filter_configs, classifier_builders=classifier_builders,
        negative_ratio_values=negative_ratio_values, n_robust_samples=n_robust_samples, seed=seed,
        time_budget_seconds=time_budget_seconds,
    )

    validation["oof_edge_policy_df"].to_csv("/kaggle/working/milestone11_oof_edge_policy_eval.csv", index=False)
    validation["oof_tracklet_df"].to_csv("/kaggle/working/milestone11_tracklet_validation_eval.csv", index=False)
    validation["per_sample_tracklet_df"].to_csv("/kaggle/working/milestone11_per_sample_eval.csv", index=False)

    print("\n=== Part H: best-candidate selection + decision report ===")
    best_candidates_df = select_best_candidates(validation["oof_tracklet_df"], validation["per_sample_tracklet_df"])
    decision = generate_decision_report(validation["oof_edge_policy_df"], validation["oof_tracklet_df"], best_candidates_df)

    print_summary_reports(
        validation["oof_edge_policy_df"], validation["oof_tracklet_df"],
        validation["per_sample_tracklet_df"], best_candidates_df, decision,
    )

    return {
        "folds": validation["folds"],
        "per_sample_edge_policy_df": validation["per_sample_edge_policy_df"],
        "oof_edge_policy_df": validation["oof_edge_policy_df"],
        "shortlist_df": validation["shortlist_df"],
        "per_sample_tracklet_df": validation["per_sample_tracklet_df"],
        "oof_tracklet_df": validation["oof_tracklet_df"],
        "best_candidates_df": best_candidates_df,
        "decision": decision,
    }


if __name__ == "__main__":
    run_milestone11_pipeline(mode="robust")
