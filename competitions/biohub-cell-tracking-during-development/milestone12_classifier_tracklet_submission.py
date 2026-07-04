"""
Biohub - Cell Tracking During Development
Milestone 12: final classifier-tracklet submission builder.

Milestone 11's real Kaggle robust OOF result succeeded: node_filter=
moderate_150_085, classifier=hist_gradient_boosting, negative_ratio=100,
edge_selection_policy=greedy_exclusive_per_frame (quantile=0.900),
node_policy=A_tracklet_nodes_only, tracklet filter (min_tracklet_length=5,
min_mean_classifier_score_quantile=0.7, max_smoothness_error_um=3,
keep_top_k_tracklets_per_sample=100) achieved aggregate OOF edge_TP=103,
edge_FP=1932, edge_FN=5759, edge_jaccard=0.013215 across n_samples=12 with
n_samples_with_tp=6 - beating the 6C/6D baseline (edge_jaccard=0.010667)
on both the Strong (beats baseline) and Stable (recurs on more than one
held-out sample, not just one) success criteria. This module builds the
FIRST real submission.csv this codebase has ever been allowed to produce.

This is a PRODUCTION BUILD, not another validation sweep: no OOF grid,
no cross-validation, one fixed config per classifier, evaluated once.
Milestone 11 already answered "does this generalize" - this module only
answers "does this fixed, already-validated recipe run cleanly end to end
on the real test data and produce a schema-correct file."

Design notes incorporating Opus architecture review:

  - Final-model training: the exact same fixed-seed ROBUST train-sample
    selection Milestone 11 used (`select_train_samples_robust`, seed=0, up
    to 12 samples) is recomputed, then ALL of it (no held-out fold - there
    is nothing left to hold out; Milestone 11 already validated
    generalization via GroupKFold) is pooled into one training-candidate
    table for node_filter=moderate_150_085, hard-negative sampled
    (negative_ratio=100), and used to fit exactly 2 classifiers: a primary
    HistGradientBoostingClassifier and a conservative-diagnostic
    ExtraTreesClassifier. Every feature is derived from `filtered_nodes`/
    the ceiling candidate pool alone (distance, node scores, local
    density, competition counts, rank, smoothness) - GT touches nothing
    but the final `label` column, so there is no leakage risk between
    training and (GT-free) test-time scoring.

  - GT-free feature parity: test data has NO ground truth at all (a real
    hidden-label competition - no .geff file exists for test samples).
    Feature computation is therefore split into two explicit steps: a
    single GT-free function (`build_candidate_feature_table`) computes the
    full ~24-feature set identically for train and test, and a separate
    `label_feature_table` step (applied ONLY to training samples) appends
    the GT-derived label column afterward. This guarantees train/test
    features are byte-for-byte identical - any drift here would silently
    break the classifier's calibration - while making it structurally
    impossible for the labeling step to accidentally run on GT-free test
    data (there is no `FEATURE_COLUMNS`-with-label list to accidentally
    select from a GT-free cache).

  - Imputation medians are TRAINED, not recomputed: ExtraTreesClassifier
    needs NaN-free input, so the per-feature median computed from the
    hard-negative-sampled TRAINING set is persisted alongside the fitted
    classifier and reapplied verbatim to every test sample's features - a
    median freshly computed from each test candidate pool would leak
    test-time distribution information and could silently differ from
    what the classifier was actually calibrated against.
    HistGradientBoostingClassifier (primary) accepts NaN natively and
    skips this step entirely.

  - Guaranteed-nonempty per-dataset fallback: `validate_submission` hard-
    requires every expected test dataset to be present. An aggressive
    filter (moderate_150_085 + a 0.900 greedy-exclusive quantile + a
    min_tracklet_length=5) meeting one quiet/short test sample could wipe
    every surviving tracklet, silently making a submission invalid by
    omission. `run_inference_for_test_sample` therefore guarantees a non-
    empty `pred_nodes` for any sample where the raw detector found at
    least one node: if classifier-driven edge selection or tracklet
    filtering wipes every edge, that sample falls back to emitting its own
    filtered (or, failing that, raw) nodes as isolated edge-less node
    rows - so a dataset can never vanish from the submission purely
    because a filter was too strict for it.

  - Primary/conservative fallback: BOTH the primary and conservative-
    diagnostic submissions are generated and validated UNCONDITIONALLY
    (never short-circuited), so a bug in one is never hidden behind the
    other's success. `submission.csv` is written from the primary config
    if it validates; only if the primary FAILS validation does the
    conservative-diagnostic config become `submission.csv` instead; if
    BOTH fail, the pipeline raises rather than writing anything - a run
    that produces no submission.csv is preferable to one that silently
    writes something invalid.

  - Global id/node_id threading: `id` and `node_id` counters run
    consecutively across the WHOLE submission (every test dataset
    concatenated), mirroring conservative_tracker.py's established
    convention. `node_id_start` is always threaded using the detector's
    OWN returned counter (already advanced past every RAW node a sample
    produced), never `len(pred_nodes)` - using the emitted-row count
    instead would silently let two different samples' node_ids collide
    whenever tracklet filtering discards some of a sample's raw nodes.

  - Every quantile threshold (`edge_selection_param`'s greedy-exclusive/
    per-source-top1 quantile, and the tracklet filter's
    `min_mean_classifier_score_quantile`) is resolved from THAT test
    sample's own score/mean_edge_score distribution alone, never pooled
    across test datasets - identical to Milestone 11's per-(classifier,
    sample) thresholding discipline.

No Kaggle submission API is ever called and Save Version is never
triggered - this module only writes files under /kaggle/working/. 6C/6D/
9/10/11's files are not modified - all reused logic (reader/detector/
node-filter/GT-matching/candidate-pool/feature-engineering/tracklet-
construction-and-filtering machinery from Milestones 10/11, the
submission-row/validation convention from conservative_tracker.py) is
copied in, not imported.
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

# 15. Part A - sample selection (FAST vs ROBUST modes) + runtime guard
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

# --------------------------------------------------------------------------- #
# 16. Feature engineering (GT-free) + a separate, explicit training-only
#     labeling step
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS_NO_LABEL = [
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
]

# Same feature set Milestones 10/11 trained on - numeric only, excludes
# identifiers/labels/metadata. HistGradientBoostingClassifier accepts NaN
# natively; ExtraTreesClassifier needs median imputation (see Part D below).
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


def build_candidate_feature_table(node_filter_cache: dict) -> pd.DataFrame:
    """Computes the full per-candidate feature table for one (sample,
    node-filter-config)'s ceiling candidate pool. IDENTICAL feature
    computation whether or not GT is available - every feature is derived
    from `filtered_nodes`/`ceiling_pool` alone (density, competition count,
    local_rank, smoothness via the reference-displacement chain), so this
    function never reads `gt_edge_set`/`match_filtered` and is safe to call
    on GT-free test-time caches. Labeling (which DOES need GT) is a
    separate, explicit step below - never bundled in here - so train and
    test features are guaranteed byte-for-byte identical.
    """
    pool = node_filter_cache["ceiling_pool"]
    if len(pool) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS_NO_LABEL)

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

    return df[FEATURE_COLUMNS_NO_LABEL].reset_index(drop=True)


def label_feature_table(feature_df: pd.DataFrame, match_filtered: dict, gt_edge_set: set) -> pd.DataFrame:
    """Appends the `label` column to an already-built (GT-free) feature
    table via GT-node matching - the ONLY GT-dependent step in the whole
    feature pipeline, applied only to TRAINING samples.
    """
    labeled = feature_df.copy()
    if len(labeled) == 0:
        labeled["label"] = pd.Series([], dtype=bool)
        return labeled
    labeled["label"] = label_candidates(labeled, match_filtered["pred_to_gt"], gt_edge_set).to_numpy()
    return labeled


def build_training_feature_table_for_sample(node_filter_cache: dict) -> pd.DataFrame:
    """Training-only convenience: builds the GT-free feature table then
    immediately labels it using this cache's GT-matching fields (present
    only for TRAINING node-filter caches, never test ones).
    """
    feature_df = build_candidate_feature_table(node_filter_cache)
    return label_feature_table(feature_df, node_filter_cache["match_filtered"], node_filter_cache["gt_edge_set"])


def build_all_training_feature_tables(node_filter_caches: Sequence[dict]) -> pd.DataFrame:
    """Builds and concatenates the labeled feature table for every training
    sample's node-filter cache.
    """
    parts = [build_training_feature_table_for_sample(c) for c in node_filter_caches]
    if not parts:
        return pd.DataFrame(columns=FEATURE_COLUMNS_NO_LABEL + ["label"])
    return pd.concat(parts, ignore_index=True)

# --------------------------------------------------------------------------- #
# 17. Node-filter config + per-sample caching (training: GT-retaining;
#     test: GT-free)
# --------------------------------------------------------------------------- #
NODE_FILTER_CONFIG = {
    "name": "moderate_150_085", "max_nodes_per_timepoint": 150,
    "score_quantile_per_timepoint": 0.85, "absolute_score_threshold": 0.05,
}
NODE_MATCH_MAX_DISTANCE_UM = 7.0


def precompute_node_filter_cache(
    raw_cache: dict, node_filter_config: dict = NODE_FILTER_CONFIG, match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """TRAINING-time cache for ONE (train sample, node-filter-config) pair:
    applies node filtering, matches filtered nodes to GT ONCE (needed for
    labeling), builds the reference-displacement chain and ceiling
    candidate-edge pool ONCE.
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


def precompute_raw_test_sample_cache(sample_zarr_dir: Path, detector_config: dict, node_id_start: int) -> tuple[dict, int]:
    """Runs the raw detector for ONE test sample - no GT exists for test
    data at all, so this never reads a .geff file. `node_id_start` threads
    a running counter across every test sample so node_ids stay globally
    unique across the WHOLE submission (mirrors conservative_tracker.py's
    established convention). The RETURNED counter is already advanced past
    every RAW node this sample produced (not just however many survive
    downstream filtering) - callers must feed that returned value into the
    next sample's `node_id_start`, never `len(pred_nodes)`, or ids from
    different samples could collide.
    """
    dataset_name = sample_zarr_dir.stem
    raw_nodes_df, next_node_id = detect_all_raw_nodes_for_sample(sample_zarr_dir, detector_config, node_id_start=node_id_start)
    n_timepoints = get_sample_timepoint_count(sample_zarr_dir)
    cache = {"dataset": dataset_name, "raw_nodes_df": raw_nodes_df, "n_timepoints": n_timepoints}
    return cache, next_node_id


def precompute_test_node_filter_cache(raw_test_cache: dict, node_filter_config: dict = NODE_FILTER_CONFIG) -> dict:
    """GT-free node-filter cache for ONE test sample - applies the same
    node filter and builds the same ceiling candidate pool as training,
    but never touches GT (there is none) and never runs GT-node matching.
    """
    filtered_nodes = apply_node_filter(
        raw_test_cache["raw_nodes_df"],
        node_filter_config["max_nodes_per_timepoint"],
        node_filter_config["score_quantile_per_timepoint"],
        node_filter_config["absolute_score_threshold"],
    )[["node_id", "t", "z", "y", "x", "score"]].reset_index(drop=True)

    ref_disp = build_reference_displacement_by_node(filtered_nodes)
    pool = build_ceiling_candidate_pool(filtered_nodes, ref_disp)

    return {
        "dataset": raw_test_cache["dataset"],
        "node_filter_name": node_filter_config["name"],
        "raw_nodes_df": raw_test_cache["raw_nodes_df"],
        "filtered_nodes": filtered_nodes,
        "n_timepoints": raw_test_cache["n_timepoints"],
        "ceiling_pool": pool,
    }

# 18. Part D - classifiers: hard-negative sampling + one-classifier-per-fold
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

# --------------------------------------------------------------------------- #
# 19. Final config definitions + final-model training (no folds - this is
#     the production fit, not a cross-validation experiment)
# --------------------------------------------------------------------------- #
PRIMARY_CONFIG = {
    "config_name": "primary",
    "node_filter": NODE_FILTER_CONFIG,
    "classifier_name": "hist_gradient_boosting",
    "negative_ratio": 100,
    "edge_selection_policy": "greedy_exclusive_per_frame",
    "edge_selection_param": 0.900,
    "node_policy": "A_tracklet_nodes_only",
    "min_tracklet_length": 5,
    "min_mean_classifier_score_quantile": 0.7,
    "max_smoothness_error_um": 3.0,
    "keep_top_k_tracklets_per_sample": 100,
}

# Diagnostic only - never written to submission.csv unless the primary
# config fails validation (see Part H).
CONSERVATIVE_CONFIG = {
    "config_name": "conservative_diagnostic",
    "node_filter": NODE_FILTER_CONFIG,
    "classifier_name": "extra_trees",
    "negative_ratio": 100,
    "edge_selection_policy": "per_source_top1_with_score_quantile",
    "edge_selection_param": 0.995,
    "node_policy": "A_tracklet_nodes_only",
    "min_tracklet_length": 2,
    "min_mean_classifier_score_quantile": 0.0,
    "max_smoothness_error_um": 3.0,
    "keep_top_k_tracklets_per_sample": 200,
}


def fit_final_classifier(train_feature_df: pd.DataFrame, classifier_name: str, negative_ratio: int, random_state: int = 0) -> dict:
    """Trains exactly ONE classifier on the FULL pooled training-sample
    feature table (hard-negative sampled) - no held-out fold, no cross-
    validation: Milestone 11 already validated generalization via
    GroupKFold; this is the final production fit using every available
    training signal, with nothing left to hold out. Persists the training-
    set median (needed to impute NaN features for classifiers that don't
    accept NaN natively) so the EXACT SAME medians get applied at test
    time - a median freshly computed from the test candidate pool would
    both leak test-time distribution information and silently differ from
    what the classifier was actually calibrated against during training.
    """
    train_sample = hard_negative_sample(train_feature_df, negative_ratio, random_state=random_state)
    if train_sample["label"].nunique() < 2:
        raise ValueError(f"training data is degenerate (single class) for classifier={classifier_name!r} - cannot fit.")

    X_train_raw = train_sample[CLASSIFIER_FEATURE_COLUMNS]
    y_train = train_sample["label"].to_numpy(dtype=int)
    medians = X_train_raw.median()
    needs_imputation = classifier_name in CLASSIFIERS_NEEDING_IMPUTATION
    X_train = X_train_raw.fillna(medians) if needs_imputation else X_train_raw

    clf = CLASSIFIER_BUILDERS[classifier_name]()
    clf.fit(X_train, y_train)

    n_train_pos = int(y_train.sum())
    return {
        "classifier_name": classifier_name, "classifier": clf, "medians": medians, "needs_imputation": needs_imputation,
        "negative_ratio": negative_ratio,
        "n_train_positive": n_train_pos, "n_train_negative": int(len(y_train) - n_train_pos),
    }


def score_candidates(feature_df: pd.DataFrame, trained_model: dict) -> np.ndarray:
    """Scores an already-built (GT-free) feature table with a trained
    model, applying the SAME imputation medians computed at training time
    (never a fresh median of the test pool itself).
    """
    X_raw = feature_df[CLASSIFIER_FEATURE_COLUMNS]
    X = X_raw.fillna(trained_model["medians"]) if trained_model["needs_imputation"] else X_raw
    return trained_model["classifier"].predict_proba(X)[:, 1]

# --------------------------------------------------------------------------- #
# 20. Edge-selection policies (the 2 policies the primary/conservative
#     configs actually use - copied from Milestone 11, GT-free)
# --------------------------------------------------------------------------- #
GREEDY_SOURCE_CAPACITY_DEFAULT = 1  # divisions are rare (M10 GT diagnosis); 2 is a diagnostic-only option


def select_per_source_top1_with_quantile(scored_df: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, float]:
    """Per source node, keep only its single best (highest-score)
    outgoing candidate, then gate on that candidate's score being at or
    above the given quantile of THIS (classifier, sample)'s own score
    distribution - never a pooled/global threshold.
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
    since edges only ever connect frame t to t+1. The threshold is always
    resolved from THIS (classifier, sample)'s own score distribution.
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


EDGE_SELECTION_FUNCTIONS = {
    "greedy_exclusive_per_frame": select_greedy_exclusive_per_frame,
    "per_source_top1_with_score_quantile": select_per_source_top1_with_quantile,
}

# --------------------------------------------------------------------------- #
# 21. Tracklet construction from classifier-selected edges (copied from
#     Milestone 11, GT-free)
# --------------------------------------------------------------------------- #
def enforce_edge_exclusivity(selected_edges_df: pd.DataFrame) -> pd.DataFrame:
    """Re-applies source/target exclusivity (capacity=1, highest-score
    first) to an ARBITRARY pre-selected edge set. `build_tracklets`
    structurally assumes at most one outgoing/incoming edge per node (a
    plain dict silently overwrites duplicates otherwise), so every edge-
    selection policy is pushed through this SAME final exclusivity gate
    before tracklet construction, even `per_source_top1_with_score_quantile`
    which only enforces source-side exclusivity on its own.
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

# --------------------------------------------------------------------------- #
# 22. Test-time inference (GT-free) with a guaranteed-nonempty fallback
# --------------------------------------------------------------------------- #
def run_inference_for_test_sample(
    sample_zarr_dir: Path, config: dict, trained_model: dict, node_id_start: int,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
) -> dict:
    """Runs the full classifier-tracklet inference pipeline for ONE test
    sample under the given config (primary or conservative), with NO GT
    involved anywhere. Guarantees a non-empty `pred_nodes` for any sample
    where the raw detector found at least one node - if classifier-driven
    edge selection or tracklet filtering wipes every edge (an aggressive
    filter meeting a quiet sample), predicted nodes fall back to that
    sample's own filtered (or, failing that, raw) nodes emitted as
    isolated edge-less node rows, so `validate_submission`'s "every
    expected test dataset is present" check can never fail purely because
    a filter was too aggressive for one quiet sample.
    """
    node_filter_config = config["node_filter"]
    raw_cache, next_node_id = precompute_raw_test_sample_cache(sample_zarr_dir, detector_config, node_id_start)
    test_cache = precompute_test_node_filter_cache(raw_cache, node_filter_config)
    filtered_nodes = test_cache["filtered_nodes"]

    diagnostics = {
        "dataset": test_cache["dataset"], "n_raw_nodes": len(raw_cache["raw_nodes_df"]),
        "n_filtered_nodes": len(filtered_nodes), "n_candidates": len(test_cache["ceiling_pool"]),
        "n_selected_edges": 0, "edge_score_threshold": None,
        "n_tracklets_built": 0, "n_tracklets_kept": 0, "min_mean_edge_score_threshold": None,
        "fallback_used": False,
    }

    def _isolated_nodes(nodes_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_nodes = nodes_source[["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
        pred_edges = pd.DataFrame(columns=["source_id", "target_id"])
        return pred_nodes, pred_edges

    def _finish(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame) -> dict:
        return {"pred_nodes": pred_nodes, "pred_edges": pred_edges, "next_node_id": next_node_id, "diagnostics": diagnostics}

    if len(filtered_nodes) == 0:
        # Node filter wiped out every raw node - fall back one more level to
        # the UNFILTERED raw detections so this dataset is never entirely
        # missing from the submission.
        diagnostics["fallback_used"] = True
        return _finish(*_isolated_nodes(raw_cache["raw_nodes_df"]))

    feature_df = build_candidate_feature_table(test_cache)
    if len(feature_df) == 0:
        diagnostics["fallback_used"] = True
        return _finish(*_isolated_nodes(filtered_nodes))

    scored_df = feature_df.copy()
    scored_df["_score"] = score_candidates(feature_df, trained_model)

    edge_selection_fn = EDGE_SELECTION_FUNCTIONS[config["edge_selection_policy"]]
    selected_edges, edge_threshold = edge_selection_fn(scored_df, config["edge_selection_param"])
    diagnostics["n_selected_edges"] = len(selected_edges)
    diagnostics["edge_score_threshold"] = edge_threshold

    if len(selected_edges) == 0:
        diagnostics["fallback_used"] = True
        return _finish(*_isolated_nodes(filtered_nodes))

    tracklets_df = build_tracklets_from_selected_edges(selected_edges, filtered_nodes)
    diagnostics["n_tracklets_built"] = len(tracklets_df)

    if len(tracklets_df) == 0:
        diagnostics["fallback_used"] = True
        return _finish(*_isolated_nodes(filtered_nodes))

    min_mean_edge_score_threshold = float(tracklets_df["mean_edge_score"].quantile(config["min_mean_classifier_score_quantile"]))
    kept = apply_tracklet_filter(
        tracklets_df, min_tracklet_length=config["min_tracklet_length"], min_mean_node_score=0.0,
        min_mean_edge_score_threshold=min_mean_edge_score_threshold,
        max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM, max_smoothness_error_um=config["max_smoothness_error_um"],
        keep_top_k=config["keep_top_k_tracklets_per_sample"],
    )
    diagnostics["n_tracklets_kept"] = len(kept)
    diagnostics["min_mean_edge_score_threshold"] = min_mean_edge_score_threshold

    if len(kept) == 0:
        diagnostics["fallback_used"] = True
        return _finish(*_isolated_nodes(filtered_nodes))

    # node_policy is "A_tracklet_nodes_only" for both configs in this
    # milestone, but dispatch on the config value rather than hardcoding.
    if config["node_policy"] == "A_tracklet_nodes_only":
        pred_nodes, pred_edges = reconstruct_policy_a(kept, filtered_nodes)
    else:
        pred_nodes, pred_edges = reconstruct_policy_b(filtered_nodes, kept)

    if len(pred_nodes) == 0:
        diagnostics["fallback_used"] = True
        return _finish(*_isolated_nodes(filtered_nodes))

    return _finish(pred_nodes, pred_edges)

# --------------------------------------------------------------------------- #
# 23. Submission row building + validation
# --------------------------------------------------------------------------- #
SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def build_submission_rows_for_sample(
    dataset_name: str, pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, next_id: int,
) -> tuple[list[dict], int]:
    """Threads a running `id` counter across every dataset in the final
    submission (never resets per-sample) - mirrors conservative_tracker.py's
    established submission-row convention.
    """
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
    return rows, next_id


def build_submission_for_config(
    test_dirs: Sequence[Path], config: dict, trained_model: dict, detector_config: dict = DEFAULT_DETECTOR_CONFIG,
) -> tuple[pd.DataFrame, list[dict]]:
    """Runs test-time inference for EVERY discovered test dataset under one
    config, threading `id`/`node_id` counters across samples so both stay
    globally unique/consecutive across the whole submission.
    """
    all_rows: list[dict] = []
    next_id = 0
    next_node_id = 0
    per_dataset_diagnostics: list[dict] = []
    for i, sample_dir in enumerate(test_dirs):
        t0 = time.time()
        result = run_inference_for_test_sample(sample_dir, config, trained_model, next_node_id, detector_config)
        dataset_name = result["diagnostics"]["dataset"]
        rows, next_id = build_submission_rows_for_sample(dataset_name, result["pred_nodes"], result["pred_edges"], next_id)
        all_rows.extend(rows)
        next_node_id = result["next_node_id"]

        diag = dict(result["diagnostics"])
        diag["elapsed_seconds"] = time.time() - t0
        diag["n_submission_nodes"] = len(result["pred_nodes"])
        diag["n_submission_edges"] = len(result["pred_edges"])
        per_dataset_diagnostics.append(diag)
        fallback_note = " [FALLBACK USED]" if diag["fallback_used"] else ""
        print(
            f"  [{i + 1}/{len(test_dirs)}] {dataset_name}: {diag['n_submission_nodes']} node(s), "
            f"{diag['n_submission_edges']} edge(s) generated in {diag['elapsed_seconds']:.1f}s{fallback_note}"
        )
    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    return submission_df, per_dataset_diagnostics


def validate_submission(df: pd.DataFrame, expected_datasets: Sequence[str]) -> tuple[bool, str]:
    """Structural validation mirroring conservative_tracker.py's
    established checks, extended with duplicate-edge and no-foreign-
    dataset-rows checks. Returns (all_passed, report) rather than raising,
    so the primary AND conservative submissions can both be validated
    unconditionally before either is chosen as the final submission.csv.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("columns match required schema", list(df.columns) == SUBMISSION_COLUMNS, f"got {list(df.columns)}")

    na_cols = df.columns[df.isna().any()].tolist() if len(df) else []
    check("no NaN values anywhere", not na_cols, f"NaN columns: {na_cols}")

    check("id column is consecutive starting at 0", df["id"].tolist() == list(range(len(df))))

    present_datasets = set(df["dataset"].unique()) if len(df) else set()
    missing = sorted(set(expected_datasets) - present_datasets)
    check("every expected test dataset is present", not missing, f"missing: {missing}")

    foreign = sorted(present_datasets - set(expected_datasets))
    check("no accidental train/foreign dataset rows in submission", not foreign, f"foreign datasets: {foreign}")

    check("row_type values are only 'node'/'edge'", set(df["row_type"].unique()) <= {"node", "edge"} if len(df) else True)

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
    check("no negative timepoints among node rows", bool((nodes["t"] >= 0).all()) if len(nodes) else True)

    bad_source_parts, bad_target_parts = [], []
    for dataset, group in edges.groupby("dataset"):
        valid_ids = set(nodes.loc[nodes["dataset"] == dataset, "node_id"])
        bad_source_parts.append(group[~group["source_id"].isin(valid_ids)])
        bad_target_parts.append(group[~group["target_id"].isin(valid_ids)])
    bad_source = pd.concat(bad_source_parts) if bad_source_parts else edges.iloc[0:0]
    bad_target = pd.concat(bad_target_parts) if bad_target_parts else edges.iloc[0:0]
    check("edge source_id references an existing node_id in the same dataset", len(bad_source) == 0, f"{len(bad_source)} bad rows")
    check("edge target_id references an existing node_id in the same dataset", len(bad_target) == 0, f"{len(bad_target)} bad rows")

    dup_edges = edges[edges.duplicated(subset=["dataset", "source_id", "target_id"])]
    check("no duplicate edges", len(dup_edges) == 0, f"{len(dup_edges)} duplicate rows")

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
    return all_passed, report


def print_submission_diagnostics(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label}: shape={df.shape} ---")
    if len(df) == 0:
        print("  (empty)")
        return
    print("  row_type counts:")
    for row_type, count in df["row_type"].value_counts().items():
        print(f"    {row_type}: {count}")
    print("  per-dataset node/edge counts:")
    for dataset, group in df.groupby("dataset"):
        n_nodes = int((group["row_type"] == "node").sum())
        n_edges = int((group["row_type"] == "edge").sum())
        print(f"    {dataset}: nodes={n_nodes} edges={n_edges}")

# --------------------------------------------------------------------------- #
# 24. Part H - orchestration: final training, submission generation for
#     both configs, validation, fallback selection, diagnostics
# --------------------------------------------------------------------------- #
def run_milestone12_submission_pipeline(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT,
    seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    working_dir: str = "/kaggle/working",
) -> dict:
    """Recomputes the exact same fixed-seed ROBUST train-sample selection
    Milestone 11 used, trains the final primary (HistGradientBoosting) and
    conservative-diagnostic (ExtraTrees) classifiers on ALL of it (no
    folds - production fit, not a validation experiment), generates a
    submission for each config against every discovered test dataset,
    validates BOTH unconditionally, and writes submission.csv from
    whichever validates (primary preferred; conservative only as a
    fallback). Never calls any Kaggle submission API and never calls Save
    Version - only writes CSVs/JSON under `working_dir`.
    """
    print("=== Milestone 12: final classifier-tracklet submission builder ===")

    print("\n=== Recomputing Milestone 11's fixed-seed ROBUST train-sample selection ===")
    train_dirs = select_train_samples_robust(n_samples=n_robust_samples, seed=seed)
    print(f"Selected {len(train_dirs)} robust train sample(s): {[d.stem for d in train_dirs]}")

    raw_caches = cache_samples_within_budget(train_dirs, detector_config, time_budget_seconds)
    print(f"Cached raw detection + GT for {len(raw_caches)} train sample(s)")

    node_filter_caches = [precompute_node_filter_cache(rc, NODE_FILTER_CONFIG) for rc in raw_caches]
    train_feature_df = build_all_training_feature_tables(node_filter_caches)
    n_positive = int(train_feature_df["label"].sum()) if len(train_feature_df) else 0
    n_total = len(train_feature_df)
    print(f"\nBuilt {n_total} labeled training candidate row(s): {n_positive} positive, {n_total - n_positive} negative")

    print("\n=== Training final classifiers (no folds - production fit on all selected samples) ===")
    primary_model = fit_final_classifier(
        train_feature_df, PRIMARY_CONFIG["classifier_name"], PRIMARY_CONFIG["negative_ratio"], random_state=seed,
    )
    print(
        f"  primary ({PRIMARY_CONFIG['classifier_name']}): trained on "
        f"{primary_model['n_train_positive']} positive(s) + {primary_model['n_train_negative']} negative(s)"
    )
    conservative_model = fit_final_classifier(
        train_feature_df, CONSERVATIVE_CONFIG["classifier_name"], CONSERVATIVE_CONFIG["negative_ratio"], random_state=seed,
    )
    print(
        f"  conservative ({CONSERVATIVE_CONFIG['classifier_name']}): trained on "
        f"{conservative_model['n_train_positive']} positive(s) + {conservative_model['n_train_negative']} negative(s)"
    )

    print(f"\nClassifier feature list ({len(CLASSIFIER_FEATURE_COLUMNS)}): {CLASSIFIER_FEATURE_COLUMNS}")
    print(f"\nPrimary config: {PRIMARY_CONFIG}")
    print(f"Conservative diagnostic config: {CONSERVATIVE_CONFIG}")

    test_dirs = find_test_zarr_dirs()
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_dirs:
        print("[error] no test .zarr datasets found. Nothing to submit.")
        return {
            "train_dirs": train_dirs, "primary_model": primary_model, "conservative_model": conservative_model,
            "primary_df": pd.DataFrame(columns=SUBMISSION_COLUMNS), "conservative_df": pd.DataFrame(columns=SUBMISSION_COLUMNS),
            "final_df": pd.DataFrame(columns=SUBMISSION_COLUMNS), "final_source": None,
            "primary_valid": False, "conservative_valid": False, "submission_report": {},
        }

    expected_datasets = [p.stem for p in test_dirs]
    print(f"\nDiscovered {len(test_dirs)} test dataset(s): {expected_datasets}")

    print("\n=== Generating PRIMARY submission ===")
    primary_df, primary_diag = build_submission_for_config(test_dirs, PRIMARY_CONFIG, primary_model, detector_config)
    print("\n=== Generating CONSERVATIVE DIAGNOSTIC submission ===")
    conservative_df, conservative_diag = build_submission_for_config(test_dirs, CONSERVATIVE_CONFIG, conservative_model, detector_config)

    print_submission_diagnostics(primary_df, "PRIMARY submission")
    print_submission_diagnostics(conservative_df, "CONSERVATIVE DIAGNOSTIC submission")

    print("\n=== Validating PRIMARY submission ===")
    primary_valid, primary_report = validate_submission(primary_df, expected_datasets)
    print(primary_report)
    print("\n=== Validating CONSERVATIVE DIAGNOSTIC submission ===")
    conservative_valid, conservative_report = validate_submission(conservative_df, expected_datasets)
    print(conservative_report)

    if primary_valid:
        final_df, final_source = primary_df, "primary"
    elif conservative_valid:
        final_df, final_source = conservative_df, "conservative_diagnostic"
        print(
            "\n[warn] PRIMARY submission FAILED validation - falling back to the CONSERVATIVE DIAGNOSTIC "
            "config for submission.csv."
        )
    else:
        raise AssertionError(
            "Both primary and conservative diagnostic submissions FAILED validation - refusing to write submission.csv."
        )

    primary_df.to_csv(out_dir / "submission_primary.csv", index=False)
    conservative_df.to_csv(out_dir / "submission_conservative_diagnostic.csv", index=False)
    final_df.to_csv(out_dir / "submission.csv", index=False)
    print(f"\nFinal submission.csv source: {final_source!r}")
    print(f"submission.csv path: {out_dir / 'submission.csv'}  shape={final_df.shape}")

    pd.DataFrame({"dataset": [d.stem for d in train_dirs]}).to_csv(out_dir / "milestone12_train_samples.csv", index=False)

    per_dataset_rows = [dict(d, config="primary") for d in primary_diag] + [dict(d, config="conservative_diagnostic") for d in conservative_diag]
    pd.DataFrame(per_dataset_rows).to_csv(out_dir / "milestone12_per_dataset_counts.csv", index=False)

    with open(out_dir / "milestone12_primary_config.json", "w") as f:
        json.dump(PRIMARY_CONFIG, f, indent=2, default=str)

    submission_report = {
        "n_train_samples": len(train_dirs), "train_datasets": [d.stem for d in train_dirs],
        "n_train_candidates": n_total, "n_train_positive": n_positive, "n_train_negative": n_total - n_positive,
        "n_test_datasets": len(test_dirs), "test_datasets": expected_datasets,
        "primary_config": PRIMARY_CONFIG, "conservative_config": CONSERVATIVE_CONFIG,
        "primary_valid": primary_valid, "conservative_valid": conservative_valid,
        "final_submission_source": final_source,
        "primary_shape": list(primary_df.shape), "conservative_shape": list(conservative_df.shape),
        "final_shape": list(final_df.shape),
    }
    with open(out_dir / "milestone12_submission_report.json", "w") as f:
        json.dump(submission_report, f, indent=2, default=str)

    print("\nsubmission.csv built and validated. Do not submit until user reviews.")

    return {
        "train_dirs": train_dirs, "primary_model": primary_model, "conservative_model": conservative_model,
        "primary_df": primary_df, "conservative_df": conservative_df, "final_df": final_df,
        "final_source": final_source, "primary_valid": primary_valid, "conservative_valid": conservative_valid,
        "submission_report": submission_report,
    }

# --------------------------------------------------------------------------- #
# 25. Tests
# --------------------------------------------------------------------------- #
def run_milestone12_tests() -> None:
    """Correctness tests for the Milestone 12 submission builder: final
    config parsing, greedy exclusivity, tracklet filtering on classifier
    edge_score, submission schema validation, dangling-edge rejection, and
    a synthetic end-to-end fixture producing a schema-valid submission -
    all on small in-memory fixtures (no disk I/O, no real zarr datasets).
    """
    # Test 1: final config parsing - both configs reference classifiers and
    # edge-selection policies that actually exist, and match the user's spec.
    required_keys = {
        "config_name", "node_filter", "classifier_name", "negative_ratio",
        "edge_selection_policy", "edge_selection_param", "node_policy",
        "min_tracklet_length", "min_mean_classifier_score_quantile",
        "max_smoothness_error_um", "keep_top_k_tracklets_per_sample",
    }
    assert required_keys <= set(PRIMARY_CONFIG.keys())
    assert required_keys <= set(CONSERVATIVE_CONFIG.keys())
    assert PRIMARY_CONFIG["classifier_name"] in CLASSIFIER_BUILDERS
    assert CONSERVATIVE_CONFIG["classifier_name"] in CLASSIFIER_BUILDERS
    assert PRIMARY_CONFIG["edge_selection_policy"] in EDGE_SELECTION_FUNCTIONS
    assert CONSERVATIVE_CONFIG["edge_selection_policy"] in EDGE_SELECTION_FUNCTIONS
    assert PRIMARY_CONFIG["node_filter"]["name"] == "moderate_150_085"
    assert PRIMARY_CONFIG["keep_top_k_tracklets_per_sample"] == 100
    assert PRIMARY_CONFIG["classifier_name"] == "hist_gradient_boosting"
    assert CONSERVATIVE_CONFIG["classifier_name"] == "extra_trees"

    # Test 2: greedy_exclusive_per_frame respects BOTH source capacity (1)
    # and target exclusivity, preferring the higher-scoring edge.
    scored2 = pd.DataFrame({
        "source_id": [1, 1, 2, 3], "target_id": [10, 11, 10, 12], "t": [0, 0, 0, 0],
        "_score": [0.9, 0.85, 0.95, 0.5],
    })
    kept2, _ = select_greedy_exclusive_per_frame(scored2, quantile=0.0, source_capacity=1)
    assert set(zip(kept2["source_id"], kept2["target_id"])) == {(2, 10), (1, 11), (3, 12)}, (
        "source 2's edge to target 10 (score 0.95) must win over source 1's competing edge to the same "
        "target (score 0.9); source 1 then gets its next-best edge to target 11 (score 0.85)"
    )

    # Test 3: tracklet filtering with the classifier's predicted score
    # written into edge_score - a lenient threshold keeps the tracklet, a
    # threshold above its own mean_edge_score filters it out.
    nodes3 = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 2, 0],
        "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 5.0], "x": [0.0] * 4, "score": [0.9, 0.8, 0.85, 0.7],
    })
    selected_edges3 = pd.DataFrame({
        "source_id": [0, 1], "target_id": [1, 2], "distance_um": [0.4, 0.4], "_score": [0.9, 0.85],
    })
    tracklets3 = build_tracklets_from_selected_edges(selected_edges3, nodes3)
    assert len(tracklets3) == 1 and tracklets3.iloc[0]["length_frames"] == 3
    lenient = apply_tracklet_filter(
        tracklets3, min_tracklet_length=2, min_mean_node_score=0.0, min_mean_edge_score_threshold=0.0,
        max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM, max_smoothness_error_um=999.0, keep_top_k=10,
    )
    assert len(lenient) == 1
    strict = apply_tracklet_filter(
        tracklets3, min_tracklet_length=2, min_mean_node_score=0.0, min_mean_edge_score_threshold=0.99,
        max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM, max_smoothness_error_um=999.0, keep_top_k=10,
    )
    assert len(strict) == 0, "the tracklet's own mean_edge_score (~0.875) should fail a 0.99 threshold"

    # Test 4: submission schema validation accepts a well-formed submission
    # and rejects a dangling edge (source/target referencing a node_id that
    # does not exist).
    good_rows = [
        {"id": 0, "dataset": "d1", "row_type": "node", "node_id": 10, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 1, "dataset": "d1", "row_type": "node", "node_id": 11, "t": 1, "z": 0.0, "y": 1.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 2, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 10, "target_id": 11},
    ]
    good_df = pd.DataFrame(good_rows, columns=SUBMISSION_COLUMNS)
    valid4, report4 = validate_submission(good_df, expected_datasets=["d1"])
    assert valid4, report4

    missing_ds_valid, missing_ds_report = validate_submission(good_df, expected_datasets=["d1", "d2"])
    assert not missing_ds_valid
    assert "missing" in missing_ds_report.lower()

    # Test 5: no dangling edges - a target_id that does not correspond to
    # any emitted node_id must fail validation with a specific message.
    dangling_rows = list(good_rows)
    dangling_rows[2] = {**dangling_rows[2], "target_id": 999}
    dangling_df = pd.DataFrame(dangling_rows, columns=SUBMISSION_COLUMNS)
    dangling_valid, dangling_report = validate_submission(dangling_df, expected_datasets=["d1"])
    assert not dangling_valid
    assert "target_id references an existing node_id" in dangling_report

    # Test 6: synthetic end-to-end fixture - a cleanly separable true track
    # (nodes 0,1,2) vs. a far-away noise track (nodes 100,101,102), pushed
    # through greedy-exclusive selection, tracklet construction+filtering,
    # node_policy A reconstruction, and submission-row building, should
    # produce a schema-VALID submission.
    nodes6 = pd.DataFrame({
        "node_id": [0, 1, 2, 100, 101, 102], "t": [0, 1, 2, 0, 1, 2],
        "z": [0.0] * 6, "y": [0.0, 1.0, 2.0, 50.0, 51.0, 52.0], "x": [0.0] * 6,
        "score": [0.9, 0.9, 0.9, 0.5, 0.5, 0.5],
    })
    candidates6 = pd.DataFrame({
        "source_id": [0, 1, 100, 101], "target_id": [1, 2, 101, 102], "t": [0, 1, 0, 1],
        "_score": [0.95, 0.9, 0.05, 0.02], "distance_um": [0.4, 0.4, 0.4, 0.4],
    })
    selected6, _ = select_greedy_exclusive_per_frame(candidates6, quantile=0.0)
    assert len(selected6) == 4, "no source/target conflicts exist in this fixture, so all 4 candidates survive"

    tracklets6 = build_tracklets_from_selected_edges(selected6, nodes6)
    assert len(tracklets6) == 2

    threshold6 = float(tracklets6["mean_edge_score"].quantile(0.5))
    kept6 = apply_tracklet_filter(
        tracklets6, min_tracklet_length=2, min_mean_node_score=0.0, min_mean_edge_score_threshold=threshold6,
        max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM, max_smoothness_error_um=999.0, keep_top_k=10,
    )
    assert len(kept6) >= 1

    pred_nodes6, pred_edges6 = reconstruct_policy_a(kept6, nodes6)
    rows6, _ = build_submission_rows_for_sample("synthetic_test", pred_nodes6, pred_edges6, next_id=0)
    submission6 = pd.DataFrame(rows6, columns=SUBMISSION_COLUMNS)
    assert len(submission6) > 0
    valid6, report6 = validate_submission(submission6, expected_datasets=["synthetic_test"])
    assert valid6, report6

    print("All milestone12_classifier_tracklet_submission tests passed (6/6).")

# --------------------------------------------------------------------------- #
# 26. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone12_pipeline(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT,
    seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
) -> dict:
    """Runs the unit tests, then the full Milestone 12 submission builder:
    recomputes Milestone 11's fixed-seed robust train-sample selection,
    trains the final primary + conservative-diagnostic classifiers on all
    of it, generates and validates both submissions, and writes
    submission.csv (from whichever validates) plus every required
    diagnostic file. Never calls any Kaggle submission API and never calls
    Save Version.
    """
    print("=== Self-test: config parsing, greedy exclusivity, tracklet filtering, submission validation, dangling-edge rejection, and synthetic end-to-end unit tests ===")
    run_milestone12_tests()
    return run_milestone12_submission_pipeline(
        n_robust_samples=n_robust_samples, seed=seed, time_budget_seconds=time_budget_seconds,
    )


if __name__ == "__main__":
    run_milestone12_pipeline()
