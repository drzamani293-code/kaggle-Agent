"""
Biohub - Cell Tracking During Development
Milestone 15: final motion-relink submission builder.

Milestone 14's real-Kaggle OOF validation (GroupKFold on the same 12
fixed-seed robust train samples Milestone 11 used) found: Experiment A
(M12 baseline, greedy_exclusive_per_frame) edge_jaccard=0.013215 (the
harness-reproduction anchor, confirmed within tolerance); Experiment C
(motion_relink only) edge_jaccard=0.019449 - the best of all 9
experiments, though its edge_FP jumped from 1932 to 8432 (a real
precision/recall tradeoff, not a free win); Experiment I_union (an
ensemble of a greedy-exclusive primary + the conservative ExtraTrees
config) edge_jaccard=0.018929 with better precision and stability
(n_samples_with_tp=7 vs 6) than raw motion_relink. Milestone 14's final
recommendation was to port motion_relink into a real submission - this
module does exactly that, with the ensemble idea kept as an explicit,
independently-validated FALLBACK rather than the primary choice, since
motion_relink's higher OOF jaccard is real signal even though its FP
count is also real and non-negligible.

This is the FIRST milestone allowed to build a submission.csv using
motion_relink. It is still a fixed-config, single-pass build - no OOF
grid, no sweeps, "no giant sweeps" - Milestone 14 already answered which
configs are worth trying; this module only answers "does this exact,
already-chosen recipe run cleanly end to end on the real test data."

Three independently-generated, independently-validated variants:

  1. **motion_relink** (new primary): identical to Milestone 12's
     PRIMARY_CONFIG pipeline (same node filter, same trained
     HistGradientBoostingClassifier, same tracklet-filter/node-policy-A
     config) with ONLY the edge-selection step swapped from
     `greedy_exclusive_per_frame` to Milestone 14's already-unit-tested
     `select_edges_motion_relink` (two-pass Hungarian assignment, tight
     gate 6.0um then relaxed gate 10.0um, cost = motion_distance_um +
     0.05*raw_distance_um - 0.75*classifier_score, a per-source reject
     column so a source with no acceptable candidate is left unmatched
     rather than force-assigned).

  2. **union_stable**: the SAME motion_relink pipeline's (pred_nodes,
     pred_edges) unioned with the conservative (ExtraTreesClassifier,
     `per_source_top1_with_score_quantile`) pipeline's own independent
     output for the SAME test sample, via Milestone 14's
     `combine_ensemble(..., mode="union")` - a deterministic PRIORITY
     rule (motion_relink's already-internally-exclusive edges are kept
     whole; the conservative pipeline's edges are added only where they
     don't conflict with an already-used source/target) that NEVER
     compares raw probability scores across the two different classifier
     types. This redefines which side is "primary" in the ensemble
     relative to Milestone 14's own OOF experiment (which used the
     greedy-exclusive config as its ensemble's primary side) - a
     deliberate choice for this milestone specifically, not an
     inconsistency: motion_relink IS this module's new primary.

  3. **m12_baseline_safe**: Milestone 12's ORIGINAL, completely unchanged
     PRIMARY_CONFIG pipeline (greedy_exclusive_per_frame @ 0.9, no graph
     repair at all) - a pure safety net with no new logic whatsoever.

`submission.csv` = whichever of the 3 validates, in priority order
motion_relink -> union_stable -> m12_baseline_safe -> raise (refuse to
write anything) if all 3 fail. All 3 are generated AND validated
UNCONDITIONALLY regardless of which one is ultimately chosen, so a bug in
one is never hidden behind another's success.

Design notes incorporating architecture review:

  - The classifier itself is trained EXACTLY ONCE per type (primary
    HistGradientBoosting, conservative ExtraTrees) and reused across every
    variant that needs it - motion_relink and m12_baseline_safe share the
    IDENTICAL trained primary classifier and therefore identical `_score`
    values per candidate; the two variants differ ONLY in which edge-
    selection procedure consumes those same scores, making the comparison
    between them a clean ablation of the edge-selection step alone.

  - `union_stable`'s node-id consistency relies on `precompute_raw_test_
    sample_cache`'s detection being fully deterministic given the same
    `(sample, node_id_start)` - confirmed (no RNG, no dict/set-iteration-
    order dependency in the detector or NMS path) - so calling the
    underlying per-sample inference twice (once per pipeline) with the
    SAME starting `node_id_start` guarantees both pipelines assign
    IDENTICAL node_ids to the same physical raw detections, making a
    node-id-keyed union valid; the two calls' returned `next_node_id`
    counters are asserted equal as a live consistency check, not just
    assumed.

  - `validate_submission` is extended (never modified in place) with
    structural topology checks reusing Milestone 13's per-dataset-scoped
    `compute_submission_stats` (dedupes edges before computing degree, so
    a duplicate can never inflate out-degree; scopes every node/dangling/
    degree lookup by (dataset, node_id), never a bare global id). The
    t-gap check is `set(t_gap_distribution_keys) <= {"1"}` - NOT a literal
    `{1: n_edges}` equality - because Milestone 12's documented fallback-
    to-isolated-nodes path can legitimately produce a 0-edge dataset for
    an unusually quiet test sample, and an empty t_gap_distribution must
    still PASS this check rather than false-failing a structurally valid
    submission.

No Kaggle submission API is ever called, Save Version is never triggered,
and no OOF sweep is run - fixed configs only. 6C/6D/9/10/11/12/13/14's
files are not modified - all reused logic (the full Milestone 12 test-
time inference/submission/validation pipeline, Milestone 14's motion-
relink/ensemble machinery, Milestone 13's submission-stats computation)
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

# --------------------------------------------------------------------------- #
# 24. Submission-stats computation (copied from Milestone 13, for the
#     extended topology validation)
# --------------------------------------------------------------------------- #
def detect_submission_schema(df: pd.DataFrame | None) -> tuple[str, str]:
    """Never assumes a foreign submission matches our SUBMISSION_COLUMNS
    layout - checks explicitly and reports WHY detection failed rather
    than guessing at node/edge semantics that could silently misclassify
    rows.
    """
    if df is None or len(df.columns) == 0:
        return "empty", "no dataframe/columns"
    missing = sorted(set(SUBMISSION_COLUMNS) - set(df.columns))
    if missing:
        return "unrecognized", f"missing columns: {missing}"
    row_types = set(df["row_type"].unique()) if len(df) else set()
    if not row_types <= {"node", "edge"}:
        return "unrecognized", f"row_type has unexpected values: {sorted(row_types)}"
    return "recognized", "matches SUBMISSION_COLUMNS schema with row_type in {node,edge}"


def compute_generic_stats(df: pd.DataFrame) -> dict:
    """Stats that make sense regardless of whether the schema is
    recognized - always reported, even for a completely foreign layout.
    """
    return {
        "shape": list(df.shape), "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "nunique": {c: int(df[c].nunique()) for c in df.columns},
    }


def compute_submission_stats(df: pd.DataFrame) -> dict:
    """Computes the full requested stat suite for a SUBMISSION_COLUMNS-
    shaped dataframe: shape/counts, node_id uniqueness, dangling/duplicate
    edges, t-gap distribution, in/out-degree distributions, division-like
    and multi-parent counts, physical edge-distance percentiles, per-
    dataset coordinate ranges, and per-timepoint node/edge density.

    All node/dangling/degree/t-gap computations are scoped per-dataset via
    an explicit (dataset, node_id) key - a foreign submission's node_id is
    NOT assumed globally unique the way ours is, so a bare global lookup
    could silently mismatch nodes across datasets that reuse small integer
    ids. Degree distributions are reindexed over the FULL node set so
    degree-0 nodes are never silently dropped. Duplicate edges are deduped
    on the ORDERED (directed) (dataset, source_id, target_id) triple,
    since tracking edges are directed in time.
    """
    schema, reason = detect_submission_schema(df)
    result: dict = {"schema": schema, "schema_reason": reason, "generic": compute_generic_stats(df)}
    empty_frames = {
        "per_dataset_counts": pd.DataFrame(), "degree_stats": pd.DataFrame(),
        "edge_distance_stats": pd.DataFrame(), "timepoint_density": pd.DataFrame(),
    }
    if schema != "recognized":
        result.update(empty_frames)
        return result

    nodes = df[df["row_type"] == "node"].copy()
    edges = df[df["row_type"] == "edge"].copy()
    datasets = sorted(df["dataset"].unique().tolist())

    result["n_node_rows"] = len(nodes)
    result["n_edge_rows"] = len(edges)
    result["datasets"] = datasets
    result["edge_to_node_ratio"] = (len(edges) / len(nodes)) if len(nodes) > 0 else float("nan")

    per_dataset_rows = []
    for dataset in datasets:
        n_nodes_d = int((nodes["dataset"] == dataset).sum())
        n_edges_d = int((edges["dataset"] == dataset).sum())
        per_dataset_rows.append({
            "dataset": dataset, "n_nodes": n_nodes_d, "n_edges": n_edges_d,
            "edge_to_node_ratio": (n_edges_d / n_nodes_d) if n_nodes_d > 0 else float("nan"),
        })
    result["per_dataset_counts"] = pd.DataFrame(per_dataset_rows)

    result["node_id_unique_global"] = bool(nodes["node_id"].is_unique)
    result["node_id_unique_per_dataset"] = {ds: bool(g["node_id"].is_unique) for ds, g in nodes.groupby("dataset")}

    # (dataset, node_id) -> t / (z,y,x) lookups - never a bare global node_id key
    node_t_lookup: dict[tuple, int] = {}
    node_coord_lookup: dict[tuple, tuple] = {}
    valid_ids_by_dataset: dict[str, set] = {}
    for dataset, group in nodes.groupby("dataset"):
        valid_ids_by_dataset[dataset] = set(group["node_id"])
        for nid, t, z, y, x in zip(group["node_id"], group["t"], group["z"], group["y"], group["x"]):
            node_t_lookup[(dataset, int(nid))] = int(t)
            node_coord_lookup[(dataset, int(nid))] = (float(z), float(y), float(x))

    dangling_mask, t_gaps = [], []
    for row in edges.itertuples():
        valid_ids = valid_ids_by_dataset.get(row.dataset, set())
        src_ok, tgt_ok = row.source_id in valid_ids, row.target_id in valid_ids
        dangling_mask.append(not (src_ok and tgt_ok))
        t_gaps.append(node_t_lookup[(row.dataset, int(row.target_id))] - node_t_lookup[(row.dataset, int(row.source_id))] if (src_ok and tgt_ok) else None)

    # Explicit dtype+index construction (not a bare list) - an EMPTY plain
    # Python list defaults to float64 dtype on assign, and pandas then
    # silently mis-handles `edges[~edges["_dangling"]]` as column
    # selection rather than boolean row-filtering, returning zero columns.
    # This only ever surfaces on a completely edge-less submission (e.g.
    # a node-only fallback dataset), which is a real, legitimate case.
    edges = edges.assign(
        _dangling=pd.Series(dangling_mask, dtype=bool, index=edges.index),
        _t_gap=pd.Series(t_gaps, dtype=object, index=edges.index),
    )
    result["dangling_edge_count"] = int(sum(dangling_mask))
    result["duplicate_edge_count"] = int(edges.duplicated(subset=["dataset", "source_id", "target_id"]).sum())

    # Degree/t-gap/distance/density stats all operate on a DEDUPED, non-
    # dangling edge set - a duplicate row must never be double-counted as
    # if it were 2 distinct edges (it would otherwise silently inflate
    # out-degree and misclassify a node as "division-like").
    valid_edges = edges[~edges["_dangling"]].drop_duplicates(subset=["dataset", "source_id", "target_id"])

    valid_t_gaps = [g for g in valid_edges["_t_gap"] if g is not None]
    if valid_t_gaps:
        t_gap_counts = pd.Series(valid_t_gaps).value_counts().sort_index()
        result["t_gap_distribution"] = {str(k): int(v) for k, v in t_gap_counts.items()}
        result["frac_edges_t_to_t1"] = float((pd.Series(valid_t_gaps) == 1).mean())
    else:
        result["t_gap_distribution"] = {}
        result["frac_edges_t_to_t1"] = float("nan")

    node_keys = pd.MultiIndex.from_frame(nodes[["dataset", "node_id"]])
    out_degree_full = valid_edges.groupby(["dataset", "source_id"]).size().reindex(node_keys, fill_value=0)
    in_degree_full = valid_edges.groupby(["dataset", "target_id"]).size().reindex(node_keys, fill_value=0)

    degree_df = pd.DataFrame({
        "dataset": node_keys.get_level_values(0), "node_id": node_keys.get_level_values(1),
        "out_degree": out_degree_full.to_numpy(), "in_degree": in_degree_full.to_numpy(),
    })
    result["degree_stats"] = degree_df
    result["out_degree_distribution"] = {str(k): int(v) for k, v in degree_df["out_degree"].value_counts().sort_index().items()}
    result["in_degree_distribution"] = {str(k): int(v) for k, v in degree_df["in_degree"].value_counts().sort_index().items()}
    result["division_like_source_count"] = int((degree_df["out_degree"] == 2).sum())
    result["multi_parent_child_count"] = int((degree_df["in_degree"] > 1).sum())

    distances = []
    for row in valid_edges.itertuples():
        src_c = node_coord_lookup.get((row.dataset, int(row.source_id)))
        tgt_c = node_coord_lookup.get((row.dataset, int(row.target_id)))
        if src_c is None or tgt_c is None:
            continue
        dz = (tgt_c[0] - src_c[0]) * VOXEL_SIZE_UM["z"]
        dy = (tgt_c[1] - src_c[1]) * VOXEL_SIZE_UM["y"]
        dx = (tgt_c[2] - src_c[2]) * VOXEL_SIZE_UM["x"]
        distances.append(float(np.sqrt(dz * dz + dy * dy + dx * dx)))

    if distances:
        dist_series = pd.Series(distances)
        result["edge_distance_summary"] = {
            "mean": float(dist_series.mean()), "median": float(dist_series.median()),
            "p90": float(dist_series.quantile(0.90)), "p95": float(dist_series.quantile(0.95)),
            "p99": float(dist_series.quantile(0.99)), "max": float(dist_series.max()),
        }
        result["edge_distance_stats"] = pd.DataFrame([result["edge_distance_summary"]])
    else:
        result["edge_distance_summary"] = {}
        result["edge_distance_stats"] = pd.DataFrame()

    coord_ranges = {}
    for dataset, group in nodes.groupby("dataset"):
        coord_ranges[dataset] = {
            "z_min": float(group["z"].min()), "z_max": float(group["z"].max()),
            "y_min": float(group["y"].min()), "y_max": float(group["y"].max()),
            "x_min": float(group["x"].min()), "x_max": float(group["x"].max()),
            "t_min": int(group["t"].min()), "t_max": int(group["t"].max()),
        }
    result["coordinate_ranges_per_dataset"] = coord_ranges

    edge_source_t_counts: dict[tuple, int] = {}
    for row in valid_edges.itertuples():
        src_t = node_t_lookup.get((row.dataset, int(row.source_id)))
        if src_t is None:
            continue
        key = (row.dataset, src_t)
        edge_source_t_counts[key] = edge_source_t_counts.get(key, 0) + 1

    tp_rows = []
    for dataset, group in nodes.groupby("dataset"):
        for t, t_group in group.groupby("t"):
            tp_rows.append({
                "dataset": dataset, "t": int(t), "n_nodes": len(t_group),
                "n_edges_starting_here": edge_source_t_counts.get((dataset, int(t)), 0),
            })
    result["timepoint_density"] = pd.DataFrame(tp_rows)

    return result

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 25. Motion-relink two-pass Hungarian assignment (copied from Milestone 14)
# --------------------------------------------------------------------------- #
# 24. Experiment C - motion-relink two-pass Hungarian assignment
# --------------------------------------------------------------------------- #
MOTION_RELINK_TIGHT_GATE_UM = 6.0
MOTION_RELINK_RELAXED_GATE_UM = 10.0
MOTION_RELINK_CLASSIFIER_BONUS = 0.75
MOTION_RELINK_RAW_DISTANCE_WEIGHT = 0.05
MOTION_RELINK_REJECT_COST = 1000.0
MOTION_RELINK_BIG_COST = 1e6


def _hungarian_pass_with_reject(candidates_df: pd.DataFrame, reject_cost: float = MOTION_RELINK_REJECT_COST) -> tuple[pd.DataFrame, set, set]:
    """One Hungarian assignment pass with a per-source REJECT column, so
    `linear_sum_assignment` can leave a source unmatched (cost=reject_cost)
    rather than being forced onto its least-bad real option - avoiding the
    forced-assignment failure mode this codebase moved away from
    (Milestone 9). `candidates_df` must already be gated to only the rows
    that pass THIS pass's motion-distance threshold (a gate-failing
    candidate is never placed in the matrix at all, so it can never
    out-compete a real gate-passing option on cost alone) and must carry a
    `cost` column. Sources/targets are sorted by id for deterministic,
    run-to-run-reproducible tie-breaking.
    """
    if len(candidates_df) == 0:
        return candidates_df.iloc[0:0], set(), set()

    sources = sorted(candidates_df["source_id"].unique())
    targets = sorted(candidates_df["target_id"].unique())
    src_idx = {s: i for i, s in enumerate(sources)}
    tgt_idx = {t: j for j, t in enumerate(targets)}
    n_s, n_t = len(sources), len(targets)

    cost_matrix = np.full((n_s, n_t + n_s), MOTION_RELINK_BIG_COST)
    row_lookup: dict[tuple, int] = {}
    for row in candidates_df.itertuples():
        i, j = src_idx[row.source_id], tgt_idx[row.target_id]
        if row.cost < cost_matrix[i, j]:
            cost_matrix[i, j] = row.cost
            row_lookup[(i, j)] = row.Index
    for i in range(n_s):
        cost_matrix[i, n_t + i] = reject_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    accepted_indices: list = []
    used_sources: set = set()
    used_targets: set = set()
    for r, c in zip(row_ind, col_ind):
        if c < n_t and (r, c) in row_lookup:
            accepted_indices.append(row_lookup[(r, c)])
            used_sources.add(sources[r])
            used_targets.add(targets[c])

    accepted_df = candidates_df.loc[accepted_indices] if accepted_indices else candidates_df.iloc[0:0]
    return accepted_df, used_sources, used_targets


def select_edges_motion_relink(
    scored_df: pd.DataFrame, tight_gate: float = MOTION_RELINK_TIGHT_GATE_UM, relaxed_gate: float = MOTION_RELINK_RELAXED_GATE_UM,
    classifier_bonus: float = MOTION_RELINK_CLASSIFIER_BONUS, raw_distance_weight: float = MOTION_RELINK_RAW_DISTANCE_WEIGHT,
) -> tuple[pd.DataFrame, dict]:
    """Two-pass Hungarian assignment per frame transition, processed
    STRICTLY in ascending-t order - a sequential/stateful algorithm:
    transition t's own tight-then-relaxed passes only depend on
    transition t's own state, and transition t+1's velocity lookups only
    depend on transitions <= t, both satisfied by a single ascending pass.
    Velocity is modeled as "repeat the last observed single-step
    displacement" - the (dz_um,dy_um,dx_um) of whichever edge was most
    recently accepted INTO a node, reused as that node's predicted
    displacement at the NEXT transition (zero-velocity/cold-start for any
    node with no accepted predecessor yet). Cost = motion_distance_um +
    raw_distance_weight*raw_distance_um - classifier_bonus*classifier_score.
    """
    velocity_by_node: dict[int, np.ndarray] = {}
    accepted_frames: list[pd.DataFrame] = []
    n_tight, n_relaxed = 0, 0

    for t in sorted(scored_df["t"].unique()):
        candidates_t = scored_df[scored_df["t"] == t].copy()
        if len(candidates_t) == 0:
            continue

        disp = candidates_t[["dz_um", "dy_um", "dx_um"]].to_numpy(dtype=float)
        vel = np.array([velocity_by_node.get(sid, np.zeros(3)) for sid in candidates_t["source_id"]])
        candidates_t["motion_distance_um"] = np.linalg.norm(disp - vel, axis=1)
        candidates_t["cost"] = (
            candidates_t["motion_distance_um"] + raw_distance_weight * candidates_t["distance_um"]
            - classifier_bonus * candidates_t["_score"]
        )

        tight_df = candidates_t[candidates_t["motion_distance_um"] <= tight_gate]
        pass1_df, used_src1, used_tgt1 = _hungarian_pass_with_reject(tight_df)
        if len(pass1_df):
            pass1_df = pass1_df.assign(assignment_pass="tight")
            accepted_frames.append(pass1_df)
            n_tight += len(pass1_df)
            for row in pass1_df.itertuples():
                velocity_by_node[row.target_id] = np.array([row.dz_um, row.dy_um, row.dx_um])

        remaining_df = candidates_t[
            (~candidates_t["source_id"].isin(used_src1)) & (~candidates_t["target_id"].isin(used_tgt1))
            & (candidates_t["motion_distance_um"] <= relaxed_gate)
        ]
        pass2_df, _, _ = _hungarian_pass_with_reject(remaining_df)
        if len(pass2_df):
            pass2_df = pass2_df.assign(assignment_pass="relaxed")
            accepted_frames.append(pass2_df)
            n_relaxed += len(pass2_df)
            for row in pass2_df.itertuples():
                velocity_by_node[row.target_id] = np.array([row.dz_um, row.dy_um, row.dx_um])

    if accepted_frames:
        selected_edges = pd.concat(accepted_frames, ignore_index=False)
    else:
        selected_edges = scored_df.iloc[0:0].copy()
        selected_edges["assignment_pass"] = pd.Series(dtype=object)

    diagnostics = {"n_tight": n_tight, "n_relaxed": n_relaxed, "n_total": n_tight + n_relaxed}
    return selected_edges, diagnostics

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 26. Primary/conservative ensemble union+intersection (copied from Milestone 14)
# --------------------------------------------------------------------------- #
# 27. Experiment I - primary/conservative ensemble diagnostic
# --------------------------------------------------------------------------- #
def ensemble_union_edges(primary_edges: pd.DataFrame, conservative_edges: pd.DataFrame) -> pd.DataFrame:
    """Union with a deterministic PRIORITY rule (primary always wins any
    conflict) - NEVER compares raw classifier scores across the two
    different classifiers (HistGradientBoosting vs ExtraTrees probability
    scales are not comparable - the same cross-classifier pooling trap
    Milestone 11 was built to avoid). Primary's already-internally-
    exclusive edge set is kept whole; conservative's edges are added only
    where they don't conflict with an already-used source or target, so
    the union stays topologically exclusive without any score comparison.
    """
    used_sources = set(primary_edges["source_id"]) if len(primary_edges) else set()
    used_targets = set(primary_edges["target_id"]) if len(primary_edges) else set()
    keep_indices = []
    for row in conservative_edges.itertuples():
        if row.source_id in used_sources or row.target_id in used_targets:
            continue
        keep_indices.append(row.Index)
        used_sources.add(row.source_id)
        used_targets.add(row.target_id)
    added_from_conservative = conservative_edges.loc[keep_indices] if keep_indices else conservative_edges.iloc[0:0]
    return pd.concat([primary_edges, added_from_conservative], ignore_index=True)


def ensemble_intersection_edges(primary_edges: pd.DataFrame, conservative_edges: pd.DataFrame) -> pd.DataFrame:
    """Intersection needs no repair pass at all - a subset of an already-
    exclusive edge set is trivially still exclusive.
    """
    if len(primary_edges) == 0 or len(conservative_edges) == 0:
        return conservative_edges.iloc[0:0]
    primary_pairs = set(zip(primary_edges["source_id"], primary_edges["target_id"]))
    mask = [(s, t) in primary_pairs for s, t in zip(conservative_edges["source_id"], conservative_edges["target_id"])]
    return conservative_edges[pd.Series(mask, index=conservative_edges.index)].reset_index(drop=True)


def combine_ensemble(
    primary_nodes: pd.DataFrame, primary_edges: pd.DataFrame, conservative_nodes: pd.DataFrame, conservative_edges: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "union":
        combined_edges = ensemble_union_edges(primary_edges, conservative_edges)
        combined_nodes = pd.concat([primary_nodes, conservative_nodes], ignore_index=True).drop_duplicates(subset="node_id").reset_index(drop=True)
        return combined_nodes, combined_edges
    if mode == "intersection":
        combined_edges = ensemble_intersection_edges(primary_edges, conservative_edges)
        referenced_ids = set(combined_edges["source_id"]) | set(combined_edges["target_id"]) if len(combined_edges) else set()
        combined_nodes = primary_nodes[primary_nodes["node_id"].isin(referenced_ids)].reset_index(drop=True)
        return combined_nodes, combined_edges
    raise ValueError(f"unknown ensemble mode {mode!r}, expected 'union' or 'intersection'")

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 27. Motion-relink config + EDGE_SELECTION_FUNCTIONS extension
# --------------------------------------------------------------------------- #
def motion_relink_edge_selection(scored_df: pd.DataFrame, _unused_param=None) -> tuple[pd.DataFrame, None]:
    """Thin adapter matching the 2-argument `(scored_df, param) ->
    (selected_edges_df, threshold)` shape `run_inference_for_test_sample`
    expects from every `EDGE_SELECTION_FUNCTIONS` entry - motion_relink
    has no single quantile threshold (it uses fixed tight/relaxed gates
    and a classifier bonus instead), so this always returns `None` for
    the threshold slot. Nothing downstream branches on that value - it is
    only ever stashed into a diagnostics dict.
    """
    selected, _diag = select_edges_motion_relink(
        scored_df, tight_gate=MOTION_RELINK_TIGHT_GATE_UM, relaxed_gate=MOTION_RELINK_RELAXED_GATE_UM,
        classifier_bonus=MOTION_RELINK_CLASSIFIER_BONUS, raw_distance_weight=MOTION_RELINK_RAW_DISTANCE_WEIGHT,
    )
    return selected, None


EDGE_SELECTION_FUNCTIONS["motion_relink"] = motion_relink_edge_selection

# Identical to PRIMARY_CONFIG in every respect except the edge-selection
# step - same node filter, same trained classifier, same tracklet-filter/
# node-policy - so motion_relink and m12_baseline_safe differ ONLY in
# which procedure consumes the SAME classifier scores.
MOTION_RELINK_CONFIG = {
    "config_name": "motion_relink",
    "node_filter": NODE_FILTER_CONFIG,
    "classifier_name": "hist_gradient_boosting",
    "negative_ratio": 100,
    "edge_selection_policy": "motion_relink",
    "edge_selection_param": None,  # unused - motion_relink uses its own fixed MOTION_RELINK_* constants
    "node_policy": "A_tracklet_nodes_only",
    "min_tracklet_length": 5,
    "min_mean_classifier_score_quantile": 0.7,
    "max_smoothness_error_um": 3.0,
    "keep_top_k_tracklets_per_sample": 100,
}

# --------------------------------------------------------------------------- #
# 28. Extended submission validation (topology checks layered on top of
#     Milestone 12's base `validate_submission`, never modifying it)
# --------------------------------------------------------------------------- #
def validate_submission_topology(df: pd.DataFrame) -> tuple[bool, str, dict]:
    """Structural topology checks reusing Milestone 13's per-dataset-
    scoped `compute_submission_stats` (dedupes edges before computing
    degree, scopes every lookup by (dataset, node_id), never a bare
    global id). The t-gap check is `keys <= {"1"}` - NOT a literal
    `{1: n_edges}` equality - because Milestone 12's documented fallback-
    to-isolated-nodes path can legitimately yield a 0-edge dataset for an
    unusually quiet test sample, and an empty t_gap_distribution must
    still PASS rather than false-failing a structurally valid submission.
    """
    stats = compute_submission_stats(df)
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if stats["schema"] != "recognized":
        check("schema recognized for topology stats", False, stats["schema_reason"])
    else:
        t_gap_keys = set(stats["t_gap_distribution"].keys())
        check(
            "t_gap distribution contains only gap=1 (or is empty)", t_gap_keys <= {"1"},
            f"t_gap_distribution={stats['t_gap_distribution']}",
        )
        out_degree_keys = [int(k) for k in stats["out_degree_distribution"].keys()]
        max_out_degree = max(out_degree_keys, default=0)
        check("max out-degree <= 1", max_out_degree <= 1, f"out_degree_distribution={stats['out_degree_distribution']}")
        check(
            "no multi-parent children (in-degree > 1)", stats["multi_parent_child_count"] == 0,
            f"multi_parent_child_count={stats['multi_parent_child_count']}",
        )

    lines = ["Topology validation report:"]
    all_passed = True
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_passed = False
        line = f"  [{mark}] {c['name']}"
        if c["detail"] and not c["passed"]:
            line += f"  -- {c['detail']}"
        lines.append(line)
    return all_passed, "\n".join(lines), stats


def validate_submission_full(df: pd.DataFrame, expected_datasets: Sequence[str]) -> tuple[bool, str, dict]:
    """Runs Milestone 12's base schema/dangling/duplicate/uniqueness
    checks AND the new topology checks, unconditionally, on the same
    submission. Both must pass for the combined result to pass.
    """
    base_ok, base_report = validate_submission(df, expected_datasets)
    topo_ok, topo_report, stats = validate_submission_topology(df)
    return (base_ok and topo_ok), base_report + "\n" + topo_report, stats

# --------------------------------------------------------------------------- #
# 29. Union-stable inference (motion_relink pipeline unioned with the
#     conservative pipeline for the SAME test sample)
# --------------------------------------------------------------------------- #
def run_union_inference_for_test_sample(
    sample_zarr_dir: Path, motion_relink_model: dict, conservative_model: dict, node_id_start: int,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
) -> dict:
    """Runs the motion_relink pipeline AND the conservative pipeline
    independently for the SAME test sample, both starting from the SAME
    `node_id_start` - detection is fully deterministic given
    `(sample, node_id_start)` (no RNG, no dict/set-iteration-order
    dependency), so both pipelines assign IDENTICAL node_ids to the same
    physical raw detections, making a node-id-keyed union valid. The two
    calls' returned `next_node_id` are asserted equal as a live
    consistency check, not just assumed.
    """
    primary_result = run_inference_for_test_sample(sample_zarr_dir, MOTION_RELINK_CONFIG, motion_relink_model, node_id_start, detector_config)
    conservative_result = run_inference_for_test_sample(sample_zarr_dir, CONSERVATIVE_CONFIG, conservative_model, node_id_start, detector_config)
    assert primary_result["next_node_id"] == conservative_result["next_node_id"], (
        "node_id threading diverged between the motion_relink and conservative pipelines for the same "
        "sample - detection is expected to be fully deterministic given the same (sample, node_id_start)."
    )

    union_nodes, union_edges = combine_ensemble(
        primary_result["pred_nodes"], primary_result["pred_edges"],
        conservative_result["pred_nodes"], conservative_result["pred_edges"], mode="union",
    )

    diagnostics = {
        "dataset": primary_result["diagnostics"]["dataset"],
        "n_motion_relink_nodes": len(primary_result["pred_nodes"]), "n_motion_relink_edges": len(primary_result["pred_edges"]),
        "n_conservative_nodes": len(conservative_result["pred_nodes"]), "n_conservative_edges": len(conservative_result["pred_edges"]),
        "n_union_nodes": len(union_nodes), "n_union_edges": len(union_edges),
        "motion_relink_fallback_used": primary_result["diagnostics"]["fallback_used"],
        "conservative_fallback_used": conservative_result["diagnostics"]["fallback_used"],
    }
    return {"pred_nodes": union_nodes, "pred_edges": union_edges, "next_node_id": primary_result["next_node_id"], "diagnostics": diagnostics}


def build_union_submission(
    test_dirs: Sequence[Path], motion_relink_model: dict, conservative_model: dict, detector_config: dict = DEFAULT_DETECTOR_CONFIG,
) -> tuple[pd.DataFrame, list[dict]]:
    all_rows: list[dict] = []
    next_id = 0
    next_node_id = 0
    per_dataset_diagnostics: list[dict] = []
    for i, sample_dir in enumerate(test_dirs):
        t0 = time.time()
        result = run_union_inference_for_test_sample(sample_dir, motion_relink_model, conservative_model, next_node_id, detector_config)
        dataset_name = result["diagnostics"]["dataset"]
        rows, next_id = build_submission_rows_for_sample(dataset_name, result["pred_nodes"], result["pred_edges"], next_id)
        all_rows.extend(rows)
        next_node_id = result["next_node_id"]

        diag = dict(result["diagnostics"])
        diag["elapsed_seconds"] = time.time() - t0
        per_dataset_diagnostics.append(diag)
        print(
            f"  [{i + 1}/{len(test_dirs)}] {dataset_name}: union={diag['n_union_nodes']} node(s)/{diag['n_union_edges']} edge(s) "
            f"(motion_relink={diag['n_motion_relink_nodes']}/{diag['n_motion_relink_edges']}, "
            f"conservative={diag['n_conservative_nodes']}/{diag['n_conservative_edges']}) in {diag['elapsed_seconds']:.1f}s"
        )
    submission_df = pd.DataFrame(all_rows, columns=SUBMISSION_COLUMNS)
    return submission_df, per_dataset_diagnostics

# --------------------------------------------------------------------------- #
# 30. Part H - orchestration: training, 3 submission variants, fallback
#     selection, diagnostics
# --------------------------------------------------------------------------- #
M14_OOF_REFERENCE = {
    "A": {"edge_TP": 103, "edge_FP": 1932, "edge_FN": 5759, "edge_jaccard": 0.013215, "n_samples": 12, "n_samples_with_tp": 6},
    "C": {"edge_TP": 278, "edge_FP": 8432, "edge_FN": 5584, "edge_jaccard": 0.019449, "precision": 0.031917, "recall": 0.047424, "n_samples": 12, "n_samples_with_tp": 5},
    "I_union": {"edge_TP": 170, "edge_FP": 3119, "edge_FN": 5692, "edge_jaccard": 0.018929, "precision": 0.051687, "recall": 0.029000, "n_samples": 12, "n_samples_with_tp": 7},
}


def select_final_variant(motion_relink_valid: bool, union_stable_valid: bool, baseline_valid: bool) -> str | None:
    """Priority order motion_relink -> union_stable -> m12_baseline_safe ->
    None (caller must raise rather than write submission.csv). Extracted
    as a small pure function so the fallback logic is directly unit-
    testable without needing a full pipeline run.
    """
    if motion_relink_valid:
        return "motion_relink"
    if union_stable_valid:
        return "union_stable"
    if baseline_valid:
        return "m12_baseline_safe"
    return None


def run_milestone15_submission_pipeline(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS, working_dir: str = "/kaggle/working",
) -> dict:
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Milestone 15: final motion-relink submission builder ===")
    print("M14 OOF reference (GroupKFold, same 12 robust train samples):")
    for exp_id, ref in M14_OOF_REFERENCE.items():
        print(f"  {exp_id}: {ref}")

    print("\n=== Training (same 12 robust samples as M11/M12/M14) ===")
    train_dirs = select_train_samples_robust(n_samples=n_robust_samples, seed=seed)
    print(f"Selected {len(train_dirs)} robust train sample(s): {[d.stem for d in train_dirs]}")
    raw_caches = cache_samples_within_budget(train_dirs, DEFAULT_DETECTOR_CONFIG, time_budget_seconds)
    node_filter_caches = [precompute_node_filter_cache(rc, NODE_FILTER_CONFIG) for rc in raw_caches]
    train_feature_df = build_all_training_feature_tables(node_filter_caches)
    n_positive = int(train_feature_df["label"].sum()) if len(train_feature_df) else 0
    print(f"Built {len(train_feature_df)} labeled training candidate row(s): {n_positive} positive, {len(train_feature_df) - n_positive} negative")

    primary_model = fit_final_classifier(train_feature_df, "hist_gradient_boosting", 100, random_state=seed)
    print(
        f"  primary (hist_gradient_boosting): trained on {primary_model['n_train_positive']} positive(s) + "
        f"{primary_model['n_train_negative']} negative(s) - shared by motion_relink AND m12_baseline_safe"
    )
    conservative_model = fit_final_classifier(train_feature_df, "extra_trees", 100, random_state=seed)
    print(f"  conservative (extra_trees): trained on {conservative_model['n_train_positive']} positive(s) + {conservative_model['n_train_negative']} negative(s)")

    test_dirs = find_test_zarr_dirs()
    if not test_dirs:
        print("[error] no test .zarr datasets found. Nothing to submit.")
        return {"final_source": None, "final_df": pd.DataFrame(columns=SUBMISSION_COLUMNS)}

    expected_datasets = [p.stem for p in test_dirs]
    print(f"\nDiscovered {len(test_dirs)} test dataset(s): {expected_datasets}")

    print("\n=== Generating MOTION_RELINK submission (new primary) ===")
    motion_relink_df, motion_relink_diag = build_submission_for_config(test_dirs, MOTION_RELINK_CONFIG, primary_model, DEFAULT_DETECTOR_CONFIG)

    print("\n=== Generating UNION_STABLE submission (fallback #1) ===")
    union_stable_df, union_stable_diag = build_union_submission(test_dirs, primary_model, conservative_model, DEFAULT_DETECTOR_CONFIG)

    print("\n=== Generating M12_BASELINE_SAFE submission (fallback #2) ===")
    baseline_df, baseline_diag = build_submission_for_config(test_dirs, PRIMARY_CONFIG, primary_model, DEFAULT_DETECTOR_CONFIG)

    print_submission_diagnostics(motion_relink_df, "MOTION_RELINK submission")
    print_submission_diagnostics(union_stable_df, "UNION_STABLE submission")
    print_submission_diagnostics(baseline_df, "M12_BASELINE_SAFE submission")

    print("\n=== Validating MOTION_RELINK submission ===")
    motion_relink_valid, motion_relink_report, motion_relink_stats = validate_submission_full(motion_relink_df, expected_datasets)
    print(motion_relink_report)
    print("\n=== Validating UNION_STABLE submission ===")
    union_stable_valid, union_stable_report, union_stable_stats = validate_submission_full(union_stable_df, expected_datasets)
    print(union_stable_report)
    print("\n=== Validating M12_BASELINE_SAFE submission ===")
    baseline_valid, baseline_report, baseline_stats = validate_submission_full(baseline_df, expected_datasets)
    print(baseline_report)

    final_source = select_final_variant(motion_relink_valid, union_stable_valid, baseline_valid)
    if final_source is None:
        raise AssertionError(
            "All 3 submission variants (motion_relink, union_stable, m12_baseline_safe) FAILED validation - "
            "refusing to write submission.csv."
        )
    final_df = {"motion_relink": motion_relink_df, "union_stable": union_stable_df, "m12_baseline_safe": baseline_df}[final_source]
    if final_source != "motion_relink":
        print(f"\n[warn] falling back to {final_source!r} - a higher-priority variant failed validation.")

    motion_relink_df.to_csv(out_dir / "submission_motion_relink.csv", index=False)
    union_stable_df.to_csv(out_dir / "submission_union_stable.csv", index=False)
    baseline_df.to_csv(out_dir / "submission_m12_baseline_safe.csv", index=False)
    final_df.to_csv(out_dir / "submission.csv", index=False)
    print(f"\nFinal submission.csv source: {final_source!r}")
    print(f"submission.csv path: {out_dir / 'submission.csv'}  shape={final_df.shape}")

    pd.DataFrame({"dataset": [d.stem for d in train_dirs]}).to_csv(out_dir / "milestone15_train_samples.csv", index=False)
    with open(out_dir / "milestone15_primary_config.json", "w") as f:
        json.dump(MOTION_RELINK_CONFIG, f, indent=2, default=str)

    variant_stats_map = {"motion_relink": motion_relink_stats, "union_stable": union_stable_stats, "m12_baseline_safe": baseline_stats}
    variant_valid_map = {"motion_relink": motion_relink_valid, "union_stable": union_stable_valid, "m12_baseline_safe": baseline_valid}
    variant_df_map = {"motion_relink": motion_relink_df, "union_stable": union_stable_df, "m12_baseline_safe": baseline_df}

    variant_counts_rows = [
        {
            "variant": name, "valid": variant_valid_map[name], "total_rows": list(df.shape)[0],
            "n_node_rows": stats.get("n_node_rows"), "n_edge_rows": stats.get("n_edge_rows"),
            "edge_to_node_ratio": stats.get("edge_to_node_ratio"),
        }
        for name, stats, df in ((n, variant_stats_map[n], variant_df_map[n]) for n in ("motion_relink", "union_stable", "m12_baseline_safe"))
    ]
    pd.DataFrame(variant_counts_rows).to_csv(out_dir / "milestone15_variant_counts.csv", index=False)

    topology_rows = [
        {
            "variant": name, "dangling_edge_count": stats.get("dangling_edge_count"), "duplicate_edge_count": stats.get("duplicate_edge_count"),
            "out_degree_distribution": json.dumps(stats.get("out_degree_distribution", {})),
            "in_degree_distribution": json.dumps(stats.get("in_degree_distribution", {})),
            "t_gap_distribution": json.dumps(stats.get("t_gap_distribution", {})),
            "multi_parent_child_count": stats.get("multi_parent_child_count"), "division_like_source_count": stats.get("division_like_source_count"),
        }
        for name, stats in variant_stats_map.items()
    ]
    pd.DataFrame(topology_rows).to_csv(out_dir / "milestone15_topology_validation.csv", index=False)

    edge_distance_rows = [{"variant": name, **(stats.get("edge_distance_summary", {}) or {})} for name, stats in variant_stats_map.items()]
    pd.DataFrame(edge_distance_rows).to_csv(out_dir / "milestone15_edge_distance_stats.csv", index=False)

    per_dataset_rows = (
        [dict(d, variant="motion_relink") for d in motion_relink_diag]
        + [dict(d, variant="union_stable") for d in union_stable_diag]
        + [dict(d, variant="m12_baseline_safe") for d in baseline_diag]
    )

    submission_report = {
        "m14_oof_reference": M14_OOF_REFERENCE,
        "n_train_samples": len(train_dirs), "train_datasets": [d.stem for d in train_dirs],
        "n_test_datasets": len(test_dirs), "test_datasets": expected_datasets,
        "variants": {
            name: {"valid": variant_valid_map[name], "shape": list(variant_df_map[name].shape)}
            for name in ("motion_relink", "union_stable", "m12_baseline_safe")
        },
        "final_submission_source": final_source, "final_shape": list(final_df.shape),
        "per_dataset_diagnostics": per_dataset_rows,
        "rationale": (
            "motion_relink is the new primary because Milestone 14's OOF validation found it achieves the "
            "highest aggregate edge_jaccard (0.019449 vs the M12 baseline's 0.013215) of all 9 graph-repair "
            "experiments - a real, GroupKFold-validated improvement, even though its edge_FP is also "
            "substantially higher (8432 vs 1932), a genuine precision/recall tradeoff, not a free win. "
            "union_stable exists as an explicit fallback because Milestone 14 found it trades some of "
            "motion_relink's raw jaccard for better precision and stability (n_samples_with_tp=7 vs "
            "motion_relink's 5) - a reasonable second choice if motion_relink's own output fails validation "
            "on the real test data. m12_baseline_safe is the final safety net: M12's original, completely "
            "unmodified recipe, with no new logic at all."
        ),
        "final_instruction": "submission.csv built and validated. Do not submit until user reviews.",
    }
    with open(out_dir / "milestone15_submission_report.json", "w") as f:
        json.dump(submission_report, f, indent=2, default=str)

    print("\nsubmission.csv built and validated. Do not submit until user reviews.")

    return {
        "train_dirs": train_dirs, "primary_model": primary_model, "conservative_model": conservative_model,
        "motion_relink_df": motion_relink_df, "union_stable_df": union_stable_df, "baseline_df": baseline_df,
        "final_df": final_df, "final_source": final_source,
        "motion_relink_valid": motion_relink_valid, "union_stable_valid": union_stable_valid, "baseline_valid": baseline_valid,
        "submission_report": submission_report,
    }

# --------------------------------------------------------------------------- #
# 31. Tests
# --------------------------------------------------------------------------- #
def run_milestone15_tests() -> None:
    """Correctness tests for Milestone 15's new machinery: motion-relink
    reject-column assignment, gate-failing-candidate rejection, no-forced-
    assignment, no-multi-parent, duplicate-directed-edge detection,
    submission schema/topology validation, fallback-selection logic, and a
    synthetic end-to-end fixture producing all 3 variants' final rows -
    all on small in-memory fixtures (no disk I/O, no real zarr datasets).
    """
    # Test 1: motion-relink assignment with reject columns - 2 sources
    # competing for 2 targets with a clear best pairing.
    scored1 = pd.DataFrame({
        "source_id": [0, 0, 1, 1], "target_id": [10, 11, 10, 11], "t": [0, 0, 0, 0],
        "dz_um": [0.0] * 4, "dy_um": [1.0, 5.0, 5.0, 1.0], "dx_um": [0.0] * 4,
        "distance_um": [1.0, 5.0, 5.0, 1.0], "_score": [0.9, 0.5, 0.5, 0.9],
    })
    selected1, _ = select_edges_motion_relink(scored1, tight_gate=6.0, relaxed_gate=10.0)
    accepted_pairs1 = set(zip(selected1["source_id"], selected1["target_id"]))
    assert (0, 10) in accepted_pairs1 and (1, 11) in accepted_pairs1

    # Test 2: gate-failing candidates can NEVER be selected, even with a
    # very high classifier score - a candidate's motion_distance_um must
    # be <= the relaxed gate to enter the cost matrix at all.
    scored2 = pd.DataFrame({
        "source_id": [0], "target_id": [99], "t": [0],
        "dz_um": [0.0], "dy_um": [50.0], "dx_um": [0.0],
        "distance_um": [50.0], "_score": [0.999],
    })
    selected2, _ = select_edges_motion_relink(scored2, tight_gate=6.0, relaxed_gate=10.0)
    assert len(selected2) == 0, "a candidate whose motion_distance (50um) exceeds even the relaxed gate must never be selected"

    # Test 3: no forced assignment - a source with zero gate-passing
    # candidates must be left unmatched (rejected), not forced onto its
    # least-bad option.
    scored3 = pd.DataFrame({
        "source_id": [0, 1], "target_id": [10, 11], "t": [0, 0],
        "dz_um": [0.0, 0.0], "dy_um": [1.0, 99.0], "dx_um": [0.0, 0.0],
        "distance_um": [1.0, 99.0], "_score": [0.9, 0.9],
    })
    selected3, _ = select_edges_motion_relink(scored3, tight_gate=6.0, relaxed_gate=10.0)
    assert set(selected3["source_id"]) == {0}, "source 1's only candidate (99um) must be rejected, not force-matched"

    # Test 4: no multi-parent across the combined tight+relaxed output -
    # every target used at most once.
    scored4 = pd.DataFrame({
        "source_id": [0, 1, 2], "target_id": [10, 10, 11], "t": [0, 0, 0],
        "dz_um": [0.0] * 3, "dy_um": [1.0, 1.2, 1.0], "dx_um": [0.0] * 3,
        "distance_um": [1.0, 1.2, 1.0], "_score": [0.9, 0.85, 0.8],
    })
    selected4, _ = select_edges_motion_relink(scored4, tight_gate=6.0, relaxed_gate=10.0)
    assert selected4["target_id"].is_unique, "no target should be assigned to more than one source"
    assert selected4["source_id"].is_unique

    # Test 5: duplicate directed edge detection - validate_submission_full
    # must FAIL a submission containing a duplicate (dataset,source,target)
    # edge, and the reported duplicate_edge_count must be exactly 1.
    good_rows5 = [
        {"id": 0, "dataset": "d1", "row_type": "node", "node_id": 0, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 1, "dataset": "d1", "row_type": "node", "node_id": 1, "t": 1, "z": 0.0, "y": 1.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 2, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 1},
        {"id": 3, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 1},
    ]
    dup_df5 = pd.DataFrame(good_rows5, columns=SUBMISSION_COLUMNS)
    dup_valid5, dup_report5, dup_stats5 = validate_submission_full(dup_df5, expected_datasets=["d1"])
    assert not dup_valid5
    assert dup_stats5["duplicate_edge_count"] == 1

    # Test 6: submission schema + topology validation on a well-formed
    # single-edge submission must pass BOTH the base and topology checks.
    good_rows6 = [
        {"id": 0, "dataset": "d1", "row_type": "node", "node_id": 0, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 1, "dataset": "d1", "row_type": "node", "node_id": 1, "t": 1, "z": 0.0, "y": 1.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 2, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 1},
    ]
    good_df6 = pd.DataFrame(good_rows6, columns=SUBMISSION_COLUMNS)
    good_valid6, good_report6, good_stats6 = validate_submission_full(good_df6, expected_datasets=["d1"])
    assert good_valid6, good_report6
    assert set(good_stats6["t_gap_distribution"].keys()) <= {"1"}

    # An entirely node-only (zero-edge) submission must ALSO pass the
    # t_gap check - an empty distribution is not a "gap != 1" violation.
    node_only_rows6 = good_rows6[:2]
    node_only_df6 = pd.DataFrame(node_only_rows6, columns=SUBMISSION_COLUMNS)
    node_only_valid6, node_only_report6, node_only_stats6 = validate_submission_full(node_only_df6, expected_datasets=["d1"])
    assert node_only_valid6, node_only_report6
    assert node_only_stats6["t_gap_distribution"] == {}

    # Test 7: fallback-selection logic - the pure priority function.
    assert select_final_variant(True, True, True) == "motion_relink"
    assert select_final_variant(False, True, True) == "union_stable"
    assert select_final_variant(False, False, True) == "m12_baseline_safe"
    assert select_final_variant(False, False, False) is None

    # Test 8: synthetic end-to-end fixture - builds all 3 variants'
    # submission ROWS for one hand-built sample (no disk I/O) and confirms
    # each produces a schema+topology-valid result.
    filtered_nodes8 = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4], "t": [0, 1, 2, 3, 4],
        "z": [0.0] * 5, "y": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [0.0] * 5, "score": [0.9] * 5,
    })
    real_chain_rows8 = [
        {"source_id": s, "target_id": s + 1, "t": s, "dz_um": 0.0, "dy_um": 1.0, "dx_um": 0.0, "distance_um": 0.4, "_score": score}
        for s, score in zip([0, 1, 2, 3], [0.95, 0.93, 0.91, 0.90])
    ]
    noise_rows8 = [
        {"source_id": 1000 + i, "target_id": 1 + (i % 4), "t": (i % 4), "dz_um": 0.0, "dy_um": -99.0, "dx_um": 0.0, "distance_um": 99.0, "_score": 0.05}
        for i in range(60)
    ]
    scored8 = pd.DataFrame(real_chain_rows8 + noise_rows8)

    def _build_via(edge_selection_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected, _ = edge_selection_fn(scored8, 0.9)
        tracklets_df = build_tracklets_from_selected_edges(selected, filtered_nodes8)
        min_thr = float(tracklets_df["mean_edge_score"].quantile(0.7)) if len(tracklets_df) else 0.0
        kept = apply_tracklet_filter(
            tracklets_df, min_tracklet_length=5, min_mean_node_score=0.0, min_mean_edge_score_threshold=min_thr,
            max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM, max_smoothness_error_um=3.0, keep_top_k=100,
        )
        return reconstruct_policy_a(kept, filtered_nodes8)

    motion_relink_nodes8, motion_relink_edges8 = _build_via(motion_relink_edge_selection)
    baseline_nodes8, baseline_edges8 = _build_via(select_greedy_exclusive_per_frame)
    union_nodes8, union_edges8 = combine_ensemble(motion_relink_nodes8, motion_relink_edges8, baseline_nodes8, baseline_edges8, mode="union")

    for label, (nodes8, edges8) in {
        "motion_relink": (motion_relink_nodes8, motion_relink_edges8), "union": (union_nodes8, union_edges8), "baseline": (baseline_nodes8, baseline_edges8),
    }.items():
        assert len(nodes8) > 0, f"{label} produced no nodes"
        rows8, _ = build_submission_rows_for_sample("synthetic_test", nodes8, edges8, next_id=0)
        submission8 = pd.DataFrame(rows8, columns=SUBMISSION_COLUMNS)
        valid8, report8, _ = validate_submission_full(submission8, expected_datasets=["synthetic_test"])
        assert valid8, f"{label} submission failed validation:\n{report8}"

    print("All milestone15_motion_relink_submission tests passed (8/8).")

# --------------------------------------------------------------------------- #
# 32. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone15_pipeline(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
) -> dict:
    """Runs the unit tests, then the full Milestone 15 submission builder:
    trains the primary + conservative classifiers once on the same 12
    robust samples, builds and validates all 3 submission variants
    (motion_relink, union_stable, m12_baseline_safe), and writes
    submission.csv from whichever validates (motion_relink preferred).
    Never calls any Kaggle submission API and never calls Save Version.
    """
    print(
        "=== Self-test: motion-relink reject-column assignment, gate-failing-candidate rejection, "
        "no-forced-assignment, no-multi-parent, duplicate-edge detection, submission validation, "
        "fallback-selection, and synthetic end-to-end unit tests ==="
    )
    run_milestone15_tests()
    return run_milestone15_submission_pipeline(n_robust_samples=n_robust_samples, seed=seed, time_budget_seconds=time_budget_seconds)


if __name__ == "__main__":
    run_milestone15_pipeline()
