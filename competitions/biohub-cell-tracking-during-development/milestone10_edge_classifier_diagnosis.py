"""
Biohub - Cell Tracking During Development
Milestone 10: edge separability + supervised edge classifier diagnostic.

Milestones 7-9 established: node filtering is the dominant bottleneck for
GT-edge recoverability (7), loosening it alone explodes edge_FP because a
forced/greedy assignment links every admitted node regardless of quality
(8), and hand-crafted hard-coded edge scoring (distance x node-score x
rank x smoothness) is insufficient to separate true from false candidates
well enough to beat even 6B's fair pre-tracklet baseline (9). This module
is a strategy pivot: stop hand-tuning thresholds/scores and instead (A)
diagnose whether the GT graph is even well-modeled by one-to-one
frame-to-frame linking (does the data contain cell divisions? do edges
ever skip a frame?), (B/C) build an unbiased candidate-edge pool with rich
per-candidate features, (D) measure the label-oracle ceiling - the best
ANY ranking-based method could ever achieve on this candidate pool, (E)
check how separable the true/false candidates already are on simple
single-feature scores, (F) train real sklearn classifiers with
leave-one-sample-out cross-validation and hard-negative sampling, and (G)
issue a decision report.

Design notes (see architecture review before implementation):

  - Candidate labeling: a raw-node candidate (source_id, target_id) is
    "positive" iff both endpoints Hungarian-match (<=7um) to GT nodes AND
    that exact (gt_source, gt_target) pair is a real GT edge. GT-node
    matching is one-to-one per timepoint, so every GT edge can be
    "recovered" by at most one candidate row - no double-counting risk.

  - Candidate-pool caching: rather than rebuilding the kNN candidate pool
    separately for every (max_link_distance_um, k_next_candidates) grid
    point (4 node-filter configs x 5 distances x 4 k-values x 3 samples =
    240 rebuilds, each up to ~75,000 rows - intractable for feature
    engineering under the runtime budget), the candidate pool is built
    ONCE per (sample, node-filter config) at the LOOSEST ceiling settings
    (max_link_distance_um=7, k_next_candidates=5), with each candidate
    tagged by its true distance_um and its local_rank (rank among that
    source's candidates by ascending distance, WITH A DETERMINISTIC
    SECONDARY SORT KEY (target_id) so ties never depend on row order).
    Every tighter grid point is then reproduced by filtering this cached
    table: `distance_um <= max_link_distance_um AND local_rank <=
    k_next_candidates` - exactly equivalent to a from-scratch rebuild
    because local_rank is assigned by ascending-distance order within the
    ceiling gate, so both filters are monotonic prefixes of the same fixed
    ordering (verified in architecture review; the deterministic secondary
    sort is what makes the equivalence exact, not just approximate).

  - Oracle semantics: "oracle at FP budget B" is deliberately reported as
    IDENTICAL across every B in [200,500,1000,2000,5000], because a
    label-oracle never needs to spend any FP budget at all - GT-node
    matching is one-to-one, so distinct positive candidates never
    structurally conflict (even at division nodes, which legitimately have
    out_degree>=2), meaning the oracle's optimal choice at every budget is
    simply "take every positive candidate, zero negatives." This is not a
    bug or a missed opportunity to make the columns differ - it IS the
    finding: the oracle ceiling is just the candidate pool's raw recall
    (`positives_kept / total_gt_edges`), independent of any FP tolerance.
    The practical consequence: if `oracle_edge_jaccard` (== recall
    ceiling, since oracle FP=0) already exceeds 6C/6D's 0.010667, then any
    nonzero FP a REAL classifier produces is purely a classifier/feature
    limitation, not evidence the candidate pool itself is insufficient. A
    budget-sensitive oracle would only exist under a genuinely constrained
    assignment (e.g. division-aware min-cost flow with capacity conflicts,
    which this module does not attempt) or for a non-label ranker (which
    is exactly what Part F's real classifiers are).

  - Train/eval split: classifiers train on a small hard-negative-sampled
    set (all positives + negatives with local_rank<=3 + a random top-up to
    a capped negative:positive ratio) but are ALWAYS evaluated by scoring
    the complete, unsampled candidate pool of the held-out validation
    sample, so reported TP/FP/FN/precision/recall reflect the true
    candidate universe, not the biased training sample. Only RANKING
    (global top-K or per-frame top-K) is ever used to pick an operating
    point here - no absolute probability threshold is applied anywhere, so
    the well-known "class_weight='balanced' + hard-negative-subsampling
    double-corrects the base rate" trap (flagged in review) cannot bite:
    ranking is invariant to any monotonic/near-monotonic shift in the
    predicted scores.

  - n=3 leave-one-sample-out CV is directional/high-variance only - the
    go/no-go decision in Part G is anchored primarily on the deterministic
    oracle ceiling (Part D), with classifier CV results (Part F) treated
    as a secondary, noisier signal, exactly as recommended in review.

  - This diagnoses candidate-edge SEPARABILITY, not full tracking - it
    does not build tracklets or a submission, and independent per-edge
    top-K selection imposes no temporal/track continuity by construction.
    Part G's decision report explicitly weighs this proxy gap: even a
    strong classifier result here is evidence for "edges are separable,"
    not proof that a full tracking submission using it would score well
    on the competition's real (track-consistency-sensitive) metric - this
    is the central argument for taking option E (division-aware min-cost
    flow) seriously rather than defaulting to "ship a classifier" the
    moment Part F clears a threshold.

No final submission is built. milestone9's / 6C/6D's files are not
modified - all reused logic is copied in, not imported. No ML training
happens outside this file's own explicit sklearn classifier calls (Part
F only) - Parts A-E are pure diagnostics.
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

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
# 13. Part A - GT graph structural diagnosis (GT-only, no detection needed -
#     cheap enough to run over EVERY discovered train sample, not just 3)
# --------------------------------------------------------------------------- #
GT_STRUCTURE_COLUMNS = [
    "dataset", "gt_node_count", "gt_edge_count",
    "frac_out_degree_0", "frac_out_degree_1", "frac_out_degree_2", "frac_out_degree_gt2",
    "frac_in_degree_0", "frac_in_degree_1", "frac_in_degree_gt1",
    "division_candidate_count", "division_candidate_frac",
    "frac_edge_time_gap_eq1", "frac_edge_time_gap_skip", "frac_edge_time_gap_nonpositive",
    "mean_time_gap", "max_time_gap",
]


def _analyze_gt_graph_structure_core(dataset_name: str, gt_nodes: pd.DataFrame, gt_edges: pd.DataFrame) -> dict:
    """Pure GT-graph analysis (out/in-degree distribution, division-like
    nodes, edge time-gap distribution) - answers whether the data is even
    well-modeled by strict one-to-one frame-to-frame linking (division
    nodes have out_degree>=2; a real skip-frame edge would break every
    candidate generator built so far, which only ever considers t->t+1).
    Takes GT tables directly (no disk I/O) so the logic is unit-testable
    on small synthetic fixtures - see `analyze_gt_graph_structure` for the
    disk-I/O-driving wrapper used on real samples.
    """
    node_ids = gt_nodes["node_id"].to_numpy()
    n_nodes = len(node_ids)
    n_edges = len(gt_edges)

    out_degree = pd.Series(0, index=node_ids, dtype=int)
    in_degree = pd.Series(0, index=node_ids, dtype=int)
    if n_edges:
        out_counts = gt_edges["source_id"].value_counts()
        in_counts = gt_edges["target_id"].value_counts()
        out_degree.update(out_counts)
        in_degree.update(in_counts)

    frac_out_0 = float((out_degree == 0).mean()) if n_nodes else float("nan")
    frac_out_1 = float((out_degree == 1).mean()) if n_nodes else float("nan")
    frac_out_2 = float((out_degree == 2).mean()) if n_nodes else float("nan")
    frac_out_gt2 = float((out_degree > 2).mean()) if n_nodes else float("nan")
    frac_in_0 = float((in_degree == 0).mean()) if n_nodes else float("nan")
    frac_in_1 = float((in_degree == 1).mean()) if n_nodes else float("nan")
    frac_in_gt1 = float((in_degree > 1).mean()) if n_nodes else float("nan")

    division_candidate_count = int((out_degree >= 2).sum())
    division_candidate_frac = float(division_candidate_count / n_nodes) if n_nodes else float("nan")

    if n_edges:
        node_t = gt_nodes.set_index("node_id")["t"]
        time_gaps = node_t.reindex(gt_edges["target_id"]).to_numpy() - node_t.reindex(gt_edges["source_id"]).to_numpy()
        frac_gap_eq1 = float(np.mean(time_gaps == 1))
        frac_gap_skip = float(np.mean(time_gaps > 1))
        frac_gap_nonpositive = float(np.mean(time_gaps <= 0))
        mean_gap = float(np.mean(time_gaps))
        max_gap = int(np.max(time_gaps))
    else:
        frac_gap_eq1 = frac_gap_skip = frac_gap_nonpositive = mean_gap = float("nan")
        max_gap = 0

    return {
        "dataset": dataset_name, "gt_node_count": n_nodes, "gt_edge_count": n_edges,
        "frac_out_degree_0": frac_out_0, "frac_out_degree_1": frac_out_1,
        "frac_out_degree_2": frac_out_2, "frac_out_degree_gt2": frac_out_gt2,
        "frac_in_degree_0": frac_in_0, "frac_in_degree_1": frac_in_1, "frac_in_degree_gt1": frac_in_gt1,
        "division_candidate_count": division_candidate_count, "division_candidate_frac": division_candidate_frac,
        "frac_edge_time_gap_eq1": frac_gap_eq1, "frac_edge_time_gap_skip": frac_gap_skip,
        "frac_edge_time_gap_nonpositive": frac_gap_nonpositive,
        "mean_time_gap": mean_gap, "max_time_gap": max_gap,
    }


def analyze_gt_graph_structure(sample_zarr_dir: Path) -> dict:
    """Disk-I/O-driving wrapper: reads GT then calls the pure core above."""
    dataset_name = sample_zarr_dir.stem
    gt_nodes, gt_edges = read_geff(sample_zarr_dir)
    return _analyze_gt_graph_structure_core(dataset_name, gt_nodes, gt_edges)


def run_part_a_gt_structure(
    sample_dirs: Sequence[Path], csv_path: str = "/kaggle/working/milestone10_gt_graph_structure.csv",
) -> pd.DataFrame:
    """Runs `analyze_gt_graph_structure` over every given sample (cheap -
    GT-only, no detection) and saves the result.
    """
    rows = [analyze_gt_graph_structure(s) for s in sample_dirs]
    df = pd.DataFrame(rows, columns=GT_STRUCTURE_COLUMNS)

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved GT graph structure diagnosis ({len(df)} row(s)) to {out_path}")

    if len(df):
        agg = df[df["gt_node_count"] > 0]
        if len(agg):
            print(
                f"  aggregate (unweighted mean across {len(agg)} sample(s)): "
                f"frac_out_degree_0={agg['frac_out_degree_0'].mean():.4f} "
                f"frac_out_degree_1={agg['frac_out_degree_1'].mean():.4f} "
                f"frac_out_degree_gt2={agg['frac_out_degree_gt2'].mean():.4f} "
                f"division_candidate_frac={agg['division_candidate_frac'].mean():.4f} "
                f"frac_edge_time_gap_skip={agg['frac_edge_time_gap_skip'].mean():.4f}"
            )
    return df
# --------------------------------------------------------------------------- #
# 14. Selected node-filter configs + per-sample/per-node-filter caching
# --------------------------------------------------------------------------- #
SELECTED_NODE_FILTER_CONFIGS = [
    {"name": "baseline_6BCD", "max_nodes_per_timepoint": 75, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.05},
    {"name": "m8_best_tradeoff", "max_nodes_per_timepoint": 75, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.03},
    {"name": "moderate_100", "max_nodes_per_timepoint": 100, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.03},
    {"name": "moderate_150_085", "max_nodes_per_timepoint": 150, "score_quantile_per_timepoint": 0.85, "absolute_score_threshold": 0.05},
]

NODE_MATCH_MAX_DISTANCE_UM = 7.0


def precompute_raw_sample_cache(sample_zarr_dir: Path, detector_config: dict = DEFAULT_DETECTOR_CONFIG) -> dict:
    """Runs the raw detector and reads GT exactly ONCE per sample - shared
    across all 4 node-filter configs and everything downstream.
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


def precompute_node_filter_cache(
    raw_cache: dict, node_filter_config: dict, match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """For ONE (sample, node-filter-config) pair: applies node filtering,
    matches filtered nodes to GT ONCE, builds the reference-displacement
    chain ONCE, and builds the ceiling candidate-edge pool ONCE - every
    downstream (distance, k) grid point and every feature/label column
    reuses this same cached pool (see section 12).
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
    """Builds `{node_filter_name: [per_sample_cache, ...]}` for every one of
    the 4 selected node-filter configs, once, up front.
    """
    caches_by_name: dict[str, list[dict]] = {}
    for node_filter_config in node_filter_configs:
        name = node_filter_config["name"]
        caches_by_name[name] = [
            precompute_node_filter_cache(rc, node_filter_config, match_max_distance_um) for rc in raw_caches
        ]
    return caches_by_name
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
# 16. Part D - oracle analysis (theoretical ceiling per node-filter/candidate config)
# --------------------------------------------------------------------------- #
DEFAULT_MAX_LINK_DISTANCE_UM_VALUES = (2, 3, 4, 5, 7)
DEFAULT_K_NEXT_CANDIDATES_VALUES = (1, 2, 3, 5)
ORACLE_FP_BUDGETS = (200, 500, 1000, 2000, 5000)

ORACLE_COLUMNS = [
    "node_filter_name", "max_link_distance_um", "k_next_candidates",
    "candidate_count", "positive_count", "positive_rate", "candidate_edge_recall",
    "pool_recall_ceiling",
] + [f"oracle_edge_jaccard_at_fp{b}" for b in ORACLE_FP_BUDGETS] + [
    "beats_6B_edge_jaccard", "beats_6C6D_edge_jaccard",
]


def run_part_d_oracle(
    feature_df: pd.DataFrame,
    node_filter_caches_by_name: dict,
    max_link_distance_um_values: Sequence[float] = DEFAULT_MAX_LINK_DISTANCE_UM_VALUES,
    k_next_candidates_values: Sequence[int] = DEFAULT_K_NEXT_CANDIDATES_VALUES,
    baseline_6B_edge_jaccard: float = 0.002215,
    baseline_6C6D_edge_jaccard: float = 0.010667,
    csv_path: str = "/kaggle/working/milestone10_edge_candidate_diagnostics.csv",
) -> pd.DataFrame:
    """For every (node-filter, max_link_distance_um, k_next_candidates)
    grid point, filters the cached master feature table (no rebuild - see
    section 12) and reports the LABEL-ORACLE ceiling: since a label-oracle
    never needs to spend any FP budget (GT-node matching is one-to-one, so
    distinct positive candidates never structurally conflict), `oracle_TP
    = positive_count` and `oracle_FP = 0` at EVERY budget - this identical-
    across-budgets outcome is the diagnostic finding, not an oversight (see
    module docstring). `pool_recall_ceiling` is the same number reported
    once, unambiguously: the best edge_jaccard ANY ranking-based method
    could ever achieve on this exact candidate pool.
    """
    rows: list[dict] = []
    total_gt_edges_by_name = {
        name: sum(c["gt_edge_count"] for c in caches) for name, caches in node_filter_caches_by_name.items()
    }

    for name in node_filter_caches_by_name:
        subset_by_name = feature_df[feature_df["node_filter_name"] == name]
        total_gt_edges = total_gt_edges_by_name[name]

        for max_dist in max_link_distance_um_values:
            for k in k_next_candidates_values:
                filtered = subset_by_name[
                    (subset_by_name["distance_um"] <= max_dist) & (subset_by_name["local_rank"] <= k)
                ]
                candidate_count = len(filtered)
                positive_count = int(filtered["label"].sum())
                positive_rate = (positive_count / candidate_count) if candidate_count else float("nan")
                fn = total_gt_edges - positive_count
                denom = positive_count + fn  # oracle FP is always 0
                pool_recall_ceiling = (positive_count / denom) if denom > 0 else 1.0

                row = {
                    "node_filter_name": name, "max_link_distance_um": max_dist, "k_next_candidates": k,
                    "candidate_count": candidate_count, "positive_count": positive_count,
                    "positive_rate": positive_rate,
                    "candidate_edge_recall": (positive_count / total_gt_edges) if total_gt_edges else float("nan"),
                    "pool_recall_ceiling": pool_recall_ceiling,
                }
                for budget in ORACLE_FP_BUDGETS:
                    row[f"oracle_edge_jaccard_at_fp{budget}"] = pool_recall_ceiling
                row["beats_6B_edge_jaccard"] = pool_recall_ceiling > baseline_6B_edge_jaccard
                row["beats_6C6D_edge_jaccard"] = pool_recall_ceiling > baseline_6C6D_edge_jaccard
                rows.append(row)

    oracle_df = pd.DataFrame(rows, columns=ORACLE_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_df.to_csv(out_path, index=False)
    print(f"Saved oracle/candidate diagnostics ({len(oracle_df)} row(s)) to {out_path}")

    if len(oracle_df):
        best_row = oracle_df.sort_values("pool_recall_ceiling", ascending=False).iloc[0]
        print(
            f"  Best pool_recall_ceiling: {best_row['pool_recall_ceiling']:.6f} "
            f"(node_filter={best_row['node_filter_name']}, max_dist={best_row['max_link_distance_um']}, "
            f"k={best_row['k_next_candidates']}) - beats_6B={bool(best_row['beats_6B_edge_jaccard'])}, "
            f"beats_6C6D={bool(best_row['beats_6C6D_edge_jaccard'])}"
        )
        print(
            f"  (reminder: oracle_edge_jaccard is identical across all FP budgets {ORACLE_FP_BUDGETS} by "
            f"construction - a label-oracle never needs to spend any FP budget; see module docstring)"
        )
    return oracle_df
# --------------------------------------------------------------------------- #
# 17. Part E - feature separability (AUC/AP for simple scores + true-vs-false
#     quantiles for every feature)
# --------------------------------------------------------------------------- #
SEPARABILITY_COLUMNS = [
    "node_filter_name", "feature_name", "auc", "ap", "n_positive", "n_negative",
    "pos_p10", "pos_p25", "pos_p50", "pos_p75", "pos_p90",
    "neg_p10", "neg_p25", "neg_p50", "neg_p75", "neg_p90",
]

# Orientation multiplier so "higher score = more likely positive" for AUC/AP
# (distance is negated since SMALLER distance should indicate a TRUE edge).
# AUC/AP are only reported for these 5 requested "simple scores" - every
# other feature still gets its true-vs-false quantile row.
SIMPLE_SCORE_ORIENTATION = {
    "distance_um": -1.0,
    "node_score_product": 1.0,
    "edge_score": 1.0,
    "rank_score": 1.0,
    "smoothness_score": 1.0,
}


def _quantiles(values: pd.Series) -> list[float]:
    values = values.dropna()
    if len(values) == 0:
        return [float("nan")] * 5
    return [float(values.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]


def compute_feature_separability(feature_df: pd.DataFrame) -> pd.DataFrame:
    """For every node-filter config (plus an "ALL" aggregate), computes
    ROC-AUC/AP for the 5 requested simple scores, and true-vs-false
    quantiles for EVERY classifier feature.
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=SEPARABILITY_COLUMNS)

    rows: list[dict] = []
    groups = list(feature_df.groupby("node_filter_name")) + [("ALL", feature_df)]

    for name, group in groups:
        y = group["label"].to_numpy(dtype=bool)
        n_pos = int(y.sum())
        n_neg = int((~y).sum())

        for feature_name in CLASSIFIER_FEATURE_COLUMNS:
            values = group[feature_name]
            auc = ap = float("nan")
            orientation = SIMPLE_SCORE_ORIENTATION.get(feature_name)
            if orientation is not None and n_pos > 0 and n_neg > 0:
                scored = orientation * values
                valid_mask = scored.notna().to_numpy()
                y_valid = y[valid_mask]
                s_valid = scored.to_numpy()[valid_mask]
                if len(s_valid) > 0 and len(set(y_valid)) == 2:
                    auc = float(roc_auc_score(y_valid, s_valid))
                    ap = float(average_precision_score(y_valid, s_valid))

            pos_q = _quantiles(values[y])
            neg_q = _quantiles(values[~y])
            row = {
                "node_filter_name": name, "feature_name": feature_name, "auc": auc, "ap": ap,
                "n_positive": n_pos, "n_negative": n_neg,
            }
            row.update(zip(["pos_p10", "pos_p25", "pos_p50", "pos_p75", "pos_p90"], pos_q))
            row.update(zip(["neg_p10", "neg_p25", "neg_p50", "neg_p75", "neg_p90"], neg_q))
            rows.append(row)

    return pd.DataFrame(rows, columns=SEPARABILITY_COLUMNS)


def run_part_e_separability(
    feature_df: pd.DataFrame, csv_path: str = "/kaggle/working/milestone10_feature_separability.csv",
) -> pd.DataFrame:
    sep_df = compute_feature_separability(feature_df)

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep_df.to_csv(out_path, index=False)
    print(f"Saved feature separability ({len(sep_df)} row(s)) to {out_path}")

    all_rows = sep_df[(sep_df["node_filter_name"] == "ALL") & (sep_df["feature_name"].isin(SIMPLE_SCORE_ORIENTATION))]
    if len(all_rows):
        print("  Simple-score AUC/AP (ALL node-filter configs combined):")
        for _, r in all_rows.sort_values("auc", ascending=False).iterrows():
            print(f"    {r['feature_name']}: AUC={r['auc']:.4f} AP={r['ap']:.4f}")
    return sep_df
# --------------------------------------------------------------------------- #
# 18. Part F - supervised edge classifiers: hard-negative sampling +
#     leave-one-sample-out CV + global/per-frame budget evaluation
# --------------------------------------------------------------------------- #
NEGATIVE_RATIO_VALUES = (10, 25, 50)
GLOBAL_TOP_K_BUDGETS = (200, 500, 1000, 2000, 5000)
PER_FRAME_TOP_K_VALUES = (1, 2, 3, 5, 10)
HARD_NEGATIVE_LOCAL_RANK_CEILING = 3

CLASSIFIER_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0),
}
# HistGradientBoostingClassifier accepts NaN features natively; the other
# two need median imputation (computed from the TRAIN fold only, applied to
# both train and validation, to avoid leaking validation-set statistics).
CLASSIFIERS_NEEDING_IMPUTATION = {"logistic_regression", "random_forest"}

CLASSIFIER_CV_COLUMNS = [
    "node_filter_name", "held_out_dataset", "classifier", "negative_ratio",
    "n_train_positive", "n_train_negative", "auc", "ap",
    "budget_type", "budget", "edge_TP", "edge_FP", "edge_FN", "edge_jaccard", "precision", "recall",
]


def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    return precision, recall


def hard_negative_sample(
    train_df: pd.DataFrame, negative_ratio: int, random_state: int = 0,
) -> pd.DataFrame:
    """All positives + hard negatives (local_rank<=3, which also covers
    "nearest false candidates" since rank 1 is the nearest) + a random
    top-up from the remaining negatives, capped at
    `negative_ratio x n_positive` total negatives.
    """
    positives = train_df[train_df["label"]]
    negatives = train_df[~train_df["label"]]
    n_positive = max(len(positives), 1)
    cap = int(negative_ratio * n_positive)

    hard = negatives[negatives["local_rank"] <= HARD_NEGATIVE_LOCAL_RANK_CEILING]
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


def _evaluate_budget(scored_df: pd.DataFrame, kept_mask: np.ndarray, gt_edge_count: int) -> dict:
    kept = scored_df[kept_mask]
    tp = int(kept["label"].sum())
    fp = len(kept) - tp
    fn = gt_edge_count - tp
    denom = tp + fp + fn
    edge_jaccard = tp / denom if denom > 0 else 1.0
    precision, recall = _precision_recall(tp, fp, fn)
    return {"edge_TP": tp, "edge_FP": fp, "edge_FN": fn, "edge_jaccard": edge_jaccard, "precision": precision, "recall": recall}


def evaluate_global_top_k(scored_df: pd.DataFrame, budget: int, gt_edge_count: int) -> dict:
    ordered = scored_df.sort_values("_score", ascending=False)
    kept_mask = np.zeros(len(ordered), dtype=bool)
    kept_mask[: min(budget, len(ordered))] = True
    return _evaluate_budget(ordered, kept_mask, gt_edge_count)


def evaluate_per_frame_top_k(scored_df: pd.DataFrame, per_frame_k: int, gt_edge_count: int) -> dict:
    ordered = scored_df.sort_values(["t", "_score"], ascending=[True, False])
    kept = ordered.groupby("t", sort=False, group_keys=False).head(per_frame_k)
    kept_mask = np.asarray(ordered.index.isin(kept.index))
    return _evaluate_budget(ordered, kept_mask, gt_edge_count)


def run_part_f_classifiers(
    feature_df: pd.DataFrame,
    node_filter_caches_by_name: dict,
    node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
    negative_ratio_values: Sequence[int] = NEGATIVE_RATIO_VALUES,
    classifier_builders: dict = CLASSIFIER_BUILDERS,
    global_top_k_budgets: Sequence[int] = GLOBAL_TOP_K_BUDGETS,
    per_frame_top_k_values: Sequence[int] = PER_FRAME_TOP_K_VALUES,
    csv_path: str = "/kaggle/working/milestone10_classifier_cv_eval.csv",
) -> pd.DataFrame:
    """Leave-one-sample-out CV (3 folds) x 3 classifiers x 3 negative-ratio
    caps x 4 node-filter configs = 108 fits, each evaluated at 5 global-
    top-K + 5 per-frame-top-K budgets. Classifiers train on a small hard-
    negative-sampled set but are ALWAYS scored against the complete,
    unsampled candidate pool of the held-out sample.
    """
    gt_edge_count_by_name_dataset = {
        (name, c["dataset"]): c["gt_edge_count"] for name, caches in node_filter_caches_by_name.items() for c in caches
    }

    rows: list[dict] = []
    n_total = (
        len(node_filter_configs) * len(negative_ratio_values) * len(classifier_builders)
        * (len(global_top_k_budgets) + len(per_frame_top_k_values))
    )
    combo_idx = 0
    t0 = time.time()

    for node_filter_config in node_filter_configs:
        name = node_filter_config["name"]
        subset = feature_df[feature_df["node_filter_name"] == name]
        datasets = sorted(subset["dataset"].unique())
        if len(datasets) < 2:
            print(f"  [warn] {name}: fewer than 2 samples available, skipping LOSO CV.")
            continue

        for held_out in datasets:
            train_df = subset[subset["dataset"] != held_out]
            val_df = subset[subset["dataset"] == held_out].copy()
            gt_edge_count = gt_edge_count_by_name_dataset[(name, held_out)]

            if train_df["label"].nunique() < 2:
                print(f"  [warn] {name}/{held_out}: training fold has only one class, skipping.")
                continue

            for neg_ratio in negative_ratio_values:
                train_sample = hard_negative_sample(train_df, neg_ratio)
                if train_sample["label"].nunique() < 2:
                    continue

                X_train_raw = train_sample[CLASSIFIER_FEATURE_COLUMNS]
                y_train = train_sample["label"].to_numpy(dtype=int)
                X_val_raw = val_df[CLASSIFIER_FEATURE_COLUMNS]
                y_val = val_df["label"].to_numpy(dtype=int)

                medians = X_train_raw.median()
                X_train_imputed = X_train_raw.fillna(medians)
                X_val_imputed = X_val_raw.fillna(medians)

                n_train_pos = int(y_train.sum())
                n_train_neg = int(len(y_train) - n_train_pos)

                for clf_name, builder in classifier_builders.items():
                    combo_idx += 1
                    clf = builder()
                    if clf_name in CLASSIFIERS_NEEDING_IMPUTATION:
                        clf.fit(X_train_imputed, y_train)
                        scores = clf.predict_proba(X_val_imputed)[:, 1]
                    else:
                        clf.fit(X_train_raw, y_train)
                        scores = clf.predict_proba(X_val_raw)[:, 1]

                    auc = float(roc_auc_score(y_val, scores)) if len(set(y_val)) == 2 else float("nan")
                    ap = float(average_precision_score(y_val, scores)) if len(set(y_val)) == 2 else float("nan")

                    scored_val = val_df.copy()
                    scored_val["_score"] = scores

                    for budget in global_top_k_budgets:
                        result = evaluate_global_top_k(scored_val, budget, gt_edge_count)
                        rows.append({
                            "node_filter_name": name, "held_out_dataset": held_out, "classifier": clf_name,
                            "negative_ratio": neg_ratio, "n_train_positive": n_train_pos, "n_train_negative": n_train_neg,
                            "auc": auc, "ap": ap, "budget_type": "global_top_k", "budget": budget, **result,
                        })
                    for k in per_frame_top_k_values:
                        result = evaluate_per_frame_top_k(scored_val, k, gt_edge_count)
                        rows.append({
                            "node_filter_name": name, "held_out_dataset": held_out, "classifier": clf_name,
                            "negative_ratio": neg_ratio, "n_train_positive": n_train_pos, "n_train_negative": n_train_neg,
                            "auc": auc, "ap": ap, "budget_type": "per_frame_top_k", "budget": k, **result,
                        })

                    if combo_idx % 10 == 0 or combo_idx == n_total:
                        elapsed = time.time() - t0
                        print(f"  [{combo_idx}/{n_total} classifier fit(s)] elapsed={elapsed:.1f}s")

    cv_df = pd.DataFrame(rows, columns=CLASSIFIER_CV_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(out_path, index=False)
    print(f"\nSaved classifier CV eval ({len(cv_df)} row(s)) to {out_path}")
    return cv_df
# --------------------------------------------------------------------------- #
# 19. Part G - final best-candidate selection + decision report
# --------------------------------------------------------------------------- #
BEST_CANDIDATES_COLUMNS = CLASSIFIER_CV_COLUMNS + [
    "meets_strong_success", "meets_practical_success_fp1000", "meets_practical_success_fp2000", "meets_tp10_fp2000",
]


def select_best_candidates(
    cv_df: pd.DataFrame,
    baseline_6C6D_edge_jaccard: float = 0.010667,
    csv_path: str = "/kaggle/working/milestone10_best_candidates.csv",
) -> pd.DataFrame:
    """Flags every classifier-CV row against Milestone 10's success
    criteria (strong: beat 6C/6D; practical: TP>4 with FP<1000/2000;
    TP>=10 with FP<2000) and saves every row meeting at least one.
    """
    if len(cv_df) == 0:
        print("[warn] no classifier CV rows to select best candidates from.")
        return pd.DataFrame(columns=BEST_CANDIDATES_COLUMNS)

    df = cv_df.copy()
    df["meets_strong_success"] = df["edge_jaccard"] > baseline_6C6D_edge_jaccard
    df["meets_practical_success_fp1000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 1000)
    df["meets_practical_success_fp2000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 2000)
    df["meets_tp10_fp2000"] = (df["edge_TP"] >= 10) & (df["edge_FP"] < 2000)

    criterion_cols = [
        "meets_strong_success", "meets_practical_success_fp1000", "meets_practical_success_fp2000", "meets_tp10_fp2000",
    ]
    any_criterion = df[criterion_cols].any(axis=1)
    best = df[any_criterion].sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)

    print(f"\n=== Best-candidate selection (of {len(df)} classifier CV rows) ===")
    print(f"  strong success (edge_jaccard>{baseline_6C6D_edge_jaccard}): {int(df['meets_strong_success'].sum())}")
    print(
        f"  practical success TP>4 & FP<1000: {int(df['meets_practical_success_fp1000'].sum())}, "
        f"FP<2000: {int(df['meets_practical_success_fp2000'].sum())}"
    )
    print(f"  TP>=10 & FP<2000: {int(df['meets_tp10_fp2000'].sum())}")
    print(f"  -> {len(best)} of {len(df)} row(s) meet at least one criterion.")

    if len(best):
        top = best.iloc[0]
        print(
            f"  Top classifier candidate: node_filter={top['node_filter_name']} classifier={top['classifier']} "
            f"neg_ratio={top['negative_ratio']} budget_type={top['budget_type']} budget={top['budget']} "
            f"held_out={top['held_out_dataset']} -> edge_jaccard={top['edge_jaccard']:.6f} "
            f"TP={top['edge_TP']} FP={top['edge_FP']}"
        )

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)
    print(f"Saved best candidates ({len(best)} row(s)) to {out_path}")
    return best


def generate_decision_report(
    gt_structure_df: pd.DataFrame, oracle_df: pd.DataFrame, separability_df: pd.DataFrame, cv_df: pd.DataFrame,
    baseline_6C6D_edge_jaccard: float = 0.010667,
) -> dict:
    """Prints the full Part G decision report and returns the underlying
    findings as a dict. The go/no-go decision is anchored primarily on the
    deterministic oracle ceiling (Part D) - n=3 leave-one-sample-out
    classifier CV (Part F) is treated as a secondary, higher-variance
    signal, per architecture review.
    """
    print("\n" + "=" * 70)
    print("MILESTONE 10 DECISION REPORT")
    print("=" * 70)

    print("\n1) GT graph structure summary:")
    valid_structure = gt_structure_df[gt_structure_df["gt_node_count"] > 0]
    division_frac = 0.0
    skip_frac = 0.0
    if len(valid_structure):
        division_frac = float(valid_structure["division_candidate_frac"].mean())
        skip_frac = float(valid_structure["frac_edge_time_gap_skip"].mean())
        print(
            f"   samples analyzed: {len(valid_structure)}, total GT nodes: {int(valid_structure['gt_node_count'].sum())}, "
            f"total GT edges: {int(valid_structure['gt_edge_count'].sum())}"
        )
        print(
            f"   mean frac_out_degree_0={valid_structure['frac_out_degree_0'].mean():.4f} "
            f"frac_out_degree_1={valid_structure['frac_out_degree_1'].mean():.4f} "
            f"frac_out_degree_gt2={valid_structure['frac_out_degree_gt2'].mean():.4f}"
        )
        print(f"   mean division_candidate_frac={division_frac:.4f}")
        print(
            f"   mean frac_edge_time_gap_eq1={valid_structure['frac_edge_time_gap_eq1'].mean():.4f} "
            f"frac_edge_time_gap_skip={skip_frac:.4f}"
        )
    else:
        print("   no GT structure rows available.")

    print("\n2) Best oracle configs (top 5 by pool_recall_ceiling):")
    best_oracle_ceiling = float("nan")
    if len(oracle_df):
        for _, r in oracle_df.sort_values("pool_recall_ceiling", ascending=False).head(5).iterrows():
            print(
                f"   node_filter={r['node_filter_name']} max_dist={r['max_link_distance_um']} "
                f"k={r['k_next_candidates']} -> pool_recall_ceiling={r['pool_recall_ceiling']:.6f}"
            )
        best_oracle_ceiling = float(oracle_df["pool_recall_ceiling"].max())

    print("\n3) Best simple-score configs (ALL node-filters combined, by AUC):")
    if len(separability_df):
        all_scores = separability_df[(separability_df["node_filter_name"] == "ALL") & separability_df["auc"].notna()]
        for _, r in all_scores.sort_values("auc", ascending=False).iterrows():
            print(f"   {r['feature_name']}: AUC={r['auc']:.4f} AP={r['ap']:.4f}")

    print("\n4) Best classifier configs (top 5 by edge_jaccard):")
    best_classifier_jaccard = float("nan")
    if len(cv_df):
        for _, r in cv_df.sort_values("edge_jaccard", ascending=False).head(5).iterrows():
            print(
                f"   node_filter={r['node_filter_name']} classifier={r['classifier']} neg_ratio={r['negative_ratio']} "
                f"budget_type={r['budget_type']} budget={r['budget']} held_out={r['held_out_dataset']} -> "
                f"edge_jaccard={r['edge_jaccard']:.6f} TP={r['edge_TP']} FP={r['edge_FP']}"
            )
        best_classifier_jaccard = float(cv_df["edge_jaccard"].max())

    oracle_beats = bool(best_oracle_ceiling > baseline_6C6D_edge_jaccard) if not np.isnan(best_oracle_ceiling) else False
    classifier_beats = bool(best_classifier_jaccard > baseline_6C6D_edge_jaccard) if not np.isnan(best_classifier_jaccard) else False
    print(f"\n5) Oracle beats 6C/6D ({baseline_6C6D_edge_jaccard}): {oracle_beats} (best ceiling={best_oracle_ceiling:.6f})")
    print(f"6) Classifier beats 6C/6D: {classifier_beats} (best classifier edge_jaccard={best_classifier_jaccard:.6f})")

    tp4_fp1000 = tp4_fp2000 = tp10_fp2000 = False
    if len(cv_df):
        tp4_fp1000 = bool(((cv_df["edge_TP"] > 4) & (cv_df["edge_FP"] < 1000)).any())
        tp4_fp2000 = bool(((cv_df["edge_TP"] > 4) & (cv_df["edge_FP"] < 2000)).any())
        tp10_fp2000 = bool(((cv_df["edge_TP"] >= 10) & (cv_df["edge_FP"] < 2000)).any())
    print(
        f"\n7) Practical thresholds met: TP>4 & FP<1000: {tp4_fp1000}, TP>4 & FP<2000: {tp4_fp2000}, "
        f"TP>=10 & FP<2000: {tp10_fp2000}"
    )

    if classifier_beats or tp4_fp1000 or tp4_fp2000 or tp10_fp2000:
        recommendation = "A"
        rationale = (
            "A real classifier already clears at least one success criterion. HOWEVER: this only "
            "establishes that true/false edges are separable in isolation - independent per-edge "
            "top-K selection imposes no track-level continuity by construction, so validate with "
            "actual tracklet construction before committing to a full submission."
        )
    elif oracle_beats:
        recommendation = "B"
        rationale = (
            "The candidate pool's theoretical ceiling already exceeds the 6C/6D target, but no "
            "classifier reached it (n=3 LOSO CV is high-variance, but the gap is the signal here) - "
            "the bottleneck is features/model, not candidate generation. Improve features or try "
            "additional model families before touching candidate generation."
        )
    elif division_frac > 0.05 or skip_frac > 0.05:
        recommendation = "E"
        rationale = (
            f"Oracle ceiling does not beat 6C/6D, AND GT structure shows meaningful division "
            f"(division_candidate_frac={division_frac:.4f}) and/or skip-frame edges "
            f"(frac_edge_time_gap_skip={skip_frac:.4f}) - a structural violation of one-to-one "
            f"frame-to-frame linking that no amount of per-edge feature engineering can fix. A "
            f"division-aware min-cost-flow / graph-optimization formulation is warranted."
        )
    else:
        recommendation = "C"
        rationale = (
            "Oracle ceiling does not beat 6C/6D and GT structure does not show strong division/"
            "skip-frame violations - the candidate pool itself (distance/k gates) is likely too "
            "narrow. Widen candidate generation (larger max_link_distance_um/k_next_candidates, or "
            "allow t->t+2 candidates) before re-attempting classification."
        )

    print(f"\n8) Final recommendation: {recommendation}")
    print(f"   {rationale}")
    print("=" * 70)

    return {
        "oracle_beats_6C6D": oracle_beats, "classifier_beats_6C6D": classifier_beats,
        "best_oracle_ceiling": best_oracle_ceiling, "best_classifier_edge_jaccard": best_classifier_jaccard,
        "meets_tp4_fp1000": tp4_fp1000, "meets_tp4_fp2000": tp4_fp2000, "meets_tp10_fp2000": tp10_fp2000,
        "division_candidate_frac": division_frac, "frac_edge_time_gap_skip": skip_frac,
        "recommendation": recommendation, "rationale": rationale,
    }
# --------------------------------------------------------------------------- #
# 20. Tests
# --------------------------------------------------------------------------- #
def run_milestone10_tests() -> None:
    """Correctness tests for GT degree analysis, edge labeling, the
    candidate-pool ceiling+filter equivalence, hard-negative sampling, the
    leave-one-sample-out split, budget evaluation, and a synthetic fixture
    where a classifier can separate true edges - all on small in-memory
    fixtures (no disk I/O).
    """
    # Test 1: GT degree analysis on a hand-built fixture with a known
    # division node (out_degree=2), a skip-frame edge, and an isolated node.
    gt_nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5], "t": [0, 1, 1, 2, 0, 2],
        "z": [0.0] * 6, "y": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0], "x": [0.0] * 6,
    })
    # node 0 (t=0) divides into 1 and 2 (t=1) AND also has a direct
    # skip-frame edge to node 3 (t=2) - out_degree(0)=3 (a division node,
    # since out_degree>=2). node 4 (t=0) is isolated (out_degree=0).
    # node 5 (t=2) is isolated (in_degree=0).
    gt_edges = pd.DataFrame({"source_id": [0, 0, 0], "target_id": [1, 2, 3]})
    result = _analyze_gt_graph_structure_core("fixture", gt_nodes, gt_edges)
    assert result["gt_node_count"] == 6 and result["gt_edge_count"] == 3
    assert result["division_candidate_count"] == 1, "only node 0 has out_degree>=2 (it has out_degree 3)"
    assert abs(result["division_candidate_frac"] - 1 / 6) < 1e-9
    assert abs(result["frac_out_degree_0"] - 5 / 6) < 1e-9, "nodes 1,2,3,4,5 all have out_degree 0"
    assert abs(result["frac_edge_time_gap_eq1"] - 2 / 3) < 1e-9, "edges (0,1) and (0,2) have gap 1"
    assert abs(result["frac_edge_time_gap_skip"] - 1 / 3) < 1e-9, "edge (0,3) has gap 2 - a skip-frame edge"
    assert result["max_time_gap"] == 2

    # Test 2: edge labeling correctness. Candidate (100,200) maps via
    # pred_to_gt to a real GT edge (0,1) -> positive. Candidate (100,201)
    # maps to (0,2), NOT a real GT edge -> negative. Candidate (999,200)
    # has an unmatched source -> negative.
    candidates = pd.DataFrame({
        "source_id": [100, 100, 999], "target_id": [200, 201, 200],
    })
    pred_to_gt = {100: 0, 200: 1, 201: 2}
    gt_edge_set = {(0, 1)}
    labels = label_candidates(candidates, pred_to_gt, gt_edge_set)
    assert list(labels) == [True, False, False]

    # Test 3: ceiling-pool + filter reproduces a from-scratch tighter build
    # exactly (the key equivalence claim from architecture review). Build
    # at the ceiling, then filter to (max_dist=1.0, k=1); separately build
    # directly with those tighter values by hand-checking the expected
    # single nearest candidate per source.
    nodes_eq = pd.DataFrame({
        "node_id": [10, 11, 12, 13], "t": [0, 1, 1, 1],
        "z": [0.0] * 4, "y": [0.0, 1.0, 2.0, 100.0], "x": [0.0] * 4,
        "score": [0.9, 0.8, 0.8, 0.8],
    })
    ceiling_pool = build_ceiling_candidate_pool(nodes_eq, {})
    # y differences of 1/2 voxels ~ 0.406/0.813um (VOXEL_SIZE_UM["y"]=0.40625).
    filtered_tight = filter_candidate_pool(ceiling_pool, max_link_distance_um=0.5, k_next_candidates=1)
    assert list(filtered_tight["target_id"]) == [11], "only node 11 (~0.406um) survives a 0.5um/k=1 filter"
    filtered_wider = filter_candidate_pool(ceiling_pool, max_link_distance_um=1.0, k_next_candidates=2)
    assert set(filtered_wider["target_id"]) == {11, 12}, "both 11 and 12 (~0.406/0.813um) survive a 1.0um/k=2 filter"

    # Test 4: hard-negative sampling includes ALL positives, prioritizes
    # local_rank<=3 negatives, and respects the negative:positive ratio cap.
    train_df = pd.DataFrame({
        "label": [True, True] + [False] * 10,
        "local_rank": [1, 1] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    sampled = hard_negative_sample(train_df, negative_ratio=2, random_state=0)
    assert sampled["label"].sum() == 2, "both positives must always be included"
    n_negatives = int((~sampled["label"]).sum())
    assert n_negatives == 4, "negative:positive ratio of 2 with 2 positives caps negatives at 4"
    hard_included = sampled[(~sampled["label"]) & (sampled["local_rank"] <= 3)]
    assert len(hard_included) == 3, "all 3 hard negatives (rank<=3) should be included before any random fill"

    # Test 5: leave-one-sample-out partitioning - for a 3-dataset table,
    # each held-out fold's train/val split is disjoint and covers everything.
    toy = pd.DataFrame({"dataset": ["s0"] * 3 + ["s1"] * 4 + ["s2"] * 2, "value": range(9)})
    datasets = sorted(toy["dataset"].unique())
    assert datasets == ["s0", "s1", "s2"]
    for held_out in datasets:
        train_part = toy[toy["dataset"] != held_out]
        val_part = toy[toy["dataset"] == held_out]
        assert set(train_part["dataset"]) | set(val_part["dataset"]) == set(datasets)
        assert set(train_part.index) & set(val_part.index) == set()
        assert held_out not in set(train_part["dataset"])
        assert set(val_part["dataset"]) == {held_out}

    # Test 6: global top-K budget evaluation correctness on a small,
    # hand-scored fixture.
    scored = pd.DataFrame({
        "label": [True, True, False, False, False],
        "_score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "t": [0, 0, 0, 0, 0],
    })
    result_top2 = evaluate_global_top_k(scored, budget=2, gt_edge_count=3)
    assert result_top2["edge_TP"] == 2 and result_top2["edge_FP"] == 0 and result_top2["edge_FN"] == 1
    result_top4 = evaluate_global_top_k(scored, budget=4, gt_edge_count=3)
    assert result_top4["edge_TP"] == 2 and result_top4["edge_FP"] == 2

    # Test 7: per-frame top-K budget evaluation correctness across 2 frames.
    scored_frames = pd.DataFrame({
        "label": [True, False, False, True, False],
        "_score": [0.9, 0.8, 0.7, 0.95, 0.85],
        "t": [0, 0, 0, 1, 1],
    })
    result_pf1 = evaluate_per_frame_top_k(scored_frames, per_frame_k=1, gt_edge_count=2)
    assert result_pf1["edge_TP"] == 2 and result_pf1["edge_FP"] == 0, "top-1/frame picks exactly the 2 true edges here"
    result_pf2 = evaluate_per_frame_top_k(scored_frames, per_frame_k=2, gt_edge_count=2)
    assert result_pf2["edge_TP"] == 2 and result_pf2["edge_FP"] == 2, "top-2/frame keeps 4 rows total, 2 true + 2 false"

    # Test 8: a synthetic fixture where distance alone cleanly separates
    # true from false candidates - a classifier should achieve perfect AUC
    # and a generous top-K budget should recover every true edge with 0 FP.
    rng = np.random.default_rng(0)
    n_pos, n_neg = 30, 30
    fixture_df = pd.DataFrame({
        "label": [True] * n_pos + [False] * n_neg,
        "distance_um": np.concatenate([rng.uniform(0.1, 1.0, n_pos), rng.uniform(5.0, 10.0, n_neg)]),
        "node_score_product": rng.uniform(0.5, 0.9, n_pos + n_neg),
        "edge_score": rng.uniform(0.1, 0.9, n_pos + n_neg),
        "rank_score": rng.uniform(0.1, 1.0, n_pos + n_neg),
        "smoothness_score": rng.uniform(0.1, 1.0, n_pos + n_neg),
    })
    auc = roc_auc_score(fixture_df["label"], -fixture_df["distance_um"])
    assert auc > 0.99, f"distance alone should near-perfectly separate this fixture, got AUC={auc}"

    clf = LogisticRegression(class_weight="balanced")
    clf.fit(fixture_df[["distance_um"]], fixture_df["label"].astype(int))
    scores = clf.predict_proba(fixture_df[["distance_um"]])[:, 1]
    fixture_scored = fixture_df.copy()
    fixture_scored["_score"] = scores
    fixture_scored["t"] = 0
    result = evaluate_global_top_k(fixture_scored, budget=n_pos, gt_edge_count=n_pos)
    assert result["edge_TP"] == n_pos and result["edge_FP"] == 0, "a top-K budget sized to n_pos should recover all true edges with 0 FP"

    print("All milestone10_edge_classifier_diagnosis tests passed (8/8).")
# --------------------------------------------------------------------------- #
# 21. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone10_pipeline(
    n_detection_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    node_filter_configs: list[dict] = SELECTED_NODE_FILTER_CONFIGS,
) -> dict:
    """Runs the tests, then the full Milestone 10 diagnostic: Part A (GT
    graph structure over EVERY discovered train sample - cheap, GT-only),
    Parts B/C (cached candidate pool + features over the first
    `n_detection_samples` samples), Part D (oracle ceiling), Part E
    (feature separability), Part F (leave-one-sample-out classifier CV),
    and Part G (decision report). Builds no submission, never calls any
    Kaggle submission API, and never mutates 6C/6D/9's files.
    """
    print("=== Self-test: GT-degree, labeling, candidate-pool equivalence, sampling, CV-split, and budget-eval unit tests ===")
    run_milestone10_tests()

    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        empty = pd.DataFrame()
        return {"gt_structure_df": empty, "oracle_df": empty, "separability_df": empty, "cv_df": empty, "best_candidates_df": empty, "decision": {}}

    print(f"\n=== Part A: GT graph structural diagnosis (all {len(train_dirs)} discovered train sample(s)) ===")
    gt_structure_df = run_part_a_gt_structure(train_dirs)

    detection_samples = train_dirs[:n_detection_samples]
    print(f"\n=== Caching raw detection + GT for {len(detection_samples)} sample(s) (Parts B-F) ===")
    raw_caches = [precompute_raw_sample_cache(s, detector_config) for s in detection_samples]
    for rc in raw_caches:
        print(f"  {rc['dataset']}: {len(rc['raw_nodes_df'])} raw node(s), GT edges={len(rc['gt_edges'])}")

    print(f"\n=== Building node-filter caches + ceiling candidate pools for {len(node_filter_configs)} config(s) ===")
    node_filter_caches_by_name = build_node_filter_caches(raw_caches, node_filter_configs)
    for name, caches in node_filter_caches_by_name.items():
        total_candidates = sum(len(c["ceiling_pool"]) for c in caches)
        print(f"  {name}: {total_candidates} ceiling candidate edge(s) total across {len(caches)} sample(s)")

    print("\n=== Parts B/C: candidate labeling + feature engineering ===")
    feature_df = build_all_feature_tables(node_filter_caches_by_name)
    print(f"  built {len(feature_df)} labeled candidate row(s), {int(feature_df['label'].sum())} positive")

    print("\n=== Part D: oracle analysis ===")
    oracle_df = run_part_d_oracle(feature_df, node_filter_caches_by_name)

    print("\n=== Part E: feature separability ===")
    separability_df = run_part_e_separability(feature_df)

    print("\n=== Part F: leave-one-sample-out classifier CV ===")
    cv_df = run_part_f_classifiers(feature_df, node_filter_caches_by_name, node_filter_configs)

    print("\n=== Final best-candidate selection ===")
    best_candidates_df = select_best_candidates(cv_df)

    decision = generate_decision_report(gt_structure_df, oracle_df, separability_df, cv_df)

    return {
        "gt_structure_df": gt_structure_df, "oracle_df": oracle_df, "separability_df": separability_df,
        "cv_df": cv_df, "best_candidates_df": best_candidates_df, "decision": decision,
    }


if __name__ == "__main__":
    run_milestone10_pipeline(n_detection_samples=3)
