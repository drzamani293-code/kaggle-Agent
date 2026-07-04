"""
Biohub - Cell Tracking During Development
Milestone 14: M12 + graph-repair hybrid OOF validation.

Milestone 13's audit of a competing high-score reference solution (a
learned UNet + node-transformer + ILP pipeline, architecturally unrelated
to ours) found its supporting artifact/weights/scripts NOT available on
disk (replication_possible_in_kaggle_offline_mode=False) and recommended
porting 2 of its ideas - line-fit coordinate smoothing and motion-relink-
style edge assignment - into OUR OWN classical-detector + gradient-
boosted-classifier pipeline instead of replicating the reference.

This module evaluates 9 experiments (A-I) - M12's exact primary config
reproduced under Milestone 11's GroupKFold OOF harness, plus graph-repair
variants layered on top - on the SAME 12 fixed-seed robust train samples
Milestone 11 used, never on test data. No submission is built.

Design notes incorporating architecture review:

  - Shared base scoring: experiments B, D, F, G, H, I all reuse the SAME
    per-fold classifier scores (M12's unchanged HistGradientBoosting
    recipe, computed ONCE per fold per held-out sample) - only C/D/F swap
    out `greedy_exclusive_per_frame` for the motion-relink procedure. The
    conservative (ExtraTrees) config is ALSO scored once per fold for
    Experiment I's ensemble, never retrained per experiment.

  - Motion-relink (C): processes each held-out sample's frame transitions
    STRICTLY in ascending-t order (a real sequential/stateful algorithm -
    the tight/relaxed two-pass assignment within transition t only depends
    on that same transition's own state, and transition t+1's velocity
    lookups only depend on transitions <= t, both already-satisfied by a
    single ascending pass). Motion distance is computed directly from the
    existing `(dz_um,dy_um,dx_um)` displacement feature diffed against a
    predicted velocity - never needing absolute node positions for this
    step. Velocity is modeled as "repeat the last observed single-step
    displacement" (a deliberate, simple interpretation of "short-window
    velocity estimation" - the reference's actual algorithm was never
    available to us, only its name, via Milestone 13's static parsing).
    Assignment uses `scipy.optimize.linear_sum_assignment` on a cost
    matrix augmented with one "reject" column PER SOURCE (cost=1000,
    dwarfing every real gate-passing cost which stays within roughly
    [-0.75, 10.35] given the cost formula and um gates) so a source with
    no acceptable candidate is left unmatched rather than force-assigned
    to a bad option - avoiding the exact forced-assignment failure mode
    this codebase moved away from in an earlier milestone. Only
    candidates that ACTUALLY pass a given pass's motion-distance gate are
    even placed in the cost matrix (everything else is a hard BIG=1e6
    cell, never the raw formula cost, so a gate-failing candidate can
    never accidentally out-compete a real one). Sources/targets/rows are
    sorted by node_id before building each matrix so ties resolve
    deterministically run-to-run.

  - Gap-closing (E) and safe-division recovery (G) both need ABSOLUTE
    (z,y,x) positions (sister-child distance, end-to-t+2 distance) that
    the displacement-only features don't carry - these are looked up from
    `filtered_nodes` and explicitly voxel-scaled via `VOXEL_SIZE_UM`
    before comparison against any micrometer gate (a real, easy-to-miss
    unit-mismatch trap: `filtered_nodes`' z/y/x are raw voxel indices, not
    micrometers).

  - Topology validation is a first-class, shared function
    (`compute_topology_stats`) used both to report the required topology
    CSV columns AND as the hard-fail gate: dangling edges and
    MULTI-PARENT (in-degree>1, a genuinely different concept from
    division/multi-CHILD out-degree>1) must NEVER occur in any of the 9
    experiments, including G - G only ever admits a second edge into a
    target with in-degree==0, so it can create controlled out-degree=2
    (division) without ever creating in-degree>1. Every experiment except
    G asserts max out-degree<=1 as a hard structural invariant (asserted
    loudly, not silently marked "failed to improve", since a violation in
    a topology-preserving experiment like B indicates a harness bug, not
    a legitimate experimental outcome).

  - Experiment A's reproduction is treated as a HARD gate per the user's
    explicit hard-fail list: if it does not approximately match the
    Milestone 11/12 anchor numbers, `harness_reproduction_ok=False` is
    set, printed loudly, and the final decision report explicitly
    withholds a "port X" recommendation (recommending only "fix the
    harness reproduction mismatch first") - experiments B-I still run
    (the underlying detection/feature caching is expensive and their
    numbers remain informative for debugging), but they are never used to
    justify a port-to-M12 decision while A itself is not trustworthy.

  - Ensemble (I) conflict resolution NEVER compares raw scores across the
    two different classifiers (HistGradientBoosting vs ExtraTrees produce
    non-comparable probability scales - the same pooling trap Milestone
    11 was built to avoid). The union variant instead uses a deterministic
    PRIORITY rule (primary's already-internally-exclusive edge set is
    kept whole; conservative's edges are added only where they don't
    conflict with an already-used source/target), guaranteeing the
    unioned result stays topologically clean without any cross-classifier
    score comparison. The intersection variant needs no repair at all -
    a subset of an already-exclusive edge set is trivially still exclusive.

No submission is ever built, no Kaggle API is ever called, Save Version
is never triggered, and test data is never used for config selection.
6C/6D/9/10/11/12/13's files are not modified - all reused logic is copied
in, not imported.
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

# --------------------------------------------------------------------------- #
# 22. Topology validation + distance/degree stats (shared across every
#     experiment - both the required CSV columns AND the hard-fail gate)
# --------------------------------------------------------------------------- #
def attach_edge_distances(pred_edges: pd.DataFrame, pred_nodes: pd.DataFrame) -> pd.DataFrame:
    """Computes/overwrites `distance_um` for every edge from absolute node
    positions. `reconstruct_policy_a`/`reconstruct_policy_b`'s edges never
    carry a distance_um column through (only source_id/target_id) - this
    is attached uniformly here (using the FINAL pred_nodes, which includes
    any gap-closing-added real/synthetic nodes) so every downstream
    topology/distance-stat computation sees a complete, consistent column
    regardless of which experiment produced the edge.
    """
    if len(pred_edges) == 0:
        result = pred_edges.copy()
        result["distance_um"] = pd.Series(dtype=float)
        return result
    node_lookup = pred_nodes.set_index("node_id")[["z", "y", "x"]]
    distances = []
    for row in pred_edges.itertuples():
        try:
            src_pos = node_lookup.loc[row.source_id].to_numpy(dtype=float)
            tgt_pos = node_lookup.loc[row.target_id].to_numpy(dtype=float)
            distances.append(physical_distance_um(src_pos, tgt_pos))
        except KeyError:
            distances.append(np.nan)
    result = pred_edges.copy()
    result["distance_um"] = distances
    return result


def compute_topology_stats(pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, allow_divisions: bool = False) -> dict:
    """Dangling/duplicate-edge checks, in/out-degree distributions, t-gap
    distribution, and edge-distance percentiles for ONE sample's final
    predicted (nodes, edges). `allow_divisions` permits out-degree==2
    (Experiment G only) without ever permitting MULTI-PARENT
    (in-degree>1, a genuinely different concept from division) - dangling
    edges and multi-parent are NEVER legitimate in any experiment.
    """
    if len(pred_nodes) == 0:
        empty = {
            "dangling_edge_count": 0, "duplicate_edge_count": 0,
            "out_degree_distribution": {}, "in_degree_distribution": {},
            "division_like_source_count": 0, "multi_parent_count": 0, "max_out_degree": 0,
            "t_gap_distribution": {}, "edge_distance_summary": {}, "topology_valid": len(pred_edges) == 0,
        }
        return empty

    node_ids = set(pred_nodes["node_id"])
    node_t_lookup = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))

    if len(pred_edges) == 0:
        dangling_count, duplicate_count = 0, 0
        valid_edges = pred_edges
    else:
        dangling_mask = [
            not (s in node_ids and t in node_ids) for s, t in zip(pred_edges["source_id"], pred_edges["target_id"])
        ]
        dangling_count = int(sum(dangling_mask))
        non_dangling = pred_edges[~pd.Series(dangling_mask, index=pred_edges.index)]
        duplicate_count = int(non_dangling.duplicated(subset=["source_id", "target_id"]).sum())
        valid_edges = non_dangling.drop_duplicates(subset=["source_id", "target_id"])

    out_degree = valid_edges.groupby("source_id").size() if len(valid_edges) else pd.Series(dtype=int)
    in_degree = valid_edges.groupby("target_id").size() if len(valid_edges) else pd.Series(dtype=int)
    all_ids = pred_nodes["node_id"]
    out_degree_full = out_degree.reindex(all_ids, fill_value=0)
    in_degree_full = in_degree.reindex(all_ids, fill_value=0)

    out_degree_distribution = {str(k): int(v) for k, v in out_degree_full.value_counts().sort_index().items()}
    in_degree_distribution = {str(k): int(v) for k, v in in_degree_full.value_counts().sort_index().items()}
    division_like_source_count = int((out_degree_full == 2).sum())
    multi_parent_count = int((in_degree_full > 1).sum())
    max_out_degree = int(out_degree_full.max()) if len(out_degree_full) else 0

    if len(valid_edges):
        t_gaps = [node_t_lookup[t] - node_t_lookup[s] for s, t in zip(valid_edges["source_id"], valid_edges["target_id"])]
        t_gap_distribution = {str(k): int(v) for k, v in pd.Series(t_gaps).value_counts().sort_index().items()}
    else:
        t_gap_distribution = {}

    if len(valid_edges) and "distance_um" in valid_edges.columns:
        dist = valid_edges["distance_um"].dropna()
        edge_distance_summary = (
            {"mean": float(dist.mean()), "median": float(dist.median()), "p90": float(dist.quantile(0.90)),
             "p95": float(dist.quantile(0.95)), "p99": float(dist.quantile(0.99)), "max": float(dist.max())}
            if len(dist) else {}
        )
    else:
        edge_distance_summary = {}

    max_allowed_out_degree = 2 if allow_divisions else 1
    topology_valid = (dangling_count == 0) and (multi_parent_count == 0) and (max_out_degree <= max_allowed_out_degree)

    return {
        "dangling_edge_count": dangling_count, "duplicate_edge_count": duplicate_count,
        "out_degree_distribution": out_degree_distribution, "in_degree_distribution": in_degree_distribution,
        "division_like_source_count": division_like_source_count, "multi_parent_count": multi_parent_count,
        "max_out_degree": max_out_degree, "t_gap_distribution": t_gap_distribution,
        "edge_distance_summary": edge_distance_summary, "topology_valid": topology_valid,
    }


def assert_topology_invariants(topology_stats: dict, experiment_id: str, sample_name: str) -> None:
    """Raises loudly - never silently marks a per-experiment row as
    'failed to improve' - if a structural invariant is violated in an
    experiment where it should be structurally impossible. This is the
    difference between a legitimate experimental outcome (didn't beat the
    baseline) and a harness bug (produced an invalid graph).
    """
    if not topology_stats["topology_valid"]:
        raise AssertionError(
            f"Experiment {experiment_id} produced an INVALID topology on sample {sample_name!r}: "
            f"dangling_edge_count={topology_stats['dangling_edge_count']}, "
            f"multi_parent_count={topology_stats['multi_parent_count']}, "
            f"max_out_degree={topology_stats['max_out_degree']}"
        )

# --------------------------------------------------------------------------- #
# 23. Experiment B - line-fit smoothing (topology-preserving)
# --------------------------------------------------------------------------- #
LINEFIT_WINDOW_DEFAULT = 2
LINEFIT_WEIGHT_DEFAULT = 0.8


def linefit_smooth_nodes(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame,
    window: int = LINEFIT_WINDOW_DEFAULT, weight: float = LINEFIT_WEIGHT_DEFAULT,
) -> pd.DataFrame:
    """Topology-preserving coordinate smoothing - NEVER adds/removes a
    node or edge, only reassigns (z,y,x) for qualifying nodes. Only
    smooths "linear interior" nodes: exactly one predecessor AND exactly
    one successor, with consecutive t (pred.t == node.t-1, succ.t ==
    node.t+1) on both sides - division nodes, branch points, and chain
    endpoints are left untouched. Walks up to `window` steps in each
    direction along the (guaranteed-simple, since exactly-one-in/one-out)
    chain, stopping early at a branch/endpoint, averages the collected
    neighbor positions, and blends `weight` x original + `(1-weight)` x
    neighbor_avg.
    """
    if len(pred_nodes) == 0 or len(pred_edges) == 0:
        return pred_nodes.copy()

    succ = dict(zip(pred_edges["source_id"], pred_edges["target_id"]))
    pred = dict(zip(pred_edges["target_id"], pred_edges["source_id"]))
    out_degree = pred_edges.groupby("source_id").size().to_dict()
    in_degree = pred_edges.groupby("target_id").size().to_dict()

    node_lookup = pred_nodes.set_index("node_id")
    t_lookup = node_lookup["t"].to_dict()

    def is_linear_interior(node_id) -> bool:
        if out_degree.get(node_id, 0) != 1 or in_degree.get(node_id, 0) != 1:
            return False
        succ_id, pred_id = succ.get(node_id), pred.get(node_id)
        if succ_id is None or pred_id is None:
            return False
        return t_lookup.get(succ_id) == t_lookup[node_id] + 1 and t_lookup.get(pred_id) == t_lookup[node_id] - 1

    def collect_neighbors(node_id, direction: str, steps: int) -> list:
        neighbors = []
        cur = node_id
        for _ in range(steps):
            nxt = succ.get(cur) if direction == "succ" else pred.get(cur)
            if nxt is None:
                break
            neighbors.append(nxt)
            if not is_linear_interior(nxt):
                break
            cur = nxt
        return neighbors

    smoothed_positions: dict = {}
    for node_id in pred_nodes["node_id"]:
        if not is_linear_interior(node_id):
            continue
        neighbor_ids = collect_neighbors(node_id, "succ", window) + collect_neighbors(node_id, "pred", window)
        if not neighbor_ids:
            continue
        neighbor_coords = node_lookup.loc[neighbor_ids, ["z", "y", "x"]].to_numpy(dtype=float)
        neighbor_avg = neighbor_coords.mean(axis=0)
        original = node_lookup.loc[node_id, ["z", "y", "x"]].to_numpy(dtype=float)
        smoothed_positions[node_id] = weight * original + (1.0 - weight) * neighbor_avg

    result = pred_nodes.copy()
    for node_id, pos in smoothed_positions.items():
        result.loc[result["node_id"] == node_id, ["z", "y", "x"]] = pos
    return result

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
# 25. Experiment E - one-frame gap closing (diagnostic; highest risk)
# --------------------------------------------------------------------------- #
GAP_CLOSE_MAX_TOTAL_DISTANCE_UM = 12.0
GAP_CLOSE_MAX_PER_STEP_DISTANCE_UM = 6.0
GAP_CLOSE_MAX_REFINEMENT_SHIFT_UM = 3.2  # documented gate for image-intensity refinement - not implemented (no image re-query in this OOF harness), kept as a named constant for parity with the reference design
GAP_CLOSE_MAX_ADDED_ABS = 2000
GAP_CLOSE_MAX_ADDED_FRAC = 0.05


def find_gap_close_candidates(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, filtered_nodes: pd.DataFrame,
    max_total_distance: float = GAP_CLOSE_MAX_TOTAL_DISTANCE_UM, max_per_step_distance: float = GAP_CLOSE_MAX_PER_STEP_DISTANCE_UM,
) -> list[dict]:
    """Identifies tracklet END nodes (out_degree==0, not at the sample's
    last timepoint) and START nodes (in_degree==0, not at the first
    timepoint) separated by exactly 2 frames, gated on total AND per-step
    physical distance (via `physical_distance_um`, which applies
    VOXEL_SIZE_UM internally - `filtered_nodes`' z/y/x are raw voxel
    indices, never already in micrometers). For each admissible pair,
    prefers reusing a real, currently-unused node-filter survivor near the
    implied t+1 midpoint over a synthetic interpolated point.
    """
    if len(pred_nodes) == 0:
        return []

    node_lookup = filtered_nodes.set_index("node_id")
    out_degree = pred_edges.groupby("source_id").size().to_dict() if len(pred_edges) else {}
    in_degree = pred_edges.groupby("target_id").size().to_dict() if len(pred_edges) else {}

    max_t, min_t = int(filtered_nodes["t"].max()), int(filtered_nodes["t"].min())
    used_pred_node_ids = set(pred_nodes["node_id"])

    end_nodes = [(nid, int(t)) for nid, t in zip(pred_nodes["node_id"], pred_nodes["t"]) if out_degree.get(nid, 0) == 0 and t < max_t]
    start_nodes = [(nid, int(t)) for nid, t in zip(pred_nodes["node_id"], pred_nodes["t"]) if in_degree.get(nid, 0) == 0 and t > min_t]
    starts_by_t: dict[int, list] = {}
    for nid, t in start_nodes:
        starts_by_t.setdefault(t, []).append(nid)

    candidates = []
    for end_id, end_t in end_nodes:
        target_t = end_t + 2
        end_pos = node_lookup.loc[end_id, ["z", "y", "x"]].to_numpy(dtype=float)
        for start_id in starts_by_t.get(target_t, []):
            start_pos = node_lookup.loc[start_id, ["z", "y", "x"]].to_numpy(dtype=float)
            total_distance = physical_distance_um(end_pos, start_pos)
            if total_distance > max_total_distance:
                continue

            mid_t = end_t + 1
            mid_pool = filtered_nodes[(filtered_nodes["t"] == mid_t) & (~filtered_nodes["node_id"].isin(used_pred_node_ids))]
            best_real_id, best_real_cost = None, None
            for mid_row in mid_pool.itertuples():
                mid_pos = np.array([mid_row.z, mid_row.y, mid_row.x], dtype=float)
                step1, step2 = physical_distance_um(end_pos, mid_pos), physical_distance_um(mid_pos, start_pos)
                if step1 <= max_per_step_distance and step2 <= max_per_step_distance:
                    cost = step1 + step2
                    if best_real_cost is None or cost < best_real_cost:
                        best_real_cost, best_real_id = cost, mid_row.node_id

            candidates.append({
                "end_node_id": end_id, "start_node_id": start_id, "mid_t": mid_t, "total_distance_um": total_distance,
                "reused_node_id": best_real_id, "provenance": "reused_real_node" if best_real_id is not None else "synthetic_midpoint",
            })
    return candidates


def gap_close(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, filtered_nodes: pd.DataFrame,
    max_total_distance: float = GAP_CLOSE_MAX_TOTAL_DISTANCE_UM, max_per_step_distance: float = GAP_CLOSE_MAX_PER_STEP_DISTANCE_UM,
    max_added_abs: int = GAP_CLOSE_MAX_ADDED_ABS, max_added_frac: float = GAP_CLOSE_MAX_ADDED_FRAC, allow_synthetic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Applies gap-closing under the total-added-node cap
    (min(max_added_abs, max_added_frac x current node count)), preferring
    real-node reuse; synthetic midpoints (a straight-line interpolation -
    no image-intensity re-query is implemented in this OOF harness) are
    used only when no real candidate is admissible and `allow_synthetic`
    is set (diagnostic default True here; the final recommendation still
    treats synthetic insertion as non-default per the milestone brief).
    Synthetic node_ids are negative (real detector node_ids are always
    >=0) so they can never collide with a real id.
    """
    candidates = find_gap_close_candidates(pred_nodes, pred_edges, filtered_nodes, max_total_distance, max_per_step_distance)
    max_added = min(max_added_abs, int(max_added_frac * max(len(pred_nodes), 1)))

    node_lookup = filtered_nodes.set_index("node_id")
    new_node_rows, new_edge_rows = [], []
    used_mid_nodes, used_end_nodes, used_start_nodes = set(), set(), set()
    n_reused, n_synthetic = 0, 0
    next_synthetic_id = -1

    for cand in sorted(candidates, key=lambda c: c["total_distance_um"]):
        if len(new_node_rows) >= max_added:
            break
        if cand["end_node_id"] in used_end_nodes or cand["start_node_id"] in used_start_nodes:
            continue

        end_pos = node_lookup.loc[cand["end_node_id"], ["z", "y", "x"]].to_numpy(dtype=float)
        start_pos = node_lookup.loc[cand["start_node_id"], ["z", "y", "x"]].to_numpy(dtype=float)

        if cand["reused_node_id"] is not None:
            if cand["reused_node_id"] in used_mid_nodes:
                continue
            mid_id = cand["reused_node_id"]
            mid_pos = node_lookup.loc[mid_id, ["z", "y", "x"]].to_numpy(dtype=float)
            n_reused += 1
        elif allow_synthetic:
            mid_pos = (end_pos + start_pos) / 2.0
            mid_id = next_synthetic_id
            next_synthetic_id -= 1
            n_synthetic += 1
        else:
            continue

        new_node_rows.append({"node_id": mid_id, "t": cand["mid_t"], "z": float(mid_pos[0]), "y": float(mid_pos[1]), "x": float(mid_pos[2])})
        new_edge_rows.append({"source_id": cand["end_node_id"], "target_id": mid_id, "distance_um": physical_distance_um(end_pos, mid_pos)})
        new_edge_rows.append({"source_id": mid_id, "target_id": cand["start_node_id"], "distance_um": physical_distance_um(mid_pos, start_pos)})
        used_mid_nodes.add(mid_id)
        used_end_nodes.add(cand["end_node_id"])
        used_start_nodes.add(cand["start_node_id"])

    result_nodes = pd.concat([pred_nodes, pd.DataFrame(new_node_rows)], ignore_index=True) if new_node_rows else pred_nodes.copy()
    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edge_rows)], ignore_index=True) if new_edge_rows else pred_edges.copy()

    diagnostics = {
        "n_gap_candidates_found": len(candidates), "n_gaps_closed": n_reused + n_synthetic,
        "n_reused_real_node": n_reused, "n_synthetic_midpoint": n_synthetic, "max_added_cap": max_added,
    }
    return result_nodes, result_edges, diagnostics

# --------------------------------------------------------------------------- #
# 26. Experiment G - safe division recovery (diagnostic)
# --------------------------------------------------------------------------- #
SAFE_DIVISION_PARENT_CHILD_MAX_UM = 4.7
SAFE_DIVISION_SISTER_MAX_UM = 7.2
SAFE_DIVISION_EXISTING_CHILD_MAX_UM = 7.8
SAFE_DIVISION_FRAME_CAP_FRAC = 0.008
SAFE_DIVISION_GLOBAL_CAP_FRAC = 0.004


def add_safe_divisions(
    pred_nodes: pd.DataFrame, pred_edges: pd.DataFrame, scored_df: pd.DataFrame, filtered_nodes: pd.DataFrame,
    parent_child_max_um: float = SAFE_DIVISION_PARENT_CHILD_MAX_UM, sister_max_um: float = SAFE_DIVISION_SISTER_MAX_UM,
    existing_child_max_um: float = SAFE_DIVISION_EXISTING_CHILD_MAX_UM,
    frame_cap_frac: float = SAFE_DIVISION_FRAME_CAP_FRAC, global_cap_frac: float = SAFE_DIVISION_GLOBAL_CAP_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Admits a SECOND outgoing edge for a source that currently has
    EXACTLY one accepted child, under strict geometric gates. Maintains
    the hard invariants: no multi-parent (the new target must currently
    have in-degree==0), source-currently-has-exactly-one-child (a source
    is never given a THIRD child), and the ORIGINAL single edge's own
    distance must itself have been <=existing_child_max_um (don't turn an
    already-stretched single link into a division). Candidates are
    admitted in classifier-score-descending order so the frame/global
    caps favor the most-confident divisions first.
    """
    if len(pred_edges) == 0:
        return pred_nodes.copy(), pred_edges.copy(), {"n_divisions_added": 0, "n_candidates_considered": 0}

    node_lookup = filtered_nodes.set_index("node_id")
    out_degree = pred_edges.groupby("source_id").size().to_dict()
    in_degree = pred_edges.groupby("target_id").size().to_dict()

    # `pred_edges` (from reconstruct_policy_a/b) carries only source_id/
    # target_id - never a distance_um column - so the existing child's
    # distance is computed here from absolute node positions, not read
    # off a column that may not exist.
    existing_child_by_source: dict[int, tuple] = {}
    for row in pred_edges.itertuples():
        if out_degree.get(row.source_id, 0) == 1:
            src_pos = node_lookup.loc[row.source_id, ["z", "y", "x"]].to_numpy(dtype=float)
            tgt_pos = node_lookup.loc[row.target_id, ["z", "y", "x"]].to_numpy(dtype=float)
            existing_child_by_source[row.source_id] = (row.target_id, physical_distance_um(src_pos, tgt_pos))

    node_t_lookup = dict(zip(pred_nodes["node_id"], pred_nodes["t"]))
    frame_totals: dict[int, int] = {}
    for t in node_t_lookup.values():
        frame_totals[t] = frame_totals.get(t, 0) + 1

    global_cap = int(global_cap_frac * max(len(pred_nodes), 1))
    frame_counts: dict[int, int] = {}
    candidate_rows = scored_df[scored_df["source_id"].isin(existing_child_by_source.keys())].sort_values("_score", ascending=False)

    new_edge_rows = []
    n_added = 0
    for row in candidate_rows.itertuples():
        if n_added >= global_cap:
            break
        source_id = row.source_id
        existing_child_id, existing_child_distance = existing_child_by_source[source_id]
        if row.target_id == existing_child_id:
            continue
        if in_degree.get(row.target_id, 0) != 0:
            continue  # would create multi-parent
        if out_degree.get(source_id, 0) != 1:
            continue  # source already divided this pass (never a 3rd child)
        if existing_child_distance > existing_child_max_um:
            continue
        if row.distance_um > parent_child_max_um:
            continue

        existing_child_pos = node_lookup.loc[existing_child_id, ["z", "y", "x"]].to_numpy(dtype=float)
        new_child_pos = node_lookup.loc[row.target_id, ["z", "y", "x"]].to_numpy(dtype=float)
        if physical_distance_um(existing_child_pos, new_child_pos) > sister_max_um:
            continue

        frame_t = row.t
        frame_cap = int(frame_cap_frac * max(frame_totals.get(frame_t, 0), 1))
        if frame_cap <= 0 or frame_counts.get(frame_t, 0) >= frame_cap:
            continue

        new_edge_rows.append({"source_id": source_id, "target_id": row.target_id, "distance_um": row.distance_um})
        out_degree[source_id] = out_degree.get(source_id, 0) + 1
        in_degree[row.target_id] = in_degree.get(row.target_id, 0) + 1
        frame_counts[frame_t] = frame_counts.get(frame_t, 0) + 1
        n_added += 1

    result_edges = pd.concat([pred_edges, pd.DataFrame(new_edge_rows)], ignore_index=True) if new_edge_rows else pred_edges.copy()
    diagnostics = {"n_divisions_added": n_added, "n_candidates_considered": len(candidate_rows)}
    return pred_nodes.copy(), result_edges, diagnostics

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
# 28. Fixed configs (M12 primary + conservative, unchanged) and M11 anchor
# --------------------------------------------------------------------------- #
PRIMARY_NODE_FILTER_NAME = "moderate_150_085"
BASE_CLASSIFIER_NAME = "hist_gradient_boosting"
BASE_NEGATIVE_RATIO = 100
BASE_EDGE_SELECTION_PARAM = 0.9  # greedy_exclusive_per_frame quantile
BASE_TRACKLET_FILTER = {"min_tracklet_length": 5, "min_mean_classifier_score_quantile": 0.7, "max_smoothness_error_um": 3.0, "keep_top_k": 100}

CONSERVATIVE_CLASSIFIER_NAME = "extra_trees"
CONSERVATIVE_EDGE_SELECTION_PARAM = 0.995  # per_source_top1_with_score_quantile quantile
CONSERVATIVE_TRACKLET_FILTER = {"min_tracklet_length": 2, "min_mean_classifier_score_quantile": 0.0, "max_smoothness_error_um": 3.0, "keep_top_k": 200}

M11_BEST_EDGE_JACCARD = 0.013215
M11_EDGE_TP = 103
M11_EDGE_FP = 1932
M11_EDGE_FN = 5759
M11_N_SAMPLES = 12
M11_N_SAMPLES_WITH_TP = 6
BASELINE_6C6D_EDGE_JACCARD = 0.010667

A_REPRODUCTION_TOLERANCE_FRAC = 0.15  # "approximately matches" - the harness is fully deterministic (same seed/recipe), so a real mismatch beyond this is a code bug, not run-to-run noise

EXPERIMENT_IDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I_union", "I_intersection"]

EXPERIMENT_DESCRIPTIONS = {
    "A": "M12 baseline reproduced (required before trusting any other experiment)",
    "B": "M12 + linefit_smoothing only (topology-preserving)",
    "C": "M12 + motion_relink only (replaces greedy_exclusive_per_frame)",
    "D": "M12 + motion_relink + linefit",
    "E": "M12 + gap_close diagnostic (highest risk)",
    "F": "M12 + motion_relink + gap_close + linefit (best-case full stack)",
    "G": "M12 + safe_divisions diagnostic",
    "H": "M12 + dense-node policy diagnostic (node_policy B)",
    "I_union": "M12 primary/conservative ensemble diagnostic - union",
    "I_intersection": "M12 primary/conservative ensemble diagnostic - intersection",
}

# --------------------------------------------------------------------------- #
# 29. Experiment dispatch - builds each experiment's (pred_nodes, pred_edges)
#     from a fold's already-scored candidate table(s)
# --------------------------------------------------------------------------- #
def _apply_base_tracklet_filter(tracklets_df: pd.DataFrame, filter_params: dict) -> pd.DataFrame:
    min_mean_edge_score_threshold = (
        float(tracklets_df["mean_edge_score"].quantile(filter_params["min_mean_classifier_score_quantile"]))
        if len(tracklets_df) else 0.0
    )
    return apply_tracklet_filter(
        tracklets_df, min_tracklet_length=filter_params["min_tracklet_length"], min_mean_node_score=0.0,
        min_mean_edge_score_threshold=min_mean_edge_score_threshold, max_mean_link_distance_um=CEILING_MAX_LINK_DISTANCE_UM,
        max_smoothness_error_um=filter_params["max_smoothness_error_um"], keep_top_k=filter_params["keep_top_k"],
    )


def _run_base_pipeline(scored_val: pd.DataFrame, filtered_nodes: pd.DataFrame, edge_selection_param: float, tracklet_filter: dict, edge_selection_fn=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fn = edge_selection_fn or select_greedy_exclusive_per_frame
    selected_edges, _ = fn(scored_val, quantile=edge_selection_param)
    tracklets_df = build_tracklets_from_selected_edges(selected_edges, filtered_nodes)
    kept = _apply_base_tracklet_filter(tracklets_df, tracklet_filter)
    pred_nodes, pred_edges = reconstruct_policy_a(kept, filtered_nodes)
    return pred_nodes, pred_edges, {}


def run_experiment(
    experiment_id: str, scored_val: pd.DataFrame, conservative_scored_val: pd.DataFrame | None, filtered_nodes: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, dict]]:
    """Builds one experiment's final (pred_nodes, pred_edges) and attaches
    a fresh `distance_um` column to every result (computed from the FINAL
    pred_nodes' absolute positions - reconstruct_policy_a/b's edges never
    carry this column through, so it is never assumed to already be
    present or consistent across differently-provenanced edges, e.g.
    gap-closing's newly-inserted edges vs. the base pipeline's).
    """
    results = _run_experiment_core(experiment_id, scored_val, conservative_scored_val, filtered_nodes)
    return [(sub_id, pred_nodes, attach_edge_distances(pred_edges, pred_nodes), diag) for sub_id, pred_nodes, pred_edges, diag in results]


def _run_experiment_core(
    experiment_id: str, scored_val: pd.DataFrame, conservative_scored_val: pd.DataFrame | None, filtered_nodes: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, dict]]:
    """Builds one experiment's final (pred_nodes, pred_edges), respecting
    the documented step order for combined experiments: edge selection ->
    tracklet construction/filter -> node-policy reconstruction -> gap-close
    (E,F) -> safe-divisions (G) -> linefit smoothing (B,D,F) LAST (a
    coordinate-only post-process that must never precede a step that adds
    new nodes/edges). Returns a LIST of (sub_experiment_id, pred_nodes,
    pred_edges, diagnostics) so Experiment I can report both its union and
    intersection variants through the same uniform interface.
    """
    if experiment_id == "I":
        primary_nodes, primary_edges, _ = _run_base_pipeline(scored_val, filtered_nodes, BASE_EDGE_SELECTION_PARAM, BASE_TRACKLET_FILTER)
        if conservative_scored_val is None:
            return [("I_union", primary_nodes, primary_edges, {"ensemble_mode": "no_conservative_available_fallback_to_primary"})]

        conservative_nodes, conservative_edges, _ = _run_base_pipeline(
            conservative_scored_val, filtered_nodes, CONSERVATIVE_EDGE_SELECTION_PARAM, CONSERVATIVE_TRACKLET_FILTER,
            edge_selection_fn=select_per_source_top1_with_quantile,
        )
        union_nodes, union_edges = combine_ensemble(primary_nodes, primary_edges, conservative_nodes, conservative_edges, mode="union")
        intersection_nodes, intersection_edges = combine_ensemble(primary_nodes, primary_edges, conservative_nodes, conservative_edges, mode="intersection")
        return [
            ("I_union", union_nodes, union_edges, {"ensemble_mode": "union", "n_primary_edges": len(primary_edges), "n_conservative_edges": len(conservative_edges)}),
            ("I_intersection", intersection_nodes, intersection_edges, {"ensemble_mode": "intersection", "n_primary_edges": len(primary_edges), "n_conservative_edges": len(conservative_edges)}),
        ]

    diag: dict = {}
    if experiment_id in ("C", "D", "F"):
        selected_edges, motion_diag = select_edges_motion_relink(scored_val)
        diag.update(motion_diag)
    else:
        selected_edges, _ = select_greedy_exclusive_per_frame(scored_val, quantile=BASE_EDGE_SELECTION_PARAM)

    tracklets_df = build_tracklets_from_selected_edges(selected_edges, filtered_nodes)
    kept = _apply_base_tracklet_filter(tracklets_df, BASE_TRACKLET_FILTER)

    if experiment_id == "H":
        pred_nodes, pred_edges = reconstruct_policy_b(filtered_nodes, kept)
    else:
        pred_nodes, pred_edges = reconstruct_policy_a(kept, filtered_nodes)

    if experiment_id in ("E", "F"):
        pred_nodes, pred_edges, gap_diag = gap_close(pred_nodes, pred_edges, filtered_nodes)
        diag.update(gap_diag)

    if experiment_id == "G":
        pred_nodes, pred_edges, div_diag = add_safe_divisions(pred_nodes, pred_edges, scored_val, filtered_nodes)
        diag.update(div_diag)

    if experiment_id in ("B", "D", "F"):
        pred_nodes = linefit_smooth_nodes(pred_nodes, pred_edges)

    return [(experiment_id, pred_nodes, pred_edges, diag)]

# --------------------------------------------------------------------------- #
# 29b. Aggregate OOF (copied from Milestone 11, unmodified)
# --------------------------------------------------------------------------- #
def aggregate_oof(per_sample_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Pooled TP/FP/FN summed across ALL held-out samples within each
    group (micro-average) - NEVER a mean of per-sample edge_jaccard values,
    which is unstable when many samples have zero or near-zero GT-positive
    candidates. Also reports `n_samples` and `n_samples_with_tp` per group.
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

# --------------------------------------------------------------------------- #
# 30. Orchestration - fold loop, per-experiment evaluation, aggregation
# --------------------------------------------------------------------------- #
def run_milestone14_experiments(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
) -> dict:
    """Recomputes Milestone 11's exact fixed-seed robust train-sample
    selection, then for every GroupKFold fold trains the primary
    (HistGradientBoosting) and conservative (ExtraTrees) classifiers ONCE
    (M12's unchanged recipe) and scores every held-out sample individually
    - every experiment that doesn't need re-scoring (B,D,F,G,H,I) reuses
    these SAME scores; only C/D/F swap out the edge-selection step itself.
    """
    node_filter_config = next(c for c in SELECTED_NODE_FILTER_CONFIGS if c["name"] == PRIMARY_NODE_FILTER_NAME)

    train_dirs = select_train_samples_robust(n_samples=n_robust_samples, seed=seed)
    print(f"Selected {len(train_dirs)} robust train sample(s): {[d.stem for d in train_dirs]}")
    raw_caches = cache_samples_within_budget(train_dirs, DEFAULT_DETECTOR_CONFIG, time_budget_seconds)
    dataset_names = [rc["dataset"] for rc in raw_caches]
    print(f"Cached raw detection + GT for {len(raw_caches)} sample(s)")

    node_filter_caches = [precompute_node_filter_cache(rc, node_filter_config) for rc in raw_caches]
    node_filter_cache_by_dataset = {c["dataset"]: c for c in node_filter_caches}
    gt_edge_count_by_dataset = {c["dataset"]: c["gt_edge_count"] for c in node_filter_caches}

    feature_df = build_all_feature_tables({PRIMARY_NODE_FILTER_NAME: node_filter_caches})
    n_positive = int(feature_df["label"].sum()) if len(feature_df) else 0
    print(f"Built {len(feature_df)} labeled candidate row(s): {n_positive} positive, {len(feature_df) - n_positive} negative")

    folds = build_folds("robust", dataset_names)
    positive_counts = compute_positive_counts_by_dataset(feature_df)
    folds = annotate_fold_degeneracy(folds, positive_counts)

    per_sample_rows: list[dict] = []
    topology_rows: list[dict] = []
    edge_distance_rows: list[dict] = []

    for fold in folds:
        if fold.get("degenerate"):
            continue
        primary_fold_result = train_and_score_fold(fold, feature_df, BASE_CLASSIFIER_NAME, BASE_NEGATIVE_RATIO, random_state=seed)
        if primary_fold_result is None:
            continue
        conservative_fold_result = train_and_score_fold(fold, feature_df, CONSERVATIVE_CLASSIFIER_NAME, BASE_NEGATIVE_RATIO, random_state=seed)

        for held_out_dataset, scored_val in primary_fold_result["scored_by_dataset"].items():
            node_filter_cache = node_filter_cache_by_dataset[held_out_dataset]
            filtered_nodes = node_filter_cache["filtered_nodes"]
            gt_edge_count = gt_edge_count_by_dataset[held_out_dataset]
            conservative_scored_val = (
                conservative_fold_result["scored_by_dataset"].get(held_out_dataset) if conservative_fold_result else None
            )

            for experiment_id in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
                exp_t0 = time.time()
                try:
                    results = run_experiment(experiment_id, scored_val, conservative_scored_val, filtered_nodes)
                except Exception as exc:
                    per_sample_rows.append({
                        "experiment_id": experiment_id, "held_out_dataset": held_out_dataset, "fold_id": fold["fold_id"],
                        "status": "ERROR", "error": str(exc), "edge_TP": 0, "edge_FP": 0, "edge_FN": gt_edge_count,
                        "edge_jaccard": 0.0, "precision": 0.0, "recall": 0.0, "runtime_seconds": time.time() - exp_t0,
                    })
                    print(f"  [error] experiment {experiment_id} on {held_out_dataset}: {exc!r}")
                    continue

                for sub_id, pred_nodes, pred_edges, exp_diag in results:
                    allow_divisions = sub_id == "G"
                    topo = compute_topology_stats(pred_nodes, pred_edges, allow_divisions=allow_divisions)
                    try:
                        assert_topology_invariants(topo, sub_id, held_out_dataset)
                    except AssertionError as exc:
                        per_sample_rows.append({
                            "experiment_id": sub_id, "held_out_dataset": held_out_dataset, "fold_id": fold["fold_id"],
                            "status": "TOPOLOGY_INVALID", "error": str(exc), "edge_TP": 0, "edge_FP": 0, "edge_FN": gt_edge_count,
                            "edge_jaccard": 0.0, "precision": 0.0, "recall": 0.0, "runtime_seconds": time.time() - exp_t0,
                        })
                        print(f"  [CRITICAL] {exc}")
                        continue

                    match = match_nodes_by_timepoint(pred_nodes, node_filter_cache["gt_nodes"], NODE_MATCH_MAX_DISTANCE_UM)
                    edge_result = compute_edge_jaccard(pred_edges, node_filter_cache["gt_edges"], match["pred_to_gt"])
                    precision, recall = _precision_recall(edge_result["edge_TP"], edge_result["edge_FP"], edge_result["edge_FN"])
                    n_timepoints = max(node_filter_cache["n_timepoints"], 1)

                    per_sample_rows.append({
                        "experiment_id": sub_id, "held_out_dataset": held_out_dataset, "fold_id": fold["fold_id"], "status": "ok",
                        **edge_result, "precision": precision, "recall": recall,
                        "node_matches": len(match["pred_to_gt"]), "unmatched_gt_nodes": len(match["unmatched_gt_nodes"]),
                        "unmatched_pred_nodes": len(match["unmatched_pred_nodes"]),
                        "n_pred_nodes": len(pred_nodes), "n_pred_edges": len(pred_edges),
                        "avg_edges_per_frame": len(pred_edges) / n_timepoints,
                        "dangling_edge_count": topo["dangling_edge_count"], "duplicate_edge_count": topo["duplicate_edge_count"],
                        "multi_parent_count": topo["multi_parent_count"], "division_like_source_count": topo["division_like_source_count"],
                        "max_out_degree": topo["max_out_degree"], "runtime_seconds": time.time() - exp_t0,
                        **exp_diag,
                    })
                    topology_rows.append({
                        "experiment_id": sub_id, "held_out_dataset": held_out_dataset,
                        "out_degree_distribution": json.dumps(topo["out_degree_distribution"]),
                        "in_degree_distribution": json.dumps(topo["in_degree_distribution"]),
                        "t_gap_distribution": json.dumps(topo["t_gap_distribution"]),
                        "dangling_edge_count": topo["dangling_edge_count"], "duplicate_edge_count": topo["duplicate_edge_count"],
                        "division_like_source_count": topo["division_like_source_count"], "multi_parent_count": topo["multi_parent_count"],
                    })
                    dist_summary = topo["edge_distance_summary"]
                    edge_distance_rows.append({
                        "experiment_id": sub_id, "held_out_dataset": held_out_dataset,
                        "mean": dist_summary.get("mean"), "median": dist_summary.get("median"),
                        "p90": dist_summary.get("p90"), "p95": dist_summary.get("p95"),
                        "p99": dist_summary.get("p99"), "max": dist_summary.get("max"),
                    })

    per_sample_df = pd.DataFrame(per_sample_rows)
    topology_df = pd.DataFrame(topology_rows)
    edge_distance_df = pd.DataFrame(edge_distance_rows)

    ok_rows = per_sample_df[per_sample_df["status"] == "ok"] if len(per_sample_df) else per_sample_df
    experiment_oof_df = aggregate_oof(ok_rows, ["experiment_id"]) if len(ok_rows) else pd.DataFrame()

    return {
        "train_dirs": train_dirs, "folds": folds, "feature_df": feature_df,
        "per_sample_df": per_sample_df, "topology_df": topology_df, "edge_distance_df": edge_distance_df,
        "experiment_oof_df": experiment_oof_df,
    }

# --------------------------------------------------------------------------- #
# 31. Best-candidate selection + decision report
# --------------------------------------------------------------------------- #
def check_harness_reproduction(experiment_oof_df: pd.DataFrame, tolerance_frac: float = A_REPRODUCTION_TOLERANCE_FRAC) -> dict:
    """Experiment A must approximately reproduce the M11/M12 anchor
    numbers - this harness is fully deterministic (same seed, same
    recipe), so a real mismatch beyond `tolerance_frac` relative error
    indicates a code bug in this module, not run-to-run noise. Per the
    user's explicit hard-fail list, this is a HARD gate: if it fails, the
    final recommendation is withheld regardless of what B-I show.
    """
    a_rows = experiment_oof_df[experiment_oof_df["experiment_id"] == "A"] if len(experiment_oof_df) else pd.DataFrame()
    if a_rows.empty:
        return {"ok": False, "reason": "Experiment A produced no aggregate OOF row at all (every fold/sample must have errored or been skipped)."}

    a = a_rows.iloc[0]
    jaccard_rel_error = abs(a["edge_jaccard"] - M11_BEST_EDGE_JACCARD) / M11_BEST_EDGE_JACCARD if M11_BEST_EDGE_JACCARD else float("inf")
    ok = jaccard_rel_error <= tolerance_frac
    return {
        "ok": bool(ok), "a_edge_jaccard": float(a["edge_jaccard"]), "a_edge_TP": int(a["edge_TP"]), "a_edge_FP": int(a["edge_FP"]),
        "a_edge_FN": int(a["edge_FN"]), "a_n_samples": int(a["n_samples"]), "a_n_samples_with_tp": int(a["n_samples_with_tp"]),
        "anchor_edge_jaccard": M11_BEST_EDGE_JACCARD, "jaccard_relative_error": float(jaccard_rel_error), "tolerance_frac": tolerance_frac,
        "reason": (
            "within tolerance" if ok else
            f"edge_jaccard relative error {jaccard_rel_error:.3f} exceeds tolerance {tolerance_frac} - "
            "this indicates a harness bug, not sampling noise (the recipe/seed are fully deterministic)."
        ),
    }


def select_best_candidates(experiment_oof_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Flags every experiment's aggregate-OOF row against Primary (beats
    the M11 anchor jaccard) and Secondary (same/slightly-lower jaccard but
    lower FP and better n_samples_with_tp than M11's 6/12) success
    criteria.
    """
    if experiment_oof_df.empty:
        pd.DataFrame().to_csv(csv_path, index=False)
        return experiment_oof_df.copy()

    df = experiment_oof_df.copy()
    df["meets_primary_success"] = df["edge_jaccard"] > M11_BEST_EDGE_JACCARD
    df["meets_secondary_success"] = (
        (df["edge_jaccard"] >= M11_BEST_EDGE_JACCARD * 0.97)
        & (df["edge_FP"] < M11_EDGE_FP) & (df["n_samples_with_tp"] > M11_N_SAMPLES_WITH_TP)
    )
    best_df = df[df["meets_primary_success"] | df["meets_secondary_success"]].sort_values("edge_jaccard", ascending=False)
    best_df.to_csv(csv_path, index=False)
    return best_df


def determine_final_recommendation(experiment_oof_df: pd.DataFrame, reproduction_check: dict) -> dict:
    """Derives one of the 7 lettered recommendations from the aggregate
    OOF numbers, checked in priority order (best-case combos first). The
    decision rule is stated explicitly in the returned dict so the
    recommendation is auditable.
    """
    options = {
        "A": "linefit only should be ported to M12 submission",
        "B": "motion_relink should be ported",
        "C": "motion_relink + linefit should be ported",
        "D": "gap_close is promising but needs safeguards",
        "E": "safe_divisions is promising but a separate milestone is needed",
        "F": "no graph-repair improvement; seek support artifact for learned-model replication",
        "G": "dense-node policy changes metric behavior but is too risky",
    }

    def jaccard_of(experiment_id: str) -> float | None:
        rows = experiment_oof_df[experiment_oof_df["experiment_id"] == experiment_id]
        return float(rows.iloc[0]["edge_jaccard"]) if len(rows) else None

    def fp_of(experiment_id: str) -> int | None:
        rows = experiment_oof_df[experiment_oof_df["experiment_id"] == experiment_id]
        return int(rows.iloc[0]["edge_FP"]) if len(rows) else None

    a_jaccard = jaccard_of("A")
    b_jaccard, c_jaccard, d_jaccard = jaccard_of("B"), jaccard_of("C"), jaccard_of("D")
    e_jaccard, g_jaccard, h_jaccard = jaccard_of("E"), jaccard_of("G"), jaccard_of("H")

    if not reproduction_check["ok"]:
        return {
            "recommendation": None, "options": options,
            "rationale": (
                "Experiment A failed to reproduce the M11/M12 anchor within tolerance - per the hard-fail "
                "criteria, no port recommendation is issued until this harness mismatch is fixed. "
                f"({reproduction_check['reason']})"
            ),
            "decision_rule": "harness_reproduction_ok == False -> withhold recommendation",
        }

    candidates = []
    if d_jaccard is not None and a_jaccard is not None and d_jaccard > a_jaccard and (c_jaccard is None or d_jaccard >= c_jaccard):
        candidates.append(("C", d_jaccard))
    if c_jaccard is not None and a_jaccard is not None and c_jaccard > a_jaccard:
        candidates.append(("B", c_jaccard))
    if b_jaccard is not None and a_jaccard is not None and b_jaccard > a_jaccard:
        candidates.append(("A", b_jaccard))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        recommendation = candidates[0][0]
        rationale = (
            f"Experiment {'D' if recommendation == 'C' else 'C' if recommendation == 'B' else 'B'} beat the "
            f"Experiment A baseline (edge_jaccard={a_jaccard:.6f}) with edge_jaccard={candidates[0][1]:.6f}."
        )
    elif e_jaccard is not None and a_jaccard is not None and e_jaccard > a_jaccard:
        recommendation = "D"
        rationale = (
            f"Experiment E (gap_close) beat the Experiment A baseline (edge_jaccard={a_jaccard:.6f} -> "
            f"{e_jaccard:.6f}) but gap-closing is structurally the highest-risk idea (can reuse/synthesize "
            "nodes the node filter specifically rejected) - needs additional safeguards before being trusted."
        )
    elif g_jaccard is not None and a_jaccard is not None and g_jaccard > a_jaccard:
        recommendation = "E"
        rationale = (
            f"Experiment G (safe_divisions) beat the Experiment A baseline (edge_jaccard={a_jaccard:.6f} -> "
            f"{g_jaccard:.6f}), a small but real gain given the rarity of true divisions - promising enough "
            "to warrant a dedicated follow-up milestone rather than folding it in immediately."
        )
    elif h_jaccard is not None and a_jaccard is not None and abs(h_jaccard - a_jaccard) > 1e-9:
        recommendation = "G"
        rationale = (
            f"Experiment H (dense-node policy) meaningfully changes metric behavior (edge_jaccard "
            f"{a_jaccard:.6f} -> {h_jaccard:.6f}) but without ground truth on the real grader's node-vs-edge "
            "weighting, it is too risky to commit to as a default policy change."
        )
    else:
        recommendation = "F"
        rationale = (
            "None of the graph-repair experiments (B-H) meaningfully beat the Experiment A baseline "
            f"(edge_jaccard={a_jaccard}) in this run - no graph-repair improvement found; consider seeking "
            "the reference solution's support artifact for a learned-model replication instead."
        )

    return {
        "recommendation": recommendation, "options": options, "rationale": rationale,
        "decision_rule": "checked in order: D-vs-C-vs-B best-of (C/B/A letters) -> E (D) -> G (E) -> H (G) -> else F",
    }

# --------------------------------------------------------------------------- #
# 32. Full pipeline - orchestration + outputs + final recommendation
# --------------------------------------------------------------------------- #
def run_milestone14_validation_and_report(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS, working_dir: str = "/kaggle/working",
) -> dict:
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Milestone 14: M12 + graph-repair hybrid OOF validation ===")
    print("(OOF VALIDATION ONLY - no submission.csv is generated, no Kaggle API is called, no Save Version is triggered.)")

    results = run_milestone14_experiments(n_robust_samples=n_robust_samples, seed=seed, time_budget_seconds=time_budget_seconds)

    results["per_sample_df"].to_csv(out_dir / "milestone14_per_sample_eval.csv", index=False)
    results["topology_df"].to_csv(out_dir / "milestone14_topology_stats.csv", index=False)
    results["edge_distance_df"].to_csv(out_dir / "milestone14_edge_distance_stats.csv", index=False)
    results["experiment_oof_df"].to_csv(out_dir / "milestone14_experiment_eval.csv", index=False)

    print("\n=== Aggregate OOF per experiment ===")
    if len(results["experiment_oof_df"]):
        for _, row in results["experiment_oof_df"].sort_values("edge_jaccard", ascending=False).iterrows():
            print(
                f"  {row['experiment_id']}: jaccard={row['edge_jaccard']:.6f} TP={row['edge_TP']} FP={row['edge_FP']} "
                f"FN={row['edge_FN']} n_samples_with_tp={row['n_samples_with_tp']}/{row['n_samples']}"
            )

    reproduction_check = check_harness_reproduction(results["experiment_oof_df"])
    print("\n=== Harness reproduction check (Experiment A vs M11/M12 anchor) ===")
    print(f"  ok={reproduction_check['ok']}: {reproduction_check['reason']}")

    best_candidates_df = select_best_candidates(results["experiment_oof_df"], out_dir / "milestone14_best_candidates.csv")
    print(f"\nBest candidates (meeting primary or secondary success): {len(best_candidates_df)}")

    decision = determine_final_recommendation(results["experiment_oof_df"], reproduction_check)

    decision_report = {
        "harness_reproduction_check": reproduction_check,
        "anchor": {
            "m11_best_edge_jaccard": M11_BEST_EDGE_JACCARD, "m11_edge_TP": M11_EDGE_TP, "m11_edge_FP": M11_EDGE_FP,
            "m11_edge_FN": M11_EDGE_FN, "m11_n_samples": M11_N_SAMPLES, "m11_n_samples_with_tp": M11_N_SAMPLES_WITH_TP,
            "baseline_6c6d_edge_jaccard": BASELINE_6C6D_EDGE_JACCARD,
        },
        "experiment_summary": results["experiment_oof_df"].to_dict(orient="records") if len(results["experiment_oof_df"]) else [],
        "n_best_candidates": len(best_candidates_df),
        "final_recommendation": decision["recommendation"], "rationale": decision["rationale"],
        "decision_rule": decision["decision_rule"], "options": decision["options"],
    }
    with open(out_dir / "milestone14_decision_report.json", "w") as f:
        json.dump(decision_report, f, indent=2, default=str)

    next_step_lines = [f"Milestone 14 recommendation: {decision['recommendation']}", "", decision["rationale"], ""]
    if decision["recommendation"] is not None:
        next_step_lines.append(f"Full description: {decision['options'][decision['recommendation']]}")
    else:
        next_step_lines.append("Fix the Experiment A harness-reproduction mismatch before proceeding to any further milestone.")
    with open(out_dir / "milestone14_next_step_prompt.txt", "w") as f:
        f.write("\n".join(next_step_lines))

    print("\n=== Final recommendation ===")
    for key, label in decision["options"].items():
        marker = "  <-- RECOMMENDED" if key == decision["recommendation"] else ""
        print(f"  {key}) {label}{marker}")
    if decision["recommendation"] is None:
        print("  ** NO RECOMMENDATION ISSUED - harness reproduction check FAILED **")
    print(f"\nDecision rule: {decision['decision_rule']}")
    print(f"Rationale: {decision['rationale']}")

    print("\nNo submission.csv was generated by Milestone 14.")
    print("Do not Save Version / Submit from this validation notebook.")

    return {**results, "reproduction_check": reproduction_check, "best_candidates_df": best_candidates_df, "decision": decision}

# --------------------------------------------------------------------------- #
# 33. Tests
# --------------------------------------------------------------------------- #
def run_milestone14_tests() -> None:
    """Correctness tests for every new Milestone 14 mechanism: line-fit
    smoothing (correctness + topology preservation), motion-relink
    Hungarian assignment (basic correctness, tight/relaxed gating,
    no-multi-parent/no-duplicate-edge), gap-closing, safe-division
    geometry gates, and a synthetic end-to-end fixture - all on small
    in-memory fixtures (no disk I/O, no real train/test samples).
    """
    # Test 1: line-fit smoothing moves a jittered interior node toward its
    # window-averaged neighbors, blended by `weight`.
    nodes1 = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4], "t": [0, 1, 2, 3, 4],
        "z": [0.0] * 5, "y": [0.0, 1.0, 2.1, 3.0, 4.0], "x": [0.0] * 5,
    })
    edges1 = pd.DataFrame({"source_id": [0, 1, 2, 3], "target_id": [1, 2, 3, 4]})
    smoothed1 = linefit_smooth_nodes(nodes1, edges1, window=2, weight=0.8)
    neighbor_avg_y = np.mean([3.0, 4.0, 1.0, 0.0])  # forward (3,4) + backward (1,0)
    expected_y2 = 0.8 * 2.1 + 0.2 * neighbor_avg_y
    smoothed_y2 = float(smoothed1.loc[smoothed1["node_id"] == 2, "y"].iloc[0])
    assert abs(smoothed_y2 - expected_y2) < 1e-9, f"expected {expected_y2}, got {smoothed_y2}"
    # endpoints (out_degree or in_degree == 0) must be untouched
    assert float(smoothed1.loc[smoothed1["node_id"] == 0, "y"].iloc[0]) == 0.0
    assert float(smoothed1.loc[smoothed1["node_id"] == 4, "y"].iloc[0]) == 4.0

    # Test 2: line-fit smoothing NEVER changes topology - same node_ids,
    # same edge set, only (z,y,x) may differ.
    assert set(smoothed1["node_id"]) == set(nodes1["node_id"])
    assert len(smoothed1) == len(nodes1)

    # Test 3: motion-relink basic correctness - 2 sources at t=0 competing
    # for 2 targets at t=1, with a clear best pairing and one source with
    # no acceptable candidate at all (must be REJECTED, not force-matched).
    scored3 = pd.DataFrame({
        "source_id": [0, 0, 1, 1, 2], "target_id": [10, 11, 10, 11, 12], "t": [0, 0, 0, 0, 0],
        "dz_um": [0.0] * 5, "dy_um": [1.0, 5.0, 5.0, 1.0, 50.0], "dx_um": [0.0] * 5,
        "distance_um": [1.0, 5.0, 5.0, 1.0, 50.0], "_score": [0.9, 0.5, 0.5, 0.9, 0.1],
    })
    selected3, diag3 = select_edges_motion_relink(scored3, tight_gate=6.0, relaxed_gate=10.0)
    accepted_pairs3 = set(zip(selected3["source_id"], selected3["target_id"]))
    assert (0, 10) in accepted_pairs3, "source 0's cheapest option (target 10, dist=1.0) should be accepted"
    assert (1, 11) in accepted_pairs3, "source 1's cheapest option (target 11, dist=1.0) should be accepted"
    assert 2 not in selected3["source_id"].values, "source 2's only candidate (dist=50um) exceeds even the relaxed gate - must be rejected, not force-matched"
    assert diag3["n_total"] == 2

    # Test 4: tight-vs-relaxed pass behavior - a candidate whose motion
    # distance is inside (tight_gate, relaxed_gate] must be accepted ONLY
    # in the relaxed pass, tagged accordingly.
    scored4 = pd.DataFrame({
        "source_id": [0], "target_id": [20], "t": [0],
        "dz_um": [0.0], "dy_um": [8.0], "dx_um": [0.0],
        "distance_um": [8.0], "_score": [0.7],
    })
    selected4, diag4 = select_edges_motion_relink(scored4, tight_gate=6.0, relaxed_gate=10.0)
    assert diag4["n_tight"] == 0 and diag4["n_relaxed"] == 1
    assert selected4.iloc[0]["assignment_pass"] == "relaxed"

    # Test 5: no multi-parent / no duplicate directed edges across the
    # FULL tight+relaxed combined output on a busier synthetic scenario.
    scored5 = pd.DataFrame({
        "source_id": [0, 1, 2, 0, 1], "target_id": [10, 10, 11, 11, 12], "t": [0, 0, 0, 0, 0],
        "dz_um": [0.0] * 5, "dy_um": [1.0, 1.2, 1.0, 5.0, 5.0], "dx_um": [0.0] * 5,
        "distance_um": [1.0, 1.2, 1.0, 5.0, 5.0], "_score": [0.9, 0.85, 0.8, 0.5, 0.5],
    })
    selected5, _ = select_edges_motion_relink(scored5, tight_gate=6.0, relaxed_gate=10.0)
    assert selected5["source_id"].is_unique, "no source should be assigned twice"
    assert selected5["target_id"].is_unique, "no target should be assigned twice (no multi-parent)"

    # Test 6: gap-closing reuses a real, currently-unused node near the
    # implied t+1 midpoint to bridge a tracklet end (t=0) to another
    # tracklet's start (t=2).
    pred_nodes6 = pd.DataFrame({
        "node_id": [0, 1], "t": [0, 2], "z": [0.0, 0.0], "y": [0.0, 4.0], "x": [0.0, 0.0],
    })
    pred_edges6 = pd.DataFrame(columns=["source_id", "target_id"])  # both nodes are isolated (end/start of nothing yet)
    filtered_nodes6 = pd.DataFrame({
        "node_id": [0, 1, 100], "t": [0, 2, 1], "z": [0.0, 0.0, 0.0], "y": [0.0, 4.0, 2.0], "x": [0.0, 0.0, 0.0], "score": [0.9, 0.9, 0.6],
    })
    new_nodes6, new_edges6, gap_diag6 = gap_close(
        pred_nodes6, pred_edges6, filtered_nodes6, max_total_distance=12.0, max_per_step_distance=6.0, max_added_frac=1.0,
    )
    assert gap_diag6["n_gaps_closed"] == 1 and gap_diag6["n_reused_real_node"] == 1
    assert 100 in set(new_nodes6["node_id"]), "the real unused node at t=1 should be reused, not synthesized"
    assert (0, 100) in set(zip(new_edges6["source_id"], new_edges6["target_id"]))
    assert (100, 1) in set(zip(new_edges6["source_id"], new_edges6["target_id"]))

    # Test 7: safe-division geometry gates - a source with one existing
    # (close) child; a second candidate within all gates is admitted, a
    # second candidate that violates the sister-distance gate is rejected.
    pred_nodes7 = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 1, 1], "z": [0.0] * 4, "y": [0.0, 1.0, 1.5, 20.0], "x": [0.0] * 4,
    })
    pred_edges7 = pd.DataFrame({"source_id": [0], "target_id": [1], "distance_um": [physical_distance_um([0, 0, 0], [0, 1.0, 0])]})
    filtered_nodes7 = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 1, 1], "z": [0.0] * 4, "y": [0.0, 1.0, 1.5, 20.0], "x": [0.0] * 4, "score": [0.9] * 4,
    })
    scored7 = pd.DataFrame({
        "source_id": [0, 0], "target_id": [2, 3], "t": [0, 0],
        "distance_um": [physical_distance_um([0, 0, 0], [0, 1.5, 0]), physical_distance_um([0, 0, 0], [0, 20.0, 0])],
        "_score": [0.8, 0.8],
    })
    new_nodes7, new_edges7, div_diag7 = add_safe_divisions(
        pred_nodes7, pred_edges7, scored7, filtered_nodes7,
        parent_child_max_um=4.7, sister_max_um=7.2, existing_child_max_um=7.8, frame_cap_frac=1.0, global_cap_frac=1.0,
    )
    accepted_pairs7 = set(zip(new_edges7["source_id"], new_edges7["target_id"]))
    assert (0, 2) in accepted_pairs7, "node 2 is close to both parent and sister - should be admitted as a safe division"
    assert (0, 3) not in accepted_pairs7, "node 3 is far from the sister (violates sister_max_um) - must be rejected"
    assert div_diag7["n_divisions_added"] == 1

    # Test 8: synthetic end-to-end fixture - a cleanly separable true
    # 5-node track (matching BASE_TRACKLET_FILTER's min_tracklet_length=5)
    # run through the full Experiment A dispatch (greedy_exclusive ->
    # tracklets -> filter -> reconstruct) must produce a topology-valid,
    # non-empty result.
    real_chain_rows8 = [
        {"source_id": s, "target_id": s + 1, "t": s, "dz_um": 0.0, "dy_um": 1.0, "dx_um": 0.0, "distance_um": 0.4, "_score": score}
        for s, score in zip([0, 1, 2, 3], [0.95, 0.93, 0.91, 0.90])
    ]
    # Pad with low-score noise candidates (distinct fake source ids, never
    # present in filtered_nodes8) so the real chain's 0.90-0.95 scores sit
    # comfortably above the 90th-percentile quantile gate `select_greedy_
    # exclusive_per_frame` applies - mirroring a real candidate pool where
    # true edges are a small minority. The noise rows are filtered out by
    # either the quantile gate or losing the greedy-exclusivity contest for
    # the same target, so they never need a corresponding real node.
    # Noise targets cycle through 1,2,3,4 (each already claimed by a real,
    # much-higher-scoring edge) - NEVER target 0, which no real edge
    # claims, so a stray noise edge can never win the greedy-exclusivity
    # contest and end up referencing a fake source node downstream.
    noise_rows8 = [
        {"source_id": 1000 + i, "target_id": 1 + (i % 4), "t": (i % 4), "dz_um": 0.0, "dy_um": -99.0, "dx_um": 0.0, "distance_um": 99.0, "_score": 0.05}
        for i in range(60)
    ]
    scored8 = pd.DataFrame(real_chain_rows8 + noise_rows8)
    filtered_nodes8 = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4], "t": [0, 1, 2, 3, 4],
        "z": [0.0] * 5, "y": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [0.0] * 5, "score": [0.9] * 5,
    })
    results8 = run_experiment("A", scored8, None, filtered_nodes8)
    assert len(results8) == 1
    sub_id8, pred_nodes8, pred_edges8, _ = results8[0]
    assert sub_id8 == "A"
    topo8 = compute_topology_stats(pred_nodes8, pred_edges8, allow_divisions=False)
    assert topo8["topology_valid"]
    assert len(pred_nodes8) > 0

    print("All milestone14_graph_repair_hybrid tests passed (8/8).")

# --------------------------------------------------------------------------- #
# 34. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone14_pipeline(
    n_robust_samples: int = ROBUST_MODE_N_SAMPLES_DEFAULT, seed: int = 0,
    time_budget_seconds: float = ROBUST_MODE_TIME_BUDGET_SECONDS,
) -> dict:
    """Runs the unit tests, then the full Milestone 14 graph-repair-hybrid
    OOF validation (9 experiments, evaluated via GroupKFold on the same 12
    fixed-seed robust train samples Milestone 11 used - never test data).
    No submission.csv is ever generated, no Kaggle API is called, no Save
    Version is triggered.
    """
    print(
        "=== Self-test: linefit smoothing, motion-relink Hungarian assignment, gap-closing, "
        "safe-division gates, and synthetic end-to-end unit tests ==="
    )
    run_milestone14_tests()
    return run_milestone14_validation_and_report(n_robust_samples=n_robust_samples, seed=seed, time_budget_seconds=time_budget_seconds)


if __name__ == "__main__":
    run_milestone14_pipeline()
