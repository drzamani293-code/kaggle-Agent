"""
Biohub - Cell Tracking During Development
Milestone 5C: centroid-aware candidate generator.

The boundary-aware comparison (boundary_detector.py) found that boundary
proximity is NOT the main bottleneck: D (boundary-aware local maxima) scored
identically to A (raw_candidate_coverage=0.827586, cap_coverage=0.741379),
and E (dense intensity ranking) performed much worse
(raw_candidate_coverage=0.385057) once actually run on real data - dense
un-clustered voxel sampling drowns the signal rather than recovering it.

The real finding: many lost GT nodes have a local intensity maximum only
9-13 um away from the annotated centroid - a peak DOES exist nearby, it is
just not centroid-aligned. This points to a systematic peak-vs-centroid
offset (e.g. an elongated/asymmetric cell whose brightest voxel sits off
to one side of its true geometric/annotated center), not a missing-signal
problem. Strict local-maximum detection (A/D) reports the single brightest
voxel; it has no mechanism to "average out" to the object's true centroid.

This module targets that specific failure mode with 4 new methods:
  - G_component_centroid: threshold + connected-component labeling, then
    emit the component's intensity-weighted centroid, geometric centroid,
    brightest voxel, and a locally-weighted center-of-mass as separate
    candidates - centroids computed over the WHOLE detected blob rather
    than a single voxel, so an elongated/asymmetric object's centroid
    candidate is pulled toward its true center of mass.
  - H_peak_to_centroid_refinement: starts from A's local maxima, then
    computes an intensity-weighted center of mass in a 7-12um patch around
    each peak - directly corrects the peak-vs-centroid offset without
    discarding the original (well-calibrated in most cases) peak detection.
  - I_multi_threshold_component_union: repeats G's per-threshold component
    extraction at 6 thresholds (0.03-0.20), unions the resulting weighted
    centroids, and deduplicates via physical NMS - tests whether varying
    the threshold recovers components missed at any single operating point.
  - J_oracle_diagnostic_only: NOT an inference method - reports, purely as
    an upper-bound diagnostic, what fraction of GT nodes have *some*
    component-centroid candidate within 7um across all 6 thresholds. Never
    used to generate predictions; exists only to bound how much headroom
    component-based methods have before even running the full pipeline.

No tracking, no final submission - candidate generation diagnostics only.
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
# 4. Train-sample discovery
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
    """Recursively locate every *.zarr folder belonging to the given split."""
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


# --------------------------------------------------------------------------- #
# 5. Shared detector primitives (normalize, NMS, matching, coverage, patch stats)
# --------------------------------------------------------------------------- #
FIXED_GAUSSIAN_SIGMA_Z = 1.0
FIXED_GAUSSIAN_SIGMA_YX = 1.5

MAX_RAW_CANDIDATES_BEFORE_NMS = 30000


def normalize_intensity(image: np.ndarray, percentile_low: float, percentile_high: float) -> np.ndarray:
    """Percentile-clip + rescale an image to [0, 1] float32."""
    lo, hi = np.percentile(image, [percentile_low, percentile_high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def _nms_by_physical_distance(candidates_df: pd.DataFrame, min_distance_um: float) -> pd.DataFrame:
    """Greedy non-maximum suppression via KD-tree: candidates must already
    be sorted by score descending.

    Uses `cKDTree.query_pairs` (one bulk C call returning every close pair
    at once) rather than a per-point `query_ball_point` loop - 3-8x faster
    with identical output (see gt_failure_analysis.py for the profiling
    that motivated this rewrite).
    """
    if len(candidates_df) <= 1:
        return candidates_df

    scaled_coords = candidates_df[["z", "y", "x"]].to_numpy(dtype=float) * _VOXEL_SCALE
    tree = cKDTree(scaled_coords)
    n = len(candidates_df)
    keep_mask = np.ones(n, dtype=bool)

    pairs = tree.query_pairs(r=min_distance_um, output_type="ndarray")
    if len(pairs):
        # pairs are (i, j) with i < j (index order == score-descending order).
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


def _match_single_timepoint(
    pred_coords_zyx: np.ndarray, gt_coords_zyx: np.ndarray, max_distance_um: float
) -> list[float]:
    """Hungarian-match one timepoint's predicted vs GT (z,y,x) coordinates;
    returns the list of accepted (<= max_distance_um) match distances.
    """
    if len(pred_coords_zyx) == 0 or len(gt_coords_zyx) == 0:
        return []
    cost = _pairwise_physical_distance_um(pred_coords_zyx, gt_coords_zyx)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [float(cost[r, c]) for r, c in zip(row_ind, col_ind) if cost[r, c] <= max_distance_um]


def physical_distance_um(p1: Sequence[float], p2: Sequence[float]) -> float:
    diff = (np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)) * _VOXEL_SCALE
    return float(np.sqrt((diff ** 2).sum()))


def _gt_coverage_mask(gt_coords_zyx: np.ndarray, candidates_df: pd.DataFrame, max_distance_um: float) -> np.ndarray:
    """Boolean array (one per GT node): is there >= 1 candidate within
    `max_distance_um` physical distance, regardless of 1-1 assignment?
    """
    if len(gt_coords_zyx) == 0:
        return np.zeros(0, dtype=bool)
    if len(candidates_df) == 0:
        return np.zeros(len(gt_coords_zyx), dtype=bool)
    cand_scaled = candidates_df[["z", "y", "x"]].to_numpy(dtype=float) * _VOXEL_SCALE
    gt_scaled = gt_coords_zyx * _VOXEL_SCALE
    tree = cKDTree(cand_scaled)
    dists, _ = tree.query(gt_scaled, k=1)
    return np.atleast_1d(dists) <= max_distance_um


def _nearest_candidate_distance_um(gt_coord_zyx: np.ndarray, candidates_df: pd.DataFrame) -> float:
    """Physical distance from a single GT coordinate to its nearest candidate; NaN if none."""
    if len(candidates_df) == 0:
        return float("nan")
    cand_scaled = candidates_df[["z", "y", "x"]].to_numpy(dtype=float) * _VOXEL_SCALE
    gt_scaled = gt_coord_zyx * _VOXEL_SCALE
    tree = cKDTree(cand_scaled)
    dist, _ = tree.query(gt_scaled, k=1)
    return float(dist)


def _collect_gt_cases(sample_dirs: Sequence[Path]) -> list[tuple[str, Path, int, pd.DataFrame]]:
    """One (sample_name, sample_dir, t, gt_nodes_t) entry per timepoint that
    has at least one GT node, across the given samples.
    """
    cases = []
    for sample_dir in sample_dirs:
        gt_nodes, _ = read_geff(sample_dir)
        if len(gt_nodes) == 0:
            continue
        for t in sorted(gt_nodes["t"].unique()):
            gt_t = gt_nodes[gt_nodes["t"] == t].reset_index(drop=True)
            cases.append((sample_dir.stem, sample_dir, int(t), gt_t))
    return cases


def _boundary_distances_um(z: float, y: float, x: float, shape_zyx: tuple[int, int, int]) -> dict:
    """Physical distance (um) from a coordinate to each of the 6 volume faces,
    plus the overall minimum (how close is this point to *any* boundary).
    """
    Z, Y, X = shape_zyx
    dz0, dz1 = z * VOXEL_SIZE_UM["z"], (Z - 1 - z) * VOXEL_SIZE_UM["z"]
    dy0, dy1 = y * VOXEL_SIZE_UM["y"], (Y - 1 - y) * VOXEL_SIZE_UM["y"]
    dx0, dx1 = x * VOXEL_SIZE_UM["x"], (X - 1 - x) * VOXEL_SIZE_UM["x"]
    return {
        "dist_to_z0_um": dz0, "dist_to_zmax_um": dz1,
        "dist_to_y0_um": dy0, "dist_to_ymax_um": dy1,
        "dist_to_x0_um": dx0, "dist_to_xmax_um": dx1,
        "min_boundary_distance_um": min(dz0, dz1, dy0, dy1, dx0, dx1),
    }


# --------------------------------------------------------------------------- #
# 6. Baseline candidate methods (A: local maxima, D: boundary-aware local
#    maxima) - carried over unchanged from boundary_detector.py for comparison.
# --------------------------------------------------------------------------- #
def _normalized_smoothed(image_zyx: np.ndarray, percentile_low: float, percentile_high: float) -> np.ndarray:
    """Shared preprocessing for A/G/H/I (percentile normalize + 3D Gaussian
    smooth, mode='reflect') - factored out so callers that need multiple of
    these methods on the same image (the full A/D/G/H/I comparison, the
    lost-node recovery analysis) can compute it once and pass it in via
    each method's optional `smoothed` parameter, instead of redundantly
    re-normalizing and re-smoothing the same volume once per method.
    """
    normalized = normalize_intensity(image_zyx, percentile_low, percentile_high)
    return gaussian_filter(normalized, sigma=(FIXED_GAUSSIAN_SIGMA_Z, FIXED_GAUSSIAN_SIGMA_YX, FIXED_GAUSSIAN_SIGMA_YX), mode="reflect")


def method_a_raw_candidates(
    image_zyx: np.ndarray, percentile_low: float, percentile_high: float, min_threshold: float,
    smoothed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Milestone 5/5B baseline: percentile normalize + 3D Gaussian smooth
    (mode='reflect') + peak_local_max with exclude_border=False. Pass a
    precomputed `smoothed` array (see `_normalized_smoothed`) to skip
    redundant preprocessing when calling multiple methods on one image.
    """
    if smoothed is None:
        smoothed = _normalized_smoothed(image_zyx, percentile_low, percentile_high)

    coords = peak_local_max(smoothed, min_distance=1, threshold_abs=min_threshold, exclude_border=False)
    if len(coords) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    scores = smoothed[tuple(coords.T)]
    candidates = pd.DataFrame({
        "z": coords[:, 0].astype(float), "y": coords[:, 1].astype(float), "x": coords[:, 2].astype(float),
        "score": scores.astype(float),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if len(candidates) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        candidates = candidates.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]
    return candidates


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
# 7. Component-centroid primitives + methods G, H, I
# --------------------------------------------------------------------------- #
# Ignore components smaller than this many voxels - at low thresholds (0.03),
# noise specks produce huge numbers of 1-2 voxel "components" that would
# otherwise flood the candidate pool without representing real signal.
MIN_COMPONENT_VOXELS = 3

# A real single cell in this dataset's voxel geometry (z=1.625um, y=x=0.40625um)
# occupies at most ~30,000 voxels even for a generous 25um-diameter cell.
# CRITICAL finding during testing: at the module's low threshold floor
# (0.03), percentile-based normalization can make background noise itself
# span nearly the full [0,1] range whenever background occupies the vast
# majority of the volume (which it does - real cells are sparse) - so a
# naive threshold-and-label pass can merge almost the ENTIRE volume into
# ONE giant "component" whose whole-component centroid is meaningless
# (dragged toward the bulk of the background, not any real cell). This
# mirrors exactly why E (dense intensity ranking, no clustering) scored
# far worse than expected on real data in the boundary-aware milestone
# (raw_candidate_coverage=0.385057) - un-clustered/over-merged low-threshold
# signal drowns real cells rather than finding them. Any component larger
# than `MAX_COMPONENT_VOXELS` is almost certainly such a background merge,
# not a single cell: for those, only the single `brightest_voxel` (always
# a locally meaningful point regardless of how large the surrounding
# thresholded region is) is kept - the whole-component averaging variants
# (weighted_centroid, geometric_centroid, local_center_of_mass) are skipped
# since they'd be dragged toward the background bulk.
MAX_COMPONENT_VOXELS = 50000

G_DEFAULT_THRESHOLD = 0.05
I_DEFAULT_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
H_REFINEMENT_RADIUS_UM = 10.0  # within the requested 7-12um range

CENTROID_CANDIDATE_TYPES = ("weighted_centroid", "geometric_centroid", "brightest_voxel", "local_center_of_mass")


def _connected_component_candidates(
    smoothed: np.ndarray, threshold: float,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
    max_component_voxels: int = MAX_COMPONENT_VOXELS,
) -> pd.DataFrame:
    """Threshold `smoothed` at `threshold`, connected-component label the
    result, and for each surviving component emit up to 4 centroid variants:
    intensity-weighted centroid, geometric (unweighted) centroid, the single
    brightest voxel, and a locally-weighted center-of-mass restricted to the
    component's top half of intensity (a middle ground between the whole-
    component weighted centroid and the single brightest voxel). All share
    the same score (the component's max intensity) and diagnostic columns
    (mean_intensity, volume, compactness = volume / bounding-box volume).

    Components larger than `max_component_voxels` (implausible for a single
    real cell - see the module-level comment above `MAX_COMPONENT_VOXELS`)
    only emit `brightest_voxel`, since the other 3 variants would average
    over background rather than any real, localized structure.
    """
    empty = pd.DataFrame(columns=["z", "y", "x", "score", "candidate_type", "mean_intensity", "volume", "compactness"])
    mask = smoothed >= threshold
    labeled, n = ndi.label(mask)
    if n == 0:
        return empty

    objects = ndi.find_objects(labeled)
    rows = []
    for comp_id in range(1, n + 1):
        sl = objects[comp_id - 1]
        if sl is None:
            continue
        sub = smoothed[sl]
        sub_mask = labeled[sl] == comp_id
        volume = int(sub_mask.sum())
        if volume < min_component_voxels:
            continue

        coords_local = np.argwhere(sub_mask).astype(float)
        offset = np.array([s.start for s in sl], dtype=float)
        intensities = sub[sub_mask]
        max_intensity = float(intensities.max())
        mean_intensity = float(intensities.mean())
        bbox_volume = int(np.prod([s.stop - s.start for s in sl]))
        compactness = volume / bbox_volume if bbox_volume > 0 else 0.0

        bright_local = coords_local[np.argmax(intensities)] + offset

        if volume > max_component_voxels:
            candidate_variants = (("brightest_voxel", bright_local),)
        else:
            geometric = coords_local.mean(axis=0) + offset
            weighted = (coords_local * intensities[:, None]).sum(axis=0) / intensities.sum() + offset
            median_intensity = float(np.median(intensities))
            top_mask = intensities >= median_intensity
            top_coords = coords_local[top_mask]
            top_weights = intensities[top_mask]
            local_com = (top_coords * top_weights[:, None]).sum(axis=0) / top_weights.sum() + offset
            candidate_variants = (
                ("weighted_centroid", weighted),
                ("geometric_centroid", geometric),
                ("brightest_voxel", bright_local),
                ("local_center_of_mass", local_com),
            )

        for ctype, coord in candidate_variants:
            rows.append({
                "z": float(coord[0]), "y": float(coord[1]), "x": float(coord[2]), "score": max_intensity,
                "candidate_type": ctype, "mean_intensity": mean_intensity, "volume": volume, "compactness": compactness,
            })

    if not rows:
        return empty
    return pd.DataFrame(rows)


def method_g_component_centroid(
    image_zyx: np.ndarray, percentile_low: float, percentile_high: float, threshold: float = G_DEFAULT_THRESHOLD,
    smoothed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Connected-component centroid extraction at a single, fixed low
    threshold: emits centroid candidates (4 variants per component) instead
    of only strict local maxima, so an elongated/asymmetric object's
    candidate position is pulled toward its true center of mass rather than
    its single brightest voxel. Pass a precomputed `smoothed` array (see
    `_normalized_smoothed`) to skip redundant preprocessing.
    """
    if smoothed is None:
        smoothed = _normalized_smoothed(image_zyx, percentile_low, percentile_high)

    candidates = _connected_component_candidates(smoothed, threshold)
    if len(candidates) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    if len(candidates) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        candidates = candidates.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]
    return candidates


def method_i_multi_threshold_component_union(
    image_zyx: np.ndarray,
    percentile_low: float,
    percentile_high: float,
    min_distance_um: float,
    thresholds: Sequence[float] = I_DEFAULT_THRESHOLDS,
    smoothed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Repeats G's connected-component extraction at 6 thresholds
    (0.03-0.20), keeping only each component's intensity-weighted centroid
    (the single most physically meaningful candidate per component - using
    all 4 variants at 6 thresholds would explode the candidate count),
    unions the results across thresholds, and deduplicates with physical
    NMS - testing whether varying the threshold recovers components missed
    at any single operating point (e.g. a dim cell only forms its own
    component at a lower threshold, before merging with a bright neighbor
    at a higher one). Pass a precomputed `smoothed` array (see
    `_normalized_smoothed`) to skip redundant preprocessing.
    """
    if smoothed is None:
        smoothed = _normalized_smoothed(image_zyx, percentile_low, percentile_high)

    per_threshold = []
    for threshold in thresholds:
        comp_df = _connected_component_candidates(smoothed, threshold)
        if len(comp_df):
            per_threshold.append(comp_df[comp_df["candidate_type"] == "weighted_centroid"])

    if not per_threshold:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    union = pd.concat(per_threshold, ignore_index=True)
    union = union.sort_values("score", ascending=False).reset_index(drop=True)
    if len(union) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        union = union.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]
    return _nms_by_physical_distance(union, min_distance_um).sort_values("score", ascending=False).reset_index(drop=True)


H_MAX_PEAKS_TO_REFINE = 2000


def method_h_peak_to_centroid_refinement(
    image_zyx: np.ndarray,
    percentile_low: float,
    percentile_high: float,
    min_threshold: float,
    refinement_radius_um: float = H_REFINEMENT_RADIUS_UM,
    smoothed: np.ndarray | None = None,
    max_peaks_to_refine: int = H_MAX_PEAKS_TO_REFINE,
) -> pd.DataFrame:
    """Starts from A's local maxima; around each peak, takes a
    `refinement_radius_um` patch and computes the intensity-weighted center
    of mass over the *entire* patch (not just component-thresholded voxels)
    - directly corrects a peak that sits off to one side of the object's
    true intensity-weighted center. Emits BOTH the original peak and the
    refined centroid as separate candidates (candidate_type column), so
    downstream matching can use whichever is closer without discarding the
    original, already-reasonable detection. Pass a precomputed `smoothed`
    array (see `_normalized_smoothed`) to skip redundant preprocessing.

    At this module's permissive reference threshold, A can produce tens of
    thousands of raw peaks per volume (mostly low-score noise); refining
    every one of them is both pointless (low-score peaks are dropped by the
    downstream threshold/NMS/cap stages anyway) and expensive (each
    refinement scans a ~15x51x51-voxel patch). Only the top
    `max_peaks_to_refine` peaks by score are refined - the rest pass
    through unrefined as "original_peak" candidates, still available to
    the coverage/matching pipeline.
    """
    if smoothed is None:
        smoothed = _normalized_smoothed(image_zyx, percentile_low, percentile_high)

    peaks = method_a_raw_candidates(image_zyx, percentile_low, percentile_high, min_threshold, smoothed=smoothed)
    if len(peaks) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    peaks_to_refine = peaks.iloc[:max_peaks_to_refine]
    peaks_passthrough = peaks.iloc[max_peaks_to_refine:]

    Z, Y, X = image_zyx.shape
    rz = int(np.ceil(refinement_radius_um / VOXEL_SIZE_UM["z"]))
    ry = int(np.ceil(refinement_radius_um / VOXEL_SIZE_UM["y"]))
    rx = int(np.ceil(refinement_radius_um / VOXEL_SIZE_UM["x"]))

    # Precompute the (2rz+1)x(2ry+1)x(2rx+1) local index grid once - reused
    # for every interior peak (the common case) instead of recomputing
    # np.unravel_index per peak; only peaks whose patch is clipped by the
    # volume boundary fall back to a per-peak grid.
    full_shape = (2 * rz + 1, 2 * ry + 1, 2 * rx + 1)
    full_idx_coords = np.array(np.meshgrid(*(np.arange(s) for s in full_shape), indexing="ij"), dtype=float).reshape(3, -1).T

    rows = []
    for prow in peaks_to_refine.itertuples():
        rows.append({"z": float(prow.z), "y": float(prow.y), "x": float(prow.x), "score": float(prow.score), "candidate_type": "original_peak"})

        zc, yc, xc = int(round(prow.z)), int(round(prow.y)), int(round(prow.x))
        z0, z1 = max(0, zc - rz), min(Z, zc + rz + 1)
        y0, y1 = max(0, yc - ry), min(Y, yc + ry + 1)
        x0, x1 = max(0, xc - rx), min(X, xc + rx + 1)
        patch = smoothed[z0:z1, y0:y1, x0:x1]
        if patch.size == 0 or patch.sum() <= 0:
            continue

        offset = np.array([z0, y0, x0], dtype=float)
        if patch.shape == full_shape:
            idx_coords = full_idx_coords
        else:
            idx_coords = np.array(np.unravel_index(np.arange(patch.size), patch.shape), dtype=float).T
        weights = patch.ravel().astype(float)
        com = (idx_coords * weights[:, None]).sum(axis=0) / weights.sum() + offset

        com_zi, com_yi, com_xi = int(round(com[0])), int(round(com[1])), int(round(com[2]))
        if 0 <= com_zi < Z and 0 <= com_yi < Y and 0 <= com_xi < X:
            refined_score = float(smoothed[com_zi, com_yi, com_xi])
        else:
            refined_score = float(prow.score)

        rows.append({"z": float(com[0]), "y": float(com[1]), "x": float(com[2]), "score": refined_score, "candidate_type": "refined_centroid"})

    for prow in peaks_passthrough.itertuples():
        rows.append({"z": float(prow.z), "y": float(prow.y), "x": float(prow.x), "score": float(prow.score), "candidate_type": "original_peak"})

    candidates = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if len(candidates) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        candidates = candidates.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]
    return candidates


# --------------------------------------------------------------------------- #
# 8. J: oracle diagnostic (NOT an inference method - never feeds GT into
#    candidate generation, only checks coverage of an already-generated pool)
# --------------------------------------------------------------------------- #
def run_oracle_diagnostic_j(
    n_samples: int = 3,
    percentile_low: float = 1.0,
    percentile_high: float = 97.0,
    thresholds: Sequence[float] = I_DEFAULT_THRESHOLDS,
    radius_um: float = 7.0,
) -> pd.DataFrame:
    """For each GT node, checks whether ANY component centroid (union of
    weighted centroids across all `thresholds`, before NMS dedup) exists
    within `radius_um`. This is a diagnostic upper bound only: it never
    conditions candidate generation on GT, it only evaluates coverage of a
    pool that was already generated candidate-by-candidate exactly as
    method I would - it tells us the best-case recall component-centroid
    methods could ever reach, before NMS/cap/threshold trade-offs enter.
    """
    train_dirs = find_train_zarr_dirs()
    cases_full = _collect_gt_cases(train_dirs[:n_samples])
    if not cases_full:
        print("[error] no GT cases found.")
        return pd.DataFrame(columns=["total_gt_nodes", "oracle_component_coverage", "oracle_lost_count"])

    total_gt = 0
    covered = 0
    image_cache: dict[tuple[str, int], np.ndarray] = {}

    for sample_name, sample_dir, t, gt_t in cases_full:
        key = (sample_name, t)
        if key not in image_cache:
            image_cache[key] = read_image_timepoint(sample_dir, t)
        image = image_cache[key]

        normalized = normalize_intensity(image, percentile_low, percentile_high)
        smoothed = gaussian_filter(normalized, sigma=(FIXED_GAUSSIAN_SIGMA_Z, FIXED_GAUSSIAN_SIGMA_YX, FIXED_GAUSSIAN_SIGMA_YX), mode="reflect")

        pools = []
        for threshold in thresholds:
            comp_df = _connected_component_candidates(smoothed, threshold)
            if len(comp_df):
                pools.append(comp_df)
        oracle_pool = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame(columns=["z", "y", "x", "score"])

        gt_coords = gt_t[["z", "y", "x"]].to_numpy(dtype=float)
        cov_mask = _gt_coverage_mask(gt_coords, oracle_pool, radius_um)
        total_gt += len(gt_coords)
        covered += int(cov_mask.sum())

    summary = pd.DataFrame([{
        "total_gt_nodes": total_gt,
        "oracle_component_coverage": covered / total_gt if total_gt else float("nan"),
        "oracle_lost_count": total_gt - covered,
    }])
    print("\n[J_oracle_diagnostic_only] upper-bound coverage if using ANY component centroid at ANY of the swept thresholds (diagnostic only, not an inference method):")
    print(summary.to_string(index=False))
    return summary


# --------------------------------------------------------------------------- #
# 9. Lost-raw-candidate recovery analysis (which of the 30 lost nodes do
#    G / H / I recover?)
# --------------------------------------------------------------------------- #
REFERENCE_CONFIG = {
    "percentile_low": 1.0,
    "percentile_high": 97.0,
    "threshold": 0.005,
    "min_distance_um": 0.75,
    "max_nodes_per_timepoint": 1000,
}

LOST_RECOVERY_COLUMNS = [
    "sample", "t", "gt_node_id", "gt_z", "gt_y", "gt_x", "min_boundary_distance_um",
    "recovered_by_G_raw", "recovered_by_H_raw", "recovered_by_I_raw",
    "best_candidate_distance_um_G", "best_candidate_distance_um_H", "best_candidate_distance_um_I",
    "recovered_by_any",
]


def analyze_lost_raw_recovery(
    n_samples: int = 3,
    reference_config: dict = REFERENCE_CONFIG,
    radius_um: float = 7.0,
    csv_path: str = "/kaggle/working/lost_raw_recovery_analysis.csv",
) -> pd.DataFrame:
    """Identifies the GT nodes A cannot cover at all (its raw candidate pool
    has nothing within `radius_um`, at the most permissive reference
    config), then checks whether G, H, or I's raw candidate pools recover
    each one - directly answering "how many of the 30 lost nodes does each
    new method recover".
    """
    train_dirs = find_train_zarr_dirs()
    cases_full = _collect_gt_cases(train_dirs[:n_samples])
    if not cases_full:
        print("[error] no GT cases found.")
        return pd.DataFrame(columns=LOST_RECOVERY_COLUMNS)

    rows = []
    for sample_name, sample_dir, t, gt_t in cases_full:
        image = read_image_timepoint(sample_dir, t)
        Z, Y, X = image.shape

        shared_smoothed = _normalized_smoothed(image, reference_config["percentile_low"], reference_config["percentile_high"])
        a_raw = method_a_raw_candidates(
            image, reference_config["percentile_low"], reference_config["percentile_high"], reference_config["threshold"],
            smoothed=shared_smoothed,
        )

        # Which GT nodes in THIS case are lost by A's raw pool?
        lost_gt_rows = [
            gt_row for _, gt_row in gt_t.iterrows()
            if not bool(_gt_coverage_mask(np.array([[gt_row["z"], gt_row["y"], gt_row["x"]]]), a_raw, radius_um)[0])
        ]
        if not lost_gt_rows:
            continue  # every GT node in this case is already covered by A - skip G/H/I entirely for this case

        # Compute G/H/I's raw pools ONCE per case (not once per lost node -
        # they don't depend on which specific GT node is being checked).
        g_raw = method_g_component_centroid(image, reference_config["percentile_low"], reference_config["percentile_high"], smoothed=shared_smoothed)
        h_raw = method_h_peak_to_centroid_refinement(
            image, reference_config["percentile_low"], reference_config["percentile_high"], reference_config["threshold"],
            smoothed=shared_smoothed,
        )
        i_raw = method_i_multi_threshold_component_union(
            image, reference_config["percentile_low"], reference_config["percentile_high"], reference_config["min_distance_um"],
            smoothed=shared_smoothed,
        )

        for gt_row in lost_gt_rows:
            gt_coord = np.array([gt_row["z"], gt_row["y"], gt_row["x"]])
            g_cov = bool(_gt_coverage_mask(gt_coord[None, :], g_raw, radius_um)[0])
            h_cov = bool(_gt_coverage_mask(gt_coord[None, :], h_raw, radius_um)[0])
            i_cov = bool(_gt_coverage_mask(gt_coord[None, :], i_raw, radius_um)[0])

            boundary_dists = _boundary_distances_um(gt_row["z"], gt_row["y"], gt_row["x"], (Z, Y, X))

            rows.append({
                "sample": sample_name, "t": t, "gt_node_id": int(gt_row["node_id"]),
                "gt_z": float(gt_row["z"]), "gt_y": float(gt_row["y"]), "gt_x": float(gt_row["x"]),
                "min_boundary_distance_um": boundary_dists["min_boundary_distance_um"],
                "recovered_by_G_raw": g_cov, "recovered_by_H_raw": h_cov, "recovered_by_I_raw": i_cov,
                "best_candidate_distance_um_G": _nearest_candidate_distance_um(gt_coord, g_raw),
                "best_candidate_distance_um_H": _nearest_candidate_distance_um(gt_coord, h_raw),
                "best_candidate_distance_um_I": _nearest_candidate_distance_um(gt_coord, i_raw),
                "recovered_by_any": bool(g_cov or h_cov or i_cov),
            })

    recovery_df = pd.DataFrame(rows, columns=LOST_RECOVERY_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_df.to_csv(out_path, index=False)

    print(f"\nSaved lost-raw-candidate recovery analysis to {out_path} ({len(recovery_df)} lost node(s) analyzed)")
    if len(recovery_df):
        print(f"  recovered by G: {int(recovery_df['recovered_by_G_raw'].sum())}/{len(recovery_df)}")
        print(f"  recovered by H: {int(recovery_df['recovered_by_H_raw'].sum())}/{len(recovery_df)}")
        print(f"  recovered by I: {int(recovery_df['recovered_by_I_raw'].sum())}/{len(recovery_df)}")
        print(f"  recovered by any: {int(recovery_df['recovered_by_any'].sum())}/{len(recovery_df)}")
        still_missed = recovery_df[~recovery_df["recovered_by_any"]]
        print(f"\nExamples still missed by all of G/H/I ({len(still_missed)} total):")
        cols = ["sample", "t", "gt_node_id", "gt_z", "gt_y", "gt_x", "min_boundary_distance_um"]
        if len(still_missed):
            print(still_missed[cols].head(5).to_string(index=False))
        else:
            print("  (none)")
    return recovery_df


# --------------------------------------------------------------------------- #
# 10. Full A / D / G / H / I evaluation
# --------------------------------------------------------------------------- #
COMPARISON_COLUMNS = [
    "method", "raw_candidate_coverage", "threshold_coverage", "nms_coverage", "cap_coverage",
    "avg_pred_nodes", "recall_on_sparse_gt", "mean_matched_distance_um", "median_matched_distance_um",
    "elapsed_seconds", "recovered_that_a_missed", "still_missed",
]

METHOD_ORDER = (
    "A_current_3d_local_max",
    "D_boundary_aware_3d_local_max",
    "G_component_centroid",
    "H_peak_to_centroid_refinement",
    "I_multi_threshold_component_union",
)


def _generate_raw(method_name: str, image: np.ndarray, config: dict, shared_smoothed: np.ndarray | None = None) -> pd.DataFrame:
    """`shared_smoothed` is the A/G/H/I-shared `_normalized_smoothed(...)`
    result (they all use identical percentile-normalize + reflect-mode
    Gaussian smoothing) - computed once per case by the caller and reused
    across all four methods instead of redundantly recomputed 4x. D uses a
    distinct padded/nearest-mode pipeline and always recomputes its own.
    """
    if method_name == "A_current_3d_local_max":
        return method_a_raw_candidates(image, config["percentile_low"], config["percentile_high"], config["threshold"], smoothed=shared_smoothed)
    if method_name == "D_boundary_aware_3d_local_max":
        return method_d_boundary_aware_3d_local_max(image, config["percentile_low"], config["percentile_high"], config["threshold"])
    if method_name == "G_component_centroid":
        return method_g_component_centroid(image, config["percentile_low"], config["percentile_high"], smoothed=shared_smoothed)
    if method_name == "H_peak_to_centroid_refinement":
        return method_h_peak_to_centroid_refinement(image, config["percentile_low"], config["percentile_high"], config["threshold"], smoothed=shared_smoothed)
    if method_name == "I_multi_threshold_component_union":
        return method_i_multi_threshold_component_union(image, config["percentile_low"], config["percentile_high"], config["min_distance_um"], smoothed=shared_smoothed)
    raise ValueError(f"unknown method {method_name!r}")


def compare_centroid_aware_methods(
    n_samples: int = 3,
    reference_config: dict = REFERENCE_CONFIG,
    match_max_distance_um: float = 7.0,
    methods: Sequence[str] = METHOD_ORDER,
    csv_path: str = "/kaggle/working/centroid_detector_comparison.csv",
) -> pd.DataFrame:
    """Evaluates A/D/G/H/I on the first `n_samples` real train samples with
    the same 4-stage coverage funnel used in prior milestones (raw /
    threshold / nms / cap), plus avg_pred_nodes, recall_on_sparse_gt,
    mean/median_matched_distance_um, elapsed_seconds, and - critically -
    per-method GT-node-level comparison against A's baseline coverage:
    how many GT nodes does this method recover that A's final (capped)
    pipeline missed, and how many does it still miss.
    """
    train_dirs = find_train_zarr_dirs()
    cases_full = _collect_gt_cases(train_dirs[:n_samples])
    if not cases_full:
        print("[error] no GT cases found.")
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    cases = [(name, d, t, gt_t[["z", "y", "x"]].to_numpy(dtype=float)) for name, d, t, gt_t in cases_full]
    total_gt = sum(len(c[3]) for c in cases)
    image_cache: dict[tuple[str, int], np.ndarray] = {}
    # A/G/H/I all share identical percentile-normalize + reflect-mode
    # Gaussian smoothing - cache it once per case so it's computed at most
    # once total (not once per method) across the whole comparison.
    smoothed_cache: dict[tuple[str, int], np.ndarray] = {}
    SHARED_SMOOTHED_METHODS = {"A_current_3d_local_max", "G_component_centroid", "H_peak_to_centroid_refinement", "I_multi_threshold_component_union"}

    def get_image(sample_name: str, sample_dir: Path, t: int) -> np.ndarray:
        key = (sample_name, t)
        if key not in image_cache:
            image_cache[key] = read_image_timepoint(sample_dir, t)
        return image_cache[key]

    def get_shared_smoothed(sample_name: str, t: int, image: np.ndarray) -> np.ndarray:
        key = (sample_name, t)
        if key not in smoothed_cache:
            smoothed_cache[key] = _normalized_smoothed(image, reference_config["percentile_low"], reference_config["percentile_high"])
        return smoothed_cache[key]

    results = []
    coverage_by_method: dict[str, np.ndarray] = {}

    for method_name in methods:
        t0 = time.time()
        raw_cov = thr_cov = nms_cov = cap_cov = 0
        sum_pred = sum_matched = 0
        all_distances: list[float] = []
        node_cap_cov: list[bool] = []

        for sample_name, sample_dir, t, gt_coords in cases:
            image = get_image(sample_name, sample_dir, t)
            shared_smoothed = get_shared_smoothed(sample_name, t, image) if method_name in SHARED_SMOOTHED_METHODS else None
            raw = _generate_raw(method_name, image, reference_config, shared_smoothed=shared_smoothed)
            thresholded = raw[raw["score"] >= reference_config["threshold"]]
            nms = _nms_by_physical_distance(thresholded, reference_config["min_distance_um"])
            capped = nms.iloc[: reference_config["max_nodes_per_timepoint"]]
            pred_coords = capped[["z", "y", "x"]].to_numpy(dtype=float)

            raw_cov += int(_gt_coverage_mask(gt_coords, raw, match_max_distance_um).sum())
            thr_cov += int(_gt_coverage_mask(gt_coords, thresholded, match_max_distance_um).sum())
            nms_cov += int(_gt_coverage_mask(gt_coords, nms, match_max_distance_um).sum())
            cap_mask = _gt_coverage_mask(gt_coords, capped, match_max_distance_um)
            cap_cov += int(cap_mask.sum())
            node_cap_cov.extend(cap_mask.tolist())

            matched = _match_single_timepoint(pred_coords, gt_coords, match_max_distance_um)
            sum_pred += len(pred_coords)
            sum_matched += len(matched)
            all_distances.extend(matched)

        elapsed = time.time() - t0
        n_cases = len(cases)
        coverage_by_method[method_name] = np.array(node_cap_cov, dtype=bool)
        results.append({
            "method": method_name,
            "raw_candidate_coverage": raw_cov / total_gt if total_gt else float("nan"),
            "threshold_coverage": thr_cov / total_gt if total_gt else float("nan"),
            "nms_coverage": nms_cov / total_gt if total_gt else float("nan"),
            "cap_coverage": cap_cov / total_gt if total_gt else float("nan"),
            "avg_pred_nodes": sum_pred / n_cases,
            "recall_on_sparse_gt": (sum_matched / total_gt) if total_gt else float("nan"),
            "mean_matched_distance_um": float(np.mean(all_distances)) if all_distances else float("nan"),
            "median_matched_distance_um": float(np.median(all_distances)) if all_distances else float("nan"),
            "elapsed_seconds": elapsed,
        })
        print(f"  {method_name}: raw_cov={results[-1]['raw_candidate_coverage']:.3f} "
              f"cap_cov={results[-1]['cap_coverage']:.3f} avg_pred={results[-1]['avg_pred_nodes']:.1f} "
              f"done in {elapsed:.1f}s")

    baseline_name = "A_current_3d_local_max"
    baseline_cov = coverage_by_method.get(baseline_name)
    for row in results:
        this_cov = coverage_by_method[row["method"]]
        if baseline_cov is not None and len(baseline_cov) == len(this_cov):
            row["recovered_that_a_missed"] = int(np.sum(this_cov & ~baseline_cov))
        else:
            row["recovered_that_a_missed"] = 0
        row["still_missed"] = int(np.sum(~this_cov))

    comparison_df = pd.DataFrame(results, columns=COMPARISON_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved centroid detector comparison to {out_path}")
    return comparison_df


# --------------------------------------------------------------------------- #
# 11. Goal assessment summary
# --------------------------------------------------------------------------- #
def summarize_against_goals(
    comparison_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    raw_goal: float = 0.90,
    raw_stretch_goal: float = 0.95,
    cap_goal: float = 0.85,
    cap_stretch_goal: float = 0.90,
    csv_path: str = "/kaggle/working/centroid_detector_summary.csv",
) -> pd.DataFrame:
    """Requirement: check each method against raw_candidate_coverage >= 0.90
    (stretch 0.95) and cap_coverage >= 0.85 (stretch 0.90), without letting
    avg_pred_nodes explode. Appends the J oracle diagnostic as a clearly
    labeled, non-comparable reference row.
    """
    if len(comparison_df) == 0:
        print("[warn] no comparison results to summarize.")
        return pd.DataFrame()

    summary_rows = []
    for _, row in comparison_df.iterrows():
        meets_raw = row["raw_candidate_coverage"] >= raw_goal
        meets_raw_stretch = row["raw_candidate_coverage"] >= raw_stretch_goal
        meets_cap = row["cap_coverage"] >= cap_goal
        meets_cap_stretch = row["cap_coverage"] >= cap_stretch_goal
        summary_rows.append({
            "method": row["method"],
            "raw_candidate_coverage": row["raw_candidate_coverage"],
            "cap_coverage": row["cap_coverage"],
            "avg_pred_nodes": row["avg_pred_nodes"],
            "meets_raw_goal_0.90": bool(meets_raw),
            "meets_raw_stretch_goal_0.95": bool(meets_raw_stretch),
            "meets_cap_goal_0.85": bool(meets_cap),
            "meets_cap_stretch_goal_0.90": bool(meets_cap_stretch),
            "recovered_that_a_missed": row["recovered_that_a_missed"],
            "still_missed": row["still_missed"],
        })

    if len(oracle_df):
        oracle_row = oracle_df.iloc[0]
        summary_rows.append({
            "method": "J_oracle_diagnostic_only (NOT an inference method - upper bound only)",
            "raw_candidate_coverage": oracle_row["oracle_component_coverage"],
            "cap_coverage": float("nan"),
            "avg_pred_nodes": float("nan"),
            "meets_raw_goal_0.90": bool(oracle_row["oracle_component_coverage"] >= raw_goal),
            "meets_raw_stretch_goal_0.95": bool(oracle_row["oracle_component_coverage"] >= raw_stretch_goal),
            "meets_cap_goal_0.85": None,
            "meets_cap_stretch_goal_0.90": None,
            "recovered_that_a_missed": None,
            "still_missed": int(oracle_row["oracle_lost_count"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved centroid detector summary to {out_path}")
    print(summary_df.to_string(index=False))

    passing = comparison_df[(comparison_df["raw_candidate_coverage"] >= raw_goal) & (comparison_df["cap_coverage"] >= cap_goal)]
    if len(passing):
        best = passing.loc[passing["avg_pred_nodes"].idxmin()]
        print(f"\nBest method meeting both goals (lowest avg_pred_nodes): {best['method']} (avg_pred_nodes={best['avg_pred_nodes']:.1f})")
    else:
        best_raw = comparison_df.loc[comparison_df["raw_candidate_coverage"].idxmax()]
        print(f"\nNo method met both goals yet. Closest on raw_candidate_coverage: {best_raw['method']} ({best_raw['raw_candidate_coverage']:.3f})")

    return summary_df


# --------------------------------------------------------------------------- #
# 12. Tests
# --------------------------------------------------------------------------- #
def run_centroid_detector_tests() -> None:
    """The core hypothesis under test: an elongated/asymmetric blob's
    brightest voxel (what A/D report) sits away from its true intensity-
    weighted center. G's weighted centroid and H's refined centroid should
    land closer to that true center than A's raw peak does.
    """
    rng = np.random.default_rng(3)
    shape = (24, 48, 48)

    def make_asymmetric_blob(center: tuple[float, float, float]) -> np.ndarray:
        """A comet-shaped blob: a bright core plus a smaller, offset bright
        satellite, so the single brightest voxel is NOT at the blob's
        intensity-weighted center of mass.
        """
        vol = rng.normal(loc=100, scale=2.0, size=shape).astype(np.float32)
        zz, yy, xx = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
        cz, cy, cx = center
        d2_core = (zz - cz) ** 2 * 4 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol += 600.0 * np.exp(-d2_core / (2 * 3.0 ** 2))
        # offset satellite, slightly brighter than the core so the true
        # weighted centroid is pulled away from the core's own peak voxel.
        sat = (cz, cy - 5, cx - 5)
        d2_sat = (zz - sat[0]) ** 2 * 4 + (yy - sat[1]) ** 2 + (xx - sat[2]) ** 2
        vol += 650.0 * np.exp(-d2_sat / (2 * 2.0 ** 2))
        return vol, np.array([cz, cy - 2.5, cx - 2.5])  # approx true "cell centroid" between the two lobes

    vol, true_centroid = make_asymmetric_blob((12.0, 24.0, 24.0))

    a_cands = method_a_raw_candidates(vol, 1.0, 99.5, 0.3)
    g_cands = method_g_component_centroid(vol, 1.0, 99.5, threshold=0.2)
    h_cands = method_h_peak_to_centroid_refinement(vol, 1.0, 99.5, 0.3)

    assert len(a_cands) >= 1, "method A produced no candidates on the asymmetric blob fixture"
    assert len(g_cands) >= 1, "method G produced no candidates on the asymmetric blob fixture"
    assert len(h_cands) >= 1, "method H produced no candidates on the asymmetric blob fixture"

    def nearest_distance(df: pd.DataFrame, point: np.ndarray) -> float:
        coords = df[["z", "y", "x"]].to_numpy(dtype=float)
        return float(np.sqrt(((coords - point) ** 2).sum(axis=1)).min())

    a_dist = nearest_distance(a_cands, true_centroid)
    g_dist = nearest_distance(g_cands[g_cands["candidate_type"] == "weighted_centroid"], true_centroid)
    h_refined_only = h_cands[h_cands["candidate_type"] == "refined_centroid"]
    h_dist = nearest_distance(h_refined_only, true_centroid) if len(h_refined_only) else float("inf")

    assert g_dist <= a_dist + 1e-6, f"G's weighted centroid ({g_dist:.2f}) should be at least as close to the true centroid as A's peak ({a_dist:.2f})"
    assert h_dist <= a_dist + 1e-6, f"H's refined centroid ({h_dist:.2f}) should be at least as close to the true centroid as A's peak ({a_dist:.2f})"

    # I: multi-threshold union should produce at least as many distinct
    # spatial clusters as G at its single default threshold, on a
    # multi-blob volume, and must respect min_distance_um deduplication.
    vol2, _ = make_asymmetric_blob((12.0, 12.0, 12.0))
    vol2b, _ = make_asymmetric_blob((12.0, 36.0, 36.0))
    combined = np.maximum(vol2, vol2b)
    i_cands = method_i_multi_threshold_component_union(combined, 1.0, 99.5, min_distance_um=2.0)
    assert len(i_cands) >= 2, "method I should find at least 2 distinct components on a 2-blob volume"
    assert set(["z", "y", "x", "score"]).issubset(i_cands.columns)

    # J oracle diagnostic must not crash on a tiny fixture and must return
    # the expected schema (checked via the connected-component primitive
    # directly rather than hitting disk I/O in this unit test).
    normalized = normalize_intensity(combined, 1.0, 99.5)
    smoothed = gaussian_filter(normalized, sigma=(1.0, 1.5, 1.5), mode="reflect")
    comp_df = _connected_component_candidates(smoothed, 0.1)
    assert set(["z", "y", "x", "score", "candidate_type", "mean_intensity", "volume", "compactness"]).issubset(comp_df.columns)

    # NMS correctness spot-check (shared primitive, carried over unchanged).
    dedup_input = pd.DataFrame({
        "z": [5.0, 5.0, 20.0], "y": [5.0, 6.0, 20.0], "x": [5.0, 5.0, 20.0], "score": [0.9, 0.8, 0.7],
    })
    deduped = _nms_by_physical_distance(dedup_input, min_distance_um=2.0)
    assert len(deduped) == 2, "NMS should merge the two nearby points and keep the isolated one"

    print("All centroid_detector tests passed (6/6): "
          f"A_dist={a_dist:.2f}um G_dist={g_dist:.2f}um H_dist={h_dist:.2f}um to true asymmetric-blob centroid.")


# --------------------------------------------------------------------------- #
# 13. Top-level driver
# --------------------------------------------------------------------------- #
def run_centroid_detector_analysis(n_samples: int = 3, reference_config: dict = REFERENCE_CONFIG) -> dict:
    print("=== Self-test: centroid-vs-peak hypothesis + G/H/I/NMS unit tests ===")
    run_centroid_detector_tests()

    print("\n=== J: oracle diagnostic (upper bound only, NOT an inference method) ===")
    oracle_df = run_oracle_diagnostic_j(n_samples=n_samples, percentile_low=reference_config["percentile_low"], percentile_high=reference_config["percentile_high"])

    print("\n=== Lost-raw-candidate recovery analysis (which of A's misses does G/H/I recover?) ===")
    recovery_df = analyze_lost_raw_recovery(n_samples=n_samples, reference_config=reference_config)

    print("\n=== Evaluate A / D / G / H / I ===")
    comparison_df = compare_centroid_aware_methods(n_samples=n_samples, reference_config=reference_config)
    print(comparison_df.to_string(index=False))

    print("\n=== Summary against goals ===")
    summary_df = summarize_against_goals(comparison_df, oracle_df)

    return {
        "oracle_df": oracle_df,
        "recovery_df": recovery_df,
        "comparison_df": comparison_df,
        "summary_df": summary_df,
    }


run_centroid_detector_analysis(n_samples=3)
