"""
Biohub - Cell Tracking During Development
GT-centered detector failure analysis.

Even the recall-preserving sweep (detector_recall_calibration.py) only
reached recall_on_sparse_gt=0.586 (72 unmatched GT nodes) at its best
observed real-data setting, with avg_pred_nodes still at 190.7. This means
the problem isn't purely a tuning/capping issue - some GT nodes may have
no viable candidate at all, at any stage, however permissive the settings.

This module inspects every GT node directly: a local 3D patch around its
exact coordinate, whether the raw candidate pool (Gaussian smooth + local
maxima, before any thresholding) contains anything nearby, and whether
that candidate then survives thresholding, NMS, and the max_nodes cap.
It also re-sweeps with an even more permissive parameter grid, and
compares the current 3D-local-maxima candidate generator against two
alternatives (2D projection + z-refinement, and multi-scale LoG blobs).

No tracking, no final submission - diagnostics + calibration only.
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
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import blob_log, peak_local_max

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
# 5. Shared detector primitives (normalize, NMS, matching, coverage)
# --------------------------------------------------------------------------- #
FIXED_GAUSSIAN_SIGMA_Z = 1.0
FIXED_GAUSSIAN_SIGMA_YX = 1.5

# Defensive cap on raw (pre-NMS) candidates per volume - bounds worst-case
# NMS/coverage cost regardless of how many local maxima a given
# (volume, percentile_high) pass produces. Larger than prior sweeps since
# this module's threshold floor (0.005) is far more permissive.
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
    at once) rather than a per-point `query_ball_point` loop - on a raw
    pool of ~20k candidates (typical at this module's permissive threshold
    floor) this is 3-8x faster while producing identical output, since NMS
    dominates the sweep's total runtime (measured ~95% of per-case cost).
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


def _collect_gt_cases(sample_dirs: Sequence[Path]) -> list[tuple[str, Path, int, pd.DataFrame]]:
    """One (sample_name, sample_dir, t, gt_nodes_t) entry per timepoint that
    has at least one GT node, across the given samples. `gt_nodes_t` keeps
    node_id alongside z,y,x (unlike the sweep modules) since the failure
    analysis reports per-GT-node rows.
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


# --------------------------------------------------------------------------- #
# 6. Candidate generation methods (A: 3D local maxima, B: 2D projection +
#    z-refinement, C: multi-scale LoG blobs)
# --------------------------------------------------------------------------- #
def method_a_raw_candidates(image_zyx: np.ndarray, percentile_low: float, percentile_high: float, min_threshold: float) -> pd.DataFrame:
    """Current baseline: percentile normalize + 3D Gaussian smooth + local
    maxima at `min_threshold`. Everything a higher threshold would find is
    a subset of this pool, filterable by score alone.
    """
    normalized = normalize_intensity(image_zyx, percentile_low, percentile_high)
    smoothed = gaussian_filter(normalized, sigma=(FIXED_GAUSSIAN_SIGMA_Z, FIXED_GAUSSIAN_SIGMA_YX, FIXED_GAUSSIAN_SIGMA_YX))

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


def method_b_projection_candidates(
    image_zyx: np.ndarray, percentile_low: float, percentile_high: float, min_threshold: float
) -> pd.DataFrame:
    """XY max-intensity-projection local maxima (collapsing the coarser,
    ambiguous z axis entirely for *detection*), then picking the best z at
    each detected (y, x) via a z-only-smoothed profile. May recover cells
    whose z-extent is too broad/flat to form a clean 3D local maximum.
    """
    normalized = normalize_intensity(image_zyx, percentile_low, percentile_high)
    mip_yx = normalized.max(axis=0)
    smoothed_2d = gaussian_filter(mip_yx, sigma=FIXED_GAUSSIAN_SIGMA_YX)

    coords_2d = peak_local_max(smoothed_2d, min_distance=1, threshold_abs=min_threshold, exclude_border=False)
    if len(coords_2d) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    smoothed_z_profile = gaussian_filter1d(normalized, sigma=FIXED_GAUSSIAN_SIGMA_Z, axis=0)
    ys, xs = coords_2d[:, 0], coords_2d[:, 1]
    z_columns = smoothed_z_profile[:, ys, xs]  # (Z, N)
    z_best = np.argmax(z_columns, axis=0)
    scores = smoothed_2d[ys, xs]

    candidates = pd.DataFrame({
        "z": z_best.astype(float), "y": ys.astype(float), "x": xs.astype(float), "score": scores.astype(float),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if len(candidates) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        candidates = candidates.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]
    return candidates


DEFAULT_LOG_CONFIG = {
    "percentile_low": 1.0,
    "percentile_high": 99.0,
    "min_sigma": (0.5, 1.5, 1.5),
    "max_sigma": (1.5, 3.0, 3.0),
    "num_sigma": 2,
    "threshold": 0.05,
    "overlap": 0.3,
}


def method_c_log_blob_candidates(image_zyx: np.ndarray, log_config: dict = DEFAULT_LOG_CONFIG) -> pd.DataFrame:
    """Multi-scale Laplacian-of-Gaussian blob detection (skimage.feature.blob_log),
    with anisotropic sigma matching this dataset's voxel geometry.

    Measured cost: ~2-3s per 64x256x256 volume at num_sigma=2 (vs ~33s at
    num_sigma=3) - still far too slow to include in a full parameter sweep,
    so this is evaluated once per case at a single fixed operating point,
    not swept.
    """
    normalized = normalize_intensity(image_zyx, log_config["percentile_low"], log_config["percentile_high"])
    blobs = blob_log(
        normalized,
        min_sigma=list(log_config["min_sigma"]),
        max_sigma=list(log_config["max_sigma"]),
        num_sigma=log_config["num_sigma"],
        threshold=log_config["threshold"],
        overlap=log_config["overlap"],
    )
    if len(blobs) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    Z, Y, X = normalized.shape
    zz = np.clip(np.round(blobs[:, 0]).astype(int), 0, Z - 1)
    yy = np.clip(np.round(blobs[:, 1]).astype(int), 0, Y - 1)
    xx = np.clip(np.round(blobs[:, 2]).astype(int), 0, X - 1)
    scores = normalized[zz, yy, xx]

    candidates = pd.DataFrame({
        "z": blobs[:, 0], "y": blobs[:, 1], "x": blobs[:, 2], "score": scores.astype(float),
    }).sort_values("score", ascending=False).reset_index(drop=True)
    return candidates


# --------------------------------------------------------------------------- #
# 7. GT-centered failure analysis (Part 1 + Part 3)
# --------------------------------------------------------------------------- #
REFERENCE_CONFIG = {
    "percentile_low": 1.0,
    "percentile_high": 97.0,
    "threshold": 0.005,
    "min_distance_um": 0.75,
    "max_nodes_per_timepoint": 1000,
}

FAILURE_COLUMNS = [
    "sample", "t", "gt_node_id", "gt_z", "gt_y", "gt_x",
    "intensity_at_gt", "patch_max_intensity", "distance_to_local_max_um", "near_boundary",
    "best_candidate_rank", "raw_candidate_within_7um", "survives_threshold", "survives_nms", "survives_cap",
]

STAGE_SUMMARY_COLUMNS = [
    "total_gt_nodes", "raw_candidate_coverage", "threshold_coverage", "nms_coverage", "cap_coverage",
    "lost_at_raw_candidate", "lost_at_threshold", "lost_at_nms", "lost_at_cap",
]


def _patch_stats(image_zyx: np.ndarray, gt_z: float, gt_y: float, gt_x: float, radius_um: float) -> dict:
    """Raw (unnormalized) intensity at the exact GT coordinate, the max raw
    intensity within a physical `radius_um` patch around it, the physical
    distance from GT to that patch-local maximum, and whether the patch
    was clipped by the volume boundary.
    """
    Z, Y, X = image_zyx.shape
    rz = int(np.ceil(radius_um / VOXEL_SIZE_UM["z"]))
    ry = int(np.ceil(radius_um / VOXEL_SIZE_UM["y"]))
    rx = int(np.ceil(radius_um / VOXEL_SIZE_UM["x"]))

    zc, yc, xc = int(round(gt_z)), int(round(gt_y)), int(round(gt_x))
    z0, z1 = max(0, zc - rz), min(Z, zc + rz + 1)
    y0, y1 = max(0, yc - ry), min(Y, yc + ry + 1)
    x0, x1 = max(0, xc - rx), min(X, xc + rx + 1)
    near_boundary = (z0 != zc - rz) or (z1 != zc + rz + 1) or (y0 != yc - ry) or (y1 != yc + ry + 1) or (x0 != xc - rx) or (x1 != xc + rx + 1)

    intensity_at_gt = float(image_zyx[
        int(np.clip(round(gt_z), 0, Z - 1)),
        int(np.clip(round(gt_y), 0, Y - 1)),
        int(np.clip(round(gt_x), 0, X - 1)),
    ])

    patch = image_zyx[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return {
            "intensity_at_gt": intensity_at_gt, "patch_max_intensity": float("nan"),
            "distance_to_local_max_um": float("nan"), "near_boundary": bool(near_boundary),
        }

    local_idx = np.unravel_index(np.argmax(patch), patch.shape)
    global_zyx = (z0 + local_idx[0], y0 + local_idx[1], x0 + local_idx[2])
    patch_max = float(patch[local_idx])
    dist_to_local_max = physical_distance_um(global_zyx, (gt_z, gt_y, gt_x))

    return {
        "intensity_at_gt": intensity_at_gt, "patch_max_intensity": patch_max,
        "distance_to_local_max_um": dist_to_local_max, "near_boundary": bool(near_boundary),
    }


def _best_nearby_rank(gt_coord_zyx: np.ndarray, candidates_df: pd.DataFrame, max_distance_um: float) -> float:
    """1-based rank (by score, best=1) of the highest-scoring candidate
    within `max_distance_um` of the GT node; NaN if none exists.
    `candidates_df` must already be sorted by score descending.
    """
    if len(candidates_df) == 0:
        return float("nan")
    coords = candidates_df[["z", "y", "x"]].to_numpy(dtype=float) * _VOXEL_SCALE
    gt_scaled = gt_coord_zyx * _VOXEL_SCALE
    dists = np.sqrt(((coords - gt_scaled) ** 2).sum(axis=1))
    within = np.where(dists <= max_distance_um)[0]
    if len(within) == 0:
        return float("nan")
    return float(within.min() + 1)


def run_gt_failure_analysis(
    n_samples: int = 3,
    reference_config: dict = REFERENCE_CONFIG,
    radius_um: float = 7.0,
    csv_path: str = "/kaggle/working/gt_failure_analysis.csv",
) -> pd.DataFrame:
    """Part 1: per-GT-node failure analysis using the most permissive
    reference configuration - if a GT node is still lost here, it isn't a
    tuning problem, the candidate-generation pipeline fundamentally never
    produces anything nearby.
    """
    train_dirs = find_train_zarr_dirs()
    cases = _collect_gt_cases(train_dirs[:n_samples])
    if not cases:
        print("[error] no GT cases found.")
        return pd.DataFrame(columns=FAILURE_COLUMNS)

    rows = []
    for sample_name, sample_dir, t, gt_t in cases:
        image = read_image_timepoint(sample_dir, t)

        raw = method_a_raw_candidates(
            image, reference_config["percentile_low"], reference_config["percentile_high"], reference_config["threshold"]
        )
        thresholded = raw[raw["score"] >= reference_config["threshold"]]
        nms = _nms_by_physical_distance(thresholded, reference_config["min_distance_um"])
        capped = nms.iloc[: reference_config["max_nodes_per_timepoint"]]

        for _, gt_row in gt_t.iterrows():
            gt_coord = np.array([gt_row["z"], gt_row["y"], gt_row["x"]])
            patch_stats = _patch_stats(image, gt_row["z"], gt_row["y"], gt_row["x"], radius_um)

            raw_cov = bool(_gt_coverage_mask(gt_coord[None, :], raw, radius_um)[0])
            thr_cov = bool(_gt_coverage_mask(gt_coord[None, :], thresholded, radius_um)[0])
            nms_cov = bool(_gt_coverage_mask(gt_coord[None, :], nms, radius_um)[0])
            cap_cov = bool(_gt_coverage_mask(gt_coord[None, :], capped, radius_um)[0])
            rank = _best_nearby_rank(gt_coord, raw, radius_um)

            rows.append({
                "sample": sample_name, "t": t, "gt_node_id": int(gt_row["node_id"]),
                "gt_z": float(gt_row["z"]), "gt_y": float(gt_row["y"]), "gt_x": float(gt_row["x"]),
                "intensity_at_gt": patch_stats["intensity_at_gt"],
                "patch_max_intensity": patch_stats["patch_max_intensity"],
                "distance_to_local_max_um": patch_stats["distance_to_local_max_um"],
                "near_boundary": patch_stats["near_boundary"],
                "best_candidate_rank": rank,
                "raw_candidate_within_7um": raw_cov,
                "survives_threshold": thr_cov,
                "survives_nms": nms_cov,
                "survives_cap": cap_cov,
            })

    failure_df = pd.DataFrame(rows, columns=FAILURE_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failure_df.to_csv(out_path, index=False)
    print(f"Saved GT failure analysis to {out_path} (shape={failure_df.shape})")
    return failure_df


def summarize_stage_coverage(
    failure_df: pd.DataFrame, csv_path: str = "/kaggle/working/stagewise_gt_coverage.csv"
) -> pd.DataFrame:
    """Part 3: single-row funnel summary derived from the Part 1 failure table."""
    total = len(failure_df)
    if total == 0:
        summary = pd.DataFrame([{c: 0 for c in STAGE_SUMMARY_COLUMNS}])
    else:
        raw_cov = int(failure_df["raw_candidate_within_7um"].sum())
        thr_cov = int(failure_df["survives_threshold"].sum())
        nms_cov = int(failure_df["survives_nms"].sum())
        cap_cov = int(failure_df["survives_cap"].sum())
        summary = pd.DataFrame([{
            "total_gt_nodes": total,
            "raw_candidate_coverage": raw_cov / total,
            "threshold_coverage": thr_cov / total,
            "nms_coverage": nms_cov / total,
            "cap_coverage": cap_cov / total,
            "lost_at_raw_candidate": total - raw_cov,
            "lost_at_threshold": raw_cov - thr_cov,
            "lost_at_nms": thr_cov - nms_cov,
            "lost_at_cap": nms_cov - cap_cov,
        }])

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved stage-wise coverage summary to {out_path}")
    print(summary.to_string(index=False))
    return summary


def print_gt_loss_examples(failure_df: pd.DataFrame, n_examples: int = 5) -> None:
    """Part 7: a few example GT nodes lost at each stage transition."""
    if len(failure_df) == 0:
        print("[warn] no failure analysis rows to inspect.")
        return

    stages = [
        ("lost at raw candidate stage (no nearby local maximum at all, however permissive)",
         ~failure_df["raw_candidate_within_7um"]),
        ("lost at threshold stage (a nearby candidate existed, but scored too low)",
         failure_df["raw_candidate_within_7um"] & ~failure_df["survives_threshold"]),
        ("lost at NMS stage (survived threshold, suppressed by a nearby higher-score peak)",
         failure_df["survives_threshold"] & ~failure_df["survives_nms"]),
        ("lost at cap stage (survived NMS, but ranked below the max_nodes_per_timepoint cutoff)",
         failure_df["survives_nms"] & ~failure_df["survives_cap"]),
    ]
    cols = ["sample", "t", "gt_node_id", "gt_z", "gt_y", "gt_x", "intensity_at_gt", "patch_max_intensity", "best_candidate_rank"]

    for label, mask in stages:
        subset = failure_df[mask]
        print(f"\nExamples {label} ({len(subset)} total):")
        if len(subset):
            print(subset[cols].head(n_examples).to_string(index=False))
        else:
            print("  (none)")


# --------------------------------------------------------------------------- #
# 8. Recall-preserving sweep v2 (Part 4) - even more permissive grid
# --------------------------------------------------------------------------- #
DEFAULT_PERCENTILE_HIGH_VALUES = (97.0, 98.0, 98.5, 99.0, 99.3, 99.5)
DEFAULT_THRESHOLD_VALUES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)
DEFAULT_MIN_DISTANCE_UM_VALUES = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
DEFAULT_MAX_NODES_VALUES = (200, 300, 400, 600, 800, 1000)

SWEEP_COLUMNS = [
    "percentile_high", "threshold", "min_distance_um", "max_nodes_per_timepoint",
    "avg_gt_nodes", "avg_pred_nodes", "avg_matched_nodes",
    "total_unmatched_gt_nodes", "total_unmatched_pred_nodes",
    "mean_matched_distance_um", "median_matched_distance_um",
    "recall_on_sparse_gt", "precision_against_sparse_gt", "score_proxy",
    "cap_gt_coverage",
]

PARAM_COLUMNS = ["percentile_high", "threshold", "min_distance_um", "max_nodes_per_timepoint"]


def run_recall_preserving_sweep_v2(
    n_samples: int = 3,
    percentile_high_values: Sequence[float] = DEFAULT_PERCENTILE_HIGH_VALUES,
    threshold_values: Sequence[float] = DEFAULT_THRESHOLD_VALUES,
    min_distance_um_values: Sequence[float] = DEFAULT_MIN_DISTANCE_UM_VALUES,
    max_nodes_values: Sequence[int] = DEFAULT_MAX_NODES_VALUES,
    match_max_distance_um: float = 7.0,
    csv_path: str = "/kaggle/working/recall_preserving_detector_sweep_v2.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Part 4: same efficient caching architecture as detector_calibration.py
    (percentile_high is the only parameter that changes the expensive
    normalize+smooth+peak-detection pass; NMS is shared across
    max_nodes_per_timepoint values), but with a much more permissive grid.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=SWEEP_COLUMNS)

    cases_full = _collect_gt_cases(train_dirs[:n_samples])
    if not cases_full:
        print("[error] none of the selected samples have GT nodes.")
        return pd.DataFrame(columns=SWEEP_COLUMNS)

    # (sample_name, sample_dir, t, gt_coords_zyx) - lean form for the sweep
    cases = [(name, d, t, gt_t[["z", "y", "x"]].to_numpy(dtype=float)) for name, d, t, gt_t in cases_full]
    total_gt = sum(len(c[3]) for c in cases)

    if verbose:
        print(f"Calibrating over {len(cases)} case(s), {total_gt} total GT node(s).")

    min_threshold = min(threshold_values)
    image_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_image(sample_name: str, sample_dir: Path, t: int) -> np.ndarray:
        key = (sample_name, t)
        if key not in image_cache:
            image_cache[key] = read_image_timepoint(sample_dir, t)
        return image_cache[key]

    sweep_start = time.time()
    rows = []

    for percentile_high in percentile_high_values:
        ph_start = time.time()
        raw_candidates_by_case = [
            method_a_raw_candidates(get_image(sample_name, sample_dir, t), 1.0, percentile_high, min_threshold)
            for sample_name, sample_dir, t, _ in cases
        ]
        if verbose:
            print(f"  percentile_high={percentile_high}: candidate pools built in {time.time() - ph_start:.1f}s")

        for threshold in threshold_values:
            filtered_by_case = [raw[raw["score"] >= threshold] for raw in raw_candidates_by_case]

            for min_distance_um in min_distance_um_values:
                nms_by_case = [
                    _nms_by_physical_distance(filtered, min_distance_um) for filtered in filtered_by_case
                ]

                for max_nodes in max_nodes_values:
                    sum_gt = sum_pred = sum_matched = cap_covered = 0
                    all_distances: list[float] = []

                    for (_, _, _, gt_coords), nms_df in zip(cases, nms_by_case):
                        capped = nms_df.iloc[:max_nodes]
                        pred_coords = capped[["z", "y", "x"]].to_numpy(dtype=float)

                        cap_covered += int(_gt_coverage_mask(gt_coords, capped, match_max_distance_um).sum())
                        matched_distances = _match_single_timepoint(pred_coords, gt_coords, match_max_distance_um)

                        sum_gt += len(gt_coords)
                        sum_pred += len(pred_coords)
                        sum_matched += len(matched_distances)
                        all_distances.extend(matched_distances)

                    n_cases = len(cases)
                    recall_on_sparse_gt = (sum_matched / sum_gt) if sum_gt > 0 else float("nan")
                    avg_pred_nodes = sum_pred / n_cases

                    rows.append({
                        "percentile_high": percentile_high,
                        "threshold": threshold,
                        "min_distance_um": min_distance_um,
                        "max_nodes_per_timepoint": max_nodes,
                        "avg_gt_nodes": sum_gt / n_cases,
                        "avg_pred_nodes": avg_pred_nodes,
                        "avg_matched_nodes": sum_matched / n_cases,
                        "total_unmatched_gt_nodes": sum_gt - sum_matched,
                        "total_unmatched_pred_nodes": sum_pred - sum_matched,
                        "mean_matched_distance_um": float(np.mean(all_distances)) if all_distances else float("nan"),
                        "median_matched_distance_um": float(np.median(all_distances)) if all_distances else float("nan"),
                        "recall_on_sparse_gt": recall_on_sparse_gt,
                        "precision_against_sparse_gt": (
                            (sum_matched / sum_pred) if sum_pred > 0 else (1.0 if sum_matched == 0 else 0.0)
                        ),
                        "score_proxy": recall_on_sparse_gt - 0.002 * avg_pred_nodes,
                        "cap_gt_coverage": cap_covered / total_gt if total_gt else float("nan"),
                    })

    sweep_df = pd.DataFrame(rows, columns=SWEEP_COLUMNS)

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_path, index=False)

    if verbose:
        print(
            f"\nSwept {len(sweep_df)} parameter combination(s) over {len(cases)} case(s) "
            f"({total_gt} GT node(s)) in {time.time() - sweep_start:.1f}s"
        )
        print(f"Saved sweep to {out_path} (shape={sweep_df.shape})")

    return sweep_df


def print_top_configs(sweep_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Part 3 (sorting): cap_gt_coverage desc, avg_pred_nodes asc, mean_matched_distance_um asc."""
    if len(sweep_df) == 0:
        print("[warn] sweep produced no rows to rank.")
        return sweep_df
    ranked = sweep_df.sort_values(
        by=["cap_gt_coverage", "avg_pred_nodes", "mean_matched_distance_um"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    top = ranked.head(top_n)
    print(f"\nTop {len(top)} config(s) (sorted by cap_gt_coverage desc, avg_pred_nodes asc, mean_matched_distance_um asc):")
    print(top.to_string(index=False))
    return top


# --------------------------------------------------------------------------- #
# 9. Candidate-generation method comparison (Part 5) - single operating
#    point per method, not swept (LoG is too slow to sweep - see method_c).
# --------------------------------------------------------------------------- #
METHOD_COMPARISON_COLUMNS = [
    "method", "avg_gt_nodes", "avg_pred_nodes", "avg_matched_nodes",
    "recall_on_sparse_gt", "precision_against_sparse_gt", "cap_gt_coverage", "elapsed_seconds",
]


def compare_candidate_generation_methods(
    n_samples: int = 3,
    reference_config: dict = REFERENCE_CONFIG,
    log_config: dict = DEFAULT_LOG_CONFIG,
    match_max_distance_um: float = 7.0,
) -> pd.DataFrame:
    """Compares method A (3D local maxima), B (2D projection + z-refinement),
    and C (multi-scale LoG blobs) at one fixed, permissive operating point
    each, all passed through the same NMS + cap + Hungarian-match pipeline.
    """
    train_dirs = find_train_zarr_dirs()
    cases_full = _collect_gt_cases(train_dirs[:n_samples])
    if not cases_full:
        print("[error] no GT cases found.")
        return pd.DataFrame(columns=METHOD_COMPARISON_COLUMNS)

    cases = [(name, d, t, gt_t[["z", "y", "x"]].to_numpy(dtype=float)) for name, d, t, gt_t in cases_full]
    total_gt = sum(len(c[3]) for c in cases)
    image_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_image(sample_name: str, sample_dir: Path, t: int) -> np.ndarray:
        key = (sample_name, t)
        if key not in image_cache:
            image_cache[key] = read_image_timepoint(sample_dir, t)
        return image_cache[key]

    def generate_a(image):
        return method_a_raw_candidates(image, reference_config["percentile_low"], reference_config["percentile_high"], reference_config["threshold"])

    def generate_b(image):
        return method_b_projection_candidates(image, reference_config["percentile_low"], reference_config["percentile_high"], reference_config["threshold"])

    def generate_c(image):
        return method_c_log_blob_candidates(image, log_config)

    methods = [("A_3d_local_max", generate_a), ("B_2d_projection", generate_b), ("C_log_blob", generate_c)]
    results = []

    for method_name, generate in methods:
        t0 = time.time()
        sum_gt = sum_pred = sum_matched = cap_covered = 0

        for sample_name, sample_dir, t, gt_coords in cases:
            image = get_image(sample_name, sample_dir, t)
            raw = generate(image)
            nms = _nms_by_physical_distance(raw, reference_config["min_distance_um"])
            capped = nms.iloc[: reference_config["max_nodes_per_timepoint"]]
            pred_coords = capped[["z", "y", "x"]].to_numpy(dtype=float)

            cap_covered += int(_gt_coverage_mask(gt_coords, capped, match_max_distance_um).sum())
            matched = _match_single_timepoint(pred_coords, gt_coords, match_max_distance_um)

            sum_gt += len(gt_coords)
            sum_pred += len(pred_coords)
            sum_matched += len(matched)

        elapsed = time.time() - t0
        n_cases = len(cases)
        results.append({
            "method": method_name,
            "avg_gt_nodes": sum_gt / n_cases,
            "avg_pred_nodes": sum_pred / n_cases,
            "avg_matched_nodes": sum_matched / n_cases,
            "recall_on_sparse_gt": (sum_matched / sum_gt) if sum_gt else float("nan"),
            "precision_against_sparse_gt": (sum_matched / sum_pred) if sum_pred else (1.0 if sum_matched == 0 else 0.0),
            "cap_gt_coverage": (cap_covered / total_gt) if total_gt else float("nan"),
            "elapsed_seconds": elapsed,
        })
        print(f"  {method_name}: done in {elapsed:.1f}s")

    return pd.DataFrame(results, columns=METHOD_COMPARISON_COLUMNS)


# --------------------------------------------------------------------------- #
# 10. Top-level driver
# --------------------------------------------------------------------------- #
def run_gt_centered_failure_analysis(
    n_samples: int = 3,
    reference_config: dict = REFERENCE_CONFIG,
    run_sweep: bool = True,
    run_method_comparison: bool = True,
) -> dict:
    print("=== Part 1: GT-centered failure analysis (per GT node) ===")
    failure_df = run_gt_failure_analysis(n_samples=n_samples, reference_config=reference_config)

    print("\n=== Part 3: stage-wise coverage summary ===")
    stage_summary_df = summarize_stage_coverage(failure_df)

    print("\n=== Part 7: examples of GT nodes lost at each stage ===")
    print_gt_loss_examples(failure_df)

    sweep_df = pd.DataFrame(columns=SWEEP_COLUMNS)
    if run_sweep:
        print("\n=== Part 4: recall-preserving sweep v2 (more permissive grid) ===")
        sweep_df = run_recall_preserving_sweep_v2(n_samples=n_samples)
        print("\n=== Part 3/7: top configs ===")
        print_top_configs(sweep_df, top_n=20)

    method_comparison_df = pd.DataFrame(columns=METHOD_COMPARISON_COLUMNS)
    if run_method_comparison:
        print("\n=== Part 5: candidate-generation method comparison (A vs B vs C) ===")
        method_comparison_df = compare_candidate_generation_methods(n_samples=n_samples)
        print(method_comparison_df.to_string(index=False))

    return {
        "failure_df": failure_df,
        "stage_summary_df": stage_summary_df,
        "sweep_df": sweep_df,
        "method_comparison_df": method_comparison_df,
    }


if __name__ == "__main__":
    run_gt_centered_failure_analysis(n_samples=3)
