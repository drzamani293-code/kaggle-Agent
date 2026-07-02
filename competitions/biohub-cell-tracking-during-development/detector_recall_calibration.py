"""
Biohub - Cell Tracking During Development
Recall-preserving detector diagnostics + second calibration sweep.

The first calibration sweep (detector_calibration.py) collapsed recall to
~0.45-0.53 (missing 80-100 GT nodes) at its best-ranked settings, even
though the uncalibrated Milestone 5 baseline matched nearly all GT nodes.
The likely cause: that sweep's threshold floor (0.35) and percentile_high
floor (99.0) can clip real-but-dim cell signal out of the *raw candidate
pool* before any of the tunable parameters even get a chance to act - no
downstream tuning can recover a GT node whose matching peak never made it
into the pool in the first place.

This module:
  Part 1 - measures, at each of 4 pipeline stages (raw candidate pool,
    after threshold, after NMS, after max_nodes cap), whether every GT
    node in the first 3 train samples still has >=1 candidate within 7 um
    - directly localizing where in the pipeline coverage is actually lost.
  Part 2 - re-sweeps with a much less aggressive parameter range
    (lower percentile_high/threshold floors, finer/smaller min_distance_um,
    larger max_nodes_per_timepoint) designed to preserve that coverage.
  Part 3 - ranks configurations by coverage first, overprediction second.

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
FIXED_PERCENTILE_LOW = 1.0
FIXED_GAUSSIAN_SIGMA_Z = 1.0
FIXED_GAUSSIAN_SIGMA_YX = 1.5

# Defensive cap on raw (pre-NMS) candidates per volume - bounds worst-case
# NMS/coverage cost regardless of how many local maxima a given
# (volume, percentile_high) pass produces. Larger than the first sweep's
# cap since a much lower threshold floor here lets more candidates through.
MAX_RAW_CANDIDATES_BEFORE_NMS = 20000

_VOXEL_SCALE = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])


def normalize_intensity(image: np.ndarray, percentile_low: float, percentile_high: float) -> np.ndarray:
    """Percentile-clip + rescale an image to [0, 1] float32."""
    lo, hi = np.percentile(image, [percentile_low, percentile_high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def _nms_by_physical_distance(candidates_df: pd.DataFrame, min_distance_um: float) -> pd.DataFrame:
    """Greedy non-maximum suppression via KD-tree: candidates must already
    be sorted by score descending. See classical_detector.py (Milestone 5).
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


def _gt_coverage_mask(gt_coords_zyx: np.ndarray, candidates_df: pd.DataFrame, max_distance_um: float) -> np.ndarray:
    """Boolean array (one per GT node): is there >= 1 candidate within
    `max_distance_um` physical distance, regardless of 1-1 assignment?
    This is a looser existence check than Hungarian matching - it upper-
    bounds the recall any matching scheme could possibly achieve at this
    pipeline stage.
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


# --------------------------------------------------------------------------- #
# 6. Recall-preserving sweep (Part 2) + stage-wise coverage (Part 1)
# --------------------------------------------------------------------------- #
DEFAULT_PERCENTILE_HIGH_VALUES = (98.0, 98.5, 99.0, 99.3, 99.5, 99.7)
DEFAULT_THRESHOLD_VALUES = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_MIN_DISTANCE_UM_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
DEFAULT_MAX_NODES_VALUES = (150, 200, 250, 300, 400, 600)

SWEEP_COLUMNS = [
    "percentile_high", "threshold", "min_distance_um", "max_nodes_per_timepoint",
    "avg_gt_nodes", "avg_pred_nodes", "avg_matched_nodes",
    "total_unmatched_gt_nodes", "total_unmatched_pred_nodes",
    "mean_matched_distance_um", "median_matched_distance_um",
    "recall_on_sparse_gt", "precision_against_sparse_gt", "score_proxy",
]

COVERAGE_COLUMNS = [
    "percentile_high", "threshold", "min_distance_um", "max_nodes_per_timepoint",
    "raw_candidate_gt_coverage", "threshold_gt_coverage", "nms_gt_coverage", "cap_gt_coverage",
    "gt_lost_at_threshold", "gt_lost_at_nms", "gt_lost_at_cap",
]


def _collect_gt_cases(sample_dirs: Sequence[Path]) -> list[tuple[str, Path, int, np.ndarray]]:
    """One (sample_name, sample_dir, t, gt_coords_zyx) entry per timepoint
    that has at least one GT node, across the given samples.
    """
    cases = []
    for sample_dir in sample_dirs:
        gt_nodes, _ = read_geff(sample_dir)
        if len(gt_nodes) == 0:
            continue
        for t in sorted(gt_nodes["t"].unique()):
            gt_t = gt_nodes[gt_nodes["t"] == t]
            cases.append((sample_dir.stem, sample_dir, int(t), gt_t[["z", "y", "x"]].to_numpy(dtype=float)))
    return cases


def _raw_candidates_for_case(image_zyx: np.ndarray, percentile_high: float, min_threshold: float) -> pd.DataFrame:
    """The expensive step (percentile normalize + Gaussian smooth + local-
    maxima detection at the sweep's lowest threshold): everything a higher
    threshold would find is a subset of this, filterable by score alone.
    """
    normalized = normalize_intensity(image_zyx, FIXED_PERCENTILE_LOW, percentile_high)
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


def _coverage_fraction(cases, candidates_by_case: Sequence[pd.DataFrame], total_gt: int, max_distance_um: float) -> int:
    """Total (not fractional) count of GT nodes with >=1 candidate within
    `max_distance_um`, across all cases; caller divides by total_gt.
    """
    covered = 0
    for (_, _, _, gt_coords), candidates_df in zip(cases, candidates_by_case):
        covered += int(_gt_coverage_mask(gt_coords, candidates_df, max_distance_um).sum())
    return covered


def run_recall_preserving_calibration(
    n_samples: int = 3,
    percentile_high_values: Sequence[float] = DEFAULT_PERCENTILE_HIGH_VALUES,
    threshold_values: Sequence[float] = DEFAULT_THRESHOLD_VALUES,
    min_distance_um_values: Sequence[float] = DEFAULT_MIN_DISTANCE_UM_VALUES,
    max_nodes_values: Sequence[int] = DEFAULT_MAX_NODES_VALUES,
    match_max_distance_um: float = 7.0,
    sweep_csv_path: str = "/kaggle/working/detector_recall_preserving_sweep.csv",
    coverage_csv_path: str = "/kaggle/working/detector_stage_coverage.csv",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep the recall-preserving parameter grid over the first `n_samples`
    train samples (timepoints with GT nodes only), computing BOTH the usual
    overprediction/recall/precision sweep metrics (Part 2) AND, for every
    combination, stage-wise GT coverage (Part 1): does every GT node still
    have >=1 candidate within `match_max_distance_um` after the raw
    candidate pool, after thresholding, after NMS, and after the
    max_nodes_per_timepoint cap.

    Returns (sweep_df, coverage_df); both share the same parameter columns
    and row order, so they can be joined on
    [percentile_high, threshold, min_distance_um, max_nodes_per_timepoint].
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=SWEEP_COLUMNS), pd.DataFrame(columns=COVERAGE_COLUMNS)

    cases = _collect_gt_cases(train_dirs[:n_samples])
    if not cases:
        print("[error] none of the selected samples have GT nodes.")
        return pd.DataFrame(columns=SWEEP_COLUMNS), pd.DataFrame(columns=COVERAGE_COLUMNS)

    total_gt = sum(len(gt_coords) for _, _, _, gt_coords in cases)

    if verbose:
        print(f"Calibrating over {len(cases)} (sample, timepoint) case(s), {total_gt} total GT node(s).")

    min_threshold = min(threshold_values)
    image_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_image(sample_name: str, sample_dir: Path, t: int) -> np.ndarray:
        key = (sample_name, t)
        if key not in image_cache:
            image_cache[key] = read_image_timepoint(sample_dir, t)
        return image_cache[key]

    sweep_start = time.time()
    sweep_rows = []
    coverage_rows = []

    for percentile_high in percentile_high_values:
        ph_start = time.time()
        raw_candidates_by_case = [
            _raw_candidates_for_case(get_image(sample_name, sample_dir, t), percentile_high, min_threshold)
            for sample_name, sample_dir, t, _ in cases
        ]
        raw_covered = _coverage_fraction(cases, raw_candidates_by_case, total_gt, match_max_distance_um)
        if verbose:
            print(
                f"  percentile_high={percentile_high}: candidate pools built in {time.time() - ph_start:.1f}s "
                f"(raw_candidate_gt_coverage={raw_covered / total_gt:.3f})"
            )

        for threshold in threshold_values:
            filtered_by_case = [raw[raw["score"] >= threshold] for raw in raw_candidates_by_case]
            threshold_covered = _coverage_fraction(cases, filtered_by_case, total_gt, match_max_distance_um)

            for min_distance_um in min_distance_um_values:
                nms_by_case = [
                    _nms_by_physical_distance(filtered, min_distance_um) for filtered in filtered_by_case
                ]
                nms_covered = _coverage_fraction(cases, nms_by_case, total_gt, match_max_distance_um)

                for max_nodes in max_nodes_values:
                    sum_gt = sum_pred = sum_matched = 0
                    all_distances: list[float] = []
                    cap_covered = 0

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
                    avg_gt_nodes = sum_gt / n_cases
                    avg_pred_nodes = sum_pred / n_cases
                    avg_matched_nodes = sum_matched / n_cases
                    total_unmatched_gt_nodes = sum_gt - sum_matched
                    total_unmatched_pred_nodes = sum_pred - sum_matched
                    mean_matched_distance_um = float(np.mean(all_distances)) if all_distances else float("nan")
                    median_matched_distance_um = float(np.median(all_distances)) if all_distances else float("nan")
                    recall_on_sparse_gt = (sum_matched / sum_gt) if sum_gt > 0 else float("nan")
                    precision_against_sparse_gt = (
                        (sum_matched / sum_pred) if sum_pred > 0 else (1.0 if sum_matched == 0 else 0.0)
                    )
                    score_proxy = recall_on_sparse_gt - 0.002 * avg_pred_nodes

                    params = {
                        "percentile_high": percentile_high,
                        "threshold": threshold,
                        "min_distance_um": min_distance_um,
                        "max_nodes_per_timepoint": max_nodes,
                    }

                    sweep_rows.append({
                        **params,
                        "avg_gt_nodes": avg_gt_nodes,
                        "avg_pred_nodes": avg_pred_nodes,
                        "avg_matched_nodes": avg_matched_nodes,
                        "total_unmatched_gt_nodes": total_unmatched_gt_nodes,
                        "total_unmatched_pred_nodes": total_unmatched_pred_nodes,
                        "mean_matched_distance_um": mean_matched_distance_um,
                        "median_matched_distance_um": median_matched_distance_um,
                        "recall_on_sparse_gt": recall_on_sparse_gt,
                        "precision_against_sparse_gt": precision_against_sparse_gt,
                        "score_proxy": score_proxy,
                    })

                    coverage_rows.append({
                        **params,
                        "raw_candidate_gt_coverage": raw_covered / total_gt,
                        "threshold_gt_coverage": threshold_covered / total_gt,
                        "nms_gt_coverage": nms_covered / total_gt,
                        "cap_gt_coverage": cap_covered / total_gt,
                        "gt_lost_at_threshold": raw_covered - threshold_covered,
                        "gt_lost_at_nms": threshold_covered - nms_covered,
                        "gt_lost_at_cap": nms_covered - cap_covered,
                    })

    sweep_df = pd.DataFrame(sweep_rows, columns=SWEEP_COLUMNS)
    coverage_df = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)

    sweep_out = Path(sweep_csv_path)
    sweep_out.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(sweep_out, index=False)

    coverage_out = Path(coverage_csv_path)
    coverage_out.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(coverage_out, index=False)

    if verbose:
        print(
            f"\nSwept {len(sweep_df)} parameter combination(s) over {len(cases)} case(s) "
            f"({total_gt} GT node(s)) in {time.time() - sweep_start:.1f}s"
        )
        print(f"Saved sweep to {sweep_out} (shape={sweep_df.shape})")
        print(f"Saved stage coverage to {coverage_out} (shape={coverage_df.shape})")

    return sweep_df, coverage_df


# --------------------------------------------------------------------------- #
# Part 3. Ranking and reporting
# --------------------------------------------------------------------------- #
PARAM_COLUMNS = ["percentile_high", "threshold", "min_distance_um", "max_nodes_per_timepoint"]


def _merge_sweep_and_coverage(sweep_df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    return sweep_df.merge(coverage_df, on=PARAM_COLUMNS, how="inner")


def print_top_recall_preserving_results(
    sweep_df: pd.DataFrame, coverage_df: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Sort by cap_gt_coverage desc, avg_pred_nodes asc, mean_matched_distance_um
    asc, and print/return the top `top_n` rows.
    """
    merged = _merge_sweep_and_coverage(sweep_df, coverage_df)
    if len(merged) == 0:
        print("[warn] sweep produced no rows to rank.")
        return merged

    ranked = merged.sort_values(
        by=["cap_gt_coverage", "avg_pred_nodes", "mean_matched_distance_um"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    top = ranked.head(top_n)
    print(
        f"\nTop {len(top)} parameter set(s) "
        f"(sorted by cap_gt_coverage desc, then avg_pred_nodes asc, then mean_matched_distance_um asc):"
    )
    print(top.to_string(index=False))
    return top


def print_high_recall_low_overprediction_configs(
    sweep_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    min_cap_gt_coverage: float = 0.95,
    min_recall_on_sparse_gt: float = 0.95,
) -> pd.DataFrame:
    """Configs meeting both cap_gt_coverage >= `min_cap_gt_coverage` and
    recall_on_sparse_gt >= `min_recall_on_sparse_gt`, sorted by avg_pred_nodes
    ascending (i.e. the least overpredicting config among those that
    actually preserve recall).
    """
    merged = _merge_sweep_and_coverage(sweep_df, coverage_df)
    qualifying = merged[
        (merged["cap_gt_coverage"] >= min_cap_gt_coverage)
        & (merged["recall_on_sparse_gt"] >= min_recall_on_sparse_gt)
    ].sort_values(by="avg_pred_nodes", ascending=True).reset_index(drop=True)

    print(
        f"\nConfigs with cap_gt_coverage >= {min_cap_gt_coverage} and "
        f"recall_on_sparse_gt >= {min_recall_on_sparse_gt} (sorted by avg_pred_nodes asc): "
        f"{len(qualifying)} found"
    )
    if len(qualifying):
        print(qualifying.to_string(index=False))
    else:
        print("  (none - consider loosening min_cap_gt_coverage / min_recall_on_sparse_gt)")
    return qualifying


def run_recall_preserving_diagnostics(
    n_samples: int = 3,
    percentile_high_values: Sequence[float] = DEFAULT_PERCENTILE_HIGH_VALUES,
    threshold_values: Sequence[float] = DEFAULT_THRESHOLD_VALUES,
    min_distance_um_values: Sequence[float] = DEFAULT_MIN_DISTANCE_UM_VALUES,
    max_nodes_values: Sequence[int] = DEFAULT_MAX_NODES_VALUES,
    match_max_distance_um: float = 7.0,
    sweep_csv_path: str = "/kaggle/working/detector_recall_preserving_sweep.csv",
    coverage_csv_path: str = "/kaggle/working/detector_stage_coverage.csv",
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full recall-preserving sweep + stage coverage diagnostics,
    then print the top-ranked configs and the high-recall/low-overprediction
    subset.
    """
    sweep_df, coverage_df = run_recall_preserving_calibration(
        n_samples=n_samples,
        percentile_high_values=percentile_high_values,
        threshold_values=threshold_values,
        min_distance_um_values=min_distance_um_values,
        max_nodes_values=max_nodes_values,
        match_max_distance_um=match_max_distance_um,
        sweep_csv_path=sweep_csv_path,
        coverage_csv_path=coverage_csv_path,
    )
    print_top_recall_preserving_results(sweep_df, coverage_df, top_n=top_n)
    print_high_recall_low_overprediction_configs(sweep_df, coverage_df)
    return sweep_df, coverage_df


if __name__ == "__main__":
    run_recall_preserving_diagnostics(n_samples=3)
