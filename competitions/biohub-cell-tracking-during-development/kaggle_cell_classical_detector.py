"""
Biohub - Cell Tracking During Development
Milestone 5: classical 3D detector baseline.

Detects candidate cell centers directly from the image volumes using
classical image processing (percentile normalization, 3D Gaussian
smoothing, local-maxima detection, intensity thresholding, and physical-
distance non-maximum suppression) - no ML, no U-Net, no division
detection, no competition submission. Built on the manual offline
Zarr/GEFF reader from Milestone 2 (no zarr/numcodecs dependency) and the
node-matching machinery from Milestone 4.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless: we only need savefig, never a display
import matplotlib.pyplot as plt
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
# 5. Node matching (Milestone 4's local metric, reused for Part C)
# --------------------------------------------------------------------------- #
def _pairwise_physical_distance_um(coords1_zyx: np.ndarray, coords2_zyx: np.ndarray) -> np.ndarray:
    """Vectorized (N,M) physical-distance cost matrix between two (z,y,x) coordinate arrays."""
    scale = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])
    diff = (coords1_zyx[:, None, :] - coords2_zyx[None, :, :]) * scale
    return np.sqrt((diff ** 2).sum(axis=-1))


def match_nodes_by_timepoint(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    max_distance_um: float = 7.0,
) -> dict:
    """Match predicted nodes to GT nodes independently per timepoint using
    the Hungarian algorithm on physical (µm) distance, rejecting matches
    farther than `max_distance_um`. See Milestone 4 (local_metric.py).
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

        unmatched_pred.extend(
            pred_t["node_id"].iloc[r] for r in range(len(pred_t)) if r not in matched_rows
        )
        unmatched_gt.extend(
            gt_t["node_id"].iloc[c] for c in range(len(gt_t)) if c not in matched_cols
        )

    matches_df = pd.DataFrame(match_rows, columns=["t", "pred_node_id", "gt_node_id", "distance_um"])

    return {
        "pred_to_gt": pred_to_gt,
        "matches_df": matches_df,
        "unmatched_pred_nodes": unmatched_pred,
        "unmatched_gt_nodes": unmatched_gt,
    }


# --------------------------------------------------------------------------- #
# Part A. Dataset statistics
# --------------------------------------------------------------------------- #
STATS_COLUMNS = [
    "sample", "t", "num_gt_nodes_total", "num_gt_edges_total", "num_timepoints_with_gt",
    "gt_node_count_at_t", "image_shape_z", "image_shape_y", "image_shape_x",
    "image_p1", "image_p50", "image_p99", "image_mean",
    "node_intensity_mean", "node_intensity_std", "node_intensity_min", "node_intensity_max",
    "estimated_density_per_mm3",
]


def _sample_intensity_at_nodes(image_zyx: np.ndarray, nodes_t: pd.DataFrame) -> np.ndarray:
    """Nearest-voxel image intensity at each node's (z,y,x) position, clipped to bounds."""
    Z, Y, X = image_zyx.shape
    zz = np.clip(np.round(nodes_t["z"].to_numpy()).astype(int), 0, Z - 1)
    yy = np.clip(np.round(nodes_t["y"].to_numpy()).astype(int), 0, Y - 1)
    xx = np.clip(np.round(nodes_t["x"].to_numpy()).astype(int), 0, X - 1)
    return image_zyx[zz, yy, xx]


def compute_dataset_stats_for_sample(sample_dir: Path) -> pd.DataFrame:
    """One row per timepoint that has GT nodes: node/edge counts, image
    intensity quantiles, intensity around GT nodes, and estimated node
    density (nodes per physical mm^3).
    """
    sample_name = sample_dir.stem
    gt_nodes, gt_edges = read_geff(sample_dir)
    num_gt_nodes_total = len(gt_nodes)
    num_gt_edges_total = len(gt_edges)

    if num_gt_nodes_total == 0:
        return pd.DataFrame(columns=STATS_COLUMNS)

    gt_timepoints = sorted(gt_nodes["t"].unique())
    num_timepoints_with_gt = len(gt_timepoints)

    rows = []
    for t in gt_timepoints:
        try:
            nodes_t = gt_nodes[gt_nodes["t"] == t]
            image_zyx = read_image_timepoint(sample_dir, t)
            Z, Y, X = image_zyx.shape

            p1, p50, p99 = (float(v) for v in np.percentile(image_zyx, [1, 50, 99]))
            node_intensities = _sample_intensity_at_nodes(image_zyx, nodes_t)

            physical_volume_mm3 = (
                (Z * VOXEL_SIZE_UM["z"]) * (Y * VOXEL_SIZE_UM["y"]) * (X * VOXEL_SIZE_UM["x"]) * 1e-9
            )
            density = (len(nodes_t) / physical_volume_mm3) if physical_volume_mm3 > 0 else float("nan")

            rows.append({
                "sample": sample_name,
                "t": t,
                "num_gt_nodes_total": num_gt_nodes_total,
                "num_gt_edges_total": num_gt_edges_total,
                "num_timepoints_with_gt": num_timepoints_with_gt,
                "gt_node_count_at_t": len(nodes_t),
                "image_shape_z": Z, "image_shape_y": Y, "image_shape_x": X,
                "image_p1": p1, "image_p50": p50, "image_p99": p99,
                "image_mean": float(image_zyx.mean()),
                "node_intensity_mean": float(node_intensities.mean()) if len(node_intensities) else float("nan"),
                "node_intensity_std": float(node_intensities.std()) if len(node_intensities) else float("nan"),
                "node_intensity_min": float(node_intensities.min()) if len(node_intensities) else float("nan"),
                "node_intensity_max": float(node_intensities.max()) if len(node_intensities) else float("nan"),
                "estimated_density_per_mm3": density,
            })
        except Exception as exc:
            print(f"[warn] could not compute stats for {sample_name!r} t={t}: {exc!r}")

    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def run_dataset_statistics(n: int = 10, csv_path: str = "/kaggle/working/detector_stats.csv") -> pd.DataFrame:
    """Compute + save dataset statistics for the first `n` train samples."""
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=STATS_COLUMNS)

    all_rows = []
    for sample_dir in train_dirs[:n]:
        df = compute_dataset_stats_for_sample(sample_dir)
        all_rows.append(df)
        print(f"{sample_dir.stem}: {len(df)} timepoint(s) with GT nodes processed")

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=STATS_COLUMNS)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved dataset statistics to {out_path} (shape={result.shape})")
    return result


# --------------------------------------------------------------------------- #
# Part B. Classical detector
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "percentile_low": 1.0,
    "percentile_high": 99.5,
    "gaussian_sigma_z": 1.0,
    "gaussian_sigma_yx": 1.5,
    "threshold": 0.5,
    "min_distance_um": 5.0,
    "max_nodes_per_timepoint": 500,
    "selected_timepoints": (0, 25, 50, 75, 90),
}

# Hard cap on raw (pre-NMS) candidates per volume - a defensive bound, not a
# tuning knob (see detect_candidates_in_volume).
MAX_RAW_CANDIDATES_BEFORE_NMS = 5000


def resolve_selected_timepoints(T: int, config: dict) -> list[int]:
    """Clip `config['selected_timepoints']` to the valid [0, T) range, or
    fall back to every timepoint if none are configured.
    """
    candidates = config.get("selected_timepoints")
    if candidates is None:
        return list(range(T))
    return sorted({t for t in candidates if 0 <= t < T})


def normalize_intensity(image: np.ndarray, percentile_low: float, percentile_high: float) -> np.ndarray:
    """Percentile-clip + rescale an image to [0, 1] float32."""
    lo, hi = np.percentile(image, [percentile_low, percentile_high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def _nms_by_physical_distance(candidates_df: pd.DataFrame, min_distance_um: float) -> pd.DataFrame:
    """Greedy non-maximum suppression: candidates must already be sorted by
    score descending. Suppresses any lower-score candidate within
    `min_distance_um` (physical distance) of an already-kept candidate.

    Uses a KD-tree (`scipy.spatial.cKDTree`) so this scales to thousands of
    raw candidates - an O(n^2) all-pairs distance matrix becomes impractical
    on noisy volumes where the threshold lets many raw peaks through.
    """
    if len(candidates_df) <= 1:
        return candidates_df

    scale = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])
    scaled_coords = candidates_df[["z", "y", "x"]].to_numpy(dtype=float) * scale
    tree = cKDTree(scaled_coords)
    keep_mask = np.ones(len(candidates_df), dtype=bool)

    for i in range(len(scaled_coords)):
        if not keep_mask[i]:
            continue
        neighbor_idx = tree.query_ball_point(scaled_coords[i], r=min_distance_um)
        for j in neighbor_idx:
            if j > i:
                keep_mask[j] = False

    return candidates_df[keep_mask]


def detect_candidates_in_volume(image_zyx: np.ndarray, config: dict) -> pd.DataFrame:
    """Classical candidate-center detection on a single (Z,Y,X) volume:
    percentile normalize -> 3D Gaussian smooth -> local maxima + intensity
    threshold -> physical-distance NMS -> cap at max_nodes_per_timepoint.

    Returns a DataFrame with columns [z, y, x, score] (score in [0, 1],
    the smoothed normalized intensity at that peak).
    """
    normalized = normalize_intensity(image_zyx, config["percentile_low"], config["percentile_high"])
    smoothed = gaussian_filter(
        normalized,
        sigma=(config["gaussian_sigma_z"], config["gaussian_sigma_yx"], config["gaussian_sigma_yx"]),
    )

    # peak_local_max finds local maxima (step 4) and applies the intensity
    # threshold (step 5) in one pass; min_distance=1 here just avoids
    # literally-adjacent duplicate voxels - the real NMS is physical-distance
    # based and happens afterward.
    coords = peak_local_max(
        smoothed,
        min_distance=1,
        threshold_abs=config["threshold"],
        exclude_border=False,
    )
    if len(coords) == 0:
        return pd.DataFrame(columns=["z", "y", "x", "score"])

    scores = smoothed[tuple(coords.T)]
    candidates = pd.DataFrame({
        "z": coords[:, 0].astype(float),
        "y": coords[:, 1].astype(float),
        "x": coords[:, 2].astype(float),
        "score": scores.astype(float),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    # Defensive cap on raw (pre-NMS) candidates: on a very noisy volume with
    # a permissive threshold, peak_local_max can return tens of thousands of
    # local maxima. NMS is still near-linear via the KD-tree, but bounding
    # the input keeps worst-case runtime predictable regardless of how noisy
    # a given volume/config combination turns out to be.
    if len(candidates) > MAX_RAW_CANDIDATES_BEFORE_NMS:
        candidates = candidates.iloc[:MAX_RAW_CANDIDATES_BEFORE_NMS]

    kept = _nms_by_physical_distance(candidates, config["min_distance_um"])

    max_n = config.get("max_nodes_per_timepoint")
    if max_n is not None and len(kept) > max_n:
        kept = kept.iloc[:max_n]

    return kept.reset_index(drop=True)


def assemble_candidates_over_timepoints(images_by_t: dict[int, np.ndarray], config: dict) -> pd.DataFrame:
    """Run `detect_candidates_in_volume` over each (t -> image) pair and
    assemble one globally node_id-unique DataFrame[node_id, t, z, y, x, score].
    """
    rows = []
    next_node_id = 0
    for t in sorted(images_by_t):
        candidates = detect_candidates_in_volume(images_by_t[t], config)
        for _, row in candidates.iterrows():
            rows.append({
                "node_id": next_node_id, "t": t,
                "z": row["z"], "y": row["y"], "x": row["x"], "score": row["score"],
            })
            next_node_id += 1
    return pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x", "score"])


def run_classical_detector_on_sample(
    sample_dir: Path,
    config: dict | None = None,
    timepoints: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run the classical detector over `timepoints` (or the config's
    `selected_timepoints`/full range if not given) for one sample.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    if timepoints is None:
        T = parse_array_metadata(sample_dir / "0")["shape"][0]
        timepoints = resolve_selected_timepoints(T, config)

    images_by_t = {t: read_image_timepoint(sample_dir, t) for t in timepoints}
    return assemble_candidates_over_timepoints(images_by_t, config)


# --------------------------------------------------------------------------- #
# Part C. Evaluation
# --------------------------------------------------------------------------- #
EVAL_COLUMNS = [
    "sample", "t", "num_gt_nodes", "num_pred_nodes", "num_matched",
    "unmatched_gt_nodes", "unmatched_pred_nodes",
    "mean_matched_distance_um", "median_matched_distance_um",
]


def build_eval_rows(
    sample_name: str,
    gt_nodes: pd.DataFrame,
    pred_nodes: pd.DataFrame,
    match_max_distance_um: float = 7.0,
) -> pd.DataFrame:
    """Pure dataframe-in, dataframe-out per-timepoint evaluation (no disk
    I/O) - shared by the real evaluation driver and the test suite.
    """
    gt_timepoints = sorted(gt_nodes["t"].unique()) if len(gt_nodes) else []
    if not gt_timepoints:
        return pd.DataFrame(columns=EVAL_COLUMNS)

    match = match_nodes_by_timepoint(pred_nodes, gt_nodes, max_distance_um=match_max_distance_um)
    matches_df = match["matches_df"]

    rows = []
    for t in gt_timepoints:
        gt_t = gt_nodes[gt_nodes["t"] == t]
        pred_t = pred_nodes[pred_nodes["t"] == t] if len(pred_nodes) else pred_nodes
        matches_t = matches_df[matches_df["t"] == t] if len(matches_df) else matches_df
        n_matched = len(matches_t)

        rows.append({
            "sample": sample_name,
            "t": t,
            "num_gt_nodes": len(gt_t),
            "num_pred_nodes": len(pred_t),
            "num_matched": n_matched,
            "unmatched_gt_nodes": len(gt_t) - n_matched,
            "unmatched_pred_nodes": len(pred_t) - n_matched,
            "mean_matched_distance_um": float(matches_t["distance_um"].mean()) if n_matched else float("nan"),
            "median_matched_distance_um": float(matches_t["distance_um"].median()) if n_matched else float("nan"),
        })

    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def evaluate_classical_detector_on_sample(
    sample_dir: Path,
    config: dict | None = None,
    match_max_distance_um: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the detector only on timepoints that have GT nodes, then
    evaluate. Returns (eval_df, gt_nodes, pred_nodes) so figures can reuse
    the same detection results without recomputing them.
    """
    gt_nodes, gt_edges = read_geff(sample_dir)
    gt_timepoints = sorted(gt_nodes["t"].unique()) if len(gt_nodes) else []

    if gt_timepoints:
        pred_nodes = run_classical_detector_on_sample(sample_dir, config=config, timepoints=gt_timepoints)
    else:
        pred_nodes = pd.DataFrame(columns=["node_id", "t", "z", "y", "x", "score"])

    eval_df = build_eval_rows(sample_dir.stem, gt_nodes, pred_nodes, match_max_distance_um)
    return eval_df, gt_nodes, pred_nodes


# --------------------------------------------------------------------------- #
# Part D. Diagnostics figures
# --------------------------------------------------------------------------- #
def save_detector_figure(
    sample_name: str, t: int, image_zyx: np.ndarray,
    gt_nodes_t: pd.DataFrame, pred_nodes_t: pd.DataFrame, matched_count: int,
    out_path: Path,
) -> None:
    """XY max-intensity projection with GT nodes (red) and predicted nodes
    (cyan) overlaid; title includes sample, t, GT/predicted/matched counts.
    """
    mip_xy = image_zyx.max(axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip_xy, cmap="gray")
    if len(gt_nodes_t):
        ax.scatter(
            gt_nodes_t["x"], gt_nodes_t["y"],
            s=50, facecolors="none", edgecolors="red", linewidths=1.4, label="GT",
        )
    if len(pred_nodes_t):
        ax.scatter(
            pred_nodes_t["x"], pred_nodes_t["y"],
            s=28, facecolors="none", edgecolors="cyan", linewidths=1.1, marker="s", label="Predicted",
        )
    if len(gt_nodes_t) or len(pred_nodes_t):
        ax.legend(loc="upper right", fontsize=8)

    ax.set_title(
        f"{sample_name} | t={t} | GT={len(gt_nodes_t)} pred={len(pred_nodes_t)} matched={matched_count}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_classical_detector_evaluation_and_figures(
    n: int = 3,
    config: dict | None = None,
    match_max_distance_um: float = 7.0,
    figures_dir: str = "/kaggle/working/detector_figures",
    csv_path: str = "/kaggle/working/classical_detector_eval.csv",
) -> pd.DataFrame:
    """Parts C + D combined: for the first `n` train samples, run the
    detector on GT timepoints, evaluate node matching, save one diagnostic
    figure per (sample, t), save the evaluation CSV, and print a clean table.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=EVAL_COLUMNS)

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    all_eval_dfs = []
    saved_figures: list[Path] = []

    for sample_dir in train_dirs[:n]:
        eval_df, gt_nodes, pred_nodes = evaluate_classical_detector_on_sample(
            sample_dir, config=config, match_max_distance_um=match_max_distance_um
        )
        all_eval_dfs.append(eval_df)

        gt_timepoints = sorted(gt_nodes["t"].unique()) if len(gt_nodes) else []
        for t in gt_timepoints:
            gt_t = gt_nodes[gt_nodes["t"] == t]
            pred_t = pred_nodes[pred_nodes["t"] == t] if len(pred_nodes) else pred_nodes
            matched_count = int(eval_df.loc[eval_df["t"] == t, "num_matched"].iloc[0]) if len(eval_df) else 0

            try:
                image_zyx = read_image_timepoint(sample_dir, t)
            except Exception as exc:
                print(f"[warn] could not read image for figure {sample_dir.stem!r} t={t}: {exc!r}")
                continue

            fig_path = figures_path / f"{sample_dir.stem}_t{t:03d}_detector.png"
            save_detector_figure(sample_dir.stem, t, image_zyx, gt_t, pred_t, matched_count, fig_path)
            saved_figures.append(fig_path)

    result = pd.concat(all_eval_dfs, ignore_index=True) if all_eval_dfs else pd.DataFrame(columns=EVAL_COLUMNS)

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"\nSaved {len(saved_figures)} figure(s) to {figures_path}")
    print(f"Saved evaluation to {out_path} (shape={result.shape})")
    print("\nClassical detector evaluation:")
    print(result.to_string(index=False))

    return result


# --------------------------------------------------------------------------- #
# Part E. Tests
# --------------------------------------------------------------------------- #
def _make_synthetic_volume(
    shape: tuple[int, int, int] = (10, 40, 40),
    blobs: Sequence[tuple[int, int, int, float]] = ((5, 20, 20, 5000.0),),
    background: float = 100.0,
    noise_sigma: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    """A small synthetic (Z,Y,X) volume with bright single-voxel blobs at
    given (z, y, x, amplitude) locations, for fast, disk-free unit tests.
    """
    rng = np.random.default_rng(seed)
    volume = background + rng.normal(0, noise_sigma, size=shape)
    volume = np.clip(volume, 0, None)
    for bz, by, bx, amplitude in blobs:
        volume[bz, by, bx] += amplitude
    return volume.astype(np.uint16)


def test_detector_columns() -> bool:
    """1. Detector returns a dataframe with the correct columns."""
    volume = _make_synthetic_volume()
    df = assemble_candidates_over_timepoints({0: volume}, DEFAULT_CONFIG)
    ok = list(df.columns) == ["node_id", "t", "z", "y", "x", "score"]
    print(f"  [{'PASS' if ok else 'FAIL'}] correct columns: {list(df.columns)}")
    return ok


def test_coordinates_in_bounds() -> bool:
    """2. Coordinates are inside the image bounds."""
    shape = (10, 40, 40)
    volume = _make_synthetic_volume(shape=shape, blobs=((2, 5, 5, 6000.0), (7, 35, 35, 6000.0), (5, 20, 20, 6000.0)))
    df = assemble_candidates_over_timepoints({0: volume}, DEFAULT_CONFIG)
    ok = (
        len(df) > 0
        and df["z"].between(0, shape[0] - 1).all()
        and df["y"].between(0, shape[1] - 1).all()
        and df["x"].between(0, shape[2] - 1).all()
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] {len(df)} candidate(s), all coordinates in bounds")
    return ok


def test_no_nan_values() -> bool:
    """3. No NaN values in the detector output."""
    volume = _make_synthetic_volume(blobs=((2, 5, 5, 6000.0), (7, 35, 35, 6000.0)))
    df = assemble_candidates_over_timepoints({0: volume, 1: volume}, DEFAULT_CONFIG)
    ok = len(df) > 0 and not df.isna().any().any()
    print(f"  [{'PASS' if ok else 'FAIL'}] no NaN values ({len(df)} rows)")
    return ok


def test_node_id_unique() -> bool:
    """4. node_id is unique across timepoints."""
    volume = _make_synthetic_volume(blobs=((2, 5, 5, 6000.0), (7, 35, 35, 6000.0)))
    df = assemble_candidates_over_timepoints({0: volume, 1: volume, 2: volume}, DEFAULT_CONFIG)
    ok = len(df) > 0 and df["node_id"].is_unique
    print(f"  [{'PASS' if ok else 'FAIL'}] node_id unique ({len(df)} rows)")
    return ok


def test_nms_reduces_duplicates() -> bool:
    """5. Physical-distance NMS collapses two very close peaks into one,
    while leaving a far-away peak untouched - tested directly against the
    NMS function (isolated from percentile-normalization/threshold
    behavior, which a single-voxel-spike-in-near-zero-noise synthetic
    volume would otherwise confound).
    """
    # Two candidates 2 voxels apart in x (~0.8125 um, well under 5 um) plus
    # one far away (~6 um in x); candidates must already be score-sorted.
    candidates = pd.DataFrame({
        "z": [5.0, 5.0, 5.0],
        "y": [20.0, 20.0, 20.0],
        "x": [20.0, 22.0, 35.0],
        "score": [0.95, 0.8, 0.9],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    merged = _nms_by_physical_distance(candidates, min_distance_um=5.0)
    unmerged = _nms_by_physical_distance(candidates, min_distance_um=0.01)

    ok = len(merged) == 2 and len(unmerged) == 3
    print(
        f"  [{'PASS' if ok else 'FAIL'}] NMS: {len(unmerged)} candidate(s) without suppression, "
        f"{len(merged)} after physical-distance NMS (close pair collapsed, far one kept)"
    )
    return ok


def test_empty_prediction_no_crash() -> bool:
    """6. Evaluation does not crash on empty predictions."""
    gt_nodes = pd.DataFrame({
        "node_id": [0, 1], "t": [0, 0], "z": [5.0, 6.0], "y": [10.0, 12.0], "x": [10.0, 12.0],
    })
    pred_nodes = pd.DataFrame(columns=["node_id", "t", "z", "y", "x", "score"])

    try:
        eval_df = build_eval_rows("synthetic_sample", gt_nodes, pred_nodes, match_max_distance_um=7.0)
        ok = (
            len(eval_df) == 1
            and int(eval_df["num_pred_nodes"].iloc[0]) == 0
            and int(eval_df["unmatched_gt_nodes"].iloc[0]) == 2
            and int(eval_df["num_matched"].iloc[0]) == 0
            and pd.isna(eval_df["mean_matched_distance_um"].iloc[0])
        )
    except Exception as exc:
        print(f"  [FAIL] evaluation raised on empty predictions: {exc!r}")
        return False

    print(f"  [{'PASS' if ok else 'FAIL'}] empty prediction handled without crashing")
    return ok


def run_classical_detector_tests() -> bool:
    print("Running classical_detector tests (synthetic, disk-free):")
    results = [
        test_detector_columns(),
        test_coordinates_in_bounds(),
        test_no_nan_values(),
        test_node_id_unique(),
        test_nms_reduces_duplicates(),
        test_empty_prediction_no_crash(),
    ]
    all_ok = all(results)
    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ({sum(results)}/{len(results)})")
    return all_ok


# --------------------------------------------------------------------------- #
# Top-level milestone driver
# --------------------------------------------------------------------------- #
def run_classical_detector_milestone(
    n_stats: int = 10, n_eval: int = 3, config: dict | None = None, match_max_distance_um: float = 7.0
) -> pd.DataFrame:
    print("=== Part A: Dataset statistics ===")
    run_dataset_statistics(n=n_stats)

    print("\n=== Parts B/C/D: Classical detector evaluation + diagnostics ===")
    return run_classical_detector_evaluation_and_figures(
        n=n_eval, config=config, match_max_distance_um=match_max_distance_um
    )


run_classical_detector_milestone(n_stats=10, n_eval=3)
