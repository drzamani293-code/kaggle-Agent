"""
Biohub - Cell Tracking During Development
Milestone 8: node-filter recall recovery with controlled edge FP.

Milestone 7's stage-by-stage diagnosis (real Kaggle data, first 3 train
samples, 174 GT nodes / 169 GT edges) found the bottleneck is emphatically
node filtering, not tracklet filtering or the per-frame edge cap:

  raw_node_coverage:       127/169 (0.751479) - detector is imperfect but fine
  filtered_node_coverage:   17/169 (0.100592) - node filter destroys 110 edges
  candidate_edge_coverage:  13/169 (0.076923) - loses only 4 more
  mnn_edge_coverage:        12/169 (0.071006) - loses only 1 more
  capped_edge_coverage:     12/169 (0.071006) - loses 0 more

Node filtering alone drops coverage from 127 to 17 - a loss of 110 GT
edges, dwarfing every downstream stage's loss combined (4+1+0=5). This
module directly attacks that bottleneck: the raw detector and its output
are reused/cached UNCHANGED for the first 3 train samples (no re-running
detection per combo), and only the node-filter parameters are swept more
permissively than the 6B/6C/6D baseline
(max_nodes_per_timepoint=75, score_quantile_per_timepoint=0.90,
absolute_score_threshold=0.05):

  max_nodes_per_timepoint:      [75, 100, 150, 200, 300, 500]
  score_quantile_per_timepoint: [0.90, 0.85, 0.80, 0.75, 0.70]
  absolute_score_threshold:     [0.03, 0.05]

6 x 5 x 2 = 60 node-filter combos - deliberately NOT a huge blind sweep.
The edge-filter stage is initially held fixed at the same real-data best
config (max_link_distance_um=7, require_mutual_nearest_neighbor=True,
max_edges_per_frame=50) so any gain can be attributed to node-filter
looseness alone, not to simultaneously loosening edge filtering too.

For every combo, both the Milestone-7-style stage diagnosis (raw/filtered/
candidate/mnn/capped GT-edge coverage) AND a full local edge_jaccard
approximation (edge_TP/FP/FN, precision, recall, using the final capped
edges as predicted edges and ALL filtered nodes as predicted nodes - the
same "keep all filtered nodes" philosophy as 6C's policy B, since
tracklet-level filtering is deliberately not applied at this milestone)
are computed and reported side by side.

Caching strategy: raw detection + GT node/edge tables + the GT-to-raw-node
match (which never changes across the sweep, since node filtering can
only ever shrink the raw pool) are computed ONCE per sample up front. Each
of the 60 combos then only re-runs node filtering + a single Hungarian
match-to-GT + one Hungarian candidate-edge pass per consecutive frame pair
- no detector re-runs, no GT re-reads.

Configs are ranked against four criteria pulled directly from Milestone 7/
8's numeric targets:
  A) filtered_node_coverage_count >= 40 (up from 17) with edge_FP < 2000
  B) edge_TP > 4 with edge_FP < 1000, < 2000, and < 5000
  C) edge_jaccard > 0.010667 (6C/6D's real-data best)
  D) recall >= 0.05 with controlled edge_FP

No final submission is built here, and this is NOT a huge blind sweep (60
combos, cached raw detection) - runtime target is well under 60 minutes.
No ML yet.
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
# 9. Conservative edge filtering (fixed at the same real-data best config)
# --------------------------------------------------------------------------- #
CANDIDATE_EDGE_COLUMNS = ["t", "source_id", "target_id", "source_score", "target_score", "distance_um", "is_mnn", "edge_score"]


def compute_candidate_edges(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Hungarian assignment + mutual-nearest-neighbor flag between every
    consecutive timepoint pair (see conservative_tracker.py for the full
    rationale) - computed once per sample, reused unfiltered/filtered as
    needed.
    """
    if len(nodes_df) == 0:
        return pd.DataFrame(columns=CANDIDATE_EDGE_COLUMNS)

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
        cost = _pairwise_physical_distance_um(cur_coords, nxt_coords)

        row_ind, col_ind = linear_sum_assignment(cost)
        nearest_nxt_for_cur = cost.argmin(axis=1)
        nearest_cur_for_nxt = cost.argmin(axis=0)

        cur_ids = cur["node_id"].to_numpy()
        cur_scores = cur["score"].to_numpy(dtype=float)
        nxt_ids = nxt["node_id"].to_numpy()
        nxt_scores = nxt["score"].to_numpy(dtype=float)

        for r, c in zip(row_ind, col_ind):
            distance = float(cost[r, c])
            is_mnn = bool(nearest_nxt_for_cur[r] == c and nearest_cur_for_nxt[c] == r)
            source_score = float(cur_scores[r])
            target_score = float(nxt_scores[c])
            edge_score = source_score * target_score / (1.0 + distance)
            rows.append({
                "t": int(t), "source_id": int(cur_ids[r]), "target_id": int(nxt_ids[c]),
                "source_score": source_score, "target_score": target_score,
                "distance_um": distance, "is_mnn": is_mnn, "edge_score": edge_score,
            })

    return pd.DataFrame(rows, columns=CANDIDATE_EDGE_COLUMNS)


def apply_edge_filter(
    candidate_edges_df: pd.DataFrame, max_link_distance_um: float, require_mnn: bool, max_edges_per_frame: int,
) -> pd.DataFrame:
    """Distance cutoff -> optional mutual-nearest-neighbor requirement ->
    top `max_edges_per_frame` edges per timepoint by edge_score. Returns
    the FULL candidate-edge rows (not just source_id/target_id) since
    tracklet-building needs distance_um/edge_score per retained edge.
    """
    if len(candidate_edges_df) == 0:
        return candidate_edges_df

    filtered = candidate_edges_df[candidate_edges_df["distance_um"] <= max_link_distance_um]
    if require_mnn:
        filtered = filtered[filtered["is_mnn"]]
    if len(filtered) == 0:
        return filtered

    filtered = filtered.sort_values(["t", "edge_score"], ascending=[True, False])
    filtered = filtered.groupby("t", sort=False, group_keys=False).head(max_edges_per_frame)
    return filtered.reset_index(drop=True)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 10. GT node matching (embedded copy of local_metric.py's evaluator)
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
# 11. Local edge_jaccard evaluation (embedded copy of local_metric.py's evaluator)
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



def _precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    return precision, recall


# --------------------------------------------------------------------------- #
# 12. GT-edge stage coverage primitives (same waterfall as Milestone 7)
# --------------------------------------------------------------------------- #
NODE_MATCH_MAX_DISTANCE_UM = 7.0

STAGE_NAMES = [
    "gt_total",
    "raw_node_coverage",
    "filtered_node_coverage",
    "candidate_edge_coverage",
    "mnn_edge_coverage",
    "capped_edge_coverage",
]


def _invert_pred_to_gt(pred_to_gt: dict) -> dict:
    """`pred_node_id -> gt_node_id` becomes `gt_node_id -> pred_node_id`. The
    underlying match is one-to-one within `max_distance_um` (Hungarian per
    timepoint, each row/column used at most once), so inversion is safe.
    """
    return {gt_id: pred_id for pred_id, gt_id in pred_to_gt.items()}


def _edge_pair_set(edges_df: pd.DataFrame) -> set[tuple[int, int]]:
    if len(edges_df) == 0:
        return set()
    return set(zip(edges_df["source_id"].astype(int), edges_df["target_id"].astype(int)))


# --------------------------------------------------------------------------- #
# 13. Per-sample raw-detection caching (computed ONCE per sample, reused by
#     every one of the 60 node-filter combos)
# --------------------------------------------------------------------------- #
def precompute_sample_cache(
    sample_zarr_dir: Path,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Runs the raw detector and reads GT exactly ONCE for one sample, and
    also computes the GT-to-raw-node match ONCE - this match is invariant
    across the whole node-filter sweep (node filtering can only ever
    shrink the raw pool, so raw_node_coverage is identical for every one
    of the 60 combos and never needs recomputing).
    """
    dataset_name = sample_zarr_dir.stem
    gt_nodes, gt_edges = read_geff(sample_zarr_dir)
    gt_edge_pairs = list(zip(gt_edges["source_id"], gt_edges["target_id"]))

    raw_nodes_df, _ = detect_all_raw_nodes_for_sample(sample_zarr_dir, detector_config, node_id_start=0)
    match_raw = match_nodes_by_timepoint(raw_nodes_df, gt_nodes, match_max_distance_um)
    gt_to_raw = _invert_pred_to_gt(match_raw["pred_to_gt"])
    raw_ok_flags = [(gu in gt_to_raw and gv in gt_to_raw) for gu, gv in gt_edge_pairs]

    n_timepoints = get_sample_timepoint_count(sample_zarr_dir)

    return {
        "dataset": dataset_name,
        "gt_nodes": gt_nodes,
        "gt_edges": gt_edges,
        "gt_edge_pairs": gt_edge_pairs,
        "raw_nodes_df": raw_nodes_df,
        "raw_ok_flags": raw_ok_flags,
        "raw_node_coverage_count": int(sum(raw_ok_flags)),
        "gt_node_count": len(gt_nodes),
        "gt_edge_count": len(gt_edge_pairs),
        "n_timepoints": n_timepoints,
    }


# --------------------------------------------------------------------------- #
# 14. Per-config evaluation (re-filters the cached raw pool; only stage
#     2-onward is recomputed - raw_node_coverage is reused from the cache)
# --------------------------------------------------------------------------- #
def evaluate_node_filter_config(
    cache: dict,
    max_nodes_per_timepoint: int,
    score_quantile_per_timepoint: float,
    absolute_score_threshold: float,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
) -> dict:
    """Evaluates ONE node-filter config against one sample's cache: applies
    node filtering to the cached raw pool, then computes filtered/
    candidate/mnn/capped GT-edge stage coverage (reusing the cache's
    already-known raw_node_coverage), plus a full local edge_jaccard/
    precision/recall using the final capped edges as predicted edges and
    ALL filtered nodes as predicted nodes (6C's "keep all filtered nodes"
    policy-B philosophy - tracklet-level filtering is deliberately not
    applied at this milestone).
    """
    filtered_nodes = apply_node_filter(
        cache["raw_nodes_df"], max_nodes_per_timepoint, score_quantile_per_timepoint, absolute_score_threshold,
    )[["node_id", "t", "z", "y", "x", "score"]].reset_index(drop=True)

    match_filtered = match_nodes_by_timepoint(filtered_nodes, cache["gt_nodes"], match_max_distance_um)
    gt_to_filtered = _invert_pred_to_gt(match_filtered["pred_to_gt"])

    candidate_edges = compute_candidate_edges(filtered_nodes)
    distance_ok_edges = candidate_edges[candidate_edges["distance_um"] <= edge_filter_config["max_link_distance_um"]]
    if edge_filter_config["require_mutual_nearest_neighbor"]:
        mnn_edges = distance_ok_edges[distance_ok_edges["is_mnn"]]
    else:
        mnn_edges = distance_ok_edges
    capped_edges = (
        mnn_edges.sort_values(["t", "edge_score"], ascending=[True, False])
        .groupby("t", sort=False, group_keys=False)
        .head(edge_filter_config["max_edges_per_frame"])
    )

    candidate_pairs = _edge_pair_set(distance_ok_edges)
    mnn_pairs = _edge_pair_set(mnn_edges)
    capped_pairs = _edge_pair_set(capped_edges)

    counts = {name: 0 for name in STAGE_NAMES}
    counts["gt_total"] = cache["gt_edge_count"]
    counts["raw_node_coverage"] = cache["raw_node_coverage_count"]

    for (gu, gv), raw_ok in zip(cache["gt_edge_pairs"], cache["raw_ok_flags"]):
        filtered_ok = gu in gt_to_filtered and gv in gt_to_filtered
        candidate_ok = mnn_ok = capped_ok = False
        if filtered_ok:
            pu, pv = gt_to_filtered[gu], gt_to_filtered[gv]
            candidate_ok = (pu, pv) in candidate_pairs
            mnn_ok = (pu, pv) in mnn_pairs
            capped_ok = (pu, pv) in capped_pairs

        if filtered_ok:
            counts["filtered_node_coverage"] += 1
        if candidate_ok:
            counts["candidate_edge_coverage"] += 1
        if mnn_ok:
            counts["mnn_edge_coverage"] += 1
        if capped_ok:
            counts["capped_edge_coverage"] += 1

    edge_result = compute_edge_jaccard(
        capped_edges[["source_id", "target_id"]], cache["gt_edges"], match_filtered["pred_to_gt"],
    )
    precision, recall = _precision_recall(edge_result["edge_TP"], edge_result["edge_FP"], edge_result["edge_FN"])

    return {
        "dataset": cache["dataset"],
        "counts": counts,
        "edge_TP": edge_result["edge_TP"], "edge_FP": edge_result["edge_FP"], "edge_FN": edge_result["edge_FN"],
        "edge_jaccard": edge_result["edge_jaccard"], "precision": precision, "recall": recall,
        "n_pred_nodes": len(filtered_nodes), "n_pred_edges": len(capped_edges),
        "n_timepoints": cache["n_timepoints"],
    }


# --------------------------------------------------------------------------- #
# 15. Full node-filter sweep (cached raw detection, 60 combos by default)
# --------------------------------------------------------------------------- #
DEFAULT_MAX_NODES_PER_TIMEPOINT_VALUES = (75, 100, 150, 200, 300, 500)
DEFAULT_SCORE_QUANTILE_PER_TIMEPOINT_VALUES = (0.90, 0.85, 0.80, 0.75, 0.70)
DEFAULT_ABSOLUTE_SCORE_THRESHOLD_VALUES = (0.03, 0.05)

EVAL_COLUMNS = [
    "max_nodes_per_timepoint", "score_quantile_per_timepoint", "absolute_score_threshold",
    "raw_node_coverage_count", "raw_node_coverage_frac",
    "filtered_node_coverage_count", "filtered_node_coverage_frac",
    "candidate_edge_coverage_count", "candidate_edge_coverage_frac",
    "mnn_edge_coverage_count", "mnn_edge_coverage_frac",
    "capped_edge_coverage_count", "capped_edge_coverage_frac",
    "edge_jaccard", "edge_TP", "edge_FP", "edge_FN", "precision", "recall",
    "avg_pred_nodes_per_timepoint", "avg_edges_per_frame",
    "runtime_seconds",
]

STAGE_BY_CONFIG_COLUMNS = [
    "max_nodes_per_timepoint", "score_quantile_per_timepoint", "absolute_score_threshold", "dataset",
    "gt_node_count", "gt_edge_count",
    "raw_node_coverage_count", "filtered_node_coverage_count", "candidate_edge_coverage_count",
    "mnn_edge_coverage_count", "capped_edge_coverage_count",
]


def run_milestone8_eval(
    n_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    max_nodes_per_timepoint_values: Sequence[int] = DEFAULT_MAX_NODES_PER_TIMEPOINT_VALUES,
    score_quantile_per_timepoint_values: Sequence[float] = DEFAULT_SCORE_QUANTILE_PER_TIMEPOINT_VALUES,
    absolute_score_threshold_values: Sequence[float] = DEFAULT_ABSOLUTE_SCORE_THRESHOLD_VALUES,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    match_max_distance_um: float = NODE_MATCH_MAX_DISTANCE_UM,
    eval_csv_path: str = "/kaggle/working/milestone8_node_filter_eval.csv",
    stage_csv_path: str = "/kaggle/working/milestone8_stage_diagnosis_by_config.csv",
    progress_every: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Caches raw detection + GT once per sample, then sweeps the
    node-filter grid (60 combos by default) - only node filtering, GT
    matching, and one Hungarian candidate-edge pass are recomputed per
    combo; the detector itself never re-runs.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame(columns=EVAL_COLUMNS), pd.DataFrame(columns=STAGE_BY_CONFIG_COLUMNS)

    samples = train_dirs[:n_samples]
    n_combos = (
        len(max_nodes_per_timepoint_values) * len(score_quantile_per_timepoint_values)
        * len(absolute_score_threshold_values)
    )
    print(f"Sweeping {n_combos} node-filter combo(s) over {len(samples)} sample(s) (cached raw detection).")

    print("Caching raw detection + GT + GT-to-raw-node match for each sample...")
    caches = []
    for sample_dir in samples:
        t0 = time.time()
        cache = precompute_sample_cache(sample_dir, detector_config, match_max_distance_um)
        caches.append(cache)
        print(
            f"  {cache['dataset']}: {len(cache['raw_nodes_df'])} raw node(s), "
            f"GT nodes={cache['gt_node_count']}, GT edges={cache['gt_edge_count']}, "
            f"raw_node_coverage={cache['raw_node_coverage_count']}/{cache['gt_edge_count']} "
            f"cached in {time.time() - t0:.1f}s"
        )

    total_timepoints = sum(c["n_timepoints"] for c in caches)
    total_frame_pairs = sum(max(0, c["n_timepoints"] - 1) for c in caches)
    total_gt_edges = sum(c["gt_edge_count"] for c in caches)

    eval_rows: list[dict] = []
    stage_rows: list[dict] = []
    combo_idx = 0
    sweep_t0 = time.time()

    for max_nodes in max_nodes_per_timepoint_values:
        for score_quantile in score_quantile_per_timepoint_values:
            for abs_threshold in absolute_score_threshold_values:
                combo_idx += 1
                row_t0 = time.time()

                per_sample_results = [
                    evaluate_node_filter_config(
                        cache, max_nodes, score_quantile, abs_threshold, edge_filter_config, match_max_distance_um,
                    )
                    for cache in caches
                ]

                agg_counts = {name: sum(r["counts"][name] for r in per_sample_results) for name in STAGE_NAMES}
                edge_TP = sum(r["edge_TP"] for r in per_sample_results)
                edge_FP = sum(r["edge_FP"] for r in per_sample_results)
                edge_FN = sum(r["edge_FN"] for r in per_sample_results)
                denom = edge_TP + edge_FP + edge_FN
                edge_jaccard = edge_TP / denom if denom > 0 else 1.0
                precision, recall = _precision_recall(edge_TP, edge_FP, edge_FN)
                total_pred_nodes = sum(r["n_pred_nodes"] for r in per_sample_results)
                total_pred_edges = sum(r["n_pred_edges"] for r in per_sample_results)

                def frac(n: int) -> float:
                    return (n / total_gt_edges) if total_gt_edges else float("nan")

                eval_rows.append({
                    "max_nodes_per_timepoint": max_nodes,
                    "score_quantile_per_timepoint": score_quantile,
                    "absolute_score_threshold": abs_threshold,
                    "raw_node_coverage_count": agg_counts["raw_node_coverage"],
                    "raw_node_coverage_frac": frac(agg_counts["raw_node_coverage"]),
                    "filtered_node_coverage_count": agg_counts["filtered_node_coverage"],
                    "filtered_node_coverage_frac": frac(agg_counts["filtered_node_coverage"]),
                    "candidate_edge_coverage_count": agg_counts["candidate_edge_coverage"],
                    "candidate_edge_coverage_frac": frac(agg_counts["candidate_edge_coverage"]),
                    "mnn_edge_coverage_count": agg_counts["mnn_edge_coverage"],
                    "mnn_edge_coverage_frac": frac(agg_counts["mnn_edge_coverage"]),
                    "capped_edge_coverage_count": agg_counts["capped_edge_coverage"],
                    "capped_edge_coverage_frac": frac(agg_counts["capped_edge_coverage"]),
                    "edge_jaccard": edge_jaccard, "edge_TP": edge_TP, "edge_FP": edge_FP, "edge_FN": edge_FN,
                    "precision": precision, "recall": recall,
                    "avg_pred_nodes_per_timepoint": (total_pred_nodes / total_timepoints) if total_timepoints else float("nan"),
                    "avg_edges_per_frame": (total_pred_edges / total_frame_pairs) if total_frame_pairs else float("nan"),
                    "runtime_seconds": time.time() - row_t0,
                })

                for r in per_sample_results:
                    stage_rows.append({
                        "max_nodes_per_timepoint": max_nodes,
                        "score_quantile_per_timepoint": score_quantile,
                        "absolute_score_threshold": abs_threshold,
                        "dataset": r["dataset"],
                        "gt_node_count": next(c["gt_node_count"] for c in caches if c["dataset"] == r["dataset"]),
                        "gt_edge_count": r["counts"]["gt_total"],
                        "raw_node_coverage_count": r["counts"]["raw_node_coverage"],
                        "filtered_node_coverage_count": r["counts"]["filtered_node_coverage"],
                        "candidate_edge_coverage_count": r["counts"]["candidate_edge_coverage"],
                        "mnn_edge_coverage_count": r["counts"]["mnn_edge_coverage"],
                        "capped_edge_coverage_count": r["counts"]["capped_edge_coverage"],
                    })

                if combo_idx % progress_every == 0 or combo_idx == n_combos:
                    elapsed = time.time() - sweep_t0
                    eta = elapsed / combo_idx * (n_combos - combo_idx)
                    print(f"  [{combo_idx}/{n_combos} node-filter combos] elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    eval_df = pd.DataFrame(eval_rows, columns=EVAL_COLUMNS)
    stage_df = pd.DataFrame(stage_rows, columns=STAGE_BY_CONFIG_COLUMNS)

    eval_out = Path(eval_csv_path)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(eval_out, index=False)
    print(f"\nSaved node-filter eval sweep ({len(eval_df)} row(s)) to {eval_out}")

    stage_out = Path(stage_csv_path)
    stage_out.parent.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(stage_out, index=False)
    print(f"Saved per-config stage diagnosis ({len(stage_df)} row(s)) to {stage_out}")

    return eval_df, stage_df


# --------------------------------------------------------------------------- #
# 16. Reporting + best-candidate selection against Milestone 8's 4 criteria
# --------------------------------------------------------------------------- #
CONFIG_COLUMNS_FOR_PRINT = [
    "max_nodes_per_timepoint", "score_quantile_per_timepoint", "absolute_score_threshold",
    "filtered_node_coverage_count", "candidate_edge_coverage_count", "mnn_edge_coverage_count",
    "capped_edge_coverage_count", "edge_jaccard", "edge_TP", "edge_FP", "edge_FN", "precision", "recall",
]


def _print_config_table(df: pd.DataFrame, max_rows: int) -> None:
    if len(df) == 0:
        print("  (no configs match)")
        return
    for _, r in df[CONFIG_COLUMNS_FOR_PRINT].head(max_rows).iterrows():
        print(
            f"  max_nodes={r['max_nodes_per_timepoint']} score_q={r['score_quantile_per_timepoint']} "
            f"abs_thresh={r['absolute_score_threshold']} | "
            f"filtered_cov={r['filtered_node_coverage_count']} candidate_cov={r['candidate_edge_coverage_count']} "
            f"mnn_cov={r['mnn_edge_coverage_count']} capped_cov={r['capped_edge_coverage_count']} | "
            f"edge_jaccard={r['edge_jaccard']:.6f} TP={r['edge_TP']} FP={r['edge_FP']} FN={r['edge_FN']} "
            f"precision={r['precision']:.6f} recall={r['recall']:.6f}"
        )


def print_summary_reports(eval_df: pd.DataFrame) -> dict:
    """Prints the requested breakdowns: top 30 by edge_jaccard, plus
    configs meeting edge_FP<1000/2000, filtered_node_coverage_count>=40,
    edge_TP>4, and recall>=0.05.
    """
    if len(eval_df) == 0:
        print("[warn] no eval rows to report on.")
        return {}

    by_jaccard = eval_df.sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)

    fp_lt_1000 = by_jaccard[by_jaccard["edge_FP"] < 1000]
    fp_lt_2000 = by_jaccard[by_jaccard["edge_FP"] < 2000]
    filtered_cov_ge_40 = by_jaccard[by_jaccard["filtered_node_coverage_count"] >= 40]
    tp_gt_4 = by_jaccard[by_jaccard["edge_TP"] > 4]
    recall_ge_05 = by_jaccard[by_jaccard["recall"] >= 0.05]

    print(f"\n=== Top 30 configs by edge_jaccard (of {len(eval_df)} total) ===")
    _print_config_table(by_jaccard, 30)

    print(f"\n=== Top configs with edge_FP < 1000 ({len(fp_lt_1000)} of {len(eval_df)}) ===")
    _print_config_table(fp_lt_1000, 15)

    print(f"\n=== Top configs with edge_FP < 2000 ({len(fp_lt_2000)} of {len(eval_df)}) ===")
    _print_config_table(fp_lt_2000, 15)

    print(f"\n=== Top configs with filtered_node_coverage_count >= 40 ({len(filtered_cov_ge_40)} of {len(eval_df)}) ===")
    _print_config_table(filtered_cov_ge_40, 15)

    print(f"\n=== Top configs with edge_TP > 4 ({len(tp_gt_4)} of {len(eval_df)}) ===")
    _print_config_table(tp_gt_4, 15)

    print(f"\n=== Top configs with recall >= 0.05 ({len(recall_ge_05)} of {len(eval_df)}) ===")
    _print_config_table(recall_ge_05, 15)

    return {
        "by_edge_jaccard": by_jaccard,
        "edge_fp_lt_1000": fp_lt_1000,
        "edge_fp_lt_2000": fp_lt_2000,
        "filtered_coverage_ge_40": filtered_cov_ge_40,
        "edge_tp_gt_4": tp_gt_4,
        "recall_ge_05": recall_ge_05,
    }


BEST_CANDIDATES_COLUMNS = EVAL_COLUMNS + [
    "meets_criterion_A", "meets_criterion_B_fp1000", "meets_criterion_B_fp2000", "meets_criterion_B_fp5000",
    "meets_criterion_C", "meets_criterion_D", "criteria_met",
]


def select_best_candidates(
    eval_df: pd.DataFrame,
    baseline_filtered_node_coverage_count: int = 17,
    target_filtered_node_coverage_count: int = 40,
    baseline_edge_jaccard: float = 0.010667,
    csv_path: str = "/kaggle/working/milestone8_best_candidates.csv",
) -> pd.DataFrame:
    """Flags every config against Milestone 8's 4 ranking criteria:
      A) filtered_node_coverage_count >= 40 (up from the 6C/6D baseline's
         17) with edge_FP < 2000
      B) edge_TP > 4 with edge_FP < 1000 / < 2000 / < 5000
      C) edge_jaccard > 0.010667 (6C/6D's real-data best)
      D) recall >= 0.05 with controlled edge_FP (< 5000)
    Saves every config meeting at least one criterion (sorted by
    edge_jaccard descending) to `csv_path`.
    """
    if len(eval_df) == 0:
        print("[warn] no eval rows to select best candidates from.")
        return pd.DataFrame(columns=BEST_CANDIDATES_COLUMNS)

    df = eval_df.copy()
    df["meets_criterion_A"] = (df["filtered_node_coverage_count"] >= target_filtered_node_coverage_count) & (df["edge_FP"] < 2000)
    df["meets_criterion_B_fp1000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 1000)
    df["meets_criterion_B_fp2000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 2000)
    df["meets_criterion_B_fp5000"] = (df["edge_TP"] > 4) & (df["edge_FP"] < 5000)
    df["meets_criterion_C"] = df["edge_jaccard"] > baseline_edge_jaccard
    df["meets_criterion_D"] = (df["recall"] >= 0.05) & (df["edge_FP"] < 5000)

    criterion_cols = [
        "meets_criterion_A", "meets_criterion_B_fp1000", "meets_criterion_B_fp2000",
        "meets_criterion_B_fp5000", "meets_criterion_C", "meets_criterion_D",
    ]
    df["criteria_met"] = df[criterion_cols].apply(lambda row: ",".join(c for c in criterion_cols if row[c]), axis=1)

    any_criterion = df[criterion_cols].any(axis=1)
    best = df[any_criterion].sort_values(
        by=["edge_jaccard", "precision", "recall"], ascending=[False, False, False]
    ).reset_index(drop=True)

    print(f"\n=== Best-candidate selection (configs meeting >= 1 of Milestone 8's 4 criteria) ===")
    print(f"  {len(best)} of {len(eval_df)} config(s) meet at least one criterion:")
    print(f"    A) filtered_node_coverage_count >= {target_filtered_node_coverage_count} (baseline {baseline_filtered_node_coverage_count}) and edge_FP < 2000: {int(df['meets_criterion_A'].sum())}")
    print(f"    B) edge_TP > 4 and edge_FP < 1000: {int(df['meets_criterion_B_fp1000'].sum())}, < 2000: {int(df['meets_criterion_B_fp2000'].sum())}, < 5000: {int(df['meets_criterion_B_fp5000'].sum())}")
    print(f"    C) edge_jaccard > {baseline_edge_jaccard}: {int(df['meets_criterion_C'].sum())}")
    print(f"    D) recall >= 0.05 with edge_FP < 5000: {int(df['meets_criterion_D'].sum())}")

    if len(best):
        top = best.iloc[0]
        print(
            f"\n  Top candidate overall (by edge_jaccard): max_nodes={top['max_nodes_per_timepoint']} "
            f"score_q={top['score_quantile_per_timepoint']} abs_thresh={top['absolute_score_threshold']} "
            f"-> edge_jaccard={top['edge_jaccard']:.6f}, edge_TP={top['edge_TP']}, edge_FP={top['edge_FP']}, "
            f"recall={top['recall']:.6f}, filtered_node_coverage_count={top['filtered_node_coverage_count']}, "
            f"criteria_met=[{top['criteria_met']}]"
        )
    else:
        print("\n  No config met any of the 4 criteria.")

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)
    print(f"Saved best candidates ({len(best)} row(s)) to {out_path}")

    return best
# --------------------------------------------------------------------------- #
# 17. Tests
# --------------------------------------------------------------------------- #
def run_milestone8_tests() -> None:
    """Correctness tests for the per-config evaluation, sweep-grid size,
    and best-candidate selection logic, using small synthetic fixtures (no
    disk I/O) so exact behavior can be verified by hand.
    """
    # Test 1: _invert_pred_to_gt / _edge_pair_set correctness (incl. empty case).
    assert _invert_pred_to_gt({}) == {}
    assert _invert_pred_to_gt({10: 100, 11: 101}) == {100: 10, 101: 11}
    assert _edge_pair_set(pd.DataFrame(columns=["source_id", "target_id"])) == set()
    assert _edge_pair_set(pd.DataFrame({"source_id": [1, 2], "target_id": [3, 4]})) == {(1, 3), (2, 4)}

    # Test 2: the default node-filter grid has the expected size
    # (6 x 5 x 2 = 60 combos - deliberately not a huge blind sweep).
    n_combos = (
        len(DEFAULT_MAX_NODES_PER_TIMEPOINT_VALUES) * len(DEFAULT_SCORE_QUANTILE_PER_TIMEPOINT_VALUES)
        * len(DEFAULT_ABSOLUTE_SCORE_THRESHOLD_VALUES)
    )
    assert n_combos == 60, f"expected 60 node-filter combos, got {n_combos}"
    assert min(DEFAULT_MAX_NODES_PER_TIMEPOINT_VALUES) == 75, "75 is the 6C/6D baseline, kept as the strictest end"
    assert max(DEFAULT_MAX_NODES_PER_TIMEPOINT_VALUES) == 500
    assert min(DEFAULT_SCORE_QUANTILE_PER_TIMEPOINT_VALUES) == 0.70, "should allow a looser quantile than the 0.90 baseline"

    # Test 3: evaluate_node_filter_config on a small hand-built cache. Two
    # GT edges: (g0,g1) whose raw detections are well within a lenient
    # node filter's cap, and (g2,g3) whose raw detections are LOW-SCORE
    # and get dropped once the node filter's absolute_score_threshold is
    # raised - this must show up specifically as a filtered_node_coverage
    # drop, with raw_node_coverage (from the cache) unaffected.
    gt_nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.0, 10.0, 200.0, 210.0], "x": [0.0] * 4,
    })
    gt_edges = pd.DataFrame({"source_id": [0, 2], "target_id": [1, 3]})
    raw_nodes_df = pd.DataFrame({
        "node_id": [100, 101, 102, 103], "t": [0, 1, 0, 1],
        "z": [0.0] * 4, "y": [0.1, 10.1, 200.1, 210.1], "x": [0.0] * 4,
        "score": [0.9, 0.9, 0.02, 0.02],
    })
    gt_edge_pairs = list(zip(gt_edges["source_id"], gt_edges["target_id"]))
    match_raw = match_nodes_by_timepoint(raw_nodes_df, gt_nodes, 7.0)
    gt_to_raw = _invert_pred_to_gt(match_raw["pred_to_gt"])
    raw_ok_flags = [(gu in gt_to_raw and gv in gt_to_raw) for gu, gv in gt_edge_pairs]
    cache = {
        "dataset": "fixture", "gt_nodes": gt_nodes, "gt_edges": gt_edges, "gt_edge_pairs": gt_edge_pairs,
        "raw_nodes_df": raw_nodes_df, "raw_ok_flags": raw_ok_flags,
        "raw_node_coverage_count": int(sum(raw_ok_flags)), "gt_node_count": len(gt_nodes),
        "gt_edge_count": len(gt_edge_pairs), "n_timepoints": 2,
    }
    assert cache["raw_node_coverage_count"] == 2, "both GT edges' endpoints are detected by the raw detector"

    lenient_result = evaluate_node_filter_config(cache, max_nodes_per_timepoint=10, score_quantile_per_timepoint=0.0, absolute_score_threshold=0.0)
    assert lenient_result["counts"]["raw_node_coverage"] == 2
    assert lenient_result["counts"]["filtered_node_coverage"] == 2, "a lenient filter keeps all 4 raw nodes"

    strict_result = evaluate_node_filter_config(cache, max_nodes_per_timepoint=10, score_quantile_per_timepoint=0.0, absolute_score_threshold=0.5)
    assert strict_result["counts"]["raw_node_coverage"] == 2, "raw coverage is unaffected by node filtering"
    assert strict_result["counts"]["filtered_node_coverage"] == 1, "only the high-score (g0,g1) pair survives abs_threshold=0.5"

    # Test 4: select_best_candidates flags configs correctly on a small
    # hand-built eval_df fixture.
    fixture_eval = pd.DataFrame([
        {  # meets A, C, D, and B at every FP threshold
            "max_nodes_per_timepoint": 300, "score_quantile_per_timepoint": 0.70, "absolute_score_threshold": 0.03,
            "raw_node_coverage_count": 127, "raw_node_coverage_frac": 0.75,
            "filtered_node_coverage_count": 60, "filtered_node_coverage_frac": 0.35,
            "candidate_edge_coverage_count": 50, "candidate_edge_coverage_frac": 0.30,
            "mnn_edge_coverage_count": 45, "mnn_edge_coverage_frac": 0.27,
            "capped_edge_coverage_count": 40, "capped_edge_coverage_frac": 0.24,
            "edge_jaccard": 0.05, "edge_TP": 20, "edge_FP": 500, "edge_FN": 30, "precision": 0.038, "recall": 0.4,
            "avg_pred_nodes_per_timepoint": 250.0, "avg_edges_per_frame": 30.0, "runtime_seconds": 1.0,
        },
        {  # meets nothing (identical to 6C/6D baseline behavior)
            "max_nodes_per_timepoint": 75, "score_quantile_per_timepoint": 0.90, "absolute_score_threshold": 0.05,
            "raw_node_coverage_count": 127, "raw_node_coverage_frac": 0.75,
            "filtered_node_coverage_count": 17, "filtered_node_coverage_frac": 0.10,
            "candidate_edge_coverage_count": 13, "candidate_edge_coverage_frac": 0.077,
            "mnn_edge_coverage_count": 12, "mnn_edge_coverage_frac": 0.071,
            "capped_edge_coverage_count": 12, "capped_edge_coverage_frac": 0.071,
            "edge_jaccard": 0.010667, "edge_TP": 4, "edge_FP": 206, "edge_FN": 165, "precision": 0.019, "recall": 0.024,
            "avg_pred_nodes_per_timepoint": 60.0, "avg_edges_per_frame": 19.0, "runtime_seconds": 1.0,
        },
    ], columns=EVAL_COLUMNS)
    best = select_best_candidates(fixture_eval, csv_path="/tmp/milestone8_best_candidates_test.csv")
    assert len(best) == 1, "only the improved config should meet any criterion"
    row = best.iloc[0]
    assert row["meets_criterion_A"] and row["meets_criterion_C"] and row["meets_criterion_D"]
    assert row["meets_criterion_B_fp1000"] and row["meets_criterion_B_fp2000"] and row["meets_criterion_B_fp5000"]

    print("All milestone8_node_filter_recovery tests passed (4/4).")


# --------------------------------------------------------------------------- #
# 18. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone8_pipeline(
    n_samples: int = 3,
    detector_config: dict = DEFAULT_DETECTOR_CONFIG,
    edge_filter_config: dict = BEST_CONSERVATIVE_EDGE_FILTER_CONFIG,
    baseline_filtered_node_coverage_count: int = 17,
    target_filtered_node_coverage_count: int = 40,
    baseline_edge_jaccard: float = 0.010667,
) -> dict:
    """Runs the tests, the 60-combo node-filter sweep (cached raw
    detection), the summary reports, and best-candidate selection against
    Milestone 8's 4 criteria. Builds no submission. Never calls any Kaggle
    submission API.
    """
    print("=== Self-test: per-config evaluation + best-candidate selection unit tests ===")
    run_milestone8_tests()

    print("\n=== Milestone 8: node-filter recall-recovery sweep (first %d train sample(s)) ===" % n_samples)
    eval_df, stage_df = run_milestone8_eval(
        n_samples=n_samples, detector_config=detector_config, edge_filter_config=edge_filter_config,
    )

    print("\n=== Summary reports ===")
    print_summary_reports(eval_df)

    print("\n=== Best-candidate selection ===")
    best_candidates_df = select_best_candidates(
        eval_df,
        baseline_filtered_node_coverage_count=baseline_filtered_node_coverage_count,
        target_filtered_node_coverage_count=target_filtered_node_coverage_count,
        baseline_edge_jaccard=baseline_edge_jaccard,
    )

    return {"eval_df": eval_df, "stage_df": stage_df, "best_candidates_df": best_candidates_df}


run_milestone8_pipeline(n_samples=3)
