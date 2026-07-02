"""
Biohub - Cell Tracking During Development
Milestone 4: local metric approximation.

A local evaluator for predicted tracking graphs against the GEFF ground
truth, read with the manual offline Zarr/GEFF reader from Milestone 2 (no
zarr/numcodecs dependency). Matches predicted nodes to GT nodes per
timepoint via the Hungarian algorithm on physical (µm) distance, then
scores edge recovery (a CTC-style edge Jaccard approximation) and division
diagnostics. No detector, no ML, no competition submission - evaluation
only.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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


# --------------------------------------------------------------------------- #
# 3. GEFF (graph exchange file format) reader
# --------------------------------------------------------------------------- #
def _looks_like_geff_root(path: Path) -> bool:
    return (path / "nodes" / "ids" / "zarr.json").exists() and (path / "edges" / "ids" / "zarr.json").exists()


def find_geff_root(sample_zarr_dir: Path) -> Path | None:
    """Locate the GEFF store paired with a sample's image `.zarr` store.

    Tries, in order: a sibling `<name>.geff` directory, `geff`/`tracks`
    subdirectories inside the image store, then a bounded recursive search
    for any directory containing both `nodes/ids` and `edges/ids` arrays.
    """
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
    """Read the GEFF node/edge graph paired with a sample's image store.

    Returns (nodes_df, edges_df) with columns node_id,t,z,y,x and
    source_id,target_id respectively. Returns empty (but correctly-shaped)
    frames with a warning if no GEFF store can be found.
    """
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
# A. Physical distance
# --------------------------------------------------------------------------- #
def physical_distance_um(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance in µm between two (z, y, x) voxel coordinates."""
    scale = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])
    diff = (np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)) * scale
    return float(np.sqrt((diff ** 2).sum()))


def _pairwise_physical_distance_um(coords1_zyx: np.ndarray, coords2_zyx: np.ndarray) -> np.ndarray:
    """Vectorized (N,M) physical-distance cost matrix between two (z,y,x) coordinate arrays."""
    scale = np.array([VOXEL_SIZE_UM["z"], VOXEL_SIZE_UM["y"], VOXEL_SIZE_UM["x"]])
    diff = (coords1_zyx[:, None, :] - coords2_zyx[None, :, :]) * scale
    return np.sqrt((diff ** 2).sum(axis=-1))


# --------------------------------------------------------------------------- #
# B. Node matching by timepoint
# --------------------------------------------------------------------------- #
def match_nodes_by_timepoint(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    max_distance_um: float = 7.0,
) -> dict:
    """Match predicted nodes to GT nodes independently per timepoint using
    the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on
    physical (µm) distance, then reject any assignment farther than
    `max_distance_um`.

    Returns a dict with:
      pred_to_gt: dict mapping matched pred_node_id -> gt_node_id
      matches_df: DataFrame[t, pred_node_id, gt_node_id, distance_um]
      unmatched_pred_nodes: list of pred_node_id with no accepted match
      unmatched_gt_nodes: list of gt_node_id with no accepted match
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
# C. Edge Jaccard
# --------------------------------------------------------------------------- #
def compute_edge_jaccard(
    pred_nodes: pd.DataFrame,
    pred_edges: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    gt_edges: pd.DataFrame,
    max_distance_um: float = 7.0,
    match: dict | None = None,
) -> dict:
    """Score predicted edges against GT edges via the node matching from B.

    A predicted edge is a true positive if both its endpoints map to GT
    nodes and that (source, target) pair is an actual GT edge. Returns
    edge_TP, edge_FP, edge_FN, and edge_jaccard = TP / (TP+FP+FN) (1.0 if
    there is nothing to predict and nothing predicted).
    """
    if match is None:
        match = match_nodes_by_timepoint(pred_nodes, gt_nodes, max_distance_um)
    pred_to_gt = match["pred_to_gt"]

    gt_edge_set = set(zip(gt_edges["source_id"], gt_edges["target_id"])) if len(gt_edges) else set()

    tp = 0
    fp = 0
    recovered_gt_edges = set()

    for source_id, target_id in zip(pred_edges["source_id"], pred_edges["target_id"]):
        mapped_source = pred_to_gt.get(source_id)
        mapped_target = pred_to_gt.get(target_id)
        if (
            mapped_source is not None
            and mapped_target is not None
            and (mapped_source, mapped_target) in gt_edge_set
        ):
            tp += 1
            recovered_gt_edges.add((mapped_source, mapped_target))
        else:
            fp += 1

    fn = len(gt_edge_set) - len(recovered_gt_edges)

    denom = tp + fp + fn
    edge_jaccard = tp / denom if denom > 0 else 1.0

    return {"edge_TP": tp, "edge_FP": fp, "edge_FN": fn, "edge_jaccard": edge_jaccard}


# --------------------------------------------------------------------------- #
# D. Division diagnostics
# --------------------------------------------------------------------------- #
def compute_division_diagnostics(
    pred_edges: pd.DataFrame,
    gt_edges: pd.DataFrame,
    pred_to_gt: dict | None = None,
) -> dict:
    """A division node has out-degree >= 2 (>=2 outgoing edges as source).

    If `pred_to_gt` (from B) is given, predicted division node ids are
    projected through it before comparing against GT division node ids;
    otherwise pred/GT node ids are assumed to already share an id space
    (true e.g. for a perfect-prediction sanity check).
    """
    pred_out_degree = pred_edges["source_id"].value_counts() if len(pred_edges) else pd.Series(dtype=int)
    gt_out_degree = gt_edges["source_id"].value_counts() if len(gt_edges) else pd.Series(dtype=int)

    pred_division_ids = set(pred_out_degree[pred_out_degree >= 2].index)
    gt_division_ids = set(gt_out_degree[gt_out_degree >= 2].index)

    if pred_to_gt:
        mapped_pred_division_ids = {pred_to_gt[n] for n in pred_division_ids if n in pred_to_gt}
    else:
        mapped_pred_division_ids = pred_division_ids

    matched_division_ids = mapped_pred_division_ids & gt_division_ids
    n_matched = len(matched_division_ids)
    n_pred = len(pred_division_ids)
    n_gt = len(gt_division_ids)

    precision = (n_matched / n_pred) if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = (n_matched / n_gt) if n_gt else (1.0 if n_pred == 0 else 0.0)
    union = n_pred + n_gt - n_matched
    jaccard = (n_matched / union) if union > 0 else 1.0

    return {
        "pred_division_count": n_pred,
        "gt_division_count": n_gt,
        "matched_division_count": n_matched,
        "division_precision": precision,
        "division_recall": recall,
        "division_jaccard": jaccard,
    }


# --------------------------------------------------------------------------- #
# E. Node count diagnostics
# --------------------------------------------------------------------------- #
def compute_node_count_diagnostics(pred_nodes: pd.DataFrame, gt_nodes: pd.DataFrame) -> dict:
    """Overall + per-timepoint predicted-vs-GT node count comparison."""
    pred_count = len(pred_nodes)
    gt_count = len(gt_nodes)
    ratio = (pred_count / gt_count) if gt_count > 0 else (0.0 if pred_count == 0 else float("inf"))

    if pred_count == 0 and gt_count == 0:
        timepoints: list = []
    else:
        timepoints = sorted(set(pred_nodes["t"].unique()) | set(gt_nodes["t"].unique()))

    rows = [
        {
            "t": t,
            "pred_count": int((pred_nodes["t"] == t).sum()),
            "gt_count": int((gt_nodes["t"] == t).sum()),
        }
        for t in timepoints
    ]
    per_timepoint_df = pd.DataFrame(rows, columns=["t", "pred_count", "gt_count"])
    if len(per_timepoint_df):
        per_timepoint_df["diff"] = per_timepoint_df["pred_count"] - per_timepoint_df["gt_count"]
    else:
        per_timepoint_df["diff"] = pd.Series(dtype=int)

    overpredicted_timepoints = int((per_timepoint_df["diff"] > 0).sum()) if len(per_timepoint_df) else 0
    total_overprediction = int(per_timepoint_df["diff"].clip(lower=0).sum()) if len(per_timepoint_df) else 0

    return {
        "pred_node_count": pred_count,
        "gt_node_count": gt_count,
        "node_count_ratio": ratio,
        "per_timepoint_counts": per_timepoint_df,
        "overpredicted_timepoints": overpredicted_timepoints,
        "total_overprediction": total_overprediction,
    }


# --------------------------------------------------------------------------- #
# F. Main evaluator
# --------------------------------------------------------------------------- #
def evaluate_sample(
    sample_zarr_dir: Path,
    pred_nodes: pd.DataFrame,
    pred_edges: pd.DataFrame,
    max_distance_um: float = 7.0,
) -> dict:
    """Evaluate a predicted (pred_nodes, pred_edges) tracking graph against
    the GEFF ground truth for `sample_zarr_dir`.
    """
    sample_name = Path(sample_zarr_dir).stem
    gt_nodes, gt_edges = read_geff(sample_zarr_dir)

    match = match_nodes_by_timepoint(pred_nodes, gt_nodes, max_distance_um)
    edge_result = compute_edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, max_distance_um, match=match)
    division_result = compute_division_diagnostics(pred_edges, gt_edges, pred_to_gt=match["pred_to_gt"])
    node_count_result = compute_node_count_diagnostics(pred_nodes, gt_nodes)

    warnings: list[str] = []
    if len(gt_nodes) == 0:
        warnings.append("sample has no GT nodes")
    if len(pred_nodes) == 0:
        warnings.append("no predicted nodes")
    if len(pred_nodes) and len(match["unmatched_pred_nodes"]) == len(pred_nodes):
        warnings.append("no predicted node matched any GT node")
    ratio = node_count_result["node_count_ratio"]
    if node_count_result["gt_node_count"] > 0 and math.isfinite(ratio):
        if ratio > 3:
            warnings.append(f"predicted node count is {ratio:.1f}x GT (possible overprediction)")
        elif ratio < 0.33:
            warnings.append(f"predicted node count is only {ratio:.1%} of GT (possible underprediction)")

    return {
        "sample": sample_name,
        "num_pred_nodes": len(pred_nodes),
        "num_gt_nodes": len(gt_nodes),
        "num_pred_edges": len(pred_edges),
        "num_gt_edges": len(gt_edges),
        "node_matches": len(match["pred_to_gt"]),
        "unmatched_pred_nodes": len(match["unmatched_pred_nodes"]),
        "unmatched_gt_nodes": len(match["unmatched_gt_nodes"]),
        "edge_TP": edge_result["edge_TP"],
        "edge_FP": edge_result["edge_FP"],
        "edge_FN": edge_result["edge_FN"],
        "edge_jaccard": edge_result["edge_jaccard"],
        "pred_divisions": division_result["pred_division_count"],
        "gt_divisions": division_result["gt_division_count"],
        "division_diagnostics": division_result,
        "node_count_diagnostics": node_count_result,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# G. Tests
# --------------------------------------------------------------------------- #
def _shifted_copy(nodes_df: pd.DataFrame, dx: float = 0, dy: float = 0, dz: float = 0) -> pd.DataFrame:
    out = nodes_df.copy()
    out["x"] = out["x"] + dx
    out["y"] = out["y"] + dy
    out["z"] = out["z"] + dz
    return out


def test_perfect_prediction(sample_dir: Path) -> bool:
    """pred == gt exactly: every node/edge must match, edge_jaccard == 1.0."""
    gt_nodes, gt_edges = read_geff(sample_dir)
    result = evaluate_sample(sample_dir, gt_nodes.copy(), gt_edges.copy())
    ok = (
        result["node_matches"] == len(gt_nodes)
        and result["edge_TP"] == len(gt_edges)
        and result["edge_FP"] == 0
        and result["edge_FN"] == 0
        and result["edge_jaccard"] == 1.0
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] perfect prediction "
        f"(node_matches={result['node_matches']}/{len(gt_nodes)}, "
        f"edge_TP={result['edge_TP']} edge_FP={result['edge_FP']} edge_FN={result['edge_FN']} "
        f"edge_jaccard={result['edge_jaccard']:.3f})"
    )
    return ok


def test_shifted_prediction(sample_dir: Path, shift_voxels: float = 1.0) -> bool:
    """A 1-voxel x/y shift is well under the 7 um threshold - matches should hold."""
    gt_nodes, gt_edges = read_geff(sample_dir)
    pred_nodes = _shifted_copy(gt_nodes, dx=shift_voxels, dy=shift_voxels)
    result = evaluate_sample(sample_dir, pred_nodes, gt_edges.copy())
    ok = result["node_matches"] == len(gt_nodes) and result["edge_jaccard"] == 1.0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] shifted prediction (+{shift_voxels} vox x/y) "
        f"(node_matches={result['node_matches']}/{len(gt_nodes)}, edge_jaccard={result['edge_jaccard']:.3f})"
    )
    return ok


def test_bad_prediction(sample_dir: Path, shift_voxels: float = 10000.0) -> bool:
    """A huge shift (>> 7 um) should leave nodes unmatched and tank edge_jaccard."""
    gt_nodes, gt_edges = read_geff(sample_dir)
    pred_nodes = _shifted_copy(gt_nodes, dx=shift_voxels)
    result = evaluate_sample(sample_dir, pred_nodes, gt_edges.copy())
    ok = result["node_matches"] == 0 and result["edge_jaccard"] == 0.0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] bad prediction (+{shift_voxels} vox x) "
        f"(node_matches={result['node_matches']}, edge_jaccard={result['edge_jaccard']:.3f})"
    )
    return ok


def test_empty_prediction(sample_dir: Path) -> bool:
    """No predicted nodes/edges at all - must not crash, edge_FN == len(gt_edges)."""
    gt_nodes, gt_edges = read_geff(sample_dir)
    pred_nodes = gt_nodes.iloc[0:0].copy()
    pred_edges = gt_edges.iloc[0:0].copy()
    result = evaluate_sample(sample_dir, pred_nodes, pred_edges)
    ok = (
        result["num_pred_nodes"] == 0
        and result["edge_FN"] == len(gt_edges)
        and result["edge_TP"] == 0
        and result["edge_FP"] == 0
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] empty prediction "
        f"(edge_FN={result['edge_FN']}/{len(gt_edges)}, no crash)"
    )
    return ok


def run_local_metric_tests(sample_dir: Path | None = None) -> bool:
    """Run all four correctness tests on one train sample (the first
    discovered one, unless `sample_dir` is given).
    """
    if sample_dir is None:
        train_dirs = find_train_zarr_dirs()
        if not train_dirs:
            print("[error] no train samples found; cannot run local_metric tests")
            return False
        sample_dir = train_dirs[0]

    print(f"Running local_metric tests on sample: {sample_dir.stem}")
    results = [
        test_perfect_prediction(sample_dir),
        test_shifted_prediction(sample_dir),
        test_bad_prediction(sample_dir),
        test_empty_prediction(sample_dir),
    ]
    all_ok = all(results)
    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ({sum(results)}/{len(results)})")
    return all_ok


# --------------------------------------------------------------------------- #
# I. Perfect-prediction table over the first n train samples
# --------------------------------------------------------------------------- #
def evaluate_perfect_prediction_on_first_n_train_samples(
    n: int = 3, max_distance_um: float = 7.0
) -> pd.DataFrame:
    """Sanity-check the evaluator itself: for the first `n` train samples,
    evaluate GT-against-GT (a perfect prediction) and print a clean summary
    table - everything should score as a perfect match.
    """
    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train samples found.")
        return pd.DataFrame()

    rows = []
    for sample_dir in train_dirs[:n]:
        gt_nodes, gt_edges = read_geff(sample_dir)
        result = evaluate_sample(sample_dir, gt_nodes.copy(), gt_edges.copy(), max_distance_um)
        rows.append({
            "sample": result["sample"],
            "num_gt_nodes": result["num_gt_nodes"],
            "num_gt_edges": result["num_gt_edges"],
            "node_matches": result["node_matches"],
            "unmatched_pred_nodes": result["unmatched_pred_nodes"],
            "unmatched_gt_nodes": result["unmatched_gt_nodes"],
            "edge_TP": result["edge_TP"],
            "edge_FP": result["edge_FP"],
            "edge_FN": result["edge_FN"],
            "edge_jaccard": result["edge_jaccard"],
            "pred_divisions": result["pred_divisions"],
            "gt_divisions": result["gt_divisions"],
            "division_jaccard": result["division_diagnostics"]["division_jaccard"],
            "num_warnings": len(result["warnings"]),
        })

    summary_df = pd.DataFrame(rows)
    print("\nPerfect-prediction evaluation (sanity check of the evaluator itself):")
    print(summary_df.to_string(index=False))
    return summary_df


evaluate_perfect_prediction_on_first_n_train_samples(n=3)
