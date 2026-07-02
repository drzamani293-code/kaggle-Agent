"""
Biohub - Cell Tracking During Development
Milestone 3: visualization diagnostics.

For the first 3 train samples, reads image timepoints with the manual
offline Zarr v3 reader (no zarr/numcodecs dependency), builds an XY
max-intensity-projection over z, overlays the GT nodes for that timepoint,
and saves one figure per timepoint plus a CSV summary. No ML, no detector -
just a fast visual sanity check of the reader's output.
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


# --------------------------------------------------------------------------- #
# 3. Whole-array and single-timepoint readers
# --------------------------------------------------------------------------- #
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
    fetching only the single chunk that covers timepoint `t`
    (e.g. `sample.zarr/0/c/{t}/0/0/0`) rather than the whole array.
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
# 4. GEFF (graph exchange file format) reader
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
# 5. Train-sample discovery
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
# 6. Timepoint selection
# --------------------------------------------------------------------------- #
def select_timepoints(
    T: int,
    nodes_df: pd.DataFrame,
    base_candidates: Sequence[int] = (0, 25, 50, 75, 90),
    max_timepoints: int = 150,
) -> list[int]:
    """Union of the fixed candidate timepoints (clipped to the valid range)
    and every timepoint where a GT node actually exists, deduplicated and
    sorted. `max_timepoints` is a defensive cap (generously above any
    realistic per-sample GT timepoint count) purely to guard against
    pathological inputs; the base candidates are always kept first.
    """
    base = sorted({t for t in base_candidates if 0 <= t < T})
    gt_ts = sorted({int(t) for t in nodes_df["t"].unique()}) if len(nodes_df) else []
    gt_ts = [t for t in gt_ts if 0 <= t < T]

    combined = sorted(set(base) | set(gt_ts))
    if len(combined) > max_timepoints:
        print(
            f"[warn] {len(combined)} candidate timepoints exceeds the cap of "
            f"{max_timepoints}; truncating to keep this fast"
        )
        keep = set(base)
        for t in gt_ts:
            if len(keep) >= max_timepoints:
                break
            keep.add(t)
        combined = sorted(keep)
    return combined


# --------------------------------------------------------------------------- #
# 7. XY max-intensity projection + figure generation
# --------------------------------------------------------------------------- #
def xy_max_intensity_projection(image_zyx: np.ndarray) -> np.ndarray:
    """Max-intensity projection over z: (Z,Y,X) -> (Y,X)."""
    return image_zyx.max(axis=0)


def make_figure(sample_name: str, t: int, mip_xy: np.ndarray, nodes_t: pd.DataFrame, out_path: Path) -> None:
    """Render the XY MIP with a GT-node overlay and save it to `out_path`."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip_xy, cmap="gray")
    if len(nodes_t):
        ax.scatter(
            nodes_t["x"], nodes_t["y"],
            s=40, facecolors="none", edgecolors="red", linewidths=1.3, label="GT node",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{sample_name} | t={t} | GT nodes={len(nodes_t)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 8. Main diagnostic driver
# --------------------------------------------------------------------------- #
SUMMARY_COLUMNS = ["sample", "t", "num_gt_nodes", "image_min", "image_max", "image_mean", "figure_path"]


def run_visualization_diagnostics(
    n_samples: int = 3,
    out_dir: str = "/kaggle/working/figures",
    csv_path: str = "/kaggle/working/visualization_summary.csv",
    base_timepoints: Sequence[int] = (0, 25, 50, 75, 90),
) -> pd.DataFrame:
    """For the first `n_samples` discovered train samples: read image
    timepoints, build an XY max-intensity projection, overlay GT nodes,
    save one figure per timepoint, and write a CSV summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs = find_train_zarr_dirs()
    if not train_dirs:
        print("[error] no train .zarr samples were found.")
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict] = []
    saved_paths: list[Path] = []

    for sample_dir in train_dirs[:n_samples]:
        sample_name = sample_dir.stem
        try:
            T = parse_array_metadata(sample_dir / "0")["shape"][0]
        except Exception as exc:
            print(f"[error] could not read image metadata for {sample_name!r}: {exc!r}")
            continue

        nodes_df, _edges_df = read_geff(sample_dir)
        timepoints = select_timepoints(T, nodes_df, base_timepoints)
        print(f"{sample_name}: T={T}, plotting {len(timepoints)} timepoint(s): {timepoints}")

        for t in timepoints:
            try:
                image_zyx = read_image_timepoint(sample_dir, t)
            except Exception as exc:
                print(f"[error] failed to read {sample_name!r} t={t}: {exc!r}")
                continue

            mip_xy = xy_max_intensity_projection(image_zyx)
            nodes_t = nodes_df[nodes_df["t"] == t] if len(nodes_df) else nodes_df

            fig_path = out_dir / f"{sample_name}_t{t:03d}.png"
            make_figure(sample_name, t, mip_xy, nodes_t, fig_path)
            saved_paths.append(fig_path)

            rows.append({
                "sample": sample_name,
                "t": t,
                "num_gt_nodes": int(len(nodes_t)),
                "image_min": int(image_zyx.min()),
                "image_max": int(image_zyx.max()),
                "image_mean": float(image_zyx.mean()),
                "figure_path": str(fig_path),
            })

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(saved_paths)} figure(s):")
    for p in saved_paths:
        print(f"  - {p}")
    print(f"\nSaved summary CSV to {csv_path} (shape={summary_df.shape})")

    return summary_df


run_visualization_diagnostics(n_samples=3)
