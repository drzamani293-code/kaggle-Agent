"""
Biohub - Cell Tracking During Development
Milestone 1: minimal, robust dummy submission pipeline.

Finds every test .zarr dataset, safely reads its volume metadata, and emits
a schema-valid dummy submission.csv (node + edge rows) with no ML involved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    import zarr
except ImportError:  # zarr may be unavailable in some environments
    zarr = None

REQUIRED_COLUMNS = [
    "id", "dataset", "row_type", "node_id",
    "t", "z", "y", "x", "source_id", "target_id",
]

# Fallback (t, z, y, x) shape used when a volume's metadata can't be read.
DEFAULT_SHAPE = (3, 64, 64, 64)

CANDIDATE_INPUT_ROOTS = [Path("/kaggle/input")]


# --------------------------------------------------------------------------- #
# 1. Discover test .zarr folders
# --------------------------------------------------------------------------- #
def find_test_zarr_dirs(roots: Sequence[Path] = CANDIDATE_INPUT_ROOTS) -> list[Path]:
    """Recursively locate every *.zarr folder belonging to the test split.

    Falls back to *all* discovered .zarr folders if none live under a path
    component literally named "test" (some datasets ship test-only data
    without an explicit "test" directory).
    """
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

    test_dirs = [p for p in all_zarr_dirs if "test" in {part.lower() for part in p.parts}]
    if test_dirs:
        return sorted(set(test_dirs))

    print(
        "[warn] no .zarr folder had a 'test' path component; "
        "treating all discovered .zarr folders as test data."
    )
    return sorted(set(all_zarr_dirs))


# --------------------------------------------------------------------------- #
# 2. Safely read Zarr volume metadata
# --------------------------------------------------------------------------- #
def read_zarr_shape(zarr_path: Path) -> tuple[int, ...]:
    """Safely read the array shape of a (possibly OME-Zarr) volume.

    Never raises: returns DEFAULT_SHAPE and prints a warning on any failure
    so a corrupt / partial / unusual store can't crash the whole pipeline.
    """
    if zarr is None:
        print(f"[warn] zarr package not installed; using default shape for {zarr_path.name}")
        return DEFAULT_SHAPE

    try:
        store = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:
        print(f"[warn] could not open {zarr_path}: {exc!r}; using default shape")
        return DEFAULT_SHAPE

    try:
        if hasattr(store, "shape"):
            shape = store.shape
        else:
            array_keys = sorted(store.array_keys())
            if not array_keys:
                raise ValueError("zarr group has no child arrays")
            shape = store[array_keys[0]].shape
    except Exception as exc:
        print(f"[warn] could not read shape for {zarr_path}: {exc!r}; using default shape")
        return DEFAULT_SHAPE

    if not shape:
        print(f"[warn] empty shape for {zarr_path}; using default shape")
        return DEFAULT_SHAPE
    return tuple(int(s) for s in shape)


def center_zyx(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Take the trailing (z, y, x) dims of `shape` and return their centers."""
    spatial = (list(DEFAULT_SHAPE[-3:]) + list(shape[-3:]))[-3:]
    return tuple(int(s) // 2 for s in spatial)


# --------------------------------------------------------------------------- #
# 3. Build the dummy node/edge rows for one dataset
# --------------------------------------------------------------------------- #
def build_dummy_rows(
    dataset_name: str,
    shape: tuple[int, ...],
    next_id: int,
    next_node_id: int,
) -> tuple[list[dict], int, int]:
    """3 node rows (t=0,1,2 at the volume center) + 2 edges linking them.

    ids/node_ids are threaded through so they stay globally consecutive /
    globally unique across every dataset in the submission.
    """
    cz, cy, cx = center_zyx(shape)
    node_ids = [next_node_id, next_node_id + 1, next_node_id + 2]
    rows: list[dict] = []

    for t, node_id in zip((0, 1, 2), node_ids):
        rows.append({
            "id": next_id, "dataset": dataset_name, "row_type": "node",
            "node_id": node_id, "t": t, "z": cz, "y": cy, "x": cx,
            "source_id": -1, "target_id": -1,
        })
        next_id += 1

    for source_id, target_id in ((node_ids[0], node_ids[1]), (node_ids[1], node_ids[2])):
        rows.append({
            "id": next_id, "dataset": dataset_name, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": source_id, "target_id": target_id,
        })
        next_id += 1

    return rows, next_id, next_node_id + 3


def build_submission(zarr_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    next_id = 0
    next_node_id = 0
    for zarr_path in zarr_dirs:
        dataset_name = zarr_path.stem
        shape = read_zarr_shape(zarr_path)
        dataset_rows, next_id, next_node_id = build_dummy_rows(
            dataset_name, shape, next_id, next_node_id
        )
        rows.extend(dataset_rows)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# --------------------------------------------------------------------------- #
# 4. Validation
# --------------------------------------------------------------------------- #
def validate_submission(df: pd.DataFrame, expected_datasets: Sequence[str]) -> None:
    errors: list[str] = []

    if list(df.columns) != REQUIRED_COLUMNS:
        errors.append(f"columns mismatch: {list(df.columns)} != {REQUIRED_COLUMNS}")

    if df.isna().any().any():
        errors.append(f"NaN values found in columns: {df.columns[df.isna().any()].tolist()}")

    if df["id"].tolist() != list(range(len(df))):
        errors.append("id column is not consecutive starting at 0")

    missing = set(expected_datasets) - set(df["dataset"].unique())
    if missing:
        errors.append(f"missing datasets in submission: {sorted(missing)}")

    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]

    bad_nodes = nodes[(nodes["source_id"] != -1) | (nodes["target_id"] != -1)]
    if len(bad_nodes):
        errors.append(f"{len(bad_nodes)} node rows have source_id/target_id != -1")

    bad_edges = edges[
        (edges["node_id"] != -1) | (edges["t"] != -1) | (edges["z"] != -1)
        | (edges["y"] != -1) | (edges["x"] != -1)
    ]
    if len(bad_edges):
        errors.append(f"{len(bad_edges)} edge rows have node_id/t/z/y/x != -1")

    valid_node_ids = set(nodes["node_id"])
    bad_source = edges[~edges["source_id"].isin(valid_node_ids)]
    bad_target = edges[~edges["target_id"].isin(valid_node_ids)]
    if len(bad_source):
        errors.append(f"{len(bad_source)} edges reference an unknown source_id")
    if len(bad_target):
        errors.append(f"{len(bad_target)} edges reference an unknown target_id")

    if errors:
        raise AssertionError("Submission validation FAILED:\n- " + "\n- ".join(errors))

    n_datasets = df["dataset"].nunique()
    print(f"Validation passed: {len(df)} rows, {n_datasets} dataset(s), no issues found.")


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    zarr_dirs = find_test_zarr_dirs()
    if not zarr_dirs:
        print("[error] no .zarr test folders were found under /kaggle/input. Nothing to submit.")
        sys.exit(1)

    print(f"Found {len(zarr_dirs)} test .zarr dataset(s):")
    for p in zarr_dirs:
        print(f"  - {p}")

    submission = build_submission(zarr_dirs)
    expected_datasets = [p.stem for p in zarr_dirs]
    validate_submission(submission, expected_datasets)

    out_path = Path("/kaggle/working/submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f"\nSaved submission to {out_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
