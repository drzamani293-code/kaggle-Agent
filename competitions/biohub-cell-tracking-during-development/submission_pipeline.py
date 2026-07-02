"""
Biohub - Cell Tracking During Development
Milestone 1: minimal, robust dummy submission pipeline.

Finds every test .zarr dataset, safely reads its volume metadata, and emits
a schema-valid dummy submission.csv (node + edge rows) with no ML involved.
"""

from __future__ import annotations

import os
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


# --------------------------------------------------------------------------- #
# 1. Auto-detect the Kaggle input root, then discover test .zarr folders
# --------------------------------------------------------------------------- #
def get_input_roots() -> list[Path]:
    """Auto-detect candidate input roots, without hardcoding any dataset path.

    Priority order:
    1. ``KAGGLE_INPUT_DIR`` env var - explicit override, e.g. for local testing.
    2. ``/kaggle/input`` - the real Kaggle competition environment.
    3. ``./input`` - a relative fallback for running outside Kaggle.

    Only roots that actually exist are returned, de-duplicated by resolved path.
    """
    candidates = []
    env_root = os.environ.get("KAGGLE_INPUT_DIR")
    if env_root:
        candidates.append(Path(env_root))
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
    """Recursively locate every *.zarr folder belonging to the given split
    (e.g. "test" or "train").

    `roots` defaults to the auto-detected input roots (see `get_input_roots`).
    Falls back to *all* discovered .zarr folders if none live under a path
    component literally matching `split` (some datasets ship split-only data
    without an explicit split directory).
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


def find_test_zarr_dirs(roots: Sequence[Path] | None = None) -> list[Path]:
    """Recursively locate every *.zarr folder belonging to the test split."""
    return find_zarr_dirs_for_split("test", roots)


# --------------------------------------------------------------------------- #
# 2. Safely read Zarr volume metadata
# --------------------------------------------------------------------------- #
def read_zarr_metadata(zarr_path: Path) -> dict:
    """Safely read metadata for a (possibly OME-Zarr) volume.

    Never raises: any failure is recorded in the returned dict (`used_fallback`
    + `note`) and `DEFAULT_SHAPE` is used instead, so one bad store can't
    crash the whole pipeline.
    """
    info = {
        "path": str(zarr_path),
        "shape": None,
        "dtype": None,
        "used_fallback": False,
        "note": "",
    }

    if zarr is None:
        info.update(shape=DEFAULT_SHAPE, used_fallback=True, note="zarr package not installed")
        return info

    try:
        store = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:
        info.update(shape=DEFAULT_SHAPE, used_fallback=True, note=f"could not open store: {exc!r}")
        return info

    try:
        if hasattr(store, "shape"):
            arr = store
        else:
            array_keys = sorted(store.array_keys())
            if not array_keys:
                raise ValueError("zarr group has no child arrays")
            arr = store[array_keys[0]]
        shape = tuple(int(s) for s in arr.shape)
        if not shape:
            raise ValueError("empty shape")
        info.update(shape=shape, dtype=str(arr.dtype))
    except Exception as exc:
        info.update(shape=DEFAULT_SHAPE, used_fallback=True, note=f"could not read shape: {exc!r}")

    return info


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


def build_submission(zarr_dirs: Sequence[Path], metadata: dict[Path, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    next_id = 0
    next_node_id = 0
    for zarr_path in zarr_dirs:
        dataset_name = zarr_path.stem
        shape = metadata[zarr_path]["shape"]
        dataset_rows, next_id, next_node_id = build_dummy_rows(
            dataset_name, shape, next_id, next_node_id
        )
        rows.extend(dataset_rows)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


# --------------------------------------------------------------------------- #
# 4. Validation
# --------------------------------------------------------------------------- #
def validate_submission(df: pd.DataFrame, expected_datasets: Sequence[str]) -> str:
    """Run every structural check and print a full pass/fail report.

    Raises AssertionError if any check fails; returns the report string
    otherwise.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("columns match required schema", list(df.columns) == REQUIRED_COLUMNS,
          f"got {list(df.columns)}")

    na_cols = df.columns[df.isna().any()].tolist()
    check("no NaN values anywhere", not na_cols, f"NaN columns: {na_cols}")

    check("id column is consecutive starting at 0", df["id"].tolist() == list(range(len(df))))

    missing = sorted(set(expected_datasets) - set(df["dataset"].unique()))
    check("every expected test dataset is present", not missing, f"missing: {missing}")

    check("row_type values are only 'node'/'edge'", set(df["row_type"].unique()) <= {"node", "edge"})

    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]

    bad_nodes = nodes[(nodes["source_id"] != -1) | (nodes["target_id"] != -1)]
    check("node rows have source_id=target_id=-1", len(bad_nodes) == 0, f"{len(bad_nodes)} bad rows")

    bad_edges = edges[
        (edges["node_id"] != -1) | (edges["t"] != -1) | (edges["z"] != -1)
        | (edges["y"] != -1) | (edges["x"] != -1)
    ]
    check("edge rows have node_id,t,z,y,x=-1", len(bad_edges) == 0, f"{len(bad_edges)} bad rows")

    check("node_id is unique among node rows", nodes["node_id"].is_unique)

    valid_node_ids = set(nodes["node_id"])
    bad_source = edges[~edges["source_id"].isin(valid_node_ids)]
    bad_target = edges[~edges["target_id"].isin(valid_node_ids)]
    check("edge source_id values reference an existing node_id", len(bad_source) == 0,
          f"{len(bad_source)} bad rows")
    check("edge target_id values reference an existing node_id", len(bad_target) == 0,
          f"{len(bad_target)} bad rows")

    lines = ["Validation report:"]
    all_passed = True
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_passed = False
        line = f"  [{mark}] {c['name']}"
        if c["detail"] and not c["passed"]:
            line += f"  -- {c['detail']}"
        lines.append(line)
    report = "\n".join(lines)
    print(report)

    if not all_passed:
        raise AssertionError("Submission validation FAILED. See report above.")
    return report


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    input_roots = get_input_roots()
    print(f"Auto-detected input root(s): {[str(r) for r in input_roots] or 'none found'}")

    zarr_dirs = find_test_zarr_dirs(input_roots)
    if not zarr_dirs:
        print("[error] no .zarr test folders were found. Nothing to submit.")
        sys.exit(1)

    print(f"\nDiscovered {len(zarr_dirs)} test dataset(s):")
    for p in zarr_dirs:
        print(f"  - {p.stem}  ({p})")

    print("\nZarr metadata per dataset:")
    metadata: dict[Path, dict] = {}
    for p in zarr_dirs:
        info = read_zarr_metadata(p)
        metadata[p] = info
        fallback_note = f"  [fallback used: {info['note']}]" if info["used_fallback"] else ""
        print(f"  - {p.stem}: shape={info['shape']} dtype={info['dtype']}{fallback_note}")

    submission = build_submission(zarr_dirs, metadata)

    print(f"\nsubmission.csv shape: {submission.shape}")
    print("\nsubmission.csv head:")
    print(submission.head(10).to_string(index=False))

    print()
    expected_datasets = [p.stem for p in zarr_dirs]
    validate_submission(submission, expected_datasets)

    out_path = Path("/kaggle/working/submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print(f"\nConfirmation: {out_path} exists -> {out_path.exists()}")


if __name__ == "__main__":
    main()
