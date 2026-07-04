"""
Biohub - Cell Tracking During Development
Milestone 13: top-solution audit, distillation, and winning-path design.

This module is an AUDIT, not a submission builder. It statically parses an
uploaded high-score reference solution package (a notebook, a run log, a
run_stats.csv, and the reference's own submission.csv) - never executing
any of the reference notebook's cells, never installing its dependencies,
never calling the Kaggle API - computes distributional statistics on the
reference's submission, compares it against our own Milestone 12
submission to quantify the density/sparsity gap, checks (via plain path
existence only) whether the reference's supporting artifact is actually
available for legal replication, writes a hand-authored technical note
ranking 6 candidate ideas that could be ported into M12 WITHOUT its deep
model, and produces a concrete Milestone 14 experiment-grid plan
evaluated on the same 12 fixed-seed robust train samples Milestone 11
used - not on test data.

Design notes incorporating architecture review:

  - Path disambiguation: the reference package's 4 files are searched for
    ONLY under configurable "reference roots" (`MILESTONE13_REFERENCE_DIR`
    env override, else `/kaggle/input`) - `/kaggle/working` is never
    searched for the reference, specifically so the reference's
    `submission.csv` (a read-only input mount) can never be confused with
    our own Milestone-12-produced `submission.csv` sitting in
    `/kaggle/working`. Milestone 12's own submission is located
    separately and explicitly at `/kaggle/working/submission.csv` (or
    `submission_primary.csv`). As a cheap sanity check (never a hard
    assertion - re-runs/different splits can shift exact row counts), a
    warning is printed if the "reference" file turns out to have FEWER
    rows than our own M12 submission, since that inversion usually means
    the paths got swapped.

  - Notebook parsing safety: the reference `.ipynb` is only ever
    `json.load()`-ed (standard nbformat) with a file-size safety cap and
    `errors="replace"` decoding - never `nbconvert --execute`, never
    `exec()`/`eval()` on any cell. Extracting post-processing function
    snippets uses `ast.parse()` (parsing only, never executes) on each
    code cell after stripping IPython magic lines (`!pip install ...`,
    `%matplotlib ...`) that would otherwise raise SyntaxError; cells that
    still fail to parse are skipped for the AST step but their raw text is
    still regex-scanned separately for config values/pip-installs/the
    prediction command/weights paths. Per-cell source length is capped
    before parsing to avoid a pathological cell blowing up parse time.

  - Schema-mismatch handling: the reference submission is NOT assumed to
    use our exact `SUBMISSION_COLUMNS` layout. `detect_submission_schema`
    checks for it explicitly; if unrecognized, generic stats (shape,
    columns, dtypes, null counts, nunique) are still reported, and the
    reason detection failed is recorded, rather than guessing at
    semantics that could silently produce wrong node/edge breakdowns.

  - Degree/dangling-edge/t-gap correctness: node_id is only unique WITHIN
    a dataset for a foreign submission of unknown convention (never
    assumed globally unique like our own), so every degree/dangling/t-gap
    computation is scoped per-dataset via an explicit (dataset, node_id)
    key - never a bare global node_id lookup, which could silently
    mismatch nodes across datasets that happen to reuse small integer
    ids. Degree distributions are reindexed over the FULL node set so
    degree-0 nodes are never silently dropped. Duplicate edges are
    deduped on the ORDERED (directed) (dataset, source_id, target_id)
    triple, since tracking edges are directed in time.

  - Parts E/F/G are hand-authored technical judgment (not derived from
    the parsed notebook, which can at best partially/fuzzily locate
    relevant functions via keyword heuristics) - grounded in the concrete
    techniques named in the milestone brief, optionally cross-referenced
    against whatever Part A's static extraction actually found.

No submission is ever built, no Kaggle API is ever called, Save Version
is never triggered, and no reference-solution dependency is ever
installed. 6C/6D/9/10/11/12's files are not modified - all reused logic
(SUBMISSION_COLUMNS, VOXEL_SIZE_UM) is copied in, not imported.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Shared constants (copied from Milestones 10-12, not imported)
# --------------------------------------------------------------------------- #
VOXEL_SIZE_UM = {"z": 1.625, "y": 0.40625, "x": 0.40625}
SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

# Milestone 11's real aggregate-OOF result, quoted verbatim - the anchor
# number every "does this beat what we already validated" comparison in
# this module (and the Milestone 14 plan) is checked against.
M11_BEST_EDGE_JACCARD = 0.013215
M11_EDGE_TP = 103
M11_EDGE_FP = 1932
M11_EDGE_FN = 5759
M11_N_SAMPLES = 12
M11_N_SAMPLES_WITH_TP = 6
BASELINE_6C6D_EDGE_JACCARD = 0.010667

M12_PRIMARY_CONFIG_SUMMARY = {
    "node_filter": "moderate_150_085", "classifier": "hist_gradient_boosting", "negative_ratio": 100,
    "edge_selection_policy": "greedy_exclusive_per_frame", "edge_selection_param": 0.9,
    "node_policy": "A_tracklet_nodes_only", "min_tracklet_length": 5,
    "min_mean_classifier_score_quantile": 0.7, "max_smoothness_error_um": 3.0,
    "keep_top_k_tracklets_per_sample": 100,
}

# --------------------------------------------------------------------------- #
# 2. Path resolution - reference package vs. our own M12 submission
# --------------------------------------------------------------------------- #
def get_reference_search_roots() -> list[Path]:
    """Roots to search for the UPLOADED reference package - deliberately
    NEVER includes `/kaggle/working` (that's where our own M12 submission
    lives; searching it here risks conflating the two `submission.csv`
    files).
    """
    roots: list[Path] = []
    env_root = os.environ.get("MILESTONE13_REFERENCE_DIR")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path("/kaggle/input"))
    return [r for r in roots if r.exists()]


def find_reference_files(roots: Sequence[Path] | None = None) -> dict:
    """Locates the reference notebook/log/run_stats/submission by filename
    pattern, searching ONLY the reference roots. Prints which roots were
    searched (so a "not found" outcome is auditable) and warns if more
    than one candidate of a given type is found.
    """
    if roots is None:
        roots = get_reference_search_roots()

    candidates: dict[str, list[Path]] = {"notebook": [], "log": [], "run_stats": [], "submission": []}
    for root in roots:
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                name_lower = path.name.lower()
                if name_lower.endswith(".ipynb"):
                    candidates["notebook"].append(path)
                elif name_lower.endswith(".txt") and "download" in name_lower:
                    candidates["log"].append(path)
                elif name_lower == "run_stats.csv":
                    candidates["run_stats"].append(path)
                elif name_lower == "submission.csv":
                    candidates["submission"].append(path)
        except OSError as exc:
            print(f"  [warn] could not scan reference root {root}: {exc!r}")

    found: dict[str, Path | None] = {}
    for key, paths in candidates.items():
        found[key] = paths[0] if paths else None
        if len(paths) > 1:
            print(f"  [warn] multiple candidate {key} file(s) found under reference roots, using {paths[0]} (others: {[str(p) for p in paths[1:]]})")

    if not roots:
        print("  [info] no reference search roots exist on disk (no MILESTONE13_REFERENCE_DIR override, no /kaggle/input).")
    elif not any(found.values()):
        print(f"  [info] searched reference root(s) {[str(r) for r in roots]} - no reference package files found.")

    return found


def find_m12_submission(working_dir: str = "/kaggle/working") -> Path | None:
    """Locates OUR OWN Milestone 12 submission - explicitly and only at
    its two known, established output paths.
    """
    for name in ("submission.csv", "submission_primary.csv"):
        p = Path(working_dir) / name
        if p.exists():
            return p
    return None


def sanity_check_reference_vs_m12_rowcounts(reference_df: pd.DataFrame | None, m12_df: pd.DataFrame | None) -> None:
    """A cheap, non-fatal sanity check (never a hard assertion - a re-run
    or different split could legitimately shift row counts): if the
    "reference" file has FEWER rows than our own M12 submission, that
    inversion is a strong signal the paths were accidentally swapped.
    """
    if reference_df is None or m12_df is None:
        return
    if len(reference_df) < len(m12_df):
        print(
            f"  [warn] reference submission ({len(reference_df)} rows) has FEWER rows than our own M12 "
            f"submission ({len(m12_df)} rows) - double-check that the reference/M12 paths were not swapped."
        )

# --------------------------------------------------------------------------- #
# 3. Part A - static notebook extraction (parse only, NEVER execute)
# --------------------------------------------------------------------------- #
NOTEBOOK_MAX_BYTES = 50 * 1024 * 1024  # 50MB safety cap against a hostile/huge notebook file
CELL_SOURCE_MAX_CHARS = 200_000  # per-cell cap before ast.parse - avoids a pathological cell

MAGIC_LINE_PATTERN = re.compile(r"^\s*[!%].*$", re.MULTILINE)
PIP_INSTALL_PATTERN = re.compile(r"^\s*[!%]?\s*pip\s+install.*$", re.MULTILINE | re.IGNORECASE)
WHEELS_PATTERN = re.compile(r"(--no-index|--find-links|wheels?/)", re.IGNORECASE)
PREDICT_COMMAND_PATTERN = re.compile(r"^\s*[!%]?.*predict_[a-zA-Z_]*\.py.*$", re.MULTILINE)
WEIGHTS_PATH_PATTERN = re.compile(r"['\"]([^'\"]*\.pth)['\"]")
HASH_PATTERN = re.compile(r"\b[a-fA-F0-9]{32,64}\b")
ARTIFACT_PATTERN = re.compile(r"['\"]([^'\"]*support-pack[^'\"]*)['\"]", re.IGNORECASE)
IMPORT_PATTERN = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.MULTILINE)
CONFIG_LINE_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$", re.MULTILINE)
KNOWN_CONFIG_KEYS = {"title", "method", "det_threshold", "use_ilp", "weights", "artifact", "split", "threshold"}

FUNCTION_CATEGORY_KEYWORDS = {
    "motion_relinking": ["motion_relink", "relink", "motion_link"],
    "gap_closing": ["gap_clos", "gap_recover", "close_gap"],
    "gap2_recovery": ["gap2", "two_frame_gap", "gap_two"],
    "safe_division_recovery": ["safe_division", "division_recover", "safe_div"],
    "line_fit_smoothing": ["line_fit", "linefit", "smooth"],
    "output_graph_filtering": ["filter_graph", "graph_filter", "prune", "output_filter"],
    "submission_writing": ["submission", "write_submission", "build_submission", "save_submission"],
    "validation_logic": ["validate", "sanity_check", "check_submission"],
}


def load_notebook_safely(path: Path) -> dict | None:
    """Only ever `json.load()`s the notebook (standard nbformat JSON) -
    never executes a cell. Guards against a hostile/huge file with a size
    cap and tolerant decoding.
    """
    try:
        size = path.stat().st_size
        if size > NOTEBOOK_MAX_BYTES:
            print(f"  [warn] notebook {path} is {size} byte(s), exceeds the {NOTEBOOK_MAX_BYTES}-byte safety cap - skipping.")
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        return json.loads(text)
    except Exception as exc:
        print(f"  [warn] could not load notebook {path}: {exc!r}")
        return None


def get_cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def strip_magic_lines(source: str) -> str:
    return MAGIC_LINE_PATTERN.sub("", source)


def classify_function_name(name: str, docstring: str) -> list[str]:
    haystack = (name + " " + (docstring or "")).lower()
    return [category for category, keywords in FUNCTION_CATEGORY_KEYWORDS.items() if any(kw in haystack for kw in keywords)]


def extract_functions_from_notebook(nb: dict) -> list[dict]:
    """Walks every code cell, strips IPython magic lines, and uses `ast`
    (parsing only - NEVER `exec`/`eval`) to find function definitions,
    classifying each against the requested post-processing categories by a
    keyword heuristic on its name/docstring.
    """
    results: list[dict] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        raw_source = get_cell_source(cell)
        if len(raw_source) > CELL_SOURCE_MAX_CHARS:
            continue
        stripped = strip_magic_lines(raw_source)
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                snippet = ast.get_source_segment(stripped, node) or ""
                docstring = ast.get_docstring(node) or ""
                results.append({
                    "name": node.name, "categories": classify_function_name(node.name, docstring),
                    "docstring": docstring[:500], "source_snippet": snippet[:3000],
                })
    return results


def extract_config_values(nb: dict) -> dict:
    """Heuristic `NAME = value` extraction for ALL_CAPS constants and any
    line whose key looks config/path/weight/artifact/threshold-like -
    text-only, never evaluated as real code (only `ast.literal_eval` on
    the isolated right-hand-side literal, which cannot execute anything).
    """
    config: dict = {}
    for cell in nb.get("cells", []):
        source = get_cell_source(cell)
        for match in CONFIG_LINE_PATTERN.finditer(source):
            key, raw_value = match.group(1), match.group(2)
            key_lower = key.lower()
            looks_relevant = (
                key.isupper() or key_lower in KNOWN_CONFIG_KEYS
                or any(kw in key_lower for kw in ("path", "weight", "artifact", "threshold"))
            )
            if not looks_relevant:
                continue
            try:
                value = ast.literal_eval(raw_value)
            except Exception:
                value = raw_value.strip().strip(",")
            config[key] = value
    return config


def extract_static_facts(nb: dict) -> dict:
    all_source = "\n".join(get_cell_source(c) for c in nb.get("cells", []))

    title = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = get_cell_source(cell).strip()
            if src.startswith("#"):
                title = src.lstrip("#").strip().splitlines()[0]
                break

    return {
        "title": title,
        "pip_installs": sorted(set(m.strip() for m in PIP_INSTALL_PATTERN.findall(all_source))),
        "wheel_mentions": sorted(set(WHEELS_PATTERN.findall(all_source))),
        "predict_commands": sorted(set(m.strip() for m in PREDICT_COMMAND_PATTERN.findall(all_source))),
        "weights_paths": sorted(set(WEIGHTS_PATH_PATTERN.findall(all_source))),
        "hashes_found": sorted(set(HASH_PATTERN.findall(all_source))),
        "artifact_paths": sorted(set(ARTIFACT_PATTERN.findall(all_source))),
        "dependency_imports": sorted(set(IMPORT_PATTERN.findall(all_source))),
    }


def run_part_a_notebook_extraction(notebook_path: Path | None, out_dir: Path) -> dict:
    """Ties together static-fact extraction, config-value extraction, and
    function classification for one reference notebook (or reports
    unavailability cleanly if no notebook was found/loadable).
    """
    if notebook_path is None:
        result = {"available": False, "reason": "no reference notebook found under the searched reference root(s)"}
        config_values: dict = {}
    else:
        nb = load_notebook_safely(notebook_path)
        if nb is None:
            result = {"available": False, "reason": f"notebook at {notebook_path} failed to load safely"}
            config_values = {}
        else:
            static_facts = extract_static_facts(nb)
            config_values = extract_config_values(nb)
            functions = extract_functions_from_notebook(nb)
            result = {
                "available": True, "notebook_path": str(notebook_path),
                "n_cells": len(nb.get("cells", [])), **static_facts,
                "n_functions_found": len(functions), "functions_found": functions,
            }

    with open(out_dir / "milestone13_static_notebook_extract.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(out_dir / "milestone13_reference_config.json", "w") as f:
        json.dump({"available": result["available"], "config_values": config_values}, f, indent=2, default=str)

    return result

# --------------------------------------------------------------------------- #
# 4. Shared library: load + schema-detect + compute distributional stats on
#    any submission-shaped CSV (used by both Part B and Part C)
# --------------------------------------------------------------------------- #
def load_submission_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"  [warn] could not read submission CSV {path}: {exc!r}")
        return None


def detect_submission_schema(df: pd.DataFrame | None) -> tuple[str, str]:
    """Never assumes a foreign submission matches our SUBMISSION_COLUMNS
    layout - checks explicitly and reports WHY detection failed rather
    than guessing at node/edge semantics that could silently misclassify
    rows.
    """
    if df is None or len(df.columns) == 0:
        return "empty", "no dataframe/columns"
    missing = sorted(set(SUBMISSION_COLUMNS) - set(df.columns))
    if missing:
        return "unrecognized", f"missing columns: {missing}"
    row_types = set(df["row_type"].unique()) if len(df) else set()
    if not row_types <= {"node", "edge"}:
        return "unrecognized", f"row_type has unexpected values: {sorted(row_types)}"
    return "recognized", "matches SUBMISSION_COLUMNS schema with row_type in {node,edge}"


def compute_generic_stats(df: pd.DataFrame) -> dict:
    """Stats that make sense regardless of whether the schema is
    recognized - always reported, even for a completely foreign layout.
    """
    return {
        "shape": list(df.shape), "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "nunique": {c: int(df[c].nunique()) for c in df.columns},
    }


def compute_submission_stats(df: pd.DataFrame) -> dict:
    """Computes the full requested stat suite for a SUBMISSION_COLUMNS-
    shaped dataframe: shape/counts, node_id uniqueness, dangling/duplicate
    edges, t-gap distribution, in/out-degree distributions, division-like
    and multi-parent counts, physical edge-distance percentiles, per-
    dataset coordinate ranges, and per-timepoint node/edge density.

    All node/dangling/degree/t-gap computations are scoped per-dataset via
    an explicit (dataset, node_id) key - a foreign submission's node_id is
    NOT assumed globally unique the way ours is, so a bare global lookup
    could silently mismatch nodes across datasets that reuse small integer
    ids. Degree distributions are reindexed over the FULL node set so
    degree-0 nodes are never silently dropped. Duplicate edges are deduped
    on the ORDERED (directed) (dataset, source_id, target_id) triple,
    since tracking edges are directed in time.
    """
    schema, reason = detect_submission_schema(df)
    result: dict = {"schema": schema, "schema_reason": reason, "generic": compute_generic_stats(df)}
    empty_frames = {
        "per_dataset_counts": pd.DataFrame(), "degree_stats": pd.DataFrame(),
        "edge_distance_stats": pd.DataFrame(), "timepoint_density": pd.DataFrame(),
    }
    if schema != "recognized":
        result.update(empty_frames)
        return result

    nodes = df[df["row_type"] == "node"].copy()
    edges = df[df["row_type"] == "edge"].copy()
    datasets = sorted(df["dataset"].unique().tolist())

    result["n_node_rows"] = len(nodes)
    result["n_edge_rows"] = len(edges)
    result["datasets"] = datasets
    result["edge_to_node_ratio"] = (len(edges) / len(nodes)) if len(nodes) > 0 else float("nan")

    per_dataset_rows = []
    for dataset in datasets:
        n_nodes_d = int((nodes["dataset"] == dataset).sum())
        n_edges_d = int((edges["dataset"] == dataset).sum())
        per_dataset_rows.append({
            "dataset": dataset, "n_nodes": n_nodes_d, "n_edges": n_edges_d,
            "edge_to_node_ratio": (n_edges_d / n_nodes_d) if n_nodes_d > 0 else float("nan"),
        })
    result["per_dataset_counts"] = pd.DataFrame(per_dataset_rows)

    result["node_id_unique_global"] = bool(nodes["node_id"].is_unique)
    result["node_id_unique_per_dataset"] = {ds: bool(g["node_id"].is_unique) for ds, g in nodes.groupby("dataset")}

    # (dataset, node_id) -> t / (z,y,x) lookups - never a bare global node_id key
    node_t_lookup: dict[tuple, int] = {}
    node_coord_lookup: dict[tuple, tuple] = {}
    valid_ids_by_dataset: dict[str, set] = {}
    for dataset, group in nodes.groupby("dataset"):
        valid_ids_by_dataset[dataset] = set(group["node_id"])
        for nid, t, z, y, x in zip(group["node_id"], group["t"], group["z"], group["y"], group["x"]):
            node_t_lookup[(dataset, int(nid))] = int(t)
            node_coord_lookup[(dataset, int(nid))] = (float(z), float(y), float(x))

    dangling_mask, t_gaps = [], []
    for row in edges.itertuples():
        valid_ids = valid_ids_by_dataset.get(row.dataset, set())
        src_ok, tgt_ok = row.source_id in valid_ids, row.target_id in valid_ids
        dangling_mask.append(not (src_ok and tgt_ok))
        t_gaps.append(node_t_lookup[(row.dataset, int(row.target_id))] - node_t_lookup[(row.dataset, int(row.source_id))] if (src_ok and tgt_ok) else None)

    edges = edges.assign(_dangling=dangling_mask, _t_gap=t_gaps)
    result["dangling_edge_count"] = int(sum(dangling_mask))
    result["duplicate_edge_count"] = int(edges.duplicated(subset=["dataset", "source_id", "target_id"]).sum())

    # Degree/t-gap/distance/density stats all operate on a DEDUPED, non-
    # dangling edge set - a duplicate row must never be double-counted as
    # if it were 2 distinct edges (it would otherwise silently inflate
    # out-degree and misclassify a node as "division-like").
    valid_edges = edges[~edges["_dangling"]].drop_duplicates(subset=["dataset", "source_id", "target_id"])

    valid_t_gaps = [g for g in valid_edges["_t_gap"] if g is not None]
    if valid_t_gaps:
        t_gap_counts = pd.Series(valid_t_gaps).value_counts().sort_index()
        result["t_gap_distribution"] = {str(k): int(v) for k, v in t_gap_counts.items()}
        result["frac_edges_t_to_t1"] = float((pd.Series(valid_t_gaps) == 1).mean())
    else:
        result["t_gap_distribution"] = {}
        result["frac_edges_t_to_t1"] = float("nan")

    node_keys = pd.MultiIndex.from_frame(nodes[["dataset", "node_id"]])
    out_degree_full = valid_edges.groupby(["dataset", "source_id"]).size().reindex(node_keys, fill_value=0)
    in_degree_full = valid_edges.groupby(["dataset", "target_id"]).size().reindex(node_keys, fill_value=0)

    degree_df = pd.DataFrame({
        "dataset": node_keys.get_level_values(0), "node_id": node_keys.get_level_values(1),
        "out_degree": out_degree_full.to_numpy(), "in_degree": in_degree_full.to_numpy(),
    })
    result["degree_stats"] = degree_df
    result["out_degree_distribution"] = {str(k): int(v) for k, v in degree_df["out_degree"].value_counts().sort_index().items()}
    result["in_degree_distribution"] = {str(k): int(v) for k, v in degree_df["in_degree"].value_counts().sort_index().items()}
    result["division_like_source_count"] = int((degree_df["out_degree"] == 2).sum())
    result["multi_parent_child_count"] = int((degree_df["in_degree"] > 1).sum())

    distances = []
    for row in valid_edges.itertuples():
        src_c = node_coord_lookup.get((row.dataset, int(row.source_id)))
        tgt_c = node_coord_lookup.get((row.dataset, int(row.target_id)))
        if src_c is None or tgt_c is None:
            continue
        dz = (tgt_c[0] - src_c[0]) * VOXEL_SIZE_UM["z"]
        dy = (tgt_c[1] - src_c[1]) * VOXEL_SIZE_UM["y"]
        dx = (tgt_c[2] - src_c[2]) * VOXEL_SIZE_UM["x"]
        distances.append(float(np.sqrt(dz * dz + dy * dy + dx * dx)))

    if distances:
        dist_series = pd.Series(distances)
        result["edge_distance_summary"] = {
            "mean": float(dist_series.mean()), "median": float(dist_series.median()),
            "p90": float(dist_series.quantile(0.90)), "p95": float(dist_series.quantile(0.95)),
            "p99": float(dist_series.quantile(0.99)), "max": float(dist_series.max()),
        }
        result["edge_distance_stats"] = pd.DataFrame([result["edge_distance_summary"]])
    else:
        result["edge_distance_summary"] = {}
        result["edge_distance_stats"] = pd.DataFrame()

    coord_ranges = {}
    for dataset, group in nodes.groupby("dataset"):
        coord_ranges[dataset] = {
            "z_min": float(group["z"].min()), "z_max": float(group["z"].max()),
            "y_min": float(group["y"].min()), "y_max": float(group["y"].max()),
            "x_min": float(group["x"].min()), "x_max": float(group["x"].max()),
            "t_min": int(group["t"].min()), "t_max": int(group["t"].max()),
        }
    result["coordinate_ranges_per_dataset"] = coord_ranges

    edge_source_t_counts: dict[tuple, int] = {}
    for row in valid_edges.itertuples():
        src_t = node_t_lookup.get((row.dataset, int(row.source_id)))
        if src_t is None:
            continue
        key = (row.dataset, src_t)
        edge_source_t_counts[key] = edge_source_t_counts.get(key, 0) + 1

    tp_rows = []
    for dataset, group in nodes.groupby("dataset"):
        for t, t_group in group.groupby("t"):
            tp_rows.append({
                "dataset": dataset, "t": int(t), "n_nodes": len(t_group),
                "n_edges_starting_here": edge_source_t_counts.get((dataset, int(t)), 0),
            })
    result["timepoint_density"] = pd.DataFrame(tp_rows)

    return result

# --------------------------------------------------------------------------- #
# 5. Part B - reference submission distribution analysis
# --------------------------------------------------------------------------- #
_PART_B_CSV_NAMES = {
    "per_dataset_counts": "milestone13_reference_per_dataset_counts.csv",
    "degree_stats": "milestone13_reference_degree_stats.csv",
    "edge_distance_stats": "milestone13_reference_edge_distance_stats.csv",
    "timepoint_density": "milestone13_reference_timepoint_density.csv",
}
_STATS_FRAME_KEYS = ("per_dataset_counts", "degree_stats", "edge_distance_stats", "timepoint_density")


def _save_empty_part_b_outputs(out_dir: Path, reason: str) -> dict:
    stats = {"available": False, "reason": reason}
    with open(out_dir / "milestone13_reference_submission_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    for csv_name in _PART_B_CSV_NAMES.values():
        pd.DataFrame().to_csv(out_dir / csv_name, index=False)
    return {"available": False, "df": None, "stats": stats}


def run_part_b_reference_stats(reference_submission_path: Path | None, out_dir: Path) -> dict:
    """Computes the full requested stat suite on the reference's
    submission.csv and saves the 5 required output files. Reports
    unavailability cleanly (still writing empty placeholder files) if no
    reference submission was found or it failed to read.
    """
    if reference_submission_path is None:
        return _save_empty_part_b_outputs(out_dir, "no reference submission.csv found under the searched reference root(s)")

    df = load_submission_csv(reference_submission_path)
    if df is None:
        return _save_empty_part_b_outputs(out_dir, f"failed to read {reference_submission_path}")

    stats = compute_submission_stats(df)
    stats["available"] = True
    stats["source_path"] = str(reference_submission_path)

    json_safe_stats = {k: v for k, v in stats.items() if k not in _STATS_FRAME_KEYS}
    with open(out_dir / "milestone13_reference_submission_stats.json", "w") as f:
        json.dump(json_safe_stats, f, indent=2, default=str)
    for key, csv_name in _PART_B_CSV_NAMES.items():
        stats[key].to_csv(out_dir / csv_name, index=False)

    return {"available": True, "df": df, "stats": stats}

# --------------------------------------------------------------------------- #
# 6. Part C - M12 vs. reference comparison (quantifying the sparsity gap)
# --------------------------------------------------------------------------- #
def compute_component_sizes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.Series:
    """Weakly-connected-component sizes (treating every edge as an
    undirected link) via union-find - a robust, degree-agnostic stand-in
    for "track-like component length" that works whether the underlying
    graph is a clean non-branching chain (ours) or a branching DAG with
    divisions/merges (a dense reference graph).
    """
    parent: dict[tuple, tuple] = {}

    def find(x: tuple) -> tuple:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple, b: tuple) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in nodes_df.itertuples():
        parent[(row.dataset, int(row.node_id))] = (row.dataset, int(row.node_id))
    for row in edges_df.itertuples():
        a, b = (row.dataset, int(row.source_id)), (row.dataset, int(row.target_id))
        if a in parent and b in parent:
            union(a, b)

    roots: dict[tuple, int] = {}
    for key in parent:
        r = find(key)
        roots[r] = roots.get(r, 0) + 1
    return pd.Series(list(roots.values()), dtype=int)


def component_summary(sizes: pd.Series) -> dict:
    if len(sizes) == 0:
        return {}
    return {"n_components": int(len(sizes)), "mean_size": float(sizes.mean()), "median_size": float(sizes.median()), "max_size": int(sizes.max())}


def build_comparison_row(label: str, stats: dict) -> dict:
    degree_df = stats.get("degree_stats", pd.DataFrame())
    isolated_frac = (
        float(((degree_df["in_degree"] == 0) & (degree_df["out_degree"] == 0)).mean())
        if len(degree_df) else float("nan")
    )
    edge_distance_summary = stats.get("edge_distance_summary", {}) or {}
    return {
        "label": label, "schema": stats.get("schema"),
        "total_rows": stats["generic"]["shape"][0] if "generic" in stats else None,
        "n_node_rows": stats.get("n_node_rows"), "n_edge_rows": stats.get("n_edge_rows"),
        "edge_to_node_ratio": stats.get("edge_to_node_ratio"),
        "n_datasets": len(stats.get("datasets") or []), "datasets": stats.get("datasets"),
        "dangling_edge_count": stats.get("dangling_edge_count"), "duplicate_edge_count": stats.get("duplicate_edge_count"),
        "frac_edges_t_to_t1": stats.get("frac_edges_t_to_t1"),
        "division_like_source_count": stats.get("division_like_source_count"),
        "multi_parent_child_count": stats.get("multi_parent_child_count"),
        "isolated_node_fraction": isolated_frac,
        "edge_distance_mean_um": edge_distance_summary.get("mean"), "edge_distance_p95_um": edge_distance_summary.get("p95"),
    }


def run_part_c_comparison(m12_bundle: dict, reference_bundle: dict, out_dir: Path) -> dict:
    """Side-by-side comparison of M12's own submission against the
    reference submission, quantifying exactly how sparse M12 is.
    """
    if not m12_bundle["available"] or not reference_bundle["available"]:
        report = {
            "available": False, "m12_available": m12_bundle["available"], "reference_available": reference_bundle["available"],
            "reason": "comparison requires both M12 and the reference submission to be available",
        }
        pd.DataFrame().to_csv(out_dir / "milestone13_m12_vs_reference_comparison.csv", index=False)
        with open(out_dir / "milestone13_density_gap_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    m12_stats, ref_stats = m12_bundle["stats"], reference_bundle["stats"]
    m12_row, ref_row = build_comparison_row("m12", m12_stats), build_comparison_row("reference", ref_stats)
    pd.DataFrame([m12_row, ref_row]).to_csv(out_dir / "milestone13_m12_vs_reference_comparison.csv", index=False)

    def safe_ratio(numerator, denominator):
        if not denominator:
            return float("nan")
        return numerator / denominator

    density_ratio_total = safe_ratio(ref_row["total_rows"], m12_row["total_rows"])
    density_ratio_nodes = safe_ratio(ref_row["n_node_rows"], m12_row["n_node_rows"])
    density_ratio_edges = safe_ratio(ref_row["n_edge_rows"], m12_row["n_edge_rows"])

    m12_df, ref_df = m12_bundle["df"], reference_bundle["df"]
    m12_components = (
        compute_component_sizes(m12_df[m12_df["row_type"] == "node"], m12_df[m12_df["row_type"] == "edge"])
        if m12_stats["schema"] == "recognized" else pd.Series(dtype=int)
    )
    ref_components = (
        compute_component_sizes(ref_df[ref_df["row_type"] == "node"], ref_df[ref_df["row_type"] == "edge"])
        if ref_stats["schema"] == "recognized" else pd.Series(dtype=int)
    )

    interpretation = "insufficient data to quantify the density gap"
    if not (isinstance(density_ratio_total, float) and np.isnan(density_ratio_total)):
        interpretation = (
            f"The reference submission has ~{density_ratio_total:.1f}x more total rows than M12's own submission "
            f"({ref_row['total_rows']} vs {m12_row['total_rows']}) - {density_ratio_nodes:.1f}x more node rows and "
            f"{density_ratio_edges:.1f}x more edge rows - confirming M12's node_policy=A_tracklet_nodes_only output "
            "is drastically sparser than the reference's dense output graph."
        )

    density_gap_report = {
        "available": True,
        "m12_total_rows": m12_row["total_rows"], "reference_total_rows": ref_row["total_rows"], "density_ratio_total_rows": density_ratio_total,
        "m12_node_rows": m12_row["n_node_rows"], "reference_node_rows": ref_row["n_node_rows"], "density_ratio_node_rows": density_ratio_nodes,
        "m12_edge_rows": m12_row["n_edge_rows"], "reference_edge_rows": ref_row["n_edge_rows"], "density_ratio_edge_rows": density_ratio_edges,
        "m12_per_dataset": m12_stats.get("per_dataset_counts", pd.DataFrame()).to_dict(orient="records"),
        "reference_per_dataset": ref_stats.get("per_dataset_counts", pd.DataFrame()).to_dict(orient="records"),
        "m12_component_summary": component_summary(m12_components), "reference_component_summary": component_summary(ref_components),
        "interpretation": interpretation,
    }
    with open(out_dir / "milestone13_density_gap_report.json", "w") as f:
        json.dump(density_gap_report, f, indent=2, default=str)

    return density_gap_report

# --------------------------------------------------------------------------- #
# 7. Part D - artifact and compliance check (path existence ONLY - no
#    downloads, no installs, no execution of any reference dependency)
# --------------------------------------------------------------------------- #
def _find_path_containing(roots: Sequence[Path], needle: str) -> Path | None:
    needle_lower = needle.lower()
    for root in roots:
        try:
            for path in root.rglob("*"):
                if needle_lower in path.name.lower():
                    return path
        except OSError:
            continue
    return None


def _find_wheels_dir(roots: Sequence[Path]) -> Path | None:
    for root in roots:
        try:
            for path in root.rglob("*"):
                if path.is_dir() and "wheel" in path.name.lower():
                    return path
        except OSError:
            continue
    return None


def run_part_d_artifact_compliance(reference_roots: Sequence[Path], out_dir: Path) -> dict:
    """Checks whether the reference's supporting artifact/weights/scripts/
    offline-wheels are actually present on disk - never assumes
    availability, never downloads or installs anything. This is a
    feasibility check for LEGAL, offline replication, not an attempt to
    run the reference pipeline.
    """
    artifact_dir = _find_path_containing(reference_roots, "biohub-tracking-support-pack-50ep-v1")
    manifest = _find_path_containing(reference_roots, "ARTIFACT_MANIFEST.json")
    weights = _find_path_containing(reference_roots, "edge_predictor_best.pth")
    scripts = _find_path_containing(reference_roots, "predict_unet_transformer.py")
    wheels_dir = _find_wheels_dir(reference_roots)
    packages_installable_offline = bool(wheels_dir and any(wheels_dir.glob("*.whl")))
    replication_possible = bool(artifact_dir and weights and scripts and packages_installable_offline)

    result = {
        "searched_roots": [str(r) for r in reference_roots],
        "artifact_path_exists": bool(artifact_dir), "artifact_path": str(artifact_dir) if artifact_dir else None,
        "artifact_manifest_exists": bool(manifest), "artifact_manifest_path": str(manifest) if manifest else None,
        "weights_exist": bool(weights), "weights_path": str(weights) if weights else None,
        "scripts_exist": bool(scripts), "scripts_path": str(scripts) if scripts else None,
        "wheels_dir_exists": bool(wheels_dir), "wheels_dir_path": str(wheels_dir) if wheels_dir else None,
        "packages_installable_offline": packages_installable_offline,
        "replication_possible_in_kaggle_offline_mode": replication_possible,
        "note": "Path-existence check ONLY - never downloads, installs, or executes any reference dependency.",
    }
    with open(out_dir / "milestone13_artifact_compliance_check.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result

# --------------------------------------------------------------------------- #
# 8. Part E - transferable ideas (hand-authored technical judgment, ranked
#    by expected gain vs. implementation risk)
# --------------------------------------------------------------------------- #
TRANSFERABLE_IDEAS = [
    {
        "idea_id": "line_fit_smoothing", "priority_rank": 1,
        "title": "Line-fit smoothing of tracklet coordinates",
        "what_reference_does": (
            "For each finalized tracklet, refits each interior node's (z,y,x) as a weighted local-window "
            "regression along the track (window=2, weight=0.8 per the reference config) - a purely "
            "topology-preserving coordinate denoise, changing no node/edge membership."
        ),
        "why_it_might_improve_metric": (
            "Edge_jaccard grading depends on GT-node matching via a Hungarian assignment at a fixed physical "
            "distance threshold (7um in our own harness). Smoothing coordinates closer to the true underlying "
            "trajectory could rescue currently-borderline node matches that fall just outside the threshold due "
            "to detector jitter, indirectly improving edge_TP with zero risk of removing anything already correct."
        ),
        "how_to_port_into_m12": (
            "Apply as a pure post-process on reconstruct_policy_a's output node chain, right before submission-row "
            "building - a self-contained ~10-line function with no new features, no retraining, and no interaction "
            "with the classifier/edge-selection pipeline."
        ),
        "oof_experiment_design": (
            "Milestone 14 Experiment B (linefit only). Measure whether node_matches/edge_TP change at all in our "
            "own local match_nodes_by_timepoint harness; even a null result is worth keeping as a zero-cost hedge "
            "against the real grader being pickier than our local 7um match radius."
        ),
        "expected_risk": "Lowest of all 6 ideas - cannot add/remove a node or edge, worst case is a pure no-op.",
        "implementation_complexity": "Low",
        "expected_gain": "Low-to-moderate, but effectively free",
    },
    {
        "idea_id": "motion_relink", "priority_rank": 2,
        "title": "Motion-relink-style two-pass assignment",
        "what_reference_does": (
            "Predicts a per-node velocity/displacement (via a learned motion module) to project an expected "
            "next-frame position, then links in two passes: a tight, low-tolerance gate on motion-corrected "
            "distance first (locking in confident links), then a relaxed gate on remaining unlinked nodes. Cost "
            "combines motion-corrected distance + raw distance minus a classifier-confidence bonus."
        ),
        "why_it_might_improve_metric": (
            "M12's only motion prior is a single-step nearest-neighbor reference-displacement chain (no fitted "
            "velocity). A smoothed velocity estimate would reduce FP under crowding (high local_density) and "
            "reduce FN for tracks whose single-step raw nearest-neighbor briefly picks a wrong candidate."
        ),
        "how_to_port_into_m12": (
            "Extend build_reference_displacement_by_node into a short-window weighted-average velocity estimate; "
            "add a motion_distance_um feature ALONGSIDE (not replacing) the existing distance_um feature; "
            "optionally add a two-pass variant of select_greedy_exclusive_per_frame (tight motion-gate pass, then "
            "the existing 0.9-quantile pass on whatever remains unlinked)."
        ),
        "oof_experiment_design": (
            "Milestone 14 Experiments C (motion_relink only) and D (motion_relink + linefit) - measure aggregate "
            "pooled OOF edge_jaccard/TP/FP delta vs the M12 baseline on the same 12 robust samples, via GroupKFold "
            "exactly as Milestone 11 did (never a raw training-fit number)."
        ),
        "expected_risk": (
            "Medium - velocity estimation on short/noisy tracklets adds a new hyperparameter surface (window, "
            "damping) that could overfit to the 12 training samples without generalizing."
        ),
        "implementation_complexity": "Medium - reuses existing candidate-pool infrastructure, no new sklearn model.",
        "expected_gain": "Moderate",
    },
    {
        "idea_id": "output_topology_repair", "priority_rank": 3,
        "title": "Output-topology repair / invariant guard",
        "what_reference_does": (
            "Post-processes the final predicted graph to guarantee structural invariants regardless of what the "
            "upstream model produced: no node with >1 parent, only t->t+1 edges except through an explicit "
            "controlled gap-handling exception, and pruning of any edge with an implausibly large implied distance."
        ),
        "why_it_might_improve_metric": (
            "Largely a correctness net, not a scoring improvement per se - M12 ALREADY structurally guarantees "
            "single-parent/single-child (enforce_edge_exclusivity) and t->t+1-only edges (the ceiling pool never "
            "spans more than one frame) by construction. The only genuinely new piece is formalizing this as an "
            "explicit, permanent regression guard for every future experiment."
        ),
        "how_to_port_into_m12": (
            "Extend validate_submission's existing dangling/duplicate/uniqueness checks with explicit degree-"
            "invariant assertions (max out-degree<=1, or <=2 only when safe_division_recovery is deliberately "
            "enabled; max in-degree<=1) - run inside every Milestone 14 experiment's evaluation loop, not just at "
            "final submission time."
        ),
        "oof_experiment_design": (
            "Not a metric experiment - a structural CI-style check applied to every experiment (A-I) to catch any "
            "variant (especially gap_close or safe_divisions) that accidentally breaks an invariant it wasn't "
            "supposed to touch."
        ),
        "expected_risk": "Lowest - a validation net; cannot regress the metric.",
        "implementation_complexity": "Low - extends existing validation checks with 2-3 more assertions.",
        "expected_gain": "None directly (protects the other ideas from silent regressions)",
    },
    {
        "idea_id": "denser_node_policy", "priority_rank": 4,
        "title": "Denser node-reconstruction policy (node_policy B)",
        "what_reference_does": (
            "Emits far more of its confidently-detected nodes even when they are not part of a long, quality-"
            "filtered tracklet - almost certainly the single biggest driver of the reference's ~217x row-count gap "
            "versus M12's node_policy=A_tracklet_nodes_only (min_tracklet_length=5)."
        ),
        "why_it_might_improve_metric": (
            "If the real grader scores per-frame node recall (not just edge recall), M12 is likely paying a large, "
            "currently invisible node-recall penalty for omitting every node in short tracklets (<5 frames) or "
            "nodes squeezed out by the keep_top_k=100 budget - information Milestone 13 cannot determine from row "
            "counts alone without knowing the real scoring formula's node-vs-edge weighting."
        ),
        "how_to_port_into_m12": (
            "Already-implemented, currently-unused-by-M12 machinery: reconstruct_policy_b (\"B_all_filtered_nodes_"
            "with_classifier_edges\" - all filtered nodes kept, only edges restricted to kept tracklets) is a "
            "one-line config swap from node_policy=A to B."
        ),
        "oof_experiment_design": (
            "Milestone 14 Experiment H (dense-node policy diagnostic) - report node-level precision/recall "
            "(node_matches/unmatched_gt_nodes/unmatched_pred_nodes, already computed by evaluate_tracklet_config) "
            "side by side with edge metrics, since this is a node-metric question M11's edge-only aggregate OOF "
            "never measured."
        ),
        "expected_risk": (
            "Low implementation risk, but HIGH metric-interpretation risk - without the real grading formula, "
            "\"denser\" could hurt (more isolated FP nodes) as easily as it could help."
        ),
        "implementation_complexity": "Low (config change), but genuinely needs real Kaggle score feedback to interpret.",
        "expected_gain": "Unknown until a real public/private score data point exists - diagnostic only for now",
    },
    {
        "idea_id": "safe_division_recovery", "priority_rank": 5,
        "title": "Safe division recovery (rare out-degree=2 events)",
        "what_reference_does": (
            "Explicitly models rare cell-division events (out-degree=2) under strict geometric gates (both "
            "children within a tight distance of the parent, similar node scores, symmetric smoothness) rather "
            "than treating single-outgoing-edge exclusivity as an absolute rule."
        ),
        "why_it_might_improve_metric": (
            "Milestone 10's GT diagnosis found division_candidate_frac~=0.0013 (real but rare). M12's "
            "GREEDY_SOURCE_CAPACITY_DEFAULT=1 hard-caps every node's outgoing edges at 1, meaning EVERY real "
            "division currently loses one of its two true child edges as a structurally-guaranteed FN - a small, "
            "currently invisible FN floor."
        ),
        "how_to_port_into_m12": (
            "Add a config-gated division_source_capacity=2 diagnostic mode to select_greedy_exclusive_per_frame/ "
            "enforce_edge_exclusivity (already documented as a \"diagnostic-only option\" in the M11/M12 "
            "docstrings) but only admit the second outgoing edge through a strict secondary gate (second-best "
            "candidate's classifier score also above a high absolute threshold AND comparable distance to the first)."
        ),
        "oof_experiment_design": (
            "Milestone 14 Experiment G (safe_divisions). Given the low base rate (~0.13% of nodes), expect a "
            "SMALL absolute TP gain; the real question is net FP/FN trade via the pooled aggregate metric, not a "
            "per-sample eyeball, and specifically checked against the Stable (n_samples_with_tp) criterion since a "
            "single lucky division event on one sample could look like a win while being noise."
        ),
        "expected_risk": "Medium - low base rate means high relative variance in per-sample impact.",
        "implementation_complexity": "Medium - a targeted, config-gated exception to an existing constraint, not a new model.",
        "expected_gain": "Small in absolute terms, given the rarity of true divisions",
    },
    {
        "idea_id": "one_frame_gap_closing", "priority_rank": 6,
        "title": "One-frame gap closing",
        "what_reference_does": (
            "When a track is broken by exactly one missing frame (nodes exist at t and t+2 but not a linked t+1), "
            "reuses an already-isolated node near the expected t+1 position or synthesizes a midpoint node, "
            "optionally refined by re-querying image-intensity centroid at the expected location."
        ),
        "why_it_might_improve_metric": (
            "Milestone 10's GT diagnosis found frac_edge_time_gap_skip=0.0 - GT edges never literally skip a "
            "frame - so gap-closing cannot recover a literal t->t+2 GT edge. Its only legitimate value is as a "
            "DETECTOR-RECALL fix in disguise: recovering a real but currently-missed/filtered t+1 node converts one "
            "missed 2-hop gap into two correctly-recoverable t->t+1 GT edges."
        ),
        "how_to_port_into_m12": (
            "At tracklet-construction time, when two accepted segments in the same dataset are separated by "
            "exactly one frame and their endpoints are within a plausible single-step distance/2, first check the "
            "RAW (pre-node-filter) detections near the expected midpoint and reuse a real raw node if one exists "
            "there (preferred over inventing coordinates); only synthesize a midpoint as a last resort."
        ),
        "oof_experiment_design": (
            "Milestone 14 Experiment E (gap_close only). Since this changes NODE composition (adds new node "
            "rows), evaluate both edge-level AND node-level precision/recall impact, and specifically check "
            "whether it increases FP more than TP."
        ),
        "expected_risk": (
            "Highest of all 6 ideas - synthesizing or reusing a node the node filter specifically rejected risks "
            "re-admitting the exact low-confidence nodes Milestone 8 found explode FP when the node filter is "
            "loosened."
        ),
        "implementation_complexity": "High - tracklet-level post-processing, new node-provenance-aware insertion, careful re-validation against Milestone 8's node-filter rationale.",
        "expected_gain": "Uncertain - plausible but the riskiest bet of the six",
    },
]

# --------------------------------------------------------------------------- #
# 9. Part E orchestration - save the structured ideas + generate the
#    prose "winning path" write-up
# --------------------------------------------------------------------------- #
def run_part_e_transferable_ideas(out_dir: Path) -> list[dict]:
    ideas = sorted(TRANSFERABLE_IDEAS, key=lambda d: d["priority_rank"])
    with open(out_dir / "milestone13_transferable_ideas.json", "w") as f:
        json.dump(ideas, f, indent=2, default=str)

    lines = [
        "# Milestone 13 - Winning Path: Transferable Ideas from the Reference Solution",
        "",
        "Ranked by priority (expected gain per unit of implementation risk), NOT by the",
        "reference solution's own architecture - none of these require the reference's",
        "learned UNet, node-transformer, edge-predictor weights, or ILP solver.",
        "",
    ]
    for idea in ideas:
        lines.append(f"## {idea['priority_rank']}. {idea['title']} (`{idea['idea_id']}`)")
        lines.append("")
        lines.append(f"**What the reference solution does:** {idea['what_reference_does']}")
        lines.append("")
        lines.append(f"**Why it might improve the metric:** {idea['why_it_might_improve_metric']}")
        lines.append("")
        lines.append(f"**How to port into M12:** {idea['how_to_port_into_m12']}")
        lines.append("")
        lines.append(f"**OOF experiment design:** {idea['oof_experiment_design']}")
        lines.append("")
        lines.append(f"**Expected risk:** {idea['expected_risk']}")
        lines.append("")
        lines.append(f"**Implementation complexity:** {idea['implementation_complexity']}")
        lines.append("")
        lines.append(f"**Expected gain:** {idea['expected_gain']}")
        lines.append("")
    with open(out_dir / "milestone13_winning_path.md", "w") as f:
        f.write("\n".join(lines))

    return ideas

# --------------------------------------------------------------------------- #
# 10. Part F - Milestone 14 implementation plan (M12 + graph-repair hybrid)
# --------------------------------------------------------------------------- #
MILESTONE14_METRICS = [
    "node_precision", "node_recall", "edge_TP", "edge_FP", "edge_FN", "edge_jaccard",
    "precision", "recall", "division_like_source_count", "multi_parent_count",
    "t_gap_distribution", "per_sample_score", "aggregate_pooled_score",
]

MILESTONE14_SUCCESS_CRITERIA = [
    f"aggregate OOF edge_jaccard > {M11_BEST_EDGE_JACCARD} (Milestone 11's validated result)",
    "OR the same edge_jaccard with lower FP and better stability (n_samples_with_tp) than Milestone 11's 6/12",
    f"n_samples_with_tp > {M11_N_SAMPLES_WITH_TP} (must beat, not just match, Milestone 11's stability)",
    "no invalid topology (max out-degree<=1 unless safe_divisions is deliberately enabled and gated; max in-degree<=1; no dangling/duplicate edges)",
    "no catastrophic per-sample failure (no held-out sample regresses to near-zero edge_jaccard relative to the M12 baseline reproduction)",
]

MILESTONE14_EXPERIMENTS = [
    {
        "experiment_id": "A", "name": "M12 baseline reproduced",
        "description": "Exact M12 primary config, re-evaluated via GroupKFold on the same 12 robust train samples - re-establishes the 0.013215 anchor number under Milestone 14's own harness before any new idea is layered on.",
        "config_deltas": {},
        "depends_on": [],
    },
    {
        "experiment_id": "B", "name": "M12 + linefit_smoothing only",
        "description": "Post-process reconstruct_policy_a's node coordinates with window=2/weight=0.8 line-fit smoothing. Topology-preserving (idea: line_fit_smoothing) - expected near-zero or small positive delta.",
        "config_deltas": {"linefit_smoothing": True},
        "depends_on": ["A"],
    },
    {
        "experiment_id": "C", "name": "M12 + motion_relink only",
        "description": "Replace the single-step reference-displacement chain with a short-window velocity estimate; add motion_distance_um as an additional classifier feature; retrain the classifier with this feature included (idea: motion_relink).",
        "config_deltas": {"motion_relink": True, "extra_features": ["motion_distance_um"]},
        "depends_on": ["A"],
    },
    {
        "experiment_id": "D", "name": "M12 + motion_relink + linefit",
        "description": "Combine C and B - tests whether the two lowest-interaction ideas stack additively.",
        "config_deltas": {"motion_relink": True, "extra_features": ["motion_distance_um"], "linefit_smoothing": True},
        "depends_on": ["B", "C"],
    },
    {
        "experiment_id": "E", "name": "M12 + gap_close",
        "description": "One-frame gap closing via raw-detection reuse (preferred) or synthetic midpoint insertion (fallback) between tracklet segments separated by exactly one frame (idea: one_frame_gap_closing) - the highest-risk experiment; report node-level precision/recall alongside edge metrics since this is the only experiment that changes node composition.",
        "config_deltas": {"gap_close": True},
        "depends_on": ["A"],
    },
    {
        "experiment_id": "F", "name": "M12 + motion_relink + gap_close + linefit",
        "description": "The full stack of the 3 edge/node-affecting ideas - only run if C, E individually show non-negative deltas; this is the 'best case' combined experiment, not a default recommendation.",
        "config_deltas": {"motion_relink": True, "extra_features": ["motion_distance_um"], "gap_close": True, "linefit_smoothing": True},
        "depends_on": ["B", "C", "E"],
    },
    {
        "experiment_id": "G", "name": "M12 + safe_divisions",
        "description": "Config-gated division_source_capacity=2 in the edge-selection/exclusivity step, admitted only through a strict secondary gate (idea: safe_division_recovery) - expect a small absolute TP change given division_candidate_frac~=0.0013; check Stable criterion carefully given the low base rate.",
        "config_deltas": {"division_source_capacity": 2},
        "depends_on": ["A"],
    },
    {
        "experiment_id": "H", "name": "M12 + dense-node policy diagnostic",
        "description": "Swap node_policy from A_tracklet_nodes_only to B_all_filtered_nodes_with_classifier_edges (idea: denser_node_policy) - a DIAGNOSTIC, not a committed change: report node-level precision/recall/isolated-node-fraction side by side with edge metrics, since M11's edge-only aggregate OOF never measured this tradeoff.",
        "config_deltas": {"node_policy": "B_all_filtered_nodes_with_classifier_edges"},
        "depends_on": ["A"],
    },
    {
        "experiment_id": "I", "name": "M12 primary/conservative ensemble diagnostic",
        "description": "Union or intersection of the primary (hist_gradient_boosting/greedy_exclusive) and conservative (extra_trees/per_source_top1) M12 configs' predicted edges per sample - tests whether the 2 already-built, already-validated configs are complementary (recover different TPs) or redundant.",
        "config_deltas": {"ensemble_of": ["primary", "conservative_diagnostic"]},
        "depends_on": ["A"],
    },
]


def run_part_f_milestone14_plan(out_dir: Path) -> dict:
    plan = {
        "base_config": M12_PRIMARY_CONFIG_SUMMARY,
        "evaluation_data": "the SAME 12 fixed-seed robust train samples from Milestone 11 (select_train_samples_robust, seed=0) - NOT test data",
        "evaluation_method": "GroupKFold aggregate pooled OOF, identical discipline to Milestone 11 (pooled TP/FP/FN, never a mean of per-sample edge_jaccard)",
        "metrics": MILESTONE14_METRICS,
        "success_criteria": MILESTONE14_SUCCESS_CRITERIA,
        "anchor_numbers": {
            "m11_best_edge_jaccard": M11_BEST_EDGE_JACCARD, "m11_edge_TP": M11_EDGE_TP, "m11_edge_FP": M11_EDGE_FP,
            "m11_edge_FN": M11_EDGE_FN, "m11_n_samples": M11_N_SAMPLES, "m11_n_samples_with_tp": M11_N_SAMPLES_WITH_TP,
            "baseline_6c6d_edge_jaccard": BASELINE_6C6D_EDGE_JACCARD,
        },
        "experiments": MILESTONE14_EXPERIMENTS,
    }
    with open(out_dir / "milestone13_milestone14_plan.json", "w") as f:
        json.dump(plan, f, indent=2, default=str)

    recommended_prompt = f"""Implement Milestone 14: M12 + graph-repair hybrid.

Base config (M12 primary, unchanged): {json.dumps(M12_PRIMARY_CONFIG_SUMMARY)}

Evaluate on the SAME 12 fixed-seed robust train samples from Milestone 11
(select_train_samples_robust, seed=0) via GroupKFold aggregate pooled OOF -
NOT on test data.

Run experiments A-I exactly as specified in milestone13_milestone14_plan.json:
A) M12 baseline reproduced
B) + linefit_smoothing only
C) + motion_relink only
D) + motion_relink + linefit
E) + gap_close
F) + motion_relink + gap_close + linefit
G) + safe_divisions
H) + dense-node policy diagnostic
I) + primary/conservative ensemble diagnostic

Report per experiment: {", ".join(MILESTONE14_METRICS)}.

Success criteria:
{chr(10).join("- " + c for c in MILESTONE14_SUCCESS_CRITERIA)}

Do not build a new submission.csv in Milestone 14 unless explicitly instructed -
this is another OOF validation milestone, matching Milestone 11's structure.
"""
    with open(out_dir / "milestone13_recommended_next_prompt.txt", "w") as f:
        f.write(recommended_prompt)

    return plan

# --------------------------------------------------------------------------- #
# 11. Part G - final human-readable recommendation
# --------------------------------------------------------------------------- #
RECOMMENDATION_OPTIONS = {
    "A": "Replicate the reference solution legally first",
    "B": "Port motion_relink + linefit into M12 first",
    "C": "Port gap_close first",
    "D": "Train a stronger detector first",
    "E": "Wait for M12's public score before coding Milestone 14",
    "F": "Run the reference solution as-is (only if the artifact is actually available)",
}


def determine_final_recommendation(artifact_compliance: dict) -> dict:
    """Derives the printed A-F recommendation from Part D's artifact-
    availability finding - the decision rule is stated explicitly (per
    architecture review) so the recommendation is auditable rather than
    looking hand-picked.
    """
    replication_possible = bool(artifact_compliance.get("replication_possible_in_kaggle_offline_mode", False))
    if replication_possible:
        recommendation = "F"
        rationale = (
            "The reference artifact/weights/scripts/offline wheels all appear present on disk, so running the "
            "reference solution as-is is feasible without any new implementation risk."
        )
    else:
        recommendation = "B"
        rationale = (
            "The reference artifact/weights/scripts were NOT found to be available on disk (this audit never "
            "assumes availability), so legally replicating the reference solution (option A/F) is not currently "
            "actionable. Porting motion_relink + linefit_smoothing into M12 (the two lowest-risk, no-deep-model "
            "ideas from Part E) is the pragmatic default: linefit_smoothing is free/topology-preserving and "
            "motion_relink is a moderate-complexity, moderate-gain extension of machinery M12 already has."
        )
    return {
        "recommendation": recommendation, "rationale": rationale,
        "decision_rule": "replication_possible_in_kaggle_offline_mode == True -> F, else -> B",
    }


def run_part_g_final_recommendation(artifact_compliance: dict, out_dir: Path) -> dict:
    decision = determine_final_recommendation(artifact_compliance)
    with open(out_dir / "milestone13_final_recommendation.json", "w") as f:
        json.dump({**decision, "options": RECOMMENDATION_OPTIONS}, f, indent=2, default=str)

    print("\n=== Part G: Final recommendation ===")
    for key, label in RECOMMENDATION_OPTIONS.items():
        marker = "  <-- RECOMMENDED" if key == decision["recommendation"] else ""
        print(f"  {key}) {label}{marker}")
    print(f"\nDecision rule: {decision['decision_rule']}")
    print(f"Rationale: {decision['rationale']}")
    print("\nNo submission was generated by Milestone 13.")
    print("Do not Save Version / Submit from this audit notebook.")
    return decision

# --------------------------------------------------------------------------- #
# 12. Orchestration - ties Parts A-G together
# --------------------------------------------------------------------------- #
def run_milestone13_audit(working_dir: str = "/kaggle/working") -> dict:
    """Runs the full Milestone 13 audit: locates the reference package (if
    any), statically extracts its notebook, computes distribution stats on
    its submission, compares against our own M12 submission, checks
    artifact availability, writes the transferable-ideas note and
    Milestone 14 plan, and prints one final recommendation. Never builds a
    submission, never calls the Kaggle API, never triggers Save Version.
    """
    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Milestone 13: Top Solution Audit and Winning Path Design ===")
    print("(AUDIT ONLY - no submission will be built, no Kaggle API is called, no Save Version is triggered.)")

    reference_roots = get_reference_search_roots()
    print(f"\nReference search roots: {[str(r) for r in reference_roots]}")
    reference_files = find_reference_files(reference_roots)
    for key, path in reference_files.items():
        print(f"  {key}: {path if path else 'NOT FOUND'}")

    print("\n=== Part A: static notebook extraction ===")
    part_a_result = run_part_a_notebook_extraction(reference_files.get("notebook"), out_dir)
    if part_a_result["available"]:
        print(f"  available=True, title={part_a_result.get('title')!r}, n_functions_found={part_a_result.get('n_functions_found')}")
    else:
        print(f"  available=False ({part_a_result.get('reason')})")

    print("\n=== Part B: reference submission distribution analysis ===")
    reference_bundle = run_part_b_reference_stats(reference_files.get("submission"), out_dir)
    if reference_bundle["available"]:
        stats = reference_bundle["stats"]
        print(f"  available=True, shape={stats['generic']['shape']}, schema={stats['schema']}")
        if stats["schema"] == "recognized":
            print(f"  n_node_rows={stats['n_node_rows']} n_edge_rows={stats['n_edge_rows']} datasets={stats['datasets']}")
    else:
        print(f"  available=False ({reference_bundle['stats'].get('reason')})")

    m12_path = find_m12_submission(working_dir)
    print(f"\nM12 own submission path: {m12_path if m12_path else 'NOT FOUND'}")
    m12_df = load_submission_csv(m12_path) if m12_path else None
    m12_stats = compute_submission_stats(m12_df) if m12_df is not None else None
    m12_bundle = {"available": m12_df is not None, "df": m12_df, "stats": m12_stats}

    sanity_check_reference_vs_m12_rowcounts(reference_bundle.get("df"), m12_bundle.get("df"))

    print("\n=== Part C: M12 vs reference comparison ===")
    density_gap_report = run_part_c_comparison(m12_bundle, reference_bundle, out_dir)
    if density_gap_report.get("available"):
        print(f"  {density_gap_report['interpretation']}")
    else:
        print(f"  available=False ({density_gap_report.get('reason')})")

    print("\n=== Part D: artifact and compliance check ===")
    artifact_compliance = run_part_d_artifact_compliance(reference_roots, out_dir)
    for key in (
        "artifact_path_exists", "weights_exist", "scripts_exist", "wheels_dir_exists",
        "packages_installable_offline", "replication_possible_in_kaggle_offline_mode",
    ):
        print(f"  {key}: {artifact_compliance[key]}")

    print("\n=== Part E: transferable ideas ===")
    ideas = run_part_e_transferable_ideas(out_dir)
    for idea in ideas:
        print(f"  [{idea['priority_rank']}] {idea['idea_id']} - expected_gain={idea['expected_gain'][:50]}")

    print("\n=== Part F: Milestone 14 plan ===")
    plan = run_part_f_milestone14_plan(out_dir)
    print(f"  {len(plan['experiments'])} experiment(s) planned: {[e['experiment_id'] for e in plan['experiments']]}")

    decision = run_part_g_final_recommendation(artifact_compliance, out_dir)

    return {
        "reference_files": reference_files, "part_a": part_a_result, "reference_bundle": reference_bundle,
        "m12_bundle": m12_bundle, "density_gap_report": density_gap_report, "artifact_compliance": artifact_compliance,
        "ideas": ideas, "milestone14_plan": plan, "final_recommendation": decision,
    }

# --------------------------------------------------------------------------- #
# 13. Tests
# --------------------------------------------------------------------------- #
def run_milestone13_tests() -> None:
    """Correctness tests for the Milestone 13 audit machinery: notebook
    static parsing on a synthetic notebook, submission-stats computation
    (degree/t-gap/dangling/duplicate-edge correctness) on synthetic
    submissions, M12-vs-reference comparison on tiny synthetic data, and
    graceful "not found" path resolution - all on small in-memory/temp-
    file fixtures, no real reference files, no notebook execution.
    """
    # Test 1: notebook static parsing on a small SYNTHETIC notebook (never
    # the real reference notebook) - config values, pip installs, weights/
    # artifact paths, and function classification by keyword heuristic.
    synthetic_nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Test Reference Notebook\n", "some descriptive text\n"]},
            {"cell_type": "code", "source": [
                "TITLE = 'unet_transformer'\n",
                "det_threshold = 0.99\n",
                "use_ilp = True\n",
                "weights_path = 'weights/unet_transformer/split_0/edge_predictor_best.pth'\n",
                "artifact = 'biohub-tracking-support-pack-50ep-v1'\n",
                "!pip install some-fake-package\n",
                "\n",
                "def motion_relink(nodes, edges):\n",
                "    '''Relinks tracks using motion prediction.'''\n",
                "    return edges\n",
                "\n",
                "def gap_closing(tracks):\n",
                "    '''Closes one-frame gaps.'''\n",
                "    return tracks\n",
                "\n",
                "def write_submission(df, path):\n",
                "    '''Writes the final submission.csv.'''\n",
                "    return df\n",
            ]},
        ],
    }
    tmp_nb_path = Path("/tmp") / "m13_synthetic_reference_notebook_test.ipynb"
    tmp_nb_path.write_text(json.dumps(synthetic_nb))
    tmp_out_dir = Path("/tmp") / "m13_test_scratch_out"
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    part_a_result = run_part_a_notebook_extraction(tmp_nb_path, tmp_out_dir)
    assert part_a_result["available"]
    assert part_a_result["title"] == "Test Reference Notebook"
    assert any("pip install some-fake-package" in p for p in part_a_result["pip_installs"])
    assert any(p.endswith("edge_predictor_best.pth") for p in part_a_result["weights_paths"])
    assert any("support-pack" in p for p in part_a_result["artifact_paths"])
    function_names_by_category: dict = {}
    for fn in part_a_result["functions_found"]:
        for cat in fn["categories"]:
            function_names_by_category.setdefault(cat, []).append(fn["name"])
    assert "motion_relink" in function_names_by_category.get("motion_relinking", [])
    assert "gap_closing" in function_names_by_category.get("gap_closing", [])
    assert "write_submission" in function_names_by_category.get("submission_writing", [])

    with open(tmp_out_dir / "milestone13_reference_config.json") as f:
        reference_config = json.load(f)
    assert reference_config["config_values"]["det_threshold"] == 0.99
    assert reference_config["config_values"]["use_ilp"] is True

    # Test 2: submission-stats computation (degree/t-gap/dangling/
    # duplicate-edge correctness) on a hand-built synthetic submission - a
    # clean 3-node chain (0->1->2), an isolated node (3), a DUPLICATE edge
    # of (0,1) that must not double-count node 0's out-degree, and a
    # DANGLING edge (source_id=99 does not exist).
    rows2 = [
        {"id": 0, "dataset": "d1", "row_type": "node", "node_id": 0, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 1, "dataset": "d1", "row_type": "node", "node_id": 1, "t": 1, "z": 0.0, "y": 1.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 2, "dataset": "d1", "row_type": "node", "node_id": 2, "t": 2, "z": 0.0, "y": 2.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 3, "dataset": "d1", "row_type": "node", "node_id": 3, "t": 0, "z": 0.0, "y": 10.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 4, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 1},
        {"id": 5, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 1, "target_id": 2},
        {"id": 6, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 1},
        {"id": 7, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 99, "target_id": 1},
    ]
    df2 = pd.DataFrame(rows2, columns=SUBMISSION_COLUMNS)
    stats2 = compute_submission_stats(df2)
    assert stats2["schema"] == "recognized"
    assert stats2["n_node_rows"] == 4 and stats2["n_edge_rows"] == 4
    assert stats2["dangling_edge_count"] == 1, "only the source_id=99 edge is dangling"
    assert stats2["duplicate_edge_count"] == 1, "the second (0,1) edge is a duplicate"
    assert stats2["division_like_source_count"] == 0, "node 0's TRUE out-degree is 1 once the duplicate is deduped, not 2"
    assert stats2["multi_parent_child_count"] == 0
    assert stats2["frac_edges_t_to_t1"] == 1.0, "both real (deduped) edges are t->t+1"
    degree_df2 = stats2["degree_stats"]
    node0_out_degree = int(degree_df2.loc[degree_df2["node_id"] == 0, "out_degree"].iloc[0])
    assert node0_out_degree == 1, "duplicate edge must not double-count out-degree"
    node3_degrees = degree_df2.loc[degree_df2["node_id"] == 3, ["in_degree", "out_degree"]].iloc[0]
    assert node3_degrees["in_degree"] == 0 and node3_degrees["out_degree"] == 0, "isolated node 3 must still appear with degree 0, not be dropped"

    # Test 3: degree/multi-parent/division-like checks on a fixture with a
    # genuine division (node 0 -> both 1 and 2) and a genuine multi-parent
    # child (node 5 <- both 3 and 4).
    rows3 = [
        {"id": 0, "dataset": "d2", "row_type": "node", "node_id": 0, "t": 0, "z": 0.0, "y": 0.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 1, "dataset": "d2", "row_type": "node", "node_id": 1, "t": 1, "z": 0.0, "y": 1.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 2, "dataset": "d2", "row_type": "node", "node_id": 2, "t": 1, "z": 0.0, "y": -1.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 3, "dataset": "d2", "row_type": "node", "node_id": 3, "t": 0, "z": 0.0, "y": 5.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 4, "dataset": "d2", "row_type": "node", "node_id": 4, "t": 0, "z": 0.0, "y": 5.5, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 5, "dataset": "d2", "row_type": "node", "node_id": 5, "t": 1, "z": 0.0, "y": 5.2, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 6, "dataset": "d2", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 1},
        {"id": 7, "dataset": "d2", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0, "target_id": 2},
        {"id": 8, "dataset": "d2", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 3, "target_id": 5},
        {"id": 9, "dataset": "d2", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 4, "target_id": 5},
    ]
    df3 = pd.DataFrame(rows3, columns=SUBMISSION_COLUMNS)
    stats3 = compute_submission_stats(df3)
    assert stats3["division_like_source_count"] == 1, "only node 0 has out_degree==2"
    assert stats3["multi_parent_child_count"] == 1, "only node 5 has in_degree>1 (=2)"
    assert stats3["dangling_edge_count"] == 0 and stats3["duplicate_edge_count"] == 0

    # Test 4: M12-vs-reference comparison on tiny synthetic data - the
    # "reference" is deliberately built denser than "m12" to confirm the
    # density-ratio arithmetic and interpretation string are correct.
    m12_bundle_test = {"available": True, "df": df2, "stats": compute_submission_stats(df2)}
    reference_rows4 = rows2 + [
        {"id": 8, "dataset": "d1", "row_type": "node", "node_id": 10, "t": 0, "z": 0.0, "y": 20.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 9, "dataset": "d1", "row_type": "node", "node_id": 11, "t": 1, "z": 0.0, "y": 21.0, "x": 0.0, "source_id": -1, "target_id": -1},
        {"id": 10, "dataset": "d1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 10, "target_id": 11},
    ]
    df4_ref = pd.DataFrame(reference_rows4, columns=SUBMISSION_COLUMNS)
    reference_bundle_test = {"available": True, "df": df4_ref, "stats": compute_submission_stats(df4_ref)}

    density_report = run_part_c_comparison(m12_bundle_test, reference_bundle_test, tmp_out_dir)
    assert density_report["available"]
    assert density_report["reference_total_rows"] > density_report["m12_total_rows"]
    assert density_report["density_ratio_total_rows"] > 1.0
    assert "more total rows" in density_report["interpretation"]

    # Test 5: graceful "not found" path resolution - searching a root with
    # nothing matching must return all-None without raising, and Part
    # B/Part A must both report `available=False` cleanly rather than
    # crashing (covers the "if available, else report clearly" branch this
    # sandbox always exercises for the real reference files).
    empty_root = Path("/tmp") / "m13_test_empty_reference_root"
    empty_root.mkdir(parents=True, exist_ok=True)
    found_none = find_reference_files([empty_root])
    assert all(v is None for v in found_none.values())
    part_a_missing = run_part_a_notebook_extraction(None, tmp_out_dir)
    assert part_a_missing["available"] is False
    part_b_missing = run_part_b_reference_stats(None, tmp_out_dir)
    assert part_b_missing["available"] is False

    print("All milestone13_top_solution_audit tests passed (5/5).")

# --------------------------------------------------------------------------- #
# 14. Top-level driver
# --------------------------------------------------------------------------- #
def run_milestone13_pipeline(working_dir: str = "/kaggle/working") -> dict:
    """Runs the unit tests, then the full Milestone 13 audit (Parts A-G).
    Never builds a submission, never calls the Kaggle API, never triggers
    Save Version.
    """
    print(
        "=== Self-test: notebook parsing, submission-stats, degree/t-gap/dangling-edge, "
        "M12-vs-reference comparison, and path-resolution unit tests ==="
    )
    run_milestone13_tests()
    return run_milestone13_audit(working_dir=working_dir)


if __name__ == "__main__":
    run_milestone13_pipeline()
