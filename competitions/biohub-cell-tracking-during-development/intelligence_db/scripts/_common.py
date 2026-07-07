"""Shared paths + DB helpers for the Biohub intelligence database.

DuckDB is the structured store; JSON files under data/ are the seed/import
source of truth; Markdown under reports/ is regenerated from the DB.

Loaders are delete-then-insert by key so every operation is idempotent:
re-running init or an update never duplicates rows.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "duckdb is required for the intelligence database. Install it with:\n"
        "  pip install duckdb\n"
        f"(import failed: {exc})"
    )

DB_ROOT = Path(__file__).resolve().parents[1]          # .../intelligence_db
SCHEMA_PATH = DB_ROOT / "schema.sql"
DB_PATH = DB_ROOT / "intelligence.duckdb"
DATA_DIR = DB_ROOT / "data"
REPORTS_DIR = DB_ROOT / "reports"
SEED_EXPERIMENTS = DATA_DIR / "seed_experiments.json"
SEED_PUBLIC_SOURCES = DATA_DIR / "public_sources.json"

# Column order for each table (keeps INSERTs explicit and stable).
EXPERIMENT_COLS = [
    "experiment_id", "milestone", "variant", "label", "notebook_name", "raw_file",
    "command_json", "det_threshold", "ilp_edge_weight", "ilp_appearance_weight",
    "ilp_disappearance_weight", "ilp_division_weight", "postprocess_ops_json",
    "public_score", "status", "created_at", "notes",
]
SUBMISSION_COLS = [
    "experiment_id", "n_rows", "n_nodes", "n_edges", "max_in_degree",
    "max_out_degree", "fallback_used", "valid", "final_source",
]
POSTPROCESS_COLS = [
    "experiment_id", "safe_divisions_added", "gap1_closed", "gap2_recovered",
    "synthetic_nodes_added", "isolated_nodes_pruned", "edges_removed",
    "nodes_before", "nodes_after", "edges_before", "edges_after", "per_dataset_json",
]
PUBLIC_SOURCE_COLS = [
    "source_id", "source_type", "title", "url", "repo", "path", "summary",
    "extracted_ideas_json", "relevance_score",
]
DECISION_COLS = [
    "decision_id", "after_experiment_id", "observation", "recommendation",
    "next_variant", "risk_level", "created_at",
]
LESSON_COLS = ["lesson_id", "topic", "lesson", "evidence_experiment_ids", "confidence"]


def connect(db_path: Path | str = DB_PATH):
    """Opens (creating if needed) the DuckDB database."""
    return duckdb.connect(str(db_path))


def apply_schema(con) -> None:
    con.execute(SCHEMA_PATH.read_text())


def _placeholders(n: int) -> str:
    return ", ".join(["?"] * n)


def _row(record: dict, cols: list[str]) -> list:
    return [record.get(c) for c in cols]


def upsert_by_key(con, table: str, cols: list[str], key_col: str, record: dict) -> None:
    """Delete-then-insert a single row keyed by `key_col` (idempotent)."""
    con.execute(f"DELETE FROM {table} WHERE {key_col} = ?", [record[key_col]])
    con.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({_placeholders(len(cols))})",
        _row(record, cols),
    )


def replace_children(con, table: str, cols: list[str], key_col: str, key_value, records: list[dict]) -> None:
    """Replace all child rows for one key (submission_stats / postprocess_stats
    have no PK; keying on experiment_id keeps updates idempotent)."""
    con.execute(f"DELETE FROM {table} WHERE {key_col} = ?", [key_value])
    for rec in records:
        con.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({_placeholders(len(cols))})",
            _row(rec, cols),
        )


def experiment_record(exp: dict) -> dict:
    """Flattens a seed/import experiment dict into the experiments-table row
    shape (command + ops serialized to JSON, the five ILP knobs lifted out)."""
    cmd = exp.get("command", {}) or {}
    return {
        "experiment_id": exp["experiment_id"],
        "milestone": exp.get("milestone"),
        "variant": exp.get("variant"),
        "label": exp.get("label"),
        "notebook_name": exp.get("notebook_name"),
        "raw_file": exp.get("raw_file"),
        "command_json": json.dumps(cmd, sort_keys=True),
        "det_threshold": cmd.get("det_threshold"),
        "ilp_edge_weight": cmd.get("ilp_edge_weight"),
        "ilp_appearance_weight": cmd.get("ilp_appearance_weight"),
        "ilp_disappearance_weight": cmd.get("ilp_disappearance_weight"),
        "ilp_division_weight": cmd.get("ilp_division_weight"),
        "postprocess_ops_json": json.dumps(exp.get("postprocess_ops", [])),
        "public_score": exp.get("public_score"),
        "status": exp.get("status") or ("scored" if exp.get("public_score") is not None else "pending"),
        "created_at": exp.get("created_at"),
        "notes": exp.get("notes"),
    }


def submission_record(experiment_id: str, stats: dict) -> dict:
    s = stats or {}
    return {
        "experiment_id": experiment_id,
        "n_rows": s.get("n_rows"), "n_nodes": s.get("n_nodes"), "n_edges": s.get("n_edges"),
        "max_in_degree": s.get("max_in_degree"), "max_out_degree": s.get("max_out_degree"),
        "fallback_used": s.get("fallback_used"), "valid": s.get("valid"),
        "final_source": s.get("final_source"),
    }


def postprocess_record(experiment_id: str, stats: dict) -> dict:
    s = stats or {}
    return {
        "experiment_id": experiment_id,
        "safe_divisions_added": s.get("safe_divisions_added", 0),
        "gap1_closed": s.get("gap1_closed", 0),
        "gap2_recovered": s.get("gap2_recovered", 0),
        "synthetic_nodes_added": s.get("synthetic_nodes_added", 0),
        "isolated_nodes_pruned": s.get("isolated_nodes_pruned", 0),
        "edges_removed": s.get("edges_removed", 0),
        "nodes_before": s.get("nodes_before"), "nodes_after": s.get("nodes_after"),
        "edges_before": s.get("edges_before"), "edges_after": s.get("edges_after"),
        "per_dataset_json": json.dumps(s.get("per_dataset", [])),
    }


def load_experiment(con, exp: dict) -> None:
    """Idempotently loads one experiment + its submission/postprocess stats."""
    upsert_by_key(con, "experiments", EXPERIMENT_COLS, "experiment_id", experiment_record(exp))
    if "submission_stats" in exp and exp["submission_stats"] is not None:
        replace_children(con, "submission_stats", SUBMISSION_COLS, "experiment_id",
                         exp["experiment_id"], [submission_record(exp["experiment_id"], exp["submission_stats"])])
    if "postprocess_stats" in exp and exp["postprocess_stats"] is not None:
        replace_children(con, "postprocess_stats", POSTPROCESS_COLS, "experiment_id",
                         exp["experiment_id"], [postprocess_record(exp["experiment_id"], exp["postprocess_stats"])])


def public_source_record(src: dict) -> dict:
    return {
        "source_id": src["source_id"], "source_type": src.get("source_type"),
        "title": src.get("title"), "url": src.get("url"), "repo": src.get("repo"),
        "path": src.get("path"), "summary": src.get("summary"),
        "extracted_ideas_json": json.dumps(src.get("extracted_ideas", [])),
        "relevance_score": src.get("relevance_score"),
    }
