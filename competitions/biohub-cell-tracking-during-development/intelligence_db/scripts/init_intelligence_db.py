#!/usr/bin/env python3
"""Initialize (or re-initialize) the Biohub intelligence DuckDB database.

Creates the schema and loads the seed experiments, public sources, decisions,
and lessons. Idempotent: re-running never duplicates rows (delete-then-insert
by key). Then regenerates all Markdown reports.

Usage:
    python init_intelligence_db.py            # build ./intelligence.duckdb
    python init_intelligence_db.py --fresh    # delete the db file first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def load_seeds(con) -> dict:
    exp_seed = json.loads(C.SEED_EXPERIMENTS.read_text())
    src_seed = json.loads(C.SEED_PUBLIC_SOURCES.read_text())

    experiments = exp_seed.get("experiments", [])
    for exp in experiments:
        C.load_experiment(con, exp)

    for dec in exp_seed.get("decisions", []):
        C.upsert_by_key(con, "decisions", C.DECISION_COLS, "decision_id", dec)

    for les in exp_seed.get("lessons", []):
        C.upsert_by_key(con, "lessons", C.LESSON_COLS, "lesson_id", les)

    for src in src_seed.get("sources", []):
        C.upsert_by_key(con, "public_sources", C.PUBLIC_SOURCE_COLS, "source_id", C.public_source_record(src))

    return {
        "experiments": len(experiments),
        "decisions": len(exp_seed.get("decisions", [])),
        "lessons": len(exp_seed.get("lessons", [])),
        "public_sources": len(src_seed.get("sources", [])),
    }


def init_db(db_path: Path = C.DB_PATH, fresh: bool = False, regenerate: bool = True) -> dict:
    if fresh and Path(db_path).exists():
        Path(db_path).unlink()
    con = C.connect(db_path)
    try:
        C.apply_schema(con)
        counts = load_seeds(con)
    finally:
        con.close()
    if regenerate:
        # Imported lazily so init works even if reports module is edited.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_reports  # noqa: E402
        generate_reports.generate_all(db_path)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize the Biohub intelligence database.")
    ap.add_argument("--fresh", action="store_true", help="delete the db file before building")
    ap.add_argument("--no-reports", action="store_true", help="skip report regeneration")
    ap.add_argument("--db", default=str(C.DB_PATH), help="path to the DuckDB file")
    args = ap.parse_args()

    counts = init_db(Path(args.db), fresh=args.fresh, regenerate=not args.no_reports)
    print(f"Initialized {args.db}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if not args.no_reports:
        print("Reports regenerated under reports/.")


if __name__ == "__main__":
    main()
