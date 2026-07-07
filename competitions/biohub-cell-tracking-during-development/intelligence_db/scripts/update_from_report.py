#!/usr/bin/env python3
"""Update the intelligence DB from our own Kaggle run reports.

Reads a milestone submission report JSON and/or a GEFF conversion JSON
(produced by the milestone runners under /kaggle/working) plus an optional
public score, upserts the experiment (idempotent), and regenerates reports.

Report shapes it understands (all our own diagnostics):
  submission report  milestone19_reference_submission_report.json
     { variant, label, ops, command, submission_shape, n_node_rows,
       n_edge_rows, max_in_degree, max_out_degree, valid, fallback_used,
       final_source, ... }
  conversion report  milestone19_geff_conversion.json
     { ops, n_nodes_before, n_nodes_after, n_edges_before, n_edges_after,
       n_edges_removed?, per_dataset: [ { postprocess: { safe_divisions:{n_divisions_added},
       gap1:{n_gap1_closed}, gap2:{n_gap2_recovered}, prune_isolated:{n_pruned} } } ] }

Usage:
  python update_from_report.py --experiment-id M19_C_FULL_CHAIN_PENDING \\
      --submission-report /kaggle/working/milestone19_reference_submission_report.json \\
      --conversion-report /kaggle/working/milestone19_geff_conversion.json \\
      --public-score 0.881
  python update_from_report.py --experiment-id M19_C_FULL_CHAIN_PENDING --public-score 0.881

Missing score => status 'pending' (public_score NULL). Running twice with the
same inputs does not duplicate rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"report not found: {path}")
    return json.loads(p.read_text())


def _aggregate_postprocess(conversion: dict) -> dict:
    """Sums the per-dataset postprocess diagnostics into the flat stat row."""
    div = gap1 = gap2 = pruned = 0
    for d in conversion.get("per_dataset", []):
        pp = d.get("postprocess", {}) or {}
        div += (pp.get("safe_divisions", {}) or {}).get("n_divisions_added", 0)
        gap1 += (pp.get("gap1", {}) or {}).get("n_gap1_closed", 0)
        gap2 += (pp.get("gap2", {}) or {}).get("n_gap2_recovered", 0)
        pruned += (pp.get("prune_isolated", {}) or {}).get("n_pruned", 0)
    return {
        "safe_divisions_added": div, "gap1_closed": gap1, "gap2_recovered": gap2,
        "synthetic_nodes_added": gap1 + 2 * gap2, "isolated_nodes_pruned": pruned,
        "edges_removed": conversion.get("n_edges_removed", 0),
        "nodes_before": conversion.get("n_nodes_before"), "nodes_after": conversion.get("n_nodes_after"),
        "edges_before": conversion.get("n_edges_before"), "edges_after": conversion.get("n_edges_after"),
        "per_dataset": conversion.get("per_dataset", []),
    }


def build_experiment_dict(
    experiment_id: str, submission: dict, conversion: dict, public_score, overrides: dict,
) -> dict:
    cmd_notes = submission.get("command_notes", {}) or {}
    command = {
        "det_threshold": _num(cmd_notes.get("det_threshold", 0.99)),
        "ilp_edge_weight": -1.0, "ilp_appearance_weight": 0.1,
        "ilp_disappearance_weight": 0.1, "ilp_division_weight": 1.0, "use_ilp": True,
    }
    ops = submission.get("ops") or conversion.get("ops") or []
    shape = submission.get("submission_shape") or []
    exp = {
        "experiment_id": experiment_id,
        "milestone": overrides.get("milestone") or ("M" + "".join(ch for ch in experiment_id if ch.isdigit())[:2] or None),
        "variant": submission.get("variant") or overrides.get("variant"),
        "label": submission.get("label") or overrides.get("label"),
        "notebook_name": overrides.get("notebook_name"),
        "raw_file": overrides.get("raw_file"),
        "command": command,
        "postprocess_ops": ops,
        "public_score": public_score,
        "status": "scored" if public_score is not None else "pending",
        "created_at": overrides.get("created_at"),
        "notes": overrides.get("notes"),
        "submission_stats": {
            "n_rows": shape[0] if shape else None,
            "n_nodes": submission.get("n_node_rows"),
            "n_edges": submission.get("n_edge_rows"),
            "max_in_degree": submission.get("max_in_degree"),
            "max_out_degree": submission.get("max_out_degree"),
            "fallback_used": submission.get("fallback_used"),
            "valid": submission.get("valid"),
            "final_source": submission.get("final_source"),
        },
        "postprocess_stats": _aggregate_postprocess(conversion) if conversion else None,
    }
    return exp


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _merge_preserving(con, exp: dict) -> dict:
    """Fill missing scalar fields from an existing row so a score-only update
    doesn't wipe previously-loaded metadata (created_at, notes, raw_file...)."""
    existing = con.execute(
        "SELECT milestone, variant, label, notebook_name, raw_file, created_at, notes "
        "FROM experiments WHERE experiment_id = ?", [exp["experiment_id"]]
    ).fetchone()
    if not existing:
        return exp
    keys = ["milestone", "variant", "label", "notebook_name", "raw_file", "created_at", "notes"]
    for k, v in zip(keys, existing):
        if exp.get(k) in (None, "") and v is not None:
            exp[k] = v
    # Preserve prior stats if this update carries none.
    if exp.get("submission_stats") and all(v is None for v in exp["submission_stats"].values()):
        exp.pop("submission_stats")
    if exp.get("postprocess_stats") is None:
        exp.pop("postprocess_stats", None)
    return exp


def update(experiment_id: str, submission_path=None, conversion_path=None,
           public_score=None, overrides=None, db_path: Path = C.DB_PATH, regenerate=True) -> dict:
    submission = _load_json(submission_path)
    conversion = _load_json(conversion_path)
    overrides = overrides or {}

    exp = build_experiment_dict(experiment_id, submission, conversion, public_score, overrides)
    con = C.connect(db_path)
    try:
        C.apply_schema(con)  # ensure tables exist even on a fresh db
        exp = _merge_preserving(con, exp)
        C.load_experiment(con, exp)
        row = con.execute(
            "SELECT experiment_id, public_score, status FROM experiments WHERE experiment_id = ?",
            [experiment_id]).fetchone()
    finally:
        con.close()
    if regenerate:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate_reports  # noqa: E402
        generate_reports.generate_all(db_path)
    return {"experiment_id": row[0], "public_score": row[1], "status": row[2]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Update the intelligence DB from a run report.")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--submission-report", default=None)
    ap.add_argument("--conversion-report", default=None)
    ap.add_argument("--public-score", type=float, default=None, help="omit for pending")
    ap.add_argument("--milestone", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--notebook-name", default=None)
    ap.add_argument("--raw-file", default=None)
    ap.add_argument("--created-at", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--db", default=str(C.DB_PATH))
    ap.add_argument("--no-reports", action="store_true")
    args = ap.parse_args()

    overrides = {k: getattr(args, k) for k in
                 ["milestone", "variant", "label", "notebook_name", "raw_file", "created_at", "notes"]}
    result = update(
        args.experiment_id, args.submission_report, args.conversion_report,
        args.public_score, overrides, Path(args.db), regenerate=not args.no_reports)
    print(f"Updated {result['experiment_id']}: score={result['public_score']} status={result['status']}")
    if not args.no_reports:
        print("Reports regenerated.")


if __name__ == "__main__":
    main()
