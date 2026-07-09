#!/usr/bin/env python3
"""Quick read-only queries against the intelligence DuckDB database.

Commands:
  python query_db.py best                    highest scored experiment
  python query_db.py timeline                chronological scores + deltas
  python query_db.py pending                 experiments awaiting a score
  python query_db.py experiment <ID>         full detail for one experiment

Read-only: never writes to the DB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

BASELINE_SCORE = 0.874


def _con(db_path=C.DB_PATH):
    if not Path(db_path).exists():
        raise SystemExit(f"database not found: {db_path}\nRun init_intelligence_db.py first.")
    return C.connect(db_path)


def cmd_best(con) -> int:
    row = con.execute("""
        SELECT experiment_id, public_score, status FROM experiments
        WHERE public_score IS NOT NULL ORDER BY public_score DESC, created_at ASC LIMIT 1
    """).fetchone()
    if not row:
        print("No scored experiments yet.")
        return 0
    print(f"Best: {row[0]}  score={row[1]:.4f}  status={row[2]}")
    return 0


def cmd_timeline(con) -> int:
    rows = con.execute("""
        SELECT experiment_id, created_at, public_score, status
        FROM experiments ORDER BY created_at, experiment_id
    """).fetchall()
    prev = None
    print(f"{'date':<12} {'experiment':<32} {'score':<9} {'delta':<8} status")
    for exp_id, created, score, status in rows:
        if score is None:
            s, d = "pending", "—"
        else:
            s = f"{score:.4f}"
            d = f"{score - prev:+.3f}" if prev is not None else "—"
            prev = score
        print(f"{str(created):<12} {exp_id:<32} {s:<9} {d:<8} {status}")
    return 0


def cmd_pending(con) -> int:
    rows = con.execute("""
        SELECT experiment_id, created_at, status FROM experiments
        WHERE public_score IS NULL AND status NOT IN ('unsafe', 'blocked', 'diagnostic', 'wrong_artifact')
        ORDER BY created_at
    """).fetchall()
    if not rows:
        print("No pending experiments (see 'blocked' M22 true-base variants awaiting artifact recovery).")
        return 0
    print("Pending (awaiting public score):")
    for exp_id, created, status in rows:
        print(f"  {exp_id}  ({created}, {status})")
    return 0


def cmd_experiment(con, experiment_id: str) -> int:
    e = con.execute("""
        SELECT experiment_id, milestone, variant, label, notebook_name, raw_file,
               command_json, postprocess_ops_json, public_score, status, created_at, notes
        FROM experiments WHERE experiment_id = ?
    """, [experiment_id]).fetchone()
    if not e:
        print(f"No such experiment: {experiment_id}")
        return 1
    (eid, milestone, variant, label, notebook, raw_file, cmd_json, ops_json,
     score, status, created, notes) = e
    print(f"# {eid}")
    print(f"  milestone/variant : {milestone} / {variant}  ({label})")
    print(f"  notebook          : {notebook}")
    print(f"  raw_file          : {raw_file}")
    print(f"  created_at        : {created}")
    print(f"  status            : {status}")
    print(f"  public_score      : {'pending' if score is None else f'{score:.4f}'}"
          + ("" if score is None else f"  (Δbaseline {score - BASELINE_SCORE:+.3f})"))
    print(f"  command           : {cmd_json}")
    print(f"  postprocess_ops   : {', '.join(json.loads(ops_json or '[]')) or '—'}")
    print(f"  notes             : {notes}")

    s = con.execute("""
        SELECT n_rows, n_nodes, n_edges, max_in_degree, max_out_degree, fallback_used, valid, final_source
        FROM submission_stats WHERE experiment_id = ?
    """, [experiment_id]).fetchone()
    if s:
        print("  submission_stats  :")
        print(f"    rows={s[0]} nodes={s[1]} edges={s[2]} max_in/out={s[3]}/{s[4]}")
        print(f"    fallback_used={s[5]} valid={s[6]} final_source={s[7]}")

    p = con.execute("""
        SELECT safe_divisions_added, gap1_closed, gap2_recovered, synthetic_nodes_added,
               isolated_nodes_pruned, edges_removed, nodes_before, nodes_after, edges_before, edges_after
        FROM postprocess_stats WHERE experiment_id = ?
    """, [experiment_id]).fetchone()
    if p:
        print("  postprocess_stats :")
        print(f"    divisions+={p[0]} gap1={p[1]} gap2={p[2]} synth_nodes={p[3]} pruned={p[4]} edges_removed={p[5]}")
        print(f"    nodes {p[6]}->{p[7]}  edges {p[8]}->{p[9]}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    con = _con()
    try:
        if cmd == "best":
            return cmd_best(con)
        if cmd == "timeline":
            return cmd_timeline(con)
        if cmd == "pending":
            return cmd_pending(con)
        if cmd == "experiment":
            if not rest:
                print("usage: query_db.py experiment <EXPERIMENT_ID>")
                return 2
            return cmd_experiment(con, rest[0])
        print(f"unknown command: {cmd}")
        print(__doc__)
        return 2
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
