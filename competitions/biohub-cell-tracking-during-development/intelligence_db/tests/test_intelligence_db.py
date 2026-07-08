#!/usr/bin/env python3
"""Tests for the Biohub intelligence database.

Covers: DB initializes, seed data inserted, reports generated, duplicate
update is idempotent, pending score handled, and best == M19-A while M19-C
is pending. Uses a temp DB file so the committed intelligence.duckdb is
never touched. Run: python tests/test_intelligence_db.py  (or via pytest).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import _common as C  # noqa: E402
import init_intelligence_db as initdb  # noqa: E402
import generate_reports as reports  # noqa: E402
import update_from_report as upd  # noqa: E402
import query_db as q  # noqa: E402


def _fresh_db(tmp: Path) -> Path:
    db = tmp / "test_intelligence.duckdb"
    initdb.init_db(db, fresh=True, regenerate=False)
    return db


def test_db_initializes_and_seeds(tmp: Path) -> None:
    db = _fresh_db(tmp)
    con = C.connect(db)
    try:
        n_exp = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        n_sub = con.execute("SELECT COUNT(*) FROM submission_stats").fetchone()[0]
        n_pp = con.execute("SELECT COUNT(*) FROM postprocess_stats").fetchone()[0]
        n_src = con.execute("SELECT COUNT(*) FROM public_sources").fetchone()[0]
        n_dec = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        n_les = con.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        ids = {r[0] for r in con.execute("SELECT experiment_id FROM experiments").fetchall()}
    finally:
        con.close()
    assert n_exp == 6, f"expected 6 seed experiments, got {n_exp}"
    assert n_sub == 6 and n_pp == 6, f"expected stats for all 6 (sub={n_sub}, pp={n_pp})"
    assert n_src == 8, f"expected 8 public sources, got {n_src}"
    assert n_dec >= 3 and n_les >= 5, f"expected seed decisions/lessons (dec={n_dec}, les={n_les})"
    for e in ["M16_BASELINE", "M17_C_DET_0985", "M18_C_EDGE_PRUNE",
              "M19_A_SAFE_DIVISIONS_PRUNE", "M19_C_FULL_CHAIN_PENDING", "M20_A_FULLCHAIN_TUNED"]:
        assert e in ids, f"missing seed experiment {e}"
    print("  ok: db initializes and seeds")


def test_reports_generated(tmp: Path) -> None:
    db = _fresh_db(tmp)
    # Redirect report output to a temp dir so the committed reports are untouched.
    orig = C.REPORTS_DIR
    tmp_reports = tmp / "reports"
    C.REPORTS_DIR = tmp_reports
    try:
        written = reports.generate_all(db)
    finally:
        C.REPORTS_DIR = orig
    for name in ["score_timeline.md", "experiment_matrix.md", "next_actions.md", "lessons_learned.md"]:
        assert name in written, f"{name} not generated"
        text = (tmp_reports / name).read_text()
        assert len(text) > 100, f"{name} looks empty"
    # Timeline must show baseline and the +0.003 steps to M19-A and M19-C.
    timeline = (tmp_reports / "score_timeline.md").read_text()
    assert "0.8740" in timeline and "0.8770" in timeline and "0.8800" in timeline and "+0.003" in timeline
    # M19-C scored 0.880 (> 0.877) -> next_actions is the "tune full_chain" branch.
    nxt = (tmp_reports / "next_actions.md").read_text()
    assert "0.8800" in nxt and "tune" in nxt.lower() and "full_chain" in nxt.lower()
    print("  ok: reports generated with expected content")


def test_scored_and_best(tmp: Path) -> None:
    db = _fresh_db(tmp)
    con = C.connect(db)
    try:
        m19c = con.execute("SELECT public_score, status FROM experiments WHERE experiment_id='M19_C_FULL_CHAIN_PENDING'").fetchone()
        m20 = con.execute("SELECT public_score, status FROM experiments WHERE experiment_id='M20_A_FULLCHAIN_TUNED'").fetchone()
        best = con.execute("SELECT experiment_id, public_score FROM experiments WHERE public_score IS NOT NULL ORDER BY public_score DESC, created_at LIMIT 1").fetchone()
        pending_ids = {r[0] for r in con.execute("SELECT experiment_id FROM experiments WHERE public_score IS NULL").fetchall()}
    finally:
        con.close()
    assert m19c[0] is not None and abs(m19c[0] - 0.880) < 1e-9 and m19c[1] == "scored", f"M19-C should be scored 0.880, got {m19c}"
    assert best[0] == "M19_C_FULL_CHAIN_PENDING" and abs(best[1] - 0.880) < 1e-9, \
        f"best should be M19-C @0.880 while M20 pending, got {best}"
    assert m20[0] is None and m20[1] == "pending", f"M20 should be pending, got {m20}"
    assert pending_ids == {"M20_A_FULLCHAIN_TUNED"}, f"only M20 should be pending, got {pending_ids}"
    print("  ok: M19-C best @0.880; M20 pending")


def test_update_idempotent(tmp: Path) -> None:
    db = _fresh_db(tmp)

    def counts():
        con = C.connect(db)
        try:
            return (con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
                    con.execute("SELECT COUNT(*) FROM submission_stats").fetchone()[0],
                    con.execute("SELECT COUNT(*) FROM postprocess_stats").fetchone()[0])
        finally:
            con.close()

    before = counts()
    # Two identical updates of an existing experiment must not add rows.
    for _ in range(2):
        upd.update("M19_A_SAFE_DIVISIONS_PRUNE", public_score=0.877, db_path=db, regenerate=False)
    after = counts()
    assert before == after, f"update not idempotent: {before} -> {after}"

    # A score-only update must not wipe preserved metadata or duplicate stats.
    con = C.connect(db)
    try:
        row = con.execute("SELECT created_at, notes FROM experiments WHERE experiment_id='M19_A_SAFE_DIVISIONS_PRUNE'").fetchone()
        n_sub = con.execute("SELECT COUNT(*) FROM submission_stats WHERE experiment_id='M19_A_SAFE_DIVISIONS_PRUNE'").fetchone()[0]
    finally:
        con.close()
    assert row[0] is not None and row[1], "score-only update wiped preserved metadata"
    assert n_sub == 1, f"score-only update duplicated submission_stats: {n_sub}"
    print("  ok: updates idempotent, metadata preserved")


def test_update_pending_score(tmp: Path) -> None:
    db = _fresh_db(tmp)
    # Update M19-C with no score -> stays pending.
    r = upd.update("M19_C_FULL_CHAIN_PENDING", public_score=None, db_path=db, regenerate=False)
    assert r["public_score"] is None and r["status"] == "pending", f"expected pending, got {r}"
    # Then provide a score -> becomes scored and best flips to M19-C if higher.
    r2 = upd.update("M19_C_FULL_CHAIN_PENDING", public_score=0.881, db_path=db, regenerate=False)
    assert r2["status"] == "scored" and abs(r2["public_score"] - 0.881) < 1e-9
    con = C.connect(db)
    try:
        best = con.execute("SELECT experiment_id FROM experiments WHERE public_score IS NOT NULL ORDER BY public_score DESC, created_at LIMIT 1").fetchone()[0]
    finally:
        con.close()
    assert best == "M19_C_FULL_CHAIN_PENDING", f"best should flip to M19-C @0.881, got {best}"
    print("  ok: pending->scored transition and best update")


def test_query_experiment(tmp: Path) -> None:
    db = _fresh_db(tmp)
    con = q._con(db)
    try:
        rc = q.cmd_experiment(con, "M19_A_SAFE_DIVISIONS_PRUNE")
        rc_missing = q.cmd_experiment(con, "NOPE")
    finally:
        con.close()
    assert rc == 0 and rc_missing == 1, "experiment query return codes wrong"
    print("  ok: query_db experiment lookups")


def run_all() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_db_initializes_and_seeds(tmp)
        test_reports_generated(tmp)
        test_scored_and_best(tmp)
        test_update_idempotent(tmp)
        test_update_pending_score(tmp)
        test_query_experiment(tmp)
    print("All intelligence_db tests passed (6/6).")


if __name__ == "__main__":
    run_all()
