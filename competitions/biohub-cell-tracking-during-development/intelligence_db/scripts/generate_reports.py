#!/usr/bin/env python3
"""Regenerate all Markdown reports from the intelligence DuckDB database.

Reports:
  - score_timeline.md    chronological scores + deltas
  - experiment_matrix.md full variant/command/postprocess/stats/score table
  - next_actions.md      data-driven next recommendation (branches on M19-C)
  - lessons_learned.md   curated lessons with evidence

Pure function of the DB: safe to run any time.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PENDING_EXP_ID = "M19_C_FULL_CHAIN_PENDING"
BASELINE_SCORE = 0.874
M19A_SCORE = 0.877


def _fetch(con, sql: str, params=None):
    return con.execute(sql, params or []).fetchall()


def _fmt_score(score) -> str:
    return f"{score:.4f}" if score is not None else "pending"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _experiments_ordered(con):
    return _fetch(con, """
        SELECT e.experiment_id, e.milestone, e.variant, e.label, e.det_threshold,
               e.postprocess_ops_json, e.public_score, e.status, e.created_at, e.notes,
               s.n_rows, s.n_nodes, s.n_edges, s.max_in_degree, s.max_out_degree,
               s.fallback_used, s.valid, s.final_source,
               p.safe_divisions_added, p.gap1_closed, p.gap2_recovered,
               p.synthetic_nodes_added, p.isolated_nodes_pruned, p.edges_removed,
               p.nodes_before, p.nodes_after, p.edges_before, p.edges_after
        FROM experiments e
        LEFT JOIN submission_stats s USING (experiment_id)
        LEFT JOIN postprocess_stats p USING (experiment_id)
        ORDER BY e.created_at, e.experiment_id
    """)


def _cols(row, names):
    return dict(zip(names, row))


_EXP_NAMES = [
    "experiment_id", "milestone", "variant", "label", "det_threshold",
    "postprocess_ops_json", "public_score", "status", "created_at", "notes",
    "n_rows", "n_nodes", "n_edges", "max_in_degree", "max_out_degree",
    "fallback_used", "valid", "final_source",
    "safe_divisions_added", "gap1_closed", "gap2_recovered",
    "synthetic_nodes_added", "isolated_nodes_pruned", "edges_removed",
    "nodes_before", "nodes_after", "edges_before", "edges_after",
]


def report_score_timeline(con) -> str:
    rows = [_cols(r, _EXP_NAMES) for r in _experiments_ordered(con)]
    best = max((r["public_score"] for r in rows if r["public_score"] is not None), default=None)

    out = ["# Score Timeline", "",
           f"_Generated {_now()} from intelligence.duckdb._", "",
           f"Baseline **{BASELINE_SCORE:.3f}** · best so far **{_fmt_score(best)}**.", "",
           "Delta = step change vs the previous **scored** experiment (chronological).", "",
           "| # | Date | Experiment | Score | Δ vs prev | Δ vs baseline | Status |",
           "|---|------|------------|-------|-----------|---------------|--------|"]
    prev_scored = None
    for i, r in enumerate(rows, 1):
        score = r["public_score"]
        if score is None:
            d_prev = d_base = "—"
        else:
            d_prev = f"{score - prev_scored:+.3f}" if prev_scored is not None else "—"
            d_base = f"{score - BASELINE_SCORE:+.3f}"
        out.append(f"| {i} | {r['created_at']} | `{r['experiment_id']}` | "
                   f"{_fmt_score(score)} | {d_prev} | {d_base} | {r['status']} |")
        if score is not None:
            prev_scored = score

    out += ["", "## Notes per experiment", ""]
    for r in rows:
        out.append(f"- **{r['experiment_id']}** ({_fmt_score(r['public_score'])}, {r['status']}): {r['notes']}")
    out.append("")
    return "\n".join(out)


def report_experiment_matrix(con) -> str:
    rows = [_cols(r, _EXP_NAMES) for r in _experiments_ordered(con)]
    out = ["# Experiment Matrix", "",
           f"_Generated {_now()} from intelligence.duckdb._", "",
           "Command differences, post-processing ops, submission stats, and score for every variant.",
           "All variants share ILP weights (edge -1.0, appearance 0.1, disappearance 0.1, division 1.0, --use-ilp);",
           "only `det_threshold` and the post-processing chain differ.", "",
           "| Experiment | det | Postprocess ops | rows | nodes | edges | div+ | gap1 | gap2 | synth | prune | edges_rm | in/out | valid | fb | score |",
           "|------------|-----|-----------------|------|-------|-------|------|------|------|-------|-------|----------|--------|-------|----|-------|"]
    for r in rows:
        ops = ", ".join(json.loads(r["postprocess_ops_json"] or "[]")) or "—"
        io = f"{r['max_in_degree']}/{r['max_out_degree']}" if r["max_in_degree"] is not None else "—"
        out.append(
            f"| `{r['experiment_id']}` | {r['det_threshold']} | {ops} | "
            f"{r['n_rows']} | {r['n_nodes']} | {r['n_edges']} | "
            f"{r['safe_divisions_added']} | {r['gap1_closed']} | {r['gap2_recovered']} | "
            f"{r['synthetic_nodes_added']} | {r['isolated_nodes_pruned']} | {r['edges_removed']} | "
            f"{io} | {r['valid']} | {r['fallback_used']} | {_fmt_score(r['public_score'])} |")
    out += ["", "Legend: div+ = safe divisions added, gap1 = single-frame gaps closed, ",
            "gap2 = two-frame gaps recovered, synth = synthetic nodes added, prune = isolated nodes pruned, ",
            "edges_rm = edges removed, in/out = max in/out degree, fb = fallback_used.", ""]
    return "\n".join(out)


def report_next_actions(con) -> str:
    rows = {r[0]: r for r in _fetch(con, "SELECT experiment_id, public_score, status FROM experiments")}
    pend = rows.get(PENDING_EXP_ID)
    m19c_score = pend[1] if pend else None
    m19c_status = pend[2] if pend else "absent"

    best = _fetch(con, """
        SELECT experiment_id, public_score FROM experiments
        WHERE public_score IS NOT NULL ORDER BY public_score DESC, created_at ASC LIMIT 1
    """)
    best_id, best_score = (best[0][0], best[0][1]) if best else (None, None)

    out = ["# Next Actions", "",
           f"_Generated {_now()} from intelligence.duckdb._", "",
           f"**Best scored experiment:** `{best_id}` at **{_fmt_score(best_score)}**.",
           f"**M19-C full_chain:** {m19c_status} (score {_fmt_score(m19c_score)}).", "",
           "## Recommendation", ""]

    if m19c_score is None:
        out += [
            "M19-C is **pending**. Do not spend the second remaining submission until its public score is known.",
            "",
            "- **Hold** the second submission.",
            "- When M19-C scores, this report auto-updates with the branch below.",
            "- Interim best to keep as the submitted baseline: "
            f"`{best_id}` ({_fmt_score(best_score)}).",
            "",
            "### Contingency (what each M19-C outcome will trigger)",
            "- **M19-C > 0.877** → tune `full_chain` gates: widen safe-division sister/parent gates slightly, ",
            "  and tune gap1/gap2 distance + velocity gates to add more valid synthetic chains without ",
            "  inflating the node penalty.",
            "- **M19-C == 0.877** → the gap recovery neither helped nor hurt net; isolate it by running ",
            "  **M19-B** (gap recovery alone, no divisions) or tune the safe-division gates further, since ",
            "  divisions are the proven lever.",
            "- **M19-C < 0.877** → the 1309 synthetic nodes cost more (node over-prediction penalty) than the ",
            "  recovered edges gained; **revert to M19-A** and tighten or remove gap recovery (drop gap2 first, ",
            "  then gap1), keeping only safe divisions + isolated prune.",
        ]
    elif m19c_score > M19A_SCORE:
        out += [
            f"M19-C **improved** to {_fmt_score(m19c_score)} (> {M19A_SCORE:.3f}). Gap recovery + smoothing add value on top of divisions.",
            "",
            "- **Tune the `full_chain` gates** (risk: medium):",
            "  - safe divisions: slightly widen sister/parent-child gates to admit more true divisions;",
            "  - gap1/gap2: sweep the distance and velocity gates to add more metric-valid synthetic chains;",
            "  - keep an eye on `synthetic_nodes_added` vs score to stay ahead of the node penalty.",
            "- Next variant: a tuned full_chain (e.g. `M20_A_FULLCHAIN_TUNED`).",
        ]
    elif abs(m19c_score - M19A_SCORE) < 1e-9:
        out += [
            f"M19-C **tied** M19-A at {_fmt_score(m19c_score)}. Gap recovery is net-neutral; divisions carry the gain.",
            "",
            "- **Isolate gap recovery**: run **M19-B** (gap recovery only, no divisions) to measure its standalone effect,",
            "  **or** tune the safe-division gates further (the proven lever).",
            "- Prefer the division-gate tuning first (lower risk than adding synthetic nodes).",
            "- Next variant: `M19_B_GAP_RECOVERY_PRUNE` or a division-gate sweep.",
        ]
    else:
        out += [
            f"M19-C **regressed** to {_fmt_score(m19c_score)} (< {M19A_SCORE:.3f}). The synthetic nodes cost more than they gained.",
            "",
            "- **Revert to M19-A** as the standing best submission.",
            "- **Tighten or remove gap recovery**: drop gap2 first (largest synthetic-node source), then gap1 if needed.",
            "- Keep only safe divisions + isolated prune; re-tune division gates for further division gains.",
            "- Next variant: `M19_A`-based with gap2 removed (e.g. `M20_A_DIVISIONS_GAP1_ONLY`).",
        ]

    # Append the standing recorded decisions for context.
    decisions = _fetch(con, """
        SELECT after_experiment_id, observation, recommendation, next_variant, risk_level, created_at
        FROM decisions ORDER BY created_at
    """)
    if decisions:
        out += ["", "## Recorded decisions (history)", ""]
        for d in decisions:
            out += [f"- **after `{d[0]}`** ({d[5]}, risk {d[4]}):",
                    f"  - observation: {d[1]}",
                    f"  - recommendation: {d[2]}  → next: `{d[3]}`"]
    out.append("")
    return "\n".join(out)


def report_lessons_learned(con) -> str:
    lessons = _fetch(con, "SELECT lesson_id, topic, lesson, evidence_experiment_ids, confidence FROM lessons ORDER BY lesson_id")
    out = ["# Lessons Learned", "",
           f"_Generated {_now()} from intelligence.duckdb._", "",
           "Distilled, evidence-linked findings driving strategy.", ""]
    for lid, topic, lesson, evidence, conf in lessons:
        out += [f"## {topic} — `{lid}`",
                f"- **Lesson:** {lesson}",
                f"- **Evidence:** {evidence}",
                f"- **Confidence:** {conf}", ""]
    # Public-source-derived context.
    srcs = _fetch(con, "SELECT title, url, relevance_score FROM public_sources ORDER BY relevance_score DESC")
    if srcs:
        out += ["## Public sources consulted (by relevance)", ""]
        for title, url, rel in srcs:
            out.append(f"- [{rel:.2f}] {title} — {url}")
        out.append("")
    return "\n".join(out)


REPORTS = {
    "score_timeline.md": report_score_timeline,
    "experiment_matrix.md": report_experiment_matrix,
    "next_actions.md": report_next_actions,
    "lessons_learned.md": report_lessons_learned,
}


def generate_all(db_path: Path = C.DB_PATH) -> list[str]:
    con = C.connect(db_path)
    written = []
    try:
        C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, fn in REPORTS.items():
            (C.REPORTS_DIR / name).write_text(fn(con))
            written.append(name)
    finally:
        con.close()
    return written


def main() -> None:
    written = generate_all()
    print("Regenerated reports:", ", ".join(written))


if __name__ == "__main__":
    main()
