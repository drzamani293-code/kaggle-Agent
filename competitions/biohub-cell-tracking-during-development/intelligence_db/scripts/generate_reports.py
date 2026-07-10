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

    # Genuinely-open work vs superseded/unsafe/blocked attempts.
    pending = _fetch(con, """
        SELECT experiment_id, created_at FROM experiments
        WHERE public_score IS NULL AND status NOT IN ('unsafe', 'blocked', 'diagnostic', 'wrong_artifact') ORDER BY created_at
    """)
    unsafe = _fetch(con, """
        SELECT e.experiment_id, p.nodes_before, p.edges_before
        FROM experiments e LEFT JOIN postprocess_stats p USING (experiment_id)
        WHERE e.status = 'unsafe' ORDER BY e.created_at
    """)
    blocked = _fetch(con, "SELECT experiment_id FROM experiments WHERE status = 'blocked' ORDER BY created_at")
    blocked_m22 = [r[0] for r in blocked if r[0].startswith("M22_")]
    final_row = _fetch(con, "SELECT experiment_id, public_score FROM experiments WHERE status = 'final' ORDER BY public_score DESC LIMIT 1")

    out = ["# Next Actions", "",
           f"_Generated {_now()} from intelligence.duckdb._", "",
           f"**Best scored experiment:** `{best_id}` at **{_fmt_score(best_score)}**.",
           f"**M19-C full_chain:** {m19c_status} (score {_fmt_score(m19c_score)}).",
           f"**Open (pending):** {len(pending)} · **blocked:** {len(blocked)} · **unsafe/superseded:** {len(unsafe)}.", "",
           "## Recommendation", ""]

    pending_ids = [r[0] for r in pending]
    m21_order = [e for e in ["M21_A_PILKWANG350_M19C_GATES", "M21_C_PILKWANG350_LIGHT_GAP",
                             "M21_B_PILKWANG350_SAFE_DIV_ONLY"] if e in pending_ids]
    m22_order = [e for e in ["M22_A_TRUEBASE_SAFE_DIV_TUNE", "M22_B_TRUEBASE_LIGHT_GAP",
                             "M22_C_TRUEBASE_DIV_PLUS_GAP1_ONLY"] if e in blocked_m22]

    if final_row and not pending:
        fid, fscore = final_row[0][0], final_row[0][1]
        out += [
            f"**FINAL — competition entry closed on the experimental track.** `{fid}` at **{_fmt_score(fscore)}** is the",
            "final recommended submission.",
            "",
            "- **Final submission: M19-C — `Biohub5-notebook015ca8d31a` version 12, public score 0.880.**",
            "- **Stop experimental submissions** unless the exact pilkwang350 **350ep / dfb848** artifact OR the",
            "  **true M19-C artifact** is recovered - both are currently unavailable/unrecoverable.",
            "",
            "Why: across the full experimental sweep, no path beat M19-C 0.880 -",
            "M21-A 0.874, M23-B 0.877, M24-A 0.873, M24-B 0.872 - and M25 could not run because the required",
            "350ep artifact was replaced by the 400ep snapshot (guard failed, DO_NOT_SUBMIT_WRONG_ARTIFACT).",
            "All M22/M23/M24/M25 experimental variants are blocked/abandoned; nothing is pending.",
        ]
    elif blocked_m22 and not pending:
        out += [
            "**Primary next action: recover the TRUE M19-C artifact** (base `n_nodes_before=131797`,",
            "`n_edges_before=118992`). It is likely a **NOTEBOOK input** from the original M19-C run",
            "(e.g. a kernel named like *\"Biohub Cell Tracking: Learned Graph w G\"*), not one of the two",
            "dataset support packs (pilkwang350 -> 142193, tom99763 -> 161098; both fail the base check).",
            "",
            "1. Run **`M22_ARTIFACT_FORENSIC_DIAGNOSTIC`** on Kaggle with ALL candidate inputs attached",
            "   (datasets AND notebooks). It enumerates every artifact root, records artifact_name +",
            "   weight_sha256, and smoke-runs each non-bad pack until the base equals 131797 / 118992.",
            "2. If it prints **TRUE_M19C_ARTIFACT_FOUND**, run the true-base M22 variants in this order,",
            "   each gated on `OK_TO_SUBMIT_TRUEBASE_EXPERIMENT` (true-artifact + public-base-count guards):",
        ]
        for i, eid in enumerate(m22_order, 1):
            out.append(f"   {i}. `{eid}`")
        out += [
            "3. If it prints **TRUE_M19C_ARTIFACT_NOT_FOUND**, attach the original M19-C Notebook input and",
            "   re-run the forensic. Do NOT create submit-ready runs until the true base is confirmed.",
            "",
            "Do **NOT** spend submissions on pilkwang350 drift variants: M21-A already scored 0.874 (< 0.880).",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** until a true-base score beats it.",
        ]
    elif [e for e in ["M25_A_PRUNE_MILD_SAFE_DIV", "M25_B_PRUNE_STRONG_SAFE_DIV",
                      "M25_C_PRUNE_M23B_PLUS_MICRO_GAP1"] if e in pending_ids]:
        m25_order = [e for e in ["M25_A_PRUNE_MILD_SAFE_DIV", "M25_B_PRUNE_STRONG_SAFE_DIV",
                                 "M25_C_PRUNE_M23B_PLUS_MICRO_GAP1"] if e in pending_ids]
        out += [
            "The 400ep path **failed** (M24-A 0.873, M24-B 0.872, below M19-C 0.880). The strongest",
            "experimental path is **M23-B** (pilkwang350 node-prune + safe-divisions, **0.877**, only 0.003",
            "behind M19-C). Tune the prune / safe-division balance around it with **M25** (pinned pilkwang350",
            "+ 350ep name + weight_sha256 dfb848.., NOT 400ep). Submit one only if its report says",
            "`OK_TO_SUBMIT_EXPERIMENTAL` (guard passed, 132000<=n_node_rows<=142500, prune<=0.07,",
            "synthetic<=400, valid, no fallback).",
            "",
            "**Submit order recommendation:**",
        ]
        for i, eid in enumerate(m25_order, 1):
            reason = {1: " (milder prune - preserve more real nodes than M23-B; zero synthetic)",
                      2: " (stronger prune - more node-penalty reduction; zero synthetic)",
                      3: " (M23-B repair + a very-light gap1 - tiny edge gain, synthetic ~150-350)"}.get(i, "")
            out.append(f"{i}. `{eid}`{reason}")
        out += [
            "",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M25 score beats 0.880.",
        ]
    elif [e for e in ["M24_A_400EP_FULLCHAIN_M19C_GATES", "M24_B_400EP_GAP1_ONLY",
                      "M24_C_400EP_SAFE_DIV_ONLY"] if e in pending_ids]:
        m24_order = [e for e in ["M24_A_400EP_FULLCHAIN_M19C_GATES", "M24_B_400EP_GAP1_ONLY",
                                 "M24_C_400EP_SAFE_DIV_ONLY"] if e in pending_ids]
        out += [
            "M23-B scored **0.877** (node-penalty repair on pilkwang350 helped but did not beat 0.880), and",
            "M23-C's guard failure **revealed a cleaner 400ep artifact** (base 127790/115694 - much closer to",
            "M19-C's true 131797/118992 than pilkwang350's 142193/127563). Test the lean **400ep base** with",
            "the proven M19-C post-processing (**M24**, pinned to the 400ep path+name+weight_sha256 12f688..,",
            "NO node-prune). Submit one only if its report says `OK_TO_SUBMIT_EXPERIMENTAL` (artifact guard",
            "passed, base & final node count in 120000-135000, synthetic <= 2200, valid, no fallback).",
            "",
            "**Submit order recommendation:**",
        ]
        for i, eid in enumerate(m24_order, 1):
            reason = {1: " (clean 400ep base + full M19-C chain - only high-upside candidate)",
                      2: " (gap1 only - isolates whether gap2 is risky on the lean base)",
                      3: " (safe-divisions only - zero-synthetic control)"}.get(i, "")
            out.append(f"{i}. `{eid}`{reason}")
        out += [
            "",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M24 score beats 0.880.",
        ]
    elif [e for e in ["M23_B_PILKWANG350_NODE_PRUNE_SAFE_DIV_ONLY", "M23_C_PILKWANG350_EDGE_NODE_BALANCED",
                      "M23_A_PILKWANG350_NODE_PRUNE_LIGHT"] if e in pending_ids]:
        m23_order = [e for e in ["M23_B_PILKWANG350_NODE_PRUNE_SAFE_DIV_ONLY",
                                 "M23_C_PILKWANG350_EDGE_NODE_BALANCED",
                                 "M23_A_PILKWANG350_NODE_PRUNE_LIGHT"] if e in pending_ids]
        out += [
            "M22 forensic returned **TRUE_M19C_ARTIFACT_NOT_FOUND** (the true 131797/118992 base is",
            "unrecoverable). So attack the pilkwang350 base's **node over-prediction** directly with the",
            "**M23 node-penalty repair** pack (pinned to pilkwang350 + weight_sha256 guard). Each variant",
            "prunes low-value detections (preserving divisions + long tracks) then applies conservative",
            "post-processing; submit one only if its report says `OK_TO_SUBMIT_EXPERIMENTAL` (artifact",
            "guard passed, not over-pruned >8%, post-repair node count in 130000-145000, valid, no fallback).",
            "",
            "**Submit order recommendation:**",
        ]
        for i, eid in enumerate(m23_order, 1):
            reason = {1: " (node-penalty reduction, ZERO synthetic nodes - lowest risk)",
                      2: " (balanced node+edge repair, gap1 only, no gap2)",
                      3: " (light repair + full_chain, only if B/C fall short)"}.get(i, "")
            out.append(f"{i}. `{eid}`{reason}")
        out += [
            "",
            "Why this order: M21-A's synthetic-heavy full_chain on pilkwang350 scored only 0.874, so test",
            "node-penalty reduction WITHOUT synthetic-node risk first.",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M23 score beats 0.880.",
        ]
    elif [e for e in ["M29_A_WINNING_ENGINE_DEFAULT", "M29_B_ENGINE_GAP12_ONLY", "M29_C_ENGINE_LINEFIT_ON",
                      "M29_D_ENGINE_RELINK_ON", "M29_E_ENGINE_DET095_SAFE"] if e in pending_ids]:
        m29_order = [e for e in ["M29_A_WINNING_ENGINE_DEFAULT", "M29_B_ENGINE_GAP12_ONLY", "M29_C_ENGINE_LINEFIT_ON",
                                 "M29_D_ENGINE_RELINK_ON", "M29_E_ENGINE_DET095_SAFE"] if e in pending_ids]
        out += [
            "**Integrate the full uploaded WINNING ENGINE (M29).** M28 was only a partial recall expansion",
            "(M28-A: nodes=132166, gap1=1604, gap2=1386, synthetic=4376). The uploaded engine",
            "(winning_postprocess.py) is a fuller recall recovery - motion relink + slot-Hungarian division",
            "rescue + per-timepoint Hungarian gap 1/2/3 stitching (constant velocity) + optional linefit +",
            "prune - profiling (submission(3).csv) to nodes=139418, edges=132871, divisions~2382, all invariants",
            "clean. It is EMBEDDED self-contained so each M29 runner is a standalone Kaggle cell (no external",
            "utility-dataset import). The uploaded submission(3).csv is never submitted.",
            "",
            "Submit each only if its report prints `OK_TO_SUBMIT_EXPERIMENTAL` (valid, no fallback, no NaN,",
            "consecutive id, no dangling, all edges t->t+1, in<=1/out<=2, 125000<=node_rows<=165000,",
            "110000<=edge_rows<=155000, synthetic<=12000 [E 18000], divisions_total<=4000,",
            "final_source=winning_postprocess_engine_reproduced).",
            "",
            "**Submit order recommendation:**",
        ]
        for i, eid in enumerate(m29_order, 1):
            reason = {1: " (full engine default - PRIMARY; soft-WARN vs the 139418/132871/2382 profile)",
                      2: " (gaps 1,2 only - isolate whether gap3 over-adds synthetic/FP)",
                      3: " (default + linefit window 2 weight 0.75 - compare with M19-C)",
                      4: " (default + motion relink 7.0/4.5 - RISKY; reports n_relinked/fails)",
                      5: " (det 0.95 - cautious detection sweep; DO_NOT_SUBMIT_NODE_EXPLOSION guard)"}.get(i, "")
            out.append(f"{i}. `{eid}`{reason}")
        out += [
            "",
            "**Decision rules:** if M29-A > 0.880, make it the new candidate and continue B/C; if A in",
            "[0.875, 0.880) test B and C; if A < 0.875 test B only, then **PAUSE and run the non-submit local-CV",
            "harness** (`M29_LOCAL_CV_HARNESS_INTEGRATION_NOT_SUBMIT`) before more submits.",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M29 variant beats 0.880.",
            "Do NOT build det 0.90/0.80 until M29-E (0.95) lands. M28/M26 variants remain secondary pending tracks.",
        ]
    elif [e for e in ["M28_A_M19C_GAP_CAPS_OPEN", "M28_C_DIVISION_CAP_OPEN",
                      "M28_D_GAP_OPEN_PLUS_DIV_OPEN_NO_LINEFIT", "M28_B_M19C_GAP3_ADDED",
                      "M28_E_DET095_M19C_SAFE"] if e in pending_ids]:
        m28_order = [e for e in ["M28_A_M19C_GAP_CAPS_OPEN", "M28_C_DIVISION_CAP_OPEN",
                                 "M28_D_GAP_OPEN_PLUS_DIV_OPEN_NO_LINEFIT", "M28_B_M19C_GAP3_ADDED",
                                 "M28_E_DET095_M19C_SAFE"] if e in pending_ids]
        out += [
            "**Pivot back to M19-C and EXPAND RECALL (M28).** M27-A (public-notebook reproduction) scored",
            "**0.859 < 0.880** - its pruning-heavy output pipeline over-prunes / lowers node recall on our",
            "base, so the pruning path is deprioritized. The uploaded winning strategy argues the opposite",
            "for M19-C: our post-processing is too **conservative** (gap caps too small, division cap too",
            "small, det-threshold 0.99 possibly too high), and recall-oriented gap/division recovery is the",
            "path beyond 0.880.",
            "",
            "M28 keeps the exact M19-C predict command + metric-aware chain and OPENS one recall lever per",
            "variant (every recovered edge stays unit-timepoint t->t+1; in<=1/out<=2). Submit each only if its",
            "report prints `OK_TO_SUBMIT_EXPERIMENTAL` (valid, no fallback, no NaN, consecutive id, no dangling,",
            "all edges t->t+1, in<=1/out<=2, node/edge counts sane, synthetic within the per-variant cap,",
            "final_source=reference_learned_graph_postprocessed) plus the per-variant explosion guard.",
            "",
            "**Submit order recommendation:**",
        ]
        for i, eid in enumerate(m28_order, 1):
            reason = {1: " (gap caps open - the strategy's strongest claim; node_rows 132000-140000, synthetic<=4500)",
                      2: " (division caps open only - isolate the division lever; DO_NOT_SUBMIT_DIVISION_EXPLOSION if divisions>2500)",
                      3: " (gaps + divisions open, linefit disabled - aggressive recall without smoothing; synthetic<=5000)",
                      4: " (adds strict-velocity gap3 = 3-frame recovery; higher upside, synthetic<=6500)",
                      5: " (det-threshold 0.95, a SAFE first det-sweep step; DO_NOT_SUBMIT_NODE_EXPLOSION if node_rows>155000)"}.get(i, "")
            out.append(f"{i}. `{eid}`{reason}")
        out += [
            "",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M28 variant beats 0.880.",
            "Do NOT build det 0.90/0.80 until the M28-E 0.95 result lands. The M26 ensemble variants remain a",
            "secondary pending track.",
        ]
    elif [e for e in ["M27_A_PUBLIC_REPRO_EXACT_SAFETY", "M27_C_PUBLIC_REPRO_MINLEN5",
                      "M27_B_PUBLIC_REPRO_NO_EDGE_VETO", "M27_D_PUBLIC_REPRO_STRICT_PRECISION"] if e in pending_ids]:
        m27_order = [e for e in ["M27_A_PUBLIC_REPRO_EXACT_SAFETY", "M27_C_PUBLIC_REPRO_MINLEN5",
                                 "M27_B_PUBLIC_REPRO_NO_EDGE_VETO", "M27_D_PUBLIC_REPRO_STRICT_PRECISION"] if e in pending_ids]
        out += [
            "**Reproduce the public high-score notebook (M27).** A public notebook",
            "(`biohub-cell-tracking-v4-unet-ilp-reproduction`) runs the SAME 400ep artifact we tested in M24",
            "(weight_sha256 `12f6881e..`, base 127790/115694) but scores much higher by replacing our thin",
            "safe_div/gap/linefit post-processing with a richer, evaluator-safe OUTPUT pipeline: motion-relink",
            "(per-frame Hungarian) -> gap close (reuse-or-insert) -> safe divisions -> a conservative logistic",
            "edge VETO (capped 1%, skip divisions) -> short-track-component filter (keep divisions) -> linefit.",
            "M24 (0.872/0.873) underperformed on 400ep only because our postprocess was too thin.",
            "",
            "M27 reproduces that LOGIC hidden-safe (never the static submission.csv), guards the 400ep artifact",
            "(name contains `400ep` AND exact weight_sha256) and adds a final safety repair. Submit each only if",
            "its report prints `OK_TO_SUBMIT_EXPERIMENTAL` (artifact guard, valid, no fallback,",
            "final_source=public_notebook_logic_reproduced, in<=1/out<=2, all edges t->t+1, no dangling, no NaN,",
            "consecutive id, 110000<=node_rows<=130000, 105000<=edge_rows<=122000, synthetic<=2600) plus the",
            "variant-specific count rule.",
            "",
            "**Submit order recommendation:**",
        ]
        for i, eid in enumerate(m27_order, 1):
            reason = {1: " (faithful public reproduction + safety repair - highest-fidelity shot)",
                      2: " (min_track_len 5 - does minlen 7 over-prune true short tracks? submit if node_rows<=126000)",
                      3: " (edge-policy veto disabled - isolate whether the veto helps or hurts)",
                      4: " (strict min_track_len 8, keep divisions - even lower node penalty; node_rows>=112000 & edge_rows>=108000)"}.get(i, "")
            out.append(f"{i}. `{eid}`{reason}")
        out += [
            "",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M27 variant beats 0.880.",
            "The M26 multi-artifact ensemble variants remain pending as a secondary track.",
        ]
    elif [e for e in ["M26_A_CONSENSUS_2OFN_PRECISION", "M26_B_PRIMARY_M19C_STYLE_PLUS_CONSENSUS_EDGES",
                      "M26_C_WEIGHTED_ENSEMBLE_FULLCHAIN_LIGHT"] if e in pending_ids]:
        m26_order = [e for e in ["M26_A_CONSENSUS_2OFN_PRECISION",
                                 "M26_B_PRIMARY_M19C_STYLE_PLUS_CONSENSUS_EDGES",
                                 "M26_C_WEIGHTED_ENSEMBLE_FULLCHAIN_LIGHT"] if e in pending_ids]
        out += [
            "**Pivot to multi-artifact ENSEMBLING (M26).** Single-model post-processing has plateaued at",
            f"**M19-C 0.880** - every substitute base (pilkwang350 142193, 400ep 127790, tom99763 161098)",
            "scored below it and the true M19-C artifact is unrecoverable. The remaining source of signal is",
            "cross-model **consensus**: run several valid artifacts on the same hidden-safe split and combine",
            "their tracking graphs.",
            "",
            "1. **Run Stage 1 first: `M26_ARTIFACT_ZOO_DIAGNOSTIC`** (non-scoring). It enumerates every mounted",
            "   artifact, records artifact_name + weight_sha256 + known-artifact match, and smoke-runs each to",
            "   report its base GEFF counts. **Only proceed to a submission if it finds >=2 viable artifacts.**",
            "2. If >=2 artifacts, run the ensemble variants in submit order, each only if its report prints",
            "   `OK_TO_SUBMIT_EXPERIMENTAL` (>=2 artifacts, valid, no fallback, ensemble_graph_postprocessed,",
            "   120000<=nodes<=145000, 110000<=edges<=135000, synthetic within the per-variant cap):",
        ]
        for i, eid in enumerate(m26_order, 1):
            reason = {1: " (2-of-N consensus, ZERO synthetic - tests whether consensus fixes node over-prediction)",
                      2: " (primary = base closest to 131797/118992 + consensus edges; micro gap1; synthetic<=600)",
                      3: " (weighted ensemble + full chain with a very light gap2; synthetic<=1000)"}.get(i, "")
            out.append(f"   {i}. `{eid}`{reason}")
        out += [
            "",
            "Each variant still writes a valid hidden-safe `submission.csv`; if <2 artifacts are found it reports",
            "`DO_NOT_SUBMIT_NOT_ENOUGH_ARTIFACTS` and must not be submitted.",
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless an M26 variant beats 0.880.",
        ]
    elif pending:
        out += [
            "With up to **5 daily submissions** available, run the controlled **M21** variants on the",
            "**pinned pilkwang** artifact (`pilkwang/biohub-tracking-support-pack-50ep-v1`). Each is",
            "artifact-guarded and experimental; submit one only if its report says",
            "`OK_TO_SUBMIT_EXPERIMENTAL` (artifact guard passed, valid, no fallback).",
            "",
        ]
        if m21_order:
            out += ["**Submit order recommendation:**"]
            for i, eid in enumerate(m21_order, 1):
                out.append(f"{i}. `{eid}`")
            out.append("")
        other_pending = [e for e in pending_ids if e not in m21_order]
        if other_pending:
            out.append("Other pending: " + ", ".join(f"`{e}`" for e in other_pending) + ".")
            out.append("")
        out += [
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as final** unless a new public score **beats 0.880**.",
            "Every M20 attempt is unsafe (baseline mismatch); M20 must not be submitted.",
        ]
    elif unsafe and not pending and best_id == PENDING_EXP_ID:
        # Every downstream tuning attempt failed the baseline guard: hold M19-C.
        out += [
            f"**Keep `{best_id}` at {_fmt_score(best_score)} as the best/final candidate.**",
            "",
            "Every M20 tuned-full_chain attempt **failed the M19-C baseline guard**: an identical predict",
            "command produced a DIFFERENT pre-post-processing base graph because the mounted support-pack /",
            "weights artifact differs from the one that produced M19-C. Gate tuning on a different base is",
            "uninterpretable and must not be submitted.",
            "",
            "Base `n_nodes_before` / `n_edges_before` vs the required **131797 / 118992**:",
        ]
        for eid, nb, eb in unsafe:
            out.append(f"- `{eid}`: {nb} / {eb}  → mismatch, DO_NOT_SUBMIT")
        out += [
            "",
            "- **Do NOT submit M20** (either support pack).",
            "- **Recover the TRUE M19-C baseline artifact** — the exact support pack / weights that yield",
            "  `n_nodes_before=131797` and `n_edges_before=118992` — then re-run the guarded M20 and submit",
            "  only if `baseline_guard_passed=True`.",
            "- If that artifact cannot be recovered, **M19-C `0.880` is final.**",
        ]
    elif m19c_score is None:
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
