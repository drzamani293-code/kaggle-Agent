# M30 — Metric-Aware Global Relinker v2 Pack

**Goal:** M29/v1 *patched* the ILP solution after linking. M30/v2 **re-solves**
unit-timepoint linking from the **raw GEFF candidate graph** using edge_prob +
physical distance + motion consistency. Built in parallel with the pending M29-A
run; M19-C **0.880** stays best/final unless an M30 variant beats it.

## The fused cost

```
cost = -log(edge_prob) + w_dist * (distance_um / 7.0) + w_motion * motion_deviation_um
```
`motion_deviation_um` = distance between a target and the source's motion-predicted
next position (velocity from its best raw-candidate predecessor).

## Critical engineering corrections (all implemented)

1. **Raw candidate preservation.** The reused `read_geff_graph` **applies the
   edge `solution` mask**, so it returns only the ILP solution — not the candidate
   graph. v2's `read_candidate_geff` reads **all** unit-timepoint candidate edges
   **without** that mask, keeping `edge_prob`/`edge_dist`. `build_base_graph` is
   never run before v2. `edge_prob` is never defaulted to 1.0; if it is missing,
   constant or degenerate the run reports `DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE`.
2. **Explicit no-link option.** Each Hungarian stage has per-source no-link dummy
   columns (`LinkConfig.null_link_cost`, default **4.5**). A real edge is taken
   only if its fused cost beats no-link **and** passes `min_link_prob` /
   `max_link_um` / `max_cost`.
3. **Two-stage linking.** *Phase 1* primary: one slot/source, each target ≤ 1
   parent, Hungarian over real targets + no-link dummies. *Phase 2* division: only
   sources that got a primary child and still have out-degree 1, only currently
   unassigned targets, **gates applied before** an independent Hungarian with
   no-division dummies + a global cap — a rejected division **never steals**
   another source's target.
4. **Sparse-candidate augmentation.** If the raw graph is sparse, add tight
   geometric candidates for **every** source (top-k ≤ 3 within `knn_max_um`,
   conservative fixed `knn_default_prob`, **no duplicates**); raw vs augmented
   counts and `links_selected_from_raw/knn` are reported.
5. **Real v2 local CV.** A non-submit harness discovers train GT GEFF, runs the v2
   engine, and attempts the official metric (`tracking_cellmot.metrics` /
   `scripts/evaluate.py`). It **never fabricates** scores; if wiring is impossible
   it reports `CV_NOT_WIRED` with the exact missing function/import/GT format.

## Stage 0 — candidate diagnostic (run FIRST, non-submit)

`M30_CANDIDATE_GRAPH_DIAGNOSTIC_NOT_SUBMIT.txt` runs det=0.99 predict on the 400ep
artifact and profiles the raw candidate graph: `edge_prob` availability + dtype +
min + p01/p05/p25/p50/p75/p95/p99/max + unique count + fraction 0/1;
candidates-per-source mean/median/p75/p90/p95/max; candidates-per-target
mean/median/p90/max; % sources with 0 / 1 / ≥2 / ≥3 candidates; per-frame &
per-dataset stats. It classifies:

| Class | Condition |
|-------|-----------|
| **FULL_CANDIDATE_GRAPH** | edge_prob non-degenerate **and** candidates/source mean ≥ 1.25 **and** %≥2 candidates ≥ 0.15 |
| **ILP_SOLUTION_LIKE** | mean candidates/source ≤ 1.10 **and** %≥2 candidates < 0.05 |
| **MODERATELY_SPARSE** | anything in between (edge_prob non-degenerate) |
| **EDGE_PROB_UNAVAILABLE** | edge_prob missing / constant / degenerate |

Writes `m30_candidate_graph_report.json` + `m30_candidate_graph_table.csv`. No
submission.

## Five submit-capable variants (candidate-mode-gated)

| Variant | Candidate mode | det | Notable config |
|---------|----------------|-----|----------------|
| **A** v2_full_candidates_balanced | raw (**FULL only**) | 0.99 | null 4.5, max_link 7.5, div 0.20/6.5/9.0/0.020, gaps (1,2) |
| **B** v2_full_candidates_gap123 | raw (**FULL only**) | 0.99 | A + gap3 |
| **C** v2_auto_sparse_knn_tight | raw + kNN (**sparse/ILP-like**) | 0.99 | null 4.2, max_link 7.0, kNN k3/4.5µm/0.20, gaps (1,2) |
| **D** v2_auto_det095 | auto (raw if FULL, kNN if sparse) | **0.95** | node-explosion guard |
| **E** v2_division_relaxed | auto | 0.99 | div_min_prob 0.15, div_global 0.030; division-explosion guard |

Plus `M30_V2_LOCAL_CV_HARNESS_NOT_SUBMIT.txt` (non-submit).

## Submit order (depends on Stage 0)

- **FULL_CANDIDATE_GRAPH** → A → B → E → D.
- **ILP_SOLUTION_LIKE / MODERATELY_SPARSE** → C → D → E (skip A/B — their guard
  reports `DO_NOT_SUBMIT_WRONG_CANDIDATE_MODE`).
- **EDGE_PROB_UNAVAILABLE** → submit nothing.

Do **not** blindly submit all variants.

## Safety gate & recommendation vocabulary

`OK_TO_SUBMIT_EXPERIMENTAL` requires: artifact guard passes, `edge_prob` present +
non-degenerate, selected candidate mode matches the Stage-0 class, valid, no
fallback, no NaN, consecutive ids, no dangling edges, every edge exactly t→t+1,
`max_in_degree ≤ 1`, `max_out_degree ≤ 2`, all dynamic test datasets present +
non-empty, `110000 ≤ n_node_rows ≤ 165000`, `100000 ≤ n_edge_rows ≤ 155000`,
`synthetic_nodes_added ≤ 12000`, `divisions_total ≤ 4000`,
`final_source = metric_aware_global_relinker_v2`. Otherwise exactly one of:
`DO_NOT_SUBMIT_EDGE_PROB_UNAVAILABLE`, `DO_NOT_SUBMIT_CANDIDATE_GRAPH_TOO_SPARSE`,
`DO_NOT_SUBMIT_WRONG_CANDIDATE_MODE`, `DO_NOT_SUBMIT_VALIDATION_FAILED`,
`DO_NOT_SUBMIT_UNSANE_NODE_COUNT`, `DO_NOT_SUBMIT_UNSANE_EDGE_COUNT`,
`DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION`, `DO_NOT_SUBMIT_DIVISION_EXPLOSION`,
`DO_NOT_SUBMIT_NODE_EXPLOSION`, `DO_NOT_SUBMIT_MISSING_OR_EMPTY_DATASET`. A valid
hidden-safe fallback submission is still written on any failure.

## Outputs (per submit runner, `/kaggle/working`)

`milestone30_reference_submission_report.json` (all required fields incl.
candidate_graph_class, edge_prob_available/non_degenerate, raw/augmented candidate
edges, candidates_per_source_mean/p90, pct_sources_ge2, null_link_cost,
primary_links_selected/no_link, division_candidates/selected, divisions_total,
links_selected_from_raw/knn, mean/quantiles of selected prob/dist/fused,
gap1/2/3_closed, synthetic, pruned, degrees, invariants, valid, fallback_used,
`final_source`, recommendation), `milestone30_candidate_graph_report.json`,
`milestone30_v2_conversion.json`, `milestone30_artifact_selection.json`,
`milestone30_failure_fallback_report.json`, `submission.csv`. The diagnostic
writes `m30_candidate_graph_report.json` + `.csv`; the CV harness writes
`m30_v2_local_cv_report.json`.

## Verification

- `milestone30_metric_aware_global_relinker_v2_runner.py` compiles; every `.txt`
  runs standalone as `__main__`; `run_milestone30_tests()` passes **10/10**: raw
  candidate preservation with edge_prob, missing/constant edge_prob guard, no-link
  dummy beats a poor edge / good edge beats the dummy, primary in≤1/out≤1,
  independent division assignment with **no target stealing**, division
  sister/probability gates, kNN top-k-within-radius no-duplicate, unit-timepoint
  gaps, end-to-end valid submission, candidate classification + mode
  recommendation + explosion guards, and the auto candidate-mode resolution.
- Static: exact predict flags (only D uses det 0.95), dynamic hidden-safe split
  discovery, **no static CSV / no supplied CSV submitted**, **no runtime
  dependency on the uploaded files** (engine embedded), self-contained one-cell
  runners, `final_source` string exactly `metric_aware_global_relinker_v2`, **no
  M16–M29 file modified**.
- intelligence_db: M19-C stays **final/best @0.880**; the v2 package added as 5
  sources; `M30_CANDIDATE_DIAGNOSTIC` (non-scoring) + `M30_A–E` (pending) added;
  `DEC_M30_GLOBAL_RELINKER_V2` recorded; M29-A kept pending; `next_actions.md`
  recommends running the candidate diagnostic first then A/C by class. Tests
  **6/6**.

**Expected runtime:** dominated by the single reference GPU predict (same as
M19/M28/M29 — roughly minutes on the Kaggle T4/P100); the v2 relinking is
per-frame Hungarian assignment over the candidate graph (seconds to low minutes on
CPU). The diagnostic and each variant each run one predict.

**Status:** M19-C **0.880** remains the recommended final submission. Run the
Stage-0 diagnostic first; then submit A (FULL) or C (sparse) per its class, only
on `OK_TO_SUBMIT_EXPERIMENTAL`. No supplied CSV is ever submitted; the diagnostic
and CV harness are never submitted; hold det 0.90/0.80 until M30-D (0.95) lands.
