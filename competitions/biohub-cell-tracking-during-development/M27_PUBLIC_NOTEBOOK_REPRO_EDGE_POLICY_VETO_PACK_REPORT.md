# M27 — Public-Notebook Reproduction + Edge-Policy-Veto Pack

**Goal:** a public high-score notebook (`biohub-cell-tracking-v4-unet-ilp-reproduction`)
runs the **same 400ep artifact** we tested in M24 but scores much higher. M24
(0.872 / 0.873) underperformed on 400ep only because our post-processing was too
thin (safe_div / gap / linefit). M27 reproduces the notebook's **richer output
pipeline logic** — hidden-safe, never its static `submission.csv` — on the guarded
400ep artifact. M19-C **0.880** stays final unless an M27 variant beats it.

## The 400ep artifact (guarded)

- `artifact_name` contains **`400ep`** — `biohub-tracking-support-pack-400ep-snapshot-v1`
- `weight_sha256` = **`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`**
- public raw base: **127790 nodes / 115694 edges**; public final: **119763 node rows
  / 115080 edge rows / 234843 submission rows**.

The guard requires **both** the name substring **and** the exact weight sha256. If
the artifact is absent or drifts, the run reports `DO_NOT_SUBMIT_WRONG_ARTIFACT`
and still writes a valid hidden-safe fallback submission.

## The reproduced output pipeline (per dataset, evaluator-valid)

Every produced edge is unit-timepoint (t→t+1); `in_degree ≤ 1`, `out_degree ≤ 2`.

1. **motion_relink_edges** — replace the raw learned edges with a per-frame
   **Hungarian assignment** (scipy `linear_sum_assignment`, deterministic greedy
   fallback) on motion-predicted distance `pos + velocity_weight·(pos − predecessor)`
   minus `learned_bonus·edge_prob`; feasible only within `relaxed_um`. Yields a
   clean `in≤1 / out≤1` skeleton. (tight 6.0, relaxed 10.0, vel_w 0.5, bonus 0.75)
2. **close_single_frame_gaps** — bridge END@t and START@t+2 with a node at t+1;
   **reuse** an existing unmatched t+1 node within `reuse_um` (fewer synthetic
   nodes) else insert a synthetic midpoint. Capped `min(2000, 0.05·N)`. Never a
   direct t→t+2 edge. (gap 6.0, reuse 3.2)
3. **add_safe_divisions_postlink** — admit a geometrically safe SECOND child
   (out 1→2) under per-frame (0.008) / global (0.004) caps. (4.7 / 7.2 / 7.8)
4. **apply_edge_policy_veto** — an embedded **logistic edge-only policy** scores
   every edge; the weakest are vetoed as a **conservative** repair, capped at
   `max_veto_frac` (**1%**) of scored edges, **never** vetoing a division source.
   (threshold 0.35)
5. **filter_short_track_components** — drop connected components shorter than
   `min_track_len` **unless** they contain a division (`keep_division_components`).
6. **linefit_smooth** — topology-preserving coordinate smoothing. (w 0.8, win 2)

Then a **final safety repair** enforces the hard invariants: all edges t→t+1, no
dangling edge, `in≤1`, `out≤2` (a >2-out source keeps its two best children by
**edge_prob → repair_policy_score → shorter distance_um**), no NaN, consecutive id.

## Four variants (submit order A → C → B → D)

| Variant | Change from A | Submit rule (beyond the universal gate) |
|---------|---------------|------------------------------------------|
| **A** `exact_safety` | faithful reproduction + safety repair | `OK_TO_SUBMIT_EXPERIMENTAL` |
| **C** `minlen5` | `min_track_len = 5` | valid **and** `n_node_rows ≤ 126000` |
| **B** `no_edge_veto` | edge-policy veto **disabled** | valid **and** counts sane |
| **D** `strict_precision` | `min_track_len = 8`, keep divisions | `n_node_rows ≥ 112000` **and** `n_edge_rows ≥ 108000` |

**Expected public diagnostic (variant A):** n_node_rows ~119000–121000, n_edge_rows
~114000–116500, gap_inserted_synthetic ~2000–2300, safe_divisions_added ~350–430,
short_track_nodes_removed ~9000–11000, repair_policy_edges_vetoed ~900–1400,
linefit_smoothed_nodes ~118000–121000.

## Safety gate

`OK_TO_SUBMIT_EXPERIMENTAL` requires **all** of: `artifact_guard_passed`, `valid`,
`fallback_used=False`, `final_source=public_notebook_logic_reproduced`, no NaN,
consecutive id, no dangling edges, all edges t→t+1, `max_in_degree ≤ 1`,
`max_out_degree ≤ 2`, `110000 ≤ n_node_rows ≤ 130000`, `105000 ≤ n_edge_rows ≤ 122000`,
synthetic gap nodes `≤ 2600`, no dataset emptied by repair, all expected dynamic
test datasets present — plus the variant-specific rule above. Otherwise a specific
`DO_NOT_SUBMIT_*` (`WRONG_ARTIFACT` / `VALIDATION_FAILED` / `UNSANE_NODE_COUNT` /
`UNSANE_EDGE_COUNT` / `SYNTHETIC_EXPLOSION` / `EMPTY_OR_MISSING_DATASET` /
`MINLEN5_NODES_TOO_HIGH` / `STRICT_PRECISION_UNDERCUT`).

## Outputs (per runner, all under `/kaggle/working`)

`milestone27_reference_submission_report.json` (every required field: variant,
artifact_name, weight_sha256, artifact_guard_passed, raw/final nodes+edges, gap_*,
safe_division_*, repair_policy_edge_*, short_track_*, linefit_smoothed_nodes,
n_node_rows, n_edge_rows, max_in/out_degree, direct_multiframe_edges, dangling_edges,
id_consecutive, no_nan, valid, fallback_used, final_source, recommendation),
`milestone27_run_stats.csv`, `milestone27_artifact_selection.json`,
`milestone27_failure_fallback_report.json`, `submission.csv`.

## Verification

- `milestone27_public_notebook_repro_runner.py` compiles; `run_milestone27_tests()`
  passes **9/9** on synthetic fixtures: motion relink (in≤1/out≤1, unit edges),
  gap close + reuse, safe divisions, edge-policy veto (1% cap + division-source
  skip), short-track filtering (keep divisions), final safety repair
  (degree/unit-t/dangling/NaN), an end-to-end valid submission, the variant
  registry, the 400ep guard decomposition (needs name **and** sha), and the
  recommendation gates. All four variants dry-run cleanly off Kaggle.
- Static checks: exact reference predict command (det 0.99, ilp-edge −1.0,
  appearance/disappearance 0.1, division 1.0, `--use-ilp`); dynamic hidden-safe
  split discovery; **no static public submission.csv**; no Kaggle-API submit;
  output to `/kaggle/working`; each `.txt` ends with the correct
  `run_milestone27_variant("A"|"B"|"C"|"D")`; **no M16–M26 file modified**.
- intelligence_db: M19-C stays **final/best @0.880**; the public v4 notebook added
  as a source; `M27_A/B/C/D` added as **pending**; a `DEC_M27_PUBLIC_REPRO_PIVOT`
  decision recorded; `next_actions.md` now recommends the M27 public reproduction
  (submit order A→C→B→D). Tests **6/6**.

**Status:** M19-C **0.880** remains the recommended final submission. M27 A–D are
experimental — submit only on `OK_TO_SUBMIT_EXPERIMENTAL` (plus the per-variant
rule), keep 0.880 unless beaten.
