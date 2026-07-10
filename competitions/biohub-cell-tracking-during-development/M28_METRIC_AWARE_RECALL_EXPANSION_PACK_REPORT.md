# M28 — Metric-Aware Recall Expansion Pack

**Goal:** M27's pruning-heavy public-notebook reproduction scored **0.859 < M19-C
0.880** (over-pruning / low node recall), so we **pivot back to the proven M19-C
full_chain** and push the *opposite* direction. The uploaded winning strategy
argues M19-C is too **conservative** — gap caps too small, division cap too small,
det-threshold 0.99 possibly too high — and that **recall-oriented gap/division
recovery** is the path beyond 0.880. This pack keeps the exact M19-C predict
command + metric-aware chain and opens **one recall lever per variant**. M19-C
0.880 stays best/final unless an M28 variant beats it.

## Metric safety (unchanged from M19-C)

Every recovered edge spans a **unit timepoint** (t→t+1); a direct multi-frame edge
is never created — gap2 inserts 2 synthetic nodes, the new **gap3** inserts 3.
Degrees stay `in ≤ 1 / out ≤ 2`. M28 runs whatever reference support pack is
mounted (like M19-C), reporting its `artifact_name` + `weight_sha256`; it does
**not** pin a specific artifact.

## Five variants (submit order A → C → D → B → E)

| Variant | Lever opened | Ops | det | Synthetic cap | Node window |
|---------|--------------|-----|-----|---------------|-------------|
| **A** gap_caps_open | gap1 6.5/0.05/100000 · gap2 4.6/10.5/0.05/100000 | div, gap1, gap2, linefit, prune | 0.99 | ≤ 4500 | 132000–140000 |
| **C** division_cap_open | safe-div 6.5 / 9.0 / frame 0.030 / global 0.020; **gaps = M19-C** | div, gap1, gap2, linefit, prune | 0.99 | ≤ 3000 | 125000–155000 |
| **D** gap+div open, no linefit | A gaps + C divisions, **linefit OFF** | div, gap1, gap2, prune | 0.99 | ≤ 5000 | 125000–142500 |
| **B** gap3_added | A + **gap3** (END@t→START@t+4, 3 synth; 4.6/14.5/cos≥0.0/normdiff 5.5/0.03) | div, gap1, gap2, gap3, linefit, prune | 0.99 | ≤ 6500 | 125000–142500 |
| **E** det095_safe | M19-C postprocess, **det 0.99→0.95** | div, gap1, gap2, linefit, prune | **0.95** | ≤ 3000 | 125000–155000 |

**Rationale:** A is the cleanest test of the strategy's strongest claim (gap caps
too low). C isolates the division lever. D combines both and drops the
potentially harmful linefit. B tests high-upside gap3 (more synthetic risk). E
starts a detection sweep **safely at 0.95** — 0.90/0.80 only after 0.95 lands.

**gap3** bridges an END@t and a START@t+4 with three synthetic nodes at t+1/t+2/t+3
and four unit-timepoint edges, gated on per-step (≤ 4.6 µm) and total (≤ 14.5 µm)
distance **and** velocity consistency (cosine ≥ 0.0, per-step magnitude within 5.5
µm) so it never bridges a reversing track.

## Expected diagnostics (relative to M19-C's 0.880 base ≈ 131797/118992 → 133106/121718)

- **A**: more gap1/gap2 closures → node_rows ~133000–140000, synthetic up to ~4500 (vs M19-C 1309).
- **C**: safe_divisions_added rises (bounded ≤ 2500 or `DO_NOT_SUBMIT_DIVISION_EXPLOSION`); node count ≈ M19-C.
- **D**: A's node growth, divisions from C, no coordinate smoothing.
- **B**: A + a modest number of gap3 recoveries (reported separately), synthetic up to ~6500, node_rows ≤ 142500.
- **E**: det 0.95 raises raw detections → node_rows up to ~155000, edge_rows up to ~145000 (else `DO_NOT_SUBMIT_NODE_EXPLOSION`).

## Safety gate

`OK_TO_SUBMIT_EXPERIMENTAL` requires **all** of: `valid`, `fallback_used=False`,
no NaN, consecutive id, no dangling edges, all edges t→t+1, `max_in_degree ≤ 1`,
`max_out_degree ≤ 2`, `125000 ≤ n_node_rows ≤ 155000` (**A tightened to
132000–140000**), `110000 ≤ n_edge_rows ≤ 145000`, `synthetic_nodes_added ≤` the
per-variant cap, all expected dynamic test datasets present,
`final_source=reference_learned_graph_postprocessed`. Otherwise a specific
`DO_NOT_SUBMIT_*` is reported: `VALIDATION_FAILED` / `UNSANE_NODE_COUNT` /
`UNSANE_EDGE_COUNT` / `SYNTHETIC_EXPLOSION` / **`DIVISION_EXPLOSION`** (C, >2500
divisions) / **`NODE_EXPLOSION`** (E, node_rows > 155000) /
`MISSING_OR_EMPTY_DATASET`. A valid hidden-safe fallback submission is still
written in every failure path.

## Outputs (per runner, `/kaggle/working`)

`milestone28_reference_submission_report.json` (all required fields: variant,
artifact_name, weight_sha256, n_nodes/edges_before/after, safe_divisions_added,
gap1/gap2/gap3 candidates + closed/recovered, synthetic_nodes_added,
divisions_added_total, isolated_nodes_pruned, linefit_enabled, det_threshold,
n_node_rows, n_edge_rows, max_in/out_degree, direct_multiframe_edges,
dangling_edges, id_consecutive, no_nan, valid, fallback_used, final_source,
recommendation), `milestone28_geff_conversion.json`,
`milestone28_artifact_selection.json`, `milestone28_failure_fallback_report.json`,
`submission.csv`.

## Verification

- `milestone28_metric_aware_recall_expansion_runner.py` compiles;
  `run_milestone28_tests()` passes **7/7** on synthetic fixtures: gap3 recovery (3
  synthetic nodes, 4 unit edges, at t+1/t+2/t+3), gap3 velocity-reversal
  rejection, the full chain with gap3 keeping every edge unit-timepoint, the gate
  presets + variant registry, the per-variant predict command det-threshold (A–D
  0.99, E 0.95), the recommendation/explosion guards, and an end-to-end valid
  submission. All five variants dry-run cleanly off Kaggle.
- Static checks: exact M19-C predict flags (ilp-edge −1.0, appearance/disappearance
  0.1, division 1.0, `--use-ilp`), only det-threshold varies (E 0.95); dynamic
  hidden-safe split discovery; **no static submission.csv**; no Kaggle-API submit;
  output to `/kaggle/working`; each `.txt` ends with the correct
  `run_milestone28_variant("A"|"B"|"C"|"D"|"E")`; **no M16–M27 file modified**;
  gap3 creates only unit-timepoint edges.
- intelligence_db: M19-C stays **final/best @0.880**; the winning-strategy doc
  added as a source; **M27-A recorded 0.859** (pruning path deprioritized: M27
  B/C/D blocked); `M28_A–E` added as **pending**; `DEC_M28_METRIC_AWARE_RECALL_PIVOT`
  recorded; `next_actions.md` now recommends M28 (submit order A→C→D→B→E). Tests
  **6/6**.

**Status:** M19-C **0.880** remains the recommended final submission. M28 A–E are
experimental — submit only on `OK_TO_SUBMIT_EXPERIMENTAL` (plus the per-variant
explosion guard), keep 0.880 unless beaten. Do **not** build det 0.90/0.80 until
M28-E (0.95) lands.
