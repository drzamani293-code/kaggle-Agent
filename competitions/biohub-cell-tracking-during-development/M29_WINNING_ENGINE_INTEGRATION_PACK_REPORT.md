# M29 — Winning Engine Integration Pack

**Goal:** integrate the uploaded winning post-processing **engine** into our
hidden-safe Kaggle runner. M28 was only a *partial* recall expansion (M28-A:
nodes=132166, gap1=1604, gap2=1386, synthetic=4376); the uploaded engine is a
*fuller* recall recovery whose `submission(3).csv` sanity-profiles to
nodes=139418, edges=132871, divisions≈2382, all invariants clean. M29 reproduces
the engine **logic** (never the static submission.csv) and **embeds** it so every
runner is a standalone Kaggle cell. M19-C **0.880** stays best/final unless an M29
variant beats it.

## Self-contained engine (embedded, no external import)

The engine is copied **in-line** into the runner — no dependency on an external
Kaggle utility dataset. Each M29 `.txt` needs only the competition dataset + the
mounted support pack. Pipeline (per dataset):

1. **build_base_graph** — M16-identical from raw GEFF (keeps edge_prob/edge_dist).
2. **motion_relink** *(optional)* — per-frame Hungarian; only edges longer than
   `relink_min_um` are eligible to be replaced, accepted within `relink_alt_max_um`
   and only if strictly closer.
3. **rescue_divisions** — slot-Hungarian orphan adoption: a source with one
   unit-t child gets a Hungarian-matched second child (division) under
   parent/sister gates + per-frame/global caps; with `div_allow_bare_end`, real
   track ends also adopt a nearest orphan (continuation).
4. **stitch_gaps 1/2/3** — per-timepoint Hungarian on a constant-velocity model:
   END@t → START@t+g+1 bridged by g synthetic nodes and g+1 **unit-timepoint**
   edges, gated on per-step / total distance and velocity consistency.
5. **linefit** *(optional)* → **prune isolated** → **relabel positive-consecutive**.

Every produced edge is unit-timepoint (t→t+1); `in_degree ≤ 1`, `out_degree ≤ 2`;
a direct multi-frame edge is never created. M29 runs whatever reference support
pack is mounted (like M19-C), reporting its `artifact_name` + `weight_sha256`; it
does **not** pin a specific artifact.

## Five submit-capable variants (submit order A → B → C → D → E)

| Variant | Change from A | det | Synthetic cap |
|---------|---------------|-----|---------------|
| **A** winning_engine_default | PPConfig defaults: gaps 1/2/3, divisions 6.5/9.0/frame 0.030/global 0.020, **no relink, no linefit** | 0.99 | ≤ 12000 |
| **B** engine_gap12_only | `gap_sizes=(1,2)` — isolate whether gap3 over-adds | 0.99 | ≤ 12000 |
| **C** engine_linefit_on | `linefit_enable=True` (window 2, weight 0.75) | 0.99 | ≤ 12000 |
| **D** engine_relink_on | `relink_enable=True` (min 7.0 µm, alt 4.5 µm) — **risky** | 0.99 | ≤ 12000 |
| **E** engine_det095_safe | `det_threshold=0.95` — cautious detection sweep | **0.95** | ≤ 18000 |

Plus a **non-submit local-CV harness**
(`M29_LOCAL_CV_HARNESS_INTEGRATION_NOT_SUBMIT.txt` →
`run_milestone29_local_cv()`): it locates train GT GEFF, smoke-runs the engine,
and **honestly reports whether the official metric is wired** — it never
fabricates CV numbers and never writes a competition submission (only
`m29_local_cv_report.json`; if the metric can't be located it names exactly what
function/path is missing).

## Expected diagnostics

- **A** (primary): n_node_rows **137000–142000**, n_edge_rows **129000–135000**,
  divisions **1800–2800**, multiframe **0**, in=1/out=2. Soft-WARN if outside the
  `submission(3).csv` profile (139418 ±3000 / 132871 ±4000 / 2382 ±800) — a WARN
  only, never a hard gate.
- **B**: fewer nodes/edges/divisions than A (no gap3).
- **C/D**: ≈ A with linefit smoothing / relinked edges (reports `n_relinked`).
- **E**: higher raw detections → node/edge counts up to the 165000/155000 ceiling.

## Safety gate

`OK_TO_SUBMIT_EXPERIMENTAL` requires **all** of: valid, no fallback, no NaN,
consecutive id, no dangling, all edges t→t+1, `max_in_degree ≤ 1`,
`max_out_degree ≤ 2`, all expected dynamic test datasets present, no dataset
emptied, `125000 ≤ n_node_rows ≤ 165000`, `110000 ≤ n_edge_rows ≤ 155000`,
`synthetic_nodes_added ≤ 12000` (E: 18000), `divisions_total ≤ 4000`,
`final_source = winning_postprocess_engine_reproduced`. Otherwise a specific
`DO_NOT_SUBMIT_*` is reported: `VALIDATION_FAILED` / `UNSANE_NODE_COUNT` /
`UNSANE_EDGE_COUNT` / `SYNTHETIC_EXPLOSION` / `DIVISION_EXPLOSION` /
`NODE_EXPLOSION` (E) / `MISSING_OR_EMPTY_DATASET`, and a valid hidden-safe fallback
submission is still written.

## Decision rules

- **M29-A > 0.880** → set A as the new candidate, continue B/C to improve.
- **M29-A ∈ [0.875, 0.880)** → test B and C.
- **M29-A < 0.875** → test B only; if B also fails, **pause M29 and run the
  local-CV harness** before more submits.
- Keep **M19-C version 12 = 0.880** final until a higher score is observed.

## Outputs (per submit-capable runner, `/kaggle/working`)

`milestone29_reference_submission_report.json` (all required fields incl.
engine_config, raw/final nodes+edges, per_dataset, gap1/2/3 candidates+closed,
synthetic_nodes_added, divisions_added, divisions_total, relink_enabled,
n_relinked, linefit_enabled, n_smoothed, isolated_nodes_pruned, degrees,
direct_multiframe_edges, dangling_edges, id_consecutive, no_nan, valid,
fallback_used, `final_source`, recommendation), `milestone29_engine_conversion.json`,
`milestone29_artifact_selection.json`, `milestone29_failure_fallback_report.json`,
`submission.csv`. The CV harness writes `m29_local_cv_report.json` only.

## Verification

- `milestone29_winning_engine_integration_runner.py` compiles; the standalone
  `.txt` runs as `__main__`; `run_milestone29_tests()` passes **7/7** on synthetic
  fixtures: gap stitching (gap1/gap3, unit edges only, synthetic at t+1..t+g),
  slot-Hungarian division rescue + bare-end adoption, motion relink (long-edge
  replacement), variant registry + per-variant predict det-threshold, the
  recommendation/explosion guards + soft warnings, and an end-to-end valid
  submission. All 5 variants + the CV harness dry-run cleanly off Kaggle.
- Static checks: exact reference predict flags (ilp-edge −1.0, appearance/
  disappearance 0.1, division 1.0, `--use-ilp`, `--unet-batch-size 4`), only
  det-threshold varies (E 0.95); dynamic hidden-safe split discovery; **no static
  submission.csv**; **no dependency on the uploaded files at Kaggle runtime**
  (engine embedded); self-contained one-cell runners; `final_source` string
  exactly `winning_postprocess_engine_reproduced`; **no M16–M28 file modified**.
- intelligence_db: M19-C stays **final/best @0.880**; the winning-engine package
  added as 6 sources (strategy, winning_postprocess.py, integration_cell.py,
  local_cv_harness.py, README, submission(3) profile); M27 pruning path already
  recorded failed (0.859); `M29_A–E` added as **pending**;
  `DEC_M29_WINNING_ENGINE_INTEGRATION` recorded; `next_actions.md` recommends M29
  (A→B→C→D→E, then the local-CV harness). Tests **6/6**.

**Status:** M19-C **0.880** remains the recommended final submission. M29 A–E are
the primary experimental path — submit A first, and only on
`OK_TO_SUBMIT_EXPERIMENTAL`. Do **not** submit the uploaded `submission(3).csv`;
do **not** build det 0.90/0.80 until M29-E (0.95) lands.
