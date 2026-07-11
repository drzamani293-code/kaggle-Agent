# M32 — Official Local CV & Model Selection (NON-SUBMIT)

**Goal:** stop blind leaderboard sweeps and choose the next pipeline with a **real
local cross-validation** on held-out TRAIN videos + TRAIN GT lineage GEFF, scored
by the repository's **official** metric. It compares, on the same clean validation
data and the same cached prediction: **P0** raw/base ILP, **P1** M19-C-style, **P2**
M29-A engine-v1, **P3** M30-C sparse-kNN relinker-v2. M32 is **exclusively
non-submit** — it never creates or uploads a competition submission. M19-C **0.880**
stays confirmed best/final.

## Honesty-first overriding rules (implemented)

1. **Official metric only.** `local_metric.py` is a **proxy** (its division +
   node-penalty + aggregation are not authoritative) and is **never** used as the
   scorer. `resolve_official_metric()` imports the actual official metric
   (`tracking_cellmot.metrics` / `scripts/evaluate.py`), records source path +
   functions + SHA256, and if it can't be invoked returns **`CV_NOT_WIRED`** — never
   fabricating or substituting a proxy score.
2. **Clean held-out only.** `resolve_split0_holdout()` reads split_0 train/val
   **dynamically** from split/config metadata, selects held-out = datasets NOT in
   split_0 training, and runs a leakage guard. No split config → **`CLEAN_HOLDOUT_UNRESOLVED`**;
   overlap → **`DATA_LEAKAGE_DETECTED`**.
3. **Same input.** One config-hash-guarded prediction cache feeds all pipelines;
   `m32_cache_is_stale()` rejects a cache when the artifact SHA or predict-config
   hash differ.
4. **No static CSV, no Kaggle submit.** No submission is written; sample_submission
   is never used; supplied CSVs are never used. CV reports only.
5. **No M16–M31 modification.** Prior pipeline families are **embedded snapshots**
   (reusing the M19 metric primitives), never modified.

> In THIS environment the official metric, train GT, and split config are not
> mounted (off Kaggle), so the audit honestly reports `OFFICIAL_METRIC_NOT_FOUND` /
> `CV_NOT_WIRED` and no ranking is produced. On Kaggle, with the repo + train GT
> attached, the resolver binds the real official metric before any score.

## Runners (all NON-SUBMIT; first Kaggle action = run M32_A only)

| Runner | Stage | Role |
|--------|-------|------|
| `M32_A_OFFICIAL_CV_AUDIT_NOT_SUBMIT` | 0 | resolve official metric + train GT + split_0 holdout + artifact + env; provenance + status |
| `M32_B_OFFICIAL_CV_REPLAY_NOT_SUBMIT` | 2 | common cache → P0/P1/P2/P3 → adapter → official scoring → robust ranking + LOO |
| `M32_C_OFFICIAL_SMALL_SWEEP_NOT_SUBMIT` | 3 | ≤10 staged configs on the official metric + LOO stability |
| `M32_D_DET_THRESHOLD_PILOT_NOT_SUBMIT` | 4 | det 0.99 vs 0.95, separate cache each |

## Exact pipeline configs (P0/P1/P2/P3)

- **P0** `m32_base_ilp_control` — cached ILP solution edges → valid graph; no gaps,
  no divisions, no synthetic nodes, no linefit, no kNN.
- **P1** `m32_m19c_style_replay` — M19-C full chain (safe_divisions → gap1 → gap2 →
  linefit → prune). **`replay_exact=False`** → labelled `M19C_REPLAY_NOT_EXACT`
  (M19-C-style on the common 400ep cache, **not** the byte-exact 0.880 submission).
- **P2** `m32_m29a_engine_v1_replay` — gap_sizes (1,2,3), gap_max_step 4.6, totals
  {1:6.5, 2:10.5, 3:14.5}, division rescue on, relink off, linefit off, prune on.
- **P3** `m32_m30c_sparse_knn_replay` — tight kNN k=3 / radius 4.5µm / prob 0.20,
  null_link_cost 4.2, max_link_um 7.0, two-stage no-link primary+division, gaps
  (1,2), no linefit, prune on.

Each pipeline goes through one **verified adapter** (`adapt_pipeline_output`):
positive-consecutive ids, no dangling/multiframe edges, in≤1/out≤2, no NaN. An
invalid output is marked `PIPELINE_INVALID` and excluded from ranking (never scored 0).

## Official metric verification

Ten synthetic fixtures assert behaviour **against the resolved official callables**
(injected in tests): perfect graph → jaccard 1.0; missing GT edge → FN; conflicting
edge → FP; correct/false/missing division; node overprediction → node-penalty
lowers the adjusted score; two-dataset aggregation matches the official summariser;
7µm gate + voxel scale. Without the official metric the fixtures report
`CV_NOT_WIRED` — never a hardcoded `local_metric.py` value.

## Cache design

`M32_PREDICT_CONFIG` (det 0.99, ilp-edge −1.0, appearance/disappearance 0.1,
division 1.0, use-ilp, batch 4, **no TTA, no DeepCenter**), `m32_predict_config_hash`
over {artifact_sha, config}, per-dataset `m32_cache_entry` (nodes / total-edges /
solution-edges / candidate-edges / edge_prob-availability / hashes / completed /
runtime), and `m32_cache_is_stale` for restart/reuse. One dataset at a time.

## Winner-selection rules

`compare_challenger_vs_baseline` (vs P1) → **`CV_WINNER_STRONG`** (agg delta ≥ +0.003,
wins ≥ 60% of datasets, no dataset delta < −0.010, valid on all), **`CV_WINNER_WEAK`**
(delta > 0 but inconsistent / < +0.003), **`NO_CV_IMPROVEMENT`** (delta ≤ 0),
**`CV_INCONCLUSIVE`** (too few clean datasets / scorer incomplete / inputs differ).
`leave_one_out_stability` reports whether the full-set winner changes when any single
held-out dataset is removed. A leaderboard submit is only ever recommended when the
metric is verified, holdout proven, inputs identical, challenger valid, and the
result is STRONG or a documented WEAK.

## Small sweep (≤10, staged)

`build_sweep_stages`: baseline → gap family {(1,2),(1,2,3)} → division cap
{0.010,0.020} → kNN mode {off, k2/4.0/0.20, k3/4.5/0.20} → null-link {3.8,4.2}.
One-factor-at-a-time; `enumerate_sweep_configs` asserts total ≤ 10; `run_staged_sweep`
carries the best forward each stage. No DeepCenter, no TTA, no ensemble. Det pilot:
0.99 vs 0.95 only (0.93/0.90/0.85 explicitly not run yet).

## Output report paths

`m32_official_cv_audit.json`, `m32_train_dataset_inventory.csv`,
`m32_split_inventory.json` (A); `m32_official_cv_results.json/.csv`,
`m32_pipeline_structural_diagnostics.json`, `m32_pipeline_ranking.json` (B);
`m32_small_sweep_results.json/.csv`, `m32_small_sweep_best.json`,
`m32_leave_one_out_stability.json` (C); `m32_det_pilot_report.json` (D). All under
`/kaggle/working`.

## Verification

- `milestone32_official_local_cv_runner.py` compiles; every `.txt` runs standalone;
  `run_milestone32_tests()` passes **28/28**: official-metric discovery (proxy never
  the scorer), perfect-graph wiring, CV_NOT_WIRED without the official metric, edge
  FP/FN + division + node-penalty wiring, multi-dataset aggregation (official
  summariser preferred), GT reader keeps all edges (no solution mask), leakage /
  clean-holdout logic, artifact SHA guard, cache config-hash invalidation, P0/P1/P2/P3
  conversions + exact configs + provenance, adapter IDs/coords/invariants (multiframe
  + dangling caught), end-to-end validity, invalid-pipeline exclusion,
  strong/weak/none/inconclusive winner logic, LOO stability, sweep size ≤10 + staging,
  no DeepCenter/TTA, det-pilot scope, no-fake-zero-on-failure.
- Static: **no M16–M31 file modified**, **no submission.csv / no Kaggle submit / no
  sample_submission** in the M32-authored code, **no proxy reported as official**, no
  fake CV score, dynamic train/GT/split discovery, exact 400ep guard, all outputs
  under `/kaggle/working`, standalone one-cell runners.
- intelligence_db: `local_metric.py` + `pp_sweep_v2.py` added as reviewed **proxy**
  sources (useful ideas, not authoritative); `M32_A/B/C/D` non-scoring;
  `DEC_M32_OFFICIAL_LOCAL_CV_FIRST` recorded (M19-C 0.880 best/final; M29-A 0.876
  failed; M30-C pending; M31 blocked); `next_actions` now leads with the official-CV
  path (A→B→C→optional D) and never recommends a blind LB submit before official CV
  succeeds. DB tests **6/6**.

**First Kaggle action:** run **`M32_A_OFFICIAL_CV_AUDIT_NOT_SUBMIT`** only — it tells
you whether the official metric + clean split_0 holdout are actually available before
any scoring is attempted. Everything downstream is gated on that audit passing.
