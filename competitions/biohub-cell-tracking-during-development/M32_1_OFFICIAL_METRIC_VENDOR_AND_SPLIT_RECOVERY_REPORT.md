# M32.1 — Official Metric Vendor & Split Recovery Fix (NON-SUBMIT)

## Why this milestone exists
On Kaggle, **M32_A** passed the exact 400ep artifact guard and discovered ~200 train GEFF
stores, but the OFFICIAL scorer could **not be imported** (`tracking_cellmot` is not on the
Kaggle image and no source was mounted) and **no split config** was present. The audit
therefore emitted `OFFICIAL_METRIC_NOT_FOUND` + `CLEAN_HOLDOUT_UNRESOLVED` and correctly
refused to fabricate a CV score. Official local CV cannot run without (a) the authoritative
scorer and (b) a provably clean `split_0` holdout.

M32.1 repairs both **honestly**:
- **Vendor** the authoritative scorer **offline, byte-for-byte** from
  `royerlab/kaggle-cell-tracking-competition` — never a rewrite, never an approximation,
  never `local_metric.py`.
- **Verify** it with 10 synthetic OFFICIAL fixtures scored by the real `evaluate()`.
- **Recover** `split_0` only from an **OBSERVED** file or checkpoint (deterministic
  reconstruction is forbidden because the repo *reads* `dataset_splits.json`; it does not
  regenerate it from an observable seed/algorithm), behind a zero-overlap leakage guard.

Everything here is **NON-SUBMIT**. `M19-C` (public **0.880**) remains final. `M32_B` official
replay may run **only** when `M32_1_B` returns `AUDIT_PASS`.

## Vendored source (byte-identical offline snapshot)
- **Upstream repo:** `https://github.com/royerlab/kaggle-cell-tracking-competition`
- **Upstream branch:** `main`
- **Upstream commit (pinned):** `7396b7e98e61844e799152ddda7e5493084cc8f3`
- **Location:** `vendor/royerlab_cellmot/` (+ `PROVENANCE.json`, per-file sha256, 12 files)
- **Compatibility patches applied:** **none** (`compatibility_patches: []`).

Core files embedded (base64, sha256-verified at runtime before import — proves the scorer is
never edited):

| file | sha256 (first 12) | bytes |
|------|-------------------|-------|
| `src/tracking_cellmot/__init__.py` | `f9df2098dd85` | 79 |
| `src/tracking_cellmot/metrics.py` | `700906a6c1cf` | 15844 |
| `src/tracking_cellmot/division_metrics.py` | `d1cf1e0a4300` | 15872 |

Full vendor tree (also committed): `__init__.py`, `metrics.py`, `division_metrics.py`,
`io.py`, `img_proc.py`, `models/{__init__,temporal_unet,simple_node_transformer}.py`,
`scripts/{evaluate,dataspec}.py`, `metrics.md`, `LICENSE`.

## Scorer facts (unchanged from upstream)
- `evaluate(graph, gt_graph, max_distance=7.0) -> EvaluationResult`; plus
  `evaluate_datasets`, `per_sample_metrics(er, n_total, node_recall)`, `summarise(rows)`.
- `adj_edge_jaccard = max(0, edge_jaccard · (1 − 0.1 · total_node_ratio))`,
  `total_node_ratio = (N_pred − N_total) / N_total`,
  `score = adj_edge_jaccard + 0.1 · division_jaccard` (`ADJUSTMENT_ALPHA=0.1`,
  `SCORE_DIVISION_WEIGHT=0.1`).
- Graph-scoring dependencies: **tracksdata, geff, polars, scipy**. `io.py` additionally pulls
  torch/dask/zarr but is **not** imported by `metrics`.

## Deliverables (both NON-SUBMIT, one-cell standalone runners)
- `M32_1_A_OFFICIAL_METRIC_VENDOR_AUDIT_NOT_SUBMIT.txt`
- `M32_1_B_OFFICIAL_CV_REAUDIT_NOT_SUBMIT.txt`
- Compiled module `milestone32_1_official_metric_vendor_runner.py` (2316 lines) +
  `kaggle_cell_milestone32_1_official_metric_vendor_runner.py`.

### M32_1_A — vendor audit
Materialise the embedded scorer to `<working>/vendor_official_metric/src/tracking_cellmot/`,
sha256-verify each file, prepend to `sys.path`, `import tracking_cellmot.metrics`, verify the
required API + `division_metrics` import, then run **10 synthetic official fixtures** through
the real `evaluate()`/`per_sample_metrics()`/`summarise()`:

1. perfect edge graph (jaccard 1.0) · 2. missing edge → FN↑ · 3. conflicting edge → FP ·
4. extra unannotated edge · 5. correct division TP · 6. false fork FP · 7. missed division FN ·
8. 7µm matching boundary (near matches, far does not) · 9. node over-prediction lowers
`adj_edge_jaccard` · 10. multi-dataset `summarise` aggregation.

Recommendation: `OFFICIAL_METRIC_VENDOR_PASS` / `OFFICIAL_METRIC_VERIFICATION_FAILED` /
`OFFICIAL_METRIC_DEPENDENCY_MISSING`. Writes `m32_1_metric_vendor_report.json`.

### M32_1_B — combined re-audit
Runs the artifact guard + vendor audit, then:
- **Normalized train inventory** — strip `.geff`/`.zarr`, exclude the `train`/`test`/`val`
  container dirs, require a paired GT GEFF (prefer a paired Zarr input), de-duplicate, read GT
  node/edge counts (no solution mask), `estimated_number_of_nodes`, and source group
  (`^([0-9a-fA-F]{4})_`). Writes `m32_1_normalized_train_inventory.csv`.
- **Exact `split_0` recovery cascade** — (A) OBSERVED split file
  (`dataset_splits.json`/manifests/config), (B) checkpoint metadata via
  `torch.load(..., weights_only=True)`, (C) deterministic reconstruction **FORBIDDEN**
  (`allowed=False`, because the repo reads the split, it does not regenerate it). Zero-overlap
  leakage guard. Writes `m32_1_split_recovery.json`.

`AUDIT_PASS` **only** when: vendored metric imports **and** all 10 fixtures pass **and**
inventory valid **and** clean `split_0` holdout proven **and** train∩test overlap = 0 **and**
400ep SHA passes. Otherwise a specific status: `OFFICIAL_METRIC_DEPENDENCY_MISSING` /
`OFFICIAL_METRIC_VERIFICATION_FAILED` / `DATA_LEAKAGE_DETECTED` / `SPLIT_DATASET_MISMATCH` /
`CLEAN_HOLDOUT_UNRESOLVED` / `WRONG_ARTIFACT`. `m32_b_replay_allowed = (AUDIT_PASS)`. Writes
`m32_1_official_cv_reaudit.json`.

## Exact Kaggle inputs required for a real (non-`DEPENDENCY_MISSING`) run
1. A dataset providing **tracksdata, geff, polars, scipy** wheels/packages (offline; no
   Internet `pip install`). Absent these → `OFFICIAL_METRIC_DEPENDENCY_MISSING` (no fake score).
2. The **400ep** weights artifact (`weight_sha256 ==
   12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`, name contains `400ep`).
3. Train **GT GEFF + paired Zarr** inputs (for the normalized inventory).
4. An **observed** `dataset_splits.json` (or split metadata inside the checkpoint) to prove a
   clean `split_0` holdout. Without it → `CLEAN_HOLDOUT_UNRESOLVED` (never a random holdout).

## Honesty guarantees
- The scorer is **vendored verbatim** and sha256-checked at runtime; it is **never** rewritten,
  approximated, or replaced by `local_metric.py`/`pp_sweep_v2.py` (those remain recorded only as
  reviewed **proxies**, never the scorer).
- No `submission.csv` is written and no Kaggle-API submit is performed by either runner.
- No random datasets are ever labelled "clean CV"; a holdout must be OBSERVED and overlap-free.
- Missing dependencies or missing split config produce **explicit statuses**, not numbers.

## Off-Kaggle verification (this environment)
- `py_compile` OK.
- `M32_1_A` and `M32_1_B` each: **23/23** self-tests pass, then the dry-run notice
  (`/kaggle/input` absent → self-tests only; on Kaggle without tracksdata/polars it reports
  `OFFICIAL_METRIC_DEPENDENCY_MISSING`, no proxy/fake).
- All 12 vendored files hash-match `PROVENANCE.json`.
- intelligence_db tests **6/6**.

## Status ledger
- `M19-C` public **0.880** — **final** (kept).
- `M32_A` — diagnostic, OBSERVED result: metric import failed + no split →
  `OFFICIAL_METRIC_NOT_FOUND` / `CLEAN_HOLDOUT_UNRESOLVED`.
- `M32_1_A` / `M32_1_B` — non-scoring vendor audit + re-audit.
- Decision `DEC_M32_1_VENDOR_AND_SPLIT_RECOVERY`: vendor → re-audit → **only if `AUDIT_PASS`**
  run `M32_B`. Never submit before official CV succeeds.
