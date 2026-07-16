# Lessons Learned

_Generated 2026-07-16 00:42 UTC from intelligence.duckdb._

Distilled, evidence-linked findings driving strategy.

## reproducibility — `LES_BASELINE_REPRO`
- **Lesson:** The pre-postprocess BASE learned graph must be pinned before comparing post-processing variants: an identical predict command does NOT guarantee the same base if a different support-pack/weights artifact is mounted. Both available 50ep packs (pilkwang, tom99763) produced different base node/edge counts (142193/127563 and 161098/137520) than M19-C's 131797/118992, making M20 gate tuning uninterpretable and unsubmittable. Always record the selected artifact/weights and hard-guard n_nodes_before/n_edges_before against the known-good baseline (tolerance 0) before trusting or submitting a variant.
- **Evidence:** M19_C_FULL_CHAIN_PENDING,M20_A_FULLCHAIN_TUNED,M20_A_FULLCHAIN_TUNED_FIXED
- **Confidence:** high

## predict command — `LES_DET_FLAT`
- **Lesson:** det-threshold changes across 0.975-0.995 are flat at 0.874: the ILP re-optimizes to essentially the same graph, so the detection threshold is not a scoring lever on the learned path.
- **Evidence:** M16_BASELINE,M17_C_DET_0985
- **Confidence:** high

## post-ILP edge prune — `LES_EDGE_PRUNE_FLAT`
- **Lesson:** A small conservative post-ILP edge prune (54 edges, AND-gate on long+low-prob) did not move the score. Tiny edge-set edits are below the metric's resolution.
- **Evidence:** M18_C_EDGE_PRUNE
- **Confidence:** high

## metric — `LES_NODE_PENALTY`
- **Lesson:** The score penalizes node over-prediction (0.1*(T_pred-T_true)/T_true), so synthetic nodes are a risk in principle. In practice M19-C's 1309 velocity/distance-gated synthetic gap nodes still improved the score (0.877 -> 0.880), so well-gated synthetic gap recovery pays off; the gates (per-step distance, velocity consistency) are what keep the added nodes matchable. Continue to track synthetic_nodes_added against score.
- **Evidence:** M19_A_SAFE_DIVISIONS_PRUNE,M19_C_FULL_CHAIN_PENDING
- **Confidence:** medium

## strategy — `LES_POSTPROCESS_OVER_ILP`
- **Lesson:** Post-GEFF metric-aware processing is more promising than blind ILP-weight changes so far: the public 0.893 notebook uses the IDENTICAL predict command to our M16, so its gain is entirely post-processing, not re-weighting the ILP.
- **Evidence:** M16_BASELINE,M19_A_SAFE_DIVISIONS_PRUNE
- **Confidence:** high

## divisions — `LES_SAFE_DIVISIONS`
- **Lesson:** Geometric safe-division recovery improved 0.874 -> 0.877 (+466 division edges, no synthetic nodes). The 0.1-weighted division term is a real, reachable lever the det-threshold cannot touch.
- **Evidence:** M19_A_SAFE_DIVISIONS_PRUNE
- **Confidence:** high

## Public sources consulted (by relevance)

- [1.00] Supplied 0.902 reference bundle (public_0902_motion_division_calibration) - USER-OBSERVED, not yet Kaggle-verified — user-supplied:0902-reference-bundle
- [1.00] lb893 learned-graph tracker with micro-safe divisions — https://github.com/dalloliogm/kaggle_competitions
- [1.00] Cell-tracking competition metric specification — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- [1.00] Vendored offline snapshot: tracking_cellmot @ 7396b7e9 (12 files) — https://github.com/royerlab/kaggle-cell-tracking-competition/tree/7396b7e98e61844e799152ddda7e5493084cc8f3
- [0.99] integration_v2.py (uploaded) — upload:integration_v2.py
- [0.99] submission(3).csv sanity profile (uploaded) — upload:submission(3).csv
- [0.99] analysis_09_strategy.md (uploaded) — upload:analysis_09_strategy.md
- [0.99] README_v2.md (uploaded) — upload:README_v2.md
- [0.99] local_cv_harness(1).py (uploaded) — upload:local_cv_harness(1).py
- [0.99] biohub-cell-tracking-v4-unet-ilp-reproduction (public high-score notebook) — kaggle:notebook/biohub-cell-tracking-v4-unet-ilp-reproduction
- [0.99] biohub_winning_strategy(1).md (uploaded) — upload:biohub_winning_strategy(1).md
- [0.99] winning_postprocess.py (uploaded engine) — upload:winning_postprocess.py
- [0.99] integration_cell.py (uploaded) — upload:integration_cell.py
- [0.99] local_cv_harness.py (uploaded) — upload:local_cv_harness.py
- [0.99] README_engine.md (uploaded) — upload:README_engine.md
- [0.99] winning_postprocess_v2.py (uploaded v2 engine) — upload:winning_postprocess_v2.py
- [0.99] winning_postprocess.py (v1, shipped alongside v2) — upload:winning_postprocess.py
- [0.97] biohub_winning_strategy.md (uploaded) — upload:biohub_winning_strategy.md
- [0.95] tracking_cellmot/division_metrics.py — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/division_metrics.py
- [0.95] tracking_cellmot/metrics.py — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/metrics.py
- [0.95] biohub-tracking-support-pack-400ep-snapshot-v1 (discovered via M23-C guard failure) — kaggle:/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1
- [0.90] evaluate.py scoring entrypoint — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/scripts/evaluate.py
- [0.90] blend_submissions.py (output-level ensemble, REVIEWED PROXY) — upload:blend_submissions.py
- [0.90] geff_ensemble.py (probability-level ensemble, REVIEWED PROXY) — upload:geff_ensemble.py
- [0.90] local_metric(1).py (PROXY scorer variant, NEVER official) — upload:local_metric(1).py
- [0.90] model_level_roadmap.md (ensemble roadmap) — upload:model_level_roadmap.md
- [0.90] Self-contained geometric TTA (M34, no external outputs) — internal:M34
- [0.85] gap2 velocity ablation — https://github.com/dalloliogm/kaggle_competitions
- [0.70] top notebooks analysis — https://github.com/dalloliogm/kaggle_competitions
- [0.60] local_metric.py (repo, reviewed PROXY) — repo:local_metric.py
- [0.60] pp_sweep_v2.py (reviewed PROXY) — upload:pp_sweep_v2.py
- [0.20] nms38 candidate — https://github.com/dalloliogm/kaggle_competitions
