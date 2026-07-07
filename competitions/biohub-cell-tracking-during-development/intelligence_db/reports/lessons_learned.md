# Lessons Learned

_Generated 2026-07-07 22:25 UTC from intelligence.duckdb._

Distilled, evidence-linked findings driving strategy.

## predict command — `LES_DET_FLAT`
- **Lesson:** det-threshold changes across 0.975-0.995 are flat at 0.874: the ILP re-optimizes to essentially the same graph, so the detection threshold is not a scoring lever on the learned path.
- **Evidence:** M16_BASELINE,M17_C_DET_0985
- **Confidence:** high

## post-ILP edge prune — `LES_EDGE_PRUNE_FLAT`
- **Lesson:** A small conservative post-ILP edge prune (54 edges, AND-gate on long+low-prob) did not move the score. Tiny edge-set edits are below the metric's resolution.
- **Evidence:** M18_C_EDGE_PRUNE
- **Confidence:** high

## metric — `LES_NODE_PENALTY`
- **Lesson:** The score penalizes node over-prediction (0.1*(T_pred-T_true)/T_true). Adding synthetic nodes (gap recovery) is a risk: they must land close enough to real GT nodes to be matched, or they inflate T_pred for no edge benefit. Prune isolated nodes; add synthetic nodes cautiously.
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

- [1.00] Cell-tracking competition metric specification — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- [1.00] lb893 learned-graph tracker with micro-safe divisions — https://github.com/dalloliogm/kaggle_competitions
- [0.95] tracking_cellmot/metrics.py — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/metrics.py
- [0.95] tracking_cellmot/division_metrics.py — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/division_metrics.py
- [0.90] evaluate.py scoring entrypoint — https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/scripts/evaluate.py
- [0.85] gap2 velocity ablation — https://github.com/dalloliogm/kaggle_competitions
- [0.70] top notebooks analysis — https://github.com/dalloliogm/kaggle_competitions
- [0.20] nms38 candidate — https://github.com/dalloliogm/kaggle_competitions
