# Score Timeline

_Generated 2026-07-07 22:25 UTC from intelligence.duckdb._

Baseline **0.874** · best so far **0.8770**.

Delta = step change vs the previous **scored** experiment (chronological).

| # | Date | Experiment | Score | Δ vs prev | Δ vs baseline | Status |
|---|------|------------|-------|-----------|---------------|--------|
| 1 | 2026-06-25 | `M16_BASELINE` | 0.8740 | — | +0.000 | scored |
| 2 | 2026-06-28 | `M17_C_DET_0985` | 0.8740 | +0.000 | +0.000 | scored |
| 3 | 2026-07-01 | `M18_C_EDGE_PRUNE` | 0.8740 | +0.000 | +0.000 | scored |
| 4 | 2026-07-04 | `M19_A_SAFE_DIVISIONS_PRUNE` | 0.8770 | +0.003 | +0.003 | scored |
| 5 | 2026-07-07 | `M19_C_FULL_CHAIN_PENDING` | pending | — | — | pending |

## Notes per experiment

- **M16_BASELINE** (0.8740, scored): Reference UNet+transformer+ILP learned graph. Establishes the 0.874 baseline that all later variants are measured against.
- **M17_C_DET_0985** (0.8740, scored): Lower det-threshold admits more detections but the ILP re-optimizes to essentially the same graph. Score identical to baseline.
- **M18_C_EDGE_PRUNE** (0.8740, scored): Post-ILP AND-gate prune (long AND low-prob) removed 54 edges. Too small to move the metric; score identical to baseline.
- **M19_A_SAFE_DIVISIONS_PRUNE** (0.8770, scored): First metric-aware post-processing. +466 safe division edges, zero synthetic nodes. Moved the score for the first time: 0.874 -> 0.877. Confirms the division term (0.1 weight) is a live lever.
- **M19_C_FULL_CHAIN_PENDING** (pending, pending): Full lb893-style chain on top of the M19-A division core: single-frame + velocity-gated two-frame gap recovery (1309 synthetic nodes) + line-fit smoothing. Submitted; public score pending. Synthetic nodes are the risk to watch (node over-prediction penalty).
