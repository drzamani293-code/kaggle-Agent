# Score Timeline

_Generated 2026-07-08 14:19 UTC from intelligence.duckdb._

Baseline **0.874** · best so far **0.8800**.

Delta = step change vs the previous **scored** experiment (chronological).

| # | Date | Experiment | Score | Δ vs prev | Δ vs baseline | Status |
|---|------|------------|-------|-----------|---------------|--------|
| 1 | 2026-06-25 | `M16_BASELINE` | 0.8740 | — | +0.000 | scored |
| 2 | 2026-06-28 | `M17_C_DET_0985` | 0.8740 | +0.000 | +0.000 | scored |
| 3 | 2026-07-01 | `M18_C_EDGE_PRUNE` | 0.8740 | +0.000 | +0.000 | scored |
| 4 | 2026-07-04 | `M19_A_SAFE_DIVISIONS_PRUNE` | 0.8770 | +0.003 | +0.003 | scored |
| 5 | 2026-07-07 | `M19_C_FULL_CHAIN_PENDING` | 0.8800 | +0.003 | +0.006 | scored |
| 6 | 2026-07-08 | `M20_A_FULLCHAIN_TUNED` | pending | — | — | unsafe |
| 7 | 2026-07-08 | `M20_A_FULLCHAIN_TUNED_FIXED` | pending | — | — | unsafe |

## Notes per experiment

- **M16_BASELINE** (0.8740, scored): Reference UNet+transformer+ILP learned graph. Establishes the 0.874 baseline that all later variants are measured against.
- **M17_C_DET_0985** (0.8740, scored): Lower det-threshold admits more detections but the ILP re-optimizes to essentially the same graph. Score identical to baseline.
- **M18_C_EDGE_PRUNE** (0.8740, scored): Post-ILP AND-gate prune (long AND low-prob) removed 54 edges. Too small to move the metric; score identical to baseline.
- **M19_A_SAFE_DIVISIONS_PRUNE** (0.8770, scored): First metric-aware post-processing. +466 safe division edges, zero synthetic nodes. Moved the score for the first time: 0.874 -> 0.877. Confirms the division term (0.1 weight) is a live lever.
- **M19_C_FULL_CHAIN_PENDING** (0.8800, scored): Full lb893-style chain on top of the M19-A division core: single-frame + velocity-gated two-frame gap recovery (1309 synthetic nodes) + line-fit smoothing. Scored 0.880 (+0.003 over M19-A, +0.006 over baseline): gap recovery + smoothing add value ON TOP of divisions, and the 1309 synthetic nodes paid off rather than costing the node penalty. New best.
- **M20_A_FULLCHAIN_TUNED** (pending, unsafe): UNSAFE - BASELINE MISMATCH: DO NOT SUBMIT. The run's pre-postprocess base learned graph changed vs M19-C (n_nodes_before 131797->142193, n_edges_before 118992->127563) despite an identical predict command - a different support-pack/weights artifact (alternate/TTA/other-epoch) was selected on Kaggle. Gate tuning on a different base is not a controlled experiment. Superseded by M20_A_FULLCHAIN_TUNED_FIXED, which adds a hard baseline guard. Original intent: widen full_chain gates (div 4.7/6.85/7.45->5.1/7.4/7.9, gap1 6.2->7.0, gap2 4.4/10.2->4.7/11.0).
- **M20_A_FULLCHAIN_TUNED_FIXED** (pending, unsafe): UNSAFE - BASELINE MISMATCH: DO NOT SUBMIT. The baseline guard worked as designed: with support pack tom99763/biohub-tracking-support-pack-50ep-v1 (artifact_name biohub-tracking-support-pack-5090-50ep-v1) the pre-postprocess base was n_nodes_before=161098 / n_edges_before=137520 vs M19-C's 131797/118992, so baseline_guard_passed=False and submission_recommendation=DO_NOT_SUBMIT_BASELINE_MISMATCH. Both available 50ep packs now fail the guard (pilkwang -> 142193/127563, tom99763 -> 161098/137520); neither reproduces the M19-C base. valid=True, fallback_used=False, final_source=reference_learned_graph_postprocessed, but NOT submitted. Keep M19-C (0.880). Recover the true M19-C baseline artifact before retrying.
