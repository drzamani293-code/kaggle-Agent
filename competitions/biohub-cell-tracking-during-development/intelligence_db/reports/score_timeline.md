# Score Timeline

_Generated 2026-07-09 01:56 UTC from intelligence.duckdb._

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
| 8 | 2026-07-08 | `M21_A_PILKWANG350_M19C_GATES` | 0.8740 | -0.006 | +0.000 | scored |
| 9 | 2026-07-08 | `M21_B_PILKWANG350_SAFE_DIV_ONLY` | pending | — | — | blocked |
| 10 | 2026-07-08 | `M21_C_PILKWANG350_LIGHT_GAP` | pending | — | — | blocked |
| 11 | 2026-07-08 | `M22_ARTIFACT_FORENSIC` | pending | — | — | diagnostic |
| 12 | 2026-07-08 | `M22_A_TRUEBASE_SAFE_DIV_TUNE` | pending | — | — | blocked |
| 13 | 2026-07-08 | `M22_B_TRUEBASE_LIGHT_GAP` | pending | — | — | blocked |
| 14 | 2026-07-08 | `M22_C_TRUEBASE_DIV_PLUS_GAP1_ONLY` | pending | — | — | blocked |
| 15 | 2026-07-09 | `M23_A_PILKWANG350_NODE_PRUNE_LIGHT` | pending | — | — | blocked |
| 16 | 2026-07-09 | `M23_B_PILKWANG350_NODE_PRUNE_SAFE_DIV_ONLY` | 0.8770 | +0.003 | +0.003 | scored |
| 17 | 2026-07-09 | `M23_C_PILKWANG350_EDGE_NODE_BALANCED` | pending | — | — | wrong_artifact |
| 18 | 2026-07-09 | `M24_A_400EP_FULLCHAIN_M19C_GATES` | pending | — | — | pending |
| 19 | 2026-07-09 | `M24_B_400EP_GAP1_ONLY` | pending | — | — | pending |
| 20 | 2026-07-09 | `M24_C_400EP_SAFE_DIV_ONLY` | pending | — | — | pending |

## Notes per experiment

- **M16_BASELINE** (0.8740, scored): Reference UNet+transformer+ILP learned graph. Establishes the 0.874 baseline that all later variants are measured against.
- **M17_C_DET_0985** (0.8740, scored): Lower det-threshold admits more detections but the ILP re-optimizes to essentially the same graph. Score identical to baseline.
- **M18_C_EDGE_PRUNE** (0.8740, scored): Post-ILP AND-gate prune (long AND low-prob) removed 54 edges. Too small to move the metric; score identical to baseline.
- **M19_A_SAFE_DIVISIONS_PRUNE** (0.8770, scored): First metric-aware post-processing. +466 safe division edges, zero synthetic nodes. Moved the score for the first time: 0.874 -> 0.877. Confirms the division term (0.1 weight) is a live lever.
- **M19_C_FULL_CHAIN_PENDING** (0.8800, scored): Full lb893-style chain on top of the M19-A division core: single-frame + velocity-gated two-frame gap recovery (1309 synthetic nodes) + line-fit smoothing. Scored 0.880 (+0.003 over M19-A, +0.006 over baseline): gap recovery + smoothing add value ON TOP of divisions, and the 1309 synthetic nodes paid off rather than costing the node penalty. New best.
- **M20_A_FULLCHAIN_TUNED** (pending, unsafe): UNSAFE - BASELINE MISMATCH: DO NOT SUBMIT. The run's pre-postprocess base learned graph changed vs M19-C (n_nodes_before 131797->142193, n_edges_before 118992->127563) despite an identical predict command - a different support-pack/weights artifact (alternate/TTA/other-epoch) was selected on Kaggle. Gate tuning on a different base is not a controlled experiment. Superseded by M20_A_FULLCHAIN_TUNED_FIXED, which adds a hard baseline guard. Original intent: widen full_chain gates (div 4.7/6.85/7.45->5.1/7.4/7.9, gap1 6.2->7.0, gap2 4.4/10.2->4.7/11.0).
- **M20_A_FULLCHAIN_TUNED_FIXED** (pending, unsafe): UNSAFE - BASELINE MISMATCH: DO NOT SUBMIT. The baseline guard worked as designed: with support pack tom99763/biohub-tracking-support-pack-50ep-v1 (artifact_name biohub-tracking-support-pack-5090-50ep-v1) the pre-postprocess base was n_nodes_before=161098 / n_edges_before=137520 vs M19-C's 131797/118992, so baseline_guard_passed=False and submission_recommendation=DO_NOT_SUBMIT_BASELINE_MISMATCH. Both available 50ep packs now fail the guard (pilkwang -> 142193/127563, tom99763 -> 161098/137520); neither reproduces the M19-C base. valid=True, fallback_used=False, final_source=reference_learned_graph_postprocessed, but NOT submitted. Keep M19-C (0.880). Recover the true M19-C baseline artifact before retrying.
- **M21_A_PILKWANG350_M19C_GATES** (0.8740, scored): SCORED 0.874 on the pilkwang drift base - FAILED to beat M19-C (0.880), and equal to the M16 baseline. Proves the newer/larger pilkwang350 base (142193 nodes) is NOT automatically better; post-processing on a drifted base does not help. Confirms the need to recover the TRUE M19-C base (131797/118992). Full_chain with ORIGINAL M19-C gates on the pilkwang base.
- **M21_B_PILKWANG350_SAFE_DIV_ONLY** (pending, blocked): BLOCKED / deprioritized: pilkwang drift base. M21-A on the same drift base scored only 0.874 (< M19-C 0.880), so drift-base variants are not worth a submission unless explicitly testing drift. Recover the true M19-C base first (see M22).
- **M21_C_PILKWANG350_LIGHT_GAP** (pending, blocked): BLOCKED / deprioritized: pilkwang drift base (same reason as M21-B; M21-A drift-base run only scored 0.874). Recover the true M19-C base first (see M22).
- **M22_ARTIFACT_FORENSIC** (pending, diagnostic): RESULT: TRUE_M19C_ARTIFACT_NOT_FOUND. The Notebook input did not contain usable repo/weights, and no mounted pack reproduced the true M19-C base (131797/118992). The true M19-C artifact could not be recovered, so M19-C 0.880 remains final and downstream tuning must run on an available (pilkwang350) base. Superseded by M23 (node-penalty repair on the pinned pilkwang350 base).
- **M22_A_TRUEBASE_SAFE_DIV_TUNE** (pending, blocked): BLOCKED until the TRUE M19-C artifact is recovered (base 131797/118992). Widened safe-division gates (4.7/6.85/7.45->5.1/7.3/7.9) + prune, ZERO synthetic nodes. Dual-guarded: true_m19c_artifact_guard + public_base_count_guard; submit only if recommendation=OK_TO_SUBMIT_TRUEBASE_EXPERIMENT. Submit order 1 of 3.
- **M22_B_TRUEBASE_LIGHT_GAP** (pending, blocked): BLOCKED until the TRUE M19-C artifact is recovered. M19-C safe_divisions + gap1 (6.6) + TIGHT gap2 (3.6/8.0, cos_min -0.05, normdiff 4.6, caps 0.0025/100) + linefit + prune; synthetic target ~900-1300. Dual-guarded. Submit order 2 of 3.
- **M22_C_TRUEBASE_DIV_PLUS_GAP1_ONLY** (pending, blocked): BLOCKED until the TRUE M19-C artifact is recovered. safe_divisions + gap1 + linefit + prune, NO gap2 - isolates whether gap2 is the risky part; synthetic target ~500-800. Dual-guarded. Submit order 3 of 3.
- **M23_A_PILKWANG350_NODE_PRUNE_LIGHT** (pending, blocked): SUPERSEDED by M24. The M23 node-prune approach on pilkwang350 only reached 0.877 (M23-B), below M19-C 0.880, and M23-C revealed a cleaner 400ep base - so pursue M24 (400ep, no prune) instead. Not submitted.
- **M23_B_PILKWANG350_NODE_PRUNE_SAFE_DIV_ONLY** (0.8770, scored): SCORED 0.877 on the pilkwang350 node-pruned base - node-penalty repair + safe divisions recovered from M21-A's 0.874 to 0.877 (matching M19-A), but did NOT beat M19-C 0.880. Confirms node-penalty reduction helps but the pilkwang350 base is still not as good as the true M19-C base. Zero synthetic nodes.
- **M23_C_PILKWANG350_EDGE_NODE_BALANCED** (pending, wrong_artifact): NOT SUBMITTED - M23 artifact guard failed (a NEW artifact was mounted at the pilkwang path). This accidentally REVEALED the 400ep snapshot: artifact_name biohub-tracking-support-pack-400ep-snapshot-v1, weight_sha256 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771, RAW base 127790/115694 (much closer to M19-C's 131797/118992 than pilkwang350). Its M23 over-repair dropped base 127790->121369 (too aggressive; wrong guard). Redirected to M24 (400ep as-is, no prune).
- **M24_A_400EP_FULLCHAIN_M19C_GATES** (pending, pending): EXPERIMENTAL on the CLEAN 400ep base (127790/115694, no prune) + M19-C full_chain gates. Highest-upside: lean base close to M19-C's true 131797/118992 + the chain that scored 0.880. Pinned to 400ep (path + artifact_name 400ep + weight 12f688..). Sanity: base/final node count in 120000-135000, synthetic<=2200. Expected final ~129000-131000, synthetic ~1000-1600. SUBMIT ORDER 1 of 3 (only high-upside candidate).
- **M24_B_400EP_GAP1_ONLY** (pending, pending): EXPERIMENTAL on the CLEAN 400ep base (no prune). safe_divisions + gap1 + linefit + prune, NO gap2 - tests whether gap2 is risky on the lean base. Expected synthetic ~400-800 (safer than A). Pinned 400ep. SUBMIT ORDER 2 of 3.
- **M24_C_400EP_SAFE_DIV_ONLY** (pending, pending): EXPERIMENTAL on the CLEAN 400ep base (no prune). safe_divisions + prune only - ZERO synthetic nodes; the zero-synthetic control on the clean base. Expected safe_divisions ~350-500. Pinned 400ep. SUBMIT ORDER 3 of 3.
