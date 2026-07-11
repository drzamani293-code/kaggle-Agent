# Experiment Matrix

_Generated 2026-07-11 07:38 UTC from intelligence.duckdb._

Command differences, post-processing ops, submission stats, and score for every variant.
All variants share ILP weights (edge -1.0, appearance 0.1, disappearance 0.1, division 1.0, --use-ilp);
only `det_threshold` and the post-processing chain differ.

| Experiment | det | Postprocess ops | rows | nodes | edges | div+ | gap1 | gap2 | synth | prune | edges_rm | in/out | valid | fb | score |
|------------|-----|-----------------|------|-------|-------|------|------|------|-------|-------|----------|--------|-------|----|-------|
| `M16_BASELINE` | 0.99 | — | 250789 | 131797 | 118992 | 0 | 0 | 0 | 0 | 0 | 0 | 1/2 | True | False | 0.8740 |
| `M17_C_DET_0985` | 0.985 | — | 255786 | 134499 | 121287 | 0 | 0 | 0 | 0 | 0 | 0 | 1/2 | True | False | 0.8740 |
| `M18_C_EDGE_PRUNE` | 0.99 | edge_prune_and_gate | 250735 | 131797 | 118938 | 0 | 0 | 0 | 0 | 0 | 54 | 1/2 | True | False | 0.8740 |
| `M19_A_SAFE_DIVISIONS_PRUNE` | 0.99 | safe_divisions, prune_isolated | 251255 | 131797 | 119458 | 466 | 0 | 0 | 0 | 0 | 0 | 1/2 | True | False | 0.8770 |
| `M19_C_FULL_CHAIN_PENDING` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | 254824 | 133106 | 121718 | 466 | 593 | 358 | 1309 | 0 | 0 | 1/2 | True | False | 0.8800 |
| `M20_A_FULLCHAIN_TUNED` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | 254824 | 144001 | 131356 | 666 | 830 | 489 | 1808 | 0 | 0 | 1/2 | True | False | pending |
| `M20_A_FULLCHAIN_TUNED_FIXED` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | 163325 | 142199 | 0 | 0 | 0 | 0 | 0 | 0 | 1/2 | True | False | pending |
| `M21_A_PILKWANG350_M19C_GATES` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | 0.8740 |
| `M21_B_PILKWANG350_SAFE_DIV_ONLY` | 0.99 | safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M21_C_PILKWANG350_LIGHT_GAP` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M22_ARTIFACT_FORENSIC` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M22_A_TRUEBASE_SAFE_DIV_TUNE` | 0.99 | safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M22_B_TRUEBASE_LIGHT_GAP` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M22_C_TRUEBASE_DIV_PLUS_GAP1_ONLY` | 0.99 | safe_divisions, gap1, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M23_A_PILKWANG350_NODE_PRUNE_LIGHT` | 0.99 | node_prune_light, safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M23_B_PILKWANG350_NODE_PRUNE_SAFE_DIV_ONLY` | 0.99 | node_prune_light, safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | 0.8770 |
| `M23_C_PILKWANG350_EDGE_NODE_BALANCED` | 0.99 | node_prune_balanced, edge_trim, safe_divisions, gap1, linefit, prune_isolated | None | 121393 | 113068 | 406 | 536 | 0 | 536 | 0 | 0 | 1/2 | True | False | pending |
| `M24_A_400EP_FULLCHAIN_M19C_GATES` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | 0.8730 |
| `M24_B_400EP_GAP1_ONLY` | 0.99 | safe_divisions, gap1, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | 0.8720 |
| `M24_C_400EP_SAFE_DIV_ONLY` | 0.99 | safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M26_ARTIFACT_ZOO` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M26_A_CONSENSUS_2OFN_PRECISION` | 0.99 | consensus2_keep, safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M26_B_PRIMARY_M19C_STYLE_PLUS_CONSENSUS_EDGES` | 0.99 | primary_plus_consensus, safe_divisions, gap1_micro, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M26_C_WEIGHTED_ENSEMBLE_FULLCHAIN_LIGHT` | 0.99 | weighted_ensemble, safe_divisions, gap1, gap2_light, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M27_A_PUBLIC_REPRO_EXACT_SAFETY` | 0.99 | motion_relink, gap_close, safe_divisions, edge_policy_veto, filter_short_tracks_minlen7, linefit, final_safety_repair | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | 0.8590 |
| `M27_B_PUBLIC_REPRO_NO_EDGE_VETO` | 0.99 | motion_relink, gap_close, safe_divisions, filter_short_tracks_minlen7, linefit, final_safety_repair | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M27_C_PUBLIC_REPRO_MINLEN5` | 0.99 | motion_relink, gap_close, safe_divisions, edge_policy_veto, filter_short_tracks_minlen5, linefit, final_safety_repair | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M27_D_PUBLIC_REPRO_STRICT_PRECISION` | 0.99 | motion_relink, gap_close, safe_divisions, edge_policy_veto, filter_short_tracks_minlen8_keepdiv, linefit, final_safety_repair | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M25_A_PRUNE_MILD_SAFE_DIV` | 0.99 | node_prune_mild, safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M25_B_PRUNE_STRONG_SAFE_DIV` | 0.99 | node_prune_strong, safe_divisions, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M25_C_PRUNE_M23B_PLUS_MICRO_GAP1` | 0.99 | node_prune_m23b, safe_divisions, gap1_micro, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M28_A_M19C_GAP_CAPS_OPEN` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M28_B_M19C_GAP3_ADDED` | 0.99 | safe_divisions, gap1, gap2, gap3, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M28_C_DIVISION_CAP_OPEN` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M28_D_GAP_OPEN_PLUS_DIV_OPEN_NO_LINEFIT` | 0.99 | safe_divisions, gap1, gap2, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M28_E_DET095_M19C_SAFE` | 0.95 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M29_A_WINNING_ENGINE_DEFAULT` | 0.99 | build_base_graph, rescue_divisions, stitch_gaps_1_2_3, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M29_B_ENGINE_GAP12_ONLY` | 0.99 | build_base_graph, rescue_divisions, stitch_gaps_1_2, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M29_C_ENGINE_LINEFIT_ON` | 0.99 | build_base_graph, rescue_divisions, stitch_gaps_1_2_3, linefit, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M29_D_ENGINE_RELINK_ON` | 0.99 | build_base_graph, motion_relink, rescue_divisions, stitch_gaps_1_2_3, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M29_E_ENGINE_DET095_SAFE` | 0.95 | build_base_graph, rescue_divisions, stitch_gaps_1_2_3, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M30_A_V2_FULL_CANDIDATES_BALANCED` | 0.99 | read_candidate_geff, two_stage_hungarian_relink, stitch_gaps, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M30_B_V2_FULL_CANDIDATES_GAP123` | 0.99 | read_candidate_geff, two_stage_hungarian_relink, stitch_gaps, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M30_CANDIDATE_DIAGNOSTIC` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M30_C_V2_AUTO_SPARSE_KNN_TIGHT` | 0.99 | read_candidate_geff, two_stage_hungarian_relink, stitch_gaps, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M30_D_V2_AUTO_DET095` | 0.95 | read_candidate_geff, two_stage_hungarian_relink, stitch_gaps, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M30_E_V2_DIVISION_RELAXED` | 0.99 | read_candidate_geff, two_stage_hungarian_relink, stitch_gaps, prune_isolated, relabel | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_A_EXACT_09_REPRO` | 0.97 | tta6, motion_relink, gap1, safe_divisions, short_track_min6, linefit0.8, prune | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_B_DEEPCENTER_SHADOW` | 0.97 | deepcenter_shadow | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_C_DEEPCENTER_GATE_BASE_CAPS` | 0.97 | tta6, motion_relink, gap1, deepcenter_gate, safe_divisions, short_track_min6, linefit0.8, prune | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_D_DEEPCENTER_GATE_GAP2_OPEN` | 0.97 | tta6, motion_relink, gap1, gap2, deepcenter_gate, safe_divisions, short_track_min6, linefit0.8, prune | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_E_DEEPCENTER_GATE_DIVISION_RELAXED` | 0.97 | tta6, motion_relink, gap1, deepcenter_gate, safe_divisions_relaxed, short_track_min6, linefit0.8, prune | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_LOCAL_CV_HARNESS` | 0.97 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M31_REFERENCE_AUDIT` | 0.97 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M32_1_A_OFFICIAL_METRIC_VENDOR_AUDIT` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M32_1_B_OFFICIAL_CV_REAUDIT` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M32_A_OFFICIAL_CV_AUDIT` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M32_B_OFFICIAL_CV_REPLAY` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M32_C_OFFICIAL_SMALL_SWEEP` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M32_D_DET_THRESHOLD_PILOT` | 0.99 | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M33_A_ENSEMBLE_INPUT_AUDIT` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M33_B_CORRECTED_OUTPUT_BLEND_DIAG` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M33_C_CORRECTED_OUTPUT_BLEND_CANDIDATE` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M33_D_PROBABILITY_FUSION_AUDIT` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |
| `M33_E_CORRECTED_PROBABILITY_ENSEMBLE` | None | — | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |

Legend: div+ = safe divisions added, gap1 = single-frame gaps closed, 
gap2 = two-frame gaps recovered, synth = synthetic nodes added, prune = isolated nodes pruned, 
edges_rm = edges removed, in/out = max in/out degree, fb = fallback_used.
