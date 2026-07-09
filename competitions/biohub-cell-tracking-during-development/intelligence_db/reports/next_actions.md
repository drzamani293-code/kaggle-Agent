# Next Actions

_Generated 2026-07-09 15:56 UTC from intelligence.duckdb._

**Best scored experiment:** `M19_C_FULL_CHAIN_PENDING` at **0.8800**.
**M19-C full_chain:** final (score 0.8800).
**Open (pending):** 3 · **blocked:** 10 · **unsafe/superseded:** 2.

## Recommendation

**Pivot to multi-artifact ENSEMBLING (M26).** Single-model post-processing has plateaued at
**M19-C 0.880** - every substitute base (pilkwang350 142193, 400ep 127790, tom99763 161098)
scored below it and the true M19-C artifact is unrecoverable. The remaining source of signal is
cross-model **consensus**: run several valid artifacts on the same hidden-safe split and combine
their tracking graphs.

1. **Run Stage 1 first: `M26_ARTIFACT_ZOO_DIAGNOSTIC`** (non-scoring). It enumerates every mounted
   artifact, records artifact_name + weight_sha256 + known-artifact match, and smoke-runs each to
   report its base GEFF counts. **Only proceed to a submission if it finds >=2 viable artifacts.**
2. If >=2 artifacts, run the ensemble variants in submit order, each only if its report prints
   `OK_TO_SUBMIT_EXPERIMENTAL` (>=2 artifacts, valid, no fallback, ensemble_graph_postprocessed,
   120000<=nodes<=145000, 110000<=edges<=135000, synthetic within the per-variant cap):
   1. `M26_A_CONSENSUS_2OFN_PRECISION` (2-of-N consensus, ZERO synthetic - tests whether consensus fixes node over-prediction)
   2. `M26_B_PRIMARY_M19C_STYLE_PLUS_CONSENSUS_EDGES` (primary = base closest to 131797/118992 + consensus edges; micro gap1; synthetic<=600)
   3. `M26_C_WEIGHTED_ENSEMBLE_FULLCHAIN_LIGHT` (weighted ensemble + full chain with a very light gap2; synthetic<=1000)

Each variant still writes a valid hidden-safe `submission.csv`; if <2 artifacts are found it reports
`DO_NOT_SUBMIT_NOT_ENOUGH_ARTIFACTS` and must not be submitted.
**Keep `M19_C_FULL_CHAIN_PENDING` at 0.8800 as final** unless an M26 variant beats 0.880.

## Recorded decisions (history)

- **after `M18_C_EDGE_PRUNE`** (2026-07-02, risk low):
  - observation: det-threshold (0.99/0.985/0.995/0.975) and a small post-ILP edge prune all score 0.874. The predict command is a flat lever; the ILP re-optimizes to the same graph.
  - recommendation: Stop tuning the predict command. Move to post-GEFF metric-aware processing targeting the node over-prediction penalty and the 0.1-weighted division term.  → next: `M19_A_SAFE_DIVISIONS_PRUNE`
- **after `M19_A_SAFE_DIVISIONS_PRUNE`** (2026-07-05, risk medium):
  - observation: Safe divisions + isolated prune moved 0.874 -> 0.877 with zero synthetic nodes. Metric-aware post-processing is the correct path.
  - recommendation: Spend the next submission on M19-C full_chain (adds gap recovery + smoothing on the same division core). Gate on valid=True and fallback_used=False before submitting.  → next: `M19_C_FULL_CHAIN_PENDING`
- **after `M19_C_FULL_CHAIN_PENDING`** (2026-07-08, risk medium):
  - observation: M19-C full_chain scored 0.880 (> M19-A 0.877). Gap recovery + line-fit smoothing add value on top of divisions, and 1309 synthetic nodes did not trip the node over-prediction penalty. full_chain is the new best.
  - recommendation: Tune the full_chain gates: slightly widen the safe-division sister/parent-child gates, and sweep the gap1/gap2 distance + velocity gates to admit more metric-valid synthetic chains while watching synthetic_nodes_added vs score to stay ahead of the node penalty.  → next: `M20_A_FULLCHAIN_TUNED`
- **after `M20_A_FULLCHAIN_TUNED`** (2026-07-08, risk medium):
  - observation: M20's pre-postprocess base learned graph changed vs M19-C (n_nodes_before 131797->142193, n_edges_before 118992->127563) despite an identical predict command - a different support-pack/weights artifact was selected on Kaggle. The gate-tuning result is not trustworthy or submittable.
  - recommendation: Use M20_A_FULLCHAIN_TUNED_FIXED, which reproduces M19-C's exact artifact/weights selection, logs the selected artifact/repo/weights/support-pack, and adds a hard baseline guard (tolerance 0) that recommends DO_NOT_SUBMIT_BASELINE_MISMATCH if the base differs. Attach ONLY the canonical 50ep-v1 support pack.  → next: `M20_A_FULLCHAIN_TUNED_FIXED`
- **after `M20_A_FULLCHAIN_TUNED_FIXED`** (2026-07-08, risk low):
  - observation: Both available support packs failed the M19-C baseline guard: pilkwang -> n_nodes_before 142193 / n_edges_before 127563, tom99763 (artifact biohub-tracking-support-pack-5090-50ep-v1) -> 161098 / 137520, vs M19-C 131797 / 118992. Neither reproduces the base learned graph that scored 0.880, so M20 cannot be safely submitted.
  - recommendation: Do NOT submit M20 (either pack). Keep M19-C at 0.880 as the best/final candidate. The tuned full_chain gains are unverifiable until the TRUE M19-C baseline support-pack artifact (the exact pack/weights yielding n_nodes_before=131797, n_edges_before=118992) is recovered; only then re-run the guarded M20. If that artifact cannot be recovered, M19-C 0.880 is final.  → next: `KEEP_M19_C_0880`
- **after `M21_A_PILKWANG350_M19C_GATES`** (2026-07-08, risk low):
  - observation: M21-A (pilkwang drift base + proven M19-C gates) scored only 0.874 - equal to baseline, below M19-C 0.880. The newer/larger pilkwang350 base is NOT automatically better; post-processing on a drifted base does not recover the M19-C result.
  - recommendation: Stop tuning on the drift base. Primary next action: recover the TRUE M19-C artifact (base 131797/118992), likely a NOTEBOOK input from the original M19-C run (e.g. a kernel named like 'Biohub Cell Tracking: Learned Graph w G'). Run M22_ARTIFACT_FORENSIC_DIAGNOSTIC to find it, then run the true-base M22 variants (A->B->C), each gated on OK_TO_SUBMIT_TRUEBASE_EXPERIMENT. Keep M19-C 0.880 final until a true-base score beats it.  → next: `M22_ARTIFACT_FORENSIC -> M22_A/B/C`
- **after `M22_ARTIFACT_FORENSIC`** (2026-07-09, risk medium):
  - observation: M22 forensic returned TRUE_M19C_ARTIFACT_NOT_FOUND: the Notebook input had no usable repo/weights and no mounted pack reproduces the true M19-C base (131797/118992). The true artifact is unrecoverable for now.
  - recommendation: Since the true base cannot be recovered, attack the pilkwang350 base's node over-prediction directly: run M23 node-penalty repair (prune low-value detections, preserve divisions+long tracks) then conservative post-processing. Submit order B -> C -> A (test node-penalty reduction with ZERO synthetic nodes first, since M21-A's synthetic-heavy full_chain scored only 0.874). Keep M19-C 0.880 final unless an M23 score beats it.  → next: `M23_B -> M23_C -> M23_A`
- **after `M23_B_PILKWANG350_NODE_PRUNE_SAFE_DIV_ONLY`** (2026-07-09, risk medium):
  - observation: M23-B (pilkwang350 node-prune + safe divisions) scored 0.877 - node-penalty reduction helped (0.874->0.877) but did not beat M19-C 0.880. Separately, M23-C's guard failure REVEALED a new 400ep artifact (biohub-tracking-support-pack-400ep-snapshot-v1, weight 12f688..) whose RAW base 127790/115694 is much closer to M19-C's true 131797/118992 than pilkwang350's 142193/127563.
  - recommendation: Pivot to the cleaner 400ep base. Run M24 (pinned to the 400ep path+name+weight hash, NO node-prune): A full_chain (highest upside), B gap1-only, C safe-div-only. Submit order A -> B -> C. Keep M19-C 0.880 final unless an M24 score beats it.  → next: `M24_A -> M24_B -> M24_C`
- **after `M25_A_PRUNE_MILD_SAFE_DIV`** (2026-07-09, risk medium):
  - observation: Single-model post-processing plateaued at M19-C 0.880; every substitute base (pilkwang350 142193, 400ep 127790, tom99763 161098) scored below 0.880, and the true M19-C artifact is unrecoverable. Post-GEFF tuning on ONE model has run out of signal.
  - recommendation: Pivot to MULTI-ARTIFACT ENSEMBLING (M26). Run Stage 1 M26_ARTIFACT_ZOO_DIAGNOSTIC first to confirm >=2 viable artifacts and read their base counts. If >=2, run the ensemble variants in submit order A (2-of-N consensus, zero synthetic) -> B (primary-closest-to-target + consensus edges, micro gap1) -> C (weighted ensemble + light full chain), each only if its report prints OK_TO_SUBMIT_EXPERIMENTAL. Keep M19-C 0.880 as final/best unless an M26 variant beats it.  → next: `M26_ARTIFACT_ZOO -> M26_A -> M26_B -> M26_C`
- **after `M24_A_400EP_FULLCHAIN_M19C_GATES`** (2026-07-10, risk medium):
  - observation: The 400ep path FAILED: M24-A 0.873, M24-B 0.872 - both below M19-C 0.880 and below M21-A 0.874. The cleaner 400ep base did not translate into a better score. The strongest experimental path remains M23-B (pilkwang350 node-prune + safe-div, 0.877), only 0.003 behind M19-C.
  - recommendation: Deprioritize 400ep. Tune around M23-B on the pinned pilkwang350 base (M25): A milder prune, B stronger prune (both safe-div-only, zero synthetic), C M23-B repair + a very-light gap1. Submit order A -> B -> C. Keep M19-C 0.880 final unless an M25 score beats it.  → next: `M25_A -> M25_B -> M25_C`
- **after `M25_A_PRUNE_MILD_SAFE_DIV`** (2026-07-10, risk low):
  - observation: M25 could not run: the required pilkwang350 350ep artifact (artifact_name biohub-tracking-support-pack-350ep-snapshot-v1, weight_sha256 dfb848..) is no longer recoverable/attachable - the mounted pack is now the 400ep snapshot (12f688..), so the M25 artifact guard failed (DO_NOT_SUBMIT_WRONG_ARTIFACT). Across M21 (0.874), M23 (0.877), M24 (0.873/0.872), no experimental path beat M19-C 0.880, and the true M19-C artifact is unrecoverable.
  - recommendation: FINAL: stop experimental submissions unless the exact 350ep/dfb848 artifact OR the true M19-C artifact is recovered. The final recommended submission is M19-C version 12 (Biohub5-notebook015ca8d31a - version 12), public score 0.880 - the strongest and safest final candidate.  → next: `KEEP_M19_C_FINAL_0880`
