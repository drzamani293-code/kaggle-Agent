# Next Actions

_Generated 2026-07-10 22:31 UTC from intelligence.duckdb._

**Best scored experiment:** `M19_C_FULL_CHAIN_PENDING` at **0.8800**.
**M19-C full_chain:** final (score 0.8800).
**Open (pending):** 19 · **blocked:** 16 · **unsafe/superseded:** 2.

## Recommendation

**TTA6 + DeepCenter REFERENCE-FIRST reproduction (M31) - run the reference audit FIRST.** The
analysis_09_strategy.md near-0.900 pipeline (400ep, 6-way TTA, det 0.97, div 0.7, pool ~2.0um,
learned motion_relink, one-frame gap, safe divisions, min_track_len 6, linefit 0.8, DeepCenter
loaded-but-inactive) is reproduced ONLY from values RESOLVED at runtime from the mounted source -
nothing is invented, and M27/M28/M29/M30 logic is never substituted.

1. **Run `M31_REFERENCE_AUDIT_NOT_SUBMIT` first (non-submit).** It resolves every reference field
   with provenance; `AUDIT_PASS` only if the source-only items (TTA flag, pool-kernel flag,
   motion_relink function, reference notebook) are OBSERVED, else `REFERENCE_CONFIG_UNRESOLVED`.
2. **`M31_A_EXACT_09_REPRO`** - submit only if `reference_config_resolved`, TTA count = 6,
   `reference_profile_pass`, and `OK_TO_SUBMIT_EXPERIMENTAL`. If the reference is not resolvable it
   reports `DO_NOT_SUBMIT_REFERENCE_CONFIG_UNRESOLVED` and writes a hidden-safe fallback (no
   substitute pipeline). Motion-relink accounting is an exact edge-set diff (`raw_replaced` == post
   edges is NOT a replacement count).
3. **`M31_B_DEEPCENTER_SHADOW_NOT_SUBMIT`** (non-submit) must prove DeepCenter is loaded, scores
   every gap/division candidate (checked > 0), and produces output byte-identical to the baseline.
   Do **not** run C/D/E until B passes.
4. Then `M31_C_DEEPCENTER_GATE_BASE_CAPS` (add-only gate) -> `M31_D` (gap2) / `M31_E` (relaxed
   divisions) as isolated ablations. `M31_LOCAL_CV_HARNESS_NOT_SUBMIT` scores A/C/D/E on train GT
   via the official metric (never fabricated).

**Never submit the audit, B shadow, or CV harness.** final_source
`tta6_motion_relink_reference_reproduced` (A) / `tta6_deepcenter_gate_*` (C/D/E).
**Keep `M19_C_FULL_CHAIN_PENDING` at 0.8800 as final** unless an M31 variant beats 0.880.
M29-A / M30-C remain pending on Kaggle; M30 diagnostic gates that family.

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
- **after `M26_A_CONSENSUS_2OFN_PRECISION`** (2026-07-09, risk medium):
  - observation: A public high-score notebook (biohub-cell-tracking-v4-unet-ilp-reproduction) uses the SAME 400ep artifact we tested in M24 (weight_sha256 12f6881e.., base 127790/115694) but scores much higher by replacing our thin safe_div/gap/linefit postprocess with a richer output pipeline: motion-relink (per-frame Hungarian), gap close with reuse, safe divisions, a conservative logistic edge VETO capped at 1%, short-track-component filtering (keep divisions), and linefit. Its public output is 119763 node rows / 115080 edge rows. M24 (0.872/0.873) failed on 400ep only because our postprocess was too thin.
  - recommendation: Reproduce the public notebook LOGIC hidden-safe as M27 (never submit its static submission.csv). Guard the 400ep artifact (name contains 400ep + exact weight_sha256) and add a final safety repair. Submit order A (exact repro) -> C (min_track_len 5) -> B (no edge veto) -> D (strict min_track_len 8), each only if its report prints OK_TO_SUBMIT_EXPERIMENTAL (or the variant-specific count rule). Keep M19-C 0.880 final unless an M27 variant beats it.  → next: `M27_A -> M27_C -> M27_B -> M27_D`
- **after `M24_A_400EP_FULLCHAIN_M19C_GATES`** (2026-07-10, risk medium):
  - observation: The 400ep path FAILED: M24-A 0.873, M24-B 0.872 - both below M19-C 0.880 and below M21-A 0.874. The cleaner 400ep base did not translate into a better score. The strongest experimental path remains M23-B (pilkwang350 node-prune + safe-div, 0.877), only 0.003 behind M19-C.
  - recommendation: Deprioritize 400ep. Tune around M23-B on the pinned pilkwang350 base (M25): A milder prune, B stronger prune (both safe-div-only, zero synthetic), C M23-B repair + a very-light gap1. Submit order A -> B -> C. Keep M19-C 0.880 final unless an M25 score beats it.  → next: `M25_A -> M25_B -> M25_C`
- **after `M25_A_PRUNE_MILD_SAFE_DIV`** (2026-07-10, risk low):
  - observation: M25 could not run: the required pilkwang350 350ep artifact (artifact_name biohub-tracking-support-pack-350ep-snapshot-v1, weight_sha256 dfb848..) is no longer recoverable/attachable - the mounted pack is now the 400ep snapshot (12f688..), so the M25 artifact guard failed (DO_NOT_SUBMIT_WRONG_ARTIFACT). Across M21 (0.874), M23 (0.877), M24 (0.873/0.872), no experimental path beat M19-C 0.880, and the true M19-C artifact is unrecoverable.
  - recommendation: FINAL: stop experimental submissions unless the exact 350ep/dfb848 artifact OR the true M19-C artifact is recovered. The final recommended submission is M19-C version 12 (Biohub5-notebook015ca8d31a - version 12), public score 0.880 - the strongest and safest final candidate.  → next: `KEEP_M19_C_FINAL_0880`
- **after `M27_A_PUBLIC_REPRO_EXACT_SAFETY`** (2026-07-10, risk medium):
  - observation: M27-A (public-notebook reproduction) scored 0.859 < M19-C 0.880 - its pruning-heavy output pipeline (short-track filtering + edge veto) over-prunes / lowers node recall on our base. The uploaded winning strategy argues the opposite problem for M19-C: our post-processing is too CONSERVATIVE (gap caps too small, division cap too small, det-threshold 0.99 possibly too high) and recall-oriented recovery is the path beyond 0.880.
  - recommendation: Stop the pruning-heavy M27 path. Pivot back to the proven M19-C full_chain and OPEN recall levers one at a time (M28): A gap caps open, C division caps open, D both + no linefit, B add strict gap3, E det 0.95. Submit order A -> C -> D -> B -> E, each only if its report prints OK_TO_SUBMIT_EXPERIMENTAL. Keep M19-C 0.880 final unless an M28 variant beats it. Build a local CV before deeper LB tuning.  → next: `M28_A -> M28_C -> M28_D -> M28_B -> M28_E`
- **after `M28_A_M19C_GAP_CAPS_OPEN`** (2026-07-10, risk medium):
  - observation: M28 was only a PARTIAL recall expansion (M28-A: nodes=132166, gap1=1604, gap2=1386, synthetic=4376). The uploaded winning-engine package (winning_postprocess.py / integration_cell.py) is a FULLER recall engine: motion relink + slot-Hungarian division rescue + per-timepoint Hungarian gap 1/2/3 stitching (constant velocity) + optional linefit + prune. Its submission(3).csv profiles to nodes=139418, edges=132871, divisions~2382, all invariants clean. M27 pruning path already failed (0.859).
  - recommendation: Integrate the full engine as M29, EMBEDDED self-contained (no external utility-dataset import) so each runner is a standalone Kaggle cell. Submit order A (default) -> B (gap12 only) -> C (linefit) -> D (relink) -> E (det 0.95), each only if its report prints OK_TO_SUBMIT_EXPERIMENTAL. If A>0.880 make it the new candidate and continue B/C; if A in [0.875,0.880) test B and C; if A<0.875 test B only, then PAUSE and run the local-CV harness before more submits. Keep M19-C version 12 0.880 final until beaten. Do not submit the uploaded submission(3).csv.  → next: `M29_A -> M29_B -> M29_C -> M29_D -> M29_E (then local_cv_harness)`
- **after `M29_A_WINNING_ENGINE_DEFAULT`** (2026-07-10, risk medium):
  - observation: A new v2 package (winning_postprocess_v2.py / integration_v2.py) RE-SOLVES unit-timepoint linking from the RAW GEFF candidate graph rather than patching the ILP solution (v1/M29). CRITICAL: the reused read_geff_graph applies the edge solution mask and returns only the ILP solution, so v2 needs read_candidate_geff (no mask). Whether v2 can help depends entirely on candidate richness - a full candidate graph enables global relinking, an ILP-solution-like store does not. M29-A is still pending on Kaggle; M30 is built in parallel.
  - recommendation: Integrate v2 as M30, embedded self-contained. RUN THE STAGE-0 CANDIDATE DIAGNOSTIC FIRST to classify the graph. If FULL_CANDIDATE_GRAPH submit A -> B -> E -> D; if ILP_SOLUTION_LIKE/MODERATELY_SPARSE submit C -> D -> E (skip A/B). Every submit requires edge_prob present+non-degenerate, the candidate mode matching the Stage-0 class, and OK_TO_SUBMIT_EXPERIMENTAL. Never submit the diagnostic or the v2 CV harness. Keep M19-C 0.880 final; keep M29-A pending until the user supplies its score.  → next: `M30_CANDIDATE_DIAGNOSTIC -> (FULL: A,B,E,D | SPARSE: C,D,E) + v2 CV harness`
- **after `M30_A_V2_FULL_CANDIDATES_BALANCED`** (2026-07-10, risk medium):
  - observation: analysis_09_strategy.md describes a near-0.900 TTA6 + learned-motion_relink + DeepCenter pipeline on 400ep. The exact reference (TTA transforms, pool-kernel flag, motion_relink params, DeepCenter model class/weights/call signature) is NOT locatable in this environment. The critical rule forbids inventing any of it or substituting M27/M28/M29/M30 logic.
  - recommendation: Build M31 REFERENCE-FIRST: resolve every field from mounted source with provenance (grounded in the actual predict script via discover_predict_flags), and REFUSE (REFERENCE_CONFIG_UNRESOLVED / DO_NOT_SUBMIT_*) rather than approximate. Operational order: audit -> A exact repro -> B DeepCenter shadow (must prove loaded+checked>0+shadow==baseline) -> C gate -> D gap2 / E division (isolated ablations) -> CV. Never submit audit/B/CV. Keep M19-C 0.880 final; M29-A and M30-C pending.  → next: `M31_REFERENCE_AUDIT -> A -> B(shadow) -> C -> D/E -> CV`
