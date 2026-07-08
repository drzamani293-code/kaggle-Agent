# Next Actions

_Generated 2026-07-08 20:31 UTC from intelligence.duckdb._

**Best scored experiment:** `M19_C_FULL_CHAIN_PENDING` at **0.8800**.
**M19-C full_chain:** scored (score 0.8800).
**Open (pending):** 0 · **blocked:** 5 · **unsafe/superseded:** 2.

## Recommendation

**Primary next action: recover the TRUE M19-C artifact** (base `n_nodes_before=131797`,
`n_edges_before=118992`). It is likely a **NOTEBOOK input** from the original M19-C run
(e.g. a kernel named like *"Biohub Cell Tracking: Learned Graph w G"*), not one of the two
dataset support packs (pilkwang350 -> 142193, tom99763 -> 161098; both fail the base check).

1. Run **`M22_ARTIFACT_FORENSIC_DIAGNOSTIC`** on Kaggle with ALL candidate inputs attached
   (datasets AND notebooks). It enumerates every artifact root, records artifact_name +
   weight_sha256, and smoke-runs each non-bad pack until the base equals 131797 / 118992.
2. If it prints **TRUE_M19C_ARTIFACT_FOUND**, run the true-base M22 variants in this order,
   each gated on `OK_TO_SUBMIT_TRUEBASE_EXPERIMENT` (true-artifact + public-base-count guards):
   1. `M22_A_TRUEBASE_SAFE_DIV_TUNE`
   2. `M22_B_TRUEBASE_LIGHT_GAP`
   3. `M22_C_TRUEBASE_DIV_PLUS_GAP1_ONLY`
3. If it prints **TRUE_M19C_ARTIFACT_NOT_FOUND**, attach the original M19-C Notebook input and
   re-run the forensic. Do NOT create submit-ready runs until the true base is confirmed.

Do **NOT** spend submissions on pilkwang350 drift variants: M21-A already scored 0.874 (< 0.880).
**Keep `M19_C_FULL_CHAIN_PENDING` at 0.8800 as final** until a true-base score beats it.

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
