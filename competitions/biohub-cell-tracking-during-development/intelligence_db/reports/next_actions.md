# Next Actions

_Generated 2026-07-07 22:25 UTC from intelligence.duckdb._

**Best scored experiment:** `M19_A_SAFE_DIVISIONS_PRUNE` at **0.8770**.
**M19-C full_chain:** pending (score pending).

## Recommendation

M19-C is **pending**. Do not spend the second remaining submission until its public score is known.

- **Hold** the second submission.
- When M19-C scores, this report auto-updates with the branch below.
- Interim best to keep as the submitted baseline: `M19_A_SAFE_DIVISIONS_PRUNE` (0.8770).

### Contingency (what each M19-C outcome will trigger)
- **M19-C > 0.877** → tune `full_chain` gates: widen safe-division sister/parent gates slightly, 
  and tune gap1/gap2 distance + velocity gates to add more valid synthetic chains without 
  inflating the node penalty.
- **M19-C == 0.877** → the gap recovery neither helped nor hurt net; isolate it by running 
  **M19-B** (gap recovery alone, no divisions) or tune the safe-division gates further, since 
  divisions are the proven lever.
- **M19-C < 0.877** → the 1309 synthetic nodes cost more (node over-prediction penalty) than the 
  recovered edges gained; **revert to M19-A** and tighten or remove gap recovery (drop gap2 first, 
  then gap1), keeping only safe divisions + isolated prune.

## Recorded decisions (history)

- **after `M18_C_EDGE_PRUNE`** (2026-07-02, risk low):
  - observation: det-threshold (0.99/0.985/0.995/0.975) and a small post-ILP edge prune all score 0.874. The predict command is a flat lever; the ILP re-optimizes to the same graph.
  - recommendation: Stop tuning the predict command. Move to post-GEFF metric-aware processing targeting the node over-prediction penalty and the 0.1-weighted division term.  → next: `M19_A_SAFE_DIVISIONS_PRUNE`
- **after `M19_A_SAFE_DIVISIONS_PRUNE`** (2026-07-05, risk medium):
  - observation: Safe divisions + isolated prune moved 0.874 -> 0.877 with zero synthetic nodes. Metric-aware post-processing is the correct path.
  - recommendation: Spend the next submission on M19-C full_chain (adds gap recovery + smoothing on the same division core). Gate on valid=True and fallback_used=False before submitting.  → next: `M19_C_FULL_CHAIN_PENDING`
