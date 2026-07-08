# Experiment Matrix

_Generated 2026-07-08 13:11 UTC from intelligence.duckdb._

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
| `M20_A_FULLCHAIN_TUNED_FIXED` | 0.99 | safe_divisions, gap1, gap2, linefit, prune_isolated | None | None | None | 0 | 0 | 0 | 0 | 0 | 0 | — | None | None | pending |

Legend: div+ = safe divisions added, gap1 = single-frame gaps closed, 
gap2 = two-frame gaps recovered, synth = synthetic nodes added, prune = isolated nodes pruned, 
edges_rm = edges removed, in/out = max in/out degree, fb = fallback_used.
