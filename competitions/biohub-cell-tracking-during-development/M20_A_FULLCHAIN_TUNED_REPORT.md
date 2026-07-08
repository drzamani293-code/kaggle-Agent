# M20_A_FULLCHAIN_TUNED — candidate report

**One primary candidate.** Tunes the M19-C `full_chain` post-processing gates
only. The learned-graph predict command is **byte-for-byte identical to
M16/M19-C** (det 0.99, ilp-edge −1.0, ilp-appearance 0.1, ilp-disappearance
0.1, ilp-division 1.0, `--use-ilp`). Nothing else changes: install/`--no-deps`,
sitecustomize Float16 patch, PYTHONPATH, dynamic list-based splits,
hidden-safe validation, fallback, GEFF conversion basics, submission schema,
Kaggle-API behavior. No motion-relink. No private data.

## Why tune (not revert)

M19-C scored **0.880** (M19-A 0.877 → +0.003; baseline 0.874 → +0.006). Its
1309 velocity/distance-gated synthetic gap nodes **did not** trip the node
over-prediction penalty — they improved the score. So the right move is to
admit *moderately more* valid structure by widening the gates, not to drop gap
recovery (M19-B). Widening is kept small (~7–15%) to stay well short of an
explosion of synthetic nodes.

## Exact changed gates vs M19-C

| Gate | M19-C | M20 | Δ | Rationale |
|------|-------|-----|---|-----------|
| `DIV_PARENT_CHILD_MAX_UM` | 4.7 | **5.1** | +8.5% | admit slightly more true second children (division term is a proven lever) |
| `DIV_SISTER_MAX_UM` | 6.85 | **7.4** | +8% | allow a marginally wider sister separation |
| `DIV_EXISTING_CHILD_MAX_UM` | 7.45 | **7.9** | +6% | let a slightly-longer existing edge still qualify as a divider |
| `DIV_FRAME_CAP_FRAC` | 0.0072 | **0.0090** | +25% | raise per-frame division density cap so widened gates aren't clipped |
| `DIV_GLOBAL_CAP_FRAC` | 0.004 | **0.0052** | +30% | raise per-dataset division cap (M19-C added 466, near the old cap) |
| `GAP1_MAX_TOTAL_UM` | 6.2 | **7.0** | +13% | gap1 adds only 1 synthetic node — safest recall lever; widen most here |
| `GAP1_CAP_FRAC` | 0.006 | **0.0075** | +25% | raise gap1 density cap |
| `GAP1_CAP_ABS` | 300 | **380** | +27% | raise gap1 absolute cap |
| `GAP2_MAX_STEP_UM` | 4.4 | **4.7** | +7% | modest per-step widening (gap2 adds 2 nodes — keep conservative) |
| `GAP2_MAX_TOTAL_UM` | 10.2 | **11.0** | +8% | modest total-distance widening |
| `GAP2_VELOCITY_NORMDIFF_UM` | 6.0 | **6.6** | +10% | allow marginally more velocity-magnitude mismatch |
| `GAP2_CAP_FRAC` | 0.0045 | **0.0052** | +16% | small gap2 density-cap raise |
| `GAP2_CAP_ABS` | 180 | **220** | +22% | small gap2 absolute-cap raise |

**Deliberately unchanged:** `GAP2_VELOCITY_COS_MIN` = −0.25 (direction gate
held — do not loosen how much a track may reverse); `linefit` window 2 /
weight 0.72 (topology-preserving, enabled); `prune_isolated` max_frac 0.6
(enabled); the safety floor of 1 on every cap.

## Why each change should improve the score

- **Divisions** feed the `0.1·division_jaccard` term directly. M19-A/M19-C
  proved recovering true divisions raises the score; a small widening should
  recover a few more without inventing false ones (gates stay geometric and
  require the existing child exactly one frame ahead, so every sister edge is
  unit-timepoint and metric-valid).
- **gap1** bridges a single missing frame with one synthetic node and two
  unit edges. It is the highest-value / lowest-risk recall lever (1 node per
  recovery), so it gets the largest relative widening.
- **gap2** bridges two missing frames (2 synthetic nodes). It is widened the
  least, and its direction gate is untouched, to control synthetic-node growth
  and keep the node penalty in check.

## Expected effects (estimates — actual depends on real GEFF geometry)

| Metric | M19-C | M20 expected | Direction |
|--------|-------|--------------|-----------|
| `safe_divisions_added` | 466 | **~600–750** | ↑ moderate |
| `gap1_closed` | 593 | **~680–820** | ↑ moderate |
| `gap2_recovered` | 358 | **~410–490** | ↑ small–moderate |
| `synthetic_nodes_added` (= gap1 + 2·gap2) | 1309 | **~1500–1800** | ↑ (within 1500–2200 guidance) |
| `n_nodes` | 133106 | **~133,300–133,600** | ↑ by synthetic nodes |
| `n_edges` (= base + div + 2·gap1 + 3·gap2) | 121718 | **~122,200–122,900** | ↑ |
| `max_in_degree` / `max_out_degree` | 1 / 2 | **1 / 2** | unchanged (invariant) |

These are risk guidance, not targets to force: if the real gates admit fewer,
that is the correct conservative outcome and still ≥ M19-C structurally.

## Risk assessment

- **Low–medium.** All widenings are ≤~15% on distance gates; cap raises only
  prevent the widened gates from being clipped. The node-penalty risk (more
  synthetic nodes) is bounded: gap2 (2 nodes each) is widened least, and the
  velocity direction gate is unchanged, so added nodes stay on plausible
  trajectories and remain matchable to GT within the 7 µm tolerance.
- **Topology is invariant by construction:** every added edge is
  unit-timepoint; divisions never create multi-parent or a third child;
  gaps insert synthetic intermediates (never a direct multi-frame edge);
  linefit only moves coordinates; pruning only removes degree-0 nodes.
  `max_in_degree ≤ 1`, `max_out_degree ≤ 2` always hold.
- **Downside if it regresses:** if 0.880 → lower, the extra synthetic nodes
  cost more than the recovered edges; the fallback is simply to keep M19-C
  (0.880) as the standing submission and, if a further attempt is available,
  dial `GAP2` back toward M19-C values first.

## Submit / no-submit criteria

Submit **only if all** hold (read from the on-Kaggle diagnostic cell):

- `valid == True`
- `fallback_used == False`
- `final_source == "reference_learned_graph_postprocessed"`
- `max_in_degree <= 1` and `max_out_degree <= 2`
- `/kaggle/working/submission.csv` exists, no NaN, `id` column consecutive
- diagnostics are sane vs guidance: `synthetic_nodes_added` roughly in
  1500–2200 (a much larger number signals the gates over-fired → do **not**
  submit; re-tighten), and divisions/gap1/gap2 each moderately above M19-C.

If any gate fails, **do not submit** — keep M19-C (0.880) as the best
confirmed submission. This candidate is only built and verified here; it is
run manually in Kaggle and reviewed before any submission decision.
