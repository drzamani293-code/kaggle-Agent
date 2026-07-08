# M20_A_FULLCHAIN_TUNED_FIXED — candidate report

Replaces the **unsafe** `M20_A_FULLCHAIN_TUNED`. Same tuned post-processing
gates, but adds a **hard baseline guard** so a drifted base graph can never be
recommended for submission.

## What went wrong with the first M20

M20 was meant to differ from M19-C **only** in post-processing gates. But its
diagnostics showed the **pre-post-processing base graph changed**:

| | M19-C (0.880) | M20 bad run |
|---|---|---|
| `n_nodes_before` | **131797** | 142193 |
| `n_edges_before` | **118992** | 127563 |

The predict command is byte-for-byte identical, so the base can only change if
a **different support-pack / weights artifact** was selected on Kaggle (an
alternate / TTA / different-epoch pack mounted at run time and resolved by
`resolve_artifact()`). Tuning gates on a different base is not a controlled
experiment — that run must not be trusted or submitted.

The runner cannot control which datasets a Kaggle notebook mounts, so the fix
is twofold: **(1)** reproduce M19-C's exact selection logic and **log exactly
what was chosen**, and **(2)** a **hard guard** that refuses to recommend
submission unless the base matches M19-C exactly.

## Exact artifact / weights / support-pack selection logic (unchanged from M19-C)

1. **Artifact resolution** — `resolve_artifact()` walks `ARTIFACT_CANDIDATE_DIRS`
   in preference order and accepts the **first** mount that contains all
   required entries (`ARTIFACT_MANIFEST.json`, `repo/scripts/predict_unet_transformer.py`,
   `weights/unet_transformer/split_0/edge_predictor_best.pth`, `wheels`):
   ```
   /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1   # canonical (preferred)
   /kaggle/input/datasets/tom99763/biohub-tracking-support-pack-50ep-v1
   /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-v1
   /kaggle/input/biohub-tracking-support-pack-50ep-v1
   /kaggle/input/biohub-tracking-support-pack-v1
   ```
   The canonical **50ep-v1** pack is first, so with only it attached the base
   is M19-C's. It never chooses an alternate/TTA pack by preference.
2. **Weights** — always `weights/unet_transformer/split_0/edge_predictor_best.pth`
   (`select_weight_split(0, …)` → split 0). No alternate split, no TTA weights.
3. **Predict command** — byte-for-byte identical to M16/M19-C:
   `--det-threshold 0.99 --ilp-edge-weight -1.0 --ilp-appearance-weight 0.1
   --ilp-disappearance-weight 0.1 --ilp-division-weight 1.0 --use-ilp`.
4. **Selection is recorded** to `/kaggle/working/milestone20_artifact_selection.json`
   with: `selected_artifact_path`, `support_pack_root`, `repo_path`,
   `weights_path`, `weights_mode`, the full `artifact_manifest`, the candidate
   preference list, and each candidate's checked entry-status — so any drift is
   visible at a glance.

## Baseline guard — code and thresholds

```python
EXPECTED_N_NODES_BEFORE = 131797   # M19-C (public 0.880) pre-postprocess node count
EXPECTED_N_EDGES_BEFORE = 118992   # M19-C (public 0.880) pre-postprocess edge count
BASELINE_TOLERANCE = 0             # exact match required

def check_baseline_guard(conversion):
    actual_nodes = conversion.get("n_nodes_before")
    actual_edges = conversion.get("n_edges_before")
    nodes_ok = actual_nodes is not None and abs(int(actual_nodes) - EXPECTED_N_NODES_BEFORE) <= BASELINE_TOLERANCE
    edges_ok = actual_edges is not None and abs(int(actual_edges) - EXPECTED_N_EDGES_BEFORE) <= BASELINE_TOLERANCE
    passed = bool(nodes_ok and edges_ok)
    return {
        "baseline_guard_passed": passed,
        "expected_n_nodes_before": 131797, "expected_n_edges_before": 118992,
        "actual_n_nodes_before": actual_nodes, "actual_n_edges_before": actual_edges,
        "baseline_tolerance": 0,
        "submission_recommendation": "OK_TO_SUBMIT_IF_GATES_PASS" if passed
                                     else "DO_NOT_SUBMIT_BASELINE_MISMATCH",
        "mismatch_detail": None if passed else "<loud message: base changed; attach only 50ep-v1>",
    }
```

- Runs on the `milestone20_geff_conversion.json` counts **before** any
  post-processing.
- **Fails closed**: a missing count or an off-by-one (±1 node or ±1 edge) also
  fails (tolerance 0).
- On failure the report is written with `baseline_guard_passed=False`,
  `submission_recommendation="DO_NOT_SUBMIT_BASELINE_MISMATCH"`, and the summary
  prints the mismatch loudly. The submission.csv is still produced and
  structurally valid (so `valid=True`, no fallback), but the **recommendation
  is explicit DO-NOT-SUBMIT**.

## Tuned post-processing gates (identical to M20 — applied only when the base matches)

| Gate | M19-C → M20(FIXED) |
|---|---|
| DIV parent-child / sister / existing-child (µm) | 4.7/6.85/7.45 → 5.1/7.4/7.9 |
| DIV frame-cap / global-cap frac | 0.0072/0.004 → 0.0090/0.0052 |
| GAP1 max-total / cap-frac / cap-abs | 6.2/0.006/300 → 7.0/0.0075/380 |
| GAP2 step / total (µm) | 4.4/10.2 → 4.7/11.0 |
| GAP2 velocity normdiff (µm) | 6.0 → 6.6 |
| GAP2 cap-frac / cap-abs | 0.0045/180 → 0.0052/220 |
| GAP2 velocity cos-min | −0.25 (held) |
| linefit (win 2 / wt 0.72), prune (0.6) | unchanged |

## Expected effects (only meaningful if the base matches 131797/118992)

- `safe_divisions_added` ~600–750 · `gap1_closed` ~680–820 · `gap2_recovered` ~410–490
- `synthetic_nodes_added` ~1500–1800 · `n_nodes_after` ~133,300–133,600 · `n_edges_after` ~122,200–122,900
- `max_in_degree` 1 / `max_out_degree` 2 (invariant)

## Risk assessment

- **The base-drift risk is now caught, not silent.** If the wrong pack is
  mounted, the run still completes and writes a valid submission, but the
  report/summary say `DO_NOT_SUBMIT_BASELINE_MISMATCH`. You cannot accidentally
  submit a different-base result believing it is a gate-tuning result.
- Gate-widening risk is unchanged from M20 and small (~7–15% on distances, caps
  raised to avoid clipping); gap2 (2 nodes each) widened least, velocity
  direction gate held.
- Topology invariants hold by construction (unit-timepoint edges only, no
  multi-parent, out-degree ≤ 2, no direct multi-frame edges).

## Operational instruction (important)

To reproduce M19-C's 0.880 base, **attach ONLY** the canonical
`biohub-tracking-support-pack-50ep-v1` dataset and **detach any alternate /
TTA / other-epoch** support packs before running. The guard is the safety net
if that is not done.

## Submit / no-submit criteria

Submit **only if ALL** hold (read from the on-Kaggle diagnostic cell):

- `baseline_guard_passed == True` **and** `submission_recommendation == "OK_TO_SUBMIT_IF_GATES_PASS"`
  (i.e. `n_nodes_before == 131797` **and** `n_edges_before == 118992`)
- `valid == True`, `fallback_used == False`
- `final_source == "reference_learned_graph_postprocessed"`
- `max_in_degree <= 1`, `max_out_degree <= 2`
- `/kaggle/working/submission.csv` exists, no NaN, `id` column consecutive
- diagnostics sane vs guidance (`synthetic_nodes_added` ≈ 1500–2200; a much
  larger number → do not submit)

If `baseline_guard_passed == False` (or any other gate fails): **DO NOT
SUBMIT.** Re-attach only the 50ep-v1 pack and re-run, or keep **M19-C (0.880)**
as the standing best confirmed submission. This candidate is built and verified
only; it is run manually on Kaggle and reviewed before any submission decision.
