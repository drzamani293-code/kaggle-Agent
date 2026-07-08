# M23 — node-penalty repair pack (pinned pilkwang350)

M22 forensic returned **TRUE_M19C_ARTIFACT_NOT_FOUND** (the Notebook input had
no usable repo/weights; the true 131797/118992 base is unrecoverable). M19-C
0.880 stays best/final. M21-A on the pilkwang350 base scored only **0.874** —
its base over-predicts nodes (142193 vs 131797) and 1503 synthetic nodes did
not help. **M23 attacks the node over-prediction directly**: prune low-value
detections *before* post-processing, then conservative post-processing.

## Pinning + guards

- **Pinned:** `/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1`
  (tom99763 excluded).
- **Artifact guard (both must hold):** selected path == pinned **and**
  `weight_sha256 == dfb848aa8e490bba8eda91ac927b9ad1d8b06296487ba8504e45a1037c5e36ec`
  (the M21-A pilkwang350 weights). Else → `DO_NOT_SUBMIT_WRONG_ARTIFACT`.
- **Safety report gates:** `prune_fraction > 8%` → `DO_NOT_SUBMIT_OVERPRUNED`;
  post-repair node count `< 130000` or `> 145000` →
  `DO_NOT_SUBMIT_UNSANE_NODE_COUNT`; invalid → `DO_NOT_SUBMIT_VALIDATION_FAILED`.
  A valid `submission.csv` is always written (hidden-safe); the report names the
  failed gate. If no node-score attribute exists, the report says
  **topology-only pruning** was used.
- Predict command identical to M16/M19-C (det 0.99, ilp −1.0/0.1/0.1/1.0, --use-ilp).

## Exact prune heuristics

Repair operates on the base graph (after M16 topology enforcement), removal-only
(never raises any degree):

1. **Protect** (never pruned): division sources (out-degree 2) and their
   children; nodes in **long tracks** (weakly-connected component size ≥
   `long_track_min`).
2. **Candidates** (eligible to prune): non-protected nodes that are **isolated**
   (degree 0) **or** in a **short component** (size ≤ `short_max`).
3. **If a node score exists** (`score`/`node_score`/`node_prob`/`prob`/`weight`/
   `confidence`/`detection_score`): restrict candidates to the **bottom
   `low_score_pct` tail** and prune **lowest-score first**. Otherwise
   **topology-only**: prune smallest-component / earliest first.
4. **Cap:** never remove more than `prune_max_frac` of base nodes. Removing a
   node also removes its incident edges (so no dangling edges; degrees only drop).
5. **C only — terminal low-confidence edge trim** (`prune_dangling_low_conf_edges`):
   if `edge_prob` exists, remove edges whose target is a track-end (`out==0`)
   and whose source is not a divider (`out==1`) and whose `edge_prob` is in the
   bottom `edge_prob_prune_pct`, capped at 1% of base nodes.

Per-variant repair config:

| Variant | short_max | prune_max_frac | low_score_pct | long_track_min | edge trim |
|---|---|---|---|---|---|
| A / B (light) | 2 | 0.045 | 60 | 4 | no |
| C (balanced) | 3 | 0.065 | 70 | 5 | yes (≤1%) |

Post-processing after repair:

| Variant | Ops after repair | Synthetic nodes |
|---|---|---|
| **A** node_prune_light | safe_divisions → gap1 → gap2 → linefit → prune (M19-C gates) | yes |
| **B** node_prune_safe_div_only | safe_divisions → prune | **0** |
| **C** edge_node_balanced | safe_divisions → gap1 → linefit → prune (no gap2) | limited |

## Expected ranges (estimates on the 142193/127563 pilkwang350 base)

| Variant | after-repair nodes | synthetic_nodes | notes |
|---|---|---|---|
| **A** | ~136000–141000 (prune ≤ 4.5%) | ~1000–1500 | full_chain on a lighter base |
| **B** | ~136000–141000 | **0** | pure node-penalty test |
| **C** | ~134000–139000 (prune ≤ 6.5%) | ~400–900 | node+edge repair, no gap2 |

All stay inside the 130000–145000 sanity window and ≤ 8% prune cap. Estimates
only — the actual gates (and whether a node score exists) decide.

## Kaggle run order (B → C → A)

Attach **only** the pilkwang `biohub-tracking-support-pack-50ep-v1` dataset,
GPU T4×2, Internet Off. Run each variant in a clean cell, review the diagnostic,
submit only if the gates pass.

1. **M23_B** — node-penalty reduction with **zero synthetic nodes** (lowest risk;
   M21-A's synthetic-heavy run only scored 0.874).
2. **M23_C** — balanced node+edge repair, gap1 only (no gap2).
3. **M23_A** — light repair + full_chain, only if B/C fall short.

Keep **M19-C 0.880** as final unless an M23 score beats it.

## Diagnostic cell for Kaggle

```python
from pathlib import Path
import json
import pandas as pd

REPORT = "/kaggle/working/milestone23_reference_submission_report.json"
CONV   = "/kaggle/working/milestone23_geff_conversion.json"
SEL    = "/kaggle/working/milestone23_artifact_selection.json"
FBACK  = "/kaggle/working/milestone23_failure_fallback_report.json"

for p in [REPORT, SEL, CONV, FBACK]:
    print("\n" + "="*100)
    print(p, "exists:", Path(p).exists())
    if Path(p).exists():
        print(Path(p).read_text(errors="replace")[-16000:])

print("\n" + "="*100 + "\nPRE-SUBMIT REVIEW")
if Path(REPORT).exists():
    r = json.loads(Path(REPORT).read_text())
    for k in ["recommendation","artifact_guard_passed","selected_artifact_path","artifact_name","weight_sha256",
              "repair_method","score_fields_available","edge_fields_available",
              "n_nodes_before_repair","n_edges_before_repair","n_nodes_after_repair","n_edges_after_repair",
              "nodes_pruned","edges_pruned","prune_fraction",
              "safe_divisions_added","gap1_closed","gap2_recovered","synthetic_nodes_added",
              "n_node_rows","n_edge_rows","valid","fallback_used","max_in_degree","max_out_degree"]:
        print(f"  {k}: {r.get(k)}")

s = Path("/kaggle/working/submission.csv")
print("\nsubmission exists:", s.exists(), "size:", s.stat().st_size if s.exists() else None)
if s.exists():
    df = pd.read_csv(s)
    print("shape:", df.shape); print(df["row_type"].value_counts())
    print("has NaN:", df.isna().any().any())
    print("id consecutive:", df["id"].tolist() == list(range(len(df))))
    print(pd.crosstab(df["dataset"], df["row_type"]))
```

## Submit / no-submit criteria (per variant)

Submit **only if**:
- `recommendation == "OK_TO_SUBMIT_EXPERIMENTAL"`
- `artifact_guard_passed == True` (pinned pilkwang350 path **and** weight hash `dfb848…36ec`)
- `prune_fraction <= 0.08` and `130000 <= n_nodes_after_repair <= 145000`
- `valid == True`, `fallback_used == False`, `final_source == "reference_learned_graph_postprocessed"`
- `max_in_degree <= 1`, `max_out_degree <= 2`, `submission.csv` no NaN, `id` consecutive

Otherwise the report shows `DO_NOT_SUBMIT_WRONG_ARTIFACT` /
`DO_NOT_SUBMIT_OVERPRUNED` / `DO_NOT_SUBMIT_UNSANE_NODE_COUNT` /
`DO_NOT_SUBMIT_VALIDATION_FAILED` → **do not submit; M19-C 0.880 remains final.**
