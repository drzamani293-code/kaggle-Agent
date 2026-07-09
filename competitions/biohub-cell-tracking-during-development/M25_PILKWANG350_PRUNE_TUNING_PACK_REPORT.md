# M25 — pilkwang350 prune-tuning pack (around M23-B)

The 400ep path failed (M24-A 0.873, M24-B 0.872). The strongest experimental
path is **M23-B** (pilkwang350 node-prune + safe-divisions, **0.877** — only
0.003 behind M19-C 0.880). M25 tunes **only the node-prune / safe-division
balance** around M23-B, on the pinned pilkwang350 artifact. M19-C 0.880 stays
best/final unless an M25 score beats it.

## Full leaderboard so far

| Experiment | base | score |
|---|---|---|
| **M19-C** full_chain (true base) | 131797/118992 | **0.880** (best) |
| M21-A pilkwang350 full_chain | 142193/127563 | 0.874 |
| **M23-B** pilkwang350 prune + safe-div | 142193→137681 | **0.877** |
| M24-A 400ep full_chain | 127790 | 0.873 |
| M24-B 400ep gap1-only | 127790 | 0.872 |

M23-B reference: base 142193/127563 → after repair 137695/125307
(nodes_pruned 4498, prune_fraction 0.0316) → final 137681/125778,
safe_divisions_added 471, synthetic 0.

## Artifact guard (same as M23-B — pilkwang350, NOT 400ep)

Passes only if **all** hold:
- path == `/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1`
- `artifact_name` contains **`350ep`** (or == `biohub-tracking-support-pack-350ep-snapshot-v1`)
- `weight_sha256` == **`dfb848aa8e490bba8eda91ac927b9ad1d8b06296487ba8504e45a1037c5e36ec`**

Else → `DO_NOT_SUBMIT_WRONG_ARTIFACT`. The 400ep name/hash (`12f688…`) is
**rejected** (verified in tests).

## Exact difference among variants

All three: pinned pilkwang350, node-penalty repair (removal-only; protect
divisions + long tracks), identical predict command. They differ in the prune
strength and post-processing:

| Variant | Repair (prune) | Post-processing ops | Synthetic |
|---|---|---|---|
| **A** `prune_mild_safe_div` | short_max 2, **cap frac 0.022** (milder than M23-B's 0.0316) | safe_divisions → prune | **0** |
| **B** `prune_strong_safe_div` | short_max 3, **cap frac 0.052**, long_track_min 5 | safe_divisions → prune | **0** |
| **C** `m23b_plus_micro_gap1` | **exact M23-B** (short_max 2, frac 0.045) | safe_divisions → **very-light gap1** → linefit → prune (no gap2) | ~150–350 |

Variant C's gap1 is much stricter than M19-C/M24-B: `max_total_um` 6.2→**5.0**,
`cap_frac` 0.006→**0.0015**, `cap_abs` 300→**90**.

## Safety gates (recommendation)

`OK_TO_SUBMIT_EXPERIMENTAL` only if: guard passed, valid, non-fallback,
`final_source == reference_learned_graph_postprocessed`, degrees ≤1/≤2, no NaN,
`id` consecutive, **`132000 ≤ n_node_rows ≤ 142500`**, **`prune_fraction ≤ 0.07`**,
**`synthetic_nodes_added ≤ 400`**. Otherwise a specific
`DO_NOT_SUBMIT_WRONG_ARTIFACT / _VALIDATION_FAILED / _UNSANE_NODE_COUNT /
_OVERPRUNED / _SYNTHETIC_EXPLOSION` is reported. A valid `submission.csv` is
always written (hidden-safe).

## Expected diagnostic ranges (pilkwang350 base 142193/127563; estimates)

| Variant | prune_fraction | final n_node_rows | synthetic | safe_divisions |
|---|---|---|---|---|
| **A** | ~0.018–0.025 | ~139000–140500 | 0 | ~460–500 |
| **B** | ~0.045–0.055 | ~134800–136500 | 0 | ~460–500 |
| **C** | ~0.031 (M23-B) | ~137700–138000 | ~150–350 | ~460–500 |

All stay in 132000–142500 and ≤ 0.07 prune. Estimates only — the real gates
(and whether a node score exists) decide; if no node score exists, the report
says topology-only pruning was used.

## Kaggle run order — A → B → C

Attach **only** the pilkwang `biohub-tracking-support-pack-50ep-v1` dataset (the
pilkwang350/dfb848 snapshot), GPU T4×2, Internet Off. Run each in a clean cell,
review, submit only if the gates pass.

1. **M25_A** — milder prune (M23-B may have removed too many real nodes).
2. **M25_B** — stronger prune (more node-penalty reduction).
3. **M25_C** — M23-B repair + a very-light gap1 (tiny edge gain).

Keep **M19-C 0.880** as final unless an M25 score beats it.

## Diagnostic cell for Kaggle

```python
from pathlib import Path
import json
import pandas as pd

REPORT = "/kaggle/working/milestone25_reference_submission_report.json"
CONV   = "/kaggle/working/milestone25_geff_conversion.json"
SEL    = "/kaggle/working/milestone25_artifact_selection.json"
FBACK  = "/kaggle/working/milestone25_failure_fallback_report.json"

for p in [REPORT, SEL, CONV, FBACK]:
    print("\n" + "="*100)
    print(p, "exists:", Path(p).exists())
    if Path(p).exists():
        print(Path(p).read_text(errors="replace")[-16000:])

print("\n" + "="*100 + "\nPRE-SUBMIT REVIEW")
if Path(REPORT).exists():
    r = json.loads(Path(REPORT).read_text())
    for k in ["recommendation","artifact_guard_passed","artifact_name","weight_sha256","repair_method",
              "n_nodes_before_repair","n_edges_before_repair","n_nodes_after_repair","n_edges_after_repair",
              "nodes_pruned","edges_pruned","prune_fraction",
              "safe_divisions_added","gap1_closed","gap2_recovered","synthetic_nodes_added","isolated_nodes_pruned",
              "n_node_rows","n_edge_rows","max_in_degree","max_out_degree","valid","fallback_used","final_source"]:
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

Submit **only if**: `recommendation == OK_TO_SUBMIT_EXPERIMENTAL` **and**
`artifact_guard_passed` (pilkwang350 path + 350ep name + weight `dfb848…36ec`)
**and** `132000 ≤ n_node_rows ≤ 142500` **and** `prune_fraction ≤ 0.07` **and**
`synthetic_nodes_added ≤ 400` **and** `valid`, `fallback_used == False`,
degrees ≤1/≤2, no NaN, `id` consecutive. Any `DO_NOT_SUBMIT_*` → do not submit;
**M19-C 0.880 remains final**.
