# M24 — 400ep clean/lean base jump pack

A real attempt at beating **M19-C 0.880** using the newly-discovered **400ep**
artifact, whose raw base is much closer to M19-C's true base than pilkwang350.

## Why 400ep

| Base | n_nodes / n_edges | score |
|---|---|---|
| **M19-C true** | 131797 / 118992 | **0.880** (best) |
| **400ep (this pack)** | **127790 / 115694** | — (to test) |
| pilkwang350 | 142193 / 127563 | M21-A 0.874 |
| tom99763 | 161098 / 137520 | over-predicting |

M23-C's guard failure accidentally revealed the 400ep snapshot
(`biohub-tracking-support-pack-400ep-snapshot-v1`, weight
`12f6881e…2fe771`). Its lean base (127790, only +? from M19-C's 131797, and
far below pilkwang350's 142193) means **no node-prune is needed** — M24 tests
it AS-IS. M23's node-prune/edge-trim is deliberately **not reused** (it
over-pruned 400ep 127790→121369).

## Exact difference among A / B / C

All three: **400ep base as-is (no prune)**, identical predict command, M19-C
gates. They differ only in the post-processing op list:

| Variant | Ops | Synthetic nodes | Purpose |
|---|---|---|---|
| **A** `400ep_fullchain_m19c_gates` | safe_divisions → gap1 → gap2 → linefit → prune | yes | **highest upside**: lean base + the chain that scored 0.880 |
| **B** `400ep_gap1_only` | safe_divisions → gap1 → linefit → prune (**no gap2**) | fewer | tests whether gap2 is risky on the lean base |
| **C** `400ep_safe_div_only` | safe_divisions → prune (**no gap1/gap2/linefit**) | **0** | zero-synthetic control |

## Artifact guard (400ep — NOT the old M23 dfb848 hash)

Passes only if **all** hold:
- selected path == `/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1`
- `artifact_name` contains **`400ep`** (or == `biohub-tracking-support-pack-400ep-snapshot-v1`)
- `weight_sha256` == **`12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`**

Else → `DO_NOT_SUBMIT_WRONG_ARTIFACT`. The old pilkwang350 hash
`dfb848…36ec` is explicitly **rejected** (the guard requires the 400ep hash).

## Sanity gates (recommendation order)

1. artifact guard fails → `DO_NOT_SUBMIT_WRONG_ARTIFACT`
2. invalid submission → `DO_NOT_SUBMIT_VALIDATION_FAILED`
3. `n_nodes_before` < 120000 or > 135000 → `DO_NOT_SUBMIT_UNEXPECTED_BASE`
4. final `n_node_rows` < 120000 or > 135000 → `DO_NOT_SUBMIT_UNSANE_NODE_COUNT`
5. `synthetic_nodes_added` > 2200 → `DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION`
6. else → `OK_TO_SUBMIT_EXPERIMENTAL`

A valid `submission.csv` is always written (hidden-safe); the report names the
failed gate.

## Expected diagnostic ranges (400ep base ~127790/115694; estimates)

| Variant | base nodes | final n_node_rows | synthetic_nodes | safe_divisions |
|---|---|---|---|---|
| **A** | ~127790 | ~129000–131000 | ~1000–1600 | ~380–520 |
| **B** | ~127790 | ~128300–129200 | ~400–800 | ~380–520 |
| **C** | ~127790 | ~127790 (+0) | **0** | ~350–500 |

All final counts stay inside the 120000–135000 sanity window; synthetic ≤ 2200.
Estimates only — the real gates decide.

## Kaggle run order — A → B → C

Attach **only** the pilkwang `biohub-tracking-support-pack-50ep-v1` dataset
(currently the 400ep snapshot), GPU T4×2, Internet Off. Run each in a clean
cell, review the diagnostic, submit only if the gates pass.

1. **M24_A** — clean 400ep base + full M19-C chain (the only high-upside candidate).
2. **M24_B** — gap1 only (isolates whether gap2 hurts on the lean base).
3. **M24_C** — safe-divisions only (zero-synthetic control).

Keep **M19-C 0.880** as final unless an M24 score beats it.

## Diagnostic cell for Kaggle

```python
from pathlib import Path
import json
import pandas as pd

REPORT = "/kaggle/working/milestone24_reference_submission_report.json"
CONV   = "/kaggle/working/milestone24_geff_conversion.json"
SEL    = "/kaggle/working/milestone24_artifact_selection.json"
FBACK  = "/kaggle/working/milestone24_failure_fallback_report.json"

for p in [REPORT, SEL, CONV, FBACK]:
    print("\n" + "="*100)
    print(p, "exists:", Path(p).exists())
    if Path(p).exists():
        print(Path(p).read_text(errors="replace")[-16000:])

print("\n" + "="*100 + "\nPRE-SUBMIT REVIEW")
if Path(REPORT).exists():
    r = json.loads(Path(REPORT).read_text())
    for k in ["recommendation","artifact_guard_passed","selected_artifact_path","artifact_name","weight_sha256",
              "n_nodes_before","n_edges_before","n_nodes_after","n_edges_after",
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

Submit **only if**:
- `recommendation == "OK_TO_SUBMIT_EXPERIMENTAL"`
- `artifact_guard_passed == True` (400ep path + name + weight `12f688…2fe771`)
- `120000 <= n_nodes_before <= 135000` (base is the expected 400ep base ~127790)
- `120000 <= n_node_rows <= 135000` and `synthetic_nodes_added <= 2200`
- `valid == True`, `fallback_used == False`, `final_source == "reference_learned_graph_postprocessed"`
- `max_in_degree <= 1`, `max_out_degree <= 2`, `submission.csv` no NaN, `id` consecutive

Otherwise the report shows `DO_NOT_SUBMIT_WRONG_ARTIFACT` /
`DO_NOT_SUBMIT_UNEXPECTED_BASE` / `DO_NOT_SUBMIT_UNSANE_NODE_COUNT` /
`DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION` / `DO_NOT_SUBMIT_VALIDATION_FAILED` →
**do not submit; M19-C 0.880 remains final.**
