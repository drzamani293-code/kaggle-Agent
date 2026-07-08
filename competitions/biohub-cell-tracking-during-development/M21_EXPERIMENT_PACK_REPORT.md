# M21 experiment pack — pilkwang base, controlled variants

Three controlled variants on the **pinned pilkwang** support pack, testing
whether post-processing on this newer/larger base graph can beat **M19-C's
0.880**. M19-C remains the best/final candidate; **M21 is experimental**.

## Pinning + artifact guard (replaces the M19-C baseline guard)

- **Pinned artifact:** `/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1`
  (candidate list = this path only; **tom99763 is excluded**).
- Records **selected path**, **artifact_name** (from `ARTIFACT_MANIFEST.json`),
  and **weight_sha256** (SHA-256 of `weights/unet_transformer/split_0/edge_predictor_best.pth`)
  into `milestone21_artifact_selection.json`.
- **Artifact guard:** if the selected path ≠ the pinned pilkwang path →
  `recommendation = DO_NOT_SUBMIT_WRONG_ARTIFACT`. The old M19-C
  `n_nodes_before/n_edges_before` baseline guard is **not** used (that base is a
  different, older artifact).
- Predict command byte-for-byte identical to M16/M19-C: `--det-threshold 0.99
  --ilp-edge-weight -1.0 --ilp-appearance-weight 0.1 --ilp-disappearance-weight
  0.1 --ilp-division-weight 1.0 --use-ilp`. Split-0 weights.

## Exact difference among A / B / C

All three share the pinned pilkwang base and the identical predict command.
They differ **only** in post-processing ops + gate preset:

| Variant | Ops | Gate preset | Synthetic nodes | Purpose |
|---|---|---|---|---|
| **A** `pilkwang350_m19c_gates` | safe_divisions → gap1 → gap2 → linefit → prune | **M19-C original** (div 4.7/6.85/7.45, gap1 6.2, gap2 4.4/10.2/−0.25/6.0) | yes (gap1+gap2) | stronger base + proven gates vs 0.880 |
| **B** `pilkwang350_safe_div_only` | safe_divisions → prune | M19-C | **0** | pure division recovery, no node-inflation risk |
| **C** `pilkwang350_light_gap` | safe_divisions → gap1 → gap2 → linefit → prune | M19-C **except gap2 tightened** (4.4/10.2→**3.8/8.5**, cos_min −0.25→**−0.10**, normdiff 6.0→**5.0**, cap_frac 0.0045→**0.0030**, cap_abs 180→**120**) | yes, **limited** | controlled middle ground below M20's 1808 |

Only gap2 changes between A and C; B simply drops all gap/linefit ops.

## Expected diagnostic ranges (estimates on the pilkwang base ~142k nodes)

Reference points: M19-C base (131797) gave div 466 / gap1 593 / gap2 358 /
synth 1309; M20 (pilkwang base, **tuned** gates) gave div 666 / gap1 830 / gap2
489 / **synth 1808**. M21 uses M19-C (non-tuned) gates, so expect **below**
M20's counts on the same base:

| Variant | safe_divisions | gap1 | gap2 | synthetic_nodes | n_nodes_after (≈ base 142193 + synth) |
|---|---|---|---|---|---|
| **A** | ~520–640 | ~640–760 | ~380–450 | **~1400–1650** | ~143.6k–143.9k |
| **B** | ~520–640 | 0 | 0 | **0** | ~142.2k (base + 0) |
| **C** | ~520–640 | ~640–760 | ~130–230 | **~900–1200** (target 800–1400) | ~143.1k–143.4k |

`n_edges_after` ≈ base 127563 + divisions + 2·gap1 + 3·gap2. Estimates only —
the real gates decide; do not force them.

## Risk assessment

- **B** is lowest risk (zero synthetic nodes; can only affect the division
  term and node-penalty favorably).
- **A** is the direct "does the bigger base + proven gates win?" test; risk is
  the extra synthetic nodes vs the node-over-prediction penalty, but gates are
  the proven M19-C ones.
- **C** deliberately caps gap2 to keep synthetic growth modest.
- Topology invariants hold by construction (unit-timepoint edges only, no
  multi-parent, out-degree ≤ 2). The artifact guard prevents submitting a
  wrong-pack result.

## Kaggle run order (5 submissions available)

Attach **only** the pilkwang `biohub-tracking-support-pack-50ep-v1` dataset,
GPU T4×2, Internet Off. Run each cell, review diagnostics, submit only if the
gates below pass.

1. **M21_A** — strongest single test (base + full proven chain).
2. **M21_C** — controlled lighter chain (isolate whether trimming gap2 helps).
3. **M21_B** — division-only floor (no synthetic-node risk).

Keep **M19-C 0.880** as final unless an M21 public score **beats 0.880**.

## Diagnostic cell for Kaggle

```python
from pathlib import Path
import json
import pandas as pd

REPORT = "/kaggle/working/milestone21_reference_submission_report.json"
CONV   = "/kaggle/working/milestone21_geff_conversion.json"
SEL    = "/kaggle/working/milestone21_artifact_selection.json"
FBACK  = "/kaggle/working/milestone21_failure_fallback_report.json"

for p in [REPORT, SEL, CONV, FBACK]:
    print("\n" + "="*100)
    print(p, "exists:", Path(p).exists())
    if Path(p).exists():
        print(Path(p).read_text(errors="replace")[-16000:])

print("\n" + "="*100 + "\nPRE-SUBMIT REVIEW")
if Path(REPORT).exists():
    r = json.loads(Path(REPORT).read_text())
    for k in ["recommendation","artifact_guard_passed","selected_artifact_path","artifact_name",
              "weight_sha256","valid","fallback_used","final_source",
              "n_nodes_before","n_edges_before","n_nodes_after","n_edges_after",
              "safe_divisions_added","gap1_closed","gap2_recovered","synthetic_nodes_added",
              "n_node_rows","n_edge_rows","max_in_degree","max_out_degree"]:
        print(f"  {k}: {r.get(k)}")

s = Path("/kaggle/working/submission.csv")
print("\nsubmission exists:", s.exists(), "size:", s.stat().st_size if s.exists() else None)
if s.exists():
    df = pd.read_csv(s)
    print("shape:", df.shape)
    print(df["row_type"].value_counts())
    print("has NaN:", df.isna().any().any())
    print("id consecutive:", df["id"].tolist() == list(range(len(df))))
    print(pd.crosstab(df["dataset"], df["row_type"]))
```

## Submit / no-submit criteria (per variant)

Submit **only if ALL** hold:
- `recommendation == "OK_TO_SUBMIT_EXPERIMENTAL"` and `artifact_guard_passed == True`
  (selected path is the pinned pilkwang pack)
- `valid == True`, `fallback_used == False`, `final_source == "reference_learned_graph_postprocessed"`
- `max_in_degree <= 1`, `max_out_degree <= 2`
- `submission.csv` exists, no NaN, `id` column consecutive
- diagnostics sane vs guidance (variant C `synthetic_nodes_added` ≈ 800–1400; a
  much larger number → re-check before submitting)

If `recommendation == "DO_NOT_SUBMIT_WRONG_ARTIFACT"` → a non-pilkwang pack was
mounted: detach it, attach only pilkwang, re-run. If any other gate fails →
do not submit. **M19-C 0.880 remains final unless an M21 score beats it.**
