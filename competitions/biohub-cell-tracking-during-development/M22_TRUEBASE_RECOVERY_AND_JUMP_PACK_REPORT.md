# M22 — true-base recovery + jump pack

A two-stage system to earn a real score jump instead of drift-base noise.
M19-C (0.880) remains the best/final candidate throughout.

## Why M22 exists

M19-C's public 0.880 came from a base learned graph of **131797 nodes /
118992 edges** (before post-processing). Neither mounted support pack
reproduces it, so tuning post-processing on them is meaningless:

| Source | base n_nodes / n_edges | weight_sha256 | result |
|---|---|---|---|
| **M19-C (true)** | **131797 / 118992** | (unknown — to recover) | **0.880** |
| pilkwang350 | 142193 / 127563 | `dfb848aa…5e36ec` | M21-A = **0.874** (drift) |
| tom99763 | 161098 / 137520 | `912ae91a…ab3f53` | base too large |

M21-A confirmed the drift base does not help (0.874, below 0.880). So the
**true M19-C artifact must be recovered** — likely a **Notebook input** from
the original run (a kernel named like *"Biohub Cell Tracking: Learned Graph w
G"*), not the two dataset packs.

## Stage 1 — forensic diagnostic (`M22_ARTIFACT_FORENSIC_DIAGNOSTIC.txt`)

Read-only, no submission. On Kaggle it:
- Enumerates every artifact root under `/kaggle/input` (datasets **and**
  notebooks; one/two/three levels deep; hidden-safe globbing).
- For each root with `repo/scripts/predict_unet_transformer.py` +
  `weights/unet_transformer/split_0/edge_predictor_best.pth`, records
  `candidate_path`, `artifact_name`, `weight_sha256`, manifest summary, repo
  path, wheels presence, required-files presence.
- Flags the two **known-bad** weight hashes so a genuinely different (candidate
  true) artifact stands out.
- **Smoke run** (`M22_FORENSIC_SMOKE_RUN=True`): for each viable candidate whose
  hash is NOT known-bad, runs the exact M19-C predict command, converts the
  GEFF to base counts (no post-processing), and checks
  `n_nodes_before==131797 AND n_edges_before==118992`. Stops at the first match.
- Writes `/kaggle/working/m22_artifact_forensic_report.json` and
  `/kaggle/working/m22_candidate_table.csv`; prints
  **TRUE_M19C_ARTIFACT_FOUND** (with selected path/name/hash/repo/weights/base
  counts) or **TRUE_M19C_ARTIFACT_NOT_FOUND**.

## Stage 2 — true-base guarded variants

Run only after the true artifact is found (or explicitly pinned by path/hash).
Broad discovery **prefers a pack whose weight hash is not known-bad**, and two
guards decide the recommendation:
- `true_m19c_artifact_guard_passed` = resolved AND weight_sha256 not known-bad.
- `public_base_count_guard_passed` = `n_nodes_before==131797` and
  `n_edges_before==118992` (tolerance 0, fail-closed).

A valid `submission.csv` is always written (hidden-safe), but the report says
`DO_NOT_SUBMIT_*` unless both guards pass.

| Variant | Ops | Gates vs M19-C | Target synthetic | Purpose |
|---|---|---|---|---|
| **A** `truebase_safe_div_tune` | safe_divisions → prune | divisions widened 4.7/6.85/7.45→**5.1/7.3/7.9**, caps up | **0** | push division term with zero node-penalty risk |
| **B** `truebase_light_gap` | safe_div → gap1 → gap2 → linefit → prune | gap1 6.2→6.6; gap2 **tightened** 4.4/10.2→3.6/8.0, cos_min −0.25→−0.05, normdiff 6.0→4.6, caps 0.0045/180→0.0025/100 | ~900–1300 | keep edge gains, trim gap2 false positives |
| **C** `truebase_div_plus_gap1_only` | safe_div → gap1 → linefit → prune (**no gap2**) | M19-C gates | ~500–800 | isolate whether gap2 is the risky part |

Predict command identical to M16/M19-C: `--det-threshold 0.99
--ilp-edge-weight -1.0 --ilp-appearance-weight 0.1 --ilp-disappearance-weight
0.1 --ilp-division-weight 1.0 --use-ilp`.

## Required Kaggle inputs (to recover M19-C)

Attach to the forensic notebook, in addition to the competition data:
- The competition dataset (test data + sample_submission).
- **The original M19-C Notebook input** — the kernel whose output was used as
  the learned-graph artifact (name similar to *"Biohub Cell Tracking: Learned
  Graph w G"*). This is the most likely home of the true 131797/118992 base.
- Any other candidate support-pack datasets you have (the forensic will
  enumerate and hash them all; known-bad ones are auto-skipped for smoke run).

## Kaggle run instructions

1. **Stage 1:** new notebook, GPU T4×2, Internet Off. Attach the competition
   data + every candidate artifact input (**including the Notebook input**).
   Paste `M22_ARTIFACT_FORENSIC_DIAGNOSTIC.txt` into one cell, Run. Read the
   verdict + `m22_candidate_table.csv`.
2. If **TRUE_M19C_ARTIFACT_FOUND**: note the `selected_artifact_path` /
   `weight_sha256`. **Stage 2:** run the three M22 variants in order A → B → C,
   each in a clean cell; review the diagnostic below; submit only if the gates
   pass.
3. If **TRUE_M19C_ARTIFACT_NOT_FOUND**: attach the original M19-C Notebook
   input and re-run Stage 1. Do not submit anything; **M19-C 0.880 stays
   final**.

## Diagnostic cell for Kaggle (Stage 2)

```python
from pathlib import Path
import json
import pandas as pd

REPORT = "/kaggle/working/milestone22_reference_submission_report.json"
CONV   = "/kaggle/working/milestone22_geff_conversion.json"
SEL    = "/kaggle/working/milestone22_artifact_selection.json"
FBACK  = "/kaggle/working/milestone22_failure_fallback_report.json"

for p in [REPORT, SEL, CONV, FBACK]:
    print("\n" + "="*100)
    print(p, "exists:", Path(p).exists())
    if Path(p).exists():
        print(Path(p).read_text(errors="replace")[-16000:])

print("\n" + "="*100 + "\nPRE-SUBMIT REVIEW")
if Path(REPORT).exists():
    r = json.loads(Path(REPORT).read_text())
    for k in ["recommendation","true_m19c_artifact_guard_passed","public_base_count_guard_passed",
              "selected_artifact_path","artifact_name","weight_sha256","valid","fallback_used",
              "final_source","n_nodes_before","n_edges_before","n_nodes_after","n_edges_after",
              "safe_divisions_added","gap1_closed","gap2_recovered","synthetic_nodes_added",
              "max_in_degree","max_out_degree"]:
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

## Submit / no-submit criteria (Stage 2, per variant)

Submit **only if**:
- `recommendation == "OK_TO_SUBMIT_TRUEBASE_EXPERIMENT"`
- `true_m19c_artifact_guard_passed == True` **and** `public_base_count_guard_passed == True`
  (i.e. `n_nodes_before==131797` and `n_edges_before==118992`)
- `valid == True`, `fallback_used == False`, `final_source == "reference_learned_graph_postprocessed"`
- `max_in_degree <= 1`, `max_out_degree <= 2`, `submission.csv` no NaN, `id` consecutive

Otherwise the report shows one of `DO_NOT_SUBMIT_ARTIFACT_NOT_RECOVERED` /
`DO_NOT_SUBMIT_BASE_MISMATCH` / `DO_NOT_SUBMIT_VALIDATION_FAILED` → **do not
submit; M19-C 0.880 remains final.**
