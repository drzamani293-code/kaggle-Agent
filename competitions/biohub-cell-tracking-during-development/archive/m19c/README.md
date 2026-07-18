# M19-C Fallback Archive (immutable)

**Status:** confirmed best/final. Public score **0.880**. Selectable at all times as
the safe fallback until a strictly higher CONFIRMED public score exists.

**Do not edit these files.** They are byte-preserved copies of the M19-C artifacts
from the commit that recorded the 0.880 public score. Any change to the working
milestone19 source files at the repository root MUST NOT touch these copies. The
`MANIFEST.sha256` beside this README is the authoritative fingerprint for later
audits (Opus 4.6 audit, hidden-rerun reproduction).

## Files (byte-identical copies)

| File | Purpose |
|---|---|
| `M19_VARIANT_C_FULL_CHAIN.txt` | One-cell Kaggle deliverable — full postprocess chain (safe divisions + gap recovery + linefit + isolated-prune + relabel). |
| `milestone19_metric_postprocess_runner.py`  | Standalone runner (module form). |
| `kaggle_cell_milestone19_metric_postprocess_runner.py` | Kaggle-cell form of the runner. |
| `milestone19_metric_postprocess_runner.ipynb` | Notebook form. |
| `MANIFEST.sha256` | SHA-256 of every file above. |

## Provenance

- Milestone: M19 (metric-aware post-processing of the reference learned graph)
- Variant: **C** = `full_chain` (label in `intelligence_db/data/seed_experiments.json`)
- Detection threshold: 0.99
- ILP weights (edge, appearance, disappearance, division): -1.0, 0.1, 0.1, 1.0
- `use_ilp = True`
- Postprocess ops (in order):
  - `safe_divisions`
  - `gap_recovery_1_2`
  - `linefit`
  - `prune_isolated`
  - `relabel`
- Public LB score: **0.880** (status = final in the intelligence DB)
- Live intelligence-DB row: `M19_C_FULL_CHAIN_PENDING` (name kept for schema
  compatibility with the seed file; the `status: final` field and `public_score:
  0.88` are the authoritative labels).

## Validation policy

M19-C submissions must satisfy the same dynamic validator we now use in M38:

- exact column order,
- discovered-dataset match (dynamic, no hardcoded stems),
- no NaN in mandatory fields,
- integer node/source/target ids,
- `id == 0..N-1` (consecutive row ids),
- no duplicate semantic rows,
- unique node ids per dataset,
- edge endpoints exist in dataset,
- `source_t < target_t`,
- max in-degree ≤ 1, max out-degree ≤ 2.

## How to restore M19-C in a Kaggle notebook

Copy `M19_VARIANT_C_FULL_CHAIN.txt` from this archive into a single Kaggle code
cell and run. It reads the reference learned graph, applies the audited
postprocess chain, and writes `/kaggle/working/submission.csv`.

Or, in a Python-runner environment, execute
`milestone19_metric_postprocess_runner.py` directly.

## Do NOT

- Do NOT edit any file in this directory.
- Do NOT regenerate the SHA manifest to match edits — regenerate only if the
  archive itself is rebuilt from an equally byte-preserved source.
- Do NOT delete this archive when refactoring the working milestone19 files;
  M19-C is the fallback candidate that must remain selectable.
