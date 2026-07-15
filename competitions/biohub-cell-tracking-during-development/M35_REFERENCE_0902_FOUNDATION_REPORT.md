# M35 — Reference 0902 Repro, Official CV & Edge-TTA Foundation

M34 (self-contained TTA on the M19-C 0.880 baseline) is superseded by the supplied
**0.902** reference bundle. M35 reproduces and builds on that reference. M34 A–D
remain `SUPERSEDED_BY_M35_REFERENCE_0902` (files kept, not promoted).

## What was genuinely executed vs. blocked
- **M35-A — manifest-driven, PASSED on Kaggle.** All critical-file SHA256 matched
  `REFERENCE_BUNDLE_MANIFEST.json`. (Off the reference environment it reports
  `REFERENCE_AUDIT_FAILED`/`bundle_not_accessible` — the bundle is only mounted on Kaggle.)
- **M35-B — now executes the FULL audited NOTEBOOK** (not predict-only, not a
  placeholder): it runs `reference/biohub-competition-solution.ipynb` end-to-end in a
  fresh **nbclient** kernel — inference (4 GEFF stores) → GEFF conversion → motion
  relink → gap-1 close → calibrated safe divisions → short-track filter → linefit →
  final CSV — then compares the **freshly generated** CSV to `evidence/submission.csv`.
  It is unit-tested end-to-end with a **real executed notebook fixture** across
  full-pass / predict-only-fail / wrong-graph-mismatch / invalid-graph /
  copied-evidence-provenance-failure / stale-rejected / older-than-start /
  wrong-cwd / altered-scientific-cell / redirection-only / C-D-E-gating cases —
  proven data-dependent, never hardcoded. It genuinely runs only where the mounted
  bundle + support-pack weights + GPU are present; elsewhere it blocks honestly.
  **Running `predict_unet_transformer.py` alone is no longer accepted.**
- **M35-C/D/E — blocked** until M35-B genuinely passes (`REPRO_PASS_EXACT` /
  `REPRO_PASS_CANONICAL`).

## Review defects fixed (commit 32abb62 → this commit)
1. **Predict-only was not reproduction.** M35-B now executes the whole notebook
   (inference **and** all postprocessing), not just the predict subprocess.
2. **Correct execution cwd.** The reference repo is materialized to
   `/kaggle/working/tracking_repo` and the notebook runs with that as cwd, so the
   notebook's relative paths (`scripts/…`, `weights/…`, split JSON) resolve.
3. **Full environment.** The notebook itself performs offline dependency install,
   repo materialization, the D4 detection-TTA patch, inference, conversion, and the
   complete postprocessing chain in a fresh kernel.
4. **No stale acceptance.** All stale outputs (`submission.csv`, `run_stats.csv`,
   `tracking_repo`, exec dir, reproduced CSV) are deleted before the run; a
   pre-existing `submission.csv` can never become the result; the only accepted
   output is the redirected `m35_b_reference_reproduced.csv`.
5. **Anti-cheating provenance** (below).

## Defects fixed (per the runtime report)
1. **Ambiguous globs → manifest-driven, exact relative paths.** The bundle is resolved
   by finding `REFERENCE_BUNDLE_MANIFEST.json`; the 8 critical assets are addressed by
   exact relative path (incl. `reference/biohub-competition-solution.ipynb` and `.log`),
   never by filename alias.
2. **No arbitrary `config*.json`.** The preset is verified from the actual reference
   notebook/log/predict source text, not from an unrelated support-pack config.
3. **Dataset-scoped node IDs.** Uniqueness is validated per `(dataset, node_id)`; IDs
   may legitimately repeat across datasets. (`node_id_unique_per_dataset`).
4. **Isolated missing-bundle tests.** `find_reference_bundle_root` searches **only** the
   supplied roots — an isolated temp root can never discover real `/kaggle` assets. All
   missing-asset and downstream-block tests pass an isolated empty root.
5. **Real M35-B.** Replaced the unconditional `REFERENCE_REPRO_MISMATCH` placeholder
   with a genuine subprocess reproduction + comparison.

Also: exactly one standalone entry point per module (no stray unconditional runner
calls); each generated one-cell `.txt` is syntax-checked and self-tests 20/20.

## M35-A (manifest-driven audit)
Verifies: manifest present + complete; every critical file's SHA256 == manifest;
preset `public_0902_motion_division_calibration` present in the real notebook/log/source;
exact submission fingerprint **128511 nodes / 124002 edges / 252513 rows / 417 divisions**;
dataset-scoped node-id uniqueness; `dangling_edges=0`, `direct_multiframe_edges=0`,
`max_in_degree≤1`, `max_out_degree≤2`, finite coordinates, consecutive row id. Emits
`/kaggle/working/m35_reference_bundle_audit.json`; recommendation `REFERENCE_AUDIT_PASS`
/ `REFERENCE_AUDIT_FAILED`. Accelerator None, Internet Off, never writes `submission.csv`.

## M35-B (real full-notebook reproduction)
Inputs: competition data, `biohub-tracking-support-pack-50ep-v1` (controlled 400ep
snapshot, SHA `12f6881e…2fe771` verified), and the 0902 bundle. GPU T4×2, Internet Off.

**How the full notebook is executed:** M35-B reads
`reference/biohub-competition-solution.ipynb`, applies an **output-redirection-only**
patch (see below), writes the patched copy to `/kaggle/working/m35_b_execution/
reference_repro.ipynb`, materializes the reference repo to `/kaggle/working/tracking_repo`,
and executes the patched notebook in a **fresh `nbclient` python3 kernel** with
`cwd=/kaggle/working/tracking_repo` and a 5-hour timeout. The notebook performs the
entire pipeline (offline deps, D4 detection TTA, inference → 4 GEFF stores, GEFF
conversion, motion relink, gap-1 close, safe divisions, short-track filter, linefit,
final CSV). Execution then compares the freshly generated
`/kaggle/working/m35_b_reference_reproduced.csv` (never `submission.csv`, never submitted)
to `evidence/submission.csv` by byte SHA256, canonical table (per-dataset counts,
node/edge/division, coordinates within tolerance, edge-set equality on
`(dataset, source_id, target_id)`), and graph invariants.

**Output-redirection-only patch (allowlist).** The only permitted change is redirecting
the two output-target literals (`/kaggle/working/submission.csv` →
`m35_b_reference_reproduced.csv`, and the run-stats target). A per-cell patch report
records the changed cells with original/patched text and SHA256. If any non-output cell
differs, the run fails `PROVENANCE_FAILURE` — scientific parameters/logic are never touched.

**Anti-cheating provenance** (a PASS is impossible without it): stale outputs deleted and
`execution_start_ns` recorded before the run; the reproduced file must not have existed
before, must have `mtime ≥ start`, `return code == 0`, ≥4 fresh prediction stores, a
final-CSV-write marker, no fallback; the generated file is hashed **before** the reference
is opened; and the executed code/patch is scanned for any read/copy of
`evidence/submission.csv` into the output → `PROVENANCE_FAILURE` on contamination. The
reference evidence CSV is read **only after** generation, solely for comparison.

Recommendation ∈ `REPRO_PASS_EXACT` / `REPRO_PASS_CANONICAL` / `REFERENCE_REPRO_MISMATCH`
/ `REFERENCE_ASSETS_NOT_ACCESSIBLE` / `RUNTIME_DEPENDENCY_FAILURE` / `INVALID_REPRODUCED_GRAPH`
/ `PROVENANCE_FAILURE`. Writes `m35_reference_0902_repro.json` (resolved paths, SHAs,
artifact verification, resolved preset, patch report + allowlist result, contamination scan,
executed cwd, return code, runtime, stdout tail, fresh prediction stores, generation
provenance, reproduced + reference fingerprints, byte + canonical + per-dataset comparisons,
graph validation, `fallback_used`, recommendation), `m35_reference_0902_repro.log`, and
`m35_b_patch_report.json`. **Confirmation: predict-only execution is no longer accepted.**

## AST-anchored output redirection + no-GPU preflight (review of 5f99999)
The real notebook does **not** contain the literal `/kaggle/working/submission.csv`; it
assigns `SUBMISSION_PATH = WORKING_DIR / "submission.csv"` (and the run-stats analog). The
patcher is therefore **AST-anchored**, not literal string replacement: it walks each code
cell's AST and redirects **only** the assignments whose single target `Name` is exactly
`SUBMISSION_PATH` or `RUN_STATS_PATH` (handling `WORKING_DIR / "…"`, literals, single/double
quotes, and optional type annotations), rewriting them to
`SUBMISSION_PATH = Path("/kaggle/working/m35_b_reference_reproduced.csv")` and
`RUN_STATS_PATH = Path("/kaggle/working/m35_b_run_stats.csv")`.

**No-GPU patch preflight** (`run_m35_b_patch_preflight` → `m35_b_real_notebook_patch_preflight.json`,
`PATCH_PREFLIGHT_PASS`/`PATCH_PREFLIGHT_FAILED`) runs **before any GPU execution** and verifies:
original notebook SHA == manifest; both assignments found; exactly two assignments patched;
patched targets point to the M35-B files; allowlist passes; scientific cells byte-unchanged.
If either assignment is not found, M35-B returns `PROVENANCE_FAILURE` /
`OUTPUT_TARGET_ASSIGNMENTS_NOT_PATCHED` and **no GPU time is spent**. Verified on a
real-structure notebook: the two exact replacements are
`SUBMISSION_PATH = WORKING_DIR / 'submission.csv'` → `SUBMISSION_PATH = Path('/kaggle/working/m35_b_reference_reproduced.csv')`
and `RUN_STATS_PATH = WORKING_DIR / 'run_stats.csv'` → `RUN_STATS_PATH = Path('/kaggle/working/m35_b_run_stats.csv')`
(2 assignments, cell 0, allowlist ok, scientific cells unchanged).

**GEFF-store counting** now counts fresh `predictions/**/*.geff` **stores** (each `.geff` may be
a directory — matching paths are counted, with a recursively-derived newest-child mtime after
`execution_start_ns`), **requires exactly 4**, and records `geff_paths` / per-store mtime +
dataset / `geff_count_expected=4` / `geff_count_match`. It no longer counts `split_0`
directories. The stdout message `Saved 4 predictions to …` is parsed only as supporting
evidence; the GEFF count is authoritative.

**Provenance honesty:** the hardcoded `hashed_before_reading_reference=True` is removed. The
report now records the real operation order — `audit_reference_read_before_generation`,
`reference_accessible_to_generation_kernel=False` (the kernel never receives the evidence path),
`generated_sha_computed_before_comparison_read=True`, `generation_notebook_references_evidence`
(the contamination-scan result).

## Reused from M34 (only)
Exact D4 XY geometry + inverse transforms + roundtrip tests; one-to-one Hungarian
canonicalization + order-invariance; no-link primary assignment; independent division
assignment; graph validation. **Not reused:** the M19-C baseline config/fingerprint,
TTA4-as-primary, or the assumption that detection TTA is missing (the 0902 reference has
D4 detection TTA). The post-fusion chain requires ops resolved from the audited bundle.

## Honesty guarantees (static-checked)
No hardcoded submission rows; no fabricated reproduction (M35-B invokes `subprocess`);
no Kaggle-API submit; `local_metric.py` never presented as the official metric (only
`extracted/tracking_repo/src/biohub_tracking/metrics.py` is authoritative); no
modification of M16–M34; no claim or guarantee of 0.975; the 0.902 is recorded
USER-OBSERVED, separate from independently verified Kaggle evidence; the M19-C
fingerprint (133106/121718) is not reused.

## Tests (20/20)
Geometry roundtrips + TTA8 safety; fusion one-to-one / order-invariance / no-link /
independent division; **exact path resolution + manifest SHA verification**;
**dataset-scoped node IDs**; manifest SHA-mismatch → FAIL; **isolated missing-bundle**;
downstream isolated block; **exact/canonical/mismatch comparison**; **real M35-B
execution (exact/canonical/mismatch)**; **invalid-graph rejection**; **M35-B
not-hardcoded (data-dependent across fixtures)**; M35-B blocks without bundle without
fabricating a fingerprint; preset/fingerprint targets; no-local-metric / no-0975.

## Status ledger
M19-C 0.880 historical; M29-A 0.876 failed; M30-C pending; M31 blocked; M32/M32.1
unresolved; M33 operationally paused; M34 A–D superseded. 0.902 recorded user-observed.
M35-A diagnostic (PASSED on Kaggle); M35-B real, blocked off-environment; M35-C/D/E
blocked pending a genuine M35-B pass. **Exact first action on Kaggle: run M35-A, then
M35-B; do not build a submission before M35-B genuinely passes.**
