# M35 — Reference 0902 Repro, Official CV & Edge-TTA Foundation

M34 (self-contained TTA on the M19-C 0.880 baseline) is superseded by the supplied
**0.902** reference bundle. M35 reproduces and builds on that reference. M34 A–D
remain `SUPERSEDED_BY_M35_REFERENCE_0902` (files kept, not promoted).

## VERIFIED on Kaggle (this update)
- **M35-A = REFERENCE_AUDIT_PASS** (VERIFIED_PASS_ON_KAGGLE).
- **M35-B = REPRO_PASS_EXACT** (VERIFIED_REPRO_PASS_EXACT_ON_KAGGLE) via a **two-phase
  CURRENT-KERNEL** runner (`nested_kernel_used=false`, `inference_rerun=true`, 4 fresh
  GEFF stores, postprocessing ≈ 3.97 min). Generated `m35_b_reference_reproduced.csv`
  SHA256 **`8c73d776…cea698`** == reference SHA256 (byte_exact, canonical_equal);
  fingerprint **exact** (rows 252513, nodes 128511, edges 124002, divisions 417);
  graph_valid. Notebook SHA256 **`beb17b03…5437e`**; controlled 400ep weight SHA256
  **`12f6881e…2fe771`**. The public LB **0.902** is USER-OBSERVED — not independently
  verified from the Kaggle leaderboard.

The verified production runner is
`M35_B_TWO_PHASE_CURRENT_KERNEL_REPRO_NOT_SUBMIT.txt` (`run_m35_b_two_phase`). It
executes the audited notebook's cells **in the current Kaggle kernel** — never a
nested notebook kernel. The old nested-nbclient path is **deprecated**
(`_run_m35_b_nested_nbclient_deprecated`); it died with `DeadKernelError` after
inference and before postprocessing.

### Two-phase current-kernel M35-B (exact five-cell index binding)
The audited reference notebook has **exactly five code cells** with a fixed role order —
`0 environment · 1 configuration · 2 dependency · 3 inference · 4 postprocess`. Cells
are bound **strictly by index** (`_m35_bind_cells_by_index`), **not by keyword**: the
postprocess cell (4) also mentions GEFF / prediction / torch terms, so any keyword
heuristic would misclassify it. The old heuristic (`_M35_INFER_MARKERS`,
`_m35_is_inference_cell`, `_m35_last_inference_index`) is **removed from the production
path**. Production also asserts the exact audited notebook sha256
(`require_reference_notebook_sha`, `M35_REF_NOTEBOOK_SHA = beb17b03…`) and reports
`REFERENCE_NOTEBOOK_SHA_MISMATCH` / `REFERENCE_NOTEBOOK_STRUCTURE_MISMATCH` when the
notebook is not the audited 0.902 reference.
- **Phase 1:** run cells `[0,1,2,3]` (env/config/dependency **+ inference once**) in the
  current kernel, validate **four** GEFF stores, write `m35_b_phase1_checkpoint.json`,
  then **release CUDA/RAM** (`torch.cuda.empty_cache` + `gc.collect`).
- **Resume:** if four complete GEFF stores already exist, run cells `[0,1,2]` to
  reconstruct the env/config/dependency namespace, **skip only the inference cell (3)**
  (`inference_rerun=false`, `resumed_from_geff=true`), and never delete a valid
  resumable `tracking_repo` (only stale final CSVs are cleared).
- **Phase 2:** run **only** the postprocess cell (4) → write
  `m35_b_reference_reproduced.csv`, hash it **before** reading the reference, then
  compare (byte + canonical) → `REPRO_PASS_EXACT` / `REPRO_PASS_CANONICAL` / mismatch.
- Only `SUBMISSION_PATH` / `RUN_STATS_PATH` are AST-patched; the no-GPU preflight still
  gates. Never submits.

### Genuine M35-C — instrumented pre-ILP candidate export (machinery, NOT yet passed)
**Correction to the prior report:** the earlier build described M35-C as “genuine”
while `export_candidates_from_reference` still raised `RuntimeError` unconditionally and
the passing tests used an injected synthetic `candidate_provider`. That was not a
genuine export. This build removes the stub and implements the real machinery, but
**M35-C has not passed — no real Kaggle export files exist and none is claimed.**

*Why a re-run is mandatory.* The four saved GEFF stores are **post-ILP** outputs
(`predict_unet_transformer.py`: `graph = build_graph(coords, edges);
graph = solver.solve(graph); save_graph(...)`). The pre-ILP learned candidates the
solver **rejected** are therefore **not recoverable** from the saved stores — the
edge/inference stage must be **re-run with instrumentation**.

*What is now genuinely implemented (section 75).*
- `_m35_instrument_predict_source` AST-injects dump hooks **immediately after
  `graph = build_graph(coords, edges)`** (pre-ILP candidate graph, before solve) and
  **after `graph = solver.solve(graph)`** (post-ILP selected set). It **raises**
  `M35CInstrumentationError` if either anchor is absent — a stub can never masquerade
  as instrumented.
- Production `export_candidates_from_reference` (no unconditional raise) **re-runs** the
  instrumented predict script into a **separate** dir
  `/kaggle/working/m35_c_instrumented_predictions/` (never overwrites the M35-B GEFF).
  Each learned candidate is tagged `selected_by_reference_solver = edge ∈ solver output`.
- Postprocess proposals (motion-relink / gap-close / safe-division) are captured by
  `M35CProposalRecorder` + `_m35_instrument_postprocess_functions`
  (`gates_passed` / `accepted_by_assignment` / `added_to_graph` / `survived_final_filter`).
- Combined table: `candidate_key = dataset|origin|source_id|target_id|proposal_stage`
  (origins/stages **never** collapsed); every raw proposal preserved
  (`m35_c_raw_proposals.csv`). **Separate recalls:** learned pre-ILP vs post-ILP
  learned; postprocess-proposal vs postprocess-added-final; **combined vs the exact
  final edges (must be 100%)**; division.

*Production vs test seam.* `candidate_provider` is a **unit-test seam only**; the
explicit production entry point `run_m35_c_production_candidate_export` forces no
provider. Off the reference environment production returns `RUNTIME_DEPENDENCY_FAILURE`
(predict deps / anchors absent) — never fabricates candidates. **What still blocks a
real pass:** the postprocess proposal instrumentation must be **wired to the mounted
repo’s actual** motion-relink / gap-close / safe-division function names, and the
instrumented re-run must run on Kaggle. Until export files exist with 100 % combined
recall, **M35-C stays blocked and is not claimed passed;** M35-D/E stay
`BLOCKED_PENDING_M35C_PASS`. No guarantee of 0.975.

*Genuine tests (not synthetic-only).* Beyond the synthetic-provider path, the
instrumentation mechanism is unit-tested on **real fixtures**: a runnable predict
script (`build_graph` → `solver.solve`) that drops low-probability edges proves the
**ILP-rejected pre-ILP candidates are exported** with `selected_by_reference_solver=false`;
a companion test proves the **post-ILP selected set is a strict subset** of the pre-ILP
candidates (so the GEFF stores alone cannot represent rejected candidates); and a test
proves a stub without anchors **cannot** be instrumented.

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
calls); each generated one-cell `.txt` is syntax-checked and self-tests 35/35.

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

## Tests (35/35)
Geometry roundtrips + TTA8 safety; fusion one-to-one / order-invariance / no-link /
independent division; **exact path resolution + manifest SHA verification**;
**dataset-scoped node IDs**; manifest SHA-mismatch → FAIL; **isolated missing-bundle**;
downstream isolated block; **exact/canonical/mismatch comparison**; **two-phase M35-B
execution (exact/canonical/mismatch)**; **invalid-graph rejection**; **M35-B
not-hardcoded (data-dependent across fixtures)**; M35-B blocks without bundle without
fabricating a fingerprint; preset/fingerprint targets; no-local-metric / no-0975. New
this build: **exact five-cell index binding** (inference always cell 3, postprocess
always cell 4, ≠5 cells rejected, keyword heuristic gone from `globals()`); **audited
notebook sha gate** → `REFERENCE_NOTEBOOK_SHA_MISMATCH`; **instrumented predict re-run
exports ILP-rejected pre-ILP candidates**; **post-ILP set is a strict subset (GEFF
alone cannot represent rejected)**; **stub without anchors cannot be instrumented**;
**M35-C without real instrumentation → `RUNTIME_DEPENDENCY_FAILURE`** (production entry
point, no provider); **postprocess proposal recorder** (accepted/added/survived, absent
function → honest error); **5-component candidate_key + separate recalls**.

## Status ledger
M19-C 0.880 historical; M29-A 0.876 failed; M30-C pending; M31 blocked; M32/M32.1
unresolved; M33 operationally paused; M34 A–D superseded. 0.902 recorded user-observed.
M35-A + M35-B **VERIFIED on Kaggle** (`REFERENCE_AUDIT_PASS`, `REPRO_PASS_EXACT`);
M35-C machinery genuine but **not yet passed** (needs the instrumented edge/inference
re-run + postprocess-function wiring on Kaggle); M35-D/E `BLOCKED_PENDING_M35C_PASS`.
**Exact next action on Kaggle: run the M35-C production entry point
(`run_m35_c_production_candidate_export`) to re-run the instrumented edge/inference
stage; do not claim M35-C until real export files exist with 100 % combined recall.**
