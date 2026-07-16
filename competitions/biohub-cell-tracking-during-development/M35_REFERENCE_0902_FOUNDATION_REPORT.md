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
**Safe resume — the dependency cell is split.** The real dependency cell (2) ends with
`ARTIFACTS = find_artifacts_root(); ensure_dependencies(ARTIFACTS);
materialize_inference_repo(ARTIFACTS)`, and `materialize_inference_repo` →
`copy_or_extract_tree` → `remove_path(REPO_DIR)` **deletes** the preserved
`tracking_repo` and all four GEFF. So `_m35_split_dependency_cell` splits cell 2 at
`ARTIFACTS = find_artifacts_root()`:
- **Fresh:** run cell 0, patched cell 1, dependency **definitions**, then the full
  activation (incl. `materialize_inference_repo`) + inference cell 3.
- **Resume:** run cell 0, patched cell 1, dependency **definitions**, then
  `find_artifacts_root()` + `ensure_dependencies()` **only — never
  `materialize_inference_repo`** — verify the predict script + exact 400ep weight, verify
  **exactly four** GEFF, verify the four **exact** dataset names
  (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1`), **skip only the
  inference cell (3)**, then run postprocess cell 4. The valid `tracking_repo`/GEFF are
  never deleted.
- **GEFF validation requires EXACTLY four** (`!= 4` fails), not `>= 4`. Phase 1 validates
  four GEFF + the four names, writes `m35_b_phase1_checkpoint.json`, and **releases
  CUDA/RAM**. Phase 2 runs **only** cell 4 → `m35_b_reference_reproduced.csv`, hashed
  **before** reading the reference → `REPRO_PASS_EXACT` / `CANONICAL` / mismatch.
- Independent manifest comparison uses the **manifest-recorded** sha (`manifest_sha256`),
  not the recomputed `actual_sha256`. Only `SUBMISSION_PATH` / `RUN_STATS_PATH` are
  AST-patched; the no-GPU preflight still gates. Never submits.

### M35-C — instrumented candidate export (machinery; `BLOCKED_NOT_YET_RUN`)
**M35-C has not run on Kaggle. No real export files exist and none is claimed.** This
build implements the full machinery and unit-tests the mechanism on genuine fixtures.

*Why a re-run is mandatory.* The four saved GEFF stores are **post-ILP** outputs
(`predict_unet_transformer.py`: `graph = build_graph(coords, edges);
graph = solver.solve(graph); save_graph(...)`), so the ILP-**rejected** pre-ILP learned
candidates are **not recoverable** from them — the edge/inference stage is **re-run with
instrumentation**.

**(B) Subprocess instrumented predict.** `export_candidates_from_reference` builds a
**separate** `/kaggle/working/m35_c_tracking_repo` (copies the controlled artifact repo +
the exact 400ep weight; the **M35-B repo is never mutated**), AST-patches its
`scripts/predict_unet_transformer.py` to dump around
`graph = build_graph(coords, edges)` / `graph = solver.solve(graph)`, writes an
**importable** `m35c_hook` module, and **runs the exact audited predict CLI as a
subprocess** (`--data-dir <test> --splits kaggle_test_splits_50ep.json --split 0
--weights …/edge_predictor_best.pth --unet-batch-size 4 --det-threshold 0.97
--ilp-edge-weight -1.0 --ilp-appearance-weight 0.1 --ilp-disappearance-weight 0.1
--ilp-division-weight 1.0 --use-ilp`) so `main()`/`predict()` genuinely executes and the
hooks fire. Dumps land in `/kaggle/working/m35_c_instrumented_predictions/`; command,
cwd, return code, stdout/stderr, runtime and hashes are recorded. (The earlier build’s
`exec(code, g)` only *defined* the script and never invoked `main()` — fixed.)

**(C) tracksdata `InMemoryGraph` adapter.** `_m35c_iter_nodes` / `_m35c_iter_edges` read
`graph.node_attrs().iter_rows(named=True)` / `graph.edge_attrs().iter_rows(named=True)`
(`node_id, t, z, y, x` / `source_id, target_id, edge_prob, edge_dist`), with a networkx
fallback. Unit-tested against a real `InMemoryGraph`-shaped fixture.

**(D) Postprocess AST instrumentation.** `_m35_instrument_postprocess_source`
AST-instruments the **exact cell-4 functions** `motion_relink_edges`,
`close_single_frame_gaps`, `add_safe_divisions_postlink`, `filter_output_graph` at their
proposal loops (snapshot at loop top, **before every `continue`**, and at loop end) so
**evaluated-but-rejected** proposals are recorded — a return-value wrapper only sees
accepted results and is **proven insufficient** by test. The instrumented cell 4 re-runs
on the M35-B **post-ILP** GEFF in an **isolated** path and **must** reproduce the exact
final-CSV sha256 `8c73d776…fe771`; otherwise it fails with
`INSTRUMENTED_POSTPROCESS_MISMATCH`.

**(E) Provenance.** The RAW proposal table preserves **every evaluation** via a
7-component key `dataset|origin|source|target|proposal_stage|proposal_pass|proposal_ordinal`
(origins/stages/passes never collapsed).

**(F) Production gates.** `CANDIDATE_EXPORT_PASS` requires **all** of: predict subprocess
ok; real tracksdata graphs recorded; ≥1 ILP-rejected learned edge; four instrumented
datasets; **M35-B GEFF hashes unchanged**; postprocess executed; instrumented final CSV
byte-exact; learned / postprocess / combined / division recall all 100 %; unique raw
keys; output files carry SHA256; and **`candidate_provider` not used**. The
`candidate_provider` is a **unit-test seam only** and can **never** yield a pass (it
returns `CANDIDATE_EXPORT_TEST_SEAM_OK`); the explicit production entry point
`run_m35_c_production_candidate_export` forces no provider and is never run at import.

Off the reference environment, production returns `RUNTIME_DEPENDENCY_FAILURE`
(deps/data absent) — never fabricates. **M35-C stays `BLOCKED_NOT_YET_RUN`; M35-D/E stay
`BLOCKED_PENDING_M35C_PASS`. Do not run M35-D/E. No guarantee of 0.975.**

*Genuine mechanism tests.* The instrumented predict CLI genuinely executes in a
subprocess and writes dumps to `m35_c_tracking_repo` (the M35-B repo is left untouched);
the tracksdata adapter is exercised; ILP-rejected pre-ILP edges are exported; the
post-ILP set is a strict subset (GEFF-alone cannot represent rejected); rejected
motion/gap/division proposals are recorded while a return-value wrapper misses them; and
the `INSTRUMENTED_POSTPROCESS_MISMATCH` SHA gate fires on a divergent CSV.

*Rev7 corrections (still `BLOCKED_NOT_YET_RUN`).*
- **GEFF dataset = store stem.** `predictions/unknown/unet_transformer/split_0/<dataset>.geff`
  → the dataset is the `.geff` **stem** (`44b6_0113de3b` …), not the `unknown` namespace
  segment (kept as `prediction_namespace`); resume matches the four exact stems.
- **Predict `--data-dir` = the `test/` dir.** `_m35_resolve_test_dir` appends `test`
  and requires **exactly four `.zarr`** datasets before launching the subprocess.
- **Real postprocess namespace.** The instrumented cell 4 runs after executing cell 0,
  a **patched cell 1** (`REPO_DIR` → `/kaggle/working/tracking_repo`, `SUBMISSION_PATH`/
  `RUN_STATS_PATH` → isolated `m35_c` paths), the dependency **definitions** +
  `find_artifacts_root()`/`ensure_dependencies()` (**never materialize**), and
  reconstructed `test_stems`/`predict_seconds` — not an empty namespace. The four M35-B
  GEFF are hashed before/after (unchanged) and the final CSV must match the exact SHA.
- **Real dataset in every proposal** (no `ALL`/empty) — snapshots carry the function's
  `dataset` argument.
- **Semantic AST anchors** instrument only the proposal loops by loop-variable name
  (motion `source_id`/`target_id`; gap `end_id`/`start_id`; safe-division
  `source_id`/`candidate_id`); `filter_output_graph` is survival-only (never snapshotted).
- **Non-vacuous postprocess recall:** postprocess-added edges = final graph **minus**
  the post-ILP learned set; every added edge must have a recorded proposal.
- **No-GPU structural preflight** (`run_m35_c_structural_preflight`,
  `M35_C_STRUCTURAL_PREFLIGHT_NOT_SUBMIT.txt`) reports the notebook SHA, five cell
  bindings, dependency-split status, postprocess functions found, semantic anchor
  counts, output-patch status, dataset propagation, and test-dir resolution →
  `M35_C_STRUCTURAL_PREFLIGHT_PASS/FAILED`, launching **no inference**. Run it first.

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
calls); each generated one-cell `.txt` is syntax-checked and self-tests 47/47.

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

## Tests (47/47)
Geometry roundtrips + TTA8 safety; fusion one-to-one / order-invariance / no-link /
independent division; **exact path resolution + manifest SHA verification**;
**dataset-scoped node IDs**; manifest SHA-mismatch → FAIL; **isolated missing-bundle**;
downstream isolated block; **exact/canonical/mismatch comparison**; **two-phase M35-B
execution (exact/canonical/mismatch)**; **invalid-graph rejection**; **M35-B
not-hardcoded (data-dependent across fixtures)**; preset/fingerprint targets;
no-local-metric / no-0975; **exact five-cell index binding** + **audited notebook sha
gate**. New this build:
- **Safe resume:** the real materialize-deleting dependency cell does **not** destroy
  the four GEFF on resume (split cell 2, no `materialize_inference_repo`); **exactly four
  GEFF + the four exact dataset names required**; wrong-named GEFF force a fresh rerun.
- **Subprocess predict genuinely executes:** a real `python` subprocess runs the
  instrumented CLI, `main()` fires, dumps land in `m35_c_tracking_repo` (the M35-B repo
  is left untouched); the tracksdata `InMemoryGraph` adapter is exercised; ILP-rejected
  pre-ILP edges are exported; post-ILP is a strict subset.
- **Postprocess AST:** rejected motion/gap/division proposals are recorded, a
  return-value wrapper is proven insufficient, unknown cell-4 fails, and the
  `INSTRUMENTED_POSTPROCESS_MISMATCH` SHA gate fires on a divergent CSV.
- **Gates:** the test seam can never yield PASS (`CANDIDATE_EXPORT_TEST_SEAM_OK`),
  production without real instrumentation → `RUNTIME_DEPENDENCY_FAILURE`, D stays blocked
  without a genuine C pass, and `_m35_geff_hashes` detects any change.

## Status ledger
M19-C 0.880 historical; M29-A 0.876 failed; M30-C pending; M31 blocked; M32/M32.1
unresolved; M33 operationally paused; M34 A–D superseded. 0.902 recorded user-observed.
**M35-A = VERIFIED_PASS_ON_KAGGLE; M35-B = VERIFIED_REPRO_PASS_EXACT_ON_KAGGLE;
M35-C = BLOCKED_NOT_YET_RUN** (machinery genuine; needs the instrumented predict
subprocess + postprocess re-run on Kaggle); M35-D/E `BLOCKED_PENDING_M35C_PASS`.
**Next action on Kaggle: run `run_m35_c_production_candidate_export` to re-run the
instrumented edge/inference stage + postprocess; do not claim M35-C until real export
files exist with 100 % combined recall; do not run M35-D/E; no 0.975 guarantee.**
