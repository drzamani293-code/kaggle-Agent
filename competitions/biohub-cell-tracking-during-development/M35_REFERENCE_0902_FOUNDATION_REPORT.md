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
- **M35-C = `M35_C_REAL_NOTEBOOK_EVENT_INSTRUMENTATION_PASS` + `M35_C_GAP_MATERIALIZED_ROW_EXPORT_PASS`
  + `M35_C_RUNTIME_SCALE_SAFETY_PASS` + `M35_C_MOTION_FEATURE_COMPLETENESS_PASS`
  (Rev12 event semantics, Rev13 gap actual-edge export, Rev14 event-level audit + gap-specific
  recall + non-vacuous gates, Rev15 scale-safe recorder + motion feature completeness) +
  `M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING` → `M35-C = BLOCKED_NOT_YET_RUN`.** Rev15 makes the
  postprocess recorder production-safe: out-of-gate Cartesian pairs (est. **65.1 M** motion
  tight pairs → the old per-pair-event architecture projects **~189 GB**, well past Kaggle's
  ~13 GB) update only a compact counter/histogram; a feasible motion candidate is created only
  at `cost_computed` (in-gate), with a hard `motion_feature_completeness_100` gate; the gap
  matrix aggregates out-of-threshold cells instead of one event per `d[i,j]`; candidate rows
  are written incrementally to disk with streaming recall/key checks; and an 800×800 stress
  probe proves state stays ≤ in-gate pairs with **zero** out-of-gate event objects. Each accepted gap bridge exports its TWO materialized graph
  edges (`source→middle`, `middle→target`) as derived `materialized_edge` rows alongside the
  preserved abstract `bridge_proposal` row (actual edges read from the SEPARATE
  `first_edge_added`/`second_edge_added` records, identity-validated, uniquely keyed). Rev14
  makes the gates non-circular: an EVENT-LEVEL gap audit (computed before candidate filtering)
  fails a one-edge / duplicate / conflicting bridge; production requires genuine gap bridges
  (`bridges_with_both_added_events > 0`, `materialized_rows > 0`); and a GAP-SPECIFIC recall
  over `materialized_edge` rows only (never satisfiable by motion/safe) must be 100%. The
  instrumentation is verified against the SUPPLIED audited notebook (`beb17b03…`, held
  locally in git-ignored `.local_reference/`). Rev11 achieved the static-AST match; Rev12
  corrects two event-semantics defects (gap `*_edge_added` must fire AFTER
  `new_edges.extend([e1, e2])`, not at the `e1`/`e2` dicts; safe-division `accepted` must
  fire AFTER the cap/conflict gates, not at the sorted-proposals loop top). On the real
  cell 4: `gap_edge_dict_e1 = gap_edge_dict_e2 = gap_extend = 1`, added events ordered
  after the extend; `safe_div_selected / global_cap_rejected / frame_cap_rejected /
  target_conflict_rejected / safe_div_acceptance = 1` each; `binding_errors = []`,
  `no_frame_t_in_ids = true`, `dataset_propagation_patch_count = 2`,
  `semantic_selftest.passed = true`. The four mounted `.zarr` dataset stems remain **not**
  verifiable locally, so the full `M35_C_STRUCTURAL_PREFLIGHT_PASS` is attainable **only**
  on Kaggle with the mounted competition `test/`. No candidate export exists; none is claimed.

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

*Rev8 — exact audited cell-4 structure.* Corrections after the real-notebook facts:
`motion_relink_edges` has **no** `dataset` param in the original — the instrumented
**copy** adds a keyword-only `dataset=None` and patches the `filter_output_graph` call
to `motion_relink_edges(..., dataset=dataset)` (`dataset_propagation_patch_count`); the
preflight checks the **patch**, not the original signature. Instrumentation now uses
**exact semantic site specs** (not every loop): motion pair-evaluation (`for i,source_id`
/ `for j,target_id`), motion Hungarian (`for r,c in zip(row_ind,col_ind)`), motion
addition (`for pass_name,gate_um`); gap matrix (`for i,ep` / `for j,sp`), gap Hungarian +
addition (`for r,c`); safe-division pair (`for candidate_id`) + acceptance
(`for _,source_id,candidate_id,parent_dist,_ in proposals`); `filter_output_graph` stays
survival-only. `dataset` is resolved across enclosing frames (`_m35c_find_dataset`) so a
snapshot inside nested `assign_pass` still gets it; numpy scalars are retained and
normalised. The preflight now reports **distinct** counts —
`motion_pair_evaluation / motion_hungarian / motion_addition / gap_matrix_pair /
gap_hungarian / gap_addition / safe_div_pair / safe_div_acceptance` +
`dataset_propagation_patch_count` — and requires every one `> 0`, all **three** cell-1
assignments patched (`REPO_DIR`/`SUBMISSION_PATH`/`RUN_STATS_PATH`), and exactly the four
`.zarr` test stems. **The audited notebook (`beb17b03…`) is not mounted in the authoring
environment**, so the real-notebook `PASS` and its genuine site counts can only be
produced by running the preflight on Kaggle; a dedicated test runs the preflight against
the real notebook when present and asserts SHA + `PASS`.

*Rev9 — explicit event architecture.* Generic locals-snapshots are replaced by an
**explicit event recorder**: `M35CEventRecorder.create_or_update(evaluation_id,
event_type, …)` appends to an ordered **event log** and maintains **one consolidated
state per evaluation_id**. Candidate identity is never inferred from arbitrary locals —
the injected call passes the **explicitly-derived** source/target; missing/`ALL` dataset
or (for identity events) missing source/target are hard errors; **conflicting identity
transitions are recorded**; numpy+Python scalars are normalised. Instrumentation is now
**statement-level**: motion pair after `raw=…` (+ `gate_rejected` before the continue,
`cost_computed` after `cost[i,j]=…`), motion Hungarian **derives**
`source_ids[int(r)]`/`target_ids[int(c)]` and records `cost[r,c]`/`raw_dist[r,c]`/
`motion_dist[r,c]`/`prob_matrix[r,c]`, `edge_added` at the append; gap matrix after
`d[i,j]=…` derives `end_ids[int(i)]`/`start_ids[int(j)]`, gap Hungarian derives
`end_ids[int(r)]`/`start_ids[int(c)]`; safe-division pair after the gate assign +
acceptance loop. Events for one proposal share ONE `evaluation_id`
(`dataset|motion|frame_t|pass_name|src|tgt`, `dataset|gap|gap|src|tgt`,
`dataset|safe_division|src|candidate`). The no-GPU preflight now validates **both** (a)
STATIC AST — the patched real cell 4 contains the required derivation expressions per
site — and (b) a **deterministic semantic self-test** that instruments+**executes** an
exact-shape fixture and asserts the one-eval_id lifecycle, rejected+accepted presence,
matrix-pairs-become-candidates, additions↔append statements, no stale/conflicting IDs,
no missing dataset/source/target, and known feature values.
`M35_C_STRUCTURAL_PREFLIGHT_PASS` requires **both**. The audited notebook (`beb17b03…`)
is **not mounted** in the authoring environment, so the real-notebook `PASS` and genuine
per-statement counts can only be produced by running the preflight on Kaggle.

*Rev10 — keyed to the real cell-4 source.* (1) Safe-division uses the real vars
`child_dist`/`parent_dist`/`sister_dist` with `SAFE_DIV_EXISTING_CHILD_MAX_UM`/
`SAFE_DIV_MAX_UM`/`SAFE_DIV_SISTER_MAX_UM` (the eval is created after
`parent_dist = edge_distance_um(…)`, updated after `sister_dist = …`, with
`gate_rejected` before the real continues and `accepted`/`edge_added` in the
proposals loop / after `added.append({…})`). (2) The gap bridge is **two** real edges
(`source→middle_id`, `middle_id→target`): `first_edge_added` + `second_edge_added` at
the actual appends, and consolidated `added_to_graph` is set only when **both** exist.
(3) Motion `edge_added` fires **only** at `selected_edges.append({…})` (dict with
`source_id`/`target_id`/`distance_um`); `frame_matches.append` emits `accepted`.
(4) The motion `evaluation_id` uses the real frame var **`t`**
(`dataset|motion|t|pass_name|src|tgt`); no production id contains `frame_t` or `|None|`.
(5) Full Hungarian lifecycle: `if cost[r,c] >= big` (motion) / `if d[r,c] > threshold_um`
(gap) → `assignment_rejected` + continue, else `hungarian_selected`. (6) The preflight
adds **AST scope/binding validation** (rejects injected names not definitely-assigned on
the enclosing path) and requires `safe_div_pair`, `gap_addition_first_edge`,
`gap_addition_second_edge`, `motion_selected_edges_append`, `motion_frame_matches_append`
all `> 0`, plus no-`frame_t`; it runs the static instrumentation against the real
notebook, and a real-notebook test asserts SHA `beb17b03…` + `PASS` when the file is
present. The audited notebook is **not mounted** here, so the real `PASS` still requires
running the preflight where the notebook lives.

*Rev11 — verified directly against the SUPPLIED real audited notebook.* The audited
notebook was provided and verified locally (`.local_reference/`, git-ignored, never
committed): SHA256 **`beb17b03682231460c3adab6069815e06c44adc23e427371c364901f8de5437e`**
— exactly **five** code cells (env / config / dependency / inference / postprocess).
Running the **unmodified** M35-C AST/semantic instrumentation against the **real** cell 4
first surfaced a genuine mismatch: `gap_addition_first_edge = 0` and
`gap_addition_second_edge = 0`. **Root cause:** the real `close_single_frame_gaps` does
not append tuples — it builds two **dict edges** `e1 = {"source_id": source_id,
"target_id": middle_id, …, "distance_um": edge_distance_um(source, middle)}` and
`e2 = {"source_id": middle_id, "target_id": target_id, …, "distance_um":
edge_distance_um(middle, target)}` and then `new_edges.extend([e1, e2])`; `middle_id` is
introduced via an `AnnAssign` (`middle_id: int | None = None`). **Fix:** the gap detector
now anchors on the dict-assignment whose `target_id` value is the Name `middle_id`
(→ `first_edge_added`, source→middle) and whose `source_id` value is `middle_id`
(→ `second_edge_added`, middle→target), and the binding tracker recognizes `AnnAssign`
targets; the deterministic self-test fixture was updated to the identical dict-edge shape.
**Verified result on the REAL cell 4** (unmodified preflight logic): `instrumented_source_
compiles = true`; all required real-source sites found; `gap_addition_first_edge = 1`,
`gap_addition_second_edge = 1`; `binding_errors = []`; `binding_ok = true`;
`no_frame_t_in_ids = true`; `dataset_propagation_patch_count = 2`;
`semantic_selftest.passed = true`; `reasons = []`. **49/49** tests pass (the real-notebook
test now instruments `.local_reference/`'s real cell 4 directly); `py_compile` passes.

**Scoped status — do NOT read this as an unconditional full pass.** The four mounted
Kaggle `.zarr` dataset stems (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`,
`6bba_05db0fb1`) were **not** verifiable locally (no mounted competition `test/`).
Therefore the recorded status is:
- **`M35_C_REAL_NOTEBOOK_AST_SEMANTIC_PREFLIGHT_PASS`** — the real cell-4 AST + semantic
  instrumentation passes against the audited `beb17b03…` notebook.
- **`M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING`** — the exact four `.zarr` stem check has not
  been run.
- **`M35-C = BLOCKED_NOT_YET_RUN`** — no candidate export exists; none is claimed.
The full production preflight may return `M35_C_STRUCTURAL_PREFLIGHT_PASS` **only** after
running on Kaggle with the mounted competition `test/` directory and confirming the exact
four `.zarr` stems. No GPU inference, no full M35-C, no M35-D/E were run; no candidate
export success and no `0.975` are claimed.

*Rev11 reporting correction.* The Rev11 commit (`9cef669`) changed **13** files (not 12).
Its accurate scoped status is **`M35_C_REAL_NOTEBOOK_STATIC_AST_MATCH_PASS`** — the static
AST match on the real cell 4 passed — with **`M35_C_EVENT_SEMANTICS_PENDING_FIX`**: two
event-semantics defects remained (gap `*_edge_added` was emitted at the `e1`/`e2` dict
assignments rather than after the real `new_edges.extend([e1, e2])`; safe-division
`accepted` was emitted at the top of the sorted-proposals loop, before the cap/conflict
gates, falsely marking rejected proposals accepted). Do not read Rev11 as a full
event-semantics pass.

*Rev12 — event semantics corrected on the real source.* (1) **Gap bridge edges enter the
graph only at the extend.** The `e1`/`e2` dict assignments now emit `first_edge_prepared`
/ `second_edge_prepared` (never `*_edge_added`); a new exact AST matcher for
`new_edges.extend([e1, e2])` (receiver `Name('new_edges')`, method `extend`, a
List/Tuple of exactly `Name('e1')`, `Name('e2')`, all of `e1`/`e2`/`source_id`/`target_id`
/`middle_id` bound) emits `first_edge_added` (`actual_edge_source_id=source_id`,
`actual_edge_target_id=middle_id`) and `second_edge_added` (`actual_edge_source_id=
middle_id`, `actual_edge_target_id=target_id`) **immediately after** the extend. The
consolidated bridge is `added_to_graph=True` only when **both** post-extend events exist.
The preflight requires `gap_edge_dict_e1 == 1`, `gap_edge_dict_e2 == 1`, `gap_extend == 1`
and verifies the `*_edge_added` calls occur after the extend and the `*_edge_prepared`
calls before it. (2) **Safe-division acceptance is post-gate.** Entering
`for _, source_id, candidate_id, parent_dist, _ in proposals:` emits only
`selected_by_sorted_order`; `global_cap_rejected` / `frame_cap_rejected` /
`target_conflict_rejected` are emitted before the respective `break`/`break`/`continue`
(matched on the enclosing `if` test); `accepted` is emitted only immediately before the
real `added.append({...})` and `edge_added` immediately after. Raw candidate logic:
`accepted_by_assignment = accepted OR edge_added` — entering the sorted loop never implies
acceptance. (3) The deterministic semantic self-test now includes a safe-division proposal
accepted+added, one rejected by global cap, one by frame cap, one by target/incoming
conflict, and a gap bridge whose `e1`/`e2` are prepared but whose extend is deliberately
skipped; it asserts cap/conflict proposals carry `selected_by_sorted_order` without
`accepted`/`edge_added`, the skipped bridge has prepared-only edges and `added_to_graph`
False, a real extend yields both `*_edge_added` with `added_to_graph` True, no identity
conflicts, and `binding_errors == []`. Verified on the real cell 4 (unmodified preflight
logic): all gap/safe sites present with the exact counts above, ordering correct,
`semantic_selftest.passed = true`, `binding_errors = []`. **49/49** tests pass;
`py_compile` passes. Status unchanged: **`M35_C_REAL_NOTEBOOK_STATIC_AST_MATCH_PASS`** (now
with correct event semantics) + **`M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING`** →
**`M35-C = BLOCKED_NOT_YET_RUN`**. No GPU inference, no full M35-C, no M35-D/E; no
candidate-export success and no `0.975` claimed.

*Rev12 status correction.* Rev12's accurate scoped status is
**`M35_C_REAL_NOTEBOOK_EVENT_INSTRUMENTATION_PASS`** (the post-extend gap events and
post-gate safe-division acceptance are correct) with **`M35_C_GAP_ACTUAL_EDGE_EXPORT_PENDING_FIX`**:
an accepted gap bridge still exported only the abstract proposal, not its two materialized
edges, so exact final-edge recall could not reach 100%. Rev13 resolves that.

*Rev13 — gap bridge ACTUAL-edge export (exact final-edge recall).* An accepted gap
bridge was previously exported only as the abstract proposal `source_id → target_id`,
while the exact final graph materializes **two** edges `source_id → middle_id` and
`middle_id → target_id`, so exact final-edge recall could never reach 100%. Fix: (1) the
consolidated **`bridge_proposal`** row is preserved unchanged (`candidate_role =
"bridge_proposal"`, keeping the original `source_id`/`target_id`/`middle_id` and gap
`evaluation_id`); (2) for every accepted bridge whose event log contains **both**
`first_edge_added` and `second_edge_added`, two derived **`materialized_edge`** rows are
emitted — Row A `gap_first_materialized_edge` (actual source→middle) and Row B
`gap_second_materialized_edge` (middle→target) — never replacing or collapsing the bridge
row. (3) The actual edges are read from the **separate** `first_edge_added` /
`second_edge_added` records in `M35CEventRecorder.events` (never the consolidated
`features`, which the second event's `features.update` overwrites), and identity-validated
(first `actual_source == bridge source_id`, `actual_target == middle_id`; second
`actual_source == middle_id`, `actual_target == bridge target_id`); conflicting/incomplete
bridges are rejected. (4) The candidate schema adds `candidate_role`, `evaluation_id`,
`bridge_evaluation_id`, `bridge_source_id`, `bridge_target_id`, `middle_id`,
`actual_edge_index`; the 7-component raw key stays unique via distinct `proposal_stage` +
`proposal_ordinal`. (5) `accepted_by_assignment` is now origin-aware — motion-relink /
safe-division = `accepted OR edge_added`; gap-close = **both** `first_edge_added` AND
`second_edge_added` — so a materialized bridge is never `accepted_by_assignment = False`
with `added_to_graph = True`. (6) New production gates require
`gap_materialized_edges_complete` (exactly two edge rows per accepted bridge),
`gap_materialized_edge_identity_valid`, `gap_materialized_edge_keys_unique`, and
`gap_final_edge_recall_100`; `CANDIDATE_EXPORT_PASS` is impossible when an accepted bridge
has zero/one/conflicting actual-edge rows. (7) A new deterministic test builds one accepted
bridge (S=0, M=100, T=10) with exact final edges `(0,100)`, `(100,10)` and asserts
postprocess proposal recall and combined final-edge recall = 1.0, the presence of
`(0,100)`@`gap_first_materialized_edge` and `(100,10)`@`gap_second_materialized_edge`, the
abstract bridge `(0,10)` present with `final_edge_present = False`, and — for a skipped
extend — the bridge proposal present with no materialized rows and `added_to_graph = False`.
**50/50** tests pass; `py_compile` passes. This remains an off-Kaggle verification: the four
mounted `.zarr` stems are still unchecked (`M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING`), so
**`M35-C = BLOCKED_NOT_YET_RUN`**. No GPU inference, no full M35-C, no M35-D/E; no
candidate-export success and no `0.975` claimed.

*Rev13 status correction.* Rev13's accurate scoped status is
**`M35_C_REAL_NOTEBOOK_EVENT_INSTRUMENTATION_PASS`** + **`M35_C_GAP_MATERIALIZED_ROW_EXPORT_PASS`**
with **`M35_C_GAP_EVENT_AUDIT_AND_SPECIFIC_RECALL_PENDING_FIX`**: the production gap gates were
still circular (they started from candidate rows already filtered by gap
`accepted_by_assignment`, which itself requires both edge events, so a one-edge bridge was
never reported incomplete) and could pass vacuously (recall reused overall combined/postprocess
recall, and empty gap sets defaulted to complete/valid=True). Rev14 fixes this.

*Rev14 — non-vacuous event-level gap audit + gap-specific recall.* (1) **Event-level audit
before candidate filtering.** `M35CEventRecorder.gap_event_audit()` reads the raw event log and
reports `bridge_evaluations_seen`, `bridges_with_prepared_events`, `bridges_with_any_added_event`,
`bridges_with_exactly_one_added_event`, `bridges_with_both_added_events`, `duplicate_first_events`,
`duplicate_second_events`, `incomplete_bridge_evaluation_ids`, `conflicting_bridge_evaluation_ids`,
`all_event_bridges_complete`, `all_event_identities_valid`. For every evaluation_id with any
edge event: exactly one first and one second must exist; datasets, bridge source/target, and
`middle_id` must agree; actual endpoints must be first `bridge_source→middle`, second
`middle→bridge_target`; and the distance features must be present. A bridge with exactly one
added event fails the gate; invalid/incomplete evidence is retained (never dropped before the
gate). The audit is threaded through `export_candidates_from_reference` aux and written to the
report. (2) **No vacuous defaults.** Production requires `bridges_with_both_added_events > 0`
and `materialized_rows > 0` (gate `gap_bridges_present_nonvacuous`); an explicit `allow_no_gap`
seam exists for gap-free test fixtures and is always `False` in production. (3) **True
gap-specific recall.** `_m35_gap_specific_recall` builds the target set from the validated
ordered event edges and the candidate pool from `candidate_origin == "gap-close"` +
`candidate_role == "materialized_edge"` only; it reports `gap_expected_materialized_edges`,
`gap_candidate_materialized_edges`, `gap_final_edges_present/missing/extra`,
`gap_materialized_recall`, `gap_materialized_recall_100`, and requires every validated event
edge to be a real final edge and every expected gap edge to have exactly one materialized row.
Motion/safe candidates can never satisfy it. (4) **Materialized-edge features.** The extend
event now records `distance_um`/`edge_prob`/`gap_closed` from `e1`/`e2`; materialized rows carry
`physical_distance_um` (never None), `learned_edge_prob` when present, and `gap_closed`
provenance; these fields are part of the event identity validation. (5) **`added_final_edges()`
fixed** to return `(dataset, bridge_source, middle)` and `(dataset, middle, bridge_target)` from
the ordered event log — never the abstract `source→target`. (6) New production gates
`gap_event_audit_complete`, `gap_event_identities_valid`, `gap_bridges_present_nonvacuous`,
`gap_final_edge_recall_100` (gap-specific). (7) New test
`_m35_t_gap_event_audit_and_specific_recall` covers only-first, only-second, duplicate-first,
conflicting-`middle_id`, zero-gap-in-production, and a valid bridge (two materialized rows with
matching `physical_distance_um`, gap-specific recall 1.0, `added_final_edges` returns S→M/M→T not
S→T). **51/51** tests pass; `py_compile` passes; real-notebook static instrumentation still
`binding_ok`/`no_frame_t`/all sites+exprs present. Status unchanged
(`M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING` → **`M35-C = BLOCKED_NOT_YET_RUN`**). No GPU
inference, no full M35-C, no M35-D/E; no candidate-export success and no `0.975` claimed.

*Rev14 status correction.* Rev14's accurate status is
**`M35_C_GAP_EVENT_AUDIT_AND_SPECIFIC_RECALL_FIXTURE_PASS`** with two production-scale gaps:
**`M35_C_RUNTIME_SCALE_SAFETY_PENDING_FIX`** (the recorder emitted `pair_evaluated` +
`gate_rejected` for every motion Cartesian pair and one event per gap `d[i,j]` — at ~65 M
tight motion pairs this projects >100 M event dicts and OOMs Kaggle) and
**`M35_C_MOTION_FEATURE_COMPLETENESS_PENDING_FIX`** (feature completeness was reported, never
gated). Rev15 fixes both.

*Rev15 — scale-safe recorder + motion feature-completeness gate.* (1) **Correct candidate
universe.** Out-of-gate motion pairs (`if raw > gate_um: continue`) now update only a compact
`count_out_of_gate(origin, dataset, t, pass_name, reason, distance)` counter (+ fixed-bin
histogram) — NO evaluation_id, state dict, or candidate row. (2) **Explicit cost_computed
candidate.** A feasible motion candidate is created immediately after the real
`cost[i, j] = motion + 0.05*raw - MOTION_RELINK_LEARNED_BONUS*prob`, carrying `matrix_i/j`,
`gate_um`, `raw_distance`, `predicted_motion_distance`, `learned_probability`, `cost`,
`within_gate=True`, `cost_computed=True`; every in-gate pair (even Hungarian-unselected) gets
it, and Hungarian/acceptance/addition update the SAME evaluation_id. (3) **Motion
feature-completeness gate.** `motion_feature_report()` requires finite
`raw_distance/predicted_motion_distance/learned_probability/cost/gate_um` + non-null
`t/pass_name/source/target` for every feasible candidate; production gates
`motion_feature_completeness_100` and `motion_feasible_candidates_present` now hard-fail.
(4) **Scale-safe recorder.** Compact audit counters (out-of-gate), feasible-candidate state
only, and a DISK-BACKED incremental writer `_m35_stream_write_candidates` (compressed JSONL
partitioned by dataset/origin) with streaming recall + key-uniqueness
(`_m35_incremental_recall_and_keys`) — never `learned_raw + postproc_raw` → one giant list →
one DataFrame. (5) **Scale preflight** `run_m35_c_scale_preflight` /
`M35_C_SCALE_PREFLIGHT_NOT_SUBMIT` computes from the real per-dataset pair counts:
`motion_tight_cartesian_pairs = 65,115,238`, `estimated_raw_gate_rejections = 64,985,008`,
`estimated_in_memory_events_old_architecture = 130,230,476`, `projected_old_memory_gb ≈ 188.8`
(> ~13 GB Kaggle RAM → OOM), `streaming_candidate_state_upper_bound = 130,230`,
`projected_streaming_memory_gb ≈ 0.26`, `projected_output_rows = 195,345`,
`projected_output_disk_gb ≈ 0.063`; it runs an architecture probe and HARD-FAILS if state
scales with Cartesian pairs. (6) **Gap matrix scaling.** The per-cell `d[i,j]` event is gone;
after `cost = np.where(d <= threshold_um, d, big)` a compact out-of-threshold counter is
recorded, and feasible gap candidates come from the bounded Hungarian loop. (7) **Retention
policies** per origin (learned → disk rows; motion → feasible state + counters; gap → feasible
state + bridge event audit; safe-division → feasible state + gate transitions). New tests:
`_m35_t_motion_out_of_gate_scale` (in-gate unselected keeps full features; out-of-gate is a
counter not a candidate; sub-100% completeness / zero feasible block the gate),
`_m35_t_scale_preflight_stress` (800×800, gate 0.5 → 800 in-gate / 639,200 out-of-gate;
`n_states ≤ in_gate`, out-of-gate individual events = 0, counter exact = 639,200, tracemalloc
peak ≈ 21 MB), `_m35_t_incremental_export_recall` (streaming preserves key-uniqueness +
combined/learned/postprocess/gap recall). **54/54** tests pass; `py_compile` passes;
real-notebook static instrumentation now shows `motion_out_of_gate=1`, `motion_cost_computed=1`,
`gap_out_of_threshold=1`, all required sites+exprs present, `binding_ok`, `no_frame_t`. Status:
**`M35_C_RUNTIME_SCALE_SAFETY_PASS`** + **`M35_C_MOTION_FEATURE_COMPLETENESS_PASS`** +
**`M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING`** → **`M35-C = BLOCKED_NOT_YET_RUN`**. No GPU
inference, no full M35-C, no M35-D/E; no candidate-export success and no `0.975` claimed.

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
- **M35-C — real-notebook AST/semantic preflight PASSED locally; Kaggle dataset check
  PENDING; still `BLOCKED_NOT_YET_RUN`.** The AST + semantic instrumentation passes
  against the audited `beb17b03…` cell 4 (`M35_C_REAL_NOTEBOOK_AST_SEMANTIC_PREFLIGHT_
  PASS`), but the four mounted `.zarr` stems were not verifiable locally
  (`M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING`). No candidate export exists; none claimed.
- **M35-D/E — `BLOCKED_PENDING_M35C_PASS`.**

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
calls); each generated one-cell `.txt` is syntax-checked and self-tests 49/49 (event architecture).

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

## Tests (49/49)
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
M35-C = BLOCKED_NOT_YET_RUN** (real-notebook event instrumentation verified locally
against the audited `beb17b03…` cell 4 = `M35_C_REAL_NOTEBOOK_EVENT_INSTRUMENTATION_PASS`
+ `M35_C_GAP_MATERIALIZED_ROW_EXPORT_PASS` + `M35_C_RUNTIME_SCALE_SAFETY_PASS` +
`M35_C_MOTION_FEATURE_COMPLETENESS_PASS`; Rev12 fixed post-extend gap events + post-gate
safe-division acceptance, Rev13 added gap actual-edge export, Rev14 added the non-vacuous
event-level gap audit + gap-specific recall + gates, and Rev15 made the recorder scale-safe
(out-of-gate counters, cost_computed candidates, motion feature-completeness gate, disk-backed
streaming writer, scale preflight ~189 GB old vs ~0.26 GB streaming); the four mounted `.zarr`
stem check is `M35_C_KAGGLE_DATASET_PREFLIGHT_PENDING`; machinery genuine; still needs the
instrumented predict subprocess + postprocess re-run on Kaggle);
M35-D/E `BLOCKED_PENDING_M35C_PASS`.
**Next action on Kaggle: run the full `run_m35_c_structural_preflight` (mounted `test/`,
four exact `.zarr` stems) then `run_m35_c_production_candidate_export`; the full
`M35_C_STRUCTURAL_PREFLIGHT_PASS` is attainable only there; do not claim M35-C until real
export files exist with 100 % combined recall; do not run M35-D/E; no 0.975 guarantee.**
