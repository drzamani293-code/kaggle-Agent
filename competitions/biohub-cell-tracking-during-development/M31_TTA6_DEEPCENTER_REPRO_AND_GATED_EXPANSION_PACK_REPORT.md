# M31 — TTA6 + DeepCenter Reference-First Reproduction & Gated Expansion

**Goal:** reproduce the `analysis_09_strategy.md` near-0.900 pipeline (400ep/split_0,
6-way TTA, det 0.97, ilp_division_weight 0.7, pool kernel ~2.0µm, learned
motion_relink, one-frame gap close, conservative safe divisions, short-track
filter min_track_len 6, linefit 0.8, DeepCenter UNet3D loaded-but-inactive) —
**reference-first, no approximation**. M19-C **0.880** stays confirmed best/final
unless an observed LB score beats it.

## The governing rule: resolve-from-source-or-refuse

The exact reference values (TTA transforms, pool-kernel flag, motion_relink
`learned_bonus`/gates, DeepCenter model class / weight / call signature, predict
CLI flags) are **never invented**. Each runner **resolves them at runtime from the
mounted source** — the predict script's own flags (via `discover_predict_flags`),
any shipped reference command (`discover_reference_commands`), and a marker-scan of
the mounted inputs for the reference notebook / motion_relink source / DeepCenter
code + `.pth` weights. Every field is recorded with provenance
(`source_file` / `section` / value / **observed-vs-inferred**). The strategy
numbers live in `M31_STRATEGY_EXPECTED` (**INFERRED**) and drive only the
reference-profile checks — never fabricated into a submission. If the reference
cannot be resolved, the runner reports `REFERENCE_CONFIG_UNRESOLVED` /
`DO_NOT_SUBMIT_REFERENCE_CONFIG_UNRESOLVED` and writes a hidden-safe fallback.
**M27/M28/M29/M30 logic is never substituted for the reference pipeline.**

> In THIS environment the reference (notebook, DeepCenter code/weights, TTA/pool
> source) is **not present** and Kaggle is not mounted, so the audit honestly
> reports `REFERENCE_CONFIG_UNRESOLVED` and A/C/D/E refuse to submit. On Kaggle,
> with the reference notebook + support pack attached, the resolver binds the real
> values before anything is built.

## Runners (operational order; never submit Stage-0 / B / CV)

| # | Runner | Role |
|---|--------|------|
| 0 | `M31_REFERENCE_AUDIT_NOT_SUBMIT` | resolve every field + provenance → `AUDIT_PASS` / `REFERENCE_CONFIG_UNRESOLVED` |
| 1 | `M31_A_EXACT_09_REPRO` | exact near-0.900 reproduction from RESOLVED values; TTA must be 6 |
| 2 | `M31_B_DEEPCENTER_SHADOW_NOT_SUBMIT` | prove DeepCenter loaded + scores every candidate + output == baseline |
| 3 | `M31_C_DEEPCENTER_GATE_BASE_CAPS` | DeepCenter as an **add-only** gate (needs B passed) |
| 4 | `M31_D_DEEPCENTER_GATE_GAP2_OPEN` | C + gap2 (max gap 2, gap_close 6→7µm) |
| 5 | `M31_E_DEEPCENTER_GATE_DIVISION_RELAXED` | C + relaxed divisions (weight 1.0, wider caps) |
| 6 | `M31_LOCAL_CV_HARNESS_NOT_SUBMIT` | official-metric CV of A/C/D/E on train GT |

## Motion-relink accounting (exact set-diff, no mislabeling)

The strategy's "~123700 baseline / 120976 raw-replaced" edges are **not** a
replacement count. `motion_relink_accounting(pre, post)` computes exact set
differences and **reconciles** (`unchanged + removed == pre`, `unchanged + added ==
post`): `edges_checked`, `edges_unchanged`, `edge_intersection_count`,
`edges_removed`, `edges_added`, `edges_replaced_target` (only sources whose actual
target changed), `actual_replacement_fraction`, and `reported_reference_counter_*`
(named "raw_replaced_edges (== post edges, NOT actual replacements)"). An edge is
labelled "replaced" **only** when its source/target relationship actually changed.

## DeepCenter: shadow → add-only gate → polarity

- **Shadow (B):** the resolved model scores every gap/division candidate but
  changes nothing; the shadow output must be byte-identical to the internal
  baseline (DataFrame equality + `submission SHA` equality). Emits
  `DEEPCENTER_WEIGHTS_MISSING` / `DEEPCENTER_NOT_WIRED` / `DEEPCENTER_SHADOW_MISMATCH`
  otherwise. C/D/E do not run until B passes.
- **Gate (C/D/E):** may only **veto proposed** gap/division additions — never
  removes a base detection or base learned edge, never touches motion_relink base
  edges or short-track decisions (verified by tests). **Score polarity must be
  explicitly resolved** (`higher_is_valid`/`lower_is_valid`) or the run reports
  `DO_NOT_SUBMIT_DEEPCENTER_POLARITY_UNRESOLVED`.
- Real loading requires the resolved model class + weights + call signature; until
  those are bound from source the scorer is `None`, checked count is 0, and the
  honest status is `DEEPCENTER_NOT_WIRED`.

## Variant A reference profile

Hard: TTA count exactly 6, correct 400ep hash, all datasets present, valid graph,
all edges t→t+1, in≤1/out≤2, no dangling, no NaN, consecutive ids, no empty
dataset, `120000 ≤ nodes ≤ 138000`, `108000 ≤ edges ≤ 138000`. Soft (profile):
`125000 ≤ nodes ≤ 133000`, gap synthetic `1400–3200`, short-track nodes removed
`3500–9000`, median edge `1.1–2.3µm`, edges>7µm `≤ 20`, division rate
`0.0015–0.0075`. Hard-pass but soft-fail → `DO_NOT_SUBMIT_REFERENCE_PROFILE_MISMATCH`;
both pass → `OK_TO_SUBMIT_EXPERIMENTAL`. final_source
`tta6_motion_relink_reference_reproduced`.

## Reports & recommendation vocabulary

Each A/C/D/E runner writes `milestone31_reference_submission_report.json`,
`milestone31_reference_config.json`, `milestone31_tta_report.json`,
`milestone31_motion_relink_report.json`, `milestone31_deepcenter_report.json`,
`milestone31_artifact_selection.json`, `milestone31_failure_fallback_report.json`,
`submission.csv`. Audit/B/CV write their own JSON only. The single recommendation
vocabulary (`M31_RECOMMENDATION_VALUES`) is: `OK_TO_SUBMIT_EXPERIMENTAL`,
`AUDIT_PASS`, `REFERENCE_CONFIG_UNRESOLVED`,
`DO_NOT_SUBMIT_REFERENCE_CONFIG_UNRESOLVED`, `DO_NOT_SUBMIT_WRONG_ARTIFACT`,
`DO_NOT_SUBMIT_TTA_NOT_ACTIVE`, `DO_NOT_SUBMIT_REFERENCE_PROFILE_MISMATCH`,
`DEEPCENTER_WEIGHTS_MISSING`, `DEEPCENTER_NOT_WIRED`, `DEEPCENTER_SHADOW_MISMATCH`,
`DO_NOT_SUBMIT_DEEPCENTER_WEIGHTS_MISSING`, `DO_NOT_SUBMIT_DEEPCENTER_NOT_WIRED`,
`DO_NOT_SUBMIT_DEEPCENTER_POLARITY_UNRESOLVED`, `DO_NOT_SUBMIT_VALIDATION_FAILED`,
`DO_NOT_SUBMIT_UNSANE_NODE_COUNT`, `DO_NOT_SUBMIT_UNSANE_EDGE_COUNT`,
`DO_NOT_SUBMIT_SYNTHETIC_EXPLOSION`, `DO_NOT_SUBMIT_DIVISION_EXPLOSION`,
`DO_NOT_SUBMIT_MISSING_OR_EMPTY_DATASET`.

## Verification

- `milestone31_tta6_deepcenter_repro_runner.py` compiles; every `.txt` runs
  standalone as `__main__`; `run_milestone31_tests()` passes **20/20**: artifact
  guard fields; strategy numbers never marked observed; TTA-count-6 + transforms
  source-only; per-variant predict flags (A/C/D det 0.97, E div 1.0);
  motion-relink accounting reconciles with **zero counter mislabeling**; DeepCenter
  missing-weight guard; shadow scores>0 + output identical to baseline; gate
  changes **only** additions and never removes base nodes/edges; polarity must
  resolve; gap2 unit-timepoint chains; short-track keeps divisions; linefit
  topology; final degrees + no multiframe; reference-profile logic;
  division-explosion guard; variant registry; audit recommendation; edge/division
  stats.
- Static: no M16–M30 file modified; no static submission.csv; no runtime
  dependency on the uploaded files (resolve-from-mounted-source); standalone
  one-cell runners; dynamic dataset discovery; exact 400ep artifact guard;
  deterministic (seed-free graph ops); all outputs under `/kaggle/working`;
  top-level fallback; no Kaggle-API submit.
- intelligence_db: M19-C stays **final/best @0.880**; `analysis_09_strategy.md`
  added as a source; `M31_REFERENCE_AUDIT` + `M31_B_DEEPCENTER_SHADOW` +
  `M31_LOCAL_CV_HARNESS` non-scoring; `M31_A_EXACT_09_REPRO` pending;
  `M31_C/D/E` **blocked until B shadow passes**;
  `DEC_M31_TTA6_DEEPCENTER_REFERENCE_FIRST` recorded; M29-A / M30-C kept pending;
  `next_actions` recommends audit → A → B → C → D/E → CV. DB tests **6/6**.

**Status:** M19-C **0.880** remains the recommended final submission. Run the
reference audit first; A submits only if the reference resolves and the profile
matches; C/D/E stay blocked until B proves DeepCenter is loaded + scores > 0 +
shadow == baseline. Nothing about the reference is approximated.
