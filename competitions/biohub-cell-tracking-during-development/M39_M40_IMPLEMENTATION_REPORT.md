# M39/M40 Implementation Report — Static Scaffolding Cycle

**Scope:** static scaffolding only. This sandbox has no GPU, no torch, no
Kaggle mount, and no vendored official metric / HOCT source. Every value that
requires the real Kaggle GPU environment is emitted with the sentinel
`PENDING_REAL_ENV`. Nothing has been submitted to Kaggle. M19-C remains the
selectable fallback.

## Branch and commits

- Branch: `claude/biohub-submission-pipeline-4pggio` (session policy).
- Parent of this cycle: `03c50a6…` (M38 hidden-rerun defect patch).
- Commits produced this cycle:
  1. **`4e4bc9f…` — M19-C immutable archive.** Byte-preserved copies of the
     four M19-C artifacts under `archive/m19c/` + `MANIFEST.sha256` + README.
     The manifest verifies at commit time.
  2. **(this commit) — M39/M40 scaffolding + tests + report + artifacts.**
     Adds the `m39_m40/` package and the pending artifacts.

## What is in this cycle

### Directory layout

```
competitions/biohub-cell-tracking-during-development/
  archive/m19c/                 # commit 4e4bc9f — do not edit
  m39_m40/                      # (this commit) scaffolding package
    __init__.py                 # exports PENDING_REAL_ENV sentinel
    runtime_guard.py            # vendored biohub_runtime_guard.py (SHA-verified)
    schemas.py                  # CV, summary, runtime, metric, split, HOCT schemas
    inventory.py                # M39 Step-0 repo inventory generator
    metric_vendor.py            # M39-A official-metric vendor/audit scaffold
    split_audit.py              # M39-B split-audit scaffold
    offline_check.py            # offline dependency import check
    hoct_adapter.py             # M40-A HOCT schema-only adapter, real inference BLOCKED
    kaggle_runner.py            # Kaggle real-env runner + preflight + fallback
    tests/                      # 74 CPU-only unit tests
      test_runtime_guard.py
      test_schemas.py
      test_hoct_adapter.py
      test_kaggle_runner.py
      test_inventory_and_cli.py
      test_timer_fallback_simulation.py
  artifacts/                    # (this commit) generated artifacts
    m39_repo_inventory.json         # honest env facts, SHA of every M-file
    m39_official_metric_audit.json  # PENDING_REAL_ENV (no vendored scorer)
    m39_split_audit.json            # PENDING_REAL_ENV (no manifest offline)
    m39_split_membership.csv        # empty header only when audit is PENDING
    m39_offline_dependency_check.json
    m39_cv_results.csv              # 5 PENDING placeholder rows
    m39_cv_summary.json             # PENDING, gates_pass=False
    m40_hoct_shadow_report.json     # PENDING scaffold
    runtime_report.json             # template runtime report (marked scaffold)
  M39_M40_IMPLEMENTATION_REPORT.md  # this file
```

### Runtime governor (M39-D)

- Vendored verbatim from the M39/M40 start pack (SHA verified after copy).
- 75/90/105/110/120/10 policy encoded in `kaggle_runner.DEFAULT_POLICY`.
- Degradation ladder tested at fake-clock GREEN → AMBER (0.68) → RED (0.82)
  → FALLBACK (0.91) → EXPIRED and via a tiny 60-second budget end-to-end.
- Finalization-reserve gate: `can_start` and `start_phase` refuse to begin a
  new phase when the estimated duration plus the 10-minute reserve would
  cross the 105-minute engineering cap. `start_phase` raises `TimeoutError`.

### Official-metric vendor/audit (M39-A)

- Scans the vendored roots (`reference/official_metric`,
  `reference/tracking_cellmot`, `vendor/tracking_cellmot`).
- Detects any `local_metric.py` inside the vendored root and hard-rejects the
  audit (test coverage).
- Records physical-scale constants (7.0 µm match distance, z=1.625, y=x=0.40625).
- All `reproduces_*_jaccard`, `synthetic_case_matches`, and `status` fields
  remain `PENDING_REAL_ENV` until the real Kaggle environment supplies a
  vendored scorer and runs the three synthetic reproduction cases.

### Split audit (M39-B)

- Scans candidate manifests (`reference/kaggle_test_splits_50ep.json`,
  `reference/splits/split_0.json`, membership CSVs).
- Refuses to reconstruct from a random seed.
- Distinguishes `split_0` (proven) from `diagnostic_LODO_*`.
- In this sandbox: verdict `PENDING_REAL_ENV` because no manifest is
  present offline. CLI exit code is non-zero for PENDING; only OK_PROVEN /
  OK_DIAGNOSTIC exit 0.

### Offline dependency check

- Honest import status per module. Currently 8/18 importable — `torch`,
  `torchvision`, `tracksdata`, `geff`, `tracking_cellmot`, `hoct`, `gurobipy`,
  and `pulp` are missing here. The Kaggle real-env runner refuses to start
  the experimental path unless all HOCT-critical modules are importable.

### HOCT adapter (M40-A)

- Schema-only. Every entry point that would touch the model
  (`build_hoct_graph`, `run_shadow_inference`) raises `HoctBlockedError`
  unless `check_prereqs()` reports no missing prerequisites AND
  `probe_hoct_source()` reports `used_official`.
- `HoctAdapterConfig` pins `max_delta_t=1`, caps `n_neighbors≤5`, forbids
  TTA and long-gap passes (test coverage).
- `probe_hoct_source()` uses `inspect.getsource` + a documented stub
  heuristic to detect an unimplemented `create_graph_from_points` (test
  covers a real synthetic stub via `sys.path`).
- Node / edge schemas explicitly list the required standardized features
  (detection score, physical coordinates, mask stats, source/target detection
  scores, displacement in µm, delta_t=1). Missing-feature and reversed-time
  edges are rejected.

### Kaggle real-env runner (M39-D / M40-A driver)

- `preflight_report()` verifies mounted competition inputs, reference
  bundle, weight file, torch importability, CUDA availability, HOCT source
  status, and M19-C archive manifest match.
- Verdict is `FALLBACK_ONLY` if any check fails; only `READY_EXPERIMENTAL`
  proceeds to the experimental path.
- `fallback_to_m19c(dry_run=True)` returns the archived one-cell path so
  Kaggle can regenerate `submission.csv` from the byte-preserved M19-C.
  In `dry_run=False` from a non-Kaggle environment the fallback REFUSES
  to fake the regeneration and returns
  `reason=fallback_regeneration_requires_kaggle_mount` (test coverage).
- `run()` returns `mode=FALLBACK_ONLY` in this sandbox without touching
  `/kaggle/working`.

## Tests

**Total: 74 tests, all pass (CPU only).**

| File | Tests | Category |
|---|---:|---|
| `test_runtime_guard.py` | 17 | GREEN/AMBER/RED/FALLBACK/EXPIRED ladder, finalization reserve, deterministic rerun, phase lifecycle. |
| `test_schemas.py` | 18 | CV row, CV summary gates, runtime-report shape, metric-audit gates (incl. wrong-scale rejection), split-audit verdicts (proven / diagnostic / rejected / pending), HOCT shadow gates. |
| `test_hoct_adapter.py` | 20 | Adapter config validation, node/edge schemas (missing fields, wrong delta_t, reversed time), stub-heuristic direct + probe classification of a fake installed stub as `stub_detected_blocked`, prereq missing list, real-inference entry points blocked. |
| `test_kaggle_runner.py` | 7 | Default policy matches spec, preflight FALLBACK_ONLY in sandbox, tampered-archive detection, fallback refuses to fake regeneration, `run()` returns fallback without raising. |
| `test_inventory_and_cli.py` | 8 | Inventory shape + counts + honest env facts, `metric_vendor` rejects local_metric fallback, split_audit PENDING when no manifest + verified when explicit manifest + leakage detected, offline check missing-module reporting. |
| `test_timer_fallback_simulation.py` | 4 | Full degradation ladder walk with a 60-second budget, `can_start` blocks near cap, `experimental_seconds_available` never negative, `run()` never raises in sandbox. |

All tests are deterministic (fake clocks, tempdirs, no network). Every run
gives byte-identical output.

### PASSED_LOCAL_STATIC vs PENDING_REAL_ENV

**PASSED_LOCAL_STATIC** (can be trusted from this cycle):

- Runtime guard degradation ladder, finalization reserve, phase lifecycle,
  deterministic-rerun equality.
- All schema validators (CV rows, CV summary, metric audit, split audit,
  runtime report, HOCT shadow).
- HOCT adapter config validation, node/edge schema validation, stub heuristic,
  prereq gate, blocked entry points.
- Kaggle runner default policy, preflight verdict in sandbox, tampered-archive
  detection, fallback dry-run reporting.
- Inventory generator (files counted, SHAs recorded, M19-C manifest verified).
- Metric-vendor rejection of any embedded `local_metric.py`.
- Split-audit PENDING vs verified vs leakage-detected verdicts.

**PENDING_REAL_ENV** (must be produced in Kaggle):

- Official-metric SHA / license / import-under-Internet-off / reproduces_*_jaccard.
- Split-audit against the real training manifest.
- Actual CV per-dataset scores, per-config totals, macro & aggregate scores,
  runtime and peak memory.
- HOCT source SHA, weight SHA, `create_graph_from_points` status
  (`used_official` / `vendored_adapter` / `stub_detected_blocked`).
- Baseline vs HOCT shadow: edge overlap, division precision/recall, official
  score delta, runtime delta, peak memory delta.
- Runtime report from a real GPU run — phase durations, memory, mode transitions.

## Pre-existing changes left untouched

Working tree was completely clean at cycle start (HEAD `03c50a6`, no
uncommitted files, no untracked files). No pre-existing changes existed to
preserve; nothing was silently rolled into the M39/M40 commits.

## Blockers before M40-A can produce a Kaggle-safe submission

1. Vendor the official `tracking_cellmot` scorer at a pinned commit; ship
   the source + license file in the reference bundle (Internet Off).
2. Locate the exact `split_0` manifest or checkpoint metadata for the
   controlled model; without it M39-B stays PENDING and any CV comparison
   is labeled diagnostic.
3. Install the official HOCT source and the `general_v0` weight file in
   the Kaggle inputs; `create_graph_from_points` must not be a stub.
4. Cache detector outputs for at least the CV split so the M40-A shadow
   comparison can reuse identical detections between baseline and HOCT.

## Required Kaggle inputs (for the real-env runner)

- `/kaggle/input/competitions/biohub-cell-tracking-during-development/`
  (test store + sample_submission).
- `/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle/`
  including the audited notebook AND
  `weights/unet_transformer/split_0/edge_predictor_best.pth`.
- Vendored `tracking_cellmot` under `reference/official_metric/` (or
  `reference/tracking_cellmot/` / `vendor/tracking_cellmot/`), with LICENSE.
- Vendored `hoct` package + `general_v0` weights.
- CUDA GPU (Kaggle P100/T4/L4/A100 all fine; the runner only asks for
  `torch.cuda.is_available()`).

## Next command in the real Kaggle GPU environment

Paste into a single Kaggle code cell (accelerator: GPU, Internet: Off):

```python
import sys
sys.path.insert(0, "/kaggle/input/datasets/USER/kaggle-agent/competitions/biohub-cell-tracking-during-development")
from pathlib import Path
from m39_m40.kaggle_runner import preflight_report, run
from m39_m40 import inventory

archive_dir = Path(sys.path[0]) / "archive/m19c"

# Step 0 — orientation
inv = inventory.build_inventory(Path(sys.path[0]))
print("kaggle_mount:", inv["env"]["kaggle_mount_present"],
      "cuda:", inv["env"]["cuda_available"],
      "m19c_manifest_ok:", inv["m19c_archive"]["manifest_ok"])

# Preflight
pf = preflight_report(archive_dir=archive_dir)
print("VERDICT:", pf.verdict, pf.reasons)

# Only proceed if READY_EXPERIMENTAL. Otherwise fall back to M19-C by
# exec'ing archive/m19c/M19_VARIANT_C_FULL_CHAIN.txt.
if pf.verdict == "READY_EXPERIMENTAL":
    result = run(archive_dir=archive_dir)
    print("mode:", result["mode"])
else:
    # M19-C fallback
    exec(Path(archive_dir / "M19_VARIANT_C_FULL_CHAIN.txt").read_text())
```

This scaffold NEVER auto-submits to Kaggle. To submit, upload the produced
`/kaggle/working/submission.csv` from the Kaggle UI after the notebook run
completes (or use `kaggle competitions submit` yourself; the runner does not).

Do NOT claim M40-A improves the public score without official-CV evidence
from the vendored `tracking_cellmot` scorer under the audited split_0.
