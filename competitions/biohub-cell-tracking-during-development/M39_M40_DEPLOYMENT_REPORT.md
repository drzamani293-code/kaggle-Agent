# M39/M40 Kaggle Deployment Bundle — Implementation Report

**Scope:** self-contained Kaggle real-environment deployment for M39/M40,
built on top of the M39/M40 static scaffolding (commit `d93d08a`) and the
byte-preserved M19-C archive (commit `4e4bc9f`).

**Sandbox reality:** no Internet, no GPU, no Kaggle mount, and no official
HOCT source available for vendoring here. The deliverables were designed
to be honest about that: the builder REFUSES to complete without the HOCT
source + weights, and every real-env value stays `PENDING_REAL_ENV` until
the Kaggle GPU cell writes it.

Nothing has been submitted to Kaggle. M19-C (0.880) remains the
selectable fallback and is the automatic outcome of any G1–G11 gate
failure.

## Branch and commits

- Branch: `claude/biohub-submission-pipeline-4pggio` (session policy).
- Parent of this cycle: `d93d08a` (M39/M40 static scaffolding).
- This cycle produces one new commit adding:
  1. `m39_m40/deployment/` package (7 modules + tests).
  2. `M39_M40_DEPLOYMENT_KAGGLE_ONE_CELL.txt` — paste-ready single Kaggle cell.
  3. `M39_M40_KAGGLE_DATASET_README.md` — operator upload handbook.
  4. `M39_M40_DEPLOYMENT_REPORT.md` — this file.

## Directory layout added this cycle

```
competitions/biohub-cell-tracking-during-development/
  M39_M40_DEPLOYMENT_KAGGLE_ONE_CELL.txt      # paste-into-Kaggle cell
  M39_M40_KAGGLE_DATASET_README.md            # operator upload steps
  M39_M40_DEPLOYMENT_REPORT.md                # this report
  m39_m40/deployment/
    __init__.py                               # BUNDLE_MARKER / BUNDLE_MANIFEST names
    bundle_manifest.py                        # SHA256 manifest schema + verifier
    bundle_builder.py                         # CLI that assembles the offline bundle
    split_recovery.py                         # OBSERVED-only split_0 cascade
    cv_harness.py                             # M39 official CV using vendored metric
    baseline_association.py                   # baseline contract + registry
    hoct_association.py                       # HOCT contract + registry (+ identity guard)
    kaggle_one_cell_runner.py                 # G1..G11-gated orchestration
    tests/
      __init__.py
      test_bundle_manifest.py                 # 7 tests (build/verify/tamper/complete)
      test_split_recovery.py                  # 12 tests (parse + cascade + refusal)
      test_cv_harness.py                      # 8 tests (path/env/aggregate)
      test_kaggle_runner_gates.py             # 7 tests (discovery + G1/registry/identity)
```

## Reused, byte-for-byte

- `vendor/royerlab_cellmot/` at upstream commit `7396b7e9` — full LICENSE,
  PROVENANCE, 12 files with per-file SHA256s all match (verified this
  cycle). This is the ONLY official metric source the runner accepts.
- `archive/m19c/` — byte-preserved M19-C fallback + MANIFEST.sha256.
- `m39_m40/` scaffolding (runtime_guard, schemas, hoct_adapter,
  kaggle_runner) — untouched.

## What is intentionally NOT in this cycle

Because the sandbox has no Internet:

- **HOCT source is not vendored here.** The bundle builder REQUIRES
  `--hoct-source <local path>` at build time. Without it, the builder's
  verdict is `REFUSED_HOCT_MISSING` (still writes a partial bundle so the
  operator can inspect it, but never claims completeness). The Kaggle
  runner's G1 gate flags `INCOMPLETE_HOCT` and falls back to M19-C.
- **HOCT `general_v0` weights are not vendored here.** Same treatment
  as source — required at build time, missing = fallback.
- **`split_0` is not fabricated.** The runner's G4 gate uses the
  observed-file cascade (mirrors M32.1): `dataset_splits.json` in the
  reference bundle → torch checkpoint metadata → explicit split file.
  Deterministic reconstruction is FORBIDDEN — the upstream repo READS
  `dataset_splits.json`; it does not regenerate the split from a seed
  (see `vendor/royerlab_cellmot/PROVENANCE.json`, key `split_generation`).

## What the single-cell Kaggle runner does (in order)

The cell paste (`M39_M40_DEPLOYMENT_KAGGLE_ONE_CELL.txt`) drives the
Kaggle side:

1. Discover the mounted bundle via `M39_M40_BUNDLE.marker`.
2. Prepend the bundle root AND `vendor/royerlab_cellmot/src` to `sys.path`.
3. Register the operator's baseline + HOCT solvers via
   `m39_m40.deployment.baseline_association.register(...)` and
   `hoct_association.register(...)`. Both are placeholders in the cell
   that raise `NotImplementedError`; the operator fills them in from the
   reference-0902 notebook.
4. Wire a `_detection_provider(dataset)` that returns the SAME cached
   detections object for baseline and HOCT (so identity holds).
5. Call `m39_m40.deployment.kaggle_one_cell_runner.main(...)` which
   enforces gates **G1..G11**:
   * G1 bundle manifest = `VERIFIED`
   * G2 preflight       = `READY_EXPERIMENTAL`
   * G3 metric env      = `OK` (`tracksdata`, `polars`, `scipy`, `geff`)
   * G4 split_0         = `OK_OBSERVED` (else refuses, no leakage)
   * G5 baseline registered
   * G6 HOCT registered + probe reports `used_official`
   * G7 detection reused across baseline and HOCT
   * G8 HOCT config = `max_delta_t=1, n_neighbors=3, no TTA, no long gap`
   * G9 HOCT aggregate score ≥ baseline aggregate score
   * G10 runtime governor never crossed 105-min engineering cap
   * G11 ≥10 min of finalization reserve preserved
6. Any G-failure → M19-C fallback (`exec` the byte-preserved one-cell
   from the mounted archive, which regenerates
   `/kaggle/working/submission.csv`).
7. Final sanity check confirms the file exists and is non-empty.

## Bundle manifest schema (BUNDLE_MANIFEST.json)

`m39_m40/deployment/bundle_manifest.py`:

```
{
  "schema_version": 1,
  "bundle_marker": "M39_M40_BUNDLE.marker",
  "provenance": {
    "built_at_utc": "<ISO8601>",
    "kaggle_agent_repo_head": "<git SHA>",
    "vendor_metric_provenance": {...},     # full royerlab_cellmot PROVENANCE.json
    "hoct_provenance": {...},              # operator-supplied
    "hoct_weights_source": "<path>",
    "marker_note": "..."
  },
  "slots": {
    "vendor_metric":     {status, files: [{path, sha256, bytes}]},
    "m39_m40_package":   {...},
    "m19c_archive":      {...},
    "hoct_source":       {...},
    "hoct_weights":      {...}
  }
}
```

`verify_manifest(bundle_root, manifest)` returns one of:

- `VERIFIED` — every present slot's files match SHA256 AND both HOCT
  slots are present.
- `INCOMPLETE_HOCT` — metric+archive+scaffold match, HOCT missing.
- `MISSING_METRIC_OR_ARCHIVE` — required non-HOCT slot missing.
- `TAMPERED` — a listed file's actual SHA256 does not match its recorded
  value.

The Kaggle runner allows ONLY `VERIFIED`.

## Local build + verify (sandbox-executed this cycle)

```
python -m m39_m40.deployment.bundle_builder \
    --comp-dir . \
    --bundle-root /tmp/bundle \
    --hoct-source /tmp/synthetic-hoct-stub \
    --hoct-weights /tmp/weights \
    --marker-note "sandbox smoke test"
```

Output:

```
{
  "bundle_root": "/tmp/bundle",
  "file_counts": {
    "vendor_metric": 13, "m39_m40_package": 29, "m19c_archive": 6,
    "hoct_source": 3,   "hoct_weights": 1
  },
  "manifest_path": "/tmp/bundle/BUNDLE_MANIFEST.json",
  "verdict": "COMPLETE"
}
```

`verify_manifest` reports `VERIFIED`, 0 mismatches, all slots present.
(The HOCT source used here was a synthetic stub — real deployment MUST
use the official MIT-licensed repo per README step 1.)

## Tests

**Total this cycle: 34 new tests, all pass.** Combined with the 74
scaffolding tests, **108/108 tests pass CPU-only, deterministic**.

| File                                   | Tests |
|----------------------------------------|------:|
| `deployment/tests/test_bundle_manifest.py`     | 7 |
| `deployment/tests/test_split_recovery.py`      | 12 |
| `deployment/tests/test_cv_harness.py`          | 8 |
| `deployment/tests/test_kaggle_runner_gates.py` | 7 |

Key coverage:

- Bundle builder refuses without HOCT unless `--allow-hoct-missing`.
- Complete bundle (with synthetic HOCT) → `VERIFIED`.
- Tamper flips the verdict to `TAMPERED`, lists mismatches.
- Vendored `tracking_cellmot` hashes still match the PROVENANCE after copy.
- Split cascade: OK_OBSERVED (upstream file / explicit split /
  kaggle_test_splits_50ep), REFUSED_LEAKAGE (overlap), REFUSED (no
  manifest), upstream wins over explicit.
- `check_metric_env` reports missing deps honestly (won't return `OK`
  in this sandbox because `tracksdata`/`polars`/`geff` are absent).
- `aggregate` stays PENDING when either audit gate is False; goes final
  only when all gates pass and rows are `computed`.
- Runner discovery walks Kaggle-like input trees.
- Runner falls back to M19-C on G1 (bundle missing) and G1 (manifest
  missing).
- HOCT identity guard: any solver that returns
  `detections_reused=False` raises `HoctIdentityBrokenError`; any solver
  that returns a different `HoctAdapterConfig` than requested raises
  `HoctBlockedError`.

## Exact Kaggle inputs to mount

1. **Competition data** (built-in):
   `/kaggle/input/competitions/biohub-cell-tracking-during-development`

2. **Reference bundle 0902**:
   `/kaggle/input/datasets/<user>/biohub-0902-reference-bundle`, containing
   at minimum:
   - `reference/biohub-competition-solution.ipynb`
   - `weights/unet_transformer/split_0/edge_predictor_best.pth`
   - `reference/dataset_splits.json` (or equivalent upstream file for G4)

3. **M39/M40 offline bundle** (produced by the builder in this cycle):
   `/kaggle/input/datasets/<user>/m39-m40-bundle`, containing:
   - `M39_M40_BUNDLE.marker`
   - `BUNDLE_MANIFEST.json`
   - `vendor/royerlab_cellmot/`  (pinned)
   - `m39_m40/`                  (scaffolding + deployment)
   - `archive/m19c/`             (immutable fallback)
   - `hoct/`                     (official MIT source)
   - `weights/hoct/general_v0/`  (official checkpoint)

## Blockers that remain (for a real experimental submission)

1. Vendor the official HOCT source at a pinned commit (MIT-licensed) with
   a `PROVENANCE.json` next to it.
2. Vendor the official HOCT `general_v0` checkpoint.
3. Rebuild the bundle via the CLI in the README, verify `VERIFIED`,
   upload as a Kaggle Dataset.
4. Provide operator-side implementations of the two placeholders in the
   single-cell script:
   * `_baseline_reference_0902(req)` — runs the reference-0902 baseline
     association on `req.cached_detections`.
   * `_hoct_general_v0(req)` — runs HOCT (`general_v0`) association on
     the SAME `req.cached_detections`; must set
     `detections_reused=True` and reuse the requested `HoctAdapterConfig`
     unchanged.
5. Provide a `_detection_provider(dataset)` that caches once per dataset
   and returns the SAME object across calls (so identity is preserved).
6. Ensure the Kaggle image has `tracksdata`, `polars`, `scipy`, `geff`
   available offline (packaged into the bundle or a companion Kaggle
   Dataset). Without them, G3 fails and the runner falls back to M19-C.

## Do NOT

- Do not claim any experimental score improvement without an
  observed-CV number produced by the vendored `tracking_cellmot.metrics`
  under the OBSERVED `split_0`.
- Do not vendor a stub HOCT and call it official.
- Do not use `torch.load(weights_only=False)` for split recovery — the
  cascade uses `weights_only=True` only.
- Do not `pip install` anything at Kaggle runtime — the notebook must
  run with Internet Off.
