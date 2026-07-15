# M35 — Reference 0902 Repro, Official CV & Edge-TTA Foundation

M34 (self-contained TTA on the M19-C 0.880 baseline) is superseded by the supplied
**0.902** reference bundle. M35 reproduces and builds on that reference. M34 A–D
remain `SUPERSEDED_BY_M35_REFERENCE_0902` (files kept, not promoted).

## What was genuinely executed vs. blocked
- **M35-A — corrected, PASSED on Kaggle.** The manifest-driven audit passed against
  `/kaggle/input/datasets/mohammadjafarzamani/biohub-0902-reference-bundle`; all
  critical-file SHA256 matched `REFERENCE_BUNDLE_MANIFEST.json`. (Off the reference
  environment, e.g. here, it honestly reports `REFERENCE_AUDIT_FAILED` with reason
  `bundle_not_accessible` — the bundle is only mounted on Kaggle.)
- **M35-B — now a REAL runner** (no longer a placeholder): it reruns the audited
  reference pipeline via subprocess and compares the output to the reference. It is
  unit-tested end-to-end (exact / canonical / mismatch / invalid fixtures) and proven
  data-dependent. It genuinely runs only where the mounted bundle + support-pack
  weights + GPU are present; elsewhere it blocks honestly.
- **M35-C/D/E — blocked** until M35-B genuinely passes (`REPRO_PASS_EXACT` or
  `REPRO_PASS_CANONICAL`).

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

## M35-B (real reproduction)
Inputs: competition data, `biohub-tracking-support-pack-50ep-v1` (controlled 400ep
snapshot, SHA `12f6881e…2fe771` verified), and the 0902 bundle. GPU T4×2, Internet Off.
Steps: audit-gate → verify the 400ep artifact SHA → resolve preset/params + the executed
command **from the audited notebook/log/source** → subprocess-run the extracted reference
pipeline → write `/kaggle/working/m35_b_reference_reproduced.csv` (never `submission.csv`,
never submitted) → compare to `evidence/submission.csv`:
- byte SHA256 (exact),
- canonical table (per-dataset counts, node/edge/division counts, coordinates within an
  explicitly justified tolerance, edge-set equality on `(dataset, source_id, target_id)`),
- graph invariants.

Recommendation ∈ `REPRO_PASS_EXACT` / `REPRO_PASS_CANONICAL` / `REFERENCE_REPRO_MISMATCH`
/ `REFERENCE_ASSETS_NOT_ACCESSIBLE` / `RUNTIME_DEPENDENCY_FAILURE` / `INVALID_REPRODUCED_GRAPH`.
Writes `m35_reference_0902_repro.json` (resolved paths, SHAs, artifact verification,
resolved preset, executed command, return code, runtime, stdout/stderr tail, reproduced
fingerprint, reference fingerprint, byte + canonical + per-dataset comparisons, graph
validation, `fallback_used`, recommendation) and `m35_reference_0902_repro.log`.

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
