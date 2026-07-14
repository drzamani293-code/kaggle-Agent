# M34 — Self-Contained Geometric TTA & Graph Fusion Pack

A fully self-contained MODEL-LEVEL ensemble inside ONE Kaggle notebook. The same
controlled **400ep** model is run under geometrically valid, **exactly invertible**
test-time transforms; every detection / probability map / graph edge is
inverse-transformed back to the original coordinate frame; detections and edge
support are fused across TTA variants; the **exact M19-C** postprocess chain is
applied; and one conservative experimental submission is written only when every
structural + geometry gate passes. No M19/M29/M30 notebook outputs are required.

**Positioning:** M34 is NOT a reproduction of the missing ~0.900 reference and makes
no claim of matching M31. It is a new, explicitly designed, fully audited TTA
ensemble. Transforms were implemented only after inspecting the real mounted
prediction source; the Z axis is never flipped or rotated. `local_metric.py` is
never the official scorer. `M19-C` (public **0.880**) stays confirmed best/final
until an OBSERVED score beats it.

## Required Kaggle inputs (only)
- Biohub competition dataset.
- Biohub Tracking Support Pack containing the controlled 400ep artifact
  (`weight_sha256 == 12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`).
Internet OFF; GPU T4 ×2. No external prior-submission inputs.

## Model source & axis order (discovered in Stage A)
Stage A locates and sha256-hashes the mounted implementation for image loading,
model class, prediction, node extraction, edge prediction, ILP graph generation,
the M19-C postprocess, and the submission writer; resolves image/model-input/output
axis order and normalization/padding/crop behaviour **from the source text** (never
invented); and reports `unresolved_items` when the source cannot be read. Physical
scale is fixed and anisotropic: **Z=1.625 µm, Y=0.40625 µm, X=0.40625 µm** (XY
isotropic). Off-Kaggle (no mounted artifact) it honestly reports
`ARTIFACT_GUARD_FAILED` while geometry verification still runs on synthetic volumes.

## Transform definitions (exactly invertible, XY only)
Each transform is a D4-in-XY element = optional XY transpose (`swap`) then optional
Y-flip / X-flip, in that order; Z is never touched. Volume, probability-map, node-
coordinate and edge-endpoint inverses are the reverse composition (flips/transpose
are involutions).

| set | members | shape |
|---|---|---|
| **TTA4 (mandatory)** | identity, flip_x, flip_y, flip_xy | shape-preserving |
| **TTA8 D4-XY (optional)** | + transpose, rot90, rot270, antitranspose | shape-swapping |

## Geometry verification (Stage A, synthetic)
identity/flip_x/flip_y/flip_xy volume + probability-map + coordinate + edge-endpoint
+ padding/unpadding roundtrips, and physical-distance invariance. **Result: all
roundtrips exact (0 error), distance-invariant.** TTA8 D4-XY is **provably safe**
(`X==Y` scale verified + all 8 roundtrips pass) → `tta8_supported=True`; if it could
not be proven safe the audit returns `USE_TTA4_ONLY`.

## M19-C reproduction config (Stage B)
Resolved from the M17/M19 source: `split_0`, controlled 400ep weight, `use_ilp=True`,
`det_threshold=0.99`, ILP weights (edge −1.0, appearance 0.1, disappearance 0.1,
division 1.0), postprocess ops `[safe_divisions, gap1, gap2, linefit, prune_isolated]`.
Identity reproduction gate vs the M19-C fingerprint (**nodes 133106, edges 121718,
divisions 483, score 0.880**): node/edge delta ≤0.5 %, division delta ≤10 %, graph
valid, `fallback_used=False`; else `BASELINE_REPRO_MISMATCH`.

## Cache layout
`/kaggle/working/m34_tta_cache/<transform>/` per dataset+transform, with a manifest
recording transform, prediction config, artifact SHA, dataset, node/edge counts,
`edge_prob` availability, source hashes, runtime, completed, and a config-derived
`cache_hash`. **Stale cache is rejected** whenever the configuration or the source/
artifact hashes differ (the cache signature is config+source-derived, not runtime).

## Node consensus
Identity nodes form a **protected backbone** (never deleted). Non-identity variants
are matched **one-to-one per frame** to the growing clusters (Hungarian, gated at
`eps_um`, default 1.0 µm; diagnostics 0.75/1.0/1.5). Two nodes of one variant can
never map to one canonical node. A TTA-only cluster is kept only when supported by
**≥2 distinct non-identity variants**. Canonical coordinates are a robust mean/median
of inverse-transformed observations. **Hard gates:** `collision_count == 0` and
order-invariance (identity first, then non-identity in sorted order → permutation of
input order yields an identical graph).

## Edge & division consensus
Per variant: translate to canonical ids, **deduplicate** (one vote per canonical
edge), reject self/non-unit-time edges, track identity support. **Primary:** keep an
identity edge by default; **replace** only on strong consensus (alt support ≥3/4,
identity support ≤1, distance ≤7 µm) via a per-frame no-link Hungarian assignment
(target ≤1 parent, source ≤1 primary child); **add** a missing edge only with support
≥2, unparented target, no primary child, distance ≤6 µm. **Division:** independent
second assignment — second-child support ≥3, fork supported by ≥3 variants, child
distance ≤6 µm, sister ≤9 µm, unparented targets only, no-division dummy so a rejected
second child cannot steal a target. **No synthetic gap nodes are created during
fusion** (the M19-C chain's own gap recovery runs afterwards). Final source:
`self_contained_tta4_fusion_plus_m19c`.

## Submission gates (Stage C)
Writes `/kaggle/working/submission.csv` only when ALL pass: all expected datasets
present; no dangling edges; direct multiframe edges = 0; max in-degree ≤1; max
out-degree ≤2; global node ids unique; row ids consecutive; no non-finite
coordinates; artifact guard passed; baseline reproduction passed; geometry passed;
`collision_count == 0`; order-invariance passed; `fallback_used == False`. **Sane
delta vs M19-C:** nodes 126000–140000, edges 115000–132000, divisions 300–1800,
identity-retained fraction ≥0.92, node delta −4..+6 %, edge delta −5..+9 %.
Recommendation `OK_TO_SUBMIT_EXPERIMENTAL` or a specific `DO_NOT_SUBMIT_*`. The user
manually reviews and submits — no Kaggle API.

## Stage D (optional TTA8 D4-XY)
Blocked unless `tta8_supported=True`, all 8 geometry tests pass, shape/padding exactly
invertible, TTA4 candidate valid, and no excessive disagreement. XY D4 group only (Z
never rotated/flipped); stricter gates (new node/edge ≥4, replace ≥6, division ≥6,
fork ≥6). Isolated model-level ablation — **no** lower det threshold, M30 kNN,
DeepCenter, gap3 opening, or relaxed division caps. Final source:
`self_contained_tta8_d4_fusion_plus_m19c`.

## Performance
One dataset + one transform at a time; cache results; release GPU memory after each
transform/dataset; never hold all volumes in RAM; Hungarian matching + graph fusion
on CPU; print progress + elapsed time; every runner restartable. TTA4 may run ~4×
identity; runtime is estimated and reported before full prediction.

## Verification (this environment)
- `py_compile` OK; all 4 runners self-test **40/40** then dry-run (`/kaggle/input`
  absent — prediction needs the mounted model + GPU).
- Geometry: TTA4 + TTA8 roundtrips exact (0 error); TTA8 provably safe.
- End-to-end synthetic fusion: identity backbone protected, division preserved
  (out-degree 2), zero collisions, order-invariant, valid graph, unique global node
  ids across datasets, consecutive row ids; sane-delta passes at the fingerprint.
- Static: exact 400ep SHA; no Kaggle-API submit; `local_metric` never imported/official;
  `submission.csv` written once, gated in Stage C; no `flip_z`/Z transform; no
  DeepCenter / `winning_postprocess_v2`; no external prior-submission inputs; no
  M16–M33 source modified.
- intelligence_db tests **6/6**.

## Status ledger
M19-C 0.880 **final**; M29-A 0.876 failed; M30-C pending (user-supplied score);
M31 blocked; M32/M32.1 official CV unresolved/in progress; **M33 operationally paused
(not technically disproven)**. M34 A/B diagnostic; C/D blocked candidates. Decision
`DEC_M34_SELF_CONTAINED_TTA_MODEL_LEVEL_NEXT`: A → B → C → D only if TTA8 is proven
safe.
