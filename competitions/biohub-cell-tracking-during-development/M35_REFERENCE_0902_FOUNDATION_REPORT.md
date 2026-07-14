# M35 — Reference 0902 Repro, Official CV & Edge-TTA Foundation

M34 (self-contained TTA on the M19-C 0.880 baseline) is technically complete but
**strategically superseded** by a newly supplied **0.902** reference bundle that is
different and stronger than M19-C. M35 is the foundation for reproducing and building
on that reference. M34 A–D are recorded `SUPERSEDED_BY_M35_REFERENCE_0902` (files kept,
not deleted, not promoted).

## Status in THIS environment — REFERENCE_ASSETS_NOT_ACCESSIBLE
The supplied 0.902 bundle (notebook + log + extracted results repository with its
official metric) is **not accessible** here. Per the hard rule, M35 does **not**
reconstruct it from the prompt. `M35_A` prints the exact missing asset patterns and
stops with `REFERENCE_ASSETS_NOT_ACCESSIBLE`; every downstream runner (B–E) is
hard-gated on the audit and refuses. No submission is built.

Missing required assets (exact patterns searched, none found):
- `reference_notebook`: `*0902*.ipynb`, `*motion_division*calibration*.ipynb`, `*public_0902*.ipynb`
- `reference_log`: `*0902*.log`, `*0902*.txt`, `*run*log*0902*`
- `results_repository`: `**/results`, `**/*results*repo*`, `**/extracted_results`
- `official_metric`: `**/tracking_cellmot/metrics.py`, `**/metric*.py`, `**/official_metric*.py`, `**/scripts/evaluate.py`
- `preset_config`: `**/*public_0902*`, `**/*motion_division*calibration*`, `**/preset*.json`, `**/config*.json`

Searched roots: `/kaggle/input`, `/kaggle/working`, the competition input dir,
`/kaggle/working/reference_0902`, `/kaggle/input/biohub-0902-reference`,
`/kaggle/input/public-0902-reference`. **To proceed, mount the 0.902 bundle and re-run M35_A.**

## The 0902 reference (EXPECTED VERIFICATION TARGETS — never reconstructed)
Preset `public_0902_motion_division_calibration`: det 0.97; **D4-XY detection TTA
already active**; pool_kernel_um 3.0; ILP edge −1.0 / app 0.1 / disapp 0.1 / div 1.0;
motion relink on (tight 6.0µm, relaxed 10.0µm, velocity_weight 0.5, learned_bonus 1.0);
gap close on (effective max gap 1); **gap2 recovery disabled**; safe division max 4.66µm,
existing child max 7.65µm, sister max 8.5µm, frame cap 0.0076, global cap 0.00375;
min_track_len 6; linefit window 2, weight 0.8; DeepCenter disabled. Reference
fingerprint: **nodes 128511, edges 124002, total rows 252513**; observed public score
**0.902** (`verified_on_kaggle=False`). These values are stored **only as verification
targets** to check an audited bundle — never used to fabricate reference outputs.

## Positioning
M35 is NOT a reproduction of the earlier missing ~0.900 reference and makes no claim
of matching M31. `local_metric.py` is never the official scorer — only the metric
inside the extracted results repository is authoritative. **No claim or guarantee of
0.975.** The 0.902 is recorded USER-OBSERVED, **separate** from independently verified
Kaggle evidence.

## Runners (standalone one-cell)
| runner | stage | gate | output |
|---|---|---|---|
| `M35_A_REFERENCE_BUNDLE_AUDIT_NOT_SUBMIT` | A | — | `m35_reference_bundle_audit.json`; STOP if not accessible |
| `M35_B_REFERENCE_0902_REPRO_NOT_SUBMIT` | B | A passes | exact 128511/124002/252513 repro + equivalence check |
| `M35_C_CANDIDATE_EDGE_EXPORT_NOT_SUBMIT` | C | B passes | full pre-ILP candidate edges + probabilities |
| `M35_D_EDGE_TTA_D4_DIAGNOSTIC_NOT_SUBMIT` | D | metric wired | feature-map D4 + edge-logit D4 vs detector-only D4, OFFICIAL CV |
| `M35_E_FULL_CANDIDATE_JOINT_SOLVER_CANDIDATE` | E | D's CV passes | joint learned+motion solver over full candidates |

## Reused from M34 (only)
Exact D4 XY geometry + inverse transforms + roundtrip tests; cache infrastructure;
one-to-one Hungarian canonicalization + order-invariance tests; no-link primary
assignment; independent division assignment; graph validation + sane-delta guards
(bands **re-based on the 0902 fingerprint**: nodes 122000–135000, edges 117000–131000,
rows 240000–266000). **Not reused:** the M19-C baseline config/fingerprint,
TTA4-as-the-primary-model, and the assumption that detection TTA is missing (the 0902
reference already has D4 detection TTA). The post-fusion postprocess chain must be
**resolved from the audited bundle** — `apply_reference_postprocess_chain` requires the
ops and raises rather than defaulting to any prompt literal.

## Verification (this environment)
- `py_compile` OK; every runner self-tests **20/20** then runs the real audit.
- `M35_A` → `REFERENCE_ASSETS_NOT_ACCESSIBLE` with exact missing paths; `M35_B–E` all
  block with the same status, no reconstruction, **no submission written**.
- Reused geometry roundtrips exact (0 error); TTA8 D4 provably safe; fusion machinery
  (one-to-one canon, order-invariance, no-link primary, independent division, graph
  validation) passes on synthetic fixtures.
- Static: no submission write anywhere; `local_metric` never imported/official; det
  0.97 target (not the 0.99 M19-C baseline); the M19-C fingerprint (133106/121718) is
  **not** reused; `verified_on_kaggle=False`; no 0.975 claim; no M16–M34 source modified.
- intelligence_db tests **6/6**.

## Status ledger
M19-C 0.880 **HISTORICAL confirmed**; M29-A 0.876 failed; M30-C pending (user-supplied
score); M31 blocked; M32/M32.1 official CV unresolved; **M33 operationally paused**;
**M34 A–D SUPERSEDED_BY_M35_REFERENCE_0902**. User-observed 0.902 recorded separately
from verified Kaggle evidence. M35 A diagnostic; B–E blocked. Decision
`DEC_M34_SUPERSEDED_BY_0902_REFERENCE`: A → B → C → D → E (E only after official CV).
**Exact first action: run `M35_A_REFERENCE_BUNDLE_AUDIT_NOT_SUBMIT` only.**
