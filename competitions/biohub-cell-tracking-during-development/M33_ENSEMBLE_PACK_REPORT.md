# M33 — Corrected Ensemble Audit & Fusion Pack

Audits the uploaded ensemble sources and **repairs** both the output-level and the
probability-level ensembles they implement. Every runner is a standalone one-cell
Kaggle notebook. A/B/D are NON-SUBMIT; C/E are hard-gated experimental candidates.
`M19-C` (public **0.880**) stays confirmed best/final until an OBSERVED score beats
it.

## Defects found (and fixed)
**blend_submissions.py** — `next_id=0` per dataset (repeated global node ids);
nearest-neighbour node snapping (not one-to-one → same-variant nodes collapse and
one variant double-votes a canonical edge); `min_votes=1` single-model forks;
canonical coords depend on variant order; no fork-level consensus.
**geff_ensemble.py** — scores `ILP_SOLUTION_LIKE` graphs as full candidate graphs;
imports the OLD `winning_postprocess_v2` linker (not the corrected M30 two-stage
linker); consensus divides by **all** models even when a model was not eligible to
propose the edge; uncalibrated probabilities; correlated det-threshold variants;
inherits the node-canonicalization bugs.

## Corrected machinery (shared)
- **Node canonicalization** — deterministic and **input-order-invariant** (variants
  processed in a fixed sorted-by-name order). Per frame, the incoming variant's
  nodes are matched **one-to-one** to the growing canonical clusters with Hungarian
  assignment gated at `eps_um` (physical µm; VOXEL_SIZE z=1.625 y=0.40625 x=0.40625).
  Two nodes of one variant **never** share a canonical node (collision count = 0,
  asserted); unmatched nodes stay distinct; canonical coordinate is a weight-robust
  mean/median over members (never "first variant wins"). eps candidates reported:
  0.75 / 1.0 / 1.5 / 2.0 µm (2.0 never used without reporting collision risk).
- **Edge voting** — per variant, translate to canonical ids, drop self/backward/
  non-unit-time edges, **deduplicate** so each variant casts at most one vote per
  canonical edge (duplicates counted), accumulate **weighted** support (M19-C 2.0,
  M29-A 1.0, M30-C 1.0; unidentified runs reported, never silently weighted).
- **Two-stage linker (corrected M30, not winning_postprocess_v2)** — primary is a
  per-frame min-cost assignment with explicit **no-link dummies** (target ≤1 parent,
  source ≤1 primary child); division is an **independent** second assignment over
  parents-with-one-child and still-unparented targets, gated by stricter support +
  multi-variant fork + parent-child + sister distance, with a no-division dummy so
  rejected second children **cannot steal** a target.
- **Probability denominator** — a model is eligible for a canonical edge only if
  both endpoint detections exist for it; `support_fraction = proposer_count /
  eligible_model_count`. A missing endpoint is **not** a zero-probability vote.

## Runners
| runner | stage | submit? | key output | recommendation set |
|---|---|---|---|---|
| `M33_A_ENSEMBLE_INPUT_AUDIT_NOT_SUBMIT` | A | no | `m33_ensemble_input_audit.json` + 2 diversity CSVs | ensemble-potential class |
| `M33_B_CORRECTED_OUTPUT_BLEND_DIAGNOSTIC_NOT_SUBMIT` | B | no | `m33_output_blend_diagnostic.json` + edge-changes CSV | B0/B1/B2/B3 stats |
| `M33_C_CORRECTED_OUTPUT_BLEND_CANDIDATE` | C | gated | `submission.csv` + report + manifest + fallback | `OK_TO_SUBMIT_EXPERIMENTAL` / `DO_NOT_SUBMIT_*` |
| `M33_D_PROBABILITY_FUSION_AUDIT_NOT_SUBMIT` | D | no | `m33_probability_fusion_audit.json` + calibration CSV | ensemble-allowed + block reasons |
| `M33_E_CORRECTED_PROBABILITY_ENSEMBLE_CANDIDATE` | E | gated | `m33_probability_ensemble_report.json` | `DO_NOT_SUBMIT_*` until candidates+calibration resolve |

### Stage A — input & diversity audit
Dynamically discovers submission CSVs, report JSONs, prediction GEFF stores, and
manifests from mounted inputs + `/kaggle/working` (no filename/dataset assumptions;
no leaderboard score inferred from filenames). Per-run structural report (sha256,
graph validity, global node-id uniqueness, consecutive row id, degree caps,
dangling/multiframe, GEFF candidate class, edge_prob availability, candidate/
solution counts) + pairwise submission diversity (node overlap, edge Jaccard,
unique/conflicting edges, division agreement) + pairwise GEFF diversity (candidate/
solution overlap, edge_prob correlation on shared eligible edges). Classes:
`INSUFFICIENT_INPUTS` / `NEAR_DUPLICATE_VARIANTS` / `MODERATE_DIVERSITY` /
`STRONG_DIVERSITY` / `INCOMPATIBLE_ARTIFACTS` / `INVALID_INPUT`.

### Stage B — corrected output-blend diagnostic (≥3 valid submissions)
B0 M19-C-alone reproduction · B1 weighted consensus (primary≥2, div≥2, fork≥2) ·
B2 stricter (≥3) · B3 controlled union (preserve M19-C on conflict unless challenger
≥2). Reports collisions (0), duplicate votes removed, order-invariance, M19-C
retained/removed/replaced/new, support histograms, per-dataset invariants. A proxy
result may be printed only under `NON_AUTHORITATIVE_PROXY_DIAGNOSTIC`.

### Stage C — corrected output-blend candidate (hard-gated)
Builds an experimental `submission.csv` from the corrected weighted ensemble with
**GLOBAL** unique node ids + a running row id across all datasets (fixes the
per-dataset reset). Gates: ≥3 valid, not `NEAR_DUPLICATE_VARIANTS`, zero
canonicalization collisions, order-invariant, no duplicate votes, valid graph, and
sane delta vs M19-C (node −5..+8 %, edge −5..+10 %, M19-C retained ≥0.90,
divisions ≤3000). `final_source = corrected_output_level_weighted_ensemble`.
Does not auto-submit — the user reviews the report first.

### Stage D — probability-fusion audit
Per GEFF variant: candidate class (`FULL_PRE_ILP_CANDIDATES` / `MODERATELY_SPARSE`
/ `ILP_SOLUTION_LIKE` / `EDGE_PROB_UNAVAILABLE`), edge_prob quantiles/non-degeneracy,
eligible-endpoint universe, calibration diffs. Reports calibration options (raw /
logit / rank-normalized / weighted-logit) but **selects none** without CV.

### Stage E — corrected probability ensemble (hard-gated)
Uses one-to-one canonicalization + eligible-model denominator + per-model dedup +
the corrected M30 two-stage linker. Refuses on `ILP_SOLUTION_LIKE`
(`DO_NOT_SUBMIT_CANDIDATE_GRAPH_TOO_SPARSE`), `EDGE_PROB_UNAVAILABLE`, insufficient
diversity, or unresolved calibration (`DO_NOT_SUBMIT_CALIBRATION_UNRESOLVED`).

## Official-CV integration
Until M32.1's vendored official metric + a clean split are available: no
`local_metric.py` result may be called official; proxy results never choose a
winner; the output blend is only an experimental LB ablation after strong
structural diagnostics; the probability ensemble stays blocked without rich
candidate graphs.

## Required Kaggle input layout
Mount ≥3 tracking outputs as datasets — each either a submission-format CSV
(`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`) or a prediction GEFF
store (for D/E, with `edges/props/edge_prob` and unmasked candidate edges). Optional
report/manifest JSONs supply run name, artifact name, weight sha256, predict config,
det threshold, and postprocess family. Nothing is assumed by filename.

## Verification (this environment)
- `py_compile` OK; all 5 runners self-test **30/30** then dry-run (`/kaggle/input`
  absent).
- End-to-end real run on 3 staged synthetic submissions: A = `STRONG_DIVERSITY`;
  B0–B3 primary=2 div=1 **collisions=0** order-invariant retain=1.00; C =
  `OK_TO_SUBMIT_EXPERIMENTAL`, valid `submission.csv` (global node ids unique,
  division preserved as out-degree 2).
- Static: no Kaggle-API submit; `local_metric` never imported/called official;
  `submission.csv` written only in the gated Stage C; `winning_postprocess_v2` /
  `global_relink` never defined/imported/called; no M16–M32 source modified.
- intelligence_db tests **6/6**.

## Status ledger
M19-C 0.880 **final**; M29-A 0.876 failed; M30-C pending; M31 blocked; M32/M32.1
official CV unresolved/in progress. M33 A/B/D non-scoring diagnostics; C/E blocked
until their audits pass. Decision `DEC_M33_CORRECTED_ENSEMBLE_FIRST`: A → B →
(optional) C → D → E only if full candidates + calibration resolve.
