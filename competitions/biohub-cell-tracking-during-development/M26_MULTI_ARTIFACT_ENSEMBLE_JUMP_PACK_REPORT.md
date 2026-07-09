# M26 — Multi-Artifact Ensemble Jump Pack

**Goal:** break the single-model plateau (M19-C **0.880**) by combining *multiple*
model artifacts on the *same* hidden-safe test split into a consensus / weighted
ensemble tracking graph — a genuinely new source of signal aimed at a real jump
toward **0.910**. M19-C 0.880 stays the best/final candidate unless an M26 variant
beats it.

## Why an ensemble now

Everything tuneable on a *single* base has been exhausted:

| Path | Base (nodes/edges) | Best public score |
|------|--------------------|-------------------|
| M19-C (true base, full chain) | 131797 / 118992 | **0.880** (final) |
| M21-A pilkwang350 + M19-C gates | 142193 / 127563 | 0.874 |
| M23-B pilkwang350 node-prune + safe-div | 142193 / 127563 | 0.877 |
| M24-A/B 400ep clean base + chain | 127790 / 115694 | 0.873 / 0.872 |
| tom99763 pack | 161098 / 137520 | — (base too large) |

The true M19-C artifact is unrecoverable and every substitute base scores below
0.880. Post-GEFF post-processing on **one** model has run out of signal. The
remaining lever is *disagreement between models*: nodes/edges that several
independent artifacts agree on are far more likely to be real, and the union of
their true detections can recover structure no single model has.

## Two stages

### Stage 1 — `M26_ARTIFACT_ZOO_DIAGNOSTIC` (non-scoring)

Enumerates **every** mounted artifact root (datasets **and** notebooks, 1–3 levels
deep, hidden-safe globbing). For each candidate it records:

- `candidate_path`, `artifact_name`, `weight_sha256`, `repo_path`, `weights_path`,
  `has_predict_script` / `has_weights` / `has_manifest` / `has_wheels`,
  manifest summary + model config, and a **known-artifact match**
  (pilkwang350 350ep `dfb848…` → base 142193/127563; 400ep `12f688…` →
  127790/115694; tom99763 `912ae91…` → 161098/137520; support-pack-v1 `347915…`).
- Then it **smoke-runs each viable artifact** with the EXACT reference predict
  command and reports the raw base GEFF counts per dataset (`n_nodes_before`,
  `n_edges_before`, `max_in/out_degree`, `has_edge_prob` / `edge_dist` /
  `node_score`).

Writes `m26_artifact_zoo_report.json` + `m26_artifact_zoo_table.csv`. **No
submission.** Run this FIRST — the ensemble is only worth submitting if it finds
**≥2 viable artifacts**.

### Stage 2 — three submit-capable ensemble variants

All three run every viable artifact on one hidden-safe split, then build the
ensemble:

**Node matching (per dataset, per timepoint).** Nodes from different artifacts are
matched by physical centroid distance **≤ 7 µm** (the evaluator's tolerance).
A cluster holds **at most one node per artifact**, so `support_count` = number of
agreeing artifacts; the cluster centroid is the artifact-weighted mean. The
primary artifact is clustered first so its nodes seed the clusters.

**Keep policy (per variant) + primary long-track preservation.**

**Edge matching.** Each artifact edge is mapped through `member → cluster`; edges
are aggregated (support, max edge-prob, min edge-dist), kept by policy, filtered to
**unit-timepoint only** (`cluster_t[target] − cluster_t[source] == 1` — never a
direct multi-frame edge), and conflict-resolved to `in_degree ≤ 1` / `out_degree ≤ 2`.
IDs are reindexed consecutively.

**Conservative metric-aware post-processing** then runs (safe divisions; optional
micro gap1; only a very light gap2 in C).

| Variant | Policy | Post-processing | Synthetic cap |
|---------|--------|-----------------|---------------|
| **A** `consensus_2ofn_precision` | keep nodes/edges with **≥2-artifact support** (+ primary long tracks) | `safe_divisions`, `prune_isolated` | **0** |
| **B** `primary_m19c_style_plus_consensus_edges` | primary = base **closest to 131797/118992**; add consensus-supported nodes/edges | `safe_divisions`, **micro gap1** (5.0µm/0.0015/90), `linefit`, `prune_isolated` | **≤ 600** |
| **C** `weighted_ensemble_fullchain_light` | weighted ensemble (closer to target → higher weight, floored 0.2; too-many/too-few penalised) | `safe_divisions`, `gap1`, **very-light gap2** (3.6/8.0/−0.05/4.6/0.0015/55), `linefit`, `prune_isolated` | **≤ 1000** |

**Submit order: A → B → C.** A tests whether cross-model consensus fixes the node
over-prediction that has capped every substitute base; B rebuilds an M19-C-style
graph on the best available primary and only *adds* consensus-supported structure;
C is the highest-upside/highest-variance weighted union.

## Safety gates

Each runner prints exactly one recommendation. `OK_TO_SUBMIT_EXPERIMENTAL` requires
**all** of:

- **≥ 2 artifacts** actually used, `valid=True`, `fallback_used=False`,
  `final_source ∈ {reference_learned_graph_postprocessed, ensemble_graph_postprocessed}`;
- `max_in_degree ≤ 1`, `max_out_degree ≤ 2`, no NaN, `id` consecutive;
- `120000 ≤ n_node_rows ≤ 145000`, `110000 ≤ n_edge_rows ≤ 135000`;
- `synthetic_nodes_added ≤` the per-variant cap (**A 0 · B 600 · C 1000**);
- no direct multi-frame edge.

Otherwise a specific `DO_NOT_SUBMIT_*` is reported:
`NOT_ENOUGH_ARTIFACTS` (< 2 viable) / `UNSANE_NODE_COUNT` / `SYNTHETIC_EXPLOSION` /
`VALIDATION_FAILED`. A valid hidden-safe `submission.csv` is still written in every
case.

## Outputs (per runner)

`milestone26_reference_submission_report.json` (all required fields: variant,
artifacts_used, artifact_names, weight_sha256 list, primary_artifact, per-artifact
base counts, ensemble_method, n_nodes/edges_per_artifact, n_consensus_clusters,
before/after counts, node/edge_support_histogram, safe_divisions_added, gap1_closed,
gap2_recovered, synthetic_nodes_added, isolated_nodes_pruned, n_node_rows,
n_edge_rows, max_in/out_degree, valid, fallback_used, final_source, recommendation),
plus `milestone26_artifact_zoo_report.json`, `milestone26_ensemble_conversion.json`,
`milestone26_failure_fallback_report.json`, and `submission.csv`.

## Required Kaggle inputs

- The **competition dataset** (`biohub-cell-tracking-during-development`).
- **≥ 2 model/support-pack artifacts** attached as inputs (datasets and/or
  notebook outputs), each containing the tracking repo + weights. Any mix of the
  known packs (pilkwang350 350ep/400ep, tom99763, support-pack-v1) works; the zoo
  diagnostic reports exactly which are viable. With < 2 the runner refuses to
  submit (`DO_NOT_SUBMIT_NOT_ENOUGH_ARTIFACTS`).

## Verification

- `milestone26_multi_artifact_ensemble_runner.py` compiles; `run_milestone26_tests()`
  passes **8/8** on synthetic 2-artifact fixtures (clustering support, keep policies,
  unit-timepoint edges, degree constraints, primary selection, weights, end-to-end
  valid submission). All three variants dry-run cleanly off Kaggle (self-test only).
- Static checks: exact reference predict command (det 0.99, ilp-edge −1.0,
  appearance/disappearance 0.1, division 1.0, `--use-ilp`); dynamic hidden-safe
  split discovery; no hardcoded public-test CSV; output to `/kaggle/working`;
  each `.txt` ends with the correct `run_milestone26_variant("A"|"B"|"C")` call;
  no M16–M25 file modified.
- intelligence_db: M19-C stays **final/best @0.880**; `M26_ARTIFACT_ZOO`
  (diagnostic, non-scoring) + `M26_A/B/C` (pending experimental) added; a
  `DEC_M26_ENSEMBLE_PIVOT` decision recorded; `next_actions.md` now recommends the
  ensemble pivot (run Stage 1, then A→B→C if ≥2 artifacts). Tests **6/6**.

**Status:** M19-C **0.880** remains the final recommended submission. M26 A/B/C are
experimental — submit only on `OK_TO_SUBMIT_EXPERIMENTAL`, keep 0.880 unless beaten.
