# Winning Strategy: Stanford RNA 3D Folding Part 2

## Competition Summary

- **Task**: Predict 3D C1' atom coordinates for RNA molecules from sequences
- **Metric**: TM-score (best-of-5 predictions averaged across targets)
- **Key Challenge**: Novel folds (no templates), RNA complexes up to 6,000 nt
- **Deadline**: March 25, 2026 | **Prize**: $75,000

---

## Evidence-Based Strategy (From Part 1 Competition)

### What Won in Part 1

| Team | Approach | Key Models |
|------|----------|-----------|
| d4t4 (top) | 3-model ensemble, scored 5/15 predictions | Protenix (0.75) + TBM (0.15) + Boltz (0.10) |
| 10th place | Ensemble selection | Protenix + DRFold2 + trRosettaRNA2 |
| Others | RhoFold+ single-model | ~0.58 avg TM-score |

**Critical insight**: Templates beat de novo methods by 0.3–0.5 TM-score when available.
For R1205 (novel fold): best team got TM-score 0.912 vs baseline 0.376.

---

## Our Strategy: 5-Phase Pipeline

### Phase 1: Template Search (BLAST against PDB)

```
RNA sequence → BLAST → PDB hits (identity > 50%) → thread coordinates
```

- **When to use**: Riboswitches, ribozymes, tRNA, rRNA (most common RNA families)
- **Expected gain**: TM-score 0.7–0.95 for hits vs 0.4–0.6 for de novo
- **Tools**: `blastn`, local PDB mirror
- **Key**: Filter by temporal cutoff to prevent data leakage

### Phase 2: MSA Generation (RNAcentral via nhmmer)

```
sequence → nhmmer → 512 homologs → .a3m file → feed to deep learning models
```

- **Database**: RNAcentral (25M+ sequences)
- **Impact**: +0.05–0.15 TM-score improvement with MSA vs no MSA
- **Even sparse MSA helps** Protenix and RhoFold+ by providing covariation signals

### Phase 3: Multi-Model Predictions

#### Model Priority & Selection

| Model | Seeds | Best For | CASP16/TM-score |
|-------|-------|----------|-----------------|
| **Protenix-v1** | 5 | Complexes, large RNA, novelty | AF3-surpassing |
| **RhoFold+** | 4 | Single-chain RNA (< 500 nt) | 0.58 avg CASP15 |
| **Boltz-1** | 3 | Diverse conformations, RNA-ligand | AF3-level |
| **Template** | 1-4 | Known RNA families | Up to 0.95 |

**Total**: Up to 16 candidate structures per target

#### Why These 3 Models?

1. **Protenix-v1** (Feb 2026) — outperforms AlphaFold3, RNA MSA+template support,
   Apache 2.0 license, 368M params. ByteDance's reproduction that beats AF3.

2. **RhoFold+** (Nature Methods 2024) — best single-chain RNA predictor,
   0.14s inference on A100, integrates RNA language model (23.7M sequences).
   Used by multiple top-10 teams in Part 1.

3. **Boltz-1** (MIT 2024) — fully open MIT license, AlphaFold3-architecture,
   Boltz-steering to fix hallucinations. Provides different error modes from Protenix.

### Phase 4: Greedy Diverse Selection

From 16 candidates → select 5 using:
```
score(j) = 0.6 × quality(j) + 0.4 × min_dist(j, already_selected)
```

Where:
- `quality(j)` = model_weight × pLDDT_confidence
- `min_dist(j, selected)` = 1 - TM_score (structural diversity bonus)
- Model weights: Protenix=0.75, Template=0.65, RhoFold=0.55, Boltz=0.45

This ensures the 5 predictions are both **high quality** AND **structurally diverse**,
maximizing the best-of-5 TM-score.

### Phase 5: Submission

Extract C1' coordinates per residue, format as:
```
ID,resname,resid,x_1,y_1,z_1,...,x_5,y_5,z_5
```

---

## Expected Performance

| Scenario | TM-score Range | Comment |
|----------|---------------|---------|
| Strong template (>70% identity) | 0.7–0.95 | Template threading |
| Known RNA family, shallow MSA | 0.55–0.75 | Protenix + MSA |
| Novel fold, no MSA | 0.4–0.6 | De novo methods |
| Very large RNA (>2000 nt) | 0.3–0.5 | Limited by compute |

**Target**: Average TM-score > 0.55 (above human expert baseline)

---

## Compute Budget (Kaggle T4 GPUs)

- 2× NVIDIA T4 (16GB each), 9-hour limit
- Protenix: ~2-5 min per sequence on T4
- RhoFold+: ~0.14s on A100, ~2-3min on T4
- Boltz-1: ~3-10 min on T4

For test sets with many targets: use only Protenix (5 seeds) if time is tight.

---

## Key Papers & References

1. **Protenix-v1**: [bioRxiv 2026.02.05.703733](https://www.biorxiv.org/content/10.64898/2026.02.05.703733v1)
   — ByteDance AF3-level model
2. **RhoFold+**: [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02487-0)
   — Best single-chain RNA predictor
3. **Boltz-1**: [PMC11601547](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601547/)
   — Open-source AF3 alternative
4. **Part 1 analysis**: [bioRxiv 2025.12.30.696949](https://www.biorxiv.org/content/10.64898/2025.12.30.696949v1.full)
   — Template-based predictions advanced through blind competition
5. **trRosettaRNA2**: [bioRxiv April 2025](https://www.biorxiv.org/content/10.1101/2025.04.09.647915v1.full)
   — CASP16 top server, secondary-structure aware

---

## Implementation Files

```
solution/rna_3d_folding/
├── STRATEGY.md                    # This file
├── kaggle_notebook.py             # Main Kaggle submission notebook
├── src/
│   ├── predictor.py               # Main orchestrator (RNA3DPredictor)
│   ├── ensemble.py                # Greedy diverse ensemble selection
│   ├── template_search.py         # BLAST template search + threading
│   ├── msa_generation.py          # nhmmer-based MSA generation
│   └── models/
│       ├── protenix_predictor.py  # Protenix-v1 wrapper
│       ├── rhofold_predictor.py   # RhoFold+ wrapper
│       └── boltz_predictor.py     # Boltz-1 wrapper
```
