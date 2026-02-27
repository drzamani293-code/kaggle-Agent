# Stanford RNA 3D Folding - Part 2 Solution

## Overview

High-performance deep learning pipeline for RNA 3D structure prediction.

**Competition:** [Stanford RNA 3D Folding - Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
**Metric:** TM-score (best-of-5, higher is better)
**Prize:** $75,000

## Architecture

```
RNA Sequence (A/C/G/U)
        │
        ▼
┌─────────────────────┐
│   RNA Encoder        │  Transformer (8 layers, 256 dim)
│   + Positional PE    │  Pre-LN, bidirectional attention
└────────┬────────────┘
         │ single repr (B, L, 256)
         ▼
┌─────────────────────┐
│   PairNet            │  Outer-product + Axial attention
│   (Evoformer-like)   │  (B, L, L, 64)
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌────────┐
│SS Head│ │DistHead│  Secondary structure + distance bins
└───────┘ └────────┘
         │
         ▼
┌─────────────────────┐
│  Structure Module    │  Iterative coordinate refinement (8 steps)
│  (IPA-inspired)      │  → C3' coordinates (B, L, 3)
└─────────────────────┘
         │
         ▼
  5 Diverse Predictions
  (greedy + dropout sampling)
         │
         ▼
  Geometry Refinement
  (secondary structure springs)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Transformer Encoder** | 8-layer pre-LN transformer, bidirectional |
| **Pairwise Representation** | Outer-product + axial attention (AlphaFold2-inspired) |
| **Secondary Structure** | Predicted and used as structural prior |
| **Distance Prediction** | 37-bin distance distribution (0–40 Å) |
| **Structure Module** | IPA-like iterative refinement (8 iterations) |
| **Diverse Predictions** | 5 predictions via dropout ensemble |
| **Geometry Refinement** | Spring-based refinement with SS restraints |
| **pLDDT Score** | Per-residue confidence (for sorting predictions) |

## Files

| File | Description |
|------|-------------|
| `rna_model.py` | Core model architecture |
| `rna_utils.py` | Sequence processing, TM-score, PDB export |
| `train.py` | Training script with cosine LR schedule |
| `predict.py` | Inference with ensemble support |
| `kaggle_submission_notebook.py` | Self-contained Kaggle notebook |

## Usage

### 1. Train

```bash
python train.py \
    --data_dir ./data \
    --epochs 100 \
    --lr 3e-4 \
    --batch_size 8 \
    --output_dir ./checkpoints
```

**Data format expected:**
- `data/train_sequences.csv` — columns: `target_id`, `sequence`
- `data/train_coords.csv` — columns: `target_id`, `residue_index`, `x`, `y`, `z`

### 2. Predict

```bash
python predict.py \
    --model_path checkpoints/best_model.pt \
    --test_csv /path/to/test.csv \
    --output submission.csv \
    --refine
```

### 3. Ensemble (multiple checkpoints)

```bash
python predict.py \
    --model_path checkpoints/best_model.pt,checkpoints/epoch_0090.pt \
    --test_csv test.csv \
    --output submission.csv
```

## Competitive Strategy

### Phase 1: Baseline (~TM 0.40–0.50)
- Train the base model on all available training data
- Use single-model predictions with geometry refinement

### Phase 2: Improved (~TM 0.50–0.60)
- Pre-train on RNA sequence databases (Rfam, RNAcentral)
- Add MSA features (multiple sequence alignments)
- Scale model (larger d_model, more layers)

### Phase 3: Top-tier (~TM 0.60+)
- Integrate RhoFold+/trRosettaRNA predictions as structural priors
- Energy-based refinement (Rosetta/OpenMM)
- Multi-state ensemble with clustering

### Tips for Winning
1. **Pre-training:** Use Masked Language Modeling on large RNA databases
2. **MSA features:** ColabFold to generate MSAs → structural co-evolution signals
3. **External tools:** RhoFold+, trRosettaRNA, ContextFold for initial predictions
4. **Refinement:** OpenMM/Amber for physics-based relaxation
5. **Ensemble:** Diverse models (different architectures, seeds, data subsets)

## Evaluation

TM-score > 0.45 → globally correct fold
TM-score > 0.60 → competitive with top methods
TM-score > 0.75 → near-experimental accuracy

## References

- [RhoFold+](https://github.com/ml4bio/e2efold-3d) — E2E RNA folding
- [trRosettaRNA](https://yanglab.qd.sdu.edu.cn/trRosettaRNA/) — Rosetta-based RNA folding
- [AlphaFold3](https://github.com/google-deepmind/alphafold3) — General biomolecular structure
- [OpenChemFold](https://github.com/GosUxD/OpenChemFold) — 3rd place Ribonanza solution
- [EternaFold](https://eternagame.org/about/software) — Secondary structure prediction
