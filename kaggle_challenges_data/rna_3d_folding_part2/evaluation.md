# Evaluation Metric: TM-score

## Primary Metric

**TM-score** (Template Modeling score) — a structural similarity metric ranging from 0.0 to 1.0 (higher = better).

### Key Properties

- **TM-score > 0.45**: Correct global fold (calibrated for RNA structures)
- **TM-score = 1.0**: Perfect prediction
- Calibrated so that scores are length-independent
- Only residues that align correctly **by numbering** are rewarded (no alignment tricks!)

### Scoring Protocol

For each target RNA:
1. Predict **5 structures**
2. Score each against experimentally determined reference structure
3. Take the **best TM-score** among the 5 predictions
4. Average across ALL targets → final score

Multiple experimental conformers may exist for some targets; score against the best-matching conformer.

## Submission Format

File: `submission.csv`

```
ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, x_4, y_4, z_4, x_5, y_5, z_5
```

Where:
- `ID`: sequence identifier
- `resname`: nucleotide name (A, U, G, C)
- `resid`: residue index (1-based)
- `x_i, y_i, z_i`: 3D coordinates of C1' atom for prediction i (in Angstroms)

## Strategy Implications

1. **Best-of-5 scoring** → generate diverse predictions, not just 5 copies of the same structure
2. **Numbering alignment** → preserve sequence alignment in PDB output (no renumbering!)
3. **No alignment tricks** → template reuse without proper modeling is penalized
4. **TM-score > 0.45 is a major breakthrough** for novel folds

## Benchmark Context

| Model | CASP15 RNA TM-score |
|-------|---------------------|
| RhoFold+ | ~0.58 avg |
| AlphaFold3 | ~0.55 avg |
| trRosettaRNA2 | ~0.60 avg (CASP16 leader) |
| Protenix-v1 | AF3-level (state of art) |
| Boltz-1 | AF3-level |
| Human experts | ~0.55 avg |
