# Data Details

## Input Format

### test_sequences.csv

| Column | Description |
|--------|-------------|
| `target_id` | Unique identifier for each RNA target |
| `sequence` | RNA sequence string (characters: A, U, G, C) |
| `temporal_cutoff` | Date after which experimental data was collected |

Example:
```
target_id,sequence,temporal_cutoff
R1116,GGGAAUAGCUCAGUC...,2024-01-01
```

## Output Format

### submission.csv

For each residue in each target, provide C1' coordinates for 5 predictions:

| Column | Description |
|--------|-------------|
| `ID` | `{target_id}_{resid}` format |
| `resname` | Nucleotide (A/U/G/C) |
| `resid` | 1-based residue index |
| `x_1` to `z_5` | 3D coordinates of C1' atom for predictions 1-5 (Angstroms) |

Example (partial):
```
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,...,x_5,y_5,z_5
R1116_1,G,1,-12.3,4.5,8.7,-11.8,4.2,9.1,...
R1116_2,G,2,-11.2,5.1,10.2,...
```

## Key Data Properties

### RNA Sequence Alphabet
- **A**: Adenine
- **U**: Uracil
- **G**: Guanine
- **C**: Cytosine
- Non-canonical: may include modified nucleotides in some targets

### RNA Structure Features
- Single-stranded RNA folds into complex 3D structures
- Secondary structure: Watson-Crick base pairs (A-U, G-C, G-U wobble)
- Tertiary structure: pseudoknots, kissing loops, ribose zippers
- **C1' atom**: The carbon atom connecting the sugar to the base (centroid of nucleotide)

### Part 2 Test Targets Include
- Short functional RNAs (50-200 nt) — aptamers, riboswitches
- Medium RNAs (200-1000 nt) — ribozymes, group I/II introns
- Large RNA assemblies (1000-6000 nt) — no existing templates
- RNA-protein complexes
- RNA-ligand complexes
- cryo-EM derived structures (novel folds, no templates)

## Training Data Sources

Participants typically use:
- **PDB RNA structures** (experimental, ~18k structures)
- **RNAcentral** (25M+ unique RNA sequences for MSA)
- **RFAM** (RNA family database for homology detection)
- **RibonanzaNet2 embeddings** (100M-param RNA foundation model)
- **CASP16 RNA targets** (previously used in Part 1 public leaderboard)

## Compute Budget (Kaggle Inference)

- **GPU**: 2x NVIDIA T4 (16GB VRAM each) — typical Kaggle inference
- **RAM**: 29GB
- **Time limit**: 9 hours for full test set inference
- **Internet**: Disabled during submission (must pre-download models!)
