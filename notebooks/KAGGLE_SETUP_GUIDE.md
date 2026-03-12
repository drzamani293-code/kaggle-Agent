# Stanford RNA 3D Folding Part 2 - Kaggle Setup Guide

## Overview

This notebook predicts RNA 3D structures using a tiered approach:
1. **RNAPro** (NVIDIA SOTA, ~0.55+ TM-score with good templates)
2. **Protenix** (ByteDance, ~0.40 TM-score baseline)
3. **Template-based** (uses precomputed templates directly)
4. **Physics fallback** (Nussinov + A-form helix geometry)

The notebook runs **offline** (no internet) on Kaggle, so all weights and
dependencies must be pre-uploaded as datasets.

---

## Step 1: Upload RNAPro Weights (~4 GB)

1. Go to https://huggingface.co/nvidia/RNAPro-Public-Best-500M
2. Download all model files (checkpoint `.pt` files)
3. Also download from https://github.com/NVIDIA-Digital-Bio/RNAPro/releases:
   - `protenix_base_default_v0.5.0.pt` (under `release_data/protenix_models/`)
4. Upload as Kaggle dataset named: **`rnapro-weights`**

## Step 2: Upload Protenix Base Checkpoint

1. Go to https://github.com/bytedance/Protenix/releases
2. Download `protenix_base_default_v0.5.0.pt`
3. Upload as Kaggle dataset named: **`protenix-base`**

## Step 3: Generate and Upload Templates (CRITICAL)

**Templates are the #1 factor for high scores.** Two options:

### Option A: Use Public Template Notebooks (Recommended)
1. Fork one of these notebooks and run them:
   - https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm
   - https://www.kaggle.com/code/nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template
2. Download the output submission CSV files
3. Convert CSV to `.pt` format using RNAPro's converter:
   ```bash
   python preprocess/convert_templates_to_pt_files.py --input_csv submission.csv
   ```
4. Upload both the CSV and `.pt` files as dataset named: **`rna-folding-templates`**

### Option B: Use MMseqs2 Template Search
1. Run MMseqs2/foldseek against PDB RNA chains for each test sequence
2. Extract top-20 templates per target
3. Convert to `.pt` format and upload

## Step 4: Upload Python Dependencies

On a machine with internet access:

```bash
mkdir rna-deps && cd rna-deps

# Clone source repos
git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
git clone https://github.com/bytedance/Protenix

# Download wheel dependencies
pip download einops ml_collections dm-tree PyYAML biopython biotite modelcif \
    gemmi scipy scikit-learn optree pydantic -d wheels/
```

Upload the entire `rna-deps` folder as dataset named: **`rna-deps`**

**Key dependencies** (from RNAPro requirements.txt):
- torch==2.7.1 (pre-installed on Kaggle)
- scipy>=1.9.0, numpy==1.26.4, pandas==2.3.1
- biopython==1.85, biotite==1.4.0, modelcif==1.4, gemmi==0.6.7
- ml_collections==1.1.0, dm-tree==0.1.9, einops
- fair-esm==2.0.0, scikit-learn==1.7.1

---

## Notebook Settings

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 x2 |
| **Language** | Python |
| **Internet** | OFF |
| **Persistence** | Files only |

## Running the Notebook

1. Create a new Kaggle notebook
2. Attach all datasets + competition data as inputs
3. Copy cells from `rna_3d_folding_v2.ipynb`
4. Set accelerator to **GPU T4 x2**
5. Turn **OFF** internet
6. Click "Save & Run All" or submit

## Expected Runtime

- **Total**: 4-9 hours (depending on test set size and model availability)
- Short sequences (<500 nt): ~1-5 min each
- Medium sequences (500-2000 nt): ~5-15 min each
- Long sequences (>2000 nt): ~15-30 min each (may fall back to templates)

## Score Expectations

| Configuration | Expected TM-score |
|--------------|-------------------|
| RNAPro + good templates + multi-seed | **0.50 - 0.65** |
| Protenix + templates | 0.35 - 0.45 |
| Templates only (from public notebooks) | 0.25 - 0.40 |
| Physics fallback only | 0.10 - 0.20 |

## Troubleshooting

### "No DL models available"
- Verify dataset paths: `/kaggle/input/rnapro-weights/`, `/kaggle/input/protenix-base/`
- Check that RNAPro source is in `/kaggle/input/rna-deps/RNAPro/`
- Check kernel logs for specific import errors

### Out of Memory (OOM)
- The notebook automatically reduces diffusion steps and cycle count
- Falls back to template-based prediction if model OOMs
- For persistent OOM: reduce `n_seeds` in the main pipeline

### Submission format errors
- Required columns: `ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5`
- The notebook validates format before saving
- All NaN values are filled with 0.0

## Architecture Details

The pipeline uses NVIDIA's RNAPro inference runner which:
1. Creates input JSON from RNA sequence via `create_input_json()`
2. Builds features via `get_inference_dataloader()` (tokenization, MSA, templates)
3. Runs `model.forward(input_feature_dict, ..., mode="inference")`
4. Extracts C1' coordinates from CIF output via `extract_c1_coordinates()`
5. Generates 5 diverse predictions per target using multi-seed ensemble
6. Selects final 5 by balancing confidence and structural diversity (Kabsch RMSD)

RNAPro model config: `rnapro_base_default` with N_cycle=10, N_step=200 diffusion steps.
