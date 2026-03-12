# Stanford RNA 3D Folding Part 2 - Kaggle Setup Guide

## Before You Start

You need to upload 5 datasets to Kaggle **before** running the notebook.
The notebook runs offline (no internet) so everything must be pre-uploaded.

---

## Step 1: Upload RNAPro Weights

1. Go to https://huggingface.co/nvidia/RNAPro-Public-Best-500M
2. Download all model files (checkpoint `.pt` files)
3. Go to https://www.kaggle.com/datasets/new
4. Upload as dataset named: **`rnapro-weights`**
5. Make it private

## Step 2: Upload RibonanzaNet2 Checkpoint

1. Search Kaggle Models for "ribonanzanet2"
   - Direct link: https://www.kaggle.com/models/daslab/ribonanzanet2
2. When creating your notebook, add this as an input model
   - Or download and upload as dataset named: **`ribonanzanet2`**

## Step 3: Upload Protenix Base Checkpoint

1. Go to https://github.com/NVIDIA-Digital-Bio/RNAPro/releases
   - Or https://github.com/bytedance/Protenix/releases
2. Download `protenix_base_default_v0.5.0.pt`
3. Upload as dataset named: **`protenix-base`**

## Step 4: Generate and Upload Templates

**This is the most critical step for a high score!**

### Option A: Use Existing Public Template Notebooks
1. Fork one of these Part 2 template notebooks:
   - https://www.kaggle.com/code/llkh0a/stanford-rna-3d-folding-part-2-protenix-tbm
   - https://www.kaggle.com/code/nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template
2. Run the template generation part
3. Download the output CSV files
4. Upload as dataset named: **`rna-folding-templates`**

### Option B: Generate Fresh Templates with MMseqs2
1. Run the MMseqs2 template search notebook from Part 1
2. Adapt it for Part 2 test sequences
3. Download and upload the results

## Step 5: Upload Python Dependencies (Wheels)

1. On your local machine with internet, run:
```bash
mkdir rna-deps
pip download protenix einops ml_collections -d rna-deps/
```
2. Also clone and include the source:
```bash
cd rna-deps
git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
git clone https://github.com/bytedance/Protenix protenix
```
3. Upload the entire `rna-deps` folder as dataset named: **`rna-deps`**

---

## Notebook Settings

| Setting | Value |
|---------|-------|
| **Accelerator** | GPU T4 x2 |
| **Language** | Python |
| **Internet** | OFF |
| **Persistence** | Files only |

## Running the Notebook

1. Create a new notebook on Kaggle
2. Attach all 5 datasets + the competition data as inputs
3. Copy-paste the cells from `rna_3d_folding_v2.ipynb`
4. Set accelerator to GPU T4 x2
5. Turn OFF internet
6. Click "Save & Run All" or submit directly

## Expected Runtime

- Total: 4-9 hours depending on number/length of test sequences
- Short sequences (<500 nt): ~1-5 min each
- Medium sequences (500-2000 nt): ~5-15 min each
- Long sequences (>2000 nt): ~15-30 min each

## Troubleshooting

### "No model available, using physics fallback"
- Check that your dataset paths are correct
- Verify weights were uploaded properly (not corrupted/incomplete)
- Check kernel logs for specific import errors

### Out of Memory (OOM)
- The notebook automatically handles OOM by reducing diffusion steps
- If persistent, try GPU P100 instead of T4 x2

### Submission format error
- Verify `submission.csv` has columns: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
- Check that all target sequences have predictions

## Score Expectations

| Model Available | Templates | Expected Score |
|----------------|-----------|----------------|
| RNAPro + templates | Yes | 0.50 - 0.65 |
| Protenix + templates | Yes | 0.40 - 0.55 |
| Physics fallback + templates | Yes | 0.20 - 0.35 |
| Physics fallback only | No | 0.10 - 0.20 |

**The #1 factor for high scores is having good templates!**
