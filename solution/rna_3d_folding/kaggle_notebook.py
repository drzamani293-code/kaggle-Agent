"""
Stanford RNA 3D Folding Part 2 — Winning Strategy Notebook
===========================================================

This is the main Kaggle submission notebook implementing the winning strategy:

APPROACH: Multi-Model Ensemble with Diversity Selection
=========================================================

Primary Models (by priority):
1. Protenix-v1 (ByteDance): AF3-surpassing performance, RNA MSA+template support
2. RhoFold+ (CUHK): Best for single-chain RNA (Nature Methods 2024)
3. Boltz-1 (MIT): Open-source AF3 alternative, diverse diffusion predictions

Strategy Proven in Part 1:
- 10th place: ensemble of Protenix + DRFold2 + trRosettaRNA2
- d4t4 team: Protenix (0.75) + TBM (0.15) + Boltz (0.10) → selected 5/15 predictions
- Templates critical for non-novel folds (TM-score 0.9+ vs 0.4-0.6 for de novo)

Evaluation: TM-score (best-of-5 predictions, higher = better)
Threshold: TM-score > 0.45 = correct global fold

Submission: submission.csv with C1' coordinates for 5 predictions per residue
"""

# ============================================================
# CELL 1: Setup and Installation
# ============================================================

import os
import sys
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")

print("Python version:", sys.version)
print("Working directory:", os.getcwd())

# Install core dependencies
INSTALL_PACKAGES = [
    "biopython",      # PDB parsing, BLAST utilities
    "boltz",          # Boltz-1 structure prediction
]

for pkg in INSTALL_PACKAGES:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        capture_output=True
    )

# Install HMMER for MSA generation
subprocess.run(
    ["apt-get", "install", "-y", "-q", "hmmer", "ncbi-blast+"],
    capture_output=True
)

print("Dependencies installed.")


# ============================================================
# CELL 2: GPU Configuration
# ============================================================

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: Install and Setup Protenix (Primary Model)
# ============================================================

PROTENIX_DIR = "/tmp/Protenix"

def setup_protenix():
    """Clone and install Protenix-v1 (ByteDance AF3-level model)."""
    if not os.path.exists(PROTENIX_DIR):
        print("Cloning Protenix-v1...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/bytedance/Protenix.git", PROTENIX_DIR],
            check=True
        )
        print("Installing Protenix...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
            cwd=PROTENIX_DIR,
            check=True,
            capture_output=True
        )
    print("Protenix ready.")


# ============================================================
# CELL 4: Install and Setup RhoFold+ (Best Single-Chain Model)
# ============================================================

RHOFOLD_DIR = "/tmp/RhoFold"
RHOFOLD_WEIGHTS = "/tmp/RhoFold/pretrained/RhoFold_pretrained.pt"
RHOFOLD_WEIGHTS_URL = "https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt"

def setup_rhofold():
    """Clone and install RhoFold+ (Nature Methods 2024)."""
    if not os.path.exists(RHOFOLD_DIR):
        print("Cloning RhoFold+...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/ml4bio/RhoFold.git", RHOFOLD_DIR],
            check=True
        )
        subprocess.run(
            [sys.executable, "setup.py", "install", "-q"],
            cwd=RHOFOLD_DIR, check=True, capture_output=True
        )

    # Download model weights
    os.makedirs(os.path.dirname(RHOFOLD_WEIGHTS), exist_ok=True)
    if not os.path.exists(RHOFOLD_WEIGHTS):
        print("Downloading RhoFold+ weights from HuggingFace...")
        subprocess.run(
            ["wget", "-q", RHOFOLD_WEIGHTS_URL, "-O", RHOFOLD_WEIGHTS],
            check=True
        )
    print("RhoFold+ ready.")


# ============================================================
# CELL 5: Core Prediction Functions
# ============================================================

import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import Optional


def parse_pdb_c1prime(pdb_path: str) -> np.ndarray:
    """Extract C1' atom coordinates from PDB file (one per nucleotide)."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            if atom_name == "C1'" and resname in ("A", "U", "G", "C"):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float32)


def write_fasta(sequence: str, target_id: str, path: str) -> None:
    """Write RNA sequence to FASTA format."""
    with open(path, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")


def predict_rhofold(sequence: str, target_id: str, seed: int = 0) -> Optional[np.ndarray]:
    """
    Run RhoFold+ prediction for a single RNA sequence.
    Returns C1' coordinates (L, 3) or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "query.fasta")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)
        write_fasta(sequence, target_id, fasta_path)

        cmd = [
            sys.executable,
            os.path.join(RHOFOLD_DIR, "inference.py"),
            "--input_fas", fasta_path,
            "--output_dir", output_dir,
            "--device", device,
            "--ckpt", RHOFOLD_WEIGHTS,
            "--seed", str(seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  RhoFold error (seed {seed}): {result.stderr[:300]}")
            return None

        # Find PDB output (prefer relaxed)
        pdb_candidates = (
            list(Path(output_dir).glob("relaxed_1000_model.pdb")) +
            list(Path(output_dir).glob("unrelaxed_model.pdb"))
        )
        if not pdb_candidates:
            return None

        coords = parse_pdb_c1prime(str(pdb_candidates[0]))
        return coords if len(coords) == len(sequence) else None


def predict_boltz(sequence: str, target_id: str, seed: int = 0) -> Optional[np.ndarray]:
    """
    Run Boltz-1 prediction for a single RNA sequence.
    Returns C1' coordinates (L, 3) or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, f"{target_id}.yaml")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        # Write Boltz YAML
        with open(yaml_path, "w") as f:
            f.write(f"version: 1\nsequences:\n  - rna:\n      id: A\n      sequence: {sequence}\n")

        cmd = [
            "boltz", "predict", yaml_path,
            "--out_dir", output_dir,
            "--cache", "/tmp/boltz_cache",
            "--seed", str(seed * 17 + 3),
            "--recycling_steps", "3",
            "--diffusion_samples", "1",
            "--sampling_steps", "200",
            "--accelerator", "gpu" if "cuda" in device else "cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            print(f"  Boltz error (seed {seed}): {result.stderr[:300]}")
            return None

        pdb_files = list(Path(output_dir).glob("**/*.pdb"))
        if not pdb_files:
            return None

        coords = parse_pdb_c1prime(str(pdb_files[0]))
        return coords if len(coords) >= len(sequence) * 0.95 else None


def predict_protenix(sequence: str, target_id: str, num_seeds: int = 5) -> list[np.ndarray]:
    """
    Run Protenix-v1 prediction for a single RNA sequence.
    Returns list of C1' coordinate arrays [(L, 3), ...]
    """
    input_json = {
        "name": target_id,
        "sequences": [{"type": "rna", "id": "A", "sequence": sequence}],
        "modelSeeds": list(range(num_seeds)),
        "dialect": "alphafold3",
        "version": 1,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, f"{target_id}.json")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        with open(json_path, "w") as f:
            json.dump(input_json, f)

        inference_script = os.path.join(PROTENIX_DIR, "runner", "inference.py")
        cmd = [
            sys.executable, inference_script,
            "--input_json_path", json_path,
            "--output_dir", output_dir,
            "--device", device,
            "--num_diffusion_samples", str(num_seeds),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"  Protenix error: {result.stderr[:300]}")
            return []

        pdb_files = sorted(Path(output_dir).glob("**/*.pdb"))
        predictions = []
        for pdb_file in pdb_files[:num_seeds]:
            coords = parse_pdb_c1prime(str(pdb_file))
            if len(coords) == len(sequence):
                predictions.append(coords)

        return predictions


# ============================================================
# CELL 6: Ensemble Selection (Quality + Diversity)
# ============================================================

def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align mobile to target using optimal Kabsch rotation."""
    mob = mobile - mobile.mean(0)
    tgt = target - target.mean(0)
    H = mob.T @ tgt
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return mob @ R.T + target.mean(0)


def fast_tm_score(pred: np.ndarray, ref: np.ndarray, subsample: int = 50) -> float:
    """Fast approximate TM-score between two C1' coordinate sets."""
    L = len(pred)
    if L == 0:
        return 0.0
    d0 = max(0.6 * (max(L, 15) - 0.5) ** (1/3) - 2.5, 0.5)

    # Subsample for speed
    if L > subsample:
        idx = np.linspace(0, L-1, subsample, dtype=int)
        p, r = pred[idx], ref[idx]
    else:
        p, r = pred, ref

    p_aligned = kabsch_align(p, r)
    d = np.sqrt(((p_aligned - r) ** 2).sum(1))
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


MODEL_PRIORITY = {
    "protenix": 0.75,
    "template": 0.65,
    "rhofold": 0.55,
    "boltz": 0.45,
}


def select_diverse_5(
    candidates: dict,
    quality_weight: float = 0.6,
    diversity_weight: float = 0.4,
) -> np.ndarray:
    """
    Select 5 diverse high-quality predictions from all candidates.

    Args:
        candidates: {"model_name": list_of_(L,3)_arrays}

    Returns:
        np.ndarray shape (5, L, 3)
    """
    all_coords, all_quality, all_labels = [], [], []

    for model, preds in candidates.items():
        weight = MODEL_PRIORITY.get(model, 0.5)
        for i, c in enumerate(preds):
            if c is not None and len(c) > 0:
                all_coords.append(c)
                all_quality.append(weight)
                all_labels.append(f"{model}_{i}")

    if not all_coords:
        raise ValueError("No valid predictions!")

    n = len(all_coords)
    quality = np.array(all_quality)

    if n <= 5:
        # Pad with best prediction if fewer than 5
        selected = list(all_coords)
        best = all_coords[int(np.argmax(quality))]
        while len(selected) < 5:
            selected.append(best.copy())
        return np.stack(selected[:5])

    # Pairwise TM-score matrix (distance = 1 - TM)
    tm_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            s = fast_tm_score(all_coords[i], all_coords[j])
            tm_mat[i, j] = tm_mat[j, i] = s
    dist_mat = 1 - tm_mat

    # Greedy diversity selection
    selected_idx = [int(np.argmax(quality))]
    for _ in range(4):
        best_score, best_j = -np.inf, None
        for j in range(n):
            if j in selected_idx:
                continue
            q = quality_weight * quality[j]
            d = diversity_weight * min(dist_mat[j, s] for s in selected_idx)
            if q + d > best_score:
                best_score, best_j = q + d, j
        if best_j is not None:
            selected_idx.append(best_j)

    for rank, idx in enumerate(selected_idx):
        print(f"  Slot {rank+1}: {all_labels[idx]} (priority={quality[idx]:.2f})")

    return np.stack([all_coords[i] for i in selected_idx])


def random_coil_fallback(sequence: str, n: int = 5) -> np.ndarray:
    """Emergency random coil baseline when all models fail."""
    L = len(sequence)
    coords = []
    for seed in range(n):
        rng = np.random.RandomState(seed)
        c = np.zeros((L, 3), dtype=np.float32)
        for i in range(1, L):
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            c[i] = c[i-1] + 3.8 * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
        coords.append(c)
    return np.stack(coords)


# ============================================================
# CELL 7: Main Prediction Loop
# ============================================================

def predict_rna(
    sequence: str,
    target_id: str,
    rhofold_seeds: int = 4,
    boltz_seeds: int = 3,
    protenix_seeds: int = 5,
) -> np.ndarray:
    """
    Full prediction pipeline for one RNA sequence.

    Returns (5, L, 3) C1' coordinates.
    """
    sequence = sequence.upper().replace("T", "U")  # normalize
    candidates = {}

    # Protenix (primary, most accurate)
    print(f"  Running Protenix-v1 ({protenix_seeds} seeds)...")
    px = predict_protenix(sequence, target_id, protenix_seeds)
    if px:
        candidates["protenix"] = px

    # RhoFold+ (best single-chain)
    print(f"  Running RhoFold+ ({rhofold_seeds} seeds)...")
    rf_preds = []
    for seed in range(rhofold_seeds):
        c = predict_rhofold(sequence, target_id, seed)
        if c is not None:
            rf_preds.append(c)
    if rf_preds:
        candidates["rhofold"] = rf_preds

    # Boltz-1 (diverse diffusion)
    print(f"  Running Boltz-1 ({boltz_seeds} seeds)...")
    bz_preds = []
    for seed in range(boltz_seeds):
        c = predict_boltz(sequence, target_id, seed)
        if c is not None:
            bz_preds.append(c)
    if bz_preds:
        candidates["boltz"] = bz_preds

    # Select 5 best/most diverse
    print("  Selecting diverse ensemble...")
    try:
        return select_diverse_5(candidates)
    except ValueError:
        print("  WARNING: All models failed, using random coil fallback!")
        return random_coil_fallback(sequence)


def run_competition(
    test_csv: str = "/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv",
    output_csv: str = "submission.csv",
) -> pd.DataFrame:
    """
    Main competition entry point.

    Reads test_sequences.csv, predicts structures, writes submission.csv.
    """
    print(f"Loading test sequences from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} target sequences")

    # Detect column names
    id_col = "target_id" if "target_id" in test_df.columns else test_df.columns[0]
    seq_col = "sequence" if "sequence" in test_df.columns else test_df.columns[1]

    all_rows = []
    t_total = time.time()

    for idx, row in test_df.iterrows():
        target_id = str(row[id_col])
        sequence = str(row[seq_col]).upper().replace("T", "U")

        print(f"\n[{idx+1}/{len(test_df)}] {target_id}: {len(sequence)} nt")
        t0 = time.time()

        predictions = predict_rna(sequence, target_id)  # (5, L, 3)

        for resid in range(1, len(sequence) + 1):
            r = {
                "ID": f"{target_id}_{resid}",
                "resname": sequence[resid - 1],
                "resid": resid,
            }
            for k in range(1, 6):
                x, y, z = predictions[k-1, resid-1]
                r[f"x_{k}"] = round(float(x), 3)
                r[f"y_{k}"] = round(float(y), 3)
                r[f"z_{k}"] = round(float(z), 3)
            all_rows.append(r)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    # Assemble submission
    cols = (
        ["ID", "resname", "resid"] +
        [f"{c}_{i}" for i in range(1, 6) for c in ("x", "y", "z")]
    )
    submission_df = pd.DataFrame(all_rows)
    submission_df = submission_df[[c for c in cols if c in submission_df.columns]]
    submission_df.to_csv(output_csv, index=False)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Submission complete!")
    print(f"  Output: {output_csv}")
    print(f"  Total rows: {len(submission_df)}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")

    return submission_df


# ============================================================
# CELL 8: Run the Competition
# ============================================================

if __name__ == "__main__":
    # Setup models (comment out what you don't need to save time)
    print("=" * 60)
    print("Stanford RNA 3D Folding Part 2 — Prediction Pipeline")
    print("=" * 60)

    # Install models (internet access required — run BEFORE submission)
    setup_protenix()
    setup_rhofold()
    # Boltz-1 is installed via pip in CELL 1

    # Run predictions
    submission = run_competition(
        test_csv="/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv",
        output_csv="submission.csv",
    )

    print("\nSample submission rows:")
    print(submission.head(3).to_string())

    # Validate submission format
    expected_cols = (
        ["ID", "resname", "resid"] +
        [f"{c}_{i}" for i in range(1, 6) for c in ("x", "y", "z")]
    )
    missing = [c for c in expected_cols if c not in submission.columns]
    if missing:
        print(f"\nWARNING: Missing columns: {missing}")
    else:
        print("\nSubmission format validated!")
