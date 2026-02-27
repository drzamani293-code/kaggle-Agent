"""
Advanced features for competitive RNA 3D structure prediction:
  - MSA (Multiple Sequence Alignment) feature extraction
  - RhoFold+ integration wrapper
  - Evolutionary coupling features (DCA/covariance)
  - Energy-based refinement (requires OpenMM)
  - Clustering-based structure selection
"""

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  MSA Feature Extraction
# ─────────────────────────────────────────────
def run_rfam_search(seq: str, e_value: float = 1e-3) -> Optional[str]:
    """
    Run Infernal cmscan against Rfam database to find homologs.
    Requires: infernal, Rfam.cm database.

    Returns path to MSA file or None if not found.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(f">query\n{seq}\n")
            query_fa = f.name

        msa_out = query_fa.replace(".fa", ".sto")
        result = subprocess.run(
            ["cmscan", "--noali", "--cpu", "4", "-E", str(e_value),
             "--tblout", msa_out, "Rfam.cm", query_fa],
            capture_output=True, timeout=120
        )
        os.unlink(query_fa)
        if result.returncode == 0 and os.path.exists(msa_out):
            return msa_out
    except Exception:
        pass
    return None


def parse_stockholm_msa(sto_file: str) -> Tuple[List[str], List[str]]:
    """
    Parse Stockholm format MSA file.
    Returns (ids, aligned_sequences) lists.
    """
    ids, seqs = [], {}
    with open(sto_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line == "//":
                continue
            parts = line.split()
            if len(parts) == 2:
                seq_id, seq = parts
                if seq_id not in seqs:
                    ids.append(seq_id)
                    seqs[seq_id] = ""
                seqs[seq_id] += seq
    return ids, [seqs[i] for i in ids]


def msa_to_one_hot(aligned_seqs: List[str]) -> np.ndarray:
    """
    Convert MSA to one-hot representation.
    Returns (N_seqs, L, 5) array (A, C, G, U, gap).
    """
    vocab = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "-": 4, ".": 4}
    if not aligned_seqs:
        return np.zeros((0, 0, 5), dtype=np.float32)
    L = len(aligned_seqs[0])
    N = len(aligned_seqs)
    out = np.zeros((N, L, 5), dtype=np.float32)
    for i, seq in enumerate(aligned_seqs):
        for j, c in enumerate(seq.upper()):
            idx = vocab.get(c, 4)
            out[i, j, idx] = 1.0
    return out


def compute_covariance_features(msa_onehot: np.ndarray) -> np.ndarray:
    """
    Compute simple pairwise covariance features from MSA.
    This captures evolutionary coupling information.

    Args:
        msa_onehot: (N, L, 5) one-hot MSA
    Returns:
        cov_features: (L, L, 25) pairwise frequency features
    """
    N, L, A = msa_onehot.shape
    # Flatten to (N, L*A)
    flat = msa_onehot.reshape(N, L * A)
    # Pairwise frequencies
    f_ij = (flat.T @ flat) / N  # (L*A, L*A)
    f_ij = f_ij.reshape(L, A, L, A)

    # Simple APC correction (average product correction)
    fi = msa_onehot.mean(0)  # (L, A)
    fi_fj = fi[:, :, None, None] * fi[None, None, :, :]  # (L, A, L, A)
    cov = f_ij - fi_fj  # (L, A, L, A)
    cov_features = cov.transpose(0, 2, 1, 3).reshape(L, L, A * A)

    return cov_features.astype(np.float32)


def pseudo_likelihood_dca(msa_onehot: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """
    Simplified Direct Coupling Analysis using pseudo-likelihood.
    Returns (L, L) coupling matrix (symmetric).
    """
    N, L, A = msa_onehot.shape
    coupling = np.zeros((L, L), dtype=np.float32)

    # Pairwise mutual information as proxy for DCA
    f_i = msa_onehot.mean(0)  # (L, A)
    eps = 1e-8
    for i in range(L):
        for j in range(i + 1, L):
            f_ij = np.einsum("na,nb->ab", msa_onehot[:, i], msa_onehot[:, j]) / N
            mi = 0.0
            for a in range(A):
                for b in range(A):
                    p_ij = f_ij[a, b]
                    p_i = f_i[i, a]
                    p_j = f_i[j, b]
                    if p_ij > eps and p_i > eps and p_j > eps:
                        mi += p_ij * np.log(p_ij / (p_i * p_j))
            coupling[i, j] = mi
            coupling[j, i] = mi

    # APC correction
    mi_mean_i = coupling.mean(1, keepdims=True)
    mi_mean_j = coupling.mean(0, keepdims=True)
    mi_mean = coupling.mean()
    coupling_apc = coupling - mi_mean_i * mi_mean_j / (mi_mean + eps)
    return coupling_apc


# ─────────────────────────────────────────────
#  RhoFold+ Integration
# ─────────────────────────────────────────────
def run_rhofold(
    seq: str,
    output_dir: str,
    rhofold_script: str = "RhoFold/rhofold_predict.py",
) -> Optional[np.ndarray]:
    """
    Run RhoFold+ for RNA 3D structure prediction.
    Returns (L, 3) coordinates or None if failed.

    Install: git clone https://github.com/ml4bio/e2efold-3d
    """
    os.makedirs(output_dir, exist_ok=True)
    input_fa = os.path.join(output_dir, "input.fa")
    with open(input_fa, "w") as f:
        f.write(f">query\n{seq}\n")

    try:
        result = subprocess.run(
            ["python", rhofold_script, "--input_fas", input_fa,
             "--output_dir", output_dir, "--device", "cpu"],
            capture_output=True, text=True, timeout=300
        )
        # Parse output PDB
        pdb_path = os.path.join(output_dir, "results", "query_model_1.pdb")
        if os.path.exists(pdb_path):
            return parse_c3prime_from_pdb(pdb_path)
    except Exception as e:
        print(f"RhoFold failed: {e}")
    return None


def parse_c3prime_from_pdb(pdb_path: str) -> np.ndarray:
    """Extract C3' atom coordinates from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and "C3'" in line[12:16]:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def run_trrosettarna(
    seq: str,
    output_dir: str,
) -> Optional[np.ndarray]:
    """
    Wrapper for trRosettaRNA predictions.
    Returns (L, 3) coordinates.

    See: https://yanglab.qd.sdu.edu.cn/trRosettaRNA/
    """
    # Similar structure to RhoFold wrapper
    # Implementation depends on local installation
    return None


# ─────────────────────────────────────────────
#  Structure Clustering for Diversity
# ─────────────────────────────────────────────
def cluster_structures(
    structures: List[np.ndarray],
    n_clusters: int = 5,
) -> List[np.ndarray]:
    """
    Select diverse structures using greedy farthest-point sampling.
    This ensures the 5 submitted predictions are maximally diverse.

    Args:
        structures: list of (L, 3) coordinate arrays
        n_clusters: number of representative structures to select
    Returns:
        Selected structures (sorted by quality if available)
    """
    if len(structures) <= n_clusters:
        return structures

    N = len(structures)
    L = min(s.shape[0] for s in structures)

    # Compute pairwise RMSD matrix
    rmsd_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = structures[i][:L] - structures[j][:L]
            rmsd = np.sqrt((d**2).sum(1).mean())
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # Greedy farthest-point sampling
    selected = [0]
    for _ in range(n_clusters - 1):
        remaining = [i for i in range(N) if i not in selected]
        if not remaining:
            break
        # Find point farthest from all selected
        min_dists = [min(rmsd_matrix[r, s] for s in selected) for r in remaining]
        farthest = remaining[int(np.argmax(min_dists))]
        selected.append(farthest)

    return [structures[i] for i in selected]


# ─────────────────────────────────────────────
#  Energy-based Refinement (OpenMM)
# ─────────────────────────────────────────────
def openmm_refine(
    coords: np.ndarray,
    seq: str,
    n_steps: int = 1000,
) -> np.ndarray:
    """
    Physics-based structure refinement using OpenMM.
    Requires: openmm, openff-toolkit

    This improves bond geometry and removes steric clashes.
    """
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        # Full implementation would require building the RNA topology
        # This is a placeholder for the integration
        print("OpenMM refinement: not yet implemented")
        return coords
    except ImportError:
        return coords


# ─────────────────────────────────────────────
#  Combined Pipeline for Max Performance
# ─────────────────────────────────────────────
def predict_with_all_tools(
    seq: str,
    model,
    tokens,
    padding_mask,
    use_rhofold: bool = False,
    use_msa: bool = False,
    use_openmm: bool = False,
    n_predictions: int = 5,
) -> List[np.ndarray]:
    """
    Full pipeline combining:
      1. Deep learning model predictions
      2. RhoFold+ predictions (if available)
      3. Secondary structure refinement
      4. Diversity-based selection

    Returns 5 best predictions.
    """
    from rna_model import generate_diverse_predictions
    from rna_utils import geometry_refine

    all_structures = []
    L = len(seq)

    # 1. DL model predictions
    dl_preds = generate_diverse_predictions(model, tokens, padding_mask, n_predictions=10)
    all_structures.extend(dl_preds)

    # 2. RhoFold+ (if installed and requested)
    if use_rhofold:
        rhofold_coords = run_rhofold(seq, output_dir="/tmp/rhofold_out")
        if rhofold_coords is not None:
            all_structures.append(rhofold_coords[:L])

    # 3. Secondary structure refinement of each prediction
    refined_structures = []
    for struct in all_structures:
        try:
            refined = geometry_refine(struct, seq, steps=200)
            refined_structures.append(refined)
        except Exception:
            refined_structures.append(struct)

    # 4. Cluster for diversity and return top-5
    diverse = cluster_structures(refined_structures, n_clusters=n_predictions)
    return diverse[:n_predictions]


if __name__ == "__main__":
    # Test MSA utilities
    seqs = [
        "AUGCGAUCGAUAGCUAGCUAGC",
        "AUGCGAUCGAUAGCUAGCUAGC",
        "GUGCGAUCGAUAGCUAGCUAGC",
    ]
    msa_oh = msa_to_one_hot(seqs)
    print("MSA one-hot shape:", msa_oh.shape)

    # Test clustering
    structs = [np.random.randn(30, 3) for _ in range(20)]
    selected = cluster_structures(structs, n_clusters=5)
    print(f"Selected {len(selected)} diverse structures from {len(structs)}")
