"""
Utility functions for RNA 3D structure prediction:
  - Sequence parsing and encoding
  - Secondary structure prediction wrappers
  - TM-score computation
  - Submission file generation
  - Multiple Sequence Alignment helpers
"""

import re
import subprocess
import tempfile
import os
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Sequence utilities
# ─────────────────────────────────────────────
NT_VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}


def encode_sequence(seq: str) -> np.ndarray:
    """Map RNA sequence string to integer array."""
    seq = seq.upper().replace("T", "U")
    return np.array([NT_VOCAB.get(c, 4) for c in seq], dtype=np.int64)


def one_hot_encode(seq: str, include_N: bool = True) -> np.ndarray:
    """One-hot encode RNA sequence. Shape: (L, 5) if include_N else (L, 4)."""
    n_channels = 5 if include_N else 4
    enc = np.zeros((len(seq), n_channels), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        idx = NT_VOCAB.get(c, 4)
        if idx < n_channels:
            enc[i, idx] = 1.0
    return enc


def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / max(len(seq), 1)


def sliding_window_features(seq: str, window: int = 5) -> np.ndarray:
    """
    Compute sliding-window nucleotide composition features.
    Returns (L, 4*window) — flattened one-hot of context window.
    """
    seq = seq.upper().replace("T", "U")
    L = len(seq)
    pad = window // 2
    padded = "N" * pad + seq + "N" * pad
    feats = []
    for i in range(L):
        window_seq = padded[i : i + window]
        oh = one_hot_encode(window_seq, include_N=False)[:, :4]
        feats.append(oh.flatten())
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────
#  Secondary Structure (Vienna RNA / fallback)
# ─────────────────────────────────────────────
def predict_secondary_structure_viennarna(seq: str) -> Tuple[str, float]:
    """
    Predict secondary structure using ViennaRNA (RNAfold).
    Falls back to a simple heuristic if not installed.

    Returns:
        dot_bracket: str  — secondary structure string
        mfe:         float — minimum free energy (kcal/mol)
    """
    try:
        import RNA  # ViennaRNA Python bindings
        fc = RNA.fold_compound(seq.upper().replace("T", "U"))
        structure, mfe = fc.mfe()
        return structure, mfe
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["RNAfold", "--noPS"],
            input=seq.upper().replace("T", "U"),
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].rsplit(" ", 1)
            structure = parts[0].strip()
            mfe_str = parts[1].strip("()") if len(parts) > 1 else "0.0"
            return structure, float(mfe_str)
    except Exception:
        pass

    # Simple heuristic fallback: all unpaired
    return "." * len(seq), 0.0


def dot_bracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """Convert dot-bracket notation to list of (i, j) base-pair indices."""
    pairs = []
    stack = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs


def pairs_to_matrix(pairs: List[Tuple[int, int]], L: int) -> np.ndarray:
    """Convert base-pair list to (L, L) binary adjacency matrix."""
    mat = np.zeros((L, L), dtype=np.float32)
    for i, j in pairs:
        mat[i, j] = 1.0
        mat[j, i] = 1.0
    return mat


# ─────────────────────────────────────────────
#  MDS-based Coordinate Initialization
# ─────────────────────────────────────────────
def mds_from_distances(dist_matrix: np.ndarray, n_dim: int = 3) -> np.ndarray:
    """
    Classical Multidimensional Scaling (CMDS) to obtain initial 3D coordinates
    from a pairwise distance matrix.

    Args:
        dist_matrix: (L, L) symmetric distance matrix
        n_dim:       number of output dimensions (3 for 3D)
    Returns:
        coords: (L, n_dim)
    """
    L = dist_matrix.shape[0]
    D2 = dist_matrix ** 2

    # Double centering
    J = np.eye(L) - np.ones((L, L)) / L
    B = -0.5 * J @ D2 @ J

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n_dim
    eigenvalues_pos = np.maximum(eigenvalues[:n_dim], 0)
    coords = eigenvectors[:, :n_dim] * np.sqrt(eigenvalues_pos)
    return coords.astype(np.float32)


def build_ideal_rna_backbone(L: int) -> np.ndarray:
    """
    Build an ideal A-form RNA helix backbone as starting coordinates.
    Used for initialization when no distance prediction is available.

    Returns: (L, 3) coordinates in Å
    """
    coords = np.zeros((L, 3), dtype=np.float32)
    # A-form RNA: rise ~2.8 Å/bp, twist ~32.7°/bp, radius ~9 Å
    rise = 2.8
    twist = np.radians(32.7)
    radius = 9.0

    for i in range(L):
        angle = i * twist
        coords[i, 0] = radius * np.cos(angle)
        coords[i, 1] = radius * np.sin(angle)
        coords[i, 2] = i * rise

    return coords


# ─────────────────────────────────────────────
#  TM-score Calculation
# ─────────────────────────────────────────────
def tm_score(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    Lref: Optional[int] = None,
) -> float:
    """
    Compute TM-score between predicted and reference 3D structures.
    Uses the standard TM-score formula with Kabsch alignment.

    Args:
        pred_coords: (L, 3) predicted coordinates
        true_coords: (L, 3) reference coordinates
        Lref:        reference length (default = len(true_coords))
    Returns:
        tm: float in (0, 1]
    """
    if Lref is None:
        Lref = len(true_coords)
    L = min(len(pred_coords), len(true_coords))
    Lref = max(Lref, 1)

    d0 = 1.24 * (Lref - 15) ** (1 / 3) - 1.8 if Lref > 19 else 0.5

    # Kabsch alignment
    P = pred_coords[:L].copy()
    Q = true_coords[:L].copy()

    P -= P.mean(axis=0)
    Q -= Q.mean(axis=0)

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, det])
    R = Vt.T @ D @ U.T
    P_aligned = P @ R.T

    # TM-score
    di2 = ((P_aligned - Q) ** 2).sum(axis=1)
    tm = (1 / (1 + di2 / d0 ** 2)).mean()
    return float(tm)


def best_of_five_tm_score(
    predictions: List[np.ndarray],
    true_coords: np.ndarray,
) -> Tuple[float, int]:
    """
    Evaluate best-of-5 TM-score (competition metric).

    Returns:
        best_tm:  float
        best_idx: int
    """
    scores = [tm_score(p, true_coords) for p in predictions]
    best_idx = int(np.argmax(scores))
    return scores[best_idx], best_idx


# ─────────────────────────────────────────────
#  Submission File Generation
# ─────────────────────────────────────────────
def format_submission(
    target_ids: List[str],
    predictions_list: List[List[np.ndarray]],
) -> pd.DataFrame:
    """
    Build the Kaggle submission DataFrame.

    The expected format (based on competition page) is:
      id, x_1, y_1, z_1 for each nucleotide across 5 models.

    Each row corresponds to one nucleotide in one model prediction.
    Columns: ID, x, y, z

    Args:
        target_ids:       list of sequence IDs
        predictions_list: list of lists of (L, 3) arrays (5 per sequence)
    Returns:
        DataFrame with submission rows
    """
    rows = []
    for seq_id, preds in zip(target_ids, predictions_list):
        for model_idx, coords in enumerate(preds):
            L = coords.shape[0]
            for nt_idx in range(L):
                rows.append(
                    {
                        "ID": f"{seq_id}_{model_idx+1}_{nt_idx+1}",
                        "x": float(coords[nt_idx, 0]),
                        "y": float(coords[nt_idx, 1]),
                        "z": float(coords[nt_idx, 2]),
                    }
                )
    return pd.DataFrame(rows)


def save_pdb(coords: np.ndarray, seq: str, filename: str, model_num: int = 1):
    """Save predicted coordinates as a PDB file for visualization."""
    atom_names = {
        "A": "C3'", "C": "C3'", "G": "C3'", "U": "C3'", "N": "C3'"
    }
    residue_names = {
        "A": "  A", "C": "  C", "G": "  G", "U": "  U", "N": "  N"
    }
    with open(filename, "w") as f:
        f.write(f"MODEL     {model_num:4d}\n")
        for i, (nt, xyz) in enumerate(zip(seq, coords)):
            nt = nt.upper()
            atom_name = atom_names.get(nt, "C3'")
            res_name = residue_names.get(nt, "  N")
            f.write(
                f"ATOM  {i+1:5d}  {atom_name:<3s} {res_name} A{i+1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n"
            )
        f.write("ENDMDL\n")


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────
def load_sequences_from_csv(
    csv_path: str,
    seq_col: str = "sequence",
    id_col: str = "target_id",
) -> Tuple[List[str], List[str]]:
    """Load sequences and IDs from a CSV file."""
    df = pd.read_csv(csv_path)
    ids = df[id_col].tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    seqs = df[seq_col].tolist()
    return ids, seqs


def load_training_coords(
    csv_path: str,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    seq_id_col: str = "target_id",
) -> Dict[str, np.ndarray]:
    """
    Load training structure coordinates grouped by sequence ID.
    Returns dict mapping ID -> (L, 3) array.
    """
    df = pd.read_csv(csv_path)
    coords_dict = {}
    for seq_id, group in df.groupby(seq_id_col):
        coords = group[[x_col, y_col, z_col]].values.astype(np.float32)
        coords_dict[str(seq_id)] = coords
    return coords_dict


if __name__ == "__main__":
    # Quick tests
    seq = "AUGCGAUCGAUAGCUAGCUAGC"
    enc = encode_sequence(seq)
    print("Encoded:", enc)

    oh = one_hot_encode(seq)
    print("One-hot shape:", oh.shape)

    ss, mfe = predict_secondary_structure_viennarna(seq)
    print(f"Secondary structure: {ss}  MFE: {mfe:.2f}")

    backbone = build_ideal_rna_backbone(len(seq))
    print("Backbone shape:", backbone.shape)

    # TM-score self-test
    tm = tm_score(backbone, backbone)
    print(f"TM-score (self): {tm:.4f}")  # should be ~1.0
