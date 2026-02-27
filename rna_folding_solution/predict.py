"""
Inference script for RNA 3D structure prediction.
Generates 5 diverse predictions per RNA sequence for submission.

Usage:
    python predict.py \
        --model_path checkpoints/best_model.pt \
        --test_csv /path/to/test.csv \
        --output submission.csv
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rna_model import RNAFoldModel, RNADataset, N_PREDICTIONS, MAX_LEN
from rna_utils import (
    load_sequences_from_csv,
    format_submission,
    generate_diverse_predictions,
    save_pdb,
    predict_secondary_structure_viennarna,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Ensemble of multiple checkpoints
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> RNAFoldModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    model = RNAFoldModel(
        d_model=args.get("d_model", 256),
        n_encoder_layers=args.get("n_encoder_layers", 8),
        n_pair_layers=args.get("n_pair_layers", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def predict_ensemble(
    models: List[RNAFoldModel],
    tokens: torch.Tensor,
    padding_mask: torch.Tensor,
    n_predictions: int = N_PREDICTIONS,
    device: torch.device = torch.device("cpu"),
) -> List[np.ndarray]:
    """
    Generate diverse predictions using an ensemble of models.

    Strategy:
      - For each model: get greedy prediction
      - For best model: get dropout-sampled predictions
      - Combine and return top-5 by pLDDT
    """
    all_predictions = []

    tokens = tokens.to(device)
    padding_mask = padding_mask.to(device)

    # Greedy predictions from each model
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(tokens, padding_mask)

        L_valid = int((~padding_mask[0]).sum().item())
        coords = out["coords"][0, -L_valid:].cpu().numpy()
        plddt = out["plddt"][0, -L_valid:].mean().item()
        all_predictions.append((plddt, coords))

    # Dropout-sampled predictions from primary model
    primary_model = models[0]
    primary_model.train()  # enable dropout for sampling
    n_extra = max(0, n_predictions - len(models))
    for _ in range(n_extra):
        with torch.no_grad():
            out = primary_model(tokens, padding_mask)
        L_valid = int((~padding_mask[0]).sum().item())
        coords = out["coords"][0, -L_valid:].cpu().numpy()
        plddt = out["plddt"][0, -L_valid:].mean().item()
        all_predictions.append((plddt, coords))
    primary_model.eval()

    # Sort by pLDDT descending and take top-5
    all_predictions.sort(key=lambda x: -x[0])
    return [c for _, c in all_predictions[:n_predictions]]


# ─────────────────────────────────────────────
#  Per-sequence post-processing
# ─────────────────────────────────────────────
def refine_with_secondary_structure(
    coords: np.ndarray,
    seq: str,
    n_steps: int = 100,
    k_bp: float = 0.5,
    k_bb: float = 2.0,
) -> np.ndarray:
    """
    Simple gradient-free energy minimization using:
      - Backbone bond length restraint (ideal: 3.8 Å for C3'-C3' consecutive)
      - Base-pair distance restraint (ideal: 7.5 Å for Watson-Crick pairs)

    This is a lightweight simulation that slightly improves geometry.
    """
    ss, _ = predict_secondary_structure_viennarna(seq)
    from rna_utils import dot_bracket_to_pairs

    pairs = dot_bracket_to_pairs(ss)
    coords = coords.copy()
    L = len(coords)

    # Target distances
    BACKBONE_DIST = 3.8  # Å between consecutive C3' atoms
    BP_DIST = 7.5  # Å between base-paired nucleotides

    learning_rate = 0.1
    for step in range(n_steps):
        forces = np.zeros_like(coords)

        # Backbone restraints
        for i in range(L - 1):
            diff = coords[i + 1] - coords[i]
            d = np.linalg.norm(diff)
            if d < 1e-8:
                continue
            # Spring force
            force_mag = k_bb * (d - BACKBONE_DIST)
            unit = diff / d
            forces[i] += force_mag * unit
            forces[i + 1] -= force_mag * unit

        # Base-pair restraints
        for i, j in pairs:
            if i >= L or j >= L:
                continue
            diff = coords[j] - coords[i]
            d = np.linalg.norm(diff)
            if d < 1e-8:
                continue
            force_mag = k_bp * (d - BP_DIST)
            unit = diff / d
            forces[i] += force_mag * unit
            forces[j] -= force_mag * unit

        # Gradient descent step
        coords += learning_rate * forces
        learning_rate *= 0.99  # Decay

    return coords


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (or comma-separated list for ensemble)",
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Path to test sequences CSV"
    )
    parser.add_argument(
        "--output", type=str, default="submission.csv", help="Output submission file"
    )
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_predictions", type=int, default=N_PREDICTIONS)
    parser.add_argument("--refine", action="store_true", help="Apply secondary structure refinement")
    parser.add_argument("--save_pdb", action="store_true", help="Save PDB files for visualization")
    parser.add_argument("--pdb_dir", type=str, default="pdb_outputs")
    parser.add_argument("--seq_col", type=str, default="sequence")
    parser.add_argument("--id_col", type=str, default="target_id")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model(s)
    checkpoint_paths = [p.strip() for p in args.model_path.split(",")]
    models = [load_model(p, device) for p in checkpoint_paths if os.path.exists(p)]

    if not models:
        # Create untrained model as fallback (for testing pipeline)
        logger.warning("No checkpoints found — using untrained model (for debug only)")
        models = [RNAFoldModel().to(device)]

    # Load test sequences
    ids, seqs = load_sequences_from_csv(
        args.test_csv, seq_col=args.seq_col, id_col=args.id_col
    )
    logger.info(f"Test sequences: {len(seqs)}")

    if args.save_pdb:
        os.makedirs(args.pdb_dir, exist_ok=True)

    # Dataset / loader
    test_ds = RNADataset(seqs, max_len=args.max_len)

    all_predictions = []

    for idx in tqdm(range(len(test_ds)), desc="Predicting"):
        item = test_ds[idx]
        tokens = item["tokens"].unsqueeze(0)  # (1, L)
        padding_mask = item["padding_mask"].unsqueeze(0)  # (1, L)
        seq = item["seq"]

        preds = predict_ensemble(
            models, tokens, padding_mask, args.n_predictions, device
        )

        # Optional: refine with secondary structure
        if args.refine:
            preds = [refine_with_secondary_structure(p, seq) for p in preds]

        all_predictions.append(preds)

        # Save PDB files
        if args.save_pdb:
            for model_idx, coords in enumerate(preds):
                pdb_path = os.path.join(args.pdb_dir, f"{ids[idx]}_model{model_idx+1}.pdb")
                save_pdb(coords, seq, pdb_path, model_num=model_idx + 1)

    # Build and save submission
    submission_df = format_submission(ids, all_predictions)
    submission_df.to_csv(args.output, index=False)
    logger.info(f"Submission saved to {args.output} ({len(submission_df)} rows)")
    logger.info(f"Preview:\n{submission_df.head(10)}")


if __name__ == "__main__":
    main()
