"""
Ensemble selection strategy for RNA 3D structure prediction.

The competition scores using the BEST-of-5 predictions. This means:
1. We want DIVERSITY across the 5 predictions (not 5 copies of same structure)
2. We want each of the 5 to be HIGH QUALITY (high pLDDT/TM-score estimates)
3. If we have 15 candidates (5 from RhoFold + 5 from Protenix + 5 from Boltz),
   we need to select the 5 best/most diverse.

Key insight from Part 1 winner (d4t4 team):
"We ran three top models and selected five of the fifteen resulting predictions
for each target based on a heuristic score that balanced diversity with
similarity to the TBM and Protenix models."

Their scoring function:
- Protenix weight: 0.75 (most reliable)
- TBM weight: 0.15 (when homologs found)
- Boltz weight: 0.10

Strategy:
1. Compute pairwise TM-score between all candidate structures
2. Select greedy diverse subset: first pick highest-quality, then repeatedly
   pick the prediction most different from already-selected set
3. Weight by model confidence (pLDDT from B-factor column)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


def tm_score_numpy(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute approximate TM-score between two sets of C1' coordinates.

    This is a fast numpy approximation. For final evaluation, use TM-align.
    Uses the simplified TM-score formula for equal-length comparisons.

    Args:
        pred: (L, 3) predicted C1' coordinates
        ref: (L, 3) reference C1' coordinates

    Returns:
        Approximate TM-score in [0, 1]
    """
    L = len(pred)
    if L == 0:
        return 0.0

    # TM-score length-dependent cutoff d0
    if L >= 30:
        d0 = 0.6 * (L - 0.5) ** (1.0 / 3.0) - 2.5
    elif L >= 24:
        d0 = 0.7
    elif L >= 20:
        d0 = 0.6
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)

    # Superimpose using Kabsch algorithm (optimal rigid body rotation)
    aligned_pred = _kabsch_align(pred, ref)

    # Compute per-residue distances
    diffs = aligned_pred - ref
    d_i = np.sqrt(np.sum(diffs ** 2, axis=1))  # (L,)

    # TM-score sum
    tm = np.mean(1.0 / (1.0 + (d_i / d0) ** 2))
    return float(tm)


def _kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align mobile coordinates to target using Kabsch algorithm.

    Returns aligned mobile coordinates (same shape as mobile).
    """
    # Center
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mob = mobile - mobile_center
    tgt = target - target_center

    # SVD for optimal rotation
    H = mob.T @ tgt
    U, S, Vt = np.linalg.svd(H)

    # Ensure right-handed coordinate system
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    # Apply rotation and translation
    aligned = mob @ R.T + target_center
    return aligned


def compute_pairwise_tm_scores(
    candidates: list[np.ndarray],
    sample_size: int = 50,
) -> np.ndarray:
    """
    Compute pairwise TM-scores between all candidate structures.

    For efficiency, subsamples positions when L is large.

    Args:
        candidates: list of (L, 3) arrays
        sample_size: subsample to this many positions for speed

    Returns:
        (N, N) matrix of pairwise TM-scores
    """
    N = len(candidates)
    L = len(candidates[0])
    tm_matrix = np.eye(N, dtype=np.float32)

    # Subsample positions for efficiency
    if L > sample_size:
        indices = np.linspace(0, L - 1, sample_size, dtype=int)
    else:
        indices = np.arange(L)

    subsampled = [c[indices] for c in candidates]

    for i in range(N):
        for j in range(i + 1, N):
            score = tm_score_numpy(subsampled[i], subsampled[j])
            tm_matrix[i, j] = score
            tm_matrix[j, i] = score

    return tm_matrix


class EnsembleSelector:
    """
    Select the 5 best/most diverse predictions from multiple model outputs.

    Selection algorithm:
    1. Score all candidates by model confidence (pLDDT weighted by model priority)
    2. Greedily select diverse subset:
       - Pick highest-score candidate first
       - Each subsequent pick maximizes: quality_score + diversity_bonus
       - Diversity bonus = min distance (TM-score) to already selected

    This balances quality with structural diversity, maximizing best-of-5 score.

    Usage:
        selector = EnsembleSelector(n_select=5)
        best_5 = selector.select(
            candidates={
                "rhofold": rhofold_coords,    # (4, L, 3)
                "protenix": protenix_coords,  # (5, L, 3)
                "boltz": boltz_coords,        # (3, L, 3)
                "template": template_coord,   # (1, L, 3) or None
            },
            plddt_scores={
                "rhofold": [0.82, 0.79, 0.85, 0.77],
                "protenix": [0.88, 0.91, 0.86, 0.89, 0.87],
                "boltz": [0.79, 0.81, 0.76],
            }
        )
        # best_5: np.ndarray of shape (5, L, 3)
    """

    # Model weights based on Part 1 competition experience
    MODEL_WEIGHTS = {
        "protenix": 0.75,
        "template": 0.65,  # very high when identity > 70%
        "rhofold": 0.55,
        "boltz": 0.45,
        "trrosettarna": 0.60,
        "baseline": 0.10,
    }

    def __init__(
        self,
        n_select: int = 5,
        diversity_weight: float = 0.4,
        quality_weight: float = 0.6,
    ):
        self.n_select = n_select
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight

    def select(
        self,
        candidates: dict[str, Optional[np.ndarray]],
        plddt_scores: Optional[dict[str, list[float]]] = None,
        template_identity: float = 0.0,
    ) -> np.ndarray:
        """
        Select n_select diverse high-quality predictions.

        Args:
            candidates: dict of {model_name: coords_array}
                coords_array shape: (n_preds, L, 3) or (L, 3) for single prediction
            plddt_scores: optional per-prediction confidence scores
            template_identity: identity of best template (boosts template weight)

        Returns:
            selected: np.ndarray of shape (n_select, L, 3)
        """
        all_coords = []
        all_quality = []
        all_labels = []

        # Boost template weight if high identity
        model_weights = dict(self.MODEL_WEIGHTS)
        if template_identity > 0.7:
            model_weights["template"] = 0.90
        elif template_identity > 0.5:
            model_weights["template"] = 0.75

        # Flatten all candidates into a single list
        for model_name, coords in candidates.items():
            if coords is None:
                continue

            # Handle single prediction (L, 3) vs batch (N, L, 3)
            if coords.ndim == 2:
                coords = coords[np.newaxis]  # (1, L, 3)

            model_weight = model_weights.get(model_name, 0.5)

            for i, c in enumerate(coords):
                if c.shape[-1] != 3 or len(c) == 0:
                    continue
                all_coords.append(c)
                all_labels.append(f"{model_name}_{i}")

                # Base quality = model weight
                quality = model_weight
                # Add pLDDT bonus if available
                if plddt_scores and model_name in plddt_scores:
                    scores = plddt_scores[model_name]
                    if i < len(scores):
                        quality = 0.6 * model_weight + 0.4 * scores[i]
                all_quality.append(quality)

        if not all_coords:
            raise ValueError("No valid candidate predictions provided!")

        quality = np.array(all_quality)
        n_candidates = len(all_coords)

        if n_candidates <= self.n_select:
            # Not enough candidates — pad with best prediction copies
            selected = list(all_coords)
            best_idx = int(np.argmax(quality))
            while len(selected) < self.n_select:
                selected.append(all_coords[best_idx].copy())
            print(
                f"[Ensemble] Only {n_candidates} candidates available, "
                f"padding to {self.n_select}"
            )
            return np.stack(selected[:self.n_select], axis=0)

        # Compute pairwise TM-score matrix for diversity measurement
        print(f"[Ensemble] Computing pairwise TM-scores for {n_candidates} candidates...")
        tm_matrix = compute_pairwise_tm_scores(all_coords)
        # Convert TM-score to distance (1 - TM-score)
        dist_matrix = 1.0 - tm_matrix

        # Greedy diverse selection
        selected_indices = []
        scores = quality.copy()

        # First: always pick the highest quality prediction
        first_idx = int(np.argmax(scores))
        selected_indices.append(first_idx)

        for _ in range(self.n_select - 1):
            # Score each remaining candidate
            best_score = -np.inf
            best_candidate = None

            for j in range(n_candidates):
                if j in selected_indices:
                    continue

                # Quality component
                q_score = self.quality_weight * quality[j]

                # Diversity component: minimum distance to already selected
                min_dist = min(
                    dist_matrix[j, sel] for sel in selected_indices
                )
                d_score = self.diversity_weight * min_dist

                combined = q_score + d_score
                if combined > best_score:
                    best_score = combined
                    best_candidate = j

            if best_candidate is not None:
                selected_indices.append(best_candidate)

        selected = np.stack(
            [all_coords[i] for i in selected_indices], axis=0
        )  # (n_select, L, 3)

        for i, idx in enumerate(selected_indices):
            print(
                f"[Ensemble] Slot {i+1}: {all_labels[idx]} "
                f"(quality={quality[idx]:.3f})"
            )

        return selected

    def select_with_fallback(
        self,
        candidates: dict[str, Optional[np.ndarray]],
        sequence: str,
        plddt_scores: Optional[dict[str, list[float]]] = None,
        template_identity: float = 0.0,
    ) -> np.ndarray:
        """
        Select with baseline fallback if no models produced results.

        Falls back to a random coil model if all predictors fail.
        """
        try:
            return self.select(candidates, plddt_scores, template_identity)
        except ValueError:
            print("[Ensemble] All models failed! Using random coil baseline...")
            return generate_random_coil_baseline(sequence, n_structures=self.n_select)


def generate_random_coil_baseline(
    sequence: str,
    n_structures: int = 5,
    bond_length: float = 3.8,  # typical C1'-C1' distance in RNA (Angstroms)
) -> np.ndarray:
    """
    Generate random coil baseline structures.

    Uses a random walk with bond length constraints as a fallback when
    all models fail. This is a last resort — TM-score will be poor.

    Returns:
        np.ndarray of shape (n_structures, L, 3)
    """
    L = len(sequence)
    all_coords = []
    rng = np.random.RandomState(42)

    for seed in range(n_structures):
        rng_s = np.random.RandomState(seed)
        coords = np.zeros((L, 3), dtype=np.float32)

        # Random walk in 3D with fixed bond length
        for i in range(1, L):
            # Random direction
            theta = rng_s.uniform(0, np.pi)
            phi = rng_s.uniform(0, 2 * np.pi)
            dx = bond_length * np.sin(theta) * np.cos(phi)
            dy = bond_length * np.sin(theta) * np.sin(phi)
            dz = bond_length * np.cos(theta)
            coords[i] = coords[i - 1] + np.array([dx, dy, dz])

        all_coords.append(coords)

    return np.stack(all_coords, axis=0)  # (n_structures, L, 3)
