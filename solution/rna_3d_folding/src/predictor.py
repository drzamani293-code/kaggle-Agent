"""
Main RNA 3D structure prediction orchestrator.

Implements the winning strategy for Stanford RNA 3D Folding Part 2:

STRATEGY OVERVIEW:
==================

Phase 1: Template Search
  - BLAST query sequence against PDB RNA structures
  - High-identity hits (>50%) → use as structural templates
  - Novel folds → skip to Phase 2

Phase 2: MSA Generation
  - Search RNAcentral database with nhmmer
  - Build multiple sequence alignment for covariation analysis
  - Even sparse MSAs help diffusion-based models

Phase 3: Multi-Model Prediction
  - Protenix-v1 (5 seeds): AF3-level, best for complexes and large RNA
  - RhoFold+ (4 seeds): Best for single-chain RNA, fast
  - Boltz-1 (3 seeds): Diverse diffusion-based predictions
  - Template-based (1-4): Direct from homology modeling
  → Total: up to 16 candidate structures

Phase 4: Ensemble Selection
  - Greedy diverse selection from 16 candidates → 5 final predictions
  - Balance quality (model confidence/pLDDT) vs diversity (TM-distance)
  - Protenix weight: 0.75, Template: 0.65, RhoFold: 0.55, Boltz: 0.45

Phase 5: Output
  - Extract C1' coordinates for each residue
  - Format submission.csv with 5 predictions per RNA
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .ensemble import EnsembleSelector, generate_random_coil_baseline
from .msa_generation import MSAGenerator
from .models import BoltzPredictor, ProtenixPredictor, RhoFoldPredictor
from .template_search import TemplateSearcher


class RNA3DPredictor:
    """
    Main orchestrator for the Stanford RNA 3D Folding Part 2 competition.

    This implements the multi-model ensemble strategy that combines:
    - Protenix-v1 (primary, AF3-level)
    - RhoFold+ (best single-chain RNA predictor)
    - Boltz-1 (diverse diffusion-based predictions)
    - Template-based modeling (when homologs exist)

    Usage:
        predictor = RNA3DPredictor(
            device="cuda:0",
            use_protenix=True,
            use_rhofold=True,
            use_boltz=True,
            use_templates=True,
        )
        predictor.setup()

        # Run on competition test set
        submission_df = predictor.predict_from_csv(
            test_csv_path="/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv",
            output_path="submission.csv",
        )
    """

    def __init__(
        self,
        device: str = "cuda:0",
        use_protenix: bool = True,
        use_rhofold: bool = True,
        use_boltz: bool = True,
        use_templates: bool = True,
        use_msa: bool = True,
        protenix_seeds: int = 5,
        rhofold_seeds: int = 4,
        boltz_seeds: int = 3,
        msa_db_path: Optional[str] = None,
        template_db_path: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        cache_dir: str = "/tmp/rna3d_cache",
    ):
        self.device = device
        self.use_protenix = use_protenix
        self.use_rhofold = use_rhofold
        self.use_boltz = use_boltz
        self.use_templates = use_templates
        self.use_msa = use_msa
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models (lazy setup)
        self._protenix = ProtenixPredictor(device=device, num_seeds=protenix_seeds)
        self._rhofold = RhoFoldPredictor(device=device, num_seeds=rhofold_seeds)
        self._boltz = BoltzPredictor(device=device, num_seeds=boltz_seeds)
        self._template_searcher = TemplateSearcher(
            pdb_dir=pdb_dir or str(self.cache_dir / "pdb"),
            blast_db_path=template_db_path,
        )
        self._msa_generator = MSAGenerator(db_path=msa_db_path)
        self._ensemble_selector = EnsembleSelector(n_select=5)

        self._setup_done = False

    def setup(self) -> None:
        """Initialize all models and databases."""
        print("Setting up RNA 3D predictor...")

        if self.use_protenix:
            print("  [1/4] Setting up Protenix-v1...")
            self._protenix.setup()

        if self.use_rhofold:
            print("  [2/4] Setting up RhoFold+...")
            self._rhofold.setup()

        if self.use_boltz:
            print("  [3/4] Setting up Boltz-1...")
            self._boltz.setup()

        if self.use_templates:
            print("  [4/4] Setting up template searcher...")
            self._template_searcher.setup()

        if self.use_msa:
            self._msa_generator.setup()

        self._setup_done = True
        print("Setup complete!")

    def predict_single(
        self,
        sequence: str,
        target_id: str,
        cutoff_date: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict 5 diverse 3D structures for a single RNA sequence.

        Args:
            sequence: RNA sequence string (A/U/G/C only)
            target_id: unique identifier for this target
            cutoff_date: competition cutoff date (exclude templates after this)

        Returns:
            coords: np.ndarray of shape (5, L, 3) — 5 C1' coordinate predictions
        """
        if not self._setup_done:
            self.setup()

        L = len(sequence)
        print(f"\n{'='*60}")
        print(f"Predicting {target_id}: {L} nucleotides")
        print(f"{'='*60}")

        candidates = {}
        plddt_scores = {}
        template_identity = 0.0

        msa_dir = self.cache_dir / "msa"
        msa_dir.mkdir(exist_ok=True)

        # Step 1: Generate MSA
        msa_path = None
        if self.use_msa:
            print("\n[Phase 1] Generating MSA...")
            msa_path = self._msa_generator.generate(sequence, target_id, msa_dir)

        # Step 2: Template search
        if self.use_templates:
            print("\n[Phase 2] Searching for structural templates...")
            template_pred = self._template_searcher.get_template_prediction(
                sequence, target_id, cutoff_date
            )
            if template_pred is not None:
                candidates["template"] = template_pred[np.newaxis]  # (1, L, 3)
                template_identity = getattr(
                    self._template_searcher, "_last_identity", 0.0
                )
            else:
                print("  No templates found — novel fold, using de novo methods only")

        # Step 3: Run Protenix (primary model)
        if self.use_protenix:
            print("\n[Phase 3a] Running Protenix-v1 (primary)...")
            t0 = time.time()
            try:
                protenix_coords = self._protenix.predict(sequence, target_id, msa_path)
                candidates["protenix"] = protenix_coords
                print(f"  Protenix: {protenix_coords.shape[0]} predictions in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  Protenix FAILED: {e}")

        # Step 4: Run RhoFold+ (best for single-chain RNA)
        if self.use_rhofold:
            print("\n[Phase 3b] Running RhoFold+...")
            t0 = time.time()
            try:
                rhofold_coords = self._rhofold.predict(sequence, target_id, msa_path)
                candidates["rhofold"] = rhofold_coords
                print(f"  RhoFold+: {rhofold_coords.shape[0]} predictions in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  RhoFold+ FAILED: {e}")

        # Step 5: Run Boltz-1 (diverse diffusion predictions)
        if self.use_boltz:
            print("\n[Phase 3c] Running Boltz-1...")
            t0 = time.time()
            try:
                boltz_coords = self._boltz.predict(sequence, target_id, msa_path)
                candidates["boltz"] = boltz_coords
                print(f"  Boltz-1: {boltz_coords.shape[0]} predictions in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  Boltz-1 FAILED: {e}")

        # Step 6: Select best 5 diverse predictions
        print("\n[Phase 4] Selecting 5 best diverse predictions...")
        selected = self._ensemble_selector.select_with_fallback(
            candidates=candidates,
            sequence=sequence,
            plddt_scores=plddt_scores,
            template_identity=template_identity,
        )

        print(f"\nDone: {target_id} → {selected.shape} predictions")
        return selected  # (5, L, 3)

    def predict_from_csv(
        self,
        test_csv_path: str,
        output_path: str = "submission.csv",
        cutoff_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run predictions on all sequences in the test CSV and generate submission.

        Args:
            test_csv_path: path to test_sequences.csv
            output_path: where to save submission.csv
            cutoff_date: competition temporal cutoff for template filtering

        Returns:
            Submission DataFrame
        """
        test_df = pd.read_csv(test_csv_path)
        print(f"Loaded {len(test_df)} test sequences from {test_csv_path}")

        # Identify column names (handle variations)
        id_col = "target_id" if "target_id" in test_df.columns else test_df.columns[0]
        seq_col = "sequence" if "sequence" in test_df.columns else test_df.columns[1]
        cutoff_col = "temporal_cutoff" if "temporal_cutoff" in test_df.columns else None

        all_rows = []
        for _, row in test_df.iterrows():
            target_id = str(row[id_col])
            sequence = str(row[seq_col]).upper().replace("T", "U")  # DNA→RNA conversion

            # Use per-sequence cutoff if available
            seq_cutoff = cutoff_date
            if cutoff_col and cutoff_col in row:
                seq_cutoff = str(row[cutoff_col])

            try:
                # Get 5 predictions: shape (5, L, 3)
                predictions = self.predict_single(sequence, target_id, seq_cutoff)

                # Build submission rows
                for resid, nt in enumerate(sequence, start=1):
                    submission_row = {
                        "ID": f"{target_id}_{resid}",
                        "resname": nt,
                        "resid": resid,
                    }
                    for pred_idx in range(5):
                        x, y, z = predictions[pred_idx, resid - 1]
                        submission_row[f"x_{pred_idx+1}"] = round(float(x), 3)
                        submission_row[f"y_{pred_idx+1}"] = round(float(y), 3)
                        submission_row[f"z_{pred_idx+1}"] = round(float(z), 3)
                    all_rows.append(submission_row)

            except Exception as e:
                print(f"FATAL: Failed to predict {target_id}: {e}")
                # Emergency fallback: random coil
                fallback = generate_random_coil_baseline(sequence, n_structures=5)
                for resid, nt in enumerate(sequence, start=1):
                    submission_row = {
                        "ID": f"{target_id}_{resid}",
                        "resname": nt,
                        "resid": resid,
                    }
                    for pred_idx in range(5):
                        x, y, z = fallback[pred_idx, resid - 1]
                        submission_row[f"x_{pred_idx+1}"] = round(float(x), 3)
                        submission_row[f"y_{pred_idx+1}"] = round(float(y), 3)
                        submission_row[f"z_{pred_idx+1}"] = round(float(z), 3)
                    all_rows.append(submission_row)

        # Build submission DataFrame
        col_order = (
            ["ID", "resname", "resid"]
            + [f"{c}_{i}" for i in range(1, 6) for c in ("x", "y", "z")]
        )
        submission_df = pd.DataFrame(all_rows)
        # Reorder columns to match expected format
        available_cols = [c for c in col_order if c in submission_df.columns]
        submission_df = submission_df[available_cols]

        submission_df.to_csv(output_path, index=False)
        print(f"\nSubmission saved: {output_path} ({len(submission_df)} rows)")
        return submission_df
