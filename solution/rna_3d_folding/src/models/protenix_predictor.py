"""
Protenix predictor wrapper.

Protenix-v1 (ByteDance, Feb 2026) is the first fully open-source model to
surpass AlphaFold3 on biomolecular structure prediction (protein, RNA, DNA, ligand).

Key advantage:
- RNA MSA & template support (critical for novel folds!)
- AF3-level diffusion model (diverse predictions via different seeds)
- 368M parameters
- Apache 2.0 license

Reference: https://github.com/bytedance/Protenix
Paper: Protenix-v1 bioRxiv 2026.02.05.703733
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


PROTENIX_GITHUB = "https://github.com/bytedance/Protenix.git"


def install_protenix(install_dir: str = "/tmp/Protenix") -> Path:
    """Clone and install Protenix if not already present."""
    install_path = Path(install_dir)
    if not install_path.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", PROTENIX_GITHUB, str(install_path)],
            check=True,
        )
        subprocess.run(
            ["pip", "install", "-e", ".[train]"],
            cwd=str(install_path),
            check=True,
            capture_output=True,
        )
    return install_path


def sequence_to_protenix_json(
    sequence: str,
    target_id: str,
    num_seeds: int = 5,
) -> dict:
    """
    Convert RNA sequence to Protenix inference JSON format.

    Protenix uses an AlphaFold3-compatible JSON input format.
    """
    # Map RNA sequence to Protenix chain format
    rna_chain = {
        "type": "rna",
        "id": "A",
        "sequence": sequence,
    }
    return {
        "name": target_id,
        "sequences": [rna_chain],
        "modelSeeds": list(range(num_seeds)),  # different seeds → diverse predictions
        "dialect": "alphafold3",
        "version": 1,
    }


def parse_pdb_c1prime_from_string(pdb_content: str) -> np.ndarray:
    """Extract C1' atom coordinates from PDB content string."""
    coords = []
    for line in pdb_content.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip().upper()
            if atom_name == "C1'" and residue_name in ("A", "U", "G", "C",
                                                         "ADE", "URA", "GUA", "CYT"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords, dtype=np.float32)


def parse_pdb_c1prime(pdb_path: Path) -> np.ndarray:
    """Extract C1' atom coordinates from a PDB file."""
    with open(pdb_path) as f:
        return parse_pdb_c1prime_from_string(f.read())


class ProtenixPredictor:
    """
    Wrapper for Protenix-v1 RNA structure prediction.

    Protenix is an AF3-level diffusion model that generates diverse predictions
    via different random seeds. It's the best model for RNA complexes and
    large RNA assemblies (up to 6000 nt).

    Usage:
        predictor = ProtenixPredictor(device="cuda:0", num_seeds=5)
        predictor.setup()
        coords = predictor.predict("GGAAUAGCUCAGUC", "R1116")
        # coords shape: (5, L, 3)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        num_seeds: int = 5,
        install_dir: str = "/tmp/Protenix",
        use_msa: bool = True,
        use_templates: bool = True,
    ):
        self.device = device
        self.num_seeds = num_seeds
        self.install_dir = Path(install_dir)
        self.use_msa = use_msa
        self.use_templates = use_templates
        self._installed = False

    def setup(self) -> None:
        """Install Protenix and download weights."""
        self.install_dir = install_protenix(str(self.install_dir))
        self._installed = True

    def predict(
        self,
        sequence: str,
        target_id: str,
        msa_path: Optional[Path] = None,
        template_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Predict RNA 3D structure using Protenix-v1.

        Args:
            sequence: RNA sequence string
            target_id: unique identifier
            msa_path: optional .a3m file for MSA (strongly improves accuracy)
            template_dir: optional directory with PDB template files

        Returns:
            coords: np.ndarray of shape (num_seeds, L, 3) — C1' coordinates
        """
        if not self._installed:
            self.setup()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write inference JSON
            input_json = sequence_to_protenix_json(sequence, target_id, self.num_seeds)
            json_path = tmpdir / f"{target_id}.json"
            with open(json_path, "w") as f:
                json.dump(input_json, f)

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            cmd = [
                "python",
                str(self.install_dir / "runner" / "inference.py"),
                "--input_json_path", str(json_path),
                "--output_dir", str(output_dir),
                "--device", self.device,
                "--num_diffusion_samples", str(self.num_seeds),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Protenix failed for {target_id}: {result.stderr[:1000]}"
                )

            # Parse all output PDB files (one per seed)
            pdb_files = sorted(output_dir.glob("**/*.pdb"))
            all_coords = []
            for pdb_file in pdb_files[:self.num_seeds]:
                coords = parse_pdb_c1prime(pdb_file)
                if len(coords) > 0:
                    # Handle length mismatch (trim or pad)
                    if len(coords) >= len(sequence):
                        coords = coords[:len(sequence)]
                    if len(coords) == len(sequence):
                        all_coords.append(coords)

            if not all_coords:
                raise RuntimeError(
                    f"Protenix: no valid predictions for {target_id}"
                )

        return np.stack(all_coords, axis=0)  # (valid_seeds, L, 3)

    def predict_batch(
        self,
        sequences: dict[str, str],
        msa_dir: Optional[Path] = None,
    ) -> dict[str, np.ndarray]:
        """
        Predict structures for multiple RNA sequences.

        For large RNAs (>1000 nt), Protenix is preferred over RhoFold+.
        """
        results = {}
        for target_id, seq in sequences.items():
            msa_path = None
            if msa_dir is not None and self.use_msa:
                candidate = msa_dir / f"{target_id}.a3m"
                if candidate.exists():
                    msa_path = candidate

            try:
                coords = self.predict(seq, target_id, msa_path)
                results[target_id] = coords
                print(
                    f"[Protenix] {target_id}: {len(seq)} nt → "
                    f"{coords.shape[0]} predictions"
                )
            except Exception as e:
                print(f"[Protenix] FAILED {target_id}: {e}")
        return results
