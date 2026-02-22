"""
RhoFold+ predictor wrapper.

RhoFold+ (Nature Methods, 2024) is the best single-chain RNA structure predictor.
Achieves ~0.58 avg TM-score on CASP15 RNA targets, faster than AlphaFold3.

Reference: https://github.com/ml4bio/RhoFold
Paper: Shen et al., Nature Methods 2024
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


RHOFOLD_GITHUB = "https://github.com/ml4bio/RhoFold.git"
RHOFOLD_WEIGHTS_URL = (
    "https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt"
)


def install_rhofold(install_dir: str = "/tmp/RhoFold") -> Path:
    """Clone and install RhoFold+ if not already present."""
    install_path = Path(install_dir)
    if not install_path.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", RHOFOLD_GITHUB, str(install_path)],
            check=True,
        )
        subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=str(install_path),
            check=True,
            capture_output=True,
        )
    # Download weights if missing
    weights_path = install_path / "pretrained" / "RhoFold_pretrained.pt"
    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["wget", "-q", RHOFOLD_WEIGHTS_URL, "-O", str(weights_path)],
            check=True,
        )
    return install_path


def write_fasta(sequence: str, target_id: str, fasta_path: Path) -> None:
    """Write RNA sequence to a FASTA file."""
    with open(fasta_path, "w") as f:
        f.write(f">{target_id}\n{sequence}\n")


def parse_pdb_c1prime(pdb_path: Path) -> np.ndarray:
    """
    Extract C1' atom coordinates from a PDB file.

    Returns shape (N, 3) array of [x, y, z] coordinates in Angstroms.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                if atom_name == "C1'" and residue_name in ("A", "U", "G", "C",
                                                             "DA", "DU", "DG", "DC"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


class RhoFoldPredictor:
    """
    Wrapper for RhoFold+ RNA structure prediction.

    Usage:
        predictor = RhoFoldPredictor(device="cuda:0")
        predictor.setup()
        coords = predictor.predict(sequence="GGAAUAGCUCAGUC", target_id="R1116")
        # coords shape: (num_seeds, L, 3)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        num_seeds: int = 4,
        install_dir: str = "/tmp/RhoFold",
        relax_steps: int = 1000,
    ):
        self.device = device
        self.num_seeds = num_seeds
        self.install_dir = Path(install_dir)
        self.relax_steps = relax_steps
        self._installed = False

    def setup(self) -> None:
        """Install RhoFold+ and download weights."""
        self.install_dir = install_rhofold(str(self.install_dir))
        self._installed = True

    def predict(
        self,
        sequence: str,
        target_id: str,
        msa_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Predict RNA 3D structure using RhoFold+.

        Args:
            sequence: RNA sequence string (A/U/G/C)
            target_id: unique identifier for logging
            msa_path: optional path to .a3m MSA file (improves accuracy significantly)

        Returns:
            coords: np.ndarray of shape (num_seeds, L, 3) — C1' coordinates
        """
        if not self._installed:
            self.setup()

        all_coords = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fasta_path = tmpdir / f"{target_id}.fasta"
            write_fasta(sequence, target_id, fasta_path)

            for seed in range(self.num_seeds):
                output_dir = tmpdir / f"seed_{seed}"
                output_dir.mkdir()

                cmd = [
                    "python",
                    str(self.install_dir / "inference.py"),
                    "--input_fas", str(fasta_path),
                    "--output_dir", str(output_dir),
                    "--device", self.device,
                    "--ckpt", str(
                        self.install_dir / "pretrained" / "RhoFold_pretrained.pt"
                    ),
                    "--seed", str(seed * 42),  # different seeds for diversity
                ]
                if msa_path is not None:
                    cmd += ["--input_a3m", str(msa_path)]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    print(f"[RhoFold] Seed {seed} failed: {result.stderr[:500]}")
                    continue

                # Parse relaxed PDB output
                pdb_files = sorted(output_dir.glob(f"relaxed_{self.relax_steps}_model.pdb"))
                if not pdb_files:
                    # Fall back to unrelaxed
                    pdb_files = sorted(output_dir.glob("unrelaxed_model.pdb"))
                if pdb_files:
                    coords = parse_pdb_c1prime(pdb_files[0])
                    if len(coords) == len(sequence):
                        all_coords.append(coords)

        if not all_coords:
            raise RuntimeError(f"RhoFold+ failed to produce any structure for {target_id}")

        return np.stack(all_coords, axis=0)  # (num_valid_seeds, L, 3)

    def predict_batch(
        self,
        sequences: dict[str, str],
        msa_dir: Optional[Path] = None,
    ) -> dict[str, np.ndarray]:
        """
        Predict structures for multiple RNA sequences.

        Args:
            sequences: dict of {target_id: sequence}
            msa_dir: optional directory containing {target_id}.a3m MSA files

        Returns:
            dict of {target_id: np.ndarray(num_seeds, L, 3)}
        """
        results = {}
        for target_id, seq in sequences.items():
            msa_path = None
            if msa_dir is not None:
                candidate = msa_dir / f"{target_id}.a3m"
                if candidate.exists():
                    msa_path = candidate
            try:
                coords = self.predict(seq, target_id, msa_path)
                results[target_id] = coords
                print(f"[RhoFold] {target_id}: {len(seq)} nt → {coords.shape[0]} predictions")
            except Exception as e:
                print(f"[RhoFold] FAILED {target_id}: {e}")
        return results
