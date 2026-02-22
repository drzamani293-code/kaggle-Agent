"""
Boltz-1 predictor wrapper.

Boltz-1 (MIT, Dec 2024) is fully open-source (MIT license) and achieves
AF3-level performance for biomolecular structure prediction.

Key advantage over RhoFold+:
- Handles RNA-protein and RNA-ligand complexes natively
- Boltz-steering: inference-time technique to fix hallucinations
- Diverse predictions via diffusion model with different seeds

Key advantage over AlphaFold3:
- Fully open source (MIT license — no commercial restrictions)
- Boltz-2: also predicts binding affinities (drug discovery)

Reference: https://github.com/jwohlwend/boltz
Paper: PMC11601547
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def install_boltz(install_dir: str = "/tmp/boltz") -> Path:
    """Install Boltz-1 from PyPI."""
    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pip", "install", "boltz", "-q"],
        check=True,
        capture_output=True,
    )
    return install_path


def write_boltz_yaml(
    sequence: str,
    target_id: str,
    yaml_path: Path,
    ligand_ccd: Optional[str] = None,
) -> None:
    """
    Write Boltz-1 YAML input file for RNA prediction.

    Boltz uses a YAML format to specify sequences and molecule types.
    """
    content = f"""version: 1
sequences:
  - rna:
      id: A
      sequence: {sequence}
"""
    if ligand_ccd is not None:
        content += f"""  - ligand:
      id: L
      ccd: {ligand_ccd}
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def parse_pdb_c1prime(pdb_path: Path) -> np.ndarray:
    """Extract C1' atom coordinates from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip().upper()
                if atom_name == "C1'" and residue_name in (
                    "A", "U", "G", "C", "ADE", "URA", "GUA", "CYT"
                ):
                    try:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        continue
    return np.array(coords, dtype=np.float32)


class BoltzPredictor:
    """
    Wrapper for Boltz-1 RNA structure prediction.

    Boltz-1 uses a diffusion-based approach similar to AlphaFold3,
    making it excellent for generating diverse structural conformations.
    Use Boltz-steering to correct physically impossible predictions.

    Usage:
        predictor = BoltzPredictor(device="cuda:0", num_seeds=3)
        predictor.setup()
        coords = predictor.predict("GGAAUAGCUCAGUC", "R1116")
        # coords shape: (3, L, 3)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        num_seeds: int = 3,
        num_recycling_steps: int = 3,
        num_diffusion_samples: int = 1,
        use_msa: bool = True,
        install_dir: str = "/tmp/boltz",
    ):
        self.device = device
        self.num_seeds = num_seeds
        self.num_recycling_steps = num_recycling_steps
        self.num_diffusion_samples = num_diffusion_samples
        self.use_msa = use_msa
        self.install_dir = Path(install_dir)
        self._installed = False

    def setup(self) -> None:
        """Install Boltz-1."""
        self.install_dir = install_boltz(str(self.install_dir))
        self._installed = True

    def predict(
        self,
        sequence: str,
        target_id: str,
        msa_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Predict RNA 3D structure using Boltz-1.

        Args:
            sequence: RNA sequence string
            target_id: unique identifier
            msa_path: optional .a3m MSA file

        Returns:
            coords: np.ndarray of shape (num_seeds, L, 3) — C1' coordinates
        """
        if not self._installed:
            self.setup()

        all_coords = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yaml_path = tmpdir / f"{target_id}.yaml"
            write_boltz_yaml(sequence, target_id, yaml_path)

            for seed in range(self.num_seeds):
                output_dir = tmpdir / f"seed_{seed}"
                output_dir.mkdir()

                cmd = [
                    "boltz", "predict",
                    str(yaml_path),
                    "--out_dir", str(output_dir),
                    "--cache", str(self.install_dir / "cache"),
                    "--seed", str(seed * 13 + 7),
                    "--recycling_steps", str(self.num_recycling_steps),
                    "--diffusion_samples", str(self.num_diffusion_samples),
                    "--sampling_steps", "200",  # more steps = better quality
                    "--accelerator", "gpu" if "cuda" in self.device else "cpu",
                ]
                if msa_path is not None and self.use_msa:
                    # Boltz supports MSA via data directory
                    cmd += ["--use_msa_server", "false"]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode != 0:
                    print(f"[Boltz] Seed {seed} failed: {result.stderr[:500]}")
                    continue

                # Parse output PDB
                pdb_files = sorted(output_dir.glob("**/*.pdb"))
                for pdb_file in pdb_files[:1]:  # take first (best ranking) per seed
                    coords = parse_pdb_c1prime(pdb_file)
                    if len(coords) >= len(sequence) * 0.95:  # allow minor discrepancy
                        coords = coords[:len(sequence)]
                        if len(coords) == len(sequence):
                            all_coords.append(coords)

        if not all_coords:
            raise RuntimeError(f"Boltz-1 failed to produce any structure for {target_id}")

        return np.stack(all_coords, axis=0)  # (num_valid_seeds, L, 3)

    def predict_batch(
        self,
        sequences: dict[str, str],
        msa_dir: Optional[Path] = None,
    ) -> dict[str, np.ndarray]:
        """Batch prediction for multiple RNA sequences."""
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
                print(
                    f"[Boltz] {target_id}: {len(seq)} nt → "
                    f"{coords.shape[0]} predictions"
                )
            except Exception as e:
                print(f"[Boltz] FAILED {target_id}: {e}")
        return results
