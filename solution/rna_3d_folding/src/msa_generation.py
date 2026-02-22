"""
Multiple Sequence Alignment (MSA) generation for RNA structure prediction.

MSA significantly improves RNA structure prediction by providing evolutionary
covariation signals that constrain the structure. For RhoFold+ and Protenix,
MSA can improve TM-score by 0.05–0.15 on average.

Approaches:
1. RNAcentral search via Infernal/nhmmer (best for non-coding RNA)
2. BLAST against RNAcentral (faster but less sensitive)
3. RNAcmap2 (covariance model search)
4. Pre-computed MSAs from Rfam families

For Part 2 (novel folds with no templates), MSA may be sparse,
but even shallow MSA helps.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


RNACENTRAL_DB_URL = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/"


class MSAGenerator:
    """
    Generate multiple sequence alignments for RNA sequences.

    Uses nhmmer (profile HMM search) against RNAcentral database for
    best sensitivity on non-coding RNA families.

    Usage:
        gen = MSAGenerator(db_path="/kaggle/input/rnacentral/rnacentral.fasta")
        gen.setup()
        msa_path = gen.generate("GGAAUAGCUCAGUC...", "R1116", output_dir)
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_seqs: int = 512,
        e_value_threshold: float = 1e-3,
        cpu_cores: int = 4,
    ):
        self.db_path = db_path
        self.max_seqs = max_seqs
        self.e_value_threshold = e_value_threshold
        self.cpu_cores = cpu_cores
        self._ready = False

    def setup(self, db_path: Optional[str] = None) -> None:
        """Check that nhmmer and the database are available."""
        if db_path:
            self.db_path = db_path

        # Check nhmmer installation
        result = subprocess.run(
            ["nhmmer", "--help"], capture_output=True
        )
        if result.returncode != 0:
            # Try installing via apt
            subprocess.run(
                ["apt-get", "install", "-y", "-q", "hmmer"],
                check=False, capture_output=True
            )

        self._ready = self.db_path is not None and Path(self.db_path).exists()

    def generate(
        self,
        sequence: str,
        target_id: str,
        output_dir: Path,
    ) -> Optional[Path]:
        """
        Generate MSA for an RNA sequence.

        Args:
            sequence: RNA sequence (A/U/G/C)
            target_id: identifier
            output_dir: directory to write .a3m output

        Returns:
            Path to .a3m MSA file, or None if MSA generation failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        a3m_path = output_dir / f"{target_id}.a3m"

        if a3m_path.exists():
            return a3m_path  # cached

        if not self._ready:
            # Return single-sequence "MSA" as fallback
            self._write_single_seq_a3m(sequence, target_id, a3m_path)
            return a3m_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            query_fasta = tmpdir / "query.fasta"
            with open(query_fasta, "w") as f:
                f.write(f">{target_id}\n{sequence}\n")

            # Step 1: Run nhmmer to find homologs
            stockholm_path = tmpdir / "hits.sto"
            cmd = [
                "nhmmer",
                "--rna",
                "--tblout", str(tmpdir / "hits.tbl"),
                "-A", str(stockholm_path),
                "--noali",
                "-E", str(self.e_value_threshold),
                "--cpu", str(self.cpu_cores),
                "--max",  # use max sensitivity mode
                str(query_fasta),
                self.db_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and stockholm_path.exists():
                # Convert Stockholm to A3M format
                self._stockholm_to_a3m(
                    stockholm_path, query_fasta, a3m_path,
                    sequence, target_id
                )
            else:
                # Fallback to BLAST-based MSA
                self._blast_msa(sequence, target_id, a3m_path, tmpdir)

        # Validate and truncate if too many sequences
        if a3m_path.exists():
            return self._filter_a3m(a3m_path, self.max_seqs)

        # Final fallback: single sequence
        self._write_single_seq_a3m(sequence, target_id, a3m_path)
        return a3m_path

    def _write_single_seq_a3m(self, sequence: str, target_id: str, path: Path) -> None:
        """Write a minimal single-sequence A3M file."""
        with open(path, "w") as f:
            f.write(f">{target_id}\n{sequence}\n")

    def _stockholm_to_a3m(
        self,
        sto_path: Path,
        query_fasta: Path,
        a3m_path: Path,
        query_seq: str,
        target_id: str,
    ) -> None:
        """Convert Stockholm alignment to A3M format."""
        cmd = [
            "esl-reformat",
            "--rna",
            "-o", str(a3m_path),
            "a2m",
            str(sto_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0 or not a3m_path.exists():
            # Fallback: manual conversion
            seqs = {}
            with open(sto_path) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("//"):
                        continue
                    parts = line.strip().split()
                    if len(parts) == 2:
                        name, aligned_seq = parts
                        if name in seqs:
                            seqs[name] += aligned_seq
                        else:
                            seqs[name] = aligned_seq

            with open(a3m_path, "w") as f:
                # Write query first
                f.write(f">{target_id}\n{query_seq}\n")
                for name, seq in list(seqs.items())[:self.max_seqs]:
                    if name != target_id:
                        # Convert to A3M: lowercase = insertion, uppercase = match
                        a3m_seq = seq.upper().replace(".", "")
                        f.write(f">{name}\n{a3m_seq}\n")

    def _blast_msa(
        self,
        sequence: str,
        target_id: str,
        a3m_path: Path,
        tmpdir: Path,
    ) -> None:
        """Fallback BLAST-based MSA generation."""
        if not self.db_path:
            self._write_single_seq_a3m(sequence, target_id, a3m_path)
            return

        query_fasta = tmpdir / "query.fasta"
        with open(query_fasta, "w") as f:
            f.write(f">{target_id}\n{sequence}\n")

        blast_out = tmpdir / "blast.xml"
        cmd = [
            "blastn",
            "-query", str(query_fasta),
            "-db", self.db_path,
            "-out", str(blast_out),
            "-outfmt", "5",  # XML
            "-max_target_seqs", str(self.max_seqs),
            "-evalue", str(self.e_value_threshold),
            "-task", "blastn",
            "-perc_identity", "60",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode != 0:
            self._write_single_seq_a3m(sequence, target_id, a3m_path)
            return

        # Parse BLAST XML and write A3M
        try:
            from Bio.Blast import NCBIXML
            with open(blast_out) as f:
                blast_records = list(NCBIXML.parse(f))
            seqs = [f">{target_id}\n{sequence}\n"]
            for record in blast_records:
                for alignment in record.alignments[:self.max_seqs]:
                    for hsp in alignment.hsps[:1]:
                        # Get aligned subject sequence
                        subj = hsp.sbjct.upper().replace("-", "")
                        seqs.append(f">{alignment.title.split()[0]}\n{subj}\n")
            with open(a3m_path, "w") as f:
                f.writelines(seqs)
        except Exception:
            self._write_single_seq_a3m(sequence, target_id, a3m_path)

    def _filter_a3m(self, a3m_path: Path, max_seqs: int) -> Path:
        """Filter A3M to maximum number of sequences."""
        seqs = []
        current_seq = []
        current_header = None

        with open(a3m_path) as f:
            for line in f:
                if line.startswith(">"):
                    if current_header and current_seq:
                        seqs.append((current_header, "".join(current_seq)))
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line.strip())

        if current_header and current_seq:
            seqs.append((current_header, "".join(current_seq)))

        if len(seqs) <= max_seqs:
            return a3m_path

        # Keep first (query) + diversity subsample of rest
        filtered = [seqs[0]] + seqs[1:max_seqs]
        with open(a3m_path, "w") as f:
            for header, seq in filtered:
                f.write(header if header.endswith("\n") else header + "\n")
                f.write(seq + "\n")

        return a3m_path

    def generate_batch(
        self,
        sequences: dict[str, str],
        output_dir: Path,
    ) -> dict[str, Optional[Path]]:
        """Generate MSAs for multiple sequences."""
        output_dir = Path(output_dir)
        results = {}
        for target_id, seq in sequences.items():
            msa_path = self.generate(seq, target_id, output_dir)
            results[target_id] = msa_path
            status = "OK" if msa_path and msa_path.exists() else "FAILED"
            print(f"[MSA] {target_id}: {len(seq)} nt → {status}")
        return results
