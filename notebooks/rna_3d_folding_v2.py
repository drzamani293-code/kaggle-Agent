#!/usr/bin/env python3
"""
Stanford RNA 3D Folding Part 2 - Competitive Kaggle Notebook
=============================================================
Approach: RNAPro (NVIDIA) / Protenix (ByteDance) + Template-Based Modeling
          + Multi-Seed Ensemble + Diverse Selection
Target Score: 0.55+ TM-score on public leaderboard

Kaggle Notebook Settings:
  - Accelerator: GPU T4 x2
  - Language: Python
  - Internet: OFF
  - Persistence: Files only
  - Session type: Batch

Required Kaggle Dataset Inputs (attach before running):
  1. rnapro-weights        -> /kaggle/input/rnapro-weights/
  2. protenix-base         -> /kaggle/input/protenix-base/
  3. rna-folding-templates -> /kaggle/input/rna-folding-templates/
  4. rna-deps              -> /kaggle/input/rna-deps/
  5. stanford-rna-3d-folding-2 -> /kaggle/input/stanford-rna-3d-folding-2/
"""

# ============================================================================
# CELL 1: ENVIRONMENT SETUP & CONFIGURATION
# ============================================================================

import os
import sys
import json
import time
import gc
import glob
import shutil
import tempfile
import warnings
import logging
import traceback

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Detect environment (Kaggle vs local) ----
IS_KAGGLE = os.path.exists("/kaggle/input")
BASE_INPUT = "/kaggle/input" if IS_KAGGLE else "./input"
BASE_OUTPUT = "/kaggle/working" if IS_KAGGLE else "./output"
os.makedirs(BASE_OUTPUT, exist_ok=True)

# ---- Paths to pre-uploaded datasets ----
RNAPRO_WEIGHTS_DIR = os.path.join(BASE_INPUT, "rnapro-weights")
PROTENIX_BASE_DIR = os.path.join(BASE_INPUT, "protenix-base")
TEMPLATES_DIR = os.path.join(BASE_INPUT, "rna-folding-templates")
DEPS_DIR = os.path.join(BASE_INPUT, "rna-deps")
COMPETITION_DATA_DIR = os.path.join(BASE_INPUT, "stanford-rna-3d-folding-2")

# Working directories for inference
WORK_DIR = os.path.join(BASE_OUTPUT, "work")
DUMP_DIR = os.path.join(BASE_OUTPUT, "predictions")
TEMP_DIR = os.path.join(BASE_OUTPUT, "temp")
for d in [WORK_DIR, DUMP_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)


# ---- Install offline dependencies ----
def install_offline_deps():
    """Install dependencies from pre-uploaded wheel files."""
    if os.path.exists(DEPS_DIR):
        # Install wheel files
        wheel_dir = DEPS_DIR
        wheels = glob.glob(os.path.join(wheel_dir, "*.whl"))
        if wheels:
            logger.info(f"Installing {len(wheels)} offline wheel packages...")
            os.system(
                f"pip install --no-deps --no-index --find-links={wheel_dir} "
                f"{' '.join(wheels)} 2>/dev/null"
            )

        # Install protenix from source if available
        protenix_src = os.path.join(DEPS_DIR, "Protenix")
        if not os.path.exists(protenix_src):
            protenix_src = os.path.join(DEPS_DIR, "protenix")
        if os.path.exists(protenix_src):
            logger.info(f"Installing protenix from: {protenix_src}")
            os.system(f"pip install -e {protenix_src} --no-deps 2>/dev/null")
            sys.path.insert(0, protenix_src)

        # Install RNAPro from source if available
        rnapro_src = os.path.join(DEPS_DIR, "RNAPro")
        if os.path.exists(rnapro_src):
            logger.info(f"Installing RNAPro from: {rnapro_src}")
            os.system(f"pip install -e {rnapro_src} --no-deps 2>/dev/null")
            sys.path.insert(0, rnapro_src)


install_offline_deps()

# ---- GPU Configuration ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

DEVICE_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_0)
DTYPE = torch.float16  # T4 does not support native bf16

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    logger.info(
        f"GPU {i}: {torch.cuda.get_device_name(i)} "
        f"({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)"
    )


# ============================================================================
# CELL 2: DATA STRUCTURES & UTILITIES
# ============================================================================

@dataclass
class RNATarget:
    """Represents a single RNA target to predict."""
    target_id: str
    sequence: str
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class Prediction:
    """A single 3D structure prediction for an RNA target."""
    target_id: str
    coords: np.ndarray  # Shape: (n_residues, 3) - C1' atom x,y,z
    confidence: float = 0.0
    seed: int = 0
    method: str = ""


@dataclass
class TargetPredictions:
    """All predictions for a single RNA target."""
    target: RNATarget
    predictions: List[Prediction] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_test_sequences() -> pd.DataFrame:
    """Load test sequences from competition data."""
    test_path = os.path.join(COMPETITION_DATA_DIR, "test_sequences.csv")
    if not os.path.exists(test_path):
        for candidate in [
            "/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv",
            "./test_sequences.csv",
            os.path.join(BASE_INPUT, "test_sequences.csv"),
        ]:
            if os.path.exists(candidate):
                test_path = candidate
                break

    logger.info(f"Loading test sequences from: {test_path}")
    df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(df)} test sequences")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(
        f"Sequence lengths: min={df['sequence'].str.len().min()}, "
        f"max={df['sequence'].str.len().max()}, "
        f"mean={df['sequence'].str.len().mean():.0f}"
    )
    return df


# ============================================================================
# CELL 3: TEMPLATE PROCESSING
# ============================================================================

class TemplateManager:
    """
    Manages precomputed template data for RNA structure prediction.

    Supports two formats:
    1. PyTorch .pt files: {target_id: coords_tensor} or per-target files
    2. CSV files in submission format with x_1,y_1,z_1,...,x_5,y_5,z_5
    """

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.pt_template_path: Optional[str] = None  # Path used by RNAPro
        self._load_templates()

    def _load_templates(self):
        """Load precomputed templates from various formats."""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        pt_files = sorted(Path(self.templates_dir).glob("**/*.pt"))
        csv_files = sorted(Path(self.templates_dir).glob("**/*.csv"))

        if pt_files:
            # Store path to first .pt file for RNAPro's template_data config
            self.pt_template_path = str(pt_files[0])
            self._load_pt_templates(pt_files)
        if csv_files:
            self._load_csv_templates(csv_files)

        logger.info(f"Loaded templates for {len(self.templates)} targets")
        if self.pt_template_path:
            logger.info(f"Primary template .pt file: {self.pt_template_path}")

    def _load_pt_templates(self, pt_files: List[Path]):
        """Load templates from PyTorch tensor files."""
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    for target_id, coords in data.items():
                        if isinstance(coords, torch.Tensor):
                            coords = coords.numpy()
                        if isinstance(coords, np.ndarray):
                            self.templates.setdefault(str(target_id), []).append(
                                coords.astype(np.float32)
                            )
                elif isinstance(data, torch.Tensor):
                    target_id = pt_file.stem
                    self.templates[target_id] = [data.numpy().astype(np.float32)]
            except Exception as e:
                logger.warning(f"Failed to load template {pt_file}: {e}")

    def _load_csv_templates(self, csv_files: List[Path]):
        """Load templates from CSV files (submission format)."""
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if "ID" not in df.columns:
                    continue

                df["target_id"] = df["ID"].apply(
                    lambda x: "_".join(str(x).split("_")[:-1])
                )

                for target_id, group in df.groupby("target_id"):
                    sort_col = "resid" if "resid" in group.columns else "ID"
                    group = group.sort_values(sort_col)
                    templates_for_target = []

                    for pred_idx in range(1, 6):
                        cols = [f"x_{pred_idx}", f"y_{pred_idx}", f"z_{pred_idx}"]
                        if all(c in group.columns for c in cols):
                            coords = group[cols].values.astype(np.float32)
                            if not np.all(np.isnan(coords)):
                                templates_for_target.append(coords)

                    if templates_for_target:
                        self.templates[str(target_id)] = templates_for_target
            except Exception as e:
                logger.warning(f"Failed to load template CSV {csv_file}: {e}")

    def get_templates(self, target_id: str, max_templates: int = 5) -> List[np.ndarray]:
        """Get templates for a specific target, up to max_templates."""
        return self.templates.get(target_id, [])[:max_templates]

    def has_templates(self, target_id: str) -> bool:
        """Check if templates exist for a target."""
        return target_id in self.templates and len(self.templates[target_id]) > 0


# ============================================================================
# CELL 4: RNAPRO INFERENCE ENGINE
# ============================================================================

class RNAProInferenceEngine:
    """
    Uses NVIDIA's RNAPro InferenceRunner for structure prediction.

    RNAPro pipeline:
    1. Create input JSON from sequence
    2. Build data features via get_inference_dataloader
    3. Run model.forward(input_feature_dict, ..., mode="inference")
    4. Extract C1' coordinates from CIF output
    """

    def __init__(self):
        self.runner = None
        self.configs = None
        self.available = False
        self._try_initialize()

    def _try_initialize(self) -> bool:
        """Try to initialize the RNAPro inference engine."""
        try:
            from rnapro.model.RNAPro import RNAPro
            from rnapro.data.infer_data_pipeline import get_inference_dataloader
            from rnapro.utils.inference import (
                update_inference_configs,
                process_sequence,
                extract_c1_coordinates,
                create_input_json,
            )
            from runner.inference import InferenceRunner

            # Find checkpoint
            ckpt_path = self._find_checkpoint(RNAPRO_WEIGHTS_DIR)
            if ckpt_path is None:
                logger.warning("No RNAPro checkpoint found")
                return False

            # Find template .pt file
            template_path = self._find_template_pt()

            logger.info(f"Initializing RNAPro with checkpoint: {ckpt_path}")

            # Build a minimal configs namespace
            import argparse
            self.configs = argparse.Namespace(
                model_name="rnapro_base_default",
                seeds=[101],
                dump_dir=DUMP_DIR,
                need_atom_confidence=False,
                sorted_by_ranking_score=True,
                load_checkpoint_dir=os.path.dirname(ckpt_path),
                num_workers=4,
                use_msa=True,
                template_data=template_path or "",
                template_idx=0,
                rna_msa_dir=os.path.join(WORK_DIR, "msa"),
                num_templates=4,
                sequences_csv="",
                use_deepspeed_evo_attention=False,
            )

            # Try to use InferenceRunner directly
            self.runner = InferenceRunner(self.configs)
            self.available = True
            logger.info("RNAPro InferenceRunner initialized successfully")
            return True

        except ImportError as e:
            logger.info(f"RNAPro not available (import error): {e}")
            return False
        except Exception as e:
            logger.warning(f"RNAPro initialization failed: {e}")
            traceback.print_exc()
            return False

    def _find_checkpoint(self, search_dir: str) -> Optional[str]:
        """Find model checkpoint file."""
        if not os.path.exists(search_dir):
            return None
        for pattern in ["**/*.pt", "**/*.ckpt", "**/*.pth"]:
            files = sorted(Path(search_dir).glob(pattern))
            # Prefer files with 'rnapro' or 'checkpoint' in name
            for f in files:
                name = f.name.lower()
                if "rnapro" in name or "checkpoint" in name:
                    return str(f)
            if files:
                return str(files[0])
        return None

    def _find_template_pt(self) -> Optional[str]:
        """Find precomputed template .pt file."""
        for search_dir in [TEMPLATES_DIR, WORK_DIR]:
            if os.path.exists(search_dir):
                for f in sorted(Path(search_dir).glob("**/*.pt")):
                    if "template" in f.name.lower() or f.name.startswith("test_"):
                        return str(f)
                # Return first .pt in templates dir
                pt_files = sorted(Path(search_dir).glob("**/*.pt"))
                if pt_files:
                    return str(pt_files[0])
        return None

    def predict(self, target_id: str, sequence: str, seed: int = 101) -> Optional[np.ndarray]:
        """
        Run RNAPro inference for a single target.

        Returns (L, 3) array of C1' coordinates, or None on failure.
        """
        if not self.available:
            return None

        try:
            from rnapro.utils.inference import (
                process_sequence,
                extract_c1_coordinates,
                create_input_json,
            )
            from rnapro.data.infer_data_pipeline import get_inference_dataloader

            # Step 1: Create input JSON
            input_json = create_input_json(sequence, target_id)
            json_dir = os.path.join(TEMP_DIR, f"{target_id}_seed{seed}")
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, f"{target_id}_input.json")
            with open(json_path, "w") as f:
                json.dump(input_json, f, indent=4)

            # Step 2: Update configs for this run
            self.configs.input_json_path = json_path
            self.configs.seeds = [seed]

            # Step 3: Get data loader
            dataloader = get_inference_dataloader(self.configs)

            # Step 4: Run inference
            for batch_data, atom_array, error_msg in dataloader:
                if error_msg:
                    logger.warning(f"Data error for {target_id}: {error_msg}")
                    continue

                prediction, _, _ = self.runner.predict(batch_data)

                # Step 5: Extract coordinates
                # The runner saves CIF files; extract C1' from them
                cif_pattern = os.path.join(
                    DUMP_DIR, f"**/{target_id}/**/predictions/*.cif"
                )
                cif_files = sorted(glob.glob(cif_pattern, recursive=True))

                if cif_files:
                    coords = extract_c1_coordinates(cif_files[-1])
                    if coords is not None:
                        # Pad/truncate to match sequence length
                        L = len(sequence)
                        if len(coords) < L:
                            padded = np.zeros((L, 3), dtype=np.float32)
                            padded[: len(coords)] = coords
                            coords = padded
                        elif len(coords) > L:
                            coords = coords[:L]
                        return coords.astype(np.float32)

                # Fallback: extract directly from prediction dict
                if isinstance(prediction, dict) and "coordinate" in prediction:
                    coord_tensor = prediction["coordinate"]
                    if isinstance(coord_tensor, torch.Tensor):
                        coord_tensor = coord_tensor.cpu().numpy()
                    # Shape: (1, N_atoms, 3) typically
                    if coord_tensor.ndim == 3:
                        coord_tensor = coord_tensor[0]
                    # Take first atom per residue (C1')
                    L = len(sequence)
                    if coord_tensor.shape[0] >= L:
                        return coord_tensor[:L].astype(np.float32)

            return None

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM for {target_id} seed={seed}")
            clear_gpu_memory()
            return None
        except Exception as e:
            logger.error(f"RNAPro prediction failed for {target_id}: {e}")
            traceback.print_exc()
            return None


# ============================================================================
# CELL 5: PROTENIX DIRECT INFERENCE ENGINE
# ============================================================================

class ProtenixInferenceEngine:
    """
    Direct Protenix (ByteDance) inference as fallback when RNAPro unavailable.

    Protenix API:
      model.forward(input_feature_dict, label_full_dict=None,
                    label_dict=None, mode="inference")
      -> (prediction_dict, log_dict, time_dict)

    prediction_dict contains: "coordinate", "plddt", "pae", "summary_confidence"
    """

    def __init__(self):
        self.model = None
        self.configs = None
        self.available = False
        self._try_initialize()

    def _try_initialize(self) -> bool:
        """Try to initialize Protenix model."""
        try:
            # Find checkpoint
            ckpt_path = None
            for search_dir in [PROTENIX_BASE_DIR, RNAPRO_WEIGHTS_DIR]:
                if os.path.exists(search_dir):
                    for pattern in ["**/*protenix*.pt", "**/*.pt"]:
                        files = sorted(Path(search_dir).glob(pattern))
                        if files:
                            ckpt_path = str(files[0])
                            break
                if ckpt_path:
                    break

            if ckpt_path is None:
                logger.info("No Protenix checkpoint found")
                return False

            # Try importing Protenix
            try:
                from protenix.model.protenix import Protenix
                from protenix.config import parse_configs
            except ImportError:
                # Try alternative import paths
                for path in [
                    os.path.join(DEPS_DIR, "Protenix"),
                    os.path.join(DEPS_DIR, "protenix"),
                ]:
                    if os.path.exists(path):
                        sys.path.insert(0, path)
                from protenix.model.protenix import Protenix
                from protenix.config import parse_configs

            logger.info(f"Loading Protenix from: {ckpt_path}")
            self.configs = parse_configs(
                model_name="protenix_base",
                load_checkpoint_path=ckpt_path,
            )
            self.model = Protenix(self.configs)
            self.model.eval()
            self.model = self.model.to(DEVICE_0)
            self.available = True
            logger.info("Protenix model loaded successfully")
            return True

        except ImportError as e:
            logger.info(f"Protenix not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Protenix initialization failed: {e}")
            traceback.print_exc()
            return False

    def predict(
        self,
        target_id: str,
        sequence: str,
        seed: int = 101,
        n_cycle: int = 10,
        n_step: int = 200,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Run Protenix inference. Returns (coords, confidence) or None.
        """
        if not self.available:
            return None

        try:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Build input features for Protenix
            # This uses Protenix's own data pipeline
            from protenix.data.infer_data_pipeline import get_inference_dataloader

            # Create input JSON in Protenix format
            input_json = [
                {
                    "sequences": [
                        {"rnaSequence": {"sequence": sequence, "count": 1}}
                    ],
                    "name": target_id,
                }
            ]

            json_dir = os.path.join(TEMP_DIR, f"protenix_{target_id}_s{seed}")
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, "input.json")
            with open(json_path, "w") as f:
                json.dump(input_json, f)

            self.configs.input_json_path = json_path

            dataloader = get_inference_dataloader(self.configs)

            for batch_data, atom_array, error_msg in dataloader:
                if error_msg:
                    logger.warning(f"Protenix data error: {error_msg}")
                    continue

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
                    prediction, log_dict, _ = self.model(
                        input_feature_dict=batch_data["input_feature_dict"],
                        label_full_dict=None,
                        label_dict=None,
                        mode="inference",
                    )

                # Extract C1' coordinates from prediction
                coords = self._extract_c1_coords(prediction, len(sequence))
                confidence = self._extract_confidence(prediction)

                if coords is not None:
                    return coords, confidence

            return None

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM for {target_id} seed={seed}")
            clear_gpu_memory()
            return None
        except Exception as e:
            logger.error(f"Protenix prediction failed: {e}")
            traceback.print_exc()
            return None

    def _extract_c1_coords(
        self, prediction: dict, n_residues: int
    ) -> Optional[np.ndarray]:
        """Extract C1' atom coordinates from Protenix output."""
        for key in [
            "coordinate",
            "final_atom_positions",
            "atom_positions",
            "coords",
            "pred_coords",
        ]:
            if key not in prediction:
                continue
            coords = prediction[key]
            if isinstance(coords, torch.Tensor):
                coords = coords.cpu().numpy()

            # Remove batch dim if present
            if coords.ndim == 4:
                coords = coords[0]

            if coords.ndim == 3:
                # Shape (N_atoms_total, 1, 3) or (N_residues, N_atoms_per_res, 3)
                # For RNA, C1' is typically the first heavy atom per residue
                # Protenix outputs all atoms; we need to select C1'
                # In standard RNA atom ordering, C1' index depends on the
                # representation. Try first atom per residue.
                if coords.shape[0] >= n_residues:
                    return coords[:n_residues, 0, :].astype(np.float32)
            elif coords.ndim == 2 and coords.shape[0] >= n_residues:
                return coords[:n_residues, :].astype(np.float32)

        return None

    def _extract_confidence(self, prediction: dict) -> float:
        """Extract confidence score from model output."""
        for key in [
            "summary_confidence",
            "plddt",
            "ranking_confidence",
            "iptm",
            "ptm",
        ]:
            if key in prediction:
                val = prediction[key]
                if isinstance(val, torch.Tensor):
                    val = (
                        val.cpu().item()
                        if val.numel() == 1
                        else val.cpu().float().mean().item()
                    )
                return float(val)
        return 0.5


# ============================================================================
# CELL 6: PHYSICS-BASED FALLBACK
# ============================================================================

class PhysicsPredictor:
    """
    Physics-based RNA 3D structure prediction fallback.

    Uses Nussinov secondary structure prediction + A-form helix geometry
    to generate coarse-grained 3D structures. Also uses templates
    directly when available.
    """

    @staticmethod
    def predict_from_template(
        sequence: str,
        template_coords: np.ndarray,
        seed: int = 42,
        noise_scale: float = 0.5,
    ) -> np.ndarray:
        """Use template coordinates with optional perturbation."""
        np.random.seed(seed)
        L = len(sequence)

        if len(template_coords) == L:
            coords = template_coords.copy()
        elif len(template_coords) > L:
            coords = template_coords[:L].copy()
        else:
            # Interpolate to match length
            coords = PhysicsPredictor._resize_coords(template_coords, L)

        # Add diversity noise
        noise = np.random.randn(L, 3).astype(np.float32) * noise_scale
        return (coords + noise).astype(np.float32)

    @staticmethod
    def predict_de_novo(sequence: str, seed: int = 42) -> np.ndarray:
        """Generate de novo structure using Nussinov + A-form geometry."""
        np.random.seed(seed)
        n = len(sequence)

        # Predict secondary structure
        pairs = PhysicsPredictor._nussinov_fold(sequence)
        paired_set = set()
        for i, j in pairs:
            paired_set.add(i)
            paired_set.add(j)

        # A-form RNA helix parameters (Angstroms/degrees)
        RISE = 2.81
        TWIST = 32.7
        RADIUS = 9.4

        coords = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            angle = np.radians(TWIST * i + seed * 37)
            x = RADIUS * np.cos(angle)
            y = RADIUS * np.sin(angle)
            z = RISE * i

            if i not in paired_set:
                x += np.random.randn() * 3.0
                y += np.random.randn() * 3.0
                z += np.random.randn() * 3.0

            coords[i] = [x, y, z]

        coords -= coords.mean(axis=0)
        coords = PhysicsPredictor._remove_clashes(coords)
        return coords

    @staticmethod
    def _nussinov_fold(
        sequence: str, min_loop: int = 4
    ) -> List[Tuple[int, int]]:
        """Simple Nussinov algorithm for RNA secondary structure."""
        n = len(sequence)
        if n > 2000:
            return []  # Too expensive for very long sequences

        can_pair = {
            ("A", "U"), ("U", "A"),
            ("G", "C"), ("C", "G"),
            ("G", "U"), ("U", "G"),
        }

        dp = np.zeros((n, n), dtype=np.int32)
        for span in range(min_loop + 1, n):
            for i in range(n - span):
                j = i + span
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
                if (sequence[i], sequence[j]) in can_pair:
                    inner = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
                    dp[i][j] = max(dp[i][j], inner + 1)
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k + 1][j])

        pairs = []
        PhysicsPredictor._traceback(dp, sequence, 0, n - 1, pairs, can_pair, min_loop)
        return pairs

    @staticmethod
    def _traceback(dp, seq, i, j, pairs, can_pair, min_loop):
        if i >= j:
            return
        if dp[i][j] == dp[i + 1][j]:
            PhysicsPredictor._traceback(dp, seq, i + 1, j, pairs, can_pair, min_loop)
        elif dp[i][j] == dp[i][j - 1]:
            PhysicsPredictor._traceback(dp, seq, i, j - 1, pairs, can_pair, min_loop)
        elif (seq[i], seq[j]) in can_pair and dp[i][j] == (
            (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0) + 1
        ):
            pairs.append((i, j))
            PhysicsPredictor._traceback(dp, seq, i + 1, j - 1, pairs, can_pair, min_loop)
        else:
            for k in range(i + 1, j):
                if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                    PhysicsPredictor._traceback(dp, seq, i, k, pairs, can_pair, min_loop)
                    PhysicsPredictor._traceback(dp, seq, k + 1, j, pairs, can_pair, min_loop)
                    break

    @staticmethod
    def _remove_clashes(
        coords: np.ndarray, min_dist: float = 3.0, n_iter: int = 100
    ) -> np.ndarray:
        n = len(coords)
        for _ in range(n_iter):
            moved = False
            for i in range(n):
                for j in range(i + 2, min(i + 10, n)):
                    diff = coords[j] - coords[i]
                    dist = np.linalg.norm(diff)
                    if 0.01 < dist < min_dist:
                        push = (min_dist - dist) / 2.0 * diff / dist
                        coords[i] -= push
                        coords[j] += push
                        moved = True
            if not moved:
                break
        return coords

    @staticmethod
    def _resize_coords(coords: np.ndarray, target_length: int) -> np.ndarray:
        """Resize coordinates via linear interpolation."""
        from scipy.interpolate import interp1d

        current_length = len(coords)
        if current_length == target_length:
            return coords

        old_idx = np.linspace(0, 1, current_length)
        new_idx = np.linspace(0, 1, target_length)

        result = np.zeros((target_length, 3), dtype=np.float32)
        for dim in range(3):
            f = interp1d(old_idx, coords[:, dim], kind="linear")
            result[:, dim] = f(new_idx)
        return result


# ============================================================================
# CELL 7: UNIFIED PREDICTION ORCHESTRATOR
# ============================================================================

class PredictionOrchestrator:
    """
    Orchestrates prediction across all model tiers with multi-seed ensemble.

    Tier 1: RNAPro (NVIDIA SOTA) - best accuracy
    Tier 2: Protenix direct (ByteDance) - strong baseline
    Tier 3: Template-based prediction - uses precomputed templates directly
    Tier 4: Physics-based fallback - de novo coarse-grained structure
    """

    # Seeds chosen for maximum diversity
    DEFAULT_SEEDS = [101, 42, 137, 256, 512, 1024, 2048, 4096, 7, 13]

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        self.rnapro_engine = None
        self.protenix_engine = None
        self.active_method = "none"

        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize model engines in priority order."""
        logger.info("Initializing RNAPro engine...")
        self.rnapro_engine = RNAProInferenceEngine()
        if self.rnapro_engine.available:
            self.active_method = "rnapro"
            logger.info("Primary engine: RNAPro (NVIDIA SOTA)")
            return

        logger.info("RNAPro unavailable. Trying Protenix...")
        self.protenix_engine = ProtenixInferenceEngine()
        if self.protenix_engine.available:
            self.active_method = "protenix"
            logger.info("Primary engine: Protenix (ByteDance)")
            return

        self.active_method = "template_physics"
        logger.warning("No DL models available. Using template/physics fallback.")

    def predict_target(
        self,
        target: RNATarget,
        n_seeds: int = 5,
        time_budget_seconds: float = 3600,
    ) -> List[Prediction]:
        """
        Generate multiple predictions for a single target.

        Returns list of Prediction objects with C1' coordinates.
        """
        predictions = []
        templates = self.template_manager.get_templates(target.target_id)
        has_templates = len(templates) > 0
        seeds = self.DEFAULT_SEEDS[:n_seeds]

        logger.info(
            f"Predicting {target.target_id} (len={target.length}, "
            f"templates={len(templates)}, method={self.active_method})"
        )

        start_time = time.time()

        # Tier 1: Try RNAPro
        if self.rnapro_engine and self.rnapro_engine.available:
            for seed in seeds:
                if time.time() - start_time > time_budget_seconds * 0.8:
                    logger.warning("Time budget nearly exceeded, stopping RNAPro seeds")
                    break

                coords = self.rnapro_engine.predict(
                    target.target_id, target.sequence, seed=seed
                )
                if coords is not None:
                    predictions.append(
                        Prediction(
                            target_id=target.target_id,
                            coords=coords,
                            confidence=0.8,
                            seed=seed,
                            method="rnapro",
                        )
                    )
                clear_gpu_memory()

        # Tier 2: Try Protenix (if RNAPro didn't produce enough)
        if len(predictions) < n_seeds and self.protenix_engine and self.protenix_engine.available:
            n_steps, n_cycle = self._adaptive_params(target.length)
            remaining_seeds = [s for s in seeds if s not in {p.seed for p in predictions}]

            for seed in remaining_seeds:
                if time.time() - start_time > time_budget_seconds * 0.8:
                    break

                result = self.protenix_engine.predict(
                    target.target_id,
                    target.sequence,
                    seed=seed,
                    n_cycle=n_cycle,
                    n_step=n_steps,
                )
                if result is not None:
                    coords, confidence = result
                    predictions.append(
                        Prediction(
                            target_id=target.target_id,
                            coords=coords,
                            confidence=confidence,
                            seed=seed,
                            method="protenix",
                        )
                    )
                clear_gpu_memory()

        # Tier 3: Template-based predictions
        if len(predictions) < n_seeds and has_templates:
            n_needed = n_seeds - len(predictions)
            used_seeds = {p.seed for p in predictions}

            for i, template in enumerate(templates[:n_needed]):
                seed = next((s for s in self.DEFAULT_SEEDS if s not in used_seeds), 42 + i)
                used_seeds.add(seed)
                noise_scale = 0.3 + 0.2 * i  # Increasing noise for diversity
                coords = PhysicsPredictor.predict_from_template(
                    target.sequence, template, seed=seed, noise_scale=noise_scale
                )
                predictions.append(
                    Prediction(
                        target_id=target.target_id,
                        coords=coords,
                        confidence=0.5 - 0.05 * i,
                        seed=seed,
                        method="template",
                    )
                )

        # Tier 4: Physics-based de novo
        if len(predictions) < n_seeds:
            n_needed = n_seeds - len(predictions)
            used_seeds = {p.seed for p in predictions}

            for i in range(n_needed):
                seed = next(
                    (s for s in range(42, 42 + 100) if s not in used_seeds), 42 + i
                )
                used_seeds.add(seed)
                coords = PhysicsPredictor.predict_de_novo(
                    target.sequence, seed=seed
                )
                predictions.append(
                    Prediction(
                        target_id=target.target_id,
                        coords=coords,
                        confidence=0.2,
                        seed=seed,
                        method="physics",
                    )
                )

        logger.info(
            f"Generated {len(predictions)} predictions for {target.target_id} "
            f"(methods: {set(p.method for p in predictions)})"
        )
        return predictions

    @staticmethod
    def _adaptive_params(seq_length: int) -> Tuple[int, int]:
        """Get (n_steps, n_cycle) adapted to sequence length for memory."""
        if seq_length > 4000:
            return 50, 4
        elif seq_length > 2000:
            return 100, 6
        elif seq_length > 1000:
            return 150, 8
        else:
            return 200, 10


# ============================================================================
# CELL 8: DIVERSE PREDICTION SELECTION
# ============================================================================

class PredictionSelector:
    """
    Select the best 5 predictions from N candidates.

    Strategy: Maximize chance that at least one prediction is close to truth
    by balancing confidence and structural diversity (competition uses best-of-5).
    """

    @staticmethod
    def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute RMSD between two coordinate sets after optimal superposition (Kabsch)."""
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)

        min_len = min(len(c1), len(c2))
        c1, c2 = c1[:min_len], c2[:min_len]

        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.eye(3)
        sign_matrix[2, 2] = np.sign(d)
        R = Vt.T @ sign_matrix @ U.T

        c2_rotated = (R @ c2.T).T
        return float(np.sqrt(np.mean(np.sum((c1 - c2_rotated) ** 2, axis=1))))

    @staticmethod
    def select_diverse_top5(
        predictions: List[Prediction],
        confidence_weight: float = 0.6,
        diversity_weight: float = 0.4,
    ) -> List[int]:
        """
        Select 5 diverse, high-confidence predictions using greedy algorithm.

        1. Pick the highest-confidence prediction first
        2. For each remaining slot, score candidates by:
           confidence * conf_weight + min_rmsd_to_selected * div_weight
        3. Pick the candidate with the highest combined score
        """
        if len(predictions) <= 5:
            return list(range(len(predictions)))

        n = len(predictions)

        # Normalize confidences
        confidences = np.array([p.confidence for p in predictions])
        conf_range = confidences.max() - confidences.min()
        norm_conf = (
            (confidences - confidences.min()) / conf_range
            if conf_range > 0
            else np.ones(n)
        )

        # Precompute pairwise RMSD
        rmsd_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                r = PredictionSelector.compute_rmsd(
                    predictions[i].coords, predictions[j].coords
                )
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = r

        max_rmsd = rmsd_matrix.max()
        norm_rmsd = rmsd_matrix / max_rmsd if max_rmsd > 0 else rmsd_matrix

        # Greedy selection
        selected = [int(np.argmax(norm_conf))]
        available = set(range(n)) - set(selected)

        for _ in range(4):
            if not available:
                break

            best_score, best_idx = -float("inf"), -1
            for idx in available:
                conf_score = norm_conf[idx] * confidence_weight
                div_score = min(norm_rmsd[idx, s] for s in selected) * diversity_weight
                total = conf_score + div_score
                if total > best_score:
                    best_score, best_idx = total, idx

            if best_idx >= 0:
                selected.append(best_idx)
                available.discard(best_idx)

        return selected


# ============================================================================
# CELL 9: SUBMISSION GENERATION
# ============================================================================

def generate_submission(
    targets: List[RNATarget],
    all_predictions: Dict[str, TargetPredictions],
    output_path: str,
) -> pd.DataFrame:
    """
    Generate submission.csv in the required format.

    Format per row: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
    where coordinates are C1' atom positions for each of 5 predictions.
    """
    rows = []

    for target in targets:
        tp = all_predictions.get(target.target_id)

        if tp is None or len(tp.predictions) == 0:
            logger.warning(f"No predictions for {target.target_id}, using zeros")
            for res_idx in range(target.length):
                row = {
                    "ID": f"{target.target_id}_{res_idx + 1}",
                    "resname": target.sequence[res_idx],
                    "resid": res_idx + 1,
                }
                for pred_idx in range(1, 6):
                    row[f"x_{pred_idx}"] = 0.0
                    row[f"y_{pred_idx}"] = 0.0
                    row[f"z_{pred_idx}"] = 0.0
                rows.append(row)
            continue

        # Get selected predictions (up to 5)
        selected = tp.selected_indices
        preds = [tp.predictions[i] for i in selected]

        # Pad to exactly 5 predictions
        while len(preds) < 5:
            if preds:
                last = preds[-1]
                np.random.seed(len(preds) + 100)
                noise = np.random.randn(*last.coords.shape).astype(np.float32) * 0.3
                preds.append(
                    Prediction(
                        target_id=target.target_id,
                        coords=last.coords + noise,
                        confidence=last.confidence * 0.9,
                        method=last.method + "_padded",
                    )
                )
            else:
                preds.append(
                    Prediction(
                        target_id=target.target_id,
                        coords=np.zeros((target.length, 3), dtype=np.float32),
                        confidence=0.0,
                        method="zero",
                    )
                )

        # Build rows for each residue
        for res_idx in range(target.length):
            row = {
                "ID": f"{target.target_id}_{res_idx + 1}",
                "resname": target.sequence[res_idx],
                "resid": res_idx + 1,
            }
            for pred_idx in range(5):
                coords = preds[pred_idx].coords
                if res_idx < len(coords):
                    row[f"x_{pred_idx + 1}"] = float(coords[res_idx, 0])
                    row[f"y_{pred_idx + 1}"] = float(coords[res_idx, 1])
                    row[f"z_{pred_idx + 1}"] = float(coords[res_idx, 2])
                else:
                    row[f"x_{pred_idx + 1}"] = 0.0
                    row[f"y_{pred_idx + 1}"] = 0.0
                    row[f"z_{pred_idx + 1}"] = 0.0
            rows.append(row)

    # Create DataFrame with correct column order
    columns = ["ID", "resname", "resid"]
    for i in range(1, 6):
        columns.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    df = pd.DataFrame(rows, columns=columns)

    # Fill any NaN with 0.0
    df = df.fillna(0.0)

    df.to_csv(output_path, index=False)

    logger.info(f"Submission saved to: {output_path}")
    logger.info(f"Shape: {df.shape}")
    n_targets = df["ID"].apply(lambda x: "_".join(x.split("_")[:-1])).nunique()
    logger.info(f"Targets in submission: {n_targets}")

    # Validate
    _validate_submission(df, targets)

    return df


def _validate_submission(df: pd.DataFrame, targets: List[RNATarget]):
    """Validate submission format and completeness."""
    issues = []

    # Check all targets present
    sub_targets = set(df["ID"].apply(lambda x: "_".join(x.split("_")[:-1])).unique())
    expected_targets = {t.target_id for t in targets}
    missing = expected_targets - sub_targets
    if missing:
        issues.append(f"Missing targets: {missing}")

    # Check for NaN/Inf
    coord_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]
    if df[coord_cols].isna().any().any():
        issues.append("NaN values found in coordinates")
    if np.isinf(df[coord_cols].values).any():
        issues.append("Inf values found in coordinates")

    # Check total rows
    expected_rows = sum(t.length for t in targets)
    if len(df) != expected_rows:
        issues.append(f"Row count mismatch: {len(df)} vs expected {expected_rows}")

    if issues:
        for issue in issues:
            logger.warning(f"Validation issue: {issue}")
    else:
        logger.info("Submission validation passed!")


# ============================================================================
# CELL 10: MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline: load data -> predict -> select -> submit."""
    start_time = time.time()

    # ---- Step 1: Load test data ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading test sequences")
    logger.info("=" * 60)

    test_df = load_test_sequences()

    targets = []
    for _, row in test_df.iterrows():
        target_id = row.get("target_id", row.get("ID", row.get("id", "")))
        sequence = row.get("sequence", row.get("seq", ""))
        targets.append(RNATarget(target_id=str(target_id), sequence=str(sequence)))

    # Sort by length (shortest first for early feedback)
    targets.sort(key=lambda t: t.length)
    logger.info(
        f"Processing {len(targets)} targets, "
        f"lengths: {[t.length for t in targets[:5]]}..."
        f"{[t.length for t in targets[-3:]]}"
    )

    # ---- Step 2: Load templates ----
    logger.info("=" * 60)
    logger.info("STEP 2: Loading templates")
    logger.info("=" * 60)

    template_manager = TemplateManager(TEMPLATES_DIR)
    n_with_templates = sum(
        1 for t in targets if template_manager.has_templates(t.target_id)
    )
    logger.info(f"Targets with templates: {n_with_templates}/{len(targets)}")

    # ---- Step 3: Initialize prediction engines ----
    logger.info("=" * 60)
    logger.info("STEP 3: Initializing prediction engines")
    logger.info("=" * 60)

    orchestrator = PredictionOrchestrator(template_manager)
    logger.info(f"Active method: {orchestrator.active_method}")

    # ---- Step 4: Generate predictions ----
    logger.info("=" * 60)
    logger.info("STEP 4: Generating predictions")
    logger.info("=" * 60)

    all_predictions: Dict[str, TargetPredictions] = {}
    selector = PredictionSelector()

    # Time management
    total_time_budget = 8.5 * 3600  # 8.5 hours
    time_per_target = total_time_budget / max(len(targets), 1)

    # Determine seeds per target
    n_seeds_base = max(5, min(10, int(time_per_target / 120)))
    logger.info(
        f"Seeds per target: {n_seeds_base}, "
        f"time budget per target: {time_per_target:.0f}s"
    )

    for target_idx, target in enumerate(targets):
        elapsed = time.time() - start_time
        remaining = total_time_budget - elapsed

        # Dynamic seed adjustment based on remaining time
        if remaining < 600:
            n_seeds = 5  # Minimum
        elif remaining < 1800:
            n_seeds = min(n_seeds_base, 5)
        else:
            n_seeds = n_seeds_base

        logger.info(
            f"\n--- Target {target_idx + 1}/{len(targets)}: "
            f"{target.target_id} (len={target.length}) ---"
        )

        # Generate predictions
        predictions = orchestrator.predict_target(
            target,
            n_seeds=n_seeds,
            time_budget_seconds=min(time_per_target, remaining * 0.9),
        )

        if not predictions:
            logger.error(f"No predictions for {target.target_id}!")
            continue

        # Select best diverse 5
        if len(predictions) > 5:
            selected_indices = selector.select_diverse_top5(predictions)
        else:
            selected_indices = list(range(len(predictions)))

        tp = TargetPredictions(
            target=target,
            predictions=predictions,
            selected_indices=selected_indices,
        )
        all_predictions[target.target_id] = tp

        logger.info(
            f"Selected {len(selected_indices)} predictions. "
            f"Methods: {[predictions[i].method for i in selected_indices]}, "
            f"Confidences: {[f'{predictions[i].confidence:.3f}' for i in selected_indices]}"
        )

        clear_gpu_memory()

    # ---- Step 5: Generate submission ----
    logger.info("=" * 60)
    logger.info("STEP 5: Generating submission")
    logger.info("=" * 60)

    submission_path = os.path.join(BASE_OUTPUT, "submission.csv")
    submission_df = generate_submission(targets, all_predictions, submission_path)

    # ---- Summary ----
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Targets processed: {len(all_predictions)}/{len(targets)}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Output: {submission_path}")

    # Method breakdown
    method_counts = {}
    for tp in all_predictions.values():
        for idx in tp.selected_indices:
            method = tp.predictions[idx].method
            method_counts[method] = method_counts.get(method, 0) + 1
    logger.info(f"Method breakdown: {method_counts}")
    logger.info("=" * 60)

    return submission_df


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    submission = main()
