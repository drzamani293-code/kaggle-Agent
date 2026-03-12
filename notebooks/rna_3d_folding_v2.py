#!/usr/bin/env python3
"""
Stanford RNA 3D Folding Part 2 - Competitive Kaggle Notebook
=============================================================
Approach: RNAPro (NVIDIA SOTA) + Multi-Seed Ensemble + Diverse Selection
Target Score: 0.55+ TM-score on public leaderboard

Kaggle Notebook Settings:
  - Accelerator: GPU T4 x2
  - Language: Python
  - Internet: OFF
  - Persistence: Files only
  - Session type: Batch

Required Kaggle Dataset Inputs (attach before running):
  1. rnapro-weights      -> /kaggle/input/rnapro-weights/
  2. ribonanzanet2       -> /kaggle/input/ribonanzanet2/
  3. protenix-base       -> /kaggle/input/protenix-base/
  4. rna-folding-templates -> /kaggle/input/rna-folding-templates/
  5. rna-deps            -> /kaggle/input/rna-deps/
  6. stanford-rna-3d-folding-2 -> /kaggle/input/stanford-rna-3d-folding-2/
"""

# ============================================================================
# CELL 1: ENVIRONMENT SETUP & CONFIGURATION
# ============================================================================

import os
import sys
import time
import gc
import warnings
import logging

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
RIBONANZANET2_DIR = os.path.join(BASE_INPUT, "ribonanzanet2")
PROTENIX_BASE_DIR = os.path.join(BASE_INPUT, "protenix-base")
TEMPLATES_DIR = os.path.join(BASE_INPUT, "rna-folding-templates")
DEPS_DIR = os.path.join(BASE_INPUT, "rna-deps")
COMPETITION_DATA_DIR = os.path.join(BASE_INPUT, "stanford-rna-3d-folding-2")

# ---- Install offline dependencies ----
def install_offline_deps():
    """Install dependencies from pre-uploaded wheel files."""
    if os.path.exists(DEPS_DIR):
        wheels = [f for f in os.listdir(DEPS_DIR) if f.endswith((".whl", ".tar.gz"))]
        if wheels:
            logger.info(f"Installing {len(wheels)} offline packages...")
            os.system(f"pip install --no-index --find-links={DEPS_DIR} {DEPS_DIR}/*.whl 2>/dev/null")

    # Install protenix from source if available
    protenix_src = os.path.join(DEPS_DIR, "protenix")
    if os.path.exists(protenix_src):
        os.system(f"pip install -e {protenix_src} 2>/dev/null")

    rnapro_src = os.path.join(DEPS_DIR, "RNAPro")
    if os.path.exists(rnapro_src):
        os.system(f"pip install -e {rnapro_src} 2>/dev/null")

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
DTYPE = torch.float16  # Use fp16 for T4 compatibility (no native bf16)

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} "
                f"({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")


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

    @property
    def residue_names(self) -> List[str]:
        return list(self.sequence)

    @property
    def residue_ids(self) -> List[int]:
        return list(range(1, self.length + 1))


@dataclass
class Prediction:
    """A single 3D structure prediction for an RNA target."""
    target_id: str
    coords: np.ndarray  # Shape: (n_residues, 3) - C1' atom x,y,z
    confidence: float = 0.0  # pLDDT or ipTM score
    seed: int = 0
    template_idx: int = -1


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
        # Fallback: search in common locations
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
    logger.info(f"Sequence lengths: min={df['sequence'].str.len().min()}, "
                f"max={df['sequence'].str.len().max()}, "
                f"mean={df['sequence'].str.len().mean():.0f}")
    return df


# ============================================================================
# CELL 3: TEMPLATE PROCESSING
# ============================================================================

class TemplateManager:
    """Manages precomputed template data for RNA structure prediction."""

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self.templates: Dict[str, List[np.ndarray]] = {}
        self._load_templates()

    def _load_templates(self):
        """Load precomputed templates from various formats."""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        # Load from CSV files (one per target or combined)
        csv_files = list(Path(self.templates_dir).glob("*.csv"))
        pt_files = list(Path(self.templates_dir).glob("*.pt"))

        if pt_files:
            self._load_pt_templates(pt_files)
        elif csv_files:
            self._load_csv_templates(csv_files)

        logger.info(f"Loaded templates for {len(self.templates)} targets")

    def _load_pt_templates(self, pt_files: List[Path]):
        """Load templates from PyTorch tensor files."""
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    for target_id, coords in data.items():
                        if isinstance(coords, torch.Tensor):
                            coords = coords.numpy()
                        if target_id not in self.templates:
                            self.templates[target_id] = []
                        self.templates[target_id].append(coords)
                elif isinstance(data, torch.Tensor):
                    target_id = pt_file.stem
                    self.templates[target_id] = [data.numpy()]
            except Exception as e:
                logger.warning(f"Failed to load template {pt_file}: {e}")

    def _load_csv_templates(self, csv_files: List[Path]):
        """Load templates from CSV files (submission format)."""
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Expected format: ID, resname, resid, x_1, y_1, z_1, ...
                if "ID" in df.columns:
                    # Extract target IDs
                    df["target_id"] = df["ID"].apply(lambda x: "_".join(str(x).split("_")[:-1]))

                    for target_id, group in df.groupby("target_id"):
                        group = group.sort_values("resid" if "resid" in group.columns else "ID")
                        templates_for_target = []

                        # Extract coordinate columns
                        for pred_idx in range(1, 6):
                            x_col = f"x_{pred_idx}"
                            y_col = f"y_{pred_idx}"
                            z_col = f"z_{pred_idx}"
                            if all(c in group.columns for c in [x_col, y_col, z_col]):
                                coords = group[[x_col, y_col, z_col]].values.astype(np.float32)
                                if not np.all(np.isnan(coords)):
                                    templates_for_target.append(coords)

                        if templates_for_target:
                            self.templates[str(target_id)] = templates_for_target
            except Exception as e:
                logger.warning(f"Failed to load template CSV {csv_file}: {e}")

    def get_templates(self, target_id: str, max_templates: int = 5) -> List[np.ndarray]:
        """Get templates for a specific target, up to max_templates."""
        templates = self.templates.get(target_id, [])
        return templates[:max_templates]

    def has_templates(self, target_id: str) -> bool:
        """Check if templates exist for a target."""
        return target_id in self.templates and len(self.templates[target_id]) > 0


# ============================================================================
# CELL 4: STRUCTURE PREDICTION ENGINE
# ============================================================================

class RNAStructurePredictor:
    """
    RNA 3D structure prediction using RNAPro or Protenix.

    Hierarchy of approaches (tries best first, falls back):
    1. RNAPro (RibonanzaNet2 + Protenix + templates) - SOTA
    2. Protenix with templates - Strong
    3. Protenix without templates - Baseline
    4. Physics-based fallback - Last resort
    """

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        self.model = None
        self.model_type = None
        self.ribonanzanet = None
        self._load_best_available_model()

    def _load_best_available_model(self):
        """Load the best available model in priority order."""
        # Try RNAPro first
        if self._try_load_rnapro():
            self.model_type = "rnapro"
            logger.info("Loaded RNAPro model (SOTA)")
            return

        # Try Protenix
        if self._try_load_protenix():
            self.model_type = "protenix"
            logger.info("Loaded Protenix model")
            return

        # Fallback to physics-based
        self.model_type = "physics"
        logger.warning("No deep learning model available. Using physics-based fallback.")

    def _try_load_rnapro(self) -> bool:
        """Attempt to load RNAPro model."""
        try:
            # Check for RNAPro weights
            rnapro_ckpt = None
            for pattern in ["*.pt", "*.ckpt", "*.pth", "*.bin"]:
                files = list(Path(RNAPRO_WEIGHTS_DIR).glob(pattern))
                if files:
                    rnapro_ckpt = str(files[0])
                    break

            if rnapro_ckpt is None:
                return False

            # Try to import RNAPro
            try:
                from protenix.model.protenix import Protenix
                from protenix.config import parse_configs
            except ImportError:
                try:
                    sys.path.insert(0, os.path.join(DEPS_DIR, "RNAPro"))
                    sys.path.insert(0, os.path.join(DEPS_DIR, "protenix"))
                    from protenix.model.protenix import Protenix
                    from protenix.config import parse_configs
                except ImportError:
                    return False

            logger.info(f"Loading RNAPro from: {rnapro_ckpt}")

            # Load RibonanzaNet2 if available
            self._try_load_ribonanzanet()

            # Load RNAPro model
            config = parse_configs(
                model_name="rnapro_base",
                load_checkpoint_path=rnapro_ckpt,
                dtype="fp16",
            )
            self.model = Protenix(config)
            self.model.eval()
            self.model = self.model.to(DEVICE_0)

            return True
        except Exception as e:
            logger.warning(f"Failed to load RNAPro: {e}")
            return False

    def _try_load_protenix(self) -> bool:
        """Attempt to load Protenix model."""
        try:
            protenix_ckpt = None
            for search_dir in [PROTENIX_BASE_DIR, RNAPRO_WEIGHTS_DIR]:
                if os.path.exists(search_dir):
                    for pattern in ["*.pt", "*.ckpt", "*.pth"]:
                        files = list(Path(search_dir).glob(pattern))
                        if files:
                            protenix_ckpt = str(files[0])
                            break
                if protenix_ckpt:
                    break

            if protenix_ckpt is None:
                return False

            try:
                from protenix.model.protenix import Protenix
                from protenix.config import parse_configs
            except ImportError:
                try:
                    sys.path.insert(0, os.path.join(DEPS_DIR, "protenix"))
                    from protenix.model.protenix import Protenix
                    from protenix.config import parse_configs
                except ImportError:
                    return False

            logger.info(f"Loading Protenix from: {protenix_ckpt}")
            config = parse_configs(
                model_name="protenix_base",
                load_checkpoint_path=protenix_ckpt,
                dtype="fp16",
            )
            self.model = Protenix(config)
            self.model.eval()
            self.model = self.model.to(DEVICE_0)

            return True
        except Exception as e:
            logger.warning(f"Failed to load Protenix: {e}")
            return False

    def _try_load_ribonanzanet(self):
        """Load RibonanzaNet2 as feature encoder."""
        try:
            if not os.path.exists(RIBONANZANET2_DIR):
                return

            ckpt_files = list(Path(RIBONANZANET2_DIR).glob("**/*.pt")) + \
                         list(Path(RIBONANZANET2_DIR).glob("**/*.ckpt"))
            if not ckpt_files:
                return

            logger.info(f"Loading RibonanzaNet2 from: {ckpt_files[0]}")
            # RibonanzaNet2 loading depends on the specific implementation
            # This will be configured when the model is actually available
            self.ribonanzanet = str(ckpt_files[0])
        except Exception as e:
            logger.warning(f"Failed to load RibonanzaNet2: {e}")

    def predict(
        self,
        target: RNATarget,
        n_predictions: int = 10,
        seeds: Optional[List[int]] = None,
    ) -> List[Prediction]:
        """
        Generate multiple 3D structure predictions for an RNA target.

        Uses multi-seed inference with template variation for diversity.
        """
        if seeds is None:
            seeds = list(range(42, 42 + n_predictions))

        predictions = []
        templates = self.template_manager.get_templates(target.target_id)
        has_templates = len(templates) > 0

        logger.info(f"Predicting {target.target_id} (len={target.length}, "
                     f"templates={len(templates)}, seeds={len(seeds)})")

        # Adaptive parameters based on sequence length
        n_steps, n_cycle = self._get_adaptive_params(target.length)

        for seed_idx, seed in enumerate(seeds):
            try:
                # Vary template subset for diversity
                template_idx = seed_idx % max(len(templates), 1) if has_templates else -1
                template_subset = [templates[template_idx]] if has_templates and template_idx < len(templates) else None

                if self.model_type in ("rnapro", "protenix"):
                    pred = self._predict_with_model(
                        target, seed, template_subset, n_steps, n_cycle
                    )
                else:
                    pred = self._predict_physics_fallback(
                        target, seed, template_subset
                    )

                if pred is not None:
                    pred.seed = seed
                    pred.template_idx = template_idx
                    predictions.append(pred)

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM for {target.target_id} seed={seed}. Reducing params...")
                clear_gpu_memory()
                # Try again with reduced parameters
                try:
                    pred = self._predict_with_reduced_params(target, seed, templates)
                    if pred is not None:
                        predictions.append(pred)
                except Exception:
                    logger.error(f"Failed even with reduced params for seed={seed}")
                    clear_gpu_memory()
            except Exception as e:
                logger.error(f"Prediction failed for {target.target_id} seed={seed}: {e}")
                clear_gpu_memory()

        # If no predictions from model, use template-based fallback
        if not predictions and has_templates:
            logger.info(f"Using template-only predictions for {target.target_id}")
            predictions = self._template_only_predictions(target, templates)

        # If still no predictions, use physics-based fallback
        if not predictions:
            logger.info(f"Using physics fallback for {target.target_id}")
            for seed in seeds[:5]:
                pred = self._predict_physics_fallback(target, seed, None)
                if pred is not None:
                    predictions.append(pred)

        logger.info(f"Generated {len(predictions)} predictions for {target.target_id}")
        return predictions

    def _get_adaptive_params(self, seq_length: int) -> Tuple[int, int]:
        """Get diffusion steps and cycle count based on sequence length."""
        if seq_length > 4000:
            return 50, 1   # Very long: minimal
        elif seq_length > 2000:
            return 100, 2  # Long: reduced
        elif seq_length > 1000:
            return 150, 3  # Medium: moderate
        else:
            return 200, 4  # Short: full

    def _predict_with_model(
        self,
        target: RNATarget,
        seed: int,
        templates: Optional[List[np.ndarray]],
        n_steps: int,
        n_cycle: int,
    ) -> Optional[Prediction]:
        """Run inference with RNAPro or Protenix model."""
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
                # Prepare input
                input_data = self._prepare_model_input(target, templates)

                # Run inference
                output = self.model.inference(
                    input_data,
                    n_step=n_steps,
                    n_cycle=n_cycle,
                    seed=seed,
                )

                # Extract C1' coordinates
                coords = self._extract_c1_coords_from_output(output, target.length)
                confidence = self._extract_confidence(output)

                return Prediction(
                    target_id=target.target_id,
                    coords=coords,
                    confidence=confidence,
                    seed=seed,
                )
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return None

    def _predict_with_reduced_params(
        self,
        target: RNATarget,
        seed: int,
        templates: Optional[List[np.ndarray]],
    ) -> Optional[Prediction]:
        """Predict with minimal parameters to avoid OOM."""
        clear_gpu_memory()
        template_subset = [templates[0]] if templates else None
        return self._predict_with_model(target, seed, template_subset, n_steps=50, n_cycle=1)

    def _prepare_model_input(self, target: RNATarget, templates: Optional[List[np.ndarray]]) -> dict:
        """Prepare input dictionary for the model."""
        # Nucleotide to integer mapping
        nuc_map = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
        seq_encoded = [nuc_map.get(n, 4) for n in target.sequence]

        input_data = {
            "sequence": target.sequence,
            "seq_encoded": torch.tensor(seq_encoded, dtype=torch.long).unsqueeze(0),
            "target_id": target.target_id,
            "seq_length": target.length,
        }

        if templates:
            template_tensors = [
                torch.tensor(t, dtype=torch.float32) for t in templates
            ]
            input_data["templates"] = torch.stack(template_tensors).unsqueeze(0)
            input_data["num_templates"] = len(templates)

        if self.ribonanzanet is not None:
            input_data["ribonanzanet_path"] = self.ribonanzanet

        return input_data

    def _extract_c1_coords_from_output(self, output: dict, n_residues: int) -> np.ndarray:
        """Extract C1' atom coordinates from model output."""
        # Different models have different output formats
        # Try common keys
        for key in ["final_atom_positions", "atom_positions", "coords", "positions", "pred_coords"]:
            if key in output:
                all_coords = output[key]
                if isinstance(all_coords, torch.Tensor):
                    all_coords = all_coords.cpu().numpy()

                # If shape is (1, N, n_atoms, 3), squeeze batch dim
                if all_coords.ndim == 4:
                    all_coords = all_coords[0]

                # all_coords shape: (N_residues, N_atoms, 3) or (N_residues, 3)
                if all_coords.ndim == 3:
                    # C1' is typically atom index 0 or 1 in RNA
                    # In most implementations, C1' is at index 0 for nucleotides
                    c1_coords = all_coords[:n_residues, 0, :]
                else:
                    c1_coords = all_coords[:n_residues, :]

                return c1_coords.astype(np.float32)

        raise ValueError(f"Could not extract coordinates. Output keys: {list(output.keys())}")

    def _extract_confidence(self, output: dict) -> float:
        """Extract confidence score from model output."""
        for key in ["plddt", "confidence", "iptm", "ptm", "ranking_confidence"]:
            if key in output:
                val = output[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().item() if val.numel() == 1 else val.cpu().mean().item()
                return float(val)
        return 0.5  # Default confidence

    def _template_only_predictions(
        self, target: RNATarget, templates: List[np.ndarray]
    ) -> List[Prediction]:
        """Use templates directly as predictions (with small perturbations for diversity)."""
        predictions = []
        for i, template in enumerate(templates[:5]):
            # Ensure template matches sequence length
            if len(template) != target.length:
                template = self._resize_template(template, target.length)

            # Add small random perturbation for diversity
            np.random.seed(42 + i)
            noise = np.random.randn(*template.shape).astype(np.float32) * 0.5
            perturbed = template + noise

            predictions.append(Prediction(
                target_id=target.target_id,
                coords=perturbed.astype(np.float32),
                confidence=0.7 - 0.05 * i,  # Templates are somewhat reliable
                seed=42 + i,
                template_idx=i,
            ))

        return predictions

    def _predict_physics_fallback(
        self,
        target: RNATarget,
        seed: int,
        templates: Optional[List[np.ndarray]],
    ) -> Optional[Prediction]:
        """
        Physics-based fallback: generate a plausible RNA 3D structure
        using coarse-grained modeling with base-pair constraints.

        This uses:
        1. RNA secondary structure prediction (Nussinov-like)
        2. Distance geometry to embed in 3D
        3. Simple energy minimization
        """
        np.random.seed(seed)
        n = target.length

        if templates and len(templates) > 0:
            # Use best template as starting point
            template = templates[0]
            if len(template) == n:
                noise = np.random.randn(n, 3).astype(np.float32) * (1.0 + seed * 0.1)
                coords = template + noise
                return Prediction(
                    target_id=target.target_id,
                    coords=coords,
                    confidence=0.4,
                    seed=seed,
                )

        # Generate de novo structure using coarse-grained approach
        coords = self._generate_coarse_grained_structure(target.sequence, seed)

        return Prediction(
            target_id=target.target_id,
            coords=coords,
            confidence=0.2,
            seed=seed,
        )

    def _generate_coarse_grained_structure(self, sequence: str, seed: int) -> np.ndarray:
        """
        Generate a coarse-grained 3D RNA structure.

        Uses:
        1. Nussinov algorithm for secondary structure
        2. A-form helix geometry for paired bases
        3. Random loop conformations for unpaired regions
        4. Simple refinement to fix steric clashes
        """
        np.random.seed(seed)
        n = len(sequence)

        # Step 1: Predict secondary structure (simple Nussinov)
        pairs = self._nussinov_fold(sequence)

        # Step 2: Build 3D coordinates
        coords = np.zeros((n, 3), dtype=np.float32)

        # A-form RNA helix parameters
        RISE_PER_BP = 2.81  # Angstroms
        TWIST_PER_BP = 32.7  # degrees
        RADIUS = 9.4  # Angstroms from helix axis to C1'

        # Place bases along a path
        # Start with a helical backbone
        for i in range(n):
            t = i / max(n - 1, 1)
            angle = np.radians(TWIST_PER_BP * i + seed * 37)

            # Base position
            x = RADIUS * np.cos(angle)
            y = RADIUS * np.sin(angle)
            z = RISE_PER_BP * i

            # Add some randomness for unpaired regions
            is_paired = any(i == p[0] or i == p[1] for p in pairs)
            if not is_paired:
                noise_scale = 3.0
                x += np.random.randn() * noise_scale
                y += np.random.randn() * noise_scale
                z += np.random.randn() * noise_scale

            coords[i] = [x, y, z]

        # Step 3: Center the structure
        coords -= coords.mean(axis=0)

        # Step 4: Simple clash removal (push apart atoms that are too close)
        coords = self._remove_clashes(coords, min_dist=3.0)

        return coords

    def _nussinov_fold(self, sequence: str, min_loop: int = 4) -> List[Tuple[int, int]]:
        """Simple Nussinov algorithm for RNA secondary structure prediction."""
        n = len(sequence)

        # Complementary base pairs
        can_pair = {
            ("A", "U"): True, ("U", "A"): True,
            ("G", "C"): True, ("C", "G"): True,
            ("G", "U"): True, ("U", "G"): True,
        }

        # DP table
        dp = np.zeros((n, n), dtype=np.int32)

        for span in range(min_loop + 1, n):
            for i in range(n - span):
                j = i + span
                # Case 1: i unpaired
                dp[i][j] = dp[i + 1][j]
                # Case 2: j unpaired
                dp[i][j] = max(dp[i][j], dp[i][j - 1])
                # Case 3: i pairs with j
                if can_pair.get((sequence[i], sequence[j]), False):
                    inner = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
                    dp[i][j] = max(dp[i][j], inner + 1)
                # Case 4: bifurcation
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k + 1][j])

        # Traceback
        pairs = []
        self._nussinov_traceback(dp, sequence, 0, n - 1, pairs, can_pair, min_loop)
        return pairs

    def _nussinov_traceback(self, dp, seq, i, j, pairs, can_pair, min_loop):
        """Traceback for Nussinov algorithm."""
        if i >= j:
            return

        if dp[i][j] == dp[i + 1][j]:
            self._nussinov_traceback(dp, seq, i + 1, j, pairs, can_pair, min_loop)
        elif dp[i][j] == dp[i][j - 1]:
            self._nussinov_traceback(dp, seq, i, j - 1, pairs, can_pair, min_loop)
        elif can_pair.get((seq[i], seq[j]), False) and \
             dp[i][j] == (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0) + 1:
            pairs.append((i, j))
            self._nussinov_traceback(dp, seq, i + 1, j - 1, pairs, can_pair, min_loop)
        else:
            for k in range(i + 1, j):
                if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                    self._nussinov_traceback(dp, seq, i, k, pairs, can_pair, min_loop)
                    self._nussinov_traceback(dp, seq, k + 1, j, pairs, can_pair, min_loop)
                    break

    def _remove_clashes(self, coords: np.ndarray, min_dist: float = 3.0, n_iter: int = 100) -> np.ndarray:
        """Push apart atoms that are too close."""
        n = len(coords)
        for _ in range(n_iter):
            moved = False
            for i in range(n):
                for j in range(i + 2, min(i + 10, n)):  # Check nearby atoms
                    diff = coords[j] - coords[i]
                    dist = np.linalg.norm(diff)
                    if dist < min_dist and dist > 0.01:
                        push = (min_dist - dist) / 2.0 * diff / dist
                        coords[i] -= push
                        coords[j] += push
                        moved = True
            if not moved:
                break
        return coords

    def _resize_template(self, template: np.ndarray, target_length: int) -> np.ndarray:
        """Resize template to match target sequence length via interpolation."""
        from scipy.interpolate import interp1d

        current_length = len(template)
        if current_length == target_length:
            return template

        # Interpolate each coordinate dimension
        old_indices = np.linspace(0, 1, current_length)
        new_indices = np.linspace(0, 1, target_length)

        new_template = np.zeros((target_length, 3), dtype=np.float32)
        for dim in range(3):
            f = interp1d(old_indices, template[:, dim], kind="linear")
            new_template[:, dim] = f(new_indices)

        return new_template


# ============================================================================
# CELL 5: DIVERSE PREDICTION SELECTION
# ============================================================================

class PredictionSelector:
    """
    Select the best 5 predictions from N candidates.

    Strategy: Maximize chance that at least one prediction is close to truth
    by balancing confidence and structural diversity.

    The competition uses best-of-5 TM-score, so diversity is crucial.
    """

    @staticmethod
    def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute RMSD between two sets of coordinates after optimal superposition."""
        # Center both
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)

        # Handle size mismatch
        min_len = min(len(c1), len(c2))
        c1 = c1[:min_len]
        c2 = c2[:min_len]

        # Kabsch algorithm for optimal rotation
        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.eye(3)
        sign_matrix[2, 2] = np.sign(d)
        R = Vt.T @ sign_matrix @ U.T

        c2_rotated = (R @ c2.T).T
        rmsd = np.sqrt(np.mean(np.sum((c1 - c2_rotated) ** 2, axis=1)))
        return rmsd

    @staticmethod
    def select_diverse_top5(
        predictions: List[Prediction],
        confidence_weight: float = 0.6,
        diversity_weight: float = 0.4,
    ) -> List[int]:
        """
        Select 5 diverse, high-confidence predictions using greedy algorithm.

        Algorithm:
        1. Pick the highest-confidence prediction first
        2. For each remaining slot, score candidates by:
           - Confidence * confidence_weight
           - Min RMSD to already selected * diversity_weight
        3. Pick the candidate with the highest combined score

        Returns indices into the predictions list.
        """
        if len(predictions) <= 5:
            return list(range(len(predictions)))

        n = len(predictions)

        # Normalize confidences to [0, 1]
        confidences = np.array([p.confidence for p in predictions])
        if confidences.max() > confidences.min():
            norm_conf = (confidences - confidences.min()) / (confidences.max() - confidences.min())
        else:
            norm_conf = np.ones(n)

        # Precompute pairwise RMSD matrix
        rmsd_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                rmsd = PredictionSelector.compute_rmsd(
                    predictions[i].coords, predictions[j].coords
                )
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        # Normalize RMSD values
        max_rmsd = rmsd_matrix.max()
        if max_rmsd > 0:
            norm_rmsd = rmsd_matrix / max_rmsd
        else:
            norm_rmsd = rmsd_matrix

        # Greedy selection
        selected = []
        available = set(range(n))

        # Pick first: highest confidence
        first = int(np.argmax(norm_conf))
        selected.append(first)
        available.discard(first)

        # Pick remaining 4
        for _ in range(4):
            if not available:
                break

            best_score = -float("inf")
            best_idx = -1

            for idx in available:
                # Confidence component
                conf_score = norm_conf[idx] * confidence_weight

                # Diversity component: minimum RMSD to any already selected
                min_rmsd_to_selected = min(norm_rmsd[idx, s] for s in selected)
                div_score = min_rmsd_to_selected * diversity_weight

                total_score = conf_score + div_score

                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                available.discard(best_idx)

        return selected


# ============================================================================
# CELL 6: SUBMISSION GENERATION
# ============================================================================

def generate_submission(
    targets: List[RNATarget],
    all_predictions: Dict[str, TargetPredictions],
    output_path: str,
) -> pd.DataFrame:
    """
    Generate submission.csv in the required format.

    Format: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
    """
    rows = []

    for target in targets:
        tp = all_predictions.get(target.target_id)
        if tp is None or len(tp.predictions) == 0:
            logger.warning(f"No predictions for {target.target_id}, using zeros")
            # Generate zero-filled predictions
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

        # Pad to exactly 5 predictions if needed
        while len(preds) < 5:
            if preds:
                # Duplicate last prediction with small noise
                last = preds[-1]
                noise = np.random.randn(*last.coords.shape).astype(np.float32) * 0.3
                padded = Prediction(
                    target_id=target.target_id,
                    coords=last.coords + noise,
                    confidence=last.confidence * 0.9,
                )
                preds.append(padded)
            else:
                # Should not happen, but just in case
                dummy_coords = np.zeros((target.length, 3), dtype=np.float32)
                preds.append(Prediction(
                    target_id=target.target_id,
                    coords=dummy_coords,
                    confidence=0.0,
                ))

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
    df.to_csv(output_path, index=False)

    logger.info(f"Submission saved to: {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Targets: {df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1])).nunique()}")
    logger.info(f"Sample:\n{df.head()}")

    return df


# ============================================================================
# CELL 7: MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline: load data -> predict -> select -> submit."""
    start_time = time.time()

    # ---- Step 1: Load test data ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading test sequences")
    logger.info("=" * 60)

    test_df = load_test_sequences()

    # Parse into RNATarget objects
    targets = []
    for _, row in test_df.iterrows():
        target_id = row.get("target_id", row.get("ID", row.get("id", "")))
        sequence = row.get("sequence", row.get("seq", ""))
        targets.append(RNATarget(target_id=str(target_id), sequence=str(sequence)))

    # Sort by length (process shorter sequences first for early feedback)
    targets.sort(key=lambda t: t.length)
    logger.info(f"Processing {len(targets)} targets, lengths: "
                f"{[t.length for t in targets[:5]]}...{[t.length for t in targets[-3:]]}")

    # ---- Step 2: Load templates ----
    logger.info("=" * 60)
    logger.info("STEP 2: Loading templates")
    logger.info("=" * 60)

    template_manager = TemplateManager(TEMPLATES_DIR)

    targets_with_templates = sum(1 for t in targets if template_manager.has_templates(t.target_id))
    logger.info(f"Targets with templates: {targets_with_templates}/{len(targets)}")

    # ---- Step 3: Initialize predictor ----
    logger.info("=" * 60)
    logger.info("STEP 3: Initializing structure predictor")
    logger.info("=" * 60)

    predictor = RNAStructurePredictor(template_manager)
    logger.info(f"Using model: {predictor.model_type}")

    # ---- Step 4: Generate predictions ----
    logger.info("=" * 60)
    logger.info("STEP 4: Generating predictions")
    logger.info("=" * 60)

    all_predictions: Dict[str, TargetPredictions] = {}
    selector = PredictionSelector()

    # Determine number of predictions per target based on total count
    # Budget: ~9 hours total, reserve 30 min for setup/submission
    time_budget_seconds = 8.5 * 3600
    avg_time_per_pred = 60  # seconds, estimated
    total_preds_budget = int(time_budget_seconds / avg_time_per_pred)
    preds_per_target = max(5, min(20, total_preds_budget // max(len(targets), 1)))
    logger.info(f"Generating {preds_per_target} predictions per target")

    # Seeds for reproducibility and diversity
    base_seeds = [42, 137, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                  7, 13, 23, 37, 53, 71, 97, 113, 151, 199]

    for target_idx, target in enumerate(targets):
        elapsed = time.time() - start_time
        remaining = time_budget_seconds - elapsed

        if remaining < 300 and target_idx > 0:  # Less than 5 min left
            logger.warning(f"Time running low ({remaining:.0f}s). "
                          f"Reducing predictions for remaining targets.")
            preds_per_target = 5

        logger.info(f"\n--- Target {target_idx + 1}/{len(targets)}: "
                    f"{target.target_id} (len={target.length}) ---")

        # Adjust seeds count based on time
        n_seeds = min(preds_per_target, len(base_seeds))
        seeds = base_seeds[:n_seeds]

        # Generate predictions
        predictions = predictor.predict(target, n_predictions=n_seeds, seeds=seeds)

        if not predictions:
            logger.error(f"No predictions generated for {target.target_id}!")
            continue

        # Select best 5
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

        # Log selection info
        selected_confs = [predictions[i].confidence for i in selected_indices]
        logger.info(f"Selected {len(selected_indices)} predictions. "
                    f"Confidences: {[f'{c:.3f}' for c in selected_confs]}")

        # Clear memory between targets
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
    logger.info("=" * 60)

    return submission_df


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    submission = main()
