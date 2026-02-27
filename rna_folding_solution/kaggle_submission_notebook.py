"""
Stanford RNA 3D Folding - Part 2
Kaggle Submission Notebook (converted to .py for readability)

This script is designed to run as a Kaggle notebook. It:
1. Installs required packages
2. Loads the test sequences
3. Predicts 5 diverse 3D structures per RNA
4. Generates the submission file

Note: For best results, use GPU (P100 or V100) in Kaggle.
"""

# ─────────────────────────────────────────────
# CELL 1: Install dependencies
# ─────────────────────────────────────────────
# !pip install -q ViennaRNA torch-geometric einops

# ─────────────────────────────────────────────
# CELL 2: Imports and constants
# ─────────────────────────────────────────────
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# Paths (Kaggle environment)
INPUT_DIR = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working"
MODEL_DIR = "/kaggle/input/rna-folding-model"  # If using pre-trained model

# Try to detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle")
DATA_DIR = INPUT_DIR if IS_KAGGLE else "./data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# ─────────────────────────────────────────────
# CELL 3: Sequence Utilities
# ─────────────────────────────────────────────
NT_VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4, "<pad>": 5}
N_NT = len(NT_VOCAB)
MAX_LEN = 512
D_MODEL = 256
N_HEADS = 8
N_ENCODER_LAYERS = 8
D_FF = 1024
D_PAIR = 64
N_PREDICTIONS = 5


def encode_sequence(seq: str) -> List[int]:
    seq = seq.upper().replace("T", "U")
    return [NT_VOCAB.get(c, NT_VOCAB["N"]) for c in seq]


# ─────────────────────────────────────────────
# CELL 4: Model Architecture
# ─────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class RNAEncoder(nn.Module):
    def __init__(self, n_nt=N_NT, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_ENCODER_LAYERS, d_ff=D_FF, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_nt, d_model, padding_idx=5)
        self.pe = SinusoidalPE(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, mask=None):
        x = self.pe(self.emb(tokens))
        return self.norm(self.enc(x, src_key_padding_mask=mask))


class PairNet(nn.Module):
    def __init__(self, d_single=D_MODEL, d_pair=D_PAIR, n_heads=4):
        super().__init__()
        self.pq = nn.Linear(d_single, d_pair)
        self.pk = nn.Linear(d_single, d_pair)
        self.outer = nn.Linear(d_pair * d_pair, d_pair)
        self.row_attn = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_pair), nn.Linear(d_pair, d_pair * 2),
            nn.GELU(), nn.Linear(d_pair * 2, d_pair)
        )

    def forward(self, s):
        B, L, _ = s.shape
        q, k = self.pq(s), self.pk(s)
        outer = torch.einsum("bid,bjd->bijd", q, k).reshape(B, L, L, -1)
        pair = self.outer(outer)
        pair_r, _ = self.row_attn(pair.reshape(B*L, L, -1), pair.reshape(B*L, L, -1), pair.reshape(B*L, L, -1))
        pair = pair_r.reshape(B, L, L, -1)
        pair_c, _ = self.col_attn(pair.permute(0,2,1,3).reshape(B*L, L, -1),
                                   pair.permute(0,2,1,3).reshape(B*L, L, -1),
                                   pair.permute(0,2,1,3).reshape(B*L, L, -1))
        pair = pair_c.reshape(B, L, L, -1).permute(0,2,1,3)
        return pair + self.ff(pair)


class DistanceHead(nn.Module):
    N_BINS = 37

    def __init__(self, d_pair=D_PAIR):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_pair), nn.Linear(d_pair, 64), nn.GELU(), nn.Linear(64, self.N_BINS)
        )
        self.register_buffer("centres", torch.linspace(1.0, 40.0, self.N_BINS))

    def forward(self, pair):
        logits = self.proj(pair)
        logits = (logits + logits.permute(0,2,1,3)) / 2
        probs = F.softmax(logits, dim=-1)
        exp_dist = (probs * self.centres).sum(-1)
        return logits, exp_dist


class SSHead(nn.Module):
    def __init__(self, d_pair=D_PAIR):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_pair), nn.Linear(d_pair, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, pair):
        logits = self.proj(pair).squeeze(-1)
        return torch.sigmoid((logits + logits.transpose(1,2)) / 2)


class StructureModule(nn.Module):
    def __init__(self, d_single=D_MODEL, d_pair=D_PAIR, n_iter=8):
        super().__init__()
        self.init = nn.Sequential(nn.LayerNorm(d_single), nn.Linear(d_single, 64), nn.GELU(), nn.Linear(64, 3))
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(d_single + d_pair + 3, 128), nn.GELU(), nn.Linear(128, 3))
            for _ in range(n_iter)
        ])
        self.plddt = nn.Sequential(nn.LayerNorm(d_single), nn.Linear(d_single, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, single, pair, exp_dist):
        coords = self.init(single)
        for upd in self.updates:
            pair_agg = pair.mean(2)
            coords = coords + upd(torch.cat([single, pair_agg, coords], -1))
        return coords, self.plddt(single).squeeze(-1)


class RNAFoldModel(nn.Module):
    def __init__(self, d_model=D_MODEL, d_pair=D_PAIR, n_enc=N_ENCODER_LAYERS, n_pair=4, n_struct=8):
        super().__init__()
        self.encoder = RNAEncoder(d_model=d_model, n_layers=n_enc)
        self.pair_net = PairNet(d_single=d_model, d_pair=d_pair)
        self.ss_head = SSHead(d_pair)
        self.dist_head = DistanceHead(d_pair)
        self.struct = StructureModule(d_model, d_pair, n_struct)

    def forward(self, tokens, mask=None):
        s = self.encoder(tokens, mask)
        p = self.pair_net(s)
        bp = self.ss_head(p)
        dist_logits, exp_dist = self.dist_head(p)
        coords, plddt = self.struct(s, p, exp_dist)
        return {"coords": coords, "bp_prob": bp, "dist_logits": dist_logits, "plddt": plddt}


# ─────────────────────────────────────────────
# CELL 5: Dataset
# ─────────────────────────────────────────────
class RNATestDataset(Dataset):
    def __init__(self, sequences: List[str], ids: List[str], max_len=MAX_LEN):
        self.sequences = sequences
        self.ids = ids
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        L = len(seq)
        tokens = torch.full((self.max_len,), fill_value=NT_VOCAB["<pad>"], dtype=torch.long)
        tokens[-L:] = torch.tensor(encode_sequence(seq), dtype=torch.long)
        mask = torch.ones(self.max_len, dtype=torch.bool)
        mask[-L:] = False
        return {"tokens": tokens, "mask": mask, "length": L, "seq": seq, "id": self.ids[idx]}


# ─────────────────────────────────────────────
# CELL 6: TM-score and Kabsch alignment
# ─────────────────────────────────────────────
def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q using Kabsch algorithm."""
    P_c = P - P.mean(0)
    Q_c = Q - Q.mean(0)
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, det])
    R = Vt.T @ D @ U.T
    return (P_c @ R.T) + Q.mean(0)


def compute_tm_score(pred: np.ndarray, ref: np.ndarray) -> float:
    L = min(len(pred), len(ref))
    Lref = len(ref)
    d0 = max(1.24 * (Lref - 15) ** (1/3) - 1.8, 0.5) if Lref > 19 else 0.5
    P_aligned = kabsch_align(pred[:L], ref[:L])
    di2 = ((P_aligned - ref[:L]) ** 2).sum(1)
    return float((1 / (1 + di2 / d0**2)).mean())


# ─────────────────────────────────────────────
# CELL 7: Secondary structure refinement
# ─────────────────────────────────────────────
def get_secondary_structure(seq: str):
    """Predict secondary structure (requires ViennaRNA or returns all dots)."""
    try:
        import RNA
        fc = RNA.fold_compound(seq.upper().replace("T", "U"))
        ss, _ = fc.mfe()
        return ss
    except Exception:
        return "." * len(seq)


def extract_pairs(ss: str):
    pairs, stack = [], []
    for i, c in enumerate(ss):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            pairs.append((stack.pop(), i))
    return pairs


def geometry_refine(coords: np.ndarray, seq: str, steps: int = 200) -> np.ndarray:
    """Spring-based structure refinement guided by secondary structure."""
    ss = get_secondary_structure(seq)
    pairs = extract_pairs(ss)
    coords = coords.copy().astype(np.float64)
    L = len(coords)

    IDEAL_BACKBONE = 3.8
    IDEAL_BP = 7.5
    LR = 0.05

    for step in range(steps):
        forces = np.zeros_like(coords)
        # Backbone springs
        for i in range(L - 1):
            d = coords[i+1] - coords[i]
            dn = np.linalg.norm(d)
            if dn < 1e-8:
                continue
            f = 2.0 * (dn - IDEAL_BACKBONE) * (d / dn)
            forces[i] += f
            forces[i+1] -= f
        # Base-pair springs
        for i, j in pairs:
            if i >= L or j >= L:
                continue
            d = coords[j] - coords[i]
            dn = np.linalg.norm(d)
            if dn < 1e-8:
                continue
            f = 0.5 * (dn - IDEAL_BP) * (d / dn)
            forces[i] += f
            forces[j] -= f
        # Clash prevention (soft repulsion)
        for i in range(L):
            for j in range(i+2, min(L, i+10)):
                d = coords[j] - coords[i]
                dn = np.linalg.norm(d)
                if dn < 4.0 and dn > 1e-8:
                    f = -0.1 * (4.0 - dn) * (d / dn)
                    forces[i] += f
                    forces[j] -= f

        coords += LR * forces
        LR *= 0.995

    return coords.astype(np.float32)


# ─────────────────────────────────────────────
# CELL 8: Diverse prediction generation
# ─────────────────────────────────────────────
def predict_diverse(
    model: RNAFoldModel,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    L: int,
    n: int = N_PREDICTIONS,
) -> List[Tuple[float, np.ndarray]]:
    """Generate n diverse predictions with pLDDT scores."""
    results = []

    # 1. Deterministic prediction
    model.eval()
    with torch.no_grad():
        out = model(tokens, mask)
    coords = out["coords"][0, -L:].cpu().numpy()
    plddt = out["plddt"][0, -L:].mean().item()
    results.append((plddt, coords))

    # 2. Stochastic (dropout-enabled) predictions
    model.train()
    for _ in range(n - 1):
        with torch.no_grad():
            out = model(tokens, mask)
        coords = out["coords"][0, -L:].cpu().numpy()
        plddt = out["plddt"][0, -L:].mean().item()
        results.append((plddt, coords))
    model.eval()

    # Sort by pLDDT descending (best first)
    results.sort(key=lambda x: -x[0])
    return results[:n]


# ─────────────────────────────────────────────
# CELL 9: Build submission DataFrame
# ─────────────────────────────────────────────
def build_submission(
    test_df: pd.DataFrame,
    all_preds: Dict[str, List[np.ndarray]],
    id_col: str = "target_id",
    seq_col: str = "sequence",
) -> pd.DataFrame:
    """
    Build the submission DataFrame.

    Competition format: one row per (sequence, model, nucleotide)
    with columns ID, x, y, z.
    """
    rows = []
    for _, row in test_df.iterrows():
        seq_id = str(row[id_col])
        preds = all_preds[seq_id]
        for m_idx, coords in enumerate(preds):
            L = coords.shape[0]
            for nt_idx in range(L):
                rows.append({
                    "ID": f"{seq_id}_{m_idx+1}_{nt_idx+1}",
                    "x": round(float(coords[nt_idx, 0]), 4),
                    "y": round(float(coords[nt_idx, 1]), 4),
                    "z": round(float(coords[nt_idx, 2]), 4),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# CELL 10: Main pipeline
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Stanford RNA 3D Folding - Prediction Pipeline")
    print("=" * 60)

    # ── Load test data ──
    test_csv = None
    for name in ["test.csv", "test_sequences.csv", "test_data.csv"]:
        path = os.path.join(DATA_DIR, "stanford-rna-3d-folding-2", name)
        if os.path.exists(path):
            test_csv = path
            break
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            test_csv = path
            break

    if test_csv is None:
        print("[WARNING] No test CSV found. Creating dummy test data for pipeline verification.")
        dummy_seqs = [
            "AUGCGAUCGAUAGCUAGCUAGCAUGC",
            "GCGAUAGCUAGCUAGCUAGCAUGCGA",
            "CUAGCUAGCUAGCAUGCGAUCGAUAG",
        ]
        test_df = pd.DataFrame({
            "target_id": [f"RNA_{i:04d}" for i in range(len(dummy_seqs))],
            "sequence": dummy_seqs,
        })
    else:
        test_df = pd.read_csv(test_csv)
        print(f"Loaded {len(test_df)} test sequences from {test_csv}")

    # Detect column names
    id_col = "target_id" if "target_id" in test_df.columns else test_df.columns[0]
    seq_col = "sequence" if "sequence" in test_df.columns else test_df.columns[1]
    print(f"Using columns: id={id_col}, seq={seq_col}")
    print(f"Sequence length range: {test_df[seq_col].str.len().min()} - {test_df[seq_col].str.len().max()}")

    # ── Initialize model ──
    print("\nInitializing model...")
    model = RNAFoldModel(
        d_model=D_MODEL,
        d_pair=D_PAIR,
        n_enc=N_ENCODER_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Load checkpoint if available ──
    ckpt_paths = [
        os.path.join(MODEL_DIR, "best_model.pt"),
        os.path.join(OUTPUT_DIR, "best_model.pt"),
        "best_model.pt",
    ]
    checkpoint_loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {ckpt_path}")
            checkpoint_loaded = True
            break
    if not checkpoint_loaded:
        print("No checkpoint found — using random weights (train first!)")

    # ── Predict ──
    print(f"\nGenerating {N_PREDICTIONS} predictions per sequence...")
    dataset = RNATestDataset(
        test_df[seq_col].tolist(),
        test_df[id_col].astype(str).tolist(),
        max_len=MAX_LEN,
    )

    all_preds = {}
    USE_REFINE = True  # Set to False for faster (but lower quality) predictions

    for i in range(len(dataset)):
        item = dataset[i]
        tokens = item["tokens"].unsqueeze(0).to(DEVICE)
        mask = item["mask"].unsqueeze(0).to(DEVICE)
        L = item["length"]
        seq_id = item["id"]
        seq = item["seq"]

        # Generate diverse predictions
        raw_preds = predict_diverse(model, tokens, mask, L, n=N_PREDICTIONS)

        # Optionally refine geometry
        final_preds = []
        for plddt, coords in raw_preds:
            if USE_REFINE and L <= 200:  # Only refine short sequences (speed)
                coords = geometry_refine(coords, seq, steps=100)
            final_preds.append(coords)

        all_preds[seq_id] = final_preds

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{len(dataset)} sequences")

    # ── Build submission ──
    print("\nBuilding submission...")
    submission = build_submission(test_df, all_preds, id_col, seq_col)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}")
    print(f"Total rows: {len(submission):,}")
    print(f"\nPreview:\n{submission.head(10)}")
    return submission


# ─────────────────────────────────────────────
# CELL 11: Run (Kaggle-compatible)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    submission = main()
    print("\nDone!")
