"""
Training script for RNA 3D Folding model.

Usage:
    python train.py --data_dir /path/to/data --epochs 100 --lr 3e-4 --batch_size 8

Data directory should contain:
    train_sequences.csv  - columns: target_id, sequence
    train_coords.csv     - columns: target_id, residue_index, x, y, z
    val_sequences.csv    (optional)
    val_coords.csv       (optional)
"""

import argparse
import os
import math
import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rna_model import (
    RNAFoldModel,
    RNADataset,
    train_epoch,
    D_MODEL,
    N_ENCODER_LAYERS,
    N_PAIR_LAYERS,
    MAX_LEN,
)
from rna_utils import (
    load_sequences_from_csv,
    load_training_coords,
    tm_score,
    generate_diverse_predictions,
    best_of_five_tm_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: RNAFoldModel,
    loader: DataLoader,
    device: torch.device,
    n_eval: int = 50,
) -> dict:
    """Compute validation FAPE loss and TM-score on a subset."""
    model.eval()
    fape_losses = []
    tm_scores = []

    dist_centres = model.dist_head.dist_centres.to(device)

    from rna_model import total_loss

    for i, batch in enumerate(loader):
        if i >= n_eval:
            break

        tokens = batch["tokens"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = model(tokens, padding_mask)
        loss = total_loss(outputs, batch_dev, dist_centres)
        fape_losses.append(loss.item())

        # TM-score on first sample of batch
        if "coords" in batch:
            L = batch["length"][0].item()
            pred_c = outputs["coords"][0, -L:].cpu().numpy()
            true_c = batch["coords"][0, -L:].cpu().numpy()
            tm = tm_score(pred_c, true_c)
            tm_scores.append(tm)

    return {
        "val_loss": float(np.mean(fape_losses)) if fape_losses else 0.0,
        "val_tm_score": float(np.mean(tm_scores)) if tm_scores else 0.0,
    }


# ─────────────────────────────────────────────
#  Cosine LR schedule with warmup
# ─────────────────────────────────────────────
def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
#  Main Training
# ─────────────────────────────────────────────
def build_dataset(data_dir: str, max_len: int = MAX_LEN):
    """Load and build train/val datasets."""
    data_dir = Path(data_dir)

    # Load sequences
    train_seq_path = data_dir / "train_sequences.csv"
    if not train_seq_path.exists():
        # Try alternative naming conventions
        for name in ["train.csv", "sequences.csv", "train_data.csv"]:
            if (data_dir / name).exists():
                train_seq_path = data_dir / name
                break

    ids, seqs = load_sequences_from_csv(str(train_seq_path))

    # Load coordinates if available
    coords_dict = {}
    for coord_name in ["train_coords.csv", "train_labels.csv", "coords.csv"]:
        coord_path = data_dir / coord_name
        if coord_path.exists():
            try:
                coords_dict = load_training_coords(str(coord_path))
            except Exception as e:
                logger.warning(f"Could not load coords from {coord_path}: {e}")
            break

    coords_list = [coords_dict.get(sid) for sid in ids]
    dataset = RNADataset(seqs, coords_list, max_len=max_len)
    return dataset, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--n_encoder_layers", type=int, default=N_ENCODER_LAYERS)
    parser.add_argument("--n_pair_layers", type=int, default=N_PAIR_LAYERS)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──
    logger.info("Loading dataset...")
    dataset, ids = build_dataset(args.data_dir, max_len=args.max_len)
    logger.info(f"Total samples: {len(dataset)}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model ──
    model = RNAFoldModel(
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_pair_layers=args.n_pair_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    start_epoch = 0
    best_val_tm = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_tm = ckpt.get("best_val_tm", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ──
    metrics_log = []
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, grad_clip=args.grad_clip
        )
        scheduler.step()

        val_metrics = validate(model, val_loader, device)
        val_tm = val_metrics["val_tm_score"]

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_tm={val_tm:.4f} | lr={lr:.2e}"
        )

        metrics_log.append(
            {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics, "lr": lr}
        )

        # Save best checkpoint
        if val_tm > best_val_tm:
            best_val_tm = val_tm
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_tm": best_val_tm,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"  ✓ Saved best model (TM={best_val_tm:.4f})")

        # Save metrics
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_log, f, indent=2)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_tm": best_val_tm,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pt"),
            )

    logger.info(f"Training complete. Best val TM-score: {best_val_tm:.4f}")


if __name__ == "__main__":
    main()
