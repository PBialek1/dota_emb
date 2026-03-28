"""train.py — SimCLR training script for Dota 2 laning representations.

Usage:
    python training/train.py --data ./data/matches
    python training/train.py --data ./data/matches.db --epochs 200 --batch-size 512
    python training/train.py --data ./data/matches --embed-dim 128 --temperature 0.1

Resuming a stopped run:
    python training/train.py --data ./data/matches --resume ./checkpoints/checkpoint_best.pt
    python training/train.py --data ./data/matches --resume ./checkpoints/checkpoint_epoch_050.pt --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader, random_split

# Allow bare imports of sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from augmentations import default_augmentation
from dataset import LaningDataset, SimCLRDataset
from loss import NTXentLoss
from model import SimCLRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SimCLR representation on Dota 2 laning features."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to JSON match directory or SQLite .db file.",
    )
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--temperature",  type=float, default=0.07,
                        help="NT-Xent temperature τ (default: 0.07).")
    parser.add_argument("--embed-dim",    type=int,   default=256,
                        help="Encoder output dimension (default: 256).")
    parser.add_argument("--save-dir",     type=str,   default="./checkpoints",
                        help="Directory for checkpoint files.")
    parser.add_argument("--val-split",    type=float, default=0.1,
                        help="Fraction of data held out for validation (default: 0.1).")
    parser.add_argument("--num-workers",  type=int,   default=0,
                        help="DataLoader worker processes. Use 0 on Windows (default: 0).")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--log-interval", type=int,   default=10,
                        help="Print training loss every N batches (default: 10).")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to a checkpoint file to resume training from.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    model: SimCLRModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
    dataset: LaningDataset,
) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "args":                 vars(args),
            "ts_mean":              dataset.ts_mean,
            "ts_std":               dataset.ts_std,
            "scalar_scaler":        dataset.scalar_scaler,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s …", args.data)
    dataset = LaningDataset(Path(args.data))
    n = len(dataset)
    logger.info("Total player samples: %d", n)

    if n < args.batch_size * 2:
        logger.warning(
            "Dataset has only %d samples; consider reducing --batch-size "
            "(currently %d) for effective contrastive learning.",
            n, args.batch_size,
        )

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    n_val   = max(1, int(n * args.val_split))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    logger.info("Split: %d train / %d val", n_train, n_val)

    # Fit normalizers on training data only (skipped when resuming — loaded from checkpoint)
    if not args.resume:
        dataset.fit_normalizers(list(train_subset.indices))
        scaler_path = save_dir / "scalar_scaler.pkl"
        with open(scaler_path, "wb") as fh:
            pickle.dump(
                {
                    "scalar_scaler": dataset.scalar_scaler,
                    "ts_mean":       dataset.ts_mean,
                    "ts_std":        dataset.ts_std,
                },
                fh,
            )
        logger.info("Saved normalizers to %s", scaler_path)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    aug = default_augmentation()
    train_ds = SimCLRDataset(train_subset, aug)
    val_ds   = SimCLRDataset(val_subset,   aug)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,   # NT-Xent requires consistent batch sizes; last batch may be size 1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler, loss
    # ------------------------------------------------------------------
    model     = SimCLRModel(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = NTXentLoss(temperature=args.temperature)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", n_params)

    start_epoch   = 1
    best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        # Restore normalizers from checkpoint so the same split/scaler is used
        dataset.ts_mean       = ckpt["ts_mean"]
        dataset.ts_std        = ckpt["ts_std"]
        dataset.scalar_scaler = ckpt["scalar_scaler"]
        # Fast-forward the scheduler to match the resumed epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        logger.info(
            "Resumed from %s (epoch %d, val_loss=%.4f)",
            ckpt_path, ckpt["epoch"], best_val_loss,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses: list[float] = []
        for batch_idx, (ts1, sc, ts2, _) in enumerate(train_loader):
            ts1, sc, ts2 = ts1.to(device), sc.to(device), ts2.to(device)

            optimizer.zero_grad()
            _, z1 = model(ts1, sc)
            _, z2 = model(ts2, sc)
            loss  = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (batch_idx + 1) % args.log_interval == 0:
                logger.debug(
                    "  batch %d/%d  loss=%.4f",
                    batch_idx + 1, len(train_loader), loss.item(),
                )

        # Validate
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for ts1, sc, ts2, _ in val_loader:
                ts1, sc, ts2 = ts1.to(device), sc.to(device), ts2.to(device)
                _, z1 = model(ts1, sc)
                _, z2 = model(ts2, sc)
                val_losses.append(criterion(z1, z2).item())

        scheduler.step()

        train_loss = mean(train_losses)
        val_loss   = mean(val_losses) if val_losses else float("nan")
        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "[Epoch %03d/%03d]  train_loss=%.4f  val_loss=%.4f  lr=%.3e  elapsed=%.1fs",
            epoch, args.epochs, train_loss, val_loss, current_lr, elapsed,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / "checkpoint_best.pt"
            _save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss, args, dataset)
            logger.info("  ✓ New best val_loss=%.4f — saved to %s", val_loss, best_path)

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            _save_checkpoint(ckpt_path, model, optimizer, epoch, train_loss, val_loss, args, dataset)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
