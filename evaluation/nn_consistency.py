"""nn_consistency.py — Nearest-neighbor label consistency check for SimCLR embeddings.

For a random sample of players, finds their k nearest neighbors in embedding space
and measures how often those neighbors share the same label (position, hero, lane
outcome, etc.) versus what you'd expect by chance.

If the geometry is meaningful, position-5 players should be reliably closest to
other position-5 players far more often than the ~20% chance baseline.

Outputs per-label summary tables:
  - recall@k     : mean fraction of k-NN sharing the same label as the query
  - chance       : base rate of that class in the dataset (random baseline)
  - enrichment   : recall@k / chance  (1.0 = no better than random)
  - n_queries    : number of query points for that class

Usage:
    python evaluation/nn_consistency.py --data ./data/matches.db
    python evaluation/nn_consistency.py --data ./data/matches.db --k 10 --sample 2000
    python evaluation/nn_consistency.py --data ./data/matches.db --labels position laneOutcome
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from model import SimCLRModel
from dataset import (
    _safe_list, _TS_DB_COLS, _SCALAR_DB_COLS, _TS_CHAN,
    N_TS_CHANNELS, N_TIMESTEPS,
)

# Reuse the aligned loader and embedding builder from umap_embeddings
sys.path.insert(0, str(Path(__file__).parent))
from umap_embeddings import load_aligned, load_model, build_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AVAILABLE_LABELS = ["position", "heroName", "laneOutcome", "lane", "isVictory", "bracket"]


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def check_consistency(
    embeddings: np.ndarray,
    labels_df: pd.DataFrame,
    label_cols: list[str],
    k: int,
    n_sample: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """
    For each label column, compute per-class recall@k vs chance.

    Returns a dict mapping label name → DataFrame with columns:
        class, recall_at_k, chance, enrichment, n_queries
    """
    n = len(embeddings)
    sample_idx = rng.choice(n, size=min(n_sample, n), replace=False)

    logger.info("Fitting NearestNeighbors (k=%d) on %d embeddings …", k, n)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(embeddings)

    logger.info("Querying %d sample points …", len(sample_idx))
    # k+1 because the first neighbor is always the point itself
    _, indices = nn.kneighbors(embeddings[sample_idx])
    neighbor_indices = indices[:, 1:]  # shape (n_sample, k) — exclude self

    results: dict[str, pd.DataFrame] = {}

    for col in label_cols:
        if col not in labels_df.columns:
            logger.warning("Label column '%s' not found, skipping.", col)
            continue

        all_labels = np.array(labels_df[col], dtype=object)
        query_labels = all_labels[sample_idx]

        # Class frequencies (chance baseline)
        classes, counts = np.unique(all_labels, return_counts=True)
        class_freq = dict(zip(classes, counts / n))

        # Per-query recall: fraction of k-NN with the same label
        rows = []
        for cls in classes:
            cls_mask = query_labels == cls
            if cls_mask.sum() == 0:
                continue
            cls_query_idx = np.where(cls_mask)[0]
            cls_neighbor_labels = all_labels[neighbor_indices[cls_query_idx]]  # (n_cls, k)
            recall = (cls_neighbor_labels == cls).mean()
            chance = class_freq[cls]
            rows.append({
                "class":       cls,
                "recall_at_k": round(float(recall), 4),
                "chance":      round(float(chance), 4),
                "enrichment":  round(float(recall / chance) if chance > 0 else 0.0, 2),
                "n_queries":   int(cls_mask.sum()),
            })

        df = pd.DataFrame(rows).sort_values("enrichment", ascending=False).reset_index(drop=True)
        results[col] = df

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: dict[str, pd.DataFrame], k: int) -> None:
    for label, df in results.items():
        mean_recall = df["recall_at_k"].mean()
        mean_chance = df["chance"].mean()
        mean_enrich = df["enrichment"].mean()

        print(f"\n{'='*60}")
        print(f"  Label: {label}   (k={k})")
        print(f"  Mean recall@{k}: {mean_recall:.3f}   "
              f"Mean chance: {mean_chance:.3f}   "
              f"Mean enrichment: {mean_enrich:.2f}×")
        print(f"{'='*60}")
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor label consistency check for SimCLR embeddings."
    )
    parser.add_argument("--data",        required=True,
                        help="SQLite .db file.")
    parser.add_argument("--checkpoint",  default="./checkpoints/checkpoint_best.pt",
                        help="Path to model checkpoint.")
    parser.add_argument("--k",           type=int, default=15,
                        help="Number of nearest neighbors to retrieve (default: 15).")
    parser.add_argument("--sample",      type=int, default=2000,
                        help="Number of query points to sample (default: 2000).")
    parser.add_argument("--labels",      nargs="+", default=["position", "heroName", "laneOutcome"],
                        choices=AVAILABLE_LABELS,
                        help="Label columns to evaluate (default: position heroName laneOutcome).")
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      type=str, default=None,
                        help="Optional path to save results as a CSV (one file per label, "
                             "e.g. './analysis/nn_position.csv').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, ckpt = load_model(Path(args.checkpoint), device)
    ts_t, sc_t, labels_df = load_aligned(Path(args.data), max_players=None, ckpt=ckpt)

    logger.info("Encoding %d players …", len(ts_t))
    embeddings = build_embeddings(model, ts_t, sc_t, args.batch_size, device)

    results = check_consistency(
        embeddings, labels_df,
        label_cols=args.labels,
        k=args.k,
        n_sample=args.sample,
        rng=rng,
    )

    print_results(results, k=args.k)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.stem
        suffix = out_path.suffix or ".csv"
        for label, df in results.items():
            save_path = out_path.parent / f"{stem}_{label}{suffix}"
            df.to_csv(save_path, index=False)
            logger.info("Saved %s → %s", label, save_path)


if __name__ == "__main__":
    main()
