"""loss.py — NT-Xent (normalized temperature-scaled cross-entropy) loss for SimCLR.

Given a batch of N samples and their two augmented views (z1, z2), both already
L2-normalized, the loss maximizes agreement between positive pairs (same sample,
different augmentation) while pushing apart negative pairs (different samples).

Reference: Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations", ICML 2020.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_MIN_BATCH_WARN = 32


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for SimCLR.

    Args:
        temperature: Softmax temperature τ. Lower values make the distribution
            sharper and the task harder. Typical range: 0.05–0.5.

    Inputs:
        z1, z2: FloatTensors of shape (N, D), both L2-normalized.
            z1[i] and z2[i] are the two views of the same sample.

    Returns:
        Scalar loss (mean NT-Xent over the 2N rows).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)

        if N < _MIN_BATCH_WARN:
            logger.warning(
                "NT-Xent batch size is %d (< %d). "
                "Each sample has only %d negatives — consider increasing --batch-size.",
                N, _MIN_BATCH_WARN, 2 * N - 2,
            )

        # Stack both views: shape (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix, scaled by temperature: shape (2N, 2N)
        # z is already L2-normalized, so z @ z.T gives cosine similarities.
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on the diagonal (set to -inf so softmax ignores it)
        sim.fill_diagonal_(float("-inf"))

        # Positive pair labels:
        #   row i   (from z1) → positive is i+N (from z2)
        #   row i+N (from z2) → positive is i   (from z1)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N,     device=z.device),
        ])  # shape (2N,)

        # Cross-entropy over the 2N-1 non-self similarities per row
        # F.cross_entropy uses the log-sum-exp trick for numerical stability.
        return F.cross_entropy(sim, labels)
