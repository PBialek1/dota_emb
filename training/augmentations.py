"""augmentations.py — Stochastic augmentations for laning-stage timeseries.

All transforms operate on a single FloatTensor of shape (C, T):
  C channels (see dataset.py for channel order) × T 15-second buckets.

Current layout (C = 28, T = 40):
  Channels 0–8   pre-normalized state/cumulative curves in [0, 1]
                 (goldNorm, xpNorm, damageDealt/Taken/Norm, csNorm,
                  towerDamageNorm, healingNorm, healthPct, manaPct).
  Channels 9–23  continuous distance features in STRATZ coords (~0–150):
                 dist_to_ally_{0..3}, dist_to_enemy_{0..4}, dist_to_tower_{0..5}.
  Channels 24–27 integer event counts (kills, deaths, assists, abilityCasts).

Augmentations are gated by a per-instance probability `p`. When the gate
fires, the transform is applied; otherwise the tensor is returned unchanged.

After per-channel timeseries standardization in LaningDataset, all channels
are approximately zero-mean, unit-variance — augmentation hyperparameters
are therefore scale-invariant.
"""

from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Add i.i.d. Gaussian noise to every element."""

    def __init__(self, sigma: float = 0.03, p: float = 0.8):
        self.sigma = sigma
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        return x + torch.randn_like(x) * self.sigma


class TemporalShift:
    """Roll all channels along the time axis by a random ±max_shift steps."""

    def __init__(self, max_shift: int = 1, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=1)


class FeatureMask:
    """Zero out randomly selected feature channels (entire rows)."""

    def __init__(self, mask_prob: float = 0.15, p: float = 0.7, max_attempts: int = 3):
        self.mask_prob = mask_prob
        self.p = p
        self.max_attempts = max_attempts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        n_channels = x.shape[0]
        for _ in range(self.max_attempts):
            mask = torch.bernoulli(torch.full((n_channels, 1), self.mask_prob))
            if mask.sum() <= n_channels // 2:
                break
        return x * (1.0 - mask)


class ScaleJitter:
    """Multiply continuous channels by independent random scale factors.

    `n_continuous` defaults to 24 — covers channels 0–23 (norm curves,
    state pcts, and all dist_to_ally/enemy/tower channels). Event-count
    channels (24–27) are left untouched because they are discrete counts.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.85, 1.15),
        p: float = 0.6,
        n_continuous: int = 24,
    ):
        self.low, self.high = scale_range
        self.p = p
        self._n_continuous = n_continuous

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        out = x.clone()
        n = min(self._n_continuous, x.shape[0])
        scale = torch.empty(n, 1).uniform_(self.low, self.high)
        out[:n] = out[:n] * scale
        return out


class TimestepDropout:
    """Zero out randomly selected timestep columns (all channels at one bucket)."""

    def __init__(
        self,
        drop_prob: float = 0.1,
        p: float = 0.5,
        max_attempts: int = 3,
        max_drop_fraction: float = 0.3,
    ):
        self.drop_prob = drop_prob
        self.p = p
        self.max_attempts = max_attempts
        self.max_drop_fraction = max_drop_fraction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        n_steps = x.shape[1]
        max_drop = max(1, int(n_steps * self.max_drop_fraction))
        for _ in range(self.max_attempts):
            mask = torch.bernoulli(torch.full((1, n_steps), self.drop_prob))
            if mask.sum() <= max_drop:
                break
        return x * (1.0 - mask)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Apply a sequence of augmentations in order."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
# Default augmentation pipeline
# ---------------------------------------------------------------------------

def default_augmentation() -> Compose:
    """
    Standard SimCLR augmentation stack for laning timeseries.

    Probabilities are set so two independent draws of this pipeline
    produce meaningfully different views that still share the same
    semantic content (role, lane, playstyle).
    """
    return Compose([
        GaussianNoise(sigma=0.03, p=0.8),
        TemporalShift(max_shift=1, p=0.5),
        FeatureMask(mask_prob=0.15, p=0.7),
        ScaleJitter(scale_range=(0.85, 1.15), p=0.6),
        TimestepDropout(drop_prob=0.1, p=0.5),
    ])
