"""model.py — Encoder and projection head for SimCLR on laning features.

Architecture overview:

  LaningEncoder
  ├── TimeseriesBranch  : Conv1d stack over (18, 10) timeseries → 256-dim
  ├── ScalarBranch      : MLP over (7,) scalars → 32-dim
  └── FusionHead        : Linear(288 → embed_dim) → embed_dim-dim embedding h

  ProjectionHead        : MLP(embed_dim → embed_dim → 128) → 128-dim projection z

  SimCLRModel           : LaningEncoder + ProjectionHead
    forward(ts, scalars) → (h, z_normalized)
      h : (B, embed_dim) — use this for downstream tasks / visualization
      z : (B, 128)       — L2-normalized; used only for NT-Xent loss during training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class TimeseriesBranch(nn.Module):
    """1-D CNN encoder for the (18, 10) timeseries tensor.

    Input : (B, 18, 10)
    Output: (B, 256)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            _conv_block(18,  64),   # (B, 64,  10)
            _conv_block(64,  128),  # (B, 128, 10)
            _conv_block(128, 256),  # (B, 256, 10)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 18, 10)
        out = self.conv(x)          # (B, 256, 10)
        out = self.pool(out)        # (B, 256, 1)
        return out.squeeze(-1)      # (B, 256)


class ScalarBranch(nn.Module):
    """Small MLP for the (7,) scalar feature vector.

    Input : (B, 7)
    Output: (B, 32)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 32)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LaningEncoder(nn.Module):
    """
    Encodes a (ts, scalars) pair into a fixed-size embedding vector h.

    Input : ts=(B, 18, 10), scalars=(B, 7)
    Output: h=(B, embed_dim)
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.ts_branch     = TimeseriesBranch()   # → (B, 256)
        self.scalar_branch = ScalarBranch()        # → (B, 32)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, ts: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        ts_feat  = self.ts_branch(ts)           # (B, 256)
        sc_feat  = self.scalar_branch(scalars)  # (B, 32)
        combined = torch.cat([ts_feat, sc_feat], dim=1)  # (B, 288)
        return self.fusion(combined)            # (B, embed_dim)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head used only during SimCLR training.
    Maps the encoder output h to the contrastive space z.

    Input : (B, embed_dim)
    Output: (B, 128)  — L2-normalized by SimCLRModel.forward
    """

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B, proj_dim)


# ---------------------------------------------------------------------------
# SimCLR model
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """
    Full SimCLR model: encoder + projection head.

    forward() returns both the encoder embedding h and the L2-normalized
    projection z:
      - Use z (via NTXentLoss) during training.
      - Use h for downstream tasks, UMAP visualization, and similarity search.
    """

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.encoder    = LaningEncoder(embed_dim=embed_dim)
        self.projection = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)

    def forward(
        self, ts: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(ts, scalars)            # (B, embed_dim)
        z = self.projection(h)                   # (B, proj_dim)
        z = F.normalize(z, dim=1, p=2)           # unit hypersphere
        return h, z

    def encode(self, ts: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Inference-time helper: returns only the embedding h."""
        with torch.no_grad():
            h, _ = self.forward(ts, scalars)
        return h
