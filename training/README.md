# training

SimCLR contrastive representation learning on Dota 2 laning-stage feature vectors.

Each player's laning phase is represented as 18 feature channels × 10 minute buckets, plus 7 scalar statistics. The model learns an embedding space where players with similar laning behaviors are close together, without any supervised labels.

---

## Method

**SimCLR** (Chen et al., 2020): two randomly augmented views of the same player are treated as a positive pair. The encoder is trained to agree on their representation while disagreeing with all other players in the batch (NT-Xent loss).

**Augmentations applied to the timeseries:**
| Augmentation | Effect | Probability |
|---|---|---|
| Gaussian noise | Small i.i.d. perturbation | 0.8 |
| Temporal shift | Roll channels ±1 minute step | 0.5 |
| Feature mask | Zero out random feature channels | 0.7 |
| Scale jitter | Multiply continuous channels by ∈ [0.85, 1.15] | 0.6 |
| Timestep dropout | Zero out up to 3 complete minute columns | 0.5 |

Scalar features (maxGold, maxXp, …) are treated as invariant context and are not augmented.

---

## Architecture

```
Input: ts (18, 10) + scalars (7,)
          │                 │
   TimeseriesBranch    ScalarBranch
   Conv1d(18→64)       Linear(7→32)
   Conv1d(64→128)      BN + ReLU
   Conv1d(128→256)          │
   AdaptiveAvgPool          │
        │                   │
        └──── concat (288) ─┘
                   │
           Linear(288→embed_dim)
           BN + ReLU
                   │
                   h  ← use this for downstream tasks / UMAP
                   │
           ProjectionHead (2-layer MLP)
                   │
                   z  ← L2-normalized; used only for NT-Xent loss
```

Default `embed_dim=256`, projection dim=128.

---

## Usage

Run from the project root:

```bash
# Train on a JSON store
python training/train.py --data ./data/matches

# Train on SQLite
python training/train.py --data ./data/matches.db

# Common options
python training/train.py \
    --data ./data/matches.db \
    --epochs 200 \
    --batch-size 512 \
    --lr 3e-4 \
    --temperature 0.07 \
    --embed-dim 256 \
    --save-dir ./checkpoints
```

Checkpoints are saved to `--save-dir`:
- `checkpoint_best.pt` — best validation loss so far
- `checkpoint_epoch_NNN.pt` — every 10 epochs
- `scalar_scaler.pkl` — normalizer stats for inference

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--data` | *(required)* | JSON directory or `.db` file |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 256 | Samples per batch (larger = more negatives = better) |
| `--lr` | 3e-4 | Adam learning rate |
| `--temperature` | 0.07 | NT-Xent temperature τ |
| `--embed-dim` | 256 | Encoder output dimension |
| `--save-dir` | `./checkpoints` | Checkpoint output directory |
| `--val-split` | 0.1 | Fraction held out for validation |
| `--num-workers` | 0 | DataLoader workers (keep 0 on Windows) |
| `--seed` | 42 | Random seed |
| `--log-interval` | 10 | Log batch loss every N steps |

---

## Loading a Checkpoint

```python
import torch
import pickle
from training.model import SimCLRModel

# Load model
ckpt = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu")
model = SimCLRModel(embed_dim=ckpt["args"]["embed_dim"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load normalizers
with open("checkpoints/scalar_scaler.pkl", "rb") as f:
    norms = pickle.load(f)
# norms["ts_mean"], norms["ts_std"], norms["scalar_scaler"]
```

---

## Files

| File | Description |
|---|---|
| `train.py` | CLI training script |
| `dataset.py` | Data loading, normalization, SimCLR dataset wrapper |
| `augmentations.py` | Stochastic timeseries augmentations |
| `model.py` | Encoder (1D CNN + MLP) and projection head |
| `loss.py` | NT-Xent contrastive loss |
