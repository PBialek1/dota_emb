"""dataset.py — Dataset loading for SimCLR training on Dota 2 laning features.

Supports the same two storage backends as extraction/store.py:
  - JSON store  : directory of {matchId}.json files
  - SQLite store: .db file with `matches` + `players` tables

Each sample is one player from one match. Returns (ts, scalars) tensors:
  - ts     : FloatTensor shape (18, 10) — 18 feature channels × 10 minute buckets
  - scalars: FloatTensor shape (7,)     — match-level scalar statistics

Channel order (18 total):
  [0–11]  timeseries: goldNorm, xpNorm, damageDealtNorm, damageTakenNorm, csNorm,
                       towerDamageNorm, healingNorm, healthPct, manaPct,
                       distToNearestAlly, distToNearestEnemy, distToNearestTower
  [12–15] events:     kills, deaths, assists, abilityCasts
  [16–17] proximity:  alliesNearby, enemiesNearby
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions — must stay in sync with extraction/feature_builder.py
# ---------------------------------------------------------------------------

TS_KEYS    = ["goldNorm", "xpNorm", "damageDealtNorm", "damageTakenNorm",
              "csNorm", "towerDamageNorm", "healingNorm", "healthPct", "manaPct",
              "distToNearestAlly", "distToNearestEnemy", "distToNearestTower"]
EVENT_KEYS = ["kills", "deaths", "assists", "abilityCasts"]
PROX_KEYS  = ["alliesNearby", "enemiesNearby"]
SCALAR_KEYS = ["maxGold", "maxXp", "maxDamageDealt", "maxDamageTaken",
               "maxCs", "maxTowerDamage", "maxHealing"]

N_TS_CHANNELS = len(TS_KEYS) + len(EVENT_KEYS) + len(PROX_KEYS)  # 18
N_TIMESTEPS   = 10
N_SCALARS     = len(SCALAR_KEYS)  # 7

# SQLite column name → feature name (mirrors extraction/store.py schema)
_TS_DB_COLS = {
    "gold_norm": "goldNorm", "xp_norm": "xpNorm",
    "damage_dealt_norm": "damageDealtNorm", "damage_taken_norm": "damageTakenNorm",
    "cs_norm": "csNorm", "tower_damage_norm": "towerDamageNorm",
    "healing_norm": "healingNorm", "health_pct": "healthPct", "mana_pct": "manaPct",
    "dist_to_nearest_ally": "distToNearestAlly", "dist_to_nearest_enemy": "distToNearestEnemy",
    "dist_to_nearest_tower": "distToNearestTower",
    "kills": "kills", "deaths": "deaths", "assists": "assists", "ability_casts": "abilityCasts",
    "allies_nearby": "alliesNearby", "enemies_nearby": "enemiesNearby",
}
_SCALAR_DB_COLS = {
    "max_gold": "maxGold", "max_xp": "maxXp",
    "max_damage_dealt": "maxDamageDealt", "max_damage_taken": "maxDamageTaken",
    "max_cs": "maxCs", "max_tower_damage": "maxTowerDamage", "max_healing": "maxHealing",
}

# Feature name → channel index (derived from channel order above)
_TS_CHAN = {k: i for i, k in enumerate(TS_KEYS + EVENT_KEYS + PROX_KEYS)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_list(val, length: int = N_TIMESTEPS) -> list:
    """Parse a JSON string or list; return a list of exactly `length` floats."""
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            val = []
    if not isinstance(val, list):
        val = []
    val = list(val)
    if len(val) < length:
        val.extend([0.0] * (length - len(val)))
    return [float(x or 0.0) for x in val[:length]]


def _parse_player_json(player: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (ts_array, scalar_array) from a JSON player dict."""
    ts = np.zeros((N_TS_CHANNELS, N_TIMESTEPS), dtype=np.float32)

    ts_src = player.get("timeseries") or {}
    ev_src = player.get("events") or {}

    for key in TS_KEYS:
        ts[_TS_CHAN[key]] = _safe_list(ts_src.get(key))
    for key in EVENT_KEYS:
        ts[_TS_CHAN[key]] = _safe_list(ev_src.get(key))
    for key in PROX_KEYS:
        ts[_TS_CHAN[key]] = _safe_list(player.get(key))

    sc_src = player.get("scalars") or {}
    scalars = np.array(
        [float(sc_src.get(k) or 0.0) for k in SCALAR_KEYS], dtype=np.float32
    )

    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
    return ts, scalars


# ---------------------------------------------------------------------------
# LaningDataset
# ---------------------------------------------------------------------------

class LaningDataset(Dataset):
    """
    Loads all player feature vectors into memory at init time.

    Normalization is applied lazily in __getitem__ once fit_normalizers() has
    been called. This allows the caller to split the data first and fit only
    on training samples.
    """

    def __init__(self, data_path: str | Path, transform: Callable | None = None):
        self.transform = transform
        self._ts_list:      list[np.ndarray] = []  # each shape (18, 10)
        self._scalar_list:  list[np.ndarray] = []  # each shape (7,)

        # Normalization stats — set by fit_normalizers()
        self.ts_mean:       np.ndarray | None = None  # shape (18,)
        self.ts_std:        np.ndarray | None = None  # shape (18,)
        self.scalar_scaler: StandardScaler | None = None

        data_path = Path(data_path)
        if data_path.is_dir():
            self._load_json(data_path)
        elif data_path.suffix in {".db", ".sqlite", ".sqlite3"}:
            self._load_sqlite(data_path)
        else:
            raise ValueError("data_path must be a directory (JSON) or a .db/.sqlite file")

        if len(self._ts_list) == 0:
            raise RuntimeError(f"No valid player samples loaded from {data_path}")

        logger.info("Loaded %d player samples.", len(self._ts_list))

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_json(self, directory: Path) -> None:
        files = sorted(directory.glob("*.json"))
        logger.info("Reading %d JSON files from %s …", len(files), directory)
        for f in files:
            try:
                try:
                    import orjson
                    data = orjson.loads(f.read_bytes())
                except ImportError:
                    data = json.loads(f.read_bytes())
            except Exception as exc:
                logger.warning("Skipping %s: %s", f.name, exc)
                continue
            for player in data.get("players") or []:
                ts, scalars = _parse_player_json(player)
                self._ts_list.append(ts)
                self._scalar_list.append(scalars)

    def _load_sqlite(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        ts_cols    = list(_TS_DB_COLS.keys())
        scalar_cols = list(_SCALAR_DB_COLS.keys())
        query = f"SELECT {', '.join(ts_cols + scalar_cols)} FROM players"
        logger.info("Querying SQLite store %s …", db_path)
        try:
            cur = conn.execute(query)
            for row in cur:
                ts_vals    = row[:len(ts_cols)]
                scalar_vals = row[len(ts_cols):]

                ts = np.zeros((N_TS_CHANNELS, N_TIMESTEPS), dtype=np.float32)
                for col_name, val in zip(ts_cols, ts_vals):
                    feat_name = _TS_DB_COLS[col_name]
                    ts[_TS_CHAN[feat_name]] = _safe_list(val)

                scalars = np.array(
                    [float(v or 0.0) for v in scalar_vals], dtype=np.float32
                )

                ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
                scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
                self._ts_list.append(ts)
                self._scalar_list.append(scalars)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def fit_normalizers(self, indices: list[int] | None = None) -> None:
        """
        Fit per-channel timeseries stats and a StandardScaler for scalars
        on the given subset of indices (default: all samples).
        """
        idx = indices if indices is not None else list(range(len(self)))

        ts_stack = np.stack([self._ts_list[i] for i in idx])  # (N, 18, 10)
        self.ts_mean = ts_stack.mean(axis=(0, 2))              # (18,)
        self.ts_std  = ts_stack.std(axis=(0, 2))               # (18,)

        scalar_stack = np.stack([self._scalar_list[i] for i in idx])  # (N, 7)
        self.scalar_scaler = StandardScaler().fit(scalar_stack)

        logger.info("Fitted normalizers on %d training samples.", len(idx))

    def _normalize_ts(self, ts: np.ndarray) -> np.ndarray:
        if self.ts_mean is None:
            return ts
        std = np.where(self.ts_std > 0, self.ts_std, 1.0)
        return ((ts - self.ts_mean[:, None]) / std[:, None]).astype(np.float32)

    def _normalize_scalars(self, scalars: np.ndarray) -> np.ndarray:
        if self.scalar_scaler is None:
            return scalars
        return self.scalar_scaler.transform(scalars.reshape(1, -1)).reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ts_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ts      = self._normalize_ts(self._ts_list[idx])
        scalars = self._normalize_scalars(self._scalar_list[idx])

        ts_t  = torch.from_numpy(ts)
        sc_t  = torch.from_numpy(scalars)

        if self.transform is not None:
            ts_t = self.transform(ts_t)

        return ts_t, sc_t


# ---------------------------------------------------------------------------
# SimCLRDataset
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """
    Wraps a Dataset (or Subset) that returns (ts, scalars) and applies two
    independent augmentations to the timeseries, returning both views.

    Returns: (ts_view1, scalars, ts_view2, scalars)
      - ts_view1, ts_view2: independently augmented ts tensors, shape (18, 10)
      - scalars: shared normalized scalar tensor, shape (7,)
        (scalars are not augmented — they are summary statistics invariant
         to within-game timing noise)
    """

    def __init__(self, base_dataset: Dataset, augmentation: Callable):
        self.base    = base_dataset
        self.augment = augmentation

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ts, scalars = self.base[idx]
        view1 = self.augment(ts.clone())
        view2 = self.augment(ts.clone())
        return view1, scalars, view2, scalars
