"""dataset.py — Dataset loading for SimCLR training on Dota 2 laning features.

Supports the same two storage backends as extraction/store.py:
  - JSON store  : directory of {matchId}.json files
  - SQLite store: .db file with `matches` + `players` tables

Each sample is one player from one match. Returns (ts, scalars) tensors:
  - ts     : FloatTensor shape (28, 40) — 28 feature channels × 40 × 15-second buckets
  - scalars: FloatTensor shape (7,)     — match-level scalar statistics

Channel order (28 total):
  [0–8]   timeseries: goldNorm, xpNorm, damageDealtNorm, damageTakenNorm, csNorm,
                       towerDamageNorm, healingNorm, healthPct, manaPct
  [9–12]  ally distances:  distToAlly0–3  (order fixed by proximity at t=90s)
  [13–17] enemy distances: distToEnemy0–4 (order fixed by proximity at t=90s)
  [18–23] tower distances: distToTower0–5 (Radiant top/mid/bot, Dire top/mid/bot)
  [24–27] events:          kills, deaths, assists, abilityCasts
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Allow importing the extraction-side feature builder so we can compute
# normalized features on the fly from raw-event SQLite stores.
_EXTRACTION_DIR = Path(__file__).resolve().parent.parent / "extraction"
if str(_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTRACTION_DIR))

from feature_builder import build_match as _build_match  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions — must stay in sync with extraction/feature_builder.py
# ---------------------------------------------------------------------------

TS_KEYS = [
    "goldNorm", "xpNorm", "damageDealtNorm", "damageTakenNorm",
    "csNorm", "towerDamageNorm", "healingNorm", "healthPct", "manaPct",
    "distToAlly0", "distToAlly1", "distToAlly2", "distToAlly3",
    "distToEnemy0", "distToEnemy1", "distToEnemy2", "distToEnemy3", "distToEnemy4",
    "distToTower0", "distToTower1", "distToTower2", "distToTower3", "distToTower4", "distToTower5",
    "kills", "deaths", "assists", "abilityCasts",
]
SCALAR_KEYS = ["maxGold", "maxXp", "maxDamageDealt", "maxDamageTaken",
               "maxCs", "maxTowerDamage", "maxHealing"]

N_TS_CHANNELS = len(TS_KEYS)  # 28
N_TIMESTEPS   = 40
N_SCALARS     = len(SCALAR_KEYS)  # 7

# SQLite column name → feature name (mirrors extraction/store.py schema)
_TS_DB_COLS = {
    "gold_norm": "goldNorm", "xp_norm": "xpNorm",
    "damage_dealt_norm": "damageDealtNorm", "damage_taken_norm": "damageTakenNorm",
    "cs_norm": "csNorm", "tower_damage_norm": "towerDamageNorm",
    "healing_norm": "healingNorm", "health_pct": "healthPct", "mana_pct": "manaPct",
    "dist_to_ally_0": "distToAlly0", "dist_to_ally_1": "distToAlly1",
    "dist_to_ally_2": "distToAlly2", "dist_to_ally_3": "distToAlly3",
    "dist_to_enemy_0": "distToEnemy0", "dist_to_enemy_1": "distToEnemy1",
    "dist_to_enemy_2": "distToEnemy2", "dist_to_enemy_3": "distToEnemy3",
    "dist_to_enemy_4": "distToEnemy4",
    "dist_to_tower_0": "distToTower0", "dist_to_tower_1": "distToTower1",
    "dist_to_tower_2": "distToTower2", "dist_to_tower_3": "distToTower3",
    "dist_to_tower_4": "distToTower4", "dist_to_tower_5": "distToTower5",
    "kills": "kills", "deaths": "deaths", "assists": "assists", "ability_casts": "abilityCasts",
}
_SCALAR_DB_COLS = {
    "max_gold": "maxGold", "max_xp": "maxXp",
    "max_damage_dealt": "maxDamageDealt", "max_damage_taken": "maxDamageTaken",
    "max_cs": "maxCs", "max_tower_damage": "maxTowerDamage", "max_healing": "maxHealing",
}

# Feature name → channel index
_TS_CHAN = {k: i for i, k in enumerate(TS_KEYS)}


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
        # Distance/state features live in player["timeseries"];
        # event counts (kills, deaths, etc.) live in player["events"].
        val = ts_src.get(key) if key in ts_src else ev_src.get(key)
        ts[_TS_CHAN[key]] = _safe_list(val)

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

    def __init__(
        self,
        data_path: str | Path,
        transform: Callable | None = None,
        *,
        cache: bool = True,
        cache_path: str | Path | None = None,
    ):
        """
        Args:
            data_path:  Path to a directory of JSON files or a .db/.sqlite file.
            transform:  Optional per-sample transform applied in __getitem__.
            cache:      If True (default), write/read a .npz cache of computed
                        features next to the source DB so subsequent runs skip
                        the feature-build cost.
            cache_path: Override the cache file location.
        """
        self.transform = transform
        self._ts_list:      list[np.ndarray] = []  # each shape (28, 40)
        self._scalar_list:  list[np.ndarray] = []  # each shape (7,)
        # DB primary key of the source `players` row for each sample (where
        # known — JSON/legacy paths leave this empty). Allows downstream
        # tools to join cached features against label columns without
        # re-running the feature build.
        self._player_ids:   list[int] = []

        # Normalization stats — set by fit_normalizers()
        self.ts_mean:       np.ndarray | None = None  # shape (28,)
        self.ts_std:        np.ndarray | None = None  # shape (28,)
        self.scalar_scaler: StandardScaler | None = None

        data_path = Path(data_path)
        cache_file = (
            Path(cache_path) if cache_path is not None
            else (data_path.parent / (data_path.stem + ".features.npz") if not data_path.is_dir() else None)
        )

        if cache and cache_file is not None and self._try_load_cache(cache_file, data_path):
            pass
        else:
            if data_path.is_dir():
                self._load_json(data_path)
            elif data_path.suffix in {".db", ".sqlite", ".sqlite3"}:
                self._load_sqlite(data_path)
            else:
                raise ValueError("data_path must be a directory (JSON) or a .db/.sqlite file")

            if cache and cache_file is not None and self._ts_list:
                self._write_cache(cache_file, data_path)

        if len(self._ts_list) == 0:
            raise RuntimeError(f"No valid player samples loaded from {data_path}")

        logger.info("Loaded %d player samples.", len(self._ts_list))

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    @staticmethod
    def _source_fingerprint(source: Path) -> tuple[int, int]:
        """(size_bytes, mtime_ns) — cheap staleness check for the cache."""
        st = source.stat()
        return int(st.st_size), int(st.st_mtime_ns)

    def _try_load_cache(self, cache_path: Path, source: Path) -> bool:
        if not cache_path.exists() or not source.exists():
            return False
        try:
            data = np.load(cache_path)
            cached_size  = int(data["source_size"])
            cached_mtime = int(data["source_mtime"])
            cur_size, cur_mtime = self._source_fingerprint(source)
            if (cached_size, cached_mtime) != (cur_size, cur_mtime):
                logger.info(
                    "Cache %s is stale (source changed); will rebuild.", cache_path
                )
                return False
            ts_stack     = data["ts"]
            scalar_stack = data["scalars"]
            if ts_stack.shape[1:] != (N_TS_CHANNELS, N_TIMESTEPS) or scalar_stack.shape[1] != N_SCALARS:
                logger.info(
                    "Cache %s has incompatible shape %s; will rebuild.",
                    cache_path, ts_stack.shape,
                )
                return False
            self._ts_list     = list(ts_stack.astype(np.float32, copy=False))
            self._scalar_list = list(scalar_stack.astype(np.float32, copy=False))
            # `player_ids` was added later — caches written before that change
            # don't include them. We still accept the cache and leave the ids
            # list empty; callers that need ids can opportunistically backfill
            # via `ensure_player_ids(db_path)`.
            if "player_ids" in data.files and len(data["player_ids"]) == len(ts_stack):
                self._player_ids = [int(x) for x in data["player_ids"]]
            else:
                self._player_ids = []
            logger.info(
                "Loaded %d samples from cache %s%s.",
                len(self._ts_list), cache_path,
                "" if self._player_ids else " (player_ids missing)",
            )
            return True
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("Failed to load cache %s (%s); will rebuild.", cache_path, exc)
            return False

    def _write_cache(self, cache_path: Path, source: Path) -> None:
        try:
            ts_stack     = np.stack(self._ts_list).astype(np.float32, copy=False)
            scalar_stack = np.stack(self._scalar_list).astype(np.float32, copy=False)
            # If player_ids weren't populated by the loader (e.g. JSON path),
            # write an empty array so consumers can detect "ids unavailable"
            # via length-zero rather than missing-key handling.
            player_ids = (
                np.array(self._player_ids, dtype=np.int64)
                if len(self._player_ids) == len(self._ts_list)
                else np.zeros(0, dtype=np.int64)
            )
            size, mtime = self._source_fingerprint(source)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            # np.savez auto-appends ".npz" when given a path string — pass an open
            # file handle instead so the temp filename is exactly what we asked for.
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    ts=ts_stack,
                    scalars=scalar_stack,
                    player_ids=player_ids,
                    source_size=np.int64(size),
                    source_mtime=np.int64(mtime),
                )
            tmp.replace(cache_path)
            logger.info(
                "Wrote feature cache: %s (%d samples, %.1f MB).",
                cache_path, len(self._ts_list),
                cache_path.stat().st_size / (1024 * 1024),
            )
        except OSError as exc:
            logger.warning("Failed to write cache %s: %s", cache_path, exc)

    def ensure_player_ids(self, db_path: str | Path) -> bool:
        """If the loaded cache is missing player_ids, fast-backfill them by
        scanning the source DB and matching cached `maxGold` values against
        recomputed per-row maxGold. On success, also rewrites the cache so
        subsequent runs are fast.

        Returns True on success, False if the cache and DB disagree.
        """
        if len(self._player_ids) == len(self._ts_list):
            return True

        db_path = Path(db_path)
        if not db_path.exists():
            logger.warning("ensure_player_ids: source DB not found at %s.", db_path)
            return False

        try:
            import orjson
            _loads = orjson.loads
        except ImportError:
            _loads = json.loads

        n_cached = len(self._ts_list)
        # scalars[0] is maxGold by SCALAR_KEYS order.
        cached_max_gold = np.array(
            [int(s[0]) for s in self._scalar_list], dtype=np.int64
        )

        kept_ids: list[int] = []
        cache_idx = 0
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT id, gold FROM players ORDER BY match_id, id"
            )
            for player_id, gold_blob in cur:
                if not gold_blob:
                    continue
                try:
                    events = (
                        _loads(gold_blob)
                        if isinstance(gold_blob, (str, bytes, bytearray))
                        else gold_blob
                    )
                except (ValueError, TypeError):
                    events = []
                if not isinstance(events, list):
                    events = []

                # Recompute maxGold the way _build_networth_curve + _normalize do:
                # the networth of the latest event with 0 <= time < LANING_SECONDS.
                latest_t = -1
                latest_nw: float = 0.0
                for e in events:
                    t = e.get("time") or 0
                    if 0 <= t < 600:
                        nw = e.get("networth")
                        if nw is not None and t >= latest_t:
                            latest_t = t
                            latest_nw = float(nw)
                max_gold = int(latest_nw)

                if max_gold <= 0:
                    continue  # filtered out at cache-build time
                if cache_idx >= n_cached:
                    logger.warning(
                        "ensure_player_ids: more kept DB rows than cached samples "
                        "(stopped at %d). Cache appears stale.", cache_idx,
                    )
                    return False
                # Verify the row aligns with this cached sample.
                expected = int(cached_max_gold[cache_idx])
                if abs(max_gold - expected) > 1:
                    logger.warning(
                        "ensure_player_ids: maxGold mismatch at sample %d "
                        "(db=%d, cached=%d). Cache appears stale.",
                        cache_idx, max_gold, expected,
                    )
                    return False
                kept_ids.append(int(player_id))
                cache_idx += 1
        finally:
            conn.close()

        if cache_idx != n_cached:
            logger.warning(
                "ensure_player_ids: filled %d ids but cache has %d samples.",
                cache_idx, n_cached,
            )
            return False

        self._player_ids = kept_ids
        logger.info(
            "Backfilled %d player_ids from %s; rewriting cache.",
            n_cached, db_path,
        )
        cache_file = db_path.parent / (db_path.stem + ".features.npz")
        self._write_cache(cache_file, db_path)
        return True

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
                sc_src = player.get("scalars") or {}
                if not float(sc_src.get("maxGold") or 0):
                    continue
                ts, scalars = _parse_player_json(player)
                self._ts_list.append(ts)
                self._scalar_list.append(scalars)

    def _load_sqlite(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(players)")}
            if "positions" in cols and "gold_norm" not in cols:
                logger.info("Detected raw-events schema in %s; computing features on load.", db_path)
                self._load_sqlite_raw(conn)
            else:
                logger.info("Detected precomputed-features schema in %s.", db_path)
                self._load_sqlite_features(conn)
        finally:
            conn.close()

    def _load_sqlite_features(self, conn: sqlite3.Connection) -> None:
        ts_cols    = list(_TS_DB_COLS.keys())
        scalar_cols = list(_SCALAR_DB_COLS.keys())
        query = f"SELECT {', '.join(ts_cols + scalar_cols)} FROM players WHERE max_gold > 0"
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

    def _load_sqlite_raw(self, conn: sqlite3.Connection) -> None:
        """
        Load from the raw-events schema: each player row stores STRATZ event
        arrays as JSON blobs (positions, health, gold, xp, last_hits, damage,
        abilities, kills, deaths, assists, healing, tower_damage). For each
        match we reconstruct the playbackData layout that feature_builder
        expects, then run build_match() to produce the same feature dict the
        old extraction pipeline used to write to disk.
        """
        try:
            import orjson
            _loads = orjson.loads
        except ImportError:
            _loads = json.loads

        # Column name in players table → playbackData event-list key.
        ev_cols = {
            "positions":    "playerUpdatePositionEvents",
            "health":       "playerUpdateHealthEvents",
            "gold":         "playerUpdateGoldEvents",
            "xp":           "experienceEvents",
            "last_hits":    "csEvents",
            "damage":       "heroDamageEvents",
            "abilities":    "abilityUsedEvents",
            "kills":        "killEvents",
            "deaths":       "deathEvents",
            "assists":      "assistEvents",
            "healing":      "healEvents",
            "tower_damage": "towerDamageEvents",
        }
        player_cols = [
            "id",
            "match_id", "steam_account_id", "hero_id", "hero_name",
            "position", "lane", "team", "is_victory",
            *ev_cols.keys(),
        ]
        # Explicit secondary sort on `id` makes the iteration order
        # deterministic across SQLite versions/query plans — the cache and
        # any later ID-backfill scan see rows in the same sequence.
        query = (
            f"SELECT {', '.join(player_cols)} FROM players "
            f"ORDER BY match_id, id"
        )

        def _parse(blob) -> list:
            if not blob:
                return []
            try:
                v = _loads(blob) if isinstance(blob, (str, bytes, bytearray)) else blob
            except (ValueError, TypeError):
                return []
            return v if isinstance(v, list) else []

        # Stream rows, accumulate one match at a time, flush when match_id changes.
        current_match: int | None = None
        buffer: list[dict] = []

        def _flush(match_id: int, rows: list[dict]) -> None:
            if not rows:
                return
            players_raw = []
            for r in rows:
                pb = {ev_cols[c]: _parse(r[c]) for c in ev_cols}
                players_raw.append({
                    "steamAccountId": r["steam_account_id"],
                    "heroId":         r["hero_id"],
                    "position":       r["position"],
                    "lane":           r["lane"],
                    "isVictory":      bool(r["is_victory"]),
                    "isRadiant":      (r["team"] == "RADIANT"),
                    "playbackData":   pb,
                })

            match_node = {
                "id":              match_id,
                "didRadiantWin":   None,
                "durationSeconds": 0,
                "players":         players_raw,
            }
            # hero_map is irrelevant downstream — we only consume timeseries/scalars.
            hero_map: dict[int, str] = {}

            try:
                result = _build_match(match_node, hero_map)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping match %s: feature_builder error: %s", match_id, exc)
                return

            # feature_builder preserves input order, so the i-th output player
            # corresponds to the i-th row in `rows`. Pair them up so we can
            # record the source DB primary key alongside each kept sample.
            for input_row, player in zip(rows, result.get("players") or []):
                sc_src = player.get("scalars") or {}
                if not float(sc_src.get("maxGold") or 0):
                    continue
                ts, scalars = _parse_player_json(player)
                self._ts_list.append(ts)
                self._scalar_list.append(scalars)
                self._player_ids.append(int(input_row["id"]))

        n_matches_total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        cur = conn.execute(query)
        col_names = [d[0] for d in cur.description]
        import time as _time
        t_start = _time.time()
        t_last_log = t_start
        matches_done = 0
        for raw_row in cur:
            row = dict(zip(col_names, raw_row))
            mid = row["match_id"]
            if current_match is None:
                current_match = mid
            if mid != current_match:
                _flush(current_match, buffer)
                matches_done += 1
                now = _time.time()
                if now - t_last_log >= 5.0:
                    rate = matches_done / max(now - t_start, 1e-6)
                    remaining = max(n_matches_total - matches_done, 0) / max(rate, 1e-6)
                    logger.info(
                        "  feature-build progress: %d / %d matches (%.0f/s, ~%.0fs remaining)",
                        matches_done, n_matches_total, rate, remaining,
                    )
                    t_last_log = now
                buffer = []
                current_match = mid
            buffer.append(row)
        if current_match is not None:
            _flush(current_match, buffer)
            matches_done += 1
        logger.info(
            "  feature-build done: %d matches in %.1fs.",
            matches_done, _time.time() - t_start,
        )

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

    @property
    def player_ids(self) -> np.ndarray:
        """DB primary key of the source `players` row for each sample.

        Returns an int64 array of length `len(self)`. Empty if the source did
        not provide identifiers (e.g. JSON directory).
        """
        if len(self._player_ids) != len(self._ts_list):
            return np.zeros(0, dtype=np.int64)
        return np.asarray(self._player_ids, dtype=np.int64)

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
