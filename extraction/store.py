"""store.py — Storage backends for raw-event match data.

Two backends:
    JsonStore   — writes one JSON file per match.
    SqliteStore — writes match + player rows into a SQLite3 database.

Interface:
    store.exists(match_id) -> bool
    store.save(match_id, result_dict) -> None
    store.close() -> None
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class MatchStore(ABC):
    @abstractmethod
    def exists(self, match_id: int) -> bool: ...

    @abstractmethod
    def save(self, match_id: int, result: dict) -> None: ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# JSON backend
# ---------------------------------------------------------------------------

class JsonStore(MatchStore):
    def __init__(self, output_dir: Path):
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, match_id: int) -> Path:
        return self._dir / f"{match_id}.json"

    def exists(self, match_id: int) -> bool:
        return self._path(match_id).exists()

    def save(self, match_id: int, result: dict) -> None:
        try:
            import orjson
            self._path(match_id).write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
        except ImportError:
            self._path(match_id).write_text(json.dumps(result, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_CREATE_MATCHES = """
CREATE TABLE IF NOT EXISTS matches (
    match_id            INTEGER PRIMARY KEY,
    duration_seconds    INTEGER,
    did_radiant_win     INTEGER,
    avg_mmr             INTEGER,
    bracket             TEXT,
    game_version_id     INTEGER,
    bottom_lane_outcome TEXT,
    mid_lane_outcome    TEXT,
    top_lane_outcome    TEXT
);
"""

_CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id         INTEGER NOT NULL REFERENCES matches(match_id),
    steam_account_id INTEGER,
    hero_id          INTEGER,
    hero_name        TEXT,
    position         TEXT,
    lane             TEXT,
    team             TEXT,
    is_victory       INTEGER,
    -- raw event sequences: JSON arrays, each element has "time" + event-specific fields
    positions        TEXT,  -- [{time, x, y}, ...]
    health           TEXT,  -- [{time, hp, maxHp, mp, maxMp}, ...]
    gold             TEXT,  -- [{time, gold, networth}, ...]
    xp               TEXT,  -- [{time, amount}, ...]
    last_hits        TEXT,  -- [{time, npcId, isCreep, isNeutral, isAncient}, ...]
    damage           TEXT,  -- [{time, attacker, target, value, ...}, ...]  damage this hero dealt
    abilities        TEXT,  -- [{time, abilityId, attacker, target}, ...]
    kills            TEXT,  -- [{time, attacker, target, gold, xp, ...}, ...]
    deaths           TEXT,  -- [{time, attacker, goldFed, timeDead, ...}, ...]
    assists          TEXT,  -- [{time, attacker, target, gold, xp, ...}, ...]
    healing          TEXT,  -- [{time, attacker, target, value, ...}, ...]
    tower_damage     TEXT,  -- [{time, npcId, damage, ...}, ...]
    items            TEXT   -- [{time, itemId}, ...]
);
"""

_INSERT_MATCH = """
INSERT OR IGNORE INTO matches
    (match_id, duration_seconds, did_radiant_win, avg_mmr, bracket, game_version_id,
     bottom_lane_outcome, mid_lane_outcome, top_lane_outcome)
VALUES
    (:match_id, :duration_seconds, :did_radiant_win, :avg_mmr, :bracket, :game_version_id,
     :bottom_lane_outcome, :mid_lane_outcome, :top_lane_outcome);
"""

_INSERT_PLAYER = """
INSERT INTO players (
    match_id, steam_account_id, hero_id, hero_name,
    position, lane, team, is_victory,
    positions, health, gold, xp, last_hits,
    damage, abilities, kills, deaths, assists, healing, tower_damage, items
) VALUES (
    :match_id, :steam_account_id, :hero_id, :hero_name,
    :position, :lane, :team, :is_victory,
    :positions, :health, :gold, :xp, :last_hits,
    :damage, :abilities, :kills, :deaths, :assists, :healing, :tower_damage, :items
);
"""


def _j(value) -> str:
    return json.dumps(value, separators=(",", ":"))


class SqliteStore(MatchStore):
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_CREATE_MATCHES + _CREATE_PLAYERS)
        self._conn.commit()

    def exists(self, match_id: int) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM matches WHERE match_id = ?", (match_id,)
        )
        return cur.fetchone() is not None

    def save(self, match_id: int, result: dict) -> None:
        meta = result["meta"]

        match_row = {
            "match_id":             meta["matchId"],
            "duration_seconds":     meta["durationSeconds"],
            "did_radiant_win":      int(bool(meta["didRadiantWin"])),
            "avg_mmr":              meta.get("avgMmr"),
            "bracket":              meta.get("bracket"),
            "game_version_id":      meta.get("gameVersionId"),
            "bottom_lane_outcome":  meta.get("bottomLaneOutcome"),
            "mid_lane_outcome":     meta.get("midLaneOutcome"),
            "top_lane_outcome":     meta.get("topLaneOutcome"),
        }

        player_rows = []
        for p in result["players"]:
            player_rows.append({
                "match_id":        match_id,
                "steam_account_id": p.get("steamAccountId"),
                "hero_id":         p.get("heroId"),
                "hero_name":       p.get("heroName", ""),
                "position":        p.get("position"),
                "lane":            p.get("lane"),
                "team":            p.get("team"),
                "is_victory":      int(bool(p.get("isVictory"))),
                "positions":       _j(p.get("positions", [])),
                "health":          _j(p.get("health", [])),
                "gold":            _j(p.get("gold", [])),
                "xp":              _j(p.get("xp", [])),
                "last_hits":       _j(p.get("lastHits", [])),
                "damage":        _j(p.get("damage", [])),
                "abilities":     _j(p.get("abilities", [])),
                "kills":         _j(p.get("kills", [])),
                "deaths":        _j(p.get("deaths", [])),
                "assists":       _j(p.get("assists", [])),
                "healing":       _j(p.get("healing", [])),
                "tower_damage":  _j(p.get("towerDamage", [])),
                "items":         _j(p.get("items", [])),
            })

        with self._conn:
            self._conn.execute(_INSERT_MATCH, match_row)
            self._conn.executemany(_INSERT_PLAYER, player_rows)

    def close(self) -> None:
        self._conn.close()
