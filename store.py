"""store.py — Storage backends for extracted match data.

Two backends are available:

    JsonStore   — writes one JSON file per match (original behaviour).
    SqliteStore — writes match + player rows into a SQLite3 database.

Both expose the same interface:
    store.exists(match_id) -> bool
    store.save(match_id, result_dict) -> None
    store.close() -> None          # no-op for JsonStore
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MatchStore(ABC):
    @abstractmethod
    def exists(self, match_id: int) -> bool: ...

    @abstractmethod
    def save(self, match_id: int, result: dict) -> None: ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# JSON backend (original behaviour)
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
    match_id          INTEGER PRIMARY KEY,
    duration_seconds  INTEGER,
    did_radiant_win   INTEGER,
    avg_mmr           INTEGER,
    bracket           INTEGER,
    game_version_id   INTEGER
);
"""

_CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id              INTEGER NOT NULL REFERENCES matches(match_id),
    steam_account_id      INTEGER,
    hero_id               INTEGER,
    hero_name             TEXT,
    position              TEXT,
    lane                  TEXT,
    team                  TEXT,
    is_victory            INTEGER,
    -- scalars
    max_gold              INTEGER,
    max_xp                INTEGER,
    max_damage_dealt      INTEGER,
    max_damage_taken      INTEGER,
    max_cs                INTEGER,
    max_tower_damage      INTEGER,
    max_healing           INTEGER,
    -- timeseries (JSON arrays, 10 elements each)
    gold_norm             TEXT,
    xp_norm               TEXT,
    damage_dealt_norm     TEXT,
    damage_taken_norm     TEXT,
    cs_norm               TEXT,
    tower_damage_norm     TEXT,
    healing_norm          TEXT,
    health_pct            TEXT,
    mana_pct              TEXT,
    dist_to_nearest_ally  TEXT,
    dist_to_nearest_enemy TEXT,
    dist_to_nearest_tower TEXT,
    -- per-minute event counts (JSON arrays, 10 elements each)
    kills                 TEXT,
    deaths                TEXT,
    assists               TEXT,
    ability_casts         TEXT,
    -- proximity counts (JSON arrays, 10 elements each)
    allies_nearby         TEXT,
    enemies_nearby        TEXT
);
"""

_INSERT_MATCH = """
INSERT OR IGNORE INTO matches
    (match_id, duration_seconds, did_radiant_win, avg_mmr, bracket, game_version_id)
VALUES
    (:match_id, :duration_seconds, :did_radiant_win, :avg_mmr, :bracket, :game_version_id);
"""

_INSERT_PLAYER = """
INSERT INTO players (
    match_id, steam_account_id, hero_id, hero_name,
    position, lane, team, is_victory,
    max_gold, max_xp, max_damage_dealt, max_damage_taken,
    max_cs, max_tower_damage, max_healing,
    gold_norm, xp_norm, damage_dealt_norm, damage_taken_norm,
    cs_norm, tower_damage_norm, healing_norm,
    health_pct, mana_pct,
    dist_to_nearest_ally, dist_to_nearest_enemy, dist_to_nearest_tower,
    kills, deaths, assists, ability_casts,
    allies_nearby, enemies_nearby
) VALUES (
    :match_id, :steam_account_id, :hero_id, :hero_name,
    :position, :lane, :team, :is_victory,
    :max_gold, :max_xp, :max_damage_dealt, :max_damage_taken,
    :max_cs, :max_tower_damage, :max_healing,
    :gold_norm, :xp_norm, :damage_dealt_norm, :damage_taken_norm,
    :cs_norm, :tower_damage_norm, :healing_norm,
    :health_pct, :mana_pct,
    :dist_to_nearest_ally, :dist_to_nearest_enemy, :dist_to_nearest_tower,
    :kills, :deaths, :assists, :ability_casts,
    :allies_nearby, :enemies_nearby
);
"""


def _j(value) -> str:
    """Serialize a list to a compact JSON string."""
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
            "match_id":         meta["matchId"],
            "duration_seconds": meta["durationSeconds"],
            "did_radiant_win":  int(bool(meta["didRadiantWin"])),
            "avg_mmr":          meta.get("avgMmr"),
            "bracket":          meta.get("bracket"),
            "game_version_id":  meta.get("gameVersionId"),
        }

        player_rows = []
        for p in result["players"]:
            sc = p["scalars"]
            ts = p["timeseries"]
            ev = p["events"]
            player_rows.append({
                "match_id":              match_id,
                "steam_account_id":      p.get("steamAccountId"),
                "hero_id":               p.get("heroId"),
                "hero_name":             p.get("heroName", ""),
                "position":              p.get("position"),
                "lane":                  p.get("lane"),
                "team":                  p.get("team"),
                "is_victory":            int(bool(p.get("isVictory"))),
                "max_gold":              sc.get("maxGold"),
                "max_xp":               sc.get("maxXp"),
                "max_damage_dealt":     sc.get("maxDamageDealt"),
                "max_damage_taken":     sc.get("maxDamageTaken"),
                "max_cs":               sc.get("maxCs"),
                "max_tower_damage":     sc.get("maxTowerDamage"),
                "max_healing":          sc.get("maxHealing"),
                "gold_norm":            _j(ts.get("goldNorm", [])),
                "xp_norm":              _j(ts.get("xpNorm", [])),
                "damage_dealt_norm":    _j(ts.get("damageDealtNorm", [])),
                "damage_taken_norm":    _j(ts.get("damageTakenNorm", [])),
                "cs_norm":              _j(ts.get("csNorm", [])),
                "tower_damage_norm":    _j(ts.get("towerDamageNorm", [])),
                "healing_norm":         _j(ts.get("healingNorm", [])),
                "health_pct":           _j(ts.get("healthPct", [])),
                "mana_pct":             _j(ts.get("manaPct", [])),
                "dist_to_nearest_ally": _j(ts.get("distToNearestAlly", [])),
                "dist_to_nearest_enemy":_j(ts.get("distToNearestEnemy", [])),
                "dist_to_nearest_tower":_j(ts.get("distToNearestTower", [])),
                "kills":                _j(ev.get("kills", [])),
                "deaths":               _j(ev.get("deaths", [])),
                "assists":              _j(ev.get("assists", [])),
                "ability_casts":        _j(ev.get("abilityCasts", [])),
                "allies_nearby":        _j(p.get("alliesNearby", [])),
                "enemies_nearby":       _j(p.get("enemiesNearby", [])),
            })

        with self._conn:
            self._conn.execute(_INSERT_MATCH, match_row)
            self._conn.executemany(_INSERT_PLAYER, player_rows)

    def close(self) -> None:
        self._conn.close()