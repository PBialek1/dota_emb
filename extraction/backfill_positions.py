"""backfill_positions.py — Re-extract matches whose player data is empty (max_gold = 0).

Players with max_gold = 0 have entirely zeroed-out feature data (scalars, timeseries,
position, lane all missing or zero). This script identifies the affected matches,
re-fetches them from STRATZ with the full playback query, reprocesses features via
build_match(), and UPDATEs the existing player rows and match lane-outcome fields in place.

Usage:
    python extraction/backfill_positions.py --db ./data/matches.db
    python extraction/backfill_positions.py --db ./data/matches.db --dry-run
    python extraction/backfill_positions.py --db ./data/matches.db --rate-limit 250
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from constants import LANING_SECONDS
from feature_builder import build_match
from stratz_client import StratzClient


class _TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
logger = logging.getLogger(__name__)


def _j(value) -> str:
    return json.dumps(value, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def get_target_match_ids(conn: sqlite3.Connection) -> list[int]:
    """Matches where any player has max_gold = 0 (fully empty feature data)."""
    rows = conn.execute(
        "SELECT DISTINCT match_id FROM players WHERE max_gold = 0"
    ).fetchall()
    ids = [r[0] for r in rows]
    logger.info("Found %d matches with max_gold = 0 players.", len(ids))
    return ids


# ---------------------------------------------------------------------------
# DB update
# ---------------------------------------------------------------------------

_UPDATE_MATCH = """
UPDATE matches
SET bottom_lane_outcome   = :bottom,
    mid_lane_outcome      = :mid,
    top_lane_outcome      = :top,
    lane_outcomes_fetched = 1
WHERE match_id = :match_id
"""

_UPDATE_PLAYER = """
UPDATE players
SET position              = :position,
    lane                  = :lane,
    max_gold              = :max_gold,
    max_xp                = :max_xp,
    max_damage_dealt      = :max_damage_dealt,
    max_damage_taken      = :max_damage_taken,
    max_cs                = :max_cs,
    max_tower_damage      = :max_tower_damage,
    max_healing           = :max_healing,
    gold_norm             = :gold_norm,
    xp_norm               = :xp_norm,
    damage_dealt_norm     = :damage_dealt_norm,
    damage_taken_norm     = :damage_taken_norm,
    cs_norm               = :cs_norm,
    tower_damage_norm     = :tower_damage_norm,
    healing_norm          = :healing_norm,
    health_pct            = :health_pct,
    mana_pct              = :mana_pct,
    dist_to_nearest_ally  = :dist_to_nearest_ally,
    dist_to_nearest_enemy = :dist_to_nearest_enemy,
    dist_to_nearest_tower = :dist_to_nearest_tower,
    kills                 = :kills,
    deaths                = :deaths,
    assists               = :assists,
    ability_casts         = :ability_casts,
    allies_nearby         = :allies_nearby,
    enemies_nearby        = :enemies_nearby
WHERE match_id = :match_id
  AND steam_account_id = :steam_account_id
"""


def apply_updates(conn: sqlite3.Connection, match_id: int, result: dict, dry_run: bool) -> int:
    """Write rebuilt features to existing match + player rows. Returns number of players updated."""
    meta = result["meta"]

    if not dry_run:
        conn.execute(_UPDATE_MATCH, {
            "match_id": match_id,
            "bottom":   meta.get("bottomLaneOutcome"),
            "mid":      meta.get("midLaneOutcome"),
            "top":      meta.get("topLaneOutcome"),
        })

    players_updated = 0
    for p in result["players"]:
        sc = p["scalars"]
        ts = p["timeseries"]
        ev = p["events"]

        row = {
            "match_id":              match_id,
            "steam_account_id":      p.get("steamAccountId"),
            "position":              p.get("position"),
            "lane":                  p.get("lane"),
            "max_gold":              sc.get("maxGold"),
            "max_xp":                sc.get("maxXp"),
            "max_damage_dealt":      sc.get("maxDamageDealt"),
            "max_damage_taken":      sc.get("maxDamageTaken"),
            "max_cs":                sc.get("maxCs"),
            "max_tower_damage":      sc.get("maxTowerDamage"),
            "max_healing":           sc.get("maxHealing"),
            "gold_norm":             _j(ts.get("goldNorm", [])),
            "xp_norm":               _j(ts.get("xpNorm", [])),
            "damage_dealt_norm":     _j(ts.get("damageDealtNorm", [])),
            "damage_taken_norm":     _j(ts.get("damageTakenNorm", [])),
            "cs_norm":               _j(ts.get("csNorm", [])),
            "tower_damage_norm":     _j(ts.get("towerDamageNorm", [])),
            "healing_norm":          _j(ts.get("healingNorm", [])),
            "health_pct":            _j(ts.get("healthPct", [])),
            "mana_pct":              _j(ts.get("manaPct", [])),
            "dist_to_nearest_ally":  _j(ts.get("distToNearestAlly", [])),
            "dist_to_nearest_enemy": _j(ts.get("distToNearestEnemy", [])),
            "dist_to_nearest_tower": _j(ts.get("distToNearestTower", [])),
            "kills":                 _j(ev.get("kills", [])),
            "deaths":                _j(ev.get("deaths", [])),
            "assists":               _j(ev.get("assists", [])),
            "ability_casts":         _j(ev.get("abilityCasts", [])),
            "allies_nearby":         _j(p.get("alliesNearby", [])),
            "enemies_nearby":        _j(p.get("enemiesNearby", [])),
        }

        if not dry_run:
            cur = conn.execute(_UPDATE_PLAYER, row)
            players_updated += cur.rowcount
        else:
            players_updated += 1

    return players_updated


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def backfill(
    db_path: Path,
    client: StratzClient,
    hero_map: dict[int, str],
    dry_run: bool,
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    match_ids = get_target_match_ids(conn)
    total = len(match_ids)

    if total == 0:
        logger.info("Nothing to backfill — no players with max_gold = 0.")
        conn.close()
        return

    success = failed = skipped = total_players = 0

    with tqdm(total=total, desc="Backfill", unit="match") as pbar:
        for match_id in match_ids:
            try:
                match_node = client.fetch_match(match_id)
            except RuntimeError as exc:
                logger.warning("Match %d fetch failed: %s", match_id, exc)
                failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=success, failed=failed, skipped=skipped)
                continue

            duration = match_node.get("durationSeconds") or 0
            if duration < LANING_SECONDS:
                logger.info("Match %d skipped: duration %ds < 600s", match_id, duration)
                skipped += 1
                pbar.update(1)
                pbar.set_postfix(ok=success, failed=failed, skipped=skipped)
                continue

            try:
                result = build_match(match_node, hero_map)
            except Exception as exc:
                logger.warning("Match %d build_match failed: %s", match_id, exc)
                failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=success, failed=failed, skipped=skipped)
                continue

            n = apply_updates(conn, match_id, result, dry_run)
            if not dry_run:
                conn.commit()

            total_players += n
            success += 1
            pbar.update(1)
            pbar.set_postfix(ok=success, failed=failed, skipped=skipped, players=total_players)

    conn.close()

    logger.info(
        "Done. matches ok=%d failed=%d skipped=%d  players updated=%d%s",
        success, failed, skipped, total_players,
        " (DRY RUN)" if dry_run else "",
    )

    if not dry_run:
        conn2 = sqlite3.connect(str(db_path))
        remaining = conn2.execute(
            "SELECT COUNT(*) FROM players WHERE max_gold = 0"
        ).fetchone()[0]
        conn2.close()
        logger.info("Remaining players with max_gold = 0: %d", remaining)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-extract matches with empty player data (max_gold = 0)."
    )
    parser.add_argument("--db",         type=Path, required=True, help="SQLite database path.")
    parser.add_argument("--rate-limit", type=int,  default=280,   help="STRATZ requests/hour.")
    parser.add_argument("--dry-run",    action="store_true",       help="Fetch but do not write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    token = os.environ.get("STRATZ_TOKEN")
    if not token:
        logger.error("STRATZ_TOKEN environment variable not set.")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — no DB writes will occur.")

    client   = StratzClient(token=token, rate_per_hour=args.rate_limit)
    hero_map = client.fetch_hero_map()

    backfill(args.db, client, hero_map, args.dry_run)


if __name__ == "__main__":
    main()
