"""
StratzMatchLaningDataFetcher.py

Fetches laning phase data for a Dota 2 match from the STRATZ GraphQL API.

Data retrieved per player (first 10 minutes / 600 seconds):
  - XY position every second
  - Cumulative hero damage dealt every second
  - Cumulative last hits every second
  - Cumulative denies every second
  - Current and max HP every second
  - Current and max mana every second
  - Cumulative gold every second
  - Cumulative experience every second
  - Hero ID, role, lane, team, and Steam account ID

Usage:
    from StratzMatchLaningDataFetcher import get_match_laning_data
    data = get_match_laning_data(8138688052)

Output:
    Returns a dict and writes/reads a cache file: cache/match_<MATCH_ID>_laning.json
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- транспортировка
# Constants
# ---------------------------------------------------------------------------

STRATZ_GRAPHQL_URL = "https://api.stratz.com/graphql"
LANING_PHASE_SECONDS = 600  # First 10 minutes
CACHE_DIR = "cache"

# ---------------------------------------------------------------------------
# GraphQL query
# ---------------------------------------------------------------------------

MATCH_QUERY = """
query GetMatchLaningData($matchId: Long!) {
  match(id: $matchId) {
    id
    didRadiantWin
    players {
      steamAccountId
      heroId
      role
      lane
      isRadiant
      stats {
        lastHitsPerMinute
        deniesPerMinute
        goldPerMinute
        experiencePerMinute
      }
      playbackData {
        playerUpdatePositionEvents {
          time
          x
          y
        }
        playerUpdateHealthEvents {
          time
          hp
          maxHp
          mp
          maxMp
        }
        heroDamageReceivedEvents: heroDamageEvents {
          time
          value
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_match(match_id: int, token: str) -> dict:
    """Send the GraphQL query to STRATZ and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "stratz-match-fetcher/1.0",
    }
    payload = {
        "query": MATCH_QUERY,
        "variables": {"matchId": match_id},
    }

    response = requests.post(STRATZ_GRAPHQL_URL, json=payload, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} from STRATZ API: {response.text[:500]}"
        )

    data = response.json()

    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")

    return data


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def accumulate_per_second(values_per_minute: list[int], max_seconds: int) -> list[int]:
    """
    Convert a per-minute list (one entry per minute) into a cumulative
    per-second list up to max_seconds.

    STRATZ returns lastHitsPerMinute / deniesPerMinute / goldPerMinute /
    experiencePerMinute as a list where index i holds the count for minute i
    (0-indexed). We spread each minute's count evenly across its 60 seconds
    to produce a per-second cumulative series.
    """
    if not values_per_minute:
        return [0] * max_seconds

    cumulative = []
    running_total = 0

    for second in range(max_seconds):
        minute_index = second // 60
        second_in_minute = second % 60

        if minute_index < len(values_per_minute):
            minute_count = values_per_minute[minute_index]
            if second_in_minute == 59:
                already_added = (minute_count * second_in_minute) // 60
                running_total += minute_count - already_added
            else:
                running_total += (minute_count * (second_in_minute + 1)) // 60 - (
                    minute_count * second_in_minute
                ) // 60

        cumulative.append(running_total)

    return cumulative


def build_positions_per_second(
    position_values: list[dict], max_seconds: int
) -> list[dict | None]:
    """
    Build a list of {x, y} dicts indexed by second (0 … max_seconds-1).
    Gaps are forward-filled from the last known position.
    """
    by_time: dict[int, dict] = {}
    for entry in position_values or []:
        t = entry.get("time")
        if t is not None and 0 <= t < max_seconds:
            by_time[t] = {"x": entry.get("x"), "y": entry.get("y")}

    result: list[dict | None] = []
    last_known: dict | None = None

    for second in range(max_seconds):
        if second in by_time:
            last_known = by_time[second]
        result.append(last_known)

    return result


def build_cumulative_damage_per_second(
    damage_events: list[dict], max_seconds: int
) -> list[int]:
    """
    Build a cumulative hero-damage list indexed by second.
    STRATZ heroDamageEvents supply a `value` field per event;
    we sum them up per second and forward-fill cumulatively.
    """
    by_time: dict[int, int] = {}
    for event in damage_events or []:
        t = event.get("time")
        v = event.get("value")
        if t is not None and v is not None and 0 <= t < max_seconds:
            by_time[t] = by_time.get(t, 0) + v

    result: list[int] = []
    last_value = 0

    for second in range(max_seconds):
        if second in by_time:
            last_value += by_time[second]
        result.append(last_value)

    return result


def build_health_mana_per_second(
    health_events: list[dict], max_seconds: int
) -> tuple[list[int | None], list[int | None], list[int | None], list[int | None]]:
    """
    Build per-second lists for hp, maxHp, mp, and maxMp from
    playerUpdateHealthEvents. Values are forward-filled from the last
    known snapshot; seconds before the first event are None.

    Note: STRATZ API uses 'mp' and 'maxMp' for mana fields
    on PlayerUpdateHealthDetailType.
    """
    by_time: dict[int, dict] = {}
    for event in health_events or []:
        t = event.get("time")
        if t is not None and 0 <= t < max_seconds:
            by_time[t] = {
                "hp": event.get("hp"),
                "maxHp": event.get("maxHp"),
                "mp": event.get("mp"),
                "maxMp": event.get("maxMp"),
            }

    hp_list: list[int | None] = []
    max_hp_list: list[int | None] = []
    mana_list: list[int | None] = []
    max_mana_list: list[int | None] = []
    last: dict | None = None

    for second in range(max_seconds):
        if second in by_time:
            last = by_time[second]
        hp_list.append(last["hp"] if last else None)
        max_hp_list.append(last["maxHp"] if last else None)
        mana_list.append(last["mp"] if last else None)
        max_mana_list.append(last["maxMp"] if last else None)

    return hp_list, max_hp_list, mana_list, max_mana_list


def process_player(player: dict, max_seconds: int) -> dict:
    """Transform a raw STRATZ player node into a clean structured dict."""
    playback = player.get("playbackData") or {}
    stats = player.get("stats") or {}

    positions = build_positions_per_second(
        playback.get("playerUpdatePositionEvents") or [], max_seconds
    )
    damage = build_cumulative_damage_per_second(
        playback.get("heroDamageReceivedEvents") or [], max_seconds
    )
    last_hits = accumulate_per_second(
        stats.get("lastHitsPerMinute") or [], max_seconds
    )
    denies = accumulate_per_second(
        stats.get("deniesPerMinute") or [], max_seconds
    )
    gold = accumulate_per_second(
        stats.get("goldPerMinute") or [], max_seconds
    )
    xp = accumulate_per_second(
        stats.get("experiencePerMinute") or [], max_seconds
    )
    hp, max_hp, mana, max_mana = build_health_mana_per_second(
        playback.get("playerUpdateHealthEvents") or [], max_seconds
    )

    return {
        "steamAccountId": player.get("steamAccountId"),
        "heroId": player.get("heroId"),
        "role": player.get("role"),
        "lane": player.get("lane"),
        "isRadiant": player.get("isRadiant"),
        # Per-second arrays, index = game second (0-based)
        "positions": positions,               # list of {x, y} or None
        "cumulativeHeroDamage": damage,       # list of int
        "cumulativeLastHits": last_hits,      # list of int
        "cumulativeDenies": denies,           # list of int
        "cumulativeGold": gold,               # list of int
        "cumulativeXp": xp,                   # list of int
        "hp": hp,                             # list of int or None
        "maxHp": max_hp,                      # list of int or None
        "mana": mana,                         # list of int or None
        "maxMana": max_mana,                  # list of int or None
    }


def process_match(raw: dict, max_seconds: int) -> dict:
    """Extract and process all match data from the raw API response."""
    match_node = raw.get("data", {}).get("match")
    if match_node is None:
        raise ValueError("No match data in API response. Check the match ID and token.")

    players = [
        process_player(p, max_seconds)
        for p in (match_node.get("players") or [])
    ]

    return {
        "matchId": match_node.get("id"),
        "didRadiantWin": match_node.get("didRadiantWin"),
        "laningPhaseSeconds": max_seconds,
        "players": players,
    }


# ---------------------------------------------------------------------------
def _cache_path(match_id: int) -> str:
    """Return the local cache file path for a given match ID."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"match_{match_id}_laning.json")


def get_match_laning_data(match_id: int) -> dict:
    """
    Return laning-phase data for the given match ID.

    Loads from local cache if available, otherwise fetches from the
    STRATZ API and saves the result to cache for future calls.

    Args:
        match_id: The Dota 2 match ID to retrieve.

    Returns:
        A dict containing matchId, didRadiantWin, laningPhaseSeconds,
        and a list of per-player arrays.
    """
    path = _cache_path(match_id)

    if os.path.exists(path):
        print(f"Loading match {match_id} from cache ({path}) …")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    token = os.environ.get("STRATZ_TOKEN", "")
    if not token:
        print(
            "ERROR: STRATZ_TOKEN not found. "
            "Please add it to your .env file as: STRATZ_TOKEN=your_token_here",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Querying STRATZ API for match {match_id} …")
    raw_response = fetch_match(match_id, token)

    print("Processing data …")
    result = process_match(raw_response, LANING_PHASE_SECONDS)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    player_count = len(result["players"])
    print(
        f"Done. {player_count} players saved to '{path}'.\n"
        f"Each player has {LANING_PHASE_SECONDS} seconds of laning-phase data."
    )

    return result


def main():
    match_id = 8716260381
    result = get_match_laning_data(match_id)
    print('done')


if __name__ == "__main__":
    main()