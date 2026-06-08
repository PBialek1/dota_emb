"""feature_builder.py — Transforms a raw STRATZ match node into the output schema."""

import logging
from bisect import bisect_left

from constants import (
    LANING_INTERVALS,
    LANING_INTERVAL_SECONDS,
    LANING_SECONDS,
    ALLY_SORT_BUCKET,
    TOWER_POSITIONS,
    euclidean,
)

# Fixed tower ordering used for all players (indices 0–5 are stable across time)
ALL_TOWERS = [
    TOWER_POSITIONS["radiant_top_t1"],
    TOWER_POSITIONS["radiant_mid_t1"],
    TOWER_POSITIONS["radiant_bot_t1"],
    TOWER_POSITIONS["dire_top_t1"],
    TOWER_POSITIONS["dire_mid_t1"],
    TOWER_POSITIONS["dire_bot_t1"],
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket(t: int) -> int | None:
    """Return 15-second interval index (0–39) for a timestamp in seconds, or None if out of range."""
    if 0 <= t < LANING_SECONDS:
        return t // LANING_INTERVAL_SECONDS
    return None


def _closest_event(events: list[dict], target_time: float, key: str = "time") -> dict | None:
    """Return the event whose `key` field is closest to target_time, or None.

    For repeated lookups on the same (sorted) event list, prefer
    `_closest_event_indexer(events)` which extracts the key array once and
    uses bisect for O(log N) lookups instead of an O(N) min scan.
    """
    if not events:
        return None
    return min(events, key=lambda e: abs((e.get(key) or 0) - target_time))


def _closest_event_indexer(events: list[dict], key: str = "time"):
    """
    Build a fast closest-event lookup over a list of events whose `key` field
    is already monotonically non-decreasing (as guaranteed by `_laning_events`).

    Returns a function `lookup(target_time) -> event | None` that uses bisect
    for O(log N) lookup.
    """
    if not events:
        return lambda _t: None
    times = [(e.get(key) or 0) for e in events]
    n = len(times)

    def lookup(target_time):
        i = bisect_left(times, target_time)
        if i == 0:
            return events[0]
        if i == n:
            return events[n - 1]
        before, after = events[i - 1], events[i]
        if abs(times[i - 1] - target_time) <= abs(times[i] - target_time):
            return before
        return after

    return lookup


def _laning_events(events: list[dict] | None) -> list[dict]:
    """Filter events to time < 600 and sort by time."""
    if not events:
        return []
    result = [e for e in events if (e.get("time") or 0) < LANING_SECONDS]
    result.sort(key=lambda e: e.get("time") or 0)
    return result


def _normalize(curve: list[float]) -> tuple[list[float], float]:
    """
    Normalize a cumulative curve by its last value.
    Returns (norm_curve, max_value).
    """
    max_val = curve[-1] if curve else 0.0
    if max_val == 0:
        return [0.0] * len(curve), 0.0
    return [v / max_val for v in curve], float(max_val)


# ---------------------------------------------------------------------------
# Per-minute cumulative builders
# ---------------------------------------------------------------------------

def _build_networth_curve(gold_events: list[dict]) -> list[float]:
    """
    Gold: take the last `networth` value within each minute bucket.
    This is a state snapshot — forward-fill missing minutes.
    """
    events = _laning_events(gold_events)
    buckets: list[float | None] = [None] * LANING_INTERVALS

    for e in events:
        b = _bucket(e.get("time") or 0)
        if b is not None:
            nw = e.get("networth")
            if nw is not None:
                buckets[b] = float(nw)

    # Forward-fill
    last = 0.0
    for i in range(LANING_INTERVALS):
        if buckets[i] is not None:
            last = buckets[i]
        else:
            buckets[i] = last

    return [b for b in buckets]


def _build_cumsum_curve(events: list[dict], value_key: str) -> list[float]:
    """
    Generic cumulative sum curve: sum `value_key` per minute → running cumulative.
    Used for XP, damage dealt, damage taken, tower damage, healing.
    """
    filtered = _laning_events(events)
    per_interval = [0.0] * LANING_INTERVALS

    for e in filtered:
        b = _bucket(e.get("time") or 0)
        if b is not None:
            per_interval[b] += float(e.get(value_key) or 0)

    # Running cumulative sum
    cumsum = []
    total = 0.0
    for v in per_interval:
        total += v
        cumsum.append(total)
    return cumsum


def _build_cs_curve(cs_events: list[dict]) -> list[float]:
    """CS: count events per minute → running cumulative."""
    filtered = _laning_events(cs_events)
    per_interval = [0] * LANING_INTERVALS

    for e in filtered:
        b = _bucket(e.get("time") or 0)
        if b is not None:
            per_interval[b] += 1

    cumsum = []
    total = 0
    for v in per_interval:
        total += v
        cumsum.append(float(total))
    return cumsum


# ---------------------------------------------------------------------------
# State curves (health / mana)
# ---------------------------------------------------------------------------

def _build_state_curves(health_events: list[dict]) -> tuple[list[float], list[float]]:
    """
    healthPct and manaPct: for each minute N, find the event closest to N*60+30.
    Forward-fill if no event in a bucket.
    Returns (healthPct[10], manaPct[10]).
    """
    events = _laning_events(health_events)
    lookup = _closest_event_indexer(events)
    health_pct: list[float] = []
    mana_pct:   list[float] = []

    last_hp, last_mp = 1.0, 1.0

    for n in range(LANING_INTERVALS):
        midpoint = n * LANING_INTERVAL_SECONDS + LANING_INTERVAL_SECONDS // 2
        e = lookup(midpoint)
        if e and e.get("maxHp") and e["maxHp"] > 0:
            last_hp = e["hp"] / e["maxHp"]
        if e and e.get("maxMp") and e["maxMp"] > 0:
            last_mp = e["mp"] / e["maxMp"]
        health_pct.append(round(last_hp, 4))
        mana_pct.append(round(last_mp, 4))

    return health_pct, mana_pct


# ---------------------------------------------------------------------------
# Position snapshot (one (x, y) per minute)
# ---------------------------------------------------------------------------

def _position_snapshots(pos_events: list[dict]) -> list[tuple[float, float] | None]:
    """
    For each minute N find the position event closest to N*60+30.
    Forward-fill dead time with last known position.
    """
    events = _laning_events(pos_events)
    lookup = _closest_event_indexer(events)
    snapshots: list[tuple[float, float] | None] = []
    last_pos: tuple[float, float] | None = None

    for n in range(LANING_INTERVALS):
        midpoint = n * LANING_INTERVAL_SECONDS + LANING_INTERVAL_SECONDS // 2
        e = lookup(midpoint)
        if e and e.get("x") is not None and e.get("y") is not None:
            last_pos = (float(e["x"]), float(e["y"]))
        snapshots.append(last_pos)

    return snapshots


# ---------------------------------------------------------------------------
# Event counters (kills, deaths, assists, ability casts)
# ---------------------------------------------------------------------------

def _count_per_minute(events: list[dict]) -> list[int]:
    """Count events per minute bucket [N*60, (N+1)*60)."""
    filtered = _laning_events(events)
    counts = [0] * LANING_INTERVALS
    for e in filtered:
        b = _bucket(e.get("time") or 0)
        if b is not None:
            counts[b] += 1
    return counts




# ---------------------------------------------------------------------------
# Cross-player positional features
# ---------------------------------------------------------------------------

def _compute_positional_features(
    all_snapshots: list[list[tuple[float, float] | None]],  # [player_idx][bucket]
    radiant_flags: list[bool],
) -> tuple[
    list[list[list[float]]],  # dist_allies  [player][rank 0–3][bucket]
    list[list[list[float]]],  # dist_enemies [player][rank 0–4][bucket]
    list[list[list[float]]],  # dist_towers  [player][tower 0–5][bucket]
]:
    n_players = len(all_snapshots)
    N_ALLIES  = 4
    N_ENEMIES = 5
    N_TOWERS  = len(ALL_TOWERS)  # 6

    # Establish ally/enemy ordering at ALLY_SORT_BUCKET (t=90s).
    # Players with no position at that bucket fall back to last known snapshot
    # (forward-fill is already applied by _position_snapshots), then to inf distance.
    sort_pos = [all_snapshots[p][ALLY_SORT_BUCKET] for p in range(n_players)]

    ally_orders:  list[list[int]] = []
    enemy_orders: list[list[int]] = []
    for p in range(n_players):
        p90 = sort_pos[p]
        allies  = [q for q in range(n_players) if q != p and radiant_flags[q] == radiant_flags[p]]
        enemies = [q for q in range(n_players) if radiant_flags[q] != radiant_flags[p]]
        if p90 is not None:
            allies  = sorted(allies,  key=lambda q, ref=p90: euclidean(ref, sort_pos[q]) if sort_pos[q] else float("inf"))
            enemies = sorted(enemies, key=lambda q, ref=p90: euclidean(ref, sort_pos[q]) if sort_pos[q] else float("inf"))
        ally_orders.append(allies)
        enemy_orders.append(enemies)

    dist_allies  = [[[0.0] * LANING_INTERVALS for _ in range(N_ALLIES)]  for _ in range(n_players)]
    dist_enemies = [[[0.0] * LANING_INTERVALS for _ in range(N_ENEMIES)] for _ in range(n_players)]
    dist_towers  = [[[0.0] * LANING_INTERVALS for _ in range(N_TOWERS)]  for _ in range(n_players)]

    for step in range(LANING_INTERVALS):
        for p in range(n_players):
            pos = all_snapshots[p][step]
            if pos is None:
                continue
            for rank, q in enumerate(ally_orders[p]):
                qpos = all_snapshots[q][step]
                if qpos is not None:
                    dist_allies[p][rank][step] = round(euclidean(pos, qpos), 1)
            for rank, q in enumerate(enemy_orders[p]):
                qpos = all_snapshots[q][step]
                if qpos is not None:
                    dist_enemies[p][rank][step] = round(euclidean(pos, qpos), 1)
            for t_idx, tower_pos in enumerate(ALL_TOWERS):
                dist_towers[p][t_idx][step] = round(euclidean(pos, tower_pos), 1)

    return dist_allies, dist_enemies, dist_towers


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_match(
    match_node: dict,
    hero_map: dict[int, str],
) -> dict:
    """
    Transform a raw STRATZ `match` node into the target output schema.

    Args:
        match_node: the `data.match` object from STRATZ GraphQL.
        hero_map:   {heroId: heroName} lookup.

    Returns:
        dict with `meta` and `players` keys, ready for serialization.
    """
    match_id            = match_node.get("id")
    duration            = match_node.get("durationSeconds") or 0
    did_radiant_win     = match_node.get("didRadiantWin")
    bracket             = match_node.get("bracket")
    game_version        = match_node.get("gameVersionId")
    avg_mmr             = match_node.get("averageRank")  # null if unavailable
    bottom_lane_outcome = match_node.get("bottomLaneOutcome")
    mid_lane_outcome    = match_node.get("midLaneOutcome")
    top_lane_outcome    = match_node.get("topLaneOutcome")

    players_raw = match_node.get("players") or []

    radiant_flags = [bool(p.get("isRadiant")) for p in players_raw]

    # heroDamageEvents in each player's playbackData contains only damage DEALT by that
    # player (attacker == hero_id). For damage TAKEN we cross-reference: collect every
    # heroDamageEvent across all players and filter by target == hero_id.
    all_hero_dmg_events: list[dict] = []
    for p in players_raw:
        all_hero_dmg_events.extend(
            (p.get("playbackData") or {}).get("heroDamageEvents") or []
        )

    # Collect per-bucket position snapshots for all players (for cross-player calcs)
    all_snapshots: list[list[tuple[float, float] | None]] = []
    for p in players_raw:
        pb = p.get("playbackData") or {}
        all_snapshots.append(_position_snapshots(pb.get("playerUpdatePositionEvents") or []))

    # Compute positional features across all players simultaneously.
    # Ally/enemy ordering is fixed by distance at ALLY_SORT_BUCKET (t=90s).
    dist_allies, dist_enemies, dist_towers = _compute_positional_features(
        all_snapshots, radiant_flags
    )

    # ------------------------------------------------------------------
    # Build per-player output
    # ------------------------------------------------------------------
    players_out = []
    for i, p in enumerate(players_raw):
        pb      = p.get("playbackData") or {}
        hero_id = p.get("heroId")
        is_rad  = radiant_flags[i]

        # attacker/target in heroDamageEvents are hero IDs (not slot IDs).
        # Each player's own heroDamageEvents = damage they dealt (attacker == hero_id).
        # Damage taken = events from other players where target == this hero_id.
        damage_dealt_events = [
            e for e in (pb.get("heroDamageEvents") or [])
            if not e.get("fromIllusion")
        ]
        damage_taken_events = [
            e for e in all_hero_dmg_events
            if e.get("target") == hero_id and not e.get("toIllusion")
        ]
        heal_events = pb.get("healEvents") or []

        # Cumulative curves
        gold_curve    = _build_networth_curve(pb.get("playerUpdateGoldEvents") or [])
        xp_curve      = _build_cumsum_curve(pb.get("experienceEvents") or [], "amount")
        dealt_curve   = _build_cumsum_curve(damage_dealt_events, "value")
        taken_curve   = _build_cumsum_curve(damage_taken_events, "value")
        tower_curve   = _build_cumsum_curve(pb.get("towerDamageEvents") or [], "damage")
        heal_curve    = _build_cumsum_curve(heal_events, "value")
        cs_curve      = _build_cs_curve(pb.get("csEvents") or [])

        # Normalize and extract scalars
        gold_norm,    max_gold    = _normalize(gold_curve)
        xp_norm,      max_xp     = _normalize(xp_curve)
        dealt_norm,   max_dealt  = _normalize(dealt_curve)
        taken_norm,   max_taken  = _normalize(taken_curve)
        tower_norm,   max_tower  = _normalize(tower_curve)
        heal_norm,    max_heal   = _normalize(heal_curve)
        cs_norm,      max_cs     = _normalize(cs_curve)

        # State curves
        health_pct, mana_pct = _build_state_curves(pb.get("playerUpdateHealthEvents") or [])

        # Kill/death/assist events are already scoped per-player in STRATZ's playbackData
        # (confirmed: assistEvents works correctly without slot filtering).
        kill_events   = pb.get("killEvents")   or []
        death_events  = pb.get("deathEvents")  or []
        assist_events = pb.get("assistEvents") or []

        kills_pm        = _count_per_minute(kill_events)
        deaths_pm       = _count_per_minute(death_events)
        assists_pm      = _count_per_minute(assist_events)
        ability_casts_pm = _count_per_minute(pb.get("abilityUsedEvents") or [])

        team = "RADIANT" if is_rad else "DIRE"

        players_out.append(
            {
                "steamAccountId": p.get("steamAccountId"),
                "heroId":         hero_id,
                "heroName":       hero_map.get(hero_id, ""),
                "position":       p.get("position"),
                "lane":           p.get("lane"),
                "team":           team,
                "isVictory":      p.get("isVictory"),
                "scalars": {
                    "maxGold":        int(max_gold),
                    "maxXp":          int(max_xp),
                    "maxDamageDealt": int(max_dealt),
                    "maxDamageTaken": int(max_taken),
                    "maxCs":          int(max_cs),
                    "maxTowerDamage": int(max_tower),
                    "maxHealing":     int(max_heal),
                },
                "timeseries": {
                    "goldNorm":        [round(v, 4) for v in gold_norm],
                    "xpNorm":          [round(v, 4) for v in xp_norm],
                    "damageDealtNorm": [round(v, 4) for v in dealt_norm],
                    "damageTakenNorm": [round(v, 4) for v in taken_norm],
                    "csNorm":          [round(v, 4) for v in cs_norm],
                    "towerDamageNorm": [round(v, 4) for v in tower_norm],
                    "healingNorm":     [round(v, 4) for v in heal_norm],
                    "healthPct":       health_pct,
                    "manaPct":         mana_pct,
                    # Per-ally distances (order fixed by proximity at t=90s; index 0 = closest)
                    "distToAlly0":     dist_allies[i][0],
                    "distToAlly1":     dist_allies[i][1],
                    "distToAlly2":     dist_allies[i][2],
                    "distToAlly3":     dist_allies[i][3],
                    # Per-enemy distances (same ordering convention)
                    "distToEnemy0":    dist_enemies[i][0],
                    "distToEnemy1":    dist_enemies[i][1],
                    "distToEnemy2":    dist_enemies[i][2],
                    "distToEnemy3":    dist_enemies[i][3],
                    "distToEnemy4":    dist_enemies[i][4],
                    # Per-tower distances: Radiant top/mid/bot (0–2), Dire top/mid/bot (3–5)
                    "distToTower0":    dist_towers[i][0],
                    "distToTower1":    dist_towers[i][1],
                    "distToTower2":    dist_towers[i][2],
                    "distToTower3":    dist_towers[i][3],
                    "distToTower4":    dist_towers[i][4],
                    "distToTower5":    dist_towers[i][5],
                },
                "events": {
                    "kills":        kills_pm,
                    "deaths":       deaths_pm,
                    "assists":      assists_pm,
                    "abilityCasts": ability_casts_pm,
                },
            }
        )

    return {
        "meta": {
            "matchId":            match_id,
            "durationSeconds":    duration,
            "didRadiantWin":      did_radiant_win,
            "avgMmr":             avg_mmr,
            "bracket":            bracket,
            "gameVersionId":      game_version,
            "bottomLaneOutcome":  bottom_lane_outcome,
            "midLaneOutcome":     mid_lane_outcome,
            "topLaneOutcome":     top_lane_outcome,
        },
        "players": players_out,
    }
