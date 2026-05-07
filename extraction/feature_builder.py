"""feature_builder.py — Filters raw STRATZ events to the first 10 minutes and structures output."""

from constants import WINDOW_END


def _filter(events: list[dict] | None) -> list[dict]:
    """Return events with time in [0, WINDOW_END), sorted by time."""
    if not events:
        return []
    return sorted(
        [e for e in events if 0 <= (e.get("time") or 0) < WINDOW_END],
        key=lambda e: e.get("time") or 0,
    )


def build_match(match_node: dict, hero_map: dict[int, str]) -> dict:
    """
    Transform a raw STRATZ match node into the raw-events output schema.

    Each player gets one list per event type, filtered to [0, WINDOW_END) and sorted by time.
    heroDamageEvents contains damage dealt by that player (STRATZ scopes it per attacker).
    """
    players_raw = match_node.get("players") or []
    players_out = []

    for p in players_raw:
        pb      = p.get("playbackData") or {}
        hero_id = p.get("heroId")

        players_out.append({
            "steamAccountId": p.get("steamAccountId"),
            "heroId":         hero_id,
            "heroName":       hero_map.get(hero_id, ""),
            "position":       p.get("position"),
            "lane":           p.get("lane"),
            "team":           "RADIANT" if p.get("isRadiant") else "DIRE",
            "isVictory":      p.get("isVictory"),
            "positions":      _filter(pb.get("playerUpdatePositionEvents")),
            "health":         _filter(pb.get("playerUpdateHealthEvents")),
            "gold":           _filter(pb.get("playerUpdateGoldEvents")),
            "xp":             _filter(pb.get("experienceEvents")),
            "lastHits":       _filter(pb.get("csEvents")),
            "damage":       _filter(pb.get("heroDamageEvents")),
            "abilities":    _filter(pb.get("abilityUsedEvents")),
            "kills":        _filter(pb.get("killEvents")),
            "deaths":       _filter(pb.get("deathEvents")),
            "assists":      _filter(pb.get("assistEvents")),
            "healing":      _filter(pb.get("healEvents")),
            "towerDamage":  _filter(pb.get("towerDamageEvents")),
            "items":        _filter(pb.get("purchaseEvents")),
        })

    return {
        "meta": {
            "matchId":           match_node.get("id"),
            "durationSeconds":   match_node.get("durationSeconds") or 0,
            "didRadiantWin":     match_node.get("didRadiantWin"),
            "avgMmr":            match_node.get("averageRank"),
            "bracket":           match_node.get("bracket"),
            "gameVersionId":     match_node.get("gameVersionId"),
            "bottomLaneOutcome": match_node.get("bottomLaneOutcome"),
            "midLaneOutcome":    match_node.get("midLaneOutcome"),
            "topLaneOutcome":    match_node.get("topLaneOutcome"),
        },
        "players": players_out,
    }
