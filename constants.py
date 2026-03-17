"""constants.py — Tower positions, map constants, and shared geometry helpers."""

import math

STRATZ_GRAPHQL_URL = "https://api.stratz.com/graphql"
STRATZ_API_BASE = "https://api.stratz.com/api/v1"

LANING_MINUTES = 10
LANING_SECONDS = 600
# STRATZ position events use a compressed coordinate system.
# Conversion from Dota 2 world units: STRATZ = (world + 8192) / 128
# Empirically confirmed: Radiant fountain ≈ (74, 74), Dire fountain ≈ (180, 177).
# 1500 world units (roughly one screen width) ≈ 12 STRATZ units.
ALLY_ENEMY_RADIUS = 12  # STRATZ units — threshold for alliesNearby / enemiesNearby

# Tier-1 tower positions converted to STRATZ coordinates via (world + 8192) / 128
TOWER_POSITIONS = {
    "radiant_top_t1": (80.0,  114.0),
    "radiant_mid_t1": (96.0,   96.0),
    "radiant_bot_t1": (114.0,  80.0),
    "dire_top_t1":    (139.0, 173.0),
    "dire_mid_t1":    (157.0, 157.0),
    "dire_bot_t1":    (173.0, 139.0),
}

RADIANT_TOWERS = [pos for key, pos in TOWER_POSITIONS.items() if key.startswith("radiant")]
DIRE_TOWERS    = [pos for key, pos in TOWER_POSITIONS.items() if key.startswith("dire")]


def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
