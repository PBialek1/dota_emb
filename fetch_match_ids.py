"""fetch_match_ids.py — Pull public Dota 2 match IDs from OpenDota filtered by patch and MMR.

OpenDota rank encoding (avg_rank_tier):
    tens digit = bracket  ones digit = stars (0–5)
    1x = Herald   2x = Guardian   3x = Crusader
    4x = Archon   5x = Legend     6x = Ancient
    7x = Divine   80 = Immortal

Usage:
    python fetch_match_ids.py --patch 7.40 --bracket legend --count 1000
    python fetch_match_ids.py --patch 7.40 --min-rank 70 --max-rank 75 --count 500
    python fetch_match_ids.py --patch 7.40 --bracket divine --count 1000 --output divine_740.txt
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OPENDOTA_BASE = "https://api.opendota.com/api"

# Game modes considered "standard" — excludes Turbo (23), arcade/custom (15, 19, 24),
# tutorial (10), event (19), and other non-competitive modes.
ALLOWED_GAME_MODES: frozenset[int] = frozenset({
    1,   # All Pick
    2,   # Captains Mode
    3,   # Random Draft
    4,   # Single Draft
    5,   # All Random
    16,  # Captains Draft
    22,  # All Draft (Ranked All Pick)
})

BRACKET_ALIASES = {
    "herald":    (10, 15),
    "guardian":  (20, 25),
    "crusader":  (30, 35),
    "archon":    (40, 45),
    "legend":    (50, 55),
    "ancient":   (60, 65),
    "divine":    (70, 75),
    "immortal":  (80, 80),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

session = requests.Session()
session.headers.update({"User-Agent": "dota-match-id-fetcher/1.0"})


# ---------------------------------------------------------------------------
# OpenDota helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None, retries: int = 3) -> dict | list:
    url = OPENDOTA_BASE + path
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            logger.warning("Request failed (%s), retrying in 5s…", exc)
            time.sleep(5)


def _resolve_patch(patch_name: str) -> tuple[int, int | None]:
    """
    Look up patch time boundaries from OpenDota constants.

    Returns (start_unix, end_unix | None).
    Accepts exact names like '7.40c' or major prefixes like '7.40'
    (prefix matches the first patch in that series, e.g. '7.40' → '7.40').
    """
    patches: list[dict] = _get("/constants/patch")
    patches = sorted(patches, key=lambda p: p.get("id", 0))

    def _parse_ts(value) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        return int(datetime.fromisoformat(value.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp())

    def _normalize(name: str) -> str:
        return str(name).lstrip("v")

    # Exact match first, then prefix
    target_idx: int | None = None
    for i, p in enumerate(patches):
        if _normalize(p.get("name", "")) == patch_name:
            target_idx = i
            break
    if target_idx is None:
        for i, p in enumerate(patches):
            if _normalize(p.get("name", "")).startswith(patch_name):
                target_idx = i
                break

    if target_idx is None:
        available = [_normalize(p.get("name", "")) for p in patches[-10:]]
        raise ValueError(f"Patch '{patch_name}' not found. Recent patches: {available}")

    start_ts = _parse_ts(patches[target_idx]["date"])

    # End = start of the next patch whose major.minor differs from patch_name
    def _major_minor(name: str) -> tuple[int, int] | None:
        parts = _normalize(name).split(".")
        try:
            return int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            return None

    base_mm = _major_minor(patch_name)
    end_ts: int | None = None
    for p in patches[target_idx + 1:]:
        mm = _major_minor(p.get("name", ""))
        if mm and base_mm and mm != base_mm:
            end_ts = _parse_ts(p["date"])
            break

    logger.info(
        "Patch %s: start=%d  end=%s",
        patch_name, start_ts, end_ts if end_ts else "present",
    )
    return start_ts, end_ts


# ---------------------------------------------------------------------------
# Fetch via /publicMatches
# ---------------------------------------------------------------------------

def fetch_match_ids(
    start_ts: int,
    end_ts: int | None,
    min_rank: int,
    max_rank: int,
    count: int,
    api_key: str | None,
    delay: float = 1.0,
) -> list[int]:
    """Paginate /publicMatches, keeping only matches within the patch time window and rank range."""
    logger.info("Fetching match IDs via /publicMatches (rank %d–%d)…", min_rank, max_rank)
    ids: list[int] = []
    less_than: int | None = None

    while len(ids) < count:
        params: dict = {"min_rank": min_rank, "max_rank": max_rank}
        if less_than is not None:
            params["less_than_match_id"] = less_than
        if api_key:
            params["api_key"] = api_key

        try:
            matches: list[dict] = _get("/publicMatches", params=params)
        except requests.RequestException as exc:
            logger.error("Failed to fetch public matches: %s", exc)
            break

        if not matches:
            logger.info("No more matches returned — stopping.")
            break

        added = 0
        oldest_ts = None
        for m in matches:
            mid = m.get("match_id")
            ts = m.get("start_time", 0)
            avg_rank = m.get("avg_rank_tier", 0)
            game_mode = m.get("game_mode", 0)

            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts

            if end_ts and ts >= end_ts:
                continue
            if ts < start_ts:
                continue
            if avg_rank < min_rank or avg_rank > max_rank:
                continue
            if game_mode not in ALLOWED_GAME_MODES:
                continue
            if mid:
                ids.append(int(mid))
                added += 1

        least_id = min(m["match_id"] for m in matches if "match_id" in m)
        if less_than is not None and least_id >= less_than:
            logger.warning("Pagination cursor did not advance — stopping.")
            break
        less_than = least_id

        logger.info(
            "Page: +%d matches (total %d / %d), oldest page ts: %s",
            added, len(ids), count, oldest_ts,
        )

        if oldest_ts is not None and oldest_ts < start_ts:
            logger.info("Reached matches older than patch start — stopping.")
            break

        time.sleep(delay)

    return ids[:count]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch public Dota 2 match IDs from OpenDota filtered by patch and MMR."
    )
    parser.add_argument(
        "--patch", required=True,
        help="Patch version string, e.g. '7.40' or '7.40c'.",
    )

    rank_group = parser.add_mutually_exclusive_group(required=True)
    rank_group.add_argument(
        "--bracket",
        choices=list(BRACKET_ALIASES),
        help="Named skill bracket (e.g. divine, legend).",
    )
    rank_group.add_argument(
        "--min-rank", type=int,
        help="Minimum avg_rank_tier (e.g. 70 for Divine 0).",
    )

    parser.add_argument(
        "--max-rank", type=int,
        help="Maximum avg_rank_tier (required with --min-rank).",
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Number of match IDs to collect (default: 1000).",
    )
    parser.add_argument(
        "--output", type=str,
        help="Output file path (default: match_ids_<patch>_<bracket>.txt).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenDota API key for higher rate limits (optional).",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between paginated requests (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bracket:
        min_rank, max_rank = BRACKET_ALIASES[args.bracket]
        bracket_label = args.bracket
    else:
        if args.max_rank is None:
            print("ERROR: --max-rank is required when using --min-rank.", file=sys.stderr)
            sys.exit(1)
        min_rank = args.min_rank
        max_rank = args.max_rank
        bracket_label = f"{min_rank}-{max_rank}"

    out_path = Path(args.output) if args.output else Path(
        f"data/match_lists/match_ids_{args.patch.replace('.', '')}_{bracket_label}.txt"
    )

    logger.info(
        "Target: patch=%s  rank=%d–%d  count=%d  output=%s",
        args.patch, min_rank, max_rank, args.count, out_path,
    )

    try:
        start_ts, end_ts = _resolve_patch(args.patch)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    ids = fetch_match_ids(
        start_ts, end_ts, min_rank, max_rank, args.count, args.api_key, args.delay
    )

    if not ids:
        logger.error("No match IDs collected — check patch name and rank range.")
        sys.exit(1)

    seen: set[int] = set()
    unique_ids: list[int] = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(mid) for mid in unique_ids) + "\n", encoding="utf-8")
    logger.info("Wrote %d match IDs to %s", len(unique_ids), out_path)


if __name__ == "__main__":
    main()
