"""extract_match.py — CLI entry point: accepts matchId(s), fetches and saves.

Usage:
    python extract_match.py --match-id 7123456789 --output-dir ./data/matches
    python extract_match.py --match-ids-file match_ids.txt --output-dir ./data/matches
    python extract_match.py --match-id 7123456789 --rate-limit 250
    python extract_match.py --match-ids-file match_ids.txt --store sqlite --db ./data/matches.db
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv

from feature_builder import build_match
from store import JsonStore, SqliteStore, MatchStore
from stratz_client import StratzClient
from constants import LANING_SECONDS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FAILED_LOG  = "data/failed_matches.txt"
SKIPPED_LOG = "data/skipped_matches.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_failed(match_id: int, reason: str) -> None:
    Path(FAILED_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{match_id}\t{reason}\n")
    logger.warning("Match %s → FAILED: %s", match_id, reason)


def _log_skipped(match_id: int, reason: str) -> None:
    Path(SKIPPED_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(SKIPPED_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{match_id}\t{reason}\n")
    logger.info("Match %s → SKIPPED: %s", match_id, reason)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_match(
    match_id: int,
    client: StratzClient,
    hero_map: dict[int, str],
    store: MatchStore,
) -> bool:
    """
    Fetch, process, and save one match.

    Returns True on success, False on skip/failure (already logged).
    """
    if store.exists(match_id):
        logger.info("Match %s already exists — skipping.", match_id)
        return True

    # Fetch
    try:
        match_node = client.fetch_match(match_id)
    except RuntimeError as exc:
        _log_failed(match_id, str(exc))
        return False

    # Skip short matches
    duration = match_node.get("durationSeconds") or 0
    if duration < LANING_SECONDS:
        _log_skipped(match_id, f"duration < 600s ({duration}s)")
        return False

    # Build features
    try:
        result = build_match(match_node, hero_map)
    except Exception as exc:  # noqa: BLE001
        _log_failed(match_id, f"feature_builder error: {exc}")
        logger.exception("Unexpected error while building match %s", match_id)
        return False

    # Save
    store.save(match_id, result)
    logger.info("Saved match %s", match_id)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract laning-stage features from STRATZ for one or more Dota 2 matches."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match-id",       type=int,  help="Single match ID to extract.")
    group.add_argument("--match-ids-file", type=str,  help="Path to file with one match ID per line.")

    parser.add_argument(
        "--store",
        choices=["json", "sqlite"],
        default="json",
        help="Storage backend: 'json' writes one file per match (default), 'sqlite' writes to a database.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/matches",
        help="Directory to write {matchId}.json files — used with --store json (default: ./data/matches).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="./data/matches.db",
        help="SQLite database path — used with --store sqlite (default: ./data/matches.db).",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=280,
        help="Max API requests per hour (default: 280; STRATZ free tier is 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = os.environ.get("STRATZ_TOKEN", "")
    if not token:
        logger.error(
            "STRATZ_TOKEN environment variable not set. "
            "Add it to your .env file as: STRATZ_TOKEN=your_token_here"
        )
        sys.exit(1)

    # Build storage backend
    if args.store == "sqlite":
        store: MatchStore = SqliteStore(Path(args.db))
        logger.info("Using SQLite store: %s", args.db)
    else:
        store = JsonStore(Path(args.output_dir))
        logger.info("Using JSON store: %s", args.output_dir)

    # Collect match IDs
    if args.match_id:
        match_ids = [args.match_id]
    else:
        ids_file = Path(args.match_ids_file)
        if not ids_file.exists():
            logger.error("match-ids-file not found: %s", ids_file)
            sys.exit(1)
        match_ids = [
            int(line.strip())
            for line in ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.strip().isdigit()
        ]
        logger.info("Loaded %d match IDs from %s", len(match_ids), ids_file)

    # Initialise shared objects
    client   = StratzClient(token=token, rate_per_hour=args.rate_limit)
    hero_map = client.fetch_hero_map()

    # Process
    success = failed = 0
    use_tqdm = len(match_ids) > 1
    iterator = tqdm(match_ids, unit="match") if use_tqdm else match_ids
    try:
        for match_id in iterator:
            ok = extract_match(match_id, client, hero_map, store)
            if ok:
                success += 1
            else:
                failed += 1
    finally:
        store.close()

    total = len(match_ids)
    logger.info(
        "Done. %d/%d succeeded. Check %s and %s for issues.",
        success, total, FAILED_LOG, SKIPPED_LOG,
    )


if __name__ == "__main__":
    main()
