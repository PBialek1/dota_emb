"""backfill_lane_outcomes.py — One-time script to add lane outcome data to existing records.

For the SQLite backend: fetches bottomLaneOutcome / midLaneOutcome / topLaneOutcome from
STRATZ for every match that currently has NULL in those columns and updates the DB.

For the JSON backend: reads each .json file, fills in the three fields under `meta`, and
rewrites the file.

Usage:
    # SQLite (default)
    python backfill_lane_outcomes.py --db ../data/matches.db

    # JSON files
    python backfill_lane_outcomes.py --json-dir ../data/matches

    # Both at once
    python backfill_lane_outcomes.py --db ../data/matches.db --json-dir ../data/matches

    # Dry-run (fetch + print, no writes)
    python backfill_lane_outcomes.py --db ../data/matches.db --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from constants import STRATZ_GRAPHQL_URL
from stratz_client import TokenBucket, MultiRateLimiter

class _TqdmHandler(logging.Handler):
    """Logging handler that routes through tqdm.write to avoid breaking progress bars."""
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batched GraphQL query using aliases
# ---------------------------------------------------------------------------

BATCH_SIZE = 50

def _build_batch_query(match_ids: list[int]) -> str:
    """Build a single GraphQL query that fetches lane outcomes for multiple matches via aliases."""
    fragments = "\n".join(
        f"  m{mid}: match(id: {mid}) {{ id bottomLaneOutcome midLaneOutcome topLaneOutcome }}"
        for mid in match_ids
    )
    return f"query {{\n{fragments}\n}}"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_lane_outcomes_batch(
    match_ids: list[int], session: requests.Session
) -> dict[int, dict]:
    """
    Fetch lane outcomes for a batch of match IDs in a single GraphQL request.

    Returns {match_id: {"bottomLaneOutcome": ..., "midLaneOutcome": ..., "topLaneOutcome": ...}}
    for each match that returned a valid result. Missing/failed matches are omitted.
    """
    payload = {"query": _build_batch_query(match_ids)}
    try:
        resp = session.post(STRATZ_GRAPHQL_URL, json=payload, timeout=60)
    except requests.RequestException as exc:
        logger.warning("Network error for batch of %d matches: %s", len(match_ids), exc)
        return {}

    if resp.status_code == 429:
        logger.warning("HTTP 429 — backing off 60s")
        time.sleep(60)
        return {}

    if resp.status_code != 200:
        logger.warning("HTTP %d for batch of %d matches", resp.status_code, len(match_ids))
        return {}

    data = resp.json().get("data") or {}
    results: dict[int, dict] = {}
    for mid in match_ids:
        node = data.get(f"m{mid}")
        if node is None:
            logger.warning("Match %d not found in batch response", mid)
            continue
        results[mid] = {
            "bottomLaneOutcome": node.get("bottomLaneOutcome"),
            "midLaneOutcome":    node.get("midLaneOutcome"),
            "topLaneOutcome":    node.get("topLaneOutcome"),
        }
    return results


# ---------------------------------------------------------------------------
# SQLite backfill
# ---------------------------------------------------------------------------

def backfill_sqlite(db_path: Path, session: requests.Session, bucket: MultiRateLimiter,
                    dry_run: bool) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Add columns if they don't exist yet (idempotent)
    for col, definition in [
        ("bottom_lane_outcome",   "TEXT"),
        ("mid_lane_outcome",      "TEXT"),
        ("top_lane_outcome",      "TEXT"),
        ("lane_outcomes_fetched", "INTEGER NOT NULL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {definition}")
            conn.commit()
            logger.info("Added column %s to matches table", col)
        except sqlite3.OperationalError:
            pass  # already exists

    rows = conn.execute(
        "SELECT match_id FROM matches WHERE lane_outcomes_fetched = 0"
    ).fetchall()

    total = len(rows)
    logger.info("SQLite: %d matches need lane outcome backfill", total)

    all_ids = [row["match_id"] for row in rows]
    updated = 0
    failed  = 0

    with tqdm(total=total, desc="SQLite backfill", unit="match") as pbar:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = all_ids[batch_start : batch_start + BATCH_SIZE]
            bucket.consume()
            outcomes_map = fetch_lane_outcomes_batch(batch, session)

            for match_id in batch:
                outcomes = outcomes_map.get(match_id)
                if outcomes is None:
                    failed += 1
                else:
                    if not dry_run:
                        conn.execute(
                            """UPDATE matches
                               SET bottom_lane_outcome = :bottom,
                                   mid_lane_outcome    = :mid,
                                   top_lane_outcome    = :top
                               WHERE match_id = :match_id""",
                            {
                                "bottom":   outcomes["bottomLaneOutcome"],
                                "mid":      outcomes["midLaneOutcome"],
                                "top":      outcomes["topLaneOutcome"],
                                "match_id": match_id,
                            },
                        )
                    updated += 1

            if not dry_run:
                # Mark all attempted matches as fetched, even those STRATZ returned null for.
                conn.executemany(
                    "UPDATE matches SET lane_outcomes_fetched = 1 WHERE match_id = ?",
                    [(mid,) for mid in batch],
                )
                conn.commit()
            pbar.update(len(batch))
            pbar.set_postfix(updated=updated, failed=failed)

    conn.close()
    logger.info("SQLite done. updated=%d failed=%d", updated, failed)


# ---------------------------------------------------------------------------
# JSON backfill
# ---------------------------------------------------------------------------

def backfill_json(json_dir: Path, session: requests.Session, bucket: MultiRateLimiter,
                  dry_run: bool) -> None:
    files = sorted(json_dir.glob("*.json"))
    total = len(files)
    logger.info("JSON: %d files to check", total)

    # Pass 1: scan files and collect those that need backfilling
    to_update: list[tuple[int, Path]] = []  # (match_id, fpath)
    skipped = 0
    scan_errors = 0

    logger.info("Scanning %d JSON files…", total)
    for fpath in files:
        try:
            data = json.loads(fpath.read_bytes())
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath.name, exc)
            scan_errors += 1
            continue
        meta = data.get("meta", {})
        if "bottomLaneOutcome" in meta:
            skipped += 1
            continue
        match_id = meta.get("matchId")
        if match_id is None:
            logger.warning("No matchId in %s, skipping", fpath.name)
            scan_errors += 1
            continue
        to_update.append((match_id, fpath))

    logger.info("%d files need update, %d already have outcomes, %d scan errors",
                len(to_update), skipped, scan_errors)

    # Pass 2: batch-fetch and write
    updated = 0
    failed  = 0

    with tqdm(total=len(to_update), desc="JSON backfill", unit="file") as pbar:
        for batch_start in range(0, len(to_update), BATCH_SIZE):
            batch = to_update[batch_start : batch_start + BATCH_SIZE]
            batch_ids = [mid for mid, _ in batch]
            bucket.consume()
            outcomes_map = fetch_lane_outcomes_batch(batch_ids, session)

            for match_id, fpath in batch:
                outcomes = outcomes_map.get(match_id)
                if outcomes is None:
                    failed += 1
                    # Still write null keys so re-runs skip this file.
                    outcomes = {"bottomLaneOutcome": None, "midLaneOutcome": None, "topLaneOutcome": None}
                else:
                    updated += 1

                if not dry_run:
                    # Re-read to avoid writing stale data if files changed.
                    data = json.loads(fpath.read_bytes())
                    data["meta"]["bottomLaneOutcome"] = outcomes["bottomLaneOutcome"]
                    data["meta"]["midLaneOutcome"]    = outcomes["midLaneOutcome"]
                    data["meta"]["topLaneOutcome"]    = outcomes["topLaneOutcome"]
                    try:
                        import orjson
                        fpath.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
                    except ImportError:
                        fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")

            pbar.update(len(batch))
            pbar.set_postfix(updated=updated, failed=failed)

    logger.info("JSON done. updated=%d skipped=%d failed=%d", updated, skipped, failed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill lane outcomes for existing matches")
    parser.add_argument("--db",       type=Path, help="Path to SQLite database")
    parser.add_argument("--json-dir", type=Path, help="Directory of per-match JSON files")
    parser.add_argument("--rate-limit", type=int, default=280,
                        help="STRATZ requests per hour (default: 280)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data but do not write anything")
    args = parser.parse_args()

    if not args.db and not args.json_dir:
        parser.error("Specify at least one of --db or --json-dir")

    token = os.environ.get("STRATZ_TOKEN")
    if not token:
        logger.error("STRATZ_TOKEN environment variable not set")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "User-Agent":    "dota-emb/1.0",
    })
    bucket = MultiRateLimiter(
        TokenBucket(20,              1),     # 20 req/second
        TokenBucket(250,             60),    # 250 req/minute
        TokenBucket(args.rate_limit, 3600),  # hourly
    )

    if args.dry_run:
        logger.info("DRY RUN — no files or DB rows will be modified")

    if args.db:
        if not args.db.exists():
            logger.error("Database not found: %s", args.db)
            sys.exit(1)
        backfill_sqlite(args.db, session, bucket, args.dry_run)

    if args.json_dir:
        if not args.json_dir.is_dir():
            logger.error("JSON directory not found: %s", args.json_dir)
            sys.exit(1)
        backfill_json(args.json_dir, session, bucket, args.dry_run)


if __name__ == "__main__":
    main()
