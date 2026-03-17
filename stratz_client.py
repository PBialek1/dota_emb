"""stratz_client.py — GraphQL client for the STRATZ API with rate limiting and retry logic."""

import json
import logging
import os
import time

import requests

from constants import STRATZ_GRAPHQL_URL, STRATZ_API_BASE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GraphQL query
# ---------------------------------------------------------------------------

MATCH_QUERY = """
query GetMatch($matchId: Long!) {
  match(id: $matchId) {
    id
    durationSeconds
    didRadiantWin
    bracket
    gameVersionId
    averageRank

    players {
      steamAccountId
      heroId
      position
      lane
      isRadiant
      isVictory

      playbackData {
        playerUpdatePositionEvents { time x y }

        playerUpdateHealthEvents { time hp maxHp mp maxMp }

        playerUpdateGoldEvents { time gold networth }

        abilityUsedEvents { time abilityId attacker target }

        killEvents   { time attacker target }
        deathEvents  { time attacker target goldFed timeDead }
        assistEvents { time attacker target }

        csEvents {
          time npcId isCreep isNeutral isAncient
        }

        heroDamageEvents {
          time attacker target value isSourceMainHero isTargetMainHero fromIllusion toIllusion
        }

        towerDamageEvents { time damage }

        healEvents { time attacker target value }

        experienceEvents { time amount }

        runeEvents { time rune action }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class TokenBucket:
    """Simple token bucket for rate limiting requests."""

    def __init__(self, rate_per_hour: int):
        self._rate_per_sec: float = rate_per_hour / 3600.0
        self._tokens: float = float(rate_per_hour)
        self._last_refill: float = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._tokens + elapsed * self._rate_per_sec, 1.0 / self._rate_per_sec * 3600)
        self._last_refill = now

    def consume(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            sleep_sec = (1.0 - self._tokens) / self._rate_per_sec
            logger.debug("Rate limit: sleeping %.2fs", sleep_sec)
            time.sleep(sleep_sec)


# ---------------------------------------------------------------------------
# STRATZ client
# ---------------------------------------------------------------------------

class StratzClient:
    """
    Wraps the STRATZ GraphQL API with:
      - Bearer token authentication
      - Token-bucket rate limiting
      - Exponential-backoff retry on transient errors
    """

    def __init__(self, token: str, rate_per_hour: int = 280, max_retries: int = 3):
        self._token = token
        self._bucket = TokenBucket(rate_per_hour)
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "dota-emb/1.0",
            }
        )
        self._hero_cache: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_match(self, match_id: int) -> dict:
        """
        Fetch raw match data from STRATZ.

        Returns the parsed `data.match` node on success.
        Raises RuntimeError on API error (caller should catch and log).
        """
        payload = {"query": MATCH_QUERY, "variables": {"matchId": match_id}}
        resp_json = self._post_with_retry(payload)

        match_node = resp_json.get("data", {}).get("match")
        if match_node is None:
            raise RuntimeError(
                f"Match {match_id} not found or returned null. "
                f"Errors: {resp_json.get('errors')}"
            )
        return match_node

    def fetch_hero_map(self) -> dict[int, str]:
        """
        Return {heroId: heroName} map, fetched from STRATZ constants and cached locally.
        """
        cache_path = "hero_names.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                self._hero_cache = {int(k): v for k, v in json.load(fh).items()}
            return self._hero_cache

        url = f"{STRATZ_API_BASE}/Hero"
        logger.info("Fetching hero list from STRATZ …")
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # STRATZ may return a dict or list
        if isinstance(data, dict):
            heroes = data.values()
        else:
            heroes = data

        result: dict[int, str] = {}
        for hero in heroes:
            hid  = hero.get("id")
            name = (hero.get("language") or {}).get("displayName") or hero.get("name") or ""
            if hid is not None:
                result[int(hid)] = name

        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in result.items()}, fh, indent=2)

        self._hero_cache = result
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _post_with_retry(self, payload: dict) -> dict:
        self._bucket.consume()

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.post(STRATZ_GRAPHQL_URL, json=payload, timeout=30)
            except requests.RequestException as exc:
                if attempt == self._max_retries:
                    raise RuntimeError(f"Network error after {attempt} attempts: {exc}") from exc
                wait = 2 ** attempt
                logger.warning("Network error (attempt %d/%d), retrying in %ds: %s",
                               attempt, self._max_retries, wait, exc)
                time.sleep(wait)
                continue

            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("HTTP 429 rate limited (attempt %d/%d), sleeping %ds",
                               attempt, self._max_retries, wait)
                time.sleep(wait)
                self._bucket.consume()
                continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from STRATZ: {resp.text[:300]}"
                )

            data = resp.json()
            if "errors" in data and not data.get("data"):
                raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'])}")

            return data

        raise RuntimeError(f"Failed after {self._max_retries} attempts.")
