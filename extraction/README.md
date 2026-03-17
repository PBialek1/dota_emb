# extraction

Fetches Dota 2 match IDs from OpenDota and extracts per-player laning-stage features from the STRATZ GraphQL API.

---

## Pipeline

```
fetch_match_ids.py   — Query OpenDota for match IDs by patch + MMR bracket
        ↓
extract_match.py     — Fetch each match from STRATZ and build feature vectors
        ↓
  data/matches/      — One JSON per match  OR  data/matches.db (SQLite)
```

---

## Usage

### 1. Fetch match IDs

```bash
python extraction/fetch_match_ids.py --patch 7.40 --bracket divine --count 1000
python extraction/fetch_match_ids.py --patch 7.40 --min-rank 70 --max-rank 75 --count 500
```

Output is written to `data/match_lists/match_ids_<patch>_<bracket>.txt`.

**Bracket reference:**

| `--bracket` | Rank range | Tier |
|---|---|---|
| `herald` | 10–15 | Herald |
| `guardian` | 20–25 | Guardian |
| `crusader` | 30–35 | Crusader |
| `archon` | 40–45 | Archon |
| `legend` | 50–55 | Legend |
| `ancient` | 60–65 | Ancient |
| `divine` | 70–75 | Divine |
| `immortal` | 80 | Immortal |

### 2. Extract features

```bash
# JSON store (one file per match)
python extraction/extract_match.py --match-ids-file data/match_lists/match_ids_740_divine.txt

# SQLite store
python extraction/extract_match.py --match-ids-file data/match_lists/match_ids_740_divine.txt \
    --store sqlite --db ./data/matches.db
```

Failed/skipped matches are logged to `data/failed_matches.txt` and `data/skipped_matches.txt`. Re-running is safe — already-stored matches are skipped automatically.

---

## Output Schema

Features cover the first 10 minutes (laning stage). All timeseries arrays have exactly 10 elements — one per minute.

```json
{
  "meta": { "matchId": 7123456789, "durationSeconds": 2340, "didRadiantWin": true, "bracket": 6 },
  "players": [
    {
      "heroName": "Luna", "position": "POSITION_1", "lane": "SAFE_LANE",
      "team": "RADIANT", "isVictory": true,
      "scalars":    { "maxGold": 4200, "maxXp": 5800, "maxCs": 87, "..." : "..." },
      "timeseries": { "goldNorm": [...], "xpNorm": [...], "healthPct": [...], "..." : "..." },
      "events":     { "kills": [...], "deaths": [...], "abilityCasts": [...] },
      "alliesNearby": [...], "enemiesNearby": [...]
    }
  ]
}
```

See the top-level README and `stratz_extraction_instructions.md` for full schema details.

---

## Files

| File | Description |
|---|---|
| `fetch_match_ids.py` | OpenDota match ID scraper |
| `extract_match.py` | CLI entry point and orchestration |
| `stratz_client.py` | STRATZ GraphQL client (auth, rate limiting, retry) |
| `feature_builder.py` | Raw STRATZ response → feature dict |
| `store.py` | Storage backends (JsonStore, SqliteStore) |
| `constants.py` | Tower positions, map constants, shared geometry |
| `hero_names.json` | Cached hero ID → name map (auto-generated on first run) |
