# dota_emb

A data pipeline for extracting laning-stage features from Dota 2 matches, designed as input for embedding models or other ML workflows. Fetches match IDs from OpenDota, pulls per-event playback data from the STRATZ GraphQL API, and outputs structured feature vectors — one per player per match.

---

## Pipeline Overview

```
fetch_match_ids.py          — Pull a list of match IDs from OpenDota (by patch + MMR)
        ↓
extract_match.py            — Fetch each match from STRATZ and extract features
        ↓
  JSON files  OR  SQLite DB — Structured per-player laning-stage feature vectors
```

---

## Setup

### Requirements

```
requests
python-dotenv
orjson          # optional but recommended — faster JSON serialization
```

Install:
```bash
pip install requests python-dotenv orjson
```

### STRATZ API Token

Create a free account at [stratz.com](https://stratz.com) and generate an API token. Add it to a `.env` file in the project root:

```
STRATZ_TOKEN=your_token_here
```

The free tier allows 300 requests/hour. The pipeline defaults to 280/hour to stay safely under the limit.

---

## Scripts

### 1. `fetch_match_ids.py` — Get match IDs from OpenDota

Queries the OpenDota public API for match IDs within a specific patch and MMR bracket, and writes them one-per-line to a text file.

```bash
# By named bracket
python fetch_match_ids.py --patch 7.37 --bracket divine --count 1000

# By explicit rank range
python fetch_match_ids.py --patch 7.37 --min-rank 70 --max-rank 75 --count 500

# Custom output file
python fetch_match_ids.py --patch 7.38 --bracket legend --count 500 --output legend_738.txt

# With an OpenDota API key (higher rate limits)
python fetch_match_ids.py --patch 7.37 --bracket divine --api-key YOUR_KEY
```

**Bracket reference:**

| `--bracket`  | Rank range | MMR tier   |
|--------------|-----------|------------|
| `herald`     | 10–15     | Herald     |
| `guardian`   | 20–25     | Guardian   |
| `crusader`   | 30–35     | Crusader   |
| `archon`     | 40–45     | Archon     |
| `legend`     | 50–55     | Legend     |
| `ancient`    | 60–65     | Ancient    |
| `divine`     | 70–75     | Divine     |
| `immortal`   | 80        | Immortal   |

The script tries the OpenDota SQL explorer endpoint first (most precise patch filtering), then falls back to paginating `/publicMatches` filtered by patch date boundaries if the explorer is unavailable.

Output is a plain text file with one match ID per line, compatible with `extract_match.py --match-ids-file`.

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--patch` | *(required)* | Patch version, e.g. `7.37` |
| `--bracket` | *(required)* | Named MMR bracket (see table above) |
| `--min-rank` / `--max-rank` | *(alternative to --bracket)* | Raw rank tier integers |
| `--count` | `1000` | Number of match IDs to collect |
| `--output` | auto-named | Output `.txt` file path |
| `--api-key` | none | OpenDota API key |
| `--method` | `auto` | `explorer`, `publicmatches`, or `auto` |
| `--delay` | `1.0` | Seconds between paginated requests |

---

### 2. `extract_match.py` — Extract features from STRATZ

Accepts one or more match IDs, queries the STRATZ GraphQL API, builds the feature vectors, and saves them.

```bash
# Single match, JSON output (default)
python extract_match.py --match-id 7123456789

# Batch from file, JSON output
python extract_match.py --match-ids-file match_ids.txt

# Batch to SQLite database
python extract_match.py --match-ids-file match_ids.txt --store sqlite

# All options
python extract_match.py --match-ids-file match_ids.txt \
    --store sqlite \
    --db ./data/matches.db \
    --rate-limit 250
```

Failed matches are logged to `failed_matches.txt`. Matches shorter than 10 minutes (early GGs) are skipped and logged to `skipped_matches.txt`. Both logs are tab-separated: `match_id\treason`. Re-running is safe — already-stored matches are skipped automatically.

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--match-id` | *(mutually exclusive)* | Single match ID |
| `--match-ids-file` | *(mutually exclusive)* | Path to text file of IDs |
| `--store` | `json` | `json` or `sqlite` |
| `--output-dir` | `./data/matches` | Directory for JSON files |
| `--db` | `./data/matches.db` | SQLite database path |
| `--rate-limit` | `280` | STRATZ API requests per hour |

---

## Output Schema

Features cover the first 10 minutes (laning stage) of each match. All timeseries arrays have exactly 10 elements — one per minute (minute 0 = seconds 0–59, minute 9 = seconds 540–599).

### JSON format

Each match is saved as `{matchId}.json` in the output directory.

```json
{
  "meta": {
    "matchId": 7123456789,
    "durationSeconds": 2340,
    "didRadiantWin": true,
    "avgMmr": 4500,
    "bracket": 6,
    "gameVersionId": 184
  },
  "players": [ /* 10 player objects */ ]
}
```

Each player object:

```json
{
  "steamAccountId": 123456,
  "heroId": 48,
  "heroName": "Luna",
  "position": "POSITION_1",
  "lane": "SAFE_LANE",
  "team": "RADIANT",
  "isVictory": true,

  "scalars": {
    "maxGold": 4200,
    "maxXp": 5800,
    "maxDamageDealt": 12000,
    "maxDamageTaken": 8000,
    "maxCs": 87,
    "maxTowerDamage": 600,
    "maxHealing": 0
  },

  "timeseries": {
    "goldNorm":            [0.0, 0.05, 0.12, "..."],
    "xpNorm":              [0.0, 0.04, 0.10, "..."],
    "damageDealtNorm":     [0.0, 0.01, 0.04, "..."],
    "damageTakenNorm":     [0.0, 0.02, 0.05, "..."],
    "csNorm":              [0.0, 0.06, 0.14, "..."],
    "towerDamageNorm":     [0.0, 0.0,  0.0,  "..."],
    "healingNorm":         [0.0, 0.0,  0.0,  "..."],
    "healthPct":           [1.0, 0.92, 0.85, "..."],
    "manaPct":             [1.0, 0.88, 0.76, "..."],
    "distToNearestAlly":   [12.5, 10.2, 18.4, "..."],
    "distToNearestEnemy":  [62.1, 48.0, 27.3, "..."],
    "distToNearestTower":  [15.0, 19.0, 22.5, "..."]
  },

  "events": {
    "kills":        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "deaths":       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "assists":      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "abilityCasts": [0, 1, 2, 2, 3, 1, 2, 2, 3, 2]
  },

  "alliesNearby":  [0, 1, 1, 2, 1, 0, 1, 1, 2, 1],
  "enemiesNearby": [0, 0, 1, 1, 0, 0, 1, 2, 0, 0]
}
```

### SQLite format (`--store sqlite`)

Two tables:

**`matches`** — one row per match:

| Column | Type | Description |
|---|---|---|
| `match_id` | INTEGER PK | Match ID |
| `duration_seconds` | INTEGER | Full match duration |
| `did_radiant_win` | INTEGER | 1 = Radiant win |
| `avg_mmr` | INTEGER | Average MMR (nullable) |
| `bracket` | INTEGER | Skill bracket integer |
| `game_version_id` | INTEGER | Patch version ID |

**`players`** — one row per player (10 per match):

| Column | Type | Description |
|---|---|---|
| `match_id` | INTEGER FK | References `matches` |
| `steam_account_id` | INTEGER | Steam ID |
| `hero_id` / `hero_name` | INTEGER / TEXT | Hero identity |
| `position` / `lane` / `team` | TEXT | Role metadata |
| `is_victory` | INTEGER | 1 = won |
| `max_gold`, `max_xp`, … | INTEGER | 7 scalar maximums |
| `gold_norm`, `xp_norm`, … | TEXT | JSON arrays (10 floats each) |
| `health_pct`, `mana_pct` | TEXT | JSON arrays (10 floats each) |
| `dist_to_nearest_ally`, … | TEXT | JSON arrays (10 floats each) |
| `kills`, `deaths`, … | TEXT | JSON arrays (10 ints each) |
| `allies_nearby`, `enemies_nearby` | TEXT | JSON arrays (10 ints each) |

Array columns are stored as compact JSON strings and can be queried with SQLite's `json_each()` / `json_extract()`.

---

## Feature Details

### Normalized cumulative curves

Each curve is built from raw STRATZ playback events, summed into per-minute buckets, then converted to a running cumulative sum. The curve is divided by its final value (minute 9) to normalize to `[0, 1]`. The pre-normalization maximum is stored as the corresponding scalar.

| Feature | Source | Construction |
|---|---|---|
| `goldNorm` | `playerUpdateGoldEvents.networth` | Last value per minute (state snapshot), forward-filled |
| `xpNorm` | `experienceEvents.amount` | Sum per minute → cumulative |
| `damageDealtNorm` | `heroDamageEvents` (attacker = this hero, `fromIllusion=false`) | Sum `value` per minute → cumulative |
| `damageTakenNorm` | `heroDamageEvents` (target = this hero, `toIllusion=false`) | Sum `value` per minute → cumulative |
| `csNorm` | `csEvents` | Count events per minute → cumulative |
| `towerDamageNorm` | `towerDamageEvents.damage` | Sum per minute → cumulative |
| `healingNorm` | `healEvents.value` | Sum per minute → cumulative |

### State curves

Sampled at the midpoint of each minute bucket (second `N*60 + 30`). Forward-filled when no event falls in a bucket (e.g. during death).

| Feature | Description |
|---|---|
| `healthPct` | `hp / maxHp` at each minute midpoint |
| `manaPct` | `mp / maxMp` at each minute midpoint |

### Position-derived features

Position is sampled at the midpoint of each minute bucket from `playerUpdatePositionEvents`. All 10 players are evaluated simultaneously so ally/enemy distances can be computed.

Coordinates use STRATZ's compressed map system: `stratz = (world + 8192) / 128`. Distances are in these compressed units (≈ 1 unit per 128 Dota world units, or roughly 1 unit per 11m).

| Feature | Description |
|---|---|
| `distToNearestAlly` | Min distance to any allied hero at each minute |
| `distToNearestEnemy` | Min distance to any enemy hero at each minute |
| `distToNearestTower` | Min distance to any friendly Tier 1 tower at each minute |
| `alliesNearby` | Count of allies within ~1500 world units (~12 compressed units) |
| `enemiesNearby` | Count of enemies within ~1500 world units (~12 compressed units) |

---

## Project Structure

```
fetch_match_ids.py          — OpenDota match ID scraper
extract_match.py            — CLI entry point and orchestration
stratz_client.py            — STRATZ GraphQL client (auth, rate limiting, retry)
feature_builder.py          — Raw STRATZ response → feature dict
store.py                    — Storage backends (JsonStore, SqliteStore)
constants.py                — Tower positions, map constants, shared geometry
hero_names.json             — Cached hero ID → name map (auto-generated)
data/
  matches/                  — JSON output directory (default)
  matches.db                — SQLite database (when using --store sqlite)
failed_matches.txt          — Match IDs that errored during extraction
skipped_matches.txt         — Match IDs skipped (e.g. early GG < 10 min)
```
