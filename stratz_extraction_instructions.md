# Instructions: STRATZ API Feature Extraction for Dota 2 Laning Embeddings

## Context

You are writing a Python script to extract laning stage features from the STRATZ GraphQL API
(https://api.stratz.com/graphql) and store them as local files, one per match.

STRATZ requires a Bearer token passed as `Authorization: Bearer <TOKEN>` in the request header.
The script should read the token from an environment variable `STRATZ_TOKEN`.

---

## Output Format

Store each match as a single JSON file named `{matchId}.json` in a configurable output directory.
The file should have two top-level keys:

```json
{
  "meta": { ... },
  "players": [ ... ]
}
```

### `meta` object

```json
{
  "matchId": 7123456789,
  "durationSeconds": 2340,
  "didRadiantWin": true,
  "avgMmr": 4500,
  "bracket": "LEGEND",
  "gameVersionId": 184
}
```

### `players` array

One object per player (10 total). Each object:

```json
{
  "steamAccountId": 123456,
  "heroId": 48,
  "heroName": "Luna",
  "position": "POSITION_1",
  "lane": "SAFE_LANE",
  "team": "RADIANT",
  "isVictory": true,
  "laneOutcome": "WIN",

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
    "goldNorm":         [0.0, 0.05, 0.12, ...],
    "xpNorm":           [0.0, 0.04, 0.10, ...],
    "damageDealtNorm":  [0.0, 0.01, 0.04, ...],
    "damageTakenNorm":  [0.0, 0.02, 0.05, ...],
    "csNorm":           [0.0, 0.06, 0.14, ...],
    "towerDamageNorm":  [0.0, 0.0,  0.0,  ...],
    "healingNorm":      [0.0, 0.0,  0.0,  ...],
    "healthPct":        [1.0, 0.92, 0.85, ...],
    "manaPct":          [1.0, 0.88, 0.76, ...],
    "distToNearestAlly":  [300, 280, 450, ...],
    "distToNearestEnemy": [800, 600, 350, ...],
    "distToNearestTower": [200, 250, 300, ...]
  },

  "events": {
    "kills":   [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "deaths":  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "assists": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "denies":  [0, 2, 3, 1, 0, 0, 0, 1, 0, 0],
    "abilityCastsOffensive": [0, 1, 2, 2, 3, 1, 2, 2, 3, 2],
    "abilityCastsDefensive": [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    "abilityCastsMobility":  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "wardsPlaced":           [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
  },

  "alliesNearby": [0, 1, 1, 2, 1, 0, 1, 1, 2, 1],
  "enemiesNearby": [0, 0, 1, 1, 0, 0, 1, 2, 0, 0]
}
```

All timeseries and event arrays have exactly **10 elements**, one per minute of the laning stage
(minute 0–9, where minute N = the interval [N, N+1) minutes).

---

## Timeseries Construction Rules

All event streams from `playbackData` are unsorted lists of timestamped events. The first step
for every feature is to sort events by `time` (in seconds) and filter to `time < 600` (10 minutes).

### Cumulative curves (gold, xp, damageDealt, damageTaken, towerDamage, healing)

Build by summing event values within each minute bucket [N*60, (N+1)*60):

**Gold** — from `playerUpdateGoldEvents`: take the last `networth` value within each minute bucket.
This is a state snapshot, not a delta. Treat as cumulative directly.

**XP** — from `experienceEvents`: sum `amount` values within each minute to get per-minute XP,
then compute running cumulative sum.

**Damage dealt** — from `heroDamageEvents` where `isSourceMainHero = true` AND `fromIllusion = false`:
sum `value` per minute, then running cumulative.

**Damage taken** — from `heroDamageEvents` where `isTargetMainHero = true` AND `toIllusion = false`:
sum `value` per minute, then running cumulative.

**Tower damage** — from `towerDamageEvents`: sum `damage` per minute, then running cumulative.

**Healing** — from `healEvents` where `attacker` matches this player's slot:
sum `value` per minute, then running cumulative. This captures healing *dealt* by the player,
which is more behaviorally meaningful than healing received.

After building each cumulative curve:
1. Record `max = curve[9]` as the scalar.
2. Normalize: `normCurve[n] = curve[n] / max` (if max == 0, all zeros).

**CS** — from `csEvents`: count events per minute to get per-minute CS, then running cumulative.
Apply the same normalize-and-store-max treatment.

### State curves (healthPct, manaPct)

From `playerUpdateHealthEvents`: for each minute N, find the event with `time` closest to `N*60 + 30`
(midpoint of the bucket). Compute:
- `healthPct[n] = hp / maxHp`
- `manaPct[n] = mp / maxMp`

If no event exists within a minute bucket, forward-fill from the previous bucket.

### Position-derived features

From `playerUpdatePositionEvents`: for each minute N, find the event with `time` closest to
`N*60 + 30`. Use that (x, y) for all position-derived calculations.

**distToNearestAlly / distToNearestEnemy** — compute across all 10 players simultaneously.
For each minute bucket, collect one (x, y) snapshot per player, then compute pairwise distances.
Nearest ally = min distance to any allied hero (excluding self).
Nearest enemy = min distance to any enemy hero.

**distToNearestTower** — compute from the player's snapshot position to the hardcoded tower
positions for their team (see constants below). Use minimum distance across all friendly towers.

**alliesNearby / enemiesNearby** — count of allied/enemy heroes within 1500 units at each
minute midpoint. Uses the same per-minute position snapshots.

### Event features (kills, deaths, assists, ability casts)

Count events within each minute window [N*60, (N+1)*60):

- **kills**: `killEvents` where `attacker` matches this player's slot index
- **deaths**: `deathEvents` where `target` matches this player's slot index
- **assists**: `assistEvents` are emitted once per assisting player — count events within each bucket
- **abilityCastsOffensive / Defensive / Mobility**: from `abilityUsedEvents`, classify each
  `abilityId` using the pre-built lookup table (see Ability Classification below), then count
  per category per minute

### Ability classification

Build a lookup `dict[abilityId → category]` at startup:

1. Fetch `https://api.stratz.com/api/v1/Ability` and cache to `ability_classifications.json`
2. Use the `attacker` and `target` slot indices in `abilityUsedEvents` as a runtime signal:
   - If `target` slot is an enemy: weight toward Offensive
   - If `target` slot is an ally or self: weight toward Defensive
   - If ability metadata indicates movement (blink, leap, dash): Mobility
3. For AOE abilities with no target, rely entirely on metadata classification
4. Unclassifiable abilities → count in a fourth bucket `abilityCastsOther` (store in output but
   note it may be noisy)

---

## STRATZ GraphQL Query

All per-event data lives under `players.playbackData` as a `MatchPlayerPlaybackDataType`.
Use a single query per match:

```graphql
query GetMatch($matchId: Long!) {
  match(id: $matchId) {
    id
    durationSeconds
    didRadiantWin
    bracket
    gameVersionId
    averageImp

    players {
      steamAccountId
      heroId
      position
      lane
      isRadiant
      isVictory
      laneOutcome

      playbackData {
        playerUpdatePositionEvents { time x y }

        playerUpdateHealthEvents { time hp maxHp mp maxMp }

        playerUpdateGoldEvents { time gold networth }

        abilityUsedEvents { time abilityId attacker target }

        killEvents   { time attacker target }
        deathEvents  { time attacker target goldFed timeDead }
        assistEvents { time target }

        csEvents {
          time npcId isCreep isNeutral isAncient
        }

        heroDamageEvents {
          time value isSourceMainHero isTargetMainHero fromIllusion toIllusion
        }

        towerDamageEvents { time damage }

        healEvents { time attacker target value }

        experienceEvents { time amount }

        runeEvents { time rune action }
      }
    }
  }
}
```

### Field notes

- **`playerUpdatePositionEvents`**: sampled position stream — NOT every tick, but frequent enough
  to interpolate per-minute snapshots. Use the event closest to the midpoint of each minute bucket.
- **`playerUpdateHealthEvents`**: sampled hp/mana stream. Same interpolation approach as position.
  Fields are `hp`, `maxHp`, `mp`, `maxMp` — compute `healthPct = hp / maxHp`.
- **`playerUpdateGoldEvents`**: use `gold` (reliable gold) to build the cumulative gold curve.
  `networth` is also available and may be more meaningful — use `networth` for the gold feature.
- **`heroDamageEvents`**: filter by `isSourceMainHero = true` for damage dealt, `isTargetMainHero = true`
  for damage taken. Exclude events where `fromIllusion = true` or `toIllusion = true` to count
  only main hero interactions.
- **`csEvents`**: each event is one last-hit. Count events per minute for the CS curve. 
  Denies are **not available** in playback data — omit the denies feature or derive it separately
  (see Notes section).
- **`abilityUsedEvents`**: use `abilityId` for classification. The `attacker` and `target` fields
  are player slot indices — use them to distinguish self-casts, ally-casts, and enemy-casts to 
  help with offensive/defensive classification.
- **`runeEvents`**: `action` distinguishes picked up vs denied. Use `action = PICKUP` events
  as a proxy for rune contest behavior (optional feature, not in core output schema).

---

## Positional Calculations

STRATZ returns x/y in Dota 2 map coordinates. To compute distances:

```python
import math

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
```

**Tower positions** — hardcode the approximate coordinates for the Tier 1 towers for each lane.
These are stable across patches. Use the nearest T1 tower for each player at each timestep.

Approximate T1 tower positions (Dota 2 world units):
```python
TOWER_POSITIONS = {
    "radiant_top_t1":    (2048, 6400),
    "radiant_mid_t1":    (4096, 4096),
    "radiant_bot_t1":    (6400, 2048),
    "dire_top_t1":       (9600, 13952),
    "dire_mid_t1":       (11904, 11904),
    "dire_bot_t1":       (13952, 9600),
}
```

For each player at each timestep, find the minimum distance to any tower on their side.

---

## Script Structure

```
extract_match.py        — CLI entry point: accepts matchId(s), fetches and saves
stratz_client.py        — GraphQL client with rate limiting and retry logic
feature_builder.py      — Transforms raw STRATZ response into the output schema
ability_classifier.py   — Maps abilityId → offensive/defensive/mobility
constants.py            — Tower positions, lane boundary definitions
```

### CLI usage
```
python extract_match.py --match-id 7123456789 --output-dir ./data/matches
python extract_match.py --match-ids-file match_ids.txt --output-dir ./data/matches
```

### Rate limiting
STRATZ free tier allows 300 requests/hour. Add a configurable `--rate-limit` flag defaulting to
280/hour. Use a token bucket or simple sleep between requests.

### Error handling
- If a match is not found or the API returns an error, log the matchId to `failed_matches.txt`
  and continue — do not crash the whole run.
- If a match is shorter than 10 minutes (early GG), skip it and log to `skipped_matches.txt`
  with reason "duration < 600s".
- If per-minute arrays from STRATZ have fewer than 10 entries, pad with the last known value
  for state features (healthPct, manaPct, positions) and with zeros for event/cumulative features.

---

## Notes and Caveats

- **laneOutcome**: STRATZ provides this directly as an enum (WIN, LOSS, TIE, UNKNOWN). Store as-is.
- **avgMmr**: STRATZ may not always have MMR data. Store null if unavailable.
- **heroName**: Resolve from heroId using the STRATZ `constants/heroes` endpoint and cache locally.
- **Denies**: `csEvents` does not include deny information in `MatchPlayerPlaybackDataType`.
  Denies are omitted from the core feature set. If needed, they may be available via the
  `match.players.stats` endpoint (separate query) — implement as an optional enrichment pass.
- **Ward placements**: Ward placement events are **not present** in `MatchPlayerPlaybackDataType`.
  Remove `wardsPlaced` from the output schema. As a weak proxy, `purchaseEvents` with observer/
  sentry ward item IDs can indicate ward-buying behavior if desired, but is not in the core schema.
- **Ability classification**: Build the classifier once by fetching the full ability list from
  STRATZ constants, classifying by behavior flags, and caching to a local JSON file. The
  `attacker`/`target` slot indices in `abilityUsedEvents` provide runtime signal to disambiguate.
- **Player slot indices**: STRATZ uses slot 0–4 for Radiant, 128–132 (or 5–9 depending on API
  version) for Dire. Verify how slot indices map to `isRadiant` before doing ally/enemy classification.
- **Position event density**: `playerUpdatePositionEvents` may have gaps during deaths (hero is
  dead, no position updates). Forward-fill from last known position during dead time, or use the
  death/respawn events to detect and flag these gaps explicitly.
- **Do not store raw positional data** — only the derived distance features. Raw x/y is not
  needed downstream and bloats the files.
- Use `orjson` for serialization — it is significantly faster than stdlib `json` for large batches.
