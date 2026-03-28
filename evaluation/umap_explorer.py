"""umap_explorer.py — Interactive Panel/HoloViz UMAP of laning features.

Loads a dataset from either a JSON store (directory of per-match .json files)
or a SQLite store (.db), projects all player feature vectors with UMAP, and
serves an interactive scatter plot where points can be coloured by role
(position), lane, team, win/loss, hero, or MMR bracket.

Usage:
    python umap_explorer.py --data ./data/matches            # JSON store
    python umap_explorer.py --data ./data/matches.db         # SQLite store
    python umap_explorer.py --data ./data/matches.db --max-players 5000
    python umap_explorer.py --data ./data/matches --n-neighbors 30 --min-dist 0.05
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import opts
from sklearn.preprocessing import StandardScaler

import umap

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature / label column definitions
# ---------------------------------------------------------------------------

TS_FEATURES = [
    "goldNorm", "xpNorm", "damageDealtNorm", "damageTakenNorm",
    "csNorm", "towerDamageNorm", "healingNorm", "healthPct", "manaPct",
    "distToNearestAlly", "distToNearestEnemy", "distToNearestTower",
]
EVENT_FEATURES = ["kills", "deaths", "assists", "abilityCasts"]
PROX_FEATURES  = ["alliesNearby", "enemiesNearby"]
SCALAR_FEATURES = [
    "maxGold", "maxXp", "maxDamageDealt", "maxDamageTaken",
    "maxCs", "maxTowerDamage", "maxHealing",
]

LABEL_COLS = ["position", "lane", "team", "isVictory", "heroName", "bracket", "laneOutcome"]

# Colour options shown in the widget, in display order
COLOR_OPTIONS = ["position", "lane", "team", "isVictory", "laneOutcome", "heroName", "bracket"]

# Bokeh categorical palette per colour dimension
_CMAP = {
    "position":    "Category10",
    "lane":        "Category10",
    "team":        "Category10",
    "isVictory":   "Category10",
    "laneOutcome": "Category10",
    "heroName":    "Category20",
    "bracket":     "Category10",
}


def _derive_lane_outcome(lane: str, team: str, bottom: str | None, mid: str | None, top: str | None) -> str:
    """Map match-level lane outcomes to the player's perspective (Win/Stomp/Loss/Tie)."""
    if lane == "MID_LANE":
        raw = mid
    elif team == "RADIANT":
        raw = bottom if lane == "SAFE_LANE" else top
    else:  # DIRE
        raw = top if lane == "SAFE_LANE" else bottom

    if not raw or not isinstance(raw, str):
        return "UNKNOWN"

    if raw == "TIE":
        return "Tie"

    winning_team = "RADIANT" if raw.startswith("RADIANT") else "DIRE"
    is_stomp     = raw.endswith("STOMP")
    player_won   = (winning_team == team)

    if player_won:
        return "Stomp Win" if is_stomp else "Win"
    else:
        return "Stomp Loss" if is_stomp else "Loss"


# ---------------------------------------------------------------------------
# Data loading — JSON store
# ---------------------------------------------------------------------------

def _flatten_player_json(player: dict, meta: dict) -> dict:
    ts = player.get("timeseries", {})
    ev = player.get("events", {})
    sc = player.get("scalars", {})
    row: dict = {}

    for col in SCALAR_FEATURES:
        row[col] = float(sc.get(col) or 0)

    for col in TS_FEATURES:
        vals = ts.get(col) or [0.0] * 10
        for i, v in enumerate(vals):
            row[f"{col}_{i}"] = float(v or 0)

    for col in EVENT_FEATURES:
        vals = ev.get(col) or [0] * 10
        for i, v in enumerate(vals):
            row[f"{col}_{i}"] = float(v or 0)

    for col in PROX_FEATURES:
        vals = player.get(col) or [0] * 10
        for i, v in enumerate(vals):
            row[f"{col}_{i}"] = float(v or 0)

    lane = player.get("lane") or "UNKNOWN"
    team = player.get("team") or "UNKNOWN"
    row["position"]    = player.get("position") or "UNKNOWN"
    row["lane"]        = lane
    row["team"]        = team
    row["isVictory"]   = "Win" if player.get("isVictory") else "Loss"
    row["heroName"]    = player.get("heroName")  or "Unknown"
    row["bracket"]     = str(meta.get("bracket") or "Unknown")
    row["laneOutcome"] = _derive_lane_outcome(
        lane, team,
        meta.get("bottomLaneOutcome"),
        meta.get("midLaneOutcome"),
        meta.get("topLaneOutcome"),
    )
    row["matchId"]     = meta.get("matchId")
    return row


def load_json_store(data_dir: Path, max_players: int | None) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.json"))
    logger.info("Loading %d JSON files from %s …", len(files), data_dir)
    rows: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_bytes())
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)
            continue
        meta = data.get("meta", {})
        for player in data.get("players", []):
            rows.append(_flatten_player_json(player, meta))
        if max_players and len(rows) >= max_players:
            break
    return pd.DataFrame(rows[:max_players] if max_players else rows)


# ---------------------------------------------------------------------------
# Data loading — SQLite store
# ---------------------------------------------------------------------------

_TS_DB_MAP = {
    "gold_norm":              "goldNorm",
    "xp_norm":                "xpNorm",
    "damage_dealt_norm":      "damageDealtNorm",
    "damage_taken_norm":      "damageTakenNorm",
    "cs_norm":                "csNorm",
    "tower_damage_norm":      "towerDamageNorm",
    "healing_norm":           "healingNorm",
    "health_pct":             "healthPct",
    "mana_pct":               "manaPct",
    "dist_to_nearest_ally":   "distToNearestAlly",
    "dist_to_nearest_enemy":  "distToNearestEnemy",
    "dist_to_nearest_tower":  "distToNearestTower",
    "kills":                  "kills",
    "deaths":                 "deaths",
    "assists":                "assists",
    "ability_casts":          "abilityCasts",
    "allies_nearby":          "alliesNearby",
    "enemies_nearby":         "enemiesNearby",
}

_SCALAR_DB_MAP = {
    "max_gold":           "maxGold",
    "max_xp":             "maxXp",
    "max_damage_dealt":   "maxDamageDealt",
    "max_damage_taken":   "maxDamageTaken",
    "max_cs":             "maxCs",
    "max_tower_damage":   "maxTowerDamage",
    "max_healing":        "maxHealing",
}


def load_sqlite_store(db_path: Path, max_players: int | None) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    limit = f"LIMIT {max_players}" if max_players else ""
    query = f"""
        SELECT
            p.position, p.lane, p.team, p.is_victory, p.hero_name,
            p.max_gold, p.max_xp, p.max_damage_dealt, p.max_damage_taken,
            p.max_cs, p.max_tower_damage, p.max_healing,
            p.gold_norm, p.xp_norm, p.damage_dealt_norm, p.damage_taken_norm,
            p.cs_norm, p.tower_damage_norm, p.healing_norm,
            p.health_pct, p.mana_pct,
            p.dist_to_nearest_ally, p.dist_to_nearest_enemy, p.dist_to_nearest_tower,
            p.kills, p.deaths, p.assists, p.ability_casts,
            p.allies_nearby, p.enemies_nearby,
            m.bracket, m.match_id,
            m.bottom_lane_outcome, m.mid_lane_outcome, m.top_lane_outcome
        FROM players p
        JOIN matches m ON p.match_id = m.match_id
        {limit}
    """
    df_raw = pd.read_sql_query(query, conn)
    conn.close()

    expanded: dict[str, object] = {}

    for db_col, feat_name in _TS_DB_MAP.items():
        parsed = df_raw[db_col].apply(
            lambda x: json.loads(x) if isinstance(x, str) else [0.0] * 10
        )
        for i in range(10):
            expanded[f"{feat_name}_{i}"] = parsed.apply(
                lambda v, i=i: float(v[i]) if len(v) > i else 0.0
            )

    for db_col, feat_name in _SCALAR_DB_MAP.items():
        expanded[feat_name] = df_raw[db_col].fillna(0).astype(float)

    expanded["position"]  = df_raw["position"].fillna("UNKNOWN")
    expanded["lane"]      = df_raw["lane"].fillna("UNKNOWN")
    expanded["team"]      = df_raw["team"].fillna("UNKNOWN")
    expanded["isVictory"] = df_raw["is_victory"].map({1: "Win", 0: "Loss"}).fillna("Unknown")
    expanded["heroName"]  = df_raw["hero_name"].fillna("Unknown")
    expanded["bracket"]   = df_raw["bracket"].fillna("Unknown").astype(str)
    expanded["laneOutcome"] = [
        _derive_lane_outcome(lane, team, bottom, mid, top)
        for lane, team, bottom, mid, top in zip(
            expanded["lane"],
            expanded["team"],
            df_raw["bottom_lane_outcome"],
            df_raw["mid_lane_outcome"],
            df_raw["top_lane_outcome"],
        )
    ]
    expanded["matchId"]   = df_raw["match_id"]

    return pd.DataFrame(expanded)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

# Expanded timeseries column names (e.g. "goldNorm_0" … "goldNorm_9")
_TS_COLS = {
    f"{feat}_{i}"
    for feat in TS_FEATURES + EVENT_FEATURES + PROX_FEATURES
    for i in range(10)
}


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    skip = set(LABEL_COLS) | {"matchId"} | _TS_COLS
    feature_cols = [c for c in df.columns if c not in skip]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    logger.info("Running UMAP on %d × %d matrix …", X.shape[0], X.shape[1])
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        verbose=True,
    )
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Panel app
# ---------------------------------------------------------------------------

def make_app(df: pd.DataFrame, embedding: np.ndarray) -> pn.viewable.Viewable:
    df = df.copy()
    df["umap_x"] = embedding[:, 0]
    df["umap_y"] = embedding[:, 1]

    color_select = pn.widgets.Select(
        name="Color by",
        options=COLOR_OPTIONS,
        value="position",
        width=200,
    )
    alpha_slider = pn.widgets.FloatSlider(
        name="Opacity", start=0.05, end=1.0, step=0.05, value=0.55, width=200,
    )
    size_slider = pn.widgets.IntSlider(
        name="Point size", start=2, end=14, step=1, value=5, width=200,
    )

    info = pn.pane.Markdown(
        f"**{len(df):,} players** · {df['matchId'].nunique():,} matches",
        width=200,
    )

    def _plot(color_by, alpha, size):
        hover_cols = [color_by, "heroName", "lane", "team", "isVictory", "laneOutcome", "matchId"]
        # deduplicate while preserving order
        seen: set[str] = set()
        vdims: list[str] = []
        for c in hover_cols:
            if c not in seen:
                seen.add(c)
                vdims.append(c)

        return hv.Points(
            df,
            kdims=["umap_x", "umap_y"],
            vdims=vdims,
        ).opts(
            opts.Points(
                color=color_by,
                cmap=_CMAP.get(color_by, "Category10"),
                alpha=alpha,
                size=size,
                tools=["hover", "box_select", "lasso_select", "reset"],
                width=1000,
                height=1000,
                legend_position="right",
                show_legend=True,
                title=f"Laning UMAP — coloured by {color_by}",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                toolbar="above",
            )
        )

    plot_pane = pn.bind(_plot, color_select, alpha_slider, size_slider)

    controls = pn.Column(
        pn.pane.Markdown("## Controls"),
        info,
        pn.layout.Divider(),
        color_select,
        alpha_slider,
        size_slider,
        width=220,
        margin=(10, 20, 10, 10),
    )

    return pn.Row(controls, pn.panel(plot_pane))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive UMAP explorer for Dota 2 laning features."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to JSON match directory or SQLite .db file.",
    )
    parser.add_argument(
        "--max-players", type=int, default=None,
        help="Cap on player rows loaded (useful for quick iteration).",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15,
        help="UMAP n_neighbors (default: 15).",
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1,
        help="UMAP min_dist (default: 0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=5006,
        help="Panel server port (default: 5006).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if data_path.is_dir():
        df = load_json_store(data_path, args.max_players)
    elif data_path.suffix in {".db", ".sqlite", ".sqlite3"}:
        df = load_sqlite_store(data_path, args.max_players)
    else:
        raise ValueError("--data must be a directory (JSON store) or a .db file (SQLite store).")

    logger.info("Loaded %d player rows.", len(df))
    if df.empty:
        logger.error("No data loaded — check your --data path.")
        return

    X = build_feature_matrix(df)
    embedding = run_umap(X, args.n_neighbors, args.min_dist)

    app = make_app(df, embedding)
    pn.serve(app, port=args.port, show=True, title="Dota 2 Laning UMAP")


if __name__ == "__main__":
    main()
