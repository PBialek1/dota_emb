"""feature_analysis.py — Univariate feature importance for lane/match outcome prediction.

Computes mutual information (MI) between each feature and each outcome label.
Timeseries features (goldNorm_0..9, etc.) are grouped by their base name and
the max MI across the 10 minute buckets is reported as that feature's score,
alongside which minute was most predictive.

Targets evaluated:
  isVictory  — win/loss
  position   — in-game role (POSITION_1..5)
  lane       — safe/mid/off/jungle/roam
  bracket    — MMR skill bracket

Outputs:
  - Ranked tables printed to stdout
  - Heatmap PNG saved alongside --output-dir (default: ./feature_importance.png)
  - CSV of all scores saved as feature_importance.csv

Usage:
    python visualization/feature_analysis.py --data ./data/matches
    python visualization/feature_analysis.py --data ./data/matches.db --top 20
    python visualization/feature_analysis.py --data ./data/matches --output-dir ./analysis
"""
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature / label definitions (mirrors umap_explorer.py)
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
LABEL_COLS = ["position", "lane", "team", "isVictory", "heroName", "bracket"]
TARGETS    = ["isVictory", "position", "lane", "bracket"]

# ---------------------------------------------------------------------------
# Data loading (mirrors umap_explorer.py)
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
    row["position"]  = player.get("position")  or "UNKNOWN"
    row["lane"]      = player.get("lane")       or "UNKNOWN"
    row["team"]      = player.get("team")       or "UNKNOWN"
    row["isVictory"] = "Win" if player.get("isVictory") else "Loss"
    row["heroName"]  = player.get("heroName")   or "Unknown"
    row["bracket"]   = str(meta.get("bracket")  or "Unknown")
    return row


def load_json_store(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.json"))
    logger.info("Loading %d JSON files …", len(files))
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_bytes())
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)
            continue
        meta = data.get("meta", {})
        for player in data.get("players", []):
            rows.append(_flatten_player_json(player, meta))
    return pd.DataFrame(rows)


_TS_DB_MAP = {
    "gold_norm": "goldNorm", "xp_norm": "xpNorm",
    "damage_dealt_norm": "damageDealtNorm", "damage_taken_norm": "damageTakenNorm",
    "cs_norm": "csNorm", "tower_damage_norm": "towerDamageNorm",
    "healing_norm": "healingNorm", "health_pct": "healthPct", "mana_pct": "manaPct",
    "dist_to_nearest_ally": "distToNearestAlly", "dist_to_nearest_enemy": "distToNearestEnemy",
    "dist_to_nearest_tower": "distToNearestTower",
    "kills": "kills", "deaths": "deaths", "assists": "assists",
    "ability_casts": "abilityCasts", "allies_nearby": "alliesNearby",
    "enemies_nearby": "enemiesNearby",
}
_SCALAR_DB_MAP = {
    "max_gold": "maxGold", "max_xp": "maxXp",
    "max_damage_dealt": "maxDamageDealt", "max_damage_taken": "maxDamageTaken",
    "max_cs": "maxCs", "max_tower_damage": "maxTowerDamage", "max_healing": "maxHealing",
}


def load_sqlite_store(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT p.position, p.lane, p.team, p.is_victory, p.hero_name,
               p.max_gold, p.max_xp, p.max_damage_dealt, p.max_damage_taken,
               p.max_cs, p.max_tower_damage, p.max_healing,
               p.gold_norm, p.xp_norm, p.damage_dealt_norm, p.damage_taken_norm,
               p.cs_norm, p.tower_damage_norm, p.healing_norm,
               p.health_pct, p.mana_pct,
               p.dist_to_nearest_ally, p.dist_to_nearest_enemy, p.dist_to_nearest_tower,
               p.kills, p.deaths, p.assists, p.ability_casts,
               p.allies_nearby, p.enemies_nearby,
               m.bracket
        FROM players p JOIN matches m ON p.match_id = m.match_id
    """
    df_raw = pd.read_sql_query(query, conn)
    conn.close()
    expanded: dict = {}
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
    return pd.DataFrame(expanded)


# ---------------------------------------------------------------------------
# Feature grouping helper
# ---------------------------------------------------------------------------

def _base_name(col: str) -> str:
    """'goldNorm_3' → 'goldNorm';  'maxGold' → 'maxGold'."""
    parts = col.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return col


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      feature, base_feature, minute (None for scalars),
      isVictory_mi, position_mi, lane_mi, bracket_mi
    """
    skip = set(LABEL_COLS) | {"matchId", "heroName", "team"}
    feature_cols = [c for c in df.columns if c not in skip]

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    le = LabelEncoder()
    mi_scores: dict[str, np.ndarray] = {}
    for target in TARGETS:
        if target not in df.columns:
            logger.warning("Target '%s' not in dataframe — skipping.", target)
            continue
        y = le.fit_transform(df[target].astype(str))
        logger.info("Computing MI for target '%s' (%d classes) …", target, len(le.classes_))
        mi_scores[target] = mutual_info_classif(X, y, random_state=42)

    rows = []
    for j, col in enumerate(feature_cols):
        base = _base_name(col)
        minute = int(col.rsplit("_", 1)[1]) if base != col else None
        row = {"feature": col, "base_feature": base, "minute": minute}
        for target, scores in mi_scores.items():
            row[f"{target}_mi"] = float(scores[j])
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_base(df_mi: pd.DataFrame) -> pd.DataFrame:
    """
    For each base feature, report:
      - max MI across minute buckets (and which minute achieved it)
      - mean MI across minute buckets
    Scalars (minute=None) pass through unchanged.
    """
    mi_cols = [c for c in df_mi.columns if c.endswith("_mi")]
    records = []

    for base, group in df_mi.groupby("base_feature", sort=False):
        if group["minute"].isna().all():
            # Scalar feature — single row
            row = group.iloc[0]
            rec = {"feature": base, "is_timeseries": False, "best_minute": None}
            for mc in mi_cols:
                rec[mc] = row[mc]
                rec[mc.replace("_mi", "_mi_mean")] = row[mc]
            records.append(rec)
        else:
            # Timeseries — aggregate over minutes
            rec = {"feature": base, "is_timeseries": True}
            for mc in mi_cols:
                best_idx = group[mc].idxmax()
                rec[mc]                          = group.loc[best_idx, mc]       # max MI
                rec[mc.replace("_mi", "_mi_mean")] = group[mc].mean()           # mean MI
            # Best minute is the one with highest sum of MI across all targets
            group = group.copy()
            group["total_mi"] = sum(group[mc] for mc in mi_cols)
            rec["best_minute"] = int(group.loc[group["total_mi"].idxmax(), "minute"])
            records.append(rec)

    return pd.DataFrame(records)


def print_ranking(df_agg: pd.DataFrame, target: str, top_n: int) -> None:
    mi_col = f"{target}_mi"
    if mi_col not in df_agg.columns:
        return
    ranked = df_agg[["feature", "is_timeseries", "best_minute", mi_col]] \
        .sort_values(mi_col, ascending=False) \
        .head(top_n) \
        .reset_index(drop=True)
    ranked.index += 1

    print(f"\n{'─'*55}")
    print(f"  Top {top_n} features → {target}")
    print(f"{'─'*55}")
    print(f"  {'#':>3}  {'Feature':<28}  {'MI':>7}  {'Best min':>8}")
    print(f"  {'─'*3}  {'─'*28}  {'─'*7}  {'─'*8}")
    for rank, r in ranked.iterrows():
        min_str = f"min {int(r['best_minute'])}" if r["is_timeseries"] else "scalar"
        print(f"  {rank:>3}  {r['feature']:<28}  {r[mi_col]:>7.4f}  {min_str:>8}")


def save_heatmap(df_agg: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not installed — skipping heatmap.")
        return

    mi_cols = [c for c in df_agg.columns if c.endswith("_mi") and "_mi_mean" not in c]
    target_labels = [c.replace("_mi", "") for c in mi_cols]

    # Sort features by sum of MI across all targets
    df_plot = df_agg.copy()
    df_plot["total"] = df_plot[mi_cols].sum(axis=1)
    df_plot = df_plot.sort_values("total", ascending=False).head(30)

    matrix = df_plot[mi_cols].values.T           # (n_targets, n_features)
    feat_labels = df_plot["feature"].tolist()

    # Normalize each target row to [0, 1] for comparable colour scale
    row_max = matrix.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1, row_max)
    matrix_norm = matrix / row_max

    fig, ax = plt.subplots(figsize=(max(12, len(feat_labels) * 0.45), 3.5))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(target_labels)))
    ax.set_yticklabels(target_labels, fontsize=9)

    # Annotate cells with raw MI values
    for i in range(len(target_labels)):
        for j in range(len(feat_labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=6, color="black" if matrix_norm[i, j] < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Normalised MI (per target)", shrink=0.8)
    ax.set_title("Feature importance (mutual information) — top 30 features by total MI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Heatmap saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Univariate feature importance analysis for laning outcome prediction."
    )
    parser.add_argument("--data", required=True,
                        help="Path to JSON match directory or SQLite .db file.")
    parser.add_argument("--top",  type=int, default=15,
                        help="Number of top features to print per target (default: 15).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save heatmap PNG and CSV (default: current dir).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if data_path.is_dir():
        df = load_json_store(data_path)
    elif data_path.suffix in {".db", ".sqlite", ".sqlite3"}:
        df = load_sqlite_store(data_path)
    else:
        logger.error("--data must be a directory or .db file.")
        sys.exit(1)

    logger.info("Loaded %d player rows.", len(df))
    if df.empty:
        logger.error("No data loaded.")
        sys.exit(1)

    df_mi  = compute_importance(df)
    df_agg = aggregate_by_base(df_mi)

    for target in TARGETS:
        print_ranking(df_agg, target, args.top)

    csv_path = out_dir / "feature_importance.csv"
    df_agg.to_csv(csv_path, index=False)
    logger.info("Full scores saved to %s", csv_path)

    save_heatmap(df_agg, out_dir / "feature_importance.png")


if __name__ == "__main__":
    main()
