"""feature_analysis.py — Univariate feature importance for lane/match outcome prediction.

Computes mutual information (MI) between each feature and each outcome label.
Timeseries features (goldNorm_0..9, etc.) are grouped by their base name and
the max MI across the 10 minute buckets is reported as that feature's score,
alongside which minute was most predictive.

Targets evaluated:
  matchOutcome — win/loss
  position     — in-game role (POSITION_1..5)
  laneOutcome  — win/loss/tie for the player's lane

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

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
LABEL_COLS = ["position", "lane", "team", "matchOutcome", "laneOutcome", "heroName", "bracket"]
TARGETS    = ["matchOutcome", "position", "laneOutcome"]

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
        vals = ts.get(col) or [0.0] * 40
        for i, v in enumerate(vals):
            row[f"{col}_{i}"] = float(v or 0)
    for col in EVENT_FEATURES:
        vals = ev.get(col) or [0] * 40
        for i, v in enumerate(vals):
            row[f"{col}_{i}"] = float(v or 0)
    for col in PROX_FEATURES:
        vals = player.get(col) or [0] * 40
        for i, v in enumerate(vals):
            row[f"{col}_{i}"] = float(v or 0)
    row["position"]     = player.get("position")  or "UNKNOWN"
    row["lane"]         = player.get("lane")       or "UNKNOWN"
    row["team"]         = player.get("team")       or "UNKNOWN"
    row["matchOutcome"] = "Win" if player.get("isVictory") else "Loss"
    row["laneOutcome"]  = player.get("laneOutcome") or "UNKNOWN"
    row["heroName"]     = player.get("heroName")   or "Unknown"
    row["bracket"]      = str(meta.get("bracket")  or "Unknown")
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


def _derive_lane_outcome(lane: str, team: str, bottom, mid, top) -> str:
    if lane == "MID_LANE":
        raw = mid
    elif team == "RADIANT":
        raw = bottom if lane == "SAFE_LANE" else top
    else:
        raw = top if lane == "SAFE_LANE" else bottom

    if not raw or not isinstance(raw, str):
        return "UNKNOWN"
    if raw == "TIE":
        return "Tie"

    winning_team = "RADIANT" if raw.startswith("RADIANT") else "DIRE"
    is_stomp = raw.endswith("STOMP")
    player_won = (winning_team == team)

    if player_won:
        return "Stomp Win" if is_stomp else "Win"
    else:
        return "Stomp Loss" if is_stomp else "Loss"


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
               m.bracket, m.bottom_lane_outcome, m.mid_lane_outcome, m.top_lane_outcome
        FROM players p JOIN matches m ON p.match_id = m.match_id
    """
    df_raw = pd.read_sql_query(query, conn)
    conn.close()
    expanded: dict = {}
    for db_col, feat_name in _TS_DB_MAP.items():
        parsed = df_raw[db_col].apply(
            lambda x: json.loads(x) if isinstance(x, str) else [0.0] * 40
        )
        for i in range(40):
            expanded[f"{feat_name}_{i}"] = parsed.apply(
                lambda v, i=i: float(v[i]) if len(v) > i else 0.0
            )
    for db_col, feat_name in _SCALAR_DB_MAP.items():
        expanded[feat_name] = df_raw[db_col].fillna(0).astype(float)
    expanded["position"]     = df_raw["position"].fillna("UNKNOWN")
    expanded["lane"]         = df_raw["lane"].fillna("UNKNOWN")
    expanded["team"]         = df_raw["team"].fillna("UNKNOWN")
    expanded["matchOutcome"] = df_raw["is_victory"].map({1: "Win", 0: "Loss"}).fillna("Unknown")
    expanded["heroName"]     = df_raw["hero_name"].fillna("Unknown")
    expanded["bracket"]      = df_raw["bracket"].fillna("Unknown").astype(str)
    expanded["laneOutcome"]  = [
        _derive_lane_outcome(lane, team, bot, mid, top)
        for lane, team, bot, mid, top in zip(
            expanded["lane"], expanded["team"],
            df_raw["bottom_lane_outcome"], df_raw["mid_lane_outcome"], df_raw["top_lane_outcome"],
        )
    ]
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

def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no valid data (maxGold == 0 indicates a failed or empty record)."""
    before = len(df)
    df = df[df["maxGold"] > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Filtered %d invalid rows (maxGold == 0), %d remaining.", dropped, len(df))
    return df


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
        counts = np.bincount(y)
        probs = counts / counts.sum()
        h_y = float(-np.sum(probs * np.log(probs + 1e-12)))
        logger.info("Computing MI for target '%s' (%d classes, H=%.3f nats) …", target, len(le.classes_), h_y)
        raw_mi = mutual_info_classif(X, y, random_state=42)
        mi_scores[target] = raw_mi / h_y if h_y > 0 else raw_mi

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


_CLF_TARGETS = ["matchOutcome", "position", "laneOutcome", "heroName"]

_LOGREG_CONFIG = {
    "matchOutcome": {"max_iter": 500,  "C": 1.0},
    "position":     {"max_iter": 500,  "C": 1.0},
    "laneOutcome":  {"max_iter": 500,  "C": 1.0},
    "heroName":     {"max_iter": 1000, "C": 0.5},
}

_XGB_CONFIG = {
    "matchOutcome": {"n_estimators": 400, "max_depth": 5,  "learning_rate": 0.1},
    "position":     {"n_estimators": 400, "max_depth": 6,  "learning_rate": 0.1},
    "laneOutcome":  {"n_estimators": 400, "max_depth": 5,  "learning_rate": 0.1},
    "heroName":     {"n_estimators": 600, "max_depth": 8,  "learning_rate": 0.1},
}


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels).plot(
        ax=ax, colorbar=True, cmap="Blues", values_format=".2f", xticks_rotation="vertical"
    )
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def run_raw_feature_classifiers(
    df: pd.DataFrame,
    out_dir: Path,
    targets: list[str] | None = None,
    models: list[str] | None = None,
    test_split: float = 0.2,
) -> pd.DataFrame:
    """Train logistic regression and XGBoost classifiers on raw features.

    Mirrors the setup in embedding_analysis.py so results are directly
    comparable against the SimCLR embedding classifiers.

    Args:
        df:         Raw player DataFrame (already filtered).
        out_dir:    Directory to save confusion matrix PNGs.
        targets:    Labels to classify. Defaults to matchOutcome, position,
                    laneOutcome, heroName.
        models:     Which classifiers to run: 'logreg', 'xgb', or both.
                    Defaults to both.
        test_split: Fraction of data held out for testing.

    Returns:
        DataFrame with one row per (target, model) and columns for accuracy.
    """
    if targets is None:
        targets = _CLF_TARGETS
    if models is None:
        models = ["logreg", "xgb"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip = set(LABEL_COLS) | {"matchId", "heroName", "team"}
    feature_cols = [c for c in df.columns if c not in skip]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_split, random_state=42,
        stratify=df["position"].values,
    )
    X_train, X_test = X[idx_train], X[idx_test]

    records = []
    for target in targets:
        if target not in df.columns:
            logger.warning("Target '%s' not in dataframe — skipping.", target)
            continue

        y = df[target].astype(str).values
        y_train, y_test = y[idx_train], y[idx_test]

        le = LabelEncoder().fit(y)
        labels = list(le.classes_)
        n_classes = len(labels)
        logger.info("Classifying '%s' (%d classes) …", target, n_classes)

        if "logreg" in models:
            cfg = _LOGREG_CONFIG[target]
            clf = LogisticRegression(max_iter=cfg["max_iter"], C=cfg["C"],
                                     solver="lbfgs", n_jobs=-1)
            logger.info("  [logreg] fitting …")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = float((y_pred == y_test).mean())
            logger.info("  [logreg] accuracy = %.4f", acc)
            print(f"\n── {target}  [logreg]  acc={acc:.4f} ──")
            print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
            _plot_confusion_matrix(
                y_test, y_pred, labels,
                title=f"{target} [logreg — raw features]  acc={acc:.3f}",
                out_path=out_dir / f"raw_confusion_{target}_logreg.png",
            )
            records.append({"target": target, "model": "logreg", "accuracy": acc,
                            "n_classes": n_classes, "n_test": len(y_test)})

        if "xgb" in models:
            cfg = _XGB_CONFIG[target]
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            clf = XGBClassifier(
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"], objective="multi:softmax",
                num_class=n_classes, tree_method="hist", device=device,
                n_jobs=-1, verbosity=0, random_state=42,
            )
            y_train_enc = le.transform(y_train)
            logger.info("  [xgb] fitting (device=%s) …", device)
            clf.fit(X_train, y_train_enc)
            y_pred = le.inverse_transform(clf.predict(X_test))
            acc = float((y_pred == y_test).mean())
            logger.info("  [xgb] accuracy = %.4f", acc)
            print(f"\n── {target}  [xgb]  acc={acc:.4f} ──")
            print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
            _plot_confusion_matrix(
                y_test, y_pred, labels,
                title=f"{target} [xgb — raw features]  acc={acc:.3f}",
                out_path=out_dir / f"raw_confusion_{target}_xgb.png",
            )
            records.append({"target": target, "model": "xgb", "accuracy": acc,
                            "n_classes": n_classes, "n_test": len(y_test)})

    summary = pd.DataFrame(records)
    csv_path = out_dir / "raw_feature_classifier_results.csv"
    summary.to_csv(csv_path, index=False)
    logger.info("Summary saved to %s", csv_path)
    return summary


def plot_timeseries_by_hero(
    df: pd.DataFrame,
    feature: str,
    heroes: list[str],
    out_path: Path | None = None,
) -> None:
    """Plot individual + mean time series curves for each hero side by side.

    Args:
        df:       Raw player DataFrame (output of load_sqlite_store / load_json_store).
        feature:  Base feature name, e.g. 'goldNorm' or 'distToNearestTower'.
        heroes:   List of hero names to plot (one subplot each).
        out_path: If provided, save the figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — cannot plot.")
        return

    # Collect the time-step columns for this feature
    step_cols = sorted(
        [c for c in df.columns if c.startswith(f"{feature}_") and c.split("_")[-1].isdigit()],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not step_cols:
        raise ValueError(f"No time series columns found for feature '{feature}'. "
                         f"Available bases: {sorted({c.rsplit('_',1)[0] for c in df.columns if '_' in c})}")

    n_steps = len(step_cols)
    x = np.arange(n_steps) * 15  # seconds

    n = len(heroes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, hero in zip(axes, heroes):
        vals = df[df["heroName"] == hero][step_cols].values
        subset = vals[~np.all(vals == 0, axis=1)]
        if len(subset) == 0:
            ax.set_title(f"{hero}\n(no data)")
            continue

        for row in subset:
            ax.plot(x, row, color="#4c78a8", alpha=0.08, linewidth=0.8)

        mean_curve = subset.mean(axis=0)
        ax.plot(x, mean_curve, color="#e45756", linewidth=2.2, label="mean")

        ax.set_title(f"{hero}\n(n={len(subset)})", fontsize=9)
        ax.set_xlabel("Seconds", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel(feature, fontsize=9)
    fig.suptitle(f"{feature} — per-hero time series", fontsize=11, y=1.01)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved to %s", out_path)
    plt.show()


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

    fig, ax = plt.subplots(figsize=(max(12, len(feat_labels) * 0.45), 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(target_labels)))
    ax.set_yticklabels(target_labels, fontsize=9)

    for i in range(len(target_labels)):
        for j in range(len(feat_labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=6, color="black" if matrix[i, j] < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="NMI = I(X;Y) / H(Y)", shrink=0.8)
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
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory to save heatmap PNG and CSV (default: figures/).")
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
    df = filter_invalid_rows(df)
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
