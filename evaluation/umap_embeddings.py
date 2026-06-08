"""umap_embeddings.py — Interactive UMAP of trained SimCLR embeddings.

Loads the trained encoder from a checkpoint, runs all players through it to
get 256-dim embeddings, projects to 2-D with UMAP, and serves an interactive
Panel scatter plot coloured by position, hero, lane, lane outcome, etc.

Usage:
    python evaluation/umap_embeddings.py --data ./data/matches.db
    python evaluation/umap_embeddings.py --data ./data/matches.db --checkpoint ./checkpoints/checkpoint_best.pt
    python evaluation/umap_embeddings.py --data ./data/matches.db --max-players 5000 --port 5006
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import opts
from bokeh.palettes import Category10, Category20
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from model import SimCLRModel
from dataset import LaningDataset

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
# Label helpers
# ---------------------------------------------------------------------------

COLOR_OPTIONS = ["position", "lane", "laneOutcome", "isVictory", "heroName", "bracket", "team"]

_PALETTE = {
    "position":    list(Category10[10]),
    "lane":        list(Category10[10]),
    "laneOutcome": list(Category10[10]),
    "isVictory":   list(Category10[10]),
    "heroName":    list(Category20[20]),
    "bracket":     list(Category10[10]),
    "team":        list(Category10[10]),
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


# ---------------------------------------------------------------------------
# Unified data loader — features (from cache) + labels in aligned row order
# ---------------------------------------------------------------------------


def _normalise_positions(raw: list[str]) -> list[str]:
    """Accept '1'-'5' or 'POSITION_1'-'POSITION_5' and return the canonical DB form."""
    out = []
    for v in raw:
        v = v.strip()
        if v.isdigit():
            out.append(f"POSITION_{v}")
        else:
            out.append(v.upper())
    return out


def _load_labels(db_path: Path, player_ids: np.ndarray) -> pd.DataFrame:
    """Fetch label columns for every cached sample and return them reordered
    to match `player_ids`. One SELECT — no `IN (?, ?, ...)` to avoid the
    SQLite parameter limit at ~100k samples.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        df_all = pd.read_sql_query(
            """
            SELECT p.id AS player_id,
                   p.position, p.lane, p.team, p.is_victory, p.hero_name,
                   p.match_id,
                   m.bracket,
                   m.bottom_lane_outcome, m.mid_lane_outcome, m.top_lane_outcome
            FROM players p
            JOIN matches m ON p.match_id = m.match_id
            """,
            conn,
        )
    finally:
        conn.close()
    # Reindex to (and so reorder by) the cache's player_ids.
    df = df_all.set_index("player_id").reindex(player_ids).reset_index()
    return df


def load_aligned(
    db_path: Path,
    max_players: int | None,
    ckpt: dict,
    heroes: list[str] | None = None,
    positions: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Returns (ts_tensor, scalar_tensor, labels_df) all aligned to the same
    row order. Features come from the LaningDataset cache (recomputed if
    missing); labels are joined back via the cached `player_id` array.

    Applies the checkpoint's saved normalizers so features match training.
    """
    logger.info("Loading features from %s (via LaningDataset cache) …", db_path)
    ds = LaningDataset(db_path)
    if ds.player_ids.size == 0:
        logger.info("Cache is missing player_ids; backfilling from %s …", db_path)
        if not ds.ensure_player_ids(db_path):
            raise RuntimeError(
                "Could not backfill player_ids — cache and DB disagree. "
                "Rebuild by deleting the .features.npz file next to the DB."
            )
    player_ids = ds.player_ids
    ts_arr = np.stack(ds._ts_list).astype(np.float32, copy=False)   # (N, C, T)
    sc_arr = np.stack(ds._scalar_list).astype(np.float32, copy=False)  # (N, 7)
    n_samples = ts_arr.shape[0]
    logger.info("Got %d cached samples (shape %s).", n_samples, ts_arr.shape[1:])

    # Pull label columns for the cached player_ids, in the same order.
    df = _load_labels(db_path, player_ids)
    if df["player_id"].isna().any():
        n_missing = int(df["player_id"].isna().sum())
        logger.warning(
            "%d cached samples had no matching DB row — cache may be stale.",
            n_missing,
        )

    # Normalize label fields up front
    df["position"]  = df["position"].fillna("UNKNOWN")
    df["lane"]      = df["lane"].fillna("UNKNOWN")
    df["team"]      = df["team"].fillna("UNKNOWN")
    df["isVictory"] = df["is_victory"].map({1: "Win", 0: "Loss"}).fillna("Unknown")
    df["heroName"]  = df["hero_name"].fillna("Unknown")
    df["bracket"]   = df["bracket"].fillna("Unknown").astype(str)
    df["laneOutcome"] = [
        _derive_lane_outcome(lane, team, bot, mid, top)
        for lane, team, bot, mid, top in zip(
            df["lane"], df["team"],
            df["bottom_lane_outcome"], df["mid_lane_outcome"], df["top_lane_outcome"],
        )
    ]
    df["matchId"] = df["match_id"]

    # Build the filter mask: hero/position filters, then drop unknowns.
    keep = np.ones(n_samples, dtype=bool)
    if heroes:
        wanted = {h.lower() for h in heroes}
        keep &= df["heroName"].str.lower().isin(wanted).to_numpy()
        logger.info("Filtering to %d heroes: %s", len(heroes), ", ".join(sorted(heroes)))
    if positions:
        keep &= df["position"].isin(positions).to_numpy()
        logger.info("Filtering to positions: %s", ", ".join(sorted(positions)))
    keep &= (df["heroName"] != "Unknown").to_numpy()
    keep &= (df["position"] != "UNKNOWN").to_numpy()
    keep &= (df["laneOutcome"] != "UNKNOWN").to_numpy()

    # Cap to max_players (applied after filtering, preserving DB order).
    if max_players is not None and keep.sum() > max_players:
        kept_idx = np.where(keep)[0][:max_players]
        keep = np.zeros(n_samples, dtype=bool)
        keep[kept_idx] = True

    n_before = n_samples
    ts_arr = ts_arr[keep]
    sc_arr = sc_arr[keep]
    df = df[keep].reset_index(drop=True)
    logger.info(
        "Filtered %d → %d rows (dropped %d via hero/position/unknown filters and cap).",
        n_before, len(df), n_before - len(df),
    )

    # Apply checkpoint normalizers (identical to training preprocessing).
    ts_mean: np.ndarray = ckpt["ts_mean"]   # shape (C,)
    ts_std:  np.ndarray = ckpt["ts_std"]    # shape (C,)
    std_safe = np.where(ts_std > 0, ts_std, 1.0)
    ts_arr = (ts_arr - ts_mean[None, :, None]) / std_safe[None, :, None]
    ts_arr = np.nan_to_num(ts_arr, nan=0.0, posinf=0.0, neginf=0.0)

    sc_arr = ckpt["scalar_scaler"].transform(sc_arr).astype(np.float32)
    sc_arr = np.nan_to_num(sc_arr, nan=0.0, posinf=0.0, neginf=0.0)

    ts_t = torch.from_numpy(ts_arr)
    sc_t = torch.from_numpy(sc_arr)

    labels_df = df[["position", "lane", "team", "isVictory", "heroName",
                    "bracket", "laneOutcome", "matchId"]].reset_index(drop=True)
    return ts_t, sc_t, labels_df


# ---------------------------------------------------------------------------
# Model loading + encoding
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    embed_dim = (
        getattr(args, "embed_dim", 256) if not isinstance(args, dict)
        else args.get("embed_dim", 256)
    )
    # Infer in_channels from the first conv weight in the encoder, so an old
    # 18-channel checkpoint loads without us having to know that out of band.
    state = ckpt["model_state_dict"]
    first_conv_key = "encoder.ts_branch.conv.0.0.weight"
    in_channels = int(state[first_conv_key].shape[1]) if first_conv_key in state else 28
    model = SimCLRModel(embed_dim=embed_dim, in_channels=in_channels)
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info(
        "Loaded checkpoint (epoch %s, embed_dim=%d, in_channels=%d).",
        ckpt.get("epoch", "?"), embed_dim, in_channels,
    )
    return model, ckpt


def build_embeddings(
    model: SimCLRModel,
    ts_t: torch.Tensor,
    sc_t: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(ts_t, sc_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    parts = []
    with torch.no_grad():
        for ts, sc in loader:
            h = model.encoder(ts.to(device), sc.to(device))
            parts.append(h.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    logger.info("Running UMAP on %d × %d embedding matrix …", X.shape[0], X.shape[1])
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
        name="Color by", options=COLOR_OPTIONS, value="position", width=200,
    )
    alpha_slider = pn.widgets.FloatSlider(
        name="Opacity", start=0.05, end=1.0, step=0.05, value=0.85, width=200,
    )
    size_slider = pn.widgets.IntSlider(
        name="Point size", start=2, end=14, step=1, value=5, width=200,
    )
    info = pn.pane.Markdown(
        f"**{len(df):,} players** · {df['matchId'].nunique():,} matches",
        width=200,
    )

    def _legend_mute_hook(plot, element):
        """Enable click-to-mute on legend entries (click a category to dim it)."""
        plot.state.legend.click_policy = "mute"
        for renderer in plot.state.renderers:
            if hasattr(renderer, "muted_glyph") and renderer.muted_glyph is not None:
                renderer.muted_glyph.fill_alpha = 0.05
                renderer.muted_glyph.line_alpha = 0.05

    def _plot(color_by, alpha, size):
        hover_cols = [color_by, "heroName", "lane", "position", "isVictory", "laneOutcome", "matchId"]
        seen: set[str] = set()
        vdims: list[str] = []
        for c in hover_cols:
            if c not in seen:
                seen.add(c)
                vdims.append(c)

        palette = _PALETTE.get(color_by, list(Category10[10]))
        unique_vals = sorted(df[color_by].dropna().unique())

        # One Points layer per category so each gets its own Bokeh renderer,
        # which is required for per-category legend muting to work.
        layers = []
        for i, val in enumerate(unique_vals):
            color = palette[i % len(palette)]
            subset = df[df[color_by] == val]
            layers.append(
                hv.Points(subset, kdims=["umap_x", "umap_y"], vdims=vdims, label=str(val)).opts(
                    opts.Points(color=color, alpha=alpha, size=size, show_legend=True)
                )
            )

        return hv.Overlay(layers).opts(
            opts.Points(
                tools=["hover", "box_select", "lasso_select", "reset"],
                width=1000,
                height=1000,
                legend_position="right",
                title=f"SimCLR Embedding UMAP — coloured by {color_by}",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                toolbar="above",
                hooks=[_legend_mute_hook],
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
        description="Interactive UMAP of trained SimCLR embeddings."
    )
    parser.add_argument("--data",        required=True,
                        help="SQLite .db file.")
    parser.add_argument("--checkpoint",  default="./checkpoints/checkpoint_best.pt",
                        help="Path to model checkpoint (default: ./checkpoints/checkpoint_best.pt).")
    parser.add_argument("--max-players", type=int, default=None,
                        help="Cap on player rows (for quick iteration).")
    parser.add_argument("--heroes", type=str, default=None,
                        help="Heroes to include. Either a comma-separated list (e.g. 'Invoker,Pudge,Luna') "
                             "or a path to a .txt file with one hero name per line. "
                             "Case-insensitive. If omitted, all heroes are included.")
    parser.add_argument("--positions", type=str, default=None,
                        help="Positions to include as a comma-separated list. "
                             "Accepts numbers (e.g. '4,5') or full names (e.g. 'POSITION_4,POSITION_5'). "
                             "If omitted, all positions are included.")
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist",    type=float, default=0.1)
    parser.add_argument("--port",        type=int, default=5006)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if args.heroes:
        p = Path(args.heroes)
        if p.is_file():
            heroes = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        else:
            heroes = [h.strip() for h in args.heroes.split(",") if h.strip()]
    else:
        heroes = None

    positions = _normalise_positions(args.positions.split(",")) if args.positions else None

    model, ckpt = load_model(ckpt_path, device)
    ts_t, sc_t, labels_df = load_aligned(data_path, args.max_players, ckpt, heroes=heroes, positions=positions)

    logger.info("Encoding %d players …", len(ts_t))
    embeddings = build_embeddings(model, ts_t, sc_t, args.batch_size, device)

    embedding_2d = run_umap(embeddings, args.n_neighbors, args.min_dist)

    app = make_app(labels_df, embedding_2d)
    pn.serve(app, port=args.port, show=True, title="Dota 2 SimCLR Embedding UMAP")


if __name__ == "__main__":
    main()
