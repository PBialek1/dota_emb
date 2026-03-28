"""embedding_classifier.py — Logistic regression + XGBoost classifiers on SimCLR embeddings.

Loads the trained encoder, encodes all players, then trains two classifiers
(logistic regression and XGBoost) for each of:
  - position    (POSITION_1 … POSITION_5)
  - heroName    (130+ heroes)
  - laneOutcome (Win / Stomp Win / Loss / Stomp Loss / Tie)

Reports accuracy, classification report, and saves confusion-matrix PNGs to
./evaluation/analysis/ for each (target, model) combination.

Usage:
    python evaluation/embedding_classifier.py --data ./data/matches.db
    python evaluation/embedding_classifier.py --data ./data/matches.db --checkpoint ./checkpoints/checkpoint_best.pt
    python evaluation/embedding_classifier.py --data ./data/matches.db --max-players 20000
    python evaluation/embedding_classifier.py --data ./data/matches.db --model xgb
    python evaluation/embedding_classifier.py --data ./data/matches.db --model logreg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from umap_embeddings import load_aligned, load_model, build_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGETS = ["position", "heroName", "laneOutcome"]

# ---------------------------------------------------------------------------
# Classifier configs
# ---------------------------------------------------------------------------

LOGREG_CONFIG = {
    "position":    {"max_iter": 500,  "C": 1.0},
    "heroName":    {"max_iter": 1000, "C": 0.5},
    "laneOutcome": {"max_iter": 500,  "C": 1.0},
}

XGB_CONFIG = {
    "position":    {"n_estimators": 400, "max_depth": 6,  "learning_rate": 0.1},
    "heroName":    {"n_estimators": 600, "max_depth": 8,  "learning_rate": 0.1},
    "laneOutcome": {"n_estimators": 400, "max_depth": 5,  "learning_rate": 0.1},
}


def make_logreg(target: str) -> LogisticRegression:
    cfg = LOGREG_CONFIG[target]
    return LogisticRegression(
        max_iter=cfg["max_iter"],
        C=cfg["C"],
        solver="lbfgs",
        n_jobs=-1,
    )


def make_xgb(target: str, n_classes: int) -> XGBClassifier:
    cfg = XGB_CONFIG[target]
    return XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        device="cuda" if _cuda_available() else "cpu",
        n_jobs=-1,
        verbosity=0,
        random_state=42,
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f", xticks_rotation="vertical")
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# Per-target, per-model evaluation
# ---------------------------------------------------------------------------

def run_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target: str,
    models: list[str],
    out_dir: Path,
) -> None:
    # Encode string labels to integers for XGBoost
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    labels = list(le.classes_)
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    n_classes   = len(labels)

    results: dict[str, tuple[float, np.ndarray]] = {}  # model_name -> (acc, y_pred_str)

    if "logreg" in models:
        logger.info("  [logreg] fitting …")
        clf = make_logreg(target)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        results["logreg"] = (acc, y_pred)

    if "xgb" in models:
        logger.info("  [xgb] fitting (n_classes=%d) …", n_classes)
        clf = make_xgb(target, n_classes)
        clf.fit(X_train, y_train_enc)
        y_pred_enc = clf.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        acc = (y_pred == y_test).mean()
        results["xgb"] = (acc, y_pred)

    # Print summary header
    print(f"\n{'='*60}")
    print(f"  Target: {target}")
    print(f"{'='*60}")
    acc_line = "  ".join(f"{m}: {acc:.4f}" for m, (acc, _) in results.items())
    print(f"  Accuracy — {acc_line}\n")

    for model_name, (acc, y_pred) in results.items():
        print(f"--- {model_name} ---")
        print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

        plot_confusion_matrix(
            y_test, y_pred, labels,
            title=f"{target} [{model_name}]  acc={acc:.3f}",
            out_path=out_dir / f"confusion_{target}_{model_name}.png",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistic regression + XGBoost classifiers on SimCLR embeddings."
    )
    parser.add_argument("--data",        required=True, help="SQLite .db file.")
    parser.add_argument("--checkpoint",  default="./checkpoints/checkpoint_best.pt")
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--test-split",  type=float, default=0.2)
    parser.add_argument(
        "--model",
        choices=["both", "logreg", "xgb"],
        default="both",
        help="Which classifier(s) to run (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(__file__).parent / "analysis"
    out_dir.mkdir(exist_ok=True)

    models = ["logreg", "xgb"] if args.model == "both" else [args.model]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, ckpt = load_model(ckpt_path, device)
    ts_t, sc_t, labels_df = load_aligned(data_path, args.max_players, ckpt)

    logger.info("Encoding %d players …", len(ts_t))
    X = build_embeddings(model, ts_t, sc_t, args.batch_size, device)

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_split, random_state=42,
        stratify=labels_df["position"].values,
    )
    X_train, X_test = X[idx_train], X[idx_test]

    for target in TARGETS:
        logger.info("Running classifiers for '%s' …", target)
        y = labels_df[target].values
        run_target(X_train, X_test, y[idx_train], y[idx_test], target, models, out_dir)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
