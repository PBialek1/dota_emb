# evaluation

Tools for visualizing and evaluating learned SimCLR embeddings.

---

## Files

| File | Description |
|---|---|
| `umap_embeddings.py` | UMAP of trained encoder embeddings (requires checkpoint) |
| `umap_explorer.py` | UMAP of raw (pre-training) feature vectors |
| `embedding_classifier.py` | Logistic regression + XGBoost classifiers on embeddings |
| `feature_analysis.py` | Mutual-information feature importance analysis |

---

## umap_embeddings.py

Loads a trained SimCLR checkpoint, encodes all players through the encoder to get 256-dim embeddings, projects them to 2D with UMAP, and serves an interactive Panel scatter plot. Points are colored by position, hero, lane, lane outcome, win/loss, or bracket.

Rows where the hero, role (position), or lane outcome is unknown are excluded before embedding and projection.

```bash
python evaluation/umap_embeddings.py --data ./data/matches.db
python evaluation/umap_embeddings.py --data ./data/matches.db --checkpoint ./checkpoints/checkpoint_best.pt
python evaluation/umap_embeddings.py --data ./data/matches.db --max-players 5000 --port 5006

# Restrict to a subset of heroes (comma-separated or a text file)
python evaluation/umap_embeddings.py --data ./data/matches.db --heroes "Invoker,Pudge,Luna"
python evaluation/umap_embeddings.py --data ./data/matches.db --heroes ./heroes.txt
```

The scatter plot colors each category as a separate layer, so clicking a legend entry mutes that category. Default opacity is 0.85.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to `.db` file |
| `--checkpoint` | `./checkpoints/checkpoint_best.pt` | Trained SimCLR checkpoint |
| `--max-players` | none | Cap on rows loaded |
| `--heroes` | none | Comma-separated hero names or path to a `.txt` file (one name per line) |
| `--batch-size` | `512` | Encoder batch size |
| `--n-neighbors` | `15` | UMAP `n_neighbors` |
| `--min-dist` | `0.1` | UMAP `min_dist` |
| `--port` | `5006` | Panel server port |

**heroes.txt format:**
```
Invoker
Pudge
Luna
Phantom Assassin
```

---

## umap_explorer.py

UMAP of raw feature vectors without a trained model — useful as a baseline before training. Works with both JSON and SQLite stores.

```bash
python evaluation/umap_explorer.py --data ./data/matches.db
python evaluation/umap_explorer.py --data ./data/matches --max-players 5000
```

---

## embedding_classifier.py

Trains logistic regression and/or XGBoost classifiers on top of frozen encoder embeddings to measure how linearly separable the learned representations are for position, hero, and lane outcome.

```bash
python evaluation/embedding_classifier.py --data ./data/matches.db
python evaluation/embedding_classifier.py --data ./data/matches.db --checkpoint ./checkpoints/checkpoint_best.pt
python evaluation/embedding_classifier.py --data ./data/matches.db --model logreg
python evaluation/embedding_classifier.py --data ./data/matches.db --model xgb --max-players 20000
```

Outputs accuracy, classification reports, and confusion-matrix PNGs saved to `./evaluation/analysis/`.

---

## feature_analysis.py

Computes mutual information between each raw feature and outcome labels (position, lane, win/loss, bracket). Timeseries features are grouped by base name and the peak MI across all timesteps is reported.

```bash
python evaluation/feature_analysis.py --data ./data/matches.db
python evaluation/feature_analysis.py --data ./data/matches.db --top 20 --output-dir ./analysis
```

Outputs a ranked table to stdout, a heatmap PNG, and a CSV of all scores.
