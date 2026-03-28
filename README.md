# dota_emb

A project for learning representations of Dota 2 player behavior during the laning stage. The pipeline covers match data collection, representation learning, and visualization/evaluation.

---

## Subprojects

| Directory | Description |
|---|---|
| [`extraction/`](extraction/README.md) | Fetch match IDs from OpenDota and extract per-player laning features from the STRATZ GraphQL API |
| [`training/`](training/README.md) | Train a SimCLR contrastive embedding model on the extracted feature vectors |
| [`evaluation/`](evaluation/README.md) | UMAP visualization, embedding classifiers, and feature importance analysis |

---

## Data Layout

```
data/
  match_lists/      — match ID text files (output of extraction/fetch_match_ids.py)
  matches/          — per-match JSON files (output of extraction/extract_match.py --store json)
  matches.db        — SQLite store (output of extraction/extract_match.py --store sqlite)
  failed_matches.txt
  skipped_matches.txt
```

---

## Setup

```bash
micromamba activate dota   # or: conda activate dota
pip install requests python-dotenv tqdm torch umap-learn panel holoviews bokeh scikit-learn xgboost
```

Create a `.env` file in the project root with your STRATZ token:

```
STRATZ_TOKEN=your_token_here
```
