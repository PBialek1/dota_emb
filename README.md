# dota_emb

A project for learning representations of Dota 2 player behavior during the laning stage. The pipeline covers match data collection, representation learning, and visualization/evaluation.

---

## Subprojects

| Directory | Description |
|---|---|
| [`extraction/`](extraction/README.md) | Fetch match IDs from OpenDota and extract per-player laning features from the STRATZ GraphQL API |
| [`training/`](training/README.md) | Train embedding models on the extracted feature vectors |
| [`visualization/`](visualization/README.md) | UMAP-based interactive explorer for inspecting learned representations |

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
conda activate dota   # or your env of choice
pip install requests python-dotenv tqdm umap-learn panel holoviews bokeh scikit-learn
```

Create a `.env` file in the project root with your STRATZ token:

```
STRATZ_TOKEN=your_token_here
```
