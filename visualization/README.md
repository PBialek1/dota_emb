# visualization

Interactive tools for exploring and evaluating learned representations.

---

## umap_explorer.py

Projects player feature vectors into 2D with UMAP and serves an interactive scatter plot. Points can be colored by role, lane, team, win/loss, hero, or MMR bracket.

```bash
# From project root:
python visualization/umap_explorer.py --data ./data/matches          # JSON store
python visualization/umap_explorer.py --data ./data/matches.db       # SQLite store
python visualization/umap_explorer.py --data ./data/matches.db --max-players 5000
python visualization/umap_explorer.py --data ./data/matches --n-neighbors 30 --min-dist 0.05
```

Opens a Panel app in the browser at `http://localhost:5006`.

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to JSON match directory or `.db` file |
| `--max-players` | none | Cap on rows loaded (useful for quick iteration) |
| `--n-neighbors` | `15` | UMAP `n_neighbors` |
| `--min-dist` | `0.1` | UMAP `min_dist` |
| `--port` | `5006` | Panel server port |
