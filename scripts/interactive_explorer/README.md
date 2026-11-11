# Interactive Semantic Cluster Explorer

An interactive web-based visualization tool for exploring Wikipedia article clusters in 2D UMAP space.

---

## Features

- **Corpus Selection**: Switch dynamically between `Vital Level 3`, `Computer Science`, `Literature`, and `Arizona`.
- **Dual Representation**: Toggle between **TF-IDF + TruncatedSVD** (100-d LSA) and **Sentence-BERT** (384-d contextual embeddings).
- **Dual Clustering Algorithms**: Compare **$K$-Means** (Spherical partition) against **HDBSCAN** (density-based with noise/outlier detection).
- **Interactive Inspection**:
  - **Hover Tooltips**: Hover over any point to see the **article title**, **page ID**, **cluster assignment**, and the **lead paragraph snippet**.
  - **Live Search**: Type article titles to locate and highlight specific Wikipedia articles instantly on the manifold.
  - **Pan & Zoom**: Scroll wheel to zoom in/out with smooth coordinates tracking; click & drag to pan.
  - **Cluster Legend & Metrics**: Sidebar displaying cluster counts, noise percentages, and silhouette scores.
- **Self-Contained**: Runs without external server dependencies or heavy JavaScript frameworks.

---

## How to Launch

You have two convenient options:

### Option 1: Direct Browser Launch (Recommended)
You can directly open the generated HTML file in your web browser:
```
scripts/interactive_explorer/web/index.html
```
Double-clicking the file or opening it with Chrome / Edge / Firefox works immediately without starting any server.

### Option 2: Run via Local Python Server
To serve it over localhost and automatically open your default browser:

```powershell
.venv\Scripts\python scripts\interactive_explorer\run_explorer.py
```
Then visit: `http://localhost:8050/index.html`

---

## Rebuilding the Dataset Bundle

If you re-run feature extraction or update cluster assignments in `data/derived/`, re-generate the explorer web bundle with:

```powershell
.venv\Scripts\python scripts\interactive_explorer\build_explorer.py
```

