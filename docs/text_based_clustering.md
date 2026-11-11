# Text-Based Clustering Pipeline Documentation

This document describes the architecture, implementation, outputs, and validation procedures for the **text-based branch** of the Wikipedia clustering project. All pipeline code for feature extraction, unsupervised clustering, and projection/visualization has been consolidated under [`scripts/text_based_clustering/`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering).

---

## 1. Overview & Architecture

The text branch extracts semantic representations from cleaned Wikipedia articles across four target corpora and performs unsupervised clustering to uncover topical groupings.

### Evaluated Corpora
- **`arizona`**: 1,455 articles (geotagged & regional entities)
- **`computer_science`**: 1,597 articles (technical domain ontology)
- **`literature`**: 1,424 articles (authors, works, genres)
- **`vital_level_3`**: 999 articles (broad foundational human knowledge)

### Pipeline Flow

```
Raw Corpora (data/derived/text/*.articles.jsonl)
       │
       ▼
[Text Cleaning] (scripts/lib/text_cleaning.py & batch_clean_text.py)
       │
       ├── data/derived/cleaned/{corpus}.cleaned.jsonl
       │
       ▼
[Feature Extraction] (scripts/text_based_clustering/)
       │
       ├── TF-IDF + TruncatedSVD (100-d) ────────┐
       └── SBERT (all-MiniLM-L6-v2, 384-d) ─────┤
                                                 ▼
                                        [UMAP 2D Projection] (project_umap.py)
                                                 │
                                                 ├── 2D Coordinates (SVD & SBERT)
                                                 ▼
                                    [Unsupervised Clustering 2x2] (run_clustering.py)
                                                 ├── Spherical K-Means (SVD)
                                                 ├── Spherical K-Means (SBERT)
                                                 ├── HDBSCAN on UMAP(SVD) 2D
                                                 └── HDBSCAN on UMAP(SBERT) 2D
                                                 │
                                                 ▼
                                      [Visual & Automated Checks]
                                                 ├── 2x2 Grid Plots (plot_umap.py)
                                                 └── Verification Suite (check_*.py)
```

---

## 2. Directory & Script Structure

All scripts supporting text extraction, clustering, visualization, and verification are organized in [`scripts/text_based_clustering/`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering):

| Script | Purpose | Input | Output |
|---|---|---|---|
| [`extract_svd.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/extract_svd.py) | TF-IDF (1-2 ngrams, sublinear TF) + 100-component TruncatedSVD (LSA) | `.cleaned.jsonl` | `{corpus}_svd100.npy`, `{corpus}_svd_top_words.json`, `{corpus}_svd_explained_variance.json` |
| [`extract_sbert.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/extract_sbert.py) | Dense 384-d embeddings (`all-MiniLM-L6-v2`, unit-normalized) | `.cleaned.jsonl` | `{corpus}_sbert.npy`, `{corpus}_page_ids.npy`, `{corpus}_titles.json` |
| [`check_features.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/check_features.py) | Validates feature shapes, lack of NaNs/Infs, unit norms, and explained variance | Feature files | Terminal test report |
| [`project_umap.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/project_umap.py) | Non-linear dimensionality reduction to 2D manifold (`cosine` metric) | `{corpus}_svd100.npy`, `{corpus}_sbert.npy` | `{corpus}_umap_svd.npy`, `{corpus}_umap_sbert.npy` |
| [`run_clustering.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/run_clustering.py) | Executes 2x2 clustering matrix (K-Means & HDBSCAN) | Feature arrays & UMAP projections | `{corpus}_clusters.csv`, `{corpus}_cluster_summary.json` |
| [`check_clustering.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/check_clustering.py) | Validates cluster coverage, size distributions, silhouette, and noise limits | Cluster files | Terminal test report |
| [`plot_umap.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/plot_umap.py) | Renders 4-panel scatter plots overlaying cluster assignments | UMAP coordinates & cluster assignments | `data/derived/figures/{corpus}_umap_clusters.png` |
| [`check_visualization.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/check_visualization.py) | Validates projection coordinates, ranges, and generated PNG files | UMAP & image files | Terminal test report |

---

## 3. Methodological Details

### 3.1 Feature Extraction
1. **Sparse / Linear Baseline (LSA)**:
   - **Vectorizer**: `TfidfVectorizer(max_features=25000, min_df=3, max_df=0.6, ngram_range=(1, 2), sublinear_tf=True)`
   - **Decomposition**: `TruncatedSVD(n_components=100, random_state=42)`
   - **Semantic inspection**: Top component term loadings exported to JSON for interpretability.
2. **Dense / Contextual Representation (SBERT)**:
   - **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
   - **Inference**: Article `lead_text` processed in batches on GPU (`cuda`). Embeddings are L2-normalized onto the unit hypersphere.

### 3.2 Unsupervised Clustering (2x2 Matrix)
- **Spherical $K$-Means on SVD & SBERT**:
  - SVD vectors are L2-normalized to emulate Spherical $K$-Means (cosine distance). SBERT embeddings are already unit-norm.
  - $K=15$ for `arizona`, `computer_science`, `literature`; $K=12$ for `vital_level_3`.
- **Density-Based Clustering (HDBSCAN)**:
  - **The High-Dimensional Distance Problem**: When HDBSCAN was initially run directly on 100-d SVD or 384-d SBERT, distance concentration collapsed density gradients, assigning nearly all articles to 1-2 giant clusters.
  - **Solution**: HDBSCAN runs on the **pre-computed UMAP 2D projections**. In 2D, density variations are preserved and local distance metrics remain discriminative.
  - **Parameters**: `min_cluster_size=25`, `min_samples=5`, `cluster_selection_method='eom'`, `metric='euclidean'`.

### 3.3 Projection & Visualization
- **UMAP Parameters**: `n_neighbors=25`, `min_dist=0.1`, `metric='cosine'`, `n_components=2`, `random_state=42`.
- Generated 4-panel figures per corpus show:
  1. Top-Left: SVD UMAP + $K$-Means overlay
  2. Top-Right: SBERT UMAP + $K$-Means overlay
  3. Bottom-Left: SVD UMAP + HDBSCAN overlay (noise highlighted in grey)
  4. Bottom-Right: SBERT UMAP + HDBSCAN overlay (noise highlighted in grey)

---

## 4. Summary of Results

### 4.1 Cluster Statistics

| Corpus | Articles | $K$-Means $K$ | Silhouette (SVD / SBERT) | HDBSCAN SVD (Clusters / Noise) | HDBSCAN SBERT (Clusters / Noise) |
|---|---|---|---|---|---|
| `arizona` | 1,455 | 15 | 0.084 / 0.034 | 3 clusters (0.0% noise) | 2 clusters (1.5% noise) |
| `computer_science` | 1,597 | 15 | 0.070 / 0.037 | 10 clusters (5.1% noise) | 15 clusters (18.9% noise) |
| `literature` | 1,424 | 15 | 0.073 / 0.043 | 5 clusters (0.5% noise) | 2 clusters (0.0% noise) |
| `vital_level_3` | 999 | 12 | 0.078 / 0.041 | 16 clusters (18.1% noise) | 9 clusters (19.9% noise) |

### 4.2 Key Observations
1. **$K$-Means Behavior**: Partitions the entire space cleanly into $K$ convex regions without noise, ensuring 100% article assignment.
2. **HDBSCAN Behavior**: Density-based clustering on UMAP projections effectively extracts distinct topological topic clusters while isolating bridge and outlier articles as noise (0% to ~20%).
3. **Internal Silhouette**: HDBSCAN non-noise silhouette scores (0.31 – 0.77) are high due to tight cluster isolation in manifold space, while high-dimensional $K$-Means scores (0.03 – 0.08) reflect standard dispersion on high-dimensional hyperspheres.

---

## 5. Artifacts and Output Locations

| Artifact Type | Directory | Naming Pattern |
|---|---|---|
| Cleaned text | `data/derived/cleaned/` | `{corpus}.cleaned.jsonl` |
| SVD features | `data/derived/features/` | `{corpus}_svd100.npy`, `{corpus}_svd_top_words.json`, `{corpus}_svd_explained_variance.json` |
| SBERT embeddings | `data/derived/features/` | `{corpus}_sbert.npy`, `{corpus}_page_ids.npy`, `{corpus}_titles.json` |
| UMAP 2D projections | `data/derived/projections/` | `{corpus}_umap_svd.npy`, `{corpus}_umap_sbert.npy` |
| Cluster tables | `data/derived/clusters/` | `{corpus}_clusters.csv` (columns: `page_id`, `title`, `kmeans_svd`, `kmeans_sbert`, `hdbscan_svd`, `hdbscan_sbert`) |
| Cluster summaries | `data/derived/clusters/` | `{corpus}_cluster_summary.json` |
| 4-Panel visual plots | `data/derived/figures/` | `{corpus}_umap_clusters.png` |

---

## 6. How to Re-run the Pipeline

Run commands using the project virtual environment (`.venv\Scripts\python`):

```powershell
# 1. Feature Extraction & Verification
.venv\Scripts\python scripts\text_based_clustering\extract_svd.py
.venv\Scripts\python scripts\text_based_clustering\extract_sbert.py
.venv\Scripts\python scripts\text_based_clustering\check_features.py

# 2. UMAP 2D Projections
.venv\Scripts\python scripts\text_based_clustering\project_umap.py

# 3. 2x2 Matrix Clustering & Verification
.venv\Scripts\python scripts\text_based_clustering\run_clustering.py
.venv\Scripts\python scripts\text_based_clustering\check_clustering.py

# 4. Visualization & Output Checks
.venv\Scripts\python scripts\text_based_clustering\plot_umap.py
.venv\Scripts\python scripts\text_based_clustering\check_visualization.py
```

