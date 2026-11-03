# Text Branch Roadmap: Wikipedia Topic Clustering & Analysis

This document outlines the overarching long-term roadmap for the **Text Branch** of the Wikipedia Clustering pipeline. The goal is to build, cluster, evaluate, and interpret semantic article representations, bridging them to the topological graph community branch via node-level divergence analysis.

---

## 1. High-Level Architecture

```
                    [data/derived/text/{corpus}.articles.jsonl]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
             [Text Preprocessing]             [Text Preprocessing]
             (Full clean text)                (Summary / Lead section)
                         │                             │
                         ▼                             ▼
              [TF-IDF + TruncatedSVD]          [SBERT Embeddings]
             (Sparse / LSA Components)         (Dense 384-d / 768-d)
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         [The 2x2 Clustering Matrix]
                  ┌─────────────────────┬─────────────────────┐
                  │  K-Means / Spherical│       HDBSCAN       │
                  └─────────────────────┴─────────────────────┘
                                        │
                                        ▼
                          [UMAP 2D Visual Projections]
                                        │
                                        ▼
                           [Quantitative Validation]
                       • Silhouette Score (Internal)
                       • NMI & ARI vs. Wikipedia Categories
                                        │
                                        ▼
                        [Bridge to Graph Branch (NDI)]
                   Compare C_sem (Text) vs. C_top (Leiden Graph)
```

---

## 2. The 5 Pipeline Steps

### Step 1: Feature Extraction & Vectorization (Dual Approaches)
Extract two complementary semantic vector spaces for each corpus:
1. **Sparse / Linear Baseline: TF-IDF + TruncatedSVD (LSA)**
   - Extract top $n$-grams ($n \in \{1, 2\}$) with sublinear term-frequency scaling and document frequency bounds (`min_df`, `max_df`).
   - Reduce dimensionality via `TruncatedSVD` (Latent Semantic Analysis, e.g., $d \in [50, 100]$ components) preserving variance.
   - **Interpretability**: Inspect top positive and negative eigenvector loadings (principal components) to map exact vocabulary driving semantic axes.
2. **Dense / Contextual Representation: Sentence-BERT (SBERT)**
   - Encode lead paragraphs or chunked article content using a transformer (e.g., `all-MiniLM-L6-v2` or `BGE-base`).
   - Normalise vectors to unit length (yielding cosine distances via Euclidean dot-product).
   - **Semantic Generalization**: Captures synonyms, conceptual relatedness, and polysemy without relying on exact surface form.

### Step 2: Unsupervised Clustering (The $2 \times 2$ Matrix)
Apply two contrasting clustering paradigms across both feature spaces:

| Clustering Algorithm | TF-IDF / SVD Vectors | SBERT Dense Embeddings |
| :--- | :--- | :--- |
| **$K$-Means / Spherical $K$-Means** | Partitions LSA space into $K$ convex Voronoi cells; evaluates global spherical variance. | Partitions dense manifold into $K$ semantic centroids using cosine/Euclidean distance. |
| **HDBSCAN** | Density-based clustering on reduced SVD components; detects non-spherical clusters. | Discovers variable-density semantic clusters, cleanly isolating outliers/noise into label `-1`. |

- **Hyperparameter Selection**:
  - Sweep $K \in [5, 50]$ (guided by corpus scale and category hierarchy).
  - Tune `min_cluster_size` and `min_samples` for HDBSCAN to manage noise ratio.

### Step 3: Projection & Visualization (UMAP)
- Run **UMAP (Uniform Manifold Approximation and Projection)** to project high-dimensional representations to a 2D Euclidean coordinate space strictly for visualization.
- Generate color-coded scatter plots overlaying:
  1. $K$-Means cluster partitions.
  2. HDBSCAN clusters (highlighting noise/outliers).
  3. Ground-truth Wikipedia Category tags.
- Verify visual cluster separation, continuous transitions, and boundary ambiguity.

### Step 4: Quantitative Evaluation
Validate clusters independently before combining with the graph branch:
1. **Internal Cohesion**:
   - **Silhouette Score**: Evaluates cluster tightness vs. separation ($s \in [-1, 1]$).
   - Calculated for non-noise points across various $K$ and HDBSCAN runs.
2. **External Alignment**:
   - **Normalized Mutual Information (NMI)**: Information-theoretic overlap with Wikipedia ground-truth categories.
   - **Adjusted Rand Index (ARI)**: Pair-counting chance-adjusted clustering similarity.
   - Evaluated against parent categories extracted from Wikipedia category metadata.

### Step 5: Bridge to the Graph Branch (Cross-Modal Comparison)
Combine textual semantics with structural hyperlink topology:
1. Export the optimal text cluster assignment vector $\mathbf{C}_{\text{sem}} \in \{1, \dots, K\}^N$ for each corpus.
2. Receive the graph community vector $\mathbf{C}_{\text{top}}$ produced by the **Leiden algorithm** on the article hyperlink adjacency graph (`data/derived/links/{corpus}.edges.tsv`).
3. Compute the **Node Divergence Index (NDI)** for each article $i$:
   $$\text{NDI}(i) = 1 - \text{Jaccard}(N_{\text{sem}}(i), N_{\text{top}}(i))$$
   or via cross-contingency membership divergence.
4. **Error Analysis / Case Studies**: Isolate high-divergence articles (e.g., articles structurally embedded in one link community but textually belonging to an entirely different topical cluster).

---

## 3. Directory & Artifact Structure

```text
clustering-wikipedia/
├── data/
│   ├── derived/
│   │   ├── text/           <- Raw JSONL article dumps
│   │   ├── links/          <- Hyperlink edge TSVs
│   │   ├── features/       <- Cached SVD (.npy/.npz) & SBERT (.npy) embeddings
│   │   └── clusters/       <- Cluster assignment CSVs (C_sem)
│   └── processed/          <- Ground truth page_ids
├── docs/
│   └── text_branch_plan.md <- This roadmap document
├── notebooks/              <- UMAP visualization and exploration notebooks
└── scripts/
    ├── clean_wikitext.py   <- Wikitext parsing and text stripping
    ├── extract_features.py <- TF-IDF + SVD and SBERT encoding
    ├── run_clustering.py   <- K-Means & HDBSCAN 2x2 grid
    └── evaluate_text.py    <- Silhouette, NMI, and ARI metrics
```

