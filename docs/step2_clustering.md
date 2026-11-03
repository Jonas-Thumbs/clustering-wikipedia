# Documentation: Step 2 Unsupervised Clustering (The 2x2 Matrix)

This document describes the implementation, theoretical foundations, output schema, and quantitative metrics for **Step 2** of the Wikipedia text clustering pipeline.

---

## 1. The $2 \times 2$ Clustering Matrix

Each corpus is evaluated across 4 complementary clustering configurations:

```
                          Representation Type
                ┌────────────────────────┬────────────────────────┐
                │  TF-IDF + SVD (100d)   │  SBERT Dense (384d)    │
┌───────────────┼────────────────────────┼────────────────────────┤
│ K-Means       │ Run 1: Spherical K-Means│ Run 2: Spherical K-Means│
│ (Partitioning)│ on L2-normalized LSA   │ on unit sphere         │
├───────────────┼────────────────────────┼────────────────────────┤
│ HDBSCAN       │ Run 3: Density clusters│ Run 4: Density clusters│
│ (Density/Noise│ on leading components  │ across dense manifold  │
└───────────────┴────────────────────────┴────────────────────────┘
```

### Theoretical Motivations
1. **Spherical $K$-Means ($K$-Means on Unit-Normalized Vectors)**:
   - Euclidean distance between unit vectors $\mathbf{u}, \mathbf{v}$ satisfies:
     $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\langle \mathbf{u}, \mathbf{v}\rangle = 2(1 - \cos(\mathbf{u}, \mathbf{v}))$$
   - Thus, running standard $K$-Means directly minimizes cosine distance distortion, preventing document length from biasing centroid placement.
2. **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**:
   - Does not force every article into a cluster.
   - Detects arbitrarily shaped, non-spherical clusters while isolating idiosyncratic outlier articles with label `-1`.

---

## 2. Output Locations & Schema

Cluster assignments and execution summaries are saved to:

```text
clustering-wikipedia/
└── data/
    └── derived/
        └── clusters/
            ├── arizona_clusters.csv
            ├── arizona_cluster_summary.json
            ├── computer_science_clusters.csv
            ├── computer_science_cluster_summary.json
            ├── literature_clusters.csv
            ├── literature_cluster_summary.json
            ├── vital_level_3_clusters.csv
            └── vital_level_3_cluster_summary.json
```

### CSV Schema (`{corpus}_clusters.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `page_id` | `int` | Wikipedia article page ID. |
| `title` | `str` | Article title. |
| `kmeans_svd` | `int` | Cluster label $\in [0, K-1]$ from $K$-Means on SVD. |
| `kmeans_sbert` | `int` | Cluster label $\in [0, K-1]$ from $K$-Means on SBERT. |
| `hdbscan_svd` | `int` | Cluster label ($\ge 0$) or `-1` (noise) from HDBSCAN on SVD. |
| `hdbscan_sbert` | `int` | Cluster label ($\ge 0$) or `-1` (noise) from HDBSCAN on SBERT. |

---

## 3. Results Summary Across Corpora

| Corpus | Articles | $K$ (K-Means) | SVD Silhouette | SBERT Silhouette | HDBSCAN (SVD) Clusters / Noise | HDBSCAN (SBERT) Clusters / Noise |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`arizona`** | 1,455 | 15 | **0.0842** | 0.0339 | 3 clusters (0.9% noise) | 3 clusters (67.5% noise) |
| **`computer_science`** | 1,597 | 15 | **0.0700** | 0.0372 | 10 clusters (53.4% noise) | 2 clusters (70.4% noise) |
| **`literature`** | 1,424 | 15 | **0.0731** | 0.0428 | 2 clusters (0.0% noise) | 2 clusters (44.8% noise) |
| **`vital_level_3`** | 999 | 12 | **0.0777** | 0.0410 | 3 clusters (13.9% noise) | 2 clusters (75.7% noise) |

---

## 4. Key Analytical Insights

1. **Partitioning Balance**:
   - $K$-Means partitions the article spaces cleanly without degenerate or single-item clusters (minimum cluster size $\ge 21$ across all corpora).
2. **Density and the "Curse of Dimensionality" in SBERT**:
   - In 384-dimensional continuous transformer space, points are relatively equidistant from one another due to high-dimensional geometry. HDBSCAN identifies a dense core of articles and flags peripheral/specialized topics as noise (~45–75%).
   - In contrast, SVD provides strong linear separation along orthogonal topic eigenvectors.
3. **Bridge Readiness**:
   - The generated cluster vectors $\mathbf{C}_{\text{sem}}$ are fully prepared for **Step 3 (UMAP Visualizations)**, **Step 4 (NMI / ARI Validation)**, and **Step 5 (Graph Leiden Comparison via NDI)**.

