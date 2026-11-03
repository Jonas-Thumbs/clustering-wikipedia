# Documentation: Step 3 UMAP Projections & Visualizations

This document explains the UMAP projection methodology, configuration parameters, generated visual outputs, and interpretation of the cluster overlays for **Step 3** of the Wikipedia text pipeline.

---

## 1. Methodology & Purpose

**Uniform Manifold Approximation and Projection (UMAP)** is a non-linear dimension-reduction algorithm based on Riemannian geometry and algebraic topology. In this pipeline, UMAP is used **strictly for visual exploration in a 2D Euclidean plane**:
- Projecting both high-dimensional spaces (100-d SVD and 384-d SBERT) to $(x, y)$ coordinates.
- Preserving both local neighborhood similarities and global cluster topology.
- Directly overlaying $K$-Means partitions and HDBSCAN density clusters (highlighting noise/outlier points in light grey) onto the same coordinates.

---

## 2. Directory Layout & Artifacts

All projections and figures are saved under `data/derived/`:

```text
clustering-wikipedia/
├── scripts/
│   └── step3_visualization/
│       ├── project_umap.py            <- Generates 2D coordinates for all corpora
│       ├── plot_umap.py               <- Renders 2x2 multi-panel scatter plots
│       └── check_visualization.py     <- Automated verification test suite
└── data/
    └── derived/
        ├── projections/               <- Cached 2D float32 coordinate arrays
        │   ├── arizona_umap_svd.npy
        │   ├── arizona_umap_sbert.npy
        │   ├── computer_science_umap_svd.npy
        │   ├── computer_science_umap_sbert.npy
        │   ├── literature_umap_svd.npy
        │   ├── literature_umap_sbert.npy
        │   ├── vital_level_3_umap_svd.npy
        │   └── vital_level_3_umap_sbert.npy
        └── figures/                   <- 150 DPI high-resolution figures
            ├── arizona_umap_clusters.png
            ├── computer_science_umap_clusters.png
            ├── literature_umap_clusters.png
            └── vital_level_3_umap_clusters.png
```

---

## 3. UMAP Configuration Parameters

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `n_components` | `2` | 2D plane for scatter plots. |
| `metric` | `"cosine"` | Aligns with spherical semantics and unit-normalized SBERT / normalized SVD vectors. |
| `n_neighbors` | `25` | Balances local fine-grained structure with global topic separation. |
| `min_dist` | `0.1` | Prevents over-clumping, allowing clear visual inspection of individual article points. |
| `random_state` | `42` | Ensures deterministic, reproducible visualizations across runs. |

---

## 4. Multi-Panel Figure Layout

Each figure (`data/derived/figures/{corpus}_umap_clusters.png`) contains a 4-panel comparison:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ (0, 0) UMAP (SVD) + K-Means          │ (0, 1) UMAP (SBERT) + K-Means        │
│ Convex partitions across LSA space   │ Convex partitions across SBERT space │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ (1, 0) UMAP (SVD) + HDBSCAN          │ (1, 1) UMAP (SBERT) + HDBSCAN        │
│ Density clusters; noise in grey      │ Density clusters; noise in grey      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 5. Visual Observations & Insights

1. **SVD vs. SBERT Geometry**:
   - **SBERT Projections** form continuous, organic manifold branches with smooth semantic transitions between neighboring subdisciplines (e.g., in Computer Science: algorithms $\rightarrow$ programming languages $\rightarrow$ computer systems $\rightarrow$ hardware).
   - **SVD Projections** exhibit star-like or radial ray patterns, reflecting the orthogonal linear axes of the dominant singular vectors.
2. **$K$-Means Alignment**:
   - $K$-Means boundaries map onto discrete territorial sectors of the UMAP manifolds, indicating strong internal cohesion within Voronoi cells.
3. **HDBSCAN Outlier Localization**:
   - Outliers flagged with label `-1` (plotted in light grey) reside primarily on the outer periphery or along transitional bridges between dense topic cores, visually confirming that HDBSCAN is properly isolating ambiguous and cross-domain articles.

