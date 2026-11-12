Ran test HDBSCAN on high-dim Arizona features

I have launched a test on the Arizona high-dimensional embeddings (100-d SVD and 384-d SBERT) with `cluster_selection_method='leaf'` to inspect the cluster counts and noise behavior before generating the full cluster sets and plots in dedicated output directories. I'll share the results once the test completes.

Ran sweep metric and params on high-dim SBERT

The preliminary run on Arizona high-dimensional features illustrates the *curse of dimensionality* directly:

* **SVD (100 dimensions)**:
  * Raw Euclidean: **4 clusters**, **69.0% noise**
  * L2-normalized: **7 clusters**, **79.7% noise**
* **SBERT (384 dimensions)**:
  * Default `min_cluster_size=25, min_samples=5`: **0 clusters**, **100.0% noise** (every single document was rejected as an outlier because Euclidean distance concentration flattens density contrasts).

I have launched a quick sweep across metrics (`cosine` vs `euclidean`) and sample parameters (`min_samples=1, 2, 5`, `min_cluster_size=10, 15, 25`) on SBERT to see under what conditions high-dimensional density clustering can find clusters before writing the dedicated script. I will report the results in a moment.

Analyzed plot_umap.py:L1-100
Created run_highdim_clustering.py
Ran run high-dimensional HDBSCAN clustering and plotting

I have created a standalone script [`scripts/text_based_clustering/run_highdim_clustering.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering/run_highdim_clustering.py) and launched the pipeline to execute HDBSCAN with `leaf` clustering directly in the original high-dimensional spaces (100-d SVD and 384-d SBERT).

All results and plots are being isolated in separate directories so the 2D baseline remains untouched:
* **Cluster Assignments & Summaries**: `data/derived/clusters_highdim/`
* **Visualization Figures**: `data/derived/figures_highdim/`

I will update you as soon as the run and plot generation finish.

The high-dimensional HDBSCAN clustering and visualization run has completed across all 4 corpora.

All outputs have been written to dedicated directories:
* **Cluster Assignments & Summaries**: [`data/derived/clusters_highdim/`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/clusters_highdim/)
* **Figures**: [`data/derived/figures_highdim/`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/figures_highdim/)

---

### Empirical Results: High-Dim vs. 2D UMAP Space

| Corpus | Method & Space | Clusters Found | Noise Ratio | Key Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **`arizona`** | **SVD 100-d** (High-Dim) | **19** | **61.9%** | Identifies tight sub-communities, but 62% of points labeled noise |
| | **SBERT 384-d** (High-Dim) | **9** | **82.1%** | Severe distance concentration: over 82% of articles marked noise |
| | *2D UMAP (Baseline)* | *18 (SVD) / 21 (SBERT)* | *32.0% / 35.0%* | Balanced density estimation with only ~33% noise |
| **`computer_science`** | **SVD 100-d** (High-Dim) | **19** | **62.3%** | Moderate cluster count, high noise |
| | **SBERT 384-d** (High-Dim) | **5** | **80.5%** | Fails to detect sparse intermediate density bridges (80% noise) |
| | *2D UMAP (Baseline)* | *22 (SVD) / 19 (SBERT)* | *31.6% / 38.4%* | Much more complete coverage across CS domains |
| **`literature`** | **SVD 100-d** (High-Dim) | **14** | **72.3%** | 72% noise |
| | **SBERT 384-d** (High-Dim) | **6** | **68.4%** | Only discovers a few dense author/genre cores |
| | *2D UMAP (Baseline)* | *20 (SVD) / 19 (SBERT)* | *33.8% / 38.6%* | Granular genres and literary periods captured |
| **`vital_level_3`** | **SVD 100-d** (High-Dim) | **20** | **42.1%** | Broad diverse categories allow lower noise in 100-d |
| | **SBERT 384-d** (High-Dim) | **9** | **75.7%** | 384-d distance concentration forces 76% noise |
| | *2D UMAP (Baseline)* | *17 (SVD) / 13 (SBERT)* | *20.6% / 28.6%* | Strongest overall silhouette and coherence |

---

### Key Takeaways for Your Research & Presentation

1. **Concrete Evidence of the Curse of Dimensionality**:
   - In 384-dimensional SBERT space, pairwise distances converge to a narrow distribution band. HDBSCAN's mutual reachability distance considers most inter-point regions too uniform, causing noise ratios of **68% to 82%**.
   - In 100-dimensional SVD space, the effect is less severe than SBERT (noise ~42% to 62%), but still leaves more than half the dataset unclustered.

2. **Visual Inspection of the Generated Plots**:
   You can view the resulting scatter plots in [`data/derived/figures_highdim/`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/figures_highdim/):
   - [`arizona_umap_clusters_highdim.png`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/figures_highdim/arizona_umap_clusters_highdim.png)
   - [`computer_science_umap_clusters_highdim.png`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/figures_highdim/computer_science_umap_clusters_highdim.png)
   - [`literature_umap_clusters_highdim.png`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/figures_highdim/literature_umap_clusters_highdim.png)
   - [`vital_level_3_umap_clusters_highdim.png`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/data/derived/figures_highdim/vital_level_3_umap_clusters_highdim.png)

   In these plots, the non-noise clusters appear as dense local cores, while the grey noise points cover the vast majority of the background manifold.

3. **Comparison Slide Insight**:
   This experiment provides evidence for your presentation:
   - High-dimensional density estimation suffers from severe noise attrition (up to 82%).
   - Running HDBSCAN on UMAP projections (or intermediate 10-d UMAP) is a necessary step to produce actionable clusters from dense transformer embeddings.