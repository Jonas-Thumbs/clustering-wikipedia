Ran list docs
Ran list derived data subdirs
Ran check graph link files
Analyzed computer_science_cluster_summary.json:L1-28

Here is a concise, slide-ready briefing of the progress, concrete numbers, and technical updates from the codebase for your 5-minute presentation.

---

### Slide 1: Datasets & Scope (Clear & Concrete)

* **Datasets (4 Wikipedia Corpora, 5,475 articles total):**
  * `vital_level_3` (**999** articles): Broad foundational world knowledge.
  * `computer_science` (**1,597** articles): Specialized technical topic hierarchy.
  * `literature` (**1,424** articles): Authors, genres, and literary works.
  * `arizona` (**1,455** articles): Geotagged geographic and regional entities.
* **Corpus Diversity:** Evaluates clustering performance across distinct domain densities, from broad universal concepts to narrow technical taxonomies and regional geography.
* **Dual-Branch Vision:** 
  * **Text branch** (semantic representations $\rightarrow$ topic clusters).
  * **Graph branch** (hyperlink adjacency networks $\rightarrow$ topological communities via Leiden).
  * **Bridge:** Node Divergence Index (NDI) to measure semantic vs. topological disagreement.

---

### Slide 2: Data Preprocessing & Engineering Challenges

* **The Challenge:** Raw Wikipedia markup contains deeply nested templates (`{{Infobox ...}}`), media links (`[[File:...]]`), HTML tags, references (`<ref>`), and category tags that distort text vector representations.
* **Implementation:** 
  * Built a custom multi-pass cleaning pipeline ([`scripts/lib/text_cleaning.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/lib/text_cleaning.py)) using `mwparserfromhell` AST parsing paired with balanced-bracket stripping.
  * Extracted both clean `lead_text` (summary definition) and clean `full_text` for each article.
  * Verified 100% completion across all 4 corpora (`.cleaned.jsonl` files).

---

### Slide 3: Feature Extraction (Dual Representations)

We implemented two complementary vectorization strategies:

1. **Linear / Interpretable Baseline (TF-IDF + LSA / TruncatedSVD)**:
   * Vocabulary: unigrams + bigrams ($1\text{--}2$ n-grams), sublinear term frequency, corpus bounds ($3 \le \text{DF} \le 60\%$).
   * Reduced to **100 dimensions** via `TruncatedSVD`.
   * **Key Feature:** Component eigenvector loadings are exported to JSON, allowing us to inspect the semantic keywords defining each principal axis.
2. **Dense / Contextual Representation (Sentence-BERT)**:
   * Pre-trained transformer: `sentence-transformers/all-MiniLM-L6-v2` (**384 dimensions**).
   * L2-normalized onto the unit hypersphere to capture deep semantic synonyms and contextual nuance.

---

### Slide 4: Unsupervised Clustering (The $2 \times 2$ Matrix) & Technical Pivot

We evaluated 4 combinations ($K$-Means vs. HDBSCAN $\times$ SVD vs. SBERT):

* **Spherical $K$-Means ($K=12\text{--}15$):**
  * Partitions space into convex regions on unit-normalized vectors (cosine geometry).
  * Guarantees 100% coverage with balanced cluster allocations.
* **HDBSCAN Pivot (Great Talking Point / "Insight"):**
  * *The Issue:* Running HDBSCAN directly on raw 100-d SVD or 384-d SBERT failed due to the *curse of dimensionality* (distance concentration made densities almost uniform, collapsing ~95%+ of points into a single cluster).
  * *The Solution:* We shifted HDBSCAN to operate on non-linear **UMAP 2D projections**. In 2D manifold space, density boundaries are preserved, producing clean, separable clusters with sensible outlier detection.

---

### Slide 5: Quantitative Preliminary Results

| Corpus | Articles | $K$-Means $K$ | HDBSCAN (SVD 2D) | HDBSCAN (SBERT 2D) | HDBSCAN Silhouette |
|---|---|---|---|---|---|
| `arizona` | 1,455 | 15 | 3 clusters (0% noise) | 2 clusters (1.5% noise) | **0.35 – 0.53** |
| `computer_science` | 1,597 | 15 | 10 clusters (5.1% noise) | 15 clusters (18.9% noise) | **0.43 – 0.46** |
| `literature` | 1,424 | 15 | 5 clusters (0.5% noise) | 2 clusters (0% noise) | **0.51 – 0.77** |
| `vital_level_3` | 999 | 12 | 16 clusters (18.1% noise) | 9 clusters (19.9% noise) | **0.31 – 0.52** |

* **Visual Evidence:** High-resolution 4-panel scatter plots generated in `data/derived/figures/{corpus}_umap_clusters.png` ready to be dropped into the presentation.

---

### Slide 6: Timeline & Next Steps (Forward-Looking Plan)

* **Completed So Far:**
  * ✅ Text extraction, cleaning, and sanitization across all 4 corpora.
  * ✅ Dual feature pipelines (SVD & SBERT) and verification tests.
  * ✅ $2 \times 2$ clustering matrix + UMAP 2D projections and diagnostic plots.
  * ✅ Consolidated pipeline under [`scripts/text_based_clustering/`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/text_based_clustering).
* **Next Steps (On Track for Final Deadline):**
  1. **Step 4: Quantitative Validation (Next 3–5 days):** Compute external ground-truth validation (NMI, ARI) using Wikipedia category taxonomies.
  2. **Step 5: Graph Branch & Leiden (Following week):** Ingest link edge lists (`.edges.tsv`), run the Leiden community detection algorithm.
  3. **Step 6: Multimodal Comparison:** Compute Node Divergence Index (NDI) to identify articles whose topical content diverges from their link topology.