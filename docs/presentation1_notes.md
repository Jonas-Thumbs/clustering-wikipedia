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

### Slide 4: Unsupervised Clustering (The $2 \times 2$ Matrix) & The "Clustering Space" Discussion

We evaluated 4 combinations ($K$-Means vs. HDBSCAN $\times$ SVD vs. SBERT):

* **Spherical $K$-Means ($K=12\text{--}15$):**
  * Operates directly on the **full high-dimensional semantic space** (100-d SVD / 384-d SBERT) normalized on the unit hypersphere (cosine geometry).
  * Guarantees 100% coverage with balanced partitions.
  * **Visual Note / 2D Overlap:** In 2D UMAP scatter plots, $K$-Means clusters visually overlap because projecting 384-d down to 2D squashes orthogonal semantic dimensions. Clusters separated in high dimensions can appear adjacent when compressed into 2D.

* **HDBSCAN Pivot & Parameter Tuning (Great Talking Point / "Insight"):**
  * *The High-Dim Dilemma:* Running density-based clustering on raw 100-d or 384-d embeddings fails due to the *curse of dimensionality* (distance concentration causes Euclidean distances to become roughly uniform, collapsing ~95%+ of points into noise or a single cluster).
  * *The 2D Solution:* Running HDBSCAN on non-linear **UMAP 2D projections** restores meaningful density gradients.
  * *From `eom` to `leaf` Selection:* Initially, default `cluster_selection_method='eom'` (Excess of Mass) formed 2–3 giant macro-clusters on continuous manifolds like Arizona. Switching to **`cluster_selection_method='leaf'`** forces HDBSCAN to extract the finest branch leaves of the condensed tree, discovering **17–22 fine-grained, localized topic sub-clusters** across corpora.
  * *Methodological Comparison (Presentation Highlight):* 
    * Is running $K$-Means in $\mathbb{R}^{384}$ vs. HDBSCAN in $\mathbb{R}^2$ unfair? Yes, they operate in different spaces!
    * **Future Improvement / Best Practice (BERTopic style):** Cluster in an intermediate 10-dimensional UMAP space ($\mathbb{R}^{10}$) — high enough to preserve complex semantic relationships, yet low enough to enable reliable density estimation.

---

### Slide 5: Quantitative Preliminary Results

| Corpus | Articles | $K$-Means $K$ | HDBSCAN (SVD 2D, `leaf`) | HDBSCAN (SBERT 2D, `leaf`) | Non-Noise Silhouette |
|---|---|---|---|---|---|
| `arizona` | 1,455 | 15 | **18 clusters** (32.0% noise) | **21 clusters** (35.0% noise) | **0.50 – 0.53** |
| `computer_science` | 1,597 | 15 | **22 clusters** (31.6% noise) | **19 clusters** (38.4% noise) | **0.53 – 0.54** |
| `literature` | 1,424 | 15 | **20 clusters** (33.8% noise) | **19 clusters** (38.6% noise) | **0.56 – 0.58** |
| `vital_level_3` | 999 | 12 | **17 clusters** (20.6% noise) | **13 clusters** (28.6% noise) | **0.48 – 0.50** |

* **Visual Evidence:** 
  * High-resolution 4-panel scatter plots generated in `data/derived/figures/{corpus}_umap_clusters.png` ready to be dropped into the presentation slides.
  * Standalone interactive browser tool in `scripts/interactive_explorer/web/index.html` for live demonstrations and point-level article inspection.

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