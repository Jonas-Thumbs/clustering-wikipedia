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

### Slide 4: Unsupervised Clustering & The Unified 10D Subspace

We evaluated 4 combinations ($K$-Means vs. HDBSCAN $\times$ SVD vs. SBERT) using an **intermediate 10-dimensional UMAP subspace**:

* **Why an Intermediate 10D Subspace? (The Methodological Core):**
  * *High-Dim Dilemma:* HDBSCAN directly on raw 100-d or 384-d vectors suffered from the *curse of dimensionality* (distances concentrated, resulting in 62%–82% noise attrition).
  * *2D Manifold Dilemma:* While 2D was great for visualization, clustering purely in 2D forced $K$-Means and HDBSCAN to compete on a heavily compressed plane where multi-faceted topic relationships were lost.
  * *The Gold-Standard Solution (BERTopic Architecture):* We reduced high-dim features to a **10D intermediate subspace** for clustering, and then projected that 10D space to **2D strictly for plotting**.
  * *Level Playing Field:* Both $K$-Means and HDBSCAN now cluster the **exact same 10-dimensional representation**, making algorithm comparisons mathematically sound and fair.

* **Clustering Configuration:**
  * **Spherical $K$-Means ($K=12\text{--}15$):** Normalized on unit 10D hypersphere; yields convex partitions with high coherence.
  * **HDBSCAN (`leaf` method, `min_cluster_size=20, min_samples=5`):** Discovers fine-grained, localized density clusters in 10D while keeping noise controlled (~18%–33%).

---

### Slide 5: Quantitative Preliminary Results (Unified 10D Subspace)

| Corpus | Articles | $K$-Means $K$ | HDBSCAN (SVD 10D) | HDBSCAN (SBERT 10D) | $K$-Means Silhouette (10D) | HDBSCAN Silhouette (10D) |
|---|---|---|---|---|---|---|
| `arizona` | 1,455 | 15 | **27 clusters** (30.4% noise) | **23 clusters** (33.7% noise) | **0.44 – 0.48** | **0.52 – 0.62** |
| `computer_science` | 1,597 | 15 | **26 clusters** (27.3% noise) | **22 clusters** (30.9% noise) | **0.48 – 0.54** | **0.53 – 0.54** |
| `literature` | 1,424 | 15 | **22 clusters** (33.6% noise) | **19 clusters** (28.2% noise) | **0.42 – 0.46** | **0.51 – 0.57** |
| `vital_level_3` | 999 | 12 | **21 clusters** (18.4% noise) | **17 clusters** (30.0% noise) | **0.43 – 0.51** | **0.52 – 0.59** |

* **Visual & Interactive Assets:** 
  * High-resolution 4-panel scatter plots generated in `data/derived/figures/{corpus}_umap_clusters.png`.
  * Standalone interactive browser tool in `scripts/interactive_explorer/web/index.html` displaying the 2D visualization manifold with live article title hover and cluster metrics.

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