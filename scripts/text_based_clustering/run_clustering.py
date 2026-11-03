"""Execute the 2x2 Unsupervised Clustering Matrix across all 4 corpora.

The 2x2 Matrix:
1. TF-IDF + SVD  x  K-Means (Spherical K-Means via L2 normalization)
2. SBERT         x  K-Means (Spherical K-Means on unit sphere)
3. TF-IDF + SVD  x  HDBSCAN (Density-based clustering on UMAP 2D projection of SVD)
4. SBERT         x  HDBSCAN (Density-based clustering on UMAP 2D projection of SBERT)

Design rationale for HDBSCAN:
  Running HDBSCAN on raw high-dimensional vectors (100-d SVD or 384-d SBERT) suffers from
  the curse of dimensionality — Euclidean distances become nearly uniform, so HDBSCAN cannot
  find meaningful density gradients and lumps almost everything into one cluster.

  Fix: run HDBSCAN on the **pre-computed UMAP 2D projections** (saved by step3_visualization).
  In 2D, density differences are sharp and Euclidean distance is meaningful again.
  Parameters tuned by grid search: eom, min_cluster_size=25, min_samples=5.

Outputs:
  data/derived/clusters/{corpus}_clusters.csv
    Columns: page_id, title, kmeans_svd, kmeans_sbert, hdbscan_svd, hdbscan_sbert
  data/derived/clusters/{corpus}_cluster_summary.json
    Metadata: k chosen, inertia, silhouette scores, HDBSCAN noise ratios & cluster counts
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"
PROJECTIONS_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "projections"
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "clusters"

# Optimal K selection per corpus reflecting scale and domain breadth
K_VALUES = {
    "arizona": 15,
    "computer_science": 15,
    "literature": 15,
    "vital_level_3": 12,
}

# HDBSCAN parameters for 2D UMAP space
# eom (excess of mass) gives more stable macro-clusters vs. leaf which over-fragments
HDBSCAN_PARAMS = dict(
    min_cluster_size=25,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cluster_corpus(corpus: str) -> None:
    log(f"=== Running 2x2 Clustering Matrix for: {corpus} ===")
    t0 = time.time()

    # --- Paths ---
    page_ids_path = FEATURES_DIR / f"{corpus}_page_ids.npy"
    titles_path = FEATURES_DIR / f"{corpus}_titles.json"
    svd_path = FEATURES_DIR / f"{corpus}_svd100.npy"
    sbert_path = FEATURES_DIR / f"{corpus}_sbert.npy"
    umap_svd_path = PROJECTIONS_DIR / f"{corpus}_umap_svd.npy"
    umap_sbert_path = PROJECTIONS_DIR / f"{corpus}_umap_sbert.npy"

    missing = [p for p in [page_ids_path, svd_path, sbert_path, umap_svd_path, umap_sbert_path] if not p.exists()]
    if missing:
        log(f"  [ERROR] Missing files for {corpus}: {[p.name for p in missing]}")
        return

    page_ids = np.load(page_ids_path)
    with open(titles_path, "r", encoding="utf-8") as f:
        titles = json.load(f)

    # --- Feature spaces ---
    X_svd = np.load(svd_path)    # (N, 100)
    X_sbert = np.load(sbert_path)  # (N, 384), already unit-normalized

    # L2-normalize SVD for Spherical K-Means (cosine geometry)
    X_svd_norm = normalize(X_svd, norm="l2")

    # UMAP 2D coordinates for HDBSCAN (density estimation is meaningful in low dim)
    umap_svd_2d = np.load(umap_svd_path)    # (N, 2)
    umap_sbert_2d = np.load(umap_sbert_path)  # (N, 2)

    k = K_VALUES.get(corpus, 15)
    summary: dict = {"corpus": corpus, "k": k, "runs": {}}

    # --- Run 1: K-Means on TF-IDF / SVD ---
    log(f"  Run 1/4: K-Means (K={k}) on SVD (L2-normalised, cosine geometry)...")
    km_svd = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km_svd = km_svd.fit_predict(X_svd_norm)
    sil_km_svd = float(silhouette_score(X_svd_norm, labels_km_svd))
    summary["runs"]["kmeans_svd"] = {
        "k": k,
        "inertia": float(km_svd.inertia_),
        "silhouette": sil_km_svd,
    }
    log(f"    Inertia: {km_svd.inertia_:.1f} | Silhouette: {sil_km_svd:.4f}")

    # --- Run 2: K-Means on SBERT ---
    log(f"  Run 2/4: K-Means (K={k}) on SBERT unit sphere...")
    km_sbert = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km_sbert = km_sbert.fit_predict(X_sbert)
    sil_km_sbert = float(silhouette_score(X_sbert, labels_km_sbert))
    summary["runs"]["kmeans_sbert"] = {
        "k": k,
        "inertia": float(km_sbert.inertia_),
        "silhouette": sil_km_sbert,
    }
    log(f"    Inertia: {km_sbert.inertia_:.1f} | Silhouette: {sil_km_sbert:.4f}")

    # --- Run 3: HDBSCAN on UMAP(SVD) 2D ---
    log("  Run 3/4: HDBSCAN on UMAP(SVD) 2D projection...")
    hdb_svd = HDBSCAN(**HDBSCAN_PARAMS)
    labels_hdb_svd = hdb_svd.fit_predict(umap_svd_2d)
    n_clusters_hdb_svd = int(len(set(labels_hdb_svd) - {-1}))
    noise_ratio_svd = float(np.mean(labels_hdb_svd == -1))
    mask_hdb_svd = labels_hdb_svd != -1
    sil_hdb_svd = (
        float(silhouette_score(umap_svd_2d[mask_hdb_svd], labels_hdb_svd[mask_hdb_svd]))
        if n_clusters_hdb_svd > 1
        else 0.0
    )
    summary["runs"]["hdbscan_svd"] = {
        "n_clusters": n_clusters_hdb_svd,
        "noise_ratio": noise_ratio_svd,
        "silhouette_non_noise": sil_hdb_svd,
        "input_space": "umap_svd_2d",
    }
    log(f"    Clusters: {n_clusters_hdb_svd} | Noise: {noise_ratio_svd:.1%} | Non-noise Silhouette: {sil_hdb_svd:.4f}")

    # --- Run 4: HDBSCAN on UMAP(SBERT) 2D ---
    log("  Run 4/4: HDBSCAN on UMAP(SBERT) 2D projection...")
    hdb_sbert = HDBSCAN(**HDBSCAN_PARAMS)
    labels_hdb_sbert = hdb_sbert.fit_predict(umap_sbert_2d)
    n_clusters_hdb_sbert = int(len(set(labels_hdb_sbert) - {-1}))
    noise_ratio_sbert = float(np.mean(labels_hdb_sbert == -1))
    mask_hdb_sbert = labels_hdb_sbert != -1
    sil_hdb_sbert = (
        float(silhouette_score(umap_sbert_2d[mask_hdb_sbert], labels_hdb_sbert[mask_hdb_sbert]))
        if n_clusters_hdb_sbert > 1
        else 0.0
    )
    summary["runs"]["hdbscan_sbert"] = {
        "n_clusters": n_clusters_hdb_sbert,
        "noise_ratio": noise_ratio_sbert,
        "silhouette_non_noise": sil_hdb_sbert,
        "input_space": "umap_sbert_2d",
    }
    log(f"    Clusters: {n_clusters_hdb_sbert} | Noise: {noise_ratio_sbert:.1%} | Non-noise Silhouette: {sil_hdb_sbert:.4f}")

    # --- Save Cluster Table ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{corpus}_clusters.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("page_id,title,kmeans_svd,kmeans_sbert,hdbscan_svd,hdbscan_sbert\n")
        for pid, title, c1, c2, c3, c4 in zip(
            page_ids, titles, labels_km_svd, labels_km_sbert, labels_hdb_svd, labels_hdb_sbert
        ):
            safe_title = f'"{title}"' if ("," in title or '"' in title) else title
            f.write(f"{pid},{safe_title},{c1},{c2},{c3},{c4}\n")

    summary_path = OUT_DIR / f"{corpus}_cluster_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"  Saved cluster assignments -> {csv_path} in {time.time() - t0:.1f}s\n")


def main() -> None:
    t_start = time.time()
    for corpus in CORPORA:
        cluster_corpus(corpus)
    log(f"=== ALL CORPORA 2x2 CLUSTERING COMPLETE in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()

