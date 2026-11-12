"""Execute the 2x2 Unsupervised Clustering Matrix in the 10D intermediate subspace.

The 2x2 Matrix (all operating in the unified 10-dimensional reduced subspace):
1. TF-IDF + SVD  (100d -> 10D) x K-Means
2. SBERT         (384d -> 10D) x K-Means
3. TF-IDF + SVD  (100d -> 10D) x HDBSCAN (leaf method)
4. SBERT         (384d -> 10D) x HDBSCAN (leaf method)

Methodological advantages of 10D clustering:
- Fair comparison: both K-Means and HDBSCAN cluster the exact same intermediate semantic geometry.
- Solves high-dim distance concentration for HDBSCAN without sacrificing multi-axis variance.
- Retains rich multi-cluster topical structures while maintaining low noise.

Outputs:
  data/derived/clusters/{corpus}_clusters.csv
    Columns: page_id, title, kmeans_svd, kmeans_sbert, hdbscan_svd, hdbscan_sbert
  data/derived/clusters/{corpus}_cluster_summary.json
    Metadata: k chosen, inertia, silhouette scores in 10D, HDBSCAN noise ratios & cluster counts
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
ROOT_DIR = Path(__file__).parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data" / "derived" / "features"
PROJECTIONS_DIR = ROOT_DIR / "data" / "derived" / "projections"
OUT_DIR = ROOT_DIR / "data" / "derived" / "clusters"

# Optimal K selection per corpus
K_VALUES = {
    "arizona": 15,
    "computer_science": 15,
    "literature": 15,
    "vital_level_3": 12,
}

# HDBSCAN parameters for 10D subspace
# 'leaf' extracts the fine-grained leaves of the condensed tree hierarchy
HDBSCAN_PARAMS = dict(
    min_cluster_size=20,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="leaf",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cluster_corpus(corpus: str) -> None:
    log(f"=== Running 10D Subspace 2x2 Clustering Matrix for: {corpus} ===")
    t0 = time.time()

    # --- Paths ---
    page_ids_path = FEATURES_DIR / f"{corpus}_page_ids.npy"
    titles_path = FEATURES_DIR / f"{corpus}_titles.json"
    umap10d_svd_path = PROJECTIONS_DIR / f"{corpus}_umap10d_svd.npy"
    umap10d_sbert_path = PROJECTIONS_DIR / f"{corpus}_umap10d_sbert.npy"

    missing = [p for p in [page_ids_path, titles_path, umap10d_svd_path, umap10d_sbert_path] if not p.exists()]
    if missing:
        log(f"  [ERROR] Missing files for {corpus}: {[p.name for p in missing]}")
        return

    page_ids = np.load(page_ids_path)
    with open(titles_path, "r", encoding="utf-8") as f:
        titles = json.load(f)

    # Load 10D intermediate representations
    X_svd_10d = np.load(umap10d_svd_path)      # (N, 10)
    X_sbert_10d = np.load(umap10d_sbert_path)  # (N, 10)

    # L2-normalize 10D representations for Spherical K-Means
    X_svd_10d_norm = normalize(X_svd_10d, norm="l2")
    X_sbert_10d_norm = normalize(X_sbert_10d, norm="l2")

    k = K_VALUES.get(corpus, 15)
    summary: dict = {
        "corpus": corpus,
        "k": k,
        "clustering_space": "10D UMAP Subspace",
        "runs": {},
    }

    # --- Run 1: K-Means on 10D SVD ---
    log(f"  Run 1/4: K-Means (K={k}) on 10D SVD...")
    km_svd = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km_svd = km_svd.fit_predict(X_svd_10d_norm)
    sil_km_svd = float(silhouette_score(X_svd_10d_norm, labels_km_svd))
    summary["runs"]["kmeans_svd"] = {
        "k": k,
        "inertia": float(km_svd.inertia_),
        "silhouette": sil_km_svd,
        "space": "10D",
    }
    log(f"    Inertia: {km_svd.inertia_:.1f} | Silhouette (10D): {sil_km_svd:.4f}")

    # --- Run 2: K-Means on 10D SBERT ---
    log(f"  Run 2/4: K-Means (K={k}) on 10D SBERT...")
    km_sbert = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km_sbert = km_sbert.fit_predict(X_sbert_10d_norm)
    sil_km_sbert = float(silhouette_score(X_sbert_10d_norm, labels_km_sbert))
    summary["runs"]["kmeans_sbert"] = {
        "k": k,
        "inertia": float(km_sbert.inertia_),
        "silhouette": sil_km_sbert,
        "space": "10D",
    }
    log(f"    Inertia: {km_sbert.inertia_:.1f} | Silhouette (10D): {sil_km_sbert:.4f}")

    # --- Run 3: HDBSCAN (leaf) on 10D SVD ---
    log("  Run 3/4: HDBSCAN (leaf) on 10D SVD...")
    hdb_svd = HDBSCAN(**HDBSCAN_PARAMS)
    labels_hdb_svd = hdb_svd.fit_predict(X_svd_10d)
    n_clusters_hdb_svd = int(len(set(labels_hdb_svd) - {-1}))
    noise_ratio_svd = float(np.mean(labels_hdb_svd == -1))
    mask_hdb_svd = labels_hdb_svd != -1
    sil_hdb_svd = (
        float(silhouette_score(X_svd_10d[mask_hdb_svd], labels_hdb_svd[mask_hdb_svd]))
        if n_clusters_hdb_svd > 1
        else 0.0
    )
    summary["runs"]["hdbscan_svd"] = {
        "n_clusters": n_clusters_hdb_svd,
        "noise_ratio": noise_ratio_svd,
        "silhouette_non_noise": sil_hdb_svd,
        "space": "10D",
    }
    log(f"    Clusters: {n_clusters_hdb_svd} | Noise: {noise_ratio_svd:.1%} | Non-noise Silhouette (10D): {sil_hdb_svd:.4f}")

    # --- Run 4: HDBSCAN (leaf) on 10D SBERT ---
    log("  Run 4/4: HDBSCAN (leaf) on 10D SBERT...")
    hdb_sbert = HDBSCAN(**HDBSCAN_PARAMS)
    labels_hdb_sbert = hdb_sbert.fit_predict(X_sbert_10d)
    n_clusters_hdb_sbert = int(len(set(labels_hdb_sbert) - {-1}))
    noise_ratio_sbert = float(np.mean(labels_hdb_sbert == -1))
    mask_hdb_sbert = labels_hdb_sbert != -1
    sil_hdb_sbert = (
        float(silhouette_score(X_sbert_10d[mask_hdb_sbert], labels_hdb_sbert[mask_hdb_sbert]))
        if n_clusters_hdb_sbert > 1
        else 0.0
    )
    summary["runs"]["hdbscan_sbert"] = {
        "n_clusters": n_clusters_hdb_sbert,
        "noise_ratio": noise_ratio_sbert,
        "silhouette_non_noise": sil_hdb_sbert,
        "space": "10D",
    }
    log(f"    Clusters: {n_clusters_hdb_sbert} | Noise: {noise_ratio_sbert:.1%} | Non-noise Silhouette (10D): {sil_hdb_sbert:.4f}")

    # --- Write Outputs ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{corpus}_clusters.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "page_id",
            "title",
            "kmeans_svd",
            "kmeans_sbert",
            "hdbscan_svd",
            "hdbscan_sbert",
        ])
        for pid, title, ks, kb, hs, hb in zip(
            page_ids,
            titles,
            labels_km_svd,
            labels_km_sbert,
            labels_hdb_svd,
            labels_hdb_sbert,
        ):
            writer.writerow([pid, title, int(ks), int(kb), int(hs), int(hb)])

    out_summary = OUT_DIR / f"{corpus}_cluster_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"  Saved 10D cluster assignments -> {out_csv} in {time.time() - t0:.1f}s\n")


def main() -> None:
    t_start = time.time()
    for corpus in CORPORA:
        cluster_corpus(corpus)
    log(f"=== ALL CORPORA 10D 2x2 CLUSTERING COMPLETE in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()
