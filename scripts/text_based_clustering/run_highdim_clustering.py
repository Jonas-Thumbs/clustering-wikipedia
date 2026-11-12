"""Run HDBSCAN clustering directly on high-dimensional representations (100d SVD and 384d SBERT).

This script provides the experimental contrast to 2D UMAP clustering:
1. Runs HDBSCAN (cluster_selection_method='leaf') directly in high-dimensional feature space:
   - SVD (100-d, Euclidean/Cosine geometry)
   - SBERT (384-d, Euclidean/Cosine geometry)
2. Saves outputs to isolated directories to avoid touching the 2D UMAP baseline:
   - data/derived/clusters_highdim/{corpus}_clusters_highdim.csv
   - data/derived/clusters_highdim/{corpus}_cluster_summary_highdim.json
   - data/derived/figures_highdim/{corpus}_umap_clusters_highdim.png
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Matplotlib cache setup
cache_dir = Path(__file__).parent.parent.parent / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib.pyplot as plt

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
ROOT_DIR = Path(__file__).parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data" / "derived" / "features"
PROJECTIONS_DIR = ROOT_DIR / "data" / "derived" / "projections"
OUT_CLUSTERS_DIR = ROOT_DIR / "data" / "derived" / "clusters_highdim"
OUT_FIG_DIR = ROOT_DIR / "data" / "derived" / "figures_highdim"

K_VALUES = {
    "arizona": 15,
    "computer_science": 15,
    "literature": 15,
    "vital_level_3": 12,
}

# HDBSCAN parameters for high-dimensional feature space
HDBSCAN_HIGHDIM_PARAMS = dict(
    min_cluster_size=15,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="leaf",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_highdim_clustering_for_corpus(corpus: str) -> dict:
    log(f"=== Running High-Dim HDBSCAN for: {corpus} ===")
    t0 = time.time()

    page_ids_path = FEATURES_DIR / f"{corpus}_page_ids.npy"
    titles_path = FEATURES_DIR / f"{corpus}_titles.json"
    svd_path = FEATURES_DIR / f"{corpus}_svd100.npy"
    sbert_path = FEATURES_DIR / f"{corpus}_sbert.npy"

    page_ids = np.load(page_ids_path)
    with open(titles_path, "r", encoding="utf-8") as f:
        titles = json.load(f)

    svd = np.load(svd_path)
    sbert = np.load(sbert_path)

    # 1. Spherical K-Means baseline (on high-dim unit sphere)
    k = K_VALUES[corpus]
    log(f"  Run 1/4: K-Means (K={k}) on SVD (normalized)...")
    svd_norm = normalize(svd)
    km_svd = KMeans(n_clusters=k, random_state=42, n_init=10).fit(svd_norm)
    labels_km_svd = km_svd.labels_

    log(f"  Run 2/4: K-Means (K={k}) on SBERT (normalized)...")
    sbert_norm = normalize(sbert)
    km_sbert = KMeans(n_clusters=k, random_state=42, n_init=10).fit(sbert_norm)
    labels_km_sbert = km_sbert.labels_

    # 2. HDBSCAN directly on high-dim SVD (100 dimensions)
    log(f"  Run 3/4: High-Dim HDBSCAN directly on 100-d SVD...")
    hdb_svd = HDBSCAN(**HDBSCAN_HIGHDIM_PARAMS).fit(svd_norm)
    labels_hdb_svd = hdb_svd.labels_
    n_c_svd = len(set(labels_hdb_svd) - {-1})
    noise_svd = float((labels_hdb_svd == -1).mean())
    log(f"    SVD High-Dim HDBSCAN: {n_c_svd} clusters | Noise: {noise_svd*100:.1f}%")

    # 3. HDBSCAN directly on high-dim SBERT (384 dimensions)
    log(f"  Run 4/4: High-Dim HDBSCAN directly on 384-d SBERT...")
    hdb_sbert = HDBSCAN(**HDBSCAN_HIGHDIM_PARAMS).fit(sbert_norm)
    labels_hdb_sbert = hdb_sbert.labels_
    n_c_sbert = len(set(labels_hdb_sbert) - {-1})
    noise_sbert = float((labels_hdb_sbert == -1).mean())
    log(f"    SBERT High-Dim HDBSCAN: {n_c_sbert} clusters | Noise: {noise_sbert*100:.1f}%")

    # Compute high-dim silhouettes if non-trivial
    def get_sil(X, labels):
        mask = labels != -1
        if len(set(labels[mask])) > 1 and mask.sum() > len(set(labels[mask])):
            try:
                return float(silhouette_score(X[mask], labels[mask]))
            except Exception:
                return None
        return None

    sil_svd = get_sil(svd_norm, labels_hdb_svd)
    sil_sbert = get_sil(sbert_norm, labels_hdb_sbert)

    # Save CSV
    out_csv = OUT_CLUSTERS_DIR / f"{corpus}_clusters_highdim.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_id", "title", "kmeans_svd", "kmeans_sbert", "hdbscan_svd_highdim", "hdbscan_sbert_highdim"])
        for pid, title, ks, kb, hs, hb in zip(page_ids, titles, labels_km_svd, labels_km_sbert, labels_hdb_svd, labels_hdb_sbert):
            writer.writerow([pid, title, int(ks), int(kb), int(hs), int(hb)])

    # Summary
    summary = {
        "corpus": corpus,
        "n_articles": len(page_ids),
        "params": HDBSCAN_HIGHDIM_PARAMS,
        "kmeans_k": k,
        "hdbscan_svd_highdim": {
            "n_clusters": n_c_svd,
            "noise_ratio": round(noise_svd, 4),
            "non_noise_silhouette": round(sil_svd, 4) if sil_svd is not None else None,
        },
        "hdbscan_sbert_highdim": {
            "n_clusters": n_c_sbert,
            "noise_ratio": round(noise_sbert, 4),
            "non_noise_silhouette": round(sil_sbert, 4) if sil_sbert is not None else None,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    out_json = OUT_CLUSTERS_DIR / f"{corpus}_cluster_summary_highdim.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"  Saved high-dim cluster data -> {out_csv.name}")
    return {
        "corpus": corpus,
        "labels_km_svd": labels_km_svd,
        "labels_km_sbert": labels_km_sbert,
        "labels_hdb_svd": labels_hdb_svd,
        "labels_hdb_sbert": labels_hdb_sbert,
        "summary": summary,
    }


def plot_highdim_clusters_for_corpus(corpus: str, data: dict) -> None:
    log(f"--- Plotting High-Dim Cluster Overlays onto UMAP 2D: {corpus} ---")
    coords_svd = np.load(PROJECTIONS_DIR / f"{corpus}_umap_svd.npy")
    coords_sbert = np.load(PROJECTIONS_DIR / f"{corpus}_umap_sbert.npy")

    km_svd_arr = data["labels_km_svd"]
    km_sbert_arr = data["labels_km_sbert"]
    hdb_svd_arr = data["labels_hdb_svd"]
    hdb_sbert_arr = data["labels_hdb_sbert"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    fig.suptitle(
        f"Corpus: {corpus.replace('_', ' ').title()} - High-Dimensional HDBSCAN (Leaf) vs K-Means\n(Cluster Assignments Projected onto UMAP 2D Manifolds)",
        fontsize=15,
        y=0.98,
    )

    # 1. K-Means SVD
    ax = axes[0, 0]
    ax.scatter(coords_svd[:, 0], coords_svd[:, 1], c=km_svd_arr, cmap="tab20", s=12, alpha=0.85, edgecolors="none")
    ax.set_title(f"TF-IDF + SVD (100d) | K-Means (K={len(set(km_svd_arr))})", fontsize=11)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 2. K-Means SBERT
    ax = axes[0, 1]
    ax.scatter(coords_sbert[:, 0], coords_sbert[:, 1], c=km_sbert_arr, cmap="tab20", s=12, alpha=0.85, edgecolors="none")
    ax.set_title(f"SBERT (384d) | K-Means (K={len(set(km_sbert_arr))})", fontsize=11)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Helper for HDBSCAN scatter with noise
    def plot_hdb(ax, coords, labels, rep_name):
        n_clusters = len(set(labels) - {-1})
        noise_mask = labels == -1
        clustered_mask = ~noise_mask
        noise_pct = noise_mask.mean() * 100

        # Plot noise first (grey)
        if np.any(noise_mask):
            ax.scatter(
                coords[noise_mask, 0],
                coords[noise_mask, 1],
                c="#94a3b8",
                s=9,
                alpha=0.35,
                edgecolors="none",
                label=f"Noise ({noise_pct:.1f}%)",
            )

        # Plot clusters
        if np.any(clustered_mask):
            ax.scatter(
                coords[clustered_mask, 0],
                coords[clustered_mask, 1],
                c=labels[clustered_mask],
                cmap="tab20",
                s=14,
                alpha=0.9,
                edgecolors="none",
            )

        ax.set_title(f"High-Dim {rep_name} | HDBSCAN (Leaf) [Clusters={n_clusters}, Noise={noise_pct:.1f}%]", fontsize=11)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.grid(True, linestyle="--", alpha=0.3)
        if np.any(noise_mask):
            ax.legend(loc="upper right", framealpha=0.85, fontsize=9)

    # 3. High-Dim HDBSCAN on SVD (100d)
    plot_hdb(axes[1, 0], coords_svd, hdb_svd_arr, "SVD (100d)")

    # 4. High-Dim HDBSCAN on SBERT (384d)
    plot_hdb(axes[1, 1], coords_sbert, hdb_sbert_arr, "SBERT (384d)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_fig = OUT_FIG_DIR / f"{corpus}_umap_clusters_highdim.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved figure -> {out_fig.name}")


def main() -> None:
    OUT_CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for corpus in CORPORA:
        data = run_highdim_clustering_for_corpus(corpus)
        plot_highdim_clusters_for_corpus(corpus, data)
        all_summaries[corpus] = data["summary"]

    with open(OUT_CLUSTERS_DIR / "all_corpora_highdim_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    log("=== ALL HIGH-DIMENSIONAL CLUSTERING AND PLOTTING COMPLETE ===")


if __name__ == "__main__":
    main()

