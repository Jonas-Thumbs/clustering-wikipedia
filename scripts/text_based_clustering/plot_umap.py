"""Generate 2x2 grid scatter plots overlaying cluster assignments onto UMAP 2D projections.

Grid layout per corpus (4 subplots):
  (Row 1, Col 1): UMAP of SVD   | Overlay: K-Means (SVD)
  (Row 1, Col 2): UMAP of SBERT | Overlay: K-Means (SBERT)
  (Row 2, Col 1): UMAP of SVD   | Overlay: HDBSCAN (SVD) - noise highlighted in grey
  (Row 2, Col 2): UMAP of SBERT | Overlay: HDBSCAN (SBERT) - noise highlighted in grey

Reads:
  data/derived/projections/{corpus}_umap_svd.npy
  data/derived/projections/{corpus}_umap_sbert.npy
  data/derived/clusters/{corpus}_clusters.csv

Writes:
  data/derived/figures/{corpus}_umap_clusters.png
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# Ensure writable matplotlib cache dir
cache_dir = Path(__file__).parent.parent.parent / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib.pyplot as plt
import numpy as np

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
PROJECTIONS_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "projections"
CLUSTERS_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "clusters"
FIG_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "figures"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plot_clusters_for_corpus(corpus: str) -> None:
    log(f"--- Plotting UMAP Cluster Overlays: {corpus} ---")
    t0 = time.time()

    umap_svd_file = PROJECTIONS_DIR / f"{corpus}_umap_svd.npy"
    umap_sbert_file = PROJECTIONS_DIR / f"{corpus}_umap_sbert.npy"
    clusters_file = CLUSTERS_DIR / f"{corpus}_clusters.csv"

    if not (umap_svd_file.exists() and umap_sbert_file.exists() and clusters_file.exists()):
        log(f"  [ERROR] Missing required files for {corpus}")
        return

    # Load 2D coordinates
    coords_svd = np.load(umap_svd_file)
    coords_sbert = np.load(umap_sbert_file)

    # Load cluster labels
    km_svd: list[int] = []
    km_sbert: list[int] = []
    hdb_svd: list[int] = []
    hdb_sbert: list[int] = []

    with open(clusters_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            km_svd.append(int(r["kmeans_svd"]))
            km_sbert.append(int(r["kmeans_sbert"]))
            hdb_svd.append(int(r["hdbscan_svd"]))
            hdb_sbert.append(int(r["hdbscan_sbert"]))

    km_svd_arr = np.array(km_svd)
    km_sbert_arr = np.array(km_sbert)
    hdb_svd_arr = np.array(hdb_svd)
    hdb_sbert_arr = np.array(hdb_sbert)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    fig.suptitle(f"Corpus: {corpus.replace('_', ' ').title()} - UMAP 2D Semantic Projections & Cluster Overlays", fontsize=16, y=0.98)

    # --- Subplot 1: SVD + K-Means ---
    ax = axes[0, 0]
    scatter = ax.scatter(
        coords_svd[:, 0],
        coords_svd[:, 1],
        c=km_svd_arr,
        cmap="tab20",
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_title(f"TF-IDF + SVD (LSA) | K-Means (K={len(set(km_svd_arr))})", fontsize=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # --- Subplot 2: SBERT + K-Means ---
    ax = axes[0, 1]
    scatter = ax.scatter(
        coords_sbert[:, 0],
        coords_sbert[:, 1],
        c=km_sbert_arr,
        cmap="tab20",
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_title(f"SBERT Dense (384-d) | K-Means (K={len(set(km_sbert_arr))})", fontsize=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    # --- Subplot 3: SVD + HDBSCAN ---
    ax = axes[1, 0]
    n_hdb_svd = len(set(hdb_svd_arr) - {-1})
    noise_svd_pct = np.mean(hdb_svd_arr == -1) * 100
    # Plot noise points first in subtle light grey
    mask_noise = hdb_svd_arr == -1
    ax.scatter(
        coords_svd[mask_noise, 0],
        coords_svd[mask_noise, 1],
        c="#d3d3d3",
        s=10,
        alpha=0.4,
        label=f"Noise ({noise_svd_pct:.1f}%)",
    )
    # Plot non-noise clusters
    mask_clustered = ~mask_noise
    if mask_clustered.any():
        ax.scatter(
            coords_svd[mask_clustered, 0],
            coords_svd[mask_clustered, 1],
            c=hdb_svd_arr[mask_clustered],
            cmap="turbo",
            s=14,
            alpha=0.9,
            edgecolors="none",
        )
    ax.set_title(f"TF-IDF + SVD | HDBSCAN ({n_hdb_svd} clusters, {noise_svd_pct:.1f}% noise)", fontsize=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="upper right", markerscale=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    # --- Subplot 4: SBERT + HDBSCAN ---
    ax = axes[1, 1]
    n_hdb_sbert = len(set(hdb_sbert_arr) - {-1})
    noise_sbert_pct = np.mean(hdb_sbert_arr == -1) * 100
    mask_noise_sb = hdb_sbert_arr == -1
    ax.scatter(
        coords_sbert[mask_noise_sb, 0],
        coords_sbert[mask_noise_sb, 1],
        c="#d3d3d3",
        s=10,
        alpha=0.4,
        label=f"Noise ({noise_sbert_pct:.1f}%)",
    )
    mask_clustered_sb = ~mask_noise_sb
    if mask_clustered_sb.any():
        ax.scatter(
            coords_sbert[mask_clustered_sb, 0],
            coords_sbert[mask_clustered_sb, 1],
            c=hdb_sbert_arr[mask_clustered_sb],
            cmap="turbo",
            s=14,
            alpha=0.9,
            edgecolors="none",
        )
    ax.set_title(f"SBERT Dense | HDBSCAN ({n_hdb_sbert} clusters, {noise_sbert_pct:.1f}% noise)", fontsize=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="upper right", markerscale=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_img = FIG_DIR / f"{corpus}_umap_clusters.png"
    plt.savefig(out_img, bbox_inches="tight")
    plt.close(fig)

    log(f"  Saved figure -> {out_img} in {time.time() - t0:.1f}s\n")


def main() -> None:
    t_start = time.time()
    for corpus in CORPORA:
        plot_clusters_for_corpus(corpus)
    log(f"=== ALL UMAP VISUALIZATIONS GENERATED in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()

