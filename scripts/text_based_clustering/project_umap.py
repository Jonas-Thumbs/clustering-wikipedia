"""Project high-dimensional TF-IDF/SVD and SBERT vectors into:
1. 10D intermediate subspace for clustering (K-Means & HDBSCAN)
2. 2D visualization manifold for scatter plots and interactive exploration

This architecture follows the modular dimensionality reduction pattern (e.g., BERTopic):
  High-dim (100d / 384d) -> [10D Subspace for Clustering] -> [2D Manifold for Plotting]

The reducer class and parameters are parameterized so alternative dimensionality reduction
techniques (e.g., PCA, PaCMAP, t-SNE, Autoencoders) can be swapped in seamlessly.

Reads:
  data/derived/features/{corpus}_svd100.npy
  data/derived/features/{corpus}_sbert.npy

Writes:
  data/derived/projections/{corpus}_umap10d_svd.npy   (N, 10)
  data/derived/projections/{corpus}_umap10d_sbert.npy (N, 10)
  data/derived/projections/{corpus}_umap_svd.npy      (N, 2)
  data/derived/projections/{corpus}_umap_sbert.npy    (N, 2)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

# Ensure writable matplotlib cache dir
cache_dir = Path(__file__).parent.parent.parent / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import numpy as np
import umap
from sklearn.preprocessing import normalize

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
ROOT_DIR = Path(__file__).parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data" / "derived" / "features"
OUT_DIR = ROOT_DIR / "data" / "derived" / "projections"

# Intermediate clustering dimensionality
INTERMEDIATE_DIM = 10
VISUALIZATION_DIM = 2


class DimensionalityReducer(Protocol):
    """Protocol for modular dimensionality reduction algorithms."""
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


def build_umap_reducer(
    n_components: int,
    metric: str = "cosine",
    n_neighbors: int = 25,
    min_dist: float = 0.1,
    random_state: int = 42,
    **kwargs: Any,
) -> umap.UMAP:
    """Factory creating a UMAP reducer instance."""
    return umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reduce_representation(
    X: np.ndarray,
    rep_name: str,
    intermediate_dim: int = INTERMEDIATE_DIM,
    vis_dim: int = VISUALIZATION_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel reduction from original feature space:
    1. High-Dim -> 10D intermediate subspace (for clustering)
    2. High-Dim -> 2D visual projection directly (avoids manifold collapse / stringy worm artifacts)
    """
    t0 = time.time()
    log(f"  Stage 1: Reducing {rep_name} ({X.shape[1]}d -> {intermediate_dim}d) for clustering...")
    reducer_10d = build_umap_reducer(n_components=intermediate_dim, metric="cosine", min_dist=0.0)
    X_10d = reducer_10d.fit_transform(X).astype(np.float32)
    log(f"    Completed {rep_name} 10D projection in {time.time() - t0:.1f}s | Shape: {X_10d.shape}")

    t1 = time.time()
    log(f"  Stage 2: Projecting {rep_name} ({X.shape[1]}d -> {vis_dim}d) directly from original features for 2D visualization...")
    reducer_2d = build_umap_reducer(n_components=vis_dim, metric="cosine", min_dist=0.1)
    X_2d = reducer_2d.fit_transform(X).astype(np.float32)
    log(f"    Completed {rep_name} 2D projection in {time.time() - t1:.1f}s | Shape: {X_2d.shape}")

    return X_10d, X_2d

    return X_10d, X_2d


def project_corpus(corpus: str) -> None:
    log(f"--- Computing 10D & 2D Projections for: {corpus} ---")
    t0 = time.time()

    svd_file = FEATURES_DIR / f"{corpus}_svd100.npy"
    sbert_file = FEATURES_DIR / f"{corpus}_sbert.npy"

    if not (svd_file.exists() and sbert_file.exists()):
        log(f"  [ERROR] Feature files missing for {corpus}")
        return

    X_svd = np.load(svd_file)
    X_sbert = np.load(sbert_file)

    # Normalize SVD components for cosine metric
    X_svd_norm = normalize(X_svd, norm="l2")

    # 1. SVD reduction: 100d -> 10d -> 2d
    svd_10d, svd_2d = reduce_representation(X_svd_norm, "SVD (100d)")

    # 2. SBERT reduction: 384d -> 10d -> 2d
    sbert_10d, sbert_2d = reduce_representation(X_sbert, "SBERT (384d)")

    # Save projections
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{corpus}_umap10d_svd.npy", svd_10d)
    np.save(OUT_DIR / f"{corpus}_umap10d_sbert.npy", sbert_10d)
    np.save(OUT_DIR / f"{corpus}_umap_svd.npy", svd_2d)
    np.save(OUT_DIR / f"{corpus}_umap_sbert.npy", sbert_2d)

    log(f"  Saved 10D & 2D projections -> {OUT_DIR} in {time.time() - t0:.1f}s\n")


def main() -> None:
    t_start = time.time()
    for corpus in CORPORA:
        project_corpus(corpus)
    log(f"=== ALL MODULAR UMAP PROJECTIONS COMPLETE in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()
