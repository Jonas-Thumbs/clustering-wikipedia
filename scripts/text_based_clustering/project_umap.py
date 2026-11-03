"""Project high-dimensional TF-IDF/SVD and SBERT vectors into a 2D plane using UMAP.

Reads:
  data/derived/features/{corpus}_svd100.npy
  data/derived/features/{corpus}_sbert.npy

Writes:
  data/derived/projections/{corpus}_umap_svd.npy  (N, 2)
  data/derived/projections/{corpus}_umap_sbert.npy (N, 2)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent.parent.parent / ".cache" / "matplotlib")

import numpy as np
import umap
from sklearn.preprocessing import normalize

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "projections"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def project_corpus(corpus: str) -> None:
    log(f"--- Computing 2D UMAP Projections for: {corpus} ---")
    t0 = time.time()

    svd_file = FEATURES_DIR / f"{corpus}_svd100.npy"
    sbert_file = FEATURES_DIR / f"{corpus}_sbert.npy"

    if not (svd_file.exists() and sbert_file.exists()):
        log(f"  [ERROR] Feature files missing for {corpus}")
        return

    X_svd = np.load(svd_file)
    X_sbert = np.load(sbert_file)

    # 1. UMAP on SVD space (using cosine metric on L2-normalized leading components)
    log("  Projecting TF-IDF / SVD vectors (100d -> 2D) via UMAP (cosine metric)...")
    X_svd_norm = normalize(X_svd, norm="l2")
    reducer_svd = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=25,
        min_dist=0.1,
        random_state=42,
        n_jobs=-1,
    )
    t_svd = time.time()
    umap_svd = reducer_svd.fit_transform(X_svd_norm)
    log(f"    SVD 2D projection completed in {time.time() - t_svd:.1f}s | Shape: {umap_svd.shape}")

    # 2. UMAP on SBERT space (using cosine metric on unit-normalized embeddings)
    log("  Projecting SBERT embeddings (384d -> 2D) via UMAP (cosine metric)...")
    reducer_sbert = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=25,
        min_dist=0.1,
        random_state=42,
        n_jobs=-1,
    )
    t_sbert = time.time()
    umap_sbert = reducer_sbert.fit_transform(X_sbert)
    log(f"    SBERT 2D projection completed in {time.time() - t_sbert:.1f}s | Shape: {umap_sbert.shape}")

    # Save projections
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{corpus}_umap_svd.npy", umap_svd.astype(np.float32))
    np.save(OUT_DIR / f"{corpus}_umap_sbert.npy", umap_sbert.astype(np.float32))
    log(f"  Saved 2D projections -> {OUT_DIR} in {time.time() - t0:.1f}s\n")


def main() -> None:
    t_start = time.time()
    for corpus in CORPORA:
        project_corpus(corpus)
    log(f"=== ALL UMAP 2D PROJECTIONS COMPLETE in {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()

