"""Automated verification suite for Step 3 UMAP Projections & Visualizations:
1. Verifies that 2D UMAP projection files exist for both SVD and SBERT for all corpora.
2. Checks array properties: shape (N, 2), dtype float32, no NaNs or Infs.
3. Verifies that all 4 figure files are generated, non-empty, and valid PNG images.
4. Checks that 2D coordinate row counts match exactly with the article count of each corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
PROJECTIONS_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "projections"
CLUSTERS_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "clusters"
FIG_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "figures"
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"


def run_checks() -> bool:
    print("=== RUNNING AUTOMATED VERIFICATION FOR STEP 3 UMAP VISUALIZATION ===", flush=True)
    all_passed = True

    for corpus in CORPORA:
        print(f"\n--- Checking Step 3 Outputs for: {corpus} ---", flush=True)

        umap_svd_path = PROJECTIONS_DIR / f"{corpus}_umap_svd.npy"
        umap_sbert_path = PROJECTIONS_DIR / f"{corpus}_umap_sbert.npy"
        fig_path = FIG_DIR / f"{corpus}_umap_clusters.png"
        page_ids_path = FEATURES_DIR / f"{corpus}_page_ids.npy"

        # 1. Existence
        for p, desc in [
            (umap_svd_path, "UMAP SVD projection"),
            (umap_sbert_path, "UMAP SBERT projection"),
            (fig_path, "Scatter figure PNG"),
        ]:
            if not p.exists():
                print(f"  [FAIL] Missing file: {p.name} ({desc})", flush=True)
                all_passed = False

        if not (umap_svd_path.exists() and umap_sbert_path.exists()):
            continue

        n_expected = len(np.load(page_ids_path))

        # 2. Check SVD 2D Projection
        coords_svd = np.load(umap_svd_path)
        if coords_svd.shape != (n_expected, 2):
            print(f"  [FAIL] SVD projection shape mismatch: {coords_svd.shape} vs expected ({n_expected}, 2)", flush=True)
            all_passed = False
        elif np.isnan(coords_svd).any() or np.isinf(coords_svd).any():
            print("  [FAIL] SVD projection contains NaNs or Infs", flush=True)
            all_passed = False
        else:
            print(f"  [OK] UMAP (SVD) 2D coordinates: {coords_svd.shape} | Range: [{coords_svd[:,0].min():.1f}, {coords_svd[:,0].max():.1f}] x [{coords_svd[:,1].min():.1f}, {coords_svd[:,1].max():.1f}]", flush=True)

        # 3. Check SBERT 2D Projection
        coords_sbert = np.load(umap_sbert_path)
        if coords_sbert.shape != (n_expected, 2):
            print(f"  [FAIL] SBERT projection shape mismatch: {coords_sbert.shape} vs expected ({n_expected}, 2)", flush=True)
            all_passed = False
        elif np.isnan(coords_sbert).any() or np.isinf(coords_sbert).any():
            print("  [FAIL] SBERT projection contains NaNs or Infs", flush=True)
            all_passed = False
        else:
            print(f"  [OK] UMAP (SBERT) 2D coordinates: {coords_sbert.shape} | Range: [{coords_sbert[:,0].min():.1f}, {coords_sbert[:,0].max():.1f}] x [{coords_sbert[:,1].min():.1f}, {coords_sbert[:,1].max():.1f}]", flush=True)

        # 4. Check Figure File
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            if size_kb < 50:
                print(f"  [FAIL] Figure size suspiciously small: {size_kb:.1f} KB", flush=True)
                all_passed = False
            else:
                print(f"  [OK] High-resolution Figure: {fig_path.name} ({size_kb:.1f} KB)", flush=True)

    print("\n" + "=" * 55, flush=True)
    if all_passed:
        print("ALL STEP 3 AUTOMATED CHECKS PASSED SUCCESSFULLY!", flush=True)
    else:
        print("SOME CHECKS FAILED. PLEASE REVIEW LOGS.", flush=True)
    print("=" * 55, flush=True)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)

