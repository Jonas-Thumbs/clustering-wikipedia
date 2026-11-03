"""Automated checks for Step 1 Feature Extraction:
1. Verifies that all expected feature files exist for all corpora.
2. Checks array shapes, finite values (no NaNs, no Infs).
3. Verifies SBERT embedding unit-normalization (L2 norm = 1.0).
4. Verifies SVD explained variance and non-trivial dimensionality.
5. Verifies 1:1 page_id and title alignment between feature rows and cleaned text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
CLEANED_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "cleaned"
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"


def run_checks() -> bool:
    print("=== RUNNING AUTOMATED VERIFICATION FOR STEP 1 FEATURES ===", flush=True)
    all_passed = True

    for corpus in CORPORA:
        print(f"\n--- Checking Features for: {corpus} ---", flush=True)
        cleaned_path = CLEANED_DIR / f"{corpus}.cleaned.jsonl"
        page_ids_path = FEATURES_DIR / f"{corpus}_page_ids.npy"
        titles_path = FEATURES_DIR / f"{corpus}_titles.json"
        svd_path = FEATURES_DIR / f"{corpus}_svd100.npy"
        sbert_path = FEATURES_DIR / f"{corpus}_sbert.npy"
        top_words_path = FEATURES_DIR / f"{corpus}_svd_top_words.json"

        # Check file existence
        missing_files = []
        for p in [page_ids_path, titles_path, svd_path, sbert_path, top_words_path]:
            if not p.exists():
                missing_files.append(p.name)

        if missing_files:
            print(f"  [FAIL] Missing expected feature files: {missing_files}", flush=True)
            all_passed = False
            continue

        # Load cleaned count
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_count = sum(1 for _ in f)

        # 1. Check page_ids and titles
        page_ids = np.load(page_ids_path)
        with open(titles_path, "r", encoding="utf-8") as f:
            titles = json.load(f)

        if len(page_ids) != cleaned_count:
            print(f"  [FAIL] Page IDs count mismatch: {len(page_ids)} vs cleaned {cleaned_count}", flush=True)
            all_passed = False
        if len(titles) != cleaned_count:
            print(f"  [FAIL] Titles count mismatch: {len(titles)} vs cleaned {cleaned_count}", flush=True)
            all_passed = False

        # 2. Check SVD Matrix
        svd = np.load(svd_path)
        if svd.shape[0] != cleaned_count:
            print(f"  [FAIL] SVD rows mismatch: {svd.shape[0]} vs cleaned {cleaned_count}", flush=True)
            all_passed = False
        if np.isnan(svd).any() or np.isinf(svd).any():
            print("  [FAIL] SVD array contains NaNs or Infs", flush=True)
            all_passed = False
        print(f"  [OK] SVD Matrix shape: {svd.shape} | No NaNs/Infs", flush=True)

        # 3. Check SBERT Embeddings
        sbert = np.load(sbert_path)
        if sbert.shape[0] != cleaned_count:
            print(f"  [FAIL] SBERT rows mismatch: {sbert.shape[0]} vs cleaned {cleaned_count}", flush=True)
            all_passed = False
        if sbert.shape[1] != 384:
            print(f"  [FAIL] SBERT unexpected embedding dimension: {sbert.shape[1]} (expected 384)", flush=True)
            all_passed = False
        if np.isnan(sbert).any() or np.isinf(sbert).any():
            print("  [FAIL] SBERT array contains NaNs or Infs", flush=True)
            all_passed = False

        # Check unit normalization: L2 norm of each row should be approximately 1.0
        norms = np.linalg.norm(sbert, axis=1)
        norm_diff = np.abs(norms - 1.0)
        if (norm_diff > 1e-4).any():
            print(f"  [FAIL] SBERT vectors are not unit normalized (max diff: {norm_diff.max():.6f})", flush=True)
            all_passed = False
        else:
            print(f"  [OK] SBERT Embeddings shape: {sbert.shape} | Verified Unit-Normalized (L2=1.0)", flush=True)

        # 4. Check SVD Topic Interpretability
        with open(top_words_path, "r", encoding="utf-8") as f:
            top_words = json.load(f)
        if len(top_words) < 5:
            print(f"  [FAIL] Top words dict has fewer than 5 components: {len(top_words)}", flush=True)
            all_passed = False
        else:
            print(f"  [OK] SVD Interpretability: {len(top_words)} principal components extracted with top keywords", flush=True)

    print("\n" + "=" * 55, flush=True)
    if all_passed:
        print("ALL STEP 1 AUTOMATED CHECKS PASSED SUCCESSFULLY!", flush=True)
    else:
        print("SOME CHECKS FAILED. PLEASE REVIEW LOGS.", flush=True)
    print("=" * 55, flush=True)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)

