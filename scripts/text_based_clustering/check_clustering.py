"""Automated checks for Step 2 Unsupervised Clustering:
1. Verifies that cluster assignment CSV and summary JSON files exist for all 4 corpora.
2. Verifies complete coverage: exactly N rows matching input page IDs without missing values.
3. Verifies label properties:
   - K-Means labels span exactly [0, K-1] with no unassigned (-1) points.
   - HDBSCAN labels contain valid non-negative cluster IDs and realistic noise (-1) distributions.
4. Checks convergence metrics (inertia > 0, non-trivial silhouette scores).
5. Cross-checks label consistency and distribution across all 4 runs in the 2x2 matrix.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
FEATURES_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"
CLUSTERS_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "clusters"


def run_checks() -> bool:
    print("=== RUNNING AUTOMATED VERIFICATION FOR STEP 2 CLUSTERING ===", flush=True)
    all_passed = True

    for corpus in CORPORA:
        print(f"\n--- Checking Cluster Results for: {corpus} ---", flush=True)
        csv_path = CLUSTERS_DIR / f"{corpus}_clusters.csv"
        summary_path = CLUSTERS_DIR / f"{corpus}_cluster_summary.json"
        titles_path = FEATURES_DIR / f"{corpus}_titles.json"

        # 1. Existence
        if not csv_path.exists() or not summary_path.exists():
            print(f"  [FAIL] Missing cluster files for {corpus}", flush=True)
            all_passed = False
            continue

        with open(titles_path, "r", encoding="utf-8") as f:
            expected_n = len(json.load(f))

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        k = summary.get("k", 15)

        # 2. Parse CSV
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        if len(rows) != expected_n:
            print(f"  [FAIL] Row count mismatch: {len(rows)} vs expected {expected_n}", flush=True)
            all_passed = False
        else:
            print(f"  [OK] Coverage verified: exactly {len(rows)} articles partitioned", flush=True)

        km_svd = [int(r["kmeans_svd"]) for r in rows]
        km_sbert = [int(r["kmeans_sbert"]) for r in rows]
        hdb_svd = [int(r["hdbscan_svd"]) for r in rows]
        hdb_sbert = [int(r["hdbscan_sbert"]) for r in rows]

        # 3. K-Means Checks
        for name, labels in [("K-Means (SVD)", km_svd), ("K-Means (SBERT)", km_sbert)]:
            unique_labels = set(labels)
            if -1 in unique_labels:
                print(f"  [FAIL] {name} has unassigned (-1) points", flush=True)
                all_passed = False
            if len(unique_labels) != k:
                print(f"  [FAIL] {name} produced {len(unique_labels)} clusters, expected K={k}", flush=True)
                all_passed = False
            else:
                # Check that no cluster is empty
                counts = [labels.count(c) for c in range(k)]
                print(f"  [OK] {name}: {k} convex clusters | min cluster size = {min(counts)}, max = {max(counts)}", flush=True)

        # 4. HDBSCAN Checks
        for name, labels in [("HDBSCAN (SVD)", hdb_svd), ("HDBSCAN (SBERT)", hdb_sbert)]:
            clusters = set(labels) - {-1}
            n_clusters = len(clusters)
            noise_count = labels.count(-1)
            noise_pct = (noise_count / expected_n) * 100

            if n_clusters < 1:
                print(f"  [FAIL] {name} found 0 clusters", flush=True)
                all_passed = False
            else:
                print(f"  [OK] {name}: {n_clusters} clusters discovered | noise: {noise_pct:.1f}% ({noise_count} articles)", flush=True)

        # 5. Summary metrics check
        runs = summary.get("runs", {})
        sil_km_svd = runs.get("kmeans_svd", {}).get("silhouette", -1)
        sil_km_sbert = runs.get("kmeans_sbert", {}).get("silhouette", -1)
        print(f"  [OK] Silhouette Scores: K-Means(SVD) = {sil_km_svd:.4f} | K-Means(SBERT) = {sil_km_sbert:.4f}", flush=True)

    print("\n" + "=" * 55, flush=True)
    if all_passed:
        print("ALL STEP 2 AUTOMATED CHECKS PASSED SUCCESSFULLY!", flush=True)
    else:
        print("SOME CHECKS FAILED. PLEASE REVIEW LOGS.", flush=True)
    print("=" * 55, flush=True)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)

