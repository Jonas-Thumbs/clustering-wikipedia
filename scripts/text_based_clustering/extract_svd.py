"""Extract sparse TF-IDF and TruncatedSVD (LSA) features for all cleaned corpora.

Reads:
  data/derived/cleaned/{corpus}.cleaned.jsonl

Writes:
  data/derived/features/{corpus}_page_ids.npy
  data/derived/features/{corpus}_titles.json
  data/derived/features/{corpus}_svd100.npy
  data/derived/features/{corpus}_svd_top_words.json
  data/derived/features/{corpus}_svd_explained_variance.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
CLEANED_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "cleaned"
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_svd_for_corpus(corpus: str, n_components: int = 100) -> None:
    cleaned_path = CLEANED_DIR / f"{corpus}.cleaned.jsonl"
    log(f"--- SVD Extraction: {corpus} ({cleaned_path.name}) ---")

    if not cleaned_path.exists():
        log(f"  [ERROR] Cleaned file not found: {cleaned_path}")
        return

    page_ids: list[int] = []
    titles: list[str] = []
    full_texts: list[str] = []

    with open(cleaned_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_ids.append(rec["page_id"])
            titles.append(rec["title"])
            full_texts.append(rec["full_text"] if rec["full_text"] else rec["title"])

    n_docs = len(page_ids)
    log(f"  Loaded {n_docs:,} articles for TF-IDF vectorization.")

    # 1. TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
    )
    t0 = time.time()
    tfidf_matrix = tfidf.fit_transform(full_texts)
    log(f"  TF-IDF matrix built in {time.time() - t0:.1f}s | Shape: {tfidf_matrix.shape}")

    # 2. TruncatedSVD (LSA)
    k = min(n_components, tfidf_matrix.shape[1] - 1, n_docs - 1)
    svd = TruncatedSVD(n_components=k, random_state=42, algorithm="randomized")
    svd_matrix = svd.fit_transform(tfidf_matrix)
    explained_var = float(svd.explained_variance_ratio_.sum())
    log(f"  TruncatedSVD shape: {svd_matrix.shape} | Explained variance ratio: {explained_var:.2%}")

    # 3. Interpretability: Extract top semantic terms per principal component
    vocab = np.array(tfidf.get_feature_names_out())
    top_words = {}
    for comp_idx in range(min(15, k)):
        comp = svd.components_[comp_idx]
        top_pos = vocab[np.argsort(comp)[-10:][::-1]].tolist()
        top_neg = vocab[np.argsort(comp)[:10]].tolist()
        top_words[f"component_{comp_idx+1}"] = {
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    # 4. Save artifacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{corpus}_page_ids.npy", np.array(page_ids, dtype=np.int64))
    with open(OUT_DIR / f"{corpus}_titles.json", "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False)

    np.save(OUT_DIR / f"{corpus}_svd100.npy", svd_matrix.astype(np.float32))

    with open(OUT_DIR / f"{corpus}_svd_top_words.json", "w", encoding="utf-8") as f:
        json.dump(top_words, f, indent=2)

    with open(OUT_DIR / f"{corpus}_svd_explained_variance.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "corpus": corpus,
                "n_components": k,
                "total_explained_variance": explained_var,
                "explained_variance_per_component": svd.explained_variance_ratio_.tolist(),
            },
            f,
            indent=2,
        )

    log(f"  Saved SVD artifacts -> {OUT_DIR / f'{corpus}_svd100.npy'}")


def main() -> None:
    t_start = time.time()
    for corpus in CORPORA:
        extract_svd_for_corpus(corpus)
    log(f"All corpora SVD extraction complete in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

