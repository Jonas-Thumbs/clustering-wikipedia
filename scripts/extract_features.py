"""Extract text features across corpora:
1. TF-IDF + TruncatedSVD (100 components, LSA)
2. SBERT embeddings (all-MiniLM-L6-v2, 384 dimensions, normalized)

Outputs:
  data/derived/features/{corpus}_page_ids.npy
  data/derived/features/{corpus}_titles.json
  data/derived/features/{corpus}_svd100.npy
  data/derived/features/{corpus}_sbert.npy
  data/derived/features/{corpus}_svd_top_words.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from text_cleaning import clean_wikitext

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
DATA_DIR = Path(__file__).parent.parent / "data" / "derived" / "text"
OUT_DIR = Path(__file__).parent.parent / "data" / "derived" / "features"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_features_for_corpus(
    corpus: str,
    sbert_model: SentenceTransformer,
    device: str,
    svd_components: int = 100,
) -> None:
    corpus_path = DATA_DIR / f"{corpus}.articles.jsonl"
    log(f"--- Processing Corpus: {corpus} ({corpus_path.name}) ---")

    page_ids: list[int] = []
    titles: list[str] = []
    leads: list[str] = []
    full_texts: list[str] = []

    t0 = time.time()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            rec = json.loads(line)
            title = rec.get("title", "")
            wikitext = rec.get("wikitext", "")
            cleaned = clean_wikitext(wikitext)
            
            lead = cleaned["lead_text"]
            full = cleaned["full_text"]

            page_ids.append(rec["page_id"])
            titles.append(title)
            # Combine title + lead summary for dense semantic representation
            leads.append(f"{title}. {lead}" if lead else title)
            # Use full cleaned text for rich vocabulary TF-IDF modeling
            full_texts.append(full if full else title)

            if idx % 500 == 0:
                log(f"  Cleaned {idx:,} articles ({idx / (time.time() - t0):.1f} arts/s)")

    n_articles = len(page_ids)
    log(f"Finished cleaning {n_articles:,} articles in {time.time() - t0:.1f}s")

    # --- 1. TF-IDF + TruncatedSVD ---
    log("Fitting TF-IDF vectorizer (ngram_range=(1, 2), min_df=3, max_df=0.6)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(full_texts)
    log(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    n_comp = min(svd_components, tfidf_matrix.shape[1] - 1, n_articles - 1)
    log(f"Fitting TruncatedSVD ({n_comp} components)...")
    svd = TruncatedSVD(n_components=n_comp, random_state=42, algorithm="randomized")
    svd_matrix = svd.fit_transform(tfidf_matrix)
    explained_var = float(svd.explained_variance_ratio_.sum())
    log(f"  TruncatedSVD shape: {svd_matrix.shape} (total explained variance: {explained_var:.1%})")

    # Extract top keywords per principal component for interpretability
    vocab = np.array(tfidf.get_feature_names_out())
    top_words = {}
    for comp_idx in range(min(15, n_comp)):
        comp = svd.components_[comp_idx]
        top_pos = vocab[np.argsort(comp)[-10:][::-1]].tolist()
        top_neg = vocab[np.argsort(comp)[:10]].tolist()
        top_words[f"component_{comp_idx+1}"] = {
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    # --- 2. SBERT Dense Embeddings ---
    log(f"Encoding {n_articles:,} article summaries with SBERT on {device}...")
    sbert_t0 = time.time()
    sbert_embeddings = sbert_model.encode(
        leads,
        batch_size=64 if device == "cuda" else 16,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )
    log(f"  SBERT encoded in {time.time() - sbert_t0:.1f}s, shape: {sbert_embeddings.shape}")

    # --- 3. Save Artifacts ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{corpus}_page_ids.npy", np.array(page_ids))
    with open(OUT_DIR / f"{corpus}_titles.json", "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False)

    np.save(OUT_DIR / f"{corpus}_svd100.npy", svd_matrix.astype(np.float32))
    np.save(OUT_DIR / f"{corpus}_sbert.npy", sbert_embeddings.astype(np.float32))
    with open(OUT_DIR / f"{corpus}_svd_top_words.json", "w", encoding="utf-8") as f:
        json.dump(top_words, f, indent=2)

    log(f"Successfully saved all feature files for {corpus} -> {OUT_DIR}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Initializing SBERT model 'all-MiniLM-L6-v2' on device: {device}...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    total_t0 = time.time()
    for corpus in CORPORA:
        extract_features_for_corpus(corpus, sbert_model, device)

    log(f"=== ALL 4 CORPORA FEATURE EXTRACTION COMPLETE in {time.time() - total_t0:.1f}s ===")


if __name__ == "__main__":
    main()

