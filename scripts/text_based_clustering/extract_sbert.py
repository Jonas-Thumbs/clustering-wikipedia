"""Extract dense contextual Sentence-BERT (SBERT) embeddings for all cleaned corpora.

Reads:
  data/derived/cleaned/{corpus}.cleaned.jsonl

Writes:
  data/derived/features/{corpus}_sbert.npy (unit-normalized 384-d vectors)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
CLEANED_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "cleaned"
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "derived" / "features"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_sbert_for_corpus(corpus: str, model: SentenceTransformer, device: str) -> None:
    cleaned_path = CLEANED_DIR / f"{corpus}.cleaned.jsonl"
    log(f"--- SBERT Extraction: {corpus} ({cleaned_path.name}) ---")

    if not cleaned_path.exists():
        log(f"  [ERROR] Cleaned file not found: {cleaned_path}")
        return

    page_ids: list[int] = []
    texts_to_encode: list[str] = []

    with open(cleaned_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_ids.append(rec["page_id"])
            title = rec.get("title", "")
            lead = rec.get("lead_text", "")
            # Encode "{title}. {lead_text}" for optimal semantic representation
            texts_to_encode.append(f"{title}. {lead}" if lead else title)

    n_docs = len(texts_to_encode)
    log(f"  Encoding {n_docs:,} articles on {device} (batch_size=64)...")

    t0 = time.time()
    embeddings = model.encode(
        texts_to_encode,
        batch_size=64 if device == "cuda" else 16,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )
    elapsed = time.time() - t0
    log(f"  Encoded {n_docs:,} docs in {elapsed:.1f}s ({n_docs / elapsed:.1f} docs/sec) | Shape: {embeddings.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{corpus}_sbert.npy"
    np.save(out_file, embeddings.astype(np.float32))
    log(f"  Saved SBERT embeddings -> {out_file}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Initializing SentenceTransformer 'all-MiniLM-L6-v2' on {device}...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    t_start = time.time()
    for corpus in CORPORA:
        extract_sbert_for_corpus(corpus, model, device)
    log(f"All corpora SBERT extraction complete in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

