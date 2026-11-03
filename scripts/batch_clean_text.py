"""Batch clean raw wikitext articles across all corpora and save the cleaned
results into persistent, ready-to-use JSONL files.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from text_cleaning import clean_wikitext

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
IN_DIR = Path(__file__).parent.parent / "data" / "derived" / "text"
OUT_DIR = Path(__file__).parent.parent / "data" / "derived" / "cleaned"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_single_record(line: str) -> str:
    rec = json.loads(line)
    title = rec.get("title", "")
    wikitext = rec.get("wikitext", "")

    cleaned = clean_wikitext(wikitext)
    lead = cleaned["lead_text"]
    full = cleaned["full_text"]

    out_record = {
        "page_id": rec["page_id"],
        "title": title,
        "lead_text": lead,
        "full_text": full,
        "lead_chars": len(lead),
        "full_chars": len(full),
    }
    return json.dumps(out_record, ensure_ascii=False)


def clean_corpus(corpus: str) -> None:
    in_path = IN_DIR / f"{corpus}.articles.jsonl"
    out_path = OUT_DIR / f"{corpus}.cleaned.jsonl"

    if not in_path.exists():
        log(f"WARNING: Source file not found: {in_path}")
        return

    # Check if already cleaned and complete
    if out_path.exists():
        with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "r", encoding="utf-8") as f_out:
            n_in = sum(1 for _ in f_in)
            n_out = sum(1 for _ in f_out)
        if n_in == n_out and n_out > 0:
            log(f"Corpus {corpus} already cleaned ({n_out:,} articles) -> skipping.")
            return

    log(f"--- Cleaning {corpus} ({in_path.name}) with multi-core parallelism ---")
    t0 = time.time()
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(clean_single_record, lines, chunksize=15))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")

    elapsed = time.time() - t0
    log(f"Finished {corpus}: {len(results):,} articles cleaned in {elapsed:.1f}s -> {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Starting batch text cleaning across {len(CORPORA)} corpora...")
    total_t0 = time.time()
    for corpus in CORPORA:
        clean_corpus(corpus)
    log(f"=== ALL CORPORA CLEANED in {time.time() - total_t0:.1f}s ===")


if __name__ == "__main__":
    main()
