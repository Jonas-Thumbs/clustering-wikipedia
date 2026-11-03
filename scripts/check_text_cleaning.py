"""Automated validation checks across all 4 corpora to verify text cleaning
quality, coverage, and readiness for feature extraction and clustering."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# Add scripts/lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))
from text_cleaning import clean_wikitext

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]
DATA_DIR = Path(__file__).parent.parent / "data" / "derived" / "text"

_RE_UNWANTED_MARKUP = [
    ("Unstripped <ref> tags", re.compile(r"<ref\b", re.IGNORECASE)),
    ("Unstripped [[Category:...]] links", re.compile(r"\[\[Category:", re.IGNORECASE)),
    ("Unstripped [[File:...]] links", re.compile(r"\[\[File:", re.IGNORECASE)),
    ("Unstripped {{templates}}", re.compile(r"\{\{[^}]*\}\}")),
    ("HTML tags remaining", re.compile(r"<(?:div|span|table|td|tr|b|i)\b[^>]*>", re.IGNORECASE)),
]


def run_checks(sample_size: int = 100) -> bool:
    print(f"=== Running Automated Checks Across {len(CORPORA)} Corpora ===", flush=True)
    overall_passed = True

    for corpus in CORPORA:
        corpus_path = DATA_DIR / f"{corpus}.articles.jsonl"
        print(f"\nEvaluating Corpus: {corpus} ({corpus_path.name})", flush=True)

        if not corpus_path.exists():
            print(f"  [FAIL] File missing: {corpus_path}", flush=True)
            overall_passed = False
            continue

        records = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if sample_size and idx >= sample_size:
                    break
                records.append(json.loads(line))

        n_total = len(records)
        print(f"  Loaded {n_total} sample articles for deep verification.", flush=True)

        empty_leads = 0
        empty_fulls = 0
        short_articles = 0
        markup_violations = {name: 0 for name, _ in _RE_UNWANTED_MARKUP}
        lead_lengths = []
        full_lengths = []

        t0 = time.time()
        for rec in records:
            wikitext = rec.get("wikitext", "")
            
            cleaned = clean_wikitext(wikitext)
            lead = cleaned["lead_text"]
            full = cleaned["full_text"]

            lead_len = len(lead)
            full_len = len(full)

            lead_lengths.append(lead_len)
            full_lengths.append(full_len)

            if lead_len == 0:
                empty_leads += 1
            if full_len == 0:
                empty_fulls += 1
            elif full_len < 150:
                short_articles += 1

            for name, pattern in _RE_UNWANTED_MARKUP:
                if pattern.search(full):
                    markup_violations[name] += 1

        elapsed = time.time() - t0
        avg_lead = sum(lead_lengths) / max(1, n_total)
        avg_full = sum(full_lengths) / max(1, n_total)

        print(f"  Cleaning throughput: {n_total / elapsed:.1f} articles/sec", flush=True)
        print(f"  Average length: Lead = {avg_lead:.0f} chars | Full = {avg_full:.0f} chars", flush=True)

        failed_checks = []
        if empty_fulls > 0:
            failed_checks.append(f"{empty_fulls} articles had empty full text")
        if empty_leads > (n_total * 0.02):
            failed_checks.append(f"{empty_leads} articles had empty lead sections")
        for name, count in markup_violations.items():
            if count > 0:
                failed_checks.append(f"{name}: {count} occurrences found in cleaned text")

        if failed_checks:
            print(f"  [FAIL] {corpus} failed checks:", flush=True)
            for fc in failed_checks:
                print(f"    - {fc}", flush=True)
            overall_passed = False
        else:
            print(f"  [PASS] All clean text quality checks passed for {corpus}!", flush=True)

    print("\n" + "=" * 55, flush=True)
    if overall_passed:
        print("ALL CORPORA PASSED AUTOMATED QUALITY CHECKS.", flush=True)
    else:
        print("SOME CORPORA FAILED QUALITY CHECKS.", flush=True)
    print("=" * 55, flush=True)
    return overall_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
