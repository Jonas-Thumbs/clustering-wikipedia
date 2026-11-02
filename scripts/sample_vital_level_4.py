"""Downsize Wikipedia's Vital Articles Level 4 list (10,012 articles --
see collect_vital_articles.py) to roughly the same scale as the other
corpora (~1,500 articles: Computer science 1,641, Literature 1,581,
Arizona 1,507) via a single seeded uniform random sample over the whole
list.

Level 4 is already a flat, finished list curated by Wikipedia across 11
topic sections (People, Biology_and_health_sciences, Geography, ...), not
something assembled by this project's own traversal -- so unlike the
category-growth trimming (grow_category_corpus.py), there's no tree
whose branching factor could bias a flat sample. A single uniform draw
over the full list naturally preserves each section's original share (a
section with ~2x the articles keeps ~2x the representation in
expectation), so no explicit stratified-by-section sampling is needed to
get that -- it falls out of uniform sampling for free.

Seeded with its own hardcoded constant, per project convention (each
pipeline stage gets its own fixed seed; no central config needed).

Reads from vital_level_4_full.page_ids.txt (the untouched full list) and
writes the sample to vital_level_4.page_ids.txt -- deliberately never the
other way around, so re-running this script (e.g. to try a different
SEED) always re-samples from the true 10,012-article population, not
from whatever was last written to the output path.
"""
from __future__ import annotations

import random
from pathlib import Path

SEED = 1504  # arbitrary but fixed, this stage's own seed

OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
TARGET = 1500


def main() -> None:
    full_path = OUT_DIR / "vital_level_4_full.page_ids.txt"
    out_path = OUT_DIR / "vital_level_4.page_ids.txt"

    if not full_path.exists():
        # First run: vital_level_4.page_ids.txt is still collect_vital_articles.py's
        # un-sampled output -- preserve it as the full list before overwriting it.
        full_path.write_text(out_path.read_text())

    full = [int(x) for x in full_path.read_text().split()]
    print(f"full Level 4 list: {len(full):,} articles")

    rng = random.Random(SEED)
    sampled = rng.sample(full, TARGET)

    out_path.write_text("\n".join(str(p) for p in sorted(sampled)) + "\n")
    print(f"wrote {len(sampled):,} page_ids (seeded uniform sample) -> {out_path}")


if __name__ == "__main__":
    main()
