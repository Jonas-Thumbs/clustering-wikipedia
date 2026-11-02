"""Filter the 5 corpus page_id lists down to real, standalone articles --
removing any member that is a non-article namespace (e.g. a Portal or
Template directly filed under a category, which categorylinks happily
lets through as a cl_type='page' membership same as any real article) or
a redirect (a one-line "#REDIRECT [[...]]" stub with no real content to
cluster on).

Why this is a separate pipeline stage: grow_category_corpus.py's BFS
collects cl_type='page' category memberships as-is from
category_index.db, which -- per its own docstring -- deliberately does
NOT resolve member page_ids to a namespace/redirect status; that's real
page-table information, out of scope for a category-graph index. This
script is the "verified against the real page table" step that index was
always designed to be followed by, as an explicit, rerunnable stage
rather than a one-off manual fixup.

Run this right after grow_category_corpus.py and before
extract_article_text.py / build_link_graph.py, so neither of those ever
sees a non-article or redirect page_id in the first place. Running it
again later on already-filtered corpora is a safe no-op: nothing left to
remove, 0 reported for every corpus.

Reads data/processed/*.page_ids.txt (the 5 corpora) and
data/raw/enwiki-{DATE}-page.sql.gz (one filtered scan).
Overwrites data/processed/*.page_ids.txt in place with the filtered list.

No randomness, no seed needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from corpora import CORPUS_FILES  # noqa: E402
from sql_dump import iter_page  # noqa: E402

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DUMP_DATE = "20260901"
PROGRESS_EVERY = 20_000_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def filter_corpora() -> None:
    corpora: dict[str, set[int]] = {}
    all_ids: set[int] = set()
    for name, filename in CORPUS_FILES.items():
        ids = {int(x) for x in (PROCESSED / filename).read_text().split()}
        corpora[name] = ids
        all_ids |= ids
    log(f"loaded {len(all_ids):,} distinct candidate page_ids across {len(CORPUS_FILES)} corpora")

    log("scanning page.sql.gz for namespace/redirect status of candidates")
    page_path = RAW / f"enwiki-{DUMP_DATE}-page.sql.gz"
    bad: dict[int, str] = {}  # page_id -> reason, for logging
    seen: set[int] = set()
    n = 0
    for page_id, ns, title, is_redirect in iter_page(str(page_path)):
        n += 1
        if page_id in all_ids:
            seen.add(page_id)
            if ns != 0:
                bad[page_id] = f"non-article namespace ns={ns} ({title!r})"
            elif is_redirect:
                bad[page_id] = f"redirect ({title!r})"
        if n % PROGRESS_EVERY == 0:
            log(f"  page rows seen: {n:,}")
    log(
        f"page scan done: {n:,} rows scanned, {len(seen):,}/{len(all_ids):,} "
        f"candidates located, {len(bad):,} excluded"
    )

    missing = all_ids - seen
    if missing:
        log(
            f"  WARNING: {len(missing):,} candidate page_ids were not found in "
            f"page.sql.gz at all: {sorted(missing)[:10]}"
        )

    for name, filename in CORPUS_FILES.items():
        ids = corpora[name]
        kept = sorted(i for i in ids if i not in bad)
        removed = sorted(i for i in ids if i in bad)
        (PROCESSED / filename).write_text("\n".join(str(i) for i in kept) + "\n")
        log(f"{name}: {len(ids):,} -> {len(kept):,} ({len(removed):,} removed)")
        for pid in removed:
            log(f"    removed {pid}: {bad[pid]}")


if __name__ == "__main__":
    filter_corpora()
