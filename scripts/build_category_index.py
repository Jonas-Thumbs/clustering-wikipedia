"""One-time ETL: build a category-graph index from the enwiki dump, used by
grow_category_corpus.py to walk a seed category outward (CS, Literature,
Arizona) and collect real article IDs until a target corpus size is hit.

Reads (from data/raw/):
  - linktarget.sql.gz   -> lt_id -> title, for namespace 14 (Category) rows
  - page.sql.gz         -> page_id -> title/is_redirect, for namespace 14 rows
  - page_props.sql.gz   -> hiddencat flag, Wikipedia's own "this category is
                            maintenance/tracking, not topical content" signal
  - categorylinks.sql.gz -> (member page, cl_type, target category) for
                             every category membership on enwiki

Writes data/processed/category_index.db (SQLite):
  - subcat_edges(parent_title, child_title)   -- "child_title is a
    subcategory of parent_title"; neither category redirects nor hidden
    (maintenance/tracking) categories are ever recorded as a child here,
    so traversal can never walk into one.
  - category_pages(category_title, page_id)   -- "page_id is a direct
    (cl_type='page') member of category_title"; page_id is not yet resolved
    to a title/namespace here, that happens when we actually collect text.

This index is title-keyed and category-only; it doesn't know which
page_ids are redirects or what namespace they're really in (categorylinks
members are overwhelmingly namespace-0 articles, but that gets verified
against the real page table during collection, not here).

No randomness in this script, so no seed is needed.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from sql_dump import (  # noqa: E402
    iter_categorylinks,
    iter_linktarget,
    iter_page,
    iter_page_props_hiddencat,
)

RAW = Path(__file__).parent.parent / "data" / "raw"
DUMP_DATE = "20260901"
DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "category_index.db"

CATEGORY_NS = 14
SUBCAT_FLUSH_EVERY = 100_000
PROGRESS_EVERY = 20_000_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("""CREATE TABLE subcat_edges (
        parent_title TEXT NOT NULL,
        child_title TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE category_pages (
        category_title TEXT NOT NULL,
        page_id INTEGER NOT NULL
    )""")
    con.commit()

    # --- pass 1: page, namespace 14 only -> page_id -> title -------------
    # Done first because it defines which category titles actually exist
    # as pages -- the only ones that can ever be reached by subcat
    # traversal (subcat children are always resolved via this table).
    log("pass 1/4: page.sql.gz (namespace 14 only)")
    page_path = RAW / f"enwiki-{DUMP_DATE}-page.sql.gz"
    page_to_title: dict[int, str] = {}
    redirect_categories: set[str] = set()
    # title -> itself, used both as a membership test for pass 2 and to
    # hand back the SAME string object so linktarget doesn't allocate a
    # second copy of every title it shares with the page table.
    canonical_title: dict[str, str] = {}
    n = 0
    for page_id, ns, title, is_redirect in iter_page(str(page_path)):
        n += 1
        if ns == CATEGORY_NS:
            page_to_title[page_id] = title
            canonical_title[title] = title
            if is_redirect:
                redirect_categories.add(title)
        if n % PROGRESS_EVERY == 0:
            log(f"  page rows seen: {n:,}  (categories kept: {len(page_to_title):,})")
    log(
        f"page done: {n:,} rows scanned, {len(page_to_title):,} category pages "
        f"({len(redirect_categories):,} are category redirects)"
    )

    # --- pass 2: page_props, hiddencat flag only --------------------------
    # Wikipedia's own maintenance/tracking signal (set via __HIDDENCAT__),
    # e.g. "Literary awards by country" -- administrative subdivisions of an
    # existing category rather than new topical content. Excluding these
    # from subcat_edges is what stops traversal from blowing up on them.
    log("pass 2/4: page_props.sql.gz (hiddencat flag)")
    pp_path = RAW / f"enwiki-{DUMP_DATE}-page_props.sql.gz"
    hidden_categories: set[str] = set()
    n = 0
    for page_id in iter_page_props_hiddencat(str(pp_path)):
        n += 1
        title = page_to_title.get(page_id)
        if title is not None:
            hidden_categories.add(title)
        if n % PROGRESS_EVERY == 0:
            log(f"  page_props hiddencat rows seen: {n:,}  (hidden categories kept: {len(hidden_categories):,})")
    log(f"page_props done: {n:,} hiddencat rows scanned, {len(hidden_categories):,} are real categories")

    # --- pass 3: linktarget, namespace 14, filtered to real categories ---
    # linktarget contains many dead/historical/red-link category targets
    # that have no page and so can never be reached by traversal; keeping
    # only titles that exist in page_to_title cuts this dict down to the
    # ~2-3M real categories instead of the ~9M raw targets, and reusing
    # canonical_title's string objects means the two dicts share title
    # storage instead of duplicating it.
    log("pass 3/4: linktarget.sql.gz (namespace 14, filtered to real categories)")
    lt_path = RAW / f"enwiki-{DUMP_DATE}-linktarget.sql.gz"
    lt_to_title: dict[int, str] = {}
    n = 0
    for lt_id, ns, title in iter_linktarget(str(lt_path)):
        n += 1
        if ns == CATEGORY_NS:
            canon = canonical_title.get(title)
            if canon is not None:
                lt_to_title[lt_id] = canon
        if n % PROGRESS_EVERY == 0:
            log(f"  linktarget rows seen: {n:,}  (resolvable categories kept: {len(lt_to_title):,})")
    log(f"linktarget done: {n:,} rows scanned, {len(lt_to_title):,} resolvable category targets")
    del canonical_title

    # --- pass 4: categorylinks -------------------------------------------
    log("pass 4/4: categorylinks.sql.gz")
    cl_path = RAW / f"enwiki-{DUMP_DATE}-categorylinks.sql.gz"
    subcat_buffer: list[tuple[str, str]] = []
    page_buffer: list[tuple[str, int]] = []
    n = 0
    skipped_unresolved = 0
    skipped_unreachable_parent = 0

    def flush_subcat() -> None:
        if subcat_buffer:
            cur.executemany(
                "INSERT INTO subcat_edges (parent_title, child_title) VALUES (?, ?)",
                subcat_buffer,
            )
            subcat_buffer.clear()

    def flush_pages() -> None:
        if page_buffer:
            cur.executemany(
                "INSERT INTO category_pages (category_title, page_id) VALUES (?, ?)",
                page_buffer,
            )
            page_buffer.clear()

    for cl_from, cl_type, cl_target_id in iter_categorylinks(str(cl_path)):
        n += 1
        parent_title = lt_to_title.get(cl_target_id)
        if parent_title is None:
            skipped_unresolved += 1
            continue
        if cl_type == "page":
            # A redirect/hidden category is never recorded as a subcat
            # child (below), so it can never be reached by traversal from
            # any seed -- anything filed only under one here would be
            # permanently orphaned in category_pages. Skip and count it
            # instead, consistent with the child-side exclusion.
            if parent_title in redirect_categories or parent_title in hidden_categories:
                skipped_unreachable_parent += 1
            else:
                page_buffer.append((parent_title, cl_from))
                if len(page_buffer) >= SUBCAT_FLUSH_EVERY:
                    flush_pages()
        elif cl_type == "subcat":
            child_title = page_to_title.get(cl_from)
            if (
                child_title is not None
                and child_title not in redirect_categories
                and child_title not in hidden_categories
            ):
                subcat_buffer.append((parent_title, child_title))
                if len(subcat_buffer) >= SUBCAT_FLUSH_EVERY:
                    flush_subcat()
        # cl_type == "file" is ignored: we don't use media-file membership.
        if n % PROGRESS_EVERY == 0:
            log(f"  categorylinks rows seen: {n:,}")

    flush_subcat()
    flush_pages()
    con.commit()
    log(
        f"categorylinks done: {n:,} rows scanned, {skipped_unresolved:,} skipped "
        f"(target not a category), {skipped_unreachable_parent:,} skipped "
        f"(parent is a redirect/hidden category, unreachable by traversal)"
    )

    log("indexing")
    cur.execute("CREATE INDEX idx_subcat_parent ON subcat_edges(parent_title)")
    cur.execute("CREATE INDEX idx_category_pages_title ON category_pages(category_title)")
    con.commit()

    n_with_pages = cur.execute("SELECT COUNT(DISTINCT category_title) FROM category_pages").fetchone()[0]
    n_with_subcats = cur.execute("SELECT COUNT(DISTINCT parent_title) FROM subcat_edges").fetchone()[0]
    log(f"distinct categories with direct pages: {n_with_pages:,}; with subcategories: {n_with_subcats:,}")

    con.close()
    log(f"done -> {DB_PATH}")


if __name__ == "__main__":
    build()
