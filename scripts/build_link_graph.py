"""Build the induced article-to-article link subgraph for each of the 5
corpora: for corpus C, every (a, b) edge where a and b are BOTH page_ids
in C, resolved from the real enwiki link structure (pagelinks + linktarget
+ page), not re-derived from wikitext -- this is what the project's
link-community side of the comparison (vs. text clustering) runs on.

Everything is filtered down to our ~7,000-article universe as early as
possible in each pass, the same approach build_category_index.py uses for
category titles, so none of the 5 full-dump scans needs to hold more than
a small fraction of the table's rows in memory.

One-level redirect resolution: pagelinks records the literal link target as
written in wikitext, which may be a redirect's title rather than the
final article's (e.g. a link written as [[AccessibleComputing]] literally
targets that title, which redirects to "Computer accessibility"). Double
redirects are disallowed/cleaned up by Wikipedia bots, so resolving one
hop (via the `redirect` table) is standard practice and sufficient --
without it, every such link would silently fail to resolve and the
induced subgraph would undercount edges in a way that's invisible later.

Passes (all reading data/raw/enwiki-{DATE}-*.sql.gz):
  1. page.sql.gz (ns=0, filtered to our target page_ids)
       -> target_pageid_to_title, target_title_to_pageid
  2. redirect.sql.gz (filtered to rows whose destination title is one of
     ours) -> {rd_from: destination_title}, i.e. which OTHER page_ids are
     redirects pointing at an article we care about
  3. page.sql.gz again (filtered to those rd_from ids, to get THEIR
     titles -- redirect.sql.gz only gives ids, but pagelinks/linktarget
     address pages by title)
       -> redirect_title_to_target_pageid
  4. linktarget.sql.gz (ns=0, filtered to titles resolvable by either of
     the above two maps) -> lt_id -> resolved target page_id
  5. pagelinks.sql.gz (the big one, ~7.1GB) -> for each row with
     pl_from_namespace=0 and pl_from in our target set, resolve
     pl_target_id through pass 4's map; if it resolves, record the edge

Writes data/derived/links/{corpus}.edges.tsv (tab-separated
"from_page_id\\tto_page_id" with a header row, directed, one file per
corpus) -- an edge is written to a corpus's file only if BOTH endpoints
are members of that specific corpus (an edge between two articles that
are each in a different single corpus, with no corpus containing both, is
correctly dropped everywhere).

No randomness, no seed needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from corpora import CORPUS_FILES  # noqa: E402
from sql_dump import iter_linktarget, iter_page, iter_pagelinks, iter_redirect  # noqa: E402

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DUMP_DATE = "20260901"
OUT_DIR = Path(__file__).parent.parent / "data" / "derived" / "links"

PROGRESS_EVERY = 20_000_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpora() -> dict[str, set[int]]:
    corpora = {}
    for name, filename in CORPUS_FILES.items():
        ids = {int(x) for x in (PROCESSED / filename).read_text().split()}
        corpora[name] = ids
        log(f"  {name}: {len(ids):,} page_ids")
    return corpora


def build() -> None:
    log("loading the 5 corpora")
    corpora = load_corpora()
    target_ids: set[int] = set()
    for ids in corpora.values():
        target_ids |= ids
    log(f"union across all corpora: {len(target_ids):,} distinct page_ids")

    # --- pass 1: page.sql.gz, our target articles' own titles -----------
    log("pass 1/5: page.sql.gz (our target articles)")
    page_path = RAW / f"enwiki-{DUMP_DATE}-page.sql.gz"
    target_pageid_to_title: dict[int, str] = {}
    target_title_to_pageid: dict[str, int] = {}
    n = 0
    n_redirect_targets = 0
    for page_id, ns, title, is_redirect in iter_page(str(page_path)):
        n += 1
        if ns == 0 and page_id in target_ids:
            target_pageid_to_title[page_id] = title
            target_title_to_pageid[title] = page_id
            if is_redirect:
                n_redirect_targets += 1
        if n % PROGRESS_EVERY == 0:
            log(f"  page rows seen: {n:,}")
    log(
        f"pass 1 done: {len(target_pageid_to_title):,}/{len(target_ids):,} target "
        f"page_ids resolved to a title ({n_redirect_targets:,} are themselves "
        f"<redirect> pages)"
    )

    # --- pass 2: redirect.sql.gz, filtered to our titles as destination -
    log("pass 2/5: redirect.sql.gz (redirects pointing AT our target articles)")
    rd_path = RAW / f"enwiki-{DUMP_DATE}-redirect.sql.gz"
    redirect_source_to_dest_title: dict[int, str] = {}
    n = 0
    for rd_from, rd_namespace, rd_title in iter_redirect(str(rd_path)):
        n += 1
        if rd_namespace == 0 and rd_title in target_title_to_pageid:
            redirect_source_to_dest_title[rd_from] = rd_title
        if n % PROGRESS_EVERY == 0:
            log(f"  redirect rows seen: {n:,}")
    log(
        f"pass 2 done: {n:,} rows scanned, {len(redirect_source_to_dest_title):,} "
        f"redirects point at one of our target articles"
    )

    # --- pass 3: page.sql.gz again, titles of those redirect sources ----
    log("pass 3/5: page.sql.gz again (titles of the redirect pages found in pass 2)")
    redirect_source_ids = set(redirect_source_to_dest_title)
    redirect_title_to_target_pageid: dict[str, int] = {}
    n = 0
    for page_id, ns, title, _is_redirect in iter_page(str(page_path)):
        n += 1
        if ns == 0 and page_id in redirect_source_ids:
            dest_title = redirect_source_to_dest_title[page_id]
            redirect_title_to_target_pageid[title] = target_title_to_pageid[dest_title]
        if n % PROGRESS_EVERY == 0:
            log(f"  page rows seen: {n:,}")
    log(
        f"pass 3 done: {len(redirect_title_to_target_pageid):,}/"
        f"{len(redirect_source_ids):,} redirect-source titles resolved"
    )

    resolvable_title_to_pageid: dict[str, int] = dict(target_title_to_pageid)
    resolvable_title_to_pageid.update(redirect_title_to_target_pageid)
    log(
        f"combined resolvable titles (direct + one-hop redirect): "
        f"{len(resolvable_title_to_pageid):,}"
    )

    # --- pass 4: linktarget.sql.gz, filtered to resolvable titles -------
    log("pass 4/5: linktarget.sql.gz (namespace 0, filtered to resolvable titles)")
    lt_path = RAW / f"enwiki-{DUMP_DATE}-linktarget.sql.gz"
    lt_to_target_pageid: dict[int, int] = {}
    n = 0
    for lt_id, ns, title in iter_linktarget(str(lt_path)):
        n += 1
        if ns == 0:
            resolved = resolvable_title_to_pageid.get(title)
            if resolved is not None:
                lt_to_target_pageid[lt_id] = resolved
        if n % PROGRESS_EVERY == 0:
            log(f"  linktarget rows seen: {n:,}")
    log(f"pass 4 done: {n:,} rows scanned, {len(lt_to_target_pageid):,} resolvable link targets")

    # --- pass 5: pagelinks.sql.gz (the big one) --------------------------
    log("pass 5/5: pagelinks.sql.gz")
    pl_path = RAW / f"enwiki-{DUMP_DATE}-pagelinks.sql.gz"
    edges: set[tuple[int, int]] = set()
    n = 0
    for pl_from, pl_from_namespace, pl_target_id in iter_pagelinks(str(pl_path)):
        n += 1
        if pl_from_namespace == 0 and pl_from in target_ids:
            resolved = lt_to_target_pageid.get(pl_target_id)
            if resolved is not None:
                edges.add((pl_from, resolved))
        if n % PROGRESS_EVERY == 0:
            log(f"  pagelinks rows seen: {n:,}  (edges found so far: {len(edges):,})")
    log(f"pass 5 done: {n:,} rows scanned, {len(edges):,} distinct edges inside our universe")

    # --- write per-corpus induced subgraphs ------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in corpora.items():
        corpus_edges = [(a, b) for a, b in edges if a in ids and b in ids]
        corpus_edges.sort()
        out_path = OUT_DIR / f"{name}.edges.tsv"
        with out_path.open("w", encoding="utf-8") as f:
            f.write("from_page_id\tto_page_id\n")
            for a, b in corpus_edges:
                f.write(f"{a}\t{b}\n")
        n_nodes_with_edges = len({x for pair in corpus_edges for x in pair})
        log(
            f"{name}: {len(corpus_edges):,} edges, {n_nodes_with_edges:,}/{len(ids):,} "
            f"articles have >=1 in-corpus link -> {out_path}"
        )


if __name__ == "__main__":
    build()
