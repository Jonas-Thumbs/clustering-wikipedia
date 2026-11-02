"""Extract full article text (raw wikitext) for exactly the page_ids the
5 corpora need, via random access into the local pages-articles-multistream
dump -- using its companion index file to seek directly to the handful of
~100-page blocks that contain our target page_ids, instead of decompressing
the full ~27GB dump or hitting the live MediaWiki API ~7,000 times (the
approach collect_vital_articles.py uses, which only makes sense for the
~30 list pages it reads, not thousands of full articles).

Reads:
  - data/processed/*.page_ids.txt (this repo)       -> union of target ids
    (excludes vital_level_4_full.page_ids.txt, which is the pre-downsample
    backup, not one of the 5 corpora)
  - data/raw/enwiki-{DATE}-pages-articles-multistream-index.txt.bz2
      "{byte_offset}:{page_id}:{title}" per page; many consecutive lines
      share the same offset (one independent bz2 stream = ~100 pages)
  - data/raw/enwiki-{DATE}-pages-articles-multistream.xml.bz2
      the dump itself, read via seek() + a fresh bz2.BZ2Decompressor per
      target block -- each block is a self-terminating bz2 member, so no
      need to know its length or end offset in advance

Writes data/derived/text/{corpus}.articles.jsonl: one JSON object per
page_id in that corpus, each `{"page_id", "title", "wikitext",
"is_redirect"}` -- one file per corpus, matching build_link_graph.py's
per-corpus edge files, so each corpus's data is self-contained and a
teammate working on just one corpus only needs its one file. An article
that's a member of two corpora (e.g. a Vital Article that's also under
Computer_science) is decompressed/parsed only once internally but is
written out to both corpora's files -- a deliberate, small (tens of MB)
duplication in exchange for not making every consumer join against a
separate membership table.
`is_redirect` just reflects whether the dump's <redirect> tag is present on
that revision -- it is NOT filtered out here; a target page_id landing on a
redirect (rather than the real article) is a fact about how the corpus was
built, worth surfacing to whoever runs preprocessing next, not something
this extraction step should silently drop or resolve. In practice
filter_corpora.py already removes all redirects before this script ever
runs, so expect this field to always read false -- it stays as a safety
net in case that filtering policy ever changes.

No randomness, no seed needed.
"""
from __future__ import annotations

import bz2
import html
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from corpora import CORPUS_FILES  # noqa: E402

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DUMP_DATE = "20260901"

INDEX_PATH = RAW / f"enwiki-{DUMP_DATE}-pages-articles-multistream-index.txt.bz2"
DUMP_PATH = RAW / f"enwiki-{DUMP_DATE}-pages-articles-multistream.xml.bz2"
OUT_DIR = Path(__file__).parent.parent / "data" / "derived" / "text"

_RE_INDEX_LINE = re.compile(r"^(\d+):(\d+):")
_RE_PAGE_BLOCK = re.compile(r"<page>.*?</page>", re.DOTALL)
_RE_TITLE_NS_ID = re.compile(
    r"<title>(.*?)</title>\s*<ns>(-?\d+)</ns>\s*<id>(\d+)</id>", re.DOTALL
)
_RE_TEXT = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
_RE_REDIRECT_TAG = re.compile(r"<redirect\b")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpora() -> dict[str, set[int]]:
    corpora: dict[str, set[int]] = {}
    for name, filename in CORPUS_FILES.items():
        ids = {int(x) for x in (PROCESSED / filename).read_text().split()}
        log(f"  {name}: {len(ids):,} page_ids")
        corpora[name] = ids
    return corpora


def find_target_offsets(target_ids: set[int]) -> dict[int, set[int]]:
    """Scan the index once; return {byte_offset: {page_ids in that block
    that we actually want}}, restricted to offsets containing >=1 target."""
    offsets: dict[int, set[int]] = {}
    n = 0
    with bz2.open(INDEX_PATH, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            m = _RE_INDEX_LINE.match(line)
            if not m:
                continue
            page_id = int(m.group(2))
            if page_id in target_ids:
                offset = int(m.group(1))
                offsets.setdefault(offset, set()).add(page_id)
            if n % 5_000_000 == 0:
                log(f"  index lines scanned: {n:,}  (target blocks found so far: {len(offsets):,})")
    log(f"index done: {n:,} lines scanned, {len(offsets):,} distinct blocks to decompress")
    return offsets


def decompress_block(offset: int) -> bytes:
    with open(DUMP_PATH, "rb") as f:
        f.seek(offset)
        dec = bz2.BZ2Decompressor()
        chunks = []
        while not dec.eof:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            chunks.append(dec.decompress(chunk))
    return b"".join(chunks)


def extract() -> None:
    log("loading target page_ids from the 5 corpora")
    corpora = load_corpora()
    target_ids: set[int] = set()
    for ids in corpora.values():
        target_ids |= ids
    log(f"total distinct target page_ids across all corpora: {len(target_ids):,}")

    log("pass 1/2: multistream index (locating blocks)")
    offsets = find_target_offsets(target_ids)

    log("pass 2/2: decompressing target blocks and extracting article text")
    found: dict[int, dict] = {}
    n_redirects = 0
    for i, (offset, wanted_ids) in enumerate(sorted(offsets.items()), start=1):
        blob = decompress_block(offset).decode("utf-8", errors="replace")
        for block in _RE_PAGE_BLOCK.findall(blob):
            m = _RE_TITLE_NS_ID.search(block)
            if not m:
                continue
            page_id = int(m.group(3))
            if page_id not in wanted_ids:
                continue
            title = html.unescape(m.group(1))
            ns = int(m.group(2))
            text_m = _RE_TEXT.search(block)
            wikitext = html.unescape(text_m.group(1)) if text_m else ""
            is_redirect = _RE_REDIRECT_TAG.search(block) is not None
            if is_redirect:
                n_redirects += 1
            if ns != 0:
                log(f"  WARNING: target page_id {page_id} ({title!r}) has ns={ns}, expected 0")
            found[page_id] = {
                "page_id": page_id,
                "title": title,
                "wikitext": wikitext,
                "is_redirect": is_redirect,
            }
        if i % 500 == 0:
            log(f"  blocks processed: {i:,}/{len(offsets):,}  (articles found so far: {len(found):,})")

    missing = target_ids - found.keys()
    log(
        f"extraction done: {len(found):,}/{len(target_ids):,} target articles found "
        f"({n_redirects:,} are marked <redirect> in the dump), {len(missing):,} missing"
    )
    if missing:
        sample = list(missing)[:10]
        log(f"  missing page_ids (sample up to 10): {sample}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in corpora.items():
        out_path = OUT_DIR / f"{name}.articles.jsonl"
        n_written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for page_id in sorted(ids):
                record = found.get(page_id)
                if record is None:
                    continue  # already logged in the "missing" summary above
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
        log(f"{name}: wrote {n_written:,}/{len(ids):,} articles -> {out_path}")


if __name__ == "__main__":
    extract()
