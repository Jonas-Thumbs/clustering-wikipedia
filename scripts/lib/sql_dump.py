"""Streaming reader for MediaWiki SQL dump files (enwiki-*.sql.gz).

These dumps are MariaDB `mysqldump` output: boilerplate/DDL, then one or
more `INSERT INTO `table` VALUES` blocks where each data row sits on its
own line (`(field,field,...),` or `(field,field,...);` on the last row of
a block). We stream line by line so we never hold more than one row's
text in memory, and extract fields with a pre-compiled regex per table
rather than a general-purpose tuple parser, since we only ever need a
handful of columns out of each row.
"""
from __future__ import annotations

import gzip
import re
from collections.abc import Iterator

_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "0": "\0", "Z": "\x1a",
    "\\": "\\", "'": "'", '"': '"',
}


def unescape(raw: str) -> str:
    """Undo mysqldump's backslash-escaping of a single-quoted string body
    (raw does not include the surrounding quotes)."""
    if "\\" not in raw:
        return raw
    out = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == "\\" and i + 1 < n:
            out.append(_ESCAPES.get(raw[i + 1], raw[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def iter_insert_lines(path: str, table: str) -> Iterator[str]:
    """Yield each raw data-row line (e.g. "(1,2,'x'),") belonging to
    `INSERT INTO `table` VALUES` blocks in the dump at `path`."""
    prefix = f"INSERT INTO `{table}` VALUES"
    in_block = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            line = line.rstrip("\n")
            if not in_block:
                if line.startswith(prefix):
                    in_block = True
                    rest = line[len(prefix):].strip()
                    if rest:
                        yield rest
                        if rest.endswith(";"):
                            in_block = False
                continue
            stripped = line.strip()
            if not stripped:
                continue
            yield stripped
            if stripped.endswith(";"):
                in_block = False


# --- table-specific row extractors -----------------------------------
# Each regex matches exactly one row tuple "(...)" from the start of a
# line; unneeded fields are consumed by non-capturing groups.

_RE_LINKTARGET = re.compile(
    r"^\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)'\)"
)

_RE_PAGE = re.compile(
    r"^\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',(\d+),\d+,[\d.eE+-]+,"
    r"'[^']*',(?:'[^']*'|NULL),\d+,\d+,(?:'[^']*'|NULL),(?:'[^']*'|NULL)\)"
)

_RE_CATEGORYLINKS = re.compile(
    r"^\((\d+),'(?:[^'\\]|\\.)*','[^']*','(?:[^'\\]|\\.)*','(\w+)',\d+,(\d+)\)"
)

_RE_PAGE_PROPS_HIDDENCAT = re.compile(r"^\((\d+),'hiddencat',")

_RE_PAGELINKS = re.compile(r"^\((\d+),(\d+),(\d+)\)")

_RE_REDIRECT = re.compile(
    r"^\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',(?:'(?:[^'\\]|\\.)*'|NULL),"
    r"(?:'(?:[^'\\]|\\.)*'|NULL)\)"
)

_RE_GEO_TAGS = re.compile(
    r"^\(\d+,(\d+),'([^']*)',(\d),"
    r"(?:NULL|-?\d+(?:\.\d+)?),(?:NULL|-?\d+(?:\.\d+)?),"
    r"(?:NULL|\d+),"
    r"(?:NULL|'(?:[^'\\]|\\.)*'),"
    r"(?:NULL|'(?:[^'\\]|\\.)*'),"
    r"(?:NULL|'([^']*)'),"
    r"(?:NULL|'([^']*)'),"
    r"(?:NULL|-?\d+),(?:NULL|-?\d+)\)"
)


def iter_linktarget(path: str) -> Iterator[tuple[int, int, str]]:
    """Yield (lt_id, lt_namespace, title) for every row of `linktarget`."""
    for line in iter_insert_lines(path, "linktarget"):
        m = _RE_LINKTARGET.match(line)
        if m:
            yield int(m.group(1)), int(m.group(2)), unescape(m.group(3))


def iter_page(path: str) -> Iterator[tuple[int, int, str, bool]]:
    """Yield (page_id, page_namespace, title, is_redirect) for every row
    of `page`."""
    for line in iter_insert_lines(path, "page"):
        m = _RE_PAGE.match(line)
        if m:
            yield (
                int(m.group(1)), int(m.group(2)), unescape(m.group(3)),
                m.group(4) == "1",
            )


def iter_categorylinks(path: str) -> Iterator[tuple[int, str, int]]:
    """Yield (cl_from, cl_type, cl_target_id) for every row of
    `categorylinks`. cl_target_id must be resolved against the
    `linktarget` table (namespace 14 subset) to get the category title."""
    for line in iter_insert_lines(path, "categorylinks"):
        m = _RE_CATEGORYLINKS.match(line)
        if m:
            yield int(m.group(1)), m.group(2), int(m.group(3))


def iter_geo_tags(path: str) -> Iterator[tuple[int, str, bool, str | None, str | None]]:
    """Yield (gt_page_id, gt_globe, gt_primary, gt_country, gt_region) for
    every row of `geo_tags`. gt_country/gt_region are None when untagged
    (common for globe != 'earth' or when no state/country breakdown was
    recorded)."""
    for line in iter_insert_lines(path, "geo_tags"):
        m = _RE_GEO_TAGS.match(line)
        if m:
            page_id, globe, primary, country, region = m.groups()
            yield int(page_id), globe, primary == "1", country, region


def iter_pagelinks(path: str) -> Iterator[tuple[int, int, int]]:
    """Yield (pl_from, pl_from_namespace, pl_target_id) for every row of
    `pagelinks`. pl_target_id must be resolved against `linktarget` to get
    the linked title; pl_from is a page_id in whatever namespace
    pl_from_namespace says (only 0 is a Wikipedia article)."""
    for line in iter_insert_lines(path, "pagelinks"):
        m = _RE_PAGELINKS.match(line)
        if m:
            yield int(m.group(1)), int(m.group(2)), int(m.group(3))


def iter_redirect(path: str) -> Iterator[tuple[int, int, str]]:
    """Yield (rd_from, rd_namespace, rd_title) for every row of
    `redirect` -- rd_from is the redirect PAGE's own page_id, rd_title is
    the literal title string it points at (not yet resolved to a
    page_id)."""
    for line in iter_insert_lines(path, "redirect"):
        m = _RE_REDIRECT.match(line)
        if m:
            yield int(m.group(1)), int(m.group(2)), unescape(m.group(3))


def iter_page_props_hiddencat(path: str) -> Iterator[int]:
    """Yield pp_page for every `page_props` row with pp_propname='hiddencat'
    -- Wikipedia's own signal (set via the __HIDDENCAT__ magic word) that a
    category page is maintenance/tracking, not topical content. A cheap
    substring check skips the regex for the many unrelated props (e.g.
    wikibase_item, defaultsort) before it's even tried."""
    needle = "'hiddencat',"
    for line in iter_insert_lines(path, "page_props"):
        if needle not in line:
            continue
        m = _RE_PAGE_PROPS_HIDDENCAT.match(line)
        if m:
            yield int(m.group(1))
