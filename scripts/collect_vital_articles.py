"""Collect Wikipedia's Vital Articles Level 3 and Level 4 lists via the
live MediaWiki API, rather than reconstructing them from a multi-hour
pagelinks.sql.gz scan of the local dump.

Why the API instead of the local dump: the pages-articles-multistream
dump excludes the Wikipedia namespace entirely (confirmed empty when
searching its index for Vital_articles), so the list pages' content/links
aren't available locally without downloading a much larger "all
namespaces" dump just for ~30 utility pages. The MediaWiki API's
generator=links resolves a list page's outgoing namespace-0 links
directly to (pageid, title) pairs in a handful of paginated calls --
seconds of wall time instead of a ~2.5-3 hour full scan of a 7.1GB table,
and it resolves page_id for us directly, so no further local lookups are
needed.

This does mean these lists reflect Wikipedia's CURRENT state rather than
the Sept 2026 dump snapshot exactly. Vital Articles lists change slowly
(periodic curation, not rapid churn), so drift over the ~2-month gap
should be minor; page_ids are stable identifiers assigned at creation, so
they should still match the local dump for virtually every article.

Source pages: by inspecting the real page table in this dump, the title
structure has changed from the "one page per topic, every level" scheme
originally assumed for this project:
  - Level 3 is now a single consolidated page: Vital_articles/Level_3
    (the old per-topic subpages are now just redirects to it)
  - Level 4 still uses 11 real per-topic subpages; the same-depth decoys
    (Removed -- a tracking list of articles removed from vital status;
    Article_alerts / Article_alerts/Archive_1 -- WikiProject
    notification pages; and the tiny root Vital_articles/Level_4 nav
    page) are deliberately excluded below.
`redirects=1` makes each lookup resilient to the source page itself
having been renamed/redirected since the dump was taken.

No randomness in this script, so no seed is needed.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "CS472-WikiClusteringCorpusBuilder/1.0 (educational coursework project)"

OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

LEVEL_3_PAGES = ["Wikipedia:Vital articles/Level_3"]
LEVEL_4_PAGES = [
    f"Wikipedia:Vital articles/Level_4/{topic}"
    for topic in [
        "Arts",
        "Biology_and_health_sciences",
        "Everyday_life",
        "Geography",
        "History",
        "Mathematics",
        "People",
        "Philosophy_and_religion",
        "Physical_sciences",
        "Society_and_social_sciences",
        "Technology",
    ]
]


def _api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 5.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"  rate limited (429), waiting {wait:.0f}s before retry...")
                time.sleep(wait)
                delay *= 2
                continue
            raise
    raise AssertionError("unreachable")  # loop always returns or raises above


def fetch_linked_page_ids(title: str) -> set[int]:
    """All namespace-0 pages linked FROM `title`, via generator=links."""
    page_ids: set[int] = set()
    params = {
        "action": "query",
        "titles": title,
        "generator": "links",
        "gplnamespace": "0",
        "gpllimit": "500",
        "redirects": "1",
        "prop": "info",
        "format": "json",
    }
    gplcontinue = None
    while True:
        q = dict(params)
        if gplcontinue:
            q["gplcontinue"] = gplcontinue
        url = API + "?" + urllib.parse.urlencode(q)
        data = _api_get(url)
        if "query" not in data:
            raise RuntimeError(f"unexpected API response for {title!r}: {data}")
        for page_id, page in data["query"].get("pages", {}).items():
            pid = int(page_id)
            if pid > 0:  # negative/missing ids mark nonexistent (red-link) pages
                page_ids.add(pid)
        cont = data.get("continue", {})
        if "gplcontinue" in cont:
            gplcontinue = cont["gplcontinue"]
            time.sleep(1.5)
        else:
            break
    return page_ids


def collect(titles: list[str], label: str) -> set[int]:
    all_ids: set[int] = set()
    for title in titles:
        ids = fetch_linked_page_ids(title)
        print(f"  {title}: {len(ids):,} linked articles")
        all_ids |= ids
        time.sleep(1.5)
    print(f"{label}: {len(all_ids):,} distinct articles across {len(titles)} source page(s)")
    return all_ids


def write(page_ids: set[int], name: str) -> None:
    out_path = OUT_DIR / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in sorted(page_ids)) + "\n")
    print(f"wrote {len(page_ids):,} page_ids -> {out_path}")


def main() -> None:
    print("=== Level 3 ===")
    level3 = collect(LEVEL_3_PAGES, "Level 3")
    write(level3, "vital_level_3.page_ids.txt")

    print("=== Level 4 ===")
    level4 = collect(LEVEL_4_PAGES, "Level 4")
    write(level4, "vital_level_4.page_ids.txt")


if __name__ == "__main__":
    main()
