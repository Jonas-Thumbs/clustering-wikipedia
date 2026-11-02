"""Collect Arizona's geotagged supplement: pages whose primary coordinate
(gt_primary=1) is tagged gt_country='US', gt_region='AZ' in geo_tags.sql.gz,
to be combined with the category-based Arizona corpus already built by
grow_category_corpus.py ("Arizona" category tree).

Why combine the two: the category tree (Category:Arizona and its subcats)
captures institutions/culture/people topically linked to Arizona, but can
miss geographic features/landmarks that are geotagged in Arizona without
ever being placed in that category tree. The geotag signal catches those;
the category signal catches non-geographic Arizona-related topics the
geotag has no way to see. Neither alone is a complete "about Arizona"
corpus, which is why the two are combined (unioned) here rather than used
in isolation.

Only gt_primary=1 rows are used -- a page's own primary location, not an
incidental location mentioned somewhere in the article body (a page can
have several non-primary geo_tags for landmarks/places it merely
discusses).

gt_region directly encodes the US state (e.g. 'AZ'), so this is an exact
membership filter, not a lat/lon bounding box -- no risk of border-area
false positives/negatives the way a box would have.

No randomness in this script, so no seed is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from sql_dump import iter_geo_tags  # noqa: E402

RAW = Path(__file__).parent.parent / "data" / "raw"
DUMP_DATE = "20260901"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def collect_geotagged() -> set[int]:
    path = RAW / f"enwiki-{DUMP_DATE}-geo_tags.sql.gz"
    page_ids: set[int] = set()
    for page_id, globe, primary, country, region in iter_geo_tags(str(path)):
        if globe == "earth" and primary and country == "US" and region == "AZ":
            page_ids.add(page_id)
    return page_ids


def main() -> None:
    geotagged = collect_geotagged()
    print(f"geotagged (US-AZ, primary location only): {len(geotagged):,} articles")

    geotagged_path = OUT_DIR / "arizona_geotagged.page_ids.txt"
    geotagged_path.parent.mkdir(parents=True, exist_ok=True)
    geotagged_path.write_text("\n".join(str(p) for p in sorted(geotagged)) + "\n")
    print(f"wrote {len(geotagged):,} page_ids -> {geotagged_path}")

    category_path = OUT_DIR / "arizona.page_ids.txt"
    if category_path.exists():
        category_ids = {int(x) for x in category_path.read_text().split()}
        overlap = geotagged & category_ids
        union = geotagged | category_ids
        print(
            f"category corpus: {len(category_ids):,}; geotagged: {len(geotagged):,}; "
            f"overlap: {len(overlap):,}; union: {len(union):,}"
        )
        combined_path = OUT_DIR / "arizona_combined.page_ids.txt"
        combined_path.write_text("\n".join(str(p) for p in sorted(union)) + "\n")
        print(f"wrote category+geotagged union -> {combined_path}")
    else:
        print(f"no existing category corpus found at {category_path}, skipping union")


if __name__ == "__main__":
    main()
