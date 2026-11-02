"""Grow a category-based corpus (Computer Science, Literature, Arizona's
category supplement) by walking data/processed/category_index.db outward
from one or more seed categories, one layer at a time, collecting real
article page_ids (deduplicated across overlapping subcategories) until the
target size is reached or the tree runs out.

This both picks the traversal depth AND produces the corpus membership in
one pass -- no separate "inspect stats, manually choose a depth" step.

Usage:
    python3 grow_category_corpus.py "Computer science" --target 1500
    python3 grow_category_corpus.py "Arizona" "Geography of Arizona" \
        --target 2000 --max-layers 4

Output: prints layer-by-layer growth (new categories found, with a sample
of their names so topic drift is visible, and new/cumulative article
counts), then writes the resulting page_id set to
data/processed/<slug>.page_ids.txt

If a layer's candidate new articles would overshoot the target, they are
NOT all included -- but the cut is made at the CATEGORY level, not the
article level: the new categories discovered at that layer are shuffled
(seeded) and added whole, one at a time, in that random order, stopping
once the cumulative total is at or past the target. Sampling individual
articles from the pooled set instead would bias the corpus toward
whichever categories happen to have the most member articles ("kids");
allocating a quota per layer-1 parent would instead bias it by how many
sibling subcategories that parent happens to have. Treating each
category as an atomic, equally-likely unit avoids both. The final size
can land a bit above or below the target -- that's expected and fine.
Traversal stops there either way (no further layer is explored from
whatever categories didn't make the cut).

Beyond the hiddencat exclusion already baked into the index (build_
category_index.py), traversal here also skips categories that look like
Wikipedia's own category-structure bookkeeping: cross-cutting
subdivisions such as "X by country/year/language" and stub-tracking
categories. These aren't flagged hiddencat (Wikipedia treats them as
normal browsable categories, not maintenance/tracking), but they cause
the same kind of runaway branching without adding new topical content --
e.g. "Literary awards" spawning "Literary awards by country/year/
language/region/genre/...". This is a tunable heuristic (unlike hiddencat,
which is ground truth from Wikipedia), so it's applied here at traversal
time rather than baked into the index, to stay cheap to adjust.

Category traversal itself is deterministic; the only randomness is the
trim-to-target draw, seeded by the SEED constant below so corpus
membership is reproducible across runs/devices.
"""
from __future__ import annotations

import argparse
import random
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "category_index.db"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

SEED = 472  # CS472/572 -- arbitrary but fixed, per-stage seed constant

# "_by_" catches cross-cutting subdivisions (e.g. "Literary_awards_by_country");
# "stubs" catches stub-tracking categories (e.g. "Africa-related_stubs") that
# slip past the hiddencat flag.
_ADMIN_CATEGORY_RE = re.compile(r"_by_|stubs", re.IGNORECASE)


def is_admin_category(title: str) -> bool:
    return bool(_ADMIN_CATEGORY_RE.search(title))


def pages_of(cur: sqlite3.Cursor, title: str) -> set[int]:
    rows = cur.execute(
        "SELECT page_id FROM category_pages WHERE category_title = ?", (title,)
    ).fetchall()
    return {r[0] for r in rows}


def children_of(cur: sqlite3.Cursor, title: str) -> tuple[list[str], int]:
    """Returns (kept children, count filtered as admin/stub categories)."""
    rows = cur.execute(
        "SELECT child_title FROM subcat_edges WHERE parent_title = ?", (title,)
    ).fetchall()
    all_children = [r[0] for r in rows]
    kept = [t for t in all_children if not is_admin_category(t)]
    return kept, len(all_children) - len(kept)


def grow(
    seeds: list[str], target: int, max_layers: int | None, con: sqlite3.Connection
) -> set[int]:
    cur = con.cursor()
    rng = random.Random(SEED)
    seed_titles = [s.replace(" ", "_") for s in seeds]
    visited = set(seed_titles)
    frontier = list(seed_titles)
    members: set[int] = set()

    for t in frontier:
        members |= pages_of(cur, t)
    print(f"layer 0 (seeds: {', '.join(seeds)}): {len(members):,} articles")

    layer = 0
    trimmed = False
    while len(members) < target and frontier and (max_layers is None or layer < max_layers):
        layer += 1
        next_frontier: list[str] = []
        filtered_count = 0
        for t in frontier:
            kids, filtered = children_of(cur, t)
            next_frontier.extend(kids)
            filtered_count += filtered
        new_titles = []
        for t in next_frontier:
            if t not in visited:
                visited.add(t)
                new_titles.append(t)

        new_article_ids: set[int] = set()
        for t in new_titles:
            new_article_ids |= pages_of(cur, t)
        new_article_ids -= members

        remaining = target - len(members)
        admin_note = f", {filtered_count:,} filtered as admin/stub" if filtered_count else ""

        if len(new_article_ids) > remaining:
            shuffled = list(new_titles)
            rng.shuffle(shuffled)
            included_titles: list[str] = []
            for t in shuffled:
                if len(members) >= target:
                    break
                members |= pages_of(cur, t)
                included_titles.append(t)
            trimmed = True
            print(
                f"layer {layer}: +{len(new_titles):,} new categories, "
                f"{len(new_article_ids):,} candidate new articles -> shuffled stop kept "
                f"{len(included_titles):,}/{len(new_titles):,} categories whole "
                f"(cumulative {len(members):,}){admin_note}"
            )
            frontier = []  # target reached; the untried categories are not explored further
        else:
            members |= new_article_ids
            print(
                f"layer {layer}: +{len(new_titles):,} new categories, "
                f"+{len(new_article_ids):,} new articles (cumulative {len(members):,}){admin_note}"
            )
            if new_titles:
                sample = ", ".join(t.replace("_", " ") for t in new_titles[:8])
                print(f"  sample new categories: {sample}" + (" ..." if len(new_titles) > 8 else ""))
            frontier = new_titles

    if trimmed:
        print(f"stopped: reached target via shuffled category stop ({len(members):,} vs target {target})")
    elif not frontier:
        print(f"stopped: category tree exhausted at layer {layer}, {len(members):,} articles total")
    elif max_layers is not None and layer >= max_layers:
        print(f"stopped: hit --max-layers={max_layers}, {len(members):,} articles total")
    else:
        print(f"stopped: reached target ({len(members):,} >= {target})")

    return members


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seeds", nargs="+", help="Seed category title(s), without 'Category:' prefix")
    ap.add_argument("--target", type=int, required=True, help="Stop once cumulative article count reaches this")
    ap.add_argument("--max-layers", type=int, default=None, help="Hard cap on traversal depth regardless of size")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--out", default=None, help="Output path for the page_id list (default: auto-named in data/processed/)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"category index not found at {args.db} -- run build_category_index.py first")

    con = sqlite3.connect(args.db)
    members = grow(args.seeds, args.target, args.max_layers, con)
    con.close()

    slug = re.sub(r"[^a-z0-9]+", "_", args.seeds[0].lower()).strip("_")
    out_path = Path(args.out) if args.out else OUT_DIR / f"{slug}.page_ids.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in sorted(members)) + "\n")
    print(f"wrote {len(members):,} page_ids -> {out_path}")


if __name__ == "__main__":
    main()
