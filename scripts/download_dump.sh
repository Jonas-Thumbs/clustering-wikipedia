#!/usr/bin/env bash
# Downloads the enwiki-20260901 dump files needed for the project.
set -euo pipefail

DUMP_DATE="20260901"
BASE_URL="https://dumps.wikimedia.org/enwiki/${DUMP_DATE}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# Ordered so the small files needed for category-tree inspection land first;
# the large multistream article dump (resumable via wget -c) comes last.
FILES=(
  "enwiki-${DUMP_DATE}-page.sql.gz"
  "enwiki-${DUMP_DATE}-categorylinks.sql.gz"
  "enwiki-${DUMP_DATE}-geo_tags.sql.gz"
  "enwiki-${DUMP_DATE}-redirect.sql.gz"
  "enwiki-${DUMP_DATE}-linktarget.sql.gz"
  "enwiki-${DUMP_DATE}-page_props.sql.gz"
  "enwiki-${DUMP_DATE}-md5sums.txt"
  "enwiki-${DUMP_DATE}-pagelinks.sql.gz"
  "enwiki-${DUMP_DATE}-pages-articles-multistream-index.txt.bz2"
  "enwiki-${DUMP_DATE}-pages-articles-multistream.xml.bz2"
)

for f in "${FILES[@]}"; do
  echo "=== Downloading $f ==="
  wget -c --tries=5 --timeout=60 "${BASE_URL}/${f}" -O "${f}"
done

echo "=== Verifying checksums ==="
# -F (fixed-string) matching against one pattern per line avoids "." in
# filenames being read as a regex wildcard; %.txt excludes md5sums.txt's
# own filename (it isn't listed inside itself) without affecting any of
# the .sql.gz/.bz2 names, which don't end in .txt.
PATTERN_FILE="$(mktemp)"
trap 'rm -f "$PATTERN_FILE"' EXIT
printf '%s\n' "${FILES[@]%.txt}" > "$PATTERN_FILE"
grep -F -f "$PATTERN_FILE" "enwiki-${DUMP_DATE}-md5sums.txt" | md5sum -c -

echo "=== Done ==="
