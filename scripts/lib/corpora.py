"""Canonical list of the 5 corpora and their page_id list files, shared by
every script that needs to iterate over "all 5 corpora" so they can't
drift out of sync with each other (e.g. one script renaming a corpus file
without the others noticing)."""
from __future__ import annotations

CORPUS_FILES: dict[str, str] = {
    "arizona": "arizona.page_ids.txt",
    "computer_science": "computer_science.page_ids.txt",
    "literature": "literature.page_ids.txt",
    "vital_level_3": "vital_level_3.page_ids.txt",
    "vital_level_4": "vital_level_4.page_ids.txt",
}
