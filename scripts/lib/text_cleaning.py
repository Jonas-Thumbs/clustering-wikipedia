"""Wikitext cleaning and section extraction utility for Wikipedia articles."""
from __future__ import annotations

import re
import mwparserfromhell

_RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_RE_MULTIPLE_SPACES = re.compile(r"[ \t]+")
_RE_MEDIA_CATEGORIES = re.compile(
    r"\[\[(?:File|Image|Category|Media):(?:[^\[\]]|\[\[[^\[\]]*\]\])*\]\]",
    re.IGNORECASE | re.DOTALL,
)
_RE_BALANCED_TEMPLATES = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_RE_REMAINING_HTML = re.compile(r"<[^>]+>")


def _strip_all_templates_exact(s: str) -> str:
    """Strip all {{...}} templates iteratively until none remain."""
    prev = None
    curr = s
    while prev != curr:
        prev = curr
        curr = _RE_BALANCED_TEMPLATES.sub("", curr)
    return curr


def clean_wikitext(wikitext: str) -> dict[str, str]:
    """Clean raw wikitext into plain text.

    Returns:
        {
            "lead_text": Clean text of the article's lead/summary section,
            "full_text": Clean text of the full article body.
        }
    """
    if not wikitext or not wikitext.strip():
        return {"lead_text": "", "full_text": ""}

    try:
        parsed = mwparserfromhell.parse(wikitext)

        # 1. Strip comments
        for c in parsed.filter_comments():
            try:
                parsed.remove(c)
            except Exception:
                pass

        # 2. Strip tags (citations, galleries, math, tables)
        for tag in parsed.filter_tags():
            if tag.tag.lower() in ("ref", "gallery", "math", "table", "timeline"):
                try:
                    parsed.remove(tag)
                except Exception:
                    pass

        # 3. Strip templates (infoboxes, navboxes, cite notes)
        for t in parsed.filter_templates():
            try:
                parsed.remove(t)
            except Exception:
                pass

        # 4. Strip media, file, and category wikilinks
        for link in parsed.filter_wikilinks():
            target = str(link.title).strip().lower()
            if target.startswith(("file:", "image:", "category:", "media:", "user:")):
                try:
                    parsed.remove(link)
                except Exception:
                    pass

        # Extract lead section before rendering
        sections = parsed.get_sections(include_lead=True, flat=True)
        if sections:
            lead_raw = str(sections[0])
            lead_clean = mwparserfromhell.parse(lead_raw).strip_code()
        else:
            lead_clean = ""

        full_clean = parsed.strip_code()

    except Exception:
        fallback = _regex_fallback_clean(wikitext)
        lead_clean = fallback["lead_text"]
        full_clean = fallback["full_text"]

    # Post-clean: remove any lingering multiline [[File:...]] that bypassed AST
    while _RE_MEDIA_CATEGORIES.search(full_clean):
        full_clean = _RE_MEDIA_CATEGORIES.sub("", full_clean)
    while _RE_MEDIA_CATEGORIES.search(lead_clean):
        lead_clean = _RE_MEDIA_CATEGORIES.sub("", lead_clean)

    # Post-clean: remove any leftover {{...}} templates that were unparsed by AST
    full_clean = _strip_all_templates_exact(full_clean)
    lead_clean = _strip_all_templates_exact(lead_clean)

    # Strip any dangling HTML tags
    full_clean = _RE_REMAINING_HTML.sub("", full_clean)
    lead_clean = _RE_REMAINING_HTML.sub("", lead_clean)

    # Strip leftover formatting ticks
    full_clean = re.sub(r"\'\'\'?", "", full_clean)
    lead_clean = re.sub(r"\'\'\'?", "", lead_clean)

    lead_text = _normalize_whitespace(lead_clean)
    full_text = _normalize_whitespace(full_clean)

    if not lead_text and full_text:
        lead_text = full_text.split("\n\n")[0]

    return {
        "lead_text": lead_text,
        "full_text": full_text,
    }


def _normalize_whitespace(text: str) -> str:
    lines = [_RE_MULTIPLE_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return _RE_MULTIPLE_NEWLINES.sub("\n\n", joined).strip()


def _regex_fallback_clean(raw: str) -> dict[str, str]:
    text = re.sub(r"<ref\b[^>]*>.*?</ref>|<ref\b[^/>]*/>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    while _RE_MEDIA_CATEGORIES.search(text):
        text = _RE_MEDIA_CATEGORIES.sub("", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = _strip_all_templates_exact(text)
    text = _RE_REMAINING_HTML.sub("", text)
    cleaned = _normalize_whitespace(text)
    lead = cleaned.split("\n\n")[0] if cleaned else ""
    return {"lead_text": lead, "full_text": cleaned}
