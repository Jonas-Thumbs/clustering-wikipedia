# Documentation: Text Preprocessing & Cleaning Pipeline

This document details the text cleaning architecture, its design decisions, output formats, and storage locations for the Wikipedia text processing pipeline.

---

## 1. Storage Location & Directory Layout

The cleaned text is saved as persistent JSON Lines (`.jsonl`) files in:

```text
clustering-wikipedia/
└── data/
    └── derived/
        ├── text/                    <- Raw wikitext from teammates (input)
        │   ├── arizona.articles.jsonl
        │   ├── computer_science.articles.jsonl
        │   ├── literature.articles.jsonl
        │   └── vital_level_3.articles.jsonl
        └── cleaned/                 <- Cleaned plaintext (output)
            ├── arizona.cleaned.jsonl
            ├── computer_science.cleaned.jsonl
            ├── literature.cleaned.jsonl
            └── vital_level_3.cleaned.jsonl
```

> **Note**: Because both `data/derived/text/` and `data/derived/cleaned/` reside under `data/derived/`, they are automatically ignored by Git (via `.gitignore`) to prevent committing large data files to the repository.

---

## 2. Record Schema

Each line in `data/derived/cleaned/{corpus}.cleaned.jsonl` is a self-contained JSON object with the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `page_id` | `int` | Wikipedia internal page ID (used for joining against link graphs and category tags). |
| `title` | `str` | Article title (e.g., `"Assembly language"`). |
| `lead_text` | `str` | Cleaned introductory summary (preceding the first `== Section ==` header). |
| `full_text` | `str` | Cleaned text of the full article body. |
| `lead_chars` | `int` | Character length of the lead section. |
| `full_chars` | `int` | Character length of the full text. |

---

## 3. Cleaning Architecture & Methodology

Cleaning raw wikitext requires preserving natural human prose while removing heavy Wikipedia markup. The cleaning pipeline in [`scripts/lib/text_cleaning.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/lib/text_cleaning.py) uses a 4-stage pipeline:

```
                  Raw Wikitext from JSONL
                            │
                            ▼
           [Stage 1: Upfront Filtering]
           • Strip HTML comments (<!-- ... -->)
           • Strip citation tags (<ref> ... </ref>)
           • Strip media galleries (<gallery> ... </gallery>)
           • Strip non-semantic tags (math, tables, timelines)
                            │
                            ▼
           [Stage 2: AST Parsing (mwparserfromhell)]
           • Parse DOM tree of remaining wikitext
           • Remove infobox & navbox templates ({{Infobox ...}})
           • Remove media & category links ([[File:...]], [[Category:...]])
           • Extract section 0 as lead_raw
           • Render plaintext via strip_code()
                            │
                            ▼
           [Stage 3: Residual Markup Post-Stripping]
           • Balanced bracket stripping for nested templates ({{code|...}}, {{cvt|...}})
           • Multiline [[File:...]] caption removal
           • Dangling HTML tag removal (<div ...>)
           • Wiki formatting tick removal (''', '')
                            │
                            ▼
           [Stage 4: Normalization & Splitting]
           • Whitespace and newline consolidation (\n{3,} -> \n\n)
           • Produce dual outputs: lead_text and full_text
```

### Why Dual Outputs (`lead_text` and `full_text`)?

1. **`lead_text` for Dense Embeddings (SBERT)**:
   - Transformer models have strict maximum sequence lengths (typically 256–512 wordpieces). Truncating an entire article to the first 512 tokens naively cuts off mid-sentence.
   - The Wikipedia lead section is explicitly written by editors as a concise, balanced summary of the whole topic. Pairing `f"{title}. {lead_text}"` captures deep semantic intent cleanly within model token budgets.
2. **`full_text` for Sparse Representations (TF-IDF + SVD)**:
   - Latent Semantic Analysis and TF-IDF benefit from full vocabulary breadth and term frequency distributions across all body sections.

---

## 4. Scripts & Utilities

- **[`scripts/lib/text_cleaning.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/lib/text_cleaning.py)**: The core cleaning function `clean_wikitext(wikitext: str) -> dict[str, str]`.
- **[`scripts/check_text_cleaning.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/check_text_cleaning.py)**: Automated verification suite testing regex violations, empty bodies, and throughput across all 4 corpora.
- **[`scripts/batch_clean_text.py`](file:///c:/Users/jonas/Documents/VSCode/clustering-wikipedia/scripts/batch_clean_text.py)**: The batch runner that iterates through `data/derived/text/` and writes the persistent cleaned JSONL files to `data/derived/cleaned/`.

