# UNC raw source

The UNC Chapel Hill **BS Economics** catalogue was fetched from the public
2024–2025 catalogue at `catalog.unc.edu` (HTML) by
`src/scraping/unc_catalog_scraper.py` on **2 June 2026**. The fetched pages are
cached and committed under `catalogue_pages/`, and the scraper parses them into
`extracted_courses.json` (code, title, credits, verbatim `prereq_raw`) and
`extracted_program.json` (`referenced_course_codes`, `source_url`,
`access_date`).

Cached pages (`catalogue_pages/`):

- `courses-econ.html` — ECON course descriptions (the major subject).
- `courses-math.html` — MATH support courses.
- `courses-stor.html` — STOR support courses.
- `courses-comp.html` — COMP support courses.
- `economics-major-bs.html` — the BS Economics programme requirement page.

These are real provenance artefacts: the scraper is cache-first, so the
extraction reproduces offline from the committed pages.
