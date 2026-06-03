# Scraping / extraction

Catalogue collection for CAP-Bench. Each module turns one institution's
**publicly published undergraduate course catalogue** into the uniform pair the
ingest layer consumes — `extracted_courses.json` (code, title, credits, verbatim
`prereq_raw`) and `extracted_program.json` (`referenced_course_codes`,
`source_url`, `access_date`). Access date for all sources: **2 June 2026**.
British English; vendor-neutral.

## Live scrapers (one per institution)

- `khalifa_catalog_scraper.py` — **Khalifa University, BSc Computer Science.**
  Fetches the live SmartCatalogIQ catalogue (`ku-ae.smartcatalogiq.com`), whose
  course pages are server-rendered static HTML carrying title, credit hours and
  verbatim prerequisite text. Pages are cached under
  `data/raw/khalifa/catalogue_pages/` and parsed into the ingest schema.
- `unc_catalog_scraper.py` — **UNC Chapel Hill, BS Economics.** Fetches the
  public 2024–2025 catalogue (`catalog.unc.edu`), where each course is a
  `div.courseblock` (`detail-code` / `detail-title` / `detail-hours` /
  `detail-requisites`) and the programme lists its required courses as `a.code`
  anchors in a `table.sc_courselist`. Caches under
  `data/raw/unc/catalogue_pages/`. Subjects: ECON (major) plus MATH / STOR / COMP
  support.
- `aus_pdf_scraper.py` — **American University of Sharjah, BSBA Information
  Systems & Business Analytics.** Parses the committed AUS 2024–2025 catalogue
  PDF (`data/raw/aus/raw_catalog_2024-2025.pdf`) with `pdfminer.six`, reading the
  ISA course descriptions (the credit value is the third number of the
  `(lecture-lab-credit)` triple).
- `aus_requirements_extract.py` — extracts the **real BSBA-IS degree-requirement
  block structure** (General Education / Business Core / Major Requirements /
  Major Electives / Free Electives) from the same AUS PDF, emitting
  `data/raw/aus/bsba_is_requirements.json` with the verbatim PDF excerpts each
  block was read from, so the structure can be audited as real. This reads the
  *programme-structure* pages — not the course-description pages parsed by
  `aus_pdf_scraper.py`. Extraction-and-report only: it modifies no dataset.

## Downstream

The `extracted_*.json` outputs feed `src/dataset/<inst>_ingest.py`
(`khalifa_ingest`, `aus_ingest`, `unc_ingest`), which normalise them into
CAP-Bench entities under `data/intermediate/<inst>/` via the shared
`src/dataset/real_ingest.py` and the `src/dataset/prereq_parser.py` prerequisite
parser.
