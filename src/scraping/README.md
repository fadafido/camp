# Scraping / extraction

This package is a **starting point** for the catalogue collection step of the
fresh rebuild. It currently contains a single recovered module:

- `aus_pdf_parser.py` — parses the AUS undergraduate catalogue **PDF** with
  pdfminer.six. **Caveats:** it targets the AUS *BSBA Management* programme (a
  *different* programme from the Information Systems major used in the benchmark),
  and it writes a debug dict to `data/intermediate/aus/aus_pdf_extracted.json`.
  It is **not** the tool that produced any previously-committed dataset; treat it
  as a reference/skeleton for AUS-PDF parsing only.

Khalifa and UNC have **no** extractor here yet. Khalifa source catalogue trees are
in `data/raw/khalifa/`; UNC must be fetched from its public catalogue. New,
clearly-authored scrapers/extractors for all three institutions are to be written
during the rebuild, producing fresh extracted inputs for the dataset pipeline.
