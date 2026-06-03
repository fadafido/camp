"""AUS catalogue scraper — BSBA Information Systems & Business Analytics.

Parses the committed AUS 2024-2025 undergraduate catalogue PDF
(``data/raw/aus/raw_catalog_2024-2025.pdf``) with ``pdfminer.six`` into the
ingest schema consumed by :mod:`src.dataset.real_ingest`:

  * ``extracted_courses.json`` — a list of course records, each with
    ``code`` ("ISA 301"), ``title``, ``credits`` and verbatim ``prereq_raw``.
  * ``extracted_program.json`` — ``referenced_course_codes`` (the courses the IS
    major requires) plus ``source_url`` / ``access_date``.

The AUS course descriptions print each course as::

    SUBJ NNN Title Words (L-L-C). Description ... Prerequisite(s): <text>.

where the credit value is the *third* number of the ``(lecture-lab-credit)``
triple. The PDF is a multi-column layout that ``pdfminer`` flattens, so course
codes appear both as headers *and* inside prerequisite text. We anchor a genuine
course header on a clean title (no ``.`` / ``(`` / ``)``) immediately followed by
the ``(L-L-C).`` credit marker, which excludes in-text references.

This module targets the IS major and emits the ingest schema. PDF parsing is
brittle: unparseable fragments are counted and logged rather than guessed.
British English. Deterministic.
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

from pdfminer.high_level import extract_text

from src.dataset import aus_is_structure as aus_is

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NON_BUSINESS_NAMED = aus_is.NON_BUSINESS_NAMED  # BUS 100, ENG 225, IEN 301
_AUS_DIR = _REPO_ROOT / "data" / "raw" / "aus"
_PDF = _AUS_DIR / "raw_catalog_2024-2025.pdf"
_SOURCE_URL = ("https://www.aus.edu/ (AUS 2024-2025 Undergraduate Catalog, "
               "cached PDF: data/raw/aus/raw_catalog_2024-2025.pdf)")

# Subjects in scope for the IS programme: the business-core home subjects plus
# the cross-listed subjects the IS major references (per aus_ingest.py).
_SUBJECTS = ["ISA", "ACC", "FIN", "MGT", "MKT", "ECO", "STA", "QBA", "BLW",
             "CMP", "COE", "EGM", "SCM", "UPL"]
_SUBJ_RE = "(?:" + "|".join(_SUBJECTS) + ")"

# A genuine course header: SUBJ NNN <clean title> (lecture-lab-credit).
_COURSE_RE = re.compile(
    rf"\b({_SUBJ_RE})\s+(\d{{3}})\s+([A-Z][^.()]{{2,90}}?)\s*"
    rf"\((\d+)-(\d+)-(\d+)\)\.\s*"
    rf"(.*?)(?=\b{_SUBJ_RE}\s+\d{{3}}\s+[A-Z][^.()]{{2,90}}?\s*\(\d+-\d+-\d+\)\.|\Z)",
    re.S,
)

# Requirement sentences kept verbatim as prereq_raw (the prereq parser handles
# the prose: grade thresholds, standing, coreqs, "or", etc.).
_REQ_RE = re.compile(
    r"(?:Prerequisites?|Co-?requisites?|Prerequisite/concurrent)\b[^.]*\.")


def _flatten(text: str) -> str:
    """Join wrapped lines so multi-line titles/prereqs become single strings."""
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return re.sub(r"\s+", " ", text)


def parse_pdf() -> tuple[list[dict], dict, str]:
    """Parse the PDF into (course_records, diagnostics, flattened_text)."""
    raw = extract_text(str(_PDF))
    flat = _flatten(raw)

    by_code: dict[str, dict] = {}
    n_no_prereq = 0
    for subj, num, title, _lec, _lab, credit, body in _COURSE_RE.findall(flat):
        code = f"{subj} {num}"
        if code in by_code:
            continue  # first (descriptions-section) occurrence wins
        title_clean = re.sub(r"\s+", " ", title).strip()
        reqs = _REQ_RE.findall(body)
        prereq_raw = " ".join(r.strip() for r in reqs).strip()
        if not prereq_raw:
            n_no_prereq += 1
        by_code[code] = {
            "code": code,
            "title": title_clean or None,
            "credits": credit,          # third number of the (L-L-C) triple
            "prereq_raw": prereq_raw,
        }

    # Named non-business-subject courses that are part of the BSBA-IS academic
    # programme (BUS 100, ENG 225, IEN 301) but whose subjects were not in the
    # business scrape. They are real catalogue courses (extracted, not invented).
    for code in _NON_BUSINESS_NAMED:
        if code in by_code:
            continue
        rec = _extract_named(flat, code)
        if rec:
            by_code[code] = rec

    courses = sorted(by_code.values(), key=lambda c: c["code"])
    diagnostics = {
        "n_courses": len(courses),
        "n_without_prereq": n_no_prereq,
        "by_subject": {s: sum(1 for c in courses if c["code"].startswith(s + " "))
                       for s in _SUBJECTS},
    }
    return courses, diagnostics, flat


def _extract_named(flat: str, code: str) -> dict | None:
    """Extract one specific named course (any subject) from the flattened text."""
    subj, num = code.split()
    m = re.search(
        rf"\b{subj}\s+{num}\s+([A-Z][^.()]{{2,90}}?)\s*\((\d+)-(\d+)-(\d+)\)\.\s*"
        rf"(.*?)(?=\b[A-Z]{{2,4}}\s+\d{{3}}\s+[A-Z][^.()]{{2,90}}?\s*\(\d+-\d+-\d+\)\.|\Z)",
        flat, re.S)
    if not m:
        return None
    reqs = _REQ_RE.findall(m.group(5))
    return {"code": code, "title": re.sub(r"\s+", " ", m.group(1)).strip() or None,
            "credits": m.group(4), "prereq_raw": " ".join(r.strip() for r in reqs).strip()}


def _prereq_closure(courses: list[dict], seed: set[str]) -> set[str]:
    """Transitive prerequisite closure of ``seed`` over real prereq references,
    restricted to courses present in the extraction."""
    by = {c["code"]: c for c in courses}
    present = set(by)
    closure = {c for c in seed if c in present}
    queue = deque(closure)
    while queue:
        raw = by.get(queue.popleft(), {}).get("prereq_raw", "") or ""
        for mm in re.finditer(r"\b([A-Z]{2,4})\s+(\d{3})\b", raw):
            ref = f"{mm.group(1)} {mm.group(2)}"
            if ref in present and ref not in closure:
                closure.add(ref)
                queue.append(ref)
    return closure


def referenced_codes(courses: list[dict], flat: str) -> list[str]:
    """The BSBA-IS academic course scope = the real degree-block courses (Business
    Core + I&E + Major Requirements incl. "or" alternatives + Major Electives pool)
    PLUS their transitive prerequisite closure, restricted to courses present in
    the extraction. ISA electives ("any 300-level+ ISA not a major requirement")
    enter scope via the ISA home subject downstream. Encodes the real catalogue
    programme (see :mod:`src.dataset.aus_is_structure`)."""
    present = {c["code"] for c in courses}
    seed = set(aus_is.all_block_codes()) & present
    return sorted(_prereq_closure(courses, seed))


def main() -> None:
    print("AUS — parsing 2024-2025 catalogue PDF (BSBA Information Systems) ...")
    if not _PDF.exists():
        raise SystemExit(f"STOP: AUS PDF not found at {_PDF}")
    courses, diag, flat = parse_pdf()
    referenced = referenced_codes(courses, flat)

    with (_AUS_DIR / "extracted_courses.json").open("w") as fh:
        json.dump(courses, fh, indent=2, ensure_ascii=False)
    programme = {
        "programme": ("Bachelor of Science in Business Administration, "
                      "Major in Information Systems and Business Analytics"),
        "university": "aus",
        "source_url": _SOURCE_URL,
        "access_date": "2026-06-02",
        "referenced_course_codes": referenced,
    }
    with (_AUS_DIR / "extracted_program.json").open("w") as fh:
        json.dump(programme, fh, indent=2, ensure_ascii=False)

    print(f"  courses={diag['n_courses']}  without_prereq={diag['n_without_prereq']}")
    print(f"  by subject: {diag['by_subject']}")
    print(f"  referenced_course_codes={len(referenced)}")
    print(f"  wrote {_AUS_DIR/'extracted_courses.json'} and extracted_program.json")


if __name__ == "__main__":
    main()
