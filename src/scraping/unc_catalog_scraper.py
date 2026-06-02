"""UNC Chapel Hill catalogue scraper — BS Economics.

Fetches the public 2024-2025 catalogue pages from ``catalog.unc.edu`` (HTML),
caches them under ``data/raw/unc/catalogue_pages/``, and parses them into the
ingest schema consumed by :mod:`src.dataset.real_ingest`:

  * ``extracted_courses.json`` — a list of course records, each with
    ``code`` ("ECON 310"), ``title``, ``credits`` and verbatim ``prereq_raw``.
  * ``extracted_program.json`` — ``referenced_course_codes`` (the courses the BS
    programme requirement table links to) plus ``source_url`` / ``access_date``.

The UNC catalogue exposes each course as a ``div.courseblock`` with three header
spans (``detail-code`` / ``detail-title`` / ``detail-hours``) and an optional
``detail-requisites`` span holding the verbatim requisite text. The programme
page exposes its required courses as ``a.code`` anchors inside a
``table.sc_courselist``.

Subjects fetched: ECON (the major) plus the MATH / STOR / COMP support subjects
the BS programme references. Scoping (ECON < 500 + referenced support) is applied
downstream by ``unc_ingest.py``; this scraper emits the full subject catalogues so
the ingest can scope faithfully.

This is a *live fetch* of a public HTML catalogue. The fetched HTML is cached so
the parse is reproducible offline. British English. Deterministic; no randomness.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / "data" / "raw" / "unc"
_CACHE_DIR = _OUT_DIR / "catalogue_pages"

_UA = {"User-Agent": "Mozilla/5.0 (CAMP research; public catalogue extraction)"}
_BASE = "https://catalog.unc.edu"

# Programme page (referenced courses) + the course-description pages we parse.
PROGRAMME_PAGE = ("economics-major-bs",
                  f"{_BASE}/undergraduate/programs-study/economics-major-bs/")
COURSE_PAGES = {
    "econ": f"{_BASE}/courses/econ/",
    "math": f"{_BASE}/courses/math/",
    "stor": f"{_BASE}/courses/stor/",
    "comp": f"{_BASE}/courses/comp/",
}


def _nbsp(s: str) -> str:
    """Normalise non-breaking spaces and collapse runs of whitespace."""
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def fetch_and_cache(force: bool = False) -> None:
    """Download the programme + course pages into the HTML cache (if missing)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pages = {PROGRAMME_PAGE[0]: PROGRAMME_PAGE[1],
             **{f"courses-{k}": v for k, v in COURSE_PAGES.items()}}
    for name, url in pages.items():
        dest = _CACHE_DIR / f"{name}.html"
        if dest.exists() and not force:
            continue
        resp = requests.get(url, headers=_UA, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        print(f"  fetched {url} -> {dest.name} ({len(resp.text)} bytes)")


def _parse_course_block(block) -> dict | None:
    """Parse one ``div.courseblock`` into a schema course record, or None."""
    code_node = block.select_one(".detail-code")
    if not code_node:
        return None
    code = _nbsp(code_node.get_text(" ")).rstrip(".").strip()
    if not re.match(r"^[A-Z]{2,5}\s*\d", code):
        return None
    title_node = block.select_one(".detail-title")
    hours_node = block.select_one(".detail-hours")
    req_node = block.select_one(".detail-requisites")

    title = _nbsp(title_node.get_text(" ")).rstrip(".").strip() if title_node else None

    credits = None
    if hours_node:
        m = re.search(r"\d+(?:[.-]\d+)?", _nbsp(hours_node.get_text(" ")))
        if m:
            credits = m.group(0)

    prereq_raw = ""
    if req_node:
        txt = _nbsp(req_node.get_text(" "))
        # Drop the leading "Requisites:" label; keep the requisite text verbatim.
        prereq_raw = re.sub(r"^Requisites?:\s*", "", txt).strip()

    return {"code": code, "title": title, "credits": credits,
            "prereq_raw": prereq_raw}


def parse_courses() -> list[dict]:
    """Parse all cached course pages into deduplicated course records."""
    seen: dict[str, dict] = {}
    for name in (f"courses-{k}" for k in COURSE_PAGES):
        html = (_CACHE_DIR / f"{name}.html").read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        for block in soup.select("div.courseblock"):
            rec = _parse_course_block(block)
            if rec and rec["code"] not in seen:
                seen[rec["code"]] = rec
    return sorted(seen.values(), key=lambda c: c["code"])


def parse_referenced_codes() -> list[str]:
    """Course codes the BS Economics requirement table links to."""
    html = (_CACHE_DIR / f"{PROGRAMME_PAGE[0]}.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    codes: list[str] = []
    seen: set[str] = set()
    for a in soup.select("table.sc_courselist a.code"):
        code = _nbsp(a.get_text(" "))
        if re.match(r"^[A-Z]{2,5}\s*\d", code) and code not in seen:
            seen.add(code)
            codes.append(code)
    return sorted(codes)


def main() -> None:
    print("UNC — fetching/parsing catalog.unc.edu (BS Economics) ...")
    fetch_and_cache()
    courses = parse_courses()
    referenced = parse_referenced_codes()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUT_DIR / "extracted_courses.json").open("w") as fh:
        json.dump(courses, fh, indent=2, ensure_ascii=False)
    programme = {
        "programme": "Bachelor of Science in Economics",
        "university": "unc",
        "source_url": PROGRAMME_PAGE[1],
        "course_source_urls": COURSE_PAGES,
        "access_date": "2026-06-02",
        "referenced_course_codes": referenced,
    }
    with (_OUT_DIR / "extracted_program.json").open("w") as fh:
        json.dump(programme, fh, indent=2, ensure_ascii=False)

    n_pre = sum(1 for c in courses if c["prereq_raw"])
    print(f"  courses={len(courses)}  with_prereq={n_pre}  referenced={len(referenced)}")
    print(f"  wrote {_OUT_DIR/'extracted_courses.json'} and extracted_program.json")


if __name__ == "__main__":
    main()
