"""Khalifa University live catalogue scraper — BSc Computer Science.

The committed raw tree (``data/raw/khalifa/raw_catalog.json``) is only a Sitecore
navigation sitemap (codes + Paths, no course detail). Those Paths map directly to
the public SmartCatalogIQ catalogue at ``ku-ae.smartcatalogiq.com``, whose course
pages ARE server-rendered static HTML containing the title, credit hours and
verbatim prerequisite text. This module fetches those live pages, caches them, and
parses them into the ingest schema consumed by :mod:`src.dataset.real_ingest`:

  * ``extracted_courses.json`` — list of records with ``code`` ("COSC 201"),
    ``title``, ``credits`` (integer) and verbatim ``prereq_raw``.
  * ``extracted_program.json`` — ``referenced_course_codes`` (the courses the BSc
    CS requirements page links to) + ``source_url`` + ``access_date``.

Course-page structure (static HTML, verified by probe)::

    <h1><span>COSC 201</span> Computer Systems Organization</h1>
    <div class="sc_credits"><div class="credits">3</div></div>
    <div class="sc_prereqs"><h3>Prerequisite</h3>COSC101</div>

Prerequisite text is kept exactly as published (e.g. "COSC301,( COSC310, or
ECCE342)") so the downstream ``prereq_parser`` does the structuring. Nothing is
fabricated: a field absent from the page is left null/empty. The fetched HTML is
cached under ``data/raw/khalifa/catalogue_pages/`` so the parse is reproducible
offline. British English. Deterministic; no randomness.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KHAL_DIR = _REPO_ROOT / "data" / "raw" / "khalifa"
_CACHE_DIR = _KHAL_DIR / "catalogue_pages"
_RAW_TREE = _KHAL_DIR / "raw_catalog.json"

_HOST = "https://ku-ae.smartcatalogiq.com"
_BASE = _HOST + "/en"
_UA = {"User-Agent": "Mozilla/5.0 (CAMP research; public catalogue extraction; contact fadi)"}

# BSc CS requirements page (its course links give the referenced support set).
_REQ_PATH = ("/2024-2025/undergraduate-catalog/college-of-computing-and-mathematical-sciences"
             "/department-of-computer-science/bachelor-of-science-in-computer-science"
             "/bachelor-of-science-in-computer-science-requirements")
_REQ_URL = _BASE + _REQ_PATH
_PROGRAMME_URL = (_BASE + "/2024-2025/undergraduate-catalog/college-of-computing-and-mathematical-sciences"
                  "/department-of-computer-science/bachelor-of-science-in-computer-science")

_COURSE_SEG_RE = re.compile(r"([a-zA-Z]{3,5})-(\d{2,4}[a-zA-Z]?)$")
_FETCH_DELAY_S = 0.4  # polite pause between requests


def _seg_to_code(url_or_href: str) -> str | None:
    """'.../courses/cosc-computer-science/200/cosc-201' -> 'COSC 201'."""
    seg = url_or_href.rstrip("/").split("/")[-1]
    m = _COURSE_SEG_RE.match(seg)
    return f"{m.group(1).upper()} {m.group(2).upper()}" if m else None


def _cache_name(url: str) -> str:
    code = _seg_to_code(url)
    return (code.replace(" ", "_") if code else url.rstrip("/").split("/")[-1]) + ".html"


def _get(url: str) -> str:
    """Fetch a URL as UTF-8 text, caching to disk; reuse cache if present."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / _cache_name(url)
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    resp = requests.get(url, headers=_UA, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    dest.write_text(resp.text, encoding="utf-8")
    time.sleep(_FETCH_DELAY_S)
    return resp.text


def _sitemap_cosc_urls() -> list[str]:
    """COSC course-page URLs derived from the committed sitemap Paths."""
    tree = json.loads(_RAW_TREE.read_text())

    def find(n, suf):
        if not isinstance(n, dict):
            return None
        if n.get("Path", "").endswith(suf):
            return n
        for c in n.get("Children", []) or []:
            r = find(c, suf)
            if r:
                return r
        return None

    cosc = find(tree, "/Courses/COSC-Computer-Science")
    urls: list[str] = []

    def leaves(n):
        ch = n.get("Children", []) or []
        if not ch and _seg_to_code(n.get("Path", "")):
            urls.append(_BASE + n["Path"].lower())
        for c in ch:
            leaves(c)

    if cosc:
        leaves(cosc)
    return urls


def _requirements_course_links(req_html: str) -> list[str]:
    """Absolute URLs of every course the requirements page links to."""
    soup = BeautifulSoup(req_html, "lxml")
    main = soup.select_one("#main") or soup
    out: list[str] = []
    seen: set[str] = set()
    for a in main.select("a[href]"):
        href = a["href"]
        if "/courses/" in href and _seg_to_code(href):
            url = href if href.startswith("http") else _HOST + href
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


def _parse_course(html: str) -> dict | None:
    """Parse a SmartCatalogIQ course page into a schema record."""
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("#main")
    if not main:
        return None
    h1 = main.find("h1")
    if not h1:
        return None
    span = h1.find("span")
    code = re.sub(r"\s+", " ", span.get_text(" ", strip=True)).strip() if span else ""
    if not re.match(r"^[A-Z]{2,5}\s*\d", code):
        return None
    # Title = the h1 text with the code span removed.
    if span:
        span.extract()
    title = re.sub(r"\s+", " ", h1.get_text(" ", strip=True)).strip() or None

    credits = None
    cr = main.select_one(".sc_credits .credits")
    if cr:
        m = re.search(r"\d+", cr.get_text(" ", strip=True))
        if m:
            credits = int(m.group(0))

    prereq_raw = ""
    pre = main.select_one(".sc_prereqs")
    if pre:
        # Keep the verbatim text after the "Prerequisite(s)" heading label.
        h3 = pre.find("h3")
        if h3:
            h3.extract()
        prereq_raw = re.sub(r"\s+", " ", pre.get_text(" ", strip=True)).strip()

    return {"code": code, "title": title, "credits": credits, "prereq_raw": prereq_raw}


def main() -> None:
    print("Khalifa — live fetch of ku-ae.smartcatalogiq.com (BSc Computer Science) ...")
    req_html = _get(_REQ_URL)
    referenced = sorted({c for c in (_seg_to_code(u)
                         for u in _requirements_course_links(req_html)) if c})

    # Course pages to fetch = sitemap COSC pages + every requirements-page course link.
    urls: list[str] = []
    seen: set[str] = set()
    for url in _sitemap_cosc_urls() + _requirements_course_links(req_html):
        if url not in seen:
            seen.add(url)
            urls.append(url)

    courses: dict[str, dict] = {}
    failed: list[str] = []
    for url in urls:
        try:
            rec = _parse_course(_get(url))
        except Exception as exc:  # network/parse failure on a single page
            failed.append(f"{url} ({type(exc).__name__})")
            continue
        if rec and rec["code"] not in courses:
            courses[rec["code"]] = rec

    course_list = sorted(courses.values(), key=lambda c: c["code"])
    with (_KHAL_DIR / "extracted_courses.json").open("w") as fh:
        json.dump(course_list, fh, indent=2, ensure_ascii=False)
    programme = {
        "programme": "Bachelor of Science in Computer Science",
        "university": "khalifa",
        "source_url": _PROGRAMME_URL,
        "requirements_url": _REQ_URL,
        "access_date": "2026-06-02",
        "referenced_course_codes": referenced,
    }
    with (_KHAL_DIR / "extracted_program.json").open("w") as fh:
        json.dump(programme, fh, indent=2, ensure_ascii=False)

    n_title = sum(1 for c in course_list if c["title"])
    n_cred = sum(1 for c in course_list if c["credits"])
    n_pre = sum(1 for c in course_list if c["prereq_raw"])
    print(f"  fetched {len(urls)} pages, parsed {len(course_list)} courses")
    print(f"  with_title={n_title}  with_credits={n_cred}  with_prereq={n_pre}  referenced={len(referenced)}")
    if failed:
        print(f"  FAILED pages ({len(failed)}): {failed[:10]}")
    print(f"  wrote {_KHAL_DIR/'extracted_courses.json'} and extracted_program.json")


if __name__ == "__main__":
    main()
