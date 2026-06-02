"""Khalifa University catalogue extractor — BSc Computer Science.

Local parse of the committed raw catalogue trees in ``data/raw/khalifa/``
(``raw_catalog.json`` / ``raw_catalog_local.json``) into the ingest schema
consumed by :mod:`src.dataset.real_ingest`.

IMPORTANT — source limitation (reported honestly, not worked around):
The committed Khalifa raw files are a **Sitecore navigation sitemap**. Every node
carries only ``Name`` / ``Path`` / ``Children``; course nodes are bare codes
(e.g. ``"COSC 201"``). The trees contain **no course titles, no credit values and
no prerequisite text**, and the "BSc CS Requirements" node is an empty leaf, so
the programme's referenced support courses are not exposed either.

Consequently this extractor can faithfully recover the **course codes** (and the
subject hierarchy) but cannot supply ``title`` / ``credits`` / ``prereq_raw`` —
those fields are emitted as ``null`` / ``""`` rather than fabricated. The result
is therefore NOT sufficient to build a meaningful CS programme (credits would be
zero, the prerequisite DAG empty). The caller must obtain a richer Khalifa source
before this institution can be ingested. British English. Deterministic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KHAL_DIR = _REPO_ROOT / "data" / "raw" / "khalifa"
# Prefer the variant with full Sitecore paths; both carry identical course nodes.
_RAW = _KHAL_DIR / "raw_catalog_local.json"

_COURSE_FOLDER = "/Courses/COSC-Computer-Science"
_CODE_RE = re.compile(r"^([A-Z]{2,5})\s*(\d{2,4}[A-Z]?)$")
_SOURCE_URL = ("https://catalog.ku.ac.ae/ (Khalifa University 2024-2025 "
               "Undergraduate Catalog; committed raw tree: "
               "data/raw/khalifa/raw_catalog_local.json)")


def _find_node(node: dict, path_suffix: str) -> dict | None:
    if not isinstance(node, dict):
        return None
    if node.get("Path", "").endswith(path_suffix):
        return node
    for child in node.get("Children", []) or []:
        found = _find_node(child, path_suffix)
        if found:
            return found
    return None


def _leaf_codes(node: dict) -> list[str]:
    """Collect bare course codes from the leaf nodes under a subject folder."""
    out: list[str] = []
    children = node.get("Children", []) or []
    if not children:
        name = (node.get("Name") or "").strip()
        if _CODE_RE.match(name):
            out.append(name)
        return out
    for child in children:
        out.extend(_leaf_codes(child))
    return out


def parse_courses() -> tuple[list[dict], dict]:
    """Recover COSC course codes. title/credits/prereq_raw are absent in source."""
    tree = json.loads(_RAW.read_text())
    cosc = _find_node(tree, _COURSE_FOLDER)
    if cosc is None:
        raise SystemExit("STOP: COSC course folder not found in Khalifa raw tree")
    codes = sorted(set(_leaf_codes(cosc)))
    courses = [{
        "code": code,
        "title": None,          # ABSENT in source — not fabricated
        "credits": None,        # ABSENT in source — not fabricated
        "prereq_raw": "",       # ABSENT in source — not fabricated
    } for code in codes]
    diagnostics = {
        "n_courses": len(courses),
        "source_limitation": (
            "Sitecore navigation sitemap: only course codes present; no titles, "
            "credits or prerequisites in the committed raw files."),
    }
    return courses, diagnostics


def main() -> None:
    print("Khalifa — parsing committed raw catalogue tree (BSc Computer Science) ...")
    courses, diag = parse_courses()

    with (_KHAL_DIR / "extracted_courses.json").open("w") as fh:
        json.dump(courses, fh, indent=2, ensure_ascii=False)
    programme = {
        "programme": "Bachelor of Science in Computer Science",
        "university": "khalifa",
        "source_url": _SOURCE_URL,
        "access_date": "2026-06-02",
        # The BSc CS Requirements node is an empty leaf in the committed tree, so
        # the programme's referenced support courses are NOT recoverable here.
        "referenced_course_codes": [],
        "source_limitation": diag["source_limitation"],
    }
    with (_KHAL_DIR / "extracted_program.json").open("w") as fh:
        json.dump(programme, fh, indent=2, ensure_ascii=False)

    print(f"  courses={diag['n_courses']} (codes only)")
    print(f"  LIMITATION: {diag['source_limitation']}")
    print(f"  wrote {_KHAL_DIR/'extracted_courses.json'} and extracted_program.json")


if __name__ == "__main__":
    main()
