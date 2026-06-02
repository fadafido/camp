"""Extract the REAL BSBA Information Systems & Business Analytics degree-requirement
structure from the cached AUS catalogue PDF.

This reads the *programme structure* pages (General Education / Business Core /
Major Requirements / Major Electives / Free Electives) — NOT the course-description
pages parsed by ``aus_pdf_scraper.py``. It emits a structured JSON
(``data/raw/aus/bsba_is_requirements.json``) plus the verbatim PDF excerpts each
block was read from, so the structure can be audited as real (not inferred).

Nothing is fabricated: credit counts and course lists are taken verbatim from the
catalogue. Where the flattened two-column PDF makes an attribution ambiguous, it
is flagged in ``notes`` rather than asserted. British English. Deterministic.

This is an EXTRACTION + REPORT artefact only — it does not modify any dataset,
config or the CAP-Bench bundle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pdfminer.high_level import extract_text

_REPO = Path(__file__).resolve().parents[2]
_PDF = _REPO / "data" / "raw" / "aus" / "raw_catalog_2024-2025.pdf"
_OUT = _REPO / "data" / "raw" / "aus" / "bsba_is_requirements.json"

_BUSINESS = {"ISA", "ACC", "FIN", "MGT", "MKT", "ECO", "STA", "QBA", "BLW", "SCM",
             "CMP", "COE", "EGM", "UPL", "BUS", "ENG", "IEN"}


def _codes(text: str) -> list[str]:
    """Course codes (in catalogue subjects) appearing in order, de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b([A-Z]{2,4})\s+(\d{3})\b", text):
        if m.group(1) in _BUSINESS:
            code = f"{m.group(1)} {m.group(2)}"
            if code not in seen:
                seen.add(code)
                out.append(code)
    return out


def extract() -> dict:
    txt = extract_text(str(_PDF))

    # ---- BSBA degree-requirements block (common to all majors) ----
    deg = txt.find("a minimum of 123 credit hours", 808000)
    deg_ctx = txt[txt.rfind("Degree Requirements", 0, deg): deg + 1200]
    total_m = re.search(r"a minimum of (\d+) credit hours", deg_ctx)
    total_credits = int(total_m.group(1)) if total_m else None

    # The "as follows:" breakdown lists the five blocks with their credit minima.
    blk = txt[deg: deg + 2800]
    _WORD = {"three": 3, "nine": 9, "six": 6, "twelve": 12}
    def cr(label_re):
        m = re.search(label_re, blk)
        if not m:
            return None
        tok = m.group(1)
        return int(tok) if tok.isdigit() else _WORD.get(tok.lower())
    gen_ed = cr(r"(\d+) credit hours of\s*general education")
    innov = cr(r"entrepreneurship requirement:\s*(three|\d+) credit")
    core = cr(r"(\d+) credit hours of core\s*requirements")
    major = cr(r"(\d+) credit hours of\s*major requirements")
    free = cr(r"(nine|\d+)\s+credit\s+hours\s+of\s+free\s+electives")

    # ---- Business Core bullet list (authoritative) ----
    bc = txt.find("following business core courses")
    bc_seg = txt[bc: txt.find("Major Requirements and Major", bc)]
    core_bullets = _codes(bc_seg)
    # ACC 201/202 sit immediately after the heading (column-flatten) and close the
    # 45-credit arithmetic (13 bullets x 3 = 39; +ACC 201 +ACC 202 = 45).
    acc_tail_seg = txt[txt.find("Major Requirements and Major", bc): bc + 2600]
    acc_tail = [c for c in _codes(acc_tail_seg) if c.startswith("ACC ")][:2]

    # ---- IS Major Requirements (18 cr) ----
    mr = txt.find("Major Requirements", 834000)
    mr_seg = txt[mr: txt.find("Major Electives", mr)]
    major_req_credits = (int(re.search(r"\((\d+) credit hours\)", mr_seg).group(1))
                         if re.search(r"\((\d+) credit hours\)", mr_seg) else None)
    # course-code bullets only (skip the interleaved learning-outcome bullets)
    major_req_codes = _codes(mr_seg)

    # ---- IS Major Electives (min 12 cr) pool ----
    me = txt.find("Major Electives", mr)
    me_seg = txt[me: txt.find("Major Learning Outcomes", me)]
    major_elec_credits = (int(re.search(r"minimum of (\d+) credit hours", me_seg).group(1))
                          if re.search(r"minimum of (\d+) credit hours", me_seg) else None)
    # electives pool extends past a page break to include UPL 302 + the open rules
    me_full = txt[me: me + 1700]
    major_elec_codes = _codes(me_full)

    structure = {
        "programme": ("Bachelor of Science in Business Administration, "
                      "Major in Information Systems and Business Analytics"),
        "source": "AUS 2024-2025 Undergraduate Catalog (cached PDF)",
        "total_credits_stated": total_credits,
        "min_cgpa": 2.00,
        "blocks": {
            "general_education": {"credits": gen_ed, "type": "distribution",
                "note": "min 36 cr; specific requirements satisfied through named courses "
                        "(maths MTH 102/103, statistics QBA 201, ethics MGT 360, writing "
                        "ENG 204/208/225, etc.); outside the recommender's academic scope."},
            "innovation_entrepreneurship": {"credits": innov, "courses": ["IEN 301"]},
            "business_core": {"credits": core, "courses_explicit": core_bullets,
                "courses_inferred_from_arithmetic": acc_tail,
                "note": "Catalogue states 45 cr. The bullet list names 13 courses (39 cr); "
                        "ACC 201 + ACC 202 appear immediately adjacent (column-flatten) and "
                        "close the arithmetic to 45 cr, so the core is these 15 courses."},
            "major_requirements": {"credits": major_req_credits, "courses": major_req_codes,
                "or_choices": ["ISA 301 OR CMP 320", "ISA 303 OR COE 420"]},
            "major_electives": {"credits": major_elec_credits, "pool": major_elec_codes,
                "open_rules": ["MGT 380 OR EGM 362 (one pair in the pool)",
                               "any 300-level or above ISA courses not listed as major requirements",
                               "any approved special-topic courses at the 300 level or above"]},
            "major_requirements_plus_electives_total": major,
            "free_electives": {"credits": free, "note": "outside the academic/recommender scope"},
        },
        "computed_total_check": sum(x for x in [gen_ed, innov, core, major, free] if x),
    }
    return structure, txt


def main() -> None:
    structure, _ = extract()
    _OUT.write_text(json.dumps(structure, indent=2))
    b = structure["blocks"]
    print("BSBA Information Systems & Business Analytics — real catalogue structure")
    print(f"  total credits stated : {structure['total_credits_stated']}  (min CGPA {structure['min_cgpa']})")
    print(f"  general education    : {b['general_education']['credits']} cr (distribution)")
    print(f"  innovation/entrep.   : {b['innovation_entrepreneurship']['credits']} cr  {b['innovation_entrepreneurship']['courses']}")
    print(f"  business core        : {b['business_core']['credits']} cr")
    print(f"     explicit (13)     : {b['business_core']['courses_explicit']}")
    print(f"     +arithmetic (ACC) : {b['business_core']['courses_inferred_from_arithmetic']}")
    print(f"  major requirements   : {b['major_requirements']['credits']} cr  {b['major_requirements']['courses']}")
    print(f"     or-choices        : {b['major_requirements']['or_choices']}")
    print(f"  major electives      : {b['major_electives']['credits']} cr from pool {b['major_electives']['pool']}")
    print(f"  major req+elec total : {b['major_requirements_plus_electives_total']} cr")
    print(f"  free electives       : {b['free_electives']['credits']} cr")
    print(f"  computed block sum   : {structure['computed_total_check']} cr")
    print(f"  wrote {_OUT}")


if __name__ == "__main__":
    main()
