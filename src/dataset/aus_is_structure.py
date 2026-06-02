"""Canonical BSBA Information Systems & Business Analytics block structure.

Single source of truth for the REAL AUS degree-requirement structure, extracted
verbatim from the 2024-2025 catalogue PDF (see ``aus_requirements_extract.py`` and
``data/raw/aus/bsba_is_requirements.json``). Used by both the AUS scraper (to
define the academic course scope) and the AUS graduation-threshold builder (to
encode the block/rule structure), so the two never drift.

Real structure (123-credit degree, min CGPA 2.00):
  - General Education      36 cr  (distribution — OUTSIDE the recommender scope)
  - Innovation & Entrep.    3 cr  (IEN 301)
  - Business Core          45 cr  (15 courses, all required)
  - Major Requirements     18 cr  (6 requirements, two as "A or B" choices)
  - Major Electives        12 cr  (>= 12 cr from a named pool)
  - Free Electives          9 cr  (unconstrained — OUTSIDE the recommender scope)

The modelled ACADEMIC scope = Business Core + I&E + Major Requirements + Major
Electives = 45 + 3 + 18 + 12 = 78 cr. General Education (36) and Free Electives
(9) — 45 cr — sit outside the recommender's rankable course universe.

DISCLOSED INFERENCE: ACC 201 + ACC 202 are taken as the two business-core courses
that close the catalogue's stated 45-cr core (the bullet list names 13 = 39 cr;
ACC 201/202 sit immediately adjacent in the flattened two-column text and in the
year-1 study plan). British English. No fabrication beyond this disclosed item.
"""

from __future__ import annotations

# Business Core — 45 cr, all required (15 courses; ACC 201/202 = disclosed inference).
BUSINESS_CORE = [
    "BLW 301", "BUS 100", "ECO 201", "ECO 202", "ENG 225", "FIN 201", "ISA 201",
    "MGT 201", "MGT 360", "MGT 406", "MKT 201", "QBA 201", "SCM 202",
    "ACC 201", "ACC 202",
]

# Innovation & Entrepreneurship — 3 cr, required.
INNOVATION = ["IEN 301"]

# Major Requirements — 18 cr; each entry is an OR-group (take any one), so the two
# "ISA 301 or CMP 320" / "ISA 303 or COE 420" catalogue choices are encoded faithfully.
MAJOR_REQUIREMENT_GROUPS = [
    ["ISA 301", "CMP 320"],
    ["ISA 303", "COE 420"],
    ["ISA 377"],
    ["ISA 388"],
    ["ISA 405"],
    ["ISA 497"],
]

# Major Electives — >= 12 cr from this named pool, PLUS any 300-level-or-above ISA
# course not listed as a major requirement (those ISA electives are added at build
# time from the scoped ISA courses). MGT 380 / EGM 362 are a single "or" pair in
# the catalogue; for an electives credit-minimum both simply count, so both are listed.
MAJOR_ELECTIVE_POOL = [
    "ACC 380", "ECO 351", "ECO 452", "FIN 375", "MGT 315", "MGT 380", "EGM 362",
    "MKT 302", "MKT 360", "SCM 310", "SCM 311", "STA 401", "UPL 302",
]

# Named courses that are NOT business-major subjects and so were not in the
# original AUS subject scrape; they are real catalogue courses in the lists above
# and are extracted specifically (not fabricated).
NON_BUSINESS_NAMED = ["BUS 100", "ENG 225", "IEN 301"]

# Credit structure (verbatim from the catalogue).
CREDITS = {
    "general_education": 36,
    "innovation_entrepreneurship": 3,
    "business_core": 45,
    "major_requirements": 18,
    "major_electives": 12,
    "free_electives": 9,
    "native_total": 123,        # full degree (incl. gen-ed + free electives)
    "academic_scope_total": 78,  # 45 + 3 + 18 + 12 (modelled / in-scope)
}


def all_block_codes() -> list[str]:
    """Every course code named in the academic blocks (flattened), de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for code in (BUSINESS_CORE + INNOVATION
                 + [c for grp in MAJOR_REQUIREMENT_GROUPS for c in grp]
                 + MAJOR_ELECTIVE_POOL):
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out
