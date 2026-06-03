"""Shared ingestion for the three real-catalogue institutions (Phase INGEST-4).

Sources arrive as ``extracted_courses.json`` + ``extracted_program.json``
(parsed from public course catalogues). The per-institution scripts
(``khalifa_ingest.py``, ``aus_ingest.py``, ``unc_ingest.py``) supply an
:class:`InstConfig` and call :func:`ingest`.

What this module does, per institution:

1. **Scope** the extracted courses by the locked Phase INGEST-4 rule
   (undergraduate-only major subject + referenced support subjects).
2. **Normalise** every course into a CAP-Bench v2.1 catalogue-course record
   (integer credits, hundreds/thousands ``level`` band, ``subject_area`` set to
   the subject code per the phase brief), parsing ``prereq_raw`` into the
   structured ``prerequisites`` AND-of-OR form via
   :mod:`src.dataset.prereq_parser`. Non-course tokens go to ``prereq_notes``;
   out-of-scope references are dropped and counted; unparseable strings are
   logged.
3. **Reconstruct** a faithful programme structure: three programme-scope blocks
   — major CORE (referenced home-subject courses), major ELECTIVES (remaining
   home-subject courses), and SUPPORT (the scoped non-home courses). Block
   ``min_credits`` are the summed credits of their courses, so the blocks sum to
   the normalised ``total_credits_required`` by construction. This is a
   documented reconstruction: the source programme pages do not expose exact
   block credit splits for these majors.

Credit normalisation: every institution already publishes semester-credit-
equivalent values; ``credits`` is the published integer (Khalifa/AUS third
number of the L-L-C triple; UNC/ISU the parsed integer). The programme's native
full-degree total is recorded separately from the normalised scoped total used
by the engine.

Module: AI503. British English. Deterministic; seed 42 governs project
randomness but nothing here draws random numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .prereq_parser import PREFIX, make_course_id, parse_prereq

_REPO_ROOT = Path(__file__).resolve().parents[2]

# subject_area is set to the subject code per the phase brief; difficulty is
# assigned from the level band as documented in the schema (§5.2).
_DIFFICULTY_BY_BAND = {0: 0.3, 100: 0.3, 200: 0.5, 300: 0.65, 400: 0.8,
                       1000: 0.3, 2000: 0.5, 3000: 0.65, 4000: 0.8, 5000: 0.8, 6000: 0.8}


@dataclass
class InstConfig:
    university: str                       # "khalifa" | "aus" | "unc"
    programme_id: str
    programme_name: str
    primary_department: str               # the major subject code
    home_subjects: set[str]               # subjects scoped "whole" (undergrad)
    major_subject: str                    # the single subject used for CORE/ELECTIVES
    support_subjects: set[str]            # scoped only if referenced
    home_max_number: int                  # undergraduate ceiling on the raw number
    digits: int                           # 3 (level //100*100) or 4 (//1000*1000)
    native_total_credits: int
    native_total_credits_note: str
    keep_any_referenced: bool = False     # AUS keeps any referenced course
    supplement_file: str | None = None    # extra raw courses merged before scoping
    version: str = "2024"
    min_gpa_required: float = 2.0
    max_terms_to_complete: int = 12
    effective_from_term: str = "2024FA"


def _num(code: str) -> int:
    m = re.search(r"(\d+)", code or "")
    return int(m.group(1)) if m else -1


def _split_code(code: str) -> tuple[str, str]:
    m = re.match(r"\s*([A-Z]{2,5})\s*(\d{2,4}[A-Z]?)", code)
    return (m.group(1), m.group(2)) if m else ("", "")


def _norm_ref(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def _level(number_str: str, digits: int) -> int:
    n = _num(number_str)
    if n < 0:
        return 0
    return (n // 1000) * 1000 if digits == 4 else (n // 100) * 100


def _credits(raw: Any) -> int:
    """First integer in the credits field ('3', '3.', '1-3' -> 1)."""
    m = re.search(r"\d+", str(raw))
    return int(m.group(0)) if m else 0


def _load(inst: str, name: str) -> Any:
    path = _REPO_ROOT / "data" / "raw" / inst / f"{name}.json"
    with path.open() as fh:
        return json.load(fh)


def scope_courses(cfg: InstConfig, raw_courses: list[dict], referenced: set[str]) -> list[dict]:
    """Apply the locked scoping rule; return the kept raw extracted records."""
    kept = []
    for c in raw_courses:
        subj, _ = _split_code(c["code"])
        n = _num(c["code"])
        in_home = subj in cfg.home_subjects and n < cfg.home_max_number
        in_support = subj in cfg.support_subjects and _norm_ref(c["code"]) in referenced
        in_ref = cfg.keep_any_referenced and _norm_ref(c["code"]) in referenced
        if in_home or in_support or in_ref:
            kept.append(c)
    return kept


def build_courses(cfg: InstConfig, scoped_raw: list[dict], scoped_ids: set[str]) -> tuple[
        list[dict], dict[str, Any]]:
    """Normalise scoped courses to CAP-Bench records; parse prerequisites.

    Returns (course_records, stats) where stats carries out-of-set and unparsed
    diagnostics for the phase log.
    """
    by_id: dict[str, dict] = {}
    n_out_of_set = 0
    unparsed: list[str] = []
    for c in scoped_raw:
        subj, number = _split_code(c["code"])
        cid = make_course_id(subj, number, cfg.university)
        if cid in by_id:
            continue  # dedupe (e.g. a duplicate extracted row)
        level = _level(number, cfg.digits)
        parsed = parse_prereq(c.get("prereq_raw", ""), cfg.university, scoped_ids)
        n_out_of_set += parsed["n_out_of_set"]
        if parsed["unparsed"]:
            unparsed.append(cid)
        by_id[cid] = {
            "course_id": cid,
            "university": cfg.university,
            "code_local": f"{subj} {number}",
            "title": (c.get("title") or "").strip() or None,
            "description_summary": None,
            "credits": _credits(c.get("credits")),
            "level": level,
            "department": subj,
            "subject_area": subj,            # subject code, per the phase brief
            "attributes": [],
            "tags": [],
            "difficulty_default": _DIFFICULTY_BY_BAND.get(level, 0.5),
            "is_repeatable": False,
            "is_active": True,
            "effective_from_term": cfg.effective_from_term,
            "effective_to_term": None,
            "prerequisites": parsed["prerequisites"],
            "prereq_notes": parsed["prereq_notes"],
        }
    stats = {
        "n_out_of_set_refs_dropped": n_out_of_set,
        "unparsed_prereqs": unparsed,
        "n_with_prereq": sum(1 for c in by_id.values() if c["prerequisites"]),
    }
    return list(by_id.values()), stats


def build_programme_side(cfg: InstConfig, courses: list[dict], referenced: set[str]) -> dict:
    """Reconstruct programme + blocks + rule groups + rules + eligible courses.

    Three programme-scope blocks (CORE / ELECTIVES / SUPPORT). Block credits are
    the summed credits of their member courses, so they sum to the normalised
    total. Documented reconstruction (see module docstring).
    """
    uni_prefix = PREFIX[cfg.university]
    home = cfg.major_subject

    def is_referenced(c: dict) -> bool:
        return _norm_ref(c["code_local"]) in referenced

    core = sorted((c for c in courses if c["subject_area"] == home and is_referenced(c)),
                  key=lambda c: c["course_id"])
    electives = sorted((c for c in courses if c["subject_area"] == home and not is_referenced(c)),
                       key=lambda c: c["course_id"])
    support = sorted((c for c in courses if c["subject_area"] != home),
                     key=lambda c: c["course_id"])

    core_cr = sum(c["credits"] for c in core)
    elec_cr = sum(c["credits"] for c in electives)
    supp_cr = sum(c["credits"] for c in support)
    normalised_total = core_cr + elec_cr + supp_cr

    programme = {
        "programme_id": cfg.programme_id,
        "university": cfg.university,
        "name": cfg.programme_name,
        "version": cfg.version,
        "effective_from_term": cfg.effective_from_term,
        "effective_to_term": None,
        "total_credits_required": normalised_total,
        "min_gpa_required": cfg.min_gpa_required,
        "max_terms_to_complete": cfg.max_terms_to_complete,
        "primary_department": cfg.primary_department,
        # Documentation fields (intermediate-only; not part of the final schema).
        "native_total_credits": cfg.native_total_credits,
        "native_total_credits_note": cfg.native_total_credits_note,
        "credit_normalisation": (
            "Published semester-credit values carried through verbatim; "
            "normalised total_credits_required is the sum of the scoped blocks "
            "(major core + electives + support). The native full-degree total "
            "(incl. general education outside the scoped subset) is recorded in "
            "native_total_credits."),
        "block_structure_note": (
            "Faithful reconstruction: source programme pages do not expose exact "
            "block credit splits for this major, so CORE = referenced home-subject "
            "courses, ELECTIVES = remaining home-subject courses, SUPPORT = scoped "
            "non-home courses; each block's min_credits is the sum of its courses' "
            "credits."),
    }

    def block(idx, bid, name, btype, min_cr):
        return {
            "block_id": bid, "programme_id": cfg.programme_id, "block_index": idx,
            "block_name": name, "block_type": btype, "is_required": True,
            "scope": "programme", "owner_university_id": None,
            "min_credits_in_block": min_cr, "description": name,
        }

    B_CORE = f"{uni_prefix}_BLOCK_CORE"
    B_ELEC = f"{uni_prefix}_BLOCK_ELECTIVES"
    B_SUPP = f"{uni_prefix}_BLOCK_SUPPORT"
    blocks = [
        block(1, B_CORE, f"{home} Major Core", "major_core", core_cr),
        block(2, B_ELEC, f"{home} Major Electives", "free_elective", elec_cr),
        block(3, B_SUPP, "Required Support Courses", "breadth", supp_cr),
    ]

    def rg(bid, name):
        return {"rule_group_id": f"{bid}_RG", "block_id": bid, "group_index": 1,
                "group_name": name, "is_required": True, "description": name}

    rule_groups = [rg(B_CORE, f"{home} core requirements"),
                   rg(B_ELEC, f"{home} elective requirements"),
                   rg(B_SUPP, "Support requirements")]

    rules = [
        {  # CORE — take all referenced home-subject core courses
            "rule_id": f"{uni_prefix}_RULE_CORE", "rule_group_id": f"{B_CORE}_RG",
            "rule_index": 1, "rule_name": f"{home} core courses",
            "rule_type": "min_courses", "min_courses": len(core), "min_credits": None,
            "min_grade_each": 1.0, "selector_type": "explicit_list",
            "attribute_filter": None, "target_block_type": None, "target_block_id": None,
            "member_group_ids": None,
        },
        {  # ELECTIVES — min credits from any home-subject course (attribute match)
            "rule_id": f"{uni_prefix}_RULE_ELECTIVES", "rule_group_id": f"{B_ELEC}_RG",
            "rule_index": 1, "rule_name": f"{home} electives — {elec_cr} credits",
            "rule_type": "min_credits", "min_courses": None, "min_credits": elec_cr,
            "min_grade_each": 1.0, "selector_type": "attribute_match",
            "attribute_filter": f"subject_area=={home}", "target_block_type": None,
            "target_block_id": None, "member_group_ids": None,
        },
        {  # SUPPORT — take all scoped support courses
            "rule_id": f"{uni_prefix}_RULE_SUPPORT", "rule_group_id": f"{B_SUPP}_RG",
            "rule_index": 1, "rule_name": "Support courses",
            "rule_type": "min_courses", "min_courses": len(support), "min_credits": None,
            "min_grade_each": 1.0, "selector_type": "explicit_list",
            "attribute_filter": None, "target_block_type": None, "target_block_id": None,
            "member_group_ids": None,
        },
    ]

    rule_eligible = (
        [{"rule_id": f"{uni_prefix}_RULE_CORE", "course_id": c["course_id"],
          "weight_credits": None} for c in core]
        + [{"rule_id": f"{uni_prefix}_RULE_SUPPORT", "course_id": c["course_id"],
            "weight_credits": None} for c in support]
    )

    return {
        "programmes": [programme], "blocks": blocks, "rule_groups": rule_groups,
        "rules": rules, "rule_eligible_courses": rule_eligible,
        "_split": {"core": len(core), "electives": len(electives), "support": len(support),
                   "normalised_total": normalised_total},
    }


def validate(entities: dict[str, list], cfg: InstConfig) -> list[str]:
    """Referential-integrity + uniqueness checks (mirrors auc_ingest.validate)."""
    failures: list[str] = []
    courses = {c["course_id"]: c for c in entities["courses"]}
    programmes = {p["programme_id"]: p for p in entities["programmes"]}
    blocks = {b["block_id"]: b for b in entities["blocks"]}
    rule_groups = {rg["rule_group_id"]: rg for rg in entities["rule_groups"]}
    rules = {r["rule_id"]: r for r in entities["rules"]}

    for name, key, lst in [
        ("course_id", courses, entities["courses"]),
        ("programme_id", programmes, entities["programmes"]),
        ("block_id", blocks, entities["blocks"]),
        ("rule_group_id", rule_groups, entities["rule_groups"]),
        ("rule_id", rules, entities["rules"]),
    ]:
        if len(key) != len(lst):
            failures.append(f"Duplicate {name}")

    for b in entities["blocks"]:
        if b["programme_id"] is not None and b["programme_id"] not in programmes:
            failures.append(f"Block {b['block_id']} -> missing programme {b['programme_id']}")
        if b["scope"] == "programme" and b["programme_id"] is None:
            failures.append(f"Block {b['block_id']} scope=programme but programme_id null")
    for rg in entities["rule_groups"]:
        if rg["block_id"] not in blocks:
            failures.append(f"RuleGroup {rg['rule_group_id']} -> missing block {rg['block_id']}")
    for r in entities["rules"]:
        if r["rule_group_id"] not in rule_groups:
            failures.append(f"Rule {r['rule_id']} -> missing rule_group {r['rule_group_id']}")
    for rec in entities["rule_eligible_courses"]:
        if rec["rule_id"] not in rules:
            failures.append(f"RuleEligibleCourse -> missing rule {rec['rule_id']}")
        if rec["course_id"] not in courses:
            failures.append(f"RuleEligibleCourse -> missing course {rec['course_id']}")

    # Every structured prerequisite must reference an in-set course.
    for c in entities["courses"]:
        for grp in c["prerequisites"]:
            for pid in grp:
                if pid not in courses:
                    failures.append(f"Course {c['course_id']} prereq -> missing course {pid}")

    # Blocks sum to the programme normalised total.
    credit_sum = sum(b["min_credits_in_block"] for b in entities["blocks"]
                     if b["min_credits_in_block"] is not None)
    total = entities["programmes"][0]["total_credits_required"]
    if credit_sum != total:
        failures.append(f"Block min_credits sum {credit_sum} != total {total}")
    return failures


def ingest(cfg: InstConfig) -> dict[str, Any]:
    """Full per-institution ingest. Writes six JSON files; returns stats."""
    raw_courses = _load(cfg.university, "extracted_courses")
    if cfg.supplement_file:
        # Additive supplement of real courses that the original gather missed but
        # the programme references but the original gather missed. The
        # protected extracted_courses.json is never modified.
        supp_path = _REPO_ROOT / "data" / "raw" / cfg.university / cfg.supplement_file
        if supp_path.exists():
            raw_courses = raw_courses + json.load(supp_path.open())
    programme_raw = _load(cfg.university, "extracted_program")
    referenced = {_norm_ref(c) for c in programme_raw.get("referenced_course_codes", [])}

    scoped_raw = scope_courses(cfg, raw_courses, referenced)
    # course_ids of the scoped set (needed so the parser can prune out-of-set refs)
    scoped_ids = {make_course_id(*_split_code(c["code"]), cfg.university)
                  for c in scoped_raw if _split_code(c["code"])[0]}

    courses, stats = build_courses(cfg, scoped_raw, scoped_ids)
    prog_side = build_programme_side(cfg, courses, referenced)

    entities = {
        "programmes": prog_side["programmes"],
        "blocks": prog_side["blocks"],
        "rule_groups": prog_side["rule_groups"],
        "rules": prog_side["rules"],
        "rule_eligible_courses": prog_side["rule_eligible_courses"],
        "courses": courses,
    }
    entities["courses"].sort(key=lambda x: x["course_id"])
    entities["rule_eligible_courses"].sort(key=lambda x: (x["rule_id"], x["course_id"]))

    failures = validate(entities, cfg)

    out_dir = _REPO_ROOT / "data" / "intermediate" / cfg.university
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "programmes": f"{cfg.university}_programmes.json",
        "blocks": f"{cfg.university}_blocks.json",
        "rule_groups": f"{cfg.university}_rule_groups.json",
        "rules": f"{cfg.university}_rules.json",
        "rule_eligible_courses": f"{cfg.university}_rule_eligible_courses.json",
        "courses": f"{cfg.university}_courses.json",
    }
    for key, fname in files.items():
        with (out_dir / fname).open("w") as fh:
            json.dump(entities[key], fh, indent=2, ensure_ascii=False)

    print(f"[{cfg.university}] courses={len(courses)} "
          f"(core={prog_side['_split']['core']}, electives={prog_side['_split']['electives']}, "
          f"support={prog_side['_split']['support']})  "
          f"with_prereq={stats['n_with_prereq']}  "
          f"out_of_set_dropped={stats['n_out_of_set_refs_dropped']}  "
          f"unparsed={len(stats['unparsed_prereqs'])}  "
          f"normalised_total={prog_side['_split']['normalised_total']}cr "
          f"native={cfg.native_total_credits}cr")
    if stats["unparsed_prereqs"]:
        print(f"  unparsed prereq course_ids: {stats['unparsed_prereqs']}")
    if failures:
        print(f"[{cfg.university}] validation: FAIL")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"[{cfg.university}] validation: PASS")
    return stats
