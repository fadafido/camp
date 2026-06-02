"""Realistic graduation-threshold construction for the four real institutions
(Phase MERGE-4, Part 1).

The ingested ``total_credits_required`` was the *scoped credit pool* (the sum of
every scoped course). Requiring all of it would force near-identical paths and
collapse the recommendation task. This module reconstructs a realistic,
graduatable threshold per institution and emits programme-side entities
(programme + blocks + rule groups + rules + rule-eligible courses) that REPLACE
the pool-based ones in the merged bundle.

Reconstruction (documented as such in the datasheet):

* **CORE** (block_type ``major_core``) — the major-subject *gateway* courses
  (courses that are a prerequisite of at least one other scoped course, i.e. the
  curricular backbone you cannot skip). ``min_courses`` = all of them.
* **SUPPORT** (``breadth``) — the support-subject gateway courses. ``min_courses``
  = all of them.
* **ELECTIVES** (``free_elective``) — a ``min_credits`` rule over the *terminal*
  courses (no dependents). The minimum is tuned so that
  required-gateway-credits + elective-minimum ≈ the native degree total, while
  leaving ≥ 40 % of the terminal pool as un-required buffer — so students choose
  *different* elective subsets and graduate without taking the whole catalogue.

``total_credits_required`` = gateway credits + elective minimum, which for all
three institutions equals the native degree total. (If a scoped subset were ever
too small to reach its native total — the native figure includes general
education outside the scoped subset — the scoped total is set below the pool and
flagged; this does not occur for the three benchmark programmes.)

A student graduates when CORE + SUPPORT + the elective credit minimum are all
met and total passed credits ≥ ``total_credits_required``.

Module: AI503. British English. Deterministic.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .prereq_parser import PREFIX

# Native published full-degree totals (incl. general education outside scope).
NATIVE_TOTAL = {"khalifa": 130, "aus": 120, "unc": 120}
# Field-appropriate elective tracks (recorded per student; soft elective bias).
TRACKS = {
    "khalifa": ["ai", "cybersecurity", "general"],
    "aus": ["business_analytics", "information_systems", "general"],
    "unc": ["quantitative", "policy", "general"],
}
TRACK_KEYWORDS = {
    "ai": ["intelligence", "machine learning", "neural", "data", "vision", "language"],
    "cybersecurity": ["security", "cryptograph", "forensic", "network"],
    "business_analytics": ["analytics", "visualization", "data", "mining"],
    "information_systems": ["systems", "database", "e-commerce", "strategy"],
    "quantitative": ["econometric", "data", "game", "financial", "mathematic"],
    "policy": ["policy", "public", "international", "development", "labor", "environment"],
    "analytics": ["analytics", "data", "visualization", "business intelligence"],
    "general": [],
}


def build_realistic_programme(
    university: str,
    programme_id: str,
    programme_name: str,
    primary_department: str,
    home_subject: str,
    courses: list[dict[str, Any]],
    graph: nx.DiGraph,
) -> dict[str, Any]:
    """Return realistic programme-side entities + a meta summary."""
    up = PREFIX[university]
    by_id = {c["course_id"]: c for c in courses}
    native = NATIVE_TOTAL[university]

    # Gateway = is a prerequisite of >=1 other course (has an incoming edge under
    # the course -> prereq convention).
    def is_gateway(cid: str) -> bool:
        return graph.in_degree(cid) > 0

    home_gate = sorted(c["course_id"] for c in courses
                       if c["subject_area"] == home_subject and is_gateway(c["course_id"]))
    supp_gate = sorted(c["course_id"] for c in courses
                       if c["subject_area"] != home_subject and is_gateway(c["course_id"]))
    terminals = sorted(c["course_id"] for c in courses if not is_gateway(c["course_id"]))

    cr = lambda ids: sum(by_id[i]["credits"] for i in ids)
    home_cr, supp_cr, term_cr = cr(home_gate), cr(supp_gate), cr(terminals)
    gateway_cr = home_cr + supp_cr

    # Elective minimum: enough to reach the native total, but never more than
    # 60 % of the terminal pool (guarantees >=40 % buffer for path diversity).
    elective_min = min(max(native - gateway_cr, 0), round(0.6 * term_cr))
    if elective_min == 0:  # gateways already exceed native — keep a real buffer
        elective_min = round(0.5 * term_cr)
    total = gateway_cr + elective_min

    programme = {
        "programme_id": programme_id, "university": university, "name": programme_name,
        "version": "2024", "effective_from_term": "2024FA", "effective_to_term": None,
        "total_credits_required": total, "min_gpa_required": 2.0,
        "max_terms_to_complete": 14, "primary_department": primary_department,
        "native_total_credits": native,
        "graduation_threshold_note": (
            "Reconstructed realistic threshold (Phase MERGE-4): CORE = major-subject "
            "gateway courses (all required), SUPPORT = support-subject gateway courses "
            "(all required), ELECTIVES = min_credits over terminal courses tuned so "
            "gateways + elective-minimum = total_credits_required, leaving >=40% of the "
            "terminal pool as un-required electives. "
            + ("total equals the native degree total." if total == native else
               f"native total {native} is unreachable from the scoped subset "
               f"(pool {gateway_cr + term_cr} cr incl. no general education); scoped "
               f"total set to {total}.")),
    }

    def block(idx, bid, name, btype, mincr):
        return {"block_id": bid, "programme_id": programme_id, "block_index": idx,
                "block_name": name, "block_type": btype, "is_required": True,
                "scope": "programme", "owner_university_id": None,
                "min_credits_in_block": mincr, "description": name}

    B_CORE, B_SUPP, B_ELEC = f"{up}_BLOCK_CORE", f"{up}_BLOCK_SUPPORT", f"{up}_BLOCK_ELECTIVES"
    blocks = [
        block(1, B_CORE, f"{home_subject} Major Core (required gateway courses)", "major_core", home_cr),
        block(2, B_SUPP, "Required Support Courses", "breadth", supp_cr),
        block(3, B_ELEC, f"Electives ({elective_min} credits from the elective pool)", "free_elective", elective_min),
    ]

    def rg(bid, name):
        return {"rule_group_id": f"{bid}_RG", "block_id": bid, "group_index": 1,
                "group_name": name, "is_required": True, "description": name}

    rule_groups = [rg(B_CORE, f"{home_subject} core"), rg(B_SUPP, "Support core"),
                   rg(B_ELEC, "Elective credits")]

    rules = [
        {"rule_id": f"{up}_RULE_CORE", "rule_group_id": f"{B_CORE}_RG", "rule_index": 1,
         "rule_name": f"All {home_subject} gateway courses", "rule_type": "min_courses",
         "min_courses": len(home_gate), "min_credits": None, "min_grade_each": 1.0,
         "selector_type": "explicit_list", "attribute_filter": None,
         "target_block_type": None, "target_block_id": None, "member_group_ids": None},
        {"rule_id": f"{up}_RULE_SUPPORT", "rule_group_id": f"{B_SUPP}_RG", "rule_index": 1,
         "rule_name": "All support gateway courses", "rule_type": "min_courses",
         "min_courses": len(supp_gate), "min_credits": None, "min_grade_each": 1.0,
         "selector_type": "explicit_list", "attribute_filter": None,
         "target_block_type": None, "target_block_id": None, "member_group_ids": None},
        {"rule_id": f"{up}_RULE_ELECTIVES", "rule_group_id": f"{B_ELEC}_RG", "rule_index": 1,
         "rule_name": f"{elective_min} elective credits", "rule_type": "min_credits",
         "min_courses": None, "min_credits": elective_min, "min_grade_each": 1.0,
         "selector_type": "explicit_list", "attribute_filter": None,
         "target_block_type": None, "target_block_id": None, "member_group_ids": None},
    ]

    rule_eligible = (
        [{"rule_id": f"{up}_RULE_CORE", "course_id": c, "weight_credits": None} for c in home_gate]
        + [{"rule_id": f"{up}_RULE_SUPPORT", "course_id": c, "weight_credits": None} for c in supp_gate]
        + [{"rule_id": f"{up}_RULE_ELECTIVES", "course_id": c, "weight_credits": None} for c in terminals]
    )

    meta = {
        "native_total": native, "scoped_total": total,
        "core_gateway_courses": len(home_gate), "core_gateway_credits": home_cr,
        "support_gateway_courses": len(supp_gate), "support_gateway_credits": supp_cr,
        "required_gateway_credits": gateway_cr,
        "elective_min_credits": elective_min,
        "elective_pool_courses": len(terminals), "elective_pool_credits": term_cr,
        "elective_buffer_credits": term_cr - elective_min,
        "uses_native_total": total == native,
    }
    return {"programmes": [programme], "blocks": blocks, "rule_groups": rule_groups,
            "rules": rules, "rule_eligible_courses": rule_eligible, "meta": meta}
