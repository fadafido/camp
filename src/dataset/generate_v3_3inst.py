"""Generate the three-institution mixed-field CAP-Bench bundle (`v3_3inst`).

Builds the published benchmark from the three ingested real-catalogue
institutions — Khalifa University (BSc Computer Science), American University of
Sharjah (BSBA Information Systems and Business Analytics), and UNC Chapel Hill
(BA/BS Economics):

  1. Per institution: build the hybrid prerequisite DAG, reconstruct a realistic
     graduation threshold (see :mod:`grad_thresholds`), and compute curricular
     features.
  2. Merge all three into ``data/cap_bench/v3_3inst/intermediate/`` (ID-collision
     guarded).
  3. Simulate 1,500 synthetic students per institution (4,500 total) with the
     realistic start-state / term-load mix, each against its own programme + DAG
     + constraint engine.
  4. Write students, flat enrolments, stratified splits, statistics and datasheet.

Each student belongs to ONE institution and is simulated only against that
institution's programme (institution-masking carries into the model phases — no
cross-institution recommendations). The catalogues are real and public; the
students are synthetic and fully reproducible from seed 42. British English.
CPU-only.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.constraint_engine import dag
from src.constraint_engine.engine import ConstraintEngine
from src.dataset import grad_thresholds as GT
from src.dataset.simulator import InstitutionConfig, MultiSimulator, write_students_and_enrolments
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("camp.generate_v3")

_REPO = Path(__file__).resolve().parents[2]
INTER = _REPO / "data" / "intermediate"
V3 = _REPO / "data" / "cap_bench" / "v3_3inst"
V3_INTER = V3 / "intermediate"
V3_STUDENTS = V3 / "students"

N_PER_INSTITUTION = {"khalifa": 1500, "aus": 1500, "unc": 1500}

# (university, digits, level_scale, programme_id, programme_name, primary_dept, home_subject)
INSTS = [
    ("khalifa", 3, 100, "KHAL_BSC_CS_v2024", "Bachelor of Science in Computer Science", "COSC", "COSC"),
    ("aus", 3, 100, "AUS_BSBA_IS_v2024",
     "Bachelor of Science in Business Administration, Major in Information Systems and Business Analytics", "ISA", "ISA"),
    ("unc", 3, 100, "UNC_BS_ECON_v2024", "Bachelor of Science in Economics", "ECON", "ECON"),
]

_FIELD_NAME = {"khalifa": "Computer Science", "aus": "Information Systems", "unc": "Economics"}


def _load(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def build_all() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build per-institution DAGs + realistic programme entities; merge to v3_3inst."""
    V3_INTER.mkdir(parents=True, exist_ok=True)
    merged = {k: [] for k in ("programmes.json", "blocks.json", "rule_groups.json",
                              "rules.json", "rule_eligible_courses.json", "courses.json",
                              "course_features.json")}
    per_inst: dict[str, Any] = {}
    graphs: dict[str, Any] = {}

    for uni, digits, _scale, pid, pname, dept, home in INSTS:
        courses = _load(INTER / uni / f"{uni}_courses.json")
        graph, _ = dag.build_real_inst_dag(courses, digits)
        graphs[uni] = graph
        prog = GT.build_realistic_programme(uni, pid, pname, dept, home, courses, graph)

        merged["programmes.json"] += prog["programmes"]
        merged["blocks.json"] += prog["blocks"]
        merged["rule_groups.json"] += prog["rule_groups"]
        merged["rules.json"] += prog["rules"]
        merged["rule_eligible_courses.json"] += prog["rule_eligible_courses"]
        merged["courses.json"] += courses
        feats = dag.compute_course_features(graph, courses)
        merged["course_features.json"] += [{**f, "university": uni} for f in feats.values()]
        per_inst[uni] = {**prog["meta"], "dag": dag.dag_summary(graph)}

    id_keys = {"programmes.json": "programme_id", "blocks.json": "block_id",
               "rule_groups.json": "rule_group_id", "rules.json": "rule_id",
               "courses.json": "course_id"}
    collisions = {}
    for fname, key in id_keys.items():
        dupes = [i for i, n in Counter(r[key] for r in merged[fname]).items() if n > 1]
        if dupes:
            collisions[fname] = dupes
    if collisions:
        raise RuntimeError(f"ID collisions in merge (STOP): {collisions}")
    bad = [c["course_id"] for c in merged["courses.json"]
           if c.get("university") not in N_PER_INSTITUTION]
    if bad:
        raise RuntimeError(f"Courses missing/invalid university (STOP): {bad[:5]}")

    for fname, rows in merged.items():
        with (V3_INTER / fname).open("w") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)

    summary = {
        "programmes": len(merged["programmes.json"]),
        "courses": len(merged["courses.json"]),
        "courses_per_institution": {u: sum(1 for c in merged["courses.json"] if c["university"] == u)
                                    for u in N_PER_INSTITUTION},
        "id_collisions": 0,
        "per_institution": per_inst,
    }
    return summary, graphs


def build_configs(graphs: dict[str, Any]) -> list[InstitutionConfig]:
    configs = []
    feats_all = _load(V3_INTER / "course_features.json")
    for uni, digits, scale, pid, pname, dept, home in INSTS:
        courses = _load(INTER / uni / f"{uni}_courses.json")
        engine = ConstraintEngine(
            courses=courses,
            blocks=[b for b in _load(V3_INTER / "blocks.json") if b["programme_id"] == pid],
            rule_groups=[g for g in _load(V3_INTER / "rule_groups.json")
                         if g["block_id"].startswith(f"{GT.PREFIX[uni]}_BLOCK")],
            rules=[r for r in _load(V3_INTER / "rules.json")
                   if r["rule_id"].startswith(f"{GT.PREFIX[uni]}_RULE")],
            rule_eligible_courses=[e for e in _load(V3_INTER / "rule_eligible_courses.json")
                                   if e["rule_id"].startswith(f"{GT.PREFIX[uni]}_RULE")],
            programmes=[p for p in _load(V3_INTER / "programmes.json") if p["programme_id"] == pid],
            graph=graphs[uni],
        )
        feats = {f["course_id"]: f for f in feats_all if f["university"] == uni}
        configs.append(InstitutionConfig(
            university=uni, programme_id=pid, engine=engine, features=feats,
            tracks=GT.TRACKS[uni], level_scale=scale,
            track_keywords={t: GT.TRACK_KEYWORDS.get(t, []) for t in GT.TRACKS[uni]},
        ))
    return configs


def _dist(values) -> dict[str, int]:
    return dict(Counter(values))


def compute_stats(students, enrolments, merge_summary) -> dict[str, Any]:
    term_sizes = [len(t["courses"]) for s in students for t in s["terms"]]
    arr = np.array(term_sizes, dtype=float)
    graded = [e for e in enrolments if not e["is_transfer"]]

    def block(rows, gr_rows):
        n = len(rows)
        grad = [r for r in rows if r["graduated"]]
        gpas = [r["final_gpa"] for r in rows]
        gterms = [r["terms_to_graduate"] for r in rows if r["terms_to_graduate"]]
        return {
            "n_students": n,
            "graduation_rate": round(len(grad) / n, 4) if n else 0.0,
            "mean_final_gpa": round(float(np.mean(gpas)), 4) if gpas else 0.0,
            "mean_terms_to_graduate": round(float(np.mean(gterms)), 4) if gterms else None,
            "population_pass_rate": round(sum(1 for e in gr_rows if e["passed"]) / len(gr_rows), 4) if gr_rows else 0.0,
        }

    per_inst = {}
    for uni in N_PER_INSTITUTION:
        rows = [s for s in students if s["university"] == uni]
        gr = [e for e in graded if e["university"] == uni]
        sets = [frozenset(e["course_id"] for e in enrolments if e["student_id"] == s["student_id"]) for s in rows[:200]]
        per_inst[uni] = {**block(rows, gr),
                         "distinct_transcripts_sample": len(set(sets)),
                         "sample_size": len(sets)}

    return {
        "n_students": len(students),
        "n_students_per_institution": dict(N_PER_INSTITUTION),
        "overall": {**block(students, graded), "total_enrolment_rows": len(enrolments),
                    "graded_enrolment_rows": len(graded)},
        "per_institution": per_inst,
        "courses_per_term": {
            "mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
            "min": int(arr.min()), "max": int(arr.max()),
            "histogram": {str(k): int(v) for k, v in sorted(Counter(term_sizes).items())},
        },
        "start_state_mix": _dist(s["start_state_type"] for s in students),
        "term_load_mix": _dist(s["term_load_type"] for s in students),
        "demographics": {
            "gender": _dist(s["demographics"]["gender"] for s in students),
            "gpa_band": _dist(s["demographics"]["gpa_band"] for s in students),
            "chosen_track": _dist(s["chosen_track"] for s in students),
        },
        "merge": merge_summary,
    }


def write_datasheet(stats: dict[str, Any]) -> None:
    cpt = stats["courses_per_term"]
    pi = stats["merge"]["per_institution"]
    rows = []
    for uni in stats["merge"]["courses_per_institution"]:
        m = pi[uni]
        rows.append(f"| {uni} | {_FIELD_NAME[uni]} | {stats['merge']['courses_per_institution'][uni]} | "
                    f"{m['scoped_total']} (native {m['native_total']}) | "
                    f"{m['core_gateway_courses']}+{m['support_gateway_courses']} gateway / "
                    f"{m['elective_min_credits']}cr from {m['elective_pool_courses']} electives | "
                    f"{stats['per_institution'][uni]['graduation_rate']} |")
    table = "\n".join(rows)
    text = f"""# CAP-Bench — Three-institution mixed-field bundle · Datasheet

**Bundle ID:** `cap_bench_v3_3inst`
**Schema version:** v2.1
**Module:** AI503 Machine Learning Assignment 2 · BUiD MSc Artificial Intelligence

## Composition — three real public catalogues, one field each, synthetic students

| Institution | Field | Courses | Grad. credits | Required / electives | Grad. rate |
|---|---|---:|---|---|---:|
{table}

Totals: 3 programmes, {stats['merge']['courses']} courses,
{stats['n_students']} synthetic students, {stats['overall']['total_enrolment_rows']} enrolment rows.
All identifiers are institution-prefixed (`KHAL_*` / `AUS_*` / `UNC_*`); **zero ID
collisions**. Each student belongs to ONE institution and is simulated only
against that institution's programme, prerequisite DAG and constraint engine
(institution-masking carries into the model phases — no cross-institution
recommendations).

## Provenance — real catalogues, synthetic students

The benchmark is built from the **publicly published undergraduate course
catalogues** of three universities — Khalifa University (BSc Computer Science),
American University of Sharjah (BSBA Information Systems and Business Analytics),
and UNC Chapel Hill (BA/BS Economics). The catalogue-derived artefacts are the
programme/requirement structures and the course prerequisites parsed from those
public catalogues. **No real student records are used.** Every student
transcript is **synthetic**, generated by the constraint-aware simulator and
fully reproducible from seed 42.

## Graduation thresholds (reconstructed)

Each programme uses a reconstructed realistic threshold: **CORE** = the
major-subject gateway courses (prerequisites of other courses — the curricular
backbone, all required), **SUPPORT** = the support-subject gateway courses (all
required), and **ELECTIVES** = a credit minimum drawn from the terminal courses,
tuned so that gateways + elective-minimum = `total_credits_required` while
leaving a substantial elective pool un-required (so students pick different
electives). The native full-degree total (incl. general education outside the
scoped subset) is recorded per programme alongside the scoped graduation total.

## Credit normalisation

Published semester-credit values are carried through as integers.

## Prerequisite strategy (real + minimal augmentation, per institution)

Prerequisites are parsed from the real catalogues (free-text → AND-of-OR groups),
with minimal curriculum-informed augmentation only for courses left without any
prerequisite at level ≥ 200. Real edges dominate everywhere
({', '.join(f"{pi[u]['dag']['real_edges']}:{pi[u]['dag']['augmented_edges']} {_FIELD_NAME[u]}" for u in stats['merge']['courses_per_institution'])}).
All DAGs are acyclic.

## Realistic student mix

- **Start state** ({_fmt(stats['start_state_mix'])}): new / mid-degree (1–4 prior
  terms) / transfer (lower-level passed courses, excluded from GPA) / returning
  (a gap term + retakes).
- **Term load** ({_fmt(stats['term_load_mix'])}): full-time (4–5), part-time (2–3),
  overload (5–6), plus occasional summer terms (~30 % of students). Term sizes are
  non-uniform: courses-per-term mean {cpt['mean']}, **std {cpt['std']}**
  (range {cpt['min']}–{cpt['max']}).
- **Demographics:** gender, GPA band, cohort, chosen field track, cold-start flag,
  start-state type, term-load type, university — all recorded for fairness work.

Selection is a ~70/30 greedy/stochastic mix; grades are band-calibrated
(population pass rate {stats['overall']['population_pass_rate']}).

## Splits

Student-level 70/15/15 train/val/test, seed 42, stratified by university (equal
representation). Disjoint and complete.

## Simplifications

- Synthetic students, real catalogues. No real transcripts.
- Whole terms only; one section per course; no capacity constraints.
- Permission-of-instructor / placement-test / standing prerequisites treated as
  notes, not hard prerequisites.
- General education is outside the scoped subset; graduation thresholds cover the
  major + required support + a major-elective credit minimum.
"""
    (V3 / "datasheet.md").write_text(text)


def _fmt(d: dict[str, int]) -> str:
    total = sum(d.values()) or 1
    order = ["new", "mid_degree", "transfer", "returning", "full_time", "part_time", "overload"]
    items = sorted(d.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
    return ", ".join(f"{k} {round(100 * v / total)}%" for k, v in items)


def main() -> dict[str, Any]:
    set_seed(42)
    V3.mkdir(parents=True, exist_ok=True)
    logger.info("Building + merging three institutions ...")
    merge_summary, graphs = build_all()
    logger.info("Merge: %d programmes, %d courses, collisions=%d",
                merge_summary["programmes"], merge_summary["courses"], merge_summary["id_collisions"])

    configs = build_configs(graphs)
    simulator = MultiSimulator(configs)
    students, enrolments = simulator.generate(N_PER_INSTITUTION)
    write_students_and_enrolments(students, enrolments, V3_STUDENTS)

    stats = compute_stats(students, enrolments, merge_summary)
    with (V3 / "dataset_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    write_datasheet(stats)
    logger.info("Done: %d students, %d enrolments, courses/term std=%.3f, pass_rate=%.3f, grad=%.3f",
                stats["n_students"], stats["overall"]["total_enrolment_rows"],
                stats["courses_per_term"]["std"], stats["overall"]["population_pass_rate"],
                stats["overall"]["graduation_rate"])
    return stats


if __name__ == "__main__":
    main()
