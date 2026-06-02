"""Unit tests for the CAMP constraint engine, on the v3_3inst benchmark.

Runnable two ways:
  * ``.venv/bin/python -m pytest src/constraint_engine/tests/test_engine.py``
  * ``.venv/bin/python src/constraint_engine/tests/test_engine.py`` (plain asserts)

The second form prints a ``passed/total`` summary and exits non-zero on any
failure. The engine is built for the Khalifa (Computer Science) programme of the
three-institution CAP-Bench bundle; the tests are structural and rely only on
that programme's CORE / SUPPORT / ELECTIVES reconstruction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a plain script (add repo root to the path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.constraint_engine import dag  # noqa: E402
from src.constraint_engine.engine import ConstraintEngine, StudentState  # noqa: E402

PROGRAMME_ID = "KHAL_BSC_CS_v2024"
_PREFIX = "KHAL"
_INTER = _REPO_ROOT / "data" / "intermediate" / "khalifa"
_BUNDLE_INTER = _REPO_ROOT / "data" / "cap_bench" / "v3_3inst" / "intermediate"


def _load(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _engine() -> ConstraintEngine:
    """Build the Khalifa engine from the v3_3inst intermediate bundle."""
    courses = _load(_INTER / "khalifa_courses.json")
    graph, _ = dag.build_real_inst_dag(courses, 3)
    return ConstraintEngine(
        courses=courses,
        blocks=[b for b in _load(_BUNDLE_INTER / "blocks.json") if b["programme_id"] == PROGRAMME_ID],
        rule_groups=[g for g in _load(_BUNDLE_INTER / "rule_groups.json")
                     if g["block_id"].startswith(f"{_PREFIX}_BLOCK")],
        rules=[r for r in _load(_BUNDLE_INTER / "rules.json")
               if r["rule_id"].startswith(f"{_PREFIX}_RULE")],
        rule_eligible_courses=[e for e in _load(_BUNDLE_INTER / "rule_eligible_courses.json")
                               if e["rule_id"].startswith(f"{_PREFIX}_RULE")],
        programmes=[p for p in _load(_BUNDLE_INTER / "programmes.json")
                    if p["programme_id"] == PROGRAMME_ID],
        graph=graph,
    )


def test_prerequisite_blocks_and_unlocks() -> None:
    """A course with an unmet prereq is not eligible; once met, it is."""
    eng = _engine()
    course_with_prereq = next(cid for cid in eng.courses_by_id if eng.prereqs.get(cid))
    prereq = eng.prereqs[course_with_prereq][0][0]

    state = StudentState(programme_id=PROGRAMME_ID)
    assert course_with_prereq not in eng.eligible_courses(state)
    assert not eng.check_prerequisites(course_with_prereq, set())

    state.passed[prereq] = 3.0
    assert eng.check_prerequisites(course_with_prereq, {prereq})
    assert course_with_prereq in eng.eligible_courses(state)


def test_per_term_credit_cap_respected() -> None:
    """Once the per-term credit budget is exhausted, no further course fits."""
    eng = _engine()
    state = StudentState(programme_id=PROGRAMME_ID)  # generous default cap
    eligible = eng.eligible_courses(state)
    assert eligible, "expected some courses eligible on an empty transcript"
    # Tighten the budget to exactly the smallest eligible course; once it is
    # selected, no further course can fit.
    smallest = min(eligible, key=lambda c: eng.credits_of(c))
    state.per_term_credit_cap = eng.credits_of(smallest)
    state.term_selected.add(smallest)
    assert eng.eligible_courses(state) == set()


def test_failed_course_is_eligible_again() -> None:
    """A failed course is offered again as a retake; a passed one is not."""
    eng = _engine()
    root = next(cid for cid in eng.courses_by_id if not eng.prereqs.get(cid))
    state = StudentState(programme_id=PROGRAMME_ID, failed={root})
    assert root in eng.eligible_courses(state)
    state.passed[root] = 2.0
    assert root not in eng.eligible_courses(state)


def test_is_graduated_empty_and_complete() -> None:
    """False on an empty transcript; True on a hand-built complete one."""
    eng = _engine()
    assert not eng.is_graduated(StudentState(programme_id=PROGRAMME_ID))

    passed = _build_complete_transcript(eng)
    complete = StudentState(programme_id=PROGRAMME_ID, passed=passed)
    assert eng.passed_credits(complete) >= eng.total_credits_required(PROGRAMME_ID)
    assert eng.is_graduated(complete)


def test_evaluate_rules_min_courses() -> None:
    """The CORE min_courses rule: unsatisfied empty, satisfied once its courses pass."""
    eng = _engine()
    rule_id = f"{_PREFIX}_RULE_CORE"
    assert eng.evaluate_rules(set())[rule_id] is False
    core_courses = {cid for cid, _ in eng._rule_eligible_set(eng.rules_by_id[rule_id])}
    assert eng.evaluate_rules(core_courses)[rule_id] is True


def test_evaluate_rules_min_credits() -> None:
    """The ELECTIVES min_credits rule needs enough credits from its eligible list."""
    eng = _engine()
    rule = eng.rules_by_id[f"{_PREFIX}_RULE_ELECTIVES"]
    need = rule["min_credits"]
    eligible = [c for c, _ in eng._rule_eligible_set(rule)]
    enough, have = set(), 0
    for cid in eligible:
        if have >= need:
            break
        enough.add(cid)
        have += eng.credits_of(cid)
    assert eng.evaluate_rules(enough)[rule["rule_id"]] is True
    short = set(list(enough)[:-1])
    if sum(eng.credits_of(c) for c in short) < need:
        assert eng.evaluate_rules(short)[rule["rule_id"]] is False


def test_evaluate_rules_attribute_filter() -> None:
    """A synthetic attribute_match rule counts credits by attribute filter."""
    eng = _engine()
    rule = {
        "rule_id": "_TEST_ATTR", "rule_type": "min_credits", "min_credits": 6,
        "selector_type": "attribute_match", "attribute_filter": "level>=300",
    }
    high = [cid for cid, c in eng.courses_by_id.items() if c["level"] >= 300]
    low = [cid for cid, c in eng.courses_by_id.items() if c["level"] < 300]
    assert eng._rule_satisfied(rule, set(high[:3])) is True   # >= 6 credits at level>=300
    assert eng._rule_satisfied(rule, set(low[:3])) is False   # none match the filter


def test_prerequisite_violations_counts() -> None:
    """Taking a course before its prerequisite is counted as a violation."""
    eng = _engine()
    # A course with a single one-member prereq group whose member is itself a
    # root (no prereqs) — so the two-term plans isolate exactly one violation.
    course, prereq = next(
        (cid, eng.prereqs[cid][0][0])
        for cid in eng.courses_by_id
        if len(eng.prereqs.get(cid, [])) == 1
        and len(eng.prereqs[cid][0]) == 1
        and not eng.prereqs.get(eng.prereqs[cid][0][0])
    )
    assert eng.prerequisite_violations([[course], [prereq]]) == 1   # dependent before prereq
    assert eng.prerequisite_violations([[prereq], [course]]) == 0   # correct order


def _build_complete_transcript(eng: ConstraintEngine) -> dict[str, float]:
    """Hand-build a passed-course set satisfying every required block (grades 3.0)."""
    passed: set[str] = set()
    for rid in (f"{_PREFIX}_RULE_CORE", f"{_PREFIX}_RULE_SUPPORT"):
        for cid, _ in eng._rule_eligible_set(eng.rules_by_id[rid]):
            passed.add(cid)

    elec = eng.rules_by_id[f"{_PREFIX}_RULE_ELECTIVES"]
    need = elec.get("min_credits") or 0
    have = sum(eng.credits_of(c) for c, _ in eng._rule_eligible_set(elec) if c in passed)
    for cid, _ in eng._rule_eligible_set(elec):
        if have >= need:
            break
        if cid not in passed:
            passed.add(cid)
            have += eng.credits_of(cid)

    total_needed = eng.total_credits_required(PROGRAMME_ID)
    for cid in sorted(eng.courses_by_id):
        if sum(eng.credits_of(c) for c in passed) >= total_needed:
            break
        passed.add(cid)
    return {c: 3.0 for c in passed}


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  PASS {t.__name__}")
    print(f"{passed}/{len(tests)} engine tests passed")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
