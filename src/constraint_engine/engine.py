"""Constraint Engine — the constraint layer of the CAMP architecture.

A reusable, deterministic engine that enforces an institution's prerequisite
chains, the per-term credit cap, retake handling, and the full graduation rule
hierarchy (block -> rule group -> rule -> eligible courses). It is consumed by
both the synthetic-student simulator and the constraint-masked RL action space
(CAMP).

The engine is constructed directly from a programme's catalogue courses,
requirement entities and a prerequisite graph (built by :mod:`dag`). For each
course it uses the explicit ``prerequisites`` field where present, falling back
to the graph's out-edges otherwise. Nothing it reads is modified.

British English throughout. The engine performs no file I/O of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

DEFAULT_PER_TERM_CREDIT_CAP = 18  # hard cap; the simulator targets ~15.
MIN_PASS_GRADE = 1.0


@dataclass
class StudentState:
    """Mutable planning state for one student.

    Attributes
    ----------
    programme_id
        The programme the student is pursuing.
    passed
        Mapping ``course_id -> best grade`` for courses passed (grade >= 1.0).
    failed
        Set of course ids the student has failed and not yet re-passed. A failed
        course is eligible again (a retake).
    current_term
        Zero-based index of the term being planned.
    term_selected
        Courses already chosen for the current term (not yet graded). Used so
        :meth:`ConstraintEngine.eligible_courses` respects the per-term credit
        budget while a term is being filled.
    per_term_credit_cap
        Maximum credits allowed in a single term.
    """

    programme_id: str
    passed: dict[str, float] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)
    current_term: int = 0
    term_selected: set[str] = field(default_factory=set)
    per_term_credit_cap: int = DEFAULT_PER_TERM_CREDIT_CAP


class ConstraintEngine:
    """Prerequisite, credit-cap, retake and graduation-rule enforcement."""

    def __init__(
        self,
        courses: list[dict[str, Any]],
        blocks: list[dict[str, Any]],
        rule_groups: list[dict[str, Any]],
        rules: list[dict[str, Any]],
        rule_eligible_courses: list[dict[str, Any]],
        programmes: list[dict[str, Any]],
        graph: nx.DiGraph,
    ) -> None:
        self.courses_by_id = {c["course_id"]: c for c in courses}
        self.blocks_by_id = {b["block_id"]: b for b in blocks}
        self.rule_groups_by_id = {g["rule_group_id"]: g for g in rule_groups}
        self.rules_by_id = {r["rule_id"]: r for r in rules}
        self.programmes_by_id = {p["programme_id"]: p for p in programmes}
        self.graph = graph

        # block_id -> [rule_group, ...] (in group_index order)
        self.groups_by_block: dict[str, list[dict[str, Any]]] = {}
        for g in sorted(rule_groups, key=lambda x: x["group_index"]):
            self.groups_by_block.setdefault(g["block_id"], []).append(g)

        # rule_group_id -> [rule, ...] (in rule_index order)
        self.rules_by_group: dict[str, list[dict[str, Any]]] = {}
        for r in sorted(rules, key=lambda x: x["rule_index"]):
            self.rules_by_group.setdefault(r["rule_group_id"], []).append(r)

        # rule_id -> [(course_id, weight_credits), ...]
        self.eligible_by_rule: dict[str, list[tuple[str, int | None]]] = {}
        for row in rule_eligible_courses:
            self.eligible_by_rule.setdefault(row["rule_id"], []).append(
                (row["course_id"], row.get("weight_credits"))
            )

        # course_id -> AND-of-OR prerequisite groups. When a course carries an
        # explicit ``prerequisites`` field, use it but filter each OR-group to
        # in-set members and drop any group whose members are all out-of-set
        # (a prerequisite that references only nonexistent courses can never be
        # satisfied and so is treated as absent). This preserves OR semantics
        # (a multi-way prerequisite choice). For a course with no explicit
        # prerequisite, derive single-member groups from the augmented DAG
        # out-edges, which carry the augmented
        # prerequisites.
        course_id_set = set(self.courses_by_id)
        self.prereqs: dict[str, list[list[str]]] = {}
        for cid in self.courses_by_id:
            explicit = self.courses_by_id[cid].get("prerequisites")
            if explicit:
                groups = [
                    [m for m in grp if m in course_id_set] for grp in explicit
                ]
                self.prereqs[cid] = [g for g in groups if g]
            else:
                self.prereqs[cid] = [[succ] for succ in self.graph.successors(cid)]

    # -------------------------------------------------------------- prerequisites
    def check_prerequisites(self, course_id: str, passed_course_ids: set[str]) -> bool:
        """True iff every AND-group has >= 1 satisfied OR-member among passed.

        A course with no prerequisites is always satisfied.
        """
        for and_group in self.prereqs.get(course_id, []):
            if not any(member in passed_course_ids for member in and_group):
                return False
        return True

    def credits_of(self, course_id: str) -> int:
        return self.courses_by_id[course_id]["credits"]

    # ----------------------------------------------------------------- eligibility
    def eligible_courses(self, state: StudentState) -> set[str]:
        """Courses the student may legally take next, given the current state.

        Applies: (a) prerequisites satisfied by passed courses, (b) not already
        passed (a *failed* course is eligible again — a retake), and (c) adding
        the course would not breach the per-term credit cap given what has
        already been selected this term.
        """
        passed_ids = set(state.passed)
        used_credits = sum(self.credits_of(c) for c in state.term_selected)
        eligible: set[str] = set()
        for cid, course in self.courses_by_id.items():
            if not course.get("is_active", True):
                continue
            if cid in state.passed or cid in state.term_selected:
                continue
            if used_credits + self.credits_of(cid) > state.per_term_credit_cap:
                continue
            if not self.check_prerequisites(cid, passed_ids):
                continue
            eligible.add(cid)
        return eligible

    # ----------------------------------------------------------------- rule logic
    def _rule_eligible_set(self, rule: dict[str, Any]) -> list[tuple[str, int | None]]:
        """Resolve the (course_id, weight) list a rule counts, by selector type."""
        selector = rule.get("selector_type")
        if selector == "explicit_list":
            return self.eligible_by_rule.get(rule["rule_id"], [])
        if selector in ("attribute_match", "attribute_filter"):
            filt = rule.get("attribute_filter")
            if not filt:
                # An attribute selector with no expression means "any course"
                # (a general-electives rule: any number of credits from any course at
                # level 100+). Every catalogue course counts.
                return [(cid, None) for cid in self.courses_by_id]
            return [
                (cid, None)
                for cid, c in self.courses_by_id.items()
                if self._matches_attribute_filter(c, filt)
            ]
        if selector == "subject_match":
            subject = rule.get("attribute_filter")
            return [
                (cid, None)
                for cid, c in self.courses_by_id.items()
                if c.get("subject_area") == subject
            ]
        # Rules without a selector (cross_block_ref, one_of_groups) carry no list.
        return self.eligible_by_rule.get(rule["rule_id"], [])

    @staticmethod
    def _matches_attribute_filter(course: dict[str, Any], filt: str | None) -> bool:
        """Evaluate a simple attribute filter such as ``level>=200``.

        Supports ``<field><op><value>`` for numeric fields (e.g. ``level``) and
        membership against the course ``attributes`` list (e.g. ``WI``).
        """
        if not filt:
            return False
        for op in (">=", "<=", "==", ">", "<"):
            if op in filt:
                field_name, _, raw = filt.partition(op)
                field_name = field_name.strip()
                raw = raw.strip()
                value = course.get(field_name)
                if value is None:
                    return False
                try:
                    target = float(raw)
                except ValueError:
                    return False
                if op == ">=":
                    return value >= target
                if op == "<=":
                    return value <= target
                if op == "==":
                    return value == target
                if op == ">":
                    return value > target
                if op == "<":
                    return value < target
        # Bare token -> attribute membership.
        return filt in course.get("attributes", [])

    def _rule_satisfied(self, rule: dict[str, Any], passed_ids: set[str]) -> bool:
        """Whether a single rule is satisfied by the passed-course set."""
        rule_type = rule["rule_type"]

        if rule_type == "cross_block_ref":
            return self._cross_block_satisfied(rule, passed_ids)

        if rule_type == "one_of_groups":
            return any(
                self._group_satisfied(self.rule_groups_by_id[gid], passed_ids)
                for gid in (rule.get("member_group_ids") or [])
            )

        eligible = self._rule_eligible_set(rule)
        counted = [(cid, w) for cid, w in eligible if cid in passed_ids]
        n_courses = len(counted)
        n_credits = sum((w if w is not None else self.credits_of(cid)) for cid, w in counted)

        if rule_type == "min_courses":
            return n_courses >= (rule.get("min_courses") or 0)
        if rule_type == "min_credits":
            return n_credits >= (rule.get("min_credits") or 0)
        if rule_type == "min_courses_and_credits":
            return n_courses >= (rule.get("min_courses") or 0) and n_credits >= (
                rule.get("min_credits") or 0
            )
        if rule_type == "all_of":
            return all(cid in passed_ids for cid, _ in eligible) and bool(eligible)
        # Unknown rule type: treat as unsatisfied rather than silently pass.
        return False

    def _group_satisfied(self, group: dict[str, Any], passed_ids: set[str]) -> bool:
        """A rule group is satisfied when all of its rules are satisfied."""
        rules = self.rules_by_group.get(group["rule_group_id"], [])
        return all(self._rule_satisfied(r, passed_ids) for r in rules)

    def _block_satisfied(
        self, block_id: str, passed_ids: set[str], _seen: set[str] | None = None
    ) -> bool:
        """A block is satisfied when all its required rule groups are satisfied.

        ``cross_block_ref`` rules recurse into the referenced block. A guard set
        prevents infinite recursion if the data ever contains a reference cycle.
        """
        _seen = _seen or set()
        if block_id in _seen:
            return True  # already being evaluated up the stack
        _seen = _seen | {block_id}
        for group in self.groups_by_block.get(block_id, []):
            if not group.get("is_required", True):
                continue
            for rule in self.rules_by_group.get(group["rule_group_id"], []):
                if not self._rule_satisfied_with_recursion(rule, passed_ids, _seen):
                    return False
        return True

    def _rule_satisfied_with_recursion(
        self, rule: dict[str, Any], passed_ids: set[str], seen: set[str]
    ) -> bool:
        if rule["rule_type"] == "cross_block_ref":
            return self._cross_block_satisfied(rule, passed_ids, seen)
        return self._rule_satisfied(rule, passed_ids)

    def _cross_block_satisfied(
        self, rule: dict[str, Any], passed_ids: set[str], seen: set[str] | None = None
    ) -> bool:
        target_id = rule.get("target_block_id")
        if target_id:
            return self._block_satisfied(target_id, passed_ids, seen)
        target_type = rule.get("target_block_type")
        if target_type:
            candidates = sorted(
                (b for b in self.blocks_by_id.values() if b["block_type"] == target_type),
                key=lambda b: b["block_index"],
            )
            if not candidates:
                return False
            return self._block_satisfied(candidates[0]["block_id"], passed_ids, seen)
        return False

    def evaluate_rules(self, passed_course_ids: set[str]) -> dict[str, bool]:
        """Return ``rule_id -> satisfied`` for every rule in the programme."""
        passed_ids = set(passed_course_ids)
        result: dict[str, bool] = {}
        for rule_id, rule in self.rules_by_id.items():
            if rule["rule_type"] == "cross_block_ref":
                result[rule_id] = self._cross_block_satisfied(rule, passed_ids)
            else:
                result[rule_id] = self._rule_satisfied(rule, passed_ids)
        return result

    # ------------------------------------------------------------------ graduation
    def _required_blocks(self, programme_id: str) -> list[str]:
        """Required block ids relevant to a programme (programme + university scope)."""
        ids = []
        for b in self.blocks_by_id.values():
            if not b.get("is_required", True):
                continue
            if b.get("scope") == "university" or b.get("programme_id") in (programme_id, None):
                ids.append(b["block_id"])
        return ids

    def total_credits_required(self, programme_id: str) -> int:
        return self.programmes_by_id[programme_id]["total_credits_required"]

    def passed_credits(self, state: StudentState) -> int:
        return sum(self.credits_of(c) for c in state.passed)

    def is_graduated(self, state: StudentState) -> bool:
        """True iff all required blocks are satisfied and credits meet the total."""
        passed_ids = set(state.passed)
        if self.passed_credits(state) < self.total_credits_required(state.programme_id):
            return False
        return all(
            self._block_satisfied(bid, passed_ids)
            for bid in self._required_blocks(state.programme_id)
        )

    def remaining_requirements(self, state: StudentState) -> dict[str, Any]:
        """Per-block unmet rule groups and the outstanding credit total.

        Used by the simulator's scoring and, later, the RL reward. For each
        required block it reports the unsatisfied required rule groups and, per
        rule, the count/credit deficit.
        """
        passed_ids = set(state.passed)
        out: dict[str, Any] = {}
        for bid in self._required_blocks(state.programme_id):
            unmet_groups = []
            for group in self.groups_by_block.get(bid, []):
                if not group.get("is_required", True):
                    continue
                if not self._group_satisfied(group, passed_ids):
                    unmet_rules = [
                        r["rule_id"]
                        for r in self.rules_by_group.get(group["rule_group_id"], [])
                        if not (
                            self._cross_block_satisfied(r, passed_ids)
                            if r["rule_type"] == "cross_block_ref"
                            else self._rule_satisfied(r, passed_ids)
                        )
                    ]
                    unmet_groups.append(
                        {"rule_group_id": group["rule_group_id"], "unmet_rules": unmet_rules}
                    )
            out[bid] = {"unmet_groups": unmet_groups}
        out["credits_remaining"] = max(
            0, self.total_credits_required(state.programme_id) - self.passed_credits(state)
        )
        return out

    def unmet_required_rule_courses(
        self,
        passed_course_ids: set[str],
        one_of_choice: dict[str, str] | None = None,
    ) -> set[str]:
        """Courses that count toward at least one unsatisfied leaf requirement rule.

        A *leaf* rule is one that lists eligible courses (``min_courses``,
        ``min_credits``, ``min_courses_and_credits``, ``all_of``, or any
        member rule of an unsatisfied ``one_of_groups``). Used by the simulator
        to reward progress toward outstanding requirements.

        ``one_of_choice`` optionally maps a ``one_of_groups`` rule_id to the
        single member ``rule_group_id`` the student has chosen to pursue (e.g. a
        Finance vs Marketing minor). When given, only that member group's
        courses are treated as outstanding for that rule; otherwise all members
        are. Graduation is unaffected — completing either member satisfies the
        rule.
        """
        passed_ids = set(passed_course_ids)
        one_of_choice = one_of_choice or {}
        target_groups: set[str] = set()
        # Every required group whose rules are not all satisfied.
        for group in self.rule_groups_by_id.values():
            if not group.get("is_required", True):
                continue
            if not self._group_satisfied(group, passed_ids):
                target_groups.add(group["rule_group_id"])
        # one_of_groups: if the parent is unsatisfied, include member groups —
        # restricted to the chosen member when a preference is supplied.
        for rule in self.rules_by_id.values():
            if rule["rule_type"] == "one_of_groups" and not self._rule_satisfied(
                rule, passed_ids
            ):
                members = rule.get("member_group_ids") or []
                chosen = one_of_choice.get(rule["rule_id"])
                if chosen and chosen in members:
                    target_groups.add(chosen)
                else:
                    target_groups.update(members)

        n_courses = len(self.courses_by_id)
        courses: set[str] = set()
        for gid in target_groups:
            for rule in self.rules_by_group.get(gid, []):
                if rule["rule_type"] in ("cross_block_ref", "one_of_groups"):
                    continue
                if self._rule_satisfied(rule, passed_ids):
                    continue
                eligible = self._rule_eligible_set(rule)
                # Skip near-universal "any course" rules (e.g. a general
                # electives, which count any course). They are satisfied
                # incidentally by ordinary progress, so steering toward them
                # would flood the bonus across the whole catalogue and wash out
                # the signal for specific compulsory courses.
                if len(eligible) >= 0.9 * n_courses:
                    continue
                for cid, _ in eligible:
                    if cid not in passed_ids:
                        courses.add(cid)

        # Steer toward the transitive prerequisites of the outstanding required
        # courses too: a required course gated behind a (possibly augmented)
        # gateway course can only be unlocked by taking that gateway. Without
        # this, large catalogues with augmented cross-subject prerequisites leave
        # required courses permanently blocked. All OR-members of a prerequisite
        # group are added (cheap over-steer; taking any one unlocks the course).
        frontier = list(courses)
        while frontier:
            c = frontier.pop()
            for group in self.prereqs.get(c, []):
                for member in group:
                    if member not in passed_ids and member not in courses:
                        courses.add(member)
                        frontier.append(member)
        return courses

    # ------------------------------------------------------- baseline scoring tools
    def prerequisite_violations(self, ordered_plan: list[list[str]]) -> int:
        """Count courses taken before their prerequisites were passed.

        ``ordered_plan`` is a list of term course-lists in chronological order.
        Courses completed in earlier terms count as passed; a prerequisite taken
        in the *same* term does not satisfy a course taken that term. Used to
        score constraint-blind baselines.
        """
        passed: set[str] = set()
        violations = 0
        for term_courses in ordered_plan:
            for cid in term_courses:
                if not self.check_prerequisites(cid, passed):
                    violations += 1
            passed.update(term_courses)
        return violations

    def graduation_compliance(self, plan: list[list[str]]) -> float:
        """Fraction of programme rules satisfied at the end of a plan, in [0, 1]."""
        passed: set[str] = set()
        for term_courses in plan:
            passed.update(term_courses)
        results = self.evaluate_rules(passed)
        if not results:
            return 0.0
        return sum(1 for ok in results.values() if ok) / len(results)
