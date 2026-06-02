"""Multi-institution synthetic-student simulator for CAP-Bench.

Generates synthetic students for one or more institutions. Each student belongs
to ONE institution and is simulated against that institution's programme,
prerequisite DAG and constraint engine. The catalogues and rules are real
(parsed from public course catalogues); the students are synthetic and fully
reproducible from seed 42 plus the student's global index.

Realism features:
  * a 60/20/15/5 start-state mix (new / mid-degree / transfer / returning),
  * variable per-student per-term loads (full-time / part-time / overload) plus
    occasional summer terms — so term sizes are not uniform,
  * a recorded field track that gently biases elective choice.

British English throughout. CPU-only.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constraint_engine.engine import ConstraintEngine, StudentState

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("camp.simulator")

SEED = 42
TERM_CAP = 16
PER_TERM_CREDIT_CAP = 18
GREEDY_PROB = 0.7  # locked policy mix: ~70% greedy, ~30% stochastic
SOFTMAX_TEMPERATURE = 1.0

# Grade model (band means; sd tuned to a ~0.96 population pass rate).
BAND_MEANS = {"low": 2.4, "med": 3.0, "high": 3.6}
GRADE_SD = 1.1
MIN_PASS_GRADE = 1.0
TRANSFER_GRADE = 3.0  # sentinel for prereq satisfaction; excluded from GPA

GPA_BANDS = ["low", "med", "high"]
GPA_BAND_P = [0.25, 0.50, 0.25]
GENDERS = ["F", "M"]
COHORTS = ["2021FA", "2022FA", "2022SP", "2023FA", "2023SP", "2024FA", "2024SP"]
COLD_START_FRACTION = 0.10

START_STATE_TYPES = ["new", "mid_degree", "transfer", "returning"]
START_STATE_P = [0.60, 0.20, 0.15, 0.05]

TERM_LOAD_TYPES = ["full_time", "part_time", "overload"]
TERM_LOAD_P = [0.70, 0.20, 0.10]
TERM_LOAD_RANGE = {"full_time": (4, 5), "part_time": (2, 3), "overload": (5, 6)}
SUMMER_FRACTION = 0.30  # students who take occasional (every 3rd) summer terms
SUMMER_RANGE = (1, 2)


@dataclass
class InstitutionConfig:
    """Everything the simulator needs to drive one institution."""

    university: str
    programme_id: str
    engine: ConstraintEngine
    features: dict[str, dict[str, Any]]
    tracks: list[str]
    level_scale: int  # 100 (3-digit) or 1000 (4-digit), for the expected-level bonus
    one_of_rule: str = ""  # the one_of_groups rule (a minor/concentration choice); "" if none
    track_to_group: dict[str, str] = field(default_factory=dict)  # track -> member rule_group_id
    track_keywords: dict[str, list[str]] = field(default_factory=dict)  # track -> elective keywords
    max_centrality: float = field(default=1.0)

    def __post_init__(self) -> None:
        self.max_centrality = max((f["centrality"] for f in self.features.values()), default=1) or 1


def expected_year(term_index: int) -> int:
    """Expected curricular year (1-4) for a term: 1,1,2,2,3,3,4,4,..."""
    return 1 + min(term_index // 2, 3)


class MultiSimulator:
    """Generates a synthetic population across one or more institutions."""

    def __init__(self, configs: list[InstitutionConfig]):
        self.configs = configs

    # ------------------------------------------------------------------ scoring
    def _score(self, cfg: InstitutionConfig, cid: str, unmet: set[str], term_index: int,
               track: str = "") -> float:
        s = 0.0
        if cid in unmet:
            s += 3.0
        s += cfg.features[cid]["centrality"] / cfg.max_centrality
        course_year = min(cfg.engine.courses_by_id[cid]["level"] // cfg.level_scale, 4)
        if course_year == expected_year(term_index):
            s += 0.5
        # Soft field-appropriate elective bias: nudge towards courses whose title
        # matches the student's chosen track (e.g. an AI-track Khalifa student
        # prefers ML/vision electives). Small, so it never overrides requirements.
        kws = cfg.track_keywords.get(track) if track else None
        if kws:
            title = (cfg.engine.courses_by_id[cid].get("title") or "").lower()
            if any(kw in title for kw in kws):
                s += 0.4
        return s

    def _fill_term(
        self,
        cfg: InstitutionConfig,
        state: StudentState,
        one_of_choice: dict[str, str],
        n_target: int,
        rng: np.random.Generator,
        track: str = "",
    ) -> list[str]:
        """Pick up to ``n_target`` courses for the current term (policy mix)."""
        selected: list[str] = []
        while len(state.term_selected) < n_target:
            eligible = sorted(cfg.engine.eligible_courses(state))
            if not eligible:
                break
            unmet = cfg.engine.unmet_required_rule_courses(
                set(state.passed) | state.term_selected, one_of_choice=one_of_choice
            )
            scores = np.array([self._score(cfg, c, unmet, state.current_term, track) for c in eligible])
            if rng.random() < GREEDY_PROB:
                choice = eligible[int(np.argmax(scores))]
            else:
                shifted = scores - scores.max()
                weights = np.exp(shifted / SOFTMAX_TEMPERATURE)
                probs = weights / weights.sum()
                choice = eligible[int(rng.choice(len(eligible), p=probs))]
            state.term_selected.add(choice)
            selected.append(choice)
        return selected

    # ------------------------------------------------------------------ helpers
    def _grade(self, gpa_band: str, rng: np.random.Generator) -> float:
        return float(np.clip(rng.normal(BAND_MEANS[gpa_band], GRADE_SD), 0.0, 4.0))

    def _n_courses_for_term(
        self, term_load_type: str, is_summer: bool, rng: np.random.Generator
    ) -> int:
        lo, hi = SUMMER_RANGE if is_summer else TERM_LOAD_RANGE[term_load_type]
        return int(rng.integers(lo, hi + 1))

    def _transfer_courses(self, cfg: InstitutionConfig, rng: np.random.Generator) -> list[str]:
        """Pick 3-6 lowest-band courses as transfer credits (prereq-satisfying)."""
        low_band = cfg.level_scale  # 100 or 1000
        candidates = sorted(
            cid
            for cid, c in cfg.engine.courses_by_id.items()
            if c["level"] // cfg.level_scale == 1 and c.get("is_active", True)
        )
        if not candidates:
            return []
        n = int(rng.integers(3, 7))
        n = min(n, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in sorted(idx)]

    # ------------------------------------------------------------------ one student
    def simulate_student(
        self, cfg: InstitutionConfig, global_idx: int, student_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rng = np.random.default_rng(SEED + global_idx)

        track = str(rng.choice(cfg.tracks))
        demo = {
            "gender": str(rng.choice(GENDERS)),
            "gpa_band": str(rng.choice(GPA_BANDS, p=GPA_BAND_P)),
            "cohort": str(rng.choice(COHORTS)),
            "major_track": track,
            "chosen_track": track,
            "cold_start": bool(rng.random() < COLD_START_FRACTION),
            "start_state_type": str(rng.choice(START_STATE_TYPES, p=START_STATE_P)),
            "term_load_type": str(rng.choice(TERM_LOAD_TYPES, p=TERM_LOAD_P)),
            "university": cfg.university,
        }
        # Retained record fields; the three benchmark programmes use neither a
        # concentration nor a minor, so both are None (the field track is in demo).
        chosen_concentration = None
        minor = None
        # one_of_groups choice only where a programme models one; the benchmark
        # programmes have no one_of rule, so the choice is empty.
        if cfg.one_of_rule and cfg.track_to_group:
            one_of_choice = {cfg.one_of_rule: cfg.track_to_group[track]}
        else:
            one_of_choice = {}
        is_summer_taker = rng.random() < SUMMER_FRACTION

        state = StudentState(programme_id=cfg.programme_id, per_term_credit_cap=PER_TERM_CREDIT_CAP)
        terms: list[dict[str, Any]] = []
        enrolment_rows: list[dict[str, Any]] = []
        best_grade: dict[str, float] = {}  # graded courses only (for GPA)
        transfer_courses: list[str] = []

        # ----- start-state pre-seeding -----
        if demo["start_state_type"] == "transfer":
            transfer_courses = self._transfer_courses(cfg, rng)
            for cid in transfer_courses:
                state.passed[cid] = TRANSFER_GRADE  # satisfies prereqs; no GPA
                enrolment_rows.append(
                    self._enrol_row(student_id, -1, cid, None, True, cfg, demo, is_transfer=True, is_repeat=False)
                )

        gap_after_term: int | None = None
        prior_terms = 0
        if demo["start_state_type"] == "mid_degree":
            prior_terms = int(rng.integers(1, 5))  # 1-4 prior terms of history
        if demo["start_state_type"] == "returning":
            gap_after_term = int(rng.integers(2, 6))  # gap after 2-5 terms

        # Cold-start students observe only a short trajectory.
        max_terms = int(rng.integers(1, 4)) if demo["cold_start"] else TERM_CAP

        graduated = False
        terms_to_graduate: int | None = None
        term_index = 0
        terms_completed = 0
        while terms_completed < max_terms:
            is_summer = is_summer_taker and (terms_completed % 3 == 2)
            n_target = self._n_courses_for_term(demo["term_load_type"], is_summer, rng)
            state.current_term = term_index
            state.term_selected = set()
            chosen = self._fill_term(cfg, state, one_of_choice, n_target, rng, track)
            if not chosen:
                break

            term_courses = []
            term_grade_sum = 0.0
            term_credits = 0
            for cid in chosen:
                grade = self._grade(demo["gpa_band"], rng)
                passed = grade >= MIN_PASS_GRADE
                credits = cfg.engine.credits_of(cid)
                is_repeat = cid in state.failed or cid in best_grade
                is_prior = terms_completed < prior_terms
                term_courses.append(
                    {"course_id": cid, "grade": round(grade, 3), "passed": passed, "credits": credits}
                )
                term_grade_sum += grade
                term_credits += credits
                best_grade[cid] = max(best_grade.get(cid, -1.0), grade)
                enrolment_rows.append(
                    self._enrol_row(student_id, term_index, cid, round(grade, 3), passed, cfg, demo, is_transfer=False, is_repeat=is_repeat, is_prior=is_prior)
                )
                if passed:
                    state.passed[cid] = max(state.passed.get(cid, 0.0), grade)
                    state.failed.discard(cid)
                else:
                    state.failed.add(cid)

            terms.append(
                {
                    "term_index": term_index,
                    "courses": term_courses,
                    "term_gpa": round(term_grade_sum / len(term_courses), 3),
                    "term_credits": term_credits,
                    "is_summer": is_summer,
                    "is_prior_history": terms_completed < prior_terms,
                }
            )
            terms_completed += 1

            if cfg.engine.is_graduated(state):
                graduated = True
                terms_to_graduate = terms_completed
                break

            # A returning student takes a one-term gap after `gap_after_term`.
            term_index += 1
            if gap_after_term is not None and terms_completed == gap_after_term:
                term_index += 1  # skip a term index to represent the break

        graded = [g for g in best_grade.values()]
        final_gpa = round(sum(graded) / len(graded), 3) if graded else 0.0
        total_credits = cfg.engine.passed_credits(state)

        record = {
            "student_id": student_id,
            "university": cfg.university,
            "programme_id": cfg.programme_id,
            "demographics": demo,
            "chosen_concentration": chosen_concentration,
            "minor": minor,
            "chosen_track": track,
            "start_state_type": demo["start_state_type"],
            "term_load_type": demo["term_load_type"],
            "transfer_courses": transfer_courses,
            "gap_after_term": gap_after_term,
            "prior_terms": prior_terms,
            "terms": terms,
            "final_gpa": final_gpa,
            "graduated": graduated,
            "terms_to_graduate": terms_to_graduate,
            "total_credits": total_credits,
        }
        return record, enrolment_rows

    @staticmethod
    def _enrol_row(
        student_id, term_index, cid, grade, passed, cfg, demo, *, is_transfer, is_repeat, is_prior=False
    ) -> dict[str, Any]:
        return {
            "student_id": student_id,
            "university": cfg.university,
            "term_index": term_index,
            "course_id": cid,
            "grade": grade,
            "passed": passed,
            "credits": cfg.engine.credits_of(cid),
            "is_repeat": is_repeat,
            "is_transfer": is_transfer,
            "is_prior_history": is_prior,
            "gender": demo["gender"],
            "gpa_band": demo["gpa_band"],
            "cohort": demo["cohort"],
            "major_track": demo["major_track"],
            "chosen_track": demo.get("chosen_track", demo["major_track"]),
            "cold_start": demo["cold_start"],
            "start_state_type": demo["start_state_type"],
            "term_load_type": demo["term_load_type"],
        }

    # ------------------------------------------------------------------ population
    def generate(self, counts: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate ``counts[university]`` students per institution."""
        students: list[dict[str, Any]] = []
        enrolments: list[dict[str, Any]] = []
        global_idx = 0
        for cfg in self.configs:
            n = counts[cfg.university]
            for local in range(n):
                student_id = f"S_{cfg.university.upper()}_{local + 1:05d}"
                rec, rows = self.simulate_student(cfg, global_idx, student_id)
                students.append(rec)
                enrolments.extend(rows)
                global_idx += 1
                if (local + 1) % 500 == 0:
                    logger.info("[%s] simulated %d/%d", cfg.university, local + 1, n)
        return students, enrolments


def make_stratified_splits(
    students: list[dict[str, Any]], seed: int = SEED
) -> dict[str, list[str]]:
    """Disjoint 70/15/15 split, stratified by university."""
    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    by_uni: dict[str, list[str]] = {}
    for s in students:
        by_uni.setdefault(s["university"], []).append(s["student_id"])
    for uni in sorted(by_uni):
        ids = sorted(by_uni[uni])
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        train += ids[:n_train]
        val += ids[n_train : n_train + n_val]
        test += ids[n_train + n_val :]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_students_and_enrolments(
    students: list[dict[str, Any]], enrolments: list[dict[str, Any]], students_dir: Path
) -> dict[str, list[str]]:
    students_dir.mkdir(parents=True, exist_ok=True)
    with (students_dir / "students.jsonl").open("w") as fh:
        for rec in students:
            fh.write(json.dumps(rec) + "\n")

    df = pd.DataFrame(enrolments)
    df.to_parquet(students_dir / "enrolments.parquet", index=False)
    df.to_csv(students_dir / "enrolments.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    splits = make_stratified_splits(students)
    with (students_dir / "splits.json").open("w") as fh:
        json.dump(splits, fh, indent=2)
    return splits


if __name__ == "__main__":  # pragma: no cover
    from src.dataset.generate_dataset import main

    main()
