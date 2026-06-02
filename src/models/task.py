"""Phase B recommendation task and data loaders.

Locked task: **next-term course-set prediction.** For each student, for each
term ``t`` (1-based ``t >= 2``; 0-based ``term_index >= 1``) the model sees the
history (all courses + grades in the earlier terms) and must predict the set of
``course_id`` the student actually took in term ``t``. A student with ``T`` terms
yields ``T - 1`` samples. A sample inherits its student's train/val/test split,
so no student crosses splits.

Also owns the shared course vocabulary (``course_id`` <-> integer index) and the
fixed categorical encodings reused across the graph, GNN and baseline modules.

British English throughout. Deterministic given the Phase A artefacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
# The three-institution mixed-field benchmark: Khalifa (Computer Science),
# AUS (Information Systems), UNC (Economics).
BUNDLE = _REPO_ROOT / "data" / "cap_bench" / "v3_3inst"
STUDENTS_PATH = BUNDLE / "students" / "students.jsonl"
SPLITS_PATH = BUNDLE / "students" / "splits.json"
COURSES_PATH = BUNDLE / "intermediate" / "courses.json"
COURSE_FEATURES_PATH = BUNDLE / "intermediate" / "course_features.json"
MODELS_DIR = BUNDLE / "models"
VOCAB_PATH = MODELS_DIR / "course_vocab.json"


def _derive_categoricals() -> tuple[list[str], list[str]]:
    """Subject-area and university vocabularies, derived from the bundle courses
    so the encodings match the active dataset exactly. In v3_3inst the
    ``subject_area`` is the fine subject code (COSC, ISA, ECON, MATH, ...)."""
    with COURSES_PATH.open() as fh:
        courses = json.load(fh)
    subjects = sorted({c["subject_area"] for c in courses})
    universities = sorted({c["university"] for c in courses})
    return subjects, universities


# Fixed categorical vocabularies (sorted for determinism), derived from v3_3inst.
SUBJECT_AREAS, UNIVERSITIES = _derive_categoricals()
GPA_BANDS = ["low", "med", "high"]
GENDERS = ["F", "M"]
COHORTS = ["2021FA", "2022FA", "2022SP", "2023FA", "2023SP", "2024FA", "2024SP"]
COHORT_INDEX = {c: i for i, c in enumerate(sorted(COHORTS))}  # scaled by order
# Institution-appropriate field tracks across the three institutions
# (Khalifa: ai/cybersecurity/general; AUS: business_analytics/information_systems/
# general; UNC: quantitative/policy/general).
MAJOR_TRACKS = ["ai", "business_analytics", "cybersecurity", "general",
                "information_systems", "policy", "quantitative"]
MIN_PASS_GRADE = 1.0


@dataclass
class Sample:
    student_id: str
    university: str
    term_index: int
    history_courses: list[str]
    history_grades: dict[str, float]
    demographics: dict[str, Any]
    target_courses: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "university": self.university,
            "term_index": self.term_index,
            "history_courses": self.history_courses,
            "history_grades": self.history_grades,
            "demographics": self.demographics,
            "target_courses": self.target_courses,
        }


def load_courses() -> list[dict[str, Any]]:
    with COURSES_PATH.open() as fh:
        return json.load(fh)


def course_university() -> dict[str, str]:
    """Map course_id -> university (for institution-masking of candidates)."""
    return {c["course_id"]: c["university"] for c in load_courses()}


def build_vocab(save: bool = True) -> dict[str, int]:
    """Course-id -> index vocabulary over all 262 courses (sorted ids)."""
    course_ids = sorted(c["course_id"] for c in load_courses())
    vocab = {cid: i for i, cid in enumerate(course_ids)}
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with VOCAB_PATH.open("w") as fh:
            json.dump(vocab, fh, indent=2)
    return vocab


def load_vocab() -> dict[str, int]:
    if VOCAB_PATH.exists():
        with VOCAB_PATH.open() as fh:
            return json.load(fh)
    return build_vocab()


def _load_students() -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in STUDENTS_PATH.read_text().splitlines()]


def _load_splits() -> dict[str, list[str]]:
    with SPLITS_PATH.open() as fh:
        return json.load(fh)


def _student_to_split() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split, ids in _load_splits().items():
        for sid in ids:
            mapping[sid] = split
    return mapping


def _samples_for_student(student: dict[str, Any]) -> list[Sample]:
    """Build (history, target) samples for one student (one per term >= 1)."""
    terms = sorted(student["terms"], key=lambda t: t["term_index"])
    samples: list[Sample] = []
    history_courses: list[str] = []
    history_grades: dict[str, float] = {}
    for pos, term in enumerate(terms):
        target = [c["course_id"] for c in term["courses"]]
        if pos >= 1:  # need at least one earlier term as history
            samples.append(
                Sample(
                    student_id=student["student_id"],
                    university=student["university"],
                    term_index=term["term_index"],
                    history_courses=list(history_courses),
                    history_grades=dict(history_grades),
                    demographics=dict(student["demographics"]),
                    target_courses=target,
                )
            )
        # Roll this term into the history (best grade per distinct course).
        for c in term["courses"]:
            cid = c["course_id"]
            if cid not in history_courses:
                history_courses.append(cid)
            history_grades[cid] = max(history_grades.get(cid, -1.0), c["grade"])
    return samples


def load_samples(split: str) -> list[dict[str, Any]]:
    """Return the next-term-prediction samples belonging to *split*."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split}")
    split_of = _student_to_split()
    samples: list[dict[str, Any]] = []
    for student in _load_students():
        if split_of.get(student["student_id"]) != split:
            continue
        samples.extend(s.as_dict() for s in _samples_for_student(student))
    return samples


def passed_history(sample: dict[str, Any]) -> list[str]:
    """Courses in the sample's history that were passed (grade >= 1.0)."""
    return [c for c in sample["history_courses"] if sample["history_grades"][c] >= MIN_PASS_GRADE]


def report() -> dict[str, Any]:
    import numpy as np

    vocab = build_vocab(save=True)
    out: dict[str, Any] = {"vocab_size": len(vocab), "splits": {}}
    for split in ("train", "val", "test"):
        samples = load_samples(split)
        sizes = np.array([len(s["target_courses"]) for s in samples], dtype=float)
        per_uni = {}
        for uni in UNIVERSITIES:
            per_uni[uni] = sum(1 for s in samples if s["university"] == uni)
        out["splits"][split] = {
            "n_samples": len(samples),
            "mean_target_set_size": round(float(sizes.mean()), 4) if len(sizes) else 0.0,
            "std_target_set_size": round(float(sizes.std()), 4) if len(sizes) else 0.0,
            "per_university": per_uni,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
