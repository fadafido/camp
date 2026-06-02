"""Stage 2 — heterogeneous graph construction (PyTorch Geometric ``HeteroData``).

Builds a course/student heterogeneous graph from the Phase A artefacts:

Nodes
  * ``course`` (262): [level (scaled), credits (scaled), subject-area one-hot,
    centrality, blocking_factor, delay_factor].
  * ``student`` (**train students only**, leakage guard): [gpa-band one-hot,
    gender one-hot, cohort (scaled), cold-start flag, completed-course count
    (scaled)].

Edges
  * ``(course, prereq_of, course)`` — from the augmented DAG (real + augmented).
  * ``(student, took, course)`` and reverse ``(course, taken_by, student)`` —
    train enrolments, passed courses only.
  * ``(course, similar_to, course)`` — co-enrolment similarity computed from
    train histories (Jaccard >= threshold, documented below).

Writes ``models/graph_summary.json`` (node/edge counts) and prints it.

British English throughout. Deterministic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.constraint_engine import dag
from src.models import task

BUNDLE = task.BUNDLE
MODELS_DIR = task.MODELS_DIR
GRAPH_SUMMARY_PATH = MODELS_DIR / "graph_summary.json"
ENROLMENTS_PATH = BUNDLE / "students" / "enrolments.parquet"

# Co-enrolment similarity threshold (Jaccard over train students' course sets).
# Two courses are linked if the fraction of students who took either that also
# took both is >= this value. 0.5 keeps the strongly co-taken compulsory core
# connected while dropping incidental elective overlaps.
SIMILARITY_JACCARD_THRESHOLD = 0.5


def _course_year(level: int) -> int:
    """Curricular year 1-5 from a 3-digit (Khalifa/AUS/UNC) or 4-digit level code."""
    return level // 1000 if level >= 1000 else level // 100


# v3_3inst institutions are all 3-digit; mapping kept explicit for clarity.
_INST_DIGITS = {"khalifa": 3, "aus": 3, "unc": 3}


def _course_node_features(vocab: dict[str, int]) -> tuple[torch.Tensor, list[str]]:
    courses = {c["course_id"]: c for c in task.load_courses()}
    feats = {f["course_id"]: f for f in json.loads(task.COURSE_FEATURES_PATH.read_text())}
    order = sorted(vocab, key=lambda c: vocab[c])

    max_cent = max(f["centrality"] for f in feats.values()) or 1
    max_block = max(f["blocking_factor"] for f in feats.values()) or 1
    max_delay = max(f["delay_factor"] for f in feats.values()) or 1

    rows = []
    for cid in order:
        c = courses[cid]
        f = feats[cid]
        # A curricular-year feature (1-5, scaled) is institution-agnostic, unlike
        # the raw level code, keeping the feature comparable across institutions.
        subject_oh = [1.0 if c["subject_area"] == s else 0.0 for s in task.SUBJECT_AREAS]
        uni_oh = [1.0 if c["university"] == u else 0.0 for u in task.UNIVERSITIES]
        rows.append(
            [_course_year(c["level"]) / 5.0, c["credits"] / 3.0]
            + subject_oh
            + [
                f["centrality"] / max_cent,
                f["blocking_factor"] / max_block,
                f["delay_factor"] / max_delay,
            ]
            + uni_oh
        )
    return torch.tensor(rows, dtype=torch.float), order


def _train_students() -> list[dict[str, Any]]:
    split_of = task._student_to_split()
    return [
        s for s in task._load_students() if split_of.get(s["student_id"]) == "train"
    ]


def _student_node_features(
    students: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, int]]:
    students = sorted(students, key=lambda s: s["student_id"])
    index = {s["student_id"]: i for i, s in enumerate(students)}

    completed = []
    for s in students:
        done = {c["course_id"] for t in s["terms"] for c in t["courses"] if c["passed"]}
        completed.append(len(done))
    max_completed = max(completed) or 1

    rows = []
    for s, n_done in zip(students, completed):
        d = s["demographics"]
        gpa_oh = [1.0 if d["gpa_band"] == b else 0.0 for b in task.GPA_BANDS]
        gender_oh = [1.0 if d["gender"] == g else 0.0 for g in task.GENDERS]
        uni_oh = [1.0 if s["university"] == u else 0.0 for u in task.UNIVERSITIES]
        cohort_scaled = task.COHORT_INDEX[d["cohort"]] / (len(task.COHORTS) - 1)
        rows.append(
            gpa_oh
            + gender_oh
            + [cohort_scaled, 1.0 if d["cold_start"] else 0.0, n_done / max_completed]
            + uni_oh
        )
    return torch.tensor(rows, dtype=torch.float), index


def _prereq_edges(vocab: dict[str, int]) -> torch.Tensor:
    """(course, prereq_of, course): edges from prereq -> dependent.

    Built from each institution's own hybrid DAG (real prerequisites + minimal
    augmentation) and unioned. There are no cross-institution prerequisite edges,
    since no DAG references another institution's courses.
    """
    all_courses = task.load_courses()
    src, dst = [], []
    for uni, digits in _INST_DIGITS.items():
        inst_courses = [c for c in all_courses if c["university"] == uni]
        graph, _ = dag.build_real_inst_dag(inst_courses, digits)
        for dependent, prereq in graph.edges():  # edge dependent -> prereq
            src.append(vocab[prereq])
            dst.append(vocab[dependent])
    return torch.tensor([src, dst], dtype=torch.long)


def _took_edges(
    vocab: dict[str, int], student_index: dict[str, int]
) -> torch.Tensor:
    """(student, took, course): train enrolments, passed courses only."""
    df = pd.read_parquet(ENROLMENTS_PATH)
    df = df[df["passed"] & df["student_id"].isin(student_index)]
    pairs = df[["student_id", "course_id"]].drop_duplicates()
    s = [student_index[sid] for sid in pairs["student_id"]]
    c = [vocab[cid] for cid in pairs["course_id"]]
    return torch.tensor([s, c], dtype=torch.long)


def _similarity_edges(
    vocab: dict[str, int], students: list[dict[str, Any]]
) -> tuple[torch.Tensor, float]:
    """(course, similar_to, course): Jaccard co-enrolment over train histories."""
    course_sets = []
    course_freq: dict[str, int] = defaultdict(int)
    for s in students:
        taken = {c["course_id"] for t in s["terms"] for c in t["courses"]}
        course_sets.append(taken)
        for cid in taken:
            course_freq[cid] += 1

    co: dict[tuple[str, str], int] = defaultdict(int)
    for taken in course_sets:
        for a, b in combinations(sorted(taken), 2):
            co[(a, b)] += 1

    src, dst = [], []
    for (a, b), n_both in co.items():
        union = course_freq[a] + course_freq[b] - n_both
        jaccard = n_both / union if union else 0.0
        if jaccard >= SIMILARITY_JACCARD_THRESHOLD:
            # Undirected -> add both directions.
            src += [vocab[a], vocab[b]]
            dst += [vocab[b], vocab[a]]
    return torch.tensor([src, dst], dtype=torch.long), SIMILARITY_JACCARD_THRESHOLD


def build_graph() -> tuple[HeteroData, dict[str, int], dict[str, int]]:
    """Build the HeteroData object plus the course and student index maps."""
    vocab = task.load_vocab()
    students = _train_students()

    data = HeteroData()
    course_x, _ = _course_node_features(vocab)
    student_x, student_index = _student_node_features(students)
    data["course"].x = course_x
    data["student"].x = student_x

    data["course", "prereq_of", "course"].edge_index = _prereq_edges(vocab)
    took = _took_edges(vocab, student_index)
    data["student", "took", "course"].edge_index = took
    data["course", "taken_by", "student"].edge_index = took.flip(0)
    sim_edges, _ = _similarity_edges(vocab, students)
    data["course", "similar_to", "course"].edge_index = sim_edges

    return data, vocab, student_index


def summarise(data: HeteroData) -> dict[str, Any]:
    summary = {
        "nodes": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "edges": {
            "__".join(et): int(data[et].edge_index.size(1)) for et in data.edge_types
        },
        "course_feature_dim": int(data["course"].x.size(1)),
        "student_feature_dim": int(data["student"].x.size(1)),
        "similarity_jaccard_threshold": SIMILARITY_JACCARD_THRESHOLD,
    }
    return summary


def build_and_save() -> tuple[HeteroData, dict[str, int], dict[str, int]]:
    data, vocab, student_index = build_graph()
    summary = summarise(data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with GRAPH_SUMMARY_PATH.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return data, vocab, student_index


if __name__ == "__main__":
    build_and_save()
