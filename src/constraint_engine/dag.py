"""Prerequisite DAG construction and curricular metrics for CAP-Bench.

Builds a directed acyclic prerequisite graph over an institution's catalogue
courses from their real (catalogue-parsed) prerequisites, applies minimal
curriculum-informed augmentation for courses left without any prerequisite, then
computes Heileman-style curricular metrics (centrality, blocking factor, delay
factor) that feed the GNN node features and the RL reward.

Edge convention
---------------
An edge ``u -> v`` means "course *u* requires course *v* as a prerequisite"
(``v`` must be passed before ``u``). With this convention the *dependents* of a
course ``c`` (courses that transitively require ``c``) are the ancestors of
``c`` reached by following edges backwards.

British English is used throughout. Seed 42 governs all project randomness,
although the augmentation here is fully deterministic and draws no random
numbers.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("camp.dag")

# Repository-relative paths. Resolve from this file so the module works
# regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_INST_DIR = _REPO_ROOT / "data" / "intermediate"


def _centrality(graph: nx.DiGraph, node: str) -> int:
    """Downstream count: number of courses that transitively depend on *node*.

    With the ``course -> prereq`` convention these are the ancestors of *node*.
    """
    return len(nx.ancestors(graph, node))


def augment_dag(graph: nx.DiGraph, courses: list[dict[str, Any]]) -> nx.DiGraph:
    """Deterministically augment the DAG with synthetic prerequisites.

    Rule (fully reproducible, documented in the datasheet):
      * Keep all real prerequisites.
      * For any course at level ``L >= 200`` with zero prerequisites, add ONE
        synthetic prerequisite: the highest-centrality course at level
        ``L - 100`` in the SAME ``subject_area``. If none exists at ``L - 100``,
        use the highest-centrality course at the lowest available level in that
        subject area. If the subject area has no lower course, leave it a root.
      * Never create a cycle (skip any edge that would).

    Processing is bottom-up by level (200, then 300, then 400) and, within a
    level, by ascending ``course_id``. Centrality is recomputed on the
    graph-so-far (real edges plus augmented edges added at lower levels), so
    gateway courses that have already accumulated dependents are preferred —
    this grows shallow chains rather than scattering edges. Ties in centrality
    are broken by ascending ``course_id`` for determinism.

    Because every augmented edge points strictly downward in level, the
    augmentation can never introduce a cycle; the explicit cycle guard is kept
    as a defensive invariant.
    """
    by_id = {c["course_id"]: c for c in courses}
    # subject_area -> {level -> [course_id, ...]}
    by_subject_level: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for c in courses:
        by_subject_level[c["subject_area"]][c["level"]].append(c["course_id"])

    def best_candidate(subject: str, target_level: int) -> str | None:
        """Highest-centrality course in *subject* at *target_level* (or below)."""
        levels = by_subject_level.get(subject, {})
        candidate_pool = levels.get(target_level, [])
        if not candidate_pool:
            # Fall back to the lowest available level below the course's level.
            lower_levels = sorted(lvl for lvl in levels if lvl < target_level + 100)
            for lvl in lower_levels:
                if lvl < target_level + 100 and levels.get(lvl):
                    candidate_pool = levels[lvl]
                    break
        if not candidate_pool:
            return None
        # Highest centrality; tie-break on ascending course_id.
        return max(
            sorted(candidate_pool),
            key=lambda cid: _centrality(graph, cid),
        )

    for level in (200, 300, 400):
        courses_at_level = sorted(
            c["course_id"] for c in courses if c["level"] == level
        )
        for cid in courses_at_level:
            if graph.out_degree(cid) > 0:
                continue  # already has a (real) prerequisite
            subject = by_id[cid]["subject_area"]
            target = best_candidate(subject, level - 100)
            if target is None or target == cid:
                continue  # no lower course in this subject area -> remains a root
            # Cycle guard: only add if it keeps the graph acyclic.
            graph.add_edge(cid, target, origin="augmented")
            if not nx.is_directed_acyclic_graph(graph):
                graph.remove_edge(cid, target)
                logger.warning("Skipped augmented edge %s -> %s (would cycle)", cid, target)
    return graph


def _longest_chain_down(graph: nx.DiGraph, memo: dict[str, int]) -> dict[str, int]:
    """Longest prerequisite chain *below* each node (nodes following out-edges)."""

    def depth(node: str) -> int:
        if node in memo:
            return memo[node]
        succ = list(graph.successors(node))
        memo[node] = 1 + max((depth(s) for s in succ), default=0)
        return memo[node]

    for n in graph.nodes:
        depth(n)
    return memo


def _longest_chain_up(graph: nx.DiGraph, memo: dict[str, int]) -> dict[str, int]:
    """Longest dependent chain *above* each node (nodes following in-edges)."""

    def height(node: str) -> int:
        if node in memo:
            return memo[node]
        pred = list(graph.predecessors(node))
        memo[node] = 1 + max((height(p) for p in pred), default=0)
        return memo[node]

    for n in graph.nodes:
        height(n)
    return memo


def _blocking_factor(graph: nx.DiGraph, node: str) -> int:
    """Courses made unreachable if *node* is removed.

    A course becomes unreachable when removing *node* leaves one of its
    AND-of-OR prerequisite groups with no satisfiable member. The augmented
    graph uses single-member groups (one prerequisite per course), so the set of
    blocked courses equals the transitive dependents of *node*; the calculation
    is written generally via a fixed-point so it stays correct if richer
    prerequisite groups are ever introduced upstream.
    """
    removed = {node}
    # Iteratively mark courses unreachable: a course is blocked if every prereq
    # edge it has points (transitively) into the removed set. Since each course
    # carries one prereq per course, an out-edge into the removed set blocks it.
    changed = True
    while changed:
        changed = False
        for n in graph.nodes:
            if n in removed:
                continue
            succ = list(graph.successors(n))
            if succ and all(s in removed for s in succ):
                removed.add(n)
                changed = True
    return len(removed) - 1  # exclude the removed node itself


def compute_course_features(
    graph: nx.DiGraph, courses: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Compute per-course curricular metrics (Heileman-style) and metadata.

    Returns one record per ``course_id`` with:
      * ``centrality`` — downstream count (transitive dependents).
      * ``blocking_factor`` — courses unreachable if this course is removed.
      * ``delay_factor`` — length (in nodes) of the longest prerequisite path
        passing through this course.
      * ``level`` and ``credits`` — copied catalogue metadata.
    """
    by_id = {c["course_id"]: c for c in courses}
    down = _longest_chain_down(graph, {})  # nodes at and below
    up = _longest_chain_up(graph, {})  # nodes at and above

    features: dict[str, dict[str, Any]] = {}
    for cid in graph.nodes:
        # Longest path through the node = chain above + chain below - 1 (the node
        # is counted in both halves).
        delay_factor = up[cid] + down[cid] - 1
        features[cid] = {
            "course_id": cid,
            "centrality": _centrality(graph, cid),
            "blocking_factor": _blocking_factor(graph, cid),
            "delay_factor": delay_factor,
            "level": by_id[cid]["level"],
            "credits": by_id[cid]["credits"],
        }
    return features


def dag_summary(graph: nx.DiGraph) -> dict[str, Any]:
    """Summary statistics over the (possibly augmented) DAG."""
    real_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get("origin") == "real")
    augmented_edges = sum(
        1 for _, _, d in graph.edges(data=True) if d.get("origin") == "augmented"
    )
    is_acyclic = nx.is_directed_acyclic_graph(graph)
    max_depth = nx.dag_longest_path_length(graph) if is_acyclic else -1
    n_with_prereq = sum(1 for n in graph.nodes if graph.out_degree(n) > 0)
    return {
        "n_courses": graph.number_of_nodes(),
        "real_edges": real_edges,
        "augmented_edges": augmented_edges,
        "is_acyclic": is_acyclic,
        "max_chain_depth": max_depth,
        "n_courses_with_prereq": n_with_prereq,
    }


# ============================================================================
# Per-institution hybrid DAGs (real prerequisites + minimal augmentation)
# ============================================================================
#
# Khalifa (CS), AUS (IS) and UNC (Economics) each ship *real* prerequisites
# parsed from their public catalogues into the ``prerequisites`` field
# (AND-of-OR-groups of in-set course_ids). The DAG is therefore mostly real;
# augmentation applies only to courses that still have zero prerequisites after
# the real ones, at curricular level >= 200. No hand-seeded prerequisites are
# used — the catalogues expose enough real ones.


def _load_real_courses(university: str) -> list[dict[str, Any]]:
    with (_REAL_INST_DIR / university / f"{university}_courses.json").open() as fh:
        return json.load(fh)


def build_real_only_dag(courses: list[dict[str, Any]]) -> tuple[nx.DiGraph, int]:
    """Build a graph from the courses' real ``prerequisites`` (no seed list).

    Adds one node per course and an edge ``course -> prereq`` (origin="real")
    for every referenced prerequisite. References outside the course set are
    ignored and counted (should be zero here — the ingester already pruned
    out-of-set references — but the guard is kept).
    """
    course_ids = {c["course_id"] for c in courses}
    graph = nx.DiGraph()
    for course in courses:
        graph.add_node(
            course["course_id"],
            level=course["level"],
            credits=course["credits"],
            subject_area=course["subject_area"],
        )
    ignored = 0
    for course in courses:
        cid = course["course_id"]
        for and_group in (course.get("prerequisites") or []):
            for prereq_id in and_group:
                if prereq_id not in course_ids:
                    ignored += 1
                    continue
                graph.add_edge(cid, prereq_id, origin="real")
    return graph, ignored


def build_real_inst_dag(courses: list[dict[str, Any]], digits: int = 3) -> tuple[nx.DiGraph, int]:
    """Hybrid DAG for a real-catalogue institution: real prereqs + augmentation.

    All three benchmark institutions use 3-digit course numbers, so augmentation
    uses the level rule (:func:`augment_dag`, levels 200/300/400). ``digits`` is
    retained for signature stability and must be 3.
    """
    real_graph, ignored = build_real_only_dag(courses)
    graph = augment_dag(real_graph, courses)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Real-institution augmented graph is cyclic — STOP.")
    return graph, ignored


def build_and_save_real_inst(university: str, digits: int = 3) -> tuple[nx.DiGraph, dict]:
    """Full pipeline for one real-catalogue institution; writes course features."""
    courses = _load_real_courses(university)
    real_graph, ignored = build_real_only_dag(courses)
    real_with_prereq = sum(1 for n in real_graph.nodes if real_graph.out_degree(n) > 0)
    logger.info(
        "%s real prerequisite density: %d/%d courses with >=1 prereq, %d real edges, "
        "max chain depth %d (ignored %d out-of-set references)",
        university, real_with_prereq, real_graph.number_of_nodes(),
        real_graph.number_of_edges(),
        nx.dag_longest_path_length(real_graph) if real_graph.number_of_edges() else 0,
        ignored,
    )
    graph, _ = build_real_inst_dag(courses, digits)
    summary = dag_summary(graph)
    logger.info("%s final DAG summary: %s", university, summary)
    if summary["augmented_edges"] > summary["real_edges"]:
        raise RuntimeError(
            f"{university}: augmented edges ({summary['augmented_edges']}) exceed real "
            f"edges ({summary['real_edges']}) — STOP (suggests a parsing problem)."
        )
    features = compute_course_features(graph, courses)
    out = _REAL_INST_DIR / university / f"{university}_course_features.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(list(features.values()), fh, indent=2)
    logger.info("Wrote %d %s course-feature records to %s", len(features), university, out)
    return graph, features


def build_khalifa_dag(courses: list[dict[str, Any]]) -> tuple[nx.DiGraph, int]:
    """Hybrid Khalifa (Computer Science) DAG."""
    return build_real_inst_dag(courses)


def build_aus_is_dag(courses: list[dict[str, Any]]) -> tuple[nx.DiGraph, int]:
    """Hybrid AUS (Information Systems) DAG."""
    return build_real_inst_dag(courses)


def build_unc_dag(courses: list[dict[str, Any]]) -> tuple[nx.DiGraph, int]:
    """Hybrid UNC (Economics) DAG."""
    return build_real_inst_dag(courses)
