"""Stage 4/5 — constraint-masked RL environment for CAMP (Gymnasium).

One episode = planning a single synthetic student's degree term by term for one
institution. The agent picks one course per step; the **action mask comes from
:meth:`ConstraintEngine.eligible_courses`**, so an illegal course (unmet
prerequisite, already passed, or breaching the per-term credit cap) can never be
selected. This is why CAMP attains ~0 prerequisite violations *by construction*.

State (observation), concatenated:
  * inductive GraphSAGE student embedding (mean of passed-course embeddings; the
    mean of all institution course embeddings when the transcript is empty) — 64-d,
  * completed-courses multi-hot over the institution's course set,
  * current term index (scaled by the term cap),
  * current GPA (scaled by 4),
  * unmet required rule-group count per required block (scaled),
  * credits completed (scaled by the programme total).

Reward (dense) — weights are module constants and recorded in the phase log.

British English throughout. CPU-only; deterministic given the seed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.constraint_engine.engine import ConstraintEngine, StudentState

# Grade model + term-load shapes mirror the simulator so episodes match the
# dataset's realistic distribution.
from src.dataset.simulator import (
    BAND_MEANS,
    GRADE_SD,
    MIN_PASS_GRADE,
    TERM_LOAD_RANGE,
    TERM_LOAD_TYPES,
    TERM_LOAD_P,
    SUMMER_RANGE,
    SUMMER_FRACTION,
)

TERM_CAP = 14
HEALTHY_TERM_CREDITS = 15

# --- PLANNING reward sub-weights (the long-horizon graduation signal). ---
W_REQ = 1.0          # picking a course that counts toward an unmet required rule
W_CENT = 0.5         # downstream-unlock bonus ~ normalised DAG centrality
W_RULE_DONE = 2.0    # each required rule group newly satisfied this term
W_GPA = 0.3          # healthy term GPA (>= 2.0)
W_OVERLOAD = 0.5     # term credits above the healthy band
W_GRAD = 10.0        # terminal graduation bonus (scaled by speed)
W_UNMET = 0.5        # terminal penalty per unmet required rule group at the cap

# --- IMITATION reward (Phase C.1): on the first post-warm-start term, reward
# picks that are in the student's actual next term and gently penalise off-target
# picks, so the eval-relevant first-term decision aligns with the real set. ---
W_IMIT_MATCH = 1.0
W_IMIT_MISS = 0.5

# --- Top-level combination weights (total = w_plan * plan + w_imit * imit). ---
# Both strictly positive: CAMP stays a genuine RL planner, not a behaviour-cloner.
# The component breakdown (planning reward >> imitation reward in magnitude, as
# planning accrues over every term while imitation scores only the first) is the
# real safeguard that planning stays active — confirmed in the phase log.
DEFAULT_W_PLAN = 1.0
DEFAULT_W_IMIT = 12.0


class CoursePlanningEnv(gym.Env):
    """Single-student, single-institution multi-term planning episode.

    Reward = ``w_plan * planning_reward + w_imit * imitation_reward``. The
    planning reward is the long-horizon graduation signal; the imitation reward
    (Phase C.1) is earned on the first post-warm-start term for picking courses
    in the student's actual next-term set. Reward components and the warm-start
    / mask behaviour are toggleable via the constructor flags so Phase D can
    ablate "no planning", "no imitation", "no mask".
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        engine: ConstraintEngine,
        programme_id: str,
        course_order: list[str],
        course_emb: np.ndarray,
        centrality: dict[str, float],
        student_records: list[dict[str, Any]],
        one_of_rule: str = "",
        track_to_group: dict[str, str] | None = None,
        seed: int = 42,
        *,
        warm_start: bool = True,
        use_planning: bool = True,
        use_imitation: bool = True,
        use_mask: bool = True,
        use_gnn: bool = True,
        w_plan: float = DEFAULT_W_PLAN,
        w_imit: float = DEFAULT_W_IMIT,
    ):
        super().__init__()
        self.engine = engine
        self.programme_id = programme_id
        self.course_order = course_order  # local idx -> course_id (sorted)
        self.cid_to_local = {c: i for i, c in enumerate(course_order)}
        self.n_courses = len(course_order)
        self.course_emb = course_emb.astype(np.float32)  # (n_courses, 64)
        self.emb_dim = course_emb.shape[1]
        self.mean_emb = self.course_emb.mean(axis=0)
        max_cent = max(centrality.values()) or 1.0
        self.norm_cent = np.array(
            [centrality[c] / max_cent for c in course_order], dtype=np.float32
        )
        self.student_records = student_records
        self.one_of_rule = one_of_rule
        self.track_to_group = track_to_group or {}
        self.required_blocks = engine._required_blocks(programme_id)
        self.total_required = engine.total_credits_required(programme_id)

        self.warm_start = warm_start
        self.use_planning = use_planning
        self.use_imitation = use_imitation
        self.use_mask = use_mask
        self.use_gnn = use_gnn
        self.w_plan = w_plan
        self.w_imit = w_imit

        self.action_space = spaces.Discrete(self.n_courses)
        obs_dim = self.emb_dim + self.n_courses + 1 + 1 + len(self.required_blocks) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.state: StudentState | None = None

    # ------------------------------------------------------------------ helpers
    def _sample_term_target(self) -> int:
        if self._is_summer_taker and (self._terms_used % 3 == 2):
            lo, hi = SUMMER_RANGE
        else:
            lo, hi = TERM_LOAD_RANGE[self._term_load_type]
        return int(self._rng.integers(lo, hi + 1))

    def _grade(self) -> float:
        return float(np.clip(self._rng.normal(BAND_MEANS[self._gpa_band], GRADE_SD), 0.0, 4.0))

    def _eligible_cids(self) -> set[str]:
        return self.engine.eligible_courses(self.state)

    def _make_one_of_choice(self, track: str) -> dict[str, str]:
        """Map a one_of_groups rule to the chosen member group, when one exists."""
        if self.one_of_rule and track in self.track_to_group:
            return {self.one_of_rule: self.track_to_group[track]}
        return {}

    def _refresh_caches(self) -> None:
        """Recompute requirement caches from the passed set (term boundaries only).

        ``remaining_requirements`` and ``unmet_required_rule_courses`` depend only
        on ``state.passed``, which is constant while a term is being filled, so
        they are cached per term rather than recomputed every step.
        """
        self._rem_cache = self.engine.remaining_requirements(self.state)
        self._unmet_cache = self.engine.unmet_required_rule_courses(
            set(self.state.passed), one_of_choice=self._one_of_choice
        )

    def _n_unmet_groups(self) -> int:
        return sum(
            len(v["unmet_groups"])
            for v in self._rem_cache.values()
            if isinstance(v, dict) and "unmet_groups" in v
        )

    def _student_embedding(self) -> np.ndarray:
        if not self.use_gnn:
            # No-GNN ablation: zero the GraphSAGE embedding so the policy relies
            # only on the raw features (completed multi-hot + scalars).
            return np.zeros(self.emb_dim, dtype=np.float32)
        passed = [self.cid_to_local[c] for c in self.state.passed if c in self.cid_to_local]
        if passed:
            return self.course_emb[passed].mean(axis=0)
        return self.mean_emb

    def _gpa(self) -> float:
        if not self._best_grade:
            return 0.0
        return float(np.mean(list(self._best_grade.values())))

    def _obs(self) -> np.ndarray:
        multihot = np.zeros(self.n_courses, dtype=np.float32)
        for c in self.state.passed:
            if c in self.cid_to_local:
                multihot[self.cid_to_local[c]] = 1.0
        rem = self._rem_cache
        block_unmet = np.array(
            [
                len(rem.get(b, {}).get("unmet_groups", [])) if isinstance(rem.get(b), dict) else 0
                for b in self.required_blocks
            ],
            dtype=np.float32,
        )
        block_unmet = block_unmet / 5.0  # coarse scaling
        credits = self.engine.passed_credits(self.state) / self.total_required
        return np.concatenate(
            [
                self._student_embedding(),
                multihot,
                np.array([self._terms_used / TERM_CAP], dtype=np.float32),
                np.array([self._gpa() / 4.0], dtype=np.float32),
                block_unmet,
                np.array([credits], dtype=np.float32),
            ]
        ).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Boolean mask of legal actions (the constraint engine's eligible set).

        With ``use_mask=False`` (Phase D ablation only) every not-passed course
        is allowed — which is what lets violations appear; the default keeps the
        constraint mask and the structural 0-violation guarantee.
        """
        if not self.use_mask:
            mask = np.ones(self.n_courses, dtype=bool)
            for c in self.state.passed:
                if c in self.cid_to_local:
                    mask[self.cid_to_local[c]] = False
            return mask
        eligible = self._eligible_cids()
        mask = np.zeros(self.n_courses, dtype=bool)
        for c in eligible:
            mask[self.cid_to_local[c]] = True
        return mask

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rec = self.student_records[int(self._rng.integers(0, len(self.student_records)))]
        demo = rec["demographics"]
        self._gpa_band = demo["gpa_band"]
        self._term_load_type = demo["term_load_type"]
        self._track = demo["major_track"]
        self._is_summer_taker = bool(self._rng.random() < SUMMER_FRACTION)
        # one_of_groups choice only where a programme models one;
        # the three v3_3inst programmes have no one_of rule, so the choice is empty.
        self._one_of_choice = self._make_one_of_choice(self._track)

        self.state = StudentState(programme_id=self.programme_id, per_term_credit_cap=18)
        self._best_grade = {}
        self._ep_plan = 0.0
        self._ep_imit = 0.0
        self._imitation_target: set[str] = set()
        self._imitation_active = False

        terms = sorted(rec["terms"], key=lambda t: t["term_index"])
        # Warm-start: cut at a random term so the agent begins from a realistic
        # mid-degree state matching the eval query distribution (every term k>=1).
        if self.warm_start and len(terms) >= 2:
            pos = int(self._rng.integers(1, len(terms)))
        else:
            pos = 0
        for t in terms[:pos]:
            for c in t["courses"]:
                cid = c["course_id"]
                self._best_grade[cid] = max(self._best_grade.get(cid, -1.0), c["grade"])
                if c["passed"]:
                    self.state.passed[cid] = max(self.state.passed.get(cid, 0.0), c["grade"])
                else:
                    self.state.failed.add(cid)
        self._terms_used = pos
        self.state.current_term = pos
        self._refresh_caches()

        # Imitation target = the student's actual next term (ground truth exists).
        if self.use_imitation and pos < len(terms):
            self._imitation_target = {c["course_id"] for c in terms[pos]["courses"]}
            self._imitation_active = True
            self._term_target = max(1, len(self._imitation_target))  # mirror eval top-m
        else:
            self._term_target = self._sample_term_target()
        return self._obs(), {}

    def _roll_term(self) -> float:
        """Assign grades to the current term's selected courses; advance term."""
        reward = 0.0
        groups_before = self._n_unmet_groups()  # from the pre-roll cache
        term_credits = 0
        grades = []
        for cid in list(self.state.term_selected):
            g = self._grade()
            grades.append(g)
            term_credits += self.engine.credits_of(cid)
            self._best_grade[cid] = max(self._best_grade.get(cid, -1.0), g)
            if g >= MIN_PASS_GRADE:
                self.state.passed[cid] = max(self.state.passed.get(cid, 0.0), g)
                self.state.failed.discard(cid)
            else:
                self.state.failed.add(cid)
        # Term-level shaping.
        if grades and float(np.mean(grades)) >= 2.0:
            reward += W_GPA
        if term_credits > HEALTHY_TERM_CREDITS:
            reward += -W_OVERLOAD

        self._terms_used += 1
        self.state.current_term += 1
        self.state.term_selected = set()
        self._term_target = self._sample_term_target()
        self._refresh_caches()  # passed set changed -> refresh requirement caches
        groups_after = self._n_unmet_groups()
        reward += W_RULE_DONE * max(0, groups_before - groups_after)
        return reward

    def step(self, action: int):
        cid = self.course_order[int(action)]
        if self.use_mask and cid not in self._eligible_cids():  # pragma: no cover
            return self._obs(), -1.0, True, False, {"illegal": True}

        plan_r = 0.0
        imit_r = 0.0
        # --- planning reward (per course) ---
        if cid in self._unmet_cache:
            plan_r += W_REQ
        plan_r += W_CENT * float(self.norm_cent[self.cid_to_local[cid]])
        # --- imitation reward (first post-warm-start term only) ---
        if self._imitation_active:
            imit_r += W_IMIT_MATCH if cid in self._imitation_target else -W_IMIT_MISS
        self.state.term_selected.add(cid)

        terminated = False
        # Roll when the load target is met, or when no legal action remains. With
        # the mask off, action_masks() (not-passed courses) is the cheap check; it
        # avoids a redundant eligible_courses() pass per step.
        term_full = len(self.state.term_selected) >= self._term_target
        if term_full or not self.action_masks().any():
            plan_r += self._roll_term()
            # Imitation only scores the first term after warm-start.
            self._imitation_active = False
            if self.engine.is_graduated(self.state):
                speed = (TERM_CAP - self._terms_used) / TERM_CAP
                plan_r += W_GRAD * (1.0 + max(0.0, speed))
                terminated = True
            elif self._terms_used >= TERM_CAP:
                plan_r += -W_UNMET * self._n_unmet_groups()
                terminated = True
            elif not self.action_masks().any():
                plan_r += -W_UNMET * self._n_unmet_groups()
                terminated = True

        plan_c = self.w_plan * plan_r if self.use_planning else 0.0
        imit_c = self.w_imit * imit_r if self.use_imitation else 0.0
        self._ep_plan += plan_c
        self._ep_imit += imit_c
        reward = plan_c + imit_c
        info = {}
        if terminated:
            info = {"plan_reward": self._ep_plan, "imit_reward": self._ep_imit}
        return self._obs(), float(reward), terminated, False, info
