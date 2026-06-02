"""Phase D Part 4 — explainability (XAI) for CAMP recommendations (Stage 6).

Three mechanisms:
  (a) Feature attribution — **permutation importance** over the policy's state
      feature groups (embedding / completed-courses / term / GPA /
      remaining-requirements / credits). SHAP is awkward on a MaskablePPO policy
      with action masking, so permutation importance is used and documented as
      the substitution. Importance = fraction of decisions whose top recommended
      course changes when a feature group is shuffled across samples.
  (b) Graph-path reasoning — for a recommended course, the prerequisite chain the
      student satisfied to unlock it, and what it unlocks downstream (DAG
      dependents + centrality).
  (c) Rule trace — masked-out courses and the unmet prerequisite that blocks
      each ("why NOT recommended"), which the baselines cannot provide.

Plus three end-to-end case studies. Writes ``results/explainability.json``.
Seed 42; British English.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from src.constraint_engine import dag
from src.constraint_engine.engine import ConstraintEngine, StudentState
from src.models import baselines as bl
from src.models import camp
from src.models import task

RESULTS_DIR = task.BUNDLE / "results"
SEED = 42
N_IMPORTANCE_SAMPLES = 250


def _feature_groups(env) -> dict[str, tuple[int, int]]:
    n, b = env.n_courses, len(env.required_blocks)
    e = env.emb_dim
    return {
        "gnn_embedding": (0, e),
        "completed_courses": (e, e + n),
        "term_index": (e + n, e + n + 1),
        "gpa": (e + n + 1, e + n + 2),
        "remaining_requirements": (e + n + 2, e + n + 2 + b),
        "credits_completed": (e + n + 2 + b, e + n + 3 + b),
    }


def _state_from_sample(env, s) -> None:
    passed = {c: s["history_grades"][c] for c in task.passed_history(s)}
    env.state = StudentState(programme_id=env.programme_id, passed=dict(passed),
                             failed={c for c in s["history_courses"] if s["history_grades"][c] < 1.0},
                             per_term_credit_cap=10_000)
    env._best_grade = {c: s["history_grades"][c] for c in s["history_courses"]}
    env._terms_used = int(s["term_index"])
    env._one_of_choice = env._make_one_of_choice(s["demographics"]["major_track"])
    env._refresh_caches()


def _top_action(model, obs, mask) -> int:
    a, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(a)


def permutation_importance(uni, model, samples, rng) -> dict[str, float]:
    env = camp.make_env(uni, "test", seed=SEED)
    groups = _feature_groups(env)
    obs_list, mask_list, top_list = [], [], []
    for s in samples:
        _state_from_sample(env, s)
        mask = env.action_masks()
        if not mask.any():
            continue
        obs = env._obs()
        obs_list.append(obs)
        mask_list.append(mask)
        top_list.append(_top_action(model, obs, mask))
    obs_arr = np.array(obs_list)
    importance = {}
    for g, (lo, hi) in groups.items():
        perm = rng.permutation(len(obs_arr))
        changed = 0
        for i in range(len(obs_arr)):
            o = obs_arr[i].copy()
            o[lo:hi] = obs_arr[perm[i]][lo:hi]  # shuffle this group across samples
            if _top_action(model, o, mask_list[i]) != top_list[i]:
                changed += 1
        importance[g] = round(changed / len(obs_arr), 4)
    return importance


def _build_dag(uni):
    # v3_3inst institutions are all 3-digit; hybrid real+augmentation DAG.
    courses = [c for c in task.load_courses() if c["university"] == uni]
    return dag.build_real_inst_dag(courses, 3)[0]


def graph_path(graph, feats, course_id, passed: set[str]) -> dict[str, Any]:
    prereqs = list(graph.successors(course_id))  # course -> prereq
    return {
        "course": course_id,
        "prerequisites": prereqs,
        "prerequisites_satisfied_by_history": [p for p in prereqs if p in passed],
        "unlocks_downstream": list(graph.predecessors(course_id))[:8],
        "centrality": feats.get(course_id, {}).get("centrality"),
        "blocking_factor": feats.get(course_id, {}).get("blocking_factor"),
    }


def masked_reasons(engine, sample, k=2) -> list[dict[str, Any]]:
    passed = set(task.passed_history(sample))
    uni = sample["university"]
    out = []
    for cid, course in engine.courses_by_id.items():
        if cid in passed:
            continue
        if engine.check_prerequisites(cid, passed):
            continue  # eligible -> not masked for prereq reasons
        # Find the first unsatisfied AND-group.
        for grp in engine.prereqs.get(cid, []):
            if not any(m in passed for m in grp):
                out.append({
                    "course": cid,
                    "blocked_reason": "unmet prerequisite",
                    "needs_one_of": grp,
                })
                break
        if len(out) >= k:
            break
    return out


def case_study(uni, model, engine, graph, feats, sample) -> dict[str, Any]:
    # CAMP's recommended next set (top-m by sequential rollout, m = actual size).
    m = max(1, len(sample["target_courses"]))
    raw = camp._camp_scores(uni, model, [sample])[0]
    vocab = task.load_vocab()
    idx2cid = {i: c for c, i in vocab.items()}
    passed = set(task.passed_history(sample))
    order = np.argsort(-raw)
    recommended = [idx2cid[j] for j in order if raw[j] > 0][:m]
    top = recommended[0] if recommended else None
    return {
        "student_id": sample["student_id"],
        "university": uni,
        "term_index": sample["term_index"],
        "profile": {
            "start_state_type": sample["demographics"]["start_state_type"],
            "major_track": sample["demographics"]["major_track"],
            "gpa_band": sample["demographics"]["gpa_band"],
            "cold_start": sample["demographics"]["cold_start"],
        },
        "n_passed": len(passed),
        "recommended_next_term": recommended,
        "actual_next_term": sample["target_courses"],
        "top_recommendation": top,
        "graph_path_for_top": graph_path(graph, feats, top, passed) if top else None,
        "masked_out_examples": masked_reasons(engine, sample, k=2),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    institutions = list(camp.INSTITUTIONS)  # khalifa, aus, unc
    engines = {u: bl._engine_for(u) for u in institutions}
    models = {u: MaskablePPO.load(task.MODELS_DIR / f"camp_{u}.zip") for u in institutions}
    feats = {f["course_id"]: f for f in json.loads(task.COURSE_FEATURES_PATH.read_text())}
    graphs = {u: _build_dag(u) for u in institutions}
    samples_test = task.load_samples("test")

    # (a) Feature importance, per institution + overall.
    importance = {}
    for uni in institutions:
        sub = [s for s in samples_test if s["university"] == uni]
        sub = [sub[i] for i in rng.permutation(len(sub))[:N_IMPORTANCE_SAMPLES]]
        importance[uni] = permutation_importance(uni, models[uni], sub, rng)
    groups = list(importance[institutions[0]].keys())
    overall = {g: round(float(np.mean([importance[u][g] for u in institutions])), 4) for g in groups}
    top5 = sorted(overall.items(), key=lambda kv: -kv[1])[:5]

    # (b,c) Three diverse case studies.
    def find(pred):
        for s in samples_test:
            if pred(s):
                return s
        return None

    # One case study per field: a new Khalifa CS student, a mid-degree AUS IS
    # student, and a UNC Econ student (transfer where available).
    cs_specs = [
        ("khalifa", lambda s: s["university"] == "khalifa" and s["demographics"]["start_state_type"] == "new" and 1 <= s["term_index"] <= 3),
        ("aus", lambda s: s["university"] == "aus" and s["demographics"]["start_state_type"] == "mid_degree"),
        ("unc", lambda s: s["university"] == "unc" and s["demographics"]["start_state_type"] == "transfer"),
    ]
    case_studies = []
    for uni_hint, pred in cs_specs:
        s = find(pred)
        if s is None:
            continue
        uni = s["university"]
        case_studies.append(case_study(uni, models[uni], engines[uni], graphs[uni], feats, s))

    out = {
        "feature_importance_method": "permutation importance (fraction of decisions whose top "
        "recommendation changes when a feature group is shuffled); SHAP substituted as it is "
        "awkward on a masked policy.",
        "feature_importance_per_institution": importance,
        "feature_importance_overall": overall,
        "top5_features": [{"feature": g, "importance": v} for g, v in top5],
        "n_case_studies": len(case_studies),
        "case_studies": case_studies,
    }
    with (RESULTS_DIR / "explainability.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {RESULTS_DIR / 'explainability.json'}")
    print("top-5 features:", top5)
    print(f"case studies saved: {len(case_studies)}")


if __name__ == "__main__":
    main()
