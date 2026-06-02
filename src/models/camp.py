"""CAMP — constraint-masked RL (MaskablePPO over GraphSAGE) trained & evaluated.

Stage 3 (GNN) + Stage 4 (RL) + Stage 5 (Constraint Engine as action mask). One
policy per institution (different action-space sizes and DAGs => distinct
environments). The action mask comes from :meth:`ConstraintEngine.eligible_courses`,
so CAMP cannot select an illegal course — a *structural* ~0 prerequisite-violation
guarantee at both training and evaluation.

Algorithm choice: **MaskablePPO** (sb3-contrib). SB3 has no native maskable DQN;
MaskablePPO integrates the constraint mask natively and is reliable on CPU. The
task explicitly permits MaskablePPO; the mask (not the optimiser family) is the
architecturally essential piece. Documented in the phase log.

Outputs (under ``data/cap_bench/v3_3inst/``):
  * ``models/camp_{khalifa,aus,unc}.zip`` — trained per-institution policies,
  * ``models/camp_training_curve.csv`` — timestep, episodic & eval reward, and
    the planning vs imitation reward components,
  * ``results/camp_results.json`` — full metric suite (same harness as baselines),
  * ``results/models_summary.json`` — 5 baselines + CAMP in one comparison.

Seed 42; CPU-only; British English.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

from src.constraint_engine.engine import ConstraintEngine, StudentState
from src.models import baselines as bl
from src.models import task
from src.models.rl_env import CoursePlanningEnv, TERM_CAP
from src.utils.seed import set_seed

MODELS_DIR = task.MODELS_DIR
RESULTS_DIR = task.BUNDLE / "results"
EMB_PATH = MODELS_DIR / "embeddings.pt"

SEED = 42
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 15_000
N_EVAL_EPISODES = 40

# v3_3inst: three per-institution policies. The reconstructed programmes carry no
# one_of_groups rule (CORE/SUPPORT/ELECTIVES only), so one_of_rule is empty and
# track is recorded but does not drive a rule-group choice.
INSTITUTIONS = {
    "khalifa": {"programme": "KHAL_BSC_CS_v2024"},
    "aus": {"programme": "AUS_BSBA_IS_v2024"},
    "unc": {"programme": "UNC_BS_ECON_v2024"},
}

PPO_HYPERPARAMS = {
    "algorithm": "MaskablePPO (sb3-contrib)",
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "total_timesteps": TOTAL_TIMESTEPS,
    "seed": SEED,
    "warm_start": True,
    "w_plan": 1.0,    # long-horizon planning reward (strictly > 0: genuine RL)
    "w_imit": 12.0,   # imitation reward on the first post-warm-start term
    "note": "MaskablePPO chosen over DQN: SB3 has no native maskable DQN; "
    "the constraint mask is the essential component and integrates natively. "
    "Phase C.1: warm-start episodes + combined planning+imitation reward.",
}


# ----------------------------------------------------------------- shared data
def _load_emb_map() -> dict[str, np.ndarray]:
    payload = torch.load(EMB_PATH, weights_only=False)
    return {cid: payload["course_emb"][i].numpy() for i, cid in enumerate(payload["course_order"])}


def _institution_courses(uni: str) -> list[str]:
    return sorted(c["course_id"] for c in task.load_courses() if c["university"] == uni)


def _features() -> dict[str, dict[str, Any]]:
    return {f["course_id"]: f for f in json.loads(task.COURSE_FEATURES_PATH.read_text())}


def _engine(uni: str) -> ConstraintEngine:
    return bl._engine_for(uni)


def _student_records(uni: str, split_ids: set[str]) -> list[dict[str, Any]]:
    """Full train/val student records (terms + demographics) for warm-start."""
    students = [json.loads(ln) for ln in (task.STUDENTS_PATH).read_text().splitlines()]
    return [
        {"terms": s["terms"], "demographics": s["demographics"]}
        for s in students
        if s["university"] == uni and s["student_id"] in split_ids
    ]


def make_env(uni: str, split: str, seed: int = SEED, **env_kwargs) -> CoursePlanningEnv:
    cfg = INSTITUTIONS[uni]
    emb_map = _load_emb_map()
    feats = _features()
    cids = _institution_courses(uni)
    course_emb = np.stack([emb_map[c] for c in cids])
    centrality = {c: feats[c]["centrality"] for c in cids}
    split_ids = set(task._load_splits()[split])
    records = _student_records(uni, split_ids)
    return CoursePlanningEnv(
        engine=_engine(uni),
        programme_id=cfg["programme"],
        course_order=cids,
        course_emb=course_emb,
        centrality=centrality,
        student_records=records,
        one_of_rule=cfg.get("one_of_rule", ""),
        track_to_group=cfg.get("tracks", {}),
        seed=seed,
        w_plan=PPO_HYPERPARAMS["w_plan"],
        w_imit=PPO_HYPERPARAMS["w_imit"],
        **env_kwargs,
    )


# ----------------------------------------------------------------- training
class CurveCallback(BaseCallback):
    """Record rollout episodic reward and a periodic greedy val-eval reward."""

    def __init__(self, val_env: CoursePlanningEnv, eval_freq: int, n_eval: int):
        super().__init__()
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.rows: list[tuple[int, float, float, float, float]] = []
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            ep_rew = self._rollout_mean()
            eval_rew = self._evaluate()
            self.rows.append((self.num_timesteps, ep_rew, eval_rew,
                              round(self.last_plan, 4), round(self.last_imit, 4)))
        return True

    def _rollout_mean(self) -> float:
        buf = self.model.ep_info_buffer
        if buf:
            return float(np.mean([e["r"] for e in buf]))
        return float("nan")

    def _evaluate(self) -> float:
        rewards, plan_parts, imit_parts = [], [], []
        for ep in range(self.n_eval):
            obs, _ = self.val_env.reset(seed=10_000 + ep)
            done = False
            total = 0.0
            info = {}
            while not done:
                mask = self.val_env.action_masks()
                if not mask.any():
                    break
                action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                obs, r, term, trunc, info = self.val_env.step(int(action))
                total += r
                done = term or trunc
            rewards.append(total)
            plan_parts.append(info.get("plan_reward", self.val_env._ep_plan))
            imit_parts.append(info.get("imit_reward", self.val_env._ep_imit))
        self.last_plan = float(np.mean(plan_parts))
        self.last_imit = float(np.mean(imit_parts))
        return float(np.mean(rewards))


def train_institution(
    uni: str,
    timesteps: int = TOTAL_TIMESTEPS,
    seed: int = SEED,
    env_kwargs: dict | None = None,
    save_path: Path | None = None,
) -> dict[str, Any]:
    env_kwargs = env_kwargs or {}
    set_seed(seed)
    env = make_env(uni, "train", seed=seed, **env_kwargs)
    val_env = make_env(uni, "val", seed=seed + 1, **env_kwargs)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_HYPERPARAMS["learning_rate"],
        n_steps=PPO_HYPERPARAMS["n_steps"],
        batch_size=PPO_HYPERPARAMS["batch_size"],
        gamma=PPO_HYPERPARAMS["gamma"],
        ent_coef=PPO_HYPERPARAMS["ent_coef"],
        seed=seed,
        verbose=0,
    )
    cb = CurveCallback(val_env, EVAL_FREQ, N_EVAL_EPISODES)
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    model.save(save_path or (MODELS_DIR / f"camp_{uni}.zip"))

    # Convergence: first vs last quartile of the eval-reward column.
    evals = [r[2] for r in cb.rows]
    q = max(1, len(evals) // 4)
    first_q = float(np.mean(evals[:q])) if evals else float("nan")
    last_q = float(np.mean(evals[-q:])) if evals else float("nan")
    return {
        "institution": uni,
        "total_timesteps": timesteps,
        "final_eval_reward": evals[-1] if evals else float("nan"),
        "first_quartile_eval_reward": round(first_q, 4),
        "last_quartile_eval_reward": round(last_q, 4),
        "converged": bool(last_q > first_q),
        "mean_planning_reward": round(getattr(cb, "last_plan", float("nan")), 4),
        "mean_imitation_reward": round(getattr(cb, "last_imit", float("nan")), 4),
        "w_plan": PPO_HYPERPARAMS["w_plan"],
        "w_imit": PPO_HYPERPARAMS["w_imit"],
        "seed": seed,
        "curve": cb.rows,
    }


# ----------------------------------------------------------------- evaluation
_ROLLOUT_DEPTH = 12  # greedy next-course picks used to form the priority ranking


def _camp_scores(
    uni: str, model: MaskablePPO, samples: list[dict[str, Any]], env_kwargs: dict | None = None
) -> np.ndarray:
    """Per-sample global (262-d) priority scores from a greedy sequential rollout.

    CAMP is a sequential planner, so a one-shot action distribution is a poor
    ranking. Instead we roll the deterministic policy forward from the student's
    current state, picking the next courses one at a time (masking each pick out
    so the next-best is chosen), and rank by pick order. The per-term credit cap
    is lifted for this read-out only, so we recover a full priority ordering of
    CAMP's preferred next courses rather than a single legal term. ``env_kwargs``
    must match the training config (e.g. ``use_gnn``) so the observation aligns.
    """
    env = make_env(uni, "test", seed=SEED, **(env_kwargs or {}))
    cids = env.course_order
    vocab = task.load_vocab()
    raw = np.zeros((len(samples), len(vocab)), dtype=float)
    for i, s in enumerate(samples):
        passed = {c: s["history_grades"][c] for c in task.passed_history(s)}
        failed = {c for c in s["history_courses"] if s["history_grades"][c] < 1.0}
        env.state = StudentState(
            programme_id=env.programme_id, passed=dict(passed), failed=set(failed),
            per_term_credit_cap=10_000,  # lifted: we want a ranking, not a real term
        )
        env._best_grade = {c: s["history_grades"][c] for c in s["history_courses"]}
        env._terms_used = int(s["term_index"])
        env._one_of_choice = env._make_one_of_choice(s["demographics"]["major_track"])
        env._refresh_caches()
        for rank in range(_ROLLOUT_DEPTH):
            mask = env.action_masks()
            if not mask.any():
                break
            obs = env._obs()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            local = int(action)
            cid = cids[local]
            # Descending priority score by pick order (first pick = highest).
            raw[i, vocab[cid]] = float(_ROLLOUT_DEPTH - rank)
            env.state.term_selected.add(cid)  # mask out; pick the next-best next
    return raw


def camp_raw_scores(
    models: dict[str, MaskablePPO], samples_test: list[dict[str, Any]], env_kwargs: dict | None = None
) -> np.ndarray:
    """Global (N×262) priority scores for CAMP over all test samples."""
    raw = np.zeros((len(samples_test), len(task.load_vocab())), dtype=float)
    for uni, model in models.items():
        idx = [i for i, s in enumerate(samples_test) if s["university"] == uni]
        sub = [samples_test[i] for i in idx]
        sub_scores = _camp_scores(uni, model, sub, env_kwargs)
        for k, i in enumerate(idx):
            raw[i] = sub_scores[k]
    return raw


def evaluate_camp(
    models: dict[str, MaskablePPO],
    samples_test: list[dict[str, Any]],
    engines,
    env_kwargs: dict | None = None,
    eligible_mask: bool = True,
    name: str = "camp",
) -> dict[str, Any]:
    raw = camp_raw_scores(models, samples_test, env_kwargs)
    return bl.evaluate_model(name, PPO_HYPERPARAMS, raw, samples_test, engines, eligible_mask=eligible_mask)


# ----------------------------------------------------------------- orchestration
def main(timesteps: int = TOTAL_TIMESTEPS) -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    engines = {uni: bl._engine_for(uni) for uni in INSTITUTIONS}

    # --- Train per institution ---
    training = {}
    models = {}
    curve_rows = []
    for uni in INSTITUTIONS:
        print(f"Training CAMP [{uni}] for {timesteps} timesteps ...")
        info = train_institution(uni, timesteps)
        training[uni] = {k: v for k, v in info.items() if k != "curve"}
        for ts, ep, ev, pl, im in info["curve"]:
            curve_rows.append((uni, ts, ep, ev, pl, im))
        models[uni] = MaskablePPO.load(MODELS_DIR / f"camp_{uni}.zip")
        print(f"  [{uni}] first-q eval {info['first_quartile_eval_reward']} -> "
              f"last-q {info['last_quartile_eval_reward']} converged={info['converged']}")

    with (MODELS_DIR / "camp_training_curve.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["institution", "timestep", "episodic_reward", "eval_reward",
                    "mean_planning_reward", "mean_imitation_reward"])
        w.writerows(curve_rows)

    # --- Evaluate CAMP on the SAME test split + harness as baselines ---
    samples_test = task.load_samples("test")
    print(f"Evaluating CAMP on {len(samples_test)} test samples ...")
    camp = evaluate_camp(models, samples_test, engines)
    camp["training"] = training
    with (RESULTS_DIR / "camp_results.json").open("w") as fh:
        json.dump(camp, fh, indent=2)
    print(f"  CAMP NDCG@10={camp['ranking_metrics']['NDCG@10']} "
          f"viol={camp['prereq_violation_rate_topk']} "
          f"grad_compliance={camp['graduation_compliance']} "
          f"pathway_feasibility={camp['pathway_feasibility']}")

    # --- Comparison: re-run the 5 baselines in-memory (augmented metrics),
    #     leaving baseline_*.json untouched, and add CAMP. ---
    split_of = task._student_to_split()
    train_students = [s for s in task._load_students() if split_of.get(s["student_id"]) == "train"]
    samples_train = task.load_samples("train")
    samples_val = task.load_samples("val")

    print("Re-scoring baselines for the augmented comparison ...")
    baseline_results = {
        "collaborative_filtering": bl.run_collaborative_filtering(train_students, samples_test, engines),
        "matrix_factorisation": bl.run_matrix_factorisation(train_students, samples_test, engines),
        "random_forest": bl.run_random_forest(samples_train, samples_test, engines),
        "deep_nn": bl.run_deep_nn(samples_train, samples_val, samples_test, engines),
        "pure_gnn": bl.run_pure_gnn(samples_test, engines),
    }

    def slim(r):
        return {
            "ranking_metrics": r["ranking_metrics"],
            "ndcg10_by_university": r.get("ndcg10_by_university", {}),
            "classification_metrics": r["classification_metrics"],
            "prereq_violation_rate_topk": r["prereq_violation_rate_topk"],
            "graduation_compliance": r["graduation_compliance"],
            "pathway_feasibility": r["pathway_feasibility"],
        }

    summary = {
        "task": "next-term course-set prediction",
        "n_test_samples": len(samples_test),
        "models": {name: slim(r) for name, r in baseline_results.items()},
    }
    summary["models"]["camp"] = slim(camp)
    with (RESULTS_DIR / "models_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {RESULTS_DIR / 'models_summary.json'}")


if __name__ == "__main__":
    main()
