"""Phase D Part 2 — statistical significance and stability.

Two analyses, both on the test split:
  1. **Stability:** retrain + evaluate CAMP-full under 5 seeds and report
     mean ± std for NDCG@10, violation rate, graduation-compliance, feasibility.
     (Seed runs use a reduced 100k steps/institution to fit the CPU budget; the
     finalised CAMP-full uses 200k — documented.) Baselines with stochastic
     training (Random Forest, Deep NN) are also re-seeded; CF and MF are
     deterministic and noted as such.
  2. **Significance:** per-sample NDCG@10 (n=7105) for the finalised models, then
     paired t-test and Wilcoxon signed-rank (CAMP vs each baseline) and a one-way
     ANOVA across all six models. Violation rate is deterministically 0 for CAMP
     (no variance) — framed accordingly.

Writes ``results/statistical_tests.json``. Seeds explicit; British English.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import stats
from sb3_contrib import MaskablePPO

from src.constraint_engine.engine import ConstraintEngine
from src.models import baselines as bl
from src.models import camp
from src.models import metrics as metrics_mod
from src.models import task

RESULTS_DIR = task.BUNDLE / "results"
SEEDS = [42, 123, 2024, 7, 99]
SEED_TIMESTEPS = 100_000


def _agg(metrics_list: list[dict[str, float]]) -> dict[str, Any]:
    out = {}
    for key in ("NDCG@10", "violation_rate", "graduation_compliance", "pathway_feasibility"):
        vals = [m[key] for m in metrics_list]
        out[key] = {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6),
                    "per_seed": [round(v, 6) for v in vals]}
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    institutions = list(camp.INSTITUTIONS)  # khalifa, aus, unc
    engines = {u: bl._engine_for(u) for u in institutions}
    samples_test = task.load_samples("test")
    split_of = task._student_to_split()
    train_students = [s for s in task._load_students() if split_of.get(s["student_id"]) == "train"]
    samples_train = task.load_samples("train")
    samples_val = task.load_samples("val")

    # ---- 1a. CAMP stability over 5 seeds (retrain) ----
    camp_seed_metrics = []
    for seed in SEEDS:
        print(f"CAMP seed {seed} ({SEED_TIMESTEPS} steps/inst) ...")
        models = {}
        for uni in institutions:
            save = task.MODELS_DIR / f"camp_seed{seed}_{uni}.zip"
            camp.train_institution(uni, SEED_TIMESTEPS, seed=seed, save_path=save)
            models[uni] = MaskablePPO.load(save)
        m = camp.evaluate_camp(models, samples_test, engines, eligible_mask=True)
        camp_seed_metrics.append({
            "NDCG@10": m["ranking_metrics"]["NDCG@10"],
            "violation_rate": m["prereq_violation_rate_topk"],
            "graduation_compliance": m["graduation_compliance"],
            "pathway_feasibility": m["pathway_feasibility"],
        })
        print(f"  seed {seed}: NDCG@10={m['ranking_metrics']['NDCG@10']:.4f} viol={m['prereq_violation_rate_topk']}")

    # ---- 1b. Stochastic baselines over 5 seeds ----
    baseline_seed = {"random_forest": [], "deep_nn": []}
    for seed in SEEDS:
        rf_raw = bl.scores_random_forest(samples_train, samples_test, seed=seed)
        rk, tg = bl.rank_samples(rf_raw, samples_test, engines, eligible_mask=False)
        baseline_seed["random_forest"].append(float(metrics_mod.ndcg_at_k_per_sample(rk, tg, 10).mean()))
        dnn_raw, _ = bl.scores_deep_nn(samples_train, samples_val, samples_test, seed=seed)
        rk, tg = bl.rank_samples(dnn_raw, samples_test, engines, eligible_mask=False)
        baseline_seed["deep_nn"].append(float(metrics_mod.ndcg_at_k_per_sample(rk, tg, 10).mean()))

    # ---- 2. Per-sample NDCG@10 for the finalised models + significance tests ----
    print("Per-sample significance tests (n=%d) ..." % len(samples_test))
    persample = {}
    # CAMP-full (finalised 200k policies).
    camp_models = {u: MaskablePPO.load(task.MODELS_DIR / f"camp_{u}.zip") for u in institutions}
    camp_raw = camp.camp_raw_scores(camp_models, samples_test)
    rk, tg = bl.rank_samples(camp_raw, samples_test, engines, eligible_mask=True)
    persample["camp"] = metrics_mod.ndcg_at_k_per_sample(rk, tg, 10)
    # Baselines.
    raw_by_model = {
        "collaborative_filtering": bl.scores_collaborative_filtering(train_students, samples_test),
        "matrix_factorisation": bl.scores_matrix_factorisation(train_students, samples_test),
        "random_forest": bl.scores_random_forest(samples_train, samples_test),
        "deep_nn": bl.scores_deep_nn(samples_train, samples_val, samples_test)[0],
        "pure_gnn": bl.scores_pure_gnn(samples_test),
    }
    for name, raw in raw_by_model.items():
        rk, tg = bl.rank_samples(raw, samples_test, engines, eligible_mask=False)
        persample[name] = metrics_mod.ndcg_at_k_per_sample(rk, tg, 10)

    camp_ps = persample["camp"]
    pairwise = {}
    for name in raw_by_model:
        t_stat, t_p = stats.ttest_rel(camp_ps, persample[name])
        # Wilcoxon needs non-zero differences; guard the degenerate case.
        try:
            w_stat, w_p = stats.wilcoxon(camp_ps, persample[name])
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        pairwise[f"camp_vs_{name}"] = {
            "camp_mean_ndcg10": round(float(camp_ps.mean()), 6),
            "other_mean_ndcg10": round(float(persample[name].mean()), 6),
            "paired_t_stat": round(float(t_stat), 4), "paired_t_p": float(t_p),
            "wilcoxon_stat": round(float(w_stat), 4), "wilcoxon_p": float(w_p),
        }
    f_stat, f_p = stats.f_oneway(*[persample[m] for m in ["camp", *raw_by_model.keys()]])

    out = {
        "n_test_samples": len(samples_test),
        "seeds": SEEDS,
        "seed_timesteps": SEED_TIMESTEPS,
        "note": "5-seed CAMP stability uses 100k steps/inst (vs 200k finalised) for CPU budget; "
        "this stability mean is DISTINCT from the 200k finalised CAMP-full NDCG@10 (0.637). "
        "CF and MF are deterministic (no training randomness). Significance tests are "
        "per-sample paired (n=n_test) on the finalised models.",
        "camp_stability_over_seeds": _agg(camp_seed_metrics),
        "baseline_ndcg10_over_seeds": {
            k: {"mean": round(float(np.mean(v)), 6), "std": round(float(np.std(v)), 6), "per_seed": [round(x, 6) for x in v]}
            for k, v in baseline_seed.items()
        },
        "camp_violation_rate_variance": 0.0,
        "significance_per_sample_ndcg10": pairwise,
        "anova_six_models": {"F": round(float(f_stat), 4), "p": float(f_p)},
    }
    with (RESULTS_DIR / "statistical_tests.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {RESULTS_DIR / 'statistical_tests.json'}")


if __name__ == "__main__":
    main()
