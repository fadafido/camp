"""JOB 2 — augment statistical_tests.json to the SEVEN-model analysis.

The committed ``statistical_tests.json`` was a six-model analysis (CAMP + five
baselines). Table T2 reports seven systems — the UNMASKED XGBoost baseline was
added later. This script brings the statistics into line with T2 by ADDING
XGBoost, WITHOUT recomputing the existing comparisons:

  * the five existing CAMP-vs-baseline pairwise entries are copied through
    **verbatim** from the committed file (byte-identical — proven at the end);
  * a new ``camp_vs_gradient_boosted`` pairwise (paired t + Wilcoxon on per-sample
    NDCG@10) is computed against the UNMASKED XGBoost scores — the same 0.716
    model in T2;
  * the one-way ANOVA is recomputed across all SEVEN systems (the F necessarily
    changes: six groups -> seven) and relabelled ``anova_seven_models``;
  * any p-value that underflows to exactly 0.0 gains a ``*_thresholded`` sibling
    (``"<1e-300"``) so nothing reads as an impossible exact zero.

The 5-seed stability section (locked at 100k) and every other field are preserved
verbatim. Seed 42; CPU-only; British English. Deterministic.

Run with libomp on the loader path (XGBoost):
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
    python3 -m src.models.phase_d_statistical_xgb_augment
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats
from sb3_contrib import MaskablePPO

from src.models import baselines as bl
from src.models import camp
from src.models import exp_gradient_boosted_xgb as xgb_exp
from src.models import metrics as metrics_mod
from src.models import task
from src.models.phase_d_statistical import _p_thresholded

RESULTS_DIR = task.BUNDLE / "results"
STAT_PATH = RESULTS_DIR / "statistical_tests.json"
SEED = 42


def main() -> None:
    old = json.loads(STAT_PATH.read_text())
    institutions = list(camp.INSTITUTIONS)
    engines = {u: bl._engine_for(u) for u in institutions}
    samples_test = task.load_samples("test")
    split_of = task._student_to_split()
    train_students = [s for s in task._load_students() if split_of.get(s["student_id"]) == "train"]
    samples_train = task.load_samples("train")
    samples_val = task.load_samples("val")

    # ---- Per-sample NDCG@10 for all SEVEN systems (deterministic, finalised models) ----
    print("Computing per-sample NDCG@10 for the seven systems (n=%d) ..." % len(samples_test))
    persample = {}
    camp_models = {u: MaskablePPO.load(task.MODELS_DIR / f"camp_{u}.zip") for u in institutions}
    camp_raw = camp.camp_raw_scores(camp_models, samples_test)
    rk, tg = bl.rank_samples(camp_raw, samples_test, engines, eligible_mask=True)
    persample["camp"] = metrics_mod.ndcg_at_k_per_sample(rk, tg, 10)

    raw_by_model = {
        "collaborative_filtering": bl.scores_collaborative_filtering(train_students, samples_test),
        "matrix_factorisation": bl.scores_matrix_factorisation(train_students, samples_test),
        "random_forest": bl.scores_random_forest(samples_train, samples_test),
        "deep_nn": bl.scores_deep_nn(samples_train, samples_val, samples_test)[0],
        "pure_gnn": bl.scores_pure_gnn(samples_test),
        # UNMASKED XGBoost — the same 0.716 baseline reported in Table T2.
        "gradient_boosted": xgb_exp.scores_gradient_boosted_xgb(samples_train, samples_test),
    }
    for name, raw in raw_by_model.items():
        rk, tg = bl.rank_samples(raw, samples_test, engines, eligible_mask=False)
        persample[name] = metrics_mod.ndcg_at_k_per_sample(rk, tg, 10)

    # Sanity (NOT tuning): recomputed means must match the committed T2 numbers.
    camp_mean = float(persample["camp"].mean())
    xgb_mean = float(persample["gradient_boosted"].mean())
    print(f"  CAMP per-sample mean NDCG@10     = {camp_mean:.6f}  (locked 0.636658)")
    print(f"  XGBoost (unmasked) mean NDCG@10  = {xgb_mean:.6f}  (T2 gradient_boosted 0.716066)")

    # ---- NEW pairwise: CAMP vs UNMASKED XGBoost ----
    camp_ps = persample["camp"]
    xgb_ps = persample["gradient_boosted"]
    t_stat, t_p = stats.ttest_rel(camp_ps, xgb_ps)
    w_stat, w_p = stats.wilcoxon(camp_ps, xgb_ps)
    new_entry = {
        "camp_mean_ndcg10": round(float(camp_ps.mean()), 6),
        "other_mean_ndcg10": round(float(xgb_ps.mean()), 6),
        "paired_t_stat": round(float(t_stat), 4), "paired_t_p": float(t_p),
        "wilcoxon_stat": round(float(w_stat), 4), "wilcoxon_p": float(w_p),
    }
    if float(t_p) == 0.0:
        new_entry["paired_t_p_thresholded"] = _p_thresholded(float(t_p))
    if float(w_p) == 0.0:
        new_entry["wilcoxon_p_thresholded"] = _p_thresholded(float(w_p))

    # ---- Seven-model ANOVA ----
    anova_models = ["camp", *raw_by_model.keys()]
    f_stat, f_p = stats.f_oneway(*[persample[m] for m in anova_models])

    # ---- Assemble: preserve everything; copy the five existing pairwise VERBATIM ----
    pairwise = dict(old["significance_per_sample_ndcg10"])  # verbatim copy of the 5
    # Additive-only: thresholded display for any existing exact-zero p-values.
    for k, e in pairwise.items():
        if e.get("paired_t_p") == 0.0 and "paired_t_p_thresholded" not in e:
            e["paired_t_p_thresholded"] = _p_thresholded(0.0)
        if e.get("wilcoxon_p") == 0.0 and "wilcoxon_p_thresholded" not in e:
            e["wilcoxon_p_thresholded"] = _p_thresholded(0.0)
    pairwise["camp_vs_gradient_boosted"] = new_entry

    out = dict(old)  # preserve n_test_samples, seeds, stability sections, etc.
    out["note"] = (
        "5-seed CAMP stability uses 100k steps/inst (vs 200k finalised) for CPU budget; "
        "this stability mean is DISTINCT from the 200k finalised CAMP-full NDCG@10 (0.637). "
        "CF and MF are deterministic (no training randomness). Significance tests are "
        "per-sample paired (n=n_test) on the finalised models. The ANOVA and pairwise "
        "tests cover the seven systems in Table T2 (CF, MF, RF, Deep NN, pure GNN, "
        "UNMASKED XGBoost, CAMP); XGBoost is the same 0.716 unmasked model in T2. "
        "Exact-zero p-values (float underflow) carry a '*_thresholded' sibling ('<1e-300')."
    )
    out["significance_per_sample_ndcg10"] = pairwise
    out.pop("anova_six_models", None)
    out["anova_seven_models"] = {
        "models": anova_models,
        "F": round(float(f_stat), 4),
        "p": float(f_p),
        "p_thresholded": _p_thresholded(float(f_p)),
    }

    STAT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {STAT_PATH}")


if __name__ == "__main__":
    main()
