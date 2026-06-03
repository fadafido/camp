"""SHAP feature importance for the gradient-boosted (XGBoost) baseline.

ADD-ONLY explainability complement. SHAP on the masked MaskablePPO policy is not
well-defined (action masking makes per-feature attributions ill-posed), so CAMP's
explainability stays the committed permutation-importance analysis
(``explainability.json``, unchanged). SHAP is provided here on the **tree
baseline** as a standard, well-defined complement — this artefact does NOT claim
SHAP for CAMP.

The XGBoost baseline is a deterministic function of seed 42 and the documented
hyperparameters/features (``exp_gradient_boosted_xgb.GB_HP`` + ``bl.summary_features``)
and is not persisted on disk, so it is **reconstructed identically** here (not
re-tuned, not a new training result) and then explained. As a correctness guard,
the reconstructed model's unmasked NDCG@10 and prerequisite-violation rate are
re-scored through the locked harness and checked against the committed
``gradient_boosted.json`` values; the committed file is never rewritten.

SHAP values are exact TreeSHAP, obtained from XGBoost's native
``predict(..., pred_contribs=True)`` (no extra dependency). Global importance per
feature = mean over the 194 one-vs-rest course models of the mean over test
samples of ``|SHAP|`` (log-odds margin units).

Writes ``results/shap_xgboost.json``. Seed 42; CPU-only; British English.

Runtime note (libomp): as with ``exp_gradient_boosted_xgb``, set
``DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`` before launching so XGBoost's
native library loads on arm64 macOS.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import xgboost as xgb

from src.models import baselines as bl
from src.models import task
from src.models.exp_gradient_boosted_xgb import GB_HP
from src.utils.seed import set_seed

RESULTS_DIR = bl.RESULTS_DIR
OUT_PATH = RESULTS_DIR / "shap_xgboost.json"
GB_COMMITTED = RESULTS_DIR / "gradient_boosted.json"
SEED = 42


def feature_names() -> list[str]:
    """Names for the 45 ``bl.summary_features`` columns, in construction order."""
    names = [f"subject:{a}" for a in task.SUBJECT_AREAS]
    names += ["mean_grade", "term_index", "n_passed"]
    names += [f"gpa_band:{b}" for b in task.GPA_BANDS]
    names += [f"gender:{g}" for g in task.GENDERS]
    names += [f"major:{m}" for m in task.MAJOR_TRACKS]
    names += [f"university:{u}" for u in task.UNIVERSITIES]
    names += ["cohort", "cold_start"]
    return names


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feats = feature_names()
    samples_train = task.load_samples("train")
    samples_test = task.load_samples("test")
    x_train = bl.summary_features(samples_train)
    y_train = bl.multilabel_targets(samples_train)
    x_test = bl.summary_features(samples_test)
    assert x_train.shape[1] == len(feats), (x_train.shape, len(feats))
    dtest = xgb.DMatrix(x_test, feature_names=feats)

    # Reconstruct the per-course XGBoost baseline (identical to the committed run)
    # and accumulate mean(|SHAP|) per feature across the course models.
    raw = np.zeros((len(samples_test), bl.N_COURSES), dtype=float)
    abs_shap_sum = np.zeros(len(feats), dtype=float)
    n_models = 0
    for j in range(bl.N_COURSES):
        yj = y_train[:, j]
        classes = np.unique(yj)
        if classes.size < 2:
            raw[:, j] = float(classes[0])
            continue
        clf = xgb.XGBClassifier(
            n_estimators=GB_HP["n_estimators"], max_depth=GB_HP["max_depth"],
            learning_rate=GB_HP["learning_rate"], tree_method=GB_HP["tree_method"],
            device=GB_HP["device"], objective=GB_HP["objective"], random_state=SEED,
        )
        clf.fit(x_train, yj)
        proba = clf.predict_proba(x_test)
        raw[:, j] = proba[:, list(clf.classes_).index(1)]
        # Exact TreeSHAP via native pred_contribs; last column is the bias term.
        contribs = clf.get_booster().predict(dtest, pred_contribs=True)
        abs_shap_sum += np.abs(contribs[:, :-1]).mean(axis=0)
        n_models += 1
    overall = abs_shap_sum / n_models
    total = float(overall.sum()) or 1.0

    importance = {f: round(float(v), 6) for f, v in zip(feats, overall)}
    normalised = {f: round(float(v / total), 6) for f, v in zip(feats, overall)}
    ranked = sorted(zip(feats, overall), key=lambda kv: -kv[1])
    top_features = [{"feature": f, "mean_abs_shap": round(float(v), 6),
                     "normalised": round(float(v / total), 6)} for f, v in ranked[:12]]

    # --- correctness guard: reconstructed model reproduces committed metrics ---
    engines = {u: bl._engine_for(u) for u in task.UNIVERSITIES}
    um = bl.evaluate_model("gradient_boosted", GB_HP, raw, samples_test, engines, eligible_mask=False)
    with GB_COMMITTED.open() as fh:
        committed = json.load(fh)["variants"]["gradient_boosted"]
    repro = {
        "reconstructed_NDCG@10": round(um["ranking_metrics"]["NDCG@10"], 6),
        "committed_NDCG@10": round(committed["ranking_metrics"]["NDCG@10"], 6),
        "reconstructed_violation_rate": round(um["prereq_violation_rate_topk"], 6),
        "committed_violation_rate": round(committed["prereq_violation_rate_topk"], 6),
    }
    repro["matches_committed"] = bool(
        abs(repro["reconstructed_NDCG@10"] - repro["committed_NDCG@10"]) < 1e-6
        and abs(repro["reconstructed_violation_rate"] - repro["committed_violation_rate"]) < 1e-6
    )

    out = {
        "experiment": "SHAP (TreeSHAP) feature importance for the gradient-boosted "
        "(XGBoost) baseline — explainability complement on the tree baseline",
        "applies_to": "gradient-boosted (XGBoost) baseline ONLY",
        "not_applicable_to": "CAMP (masked MaskablePPO policy): SHAP is not "
        "well-defined under action masking; CAMP uses permutation importance "
        "(explainability.json, unchanged).",
        "method": "exact TreeSHAP via XGBoost native predict(pred_contribs=True); "
        "global importance = mean over 194 one-vs-rest course models of the mean "
        "over test samples of |SHAP| per feature (log-odds margin units).",
        "library": "XGBoost", "library_version": xgb.__version__,
        "seed": SEED, "n_test_samples": len(samples_test),
        "n_features": len(feats), "n_course_models": n_models,
        "reconstruction_guard": repro,
        "feature_importance_meanabs_shap": importance,
        "feature_importance_normalised": normalised,
        "top_features": top_features,
    }
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"  reconstruction matches committed gradient_boosted.json: {repro['matches_committed']}")
    print(f"    NDCG@10 {repro['reconstructed_NDCG@10']} vs {repro['committed_NDCG@10']}; "
          f"viol {repro['reconstructed_violation_rate']} vs {repro['committed_violation_rate']}")
    print("  top SHAP features:")
    for t in top_features[:6]:
        print(f"    {t['feature']:22s} {t['mean_abs_shap']:.4f}  ({t['normalised']*100:.1f}%)")


if __name__ == "__main__":
    main()
