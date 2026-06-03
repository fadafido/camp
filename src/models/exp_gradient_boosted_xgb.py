"""Gradient-boosted baseline — **XGBoost** estimator, unmasked + masked.

The single gradient-boosted baseline reported in the paper: an XGBoost
``XGBClassifier`` (CPU). On arm64 macOS XGBoost's native library needs the
OpenMP runtime ``libomp.dylib`` (``brew install libomp``) on the loader path —
see the runtime note below.

Runtime note (libomp)
---------------------
``libomp`` is keg-only on Homebrew (not symlinked into ``/usr/local``). XGBoost's
native library resolves ``libomp.dylib`` at *import* time, so the loader must be
able to find it. Run this module with::

    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
    .venv/bin/python -m src.models.exp_gradient_boosted_xgb

(The env var must be set *before* the Python process starts — dyld reads it at
launch, so setting ``os.environ`` from inside the process would be too late.)

Method
------
One-vs-rest XGBoost over the 194 courses — one ``XGBClassifier`` per course —
using the SAME features the Random Forest baseline uses (``bl.summary_features``) and
``bl.multilabel_targets`` (no new features invented). Same train split, seed 42.
Per-course probabilities form the ``(n_test, 194)`` score matrix, scored through
the IDENTICAL harness (``bl.evaluate_model``) on the SAME 6,606 test split with
institution-masking, in two variants:
  * ``gradient_boosted``        — unmasked (constraint-blind),
  * ``gradient_boosted_masked`` — hard eligibility filter (``eligible_mask=True``,
    the same locked masking path used elsewhere).

Hyperparameters are standard, reasonable defaults for a strong tabular baseline
(this is a baseline, not a tuning exercise): ``n_estimators=300``,
``max_depth=6``, ``learning_rate=0.1``, ``tree_method='hist'``, ``device='cpu'``,
``random_state=42``. CPU-only; deterministic given the seed. British English.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import xgboost as xgb

from src.models import baselines as bl
from src.models import task
from src.utils.seed import set_seed

OUT_PATH = bl.RESULTS_DIR / "gradient_boosted.json"

# XGBoost hyperparameters (documented; deterministic given seed). Standard,
# reasonable defaults for a strong gradient-boosted tabular baseline — CPU-only.
GB_HP = {
    "model": "XGBoost XGBClassifier (one-vs-rest over 194 courses)",
    "library": "XGBoost",
    "library_version": xgb.__version__,
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "device": "cpu",
    "objective": "binary:logistic",
    "random_state": 42,
    "features": "bl.summary_features (same as the Random Forest baseline)",
    "note": "Per-course binary head; ranks by per-course P(taken). Courses with a "
    "single class in train get a constant score (mirrors the RF/HGB baselines). "
    "Supersedes the prior scikit-learn HistGradientBoosting run now that libomp "
    "is installed and XGBoost can load.",
}


def scores_gradient_boosted_xgb(samples_train, samples_test, seed: int = 42) -> np.ndarray:
    """Per-course XGBoost probabilities for the test split (constraint-blind).

    One ``XGBClassifier`` per course over the shared RF features. A course with
    only one class in the training targets cannot be fit (no positive/negative
    contrast); its score is set to that constant class value, exactly as the
    Random Forest / HistGradientBoosting baselines handle single-class columns.
    """
    set_seed(seed)
    x_train = bl.summary_features(samples_train)
    y_train = bl.multilabel_targets(samples_train)  # (n_train, 194) int {0,1}
    x_test = bl.summary_features(samples_test)
    raw = np.zeros((len(samples_test), bl.N_COURSES), dtype=float)
    for j in range(bl.N_COURSES):
        yj = y_train[:, j]
        classes = np.unique(yj)
        if classes.size < 2:
            raw[:, j] = float(classes[0])  # constant label (typically all-0 -> 0.0)
            continue
        clf = xgb.XGBClassifier(
            n_estimators=GB_HP["n_estimators"],
            max_depth=GB_HP["max_depth"],
            learning_rate=GB_HP["learning_rate"],
            tree_method=GB_HP["tree_method"],
            device=GB_HP["device"],
            objective=GB_HP["objective"],
            random_state=seed,
        )
        clf.fit(x_train, yj)
        proba = clf.predict_proba(x_test)
        idx1 = list(clf.classes_).index(1)
        raw[:, j] = proba[:, idx1]
    return raw


def run() -> dict[str, Any]:
    set_seed(42)
    bl.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    engines = {uni: bl._engine_for(uni) for uni in task.UNIVERSITIES}
    samples_train = task.load_samples("train")
    samples_test = task.load_samples("test")

    raw = scores_gradient_boosted_xgb(samples_train, samples_test)

    unmasked = bl.evaluate_model("gradient_boosted", GB_HP, raw, samples_test, engines,
                                 eligible_mask=False)
    masked = bl.evaluate_model("gradient_boosted_masked", GB_HP, raw, samples_test, engines,
                               eligible_mask=True)

    out = {
        "experiment": "gradient-boosted baseline (XGBoost; unmasked + mask-applied)",
        "n_test_samples": unmasked["n_test_samples"],
        "library": "XGBoost",
        "library_version": xgb.__version__,
        "library_note": "XGBoost " + xgb.__version__ + " — supersedes the prior "
        "scikit-learn HistGradientBoosting run; libomp now installed so XGBoost loads",
        "masking_note": "masked variant uses the identical locked path "
        "(evaluate_model eligible_mask=True -> ConstraintEngine.eligible_courses); "
        "institution-masking preserved in both variants",
        "variants": {
            "gradient_boosted": unmasked,
            "gradient_boosted_masked": masked,
        },
    }
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT_PATH}")
    return out


def verify() -> None:
    """Re-read the written file and print the verification block."""
    with OUT_PATH.open() as fh:
        out = json.load(fh)
    um = out["variants"]["gradient_boosted"]
    mk = out["variants"]["gradient_boosted_masked"]
    n = out["n_test_samples"]

    u_ndcg = um["ranking_metrics"]["NDCG@10"]
    u_viol = um["prereq_violation_rate_topk"]
    m = mk["ranking_metrics"]
    mc = mk["classification_metrics"]

    print("\n================ VERIFICATION BLOCK (re-read from file) ================")
    print(f"n_test actually scored      : {n}   (must be 6606)")
    print(f"library                     : {out['library']} {out['library_version']}")
    print("\n-- UNMASKED gradient_boosted (sanity vs RF 0.723/0.141, Deep NN 0.775/0.047,")
    print("   prior HGB 0.694/0.114) --")
    print(f"  NDCG@10                   : {u_ndcg:.6f}")
    print(f"  Recall@10                 : {um['ranking_metrics']['Recall@10']:.6f}")
    print(f"  F1 (micro)                : {um['classification_metrics']['f1_micro']:.6f}")
    print(f"  ROC-AUC (micro)           : {um['classification_metrics']['roc_auc_micro']:.6f}")
    print(f"  prereq-violation rate     : {u_viol:.6f}")
    print(f"  pathway-feasibility       : {um['pathway_feasibility']:.6f}")
    print(f"  graduation-compliance     : {um['graduation_compliance']:.6f}")
    print("\n-- MASKED gradient_boosted_masked --")
    print(f"  NDCG@10                   : {m['NDCG@10']:.6f}")
    print(f"  Recall@10                 : {m['Recall@10']:.6f}")
    print(f"  F1 (micro)                : {mc['f1_micro']:.6f}")
    print(f"  ROC-AUC (micro)           : {mc['roc_auc_micro']:.6f}")
    print(f"  prereq-violation rate     : {mk['prereq_violation_rate_topk']:.6f}   (MUST be 0.000)")
    print(f"  pathway-feasibility       : {mk['pathway_feasibility']:.6f}   (expect 1.000)")
    print(f"  graduation-compliance     : {mk['graduation_compliance']:.6f}")
    print("\n-- one-liner --")
    print(f"  masked - unmasked NDCG@10 = {m['NDCG@10'] - u_ndcg:+.6f}   |   "
          f"masked violation rate = {mk['prereq_violation_rate_topk']:.6f}")
    print("==========================================================================\n")

    fail = []
    if n != 6606:
        fail.append(f"n_test={n} != 6606")
    if mk["prereq_violation_rate_topk"] != 0.0:
        fail.append(f"masked violation rate = {mk['prereq_violation_rate_topk']} (must be 0.000)")
    if fail:
        raise SystemExit("STOP — verification failed:\n  - " + "\n  - ".join(fail))
    print("VERIFICATION: PASS")


if __name__ == "__main__":
    run()
    verify()
