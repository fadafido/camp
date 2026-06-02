"""Part 5 — the five Phase B baseline models.

Each baseline trains on the train split, (tunes on val where relevant), predicts
a ranked course list per test sample, and is scored through the shared harness
(:mod:`src.models.metrics`). Every model additionally reports the
**prerequisite-violation rate** of its top-k recommendations via the Phase A
:class:`ConstraintEngine` — baselines are constraint-blind, so this is expected
to be > 0 and is a key result for the paper.

Models: (a) item-item Collaborative Filtering, (b) Matrix Factorisation
(TruncatedSVD), (c) Random Forest (multi-output multilabel), (d) Deep Neural
Network (MLP), (e) Pure GNN (GraphSAGE dot-product, no constraints).

Results are written per model to ``results/baseline_<name>.json``. Seed 42
throughout; CPU-only. British English.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from src.constraint_engine import dag
from src.constraint_engine.engine import ConstraintEngine, StudentState
from src.models import gnn as gnn_mod
from src.models import metrics as metrics_mod
from src.models import task
from src.utils.seed import set_seed

BUNDLE = task.BUNDLE
RESULTS_DIR = BUNDLE / "results"
INTER_DIR = BUNDLE / "intermediate"
TOPK_FOR_VIOLATIONS = 10

# ----------------------------------------------------------------- shared setup
VOCAB = task.load_vocab()
IDX2CID = {i: c for c, i in VOCAB.items()}
N_COURSES = len(VOCAB)
# University per vocab index, for institution-masking of candidate courses.
_CU = task.course_university()
COURSE_UNI = np.array([_CU[IDX2CID[i]] for i in range(N_COURSES)])
# v3_3inst: three institutions, all 3-digit course numbers.
_PROG = {"khalifa": "KHAL_BSC_CS_v2024", "aus": "AUS_BSBA_IS_v2024", "unc": "UNC_BS_ECON_v2024"}
_PREFIX = {"khalifa": "KHAL", "aus": "AUS", "unc": "UNC"}
_DIGITS = {"khalifa": 3, "aus": 3, "unc": 3}


def _engine_for(uni: str) -> ConstraintEngine:
    """Construct the constraint engine for one institution from the v3_3inst
    intermediate bundle (filtered to that institution) + its hybrid DAG."""
    courses = [c for c in task.load_courses() if c["university"] == uni]
    graph, _ = dag.build_real_inst_dag(courses, _DIGITS[uni])
    pid, up = _PROG[uni], _PREFIX[uni]

    def L(name: str):
        with (INTER_DIR / name).open() as fh:
            return json.load(fh)

    return ConstraintEngine(
        courses=courses,
        blocks=[b for b in L("blocks.json") if b["programme_id"] == pid],
        rule_groups=[g for g in L("rule_groups.json") if g["block_id"].startswith(f"{up}_BLOCK")],
        rules=[r for r in L("rules.json") if r["rule_id"].startswith(f"{up}_RULE")],
        rule_eligible_courses=[e for e in L("rule_eligible_courses.json")
                               if e["rule_id"].startswith(f"{up}_RULE")],
        programmes=[p for p in L("programmes.json") if p["programme_id"] == pid],
        graph=graph,
    )


def _minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise min-max to [0, 1] so non-probabilistic scores are comparable."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return (scores - lo) / span


def _passed_idx(sample: dict[str, Any]) -> list[int]:
    return [VOCAB[c] for c in task.passed_history(sample)]


def _target_idx(sample: dict[str, Any]) -> set[int]:
    return {VOCAB[c] for c in sample["target_courses"]}


def _ndcg10_by_university(
    rankings: list[list[int]], targets: list[set[int]], unis: list[str]
) -> dict[str, float]:
    out = {}
    for uni in task.UNIVERSITIES:
        idx = [i for i, u in enumerate(unis) if u == uni]
        if not idx:
            continue
        sub = metrics_mod.evaluate_ranking([rankings[i] for i in idx], [targets[i] for i in idx], k_values=[10])
        out[uni] = sub["NDCG@10"]
    return out


def rank_samples(raw_scores, samples, engines, eligible_mask: bool = False):
    """Per-sample ranked course-index lists + true target sets (same masking as
    :func:`evaluate_model`). Used for per-sample significance tests."""
    rankings, targets = [], []
    for i, sample in enumerate(samples):
        row = raw_scores[i].copy()
        row[_passed_idx(sample)] = -np.inf
        row[np.where(COURSE_UNI != sample["university"])[0]] = -np.inf
        if eligible_mask:
            elig = _eligible_global_idx(engines[sample["university"]], sample)
            row[np.array([j for j in range(N_COURSES) if j not in elig], dtype=int)] = -np.inf
        rankings.append(np.argsort(-row).tolist())
        targets.append(_target_idx(sample))
    return rankings, targets


def _eligible_global_idx(engine: ConstraintEngine, sample: dict[str, Any]) -> set[int]:
    """Global vocab indices of courses that are prerequisite-eligible now."""
    passed = {c: sample["history_grades"][c] for c in task.passed_history(sample)}
    state = StudentState(programme_id=sample.get("programme_id") or _PROG[sample["university"]], passed=passed)
    return {VOCAB[c] for c in engine.eligible_courses(state)}


def evaluate_model(
    name: str,
    hyperparams: dict[str, Any],
    raw_scores: np.ndarray,
    samples: list[dict[str, Any]],
    engines: dict[str, ConstraintEngine],
    eligible_mask: bool = False,
) -> dict[str, Any]:
    """Score a model's per-sample course scores through the full harness.

    ``raw_scores`` is an ``(n_samples, 262)`` array in [0, 1]. For each sample,
    courses already passed and **courses from the other institution** are removed
    from candidacy (institution-masking). When ``eligible_mask`` is True (CAMP),
    prerequisite-ineligible courses are *also* removed — so CAMP can only ever
    recommend a legal course, giving a structural ~0 violation rate. Baselines
    use ``eligible_mask=False`` (constraint-blind).

    Adds ``graduation_compliance`` (fraction of programme rules satisfied once
    the top-k recommendations are added to the history) and
    ``pathway_feasibility`` (fraction of top-k recommendations whose
    prerequisites are met) to the standard ranking/classification/violation suite.
    """
    scores = raw_scores.copy()
    rankings: list[list[int]] = []
    targets: list[set[int]] = []
    topk_for_viol: list[list[str]] = []
    passed_lists: list[list[str]] = []
    unis: list[str] = []

    for i, sample in enumerate(samples):
        passed = _passed_idx(sample)
        other_inst = np.where(COURSE_UNI != sample["university"])[0]
        row = scores[i].copy()
        row[passed] = -np.inf
        row[other_inst] = -np.inf
        if eligible_mask:
            elig = _eligible_global_idx(engines[sample["university"]], sample)
            blocked = np.array([j for j in range(N_COURSES) if j not in elig], dtype=int)
            row[blocked] = -np.inf
            scores[i, blocked] = 0.0
        order = np.argsort(-row).tolist()
        rankings.append(order)
        targets.append(_target_idx(sample))
        # Only finite-score (legal) courses may be "recommended".
        topk = [j for j in order[:TOPK_FOR_VIOLATIONS] if np.isfinite(row[j])]
        topk_for_viol.append([IDX2CID[j] for j in topk])
        passed_lists.append([IDX2CID[j] for j in passed])
        unis.append(sample["university"])
        scores[i, passed] = 0.0
        scores[i, other_inst] = 0.0

    ranking_metrics = metrics_mod.evaluate_ranking(rankings, targets, k_values=[1, 3, 5, 10])
    classification_metrics = metrics_mod.evaluate_classification(scores, targets, N_COURSES)
    ndcg10_by_uni = _ndcg10_by_university(rankings, targets, unis)

    total_viol = 0
    total_reco = 0
    feasible_sum = 0.0
    compliance_sum = 0.0
    for passed, topk, uni in zip(passed_lists, topk_for_viol, unis):
        eng = engines[uni]
        with_topk = eng.prerequisite_violations([passed, topk])
        history_only = eng.prerequisite_violations([passed])
        total_viol += with_topk - history_only
        total_reco += len(topk)
        # Pathway feasibility: fraction of recommendations whose prereqs are met.
        if topk:
            feas = sum(1 for c in topk if eng.check_prerequisites(c, set(passed))) / len(topk)
        else:
            feas = 1.0
        feasible_sum += feas
        # Graduation compliance: rules satisfied once recommendations are added.
        compliance_sum += eng.graduation_compliance([passed, topk])
    viol_rate = round(total_viol / total_reco, 6) if total_reco else 0.0
    n = len(samples)

    return {
        "model": name,
        "hyperparameters": hyperparams,
        "n_test_samples": n,
        "ranking_metrics": ranking_metrics,
        "ndcg10_by_university": ndcg10_by_uni,
        "classification_metrics": classification_metrics,
        "prereq_violation_rate_topk": viol_rate,
        "prereq_violation_topk_k": TOPK_FOR_VIOLATIONS,
        "total_prereq_violations": total_viol,
        "total_recommended": total_reco,
        "graduation_compliance": round(compliance_sum / n, 6),
        "pathway_feasibility": round(feasible_sum / n, 6),
    }


# --------------------------------------------------------- feature construction
def passed_matrix(students: list[dict[str, Any]]) -> np.ndarray:
    """Train student×course passed-incidence matrix (rows = students)."""
    mat = np.zeros((len(students), N_COURSES), dtype=float)
    for r, s in enumerate(students):
        for t in s["terms"]:
            for c in t["courses"]:
                if c["passed"]:
                    mat[r, VOCAB[c["course_id"]]] = 1.0
    return mat


def history_vectors(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample passed multi-hot (262) and per-course mean grade (262)."""
    multihot = np.zeros((len(samples), N_COURSES), dtype=float)
    grades = np.zeros((len(samples), N_COURSES), dtype=float)
    for i, s in enumerate(samples):
        for cid in task.passed_history(s):
            j = VOCAB[cid]
            multihot[i, j] = 1.0
            grades[i, j] = s["history_grades"][cid]
    return multihot, grades


def _demographic_features(sample: dict[str, Any]) -> list[float]:
    d = sample["demographics"]
    gpa = [1.0 if d["gpa_band"] == b else 0.0 for b in task.GPA_BANDS]
    gender = [1.0 if d["gender"] == g else 0.0 for g in task.GENDERS]
    major = [1.0 if d["major_track"] == m else 0.0 for m in task.MAJOR_TRACKS]
    uni = [1.0 if sample["university"] == u else 0.0 for u in task.UNIVERSITIES]
    cohort = task.COHORT_INDEX[d["cohort"]] / (len(task.COHORTS) - 1)
    cold = 1.0 if d["cold_start"] else 0.0
    return gpa + gender + major + uni + [cohort, cold]


def summary_features(samples: list[dict[str, Any]]) -> np.ndarray:
    """Compact per-sample features for the Random Forest baseline."""
    courses = {c["course_id"]: c for c in task.load_courses()}
    rows = []
    for s in samples:
        passed = task.passed_history(s)
        subj_counts = {area: 0 for area in task.SUBJECT_AREAS}
        for cid in passed:
            subj_counts[courses[cid]["subject_area"]] += 1
        n = max(len(passed), 1)
        subj_norm = [subj_counts[a] / n for a in task.SUBJECT_AREAS]
        mean_grade = (
            float(np.mean([s["history_grades"][c] for c in passed])) if passed else 0.0
        )
        rows.append(
            subj_norm
            + [mean_grade, s["term_index"] / 12.0, len(passed) / 60.0]
            + _demographic_features(s)
        )
    return np.asarray(rows, dtype=float)


def multilabel_targets(samples: list[dict[str, Any]]) -> np.ndarray:
    y = np.zeros((len(samples), N_COURSES), dtype=int)
    for i, s in enumerate(samples):
        for cid in s["target_courses"]:
            y[i, VOCAB[cid]] = 1
    return y


# ----------------------------------------------------------------- the baselines
def scores_collaborative_filtering(train_students, samples_test) -> np.ndarray:
    mat = passed_matrix(train_students)
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = mat / norms
    sim = normed.T @ normed
    np.fill_diagonal(sim, 0.0)
    test_multihot, _ = history_vectors(samples_test)
    return _minmax_rows(test_multihot @ sim.T)


def run_collaborative_filtering(train_students, samples_test, engines) -> dict[str, Any]:
    raw = scores_collaborative_filtering(train_students, samples_test)
    hp = {"method": "item-item cosine", "matrix": "train passed incidence"}
    return evaluate_model("collaborative_filtering", hp, raw, samples_test, engines)


def scores_matrix_factorisation(train_students, samples_test, k: int = 32) -> np.ndarray:
    mat = passed_matrix(train_students)
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(mat)
    v = svd.components_
    item_item = v.T @ v
    test_multihot, _ = history_vectors(samples_test)
    return _minmax_rows(test_multihot @ item_item)


def run_matrix_factorisation(train_students, samples_test, engines, k: int = 32) -> dict[str, Any]:
    raw = scores_matrix_factorisation(train_students, samples_test, k)
    hp = {"method": "TruncatedSVD", "n_components": k, "random_state": 42}
    return evaluate_model("matrix_factorisation", hp, raw, samples_test, engines)


def scores_random_forest(samples_train, samples_test, seed: int = 42) -> np.ndarray:
    x_train = summary_features(samples_train)
    y_train = multilabel_targets(samples_train)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    rf.fit(x_train, y_train)
    x_test = summary_features(samples_test)
    proba = rf.predict_proba(x_test)
    raw = np.zeros((len(samples_test), N_COURSES), dtype=float)
    for j, p in enumerate(proba):
        classes = rf.classes_[j] if isinstance(rf.classes_, list) else rf.classes_
        if p.shape[1] == 1:
            raw[:, j] = float(classes[0])
        else:
            raw[:, j] = p[:, list(classes).index(1)]
    return raw


def run_random_forest(samples_train, samples_test, engines) -> dict[str, Any]:
    raw = scores_random_forest(samples_train, samples_test)
    hp = {
        "model": "RandomForestClassifier (multi-output multilabel)",
        "n_estimators": 200, "max_depth": None, "min_samples_leaf": 2,
        "random_state": 42, "n_jobs": -1,
        "note": "Multi-output multilabel (81 binary heads); ranks by per-course probability.",
    }
    return evaluate_model("random_forest", hp, raw, samples_test, engines)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _dnn_features(samples) -> np.ndarray:
    multihot, grades = history_vectors(samples)
    demo = np.asarray([_demographic_features(s) for s in samples], dtype=float)
    term = np.asarray([[s["term_index"] / 12.0] for s in samples], dtype=float)
    return np.concatenate([multihot, grades, demo, term], axis=1)


def scores_deep_nn(samples_train, samples_val, samples_test, seed: int = 42) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    hp = {
        "architecture": "MLP [in -> 256 -> 128 -> 81], ReLU, dropout 0.3",
        "loss": "BCEWithLogits (multi-label)",
        "optimizer": "Adam",
        "lr": 1e-3,
        "batch_size": 256,
        "max_epochs": 50,
        "early_stopping_patience": 7,
        "seed": seed,
    }
    x_train = torch.tensor(_dnn_features(samples_train), dtype=torch.float)
    y_train = torch.tensor(multilabel_targets(samples_train), dtype=torch.float)
    x_val = torch.tensor(_dnn_features(samples_val), dtype=torch.float)
    y_val = torch.tensor(multilabel_targets(samples_val), dtype=torch.float)
    x_test = torch.tensor(_dnn_features(samples_test), dtype=torch.float)

    model = MLP(x_train.size(1), [256, 128], N_COURSES, dropout=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    n = x_train.size(0)
    gen = torch.Generator().manual_seed(seed)
    best_val = float("inf")
    best_state = None
    since_best = 0
    epochs_run = 0
    for epoch in range(hp["max_epochs"]):
        epochs_run = epoch + 1
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, hp["batch_size"]):
            idx = perm[start : start + hp["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        if val_loss < best_val - 1e-5:
            best_val, best_state, since_best = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since_best += 1
            if since_best >= hp["early_stopping_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = torch.sigmoid(model(x_test)).numpy()
    hp["epochs_trained"] = epochs_run
    hp["best_val_loss"] = round(best_val, 6)
    return raw, hp


def run_deep_nn(samples_train, samples_val, samples_test, engines) -> dict[str, Any]:
    raw, hp = scores_deep_nn(samples_train, samples_val, samples_test)
    return evaluate_model("deep_nn", hp, raw, samples_test, engines)


def scores_pure_gnn(samples_test) -> np.ndarray:
    payload = torch.load(gnn_mod.EMBEDDINGS_PATH, weights_only=False)
    course_emb = payload["course_emb"]
    course_order = payload["course_order"]
    emb_by_cid = {cid: course_emb[i] for i, cid in enumerate(course_order)}
    course_mat = torch.stack([emb_by_cid[IDX2CID[j]] for j in range(N_COURSES)])
    raw = np.zeros((len(samples_test), N_COURSES), dtype=float)
    for i, s in enumerate(samples_test):
        passed = _passed_idx(s)
        student_repr = course_mat[passed].mean(dim=0) if passed else course_mat.mean(dim=0)
        raw[i] = torch.sigmoid(course_mat @ student_repr).numpy()
    return raw


def run_pure_gnn(samples_test, engines) -> dict[str, Any]:
    raw = scores_pure_gnn(samples_test)
    payload = torch.load(gnn_mod.EMBEDDINGS_PATH, weights_only=False)
    hp = {
        "method": "GraphSAGE dot-product (inductive student = mean of history course embeddings)",
        "embedding_dim": payload["hyperparams"]["embedding_dim"],
        "final_val_auc": round(float(payload["final_val_auc"]), 6),
        "epochs_trained": payload["epochs_trained"],
    }
    return evaluate_model("pure_gnn", hp, raw, samples_test, engines)


def main() -> None:
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # One constraint engine per institution (for institution-aware violations).
    engines = {uni: _engine_for(uni) for uni in task.UNIVERSITIES}

    split_of = task._student_to_split()
    train_students = [
        s for s in task._load_students() if split_of.get(s["student_id"]) == "train"
    ]
    samples_train = task.load_samples("train")
    samples_val = task.load_samples("val")
    samples_test = task.load_samples("test")

    results: dict[str, dict[str, Any]] = {}
    runners = [
        ("collaborative_filtering", lambda: run_collaborative_filtering(train_students, samples_test, engines)),
        ("matrix_factorisation", lambda: run_matrix_factorisation(train_students, samples_test, engines)),
        ("random_forest", lambda: run_random_forest(samples_train, samples_test, engines)),
        ("deep_nn", lambda: run_deep_nn(samples_train, samples_val, samples_test, engines)),
        ("pure_gnn", lambda: run_pure_gnn(samples_test, engines)),
    ]
    for name, fn in runners:
        print(f"Running baseline: {name} ...")
        res = fn()
        results[name] = res
        with (RESULTS_DIR / f"baseline_{name}.json").open("w") as fh:
            json.dump(res, fh, indent=2)
        print(
            f"  {name}: NDCG@10={res['ranking_metrics']['NDCG@10']} "
            f"Recall@10={res['ranking_metrics']['Recall@10']} "
            f"F1micro={res['classification_metrics']['f1_micro']} "
            f"viol_rate={res['prereq_violation_rate_topk']}"
        )

    # Comparison summary across all five models.
    summary = {
        "task": "next-term course-set prediction",
        "n_test_samples": len(samples_test),
        "models": {
            name: {
                "ranking_metrics": r["ranking_metrics"],
                "classification_metrics": r["classification_metrics"],
                "prereq_violation_rate_topk": r["prereq_violation_rate_topk"],
                "hyperparameters": r["hyperparameters"],
            }
            for name, r in results.items()
        },
    }
    with (RESULTS_DIR / "baselines_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {RESULTS_DIR / 'baselines_summary.json'}")

    # Task summary (samples per split, vocab size, mean target-set size).
    with (task.MODELS_DIR / "task_summary.json").open("w") as fh:
        json.dump(task.report(), fh, indent=2)

    return results


if __name__ == "__main__":
    main()
