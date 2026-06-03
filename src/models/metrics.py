"""Shared, deterministic evaluation harness for Phase B (and beyond).

All baselines and CAMP are scored through these functions on identical
samples, so model comparison is contamination-free.

Two families:
  * :func:`evaluate_ranking` — Recall/Precision/NDCG/MAP/HitRatio @ K over ranked
    course-index lists vs. true target sets.
  * :func:`evaluate_classification` — multi-label metrics over the 194 courses
    (micro/macro Accuracy, Precision, Recall, F1, ROC-AUC), via sklearn.

Run ``python -m src.models.metrics`` for the self-test on a hand-built example
with known answers.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate_ranking(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Iterable[int]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Ranking metrics averaged over all samples.

    Parameters
    ----------
    predictions
        One ranked list of course indices per sample (best first).
    targets
        One iterable of true course indices per sample.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must align")
    target_sets = [set(t) for t in targets]
    n = len(predictions)
    out: dict[str, float] = {}

    for k in k_values:
        recall = precision = ndcg = ap = hit = 0.0
        for preds, tgt in zip(predictions, target_sets):
            if not tgt:
                continue
            topk = list(preds[:k])
            rel = [1 if c in tgt else 0 for c in topk]
            n_hit = sum(rel)
            recall += n_hit / len(tgt)
            precision += n_hit / k
            hit += 1.0 if n_hit > 0 else 0.0
            ideal = [1] * min(k, len(tgt))
            idcg = _dcg(ideal)
            ndcg += (_dcg(rel) / idcg) if idcg > 0 else 0.0
            # Average precision at K.
            hits_so_far = 0
            ap_sum = 0.0
            for i, r in enumerate(rel):
                if r:
                    hits_so_far += 1
                    ap_sum += hits_so_far / (i + 1)
            denom = min(k, len(tgt))
            ap += (ap_sum / denom) if denom > 0 else 0.0
        out[f"Recall@{k}"] = round(recall / n, 6)
        out[f"Precision@{k}"] = round(precision / n, 6)
        out[f"NDCG@{k}"] = round(ndcg / n, 6)
        out[f"MAP@{k}"] = round(ap / n, 6)
        out[f"HitRatio@{k}"] = round(hit / n, 6)
    return out


def ndcg_at_k_per_sample(
    predictions: Sequence[Sequence[int]], targets: Sequence[Iterable[int]], k: int = 10
) -> np.ndarray:
    """Per-sample NDCG@k (one value per sample), for paired significance tests."""
    out = []
    for preds, tgt in zip(predictions, targets):
        tgt = set(tgt)
        if not tgt:
            out.append(0.0)
            continue
        rel = [1 if c in tgt else 0 for c in list(preds[:k])]
        idcg = _dcg([1] * min(k, len(tgt)))
        out.append((_dcg(rel) / idcg) if idcg > 0 else 0.0)
    return np.asarray(out, dtype=float)


def _binarise(targets: Sequence[Iterable[int]], vocab_size: int) -> np.ndarray:
    y = np.zeros((len(targets), vocab_size), dtype=int)
    for i, tgt in enumerate(targets):
        for c in tgt:
            y[i, c] = 1
    return y


def evaluate_classification(
    scores: np.ndarray,
    targets: Sequence[Iterable[int]],
    vocab_size: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Multi-label classification metrics over the course vocabulary.

    ``scores`` is an ``(n_samples, vocab_size)`` array of per-course
    probabilities. Predicted labels use ``threshold``. ROC-AUC uses the raw
    scores; macro ROC-AUC is averaged only over courses that have both a
    positive and a negative in the targets (others are undefined).
    """
    scores = np.asarray(scores, dtype=float)
    y_true = _binarise(targets, vocab_size)
    y_pred = (scores >= threshold).astype(int)

    out: dict[str, float] = {
        "accuracy_micro": round(float((y_pred == y_true).mean()), 6),
        "accuracy_macro": round(float((y_pred == y_true).mean(axis=0).mean()), 6),
        "precision_micro": round(
            float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 6
        ),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "recall_micro": round(
            float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 6
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }

    # ROC-AUC. Micro over the flattened arrays; macro over columns with both
    # classes present.
    try:
        out["roc_auc_micro"] = round(
            float(roc_auc_score(y_true.ravel(), scores.ravel())), 6
        )
    except ValueError:
        out["roc_auc_micro"] = float("nan")

    valid_cols = [j for j in range(vocab_size) if 0 < y_true[:, j].sum() < len(y_true)]
    if valid_cols:
        out["roc_auc_macro"] = round(
            float(
                roc_auc_score(
                    y_true[:, valid_cols], scores[:, valid_cols], average="macro"
                )
            ),
            6,
        )
        out["roc_auc_macro_n_courses"] = len(valid_cols)
    else:
        out["roc_auc_macro"] = float("nan")
        out["roc_auc_macro_n_courses"] = 0
    return out


def _self_test() -> None:
    # Sample 0: perfect ranking. Sample 1: one of two hit at rank 1.
    predictions = [[2, 0, 1, 3], [1, 4, 0, 2]]
    targets = [{0, 2}, {1, 3}]
    r = evaluate_ranking(predictions, targets, k_values=[1, 2])

    # Hand-computed expectations.
    # @1: s0 top1={2} hit -> recall 1/2, prec 1/1; s1 top1={1} hit -> recall 1/2, prec 1.
    assert abs(r["Recall@1"] - 0.5) < 1e-9, r["Recall@1"]
    assert abs(r["Precision@1"] - 1.0) < 1e-9, r["Precision@1"]
    assert abs(r["HitRatio@1"] - 1.0) < 1e-9, r["HitRatio@1"]
    # @2: s0 top2={2,0} both hit -> recall 1, prec 1, ndcg 1, ap 1.
    #     s1 top2={1,4} one hit at rank1 -> recall 1/2, prec 1/2,
    #       ndcg = (1/log2(2)) / (1/log2(2)+1/log2(3)) = 1/1.63093 = 0.613147,
    #       ap = (1/1)/min(2,2) = 0.5.
    assert abs(r["Recall@2"] - 0.75) < 1e-6, r["Recall@2"]
    assert abs(r["Precision@2"] - 0.75) < 1e-6, r["Precision@2"]
    ndcg2_expected = (1.0 + (1.0 / (1.0 + 1.0 / math.log2(3)))) / 2
    assert abs(r["NDCG@2"] - round(ndcg2_expected, 6)) < 1e-5, (r["NDCG@2"], ndcg2_expected)
    assert abs(r["MAP@2"] - 0.75) < 1e-6, r["MAP@2"]

    # Classification: 3 courses, perfect scores -> all metrics 1.0.
    scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.95, 0.7]])
    ctargets = [{0, 2}, {1, 2}]
    c = evaluate_classification(scores, ctargets, vocab_size=3)
    assert abs(c["f1_micro"] - 1.0) < 1e-9, c["f1_micro"]
    assert abs(c["recall_micro"] - 1.0) < 1e-9, c["recall_micro"]
    assert 0.0 <= c["roc_auc_micro"] <= 1.0

    # Range sanity on the ranking metrics.
    for kk, vv in r.items():
        assert 0.0 <= vv <= 1.0, (kk, vv)

    print("metrics self-test: PASS")
    print("ranking:", r)
    print("classification:", c)


if __name__ == "__main__":
    _self_test()
