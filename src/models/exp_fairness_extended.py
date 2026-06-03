"""Fairness extension — demographic-parity and equal-opportunity *differences*.

ADD-ONLY re-evaluation. The committed ``results/fairness.json`` already records,
per protected attribute, each group's selection/outcome rates (HitRatio@10,
Recall@10, NDCG@10, ...) computed by the fairness pipeline from the per-student
CAMP predictions. This module *derives* two further standard group-fairness
summaries from those committed per-group rates, without re-running any model and
without changing any committed value, and writes them to a NEW file
``results/fairness_extended.json``.

Definitions (stated precisely so the artefact is self-contained)
---------------------------------------------------------------
For a protected attribute with groups ``g`` and a per-group rate ``r_g`` already
in fairness.json:

  * **Demographic-parity difference** — the max-min spread of the *selection
    rate* across groups. Selection rate is operationalised as **HitRatio@10**:
    the proportion of students in the group who receive at least one truly
    relevant course among the top-10 recommendations (the favourable
    recommendation outcome). ``DP_diff = max_g HitRatio@10_g - min_g HitRatio@10_g``.

  * **Equal-opportunity difference** — the max-min spread of the *true-positive
    rate on relevant items* across groups. TPR-on-relevant is operationalised as
    **Recall@10**: the mean fraction of a student's actually-taken (relevant)
    next-term courses that appear in the top-10. ``EO_diff = max_g Recall@10_g -
    min_g Recall@10_g``.

Because the per-group HitRatio@10 / Recall@10 in fairness.json are themselves the
means over each group's per-student predictions, the max-min of those committed
per-group means equals the group-level DP/EO difference computed from the raw
predictions — so this derivation introduces no new modelling and cannot alter a
committed value.

This is complementary to (and does not replace) the existing
``demographic_parity.ndcg10_gap`` and ``equal_opportunity`` (HitRatio@10 among
graduated students) keys already in fairness.json, which are left untouched.

Per-subgroup prerequisite-violation stays 0.000 (copied from fairness.json).
Seed 42; CPU-only; British English.
"""

from __future__ import annotations

import json
from typing import Any

from src.models import task

RESULTS_DIR = task.BUNDLE / "results"
FAIRNESS_PATH = RESULTS_DIR / "fairness.json"
OUT_PATH = RESULTS_DIR / "fairness_extended.json"

DP_METRIC = "HitRatio@10"   # selection / favourable-recommendation rate
EO_METRIC = "Recall@10"     # true-positive rate on relevant items


def build() -> dict[str, Any]:
    with FAIRNESS_PATH.open() as fh:
        fair = json.load(fh)

    attributes: dict[str, Any] = {}
    max_subgroup_viol = 0.0
    for attr, info in fair["attributes"].items():
        per_group = info["per_group"]
        sel = {g: per_group[g][DP_METRIC] for g in per_group}
        tpr = {g: per_group[g][EO_METRIC] for g in per_group}
        viol = {g: per_group[g]["violation_rate"] for g in per_group}
        max_subgroup_viol = max(max_subgroup_viol, max(viol.values()))
        sel_vals = list(sel.values())
        tpr_vals = list(tpr.values())
        attributes[attr] = {
            "groups": sorted(per_group),
            "demographic_parity_difference": {
                "definition": f"max-min of the per-group selection rate "
                f"(selection rate = {DP_METRIC}: share of students receiving "
                f">=1 relevant course in the top-10).",
                "selection_rate_per_group": {g: round(sel[g], 6) for g in sel},
                "max_group": max(sel, key=sel.get),
                "min_group": min(sel, key=sel.get),
                "difference": round(max(sel_vals) - min(sel_vals), 6),
            },
            "equal_opportunity_difference": {
                "definition": f"max-min of the per-group true-positive rate on "
                f"relevant items (TPR = {EO_METRIC}: mean fraction of a student's "
                f"actually-taken next-term courses recovered in the top-10).",
                "tpr_relevant_per_group": {g: round(tpr[g], 6) for g in tpr},
                "max_group": max(tpr, key=tpr.get),
                "min_group": min(tpr, key=tpr.get),
                "difference": round(max(tpr_vals) - min(tpr_vals), 6),
            },
            "max_violation_rate_across_groups": round(max(viol.values()), 6),
        }

    return {
        "source": "derived from results/fairness.json committed per-group rates "
        "(no model re-run; existing values unchanged)",
        "n_test_samples": fair["n_test_samples"],
        "selection_rate_metric": DP_METRIC,
        "tpr_relevant_metric": EO_METRIC,
        "note": "Complements the existing demographic_parity.ndcg10_gap and "
        "equal_opportunity (HitRatio@10 among graduated) keys in fairness.json, "
        "which are unchanged. These are the selection-rate DP difference and the "
        "TPR-on-relevant EO difference, defined per attribute.",
        "attributes": attributes,
        "violation_rate_zero_in_every_subgroup": bool(max_subgroup_viol == 0.0),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = build()
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT_PATH}")
    for attr, info in out["attributes"].items():
        dp = info["demographic_parity_difference"]["difference"]
        eo = info["equal_opportunity_difference"]["difference"]
        print(f"  {attr:18s} DP(selection)diff={dp:.4f}  EO(TPR)diff={eo:.4f}  "
              f"max_viol={info['max_violation_rate_across_groups']}")
    print(f"violation 0 in every subgroup: {out['violation_rate_zero_in_every_subgroup']}")


if __name__ == "__main__":
    main()
