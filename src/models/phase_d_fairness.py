"""Phase D Part 3 — fairness evaluation of CAMP across demographic groups.

Evaluates the finalised CAMP policies on the test split, grouped by gpa_band,
gender, major_track, cold_start, start_state_type and university. Per group:
NDCG@10, Recall@10, HitRatio@10, violation rate, graduation-compliance, pathway
feasibility.

Fairness metrics per attribute:
  * Demographic parity — max-min gap (and min/max ratio) in mean NDCG@10 and in
    graduation-compliance across the attribute's groups.
  * Equal opportunity — among students who actually graduated (the favourable
    outcome), the HitRatio@10 (does the top-10 contain a real next-term course)
    per group; the max-min gap is the equal-opportunity disparity.
  * Confirms the violation rate is 0 in every subgroup (the mask is
    group-independent — a fairness strength).

Writes ``results/fairness.json``. British English; seed 42.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from src.constraint_engine.engine import ConstraintEngine
from src.models import baselines as bl
from src.models import camp
from src.models import task

RESULTS_DIR = task.BUNDLE / "results"
ATTRIBUTES = ["gpa_band", "gender", "major_track", "cold_start", "start_state_type", "university"]


def _attr(sample: dict[str, Any], attr: str) -> str:
    if attr == "university":
        return sample["university"]
    return str(sample["demographics"][attr])


def _metrics_for(raw, samples, engines, idx) -> dict[str, float]:
    sub_raw = raw[idx]
    sub = [samples[i] for i in idx]
    m = bl.evaluate_model("camp", {}, sub_raw, sub, engines, eligible_mask=True)
    return {
        "n": len(idx),
        "NDCG@10": m["ranking_metrics"]["NDCG@10"],
        "Recall@10": m["ranking_metrics"]["Recall@10"],
        "HitRatio@10": m["ranking_metrics"]["HitRatio@10"],
        "violation_rate": m["prereq_violation_rate_topk"],
        "graduation_compliance": m["graduation_compliance"],
        "pathway_feasibility": m["pathway_feasibility"],
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    institutions = list(camp.INSTITUTIONS)  # khalifa, aus, unc
    engines = {u: bl._engine_for(u) for u in institutions}
    samples_test = task.load_samples("test")
    models = {u: MaskablePPO.load(task.MODELS_DIR / f"camp_{u}.zip") for u in institutions}
    raw = camp.camp_raw_scores(models, samples_test)

    # Student graduation status (favourable outcome for equal opportunity).
    grad_of = {s["student_id"]: s["graduated"] for s in task._load_students()}

    out: dict[str, Any] = {"n_test_samples": len(samples_test), "attributes": {}}
    all_subgroup_viol = []
    for attr in ATTRIBUTES:
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(samples_test):
            groups.setdefault(_attr(s, attr), []).append(i)
        per_group = {}
        eo = {}  # equal-opportunity: HitRatio@10 on graduated-student samples
        for g, idx in sorted(groups.items()):
            per_group[g] = _metrics_for(raw, samples_test, engines, idx)
            all_subgroup_viol.append(per_group[g]["violation_rate"])
            grad_idx = [i for i in idx if grad_of.get(samples_test[i]["student_id"])]
            if grad_idx:
                eo[g] = _metrics_for(raw, samples_test, engines, grad_idx)["HitRatio@10"]
        ndcgs = [v["NDCG@10"] for v in per_group.values()]
        comps = [v["graduation_compliance"] for v in per_group.values()]
        eo_vals = list(eo.values())
        out["attributes"][attr] = {
            "per_group": per_group,
            "demographic_parity": {
                "ndcg10_gap": round(max(ndcgs) - min(ndcgs), 6),
                "ndcg10_ratio": round(min(ndcgs) / max(ndcgs), 6) if max(ndcgs) else 0.0,
                "grad_compliance_gap": round(max(comps) - min(comps), 6),
            },
            "equal_opportunity": {
                "metric": "HitRatio@10 among graduated students",
                "per_group": {k: round(v, 6) for k, v in eo.items()},
                "gap": round(max(eo_vals) - min(eo_vals), 6) if eo_vals else None,
            },
            "max_violation_rate_across_groups": round(max(v["violation_rate"] for v in per_group.values()), 6),
        }

    out["violation_rate_zero_in_every_subgroup"] = bool(max(all_subgroup_viol) == 0.0)
    with (RESULTS_DIR / "fairness.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {RESULTS_DIR / 'fairness.json'}")
    print(f"violation rate 0 across ALL subgroups: {out['violation_rate_zero_in_every_subgroup']}")
    for attr, info in out["attributes"].items():
        print(f"  {attr}: NDCG@10 gap {info['demographic_parity']['ndcg10_gap']:.4f} "
              f"max_viol {info['max_violation_rate_across_groups']}")


if __name__ == "__main__":
    main()
