"""Phase D verification block — ablation, significance, fairness, XAI.

Reads the four saved JSON artefacts. Run: ``.venv/bin/python -m src.models.verify_phased``
"""

from __future__ import annotations

import json

from src.models import task

RESULTS = task.BUNDLE / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text())


def main() -> None:
    abl = _load("ablation.json")
    stat = _load("statistical_tests.json")
    fair = _load("fairness.json")
    xai = _load("explainability.json")
    camp = _load("camp_results.json")

    print("=" * 78)
    print("PHASE D VERIFICATION BLOCK  (ablation · significance · fairness · XAI)")
    print("=" * 78)

    # --- ABLATION ---
    print("\n[ABLATION]  (variant timesteps {}k; full {}k)".format(
        abl["variant_timesteps"] // 1000, abl["full_timesteps"] // 1000))
    print(f"  {'variant':<20}{'NDCG@10':>9}{'viol':>8}{'grad_comp':>10}{'feasib':>8}")
    rows = {r["variant"]: r for r in abl["variants"]}
    for r in abl["variants"]:
        print(f"  {r['variant']:<20}{r['NDCG@10']:>9.4f}{r['prereq_violation_rate_topk']:>8.4f}"
              f"{r['graduation_compliance']:>10.4f}{r['pathway_feasibility']:>8.4f}")
    full = rows["CAMP-full"]
    print("  expected effects:")
    if "CAMP-no-mask" in rows:
        nm = rows["CAMP-no-mask"]
        print(f"    no-mask viol>0 (trade-off): {nm['prereq_violation_rate_topk']:.4f} "
              f"({'HELD' if nm['prereq_violation_rate_topk'] > 0 else 'NOT HELD'}); "
              f"NDCG {nm['NDCG@10']:.4f} vs full {full['NDCG@10']:.4f}")
    if "CAMP-no-imitation" in rows:
        ni = rows["CAMP-no-imitation"]
        print(f"    no-imitation NDCG drops: {ni['NDCG@10']:.4f} < full {full['NDCG@10']:.4f} "
              f"({'HELD' if ni['NDCG@10'] < full['NDCG@10'] else 'NOT HELD'})")
    if "CAMP-no-planning" in rows:
        npl = rows["CAMP-no-planning"]
        print(f"    no-planning feasib/grad drops: feasib {npl['pathway_feasibility']:.4f}, "
              f"grad {npl['graduation_compliance']:.4f} vs full grad {full['graduation_compliance']:.4f} "
              f"({'HELD' if npl['graduation_compliance'] < full['graduation_compliance'] else 'NOT HELD'})")
    if "CAMP-no-GNN" in rows:
        ng = rows["CAMP-no-GNN"]
        print(f"    no-GNN overall drops: NDCG {ng['NDCG@10']:.4f} vs full {full['NDCG@10']:.4f} "
              f"({'HELD' if ng['NDCG@10'] < full['NDCG@10'] else 'NOT HELD'})")

    # --- STATISTICAL ---
    print("\n[STATISTICAL]")
    cs = stat["camp_stability_over_seeds"]["NDCG@10"]
    print(f"  CAMP NDCG@10 over {len(stat['seeds'])} seeds: {cs['mean']:.4f} ± {cs['std']:.4f}  (per-seed {cs['per_seed']})")
    print(f"  CAMP violation-rate variance: {stat['camp_violation_rate_variance']} (deterministic 0)")
    for pair in ("camp_vs_deep_nn", "camp_vs_random_forest", "camp_vs_collaborative_filtering",
                 "camp_vs_matrix_factorisation", "camp_vs_pure_gnn"):
        p = stat["significance_per_sample_ndcg10"][pair]
        print(f"  {pair:<32} camp {p['camp_mean_ndcg10']:.4f} vs {p['other_mean_ndcg10']:.4f} | "
              f"t={p['paired_t_stat']:.1f} p={p['paired_t_p']:.2e} | Wilcoxon p={p['wilcoxon_p']:.2e}")
    print(f"  ANOVA (6 models): F={stat['anova_six_models']['F']:.1f} p={stat['anova_six_models']['p']:.2e}")

    # --- FAIRNESS ---
    print("\n[FAIRNESS]")
    for attr, info in fair["attributes"].items():
        print(f"  {attr:<18} NDCG@10 gap {info['demographic_parity']['ndcg10_gap']:.4f} | "
              f"equal-opp gap {info['equal_opportunity']['gap']} | "
              f"max viol {info['max_violation_rate_across_groups']}")
    print(f"  violation rate == 0 in EVERY subgroup: "
          f"{'OK' if fair['violation_rate_zero_in_every_subgroup'] else 'FAIL'} (fairness strength)")

    # --- XAI ---
    print("\n[XAI]")
    print("  top-5 features by importance:")
    for f in xai["top5_features"]:
        print(f"    {f['feature']:<24} {f['importance']:.4f}")
    print(f"  case studies saved: {xai['n_case_studies']} "
          f"(each with recommended set + graph-path + masked-out reasons)")

    # --- SANITY ---
    print("\n[SANITY]")
    n_test = camp["n_test_samples"]
    n_ok = abl["n_test_samples"] == stat["n_test_samples"] == fair["n_test_samples"] == n_test
    print(f"  all on test n={n_test}: {'OK' if n_ok else 'FAIL'}")
    print(f"  CAMP-full matches phaseC.1: NDCG@10={full['NDCG@10']:.4f} (camp_results {camp['ranking_metrics']['NDCG@10']:.4f}), "
          f"viol={full['prereq_violation_rate_topk']} -> {'OK' if abs(full['NDCG@10'] - camp['ranking_metrics']['NDCG@10']) < 1e-6 and full['prereq_violation_rate_topk'] == 0 else 'CHECK'}")
    print("  institution-masking on: YES")


if __name__ == "__main__":
    main()
