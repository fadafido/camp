"""Generate the seven paper figures (F1-F7) at 300 dpi from the v3_3inst results.

Data-driven figures (F2, F3, F4, F6, F7) read their numbers from the result
JSONs / training-curve CSVs — no metric values are hard-coded in the plotting
code. F1 (architecture) and F5 (prerequisite-graph structure) are schematic /
structure-drawn. British English labels; a colour-blind-safe palette; no
proprietary product names. Run: ``.venv/bin/python -m src.paper.make_figures``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx

from src.constraint_engine import dag

_REPO = Path(__file__).resolve().parents[2]
RES = _REPO / "data" / "cap_bench" / "v3_3inst" / "results"
MODELS = _REPO / "data" / "cap_bench" / "v3_3inst" / "models"
INTER = _REPO / "data" / "intermediate" / "khalifa"
OUT = _REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

# Colour-blind-safe (Wong) palette.
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "grey": "#999999"}
plt.rcParams.update({"font.size": 11, "axes.grid": False,
                     "figure.dpi": DPI, "savefig.dpi": DPI})

MODEL_LABEL = {"collaborative_filtering": "CF", "matrix_factorisation": "MF",
               "random_forest": "RF", "gradient_boosted": "XGBoost",
               "deep_nn": "Deep NN", "pure_gnn": "Pure GNN", "camp": "CAMP"}
# Seven-model order: XGBoost in the strong-classifier slot just below RF; CAMP last.
MODEL_ORDER = ["collaborative_filtering", "matrix_factorisation", "random_forest",
               "gradient_boosted", "deep_nn", "pure_gnn", "camp"]


def _load(p):
    with p.open() as fh:
        return json.load(fh)


def _save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------- F1 architecture
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
    boxes = [
        (0.2, "Student\nhistory\n(transcript)", CB["grey"]),
        (2.1, "GraphSAGE\nencoder\n(prereq graph)", CB["blue"]),
        (4.0, "State vector\n(emb + completed\n+ scalars)", CB["sky"]),
        (6.0, "MaskablePPO\npolicy", CB["green"]),
        (8.0, "Constraint mask\n(eligible courses)", CB["orange"]),
        (10.0, "Next-term\ncourse set", CB["purple"]),
    ]
    w = 1.6
    for x, label, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.2, boxstyle="round,pad=0.05",
                                    fc=c, ec="black", alpha=0.85))
        ax.text(x + w / 2, 1.6, label, ha="center", va="center", fontsize=9, color="white", weight="bold")
    for x, _, _ in boxes[:-1]:
        ax.add_patch(FancyArrowPatch((x + w, 1.6), (x + 1.9, 1.6),
                                     arrowstyle="-|>", mutation_scale=16, color="black"))
    ax.text(8.8, 0.55, "0 prerequisite violations by construction", ha="center",
            fontsize=9, style="italic", color=CB["red"])
    ax.text(5.5, 2.85, "CAMP — Constraint-Aware Multi-term Planner",
            ha="center", fontsize=12, weight="bold")
    _save(fig, "F1_architecture")


# ------------------------------------------------------- F2 RL training curves
def fig2_rl_training_curves():
    rows = list(csv.DictReader((MODELS / "camp_training_curve.csv").open()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = {"khalifa": CB["blue"], "aus": CB["orange"], "unc": CB["green"]}
    label = {"khalifa": "Khalifa (CS)", "aus": "AUS (IS)", "unc": "UNC (Econ)"}
    for uni, c in cols.items():
        sub = [(int(r["timestep"]), float(r["eval_reward"])) for r in rows if r["institution"] == uni]
        sub.sort()
        ax.plot([t for t, _ in sub], [v for _, v in sub], marker="o", ms=3,
                color=c, label=label[uni])
    ax.set_xlabel("Training timesteps"); ax.set_ylabel("Evaluation reward")
    ax.set_title("CAMP training convergence (three per-institution policies, 200k)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "F2_rl_training_curves")


# ----------------------------------------------- F3 model comparison (trade-off)
def fig3_model_comparison_bars():
    summ = _load(RES / "models_summary.json")["models"]
    # XGBoost = the UNMASKED gradient_boosted variant (shown unmasked, like the
    # other baselines); the five baselines + CAMP come from models_summary.json.
    rec = dict(summ)
    rec["gradient_boosted"] = _load(RES / "gradient_boosted.json")["variants"]["gradient_boosted"]
    labels = [MODEL_LABEL[m] for m in MODEL_ORDER]
    ndcg = [rec[m]["ranking_metrics"]["NDCG@10"] for m in MODEL_ORDER]
    viol = [rec[m]["prereq_violation_rate_topk"] for m in MODEL_ORDER]
    import numpy as np
    x = np.arange(len(labels)); wbar = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    b1 = ax.bar(x - wbar / 2, ndcg, wbar, label="NDCG@10", color=CB["blue"])
    b2 = ax.bar(x + wbar / 2, viol, wbar, label="Prereq-violation rate", color=CB["red"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Value"); ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy vs constraint compliance — the key trade-off (seven models)")
    for r in b1:
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=7)
    # Violation bars: annotate every bar including zeros, and draw a thin baseline
    # tick under any zero-height bar so CAMP's measured 0.000 is visible (honest
    # zero — never a faked height). Matches F4's styling.
    for r in b2:
        h = r.get_height()
        if h == 0:
            ax.plot([r.get_x(), r.get_x() + r.get_width()], [0, 0],
                    color=CB["red"], linewidth=2.0, solid_capstyle="butt")
        ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=7)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "F3_model_comparison_bars")


# --------------------------------------------------------------- F4 ablation
def fig4_ablation():
    variants = {r["variant"]: r for r in _load(RES / "ablation.json")["variants"]}
    order = ["CAMP-full", "CAMP-no-mask", "CAMP-no-planning", "CAMP-no-imitation", "CAMP-no-GNN"]
    ndcg = [variants[v]["NDCG@10"] for v in order]
    viol = [variants[v]["prereq_violation_rate_topk"] for v in order]
    import numpy as np
    x = np.arange(len(order)); wbar = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - wbar / 2, ndcg, wbar, label="NDCG@10", color=CB["blue"])
    b2 = ax.bar(x + wbar / 2, viol, wbar, label="Prereq-violation rate", color=CB["red"])
    ax.set_xticks(x); ax.set_xticklabels([v.replace("CAMP-", "") for v in order], rotation=15)
    ax.set_ylabel("Value"); ax.set_ylim(0, 0.75)
    ax.set_title(f"Ablation: removing the mask flips violations 0 → {viol[1]:.3f}")
    # Annotate every NDCG bar with its value (matches F3's style).
    for r in b1:
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=7)
    # Annotate every violation bar with its value — crucially including the four
    # explicit "0.000" labels, so a reader sees those variants were *measured* at
    # zero (deliberate, mask-enforced) rather than omitted. Zero-height bars are
    # otherwise invisible. We never draw a bar taller than its true value; for the
    # zero bars we add a thin baseline tick at y=0 to mark the bar's x-position
    # without faking any height.
    for r in b2:
        h = r.get_height()
        cx = r.get_x() + r.get_width() / 2
        if h == 0:
            # 1px-equivalent baseline tick sitting exactly on y=0 (no fake height).
            ax.plot([r.get_x(), r.get_x() + r.get_width()], [0, 0],
                    color=CB["red"], linewidth=2.0, solid_capstyle="butt")
        ax.annotate(f"{h:.3f}", (cx, h), ha="center", va="bottom", fontsize=7)
    ax.annotate("mask off → violations appear",
                (1 + wbar / 2, viol[1]), xytext=(1.6, 0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=CB["red"]))
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "F4_ablation")


# --------------------------------------------- F5 prerequisite graph (structure)
def fig5_prereq_graph():
    courses = _load(INTER / "khalifa_courses.json")
    graph, _ = dag.build_real_inst_dag(courses, 3)
    by = {c["course_id"]: c for c in courses}
    # The CS spine: keep COSC courses (the major) for a readable structure figure.
    spine = [c["course_id"] for c in courses if c["subject_area"] == "COSC"]
    sub = graph.subgraph(spine).copy()
    # edge convention course -> prereq; draw prereq -> course (left to right by level)
    H = nx.DiGraph()
    for u, v in sub.edges():  # u requires v
        H.add_edge(v, u)
    H.add_nodes_from(sub.nodes())
    pos = {}
    from collections import defaultdict
    by_level = defaultdict(list)
    for n in H.nodes():
        by_level[by[n]["level"]].append(n)
    for lvl, ns in sorted(by_level.items()):
        for i, n in enumerate(sorted(ns)):
            pos[n] = (lvl / 100.0, i)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    levels = [by[n]["level"] for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=420, node_color=levels, cmap="viridis", ax=ax)
    nx.draw_networkx_edges(H, pos, arrowstyle="-|>", arrowsize=10, edge_color=CB["grey"],
                           width=0.8, ax=ax, node_size=420)
    nx.draw_networkx_labels(H, pos, {n: n.replace("KHAL_COSC_", "") for n in H.nodes()},
                            font_size=7, ax=ax)
    ax.set_title("Khalifa Computer Science prerequisite structure (COSC spine)\n"
                 "left → right = prerequisite → dependent; colour = course level")
    ax.set_xlabel("Course level (×100)"); ax.set_yticks([])
    ax.grid(False)
    _save(fig, "F5_prereq_graph")


# --------------------------------------------------------------- F6 fairness
def fig6_fairness():
    fair = _load(RES / "fairness.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    # left: per-university NDCG@10 (illustrative subgroup), with viol annotated
    uni = fair["attributes"]["university"]["per_group"]
    import numpy as np
    names = list(uni); vals = [uni[n]["NDCG@10"] for n in names]
    axes[0].bar(names, vals, color=[CB["blue"], CB["orange"], CB["green"]])
    axes[0].set_ylim(0, 1.0); axes[0].set_ylabel("CAMP NDCG@10")
    axes[0].set_title("CAMP NDCG@10 by institution (violation rate 0 in every group)")
    for i, n in enumerate(names):
        axes[0].annotate(f"{uni[n]['NDCG@10']:.3f}\nviol {uni[n]['violation_rate']:.0f}",
                         (i, uni[n]["NDCG@10"]), ha="center", va="bottom", fontsize=8)
    # right: NDCG@10 disparity per attribute
    attrs = list(fair["attributes"])
    gaps = [fair["attributes"][a]["demographic_parity"]["ndcg10_gap"] for a in attrs]
    axes[1].barh(attrs, gaps, color=CB["purple"])
    axes[1].set_xlabel("NDCG@10 disparity (max−min)")
    axes[1].set_title("Demographic-parity gaps")
    for i, g in enumerate(gaps):
        axes[1].annotate(f"{g:.3f}", (g, i), va="center", fontsize=8)
    fig.suptitle("Fairness: zero prerequisite violations in every subgroup", fontsize=12, weight="bold")
    _save(fig, "F6_fairness")


# --------------------------------------------------------------- F7 XAI
def fig7_xai():
    xai = _load(RES / "explainability.json")
    feats = xai["feature_importance_overall"]
    items = sorted(feats.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].barh(names, vals, color=CB["sky"])
    axes[0].set_xlabel("Permutation importance")
    axes[0].set_title("CAMP state-feature importance")
    for i, v in enumerate(vals):
        axes[0].annotate(f"{v:.3f}", (v, i), va="center", fontsize=8)
    # right: a worked "why-not" from the Khalifa case study
    cs = next(c for c in xai["case_studies"] if c["university"] == "khalifa")
    axes[1].axis("off")
    lines = [f"Worked example — {cs['university']} ({cs['profile'].get('major_track')}), term {cs['term_index']}",
             "", f"Top recommendation: {cs['top_recommendation'].replace('KHAL_','')}",
             f"  unlocked by: {', '.join(p.replace('KHAL_','') for p in cs['graph_path_for_top']['prerequisites']) or '(none)'}",
             "", "Masked out (“why not”):"]
    for m in cs["masked_out_examples"]:
        need = ", ".join(x.replace("KHAL_", "") for x in (m.get("needs_one_of") or []))
        lines.append(f"  {m['course'].replace('KHAL_','')} — needs {need}")
    axes[1].text(0.0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("Explainability: feature attribution + constraint “why-not”",
                 fontsize=12, weight="bold")
    _save(fig, "F7_xai")


def main() -> None:
    fig1_architecture()
    fig2_rl_training_curves()
    fig3_model_comparison_bars()
    fig4_ablation()
    fig5_prereq_graph()
    fig6_fairness()
    fig7_xai()
    print("All 7 figures written to", OUT)


if __name__ == "__main__":
    main()
