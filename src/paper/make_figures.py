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
# Figures are sized for a single Springer column (~6.3 in / 16 cm printed width)
# so every element stays legible (>= ~9 pt) without down-scaling. Base font 10 pt.
plt.rcParams.update({"font.size": 10, "axes.grid": False,
                     "axes.titlesize": 11, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "figure.dpi": DPI, "savefig.dpi": DPI})
COL_W = 6.3  # single-column printed width target (inches)

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
    # ~16:6.6 aspect at column width; box x-positions are computed so every box
    # (including the final "Next-term course set") sits fully inside the canvas.
    fig, ax = plt.subplots(figsize=(COL_W, 2.85), layout="constrained")
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.0); ax.axis("off")
    # Labels are wrapped to short lines so they fit inside the narrow column-width
    # boxes without spilling over the edges or into the arrow gaps.
    boxes = [
        ("Student\nhistory\n(transcript)", CB["grey"]),
        ("GraphSAGE\nencoder\n(prereq\ngraph)", CB["blue"]),
        ("State vector\n(emb +\ncompleted\n+ scalars)", CB["sky"]),
        ("MaskablePPO\npolicy", CB["green"]),
        ("Constraint\nmask\n(eligible\ncourses)", CB["orange"]),
        ("Next-term\ncourse set", CB["purple"]),
    ]
    n = len(boxes); w = 2.30; h = 3.0; left = 0.30; y0 = 1.9
    gap = (16 - 2 * left - n * w) / (n - 1)
    xs = [left + i * (w + gap) for i in range(n)]
    for x, (label, c) in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.05",
                                    fc=c, ec="black", alpha=0.9))
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center",
                fontsize=7.5, color="white", weight="bold")
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=12, color="black"))
    ax.text(8, 1.0, "0 prerequisite violations by construction", ha="center",
            fontsize=8, style="italic", color=CB["red"])
    ax.text(8, 6.5, "CAMP — Constraint-Aware Multi-term Planner",
            ha="center", fontsize=11, weight="bold")
    _save(fig, "F1_architecture")


# ------------------------------------------------------- F2 RL training curves
def fig2_rl_training_curves():
    rows = list(csv.DictReader((MODELS / "camp_training_curve.csv").open()))
    fig, ax = plt.subplots(figsize=(COL_W, 3.9), layout="constrained")
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
    fig, ax = plt.subplots(figsize=(COL_W, 4.0), layout="constrained")
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
    fig, ax = plt.subplots(figsize=(COL_W, 4.2), layout="constrained")
    b1 = ax.bar(x - wbar / 2, ndcg, wbar, label="NDCG@10", color=CB["blue"])
    b2 = ax.bar(x + wbar / 2, viol, wbar, label="Prereq-violation rate", color=CB["red"])
    ax.set_xticks(x); ax.set_xticklabels([v.replace("CAMP-", "") for v in order], rotation=15)
    # Headroom above the tallest bar (no-planning 0.660) so value labels and the
    # upper-left legend do not collide with any bar or its label.
    ax.set_ylabel("Value"); ax.set_ylim(0, 0.82)
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
    # Point at the body of the no-mask violation bar (not its top), so the arrow
    # never crosses the "0.368" value label sitting just above the bar.
    ax.annotate("mask off → violations appear",
                (1 + wbar / 2, 0.20), xytext=(2.15, 0.52), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=CB["red"]))
    ax.legend(loc="upper left"); ax.grid(True, axis="y", alpha=0.3)
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
    from collections import defaultdict
    by_level = defaultdict(list)
    for n in H.nodes():
        by_level[by[n]["level"]].append(n)
    # Layered left->right layout: one column per course level, nodes centred and
    # evenly spaced within a column. The densest column sets the figure height so
    # every label has room — no two labels overlap.
    ranks = sorted(by_level)
    rank_idx = {lvl: i for i, lvl in enumerate(ranks)}
    n_max = max(len(v) for v in by_level.values())
    num = lambda n: int(n.replace("KHAL_COSC_", ""))
    pos = {}
    for lvl, ns in by_level.items():
        ordered = sorted(ns, key=num)
        count = len(ordered)
        for i, n in enumerate(ordered):
            pos[n] = (rank_idx[lvl] * 1.0, i - (count - 1) / 2.0)
    height = max(7.0, n_max * 0.24 + 1.6)
    fig, ax = plt.subplots(figsize=(COL_W, height), layout="constrained")
    levels = [by[n]["level"] for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=90, node_color=levels, cmap="viridis", ax=ax)
    nx.draw_networkx_edges(H, pos, arrowstyle="-|>", arrowsize=7, edge_color=CB["grey"],
                           width=0.6, alpha=0.6, ax=ax, node_size=90)
    # Labels sit just to the right of each node (not on the coloured marker), so
    # they stay black-on-white and legible; columns are far enough apart that a
    # label never reaches the next column.
    label_pos = {n: (x + 0.12, y) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(H, label_pos, {n: n.replace("KHAL_COSC_", "") for n in H.nodes()},
                            font_size=9, horizontalalignment="left", ax=ax)
    top = (n_max - 1) / 2.0
    for lvl in ranks:
        ax.text(rank_idx[lvl] * 1.0, top + 1.1, f"{lvl}-level", ha="center",
                va="bottom", fontsize=9, weight="bold")
    ax.set_xlim(-0.45, (len(ranks) - 1) + 0.85)
    ax.set_ylim(-top - 1.0, top + 2.2)
    ax.axis("off")
    ax.set_title("Khalifa Computer Science prerequisite structure (COSC spine)\n"
                 "left → right = prerequisite → dependent; colour = course level",
                 fontsize=10)
    _save(fig, "F5_prereq_graph")


# --------------------------------------------------------------- F6 fairness
def fig6_fairness():
    fair = _load(RES / "fairness.json")
    # constrained_layout reserves space for the suptitle, so it no longer collides
    # with the two subplot titles.
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")
    # left: per-university NDCG@10 (illustrative subgroup), with viol annotated
    uni = fair["attributes"]["university"]["per_group"]
    names = list(uni); vals = [uni[n]["NDCG@10"] for n in names]
    axes[0].bar(names, vals, color=[CB["blue"], CB["orange"], CB["green"]])
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("CAMP NDCG@10")
    axes[0].set_title("NDCG@10 by institution", fontsize=10)
    for i, n in enumerate(names):
        axes[0].annotate(f"{uni[n]['NDCG@10']:.3f}\nviol {uni[n]['violation_rate']:.0f}",
                         (i, uni[n]["NDCG@10"]), ha="center", va="bottom", fontsize=8)
    # right: NDCG@10 disparity per attribute
    attrs = list(fair["attributes"])
    gaps = [fair["attributes"][a]["demographic_parity"]["ndcg10_gap"] for a in attrs]
    axes[1].barh(attrs, gaps, color=CB["purple"])
    axes[1].set_xlabel("NDCG@10 disparity (max−min)")
    axes[1].set_title("Demographic-parity gaps", fontsize=10)
    axes[1].set_xlim(0, max(gaps) * 1.28)  # room for the value labels
    for i, g in enumerate(gaps):
        axes[1].annotate(f"{g:.3f}", (g, i), va="center", fontsize=8)
    fig.suptitle("Fairness: zero prerequisite violations in every subgroup",
                 fontsize=11, weight="bold")
    _save(fig, "F6_fairness")


# --------------------------------------------------------------- F7 XAI
def fig7_xai():
    xai = _load(RES / "explainability.json")
    feats = xai["feature_importance_overall"]
    items = sorted(feats.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    # Stacked panels (chart on top, worked example below): both get the full column
    # width, so the bar chart and the "why-not" text are each legible and balanced.
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 6.4),
                             gridspec_kw={"height_ratios": [1.0, 0.85]}, layout="constrained")
    axes[0].barh(names, vals, color=CB["sky"])
    axes[0].set_xlabel("Permutation importance")
    axes[0].set_title("CAMP state-feature importance", fontsize=10)
    axes[0].set_xlim(0, max(vals) * 1.15)  # room for the value labels
    for i, v in enumerate(vals):
        axes[0].annotate(f"{v:.3f}", (v, i), va="center", ha="left", fontsize=8)
    # lower panel: a worked "why-not" from the Khalifa case study
    cs = next(c for c in xai["case_studies"] if c["university"] == "khalifa")
    axes[1].axis("off")
    lines = [f"Worked example — {cs['university']} ({cs['profile'].get('major_track')}), term {cs['term_index']}",
             "", f"Top recommendation: {cs['top_recommendation'].replace('KHAL_','')}",
             f"  unlocked by: {', '.join(p.replace('KHAL_','') for p in cs['graph_path_for_top']['prerequisites']) or '(none)'}",
             "", "Masked out (“why not”):"]
    for m in cs["masked_out_examples"]:
        need = ", ".join(x.replace("KHAL_", "") for x in (m.get("needs_one_of") or []))
        lines.append(f"  {m['course'].replace('KHAL_','')} — needs {need}")
    axes[1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10,
                 family="monospace", transform=axes[1].transAxes)
    fig.suptitle("Explainability: feature attribution + constraint “why-not”",
                 fontsize=11, weight="bold")
    _save(fig, "F7_xai")


# ----------------------------------------------- F8 RL environment loop (schematic)
def fig8_rl_env_flow():
    # Schematic of the constraint-masked RL loop; reward sub-weights are read from
    # the env module constants so the figure matches the code exactly.
    from src.models import rl_env as R
    fig, ax = plt.subplots(figsize=(COL_W, 3.7), layout="constrained")
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.2); ax.axis("off")
    boxes = [
        ("State  sₜ\n• GraphSAGE embedding\n• completed multi-hot\n• term, GPA, credits\n"
         "• unmet-rule counts", CB["sky"]),
        ("MaskablePPO policy\n+ constraint mask\n(eligible courses only)\n→ action aₜ\n"
         "(select one course)", CB["green"]),
        (f"Reward  rₜ\n+ requirement (W={R.W_REQ:g})\n+ rule done (W={R.W_RULE_DONE:g})\n"
         f"+ centrality (W={R.W_CENT:g})\n+ healthy GPA (W={R.W_GPA:g})\n"
         f"− overload (W={R.W_OVERLOAD:g})\n+ grad × speed (W={R.W_GRAD:g})", CB["orange"]),
        ("Environment step\ngrades sampled,\nterm t → t+1", CB["blue"]),
    ]
    n = len(boxes); w = 3.6; h = 4.2; left = 0.3; y0 = 2.7
    gap = (16 - 2 * left - n * w) / (n - 1)
    xs = [left + i * (w + gap) for i in range(n)]
    for x, (label, c) in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.04",
                                    fc=c, ec="black", alpha=0.9))
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center",
                fontsize=7, color="white", weight="bold")
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=12, color="black"))
    # Feedback arrow: next state sₜ₊₁ back to the State box (the per-term loop).
    ax.add_patch(FancyArrowPatch((xs[-1] + w / 2, y0), (xs[0] + w / 2, y0),
                                 connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                                 mutation_scale=14, color=CB["grey"], lw=1.4))
    ax.text(8, 0.55, "next state sₜ₊₁ — repeat each term until graduation or the "
            f"term cap (TERM_CAP={R.TERM_CAP})", ha="center", fontsize=8,
            style="italic", color=CB["grey"])
    ax.text(8, 7.7, "CAMP RL environment loop (constraint-masked MaskablePPO)",
            ha="center", fontsize=11, weight="bold")
    _save(fig, "F8_rl_env_flow")


# ------------------------------------------ F9 heterogeneous graph (schematic)
def fig9_graph_construction():
    from src.models import graph as G
    fig, ax = plt.subplots(figsize=(COL_W, 4.3), layout="constrained")
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
    # Two node-type clusters.
    ax.add_patch(FancyBboxPatch((0.5, 5.6), 6.6, 4.2, boxstyle="round,pad=0.1",
                                fc="#EAF3FA", ec=CB["blue"], lw=1.5))
    ax.add_patch(FancyBboxPatch((8.9, 5.6), 6.6, 4.2, boxstyle="round,pad=0.1",
                                fc="#FBF0E3", ec=CB["orange"], lw=1.5))
    ax.text(3.8, 9.45, "course nodes (194)", ha="center", fontsize=9.5, weight="bold", color=CB["blue"])
    ax.text(12.2, 9.45, "student nodes (train only;\nleakage-guarded)", ha="center",
            fontsize=9.5, weight="bold", color=CB["orange"])
    # A few representative node markers.
    cxs = [(1.6, 8.4), (3.0, 7.2), (4.6, 8.5), (5.9, 7.0), (2.4, 6.3), (4.9, 6.2)]
    for (x, y) in cxs:
        ax.add_patch(plt.Circle((x, y), 0.34, fc=CB["blue"], ec="black", zorder=3))
    sxs = [(10.2, 8.3), (11.8, 7.2), (13.4, 8.4), (12.6, 6.3), (10.6, 6.5)]
    for (x, y) in sxs:
        ax.add_patch(plt.Circle((x, y), 0.34, fc=CB["orange"], ec="black", zorder=3))
    # prereq_of edges (course -> course, within the course cluster).
    for a, b in [(0, 1), (1, 4), (2, 3), (0, 2), (3, 5)]:
        ax.add_patch(FancyArrowPatch(cxs[a], cxs[b], arrowstyle="-|>",
                                     mutation_scale=10, color=CB["blue"], lw=1.0, zorder=2))
    # similar_to edges (course <-> course, undirected co-enrolment).
    for a, b in [(0, 2), (1, 5)]:
        ax.add_patch(FancyArrowPatch(cxs[a], cxs[b], arrowstyle="-", lw=1.0,
                                     color=CB["green"], linestyle=(0, (4, 3)), zorder=1))
    # took / taken_by edges (student <-> course, between clusters).
    for s_i, c_i in [(0, 2), (1, 3), (2, 2), (3, 5), (4, 4)]:
        ax.add_patch(FancyArrowPatch(sxs[s_i], cxs[c_i], arrowstyle="-|>",
                                     mutation_scale=9, color=CB["grey"], lw=0.9, zorder=1))
    # Legend / edge-type key.
    ax.text(0.5, 4.7, "Edge types (heterogeneous):", fontsize=9, weight="bold")
    key = [
        (CB["blue"], "-|>", "(course) –prereq_of→ (course)        from the augmented prerequisite DAG"),
        (CB["grey"], "-|>", "(student) –took→ (course)  +  reverse (course) –taken_by→ (student)"),
        (CB["green"], "-", f"(course) –similar_to– (course)        co-enrolment Jaccard ≥ {G.SIMILARITY_JACCARD_THRESHOLD:g}"),
    ]
    for i, (col, style, txt) in enumerate(key):
        y = 4.0 - i * 0.85
        ax.add_patch(FancyArrowPatch((0.7, y), (2.2, y), arrowstyle=style, mutation_scale=10,
                                     color=col, lw=1.4,
                                     linestyle=(0, (4, 3)) if style == "-" else "solid"))
        ax.text(2.5, y, txt, va="center", fontsize=8)
    ax.text(8, 10.6, "Heterogeneous graph construction (GraphSAGE input)",
            ha="center", fontsize=11, weight="bold")
    _save(fig, "F9_graph_construction")


# ----------------------------------------------------- F10 scalability
def fig10_scalability():
    import numpy as np
    sc = _load(RES / "scalability.json")
    ps = sc["by_problem_size"]; vs = sc["by_vocabulary_size"]
    ns = [d["n_samples"] for d in ps]
    wall = [d["wall_seconds"] for d in ps]
    mspp = [d["ms_per_sample"] for d in ps]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 3.3), layout="constrained")
    # Left: total wall-clock (linear in N) + per-sample cost (amortising).
    ax0 = axes[0]
    ax0.plot(ns, wall, "o-", color=CB["blue"], label="wall-clock (s)")
    ax0.set_xlabel("Number of test samples (Khalifa)")
    ax0.set_ylabel("Inference wall-clock (s)", color=CB["blue"])
    ax0.tick_params(axis="y", labelcolor=CB["blue"])
    ax0.set_title("Inference time vs problem size", fontsize=10)
    ax0b = ax0.twinx()
    ax0b.plot(ns, mspp, "s--", color=CB["orange"], label="ms / sample")
    ax0b.set_ylabel("ms / sample", color=CB["orange"])
    ax0b.tick_params(axis="y", labelcolor=CB["orange"])
    pk = ps[-1]["peak_python_mb"]
    ax0.annotate(f"peak Python memory ≈ {pk:.0f} MB\n(flat in N)", (0.04, 0.92),
                 xycoords="axes fraction", fontsize=8, va="top")
    # Right: per-sample cost vs candidate-course vocabulary / graph size.
    labels = [f"{d['institution']}\n({d['vocabulary_size']})" for d in vs]
    vals = [d["ms_per_sample"] for d in vs]
    axes[1].bar(labels, vals, color=[CB["blue"], CB["orange"], CB["green"]])
    axes[1].set_ylabel("ms / sample")
    axes[1].set_xlabel("Institution (vocabulary size)")
    axes[1].set_title("Cost vs vocabulary / graph size", fontsize=10)
    axes[1].set_ylim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        axes[1].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
    fig.suptitle("CAMP inference scalability (CPU; existing policies)",
                 fontsize=11, weight="bold")
    _save(fig, "F10_scalability")


# ----------------------------------------------------- F11 SHAP (XGBoost baseline)
def fig11_shap():
    sh = _load(RES / "shap_xgboost.json")
    top = sh["top_features"][:10][::-1]  # ascending for barh
    names = [t["feature"] for t in top]
    vals = [t["mean_abs_shap"] for t in top]
    fig, ax = plt.subplots(figsize=(COL_W, 4.2), layout="constrained")
    ax.barh(names, vals, color=CB["purple"])
    ax.set_xlabel("mean |SHAP| (log-odds margin units)")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_title("XGBoost baseline — global SHAP feature importance (TreeSHAP)",
                 fontsize=10)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (v, i), va="center", ha="left", fontsize=8)
    # Place the caveat in the clear right-hand whitespace (below the two longest
    # bars) so it never overlaps a bar.
    ax.text(0.97, 0.42, "Tree-baseline complement.\nCAMP uses permutation\nimportance"
            " (SHAP is ill-defined\nunder action masking).", transform=ax.transAxes,
            ha="right", va="center", fontsize=8, style="italic", color=CB["grey"])
    _save(fig, "F11_shap")


def main() -> None:
    fig1_architecture()
    fig2_rl_training_curves()
    fig3_model_comparison_bars()
    fig4_ablation()
    fig5_prereq_graph()
    fig6_fairness()
    fig7_xai()
    fig8_rl_env_flow()
    fig9_graph_construction()
    fig10_scalability()
    fig11_shap()
    print("All 11 figures written to", OUT)


if __name__ == "__main__":
    main()
