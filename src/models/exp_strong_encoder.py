"""Add-only ablation-style experiment — does a STRONGER encoder help CAMP?

Question: is CAMP's NDCG@10 gap to the supervised baselines an artefact of a
weak state encoder, or structural (the cost of the mask + planning objective +
single-term imitation)? We build ONE stronger GraphSAGE encoder, drop it into
CAMP's RL state with everything else identical, and re-measure. Framed as a CAMP
variant (CAMP vs CAMP-with-stronger-encoder) — NOT a new baseline, NOT a headline.

Stronger encoder (the only architectural change)
-------------------------------------------------
Committed encoder: 2-layer heterogeneous GraphSAGE, 64-d (its val link-prediction
AUC is read at runtime from the freshly-trained ``embeddings.pt`` artefact — no
number is hard-coded here). This variant: **3-layer, 128-d** GraphSAGE — deeper AND wider —
trained on the SAME heterogeneous graph, SAME link-prediction objective, SAME
val-edge split, SAME optimiser/lr/aggregator, seed 42. Saved to a NEW file
``models/embeddings_strong.pt`` (the committed ``embeddings.pt`` is untouched).

Apples-to-apples budget
-----------------------
CAMP is retrained at the reduced **100k timesteps/institution** (the ablation
budget) to fit the CPU budget. The 200k finalised CAMP (NDCG@10 0.637) is
therefore NOT the comparator. We freshly train **CAMP-full at 100k/64d/seed 42**
in this same run as the matched comparator, and report the delta against THAT.
(The Phase D 5-seed study's 100k CAMP-full mean 0.568 ± 0.008 corroborates the
comparator's scale.) Everything else — reward weights w_plan=1.0/w_imit=12.0,
PPO hyperparameters, mask, warm-start — is identical between the two arms.

ADD-ONLY: new encoder file + this module + ``results/camp_strong_encoder.json``.
No locked artefact (embeddings.pt, course vocab, camp_*.zip, results/*.json,
dataset bundle) is modified. Seed 42; CPU-only; British English.
"""

from __future__ import annotations

import copy
import csv
import json
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sb3_contrib import MaskablePPO
from torch_geometric.nn import SAGEConv, to_hetero

from src.models import baselines as bl
from src.models import camp
from src.models import gnn as gnn_mod
from src.models import graph as graph_mod
from src.models import task
from src.utils.seed import set_seed

MODELS_DIR = task.MODELS_DIR
RESULTS_DIR = task.BUNDLE / "results"
ORIG_EMB_PATH = MODELS_DIR / "embeddings.pt"
STRONG_EMB_PATH = MODELS_DIR / "embeddings_strong.pt"
STRONG_CURVE_PATH = MODELS_DIR / "gnn_strong_training_curve.csv"
OUT_PATH = RESULTS_DIR / "camp_strong_encoder.json"

BUDGET = 100_000  # matched to the ablation budget (CPU); comparator trained here too
SEED = 42


def _committed_encoder_auc() -> float | None:
    """Val link-prediction AUC of the committed 64-d encoder, read at runtime from
    the freshly-trained ``embeddings.pt`` artefact. Returns None if absent — never
    a hard-coded historical value."""
    if not ORIG_EMB_PATH.exists():
        return None
    blob = torch.load(ORIG_EMB_PATH, map_location="cpu", weights_only=False)
    auc = blob.get("final_val_auc")
    return float(auc) if auc is not None else None

# Stronger encoder: deeper (3 layers) and wider (128-d); else identical to gnn.py.
STRONG_HP = {
    "embedding_dim": 128, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
    "optimizer": "Adam", "lr": 0.01, "max_epochs": 100, "early_stopping_patience": 10,
    "val_edge_fraction": 0.15, "aggr": "sum",
    "objective": "link_prediction (student-took-course)", "seed": 42,
    "note": "Deeper+wider vs committed 2-layer/64-d; same objective/splits/seed.",
}


class SAGE3(torch.nn.Module):
    """3-layer GraphSAGE (deeper + wider); made heterogeneous via ``to_hetero``."""

    def __init__(self, hidden: int, out: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        self.conv3 = SAGEConv((-1, -1), out)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x


def train_strong_encoder() -> dict[str, Any]:
    """Train the 3-layer/128-d encoder on the SAME graph/objective/splits/seed.

    Mirrors ``gnn.train_gnn`` exactly except for the network (SAGE3, 128-d) and
    the output path. Reuses ``gnn._score`` and ``gnn._sample_negatives`` so the
    link-prediction objective and negative sampling are byte-for-byte the same.
    """
    data, vocab, student_index = graph_mod.build_graph()
    set_seed(42)
    gen = torch.Generator().manual_seed(42)

    took = data["student", "took", "course"].edge_index
    n_edges = took.size(1)
    n_students = data["student"].num_nodes
    n_courses = data["course"].num_nodes
    took_set = {(int(s), int(c)) for s, c in zip(took[0].tolist(), took[1].tolist())}

    perm = torch.randperm(n_edges, generator=gen)
    n_val = int(round(STRONG_HP["val_edge_fraction"] * n_edges))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_took, val_took = took[:, train_idx], took[:, val_idx]

    train_data = copy.copy(data)
    train_data["student", "took", "course"].edge_index = train_took
    train_data["course", "taken_by", "student"].edge_index = train_took.flip(0)

    model = to_hetero(
        SAGE3(STRONG_HP["hidden_dim"], STRONG_HP["embedding_dim"], STRONG_HP["dropout"]),
        data.metadata(), aggr=STRONG_HP["aggr"],
    )
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=STRONG_HP["lr"])
    val_neg = gnn_mod._sample_negatives(took_set, n_students, n_courses, val_took.size(1), gen)

    best_auc, best_state, best_epoch, since_best = -1.0, None, 0, 0
    curve: list[tuple[int, float, float]] = []
    for epoch in range(1, STRONG_HP["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        z = model(train_data.x_dict, train_data.edge_index_dict)
        train_neg = gnn_mod._sample_negatives(took_set, n_students, n_courses, train_took.size(1), gen)
        pos = gnn_mod._score(z["student"], z["course"], train_took)
        neg = gnn_mod._score(z["student"], z["course"], train_neg)
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z_val = model(train_data.x_dict, train_data.edge_index_dict)
            vp = gnn_mod._score(z_val["student"], z_val["course"], val_took)
            vn = gnn_mod._score(z_val["student"], z_val["course"], val_neg)
            scores = torch.cat([vp, vn]).sigmoid().numpy()
            ys = torch.cat([torch.ones_like(vp), torch.zeros_like(vn)]).numpy()
            val_auc = float(roc_auc_score(ys, scores))
        curve.append((epoch, float(loss.item()), val_auc))
        if val_auc > best_auc:
            best_auc, best_epoch, since_best = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= STRONG_HP["early_stopping_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_final = model(data.x_dict, data.edge_index_dict)
    torch.save(
        {
            "course_emb": z_final["course"].detach(),
            "student_emb": z_final["student"].detach(),
            "course_order": sorted(vocab, key=lambda c: vocab[c]),
            "student_ids": sorted(student_index, key=lambda s: student_index[s]),
            "hyperparams": STRONG_HP,
            "final_val_auc": best_auc,
            "best_epoch": best_epoch,
            "epochs_trained": len(curve),
        },
        STRONG_EMB_PATH,
    )
    with STRONG_CURVE_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_auc"])
        w.writerows(curve)
    committed_auc = _committed_encoder_auc()  # read from this run's embeddings.pt
    return {
        "architecture": "3-layer heterogeneous GraphSAGE, 128-d (deeper+wider)",
        "final_val_auc": round(best_auc, 6),
        "committed_encoder_val_auc": round(committed_auc, 6) if committed_auc is not None else None,
        "best_epoch": best_epoch, "epochs_trained": len(curve),
        "embedding_dim": STRONG_HP["embedding_dim"], "n_layers": STRONG_HP["n_layers"],
    }


def _train_eval_camp(emb_path, tag: str, engines, samples_test) -> tuple[dict, dict]:
    """Train CAMP's three policies at BUDGET with the encoder at ``emb_path`` and
    evaluate through the identical harness. Restores ``camp.EMB_PATH`` after."""
    saved = camp.EMB_PATH
    camp.EMB_PATH = emb_path  # make_env/_load_emb_map read this global at call time
    try:
        models, training = {}, {}
        for uni in camp.INSTITUTIONS:
            save_path = MODELS_DIR / f"strongenc_{tag}_{uni}.zip"  # NEW file (never camp_*.zip)
            info = camp.train_institution(uni, BUDGET, seed=SEED, save_path=save_path)
            training[uni] = {k: v for k, v in info.items() if k != "curve"}
            models[uni] = MaskablePPO.load(save_path)
            print(f"  [{tag}/{uni}] eval reward {info['first_quartile_eval_reward']} -> "
                  f"{info['last_quartile_eval_reward']} converged={info['converged']}")
        res = camp.evaluate_camp(models, samples_test, engines, eligible_mask=True, name=f"camp_{tag}")
        return res, training
    finally:
        camp.EMB_PATH = saved


def _suite(res: dict) -> dict[str, Any]:
    return {
        "NDCG@10": res["ranking_metrics"]["NDCG@10"],
        "Recall@10": res["ranking_metrics"]["Recall@10"],
        "f1_micro": res["classification_metrics"]["f1_micro"],
        "roc_auc_micro": res["classification_metrics"]["roc_auc_micro"],
        "prereq_violation_rate_topk": res["prereq_violation_rate_topk"],
        "pathway_feasibility": res["pathway_feasibility"],
        "graduation_compliance": res["graduation_compliance"],
        "ndcg10_by_university": res.get("ndcg10_by_university", {}),
    }


def run() -> dict[str, Any]:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    engines = {uni: bl._engine_for(uni) for uni in camp.INSTITUTIONS}
    samples_test = task.load_samples("test")

    print("Training stronger encoder (3-layer, 128-d) ...")
    enc = train_strong_encoder()
    print(f"  strong encoder val AUC {enc['final_val_auc']} (committed {enc['committed_encoder_val_auc']})")

    print(f"Training matched comparator: CAMP-full @ {BUDGET} steps, 64-d encoder ...")
    res_full, train_full = _train_eval_camp(ORIG_EMB_PATH, "full64d_100k", engines, samples_test)
    print(f"Training CAMP with stronger encoder @ {BUDGET} steps, 128-d encoder ...")
    res_strong, train_strong = _train_eval_camp(STRONG_EMB_PATH, "strong128d_100k", engines, samples_test)

    out = {
        "experiment": "CAMP with a stronger (deeper+wider) GraphSAGE encoder — ablation-style",
        "framing": "CAMP variant (encoder swap); NOT a new baseline or headline claim",
        "n_test_samples": len(samples_test),
        "budget_timesteps_per_institution": BUDGET,
        "comparator_note": (
            "Matched-budget comparator = CAMP-full trained at 100k/seed42/64-d IN THIS RUN. "
            "The 200k finalised CAMP (NDCG@10 0.637) is NOT the comparator. Phase D 5-seed "
            "100k CAMP-full mean 0.568 +/- 0.008 corroborates the comparator scale."
        ),
        "stronger_encoder": enc,
        "arms": {
            "camp_full_64d_100k": {"metrics": _suite(res_full), "training": train_full},
            "camp_strong_128d_100k": {"metrics": _suite(res_strong), "training": train_strong},
        },
        "ndcg10_delta_strong_minus_full": round(
            res_strong["ranking_metrics"]["NDCG@10"] - res_full["ranking_metrics"]["NDCG@10"], 6),
    }
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT_PATH}")
    return out


def verify() -> None:
    with OUT_PATH.open() as fh:
        out = json.load(fh)
    enc = out["stronger_encoder"]
    full = out["arms"]["camp_full_64d_100k"]
    strong = out["arms"]["camp_strong_128d_100k"]
    fm, sm = full["metrics"], strong["metrics"]

    print("\n================ VERIFICATION BLOCK (re-read from file) ================")
    print(f"stronger encoder: {enc['architecture']}")
    print(f"  val link-prediction AUC : {enc['final_val_auc']}  (committed 2-layer/64-d: {enc['committed_encoder_val_auc']})")
    print(f"budget used               : {out['budget_timesteps_per_institution']} timesteps/institution")
    print(f"matched comparator        : CAMP-full @100k @64d (this run) — NOT the 200k 0.637")
    print("\n-- convergence (eval reward first-q -> last-q; converged flag) --")
    for arm_name, arm in (("full64d", full), ("strong128d", strong)):
        for uni, t in arm["training"].items():
            print(f"  [{arm_name}/{uni}] {t['first_quartile_eval_reward']} -> "
                  f"{t['last_quartile_eval_reward']}  converged={t['converged']}")
    print("\n-- CAMP-strong (128d) full suite --")
    for k in ["NDCG@10", "Recall@10", "f1_micro", "roc_auc_micro",
              "prereq_violation_rate_topk", "pathway_feasibility", "graduation_compliance"]:
        print(f"  {k:<28}: {sm[k]}")
    print(f"  per-institution NDCG@10     : {sm['ndcg10_by_university']}")
    print("\n-- CAMP-full (64d, matched budget) for reference --")
    print(f"  NDCG@10={fm['NDCG@10']}  viol={fm['prereq_violation_rate_topk']}  "
          f"feasib={fm['pathway_feasibility']}  per-uni={fm['ndcg10_by_university']}")
    print(f"\n-- one-liner --")
    print(f"  strong-encoder CAMP NDCG@10 {sm['NDCG@10']} minus matched 64d-100k CAMP-full "
          f"{fm['NDCG@10']} = {out['ndcg10_delta_strong_minus_full']:+.6f}")
    print("==========================================================================\n")

    fail = []
    if sm["prereq_violation_rate_topk"] != 0.0:
        fail.append(f"STRONG arm violation rate = {sm['prereq_violation_rate_topk']} (must be 0.000 — mask broken)")
    if fm["prereq_violation_rate_topk"] != 0.0:
        fail.append(f"FULL(64d) arm violation rate = {fm['prereq_violation_rate_topk']} (must be 0.000)")
    not_conv = [f"{arm}/{uni}" for arm, a in (("full64d", full), ("strong128d", strong))
                for uni, t in a["training"].items() if not t["converged"]]
    if fail:
        raise SystemExit("STOP — verification failed:\n  - " + "\n  - ".join(fail))
    if not_conv:
        print(f"WARNING (report honestly, not a hard STOP): non-converging policies: {not_conv}")
    print("VERIFICATION: PASS (guards clean; result reported honestly above)")


if __name__ == "__main__":
    run()
    verify()
