"""Stage 3 — GraphSAGE embeddings via self-supervised link prediction.

Trains a 2-layer heterogeneous GraphSAGE on the course/student graph to predict
held-out ``(student, took, course)`` edges. The learned course and student
embeddings capture enrolment structure and course relationships; they are used
both as the "Pure GNN" baseline (Part 5e) and, later, as CAMP's state encoder
(Phase C).

Hyperparameters (recorded in ``embeddings.pt`` and the results JSON):
  embedding dim 64 · 2 SAGE layers · dropout 0.3 · Adam lr 0.01 · up to 100
  epochs · early stopping (patience 10) on validation link-prediction AUC.

To avoid leakage, the held-out validation edges are removed from the
message-passing graph during training; final saved embeddings are computed on
the full graph. Seed 42 throughout. CPU-only.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv, to_hetero

from src.models import graph as graph_mod
from src.models import task
from src.utils.seed import set_seed

MODELS_DIR = task.MODELS_DIR
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.pt"
CURVE_PATH = MODELS_DIR / "gnn_training_curve.csv"

HYPERPARAMS = {
    "embedding_dim": 64,
    "hidden_dim": 64,
    "n_layers": 2,
    "dropout": 0.3,
    "optimizer": "Adam",
    "lr": 0.01,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "val_edge_fraction": 0.15,
    "aggr": "sum",
    "objective": "link_prediction (student-took-course)",
    "seed": 42,
}


class SAGE(torch.nn.Module):
    """Plain 2-layer GraphSAGE; made heterogeneous via ``to_hetero``."""

    def __init__(self, hidden: int, out: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), out)
        # An nn.Dropout module (not F.dropout) so to_hetero/FX tracing respects
        # the train/eval flag — otherwise dropout would stay on during eval and
        # perturb the saved embeddings.
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def _score(z_student: torch.Tensor, z_course: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return (z_student[edges[0]] * z_course[edges[1]]).sum(dim=-1)


def _sample_negatives(
    took_set: set[tuple[int, int]],
    n_students: int,
    n_courses: int,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``n`` (student, course) pairs that are not real took-edges."""
    s_out, c_out = [], []
    while len(s_out) < n:
        k = 3 * (n - len(s_out))
        ss = torch.randint(0, n_students, (k,), generator=generator)
        cc = torch.randint(0, n_courses, (k,), generator=generator)
        for s, c in zip(ss.tolist(), cc.tolist()):
            if (s, c) not in took_set:
                s_out.append(s)
                c_out.append(c)
                if len(s_out) == n:
                    break
    return torch.tensor([s_out, c_out], dtype=torch.long)


def train_gnn(data, vocab: dict[str, int], student_index: dict[str, int]) -> dict[str, Any]:
    set_seed(42)
    gen = torch.Generator().manual_seed(42)

    took = data["student", "took", "course"].edge_index
    n_edges = took.size(1)
    n_students = data["student"].num_nodes
    n_courses = data["course"].num_nodes
    took_set = {(int(s), int(c)) for s, c in zip(took[0].tolist(), took[1].tolist())}

    # Split took-edges into message/supervision-train and held-out validation.
    perm = torch.randperm(n_edges, generator=gen)
    n_val = int(round(HYPERPARAMS["val_edge_fraction"] * n_edges))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_took = took[:, train_idx]
    val_took = took[:, val_idx]

    # Message-passing graph excludes the validation edges (both directions).
    train_data = copy.copy(data)
    train_data["student", "took", "course"].edge_index = train_took
    train_data["course", "taken_by", "student"].edge_index = train_took.flip(0)

    model = to_hetero(
        SAGE(HYPERPARAMS["hidden_dim"], HYPERPARAMS["embedding_dim"], HYPERPARAMS["dropout"]),
        data.metadata(),
        aggr=HYPERPARAMS["aggr"],
    )
    # Lazy-module warm-up so the optimizer sees initialised parameters.
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])

    # Fixed validation negatives for a stable AUC signal.
    val_neg = _sample_negatives(took_set, n_students, n_courses, val_took.size(1), gen)

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    patience = HYPERPARAMS["early_stopping_patience"]
    since_best = 0
    curve: list[tuple[int, float, float]] = []

    for epoch in range(1, HYPERPARAMS["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        z = model(train_data.x_dict, train_data.edge_index_dict)
        train_neg = _sample_negatives(took_set, n_students, n_courses, train_took.size(1), gen)
        pos = _score(z["student"], z["course"], train_took)
        neg = _score(z["student"], z["course"], train_neg)
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        # Validation AUC on held-out edges (message passing on train graph only).
        model.eval()
        with torch.no_grad():
            z_val = model(train_data.x_dict, train_data.edge_index_dict)
            vp = _score(z_val["student"], z_val["course"], val_took)
            vn = _score(z_val["student"], z_val["course"], val_neg)
            scores = torch.cat([vp, vn]).sigmoid().numpy()
            ys = torch.cat([torch.ones_like(vp), torch.zeros_like(vn)]).numpy()
            val_auc = float(roc_auc_score(ys, scores))
        curve.append((epoch, float(loss.item()), val_auc))

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    # Restore best and compute final embeddings on the FULL graph.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_final = model(data.x_dict, data.edge_index_dict)
    course_emb = z_final["course"].detach()
    student_emb = z_final["student"].detach()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "course_emb": course_emb,
            "student_emb": student_emb,
            "course_order": sorted(vocab, key=lambda c: vocab[c]),
            "student_ids": sorted(student_index, key=lambda s: student_index[s]),
            "hyperparams": HYPERPARAMS,
            "final_val_auc": best_auc,
            "best_epoch": best_epoch,
            "epochs_trained": len(curve),
        },
        EMBEDDINGS_PATH,
    )
    with CURVE_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_auc"])
        w.writerows(curve)

    return {
        "final_val_auc": round(best_auc, 6),
        "best_epoch": best_epoch,
        "epochs_trained": len(curve),
        "embedding_dim": HYPERPARAMS["embedding_dim"],
    }


def main() -> None:
    data, vocab, student_index = graph_mod.build_graph()
    info = train_gnn(data, vocab, student_index)
    print("GNN training:", info)


if __name__ == "__main__":
    main()
