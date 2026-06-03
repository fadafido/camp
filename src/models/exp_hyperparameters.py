"""Collect the (already-recorded) hyperparameters into one result file.

ADD-ONLY documentation. Reads the real settings from the code constants and the
committed artefacts — nothing is invented or re-tuned:
  * GraphSAGE encoder config + committed link-prediction AUC  [models/embeddings.pt]
  * MaskablePPO config + combination weights                  [camp.PPO_HYPERPARAMS]
  * RL reward sub-weights + term cap                          [rl_env constants]
  * XGBoost baseline config                                   [gradient_boosted.json]
  * training (seed 42, CPU) + student split (3150/675/675)    [splits.json]

Writes ``results/hyperparameters.json`` (consumed by make_tables.py for T8).
Seed 42; CPU-only; British English.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from src.models import camp
from src.models import rl_env as R
from src.models import task

RESULTS_DIR = task.BUNDLE / "results"
OUT_PATH = RESULTS_DIR / "hyperparameters.json"


def build() -> dict[str, Any]:
    emb = torch.load(camp.EMB_PATH, weights_only=False)
    g = emb["hyperparams"]
    gb_hp = json.load((RESULTS_DIR / "gradient_boosted.json").open())["variants"][
        "gradient_boosted"]["hyperparameters"]
    splits = json.load(task.SPLITS_PATH.open())
    ppo = camp.PPO_HYPERPARAMS

    return {
        "source": "code constants + committed artefacts (no re-tuning, no model run)",
        "gnn_graphsage": {
            "n_layers": g["n_layers"],
            "hidden_dim": g["hidden_dim"],
            "embedding_dim": g["embedding_dim"],
            "dropout": g["dropout"],
            "aggregation": g["aggr"],
            "optimizer": g["optimizer"],
            "learning_rate": g["lr"],
            "objective": g["objective"],
            "val_link_prediction_auc": round(float(emb["final_val_auc"]), 6),
            "epochs_trained": emb["epochs_trained"],
            "seed": g["seed"],
        },
        "rl_maskable_ppo": {
            "algorithm": ppo["algorithm"],
            "policy": ppo["policy"],
            "learning_rate": ppo["learning_rate"],
            "n_steps": ppo["n_steps"],
            "batch_size": ppo["batch_size"],
            "gamma": ppo["gamma"],
            "ent_coef": ppo["ent_coef"],
            "total_timesteps": ppo["total_timesteps"],
            "w_plan": ppo["w_plan"],
            "w_imit": ppo["w_imit"],
            "warm_start": ppo["warm_start"],
            "seed": ppo["seed"],
        },
        "rl_reward_weights": {
            "W_REQ_requirement": R.W_REQ,
            "W_CENT_centrality": R.W_CENT,
            "W_RULE_DONE_rule_group": R.W_RULE_DONE,
            "W_GPA_healthy_gpa": R.W_GPA,
            "W_OVERLOAD_penalty": R.W_OVERLOAD,
            "W_GRAD_graduation": R.W_GRAD,
            "W_UNMET_terminal_penalty": R.W_UNMET,
            "W_IMIT_MATCH": R.W_IMIT_MATCH,
            "W_IMIT_MISS": R.W_IMIT_MISS,
            "term_cap": R.TERM_CAP,
            "healthy_term_credits": R.HEALTHY_TERM_CREDITS,
        },
        "xgboost_baseline": {
            "n_estimators": gb_hp["n_estimators"],
            "max_depth": gb_hp["max_depth"],
            "learning_rate": gb_hp["learning_rate"],
            "tree_method": gb_hp["tree_method"],
            "objective": gb_hp["objective"],
            "random_state": gb_hp["random_state"],
        },
        "training": {"seed": 42, "device": "CPU-only",
                     "n_test_samples": len(task.load_samples("test"))},
        "data_split_students": {k: len(v) for k, v in splits.items()},
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = build()
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"  GraphSAGE: {out['gnn_graphsage']['n_layers']}-layer "
          f"{out['gnn_graphsage']['embedding_dim']}-d, val AUC "
          f"{out['gnn_graphsage']['val_link_prediction_auc']}")
    print(f"  MaskablePPO: {out['rl_maskable_ppo']['total_timesteps']} timesteps; "
          f"split {out['data_split_students']}")


if __name__ == "__main__":
    main()
