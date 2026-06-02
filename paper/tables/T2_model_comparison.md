| Model | NDCG@10 | Recall@10 | F1 (micro) | ROC-AUC (micro) | Prereq-violation rate | Graduation-compliance | Pathway-feasibility |
|---|---|---|---|---|---|---|---|
| Collaborative filtering | 0.535 | 0.616 | 0.183 | 0.950 | 0.242 | 0.546 | 0.758 |
| Matrix factorisation | 0.512 | 0.646 | 0.317 | 0.947 | 0.145 | 0.423 | 0.855 |
| Random forest | 0.736 | 0.792 | 0.410 | 0.977 | 0.153 | 0.373 | 0.847 |
| Gradient-boosted (XGBoost) | 0.716 | 0.779 | 0.517 | 0.977 | 0.119 | 0.360 | 0.881 |
| Deep NN | 0.793 | 0.835 | 0.591 | 0.984 | 0.047 | 0.377 | 0.953 |
| Pure GNN | 0.188 | 0.286 | 0.153 | 0.894 | 0.373 | 0.355 | 0.627 |
| CAMP (ours) | 0.637 | 0.712 | 0.363 | 0.864 | 0.000 | 0.405 | 1.000 |

_All baselines (including Gradient-boosted / XGBoost) are shown UNMASKED. Applying the same eligibility mask to XGBoost drives its prereq-violation rate to 0.000 (from 0.119), confirming the mask is model-agnostic. [src] gradient_boosted.json :: variants._
