| Variant | NDCG@10 | Prereq-violation rate | Graduation-compliance | Pathway-feasibility |
|---|---|---|---|---|
| CAMP-full | 0.637 | 0.000 | 0.405 | 1.000 |
| CAMP-no-mask | 0.336 | 0.368 | 0.364 | 0.632 |
| CAMP-no-planning | 0.660 | 0.000 | 0.372 | 1.000 |
| CAMP-no-imitation | 0.415 | 0.000 | 0.368 | 1.000 |
| CAMP-no-GNN | 0.625 | 0.000 | 0.403 | 1.000 |
| CAMP-full (64-d encoder, 100k)† | 0.617 | 0.000 | 0.401 | 1.000 |
| CAMP-strong-encoder (128-d, 100k)† | 0.616 | 0.000 | 0.456 | 1.000 |

† Encoder-capacity test at a **matched 100k-timestep budget** (seed 42), NOT the 200k finalised CAMP-full (NDCG@10 0.637) at the top of the table. A deeper+wider 128-d 3-layer GraphSAGE gives NDCG@10 0.616 vs the standard 64-d 2-layer 0.617 (Δ -0.001); violations stay 0.000. No NDCG gain from a stronger encoder. [src] camp_strong_encoder.json :: arms._
