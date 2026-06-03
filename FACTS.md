# CAMP — FACTS (fresh run, dataset-locked-v3)

Single source of verified numbers. Every value is read from the committed result
JSON named in brackets. Seed 42; CPU-only; British English. Do not hand-edit —
regenerate from the result files.

## Dataset  [dataset_stats.json]
- Students: 4500 (1500/institution); splits 3150/675/675; sample-level test n=6454.
- Courses: 194 (Khalifa 59, AUS 61 real-structure, UNC 74).
- Overall graduation 0.7671, mean GPA 2.9497, pass rate 0.955, mean terms-to-grad 10.7005.
  - khalifa: grad 0.7273, GPA 2.952, terms 10.9423.
  - aus: grad 0.8153, GPA 2.9449, terms 10.4963.
  - unc: grad 0.7587, GPA 2.9521, terms 10.688.

## Encoder  [models/embeddings.pt; camp_strong_encoder.json]
- Committed GNN encoder val link-prediction AUC: 0.97664.
- Strong 3-layer/128-d encoder AUC: 0.657395 (WORSE than committed).
- CAMP NDCG delta strong-minus-full (100k matched): -0.00124 (no benefit).

## Model comparison — UNMASKED baselines, n_test=6454  [baselines_summary.json; gradient_boosted.json; camp_results.json]
model                       NDCG@10    viol  ROC-AUC  feasib
collaborative_filtering      0.5346  0.2417   0.9499     0.758279
matrix_factorisation         0.5119  0.1448   0.9468     0.855149
random_forest                0.7358  0.1530   0.9772     0.846968
deep_nn                      0.7930  0.0475   0.9838     0.95253
pure_gnn                     0.1883  0.3731   0.8938     0.626934
xgboost (unmasked)           0.7161  0.1186   0.9774 0.88135
xgboost (masked)             0.7380  0.0000   0.9712     1.0
CAMP (ours, 200k)            0.6367  0.0000   0.8644     1.0

## CAMP per-institution  [camp_results.json]
- NDCG@10: {'aus': 0.717262, 'khalifa': 0.66271, 'unc': 0.540668}; violation 0.000 everywhere (total 0); grad-compliance 0.40502.
- Convergence: khalifa=True; aus=True; unc=True

## Ablation  [ablation.json] (variants retrained at 200k, MATCHED to CAMP-full 200k)
- CAMP-full: NDCG@10 0.6367, viol 0.0, feasib 1.0
- CAMP-no-mask: NDCG@10 0.3364, viol 0.3683, feasib 0.6317 (inference-time mask removal on the 200k policy)
- CAMP-no-planning: NDCG@10 0.6600, viol 0.0, feasib 1.0
- CAMP-no-imitation: NDCG@10 0.4151, viol 0.0, feasib 1.0
- CAMP-no-GNN: NDCG@10 0.6251, viol 0.0, feasib 1.0

## 5-seed stability + ANOVA  [statistical_tests.json]
- CAMP 100k 5-seed NDCG@10 mean 0.605924 +/- 0.004067 (per-seed [0.608884, 0.606458, 0.607445, 0.597997, 0.608837]); violation 0.0+/-0.0; feasibility 1.0+/-0.0.
- ANOVA seven models (T2 systems incl. unmasked XGBoost): F=3973.8438, p<1e-300.
- CAMP vs XGBoost (unmasked) per-sample NDCG@10: CAMP 0.636658 < XGBoost 0.716066; paired t=-21.337 (p~1.14e-97), Wilcoxon p~2.35e-97 (CAMP honestly mid-pack on ranking).

## Fairness  [fairness.json]
- violation 0.000 in EVERY subgroup: True.
- Largest NDCG gap: major_track 0.1893; all gaps: gpa_band 0.0074, gender 0.0286, major_track 0.1893, cold_start 0.123, start_state_type 0.1253, university 0.1766

## XAI  [explainability.json]
- Permutation-importance top features: [{'feature': 'completed_courses', 'importance': 0.6413}, {'feature': 'gnn_embedding', 'importance': 0.024}, {'feature': 'term_index', 'importance': 0.0067}, {'feature': 'remaining_requirements', 'importance': 0.0067}, {'feature': 'credits_completed', 'importance': 0.004}]
- Case studies generated: 3.

## Structural guarantee
- CAMP prereq-violation 0.000 and pathway-feasibility 1.000 on test, per-institution, every subgroup, and across 5 seeds. Removing the mask at inference (CAMP-no-mask) breaks it (viol 0.368, feasib 0.632) — the guarantee is the mask.
