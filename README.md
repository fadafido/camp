# CAMP

Constraint-Aware Multi-term course recommendation — research project.

This is a **fresh rebuild**: it contains the reusable pipeline code only. All
data and results are regenerated from scratch by the pipeline; nothing in this
repository is a previously-computed dataset, figure, table, or paper number.

## Layout

- `src/constraint_engine/` — prerequisite DAG and constraint mask.
- `src/dataset/` — catalogue ingestion, prerequisite parsing, the student-cohort
  simulator, and the dataset-build orchestrator.
- `src/models/` — heterogeneous graph + GraphSAGE encoder, baselines, the CAMP
  model, evaluation metrics, and the analysis modules (ablation, statistical
  significance, fairness, explainability).
- `src/paper/` — figure and table **generators** (outputs are produced at run
  time, not stored here).
- `src/scraping/` — catalogue collection starting point (see its README).
- `src/utils/` — shared helpers (deterministic seeding).
- `data/raw/` — raw catalogue **source** inputs only (no extracted outputs).
- `data/intermediate/`, `data/cap_bench/` — empty; populated by the pipeline.

British English throughout. Deterministic; seed 42 governs project randomness.
