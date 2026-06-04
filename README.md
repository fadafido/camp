# CAMP / CAP-Bench

A reproducible **dataset + provenance + experiment-code** repository for
constraint-aware multi-term course planning. It contains everything needed to
rebuild the **CAP-Bench** benchmark from public course catalogues and to re-run
every experiment end to end:

- the **scrapers** that collect three real public catalogues,
- the **cleaning / parsing / ingest** pipeline that normalises them,
- the **constraint engine** that builds each programme's prerequisite DAG,
- the **synthetic-student simulator** (seed 42),
- the **CAP-Bench bundle** itself,
- all **model / experiment code** and the **result JSONs** they produce, and
- the **figure and table generators** that render the result artefacts.

British English throughout. Deterministic and CPU-only; seed 42 governs all
project randomness. Vendor-neutral.

> **Student data is fully synthetic.** Only the course-catalogue *structures*
> (courses, credits, prerequisites, programme requirements) are catalogue-derived.
> No real student records are used.

---

## 1. Data sources — three real public catalogues

CAP-Bench draws one undergraduate programme, in a different field, from each of
three publicly published university course catalogues (access date **2 June
2026**):

| Institution | Programme | Field | Source format |
|---|---|---|---|
| Khalifa University | BSc Computer Science | Computer Science | SmartCatalogIQ static HTML (`ku-ae.smartcatalogiq.com`) |
| American University of Sharjah | BSBA Information Systems & Business Analytics | Information Systems | 2024–2025 undergraduate catalogue PDF |
| UNC Chapel Hill | BS Economics | Economics | 2024–2025 catalogue HTML (`catalog.unc.edu`) |

Each source exposes course titles, credit hours and **verbatim prerequisite
text**, plus the programme's referenced-course / requirement structure. Raw
inputs live under `data/raw/<inst>/` (see `data/raw/README.md`).

### Scraper modules (`src/scraping/`)

- `khalifa_catalog_scraper.py` — fetches the live SmartCatalogIQ course pages
  (server-rendered static HTML), caches them under
  `data/raw/khalifa/catalogue_pages/`, and parses title / credits / `prereq_raw`.
- `unc_catalog_scraper.py` — fetches the public `catalog.unc.edu` pages
  (`div.courseblock` / `table.sc_courselist`), caches them under
  `data/raw/unc/catalogue_pages/`, and parses the ECON major plus MATH / STOR /
  COMP support subjects.
- `aus_pdf_scraper.py` — parses the cached AUS 2024–2025 catalogue PDF
  (`data/raw/aus/raw_catalog_2024-2025.pdf`) with `pdfminer.six` for the BSBA
  Information Systems & Business Analytics course descriptions.
- `aus_requirements_extract.py` — extracts the AUS BSBA IS **programme-structure**
  pages (General Education / Business Core / Major / electives) from the same PDF,
  with verbatim excerpts so the structure can be audited as real.
- `aus_pdf_parser.py` — a reference/skeleton PDF parser (targets the AUS BSBA
  *Management* programme); not used to build the benchmark.

Each scraper writes a uniform pair consumed by the ingest layer:
`extracted_courses.json` (code, title, credits, verbatim `prereq_raw`) and
`extracted_program.json` (`referenced_course_codes`, `source_url`,
`access_date`).

---

## 2. Cleaning / parsing / ingest pipeline

`extracted_*.json` → structured CAP-Bench entities, via:

1. **`src/dataset/prereq_parser.py`** — shared free-text prerequisite parser.
   Parses each `prereq_raw` string into an AND-of-OR-groups array of prefixed
   `course_id`s. Faithful and never fabricated: grade thresholds, standing,
   permission, placement and other non-course tokens are stripped into
   `prereq_notes`; corequisites are kept separate; out-of-set references are
   dropped and counted; strings that cannot be reduced to AND-of-OR are logged.
2. **`src/dataset/real_ingest.py`** + per-institution configs
   (`khalifa_ingest.py`, `aus_ingest.py`, `unc_ingest.py`) — scope the courses
   to the major subject plus referenced support subjects, normalise to CAP-Bench
   v2.1 catalogue-course records (integer credits, level band, subject area), and
   apply the prerequisite parser. Outputs go to `data/intermediate/<inst>/`.
3. **`src/constraint_engine/`** — `dag.py` builds the hybrid prerequisite **DAG**
   for each programme; `engine.py` exposes `ConstraintEngine.eligible_courses`,
   the feasibility mask used throughout the model phases.

---

## 3. Synthetic-student simulator

`src/dataset/simulator.py` generates synthetic students, each belonging to **one**
institution and simulated against that institution's programme, prerequisite DAG
and constraint engine. Fully reproducible from **seed 42** plus the student's
global index. Realism features: a 60/20/15/5 start-state mix
(new / mid-degree / transfer / returning), variable per-term loads
(full-time / part-time / overload) with occasional summer terms, and a recorded
field track that gently biases elective choice. `grad_thresholds.py` reconstructs
a realistic per-programme graduation threshold.

---

## 4. The CAP-Bench bundle (`data/cap_bench/v3_3inst/`)

Built by **`src/dataset/generate_v3_3inst.py`**, which merges the three ingested
institutions (ID-collision guarded), simulates 1,500 students each (4,500 total),
and writes splits, statistics and a datasheet.

- `intermediate/` — merged courses, programmes, rules, rule groups, eligible
  courses and course features.
- `students/` — `students.jsonl`, flat `enrolments.csv` / `.parquet`, and
  stratified `splits.json`.
- `models/` — trained encoder embeddings, per-institution policies (incl.
  multi-seed and ablation variants), training curves and vocabularies.
- `results/` — every experiment's result JSON (see below).
- `dataset_stats.json`, `datasheet.md` — bundle composition and provenance.

Headline composition: 3 programmes, **194 courses** (Khalifa 59 / AUS 61 /
UNC 74), **4,500 synthetic students**, ~179k enrolment rows. See `datasheet.md`.

---

## 5. Experiment code and result files

Model and experiment code lives in `src/models/`:

- `graph.py`, `gnn.py` — heterogeneous course graph + GraphSAGE encoder.
- `camp.py` — CAMP: constraint-masked RL (MaskablePPO over the GraphSAGE
  encoder), one policy per institution; the constraint mask gives a structural
  ~0 prerequisite-violation guarantee at training and evaluation.
- `baselines.py` — collaborative filtering, matrix factorisation, random forest,
  deep NN and pure-GNN baselines.
- `exp_gradient_boosted_xgb.py` — gradient-boosted (XGBoost) baseline,
  masked and unmasked.
- `exp_strong_encoder.py` — encoder-capacity test (64-d/2-layer vs 128-d/3-layer)
  at a matched 100k-timestep budget.
- `phase_d_ablation.py`, `phase_d_statistical.py`, `phase_d_fairness.py`,
  `phase_d_xai.py` — ablation, 5-seed stability / ANOVA, fairness across
  subgroups, and permutation-importance explainability.
- `exp_paper_checks.py`, `verify_phased.py` — read-only audits / verification
  over the saved artefacts. `metrics.py`, `rl_env.py`, `task.py` are shared.

Each writes a JSON under `data/cap_bench/v3_3inst/results/` — e.g.
`baselines_summary.json`, `gradient_boosted.json`, `camp_results.json`,
`camp_strong_encoder.json`, `ablation.json`, `statistical_tests.json`,
`fairness.json`, `explainability.json`, plus `scalability.json` (inference
scalability), `fairness_extended.json` (demographic-parity / equal-opportunity
fairness), `shap_xgboost.json` (TreeSHAP for the gradient-boosted baseline) and
`hyperparameters.json` (full hyperparameter record). **`FACTS.md`** is the single
source of verified numbers, every value traceable to the result JSON named in
brackets.

### Result artefacts — figures and tables

`src/paper/` renders the result artefacts directly from the result JSONs
(no metric values are hard-coded in the generators):

- `make_figures.py` → `paper/figures/F1–F11` (300 dpi): architecture, RL training
  curves, model comparison, ablation, prerequisite-graph structure, fairness,
  explainability (F1–F7), plus the RL-environment loop (`F8_rl_env_flow`),
  heterogeneous-graph construction (`F9_graph_construction`), inference
  scalability (`F10_scalability`) and XGBoost TreeSHAP (`F11_shap`).
- `make_tables.py` → `paper/tables/T1–T9` (`.csv` + `.md`): dataset overview,
  model comparison, ablation, statistical significance, fairness, and
  per-institution breakdown (T1–T6), plus recommendation-ranking metrics
  (`T7_recommendation_metrics`), multilabel classification metrics
  (`T7_classification_metrics`), hyperparameter settings (`T8_hyperparameters`)
  and the inference-scalability profile (`T9_scalability`).

---

## 6. Reproduce end to end

> **Exact stack:** `requirements.txt` is fully pinned to the exact versions every
> committed result ran on (CPython 3.13.5, CPU-only, seed 42 — torch 2.12.0,
> torch-geometric 2.7.0, sb3-contrib 2.8.0, gymnasium 1.2.3, scikit-learn 1.6.1,
> xgboost 2.1.4, plus the scientific-Python core), and matches
> [`environment-lock.txt`](environment-lock.txt), the bit-for-bit reproducibility
> lock. PyTorch / PyTorch Geometric may need the CPU wheel index and, on macOS
> arm64, XGBoost needs `libomp` — both noted at the top of `requirements.txt`.

```bash
pip install -r requirements.txt   # fully pinned; mirrors environment-lock.txt

# 1. Collect catalogues (re-scrape; or use the cached data/raw/ inputs as-is)
python -m src.scraping.khalifa_catalog_scraper
python -m src.scraping.unc_catalog_scraper
python -m src.scraping.aus_pdf_scraper
python -m src.scraping.aus_requirements_extract

# 2. Ingest each institution into CAP-Bench entities
python -m src.dataset.khalifa_ingest
python -m src.dataset.aus_ingest
python -m src.dataset.unc_ingest

# 3. Build the three-institution bundle (merge + simulate students, seed 42)
python -m src.dataset.generate_v3_3inst

# 4. Encoder + baselines + CAMP
python -m src.models.gnn
python -m src.models.baselines
python -m src.models.exp_gradient_boosted_xgb
python -m src.models.camp
python -m src.models.exp_strong_encoder

# 5. Analysis (ablation, statistics, fairness, explainability) + verification
python -m src.models.phase_d_ablation
python -m src.models.phase_d_statistical
python -m src.models.phase_d_fairness
python -m src.models.phase_d_xai
python -m src.models.verify_phased

# 6. Render the result artefacts (figures + tables) from the result JSONs
python -m src.paper.make_tables
python -m src.paper.make_figures
```

Catalogue contents change over time; re-scraping may diverge from the cached
`data/raw/` inputs. To reproduce the committed benchmark exactly, build from the
cached raw inputs (skip step 1).

---

## Repository layout

- `src/scraping/` — catalogue scrapers / extractors (Khalifa, AUS, UNC).
- `src/dataset/` — prerequisite parser, ingest, simulator, bundle builder.
- `src/constraint_engine/` — prerequisite DAG and feasibility mask (+ tests).
- `src/models/` — encoder, baselines, CAMP, and the analysis experiments.
- `src/paper/` — figure and table generators (read live from result JSONs).
- `src/utils/` — deterministic seeding helpers.
- `data/raw/` — raw catalogue source inputs per institution.
- `data/intermediate/` — per-institution ingested entities.
- `data/cap_bench/v3_3inst/` — the CAP-Bench bundle (data, models, results).
- `paper/figures/`, `paper/tables/` — rendered result artefacts.
- `FACTS.md` — single source of verified numbers, each traceable to a result JSON.
