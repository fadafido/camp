# CAMP: Constraint-Aware Multi-term Planning for Prerequisite-Legal, Fairness-Evaluated Course Recommendation Across Three Institutions

*Module AI503 — Machine Learning, Assignment 2. British English throughout. All numerical claims are traceable to the locked results file `FACTS.md` and the underlying result JSONs (`data/cap_bench/v3_3inst/results/`). Seed 42; CPU-only.*

---

## 1. Abstract

**Purpose** – Academic course recommendation must do more than predict what a student is likely to take next: in a real degree programme a recommendation is only useful if it is *legal*, that is, if its prerequisites are satisfied and it fits the term's credit budget. Conventional recommenders optimise ranking accuracy and routinely surface high-scoring but ineligible courses; rule-based academic planning systems guarantee legality but cannot rank or personalise. This paper asks whether a single learned framework can deliver personalised, multi-term recommendations while *guaranteeing* prerequisite legality and pathway feasibility across heterogeneous institutions.

**Methodology** – We present CAMP (Constraint-Aware Multi-term Planning), which couples a heterogeneous GraphSAGE encoder, a constraint-masked reinforcement-learning planner (MaskablePPO), and a hard prerequisite-and-credit constraint engine. We build a reproducible three-institution benchmark from real public 2024–2025 undergraduate catalogues — Khalifa University (BSc Computer Science), the American University of Sharjah (BSBA Information Systems and Business Analytics), and the University of North Carolina at Chapel Hill (BS Economics) — comprising 194 catalogue courses and 4,500 synthetic, fully reproducible student transcripts. CAMP is compared against six baselines (collaborative filtering, matrix factorisation, random forest, deep neural network, a pure graph neural network, and gradient-boosted trees) through an institution-masked evaluation harness, with ablation, five-seed stability, analysis of variance, fairness auditing, and permutation-importance explainability.

**Findings** – CAMP attains a prerequisite-violation rate of 0.000 and a pathway-feasibility of 1.000 on the held-out test set, in every institution, in every fairness subgroup, and across all five random seeds — a structural guarantee that no unmasked baseline matches (the best baseline, the deep neural network, still violates prerequisites on 0.047 of its recommendations). This guarantee is delivered at a measured ranking cost: CAMP reaches NDCG@10 of 0.637, mid-pack on raw ranking and below the unconstrained deep network (0.793). Ablation shows the guarantee is owed entirely to the constraint mask: removing the mask at inference on the identical policy raises the violation rate to 0.368 and drops feasibility to 0.632.

**Conclusion** – CAMP demonstrates that prerequisite legality and multi-term feasibility can be guaranteed by construction within a learned recommender, generalising across three institutions and two academic fields, at a quantified and modest accuracy price.

**Limitations & future work** – The headline ranking is mid-pack; graduation-compliance reflects legal next-term advice rather than greedy whole-degree completion; transcripts are synthetic; and two of the three programmes use a reconstructed graduation threshold. To the best of our knowledge, this is the first framework integrating a GNN encoder, a constraint-masked RL planner and a hard prerequisite engine for explainable, fairness-evaluated multi-term course recommendation.

---

## 2. Introduction and Refined Research Gap

Choosing courses each term is one of the most consequential and least supported decisions an undergraduate makes. A good plan must respect a web of hard constraints — prerequisite chains, corequisites, credit-load limits, programme-specific requirement blocks — while also being personalised to the student's history, pace and interests, and while looking several terms ahead rather than one. The decision is genuinely multi-term: taking an introductory course this term unlocks an intermediate course next term and a capstone the term after, so a myopic "what is the single best next course" view systematically mis-serves students who need a coherent route to graduation.

The first assignment (A1) of this project surveyed the recommender-systems and educational-data-mining literature and identified three gaps. **Gap 1 — multi-term planning:** the dominant recommendation paradigms (collaborative filtering, matrix factorisation, sequential deep models, and graph neural networks) are tuned to predict the immediate next interaction and do not plan a multi-term trajectory toward a degree. **Gap 2 — guaranteed feasibility and constraint modelling:** learned recommenders treat prerequisites, if at all, as soft features, and therefore cannot guarantee that a recommendation is legal; conversely, rule-based academic planning systems guarantee legality but offer no learning, ranking or personalisation. **Gap 3 — deployment realities (fairness, explainability, reproducibility):** course-recommendation studies rarely audit subgroup fairness, rarely explain their recommendations, and rarely release a reproducible multi-institution benchmark.

This paper addresses Gaps 1 and 2 in full and Gap 3 in part. We argue the gap is real and not an artefact of metric choice: a recommender that scores well on ranking accuracy can still place an ineligible course at rank one, and in our experiments the strongest accuracy baseline does exactly that on roughly one recommendation in twenty. The contribution is therefore not a marginal ranking improvement but a *categorical* one on the safety axis — moving the violation rate to exactly zero by construction — together with the benchmark and evaluation protocol needed to measure that axis honestly.

We formalise the problem as constrained multi-term planning. A student is described by their completed courses and grades, their programme, and their academic state (start-state type, term-load type, accumulated credits). The system must output, for the next term, a ranked set of courses that (i) is prerequisite-legal given completed coursework, (ii) respects the term credit budget, and (iii) advances the student toward the programme's graduation requirements. We translate the A1 gaps into three research questions:

- **RQ1.** Can a learned, constraint-masked planner guarantee prerequisite legality and term feasibility (violation rate 0.000, feasibility 1.000) on unseen students?
- **RQ2.** What is the ranking-accuracy cost of that guarantee relative to unconstrained baselines that are free to recommend ineligible courses?
- **RQ3.** Does the guarantee hold uniformly across institutions, academic fields and demographic subgroups, and is the framework reproducible?

Our working hypothesis is that a hard constraint mask integrated into the policy's action space will drive the violation rate to zero independently of how well the policy ranks, so that legality and accuracy become separable axes rather than a single trade-off curve.

---

## 3. Research Objectives

The experiments are designed to test the three research questions through five concrete objectives. **O1 — Benchmark.** Construct a reproducible, multi-institution, mixed-field course-planning benchmark from real public catalogues, with synthetic but realistic student transcripts and a documented constraint structure. **O2 — Framework.** Implement CAMP as a GNN encoder, a constraint-masked RL planner and a hard constraint engine, and train one policy per institution under a fixed seed and compute budget. **O3 — Comparison.** Evaluate CAMP against six baselines through a shared, institution-masked harness on identical metrics, separating ranking accuracy from constraint-safety metrics. **O4 — Validity.** Establish that CAMP's guarantee is structural and stable through an ablation of its components, a five-seed stability study, and an analysis of variance across models. **O5 — Deployment readiness.** Audit subgroup fairness and provide model explanations, so the framework's behaviour is transparent and its disparities are quantified rather than assumed away.

---

## 4. Literature Review

**Collaborative filtering and matrix factorisation.** The foundational recommender paradigm infers preferences from co-occurrence in the user–item matrix. Neighbourhood methods recommend items taken by similar users [1], while latent-factor matrix-factorisation methods decompose the interaction matrix into low-rank user and item embeddings [2]. These methods are strong general recommenders but are constraint-blind: nothing prevents them from ranking a course whose prerequisites the student has not met, and they model neither sequence nor degree structure. In our study collaborative filtering and matrix factorisation are the weakest ranking baselines and among the highest violators.

**Deep-learning recommenders.** Neural collaborative filtering replaces the inner product with a learned interaction function [3], and multilayer perceptrons over engineered interaction features remain a robust workhorse. Such models capture richer non-linearities and, in our experiments, the deep neural network is the strongest baseline on ranking accuracy — but it still has no notion of legality and violates prerequisites on a non-trivial fraction of recommendations.

**Graph neural networks.** Because the student–course interaction structure and the prerequisite structure are naturally graphs, graph neural networks are a natural fit. Graph convolutional networks [4] and the inductive GraphSAGE aggregator [5] learn node embeddings by message passing over neighbourhoods; graph-based recommenders such as neural graph collaborative filtering [6] and its simplified successor [7] propagate collaborative signal along the user–item graph. We use a heterogeneous GraphSAGE encoder over a student–course graph to produce the course and student embeddings that condition the planner. A *pure* GNN recommender (embeddings scored directly) is included as a baseline and, tellingly, ranks poorly on its own — the graph encoder is valuable as a representation for the planner, not as a stand-alone ranker.

**Reinforcement learning for recommendation and planning.** Reinforcement learning frames recommendation as sequential decision-making, optimising long-horizon reward rather than one-step accuracy [8], [9]. Proximal Policy Optimization [10] is a stable on-policy algorithm widely used for discrete control, and value-based methods such as deep Q-networks [11] are common in interactive recommendation. RL is well suited to multi-term planning because the reward can encode progress toward graduation across many steps. Crucially for our purpose, action masking lets a policy operate over a restricted, state-dependent legal action set, which is the mechanism we exploit to guarantee legality.

**Knowledge graphs, ontologies and curriculum modelling.** A parallel line of work encodes domain knowledge — prerequisite relations, curricular ontologies, degree requirements — as explicit graphs or rule bases to inform or constrain recommendation [12], [13]. Educational-data-mining studies on course and grade prediction [14], [15] and on degree planning [16] motivate modelling the curriculum explicitly. CAMP follows this tradition by compiling the catalogue's prerequisites into a hard constraint engine, but unlike purely symbolic planners it pairs the engine with a learned ranker.

**Fairness and explainability in recommendation.** As recommenders enter consequential domains, subgroup fairness and explainability have become first-class concerns [17], [18]. Course recommendation directly affects student progression, so disparities across gender, prior attainment or entry route matter. We therefore audit per-subgroup violation and ranking, and provide permutation-importance explanations and worked case studies.

Across all five families, the literature optimises *what a student will do* and seldom guarantees *what a student is allowed to do*. CAMP targets exactly that gap, and the benchmark we release lets the safety axis be measured rather than assumed.

---

## 5. Proposed Methodology

CAMP is a three-stage framework — a graph encoder, a constraint-masked RL planner, and a hard constraint engine — whose overall architecture is shown in **Figure F1** and whose end-to-end workflow follows the canonical Data → Preprocessing → Model → Evaluation pipeline.

**Stage 1 — Heterogeneous GNN encoder.** A two-layer heterogeneous GraphSAGE network is trained over the student–course graph on a self-supervised link-prediction objective (predicting which student took which course). The encoder uses a 64-dimensional embedding, sum aggregation, dropout 0.3, the Adam optimiser at learning rate 0.01, and a 15% validation-edge split, all under seed 42. On held-out edges it reaches a link-prediction validation AUC of 0.97664, confirming that the learned course and student embeddings capture real co-enrolment structure. These embeddings condition the planner's state representation.

**Stage 2 — Constraint-masked RL planner.** One policy is trained per institution with MaskablePPO (the maskable Proximal Policy Optimization variant from the stable-baselines3-contrib library) on an MLP policy. At every step the environment exposes only the *legal* actions — courses whose prerequisites are satisfied by the student's completed coursework and which fit the remaining term credit budget — by supplying an action mask; the policy can therefore never select an ineligible course. The reward combines a planning term (progress toward outstanding graduation requirements) and an imitation term (agreement with the observed next-term set), weighted `w_plan = 1.0` and `w_imit = 12.0`, with warm-start episodes seeded from real mid-trajectory states. Training runs for 200,000 timesteps per institution (`n_steps = 2048`, batch size 64, discount 0.99, entropy coefficient 0.01, learning rate 0.001), seed 42, CPU-only.

**Stage 3 — Hard constraint engine.** A deterministic engine compiles each programme's prerequisite structure into an AND-of-OR form and enforces, at both training and inference, prerequisite satisfaction, credit-load limits and the programme's requirement blocks. The engine is the single source of the legality mask and of the evaluation's violation and feasibility metrics, so the same rules that constrain the policy also score it.

**Why the guarantee holds, and that it is model-agnostic.** The zero-violation result is not learned; it is a property of the action space. Because the mask removes ineligible courses before the policy ever ranks them, no legal-set recommendation can violate a prerequisite. The mechanism is not specific to CAMP: applying the identical eligibility mask to the gradient-boosted baseline at inference also drives its violation rate to exactly 0.000 (from 0.119), confirming that the mask, not the learner, is the source of the guarantee.

**Honest disclosures.** Four methodological choices are disclosed to avoid over-claiming. First, the *no-mask* ablation removes the mask at inference on the finalised policy rather than retraining a maskless policy; retraining without the mask is prohibitively slow because unmasked episodes rarely terminate. Second, the five-seed stability study trains at 100,000 timesteps per institution, half the 200,000-timestep headline budget, and its numbers are reported separately and never blended with the headline. Third, feature importance is computed by permutation importance rather than by SHAP. Fourth, the prerequisite graph is a real-plus-augmented hybrid: edges are parsed from the catalogues wherever the text provides them and a small number of curriculum-informed edges are added only where a higher-level course was left with no parsed prerequisite (real edges dominate everywhere: 56 real to 11 augmented at Khalifa, 97 to 2 at the American University of Sharjah, 104 to 10 at UNC).

---

## 6. Dataset

The benchmark is the core empirical contribution and is described here in full: how the data were collected, how courses were scoped, what the corpus contains, how it was preprocessed and annotated, the real programme structures it encodes, and the ethics of its construction. Its headline composition is summarised in **Table T1**.

### 6.1 Data collection

The catalogue structures are real and public; only the student transcripts are synthetic. Three 2024–2025 undergraduate catalogues were used, one academic field each, with an access date of 2 June 2026. **Khalifa University — BSc Computer Science.** The catalogue is served by a content-management system whose course pages are server-rendered static HTML; a custom Python scraper (`requests` + BeautifulSoup) fetched each course-detail page and the programme-requirements page directly from the live catalogue, with every fetched page cached under `data/raw/khalifa/` for provenance. Each page exposes the course code, title, credit hours and a verbatim prerequisite string. **The American University of Sharjah — BSBA Information Systems and Business Analytics.** The catalogue is distributed as a single PDF; a custom parser built on `pdfminer.six` extracted course codes, titles, the lecture–lab–credit triple, and verbatim prerequisite text from the flattened text, and separately extracted the degree-requirement structure (the programme blocks and their credit totals). **UNC Chapel Hill — BS Economics.** The catalogue is a public HTML site; a `requests` + BeautifulSoup scraper fetched the BS Economics programme page and the Economics, Mathematics, Statistics-and-Operations-Research and Computer-Science course pages, caching them under `data/raw/unc/`. All three scrapers are committed under `src/scraping/`, so the extraction is reproducible. Prerequisite extraction was clean throughout: zero prerequisite strings failed to parse.

### 6.2 Filtering and scoping

The raw extraction yields far more courses than any one programme requires, so each institution's catalogue is scoped to the *academic course universe* of its target programme: the major subject, the support courses the programme actually references, and the transitive prerequisite closure of those courses. General-education distribution requirements and unconstrained free electives — which are not rankable programme courses — are deliberately excluded from the recommender's scope. The scoped totals are 59 courses for Khalifa Computer Science, 61 for the American University of Sharjah Information Systems programme, and 74 for UNC Economics, giving **194 courses** in total. The compiled prerequisite graph for each institution is a directed acyclic graph (verified acyclic for all three): Khalifa has 56 real and 11 augmented edges with a maximum prerequisite chain depth of six; the American University of Sharjah has 97 real and 2 augmented edges, depth five; UNC has 104 real and 10 augmented edges, depth four.

### 6.3 Dataset description

The corpus pairs the 194 real catalogue courses with **4,500 synthetic students** — 1,500 per institution — each simulated only against their own programme, prerequisite graph and constraint engine, so there are no cross-institution recommendations. Each student has a full multi-term transcript; in total the corpus contains 179,098 enrolment records (176,534 graded). The domain is higher-education course planning and the language is English. At the student level the data are split 3,150 / 675 / 675 into train / validation / test, stratified by institution; at the prediction (history → next-term) sample level this yields 29,940 training, 6,320 validation and 6,454 test samples. The population is realistic and diverse: graduation rate 0.767 overall (Khalifa 0.727, the American University of Sharjah 0.815, UNC 0.759), mean final GPA 2.95, population pass rate 0.955, and mean terms-to-graduate 10.70. Term loads are non-uniform, averaging 3.74 courses per term (standard deviation 1.29, range one to six). The subgroup mix spans start-state type (2,682 new, 879 mid-degree, 700 transfer, 239 returning), term-load type (3,124 full-time, 918 part-time, 458 overload), gender (2,253 female, 2,247 male), GPA band (2,252 medium, 1,156 high, 1,092 low), and seven field-appropriate elective tracks. These attributes are recorded per student to enable the fairness audit of Section 8.

### 6.4 Example records

A representative course record is Khalifa's COSC 434, *Introduction to Machine Learning*, a 400-level, three-credit course whose parsed prerequisites are the AND-of-OR structure `[[COSC 330], [MATH 204], [MATH 243]]` — that is, COSC 330 *and* MATH 204 *and* MATH 243, each a single-alternative group. A genuine disjunction appears in the American University of Sharjah's ISA 303, *Introduction to Systems Analysis and Design*, whose prerequisite is `[[ISA 301, CMP 320]]` — ISA 301 *or* CMP 320 — encoding the catalogue's explicit "or" choice. A representative prediction sample is the second-term record of synthetic student `S_KHALIFA_00001`: the history is `{CHEM 115: 4.0, COSC 114: 3.5, MATH 111: 2.1, COSC 011: 3.4}` and the target — the set the student actually took next term — is `{MATH 112, COSC 202, PHYS 121, COSC 101, COSC 230}`. The recommender is scored on how well its ranked next-term recommendation recovers that target.

### 6.5 Preprocessing and annotation

Every course is normalised to a common schema: integer credit hours, a hundreds-based level band (100/200/300/400), a subject-area code, and a structured prerequisite field. Prerequisite text is the most demanding step. A recursive-descent parser tokenises each verbatim prerequisite string and compiles its course-code logic into AND-of-OR groups, handling glued and spaced course codes, comma-as-AND, explicit "and"/"or" connectives, parenthesised grouping, cross-listed codes and bare-number inheritance. Non-course tokens — grade thresholds ("a grade of C or better"), standing or classification requirements, permission-of-instructor clauses, placement tests and advisory "recommended" courses — are stripped into a separate notes field and never treated as hard prerequisites, and corequisites ("credit or enrolment in") are likewise separated. Annotation is therefore automatic and rule-based; across all three institutions, **zero** prerequisite strings were unparseable, so no course required manual prerequisite annotation.

### 6.6 The real American-University-of-Sharjah programme structure

A distinctive feature of the benchmark is that one institution's graduation requirement is encoded from the catalogue's *actual* block structure rather than reconstructed. The American University of Sharjah BSBA Information Systems and Business Analytics degree is a 123-credit programme (minimum CGPA 2.00) composed of General Education (36 credits), an Innovation and Entrepreneurship requirement (3 credits), a Business Core (45 credits), Major Requirements (18 credits, including the two "ISA 301 or CMP 320" and "ISA 303 or COE 420" choices), Major Electives (a minimum of 12 credits from a named pool), and Free Electives (9 credits). The recommender's modelled academic scope is the 78 credits that are rankable programme courses — the Business Core, the Innovation and Entrepreneurship requirement, the Major Requirements and the Major Electives — while the 36 credits of general-education distribution and 9 credits of free electives sit outside scope. One disclosure applies: the catalogue's bulleted Business Core lists thirteen courses (39 credits), and ACC 201 and ACC 202 are taken as the two further courses that complete the stated 45-credit core, an inference supported by their immediate adjacency in the flattened catalogue text, their place in the first-year study plan, and the exact arithmetic closure (39 + 6 = 45). Khalifa Computer Science and UNC Economics, whose catalogues do not expose clean block-credit splits, instead use a reconstructed realistic threshold — the curricular *gateway* courses (those that are prerequisites of other courses) are all required, plus an elective-credit minimum tuned to the native degree total — which is disclosed as a reconstruction in the dataset's datasheet.

### 6.7 Ethics

The benchmark uses no real student records. Every transcript is synthetic, generated by the constraint-aware simulator and fully reproducible from seed 42; the only real data are public course-catalogue structures. There is therefore no privacy exposure, and the dataset can be released and re-run without consent or de-identification concerns.

---

## 7. Experimental Setup

**Models.** Seven systems are evaluated under one harness. The six baselines are collaborative filtering, matrix factorisation (32 latent factors), a random forest (200 trees, minimum two samples per leaf), a deep neural network (a multilayer perceptron with 256- and 128-unit hidden layers, ReLU activations, dropout 0.3, trained with binary cross-entropy and the Adam optimiser at learning rate 0.001, batch size 256, with early stopping), a pure GNN recommender (the GraphSAGE embeddings scored directly), and gradient-boosted trees (XGBoost, version 2.1.4, 300 trees, maximum depth 6, learning rate 0.1, histogram tree method, a one-vs-rest binary head per course, seed 42). The proposed system is CAMP, as specified in Section 5.

**Training setup.** The GNN encoder and every model are trained under seed 42 on CPU. CAMP trains MaskablePPO for 200,000 timesteps per institution with the hyperparameters given in Section 5. The software stack is PyTorch 2.12, PyTorch Geometric 2.7, stable-baselines3-contrib 2.8 (MaskablePPO), gymnasium 1.2, scikit-learn 1.6, and XGBoost 2.1.4. All numbers in this paper are re-read from the written result JSONs rather than from in-memory values.

**Evaluation metrics.** Recommendation quality is measured by NDCG@10 (ranking) and Recall@10. Constraint safety is measured by two metrics computed by the constraint engine. The **prerequisite-violation rate** is the fraction of top-k recommended courses whose prerequisites are not satisfied by the student's completed coursework. **Pathway-feasibility** is the mean fraction of the top-k recommended courses that are eligible under the encoded prerequisite and term-load constraints given the student's completed coursework; a feasibility of 1.000 means every recommended course could legally be taken next term. We additionally report **graduation-compliance** (the mean fraction of recommended courses that count toward an outstanding programme requirement) and the multilabel ROC-AUC (micro-averaged). Fairness is evaluated as the per-subgroup violation rate and NDCG across the recorded attributes, and statistical significance by a one-way analysis of variance across the models' per-sample NDCG. **Institution-masked scoring** is applied throughout: each student is scored only against the courses of their own institution, so a model is never credited or penalised for ranking a course a student could not take because it belongs to another programme. The held-out test set contains 6,454 prediction samples.

---

## 8. Results and Analysis

### 8.1 Model comparison

**Table T2** reports the seven systems on the held-out test set (n = 6,454), with the baselines evaluated unmasked, and **Figure F3** visualises the comparison. CAMP attains NDCG@10 0.637, Recall@10 0.712, ROC-AUC 0.864, **prerequisite-violation rate 0.000**, and **pathway-feasibility 1.000**. The deep neural network is the strongest ranker (NDCG@10 0.793) but violates prerequisites on 0.047 of its recommendations and reaches feasibility only 0.953; the random forest (0.736) and gradient-boosted trees (0.716 unmasked) rank well but violate at 0.153 and 0.119 respectively; collaborative filtering (0.535) and matrix factorisation (0.512) rank weakly and violate at 0.242 and 0.145; the pure GNN ranks worst (0.188) and violates most (0.373).

| Model | NDCG@10 | Violation rate | Pathway-feasibility | ROC-AUC |
|---|--:|--:|--:|--:|
| Collaborative filtering | 0.535 | 0.242 | 0.758 | 0.950 |
| Matrix factorisation | 0.512 | 0.145 | 0.855 | 0.947 |
| Random forest | 0.736 | 0.153 | 0.847 | 0.977 |
| Gradient-boosted (XGBoost, unmasked) | 0.716 | 0.119 | 0.881 | 0.977 |
| Deep neural network | 0.793 | 0.047 | 0.953 | 0.984 |
| Pure GNN | 0.188 | 0.373 | 0.627 | 0.894 |
| **CAMP (ours)** | **0.637** | **0.000** | **1.000** | 0.864 |

### 8.2 Interpretation: why the results came out as they did

The central finding is that legality and ranking accuracy behave as separable axes. On the **safety axis CAMP improves over every baseline categorically**, not marginally: its violation rate of 0.000 is below even the best baseline's 0.047 (the deep network), and its feasibility of 1.000 exceeds the best baseline's 0.953. This is by construction — the mask removes ineligible courses before ranking — and the ablation in Section 8.4 confirms it is owed to the mask rather than to learning. On the **ranking axis CAMP is mid-pack** (0.637), below the unconstrained deep network and the tree baselines. The reason is structural rather than a failure of the encoder: guaranteeing legality removes from CAMP's candidate set exactly the high-scoring-but-illegal courses that the unconstrained baselines are free to place near the top of their lists. A baseline can earn ranking credit by recommending a popular advanced course a student is not yet eligible for; CAMP cannot, because that course is masked out. The accuracy gap is therefore the *measured price of the guarantee*, and it is modest. The pure-GNN result is also instructive: graph embeddings scored directly rank poorly (0.188), which shows the encoder's value lies in conditioning the planner, not in standalone ranking — consistent with the strong-encoder finding below.

### 8.3 Per-institution behaviour

CAMP's guarantee holds in every institution while its ranking varies by programme, as shown in **Table T6**. Per-institution NDCG@10 is 0.717 for the American University of Sharjah Information Systems programme, 0.663 for Khalifa Computer Science, and 0.541 for UNC Economics, with a violation rate of 0.000 in every institution (the global total of violations is zero). The lower UNC figure tracks that programme's larger, more elective-heavy course universe (74 courses) and shallower prerequisite depth, which admit more simultaneously-legal courses and so make the single observed next-term path harder to rank first. All three institutions' policies converged (the evaluation reward rose monotonically and stabilised under training), reported honestly: UNC trains to the lowest reward but its curve rises and stabilises rather than collapsing.

### 8.4 Ablation

**Table T3** and **Figure F4** report the ablation, with retrained variants trained at 100,000 timesteps per institution and CAMP-full reused at the 200,000-timestep headline. Removing the mask at inference on the finalised policy (CAMP-no-mask) is the only variant that breaks the guarantee: NDCG@10 falls to 0.336, the violation rate rises to 0.368, and feasibility drops to 0.632 — the identical policy becomes unsafe the moment the mask is removed, which is the cleanest possible demonstration that the mask is the guarantee. Among the retrained variants, all of which keep the mask and therefore retain violation 0.000 and feasibility 1.000, removing the imitation signal hurts ranking most (CAMP-no-imitation, 0.382), removing the GNN encoder costs moderately (CAMP-no-GNN, 0.601), and removing the planning reward costs least (CAMP-no-planning, 0.622, against the full 0.637). A complementary encoder study reinforces the encoder finding: a deliberately *stronger* three-layer, 128-dimensional GraphSAGE encoder reached a lower link-prediction AUC (0.657) than the committed two-layer, 64-dimensional encoder (0.977) and changed matched-budget CAMP ranking by only −0.001, so a larger encoder did not help.

### 8.5 Stability and significance

**Table T4** reports stability and significance. Retraining CAMP under five seeds (42, 123, 2024, 7, 99) at 100,000 timesteps gives a mean NDCG@10 of **0.6059 ± 0.0041**, with the violation rate and feasibility constant at 0.0 ± 0.0 and 1.0 ± 0.0 respectively — the guarantee is invariant to the seed, and the ranking is stable. These five-seed figures are kept strictly distinct from the 200,000-timestep headline of 0.6367 and are never blended with it. A one-way analysis of variance across the per-sample NDCG of the six comparison models is highly significant (F = 4383.88, p < 0.001), confirming that the differences in ranking accuracy between models are not due to chance.

### 8.6 Fairness

**Table T5** and **Figure F6** report the fairness audit. The headline fairness result mirrors the headline safety result: the **prerequisite-violation rate is 0.000 in every subgroup** of every attribute — gender, GPA band, start-state type, term-load, entry route, field track and institution — so the legality guarantee is uniform and not purchased at some subgroup's expense. Ranking quality is not perfectly uniform: the largest NDCG gap is across field track at 0.189 (from 0.522 for the policy track, n = 748, to 0.711 for the business-analytics track, n = 639), followed by institution (0.177), start-state type (0.125), cold-start status (0.123), gender (0.029) and GPA band (0.007). The small gender and GPA-band gaps are reassuring; the larger track, institution and entry-route gaps point to where ranking, not legality, is hardest.

### 8.7 Error analysis

CAMP's ranking errors are concentrated, and understanding where clarifies what the model does and does not do. Two regimes dominate. First, **elective-rich, late-degree states**: once a student has cleared the prerequisite backbone, many courses are simultaneously legal, but NDCG rewards only the single path the student actually took, so a recommendation that is entirely sensible and legal is scored as an error simply because the student chose a different legal elective. This is the main driver of the lower UNC figure and of the field-track fairness gap. Second, **cold-start and thin-history states**, where the model has little signal about the individual and falls back toward programme-typical recommendations; these states carry the largest fairness gap (the field-track gap of 0.189, and the cold-start gap of 0.123). Critically, in *both* regimes the recommendations remain prerequisite-legal: the errors are ranking deviations among legal options, never feasibility failures. The error profile is therefore qualitatively different from that of the unconstrained baselines, whose errors include recommending courses the student cannot take at all.

### 8.8 Explainability

Permutation importance (**Figure F7**) identifies the student's completed-courses signal as overwhelmingly the most informative feature (importance 0.641), far ahead of the GNN embedding (0.024), the term index (0.0067), the remaining-requirements signal (0.0067) and completed credits (0.004). This is intuitive and reassuring: the dominant determinant of a sound next-term recommendation is what the student has already passed, which is precisely the information that determines eligibility. Three worked case studies were generated to illustrate how a given student state maps to a recommended legal next-term set; they confirm that recommendations track academic state rather than collapsing to a single programme template.

### 8.9 On graduation-compliance

Graduation-compliance is 0.405 for CAMP, and it should be read carefully. CAMP optimises *legal, requirement-advancing next-term* recommendations under an imitation signal toward observed behaviour; it is not a greedy whole-degree completion planner that would maximise the share of recommendations counting toward unmet requirements at the expense of realism. A compliance of 0.405 therefore reflects that a realistic next term mixes requirement-advancing courses with legitimate electives and exploration, not a hidden failure of the planner. We flag it as a limitation of the present objective rather than disguise it.

---

## 9. Research Gap and Future Work

The experiments refine the A1 gaps into empirical, evidence-grounded limitations and a corresponding agenda. The clearest limitation is **mid-pack ranking under the constraint**: CAMP guarantees legality but ranks below unconstrained accuracy baselines, and closing that gap *without* relaxing the guarantee is the central open problem — for example by stronger imitation or contrastive objectives over the legal action set, or by ranking-aware fine-tuning that treats the many simultaneously-legal late-degree courses more gracefully than a single-path NDCG target allows. A second limitation is the **greedy reading of graduation-compliance**: a longer-horizon objective that explicitly trades next-term realism against whole-degree completion would let the framework target time-to-degree directly. A third is the use of **synthetic transcripts**: although they are realistic and reproducible, validation on real (consented, de-identified) enrolment histories is needed to confirm the ranking findings transfer, even though the legality guarantee is independent of the data. A fourth is the **reconstructed graduation thresholds** for two of the three institutions: learning or extracting exact requirement blocks for Khalifa and UNC, as we did for the American University of Sharjah, would remove a documented approximation. Beyond these, the framework should be extended with **capacity and timetabling constraints** (section availability, clashes, seat limits), tested for **scalability** to larger course universes and more institutions, and evaluated for **generalisation** to programmes and fields beyond the three studied here, as a step toward real-world deployment. Each of these goes beyond the A1 survey because it is identified by, and quantified against, the present experiments rather than anticipated in the abstract.

---

## 10. Conclusion

CAMP closes A1 Gaps 1 and 2 in full and Gap 3 in part. It contributes a reproducible, three-institution, mixed-field course-planning benchmark built from real public catalogues with fully synthetic, seed-reproducible transcripts, and a framework that *guarantees* prerequisite-legal, term-feasible multi-term recommendations across institutions, academic fields and demographic subgroups. On the held-out test set CAMP achieves a prerequisite-violation rate of 0.000 and a pathway-feasibility of 1.000 — uniformly across all three institutions, all five seeds and every fairness subgroup — making it the only one of the seven evaluated systems with zero violations, at a measured and modest ranking cost (NDCG@10 0.637 against the strongest unconstrained baseline's 0.793). The ablation shows the guarantee is structural: it is owed entirely to the constraint mask, and it is model-agnostic. The honest reading is that CAMP does not win on raw ranking; it wins on the axis that determines whether a recommendation is usable at all, and it does so by construction.

---

## 11. References

[1] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in *Proc. 10th Int. Conf. World Wide Web (WWW)*, 2001, pp. 285–295.

[2] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," *Computer*, vol. 42, no. 8, pp. 30–37, 2009.

[3] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in *Proc. 26th Int. Conf. World Wide Web (WWW)*, 2017, pp. 173–182.

[4] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.

[5] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 1024–1034.

[6] X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua, "Neural graph collaborative filtering," in *Proc. 42nd Int. ACM SIGIR Conf.*, 2019, pp. 165–174.

[7] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, "LightGCN: Simplifying and powering graph convolution network for recommendation," in *Proc. 43rd Int. ACM SIGIR Conf.*, 2020, pp. 639–648.

[8] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[9] M. M. Afsar, T. Crump, and B. Far, "Reinforcement learning based recommender systems: A survey," *ACM Computing Surveys*, vol. 55, no. 7, pp. 1–38, 2022.

[10] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv:1707.06347*, 2017.

[11] V. Mnih *et al.*, "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015.

[12] H. Wang *et al.*, "Knowledge graph convolutional networks for recommender systems," in *Proc. World Wide Web Conf. (WWW)*, 2019, pp. 3307–3313.

[13] Q. Guo *et al.*, "A survey on knowledge graph-based recommender systems," *IEEE Trans. Knowl. Data Eng.*, vol. 34, no. 8, pp. 3549–3568, 2022.

[14] A. Polyzou and G. Karypis, "Grade prediction with models specific to students and courses," *Int. J. Data Science and Analytics*, vol. 2, pp. 159–171, 2016.

[15] S. Morsy and G. Karypis, "Sparse neural attentive knowledge-based models for grade prediction," in *Proc. Int. Conf. Educational Data Mining (EDM)*, 2019.

[16] Z. A. Pardos and W. Jiang, "Designing for serendipity in a university course recommendation system," in *Proc. Int. Learning Analytics and Knowledge Conf. (LAK)*, 2020, pp. 350–359.

[17] M. D. Ekstrand *et al.*, "All the cool kids, how do they fit in? Popularity and demographic biases in recommender evaluation and effectiveness," in *Proc. Conf. Fairness, Accountability and Transparency (FAccT)*, 2018, pp. 172–186.

[18] Y. Zhang and X. Chen, "Explainable recommendation: A survey and new perspectives," *Foundations and Trends in Information Retrieval*, vol. 14, no. 1, pp. 1–101, 2020.

---

### Numerical self-check

Every numeric claim in this paper is sourced to `FACTS.md` or a committed result JSON: dataset composition, splits, graduation/GPA/pass/terms and subgroup mix (`dataset_stats.json`); DAG edge counts and depths and the 0-unparsed annotation (`dataset_stats.json`, ingest logs); the AUS 36/3/45/18/12/9 = 123 and 78-credit scope (`bsba_is_requirements.json`, `aus_is_structure.py`); encoder AUCs 0.97664 / 0.657 and the −0.001 delta (`embeddings.pt`, `camp_strong_encoder.json`); the seven-model table and n = 6,454 (`baselines_summary.json`, `gradient_boosted.json`, `camp_results.json`); CAMP per-institution 0.717/0.663/0.541 and graduation-compliance 0.405 (`camp_results.json`); the ablation figures (`ablation.json`); the five-seed 0.6059 ± 0.0041 and ANOVA F = 4383.88, p < 0.001 (`statistical_tests.json`); the fairness 0.000-everywhere result and the 0.189 maximum gap (`fairness.json`); the permutation-importance values and three case studies (`explainability.json`); and the CAMP/baseline hyperparameters (`camp_results.json`, baseline JSONs). No numeric claim in the paper is unsourced. Where a precise figure was unavailable in the results (for example, an exact decomposition of graduation-compliance, or course-universe sizes beyond the scoped 59/61/74/194), the text describes the quantity qualitatively rather than inventing a number.
