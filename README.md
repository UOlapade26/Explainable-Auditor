# The Explainable Auditor

**Evaluating post-hoc interpretability for machine-learning-based financial transaction fraud detection.**

This repository holds the complete computational work behind my dissertation. It
trains six fraud-detection models on the PaySim mobile-money dataset, evaluates how
*stable* their explanations are rather than how accurate their predictions are, and
packages the result into a deployable four-tier auditing application.

Everything in the results chapter is produced by
[`paysim_ecgb_pipeline.ipynb`](paysim_ecgb_pipeline.ipynb). I have left the cell
outputs in the notebook so that every figure and table can be traced back to the run
that produced it.

---

## The question I was actually asking

The literature already establishes that gradient boosting detects fraud well. My
question was a different one: can the *explanations* we attach to such a model be
trusted?

An explanation that changes arbitrarily between two near-identical transactions is of
no use to an investigator and no defence to a regulator, regardless of how accurate
the underlying prediction is. So I measure explanation consistency directly, and I
propose a model that tries to earn it during training rather than smooth it on
afterwards.

**ECGB (Explanation-Consistent Gradient Boosting)** adds a consistency penalty to the
XGBoost training objective. Every synthetic row SMOTE generates is paired with the
real minority row it was derived from, and the penalty pulls the model's prediction
for the synthetic row toward its parent's. At `lambda = 0` it reduces exactly to
ordinary logistic boosting, which is what makes the ablation a clean comparison.

**DASH (Diverse-Aggregated SHAP)** is my control condition. It trains five diverse
CatBoost models and averages their post-hoc SHAP values, the obvious cheap way to
stabilise explanations, requiring no custom objective at all. If ECGB cannot beat
DASH, the custom objective does not earn its complexity.

---

## What I found

All figures below are from the run recorded in the notebook.

### Predictive performance (held-out test set)

| Model | Recall | Precision | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| CatBoost | 0.9976 | 0.9414 | 0.9687 | **0.9986** | 0.9996 |
| LightGBM | 0.9976 | 0.9664 | 0.9817 | 0.9984 | 0.9998 |
| XGBoost | 0.9976 | 0.9568 | 0.9768 | 0.9983 | 0.9999 |
| Decision Tree | 0.9982 | 0.8691 | 0.9292 | 0.9980 | 0.9993 |
| **ECGB (mine)** | 0.9805 | **1.0000** | **0.9902** | 0.9976 | 0.9995 |
| MLP | 0.9927 | 0.2883 | 0.4468 | 0.9643 | 0.9992 |

ECGB has the best F1 and perfect precision but not the best PR-AUC. More importantly,
**none** of the differences between ECGB and the boosting baselines are statistically
significant: a paired Wilcoxon test across the five cross-validation folds returns
p = 0.0625 for every comparison, which is simply the smallest value the test can
return at n = 5. I report this rather than leaning on the leaderboard ordering,
because with five folds the test cannot resolve differences this small either way.

### Explanation consistency - fraud class

| Model | Cosine | Jaccard@5 | Random-pair Jaccard | Gap |
|---|---|---|---|---|
| **DASH** | 0.9442 | **0.8279** | - | +0.1738 |
| CatBoost | 0.8723 | 0.7663 | 0.5978 | +0.1686 |
| LightGBM | 0.9227 | 0.7473 | 0.5633 | +0.1840 |
| XGBoost | 0.9215 | 0.7356 | 0.5244 | **+0.2112** |
| Decision Tree | 0.9048 | 0.7263 | 0.5879 | +0.1385 |
| ECGB | 0.8813 | 0.7040 | - | +0.1989 |
| MLP | 0.8695 | 0.6618 | - | +0.1947 |

This is the finding I did not expect and did not want. **On the fraud class, ECGB has
the lowest Jaccard consistency of the tree models, and DASH the cheap post-hoc
control - has the highest.** ECGB does lead on the *legitimate* class (Jaccard 0.7955,
the highest of any model), and its neighbour-versus-random gap is second only to
XGBoost, meaning its explanations do track transaction similarity rather than
collapsing onto one generic answer. But the headline claim I set out to make about
the fraud class is not supported by these numbers, and the dissertation says so.

The gap column is the reason I did not stop at the raw consistency score. A model
returning the same explanation for every fraud transaction scores perfectly on raw
consistency while being useless; only the gap distinguishes genuine local stability
from homogenisation.

### The mechanism test

The consistency table above measures k-nearest-neighbour pairs in the test set, which
is the right measure for the research question but does not test the mechanism I claim
is responsible. So I tested the mechanism on its own terms: the trained ECGB against
an identical `lambda = 0` twin, compared on the 1,000 synthetic-parent pairs the
penalty actually operates on.

| Metric | ECGB | λ = 0 twin | Difference | p (paired Wilcoxon, n = 1000) |
|---|---|---|---|---|
| Cosine | 0.9850 | 0.9788 | +0.0062 | 4.2 × 10⁻⁸ |
| Jaccard@5 | 0.9418 | 0.9294 | +0.0124 | 2.5 × 10⁻³ |

The penalty does what I claim it does, significantly and on the pairs it acts on. The
effect is small, and it does not transfer to fraud-class consistency in the test set
which is itself the interesting result, and the one I spend the discussion chapter on.

### SHAP and LIME disagree with each other

| Model | LIME local fidelity (R²) | SHAP–LIME top-5 Jaccard |
|---|---|---|
| ECGB | **0.3791** | 0.3440 |
| MLP | 0.1840 | **0.4954** |
| Decision Tree | 0.1538 | 0.3974 |
| XGBoost | 0.1273 | 0.2990 |
| LightGBM | 0.1138 | 0.3323 |
| CatBoost | 0.0840 | 0.4259 |

Two post-hoc methods applied to the same model agree on roughly a third of the top-5
features. LIME's own surrogate fidelity is below 0.4 everywhere and below 0.15 for
most models. Neither method announces which of the two is misleading its reader. I
regard this as the most practically consequential result in the project, and it is
independent of whether ECGB works.

---

## Repository layout

```
paysim_ecgb_pipeline.ipynb   The complete pipeline, with outputs preserved
requirements.txt             Pinned versions matching the recorded run
README.md                    This file
```

Running the notebook creates a working directory (`Dissertation_ECGB/` by default)
containing `data/`, `checkpoints/`, `models/`, `logs/`, `figures/`,
`optuna_studies/` and `deployment/`. None of it is committed — see `.gitignore`.

## Notebook sections

| Section | Content |
|---|---|
| 1 | Environment, reproducibility, shared helpers |
| 2 | PaySim and the exploratory analysis that motivated the preprocessing |
| 3 | Feature engineering, partitioning, SMOTE |
| 4 | XGBoost, LightGBM, CatBoost, Decision Tree, MLP |
| 5 | ECGB: core, search, ablation, cross-validation, final model, mechanism test |
| 6 | Explanation consistency - the primary research question |
| 7 | DASH, the post-hoc aggregation control |
| 8 | LIME fidelity, and its agreement with SHAP |
| 9 | Feature importance across all models |
| 10 | Leaderboard and significance testing |
| 11 | The deployed four-tier auditor application |
| 12 | Closing inventory of everything the pipeline produced |

---

## Running it

### Data

PaySim is not redistributed here. Download `PS_20174392719_1491204439457_log.csv`
from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), rename it to
`paysim.csv`, and either place it in the `data/` directory the notebook creates or
point the `PAYSIM_CSV` environment variable at it.

### On Colab

Open the notebook, put `paysim.csv` in your Drive root, and run from the top. A GPU
runtime is assumed the boosting sections are configured for CUDA and the full
pipeline takes several hours on a T4.

### Locally

```bash
pip install -r requirements.txt
export ECGB_BASE_DIR=./Dissertation_ECGB
export PAYSIM_CSV=/path/to/paysim.csv
jupyter lab paysim_ecgb_pipeline.ipynb
```

The notebook detects that it is not on Colab and uses local paths. Without a CUDA
GPU you will need to change `device: 'cuda'` to `device: 'cpu'` in the XGBoost and
ECGB parameter dictionaries and `task_type` to `'CPU'` for CatBoost.

### Interruptions

I built the pipeline around checkpoints because Colab sessions kept dropping during
the longer searches. Optuna studies resume from their SQLite files and never repeat a
completed trial; every cell reloads what it needs from disk. After a disconnection,
re-run Section 1 and then the resume report, which prints exactly what already exists
and therefore which section to continue from.

---

## The deployed application

Section 11 generates a self-contained four-tier application under
`deployment/explainable_auditor/`:

- **Presentation**- a Dash interface, styled as a case file, where an investigator
  submits a transaction and sees the prediction with SHAP and LIME panels
- **Application**- a Flask REST API (`/predict`, `/explain`, `/health`,
  `/audit/export/<id>`)
- **Inference**- the saved ECGB booster with a SHAP `TreeExplainer` and a LIME local
  surrogate, returning the surrogate's fidelity R² alongside the explanation
- **Persistence**- a MongoDB audit store recording model id, inputs, prediction,
  explanation, timestamp and user, which is what lets the system satisfy the EU AI
  Act transparency duty and the GDPR Article 22 right to a meaningful explanation

The model is wrapped exactly as evaluated in Section 5 - nothing is retrained or
re-thresholded on the way into the container, so the deployed system behaves the way
the measured one does.

```bash
cd Dissertation_ECGB/deployment/explainable_auditor
docker compose up --build
# investigator interface at http://localhost:8050
```

The notebook verifies all four tiers in-process first, using `mongomock` in place of a
live MongoDB. The only difference between that and production is the `MONGO_URI`
environment variable.

---

## Two things I got wrong, and what they cost

I record these because both changed my results, and because the second changed a
conclusion I had already drawn.

**Early stopping ran backwards.** My first ECGB implementation trained zero to three
trees regardless of configuration. `xgb.train()` was using my custom PR-AUC function
for early stopping and I had not passed `maximize=True`; XGBoost minimises a custom
metric unless told otherwise, so it was comparing PR-AUC in the wrong direction and
abandoning training almost immediately. One keyword argument separated a model that
trained three trees from one that trained properly. Section 5.2 now checks the
distribution of `best_iteration` explicitly so the failure cannot recur silently.

**Ranking by PR-AUC alone selected a degenerate model.** My selection step originally
sorted candidates by raw PR-AUC, and on one run chose a trial scoring 0.997 PR-AUC at
0.028 precision - a model flagging nearly every transaction as fraud. PR-AUC is
threshold-independent and does not penalise that at the threshold I actually deploy
at. Selection now requires a minimum precision before ranking survivors by F1, and
prints a warning whenever the highest-PR-AUC trial is not the one chosen.

Section 9.2 is the one cell without saved output: its last run raised on a checkpoint
key I had misremembered. The lookup is fixed in the committed source and re-running
regenerates the figure. I left it unexecuted rather than pasting in output from a
different run.

---

## Reproducibility

A single seed (42) is threaded through every split, resampler, sampler and model.
Every model goes through the identical search / cross-validate / finalise code path,
so differences in the results are attributable to the models rather than to
inconsistencies in how I evaluated them. Library versions from the recorded run are
printed at the end of Section 1 and pinned in `requirements.txt`.

Two caveats on exact reproduction. Optuna's TPE sampler is seeded but its trial order
depends on how many trials were already in the study, so resuming an interrupted search
can produce a different sequence from a clean run. And GPU floating-point reductions
are not bit-deterministic, so the metrics reproduce to roughly four decimal places
rather than exactly.

---

## Citation and licence

If you use this code, please cite the dissertation:

```bibtex
@mastersthesis{explainable_auditor,
  title  = {The Explainable Auditor: Evaluating Post-Hoc Interpretability for
            Machine Learning-Based Financial Transaction Fraud Detection},
  author = {<YOUR NAME>},
  school = {<YOUR INSTITUTION>},
  year   = {<YEAR>}
}
```

The code is released under the MIT licence (see `LICENSE`; add your name to it before
publishing). PaySim is the work of Lopez-Rojas, Elmir and Axelsson and is subject to
its own terms on Kaggle.

## Acknowledgements

PaySim was introduced in Lopez-Rojas, E. A., Elmir, A. and Axelsson, S. (2016),
*PaySim: A financial mobile money simulator for fraud detection*, 28th European
Modeling and Simulation Symposium. The two engineered balance-error features follow
their observation that reconciliation failures are a strong fraud signal.
