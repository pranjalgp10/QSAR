# Experiment 3: stacking architecture optimization

## Question

Which predeclared combinations of Random Forest, Extra Trees, SVR, XGBoost and HistGradientBoosting work best with OLS, Ridge or ElasticNet meta-regression for scaffold-held-out Syk pIC50 prediction?

## Design

The curated structures use Morgan fingerprints (radius 2, 2048 bits). A single shuffled five-fold `GroupKFold` allocation groups Bemis–Murcko scaffolds. Within each outer training fold, three scaffold-held-out meta folds generate base predictions. For every meta fold and each base learner, hyperparameter search is run only on the other meta folds using three-fold grouped CV; therefore the meta-validation labels do not select that fold's base hyperparameters. Base learners are separately tuned on the full outer-training fold before predicting the outer fold. All stack candidates use the same folds and predictions.

The five declared base combinations and eight declared meta-model settings are in `config.json`. The SVR-containing candidate is motivated by lower residual correlation observed in Experiment 1, while its weaker standalone error makes it an exploratory complementarity comparison, not a presumed winner. Meta-model settings are fixed in advance; outer-fold outcomes do not tune their hyperparameters.

## Interpretation and limits

The nested OOF design controls base-model leakage into meta fitting. However, comparing many candidates on the same outer folds and reporting the lowest mean error creates selection optimism. Treat those results as architecture-screening evidence, not an unbiased estimate for the selected architecture. This run uses one outer allocation and two randomized base-search trials per fit to keep nested computation bounded. The dataset has no independent external validation cohort, and prior dataset evaluations have been inspected. Do not claim external generalization or a virgin test result.

## Run and resume

From the repository root:

```powershell
.venv\Scripts\python.exe src\qsar\experiment_03.py
```

The script creates a unique run folder under `results/qsar/experiment_03_stacking_optimisation/`, checkpointing completed folds. Resume with:

```powershell
.venv\Scripts\python.exe src\qsar\experiment_03.py --resume results\qsar\experiment_03_stacking_optimisation\<run-id>
```

The run saves configuration, fold assignments, OOF meta-features, outer predictions, fold metrics, meta coefficients and reports. It does not overwrite earlier experiments or the preserved reference benchmark.
