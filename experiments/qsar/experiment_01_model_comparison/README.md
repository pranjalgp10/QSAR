# Experiment 1: repeated scaffold model comparison

## Question

Does stacking consistently improve continuous pIC50 prediction over the individual RF, Extra Trees, SVR, XGBoost and HistGradientBoosting models on unseen Bemis–Murcko scaffolds?

## Method

Two repeats of five-fold scaffold-grouped outer CV are shared by every candidate. Each individual learner is tuned only within its outer-training fold using grouped inner CV and RMSE. The reference stacking architecture uses fixed, predeclared base parameters and grouped out-of-fold meta-features; no outer-test fold contributes to meta-model fitting. Experiment 3 is reserved for architecture optimization. Fold-level metrics and scaffold-cluster bootstrap paired differences are reported.

## Run

From repository root: `.venv/Scripts/python.exe src/qsar/experiment_01.py`. Each run gets a unique UTC run folder beneath `results/qsar/experiment_01_model_comparison/`. Intermediate predictions and manifests are checkpointed after each fold.

## Limits

The curated CSV contains 3,051 unique structures, but assay/source metadata are unavailable and activity curation followed the Syk paper's potency-favoring duplicate rule. Previous evaluations on this dataset were inspected, so these results are internal scaffold validation, not independent external validation. The two-repeat design is reduced from the preferred three repeats because of fit cost; this limitation will be retained in the report.
