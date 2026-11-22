# Methods — planned protocol, not executed

The requested final QSAR pipeline was not run because the data-integrity gate failed. This document records the intended analysis so it can be resumed without silently changing the plan.

## Data and split

Use the exact hash-verified curated Syk table only after units, source provenance, upstream filtering, disconnected structures, and InChIKey-equivalent rows have an approved disposition. Use Bemis–Murcko grouping with the existing acyclic fallback. Reproduce Experiment 1's two five-fold shuffled scaffold-grouped CV allocations (seeds 140 and 141) if all model-fitting and calibration steps can be made fold-local. Report each fold and pooled OOF metrics; use scaffold-cluster bootstrap intervals.

## Regression

Compare Random Forest, ExtraTrees, SVR, XGBoost, HistGradientBoosting, and the provisional five-model Ridge stack. Include the Experiment 3 SVR-complementary Ridge(100), boosting-pair Ridge, and all-five Ridge candidates as predeclared stack comparisons. Morgan/ECFP radius 2, 2,048 bits remains primary. All base tuning and stacking meta-features must be nested within outer-training scaffolds; outer labels must not influence tuning, architecture choice, or calibration.

## Binary activity task

For each fold, compute the threshold as the median observed pIC50 of that fold's model-training partition only. Assign active to pIC50 greater than or equal to that threshold; apply the same threshold to calibration and outer-test rows. Report ties and class counts. Evaluate regression-derived binary scores using the continuous prediction, and direct RF, XGBoost, and regularized logistic classifiers as secondary baselines. Calibration of probabilities must use training-only cross-fitting or a reserved scaffold calibration subset.

## Reliability and uncertainty

Retain the Experiment 4 predictors: ensemble disagreement, maximum Morgan Tanimoto to model-fit structures, applicability distance, and a calibration-referenced combined rank. Evaluate regression and classification error, active/inactive false labels, and precision–coverage. Calibrate intervals only on a scaffold-separated calibration partition. Compare no filtering, AD-only, uncertainty-only, and combined filtering as curves; do not choose an operating cutoff until a precision target and minimum coverage are declared before viewing the relevant held-out results.

## Additional controls

Use training-derived median labels for classification. Add mean/median regression and majority-class baselines, Y-scrambling, scaffold-aware learning curves, residual/error analysis, threshold sensitivity, and limited predeclared representation sensitivity analyses. Preserve the primary Morgan radius-2 representation. Fit a final model only after internal evaluation and selection are complete and the dataset integrity gate passes.

None of the above model evaluation steps has been run in this run directory.
