# Methods

## Evaluation protocol

Use the expected-hash curated table (3,051 rows) and Morgan radius-2, 2,048-bit fingerprints. Reproduce Experiment 1's repeated shuffled five-fold scaffold-grouped outer CV (seeds 140 and 141); Bemis–Murcko scaffolds remain intact, with the repository's molecule-specific fallback for acyclic structures. Every model and stack variant shares identical outer folds. All reported scores are internal scaffold validation because prior results on these molecules were inspected.

Within each outer fold, reserve a scaffold-disjoint calibration subset from the outer-training data. Fit models only on the remaining fit subset. For the median activity task, compute the median threshold from the fit subset's observed pIC50 labels and use that unchanged threshold for calibration and outer-test labels. Active means pIC50 >= threshold; inactive means below threshold. Report ties and per-partition prevalence. The final deployment threshold is the median of all approved development labels after model selection.

## Regression candidates and stack construction

The primary predeclared candidate is the five-model Ridge stack supported by Experiment 1. Compare RF, ExtraTrees, SVR, XGBoost, HistGradientBoosting, and the three stack configurations: five-model Ridge(alpha=1), SVR-containing RF/XGBoost/HistGradientBoosting/SVR Ridge(alpha=100), and XGBoost/HistGradientBoosting Ridge(alpha=1). Base hyperparameters are frozen from the preserved benchmark report. For each outer fold, generate all meta-training columns from scaffold-grouped OOF base predictions; fit meta-models only to those cross-fitted columns. Outer test structures never contribute to base fitting, meta fitting, model selection, thresholds, or calibration.

Report MAE, RMSE, R², median absolute error, Pearson/Spearman correlation, fold variability, pooled OOF metrics, and scaffold-cluster bootstrap intervals. Include median/mean and majority-class baselines.

## Binary classification

The primary binary result thresholds the regression model's continuous pIC50 predictions with the fit-partition median. Report confusion matrix, accuracy, balanced accuracy, precision, sensitivity, specificity, F1, MCC, ROC-AUC and PR-AUC; use continuous scores for AUC. Evaluate RF, XGBoost and regularized logistic regression as direct classifier benchmarks using labels from the same fit-derived median. Calibrate direct-classifier probabilities only on the separate calibration partition, and report Brier score, calibration curve and calibration error. Do not call regression scores probabilities.

Evaluate thresholds at the primary median, ±0.25 pIC50 around it, and fit-set 40th/60th percentiles as sensitivity analyses. Do not select a threshold from outer-test results. Report performance by distance to the primary median.

## Calibration, applicability and filtering

Evaluate ensemble disagreement, maximum Morgan/Tanimoto to fit structures, applicability distance, and a combined percentile rank referenced to the calibration subset. Calibrate nominal 80/90/95% absolute-residual intervals using calibration labels; include scaffold-maximum conformal scores as a conservative comparison. Report empirical molecule and whole-scaffold coverage, width and scaffold-bootstrap intervals.

Compare no filtering, AD-only, uncertainty-only and combined rankings at predeclared retained fractions from 10% to 100%. Report retained count, regression error, active/inactive precision, false-active/false-inactive rates and retained scaffold diversity. These curves are descriptive; no pseudolabel cutoff is chosen without an agreed target precision and minimum coverage.

## Robustness and analysis

Run a reduced Y-scrambling control with fixed model and scaffold splits; the permuted-label distribution should be near the null baseline. Build scaffold-aware learning curves; compare Morgan radius 3 and Morgan radius 2 plus selected descriptors as representation sensitivities. Analyze residuals by activity range, scaffold size, nearest-neighbour similarity and high-error scaffold; save the worst-error examples. Retain the primary Morgan-2 representation regardless of secondary results.

## Final fitting and inference

After evaluation and selection, fit the primary model on all approved development data and set the full-data median threshold. Save cross-fitted calibration summaries with an explicit limitation if they do not provide exact finite-sample guarantees for the full-data refit. Inference accepts SMILES and returns canonical SMILES, pIC50, IC50 in nM, binary median-split label, nearest training similarity, disagreement, interval (if supported), reliability status and rejection reasons. Since no validated operating threshold is established, valid molecules must not be tagged accepted for pseudolabelling.
