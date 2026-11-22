# Experiment 4: QSAR reliability and applicability domain

## Question

Do model disagreement and chemical similarity to the model-fit data identify scaffold-held-out molecules with larger QSAR errors, and how do scaffold-aware calibration methods perform for prediction intervals?

## Design

Use the 3,051 curated structures, Morgan radius 2 / 2,048-bit fingerprints, and a single shuffled five-fold scaffold-grouped outer evaluation. In each outer fold, set aside a scaffold-grouped calibration subset from outer-training data. Fit the frozen reference stacking model on the remaining structures; use the calibration subset only to estimate interval widths. Evaluate all predictions and intervals on outer-test scaffolds. The model family and hyperparameters are frozen from the preserved benchmark report; this experiment does not tune models or use historical test results.

Uncertainty signals are (1) standard deviation across the five base-model predictions, (2) one minus maximum Morgan/Tanimoto similarity to model-fit structures, and (3) the average percentile rank of those two signals. These are diagnostics, not probabilities. The primary interval is nominal 90% split conformal using calibration absolute residuals. A second scaffold-level variant uses the largest calibration residual within each scaffold as the calibration score; this is more conservative and targets whole-scaffold containment. Nominal 80% and 95% intervals are secondary.

Precision–coverage curves retain molecules in order of lower uncertainty and report MAE, RMSE, 90th-percentile absolute error, and descriptive fractions within 0.5 and 1.0 pIC50. These error tolerances describe regression accuracy; they are not pIC50 activity thresholds and do not select a pseudolabel cutoff.

## Interpretation limits

Only one five-fold allocation is used. Calibration and evaluation are scaffold-separated within each fold, but the dataset is small and lacks assay/source metadata. Previous results on this dataset have been inspected, so this is internal cross-validation rather than independent external validation. Do not treat disagreement as calibrated probability or claim that conformal coverage transfers to generated molecules or distribution-shifted scaffolds without further validation. No pseudolabel acceptance threshold is selected.

## Run and resume

From the repository root:

```powershell
.venv\Scripts\python.exe src\qsar\experiment_04_reliability.py
```

Outputs are saved in a unique run folder below `results/qsar/experiment_04_reliability/`. Resume an interrupted run with `--resume results/qsar/experiment_04_reliability/<run-id>`.
