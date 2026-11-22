# QSAR repository and data audit

## Snapshot inspected

- The project contains `AGENTS.md`, `Scientific_Context.md`, `processed_df.csv`, a project `.venv`, `src/qsar/pipeline.py`, and generated QSAR artifacts under `results/qsar_publication/`.
- No README, configuration files, tests, separate raw/external dataset, package lockfile, assay metadata, molecule identifiers, or VECTOR+ source implementation were present in this inspected snapshot.
- Installed environment observed: Python 3.14.8, pandas 3.1.1, NumPy 2.5.5, scikit-learn 1.9.1, RDKit 2026.09.2, XGBoost 3.4.1.

## Data audit

- `processed_df.csv`: 3,176 rows; columns are `Smiles`, `IC50`, `pIC50`.
- No missing values, invalid SMILES, or repeated literal SMILES were detected.
- Canonical isomeric SMILES produces 3,051 distinct structures: 117 duplicated-structure groups and 125 excess rows.
- All pIC50 values numerically equal `9 - log10(IC50)`, consistent with IC50 in nM. This is an internal arithmetic check, not independent confirmation of source units.
- The CSV has no source/assay identifiers, censoring flags, measurement error, or record IDs. Therefore assay consistency, censored measurements, and scientifically appropriate replicate aggregation cannot be established from this file alone.
- Six SMILES contain disconnected components. They require inspection before any salt/fragment handling; do not strip fragments automatically.

## Existing model and evaluation history

The prior implementation used Morgan radius-2, 2,048-bit fingerprints; RF, SVR, XGBoost, HistGradientBoosting and Ridge stacking; scaffold-grouped holdout partitions; and a small randomized search. Reported stacking test metrics were MAE 0.496, RMSE 0.674, MSE 0.455, R² 0.633. The prior outputs and split have already been examined in this project. They are historical internal-holdout results and cannot be represented as external validation or as a newly untouched test. The Syk article's random-split result is not directly comparable to scaffold holdout.

Prior code has a methodological issue: it tunes base models only within the training partition but uses the validation partition to choose among models, then reports every candidate on the test set. That is a legitimate final comparison only if specified before opening test outcomes; here outcomes have already been viewed. The prior reported ensemble-disagreement calibration uses validation errors and has only five bins, with non-monotone errors. Agreement is therefore not demonstrated to be calibrated confidence.

## Consequences and plan

1. Preserve the source CSV. Retain this audit as historical record.
2. Remove obsolete serialized QSAR models, predictions and reports after verifying their exact paths.
3. Build auditable canonicalization/duplicate review before new modeling. Since replicate source metadata are absent, do not silently average/choose activities; produce a duplicate table for scientific review and keep duplicate structures in one partition meanwhile.
4. Use scaffold-aware partitions and nested/grouped development validation. State clearly that this dataset has no virgin external test because earlier outcomes were examined; external validation requires independently sourced measurements.
5. Compare RF, SVR, XGBoost, HistGradientBoosting, and stacking. Stacking must use grouped out-of-fold predictions and be retained only if the gain is stable under resampling.
6. Treat ensemble spread as a candidate uncertainty feature, not calibrated confidence by itself. Calibrate and assess it against held-out errors, include chemical similarity/applicability domain, and report reliability versus coverage. Do not set a pseudolabel cutoff or binary activity threshold without scientific approval.

## Open evidence gaps

- Confirm dataset source/version and IC50 units from an authoritative source record.
- Obtain assay and replicate metadata to resolve duplicate-structure observations.
- Obtain an independent external Syk dataset for a publication-grade external generalization claim.
- Establish threshold and reliability/coverage targets with project scientific rationale before pseudolabel export.
