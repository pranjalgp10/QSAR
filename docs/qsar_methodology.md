# QSAR benchmark methodology (implementation protocol)

## Purpose and reference

The model predicts continuous experimental pIC50 as an enabling component for later QSAR-guided VECTOR+ augmentation. The Syk study used ECFP features and evaluated Random Forest, SVR, XGBoost and histogram gradient boosting before a stacking ensemble. This implementation uses the same model families as benchmark candidates, while replacing random-row validation with structure/scaffold grouped development splits. The paper's random-split R² must not be treated as a directly comparable target.

## Data and curation gate

Input is `processed_df.csv`, with `Smiles`, `IC50`, `pIC50`. RDKit parses each input and generates canonical isomeric SMILES and a Bemis–Murcko scaffold key. Empty scaffolds receive molecule-specific canonical keys so acyclic compounds are not all grouped together. The script checks the numerical nM conversion but this does not independently authenticate source units. It writes duplicate detail and summary files. Training stops when canonical duplicate structures exist unless an explicit reviewed resolution file is supplied. This avoids silent duplicate weighting or invented assay aggregation. The resolution CSV requires `canonical_smiles,resolved_pIC50,rationale`; its scientific rationale must be based on source/assay review.

## Features and models

Morgan/ECFP radius 2, 2,048 bits, RDKit. Candidates: Random Forest, Extra Trees as an additional tree baseline, SVR with scaling, XGBoost, HistGradientBoosting, and Ridge stacking. Base learner hyperparameters are searched with grouped CV restricted to the training partition. Stack meta-features are generated through grouped out-of-fold predictions. Validation is subdivided into model-selection and calibration folds; the test partition does not set model parameters or cutoffs.

## Split and metrics

Deterministic group holdout aims for 64/16/20 train/validation/test; actual counts and scaffold disjointness are recorded. Canonical duplicates must be resolved before this step. Regression metrics include MAE, MSE, RMSE, and R². Test intervals include scaffold-level bootstrap percentile intervals. Activity thresholding is intentionally not implemented because a scientific cutoff has not been approved.

## Confidence and applicability domain

Inter-model disagreement is the sample standard deviation of predictions from the independently tuned RF, Extra Trees, SVR, XGBoost and HistGradientBoosting models. It is reported as a diagnostic, not a probability of correctness. Validation calibration tables relate disagreement to absolute residuals. Applicability domain is separately summarized by maximum Morgan/RDK fingerprint Tanimoto similarity to training molecules. A conformal-style absolute-residual interval is calibrated on a validation calibration subset and its coverage reported on the internal holdout. No acceptance cutoff is selected. Reliability and coverage must be jointly considered before pseudolabel use.

## Interpretation limits

Earlier experiments on this CSV were evaluated and discussed. A newly generated split is not virgin external validation. Results must be labeled internal scaffold holdout. Strong publication claims require independent Syk measurements and assay provenance. The current CSV lacks assay IDs, source IDs, censoring information and replicate metadata. No binary pIC50 threshold or pseudolabel acceptance criterion is authorized by this implementation.
