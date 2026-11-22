# Model card: Syk pIC50 QSAR development model

## Model

Five-model regression stack using Random Forest, ExtraTrees, SVR, XGBoost and HistGradientBoosting predictions as inputs to Ridge regression (alpha 1). Each base model uses Morgan radius-2, 2,048-bit fingerprints. Architecture was predeclared from prior internal scaffold experiments.

## Intended use

Research development and cautious ranking of Syk-like structures for the VECTOR+ pseudolabelling study. Predictions remain computational estimates. The inference interface reports relative median-split activity, estimated IC50 in nM, 90% internal residual interval, nearest-training Tanimoto similarity, ensemble disagreement and reliability status.

## Activity label

The deployment threshold is the median pIC50 of the 3,051-row hash-pinned development table. `active` is pIC50 greater than or equal to the threshold; ties are active. This is a relative dataset-specific label, not a universal pharmacological potency cutoff.

## Training and evaluation data

Curated dataset SHA-256: `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25`. Evaluation uses repeated, scaffold-grouped five-fold CV with a separate scaffold calibration subset inside each training fold. This is internal validation; prior experiments evaluated the same data.

## Reliability and limitations

Valid predictions return `caution`, because no pseudolabel precision/coverage operating threshold has been validated prospectively. Invalid SMILES return `reject`. The 90% interval is based on internal scaffold-separated residual calibration and has no guarantee on generated molecules. No external assay-matched validation or generated-molecule validation is available. The raw source, assay IDs/conditions, replicate/censoring details, and exact VECTOR+ implementation are not present. See `limitations.md` and `final_report.md`.

## Prohibited interpretations

- Do not describe a predicted active label as experimentally confirmed activity.
- Do not use these probabilities as calibrated regression-derived probabilities.
- Do not claim external validation or experimentally validated potency.
- Do not accept pseudolabels solely from this model until an operating criterion and generated-molecule validation are established.
