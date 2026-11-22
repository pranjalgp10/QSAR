# Final QSAR development — preflight outcome

**Status: blocked before model fitting.** The attached protocol requires verified IC50 units and a fail-closed response when units are unknown. The supplied tables contain no unit, assay, or source metadata. Their pIC50 values exactly match the arithmetic for IC50 in nM, but that only confirms internal consistency; it does not verify the source unit. No regression or classification model, median threshold, calibration object, or deployment artifact was created.

## Completed

- Inspected the completed Experiment 1, 3 and 4 outputs and verified the expected curated-dataset SHA-256: `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25`.
- Audited the raw 3,176-row table and 3,051-row curated table. There are no missing fields or invalid SMILES. Formula arithmetic matches `pIC50 = 9 - log10(IC50[nM])` to floating-point precision.
- Identified six disconnected structures and retained them unchanged in the audit. Four InChIKey-equivalent tautomer pairs also remain as separate rows in the expected-hash dataset; each pair has identical IC50 in the supplied table.
- Recorded 117 canonical-structure duplicate groups and the prior potency-favoring duplicate-resolution method. The source assay records are unavailable, so discordant measurements cannot be independently adjudicated.

## Prior results and provisional architecture assessment

Experiment 1's repeated scaffold CV supports the five-model Ridge stack: mean RMSE 0.6483 ± 0.0365 over ten outer folds, compared with 0.6606 ± 0.0347 for XGBoost and 0.6629 ± 0.0435 for HistGradientBoosting. Stacking had lower fold RMSE in 9/10 folds against XGBoost and 10/10 against HistGradientBoosting.

Experiment 3's lowest-scoring candidate was the SVR-containing four-model Ridge(100) stack (mean RMSE 0.6427 ± 0.0452). Its advantage over the boosting pair and all-five stack was negligible, with paired scaffold-bootstrap intervals including zero. SVR has less correlated residuals but substantially weaker standalone performance.

Experiment 4 found pooled stacking RMSE 0.6767, MAE 0.5053, R² 0.6241. Disagreement and applicability-domain signals had only modest association with error. A calibration-referenced combined rank improved error in its lowest-uncertainty 10%, but this is internal evidence and does not validate thresholds on generated molecules.

These results make the five-model Ridge stack a provisional candidate because it performed consistently in Experiment 1 and has reliability diagnostics in Experiment 4. The SVR-containing architecture remains an ablation candidate; Experiment 3 does not establish it as superior. No final architecture is frozen here.

## Why work stopped

The missing unit/source provenance is a critical label-integrity issue. A model trained now could return numerically plausible pIC50 estimates on incorrectly scaled targets. In addition, the unavailable upstream raw source prevents auditing the prior IQR filtering, the four InChIKey-equivalent rows require a deliberate standardization/weighting decision, and the exact VECTOR+ code is absent. These issues were not resolved by silently changing structures, dropping records, or assuming units.

## To resume

Provide source-linked evidence establishing the IC50 unit and assay context for this dataset, together with the upstream raw observations and processing record needed to audit IQR filtering. Confirm whether tautomer-equivalent rows should be merged or retained under a documented policy. Then resume with an updated, versioned dataset and hash. Only after those gates pass should the regression/classification protocol, calibration, robustness controls, model package, and inference tests run.

This is a preflight report, not a completed QSAR evaluation. No external validation or generated-molecule performance is claimed.
