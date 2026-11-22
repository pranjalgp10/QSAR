# Final QSAR preflight data audit

Run: `final_qsar_20261122T143000`

Raw table: 3,176 rows; SHA-256 `52dad48b3308134a553c8db38b4fc0c188eff0ccdf51e7477e8f6252a3679532`.
Curated table: 3,051 structures; SHA-256 `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25` (expected hash matches: True).

## Audit findings

- Missing values: raw 0; curated 0.
- Invalid SMILES: raw 0; curated 0.
- Raw literal duplicate excess rows: 0; canonical duplicate groups resolved in curation: 117; curated duplicate canonical structures: 0.
- Duplicate InChIKey groups: raw 121; curated 4.
- Disconnected structures retained unchanged: 6. Their fragments and source-row mappings are in `data_audit.json`.
- Structures with unassigned potential chiral centers: 364. No source stereo metadata are available to adjudicate them.
- Internal formula check maximum absolute error: raw 1.78e-15; curated 0. This verifies arithmetic only, not the unit.
- Tukey 1.5-IQR outlier flags: 0; none were excluded.
- Discordant duplicate groups: 20; groups resolved by the potency-favoring minimum-IC50 branch: 3.

## Blocking provenance issue

The available CSVs contain only SMILES, IC50 and pIC50. The conversion is internally consistent with nM, but no per-record unit, assay ID, source, censoring, or replicate metadata are present. The prior processing record also says the raw source file is unavailable and that the supplied input had already undergone IQR filtering. Therefore units, upstream exclusions and conflicting measurements cannot be independently verified.

The attached protocol requires the pipeline to fail when IC50 units are unknown and says to stop on critical methodological issues. **No regression/classification training, threshold calculation, calibration fitting, Y-scrambling, or deployment model was run.** Training on these labels could produce numerically plausible but chemically mis-scaled predictions.

The six disconnected structures were inspected and retained in the audit; none was silently modified or removed. Their mixture/salt interpretation needs source review. Duplicate handling is documented but remains assay-unverified.

## Prior experiments inspected

Experiment 1 outputs exist and were read: two repeats of five-fold scaffold CV; stacking mean RMSE 0.6483 ± 0.0365, versus XGBoost 0.6606 ± 0.0347 and HistGradientBoosting 0.6629 ± 0.0435. Stacking beat HistGradientBoosting in 10/10 folds and XGBoost in 9/10. This supports stacking as a candidate, but differences are modest and all results are internal.

Experiment 3's screened lowest mean-RMSE configuration was the SVR-containing four-model Ridge(100) stack (mean RMSE 0.6427 ± 0.0452). Paired scaffold-bootstrap comparisons with the boosting pair and all-five Ridge stack included zero; therefore the nominal winner is not clearly superior. SVR had less correlated residuals but weaker standalone RMSE.

Experiment 4's five-fold pooled stack RMSE was 0.6767, MAE 0.5053, R² 0.6241. Disagreement, AD distance, and combined-ranking Spearman correlations with absolute error were 0.185, 0.225, and 0.251. The lowest-uncertainty 10% had RMSE 0.454 (n=306). Molecule-level 90% interval coverage was 88.8%, mean width 2.135 pIC50. These support uncertainty filtering as a diagnostic, not calibrated confidence on generated molecules.

No VECTOR+ implementation or source label-generation code was found in this repository; median splitting can be implemented from the attached specification after data provenance is resolved, but source-level parity cannot be verified here.

## Required next input

Provide the original source file or a source-linked data dictionary establishing IC50 units and assay provenance, plus the upstream raw data needed to audit IQR filtering. Resolve how mixed salts/disconnected structures and discordant duplicate measurements should be treated from source metadata. After those checks, the final model evaluation can proceed with a fail-closed data hash and preserved provenance.
