# Final QSAR preflight data audit

Run: `final_qsar_20261122T091131Z`

Raw table: 3,176 rows; SHA-256 `52dad48b3308134a553c8db38b4fc0c188eff0ccdf51e7477e8f6252a3679532`.
Curated table: 3,051 structures; SHA-256 `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25` (expected hash matches: True).

## Audit findings

- Missing values: raw 0; curated 0.
- Invalid SMILES: raw 0; curated 0.
- Raw literal duplicate excess rows: 0; canonical duplicate groups resolved in curation: 117; curated duplicate canonical structures: 0.
- Duplicate InChIKey groups: raw 121; curated 4. The four curated pairs are different tautomeric SMILES with identical IC50 values; retained separately to preserve the expected hash, with minor duplicate weighting documented.
- Disconnected structures retained unchanged: 6. Their fragments and source-row mappings are in `data_audit.json`.
- Structures with unassigned potential chiral centers: 364. No source stereo metadata are available to adjudicate them.
- Internal formula check maximum absolute error: raw 1.78e-15; curated 0.
- Tukey 1.5-IQR outlier flags: 0; none were excluded.
- Discordant duplicate groups: 20; groups resolved by the potency-favoring minimum-IC50 branch: 3.

## Unit provenance and remaining data limitations

The previously shared Data_processing.ipynb is available at `C:\Users\pranj\Downloads\Data_processing.ipynb`. Its saved Standard Units output records 3485 nM and 28 pIC50 rows in a 3,513-row input; its code converts pIC50 values to nM before recalculating pIC50. The code comment says μM, but the 10^9 factor assumes the conventional molar pIC50 scale. ChEMBL documents pChEMBL as -log10(molar concentration), where 1 nM is 9 ([ChEMBL activity FAQ](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)). The mismatch is likely a comment error, but the missing raw_data.csv prevents row-level verification of the 28 source pIC50 values. It records a 3,176-row final IQR-filtered output. `all_mols.csv` and `processed_df.csv` are byte-identical 3,176-row tables, and their pIC50 arithmetic matches nM conversion exactly.

The six disconnected structures were inspected and retained unchanged. Four InChIKey-equivalent tautomer pairs have identical IC50 labels and remain separate to preserve the stipulated dataset hash; scaffold grouping prevents their split across evaluation folds. Assay IDs, source/censoring metadata and replicate identities remain unavailable, and upstream IQR exclusions cannot be reconstructed. These are publication limitations, not grounds for silently dropping rows.

## Prior experiments inspected

Experiment 1 outputs exist and were read: two repeats of five-fold scaffold CV; stacking mean RMSE 0.6483 ± 0.0365, versus XGBoost 0.6606 ± 0.0347 and HistGradientBoosting 0.6629 ± 0.0435. Stacking beat HistGradientBoosting in 10/10 folds and XGBoost in 9/10. This supports stacking as a candidate, but differences are modest and all results are internal.

Experiment 3's screened lowest mean-RMSE configuration was the SVR-containing four-model Ridge(100) stack (mean RMSE 0.6427 ± 0.0452). Paired scaffold-bootstrap comparisons with the boosting pair and all-five Ridge stack included zero; therefore the nominal winner is not clearly superior. SVR had less correlated residuals but weaker standalone RMSE.

Experiment 4's five-fold pooled stack RMSE was 0.6767, MAE 0.5053, R² 0.6241. Disagreement, AD distance, and combined-ranking Spearman correlations with absolute error were 0.185, 0.225, and 0.251. The lowest-uncertainty 10% had RMSE 0.454 (n=306). Molecule-level 90% interval coverage was 88.8%, mean width 2.135 pIC50. These support uncertainty filtering as a diagnostic, not calibrated confidence on generated molecules.

No VECTOR+ implementation or source label-generation code was found in the repository or inspected user references. This run follows the user's explicit median-split definition and will label it VECTOR+-compatible, not an exact source-code reproduction.

## Audit disposition

Critical preflight blockers: 0. Internal scaffold evaluation is allowed against the expected hash-pinned data with the above limitations. Publication-grade external validity is not established.
