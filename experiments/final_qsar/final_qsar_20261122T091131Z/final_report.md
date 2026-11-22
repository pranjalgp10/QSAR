# Final QSAR development report

Run: `final_qsar_20261122T091131Z`. All reported evaluation is internal scaffold validation on the curated dataset; there is no virgin external validation or generated-molecule validation.

## Executive result

The predeclared primary model is the five-model Ridge stack (Random Forest, ExtraTrees, SVR, XGBoost, HistGradientBoosting; Ridge alpha 1). It was frozen from the preceding experiments before this run's outer-fold predictions were inspected. The run compares it with individual learners, the SVR-complementary stack, and the boosting-pair stack; these outer-fold comparisons are descriptive and were not used to change the architecture.

- Continuous pIC50: MAE 0.499, RMSE 0.669 (scaffold-bootstrap 95% CI 0.639–0.699), R² 0.633.
- Median-derived classification: accuracy 0.847, balanced accuracy 0.847, active precision 0.846, inactive precision 0.849, MCC 0.695, ROC-AUC 0.914, PR-AUC 0.910.
- Full-development median threshold: 7.37675 pIC50; active means greater than or equal to this relative dataset median. Counts: 1529 active and 1522 inactive; ties at median: 5.
- A 90% molecule interval had empirical coverage 0.910, mean width 2.284 pIC50. This is internal calibration/validation evidence and does not establish coverage on generated chemistry.
- No pseudolabel-acceptance cutoff is established. Inference returns `caution` for every valid molecule; invalid structures are rejected.

## Architecture evidence from Experiments 1, 3 and 4

Experiment 1's repeated scaffold evaluation favored the five-model stack on stability (RMSE 0.6483 ± 0.0365 across ten folds; better than XGBoost in 9/10 and HistGradientBoosting in 10/10 folds). Experiment 3's SVR-containing Ridge(100) stack was nominally best (RMSE 0.6427 ± 0.0452), but its advantage over the boosting-pair and all-five stacks was about 0.001 and paired scaffold intervals included zero; SVR showed complementary residuals but was a weak standalone learner. Experiment 4's five-model stack had pooled RMSE 0.6767, MAE 0.5053, R² 0.6241; disagreement, AD distance and combined rank correlated only modestly with absolute error (0.185, 0.225 and 0.251), although its lowest-uncertainty 10% had RMSE 0.454. These are all internal results.

The final primary architecture therefore follows Experiment 1's repeated-fold stability, with both Experiment 3 alternatives retained as ablations. The primary architecture was not selected by this run's held-out folds.

## Regression results

| model | n | MAE | RMSE | R2 | median_absolute_error | Pearson_r | Spearman_r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 6102.000 | 0.508 | 0.685 | 0.615 | 0.379 | 0.785 | 0.792 |
| ExtraTrees | 6102.000 | 0.516 | 0.702 | 0.596 | 0.381 | 0.773 | 0.780 |
| SVR | 6102.000 | 0.596 | 0.789 | 0.489 | 0.453 | 0.705 | 0.741 |
| XGBoost | 6102.000 | 0.507 | 0.677 | 0.623 | 0.387 | 0.790 | 0.794 |
| HistGradientBoosting | 6102.000 | 0.512 | 0.684 | 0.616 | 0.384 | 0.785 | 0.789 |
| five_model_ridge_alpha_1 | 6102.000 | 0.499 | 0.669 | 0.633 | 0.377 | 0.796 | 0.803 |
| svr_complementary_ridge_alpha_100 | 6102.000 | 0.499 | 0.668 | 0.634 | 0.376 | 0.797 | 0.804 |
| boosting_pair_ridge_alpha_1 | 6102.000 | 0.504 | 0.673 | 0.628 | 0.380 | 0.793 | 0.797 |
| training_median_baseline | 6102.000 | 0.916 | 1.114 | -0.019 | 0.804 | -0.103 | -0.114 |
| training_mean_baseline | 6102.000 | 0.912 | 1.109 | -0.010 | 0.805 | -0.103 | -0.108 |

Scaffold-cluster bootstrap intervals are reported in `scaffold_bootstrap_intervals.csv`. Fold-level metrics and mean/SD summaries are in `fold_metrics.csv` and `fold_metric_summary.csv`; pooled out-of-fold estimates combine two repeated scaffold partitions and should not be interpreted as independent observations. Median/mean regression and majority/dummy classification baselines are included.

## Median-split activity results

| model | n | prevalence_active | accuracy | balanced_accuracy | active_precision | inactive_precision | active_recall_sensitivity | specificity | F1 | MCC | ROC_AUC | PR_AUC | Brier | ECE_10bin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression_derived | 6102.000 | 0.508 | 0.847 | 0.847 | 0.846 | 0.849 | 0.856 | 0.839 | 0.851 | 0.695 | 0.914 | 0.910 | NA | NA |
| RandomForestClassifier | 6102.000 | 0.508 | 0.837 | 0.836 | 0.803 | 0.882 | 0.900 | 0.772 | 0.849 | 0.679 | 0.902 | 0.885 | 0.124 | 0.031 |
| XGBoostClassifier | 6102.000 | 0.508 | 0.841 | 0.841 | 0.816 | 0.872 | 0.887 | 0.794 | 0.850 | 0.685 | 0.911 | 0.901 | 0.117 | 0.023 |
| LogisticRegression | 6102.000 | 0.508 | 0.782 | 0.782 | 0.769 | 0.797 | 0.816 | 0.748 | 0.792 | 0.565 | 0.856 | 0.835 | 0.154 | 0.030 |
| majority_class_baseline | 6102.000 | 0.508 | 0.501 | 0.496 | 0.506 | 0.482 | 0.799 | 0.193 | 0.619 | -0.010 | 0.496 | 0.506 | NA | NA |
| dummy_prior_baseline | 6102.000 | 0.508 | 0.508 | 0.500 | 0.508 | 0.000 | 1.000 | 0.000 | 0.674 | 0.000 | 0.516 | 0.519 | 0.250 | 0.007 |

The regression-derived classifier thresholds continuous predicted pIC50 at each fold's fit-only median; its score-based ROC-AUC/PR-AUC use continuous pIC50, not a calibrated probability. Direct classifier probabilities were Platt-calibrated on a separate calibration-scaffold partition and are secondary benchmarks. Fold-specific threshold and class counts are in `median_thresholds_by_fold.csv`; the label manifest preserves fit/calibration/test provenance.

## Calibration, uncertainty and pseudolabel filtering

| method | nominal_coverage | empirical_molecule_coverage | mean_width | whole_scaffold_containment | coverage_scaffold_bootstrap_lower | coverage_scaffold_bootstrap_upper | whole_scaffold_bootstrap_lower | whole_scaffold_bootstrap_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| molecule | 0.800 | 0.813 | 1.637 | 0.744 | 0.793 | 0.833 | 0.727 | 0.761 |
| molecule | 0.900 | 0.910 | 2.284 | 0.859 | 0.898 | 0.923 | 0.845 | 0.872 |
| molecule | 0.950 | 0.956 | 2.918 | 0.919 | 0.948 | 0.964 | 0.908 | 0.929 |
| scaffold_max | 0.800 | 0.870 | 1.957 | 0.810 | 0.856 | 0.885 | 0.794 | 0.825 |
| scaffold_max | 0.900 | 0.951 | 2.804 | 0.911 | 0.942 | 0.959 | 0.900 | 0.922 |
| scaffold_max | 0.950 | 0.979 | 3.545 | 0.957 | 0.972 | 0.984 | 0.949 | 0.964 |

| score | outcome | spearman_r | p_value_descriptive |
| --- | --- | --- | --- |
| disagreement_sd | absolute_regression_error | 0.192 | 0.000 |
| disagreement_sd | binary_classification_error | -0.009 | 0.472 |
| disagreement_sd | false_active | -0.026 | 0.040 |
| disagreement_sd | false_inactive | 0.014 | 0.258 |
| ad_distance | absolute_regression_error | 0.225 | 0.000 |
| ad_distance | binary_classification_error | 0.057 | 0.000 |
| ad_distance | false_active | -0.005 | 0.684 |
| ad_distance | false_inactive | 0.085 | 0.000 |
| combined_rank | absolute_regression_error | 0.254 | 0.000 |
| combined_rank | binary_classification_error | 0.036 | 0.005 |
| combined_rank | false_active | -0.015 | 0.253 |
| combined_rank | false_inactive | 0.065 | 0.000 |

| strategy | target_retained_fraction | actual_retained_fraction | n_retained | n_unique_molecules | regression_MAE | regression_RMSE | active_precision | inactive_precision | false_active_rate | false_inactive_rate | unique_scaffolds_retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_AD_uncertainty | 0.100 | 0.100 | 611.000 | 457.000 | 0.308 | 0.422 | 0.824 | 0.902 | 0.204 | 0.083 | 303.000 |
| combined_AD_uncertainty | 0.500 | 0.500 | 3051.000 | 1842.000 | 0.395 | 0.528 | 0.849 | 0.882 | 0.163 | 0.109 | 876.000 |
| combined_AD_uncertainty | 1.000 | 1.000 | 6102.000 | 3051.000 | 0.499 | 0.669 | 0.846 | 0.849 | 0.161 | 0.144 | 1326.000 |

The filter curves describe risk versus retained coverage; counts are held-out prediction records across the two CV repeats, and `n_unique_molecules` discloses the corresponding unique structures. They do not select an operating point. Small error correlations and internal calibration do not establish generated-molecule precision. AD results are stratified by nearest-training Tanimoto similarity in `applicability_domain_analysis.csv`.

## Robustness, learning and error analysis

Y-scrambling: mean RMSE 1.149 (range 1.135–1.166) across 20 permutations; compare with unshuffled reference RMSE 0.684. The empirical lower-tail permutation p-value is 0.048. Learning curves and representation sensitivities are saved in `learning_curves.csv` and `representation_sensitivity.csv`.

The 100 largest molecule-level errors are in `error_analysis.csv`, including SMILES, observed/predicted pIC50, scaffold, similarity, disagreement and median-label error. Scaffold, activity-range, scaffold-size, molecular-weight and lipophilicity analyses are saved separately. Residual/parity figures are under `figures/`.

## Scientific interpretation and limitations

This evidence supports a reproducible, internally scaffold-evaluated QSAR development model, not a publication-ready or experimentally validated predictor. The curated file hash is `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25`. The saved processing notebook documents source counts of 3,485 nM and 28 pIC50 records. Its conversion factor follows conventional molar pIC50, but the code comment says μM; ChEMBL documents pChEMBL as −log10(molar concentration), where 1 nM maps to 9 ([ChEMBL activity FAQ](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)). The absent raw source file prevents row-level verification of those 28 values, so this unit-comment mismatch remains a provenance limitation. Assay IDs, assay conditions, censoring and replicate metadata are absent; upstream filtering and duplicate handling cannot be independently reconstructed. Six disconnected records and four same-InChIKey tautomer pairs remain unchanged; 364 structures have unassigned potential stereocenters. The exact VECTOR+ implementation is unavailable, so only the stipulated median tie convention is followed.

The test folds are internal scaffold validation because previous experiments have already examined this dataset. Calibration is based on experimental molecules, not generated molecules. No external assay-matched validation, prospective pseudolabel precision, or independent evaluation of generated candidates exists. Therefore this package must not describe predictions as experimentally active or treat them as accepted pseudolabels. New assay metadata and an independent external set are required for stronger publication claims.

## Direct answers

1. **Architecture:** predeclared five-model Ridge stack; its rationale is summarized above and in `model_selection_rationale.md`.
2. **Experiment 1:** supplied the repeated-fold stability evidence that set the primary model.
3. **Experiment 3:** showed a small nominal SVR-stack advantage and residual complementarity, but no clear paired superiority.
4. **Experiment 4:** showed modest uncertainty ranking and internal interval behavior, insufficient for a pseudolabel cutoff.
5. **Continuous prediction:** see regression metrics and scaffold-bootstrap intervals above.
6. **Median labels:** see classification metrics; threshold is relative to each fit partition, then the full development median for inference.
7. **Class-specific reliability:** active/inactive precision and error rates are reported separately; neither is externally validated.
8. **Precision/coverage:** see filter curves. No target precision or minimum coverage was pre-approved, so no final cutoff is claimed.
9. **Applicability domain:** performance by similarity bins is in `applicability_domain_analysis.csv`; no universal similarity threshold is justified.
10. **Failure modes:** sparse scaffolds, chemical extrapolation, assay heterogeneity, missing stereochemical/source metadata, and the 100 largest errors.
11. **Controlled VECTOR+ use:** suitable only for cautious research exploration with explicit provenance; current model does not mark valid generated structures as accepted.
12. **Missing evidence:** row-level assay provenance, raw source replay, independent assay-matched validation, exact VECTOR+ implementation, and validation on generated molecules.
