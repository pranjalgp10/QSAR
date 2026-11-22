"""Assemble a results-grounded report after evaluation and postprocessing."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parent


def frame(path):
    p = RUN / path
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fmt(value, digits=3):
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def md_table(df, columns=None):
    if df.empty:
        return "_Not produced._"
    view = df[columns] if columns else df
    headers = list(view.columns)
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value, 3) if isinstance(value, (float, int)) else str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    reg, cls = frame("regression_metrics.csv"), frame("classification_metrics.csv")
    boot, coverage = frame("scaffold_bootstrap_intervals.csv"), frame("interval_coverage.csv")
    threshold = json.loads((RUN / "final_activity_threshold.json").read_text(encoding="utf-8"))
    audit = json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))
    uncertainty, ad = frame("uncertainty_analysis.csv"), frame("applicability_domain_analysis.csv")
    risk, scramble = frame("precision_coverage.csv"), frame("y_scrambling_results.csv")
    learning, reps = frame("learning_curves.csv"), frame("representation_sensitivity.csv")
    cls_row = cls.loc[cls.model == "regression_derived"].iloc[0] if not cls.empty else pd.Series(dtype=float)
    primary = reg.loc[reg.model == "five_model_ridge_alpha_1"].iloc[0] if not reg.empty else pd.Series(dtype=float)
    primary_ci = boot.loc[(boot.task == "regression") & (boot.model == "five_model_ridge_alpha_1") & (boot.metric == "RMSE")]
    ci = primary_ci.iloc[0] if not primary_ci.empty else pd.Series(dtype=float)
    interval90 = coverage.loc[(coverage.method == "molecule") & (coverage.nominal_coverage == .9)]
    interval90 = interval90.iloc[0] if not interval90.empty else pd.Series(dtype=float)
    scramble_p = ((1 + int((scramble.RMSE <= scramble.true_unshuffled_reference_HGB_RMSE).sum())) / (len(scramble) + 1)) if not scramble.empty else float("nan")
    rel = risk.loc[risk.target_retained_fraction.isin([.1, .5, 1.0])] if not risk.empty else risk
    rel = rel.loc[rel.strategy == "combined_AD_uncertainty"] if not rel.empty else rel
    regcols = ["model", "n", "MAE", "RMSE", "R2", "median_absolute_error", "Pearson_r", "Spearman_r"]
    clcols = ["model", "n", "prevalence_active", "accuracy", "balanced_accuracy", "active_precision", "inactive_precision", "active_recall_sensitivity", "specificity", "F1", "MCC", "ROC_AUC", "PR_AUC", "Brier", "ECE_10bin"]
    report = f"""# Final QSAR development report

Run: `{RUN.name}`. All reported evaluation is internal scaffold validation on the curated dataset; there is no virgin external validation or generated-molecule validation.

## Executive result

The predeclared primary model is the five-model Ridge stack (Random Forest, ExtraTrees, SVR, XGBoost, HistGradientBoosting; Ridge alpha 1). It was frozen from the preceding experiments before this run's outer-fold predictions were inspected. The run compares it with individual learners, the SVR-complementary stack, and the boosting-pair stack; these outer-fold comparisons are descriptive and were not used to change the architecture.

- Continuous pIC50: MAE {fmt(primary.get('MAE'))}, RMSE {fmt(primary.get('RMSE'))} (scaffold-bootstrap 95% CI {fmt(ci.get('lower_95'))}–{fmt(ci.get('upper_95'))}), R² {fmt(primary.get('R2'))}.
- Median-derived classification: accuracy {fmt(cls_row.get('accuracy'))}, balanced accuracy {fmt(cls_row.get('balanced_accuracy'))}, active precision {fmt(cls_row.get('active_precision'))}, inactive precision {fmt(cls_row.get('inactive_precision'))}, MCC {fmt(cls_row.get('MCC'))}, ROC-AUC {fmt(cls_row.get('ROC_AUC'))}, PR-AUC {fmt(cls_row.get('PR_AUC'))}.
- Full-development median threshold: {fmt(threshold['threshold_pIC50'], 5)} pIC50; active means greater than or equal to this relative dataset median. Counts: {threshold['active_count']} active and {threshold['inactive_count']} inactive; ties at median: {threshold['ties_at_threshold']}.
- A 90% molecule interval had empirical coverage {fmt(interval90.get('empirical_molecule_coverage'))}, mean width {fmt(interval90.get('mean_width'))} pIC50. This is internal calibration/validation evidence and does not establish coverage on generated chemistry.
- No pseudolabel-acceptance cutoff is established. Inference returns `caution` for every valid molecule; invalid structures are rejected.

## Architecture evidence from Experiments 1, 3 and 4

Experiment 1's repeated scaffold evaluation favored the five-model stack on stability (RMSE 0.6483 ± 0.0365 across ten folds; better than XGBoost in 9/10 and HistGradientBoosting in 10/10 folds). Experiment 3's SVR-containing Ridge(100) stack was nominally best (RMSE 0.6427 ± 0.0452), but its advantage over the boosting-pair and all-five stacks was about 0.001 and paired scaffold intervals included zero; SVR showed complementary residuals but was a weak standalone learner. Experiment 4's five-model stack had pooled RMSE 0.6767, MAE 0.5053, R² 0.6241; disagreement, AD distance and combined rank correlated only modestly with absolute error (0.185, 0.225 and 0.251), although its lowest-uncertainty 10% had RMSE 0.454. These are all internal results.

The final primary architecture therefore follows Experiment 1's repeated-fold stability, with both Experiment 3 alternatives retained as ablations. The primary architecture was not selected by this run's held-out folds.

## Regression results

{md_table(reg, regcols)}

Scaffold-cluster bootstrap intervals are reported in `scaffold_bootstrap_intervals.csv`. Fold-level metrics and mean/SD summaries are in `fold_metrics.csv` and `fold_metric_summary.csv`; pooled out-of-fold estimates combine two repeated scaffold partitions and should not be interpreted as independent observations. Median/mean regression and majority/dummy classification baselines are included.

## Median-split activity results

{md_table(cls, clcols)}

The regression-derived classifier thresholds continuous predicted pIC50 at each fold's fit-only median; its score-based ROC-AUC/PR-AUC use continuous pIC50, not a calibrated probability. Direct classifier probabilities were Platt-calibrated on a separate calibration-scaffold partition and are secondary benchmarks. Fold-specific threshold and class counts are in `median_thresholds_by_fold.csv`; the label manifest preserves fit/calibration/test provenance.

## Calibration, uncertainty and pseudolabel filtering

{md_table(coverage)}

{md_table(uncertainty)}

{md_table(rel, ["strategy", "target_retained_fraction", "actual_retained_fraction", "n_retained", "n_unique_molecules", "regression_MAE", "regression_RMSE", "active_precision", "inactive_precision", "false_active_rate", "false_inactive_rate", "unique_scaffolds_retained"])}

The filter curves describe risk versus retained coverage; counts are held-out prediction records across the two CV repeats, and `n_unique_molecules` discloses the corresponding unique structures. They do not select an operating point. Small error correlations and internal calibration do not establish generated-molecule precision. AD results are stratified by nearest-training Tanimoto similarity in `applicability_domain_analysis.csv`.

## Robustness, learning and error analysis

Y-scrambling: mean RMSE {fmt(scramble.RMSE.mean())} (range {fmt(scramble.RMSE.min())}–{fmt(scramble.RMSE.max())}) across {len(scramble)} permutations; compare with unshuffled reference RMSE {fmt(scramble.true_unshuffled_reference_HGB_RMSE.iloc[0] if not scramble.empty else None)}. The empirical lower-tail permutation p-value is {fmt(scramble_p, 3)}. Learning curves and representation sensitivities are saved in `learning_curves.csv` and `representation_sensitivity.csv`.

The 100 largest molecule-level errors are in `error_analysis.csv`, including SMILES, observed/predicted pIC50, scaffold, similarity, disagreement and median-label error. Scaffold, activity-range, scaffold-size, molecular-weight and lipophilicity analyses are saved separately. Residual/parity figures are under `figures/`.

## Scientific interpretation and limitations

This evidence supports a reproducible, internally scaffold-evaluated QSAR development model, not a publication-ready or experimentally validated predictor. The curated file hash is `{audit['curated_sha256']}`. The saved processing notebook documents source counts of 3,485 nM and 28 pIC50 records. Its conversion factor follows conventional molar pIC50, but the code comment says μM; ChEMBL documents pChEMBL as −log10(molar concentration), where 1 nM maps to 9 ([ChEMBL activity FAQ](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)). The absent raw source file prevents row-level verification of those 28 values, so this unit-comment mismatch remains a provenance limitation. Assay IDs, assay conditions, censoring and replicate metadata are absent; upstream filtering and duplicate handling cannot be independently reconstructed. Six disconnected records and four same-InChIKey tautomer pairs remain unchanged; 364 structures have unassigned potential stereocenters. The exact VECTOR+ implementation is unavailable, so only the stipulated median tie convention is followed.

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
"""
    (RUN / "final_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
