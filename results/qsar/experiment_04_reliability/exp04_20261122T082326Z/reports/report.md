# Experiment 4 results

Run: `exp04_20261122T082326Z`

Dataset: 3,051 curated structures; SHA-256 `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25`.

Evaluation: five-fold scaffold-grouped OOF predictions. Each fold also has a scaffold-separated fit, calibration, and outer-test partition.

## Pooled model performance

| Model | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Stacking | 3051 | 0.5053 | 0.6767 | 0.6241 |
| RandomForest | 3051 | 0.5146 | 0.6917 | 0.6072 |
| ExtraTrees | 3051 | 0.5176 | 0.7036 | 0.5936 |
| SVR | 3051 | 0.5925 | 0.7850 | 0.4941 |
| XGBoost | 3051 | 0.5109 | 0.6838 | 0.6161 |
| HistGradientBoosting | 3051 | 0.5172 | 0.6903 | 0.6088 |

## Reliability and intervals

Spearman association between absolute error and disagreement: 0.185; AD distance: 0.225; combined rank: 0.251.

See `../metrics/uncertainty_reliability_bins.csv` and `../metrics/precision_coverage.csv` for decile reliability and retained-molecule error summaries.

| Interval | Nominal | Molecule coverage | Whole-scaffold containment | Mean width (pIC50) |
|---|---:|---:|---:|---:|
| molecule | 80% | 0.789 | 0.713 | 1.550 |
| molecule | 90% | 0.888 | 0.833 | 2.135 |
| molecule | 95% | 0.943 | 0.905 | 2.758 |
| scaffold | 80% | 0.870 | 0.811 | 2.005 |
| scaffold | 90% | 0.942 | 0.905 | 2.706 |
| scaffold | 95% | 0.974 | 0.952 | 3.385 |

The primary molecule-level 90% interval has cluster-bootstrap coverage CI 0.872–0.904; scaffold-max interval whole-scaffold containment CI 0.888–0.920.

Among the lowest-uncertainty 10% by combined calibration-referenced rank, RMSE was 0.454 for n=306 (compare pooled RMSE 0.677). This is a descriptive ranking result, not an acceptance rule.

Across all OOF predictions, fraction within ±0.5 pIC50: 0.610; within ±1.0 pIC50: 0.872. These are descriptive regression tolerances, not activity or pseudolabel cutoffs.

Internal scaffold CV on a dataset whose earlier model/test results have been inspected; no independent external cohort or assay metadata. Disagreement and similarity are diagnostics rather than probabilities. Empirical coverage does not establish calibration on generated molecules or under scaffold distribution shift. No cutoff was selected from these curves.
