# Experiment 3 results

Run: `exp03_20261122T063732Z`

Data: 3,051 curated structures; SHA-256 `541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25`.

Outer evaluation: 5-fold scaffold-grouped CV. No independent external cohort is available.

## Candidate summary

| Candidate | Folds | Mean RMSE | SD RMSE | Mean MAE | Mean R² |
|---|---:|---:|---:|---:|---:|
| complementarity_svR__Ridge_a100 | 5 | 0.6427 | 0.0452 | 0.4782 | 0.6559 |
| complementarity_svR__ElasticNet_l10.1 | 5 | 0.6434 | 0.0449 | 0.4787 | 0.6552 |
| boosting_pair__Ridge_a10 | 5 | 0.6436 | 0.0449 | 0.4787 | 0.6551 |
| boosting_pair__Ridge_a1 | 5 | 0.6436 | 0.0450 | 0.4787 | 0.6551 |
| boosting_pair__ElasticNet_l10.1 | 5 | 0.6436 | 0.0447 | 0.4789 | 0.6550 |
| boosting_pair__Ridge_a0.1 | 5 | 0.6436 | 0.0450 | 0.4786 | 0.6551 |
| boosting_pair__OLS | 5 | 0.6436 | 0.0450 | 0.4786 | 0.6551 |
| complementarity_svR__ElasticNet_l10.5 | 5 | 0.6436 | 0.0450 | 0.4788 | 0.6549 |
| complementarity_svR__Ridge_a10 | 5 | 0.6437 | 0.0448 | 0.4789 | 0.6549 |
| boosting_pair__ElasticNet_l10.5 | 5 | 0.6438 | 0.0446 | 0.4791 | 0.6549 |
| complementarity_svR__ElasticNet_l10.9 | 5 | 0.6439 | 0.0449 | 0.4788 | 0.6547 |
| all5__Ridge_a100 | 5 | 0.6439 | 0.0444 | 0.4777 | 0.6546 |
| boosting_pair__ElasticNet_l10.9 | 5 | 0.6440 | 0.0445 | 0.4793 | 0.6546 |
| complementarity_svR__Ridge_a1 | 5 | 0.6440 | 0.0447 | 0.4792 | 0.6546 |
| complementarity_svR__Ridge_a0.1 | 5 | 0.6440 | 0.0447 | 0.4792 | 0.6545 |
| complementarity_svR__OLS | 5 | 0.6440 | 0.0447 | 0.4792 | 0.6545 |
| boosting_pair__Ridge_a100 | 5 | 0.6442 | 0.0440 | 0.4800 | 0.6543 |
| rf_boosting__Ridge_a100 | 5 | 0.6448 | 0.0448 | 0.4789 | 0.6537 |
| all5__ElasticNet_l10.1 | 5 | 0.6449 | 0.0437 | 0.4783 | 0.6535 |
| all5__ElasticNet_l10.5 | 5 | 0.6451 | 0.0436 | 0.4782 | 0.6533 |
| rf_boosting__ElasticNet_l10.1 | 5 | 0.6451 | 0.0447 | 0.4788 | 0.6534 |
| rf_boosting__ElasticNet_l10.5 | 5 | 0.6452 | 0.0445 | 0.4789 | 0.6533 |
| rf_boosting__ElasticNet_l10.9 | 5 | 0.6453 | 0.0444 | 0.4790 | 0.6532 |
| rf_boosting__Ridge_a10 | 5 | 0.6453 | 0.0447 | 0.4789 | 0.6532 |
| all5__ElasticNet_l10.9 | 5 | 0.6453 | 0.0434 | 0.4783 | 0.6530 |
| rf_boosting__Ridge_a1 | 5 | 0.6455 | 0.0446 | 0.4791 | 0.6529 |
| rf_boosting__Ridge_a0.1 | 5 | 0.6456 | 0.0446 | 0.4791 | 0.6529 |
| rf_boosting__OLS | 5 | 0.6456 | 0.0446 | 0.4791 | 0.6529 |
| all5__Ridge_a10 | 5 | 0.6466 | 0.0435 | 0.4794 | 0.6517 |
| all5__Ridge_a1 | 5 | 0.6490 | 0.0437 | 0.4811 | 0.6489 |
| all5__Ridge_a0.1 | 5 | 0.6495 | 0.0438 | 0.4814 | 0.6484 |
| all5__OLS | 5 | 0.6495 | 0.0438 | 0.4814 | 0.6484 |
| rf_et_xgb__Ridge_a100 | 5 | 0.6515 | 0.0451 | 0.4819 | 0.6465 |
| rf_et_xgb__ElasticNet_l10.1 | 5 | 0.6527 | 0.0450 | 0.4824 | 0.6452 |
| rf_et_xgb__ElasticNet_l10.5 | 5 | 0.6534 | 0.0449 | 0.4829 | 0.6443 |
| rf_et_xgb__ElasticNet_l10.9 | 5 | 0.6536 | 0.0447 | 0.4831 | 0.6440 |
| rf_et_xgb__Ridge_a10 | 5 | 0.6540 | 0.0452 | 0.4833 | 0.6436 |
| rf_et_xgb__Ridge_a1 | 5 | 0.6567 | 0.0464 | 0.4852 | 0.6406 |
| rf_et_xgb__Ridge_a0.1 | 5 | 0.6571 | 0.0466 | 0.4855 | 0.6400 |
| rf_et_xgb__OLS | 5 | 0.6572 | 0.0467 | 0.4855 | 0.6399 |

## Pooled out-of-fold scores

The pooled out-of-fold table is saved at `../metrics/pooled_oof_metrics.csv`; these metrics aggregate every held-out molecule once.

## Interpretation

Lowest mean outer-fold RMSE in this screening: **complementarity_svR__Ridge_a100**. This is a candidate selected from comparisons on the same outer folds; its displayed performance is subject to selection optimism and must not be reported as an unbiased final estimate.

Paired scaffold-bootstrap comparisons with selected reference candidates are in `../metrics/screened_best_paired_scaffold_bootstrap.csv`. They are descriptive and do not correct for choosing the winner on these folds. Base-error correlations were calculated from leakage-controlled nested OOF predictions. See `../metrics/oof_residual_correlation.csv` and coefficient stability output.

One five-fold scaffold allocation and two hyperparameter search trials per fit were used to bound compute. Candidate rankings are exploratory model-selection results; no independent external test exists. Because the same outer folds screen multiple candidates, the lowest-ranked candidate metric is selection-optimistic and is not an unbiased final estimate. Pooled OOF metrics combine outer-fold held-out predictions once per structure. Scaffold-cluster bootstrap intervals compare the screened winner with a few predeclared alternatives but do not correct winner-selection or multiple-comparison optimism. Strictly nested OOF base tuning excludes each meta fold; meta-model settings were fixed before outer-fold evaluation.
