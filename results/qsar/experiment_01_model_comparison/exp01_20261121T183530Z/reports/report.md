# Experiment 1 — repeated scaffold-aware model comparison

## Question and design

Does stacking consistently improve continuous pIC50 prediction over individual RF, Extra Trees, SVR, XGBoost and HistGradientBoosting models on previously unseen Bemis–Murcko scaffolds?

The curated 3,051-structure dataset was evaluated with two independent shuffled five-fold `GroupKFold` allocations (seeds 140 and 141). Each allocation uses every structure once as an outer held-out prediction; the same folds were used for every individual model and stacking. Within each outer-training set, three-candidate randomized hyperparameter search used three-fold scaffold CV and RMSE. Stack meta-training used cross-fitted predictions from five predeclared reference base learners and Ridge; each outer-test fold was excluded from all meta-fitting. The original benchmark test outcomes were not used to select this experiment's individual-model parameters.

The 2-repeat design is a compute-reduced alternative to the preferred three or more repeats. Across the ten outer folds, held-out molecule counts ranged from 505 to 755 because GroupKFold balances scaffold groups rather than molecule counts. Each test fold held out 265–266 scaffolds; training sets held 1,060–1,061 scaffolds. Exact assignments are in `manifests/folds.json`.

## Aggregate fold metrics

Values below are means of the ten outer-fold scores; SD is across folds and should be read as split-to-split variability, not a classical independent-sample standard error.

| Model | RMSE mean ± SD | MAE mean | R² mean |
|---|---:|---:|---:|
| Stacking | **0.648 ± 0.037** | **0.483** | **0.648** |
| XGBoost | 0.661 ± 0.035 | 0.492 | 0.635 |
| HistGradientBoosting | 0.663 ± 0.043 | 0.492 | 0.632 |
| Random Forest | 0.665 ± 0.038 | 0.495 | 0.630 |
| Extra Trees | 0.676 ± 0.046 | 0.503 | 0.617 |
| SVR | 0.777 ± 0.037 | 0.589 | 0.495 |

Median and interquartile fold RMSE, all per-fold metrics, best parameters and inner-CV scores are in `metrics/summary.csv` and `metrics/fold_metrics.csv`.

## Paired comparisons

Stacking had lower outer-fold RMSE in 10/10 folds than HistGradientBoosting, 9/10 than XGBoost, 9/10 than Random Forest, 10/10 than Extra Trees, and 10/10 than SVR. Mean paired fold RMSE differences (Stacking minus comparator) were −0.0146 versus HistGradientBoosting, −0.0123 versus XGBoost, −0.0162 versus Random Forest, −0.0273 versus Extra Trees, and −0.1286 versus SVR.

Scaffold-cluster bootstrap intervals for pooled paired prediction differences exclude zero for these comparisons. These intervals condition on the evaluated data and fitted folds; repeated CV allocations are dependent, and the intervals do not establish universal superiority under distribution shift. No p-values or claims of definitive significance are made. Full MAE, RMSE and R² differences and bootstrap intervals are in `metrics/paired_rmse_differences.csv`.

## Interpretation

The fixed Ridge stack is the strongest candidate in this experiment, but its advantage over XGBoost and HistGradientBoosting is modest. The large gap to SVR does not justify claims about all Syk chemistry, and the approximately 0.65 pIC50 RMSE corresponds to a typical several-fold concentration error on the logarithmic scale. Experiment 3 should test whether the small gain persists when base combinations and meta-models are optimized with stricter nested cross-fitting.

These results do **not** show that pseudolabels are reliable. Experiment 4 must evaluate prediction intervals, disagreement, applicability domain, and precision-versus-coverage on held-out data. No binary activity cutoff or pseudolabel acceptance cutoff was selected here.

## Limitations

- Only two scaffold allocations were run (ten outer folds); this is fewer than the preferred three repetitions.
- The dataset lacks assay/source metadata. Duplicate groups were resolved with the Syk-paper rule, including retaining the lowest IC50 when the group range exceeded 10% of its median; this potency-favoring decision may bias labels.
- The input appears to have already undergone IQR outlier removal in the source notebook; the unavailable raw source file prevents review or reversal.
- Earlier results from this CSV have been viewed. This experiment is internal scaffold validation, not external validation.
- Outer-fold metrics are descriptive. Scaffold-bootstrap intervals quantify resampling of the observed scaffolds but do not account fully for every modeling or dataset shift uncertainty.

## Reproducibility

Run `src/qsar/experiment_01.py`; outputs are written to a unique run directory. This run used config snapshot `configs/config.json`, split manifest `manifests/folds.json`, outer predictions `predictions/outer_oof_predictions.csv`, and scikit-learn 1.9.1, RDKit 2026.09.2, XGBoost 3.4.1, Python 3.14.8. Checkpoint/resume logic retained six completed folds after an interrupted attempt; the final run completed all ten folds. Earlier failed/interrupted attempt directories remain as run records.
