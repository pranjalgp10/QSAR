# Model-selection rationale

## Evidence considered before fitting

Experiment 1 used two repeated five-fold scaffold allocations. The five-model Ridge stack had mean RMSE 0.6483 ± 0.0365 over ten outer folds, compared with 0.6606 ± 0.0347 for XGBoost and 0.6629 ± 0.0435 for HistGradientBoosting. It beat XGBoost in 9/10 and HistGradientBoosting in 10/10 folds. This is the best repeat-level stability evidence available.

Experiment 3 screened 40 stack configurations on a single five-fold allocation. The SVR-containing four-model Ridge(100) candidate had mean RMSE 0.6427 ± 0.0452. Its differences versus the boosting-pair Ridge and all-five Ridge stack were approximately 0.001 and scaffold-bootstrap intervals included zero. SVR residuals were less correlated with other base errors and it had a stable positive stack coefficient, but SVR alone performed substantially worse than the tree/boosting models.

Experiment 4 evaluated a frozen five-model stack: pooled RMSE 0.6767, MAE 0.5053, R² 0.6241. Disagreement/AD/combined error-rank correlations were 0.185/0.225/0.251. The lowest-uncertainty 10% RMSE was 0.454; this is encouraging ranking evidence but only from the same internal dataset.

## Frozen choice before this run

The five-model Ridge stack was frozen as the primary architecture before this run's outer-fold metrics were generated, because Experiment 1's repeated scaffold results provide the strongest stability evidence. The Experiment 3 SVR-complementary Ridge(100) and boosting-pair Ridge are fixed ablations on identical outer folds. Their scores are reported as descriptive comparisons; the held-out folds do not trigger model selection or hyperparameter changes. This avoids selecting a model on the same outer results later presented as its evaluation.
