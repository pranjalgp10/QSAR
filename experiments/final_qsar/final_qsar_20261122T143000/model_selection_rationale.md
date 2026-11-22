# Provisional model-selection rationale

No model was selected or fit in this run because data provenance failed the required preflight gate. The following reconciles prior internal findings without treating them as external confirmation.

- **Experiment 1:** the five-model Ridge stack had the best repeated scaffold-CV mean RMSE (0.6483 ± 0.0365 across ten folds). It beat XGBoost in 9/10 and HistGradientBoosting in 10/10 paired folds. This is the strongest stability evidence among completed experiments.
- **Experiment 3:** an SVR-containing stack with Ridge α=100 ranked first in a single five-fold architecture screen (mean RMSE 0.6427 ± 0.0452). However, its paired scaffold-bootstrap RMSE differences versus the boosting-pair stack and all-five Ridge stack included zero. The small apparent advantage does not justify selecting it solely by rank.
- **SVR complementarity:** SVR residuals were less correlated with other base learners, and its selected-stack coefficient was positive and stable across five folds. SVR standalone RMSE was much worse, so it is a plausible complement rather than an established necessary component.
- **Experiment 4:** the frozen five-model stack had pooled OOF RMSE 0.6767. Reliability ranking was modest (combined-score Spearman ρ=0.251); a small lowest-uncertainty subset had lower error, but this does not establish a deployable threshold.

If the dataset gate passes, the five-model Ridge stack is the provisional primary because of Experiment 1's repeat-level consistency and available reliability analysis. The Experiment 3 SVR-complementary Ridge(100), boosting-pair Ridge, and all-five Ridge candidates should be evaluated on identical nested outer folds. If their differences remain negligible, prefer the simpler candidate whose performance is stable. This is a preregistered rationale for future evaluation, not a final selection.
