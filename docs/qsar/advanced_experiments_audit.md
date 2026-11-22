# Advanced QSAR experiments: repository audit

## Scope and preservation

The previous benchmark at `results/qsar_benchmark/` is complete and is treated as the immutable reference for the three requested follow-up experiments. Its report, deployment model, split manifest, nested OOF predictions and reliability files were present during audit. Hashes recorded during audit:

- `report.json`: `20873B5E5AA2A897BE80D4E30CAECEED1C0…`
- `deployment_model.joblib`: `6231490B4951880211DE15FCAB4A2341F94…`
- `split_manifest.csv`: `1299B1507A28292EABC7C3FFACE87607FD8…`

No changes were made to `results/qsar_benchmark/`, the curated dataset, raw input, benchmark implementation, or scientific-context documents.

## Existing reference results

The existing report records 3,051 canonical structures and a scaffold-grouped split of 2,006 training, 525 validation and 520 test structures. The validation-selection subset has 202 structures and calibration subset has 323. HistGradientBoosting was selected on validation RMSE (0.706); stacking was close (0.715). On the internal test partition, stacking RMSE was 0.674 and HistGradientBoosting RMSE was 0.684. Nested training scaffold CV RMSE was 0.650 for stacking, 0.666 for HistGradientBoosting and 0.661 for XGBoost. Differences are modest and rankings do not establish a stable winner.

The original report describes uncertainty diagnostics with weak association to absolute error (disagreement Spearman 0.148; nearest-similarity Spearman -0.229) and a broad nominal 90% interval (half-width 1.323 pIC50, observed coverage 93.65%). This is a useful reference, not proof of calibrated prediction reliability. The test partition has already been examined and is internal, not external validation.

## Repository inventory and cleanup

Core files present: `AGENTS.md`, `Scientific_Context.md`, `processed_df.csv`, `data/processed/syk_qsar_curated.csv`, curation records, `src/data/curate_syk.py`, `src/qsar/benchmark.py`, tests, environment requirements and existing benchmark outputs. These contain useful scientific or reproducibility information and were preserved.

Removed only disposable project-level Python/pytest caches (`src/qsar/__pycache__`, `tests/__pycache__`, `.pytest_cache`). The `.venv` was preserved; its package caches are part of the environment and are not redundant project artifacts. No other repository items were identified as safely deletable.

## Follow-up experiment plan

Create new, independent directories and outputs for Experiment 1 (repeated paired scaffold-CV comparisons), Experiment 3 (leakage-controlled stacking combinations and meta-models), and Experiment 4 (reliability, applicability domain, interval and precision-coverage analyses). All use the curated dataset and fingerprint settings. None may tune on the original benchmark test outcomes. Use fresh group folds; label every result internal scaffold validation. Do not select pseudolabel thresholds without validation-based evidence and a documented decision.

Implementation status: Experiments 1, 3 and 4 have separate scripts/configurations/output roots. Experiment 4 uses frozen benchmark settings, scaffold-disjoint fit/calibration/outer-test subsets, molecular and scaffold-maximum conformal intervals, and descriptive precision–coverage curves. Its results remain internal cross-validation because previous evaluations on this dataset have been inspected; it does not set a pseudolabel cutoff.
