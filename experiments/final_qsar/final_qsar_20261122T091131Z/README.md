# Final QSAR experiment

This run completes the final regression, median-split classification, calibration, applicability-domain, robustness, inference and reporting workflow. Outputs are isolated in this timestamped folder; previous experiment results and source datasets are preserved.

## Reproduce

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe experiments\final_qsar\final_qsar_20261122T091131Z\run_pipeline.py
```

The command resumes completed outer folds from checkpoints, then writes postprocessed scaffold-bootstrap/error analyses, assembles `final_report.md`, and runs the test suite. The tested environment is pinned in `requirements-lock.txt`; the model dataset hash is checked before training.

## Main outputs

- `final_report.md`: results and scientific interpretation.
- `data_audit.md` and `data_audit.json`: data provenance and integrity.
- `pooled_oof_predictions.csv`, `fold_metrics.csv`: held-out scaffold predictions and metrics.
- `scaffold_bootstrap_intervals.csv`, `interval_coverage.csv`: clustered uncertainty intervals.
- `precision_coverage.csv`, `applicability_domain_analysis.csv`: descriptive reliability/filter analyses.
- `model.joblib`, `model_metadata.json`, `inference.py`: final fitted model package.
- `test_results.txt`: automated validation results.

This run is internal scaffold validation. It is not external assay validation or generated-molecule validation. No reliability cutoff for pseudolabel acceptance was selected.
