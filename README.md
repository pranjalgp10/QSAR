# QSAR-Guided Data Augmentation for VECTOR+

This repository develops a Syk inhibitor QSAR predictor as one component of a larger study: whether reliable QSAR pseudolabels can improve VECTOR+-style molecular generation. The project evaluates the predictor, its uncertainty and applicability domain, then provides a controlled inference interface. Predicted activity is computational evidence; it is not experimental potency.

## Current status

The final QSAR development run is complete. It used the 3,051-row curated Syk dataset and repeated five-fold scaffold-grouped validation. The primary model is a five-model Ridge stack using Random Forest, ExtraTrees, SVR, XGBoost and HistGradientBoosting. Its pooled internal scaffold results were MAE 0.499, RMSE 0.669 (scaffold-bootstrap 95% CI 0.639–0.699), and R² 0.633. The regression-derived median-split classifier had active precision 0.846, inactive precision 0.849, MCC 0.695 and ROC-AUC 0.914.

These are **internal scaffold validation results**, not external validation. Prior experiments examined the same dataset. The model returns `caution` for valid molecules because no validated pseudolabel-acceptance threshold or generated-molecule performance has been established. See the [final report](experiments/final_qsar/final_qsar_20261122T091131Z/final_report.md) and [limitations](experiments/final_qsar/final_qsar_20261122T091131Z/limitations.md).

## Scientific objective and experimental design

The research question is whether QSAR-guided molecular augmentation improves generated-molecule quality. The planned comparison includes:

1. Original VECTOR+ baseline.
2. Additional contrastive training without pseudolabelled molecules.
3. QSAR-guided augmentation with reliability filtering.
4. No-confidence-filtering ablation.
5. Disjoint versus shared experimental training pools for QSAR and VECTOR+.

The current repository contains the QSAR development and reliability experiments. A VECTOR+ implementation is not present, so the generation and augmentation comparisons remain future work. The exact VECTOR+ activity-label source implementation has not been independently verified; the current classifier follows the specified training-median convention.

The primary QSAR task is continuous pIC50 regression. For the VECTOR+-compatible binary task, each fit partition sets its own observed-label median; `pIC50 >= fit median` is active and ties are active. The matching threshold is applied unchanged to calibration and held-out labels. The saved deployment threshold is 7.37675 pIC50, a relative dataset-specific split—not a universal potency threshold.

## Repository layout

| Path | Purpose |
| --- | --- |
| `processed_df.csv` | Processed 3,176-row source table supplied for curation; retained unchanged. |
| `data/processed/` | Hash-pinned curated structures, duplicate-resolution table and curation report. |
| `docs/` | Data handling, QSAR methodology, audits and experiment design notes. |
| `src/data/` | Reproducible Syk dataset curation. |
| `src/qsar/` | Earlier benchmark, model comparison, stacking and reliability experiment code. |
| `experiments/qsar/` | Experiment configurations and reproduction notes. |
| `experiments/final_qsar/<run_id>/` | Isolated final QSAR run, reports, metrics, figures, model, inference and tests. |
| `results/qsar/` | Preserved outputs for Experiments 1, 3 and 4. |
| `results/qsar_benchmark/` | Earlier benchmark outputs; retained as historical results. |
| `tests/` | Tests for the earlier benchmark implementation. |
| `.venv/` | Local Python environment; do not commit. |

The final run folder contains its own README, frozen configuration, random seeds, environment versions, data audit, fold assignments, out-of-fold predictions, bootstrap intervals, model card and execution log. Do not overwrite historical experiment directories or raw data.

## Data and provenance

The primary modeling dataset is `data/processed/syk_qsar_curated.csv` with 3,051 structures. Its SHA-256 is:

```text
541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25
```

The curation code and duplicate decisions are in `src/data/curate_syk.py`, `docs/data_handling.md`, and `data/processed/syk_duplicate_resolution.csv`. The final audit documents invalid structures, duplicates, disconnected structures, stereochemistry, pIC50 arithmetic and provenance limitations.

The saved processing notebook indicates 3,485 nM and 28 pIC50 source rows and converts pIC50 using a molar-scale factor. Its comment says μM, which conflicts with that factor. The processed arithmetic is internally consistent, but the referenced raw source file is unavailable for row-level verification of the 28 pIC50 entries and upstream filtering. Assay IDs, replicate/censoring details and assay conditions are also absent. Accordingly, the final report marks the dataset **not publication-ready** despite permitting internal model evaluation.

## Environment setup

The recorded environment is Python 3.14.8. From the repository root in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-qsar.txt
```

For the exact final-run package versions, use its `requirements-lock.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments\final_qsar\final_qsar_20261122T091131Z\requirements-lock.txt
```

The final experiment pins NumPy, pandas, scikit-learn, RDKit, XGBoost, SciPy, Matplotlib, joblib and pytest. Matplotlib is used for the saved figures. Confirm the installed versions against `experiments/final_qsar/final_qsar_20261122T091131Z/environment_versions.txt`.

## Reproduce the final QSAR evaluation

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\final_qsar\final_qsar_20261122T091131Z\run_pipeline.py
```

The command checks the expected dataset hash, resumes any completed scaffold folds, runs evaluation and final fitting when needed, postprocesses clustered intervals and error analyses, builds the report, and runs the final-run tests. The completed run will skip refitting and rerun postprocessing/report/tests. Configuration and seeds are in the run directory. The outputs include:

- `final_report.md`, `methods.md`, `limitations.md`, and `model_selection_rationale.md`.
- `data_audit.json`, `data_audit.md`, `dataset_manifest.json`, and duplicate/exclusion tables.
- Regression/classification/fold metrics and pooled held-out predictions.
- Scaffold-bootstrap intervals, calibration coverage, threshold sensitivity, precision/coverage, applicability and uncertainty analyses.
- Y-scrambling, learning curves, representation sensitivity, error-analysis files and figures.
- `model.joblib`, `model_metadata.json`, `activity_threshold.json`, `inference.py`, `model_card.md`, and `test_results.txt`.

Prior experiments can be reproduced with:

```powershell
.\.venv\Scripts\python.exe src\qsar\experiment_01.py
.\.venv\Scripts\python.exe src\qsar\experiment_03.py
.\.venv\Scripts\python.exe src\qsar\experiment_04_reliability.py
```

Those scripts create run outputs under `results/qsar/`. Inspect each experiment README/configuration before rerunning; do not replace the preserved reference outputs.

## Use the final inference interface

The inference implementation and serialized model are in `experiments/final_qsar/final_qsar_20261122T091131Z/`. For example:

```powershell
.\.venv\Scripts\python.exe experiments\final_qsar\final_qsar_20261122T091131Z\inference.py "CCO" "c1ccccc1"
```

Or import `predict_smiles`:

```python
from experiments.final_qsar.final_qsar_20261122T091131Z.inference import predict_smiles

predictions = predict_smiles(["CCO", "c1ccccc1"])
print(predictions)
```

The output includes canonical SMILES, predicted pIC50, predicted IC50 in nM, the saved median-split label, nearest-training Tanimoto similarity, applicability-domain distance, model disagreement, combined held-out-reference rank, and an approximate internal interval. Invalid SMILES are rejected. All valid predictions carry `caution` because no pseudolabel cutoff is validated. Do not call a predicted active molecule experimentally active or use the confidence score as proof of correctness.

## Evaluation procedure and interpretation

The final run uses two repeated five-fold Bemis–Murcko scaffold partitions (seeds 140 and 141). Within every outer training partition, calibration scaffolds are separated from model-fit scaffolds. Base-model stacking features are generated strictly out of fold; the outer held-out scaffolds do not train the base learners, meta-learner, binary threshold or calibration. The reported pooled set contains 6,102 held-out prediction records for 3,051 unique molecules because each molecule is held out once per repeat.

Architecture selection was predeclared from prior evidence: Experiment 1 supported stability of the five-model Ridge stack; Experiment 3 found a small, statistically unresolved advantage for an SVR-containing stack; Experiment 4 showed modest uncertainty/error association. The final outer-fold results are reported as evaluation, not used to retune the primary architecture. A median/mean regression and majority/dummy classification baseline are included.

The 90% molecule interval achieved 91.0% internal held-out coverage at mean width 2.284 pIC50; the scaffold-maximum interval is wider. Combined AD/uncertainty filtering is descriptive and no operating threshold was chosen. These results do not demonstrate precision on generated molecules, external assay generalization, or that augmentation improves generation.

## Development and reporting rules

- Preserve source data and prior experiment outputs.
- Keep experimental measurements separate from predictions and pseudolabels.
- Record dataset hashes, split provenance, seeds, versions and model settings.
- Keep all new run outputs under a timestamped experiment folder.
- Do not use held-out labels for training, tuning, threshold selection or reliability cutoffs.
- Report scaffold-aware evaluation as internal validation when prior work has examined the dataset.
- Do not claim external validation, experimentally confirmed potency, or publication-grade validity without independent data and assay provenance.
- VECTOR+ generation, matched controls, pseudolabel augmentation and downstream molecular evaluation are not implemented in this repository yet.
