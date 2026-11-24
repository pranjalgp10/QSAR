# QSAR Modelling of SYK Inhibitors

**A machine-learning framework for predicting the potency of SYK inhibitors using molecular fingerprints, ensemble regression, and scaffold-aware validation.**

## Overview

This project develops a Quantitative Structure–Activity Relationship (QSAR) model to predict the biological potency of small molecules targeting **Spleen Tyrosine Kinase (SYK)**.

The primary objective is to build a reliable potency-prediction framework that can eventually support molecular screening and activity-guided molecular generation.

The project explores multiple machine-learning algorithms, ensemble learning, uncertainty estimation, and chemical applicability-domain analysis. It is also intended to serve as a foundation for integrating QSAR-derived activity predictions into a future VECTOR+-inspired molecular generation workflow.

### Key objectives

- Develop regression models to predict SYK inhibitor potency.
- Compare different machine-learning algorithms and ensemble strategies.
- Evaluate generalisation to structurally distinct molecules using scaffold-based validation.
- Investigate uncertainty estimation and applicability-domain assessment.
- Explore the feasibility of converting continuous potency predictions into activity labels for downstream molecular generation.

---

## Dataset

The dataset was obtained from the study:

*Integrating QSAR modelling with reinforcement learning for Syk inhibitor discovery* (2025).

The corresponding target is **CHEMBL2599** from ChEMBL.

### Dataset curation

| Property | Value |
|---|---:|
| Original records | 3,176 |
| Curated unique molecules | 3,051 |
| Molecular representation | Canonical SMILES |
| Target | SYK |
| Activity measurement | IC50 |
| Prediction target | pIC50 |

The activity values were converted into pIC50:

\[
pIC_{50}=-\log_{10}(IC_{50}\text{ in M})
\]

For IC50 values expressed in nM:

\[
pIC_{50}=9-\log_{10}(IC_{50}\text{ in nM})
\]

This transformation converts potency into a logarithmic scale, where higher pIC50 values indicate greater potency.

### Molecular representation

Molecules were represented using **Morgan fingerprints** generated through RDKit.

| Parameter | Value |
|---|---|
| Fingerprint type | Morgan |
| Radius | 2 |
| Fingerprint size | 2,048 bits |
| Representation | Binary molecular fingerprints |

Morgan fingerprints encode local molecular environments and provide a fixed-length numerical representation suitable for traditional machine-learning algorithms.

---

## Methodology

The modelling pipeline consists of molecular representation, regression modelling, ensemble learning, and scaffold-aware evaluation.

### 1. Molecular representation

Canonical SMILES were processed using RDKit to generate Morgan fingerprints.

Each molecule is represented by a 2,048-dimensional binary vector.

### 2. Regression models

The following algorithms were investigated:

| Model | Description |
|---|---|
| Random Forest | Bagging-based ensemble of decision trees |
| Extra Trees | Extremely randomized tree ensemble |
| Support Vector Regression | Nonlinear regression using an RBF kernel |
| XGBoost | Gradient-boosted decision trees |
| HistGradientBoosting | Histogram-based gradient boosting |

These models provide different approaches to learning nonlinear relationships between molecular structure and biological activity.

### 3. Stacked ensemble learning

A stacking framework was developed to combine predictions from the five base models.

The procedure uses out-of-fold (OOF) predictions from scaffold-grouped cross-validation to train a Ridge regression meta-model.

Conceptually:

\[
\hat y_{\text{stack}}
=
f_{\text{Ridge}}
\left(
\hat y_{\text{RF}},
\hat y_{\text{ET}},
\hat y_{\text{SVR}},
\hat y_{\text{XGB}},
\hat y_{\text{HGB}}
\right)
\]

The meta-model learns how to combine the predictions of the individual models.

The use of out-of-fold predictions helps prevent the meta-model from learning directly from in-sample predictions.

### 4. Scaffold-aware validation

Random train-test splits can overestimate the generalisation performance of molecular machine-learning models because structurally similar compounds may appear in both training and test sets.

To address this, the project uses **Bemis–Murcko scaffold-grouped cross-validation**.

Molecules sharing the same scaffold are kept within the same partition.

The final evaluation uses two repeats of five-fold scaffold-grouped evaluation, with separate fitting, calibration, and test partitions.

This provides a more challenging assessment of the model's ability to generalise across different molecular scaffolds.

### 5. Uncertainty estimation

Two complementary uncertainty indicators were investigated:

- **Ensemble disagreement:** variation among predictions from the constituent models.
- **Applicability-domain distance:** structural dissimilarity between a query molecule and the training dataset.

For binary Morgan fingerprints, the Tanimoto similarity is calculated as:

\[
T(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

The applicability-domain distance is defined as:

\[
d_{\mathrm{AD}}(x)
=
1-\max_j T(x,x_j)
\]

A higher distance indicates that a molecule is less structurally similar to the training molecules.

These indicators are used to assess prediction reliability, not to guarantee prediction accuracy.

### 6. Prediction intervals

Prediction intervals were constructed using calibration-set absolute errors.

For a predicted potency \(\hat y\), an interval can be expressed as:

\[
[\hat y-q,\hat y+q]
\]

where \(q\) is a calibration-derived error quantile.

Different nominal coverage levels were investigated to understand the trade-off between interval width and empirical coverage.

### 7. Binary activity classification

In addition to regression, binary activity classification was investigated.

Two approaches were explored:

1. **Regression-derived classification:** predicting continuous pIC50 and applying an activity threshold.
2. **Direct classification:** training dedicated Random Forest, XGBoost, and Logistic Regression classifiers.

The full-development dataset median pIC50 threshold was:

\[
pIC_{50}=7.37675
\]

This threshold was used to distinguish the two activity groups in the regression-derived classification analysis.

---

## Results

### Regression performance

The final repeated scaffold-validation evaluation produced the following results for the primary five-model Ridge stacking ensemble.

| Metric | Result |
|---|---:|
| MAE | 0.499 |
| RMSE | 0.669 |
| R² | 0.633 |
| Pearson correlation | 0.796 |
| Spearman correlation | 0.803 |

The RMSE scaffold-bootstrap 95% confidence interval was:

\[
[0.639,\ 0.699]
\]

### Individual model comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.508 | 0.685 | 0.615 |
| Extra Trees | 0.516 | 0.702 | 0.596 |
| SVR | 0.596 | 0.789 | 0.489 |
| XGBoost | 0.507 | 0.677 | 0.623 |
| HistGradientBoosting | 0.512 | 0.684 | 0.616 |
| **Ridge Stacking Ensemble** | **0.499** | **0.669** | **0.633** |

The ensemble achieved the strongest overall performance among the reported models, although its improvement over the strongest individual models was relatively modest.

### Classification performance

The regression-derived classifier and three direct classifiers were also evaluated.

| Approach | Accuracy | ROC-AUC | MCC | F1 |
|---|---:|---:|---:|---:|
| Random Forest Classifier | 83.74% | 0.9021 | 0.6793 | 0.8491 |
| XGBoost Classifier | 84.14% | 0.9106 | 0.6849 | 0.8504 |
| Logistic Regression | 78.24% | 0.8558 | 0.5653 | 0.7921 |
| **Regression-derived classifier** | **84.74%** | **0.9144** | **0.6948** | **0.8507** |

The regression-derived approach performed strongly across the reported classification metrics.

ROC-AUC was calculated using continuous prediction scores, rather than treating calibrated probabilities as the ranking scores.

### Prediction interval calibration

| Nominal coverage | Empirical coverage | Mean interval width |
|---|---:|---:|
| 80% | 81.3% | 1.637 |
| 90% | 91.0% | 2.284 |
| 95% | 95.6% | 2.918 |

The results demonstrate that wider intervals generally provide higher empirical coverage.

These calibration results apply to the evaluated data distribution and should not be interpreted as guaranteed coverage for newly generated molecules.

---

## Current Research Direction: Integration with VECTOR+

An important motivation behind this project is the possibility of integrating QSAR predictions into a VECTOR+-inspired molecular generation framework.

The proposed direction is to investigate whether QSAR-derived activity estimates can provide additional property information during molecular generation.

A potential workflow is:

1. Generate candidate molecular structures.
2. Evaluate candidates using the trained SYK QSAR model.
3. Estimate potency and prediction reliability.
4. Convert suitable predictions into activity labels.
5. Investigate the use of these labels in contrastive representation learning.
6. Evaluate whether the resulting latent representation improves targeted molecular generation.

This is a planned research direction, not an established result of the current QSAR experiments.

In particular, predicted activity labels are not equivalent to experimentally validated activity measurements. Their reliability, potential biases, and effect on downstream generation require independent investigation.

---

## Limitations

Several limitations remain important:

- **Dataset limitations:** the model depends on the quality and consistency of reported IC50 measurements.
- **Assay heterogeneity:** incomplete assay metadata and inconsistent experimental conditions may introduce noise.
- **Structural generalisation:** scaffold-based validation is more demanding than random splitting, but does not replace external validation.
- **Applicability domain:** structurally distant molecules may produce unreliable predictions.
- **Prediction uncertainty:** ensemble disagreement and applicability-domain distance are imperfect indicators of individual prediction error.
- **Generated molecules:** the current model has not been independently validated on a set of newly generated SYK inhibitors.
- **Experimental validation:** computational predictions cannot establish biological activity without experimental testing.

The current results therefore demonstrate internal modelling performance, not experimentally confirmed SYK inhibitor discovery.

---

## Technologies

- **Python**
- **RDKit** – molecular processing and fingerprint generation
- **NumPy** – numerical computation
- **Pandas** – dataset processing
- **Scikit-learn** – machine learning, model evaluation and calibration
- **XGBoost** – gradient boosting
- **SciPy** – scientific computation
- **Matplotlib / Seaborn** – visualisation
- **PyTorch** – available for future deep-learning extensions
- **Hugging Face Transformers** – available for future molecular language-model work

---

## Reproducibility

The curated dataset is identified by the following SHA-256 checksum:

```text
541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25
```

This can be used to verify that the underlying dataset has not changed.

For reproducible evaluation, the project uses scaffold-aware data partitioning and fixed evaluation seeds (140 and 141).

Exact preprocessing, model configurations, data splits, and evaluation scripts should be retained alongside future repository releases.

---

## Future Work

Potential extensions include:

- External validation using independent SYK inhibitor datasets.
- Improved assay-level data curation and standardisation.
- More robust uncertainty estimation and applicability-domain thresholds.
- Comparison with graph neural networks and molecular language models.
- Investigation of continuous versus binary activity information.
- Integration with contrastive learning and GMM-based latent-space sampling.
- Evaluation of generated candidates using docking and molecular dynamics.
- Experimental validation of promising candidate molecules.

---

## Disclaimer

This repository is intended for academic and research purposes. Model predictions are computational estimates and should not be interpreted as experimentally verified biological activity or as evidence of therapeutic efficacy.

**The central research question:** Can a carefully validated QSAR model provide useful activity information for targeted molecular generation, and under what conditions can that information be trusted?


