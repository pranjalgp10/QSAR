# AGENTS.md — QSAR-Guided Data Augmentation for VECTOR+

## 1. Project identity and primary objective

This repository contains the implementation of a research project investigating whether QSAR-guided molecular data augmentation can improve a VECTOR+-style molecular generation pipeline.

**Primary dataset:** Syk inhibitor dataset, containing approximately 3,176 curated molecules with experimentally measured IC50 values converted to pIC50.

**Primary research objective:** Investigate whether generated molecules can be assigned QSAR-predicted activity labels, filtered for reliability, incorporated into a second stage of contrastive learning, and used to improve subsequent molecular generation.

The central hypothesis is that carefully selected QSAR-pseudolabelled molecules can enrich the molecular representation learned by the original model and improve the quality of generated molecules.

The project is primarily about improving generated-molecule quality. It is not simply an exercise in maximising QSAR prediction accuracy or improving latent-space representations in isolation.

The QSAR predictor is an important enabling component. Its independent generalisation performance and the reliability of its pseudolabels must be assessed.

## 2. Primary dataset: Syk inhibitors

Use the Syk inhibitor dataset as the primary dataset for the initial implementation.

Relevant background:

- Approximately 3,176 curated molecules.
- Experimental IC50 measurements converted into continuous pIC50 values.
- Biological target: Syk (spleen tyrosine kinase).
- The associated published study identifies PDB structure 3FQS, containing Syk with R406, as a docking reference.

The dataset will support two distinct modelling components:

1. QSAR property prediction.
2. VECTOR+-style molecular representation learning and generation.

An important experimental-design question is whether these two components should use separate or overlapping experimental molecules.

Do not automatically assume that the dataset must be divided 50:50 between QSAR and VECTOR+.

Investigate both:

- A disjoint-data design, in which QSAR and VECTOR+ initially train on separate experimental molecules.
- A shared-data design, in which both components can use the full experimental dataset, subject to appropriate separation of QSAR training and evaluation data.

The rationale, benefits, limitations and consequences of each design must be documented before selecting the primary experimental configuration.

The final allocation must consider dataset size, scaffold distribution, activity distribution, model performance and the scientific validity of the pseudolabelling experiment.

## 3. Continuous and binary activity labels

The activity data are initially continuous pIC50 values.

The QSAR property-prediction model must initially be trained as a regression model using continuous pIC50 values. Do not discard the continuous information during initial data processing.

The continuous activity values will eventually be converted into binary activity labels for the relevant contrastive-learning and molecular augmentation workflow.

This conversion is a deliberate methodological step and must be implemented explicitly.

Important requirements:

- Do not choose an arbitrary binary activity threshold without documenting and justifying it.
- Investigate the pIC50 distribution before deciding on a threshold.
- Record the threshold, its scientific rationale and the resulting class balance.
- Keep the continuous pIC50 predictions available even after binary labels are created.
- Make the conversion reproducible and configurable.
- Do not use the untouched QSAR test set to select the activity threshold.
- Apply the same pre-established threshold consistently to experimentally measured and pseudolabelled activities.
- Investigate the effects of uncertain predictions close to the binary decision boundary.
- Do not silently change thresholds between experiments.

The precise threshold and whether additional confidence criteria are needed remain research decisions. Ask before fixing them.

## 4. Proposed research pipeline

The intended workflow is:

### Stage 1 — Dataset preparation

1. Load and inspect the Syk inhibitor dataset.
2. Validate SMILES and remove invalid structures where justified.
3. Standardise molecular representations.
4. Inspect duplicate compounds and repeated activity measurements.
5. Verify activity units and convert IC50 to pIC50.
6. Examine the activity distribution and chemical-space coverage.
7. Create documented data partitions.

### Stage 2 — QSAR property prediction

1. Train QSAR regression models on continuous pIC50 values.
2. Investigate molecular representations such as Morgan/ECFP fingerprints.
3. Establish classical machine-learning baselines:
   - Random Forest
   - XGBoost
   - Support Vector Regression
4. Investigate stacking if validation results justify it.
5. Use scaffold-aware validation to assess generalisation.
6. Evaluate the final predictor on an untouched experimental test set.

The untouched test set is primarily intended to evaluate the QSAR property predictor's ability to generalise to unseen experimental molecules.

It must remain independent of model development and augmentation.

Do not use this test set to:
- Train the QSAR model.
- Train the VECTOR+ model.
- Generate pseudolabels.
- Select pseudolabel confidence thresholds.
- Select the binary activity threshold.
- Tune the augmentation procedure.
- Choose the best model configuration.
- Make repeated decisions based on test performance.

Use training and validation data for development and model selection. Keep the test set for the final independent assessment of the property predictor.

The exact split proportions and scaffold-assignment procedure must be documented and justified before implementation.

### Stage 3 — VECTOR+ baseline

1. Reproduce the original VECTOR+ methodology as accurately as possible.
2. Establish a baseline using the Syk dataset and the appropriate activity-label representation.
3. Record the original model architecture, preprocessing, training configuration and random seeds.
4. Generate an initial population of molecules.
5. Evaluate the baseline generation results.

Do not claim an exact reproduction until the implementation has been checked against the original paper and supplementary material.

### Stage 4 — QSAR-guided pseudolabelling

1. Generate molecules using the baseline VECTOR+-style model.
2. Check molecular validity and remove unsuitable structures.
3. Predict continuous pIC50 values using the trained QSAR model.
4. Assess prediction reliability using an appropriate confidence and applicability-domain strategy.
5. Convert accepted predictions into binary activity labels using the pre-established threshold.
6. Record why each generated molecule was accepted or rejected.

Do not treat predicted activity as experimentally measured activity.

Do not assume that model confidence alone guarantees correctness.

Investigate whether generated molecules lie within the chemical applicability domain of the QSAR training data.

### Stage 5 — Second-stage contrastive learning

1. Combine the original experimentally labelled training molecules with accepted generated molecules carrying pseudolabels.
2. Preserve the distinction between experimental and predicted labels in the data.
3. Perform the second stage of contrastive learning.
4. Retrain or refit the relevant generative components, including the GMM and decoder, as required by the original architecture.
5. Generate a new molecular population.
6. Save the complete configuration and checkpoints.

Do not silently change the original VECTOR+ architecture or training procedure.

### Stage 6 — Final evaluation

Compare the original and augmented generation pipelines using matched evaluation conditions.

Relevant metrics include:

- Molecular validity
- Chemical plausibility
- Novelty
- Diversity
- QED
- Synthetic accessibility
- QSAR-predicted activity
- Molecular docking scores
- Molecular dynamics analysis for selected candidates

Docking and molecular dynamics are computational evaluation tools. They do not establish experimentally confirmed binding or potency.

Use equivalent generation budgets and evaluation protocols for all comparison groups.

MD should be performed on a carefully selected subset of representative candidates rather than automatically on every generated molecule.

## 5. Experimental controls

At minimum, investigate these experimental groups:

**A — Original VECTOR+ baseline**

The original model trained without generated pseudolabelled augmentation.

**B — Additional contrastive training control**

Additional contrastive training without introducing generated pseudolabelled molecules. This helps distinguish the effect of augmentation from the effect of extra training.

**C — Main augmentation experiment**

QSAR-predicted activity labels, reliability filtering, binary label conversion and second-stage contrastive learning.

**D — No-confidence-filtering ablation**

**E — Dataset allocation ablation**

Compare the effect of using disjoint versus shared experimental training data for QSAR and VECTOR+.

The purpose is to investigate whether the independence of the QSAR predictor's training data affects the usefulness of pseudolabel-guided augmentation.

Where possible, keep the total experimental data, model configurations, generation budgets and evaluation protocols controlled.

Report differences in QSAR performance, pseudolabel acceptance, chemical diversity and final generated-molecule quality.

Do not interpret a difference between these designs as an augmentation effect unless the appropriate non-augmentation controls are also included.

QSAR-pseudolabelled augmentation without the confidence/applicability-domain filtering strategy.

These are proposed controls. Evaluate whether additional ablations are necessary, such as alternative thresholds, different pseudolabel proportions or different confidence criteria.

Use matched generation and evaluation budgets wherever possible.

## 6. Data integrity, model separation and leakage prevention

Maintain a strict distinction between:

1. Experimental QSAR training data.
2. Experimental QSAR validation data.
3. Untouched experimental QSAR test data.
4. VECTOR+ training molecules.
5. Generated molecules.
6. Generated molecules with QSAR-predicted continuous activity.
7. Generated molecules with accepted binary pseudolabels.

### QSAR test-set isolation

The untouched QSAR test set is primarily intended to independently evaluate the QSAR property's predictive generalisation on unseen experimental molecules.

Its experimental pIC50 values must never be used for:

- QSAR training.
- Hyperparameter optimisation.
- Pseudolabelling.
- Activity threshold selection.
- Confidence threshold selection.
- Applicability-domain threshold tuning.
- Augmentation decisions.
- Model selection.

Its molecular structures may be used for structure-only or unsupervised VECTOR+ learning, provided no experimental activity information or labels derived from those measurements are exposed.

If test-set structures are used in VECTOR+ training, report this explicitly. Do not claim that the resulting generator has never encountered those molecular structures.

### QSAR and VECTOR+ training separation

It is not mandatory for QSAR and VECTOR+ to use completely disjoint experimental training molecules.

However, their overlap can affect how independently the QSAR predictor evaluates molecules generated by VECTOR+ and how we interpret the pseudolabelling results.

Investigate two experimental designs:

**Design A — Disjoint experimental training pools**

- Divide the eligible experimental dataset into separate QSAR and VECTOR+ training pools.
- Train QSAR using the QSAR pool.
- Train the initial VECTOR+ model using the VECTOR+ pool.
- Generate molecules using VECTOR+.
- Evaluate and pseudolabel them using QSAR.

This design tests whether a predictor trained on a separate set of experimental molecules can provide useful activity information to VECTOR+.

**Design B — Shared experimental training data**

- Allow QSAR and VECTOR+ to use overlapping or shared experimental training molecules.
- Maintain an independently held-out QSAR test set.
- Ensure that the test-set experimental activity values never enter VECTOR+ training or augmentation.

This design allows both components to benefit from more experimental data but may reduce the independence of the QSAR signal used for augmentation.

### Experimental comparison

Where feasible, compare both designs under controlled conditions.

Investigate:

- Whether disjoint training reduces QSAR predictive performance.
- Whether a larger VECTOR+ training pool improves generation.
- Whether pseudolabelling provides a measurable improvement in either design.
- Whether generated molecules resemble the QSAR training data too closely.
- Whether apparent improvements reflect genuine generation improvements or reinforcement of the QSAR model's learned biases.

Do not assume that a disjoint 50:50 split is optimal. Determine a defensible allocation based on scaffold distribution, activity distribution and model performance.

### General leakage prevention

Use scaffold-aware splitting wherever feasible.

Never overwrite experimental measurements with predicted values.

Preserve the provenance of every molecular activity label.

For stacking, use out-of-fold predictions to train the meta-model.

Document all dataset overlaps between QSAR, VECTOR+ and evaluation partitions.

The final experimental design must make clear which molecular structures and which activity labels were accessible to each model during every training stage.

## 7. Scientific risks to investigate

Pay particular attention to:

- QSAR extrapolation beyond its applicability domain.
- Confirmation bias from using a QSAR model to label molecules generated by a related model.
- Reinforcement of the original training-data distribution.
- Loss of information when continuous pIC50 values are converted into binary labels.
- Class imbalance after binary conversion.
- Pseudolabel noise and incorrect high-confidence predictions.
- Scaffold leakage and overly optimistic random-split performance.
- Changes in chemical diversity after augmentation.
- Whether improved predicted activity corresponds to improved independent computational evaluation.
- Whether improvements are caused by augmentation or simply additional training.

Do not assume that a more active-looking latent space necessarily produces better molecules.

## 8. Clean repository and file-structure requirements

Maintain a clean, modular and predictable repository structure throughout development.

Use the existing directory structure unless there is a strong reason to change it.

Recommended organisation:

- `docs/`: research context, methodology, experimental design and decisions.
- `data/raw/`: original, unmodified source datasets.
- `data/processed/`: cleaned and standardised datasets.
- `data/external/`: external reference data.
- `src/data/`: data loading, validation and preprocessing.
- `src/qsar/`: molecular representations, QSAR training, prediction and validation.
- `src/vector_plus/`: VECTOR+ implementation and training.
- `src/augmentation/`: pseudolabelling, filtering and second-stage learning.
- `src/evaluation/`: metrics, docking and molecular dynamics evaluation.
- `src/utils/`: shared utilities.
- `configs/`: experiment configurations.
- `experiments/`: experiment-specific scripts and run definitions.
- `notebooks/`: exploratory analysis and visualisation only.
- `models/`: model checkpoints and saved model artefacts.
- `results/`: metrics, plots and experiment outputs.
- `tests/`: automated tests.

Rules:

- Do not put all project code in one large Python file.
- Keep notebooks for exploration, not as the only location of essential pipeline logic.
- Do not duplicate implementations across notebooks and source files.
- Do not save generated datasets, model checkpoints or temporary files in the repository root.
- Never modify raw source data.
- Use descriptive file and directory names.
- Keep configuration separate from implementation.
- Make output paths explicit.
- Avoid unnecessary dependencies and redundant scripts.
- Remove obsolete files only after checking that they are not needed.
- Update documentation when the repository structure changes.
- Add a README explaining setup, execution and experiment reproduction.
- Use Git-friendly practices and do not commit large data files or secrets.

Before creating a new file, inspect the repository and decide where it belongs.

## 9. Reproducibility and implementation standards

Use Python and modular, reusable code.

For every experiment, record:
- Random seed.
- Dataset version and source.
- Data split identifiers.
- Molecular representation settings.
- Model hyperparameters.
- Training configuration.
- Pseudolabelling threshold.
- Binary activity threshold.
- Confidence and applicability-domain criteria.
- Number of molecules generated.
- Number accepted and rejected.
- Evaluation configuration.
- Software and dependency versions.

Save model checkpoints, prediction files, evaluation metrics and configuration files.

Add tests for important data transformations and scientific calculations.

Never invent data, experimental results, citations or model performance.

Clearly label all values as experimental, predicted, or computationally estimated.

## 10. Working instructions for Codex

Act as a research engineering collaborator.

Before substantial implementation:

1. Read the relevant research documents.
2. Inspect the existing repository.
3. Explain the proposed approach.
4. Identify assumptions and unresolved scientific decisions.
5. Flag potential statistical or chemical problems.
6. Propose a modular implementation plan.

Do not make major scientific decisions without discussing them.

Do not silently change the research hypothesis, model architecture, data partitions or evaluation protocol.

When a task is completed:
- Summarise the changes.
- List files created or modified.
- Explain how to run the implementation.
- Report tests performed and their results.
- Identify limitations and unresolved decisions.
- Update relevant documentation.

Prioritise scientific correctness, clean architecture, reproducibility and defensible experiments over speed or unnecessary complexity.