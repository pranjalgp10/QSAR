# Scientific Context: VECTOR+ and Syk Inhibitor QSAR

## 1. Purpose of this document

This document provides scientific background for the QSAR-guided data augmentation project.

It covers:

1. The VECTOR+ molecular-generation methodology as discussed during project planning.
2. The published Syk inhibitor QSAR and reinforcement-learning study.
3. How the two methodologies relate.
4. The proposed extension being investigated.
5. Scientific limitations and details that must be verified against the original publications.

This document is not a replacement for the original papers, supplementary information or source code. When implementing either methodology, consult the original sources and verify exact architectural and mathematical details.

---

# PART I — VECTOR+ METHODOLOGY

## 2. General research context

VECTOR+ is the methodological foundation for the molecular representation and generation component of this project.

The central idea relevant to our research is to learn a molecular representation using contrastive learning and then use the learned representation as the basis for molecular generation.

The project discussions have focused on the following conceptual components:

- Molecular representation learning.
- Contrastive learning.
- Latent-space organisation.
- Gaussian Mixture Model (GMM) fitting.
- Sampling from the learned latent distribution.
- Molecular decoding and generation.
- Evaluation of generated molecular structures and properties.

The exact VECTOR+ architecture, loss functions, molecular representation, positive/negative pair construction, sampling procedure, decoder design and training schedule must be verified from the original VECTOR+ publication and supplementary information before implementation.

Do not invent or assume these details.

## 3. Contrastive learning: relevant concepts

Contrastive learning trains a representation model by learning relationships between examples.

A molecular encoder maps a molecule to a latent representation:

z = f_theta(x)

where:

- x is the molecular input.
- f_theta is the molecular encoder.
- z is the learned latent representation.
- theta represents the trainable model parameters.

A contrastive objective generally encourages representations of related or positive examples to be closer, while representations of negative examples are separated.

The precise definition of positive and negative molecular pairs, the similarity function, temperature parameter and loss function must follow the original VECTOR+ methodology.

For this project, the important research question is whether adding reliable QSAR-pseudolabelled generated molecules and performing another stage of contrastive learning changes the latent representation in a way that improves subsequent molecular generation.

Do not assume that a better contrastive-learning loss necessarily means better generated molecules.

## 4. Latent-space modelling with a Gaussian Mixture Model

A Gaussian Mixture Model represents a probability distribution as a weighted combination of Gaussian components.

A general GMM has the form:

p(z) = sum_k pi_k N(z | mu_k, Sigma_k)

where:

- z is a latent molecular representation.
- k indexes the mixture components.
- pi_k is the mixture weight for component k.
- mu_k is the mean of component k.
- Sigma_k is its covariance matrix.
- N denotes a multivariate Gaussian distribution.

The GMM can model different regions or clusters in the learned latent space.

Sampling from the fitted distribution provides latent vectors that can subsequently be passed through the relevant decoder.

The exact number of components, covariance structure, fitting procedure, component-selection criteria and sampling strategy must be obtained from the original VECTOR+ implementation.

## 5. Molecular generation

At a conceptual level, the generation process is:

Molecule -> Encoder -> Latent representation -> Latent distribution modelling -> Sampling -> Decoder -> Generated molecule

The generated molecule must be assessed for:
- Structural validity.
- Chemical plausibility.
- Similarity to known molecules.
- Novelty.
- Diversity.
- Predicted molecular properties.

Generation and property prediction are separate tasks. A molecule that can be generated successfully is not automatically chemically useful or biologically active.

## 6. How the proposed research extends the methodology

The proposed research introduces an iterative data-augmentation stage.

The conceptual pipeline is:

1. Train the initial VECTOR+-style model using the original experimental training data.
2. Generate an initial population of molecules.
3. Use an independently trained QSAR property predictor to estimate their continuous pIC50 values.
4. Assess the reliability of predictions.
5. Filter molecules using confidence and applicability-domain criteria.
6. Convert accepted predicted activities into binary labels.
7. Combine these generated molecules and pseudolabels with the original experimental training data.
8. Perform a second stage of contrastive learning.
9. Refit the latent-space distribution model.
10. Retrain or update the decoder as required.
11. Generate a new molecular population.
12. Compare the new population against the original baseline.

The purpose is to investigate whether the new pseudolabelled examples improve the learned representation and, ultimately, generated-molecule quality.

This is a proposed extension. It must not be described as an existing feature of VECTOR+ unless the original paper explicitly establishes that.

## 7. Important distinction: continuous QSAR values and binary contrastive labels

The QSAR model is initially a regression model trained on continuous pIC50 values.

The augmentation workflow will eventually convert continuous activity values into binary activity labels.

The two representations serve different purposes:

- Continuous pIC50: preserves potency information and is the target for QSAR regression.
- Binary activity label: provides a categorical activity representation for the relevant contrastive-learning workflow.

The binary threshold must be justified, documented and fixed using training/validation information rather than the untouched test set.

The continuous predictions should be retained even after binary labels are assigned.

---

# PART II — THE SYK INHIBITOR QSAR STUDY

## 8. Published study

The main QSAR reference is:

"Integrating QSAR modelling with reinforcement learning for Syk inhibitor discovery"

Published in the Journal of Cheminformatics in 2025.

DOI: 10.1186/s13321-025-00998-2

The study combines QSAR modelling with reinforcement learning for the discovery of Syk inhibitors.

It is relevant to this project because it demonstrates a workflow in which a molecular property predictor is trained on experimentally measured activity data and subsequently used to evaluate or guide the generation of molecules.

The study is a methodological reference. Our proposed iterative contrastive-learning augmentation approach is not identical to its reinforcement-learning procedure.

## 9. Biological target and activity data

The target is Syk (spleen tyrosine kinase).

The study initially collected approximately 3,513 Syk inhibitors and retained approximately 3,176 molecules after curation.

The activity endpoint is IC50.

IC50 is the concentration of a compound required to produce 50% inhibition under a specified experimental assay.

The study converts IC50 into pIC50:

pIC50 = -log10(IC50 in M)

For example:

- IC50 = 1 micromolar = 10^-6 M.
- pIC50 = 6.

Higher pIC50 indicates greater potency.

Important: IC50 measurements can depend on assay conditions. They should not automatically be treated as directly interchangeable across different experimental assays.

Inspect the actual dataset and its source information to understand the assay provenance and measurement consistency.

## 10. Data curation described in the study

The published workflow includes molecular data processing and treatment of repeated activity measurements.

The methodology discussed from the paper and its associated notebooks includes:

- Collection of Syk inhibitor molecules.
- Molecular standardisation and data cleaning.
- Conversion of IC50 to pIC50.
- Handling of repeated measurements.
- Selection of molecular representations for QSAR modelling.

The described duplicate-measurement strategy involved comparing repeated IC50 measurements against their median. Measurements within a 10% range were averaged; when variation exceeded the stated criterion, the lowest IC50 was retained.

This is a potentially important methodological choice because retaining the lowest IC50 can bias the resulting data towards greater apparent potency.

Do not reproduce this rule blindly. Inspect the actual measurements and consider whether a more statistically defensible aggregation strategy is appropriate.

Record any deviation from the paper.

## 11. Molecular representations

The study investigated multiple molecular representations, including:

- ECFP/Morgan fingerprints.
- MACCS fingerprints.
- PubChem fingerprints.
- MAP4.
- Other fingerprint representations described in the associated notebooks.

ECFP was selected as a strong representation for the QSAR modelling workflow.

A molecular fingerprint converts a molecular structure into a numerical representation that can be used by machine-learning algorithms.

For binary fingerprints:

x in {0,1}^d

where d is the fingerprint length and each position indicates the presence or absence of a particular molecular feature.

The exact ECFP radius, bit length, hashing configuration and preprocessing settings must be checked in the original notebooks. Do not assume a particular fingerprint configuration without verifying the code.

## 12. QSAR models investigated

The published study compared several regression algorithms, including:

- Random Forest.
- Histogram-based Gradient Boosting.
- XGBoost.
- Support Vector Regression.

The study used hyperparameter optimisation and investigated a stacking ensemble.

The stacking model combined predictions from base learners, with a Linear Regression meta-model.

Conceptually:

1. Train multiple base regressors.
2. Obtain out-of-fold predictions from those regressors.
3. Use these predictions as features for a meta-regressor.
4. Train the meta-regressor to combine the base predictions.

The exact implementation must use out-of-fold predictions to avoid information leakage.

## 13. Validation and reported results

The published workflow used a random 80/20 train/test split with random_state=140 and five-fold shuffled cross-validation on the training partition.

The study reported representative individual-model test results approximately as follows:

| Model | MSE | R² |
|---|---:|---:|
| Random Forest | 0.3482 | 0.7293 |
| Histogram Gradient Boosting | 0.3789 | 0.7055 |
| XGBoost | 0.3902 | 0.6967 |
| SVR | 0.3682 | 0.7138 |

The final stacking model was reported to achieve approximately R² = 0.78 and MSE = 0.27.

These are published study results, not expected performance guarantees for our implementation.

Important limitation:

A random split does not establish strong generalisation to unseen molecular scaffolds.

Our project should investigate scaffold-aware validation and should not assume that the published random-split results demonstrate scaffold generalisation.

The reported metrics must be checked against the paper and its supplementary material before being used in a publication or formal comparison.

## 14. QSAR prediction in the published study

The trained QSAR predictor was used to estimate activity for generated molecules.

The study also used predicted activity in its reinforcement-learning workflow.

This is relevant because a learned molecular property predictor can provide a computational signal for guiding molecular generation.

However, predicted activity is not equivalent to experimentally measured activity.

A QSAR model can produce unreliable predictions when a generated molecule is structurally dissimilar to the molecules on which it was trained.

This makes applicability-domain analysis and uncertainty assessment particularly important in our project.

## 15. Associated computational resources

The associated public repository and notebooks discussed during project planning include material covering:

- Data processing.
- Molecular representations.
- QSAR model training.
- Prediction.
- Generation and property analysis.

Notebooks previously identified include:

- `Data_processing.ipynb`
- `Molecules_representation.ipynb`
- `Predictive_model.ipynb`
- `IC50_predictor.ipynb`

Verify the current repository contents and code before relying on these names or assuming that pretrained model artefacts are available.

The existence of an interactive prediction notebook does not establish that downloadable, reusable pretrained model weights are provided.

## 16. Docking reference

The Syk study identifies PDB structure 3FQS, containing Syk with R406, as a docking structure.

This provides a possible starting point for the project's docking workflow.

Before using the structure, inspect:
- Protein chain and biological assembly.
- Bound ligand identity.
- Missing residues and structural quality.
- Relevant cofactors and waters.
- Protein preparation requirements.
- Ligand preparation requirements.
- Binding-site definition.
- Protonation states.
- Whether the selected structure is appropriate for the compounds being evaluated.

Do not assume that a docking protocol is valid merely because a structure was used in the reference paper.

Where feasible, validate the docking protocol by redocking the crystallographic ligand and comparing the predicted pose with the experimental pose.

Docking scores are approximate scoring-function outputs, not experimentally measured binding affinities.

---

# PART III — RELATIONSHIP BETWEEN THE TWO METHODS

## 17. Similarities

Both approaches use molecular property prediction to support molecular discovery.

Both involve generated molecules and computational estimates of molecular activity.

Both recognise that molecular generation needs a way to assess whether generated candidates have desirable properties.

## 18. Differences

The Syk study uses QSAR predictions as part of a reinforcement-learning-based molecular discovery workflow.

Our proposed project investigates a different mechanism:

- Generate molecules with a VECTOR+-style model.
- Use a QSAR model to assign predicted activity.
- Filter the generated molecules based on prediction reliability and applicability domain.
- Convert accepted continuous predictions into binary pseudolabels.
- Add these examples to the original training data.
- Perform another stage of contrastive learning.
- Refit the generative components.
- Test whether the updated representation and generator produce better molecules.

The central methodological distinction is the use of QSAR-guided pseudolabelled data to update the contrastive-learning representation and subsequent generation process.

Do not describe this as equivalent to reinforcement learning. The training objective and mechanism are different.

## 19. Important potential circularity

The proposed method has a potential feedback-loop problem:

1. The generator produces molecules from a learned chemical distribution.
2. The QSAR model predicts activity for those molecules.
3. High-predicted-activity molecules are selected.
4. The selected molecules are used to update the representation.
5. The updated generator may produce more molecules resembling the selected region.
6. The same QSAR model may then assign high activity to similar generated molecules.

This can reinforce the QSAR model's existing biases rather than discover genuinely better compounds.

Mitigations to investigate:

- Applicability-domain filtering.
- Prediction uncertainty or ensemble disagreement.
- Independent validation.
- Scaffold-aware evaluation.
- Chemical diversity monitoring.
- Controls without pseudolabelled molecules.
- Controls without confidence filtering.
- Evaluation using methods independent of the QSAR predictor.
- Experimental validation if feasible.

Ensemble agreement is not automatically calibrated uncertainty. It should be evaluated rather than assumed to guarantee correctness.

## 20. Main research questions

The project should investigate:

1. Does QSAR-guided pseudolabelling improve generated-molecule quality?
2. Does a second stage of contrastive learning improve the representation in a useful way?
3. Does the augmented model generate molecules with improved predicted activity?
4. Does the improvement remain when evaluated using docking and other metrics that are not simply the same QSAR predictor?
5. What is the effect of confidence and applicability-domain filtering?
6. Does augmentation reduce chemical diversity?
7. Does the model simply reproduce QSAR biases?
8. How sensitive are results to the binary activity threshold?
9. How does the amount of pseudolabelled data affect the results?
10. Are observed improvements reproducible across random seeds?

---

# PART IV — EVALUATION AND INTERPRETATION

## 21. QSAR predictor evaluation

The primary role of the untouched experimental test set is to assess the generalisation of the continuous pIC50 property predictor.

Use suitable regression metrics:

- MAE.
- RMSE.
- R².
- Residual analysis.
- Scaffold-aware performance.
- Applicability-domain coverage.

Do not use the test set for augmentation or model selection.

## 22. Generated-molecule evaluation

Generated-molecule evaluation is a separate part of the project.

Compare baseline and augmented populations using:

- Validity.
- Uniqueness.
- Novelty.
- Diversity.
- Chemical plausibility.
- QED.
- Synthetic accessibility.
- Predicted activity.
- Docking.
- Molecular dynamics for selected candidates.

The evaluation should distinguish predictor performance from generator performance.

A higher predicted pIC50 among generated molecules is not sufficient evidence that the augmented generator has improved.

## 23. Docking and molecular dynamics

Docking can provide an independent computational assessment relative to the QSAR model, provided the docking protocol is properly validated.

Molecular dynamics can be used to investigate the stability and behaviour of selected protein-ligand complexes.

Neither docking nor MD establishes experimental potency.

MD should be applied to a carefully selected, chemically diverse subset of candidates, including suitable reference compounds where possible.

Do not select candidates only because they have the best docking score. This can introduce selection bias.

## 24. Interpretation standards

Always distinguish:

- Experimentally measured pIC50.
- QSAR-predicted continuous pIC50.
- Binary activity pseudolabel.
- Docking score.
- Molecular dynamics observations.
- Hypotheses about binding or activity.

Never present predicted activity as experimental activity.

Never claim that a molecule is experimentally validated based solely on QSAR, docking or MD.

## 25. References and verification requirements

Primary reference:

"Integrating QSAR modelling with reinforcement learning for Syk inhibitor discovery", Journal of Cheminformatics (2025), DOI: 10.1186/s13321-025-00998-2.

Additional primary reference:

The original VECTOR+ publication and its supplementary information.

Codex must inspect the original VECTOR+ source material before implementing its exact architecture or loss functions.

For both references:
- Preserve correct attribution.
- Verify reported numbers against the original source.
- Record which implementation details come from the paper and which are project adaptations.
- Do not assume the licence of a paper also grants an open-source licence to its associated code or pretrained models.
- Check code and model licences before reusing or redistributing them.

This document should be updated when the original papers, supplementary materials or source code establish more precise implementation details.