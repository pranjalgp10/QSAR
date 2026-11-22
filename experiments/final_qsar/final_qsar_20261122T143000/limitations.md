# Limitations and blockers

## Critical data blockers

1. The source table has only `Smiles`, `IC50`, and `pIC50`. Units, assay identifiers, source citations, censoring, and replicate metadata are absent. The nM conversion is internally exact but cannot establish that the original measurements were nM.
2. The `raw_data.csv` referenced by the prior processing record and the upstream processing notebook are not in the repository. The input appears to have already undergone IQR filtering; excluded raw records cannot be inspected or restored.
3. Six structures contain multiple disconnected components (salts, waters, or other fragments). They are listed and retained unchanged because source chemistry context is unavailable.
4. Four pairs of distinct canonical SMILES share an InChIKey and identical supplied IC50 values, consistent with tautomeric duplicates. The hash-pinned curated dataset keeps both rows; merging would change the dataset and molecule weighting, so a policy decision is required.
5. Duplicate IC50 aggregation followed a potency-favoring Syk-paper adaptation. There are 117 canonical duplicate groups, including discordant groups; no assay metadata exist to validate those resolutions.
6. The exact VECTOR+ implementation and source activity-label code are absent. The requested median split can be implemented from the attached specification, but original-source parity cannot be verified.

## Evaluation limitations

Earlier models, validation/test results, and all three follow-up experiment outputs have already been inspected. Future scaffold CV on this dataset is internal validation, not an untouched external evaluation. No available result establishes calibration on generated molecules, model performance under scaffold distribution shift, or experimentally confirmed potency.

## Consequence

The run failed closed before training, median-threshold computation, classification, calibration, Y-scrambling, learning curves, or serialization. There is no final model or pseudolabel acceptance rule.
