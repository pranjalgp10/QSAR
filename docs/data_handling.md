# Syk data handling and duplicate-resolution record

## Input and constraints

`processed_df.csv` is the preserved input (3,176 rows; only `Smiles`, `IC50`, `pIC50`). The Syk Data_processing notebook explains its historical processing: IC50 normalized to nM, duplicate literal-SMILES handling, and IQR deletion of pIC50 observations. The notebook reports that discrepant duplicate values were resolved by keeping minimum IC50 and that IQR filtering produced the processed file. The raw `raw_data.csv` referenced by that notebook was not supplied. Consequently original assay sources, units per record, censored observations, and the removed IQR observations cannot be checked or restored.

## Implemented duplicate rule

The current working table is grouped by canonical **isomeric** SMILES, which captures equivalent text encodings without merging stereoisomers. For 117 duplicate groups:

- Exact repeated IC50 values collapse to one structure/value, with all input row positions retained in the map.
- For nonidentical values whose full IC50 range is at most 10% of the group median, the mean IC50 is retained, following the notebook's rule.
- Otherwise the lowest IC50 is retained, also following that rule. This intentionally reproduces the paper's choice but is potency-favoring and may bias pIC50 upward; sensitivity analyses should compare it to excluding discordant groups or a robust log-scale aggregate when record provenance becomes available.

IC50 is aggregated before log conversion as in the source notebook; pIC50 is recalculated from resolved IC50 using `9-log10(IC50[nM])`. Source units are internally consistent with the provided pIC50 column but are not independently verified. No structures were removed for IQR criteria in this curation step; the input had already been filtered upstream.

## Outputs

- `data/processed/syk_qsar_curated.csv`: one row per canonical structure plus scaffold and source-row map.
- `data/processed/syk_duplicate_resolution.csv`: group counts, measurement ranges, resolved values, and rule applied.
- `data/processed/syk_curation_report.json`: counts, activity distribution, assumptions, and limitations.

These outputs are derived data; `processed_df.csv` remains unchanged. The benchmark split keeps scaffolds together. Since prior modeling results from this CSV have been examined, a future test split from this source is an internal holdout rather than external validation.
