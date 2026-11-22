# Limitations

- The IC50 unit evidence comes from the saved processing notebook output (3,485 nM and 28 pIC50) and its conversion code; the referenced `raw_data.csv` is unavailable for row-level replay.
- The upstream IQR step produced 3,176 records, but the 23 removed rows cannot be reconstructed from the missing raw source. This may affect the activity range and downstream generalization.
- Assay identifiers, assay conditions, source citations, censoring and replicate provenance are unavailable. Measurements may not be assay-harmonized.
- The duplicate-resolution rule is documented, but discordant groups cannot be checked against raw assay records. The minimum-IC50 branch is potency-favoring.
- Six disconnected structures are retained; mixture/salt/hydrate interpretation is uncertain. Four InChIKey-equivalent tautomer pairs also remain separate under the expected hash, creating small duplicate weighting; their labels match and scaffold splitting keeps each pair together.
- 364 structures have potential unassigned stereocenters. Source stereochemistry cannot establish whether these are underspecified or erroneous.
- The exact VECTOR+ source implementation is absent. The binary procedure follows the user's stated median convention, not a verified code reproduction.
- Earlier model/test results on this same table were inspected. Repeated scaffold CV and calibration are internal evidence, not external validation.
- No assessment on generated molecules demonstrates transfer of QSAR errors, calibration, AD scores or activity-label precision to VECTOR+ outputs.
- Any final model is a research-development artifact, not an experimentally validated potency predictor. Independent, assay-matched external Syk data are required for strong publication claims.
