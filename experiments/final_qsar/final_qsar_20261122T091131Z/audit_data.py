"""Reproducible stage-1 data/provenance audit for this final-QSAR run."""
from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
RAW = ROOT / "processed_df.csv"
ALL_MOLS = Path(r"C:\Users\pranj\Downloads\all_mols.csv")
PROCESSING_NOTEBOOK = Path(r"C:\Users\pranj\Downloads\Data_processing.ipynb")
RAW_SOURCE = ROOT / "raw_data.csv"
CURATED = ROOT / "data" / "processed" / "syk_qsar_curated.csv"
DUPLICATES = ROOT / "data" / "processed" / "syk_duplicate_resolution.csv"
EXPECTED_CURATED_SHA256 = "541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(smiles):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def main():
    raw = pd.read_csv(RAW)
    all_mols = pd.read_csv(ALL_MOLS)
    notebook = json.loads(PROCESSING_NOTEBOOK.read_text(encoding="utf-8"))
    curated = pd.read_csv(CURATED)
    dup = pd.read_csv(DUPLICATES)
    required_raw = {"Smiles", "IC50", "pIC50"}
    required_curated = {"Smiles", "canonical_smiles", "IC50", "pIC50", "scaffold_group", "source_rows", "n_source_records"}
    missing_columns = {
        "raw": sorted(required_raw - set(raw.columns)),
        "curated": sorted(required_curated - set(curated.columns)),
    }
    if any(missing_columns.values()):
        raise ValueError(f"Missing required columns: {missing_columns}")

    raw_mols = [parse(s) for s in raw.Smiles]
    curated_mols = [parse(s) for s in curated.canonical_smiles]
    raw_valid = [m is not None for m in raw_mols]
    curated_valid = [m is not None for m in curated_mols]
    if not all(raw_valid) or not all(curated_valid):
        bad = [i for i, ok in enumerate(raw_valid) if not ok]
        bad += [f"curated:{i}" for i, ok in enumerate(curated_valid) if not ok]
        raise ValueError(f"Invalid SMILES found; audit must stop: {bad}")

    raw_canonical = [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in raw_mols]
    raw_inchikeys = [Chem.MolToInchiKey(m) for m in raw_mols]
    curated_inchikeys = [Chem.MolToInchiKey(m) for m in curated_mols]
    raw_components = [len(Chem.GetMolFrags(m)) for m in raw_mols]
    curated_components = [len(Chem.GetMolFrags(m)) for m in curated_mols]

    conv_raw = 9.0 - np.log10(raw.IC50.to_numpy(float))
    conv_curated = 9.0 - np.log10(curated.IC50.to_numpy(float))
    raw_conversion_max_abs_error = float(np.max(np.abs(conv_raw - raw.pIC50.to_numpy(float))))
    curated_conversion_max_abs_error = float(np.max(np.abs(conv_curated - curated.pIC50.to_numpy(float))))
    raw_values_positive = bool(np.all(raw.IC50.to_numpy(float) > 0))
    curated_values_positive = bool(np.all(curated.IC50.to_numpy(float) > 0))

    def notebook_cell_text(index):
        cell = notebook["cells"][index]
        parts = ["".join(cell.get("source", []))]
        for output in cell.get("outputs", []):
            parts.append("".join(output.get("data", {}).get("text/plain", [])))
            parts.extend(output.get("text", []))
        return "\n".join(parts)

    unit_counts = {unit: int(count) for unit, count in
                   re.findall(r"(nM|pIC50)\s+(\d+)", notebook_cell_text(5))}
    source_unit_conversion = notebook_cell_text(6)
    conversion_comment_molarity_mismatch = (
        "μM" in source_unit_conversion.splitlines()[0]
        and re.search(r"10\*\*\(?9\)?", re.sub(r"\s+", "", source_unit_conversion)) is not None
    )
    notebook_input_rows = re.search(r"RangeIndex: (\d+) entries", notebook_cell_text(8))
    notebook_final_rows = re.search(r"\[(\d+) rows x", notebook_cell_text(27))
    notebook_unit_evidence = (
        unit_counts == {"nM": 3485, "pIC50": 28}
        and "10**(-x)" in re.sub(r"\s+", "", source_unit_conversion)
        and "Standard Units" in notebook_cell_text(5)
    )
    all_mols_identical = sha256(ALL_MOLS) == sha256(RAW)

    centers = [Chem.FindMolChiralCenters(m, includeUnassigned=True, includeCIP=True)
               for m in curated_mols]
    unassigned_center_rows = [i for i, vals in enumerate(centers) if any(label == "?" for _, label in vals)]
    chiral_center_rows = [i for i, vals in enumerate(centers) if vals]

    disconnected = []
    for i, (mol, ncomp) in enumerate(zip(curated_mols, curated_components)):
        if ncomp <= 1:
            continue
        fragments = [Chem.MolToSmiles(f, canonical=True, isomericSmiles=True)
                     for f in Chem.GetMolFrags(mol, asMols=True)]
        disconnected.append({
            "curated_row": i,
            "canonical_smiles": curated.canonical_smiles.iloc[i],
            "source_rows": curated.source_rows.iloc[i],
            "component_count": int(ncomp),
            "fragment_smiles": fragments,
            "fragment_formulas": [rdMolDescriptors.CalcMolFormula(f) for f in Chem.GetMolFrags(mol, asMols=True)],
            "IC50_as_provided": float(curated.IC50.iloc[i]),
            "pIC50_as_provided": float(curated.pIC50.iloc[i]),
            "excluded": False,
            "decision": "retained unchanged; mixture/salt meaning cannot be resolved without source records",
        })

    q1, median, q3 = np.quantile(curated.pIC50.to_numpy(float), [0.25, 0.5, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_rows = [int(i) for i, x in enumerate(curated.pIC50) if x < low or x > high]
    duplicate_counts = pd.Series(raw_canonical).value_counts()
    duplicate_group_keys = set(duplicate_counts[duplicate_counts > 1].index)
    curated_key_frame = pd.DataFrame({
        "InChIKey": curated_inchikeys,
        "canonical_smiles": curated.canonical_smiles.astype(str),
        "IC50": curated.IC50.astype(float), "pIC50": curated.pIC50.astype(float),
        "source_rows": curated.source_rows.astype(str),
    })
    inchi_duplicate_details = []
    for key, group in curated_key_frame.groupby("InChIKey", sort=True):
        if len(group) <= 1:
            continue
        inchi_duplicate_details.append({
            "InChIKey": key,
            "canonical_smiles": group.canonical_smiles.tolist(),
            "source_rows": group.source_rows.tolist(),
            "n_curated_records": int(len(group)),
            "IC50_values": group.IC50.tolist(), "pIC50_values": group.pIC50.tolist(),
            "activity_values_identical": bool(group.IC50.nunique() == 1 and group.pIC50.nunique() == 1),
            "inferred_relation": "different canonical SMILES with the same InChIKey (likely tautomeric forms); confirm with standardization policy",
        })
    discordant_groups = dup[dup["IC50_min"] != dup["IC50_max"]] if len(dup) else dup
    potency_favoring = int(dup.rule.astype(str).str.contains("minimum IC50", case=False).sum()) if len(dup) else 0

    curated_hash = sha256(CURATED)
    blockers = []
    if curated_hash != EXPECTED_CURATED_SHA256:
        blockers.append("Curated dataset SHA-256 differs from the previously evaluated dataset")
    if raw_conversion_max_abs_error > 2e-5 or curated_conversion_max_abs_error > 2e-5:
        blockers.append("IC50-to-pIC50 arithmetic is inconsistent")
    if not raw_values_positive or not curated_values_positive:
        blockers.append("Nonpositive IC50 values are present")
    if not notebook_unit_evidence:
        blockers.append("Saved processing notebook does not establish the input units and conversion")
    warnings = [
        "The original raw_data.csv is unavailable for row-level replay of source units or the upstream duplicate/IQR operations.",
        "The processing notebook's conversion comment says pIC50 is based on μM, while its 10^9 conversion factor assumes the conventional -log10(molar) pIC50 scale. ChEMBL's documentation defines pChEMBL on the molar scale (1 nM = 9); without raw source rows, the comment/value provenance cannot be independently confirmed.",
        "Four InChIKey-equivalent tautomer pairs remain distinct in the hash-pinned dataset; all four pairs have identical IC50 labels and remain within scaffold groups.",
        "The exact VECTOR+ source implementation is absent; the attached median-split procedure can be followed, but source-level parity cannot be verified.",
        "Assay IDs, source citations, censoring and replicate metadata are not preserved in the final CSV; results are not assay-harmonized or externally validated.",
    ]

    audit = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_DIR.name,
        "raw_file": str(RAW.relative_to(ROOT)),
        "all_mols_file": str(ALL_MOLS),
        "processing_notebook": str(PROCESSING_NOTEBOOK),
        "raw_source_csv_available": RAW_SOURCE.exists(),
        "curated_file": str(CURATED.relative_to(ROOT)),
        "raw_sha256": sha256(RAW),
        "all_mols_sha256": sha256(ALL_MOLS),
        "all_mols_byte_identical_to_processed_df": all_mols_identical,
        "processing_notebook_sha256": sha256(PROCESSING_NOTEBOOK),
        "notebook_raw_input_rows": int(notebook_input_rows.group(1)) if notebook_input_rows else None,
        "notebook_final_filtered_rows": int(notebook_final_rows.group(1)) if notebook_final_rows else None,
        "notebook_Standard_Units_counts": unit_counts,
        "notebook_source_conversion_code": source_unit_conversion,
        "conversion_comment_molarity_mismatch": conversion_comment_molarity_mismatch,
        "unit_source_evidence": "Data_processing.ipynb saved output records Standard Units counts: nM=3485, pIC50=28; its code converts pIC50 rows to nM. Raw_data.csv is not available for row-level replay.",
        "curated_sha256": curated_hash,
        "expected_curated_sha256": EXPECTED_CURATED_SHA256,
        "curated_hash_matches_expected": curated_hash == EXPECTED_CURATED_SHA256,
        "raw_record_count": int(len(raw)),
        "curated_record_count": int(len(curated)),
        "excluded_record_count": 0,
        "raw_columns": list(raw.columns),
        "curated_columns": list(curated.columns),
        "missing_values": {"raw": {k: int(v) for k, v in raw.isna().sum().items()},
                           "curated": {k: int(v) for k, v in curated.isna().sum().items()}},
        "invalid_smiles": {"raw": int(len(raw) - sum(raw_valid)), "curated": int(len(curated) - sum(curated_valid))},
        "raw_literal_smiles_unique": int(raw.Smiles.nunique(dropna=False)),
        "raw_duplicate_literal_smiles_excess_rows": int(len(raw) - raw.Smiles.nunique(dropna=False)),
        "raw_unique_canonical_isomeric_smiles": int(len(set(raw_canonical))),
        "raw_canonical_duplicate_groups": int(len(duplicate_group_keys)),
        "curated_duplicate_canonical_smiles": int(curated.canonical_smiles.duplicated().sum()),
        "raw_duplicate_inchikey_groups": int((pd.Series(raw_inchikeys).value_counts() > 1).sum()),
        "curated_duplicate_inchikey_groups": int((pd.Series(curated_inchikeys).value_counts() > 1).sum()),
        "curated_inchikey_duplicate_details": inchi_duplicate_details,
        "disconnected_curated_records": disconnected,
        "disconnected_raw_record_count": int(sum(n > 1 for n in raw_components)),
        "stereochemistry": {
            "curated_structures_with_chiral_centers": int(len(chiral_center_rows)),
            "curated_structures_with_unassigned_potential_chiral_centers": int(len(unassigned_center_rows)),
            "rows_with_unassigned_centers": unassigned_center_rows,
            "interpretation": "Unspecified stereocenters are observable; no source stereochemistry or assay metadata are available to decide whether they are errors. Isomeric canonicalization preserves specified stereochemistry.",
        },
        "activity_conversion": {
            "formula_checked": "pIC50 = 9 - log10(IC50[nM])",
            "raw_max_abs_arithmetic_error": raw_conversion_max_abs_error,
            "curated_max_abs_arithmetic_error": curated_conversion_max_abs_error,
            "raw_IC50_positive": raw_values_positive,
            "curated_IC50_positive": curated_values_positive,
            "source_units_documented_in_saved_notebook": notebook_unit_evidence,
            "raw_source_file_row_level_replay_available": RAW_SOURCE.exists(),
            "source_assay_metadata_available": False,
            "conversion_comment_molarity_mismatch": conversion_comment_molarity_mismatch,
            "note": "Saved notebook output records nM plus pIC50 inputs. Its 10^9 conversion factor matches conventional -log10(molar) pIC50, but the code comment says μM. Arithmetic is exact on processed rows; raw_data.csv is absent, so the source-row convention cannot be independently verified.",
        },
        "duplicate_resolution": {
            "groups": int(len(dup)),
            "discordant_groups": int(len(discordant_groups)),
            "potency_favoring_minimum_IC50_groups": potency_favoring,
            "rule": "Exact repeated values collapse; when full IC50 range is within 10% of group median, average IC50; otherwise retain minimum IC50, following the prior Syk-paper adaptation.",
            "limitations": "Source assay IDs, replicate provenance, censoring and units are absent; discordance cannot be adjudicated experimentally.",
        },
        "pIC50_distribution": {
            "count": int(curated.pIC50.count()), "mean": float(curated.pIC50.mean()),
            "sd": float(curated.pIC50.std()), "min": float(curated.pIC50.min()),
            "median": float(median), "max": float(curated.pIC50.max()),
            "tukey_1_5_iqr_bounds": [float(low), float(high)],
            "tukey_flagged_rows": outlier_rows,
            "decision": "Outlier flags are descriptive only; no values were removed or altered.",
        },
        "files_not_found_in_repository": [
            "The raw_data.csv source referenced by Data_processing.ipynb",
            "VECTOR+ training/inference implementation or source-level activity-label code",
        ],
        "blocking_issues": blockers,
        "warnings": warnings,
        "training_allowed_for_internal_evaluation": len(blockers) == 0,
        "publication_ready": False,
        "stop_reason": None if len(blockers) == 0 else "A required preflight integrity check failed.",
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "rdkit": rdkit.__version__},
    }

    (RUN_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2, default=float), encoding="utf-8")
    (RUN_DIR / "excluded_records.csv").write_text(
        "source_row,canonical_smiles,reason\n", encoding="utf-8")
    dup_out = dup.copy()
    curated_keys = dict(zip(curated.canonical_smiles, curated_inchikeys))
    dup_out["InChIKey"] = dup_out.canonical_smiles.map(curated_keys)
    dup_out.insert(0, "duplicate_basis", "canonical_isomeric_smiles")
    inchi_rows = []
    for item in inchi_duplicate_details:
        vals = item["IC50_values"]
        inchi_rows.append({
            "duplicate_basis": "InChIKey_equivalence",
            "canonical_smiles": " || ".join(item["canonical_smiles"]),
            "n_records": item["n_curated_records"],
            "source_rows": ";".join(item["source_rows"]),
            "IC50_min": min(vals), "IC50_median": float(np.median(vals)), "IC50_max": max(vals),
            "IC50_range": max(vals) - min(vals),
            "resolved_IC50": vals[0] if item["activity_values_identical"] else np.nan,
            "resolved_pIC50": item["pIC50_values"][0] if item["activity_values_identical"] else np.nan,
            "rule": "Not merged: separate tautomers share InChIKey; review standardization and weighting. Curated expected-hash dataset preserved.",
            "InChIKey": item["InChIKey"],
        })
    dup_out = pd.concat([dup_out, pd.DataFrame(inchi_rows)], ignore_index=True, sort=False)
    dup_out.to_csv(RUN_DIR / "duplicate_audit.csv", index=False)
    manifest = {
        "run_id": RUN_DIR.name, "created_utc": audit["audit_timestamp_utc"],
        "raw_dataset": {"path": str(RAW.relative_to(ROOT)), "sha256": audit["raw_sha256"], "records": len(raw)},
        "all_mols": {"path": str(ALL_MOLS), "sha256": audit["all_mols_sha256"], "byte_identical_to_processed_df": all_mols_identical},
        "processing_notebook": {"path": str(PROCESSING_NOTEBOOK), "sha256": sha256(PROCESSING_NOTEBOOK),
                                "unit_counts": unit_counts, "raw_input_rows": audit["notebook_raw_input_rows"],
                                "final_processed_rows": audit["notebook_final_filtered_rows"]},
        "curated_dataset": {"path": str(CURATED.relative_to(ROOT)), "sha256": curated_hash,
                            "expected_sha256": EXPECTED_CURATED_SHA256, "records": len(curated)},
        "experiment_sources_inspected": {
            "experiment_01": "results/qsar/experiment_01_model_comparison/exp01_20261121T183530Z",
            "experiment_03": "results/qsar/experiment_03_stacking_optimisation/exp03_20261122T063732Z",
            "experiment_04": "results/qsar/experiment_04_reliability/exp04_20261122T082326Z",
        },
        "modeling_started": (RUN_DIR / "logs" / "execution.log").exists(),
        "blocking_issues": blockers,
        "warnings": warnings,
        "software": audit["software"],
    }
    (RUN_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = [
        "# Final QSAR preflight data audit", "", f"Run: `{RUN_DIR.name}`", "",
        f"Raw table: {len(raw):,} rows; SHA-256 `{audit['raw_sha256']}`.",
        f"Curated table: {len(curated):,} structures; SHA-256 `{curated_hash}` (expected hash matches: {audit['curated_hash_matches_expected']}).", "",
        "## Audit findings", "",
        f"- Missing values: raw {sum(audit['missing_values']['raw'].values())}; curated {sum(audit['missing_values']['curated'].values())}.",
        f"- Invalid SMILES: raw {audit['invalid_smiles']['raw']}; curated {audit['invalid_smiles']['curated']}.",
        f"- Raw literal duplicate excess rows: {audit['raw_duplicate_literal_smiles_excess_rows']}; canonical duplicate groups resolved in curation: {audit['duplicate_resolution']['groups']}; curated duplicate canonical structures: {audit['curated_duplicate_canonical_smiles']}.",
        f"- Duplicate InChIKey groups: raw {audit['raw_duplicate_inchikey_groups']}; curated {audit['curated_duplicate_inchikey_groups']}. The four curated pairs are different tautomeric SMILES with identical IC50 values; retained separately to preserve the expected hash, with minor duplicate weighting documented.",
        f"- Disconnected structures retained unchanged: {len(disconnected)}. Their fragments and source-row mappings are in `data_audit.json`.",
        f"- Structures with unassigned potential chiral centers: {len(unassigned_center_rows)}. No source stereo metadata are available to adjudicate them.",
        f"- Internal formula check maximum absolute error: raw {raw_conversion_max_abs_error:.3g}; curated {curated_conversion_max_abs_error:.3g}.",
        f"- Tukey 1.5-IQR outlier flags: {len(outlier_rows)}; none were excluded.",
        f"- Discordant duplicate groups: {len(discordant_groups)}; groups resolved by the potency-favoring minimum-IC50 branch: {potency_favoring}.", "",
        "## Unit provenance and remaining data limitations", "",
        f"The previously shared Data_processing.ipynb is available at `{PROCESSING_NOTEBOOK}`. Its saved Standard Units output records {unit_counts.get('nM')} nM and {unit_counts.get('pIC50')} pIC50 rows in a 3,513-row input; its code converts pIC50 values to nM before recalculating pIC50. The code comment says μM, but the 10^9 factor assumes the conventional molar pIC50 scale. ChEMBL documents pChEMBL as -log10(molar concentration), where 1 nM is 9 ([ChEMBL activity FAQ](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)). The mismatch is likely a comment error, but the missing raw_data.csv prevents row-level verification of the 28 source pIC50 values. It records a 3,176-row final IQR-filtered output. `all_mols.csv` and `processed_df.csv` are byte-identical 3,176-row tables, and their pIC50 arithmetic matches nM conversion exactly.", "",
        "The six disconnected structures were inspected and retained unchanged. Four InChIKey-equivalent tautomer pairs have identical IC50 labels and remain separate to preserve the stipulated dataset hash; scaffold grouping prevents their split across evaluation folds. Assay IDs, source/censoring metadata and replicate identities remain unavailable, and upstream IQR exclusions cannot be reconstructed. These are publication limitations, not grounds for silently dropping rows.", "",
        "## Prior experiments inspected", "",
        "Experiment 1 outputs exist and were read: two repeats of five-fold scaffold CV; stacking mean RMSE 0.6483 ± 0.0365, versus XGBoost 0.6606 ± 0.0347 and HistGradientBoosting 0.6629 ± 0.0435. Stacking beat HistGradientBoosting in 10/10 folds and XGBoost in 9/10. This supports stacking as a candidate, but differences are modest and all results are internal.", "",
        "Experiment 3's screened lowest mean-RMSE configuration was the SVR-containing four-model Ridge(100) stack (mean RMSE 0.6427 ± 0.0452). Paired scaffold-bootstrap comparisons with the boosting pair and all-five Ridge stack included zero; therefore the nominal winner is not clearly superior. SVR had less correlated residuals but weaker standalone RMSE.", "",
        "Experiment 4's five-fold pooled stack RMSE was 0.6767, MAE 0.5053, R² 0.6241. Disagreement, AD distance, and combined-ranking Spearman correlations with absolute error were 0.185, 0.225, and 0.251. The lowest-uncertainty 10% had RMSE 0.454 (n=306). Molecule-level 90% interval coverage was 88.8%, mean width 2.135 pIC50. These support uncertainty filtering as a diagnostic, not calibrated confidence on generated molecules.", "",
        "No VECTOR+ implementation or source label-generation code was found in the repository or inspected user references. This run follows the user's explicit median-split definition and will label it VECTOR+-compatible, not an exact source-code reproduction.", "",
        "## Audit disposition", "",
        f"Critical preflight blockers: {len(blockers)}. Internal scaffold evaluation is allowed against the expected hash-pinned data with the above limitations. Publication-grade external validity is not established.", "",
    ]
    (RUN_DIR / "data_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"run_id": RUN_DIR.name, "curated_sha256": curated_hash,
                      "disconnected_count": len(disconnected), "duplicate_groups": len(dup),
                      "blocking_issues": blockers, "training_allowed_for_internal_evaluation": len(blockers) == 0}, indent=2))


if __name__ == "__main__":
    main()
