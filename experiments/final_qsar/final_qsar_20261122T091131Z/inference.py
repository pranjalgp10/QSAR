"""Inference interface for the final Syk QSAR model.

Predictions are computational estimates. This module deliberately returns
``caution`` for every valid molecule because no prospective pseudolabel
acceptance criterion has yet been validated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def _fingerprints(smiles: list[str], radius: int, bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    X = np.zeros((len(smiles), bits), dtype=np.uint8)
    mols, canonical = [], []
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            mols.append(None)
            canonical.append(None)
            continue
        mols.append(mol)
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        ConvertToNumpyArray(generator.GetFingerprint(mol), X[i])
    return X, mols, canonical


def _max_tanimoto(query: np.ndarray, packed_reference: np.ndarray) -> np.ndarray:
    reference = np.unpackbits(np.asarray(packed_reference, dtype=np.uint8), axis=1)[:, :query.shape[1]].astype(np.uint32)
    q = query.astype(np.uint32)
    ref_counts = reference.sum(axis=1)
    q_counts = q.sum(axis=1)
    output = np.zeros(len(q), dtype=float)
    for i, row in enumerate(q):
        intersection = reference @ row
        union = q_counts[i] + ref_counts - intersection
        similarities = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        output[i] = float(similarities.max(initial=0.0))
    return output


def predict_smiles(smiles_list: Iterable[str], model_path: str | Path | None = None) -> pd.DataFrame:
    """Predict activity, domain distance, disagreement and internal intervals.

    Invalid SMILES return ``reliability_status='reject'`` with a reason.
    Valid molecules return ``caution`` because current internal validation
    cannot justify a pseudolabel acceptance cutoff.
    """
    if smiles_list is None:
        raise TypeError("smiles_list must be an iterable of SMILES strings")
    inputs = list(smiles_list)
    if any(not isinstance(s, str) for s in inputs):
        raise TypeError("Every input must be a SMILES string")
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    bundle = joblib.load(path)
    bits, radius = int(bundle["bits"]), int(bundle["radius"])
    X, _, canonical = _fingerprints(inputs, radius, bits)
    valid = np.array([s is not None for s in canonical], dtype=bool)
    result = pd.DataFrame({
        "input_smiles": inputs,
        "canonical_smiles": canonical,
        "predicted_pIC50": np.nan,
        "predicted_IC50_nM": np.nan,
        "activity_threshold_pIC50": float(bundle["activity_threshold"]),
        "predicted_activity": None,
        "applicability_domain_distance": np.nan,
        "nearest_training_tanimoto": np.nan,
        "ensemble_disagreement_sd": np.nan,
        "combined_uncertainty_ad_rank": np.nan,
        "prediction_interval_90_low_pIC50": np.nan,
        "prediction_interval_90_high_pIC50": np.nan,
        "reliability_status": "reject",
        "rejection_reasons": "Invalid or unparsable SMILES",
    })
    if not valid.any():
        return result

    Xv = X[valid]
    base = np.column_stack([bundle["base_models"][name].predict(Xv) for name in bundle["base_names"]])
    cols = [bundle["base_names"].index(name) for name in bundle["meta_members"]]
    predicted = bundle["meta_model"].predict(base[:, cols])
    similarity = _max_tanimoto(Xv, bundle["reference_fingerprints_packed"])
    disagreement = base.std(axis=1, ddof=1)
    disagreement_rank = np.searchsorted(np.sort(np.asarray(bundle["disagreement_reference"], float)), disagreement, side="right") / len(bundle["disagreement_reference"])
    ad_distance = 1.0 - similarity
    ad_rank = np.searchsorted(np.sort(np.asarray(bundle["ad_distance_reference"], float)), ad_distance, side="right") / len(bundle["ad_distance_reference"])
    half_width = float(bundle["molecule_interval_half_width"])
    indices = np.flatnonzero(valid)
    result.loc[valid, "predicted_pIC50"] = predicted
    result.loc[valid, "predicted_IC50_nM"] = np.power(10.0, 9.0 - predicted)
    result.loc[valid, "predicted_activity"] = np.where(predicted >= bundle["activity_threshold"], "active", "inactive")
    result.loc[valid, "applicability_domain_distance"] = ad_distance
    result.loc[valid, "nearest_training_tanimoto"] = similarity
    result.loc[valid, "ensemble_disagreement_sd"] = disagreement
    result.loc[valid, "combined_uncertainty_ad_rank"] = (disagreement_rank + ad_rank) / 2.0
    result.loc[valid, "prediction_interval_90_low_pIC50"] = predicted - half_width
    result.loc[valid, "prediction_interval_90_high_pIC50"] = predicted + half_width
    result.loc[valid, "reliability_status"] = "caution"
    result.loc[valid, "rejection_reasons"] = "No validated pseudolabel acceptance threshold; internal scaffold validation only"
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="*", help="SMILES; omit to read one per stdin line")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    values = args.smiles if args.smiles else [line.strip() for line in __import__("sys").stdin if line.strip()]
    print(predict_smiles(values, args.model).to_json(orient="records", indent=2))
