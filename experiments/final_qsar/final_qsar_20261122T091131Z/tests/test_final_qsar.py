from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))

from data_utils import EXPECTED_SHA256, load_dataset, median_split, sha256  # noqa: E402
from evaluation_metrics import conformal_quantile, regression_metrics  # noqa: E402
from inference import predict_smiles  # noqa: E402


def test_dataset_hash_validity_and_fingerprint_shape():
    df, y, groups, X, mols, _ = load_dataset()
    assert sha256(RUN.parents[2] / "data" / "processed" / "syk_qsar_curated.csv") == EXPECTED_SHA256
    assert len(df) == len(y) == len(groups) == len(mols) == X.shape[0] == 3051
    assert X.shape[1] == 2048
    assert np.isfinite(y).all()


def test_scaffold_folds_have_no_overlap():
    _, y, groups, X, _, _ = load_dataset()
    for fit, heldout in GroupKFold(n_splits=5).split(X, y, groups):
        assert not set(groups[fit]) & set(groups[heldout])


def test_fit_calibration_test_scaffolds_are_disjoint():
    manifest_path = RUN / "manifests" / "folds_checkpoint.json"
    if not manifest_path.exists():
        pytest.skip("Fold manifest is created by train_evaluate.py")
    _, _, groups, _, _, _ = load_dataset()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == 10
    for record in records:
        fit = set(groups[np.asarray(record["fit_indices"], dtype=int)])
        cal = set(groups[np.asarray(record["calibration_indices"], dtype=int)])
        test = set(groups[np.asarray(record["test_indices"], dtype=int)])
        assert not fit & cal and not fit & test and not cal & test


def test_training_only_median_and_ties():
    threshold, assigned = median_split([1.0, 2.0, 3.0], [2.0, 1.999, 99.0])
    assert threshold == 2.0
    assert assigned.tolist() == [1, 0, 1]
    other_threshold, _ = median_split([1.0, 2.0, 3.0], [-1000.0, 1000.0])
    assert other_threshold == threshold


def test_metric_and_conformal_calculation():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert m["MAE"] == pytest.approx(1 / 3)
    assert m["RMSE"] == pytest.approx((1 / 3) ** 0.5)
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.25) == pytest.approx(0.4)


def test_model_roundtrip_and_inference_edges():
    model_path = RUN / "model.joblib"
    if not model_path.exists():
        pytest.skip("Final model is created by train_evaluate.py")
    one = predict_smiles(["CCO"], model_path)
    assert len(one) == 1 and one.loc[0, "reliability_status"] == "caution"
    assert np.isfinite(one.loc[0, "predicted_pIC50"])
    assert np.isfinite(one.loc[0, "nearest_training_tanimoto"])
    assert 0.0 <= one.loc[0, "combined_uncertainty_ad_rank"] <= 1.0
    batch = predict_smiles(["CCO", "not-a-smiles", "CCO", "CCCC"], model_path)
    assert len(batch) == 4
    assert batch.loc[1, "reliability_status"] == "reject"
    assert batch.loc[1, "predicted_pIC50"] != batch.loc[1, "predicted_pIC50"]
    assert batch.loc[0, "predicted_pIC50"] == pytest.approx(batch.loc[2, "predicted_pIC50"])
    assert batch.loc[3, "canonical_smiles"] == "CCCC"
    training_df, _, groups, _, _, _ = load_dataset()
    training_canonical = set(training_df.canonical_smiles.astype(str))
    assert batch.loc[0, "canonical_smiles"] not in training_canonical
    candidate = Chem.MolFromSmiles("CCO")
    candidate_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=candidate, includeChirality=False) or f"ACYCLIC::{batch.loc[0, 'canonical_smiles']}"
    assert candidate_scaffold not in set(groups)  # unseen scaffold
    assert batch.loc[0, "prediction_interval_90_low_pIC50"] <= batch.loc[0, "predicted_pIC50"]
    assert batch.loc[0, "predicted_pIC50"] <= batch.loc[0, "prediction_interval_90_high_pIC50"]
    out_of_domain = predict_smiles(["C" * 80], model_path).iloc[0]
    assert out_of_domain["reliability_status"] == "caution"
    assert 0.0 <= out_of_domain["nearest_training_tanimoto"] <= 1.0
    assert len(predict_smiles([], model_path)) == 0


def test_inference_in_fresh_python_process():
    model_path = RUN / "model.joblib"
    if not model_path.exists():
        pytest.skip("Final model is created by train_evaluate.py")
    completed = subprocess.run(
        [sys.executable, str(RUN / "inference.py"), "--model", str(model_path), "CCO"],
        check=True, capture_output=True, text=True, timeout=120,
    )
    rows = json.loads(completed.stdout)
    assert len(rows) == 1 and rows[0]["reliability_status"] == "caution"
