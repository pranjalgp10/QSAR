"""Hash-gated data loading and Morgan feature construction."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "syk_qsar_curated.csv"
EXPECTED_SHA256 = "541d081a227ea342b4c01b766ea4c1d1b782329f95f685c22c4de5adc1a51f25"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def median_split(training_labels, labels_to_assign):
    """Return a threshold derived only from training labels and >=-tie labels."""
    train = np.asarray(training_labels, dtype=float)
    apply = np.asarray(labels_to_assign, dtype=float)
    if train.size == 0 or not np.isfinite(train).all() or not np.isfinite(apply).all():
        raise ValueError("Median split requires finite, non-empty training labels")
    threshold = float(np.median(train))
    return threshold, (apply >= threshold).astype(np.int8)


def load_dataset():
    if sha256(DATA) != EXPECTED_SHA256:
        raise RuntimeError("Curated dataset hash mismatch; stopping before training")
    df = pd.read_csv(DATA)
    required = {"Smiles", "canonical_smiles", "IC50", "pIC50", "scaffold_group", "source_rows"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing dataset columns: {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any() or not np.isfinite(df.pIC50).all():
        raise ValueError("Missing or non-finite curated data values")
    if df.canonical_smiles.duplicated().any():
        raise ValueError("Duplicate canonical SMILES remain in the curated table")
    conversion = 9.0 - np.log10(df.IC50.to_numpy(float))
    if np.max(np.abs(conversion - df.pIC50.to_numpy(float))) > 2e-5:
        raise ValueError("pIC50 conversion is inconsistent with nM")
    mols, fps = [], []
    X = np.zeros((len(df), 2048), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for i, smiles in enumerate(df.canonical_smiles.astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES at row {i}")
        mols.append(mol)
        fp = generator.GetFingerprint(mol)
        fps.append(fp)
        DataStructs.ConvertToNumpyArray(fp, X[i])
    groups = df.scaffold_group.astype(str).to_numpy()
    return df, np.asarray(df.pIC50, dtype=float), groups, X, mols, fps


def max_train_similarity(query_fps, train_fps):
    if not train_fps:
        raise ValueError("Applicability domain requires at least one fit fingerprint")
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        for fp in query_fps
    ], dtype=float)
