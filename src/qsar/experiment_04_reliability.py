"""Experiment 4: scaffold-held-out reliability, applicability and intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "syk_qsar_curated.csv"
CFG = ROOT / "experiments" / "qsar" / "experiment_04_reliability" / "config.json"
LOG = logging.getLogger("experiment04")


def base_models(seed: int) -> dict:
    """Frozen model settings from the preserved benchmark report."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=28, min_samples_leaf=1,
            max_features=0.5, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=28, min_samples_leaf=1,
            max_features=0.5, random_state=seed, n_jobs=-1),
        "SVR": make_pipeline(StandardScaler(), SVR(C=5, epsilon=0.1, gamma="scale")),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", n_estimators=250, max_depth=7,
            learning_rate=0.03, subsample=0.8, random_state=seed,
            n_jobs=1, verbosity=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=2.0, random_state=seed),
    }


def fit_stack(X, y, groups, seed):
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two fit scaffolds are required for stacking")
    models = base_models(seed)
    cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    stack = StackingRegressor(
        estimators=list(models.items()), final_estimator=Ridge(alpha=1.0),
        cv=cv, n_jobs=1)
    stack.fit(X, y)
    return stack, list(models)


def predict_components(stack, names, X):
    base = np.column_stack([model.predict(X) for model in stack.estimators_])
    return base, stack.predict(X), base.std(axis=1, ddof=1)


def max_morgan_similarity(query_fps, train_fps):
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        for fp in query_fps
    ], dtype=float)


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        raise ValueError("Cannot calibrate an interval without calibration scores")
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def metric_set(y, pred):
    e = np.asarray(pred, float) - np.asarray(y, float)
    return {
        "n": int(len(y)),
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mean_squared_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred)),
        "median_absolute_error": float(np.median(np.abs(e))),
        "residual_mean": float(np.mean(e)),
        "residual_sd": float(np.std(e, ddof=1)),
    }


def scaffold_bootstrap(pred_df, n_boot=2000, seed=4104):
    """Cluster-bootstrap held-out molecules by Bemis-Murcko scaffold."""
    rng = np.random.default_rng(seed)
    groups, group_ix = np.unique(pred_df.scaffold.astype(str), return_inverse=True)
    y = pred_df.observed_pIC50.to_numpy(float)
    p = pred_df.stack_prediction.to_numpy(float)
    abs_err = np.abs(p - y)
    covered = pred_df.molecule_90_covered.to_numpy(bool)
    scaffold_groups = pred_df.groupby("scaffold").scaffold_90_covered.first()
    scaffold_covered = scaffold_groups.to_numpy(bool)
    vals = []
    for _ in range(n_boot):
        draws = rng.integers(0, len(groups), size=len(groups))
        weights = np.bincount(draws, minlength=len(groups))[group_ix]
        total = weights.sum()
        scaffold_draws = rng.integers(0, len(scaffold_covered), size=len(scaffold_covered))
        scaffold_weights = np.bincount(scaffold_draws, minlength=len(scaffold_covered))
        vals.append([
            np.sqrt(np.sum(weights * (p - y) ** 2) / total),
            np.sum(weights * abs_err) / total,
            np.sum(weights * covered) / total,
            np.sum(scaffold_weights * scaffold_covered) / scaffold_weights.sum(),
        ])
    arr = np.asarray(vals)
    labels = ["RMSE", "MAE", "molecule_90_coverage", "scaffold_90_full_containment"]
    return {
        name: {"estimate": float(np.mean(arr[:, i])),
               "cluster_bootstrap_95CI": np.quantile(arr[:, i], [0.025, 0.975]).tolist()}
        for i, name in enumerate(labels)
    }


def reliability_bins(df, score, bins=10):
    temp = df[["absolute_error", score]].dropna().copy()
    temp["bin"] = pd.qcut(temp[score], q=min(bins, temp[score].nunique()),
                           duplicates="drop")
    return temp.groupby("bin", observed=True).agg(
        n=("absolute_error", "size"), mean_score=(score, "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=("absolute_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        p90_absolute_error=("absolute_error", lambda x: float(np.quantile(x, 0.9))),
    ).reset_index().assign(score_name=score)


def precision_coverage(df, scores, retained_fractions):
    rows = []
    for name in scores:
        ordered = df.sort_values(name, ascending=True).reset_index(drop=True)
        for fraction in retained_fractions:
            n = max(1, int(np.ceil(float(fraction) * len(ordered))))
            part = ordered.iloc[:n]
            err = part.absolute_error.to_numpy(float)
            rows.append({
                "uncertainty_score": name, "target_retained_fraction": float(fraction),
                "actual_retained_fraction": n / len(ordered), "n_retained": n,
                "MAE": float(np.mean(err)), "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "p90_absolute_error": float(np.quantile(err, 0.9)),
                "fraction_within_0.5_pIC50": float(np.mean(err <= 0.5)),
                "fraction_within_1.0_pIC50": float(np.mean(err <= 1.0)),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path,
                        help="Resume a run folder after its last completed outer fold")
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    run_id = args.resume.name if args.resume else datetime.now(timezone.utc).strftime("exp04_%Y%m%dT%H%M%SZ")
    out = args.resume if args.resume else ROOT / "results" / "qsar" / "experiment_04_reliability" / run_id
    if args.resume and not out.is_absolute():
        out = ROOT / out
    for sub in ["configs", "predictions", "metrics", "manifests", "logs", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(out / "logs" / "run.log", encoding="utf-8"),
                  logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(message)s")
    config_copy = out / "configs" / "config.json"
    if args.resume and config_copy.exists() and json.loads(config_copy.read_text(encoding="utf-8")) != cfg:
        raise ValueError("Cannot resume: configuration differs from saved run")
    config_copy.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    df = pd.read_csv(DATA)
    y = df.pIC50.to_numpy(float)
    groups = df.scaffold_group.astype(str).to_numpy()
    if not np.isfinite(y).all() or len(df) != len(np.unique(df.canonical_smiles)):
        raise ValueError("Curated data has non-finite targets or duplicate canonical structures")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = []
    fps = []
    X = np.zeros((len(df), 2048), dtype=np.uint8)
    from rdkit import Chem
    from rdkit import DataStructs as DS
    for i, smiles in enumerate(df.canonical_smiles.astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES in row {i}")
        mols.append(mol)
        fp = fpgen.GetFingerprint(mol)
        fps.append(fp)
        DS.ConvertToNumpyArray(fp, X[i])

    seed = int(cfg["outer_split"]["seed"])
    folds = list(GroupKFold(n_splits=cfg["outer_split"]["folds"], shuffle=True,
                            random_state=seed).split(X, y, groups))
    for tr, te in folds:
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Scaffold leakage across outer split")

    pred_path = out / "predictions" / "oof_predictions_checkpoint.csv"
    metric_path = out / "metrics" / "fold_metrics_checkpoint.csv"
    cal_path = out / "metrics" / "calibration_checkpoint.csv"
    manifest_path = out / "manifests" / "folds_checkpoint.json"
    pred_rows = pd.read_csv(pred_path).to_dict("records") if args.resume and pred_path.exists() else []
    metric_rows = pd.read_csv(metric_path).to_dict("records") if args.resume and metric_path.exists() else []
    cal_rows = pd.read_csv(cal_path).to_dict("records") if args.resume and cal_path.exists() else []
    fold_records = json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume and manifest_path.exists() else []
    completed = {int(record["fold"]) for record in fold_records}
    alpha_levels = [0.20, 0.10, 0.05]

    for fi, (outer_train, outer_test) in enumerate(folds):
        fold = fi + 1
        if fold in completed:
            LOG.info("Skipping completed outer fold %d", fold)
            continue
        local_fit, local_cal = next(GroupShuffleSplit(
            n_splits=1, test_size=cfg["calibration_split"]["fraction_of_outer_train"],
            random_state=seed + int(cfg["calibration_split"]["seed_offset"]) + fold,
        ).split(X[outer_train], y[outer_train], groups[outer_train]))
        fit_ix, cal_ix = outer_train[local_fit], outer_train[local_cal]
        if set(groups[fit_ix]) & set(groups[cal_ix]) or set(groups[fit_ix]) & set(groups[outer_test]) or set(groups[cal_ix]) & set(groups[outer_test]):
            raise AssertionError("Scaffold leakage across fit/calibration/test partitions")
        LOG.info("Fold %d/%d: fit=%d cal=%d test=%d scaffolds=%d/%d/%d",
                 fold, len(folds), len(fit_ix), len(cal_ix), len(outer_test),
                 len(set(groups[fit_ix])), len(set(groups[cal_ix])), len(set(groups[outer_test])))
        started = time.perf_counter()
        stack, names = fit_stack(X[fit_ix], y[fit_ix], groups[fit_ix], seed + fold)
        _, cal_pred, cal_sd = predict_components(stack, names, X[cal_ix])
        test_base, test_pred, test_sd = predict_components(stack, names, X[outer_test])
        cal_abs = np.abs(y[cal_ix] - cal_pred)
        test_abs = np.abs(y[outer_test] - test_pred)
        cal_maxsim = max_morgan_similarity([fps[i] for i in cal_ix], [fps[i] for i in fit_ix])
        test_maxsim = max_morgan_similarity([fps[i] for i in outer_test], [fps[i] for i in fit_ix])
        cal_ad_dist = 1.0 - cal_maxsim
        test_ad_dist = 1.0 - test_maxsim
        cal_scaffold_scores = pd.DataFrame({"scaffold": groups[cal_ix], "abs_error": cal_abs}).groupby("scaffold").abs_error.max()
        test_scaffold_max = pd.DataFrame({"scaffold": groups[outer_test], "abs_error": test_abs}).groupby("scaffold").abs_error.max()
        q_mol = {a: conformal_quantile(cal_abs, a) for a in alpha_levels}
        q_scaf = {a: conformal_quantile(cal_scaffold_scores.to_numpy(), a) for a in alpha_levels}
        cal_rows.extend([
            {"outer_fold": fold, "method": "molecule", "alpha": a,
             "nominal_coverage": 1-a, "calibration_units": len(cal_abs), "half_width": q}
            for a, q in q_mol.items()
        ])
        cal_rows.extend([
            {"outer_fold": fold, "method": "scaffold_maximum", "alpha": a,
             "nominal_coverage": 1-a, "calibration_units": len(cal_scaffold_scores), "half_width": q}
            for a, q in q_scaf.items()
        ])

        # Percentile ranks are used only to compare diagnostic ranking quality; no cutoff is selected.
        test_combined = (
            np.searchsorted(np.sort(cal_sd), test_sd, side="right") / len(cal_sd)
            + np.searchsorted(np.sort(cal_ad_dist), test_ad_dist, side="right") / len(cal_ad_dist)
        ) / 2
        for j, ix in enumerate(outer_test):
            row = {
                "outer_fold": fold, "row_index": int(ix),
                "canonical_smiles": df.canonical_smiles.iloc[ix], "scaffold": groups[ix],
                "observed_pIC50": float(y[ix]), "stack_prediction": float(test_pred[j]),
                "absolute_error": float(test_abs[j]), "disagreement_sd": float(test_sd[j]),
                "max_train_morgan_similarity": float(test_maxsim[j]),
                "applicability_domain_distance": float(test_ad_dist[j]),
                "combined_uncertainty_rank": float(test_combined[j]),
                "molecule_90_half_width": q_mol[0.10], "scaffold_90_half_width": q_scaf[0.10],
                "molecule_90_covered": bool(test_abs[j] <= q_mol[0.10]),
                "scaffold_90_covered": bool(test_scaffold_max.loc[groups[ix]] <= q_scaf[0.10]),
            }
            for name_i, name in enumerate(names):
                row[f"prediction_{name}"] = float(test_base[j, name_i])
            for a in alpha_levels:
                suffix = int(round((1-a) * 100))
                row[f"molecule_{suffix}_lower"] = float(test_pred[j] - q_mol[a])
                row[f"molecule_{suffix}_upper"] = float(test_pred[j] + q_mol[a])
                row[f"scaffold_{suffix}_lower"] = float(test_pred[j] - q_scaf[a])
                row[f"scaffold_{suffix}_upper"] = float(test_pred[j] + q_scaf[a])
            pred_rows.append(row)

        for name_i, name in enumerate(names):
            metric_rows.append({"outer_fold": fold, "model": name,
                                **metric_set(y[outer_test], test_base[:, name_i])})
        metric_rows.append({"outer_fold": fold, "model": "Stacking",
                            **metric_set(y[outer_test], test_pred)})
        fold_records.append({
            "fold": fold, "seed": seed, "fit_indices": fit_ix.tolist(),
            "calibration_indices": cal_ix.tolist(), "test_indices": outer_test.tolist(),
            "fit_scaffolds": int(len(set(groups[fit_ix]))),
            "calibration_scaffolds": int(len(set(groups[cal_ix]))),
            "test_scaffolds": int(len(set(groups[outer_test]))),
            "scaffold_overlap": 0,
            "molecule_interval_half_widths": {str(1-a): q for a, q in q_mol.items()},
            "scaffold_interval_half_widths": {str(1-a): q for a, q in q_scaf.items()},
            "fit_seconds": time.perf_counter() - started,
        })
        pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
        pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
        pd.DataFrame(cal_rows).to_csv(cal_path, index=False)
        manifest_path.write_text(json.dumps(fold_records), encoding="utf-8")
        LOG.info("Completed fold %d in %.1f seconds", fold, time.perf_counter() - started)

    pred = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows)
    if pred.row_index.nunique() != len(df) or len(pred) != len(df):
        raise AssertionError("Outer-fold predictions must cover each structure exactly once")
    pred.to_csv(out / "predictions" / "oof_predictions.csv", index=False)
    metrics.to_csv(out / "metrics" / "fold_metrics.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / "metrics" / "calibration.csv", index=False)
    (out / "manifests" / "folds.json").write_text(json.dumps(fold_records, indent=2), encoding="utf-8")

    pooled_metrics = []
    for name in ["Stacking"] + list(base_models(seed)):
        pred_col = "stack_prediction" if name == "Stacking" else f"prediction_{name}"
        pooled_metrics.append({"model": name, **metric_set(pred.observed_pIC50, pred[pred_col])})
    pooled = pd.DataFrame(pooled_metrics)
    pooled.to_csv(out / "metrics" / "pooled_oof_metrics.csv", index=False)
    score_names = ["disagreement_sd", "applicability_domain_distance", "combined_uncertainty_rank"]
    reliability = pd.concat([reliability_bins(pred, score) for score in score_names], ignore_index=True)
    reliability.to_csv(out / "metrics" / "uncertainty_reliability_bins.csv", index=False)
    risk = precision_coverage(pred, score_names, cfg["precision_coverage"]["retained_fraction"])
    risk.to_csv(out / "metrics" / "precision_coverage.csv", index=False)
    interval_rows = []
    for method in ["molecule", "scaffold"]:
        for nominal in [80, 90, 95]:
            lo, hi = f"{method}_{nominal}_lower", f"{method}_{nominal}_upper"
            covered = (pred.observed_pIC50 >= pred[lo]) & (pred.observed_pIC50 <= pred[hi])
            group_coverage = pd.DataFrame({"scaffold": pred.scaffold, "covered": covered}).groupby("scaffold").covered.all()
            interval_rows.append({
                "method": method, "nominal_coverage": nominal / 100,
                "empirical_molecule_coverage": float(covered.mean()),
                "mean_interval_width": float((pred[hi] - pred[lo]).mean()),
                "scaffolds_with_all_molecules_covered": int(group_coverage.sum()),
                "n_test_scaffolds": int(len(group_coverage)),
                "empirical_whole_scaffold_containment": float(group_coverage.mean()),
            })
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(out / "metrics" / "interval_coverage.csv", index=False)
    correlations = {
        score: float(pred[score].corr(pred.absolute_error, method="spearman"))
        for score in score_names
    }
    tolerance = {
        str(t): float(np.mean(pred.absolute_error <= t))
        for t in cfg["precision_coverage"]["descriptive_absolute_error_tolerances_pIC50"]
    }
    ci = scaffold_bootstrap(pred)
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    precision_10 = risk[np.isclose(risk.target_retained_fraction.astype(float), 0.10)].to_dict(orient="records")
    combined_10 = next(row for row in precision_10 if row["uncertainty_score"] == "combined_uncertainty_rank")
    report = {
        "experiment": "04 reliability, applicability domain, intervals, and precision-coverage",
        "run_id": run_id, "dataset_rows": len(df), "dataset_sha256": data_hash,
        "outer_split": cfg["outer_split"], "fit_calibration_test_scaffold_disjoint": True,
        "pooled_oof_metrics": pooled.to_dict(orient="records"),
        "spearman_score_vs_absolute_error": correlations,
        "descriptive_full_coverage_error_tolerances": tolerance,
        "interval_coverage": intervals.to_dict(orient="records"),
        "precision_coverage_at_10_percent_retained": precision_10,
        "scaffold_cluster_bootstrap_95CI": ci,
        "selected_pseudolabel_threshold": None,
        "limitations": "Internal scaffold CV on a dataset whose earlier model/test results have been inspected; no independent external cohort or assay metadata. Disagreement and similarity are diagnostics rather than probabilities. Empirical coverage does not establish calibration on generated molecules or under scaffold distribution shift. No cutoff was selected from these curves.",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "rdkit": rdkit.__version__, "xgboost": xgboost.__version__},
    }
    (out / "reports" / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    md = ["# Experiment 4 results", "", f"Run: `{run_id}`", "",
          f"Dataset: {len(df):,} curated structures; SHA-256 `{data_hash}`.", "",
          "Evaluation: five-fold scaffold-grouped OOF predictions. Each fold also has a scaffold-separated fit, calibration, and outer-test partition.", "",
          "## Pooled model performance", "", "| Model | n | MAE | RMSE | R² |", "|---|---:|---:|---:|---:|"]
    for _, row in pooled.iterrows():
        md.append(f"| {row.model} | {int(row.n)} | {row.MAE:.4f} | {row.RMSE:.4f} | {row.R2:.4f} |")
    md += ["", "## Reliability and intervals", "",
           f"Spearman association between absolute error and disagreement: {correlations['disagreement_sd']:.3f}; AD distance: {correlations['applicability_domain_distance']:.3f}; combined rank: {correlations['combined_uncertainty_rank']:.3f}.", "",
           "See `../metrics/uncertainty_reliability_bins.csv` and `../metrics/precision_coverage.csv` for decile reliability and retained-molecule error summaries.", "",
           "| Interval | Nominal | Molecule coverage | Whole-scaffold containment | Mean width (pIC50) |", "|---|---:|---:|---:|---:|"]
    for _, row in intervals.iterrows():
        md.append(f"| {row.method} | {row.nominal_coverage:.0%} | {row.empirical_molecule_coverage:.3f} | {row.empirical_whole_scaffold_containment:.3f} | {row.mean_interval_width:.3f} |")
    md += ["", "The primary molecule-level 90% interval has cluster-bootstrap coverage CI "
           f"{ci['molecule_90_coverage']['cluster_bootstrap_95CI'][0]:.3f}–{ci['molecule_90_coverage']['cluster_bootstrap_95CI'][1]:.3f}; scaffold-max interval whole-scaffold containment CI "
           f"{ci['scaffold_90_full_containment']['cluster_bootstrap_95CI'][0]:.3f}–{ci['scaffold_90_full_containment']['cluster_bootstrap_95CI'][1]:.3f}.", "",
           "Among the lowest-uncertainty 10% by combined calibration-referenced rank, RMSE was "
           f"{combined_10['RMSE']:.3f} for n={combined_10['n_retained']} (compare pooled RMSE {pooled.loc[pooled.model == 'Stacking', 'RMSE'].iloc[0]:.3f}). This is a descriptive ranking result, not an acceptance rule.", "",
           f"Across all OOF predictions, fraction within ±0.5 pIC50: {tolerance['0.5']:.3f}; within ±1.0 pIC50: {tolerance['1.0']:.3f}. These are descriptive regression tolerances, not activity or pseudolabel cutoffs.", "",
           report["limitations"], ""]
    (out / "reports" / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({
        "status": "complete", "run_id": run_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outer_folds_completed": len(fold_records), "outer_oof_predictions": len(pred),
    }, indent=2), encoding="utf-8")
    LOG.info("Experiment 4 complete: %s", out)
    print(out)


if __name__ == "__main__":
    main()
