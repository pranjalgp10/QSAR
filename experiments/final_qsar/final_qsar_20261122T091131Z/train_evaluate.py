"""Final QSAR internal evaluation, median-split task and controlled deployment fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from data_utils import DATA, EXPECTED_SHA256, load_dataset, max_train_similarity, median_split, sha256
from evaluation_metrics import (classification_metrics, conformal_quantile,
                                regression_metrics, scaffold_bootstrap)
from modeling import (BASE_NAMES, STACK_SPECS, fit_regression_ensemble,
                      make_direct_classifiers, make_stack_predictions,
                      raw_classifier_score)

RUN_DIR = Path(__file__).resolve().parent
CFG_PATH = RUN_DIR / "config.json"
LOG = logging.getLogger("final_qsar")


def logit_score(values, model_name):
    z = np.asarray(values, float)
    if model_name in {"RandomForestClassifier", "XGBoostClassifier"}:
        z = np.log(np.clip(z, 1e-6, 1 - 1e-6) / np.clip(1 - z, 1e-6, 1))
    return z


def metric_record(repeat, fold, task, model, values):
    return {"repeat": repeat, "fold": fold, "task": task, "model": model, **values}


def scaffold_overlap_check(groups, fit_ix, cal_ix, test_ix):
    sets = [set(groups[ix]) for ix in (fit_ix, cal_ix, test_ix)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Scaffold overlap across fit/calibration/test partitions")
    if len(np.unique(np.concatenate([fit_ix, cal_ix, test_ix]))) != len(groups):
        raise AssertionError("Fold partitions do not cover each row exactly once")


def percentile_rank(reference, values):
    sorted_reference = np.sort(np.asarray(reference, float))
    return np.searchsorted(sorted_reference, np.asarray(values, float), side="right") / len(sorted_reference)


def fit_platt(raw_cal, y_cal):
    if len(np.unique(y_cal)) < 2:
        raise ValueError("Calibration subset contains one class; cannot calibrate direct classifier")
    scaler = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    scaler.fit(np.asarray(raw_cal).reshape(-1, 1), np.asarray(y_cal, int))
    return scaler


def write_confusion_matrices(predictions):
    rows = []
    for model in ["regression_derived"] + [
        "RandomForestClassifier", "XGBoostClassifier", "LogisticRegression"]:
        if model == "regression_derived":
            score_col, pred_col = "primary_prediction", "regression_derived_active"
        else:
            score_col, pred_col = f"prob_{model}", f"active_{model}"
        y = predictions.observed_active.astype(int).to_numpy()
        p = predictions[pred_col].astype(int).to_numpy()
        met = classification_metrics(y, predictions[score_col].to_numpy(float),
                                     probability=None if model == "regression_derived" else predictions[score_col].to_numpy(float),
                                     hard=p)
        for key in ["TN", "FP", "FN", "TP"]:
            rows.append({"model": model, "cell": key, "count": met[key]})
    pd.DataFrame(rows).to_csv(RUN_DIR / "confusion_matrix.csv", index=False)


def evaluate_filter_curves(pred):
    strategies = {
        "no_filter": None,
        "AD_only": "ad_distance",
        "uncertainty_only": "disagreement_sd",
        "combined_AD_uncertainty": "combined_rank",
    }
    rows = []
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for strategy, score in strategies.items():
        ordered = pred if score is None else pred.sort_values(score, ascending=True)
        for fraction in fractions:
            n = max(1, int(np.ceil(fraction * len(ordered))))
            sample = ordered.iloc[:n]
            y = sample.observed_pIC50.to_numpy(float)
            p = sample.primary_prediction.to_numpy(float)
            labels = sample.observed_active.astype(int).to_numpy()
            hard = sample.regression_derived_active.astype(int).to_numpy()
            cls = classification_metrics(labels, p, hard=hard)
            rows.append({
                "strategy": strategy, "target_retained_fraction": fraction,
                "actual_retained_fraction": n / len(pred), "n_retained": n,
                "regression_MAE": float(np.mean(np.abs(p-y))),
                "regression_RMSE": float(np.sqrt(np.mean((p-y)**2))),
                "active_precision": cls["active_precision"],
                "inactive_precision": cls["inactive_precision"],
                "false_active_rate": cls["false_active_rate"],
                "false_inactive_rate": cls["false_inactive_rate"],
                "accuracy": cls["accuracy"], "balanced_accuracy": cls["balanced_accuracy"],
                "unique_scaffolds_retained": int(sample.scaffold.nunique()),
            })
    return pd.DataFrame(rows)


def run_y_scrambling(X, y, groups, cfg, true_hgb_rmse):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = []
    perms = int(cfg["y_scrambling"]["permutations"])
    nfold = int(cfg["y_scrambling"]["scaffold_folds"])
    for perm in range(perms):
        seed = int(cfg["y_scrambling"]["seed"]) + perm
        shuffled = np.random.default_rng(seed).permutation(y)
        oof = np.full(len(y), np.nan)
        splits = GroupKFold(n_splits=nfold, shuffle=True, random_state=seed)
        for fold, (itr, ite) in enumerate(splits.split(X, shuffled, groups), 1):
            if set(groups[itr]) & set(groups[ite]):
                raise AssertionError("Scaffold leakage in Y-scrambling")
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=15, learning_rate=0.04,
                l2_regularization=2.0, random_state=seed + fold)
            model.fit(X[itr], shuffled[itr])
            oof[ite] = model.predict(X[ite])
        rmse = float(np.sqrt(np.mean((oof - shuffled) ** 2)))
        out.append({"permutation": perm + 1, "seed": seed, "RMSE": rmse,
                    "R2": float(1 - np.sum((oof-shuffled)**2) / np.sum((shuffled-shuffled.mean())**2)),
                    "true_unshuffled_reference_HGB_RMSE": true_hgb_rmse})
        LOG.info("Y-scramble %d/%d RMSE %.4f", perm+1, perms, rmse)
    return pd.DataFrame(out)


def run_learning_curve(X, y, groups, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    seed = int(cfg["outer_cv"]["seeds"][0])
    splits = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fractions = cfg["learning_curve"]["fractions"]
    for fold, (itr, ite) in enumerate(splits.split(X, y, groups), 1):
        train_groups = np.unique(groups[itr])
        for fi, fraction in enumerate(fractions):
            if fraction >= 1:
                subset = itr
            else:
                part, _ = next(GroupShuffleSplit(
                    n_splits=1, test_size=1-fraction,
                    random_state=seed + fold*100 + fi).split(X[itr], y[itr], groups[itr]))
                subset = itr[part]
            if set(groups[subset]) & set(groups[ite]):
                raise AssertionError("Scaffold leakage in learning curve")
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=15, learning_rate=0.04,
                l2_regularization=2.0, random_state=seed + fold + fi)
            model.fit(X[subset], y[subset])
            pred = model.predict(X[ite])
            rows.append({"fold": fold, "training_fraction_target": fraction,
                         "training_rows": len(subset), "training_scaffolds": len(set(groups[subset])),
                         "test_rows": len(ite), **regression_metrics(y[ite], pred)})
        LOG.info("Learning-curve fold %d/%d complete", fold, 5)
    return pd.DataFrame(rows)


def physicochemical_features(mols):
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    return np.asarray([[Descriptors.MolWt(m), Descriptors.MolLogP(m),
                        rdMolDescriptors.CalcTPSA(m), Lipinski.NumHDonors(m),
                        Lipinski.NumHAcceptors(m), Lipinski.NumRotatableBonds(m)]
                       for m in mols], dtype=float)


def run_representation_sensitivity(df, y, groups, mols, cfg):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.ensemble import HistGradientBoostingRegressor
    X3 = np.zeros((len(mols), 2048), dtype=np.uint8)
    gen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    for i, mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(gen3.GetFingerprint(mol), X3[i])
    X2 = np.zeros((len(mols), 2048), dtype=np.uint8)
    gen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for i, mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(gen2.GetFingerprint(mol), X2[i])
    features = {
        "Morgan_radius3": X3,
        "Morgan_radius2_plus_physchem": np.column_stack([X2, physicochemical_features(mols)]),
    }
    rows = []
    seed = int(cfg["outer_cv"]["seeds"][0])
    splits = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, Xalt in features.items():
        for fold, (itr, ite) in enumerate(splits.split(Xalt, y, groups), 1):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_leaf_nodes=15, learning_rate=0.04,
                l2_regularization=2.0, random_state=seed + fold)
            model.fit(Xalt[itr], y[itr])
            rows.append({"representation": name, "fold": fold,
                         **regression_metrics(y[ite], model.predict(Xalt[ite]))})
        LOG.info("Representation sensitivity complete: %s", name)
    return pd.DataFrame(rows)


def save_figures(pred, fold_metrics, lc, y_scramble, ad_analysis, calibration_curve_df):
    import matplotlib.pyplot as plt
    figdir = RUN_DIR / "figures"
    figdir.mkdir(exist_ok=True)
    y = pred.observed_pIC50.to_numpy(float)
    p = pred.primary_prediction.to_numpy(float)
    err = p - y
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(y, p, alpha=.22, s=10); lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); ax.plot([lo,hi],[lo,hi],'k--'); ax.set(xlabel='Observed pIC50',ylabel='Predicted pIC50',title='Scaffold-held-out parity'); fig.tight_layout(); fig.savefig(figdir/'regression_parity.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(p, err, alpha=.22, s=10); ax.axhline(0,color='k',linestyle='--'); ax.set(xlabel='Predicted pIC50',ylabel='Residual (predicted - observed)',title='Residuals vs prediction'); fig.tight_layout(); fig.savefig(figdir/'residual_plot.png',dpi=180); plt.close(fig)
    cm = np.array([[((pred.observed_active==0)&(pred.regression_derived_active==0)).sum(),((pred.observed_active==0)&(pred.regression_derived_active==1)).sum()],[((pred.observed_active==1)&(pred.regression_derived_active==0)).sum(),((pred.observed_active==1)&(pred.regression_derived_active==1)).sum()]])
    fig, ax = plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap='Blues'); ax.set_xticks([0,1],['Inactive','Active']); ax.set_yticks([0,1],['Inactive','Active']); ax.set(xlabel='Predicted',ylabel='Observed',title='Median-split confusion matrix'); [ax.text(j,i,str(cm[i,j]),ha='center',va='center') for i in range(2) for j in range(2)]; fig.tight_layout(); fig.savefig(figdir/'confusion_matrix.png',dpi=180); plt.close(fig)
    from sklearn.metrics import roc_curve, precision_recall_curve
    yt=pred.observed_active.astype(int).to_numpy();
    fig, ax = plt.subplots(figsize=(6,5)); fpr,tpr,_=roc_curve(yt,p); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC — regression-derived score'); fig.tight_layout(); fig.savefig(figdir/'roc_curve.png',dpi=180); plt.close(fig)
    precision,recall,_=precision_recall_curve(yt,p)
    fig, ax = plt.subplots(figsize=(6,5)); ax.plot(recall,precision); ax.set(xlabel='Recall',ylabel='Precision',title='Precision–recall — regression-derived score'); fig.tight_layout(); fig.savefig(figdir/'precision_recall_curve.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5)); ax.plot(calibration_curve_df.mean_probability,calibration_curve_df.observed_fraction,'o-'); ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Mean calibrated probability',ylabel='Observed active fraction',title='Direct-classifier calibration'); fig.tight_layout(); fig.savefig(figdir/'calibration_curve.png',dpi=180); plt.close(fig)
    risk=pd.read_csv(RUN_DIR/'precision_coverage.csv')
    fig, ax = plt.subplots(figsize=(7,5));
    for strategy,part in risk.groupby('strategy'): ax.plot(part.actual_retained_fraction,part.regression_RMSE,marker='o',label=strategy)
    ax.set(xlabel='Retained fraction',ylabel='RMSE',title='Regression error vs retained coverage'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'precision_coverage_curve.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(lc.groupby('training_fraction_target').training_rows.mean(),lc.groupby('training_fraction_target').RMSE.mean(),'o-'); ax.set(xlabel='Mean training rows',ylabel='RMSE',title='Scaffold-aware learning curve'); fig.tight_layout(); fig.savefig(figdir/'learning_curve.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5));
    grouped=ad_analysis.groupby('similarity_bin',observed=True).agg(sim=('mean_similarity','mean'),rmse=('RMSE','mean'))
    ax.plot(grouped.sim,grouped.rmse,'o-'); ax.set(xlabel='Nearest training Morgan similarity',ylabel='RMSE',title='Applicability-domain stress test'); fig.tight_layout(); fig.savefig(figdir/'applicability_domain_performance.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.hist(y_scramble.RMSE,bins=10,alpha=.7); ax.axvline(y_scramble.true_unshuffled_reference_HGB_RMSE.iloc[0],color='red',label='Unshuffled CV RMSE'); ax.set(xlabel='RMSE after label permutation',ylabel='Count',title='Y-scrambling null distribution'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'y_scrambling_distribution.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last completed outer fold")
    args = parser.parse_args()
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if sha256(DATA) != EXPECTED_SHA256:
        raise RuntimeError("Dataset hash differs from frozen expected hash")
    for sub in ["predictions", "metrics", "manifests", "logs", "figures"]:
        (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)
    if not args.resume and (RUN_DIR / "manifests" / "folds_checkpoint.json").exists():
        raise FileExistsError("Run already has checkpoints; pass --resume to continue")
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(RUN_DIR/"logs"/"execution.log",encoding="utf-8"),logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")
    df, y, groups, X, mols, fps = load_dataset()
    repeats = cfg["outer_cv"]["seeds"]
    pred_cp = RUN_DIR/"predictions"/"pooled_oof_predictions_checkpoint.csv"
    fold_cp = RUN_DIR/"metrics"/"fold_metrics_checkpoint.csv"
    threshold_cp = RUN_DIR/"median_thresholds_by_fold.csv"
    labels_cp = RUN_DIR/"classification_label_manifest.csv"
    calibration_cp = RUN_DIR/"calibration_metrics.csv"
    manifest_cp = RUN_DIR/"manifests"/"folds_checkpoint.json"
    pred_rows = pd.read_csv(pred_cp).to_dict("records") if args.resume and pred_cp.exists() else []
    fold_rows = pd.read_csv(fold_cp).to_dict("records") if args.resume and fold_cp.exists() else []
    threshold_rows = pd.read_csv(threshold_cp).to_dict("records") if args.resume and threshold_cp.exists() else []
    label_rows = pd.read_csv(labels_cp).to_dict("records") if args.resume and labels_cp.exists() else []
    cal_rows = pd.read_csv(calibration_cp).to_dict("records") if args.resume and calibration_cp.exists() else []
    fold_records = json.loads(manifest_cp.read_text(encoding="utf-8")) if args.resume and manifest_cp.exists() else []
    done = {f"r{rec['repeat']}_f{rec['fold']}" for rec in fold_records}
    interval_cp = RUN_DIR / "calibration_interval_widths_checkpoint.csv"
    q_intervals = (pd.read_csv(interval_cp).to_dict("records")
                   if args.resume and interval_cp.exists() else [])
    for rep, seed in enumerate(repeats, start=1):
        outer = list(GroupKFold(n_splits=cfg["outer_cv"]["folds"], shuffle=True,
                                random_state=seed).split(X, y, groups))
        for fi, (outer_fit, outer_test) in enumerate(outer, start=1):
            key = f"r{rep}_f{fi}"
            if key in done:
                LOG.info("Skipping checkpointed %s", key)
                continue
            local_fit, local_cal = next(GroupShuffleSplit(
                n_splits=1, test_size=cfg["calibration"]["fraction"],
                random_state=seed + 7000 + fi).split(X[outer_fit], y[outer_fit], groups[outer_fit]))
            fit_ix, cal_ix = outer_fit[local_fit], outer_fit[local_cal]
            scaffold_overlap_check(groups, fit_ix, cal_ix, outer_test)
            median, yfit_binary = median_split(y[fit_ix], y[fit_ix])
            _, ycal_binary = median_split(y[fit_ix], y[cal_ix])
            _, ytest_binary = median_split(y[fit_ix], y[outer_test])
            LOG.info("%s fit=%d cal=%d test=%d scaffolds=%d/%d/%d median=%.5f",
                     key,len(fit_ix),len(cal_ix),len(outer_test),len(set(groups[fit_ix])),len(set(groups[cal_ix])),len(set(groups[outer_test])),median)
            start = time.perf_counter()
            base_models, meta_oof = fit_regression_ensemble(
                X[fit_ix], y[fit_ix], groups[fit_ix], seed + fi + rep*100, cfg["stack_oof_folds"])
            base_cal, stacks_cal, disagreement_cal = make_stack_predictions(
                base_models, meta_oof, y[fit_ix], X[cal_ix])
            base_test, stacks_test, disagreement_test = make_stack_predictions(
                base_models, meta_oof, y[fit_ix], X[outer_test])
            similarity_cal = max_train_similarity([fps[i] for i in cal_ix], [fps[i] for i in fit_ix])
            similarity_test = max_train_similarity([fps[i] for i in outer_test], [fps[i] for i in fit_ix])
            ad_cal, ad_test = 1-similarity_cal, 1-similarity_test
            combined_test = (percentile_rank(disagreement_cal, disagreement_test) +
                             percentile_rank(ad_cal, ad_test)) / 2

            # Absolute-residual intervals use labels in a disjoint scaffold calibration subset.
            primary_cal = stacks_cal[cfg["primary_model"]]["prediction"]
            primary_test = stacks_test[cfg["primary_model"]]["prediction"]
            cal_abs = np.abs(y[cal_ix] - primary_cal)
            scaf_cal = pd.DataFrame({"scaffold": groups[cal_ix], "abs_error": cal_abs}).groupby("scaffold").abs_error.max()
            qmol = {alpha: conformal_quantile(cal_abs, alpha) for alpha in [0.2,0.1,0.05]}
            qscaf = {alpha: conformal_quantile(scaf_cal.to_numpy(), alpha) for alpha in [0.2,0.1,0.05]}
            q_intervals.extend([
                {"repeat":rep,"fold":fi,"method":"molecule","nominal":1-a,"half_width":q}
                for a,q in qmol.items()]+[
                {"repeat":rep,"fold":fi,"method":"scaffold_max","nominal":1-a,"half_width":q}
                for a,q in qscaf.items()])

            fold_predictions = {name: base_test[:,j] for j,name in enumerate(BASE_NAMES)}
            for name, result in stacks_test.items():
                fold_predictions[name] = result["prediction"]
            for name, prediction in fold_predictions.items():
                fold_rows.append(metric_record(rep,fi,"regression",name,regression_metrics(y[outer_test],prediction)))

            # Direct classifiers train on fit-only median labels and use calibration-only Platt scaling.
            direct_predictions = {}
            for cls_name, classifier in make_direct_classifiers(seed+rep*100+fi).items():
                classifier.fit(X[fit_ix], yfit_binary)
                raw_cal = raw_classifier_score(classifier, X[cal_ix])
                raw_test = raw_classifier_score(classifier, X[outer_test])
                calibrator = fit_platt(logit_score(raw_cal, cls_name), ycal_binary)
                pcal = calibrator.predict_proba(logit_score(raw_cal, cls_name).reshape(-1,1))[:,1]
                ptest = calibrator.predict_proba(logit_score(raw_test, cls_name).reshape(-1,1))[:,1]
                direct_predictions[cls_name] = (classifier, calibrator, ptest)
                cm = classification_metrics(ytest_binary, ptest, probability=ptest, hard=(ptest>=.5).astype(int))
                fold_rows.append(metric_record(rep,fi,"direct_binary_classification",cls_name,cm))
                cal_rows.append({"repeat":rep,"fold":fi,"classifier":cls_name,
                                 "calibration_n":len(cal_ix),"calibration_prevalence":float(ycal_binary.mean()),
                                 "calibration_Brier":float(np.mean((pcal-ycal_binary)**2)),
                                 "outer_test_Brier":float(np.mean((ptest-ytest_binary)**2)),
                                 "outer_test_ECE_10bin":classification_metrics(ytest_binary,ptest,probability=ptest,hard=(ptest>=.5).astype(int))["ECE_10bin"],
                                 "platt_slope":float(calibrator.coef_[0,0]),"platt_intercept":float(calibrator.intercept_[0])})

            primary_cls = classification_metrics(ytest_binary, primary_test,
                hard=(primary_test>=median).astype(int))
            fold_rows.append(metric_record(rep,fi,"regression_derived_median_classification","regression_derived",primary_cls))
            threshold_set = {
                "training_median": median,
                "median_minus_0.25": median-0.25,
                "median_plus_0.25": median+0.25,
                "training_q40": float(np.quantile(y[fit_ix],0.4)),
                "training_q60": float(np.quantile(y[fit_ix],0.6)),
            }
            for threshold_name, threshold in threshold_set.items():
                yt = (y[outer_test]>=threshold).astype(int)
                yp = (primary_test>=threshold).astype(int)
                tm = classification_metrics(yt,primary_test,hard=yp)
                threshold_rows.append({"repeat":rep,"fold":fi,"threshold_name":threshold_name,
                    "fit_threshold":threshold,"fit_class_count_active":int((y[fit_ix]>=threshold).sum()),
                    "fit_class_count_inactive":int((y[fit_ix]<threshold).sum()),
                    "test_class_count_active":int(yt.sum()),"test_class_count_inactive":int((1-yt).sum()),**tm})
            for role, ix, labels in [("fit",fit_ix,yfit_binary),("calibration",cal_ix,ycal_binary),("test",outer_test,ytest_binary)]:
                label_rows.extend({"repeat":rep,"fold":fi,"row_index":int(i),"partition":role,
                    "pIC50":float(y[i]),"training_median_threshold":median,
                    "activity_label":"active" if int(label)==1 else "inactive",
                    "tie_at_threshold":bool(np.isclose(y[i],median))}
                    for i,label in zip(ix,labels))

            for j, ix in enumerate(outer_test):
                row = {"repeat":rep,"fold":fi,"row_index":int(ix),
                    "canonical_smiles":df.canonical_smiles.iloc[ix],"scaffold":groups[ix],
                    "scaffold_size_test":int(np.sum(groups[outer_test]==groups[ix])),
                    "observed_pIC50":float(y[ix]),"training_median_threshold":median,
                    "observed_active":int(ytest_binary[j]),
                    "primary_prediction":float(primary_test[j]),
                    "absolute_error":float(abs(primary_test[j]-y[ix])),
                    "regression_derived_active":int(primary_test[j]>=median),
                    "disagreement_sd":float(disagreement_test[j]),
                    "max_train_similarity":float(similarity_test[j]),"ad_distance":float(ad_test[j]),
                    "combined_rank":float(combined_test[j]),
                    "molecule_90_half_width":qmol[0.1],"scaffold_90_half_width":qscaf[0.1],
                    "molecule_90_covered":bool(abs(primary_test[j]-y[ix])<=qmol[0.1]),
                    "scaffold_90_covered":bool(abs(primary_test[j]-y[ix])<=qscaf[0.1])}
                for k,name in enumerate(BASE_NAMES): row[f"pred_{name}"]=float(base_test[j,k])
                for name,result in stacks_test.items(): row[f"pred_{name}"]=float(result["prediction"][j])
                for cls_name,(_,_,probs) in direct_predictions.items():
                    row[f"prob_{cls_name}"]=float(probs[j]);row[f"active_{cls_name}"]=int(probs[j]>=.5)
                pred_rows.append(row)
            fold_records.append({"repeat":rep,"fold":fi,"fold_key":key,"seed":seed,
                "fit_indices":fit_ix.tolist(),"calibration_indices":cal_ix.tolist(),"test_indices":outer_test.tolist(),
                "fit_scaffolds":len(set(groups[fit_ix])),"calibration_scaffolds":len(set(groups[cal_ix])),
                "test_scaffolds":len(set(groups[outer_test])),"scaffold_overlap":0,
                "threshold":median,"training_threshold_label_counts":{"active":int(yfit_binary.sum()),"inactive":int((1-yfit_binary).sum())},
                "fit_seconds":time.perf_counter()-start})
            pd.DataFrame(pred_rows).to_csv(pred_cp,index=False)
            pd.DataFrame(fold_rows).to_csv(fold_cp,index=False)
            pd.DataFrame(threshold_rows).to_csv(threshold_cp,index=False)
            pd.DataFrame(label_rows).to_csv(labels_cp,index=False)
            pd.DataFrame(cal_rows).to_csv(calibration_cp,index=False)
            pd.DataFrame(q_intervals).to_csv(RUN_DIR/"calibration_interval_widths_checkpoint.csv",index=False)
            manifest_cp.write_text(json.dumps(fold_records),encoding="utf-8")
            done.add(key)
            LOG.info("%s complete in %.1fs",key,time.perf_counter()-start)

    pred = pd.DataFrame(pred_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    if len(pred) != len(df)*len(repeats) or pred.groupby(["repeat","row_index"]).size().max()!=1:
        raise AssertionError("OOF predictions do not cover each row exactly once per repeat")
    pred.to_csv(RUN_DIR/"pooled_oof_predictions.csv",index=False)
    fold_metrics.to_csv(RUN_DIR/"fold_metrics.csv",index=False)
    pd.DataFrame(threshold_rows).to_csv(RUN_DIR/"median_thresholds_by_fold.csv",index=False)
    pd.DataFrame(label_rows).to_csv(RUN_DIR/"classification_label_manifest.csv",index=False)
    pd.DataFrame(cal_rows).to_csv(RUN_DIR/"calibration_metrics.csv",index=False)
    pd.DataFrame(q_intervals).to_csv(RUN_DIR/"calibration_interval_widths.csv",index=False)
    manifest_cp.write_text(json.dumps(fold_records,indent=2),encoding="utf-8")

    # Pooled regression metrics and confidence intervals.
    reg_models = BASE_NAMES + list(STACK_SPECS)
    reg_rows, boot_rows = [], []
    for model in reg_models:
        col = "primary_prediction" if model==cfg["primary_model"] else f"pred_{model}"
        reg_rows.append({"model":model,**regression_metrics(pred.observed_pIC50,pred[col])})
        ci_fn=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
        ci=scaffold_bootstrap(pred.observed_pIC50.to_numpy(),pred[col].to_numpy(),pred.scaffold.to_numpy(),ci_fn,seed=140)
        boot_rows.append({"task":"regression","model":model,"metric":"RMSE","lower_95":ci[0],"upper_95":ci[1],"cluster":"Bemis-Murcko scaffold"})
    pd.DataFrame(reg_rows).to_csv(RUN_DIR/"regression_metrics.csv",index=False)

    # Pooled classification metrics and confusion matrix.
    class_rows=[]; confusion=[]
    for model in ["regression_derived"]+list(make_direct_classifiers(140)):
        if model=="regression_derived":
            score=pred.primary_prediction.to_numpy(float);hard=pred.regression_derived_active.to_numpy(int);prob=None
        else:
            score=pred[f"prob_{model}"].to_numpy(float);hard=pred[f"active_{model}"].to_numpy(int);prob=score
        met=classification_metrics(pred.observed_active.to_numpy(int),score,probability=prob,hard=hard)
        class_rows.append({"model":model,**met})
        for cell in ["TN","FP","FN","TP"]: confusion.append({"model":model,"cell":cell,"count":met[cell]})
    pd.DataFrame(class_rows).to_csv(RUN_DIR/"classification_metrics.csv",index=False)
    pd.DataFrame(confusion).to_csv(RUN_DIR/"confusion_matrix.csv",index=False)
    for model in ["regression_derived"]+list(make_direct_classifiers(140)):
        scorecol="primary_prediction" if model=="regression_derived" else f"prob_{model}"
        hardcol="regression_derived_active" if model=="regression_derived" else f"active_{model}"
        ytrue=pred.observed_active.to_numpy(int); yhat=pred[hardcol].to_numpy(int)
        for name,fn in [("accuracy",lambda a,b:float(np.mean(a==b))),
                        ("balanced_accuracy",lambda a,b:float(__import__('sklearn.metrics').metrics.balanced_accuracy_score(a,b))),
                        ("active_precision",lambda a,b:float(np.sum((a==1)&(b==1))/max(1,np.sum(b==1)))),
                        ("inactive_precision",lambda a,b:float(np.sum((a==0)&(b==0))/max(1,np.sum(b==0)))),
                        ("active_recall",lambda a,b:float(np.sum((a==1)&(b==1))/max(1,np.sum(a==1)))),
                        ("specificity",lambda a,b:float(np.sum((a==0)&(b==0))/max(1,np.sum(a==0)))),
                        ("MCC",lambda a,b:float(__import__('sklearn.metrics').metrics.matthews_corrcoef(a,b)))]:
            ci=scaffold_bootstrap(ytrue,yhat,pred.scaffold.to_numpy(),lambda a,b:fn(a,b),seed=141)
            boot_rows.append({"task":"classification","model":model,"metric":name,"lower_95":ci[0],"upper_95":ci[1],"cluster":"Bemis-Murcko scaffold"})
    pd.DataFrame(boot_rows).to_csv(RUN_DIR/"scaffold_bootstrap_intervals.csv",index=False)

    # Error, uncertainty, applicability, risk/coverage and threshold analyses.
    pred.to_csv(RUN_DIR/"pooled_oof_predictions.csv",index=False)
    risk=evaluate_filter_curves(pred);risk.to_csv(RUN_DIR/"precision_coverage.csv",index=False)
    corr_rows=[]
    for score in ["disagreement_sd","ad_distance","combined_rank"]:
        corr_rows.append({"score":score,"spearman_abs_regression_error":float(spearmanr(pred[score],pred.absolute_error).statistic),
            "spearman_binary_error":float(spearmanr(pred[score],(pred.observed_active!=pred.regression_derived_active).astype(int)).statistic)})
    pd.DataFrame(corr_rows).to_csv(RUN_DIR/"uncertainty_analysis.csv",index=False)
    pred["similarity_bin"]=pd.qcut(pred.max_train_similarity,q=5,duplicates="drop")
    ad=[]
    for bin_value,part in pred.groupby("similarity_bin",observed=True):
        yhat=part.primary_prediction.to_numpy(float); yv=part.observed_pIC50.to_numpy(float)
        cls=classification_metrics(part.observed_active.astype(int),yhat,hard=part.regression_derived_active.astype(int))
        ad.append({"similarity_bin":str(bin_value),"n":len(part),"mean_similarity":float(part.max_train_similarity.mean()),
            "mean_ad_distance":float(part.ad_distance.mean()),**regression_metrics(yv,yhat),
            "binary_accuracy":cls["accuracy"],"active_precision":cls["active_precision"],"inactive_precision":cls["inactive_precision"],
            "false_active_rate":cls["false_active_rate"],"false_inactive_rate":cls["false_inactive_rate"],"scaffolds":part.scaffold.nunique()})
    ad_df=pd.DataFrame(ad);ad_df.to_csv(RUN_DIR/"applicability_domain_analysis.csv",index=False)
    sens=pd.DataFrame(threshold_rows);sens.to_csv(RUN_DIR/"threshold_sensitivity.csv",index=False)
    dist_bins=pd.cut(np.abs(pred.observed_pIC50-pred.training_median_threshold),bins=[-1e-8,.25,.5,np.inf],labels=["<=0.25","0.25-0.5",">0.5"])
    near=[]
    for b in dist_bins.cat.categories:
        part=pred.loc[dist_bins==b]
        if not len(part): continue
        near.append({"distance_from_fold_median_bin":str(b),"n":len(part),"accuracy":float(np.mean(part.observed_active==part.regression_derived_active)),
                     "active_precision":float(np.mean(part.loc[part.regression_derived_active==1,"observed_active"])) if (part.regression_derived_active==1).any() else np.nan})
    pd.DataFrame(near).to_csv(RUN_DIR/"median_margin_performance.csv",index=False)
    by_scaffold=pred.groupby("scaffold").agg(n=("row_index","size"),observed_mean=("observed_pIC50","mean"),predicted_mean=("primary_prediction","mean"),MAE=("absolute_error","mean"),mean_similarity=("max_train_similarity","mean"),disagreement=("disagreement_sd","mean"),label_error=("observed_active",lambda x:float(np.mean(x.to_numpy()!=pred.loc[x.index,"regression_derived_active"].to_numpy())))).reset_index().sort_values("MAE",ascending=False)
    by_scaffold.to_csv(RUN_DIR/"error_by_scaffold.csv",index=False)
    pred.sort_values("absolute_error",ascending=False).head(100).to_csv(RUN_DIR/"error_analysis.csv",index=False)

    # Calibration and interval coverage on all held-out predictions.
    interval_rows=[]
    for method, qcol in [("molecule","molecule_90_half_width"),("scaffold_max","scaffold_90_half_width")]:
        covered=pred.absolute_error<=pred[qcol]
        group_cover=pred.assign(covered=covered).groupby(["repeat","scaffold"]).covered.all()
        ci=scaffold_bootstrap(covered.astype(int).to_numpy(),covered.astype(int).to_numpy(),pred.scaffold.to_numpy(),
                              lambda a,b:float(np.mean(a)),seed=814)
        interval_rows.append({"method":method,"nominal_coverage":.90,"empirical_molecule_coverage":float(covered.mean()),
            "mean_width":float(2*pred[qcol].mean()),"whole_scaffold_containment":float(group_cover.mean()),
            "coverage_scaffold_bootstrap_lower":ci[0],"coverage_scaffold_bootstrap_upper":ci[1]})
    pd.DataFrame(interval_rows).to_csv(RUN_DIR/"interval_coverage.csv",index=False)
    direct_prob_cols=[f"prob_{m}" for m in make_direct_classifiers(140)]
    calibration_curve_rows=[]
    for col in direct_prob_cols:
        temp=pred[[col,"observed_active"]].copy();temp["bin"]=pd.qcut(temp[col].rank(method="first"),q=10,labels=False,duplicates="drop")
        for bin_id,part in temp.groupby("bin"):
            calibration_curve_rows.append({"classifier":col[5:],"bin":int(bin_id),"n":len(part),"mean_probability":float(part[col].mean()),"observed_fraction":float(part.observed_active.mean())})
    cal_curve=pd.DataFrame(calibration_curve_rows);cal_curve.to_csv(RUN_DIR/"calibration_curve.csv",index=False)

    # Predeclared reduced null, learning-curve and representation controls.
    true_hgb=float(pred["pred_HistGradientBoosting"].sub(pred.observed_pIC50).pow(2).mean()**.5)
    yscr=run_y_scrambling(X,y,groups,cfg,true_hgb);yscr.to_csv(RUN_DIR/"y_scrambling_results.csv",index=False)
    learning=run_learning_curve(X,y,groups,cfg);learning.to_csv(RUN_DIR/"learning_curves.csv",index=False)
    reps=run_representation_sensitivity(df,y,groups,mols,cfg);reps.to_csv(RUN_DIR/"representation_sensitivity.csv",index=False)

    # Final all-data fit after evaluation and predeclared architecture decision.
    threshold=float(np.median(y)); tie_count=int(np.isclose(y,threshold).sum())
    final_models, final_oof=fit_regression_ensemble(X,y,groups,seed=1717,oof_folds=5)
    final_eval, final_stacks, final_disagreement=make_stack_predictions(final_models,final_oof,y,X[:1])
    primary_members, primary_alpha=STACK_SPECS[cfg["primary_model"]]
    from sklearn.linear_model import Ridge
    cols=[BASE_NAMES.index(member) for member in primary_members]
    final_meta=Ridge(alpha=primary_alpha).fit(final_oof[:,cols],y)
    # Cross-fitted residual quantile is a heuristic for the full-data refit, not a finite-sample guarantee.
    q90=conformal_quantile(pred.absolute_error,0.10)
    q90_scaffold=conformal_quantile(pred.groupby("scaffold").absolute_error.max().to_numpy(),0.10)
    reference_bits=np.packbits(X.astype(np.uint8),axis=1)
    bundle={"base_models":final_models,"base_names":BASE_NAMES,"meta_model":final_meta,
        "meta_members":primary_members,"meta_alpha":primary_alpha,"radius":2,"bits":2048,
        "activity_threshold":threshold,"molecule_interval_half_width":q90,
        "scaffold_interval_half_width":q90_scaffold,"reference_fingerprints_packed":reference_bits,
        "disagreement_reference":pred.disagreement_sd.to_numpy(float),
        "ad_distance_reference":pred.ad_distance.to_numpy(float),
        "dataset_sha256":EXPECTED_SHA256,"calibration_note":"90% interval half-width derived from repeated scaffold OOF residuals; empirical internal coverage, no guarantee on generated molecules.",
        "reliability_status_policy":"caution for every valid input until prospective pseudolabel precision/coverage operating criterion is approved"}
    model_path=RUN_DIR/"model.joblib";joblib.dump(bundle,model_path,compress=3)
    threshold_out={"definition":"median pIC50 of full approved development dataset","threshold_pIC50":threshold,
        "active_rule":"pIC50 >= threshold","inactive_rule":"pIC50 < threshold","active_count":int((y>=threshold).sum()),
        "inactive_count":int((y<threshold).sum()),"ties_at_threshold":tie_count,"n":len(y),
        "interpretation":"Relative dataset-dependent VECTOR+-compatible split, not a universal potency threshold."}
    (RUN_DIR/"activity_threshold.json").write_text(json.dumps(threshold_out,indent=2),encoding="utf-8")
    (RUN_DIR/"final_activity_threshold.json").write_text(json.dumps(threshold_out,indent=2),encoding="utf-8")
    metadata={"model":"Five-model Ridge stack alpha=1","architecture_selection":"predeclared from Experiment 1 repeated-fold stability; not selected by outer metrics in this run",
        "dataset_sha256":EXPECTED_SHA256,"n_training":len(y),"fingerprint":{"type":"Morgan","radius":2,"bits":2048},
        "threshold":threshold_out,"intervals":{"molecule_q90_half_width":q90,"scaffold_q90_half_width":q90_scaffold,
        "source":"repeated scaffold OOF residuals; approximate for full refit"},"direct_classifier_retained":False,
        "reliability_status":"all valid molecules return caution; no validated pseudolabel acceptance threshold was established",
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"rdkit":rdkit.__version__,"xgboost":xgboost.__version__},
        "training_seed":1717}
    (RUN_DIR/"model_metadata.json").write_text(json.dumps(metadata,indent=2,default=float),encoding="utf-8")
    runmanifest={"run_id":RUN_DIR.name,"completed_utc":datetime.now(timezone.utc).isoformat(),
        "status":"evaluation_complete_internal_only","dataset_sha256":EXPECTED_SHA256,
        "outer_folds":len(repeats)*cfg["outer_cv"]["folds"],"oof_predictions":len(pred),
        "model_path":"model.joblib","not_external_validation":True,"configs":"config.json"}
    (RUN_DIR/"run_manifest.json").write_text(json.dumps(runmanifest,indent=2),encoding="utf-8")
    pd.DataFrame(q_intervals).to_csv(RUN_DIR/"calibration_intervals_by_fold.csv",index=False)
    calibration_table= pd.DataFrame(cal_rows)
    calibration_table.to_csv(RUN_DIR/"calibration_metrics.csv",index=False)
    save_figures(pred,fold_metrics,learning,yscr,ad_df,cal_curve)
    LOG.info("Training/evaluation complete in %s",RUN_DIR)


if __name__ == "__main__":
    main()
