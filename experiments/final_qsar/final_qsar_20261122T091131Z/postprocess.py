"""Create clustered intervals and detailed error tables from final OOF files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata, spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors

from data_utils import DATA, load_dataset
from evaluation_metrics import regression_metrics, classification_metrics

RUN = Path(__file__).resolve().parent


def clustered_mean_ci(values, groups, seed=404, n_boot=2000):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        selected = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(selected, minlength=len(unique))[inverse]
        means[b] = np.average(values, weights=weights)
    return np.quantile(means, [0.025, 0.975]).tolist()


def cluster_bootstrap_weights(groups, n_boot, seed):
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    cluster_draws = np.random.default_rng(seed).multinomial(
        len(unique), np.full(len(unique), 1.0 / len(unique)), size=n_boot)
    return cluster_draws[:, inverse].astype(np.int16)


def classification_from_counts(tn, fp, fn, tp):
    specificity = tn / (tn + fp) if tn + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    active_precision = tp / (tp + fp) if tp + fp else 0.0
    inactive_precision = tn / (tn + fn) if tn + fn else 0.0
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn-fp*fn)/denom if denom else 0.0
    accuracy = (tn+tp)/(tn+fp+fn+tp) if tn+fp+fn+tp else 0.0
    return {"accuracy":accuracy,"balanced_accuracy":(sensitivity+specificity)/2,
            "active_precision":active_precision,"inactive_precision":inactive_precision,
            "sensitivity":sensitivity,"specificity":specificity,"MCC":mcc}


def weighted_regression_metrics(y, p, weights):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rank_y, rank_p = rankdata(y, method="average"), rankdata(p, method="average")
    out = {key: [] for key in ["MAE","RMSE","R2","median_absolute_error","Pearson_r","Spearman_r"]}
    abs_err = np.abs(y-p); sq_err=(y-p)**2
    for w in weights:
        total=float(w.sum())
        mean_y=float(np.dot(w,y)/total); mean_p=float(np.dot(w,p)/total)
        mae=float(np.dot(w,abs_err)/total); mse=float(np.dot(w,sq_err)/total)
        variance=float(np.dot(w,(y-mean_y)**2)/total)
        covariance=float(np.dot(w,(y-mean_y)*(p-mean_p))/total)
        pvar=float(np.dot(w,(p-mean_p)**2)/total)
        sorted_error=np.argsort(abs_err); cum=np.cumsum(w[sorted_error])
        median_abs=float(abs_err[sorted_error[min(np.searchsorted(cum,total/2,side="left"),len(y)-1)]])
        rank_y_mean=float(np.dot(w,rank_y)/total); rank_p_mean=float(np.dot(w,rank_p)/total)
        rank_cov=float(np.dot(w,(rank_y-rank_y_mean)*(rank_p-rank_p_mean))/total)
        rank_y_var=float(np.dot(w,(rank_y-rank_y_mean)**2)/total); rank_p_var=float(np.dot(w,(rank_p-rank_p_mean)**2)/total)
        out["MAE"].append(mae); out["RMSE"].append(float(np.sqrt(mse)))
        out["R2"].append(float(1-mse/variance) if variance else np.nan)
        out["median_absolute_error"].append(median_abs)
        out["Pearson_r"].append(covariance/np.sqrt(variance*pvar) if variance and pvar else np.nan)
        out["Spearman_r"].append(rank_cov/np.sqrt(rank_y_var*rank_p_var) if rank_y_var and rank_p_var else np.nan)
    return out


def save_supplementary_figures(pred):
    import matplotlib.pyplot as plt
    figdir = RUN / "figures"
    figdir.mkdir(exist_ok=True)
    residual = pred.primary_prediction.to_numpy(float) - pred.observed_pIC50.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred.observed_pIC50, residual, alpha=0.22, s=10)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="Observed pIC50", ylabel="Residual (predicted - observed)", title="Residuals vs observed pIC50")
    fig.tight_layout(); fig.savefig(figdir / "residuals_vs_observed.png", dpi=180); plt.close(fig)
    risk = pd.read_csv(RUN / "precision_coverage.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, part in risk.groupby("strategy"):
        ax.plot(part.actual_retained_fraction, part.active_precision, marker="o", label=f"{strategy}: active precision")
        ax.plot(part.actual_retained_fraction, part.inactive_precision, marker=".", linestyle="--", label=f"{strategy}: inactive precision")
    ax.set(xlabel="Retained fraction", ylabel="Class precision", ylim=(0, 1), title="Median-label precision vs retained coverage")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(figdir / "classification_precision_coverage.png", dpi=180); plt.close(fig)


def main():
    pred = pd.read_csv(RUN / "pooled_oof_predictions.csv")
    interval = pd.read_csv(RUN / "calibration_interval_widths.csv")
    df, all_y, _, _, _, _ = load_dataset()
    save_supplementary_figures(pred)
    fold_records = json.loads((RUN / "manifests" / "folds_checkpoint.json").read_text(encoding="utf-8"))
    fold_baselines = []
    for rec in fold_records:
        fit, test = np.asarray(rec["fit_indices"], dtype=int), np.asarray(rec["test_indices"], dtype=int)
        yfit, ytest = all_y[fit], all_y[test]
        median, mean = float(np.median(yfit)), float(np.mean(yfit))
        active_fit, active_test = (yfit >= median).astype(int), (ytest >= median).astype(int)
        majority, prevalence = int(np.bincount(active_fit, minlength=2).argmax()), float(active_fit.mean())
        subset = pred.loc[(pred.repeat == rec["repeat"]) & (pred.fold == rec["fold"])]
        if set(subset.row_index.astype(int)) != set(test.tolist()):
            raise AssertionError("Baseline postprocessing found mismatch between saved test and prediction rows")
        pred.loc[subset.index, "pred_training_median"] = median
        pred.loc[subset.index, "pred_training_mean"] = mean
        pred.loc[subset.index, "majority_class_prediction"] = majority
        pred.loc[subset.index, "dummy_prior_probability"] = prevalence
        pred.loc[subset.index, "dummy_prior_prediction"] = int(prevalence >= 0.5)
        fold_baselines.extend([
            {"repeat": rec["repeat"], "fold": rec["fold"], "task": "regression", "model": "training_median_baseline", **regression_metrics(ytest, np.full(len(test), median))},
            {"repeat": rec["repeat"], "fold": rec["fold"], "task": "regression", "model": "training_mean_baseline", **regression_metrics(ytest, np.full(len(test), mean))},
            {"repeat": rec["repeat"], "fold": rec["fold"], "task": "binary_classification", "model": "majority_class_baseline", **classification_metrics(active_test, np.full(len(test), majority), hard=np.full(len(test), majority))},
            {"repeat": rec["repeat"], "fold": rec["fold"], "task": "binary_classification", "model": "dummy_prior_baseline", **classification_metrics(active_test, np.full(len(test), prevalence), probability=np.full(len(test), prevalence), hard=np.full(len(test), int(prevalence >= 0.5)))}
        ])
    pred.to_csv(RUN / "pooled_oof_predictions.csv", index=False)
    model_path = RUN / "model.joblib"
    if model_path.exists():
        bundle = joblib.load(model_path)
        bundle["disagreement_reference"] = pred.disagreement_sd.to_numpy(float)
        bundle["ad_distance_reference"] = pred.ad_distance.to_numpy(float)
        joblib.dump(bundle, model_path, compress=3)
        metadata_path = RUN / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        metadata["combined_reliability_reference"] = "empirical distributions of held-out repeated scaffold OOF disagreement and AD distances; no pseudolabel acceptance cutoff"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    fold_path = RUN / "fold_metrics.csv"
    fold_existing = pd.read_csv(fold_path)
    fold_existing = fold_existing.loc[~fold_existing.model.isin(["training_median_baseline", "training_mean_baseline", "majority_class_baseline", "dummy_prior_baseline"])]
    fm = pd.concat([fold_existing, pd.DataFrame(fold_baselines)], ignore_index=True)
    fm.to_csv(fold_path, index=False)
    metrics_to_summarize=[c for c in ["MAE","RMSE","R2","median_absolute_error","Pearson_r","Spearman_r","accuracy","balanced_accuracy","active_precision","inactive_precision","active_recall_sensitivity","specificity","F1","MCC","ROC_AUC","PR_AUC","Brier","ECE_10bin"] if c in fm.columns]
    fm.groupby(["task","model"],dropna=False)[metrics_to_summarize].agg(["mean","std"]).to_csv(RUN/"fold_metric_summary.csv")
    regression_path = RUN / "regression_metrics.csv"
    reg = pd.read_csv(regression_path)
    reg = reg.loc[~reg.model.isin(["training_median_baseline", "training_mean_baseline"])]
    for model, col in [("training_median_baseline", "pred_training_median"), ("training_mean_baseline", "pred_training_mean")]:
        reg = pd.concat([reg, pd.DataFrame([{"model": model, **regression_metrics(pred.observed_pIC50, pred[col])}])], ignore_index=True)
    reg.to_csv(regression_path, index=False)
    class_path = RUN / "classification_metrics.csv"
    cls = pd.read_csv(class_path)
    cls = cls.loc[~cls.model.isin(["majority_class_baseline", "dummy_prior_baseline"])]
    for model, score, hard, probability in [
        ("majority_class_baseline", pred.majority_class_prediction, pred.majority_class_prediction, None),
        ("dummy_prior_baseline", pred.dummy_prior_probability, pred.dummy_prior_prediction, pred.dummy_prior_probability),
    ]:
        cls = pd.concat([cls, pd.DataFrame([{"model": model, **classification_metrics(pred.observed_active.astype(int), score, probability=probability, hard=hard)}])], ignore_index=True)
    cls.to_csv(class_path, index=False)
    from sklearn.metrics import confusion_matrix
    cm_path = RUN / "confusion_matrix.csv"
    cm = pd.read_csv(cm_path)
    cm = cm.loc[~cm.model.isin(["majority_class_baseline", "dummy_prior_baseline"])]
    for model, hard in [("majority_class_baseline", pred.majority_class_prediction.astype(int)),
                        ("dummy_prior_baseline", pred.dummy_prior_prediction.astype(int))]:
        tn, fp, fn, tp = confusion_matrix(pred.observed_active.astype(int), hard, labels=[0, 1]).ravel()
        cm = pd.concat([cm, pd.DataFrame([{"model": model, "cell": cell, "count": val}
                                         for cell, val in [("TN", tn), ("FP", fp), ("FN", fn), ("TP", tp)]])], ignore_index=True)
    cm.to_csv(cm_path, index=False)
    uncertainty_rows=[]
    true_active=pred.observed_active.to_numpy(int)
    predicted_active=pred.regression_derived_active.to_numpy(int)
    error_targets={
        "absolute_regression_error":pred.absolute_error.to_numpy(float),
        "binary_classification_error":(true_active!=predicted_active).astype(int),
        "false_active":((true_active==0)&(predicted_active==1)).astype(int),
        "false_inactive":((true_active==1)&(predicted_active==0)).astype(int),
    }
    for score in ["disagreement_sd","ad_distance","combined_rank"]:
        for target_name,target in error_targets.items():
            score_values=pred[score].to_numpy(float)
            if np.std(score_values)==0 or np.std(target)==0:
                corr,pvalue=np.nan,np.nan
            else:
                stat=spearmanr(score_values,target);corr,pvalue=float(stat.statistic),float(stat.pvalue)
            uncertainty_rows.append({"score":score,"outcome":target_name,"spearman_r":corr,"p_value_descriptive":pvalue})
    pd.DataFrame(uncertainty_rows).to_csv(RUN/"uncertainty_analysis.csv",index=False)
    filter_rows=[]
    filtering={"no_filter":None,"AD_only":"ad_distance","uncertainty_only":"disagreement_sd","combined_AD_uncertainty":"combined_rank"}
    for strategy,score in filtering.items():
        ordered=pred if score is None else pred.sort_values(score,ascending=True)
        for fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            n=max(1,int(np.ceil(fraction*len(ordered))));sample=ordered.iloc[:n]
            cls_metrics=classification_metrics(sample.observed_active.astype(int),sample.primary_prediction.astype(float),
                                               hard=sample.regression_derived_active.astype(int))
            filter_rows.append({"strategy":strategy,"target_retained_fraction":fraction,
                "actual_retained_fraction":n/len(pred),"n_retained":n,"n_unique_molecules":int(sample.row_index.nunique()),
                "regression_MAE":float(sample.absolute_error.mean()),
                "regression_RMSE":float(np.sqrt(np.mean(sample.absolute_error.to_numpy(float)**2))),
                "active_precision":cls_metrics["active_precision"],"inactive_precision":cls_metrics["inactive_precision"],
                "false_active_rate":cls_metrics["false_active_rate"],"false_inactive_rate":cls_metrics["false_inactive_rate"],
                "accuracy":cls_metrics["accuracy"],"balanced_accuracy":cls_metrics["balanced_accuracy"],
                "unique_scaffolds_retained":int(sample.scaffold.nunique())})
    pd.DataFrame(filter_rows).to_csv(RUN/"precision_coverage.csv",index=False)
    boot = []
    groups = pred.scaffold.astype(str).to_numpy()
    class_weights = cluster_bootstrap_weights(groups, 2000, 410)
    model_hard = {"regression_derived": pred.regression_derived_active.to_numpy(int)}
    for col in pred.columns:
        if col.startswith("active_"):
            model_hard[col[len("active_"):]] = pred[col].to_numpy(int)
    model_hard["majority_class_baseline"] = pred.majority_class_prediction.to_numpy(int)
    model_hard["dummy_prior_baseline"] = pred.dummy_prior_prediction.to_numpy(int)
    ycls = pred.observed_active.to_numpy(int)
    for model, hard in model_hard.items():
        samples = {key:[] for key in ["accuracy","balanced_accuracy","active_precision","inactive_precision","sensitivity","specificity","MCC"]}
        true_positive=ycls==1; false_positive=ycls==0
        pred_positive=hard==1; pred_negative=hard==0
        for weights in class_weights:
            tn=float(weights[false_positive & pred_negative].sum()); fp=float(weights[false_positive & pred_positive].sum())
            fn=float(weights[true_positive & pred_negative].sum()); tp=float(weights[true_positive & pred_positive].sum())
            for metric,value in classification_from_counts(tn,fp,fn,tp).items(): samples[metric].append(float(value))
        for metric,vals in samples.items():
            lo, hi = np.quantile(vals, [0.025, 0.975])
            boot.append({"task": "classification", "model": model, "metric": metric,
                         "lower_95": float(lo), "upper_95": float(hi),
                         "cluster": "Bemis-Murcko scaffold"})
    y = pred.observed_pIC50.to_numpy(float)
    reg_weights=cluster_bootstrap_weights(groups,2000,420)
    for model in ["RandomForest", "ExtraTrees", "SVR", "XGBoost", "HistGradientBoosting",
                  "five_model_ridge_alpha_1", "svr_complementary_ridge_alpha_100", "boosting_pair_ridge_alpha_1",
                  "training_median_baseline", "training_mean_baseline"]:
        col = ("primary_prediction" if model == "five_model_ridge_alpha_1" else
               "pred_training_median" if model == "training_median_baseline" else
               "pred_training_mean" if model == "training_mean_baseline" else f"pred_{model}")
        p = pred[col].to_numpy(float)
        draws=weighted_regression_metrics(y,p,reg_weights)
        for name, values in draws.items():
            lo,hi=np.nanquantile(values,[.025,.975])
            boot.append({"task":"regression","model":model,"metric":name,"lower_95":float(lo),"upper_95":float(hi),"cluster":"Bemis-Murcko scaffold"})
    pd.DataFrame(boot).to_csv(RUN / "scaffold_bootstrap_intervals.csv", index=False)

    pred.sort_values("absolute_error", ascending=False).head(100).to_csv(RUN / "error_analysis.csv", index=False)
    pred.groupby("scaffold", as_index=False).agg(
        n=("row_index", "size"), observed_mean=("observed_pIC50", "mean"),
        predicted_mean=("primary_prediction", "mean"), MAE=("absolute_error", "mean"),
        mean_similarity=("max_train_similarity", "mean"), disagreement=("disagreement_sd", "mean"),
        label_error=("observed_active", lambda z: float(np.mean(z.to_numpy() != pred.loc[z.index, "regression_derived_active"].to_numpy()))),
    ).sort_values("MAE", ascending=False).to_csv(RUN / "error_by_scaffold.csv", index=False)

    interval_rows=[]
    for (method, nominal), cal in interval.groupby(["method", "nominal"]):
        widths=cal.set_index(["repeat","fold"])["half_width"]
        keys=list(zip(pred.repeat,pred.fold))
        half=np.array([widths.loc[k] for k in keys],dtype=float)
        covered=(pred.absolute_error.to_numpy(float)<=half)
        ci=clustered_mean_ci(covered.astype(float),groups,seed=500+int(nominal*100)+len(interval_rows))
        joined=pred.assign(covered=covered)
        whole_series=joined.groupby(["repeat","scaffold"]).covered.all().astype(float)
        whole=whole_series.to_numpy()
        whole_groups=np.asarray([f"{repeat}::{scaffold}" for repeat,scaffold in whole_series.index],dtype=str)
        scaffold_ci=clustered_mean_ci(whole,whole_groups)
        interval_rows.append({"method":method,"nominal_coverage":float(nominal),
            "empirical_molecule_coverage":float(covered.mean()),"mean_width":float(2*half.mean()),
            "whole_scaffold_containment":float(whole.mean()),"coverage_scaffold_bootstrap_lower":ci[0],
            "coverage_scaffold_bootstrap_upper":ci[1],"whole_scaffold_bootstrap_lower":scaffold_ci[0],
            "whole_scaffold_bootstrap_upper":scaffold_ci[1]})
    pd.DataFrame(interval_rows).to_csv(RUN / "interval_coverage.csv",index=False)

    # Residual summaries stratified by activity and held-out scaffold size.
    pred["activity_bin"] = pd.cut(pred.observed_pIC50, bins=[-np.inf,6,7,8,9,np.inf],
                                   labels=["<6","6-7","7-8","8-9",">=9"])
    pred.groupby("activity_bin",observed=True).agg(n=("absolute_error","size"),MAE=("absolute_error","mean"),
        signed_residual=("primary_prediction",lambda z:float(np.mean(z.to_numpy()-pred.loc[z.index,"observed_pIC50"].to_numpy())))).reset_index().to_csv(RUN/"error_by_activity_range.csv",index=False)
    pred["scaffold_size_bin"] = pd.cut(pred.scaffold_size_test,bins=[0,1,4,np.inf],labels=["1","2-4","5+"])
    pred.groupby("scaffold_size_bin",observed=True).agg(n=("absolute_error","size"),MAE=("absolute_error","mean"),RMSE=("absolute_error",lambda z:float(np.sqrt(np.mean(z.to_numpy()**2))))).reset_index().to_csv(RUN/"error_by_scaffold_size.csv",index=False)

    df,_,_,_,mols,_=load_dataset()
    props=pd.DataFrame({"row_index":np.arange(len(df)),"MolWt":[Descriptors.MolWt(m) for m in mols],
                        "MolLogP":[Descriptors.MolLogP(m) for m in mols]})
    base=pred.merge(props,on="row_index",how="left")
    for prop in ["MolWt","MolLogP"]:
        base[prop+"_bin"]=pd.qcut(base[prop].rank(method="first"),q=5,labels=False,duplicates="drop")
        base.groupby(prop+"_bin",observed=True).agg(n=("absolute_error","size"),MAE=("absolute_error","mean"),
            signed_residual=("primary_prediction",lambda z:float(np.mean(z.to_numpy()-base.loc[z.index,"observed_pIC50"].to_numpy())))).reset_index().to_csv(RUN/f"error_by_{prop}.csv",index=False)


if __name__ == "__main__":
    main()
