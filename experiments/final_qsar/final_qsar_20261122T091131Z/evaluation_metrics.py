"""Regression, classification, calibration and cluster-bootstrap metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)


def regression_metrics(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    error = prediction - y
    nonconstant = np.ptp(y) > 1e-12 and np.ptp(prediction) > 1e-12
    pearson = float(np.corrcoef(y, prediction)[0, 1]) if nonconstant else float("nan")
    spearman = float(pd.Series(y).corr(pd.Series(prediction), method="spearman")) if nonconstant else float("nan")
    return {
        "n": int(len(y)), "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(mean_squared_error(y, prediction) ** 0.5),
        "MSE": float(mean_squared_error(y, prediction)),
        "R2": float(r2_score(y, prediction)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "Pearson_r": pearson,
        "Spearman_r": spearman,
        "residual_mean": float(np.mean(error)),
        "residual_sd": float(np.std(error, ddof=1)),
    }


def classification_metrics(y_true, score, probability=None, hard=None):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    hard = (score >= 0.5).astype(int) if hard is None else np.asarray(hard, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, hard, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y_true)), "prevalence_active": float(np.mean(y_true == 1)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, hard)),
        "active_precision": float(precision_score(y_true, hard, zero_division=0)),
        "active_recall_sensitivity": float(recall_score(y_true, hard, zero_division=0)),
        "inactive_precision": float(tn / (tn + fn)) if tn + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1": float(f1_score(y_true, hard, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, hard)),
        "false_active_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_inactive_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "ROC_AUC": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "PR_AUC": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan"),
    }
    if probability is not None:
        p = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
        result["Brier"] = float(brier_score_loss(y_true, p))
        result["ECE_10bin"] = expected_calibration_error(y_true, p, bins=10)
    return result


def expected_calibration_error(y, probability, bins=10):
    y = np.asarray(y, int)
    p = np.asarray(probability, float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.any():
            ece += mask.sum() / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def conformal_quantile(abs_residuals, alpha):
    scores = np.asarray(abs_residuals, dtype=float)
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def scaffold_bootstrap(y, prediction, scaffold, metric_fn, n_boot=2000, seed=140):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    scaffold = np.asarray(scaffold).astype(str)
    groups, inverse = np.unique(scaffold, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = rng.integers(0, len(groups), size=len(groups))
        weights = np.bincount(sample, minlength=len(groups))[inverse]
        idx = np.repeat(np.arange(len(y)), weights)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(metric_fn(y[idx], prediction[idx]))
    return np.quantile(np.asarray(draws), [0.025, 0.975]).tolist()
