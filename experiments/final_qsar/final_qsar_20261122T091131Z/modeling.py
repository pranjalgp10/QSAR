"""Fold-local fixed QSAR models and strictly cross-fitted stacking features."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

BASE_NAMES = ["RandomForest", "ExtraTrees", "SVR", "XGBoost", "HistGradientBoosting"]
STACK_SPECS = {
    "five_model_ridge_alpha_1": (BASE_NAMES, 1.0),
    "svr_complementary_ridge_alpha_100": (["RandomForest", "XGBoost", "HistGradientBoosting", "SVR"], 100.0),
    "boosting_pair_ridge_alpha_1": (["XGBoost", "HistGradientBoosting"], 1.0),
}


def make_base_regressors(seed: int):
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
            learning_rate=0.03, subsample=0.8, random_state=seed, n_jobs=1,
            verbosity=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=15, learning_rate=0.04,
            l2_regularization=2.0, random_state=seed),
    }


def fit_regression_ensemble(X, y, groups, seed: int, oof_folds: int = 3):
    """Return fold-local base fits and base OOF matrix for meta fitting."""
    unique_groups = len(np.unique(groups))
    n_splits = min(oof_folds, unique_groups)
    if n_splits < 2:
        raise ValueError("Stacking requires at least two distinct fit scaffolds")
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    oof = np.full((len(y), len(BASE_NAMES)), np.nan, dtype=float)
    final_models = {}
    for col, name in enumerate(BASE_NAMES):
        for itr, iva in splits:
            if set(groups[itr]) & set(groups[iva]):
                raise AssertionError("Scaffold leak in meta-feature generation")
            learner = make_base_regressors(seed + col)[name]
            learner.fit(X[itr], y[itr])
            oof[iva, col] = learner.predict(X[iva])
        learner = make_base_regressors(seed + col)[name]
        learner.fit(X, y)
        final_models[name] = learner
    if not np.isfinite(oof).all():
        raise AssertionError("OOF meta-feature matrix is incomplete")
    return final_models, oof


def make_stack_predictions(models, oof, y_fit, X_eval):
    base_eval = np.column_stack([models[name].predict(X_eval) for name in BASE_NAMES])
    results = {}
    for name, (members, alpha) in STACK_SPECS.items():
        cols = [BASE_NAMES.index(member) for member in members]
        meta = Ridge(alpha=alpha).fit(oof[:, cols], y_fit)
        results[name] = {
            "prediction": meta.predict(base_eval[:, cols]),
            "meta_model": meta,
            "members": members,
        }
    disagreement = base_eval.std(axis=1, ddof=1)
    return base_eval, results, disagreement


def make_direct_classifiers(seed: int):
    return {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400, max_depth=28, min_samples_leaf=1,
            max_features=0.5, random_state=seed, n_jobs=-1),
        "XGBoostClassifier": XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=300,
            max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000,
                                                 solver="lbfgs", random_state=seed)),
    }


def raw_classifier_score(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
