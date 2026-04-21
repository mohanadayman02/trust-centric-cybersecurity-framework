"""Training and cross-validation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate


def _safe_cross_validate(model, x_train, y_train, cv_folds: int, scoring: Dict[str, object]):
    """Run CV with parallel backend, fallback to single-process if restricted."""
    try:
        return cross_validate(
            model,
            x_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            error_score=np.nan,
        )
    except PermissionError:
        return cross_validate(
            model,
            x_train,
            y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=1,
            error_score=np.nan,
        )


def train_agent(model, x_train, y_train):
    """Train a model and return the fitted model."""
    model.fit(x_train, y_train)
    return model


def run_cross_validation(model, x_train, y_train, cv_folds: int) -> Dict[str, float]:
    """Run cross-validation and return mean accuracy/precision/recall/f1."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "roc_auc": "roc_auc",
    }
    cv_results = _safe_cross_validate(model, x_train, y_train, cv_folds, scoring=scoring)

    return {
        "cv_accuracy": float(np.nanmean(cv_results["test_accuracy"])),
        "cv_precision": float(np.nanmean(cv_results["test_precision"])),
        "cv_recall": float(np.nanmean(cv_results["test_recall"])),
        "cv_f1": float(np.nanmean(cv_results["test_f1"])),
        "cv_roc_auc": (
            float(np.nanmean(cv_results["test_roc_auc"]))
            if not np.isnan(cv_results["test_roc_auc"]).all()
            else np.nan
        ),
    }
