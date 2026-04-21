"""Evaluation utilities for IDS agents."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _compute_metrics(y_test, predictions, y_prob=None) -> Dict:
    """Compute binary IDS metrics for provided predictions."""
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((tpr + tnr) / 2.0)
    error_rate = float((fp + fn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    roc_auc = np.nan
    pr_auc = np.nan
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:  # pylint: disable=broad-except
            roc_auc = np.nan
        try:
            pr_auc = float(average_precision_score(y_test, y_prob))
        except Exception:  # pylint: disable=broad-except
            pr_auc = np.nan

    precision = float(precision_score(y_test, predictions, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, predictions, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, predictions, pos_label=1, zero_division=0))

    return {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": tnr,
        "balanced_accuracy": balanced_accuracy,
        "error_rate": error_rate,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "support_total": int(len(y_test)),
        "support_attack": int((y_test == 1).sum()),
        "support_normal": int((y_test == 0).sum()),
    }


def evaluate_model(model, x_test, y_test, predictions=None, y_prob=None) -> Dict:
    """Evaluate a trained model on test data."""
    if predictions is None:
        predictions_output = model.predict(x_test)
        predictions = (
            predictions_output["y_pred"]
            if isinstance(predictions_output, dict) and "y_pred" in predictions_output
            else predictions_output
        )
    return _compute_metrics(y_test, predictions, y_prob=y_prob)


def evaluate_predictions(y_test, predictions, y_prob=None) -> Dict:
    """Evaluate already available predictions without a model object."""
    return _compute_metrics(y_test, predictions, y_prob=y_prob)
