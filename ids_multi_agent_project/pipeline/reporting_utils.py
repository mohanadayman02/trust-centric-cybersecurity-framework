from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_oracle_upper_bound(y_true: np.ndarray, model_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    if not model_predictions:
        return {
            "oracle_accuracy": 0.0,
            "at_least_one_correct_count": 0,
            "all_wrong_count": int(y_true_arr.shape[0]),
            "maximum_possible_improvement": 0.0,
        }

    correct_any = None
    all_wrong = None
    for predictions in model_predictions.values():
        prediction_array = np.asarray(predictions, dtype=int)
        correct = prediction_array == y_true_arr
        correct_any = correct if correct_any is None else (correct_any | correct)
        all_wrong = (~correct) if all_wrong is None else (all_wrong & ~correct)

    oracle_accuracy = float(np.mean(correct_any))
    all_wrong_count = int(np.sum(all_wrong))
    best_baseline_accuracy = max(float(np.mean(np.asarray(predictions, dtype=int) == y_true_arr)) for predictions in model_predictions.values())
    return {
        "oracle_accuracy": oracle_accuracy,
        "at_least_one_correct_count": int(np.sum(correct_any)),
        "all_wrong_count": all_wrong_count,
        "maximum_possible_improvement": float(oracle_accuracy - best_baseline_accuracy),
    }


def compute_model_diversity_report(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=int)
    rows = []
    for model_a, model_b in combinations(model_predictions.keys(), 2):
        preds_a = np.asarray(model_predictions[model_a], dtype=int)
        preds_b = np.asarray(model_predictions[model_b], dtype=int)
        a_correct = preds_a == y_true_arr
        b_correct = preds_b == y_true_arr
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "agreement_rate": float(np.mean(preds_a == preds_b)),
                "disagreement_rate": float(np.mean(preds_a != preds_b)),
                "a_correct_b_wrong": int(np.sum(a_correct & ~b_correct)),
                "b_correct_a_wrong": int(np.sum(~a_correct & b_correct)),
                "both_wrong": int(np.sum(~a_correct & ~b_correct)),
                "both_correct": int(np.sum(a_correct & b_correct)),
            }
        )
    return pd.DataFrame(rows)


def summarize_prediction_distribution(predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, preds in predictions.items():
        pred_array = np.asarray(preds, dtype=int)
        attack_rate = float(np.mean(pred_array == 1))
        normal_rate = float(np.mean(pred_array == 0))
        rows.append(
            {
                "method": name,
                "attack_rate": attack_rate,
                "normal_rate": normal_rate,
                "almost_all_attack": bool(attack_rate >= 0.95),
                "almost_all_normal": bool(normal_rate >= 0.95),
            }
        )
    return pd.DataFrame(rows)
