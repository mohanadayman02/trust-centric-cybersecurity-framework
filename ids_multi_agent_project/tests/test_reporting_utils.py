import inspect

import numpy as np

from pipeline.reporting_utils import compute_model_diversity_report, compute_oracle_upper_bound
from pipeline import trust_methods


def test_oracle_upper_bound_computation():
    y_true = np.array([0, 1, 1, 0])
    preds = {
        "m1": np.array([0, 1, 0, 0]),
        "m2": np.array([1, 1, 1, 1]),
    }
    report = compute_oracle_upper_bound(y_true, preds)
    assert report["oracle_accuracy"] == 1.0
    assert report["at_least_one_correct_count"] == 4
    assert report["all_wrong_count"] == 0
    assert report["maximum_possible_improvement"] >= 0.0


def test_model_diversity_counts_are_correct():
    y_true = np.array([0, 1, 1, 0])
    preds = {
        "m1": np.array([0, 1, 0, 0]),
        "m2": np.array([0, 0, 1, 1]),
    }
    report = compute_model_diversity_report(y_true, preds)
    row = report.iloc[0]
    assert row["agreement_rate"] == 0.25
    assert row["disagreement_rate"] == 0.75
    assert row["a_correct_b_wrong"] == 2
    assert row["b_correct_a_wrong"] == 1
    assert row["both_wrong"] == 0
    assert row["both_correct"] == 1


def test_confidence_margin_trust_prefers_high_confidence_model():
    preds = {"trusty": np.array([1, 1]), "cautious": np.array([1, 0])}
    probs = {
        "trusty": np.array([[0.05, 0.95], [0.10, 0.90]]),
        "cautious": np.array([[0.45, 0.55], [0.55, 0.45]]),
    }
    metrics = {
        "trusty": {"test_accuracy": 0.90},
        "cautious": {"test_accuracy": 0.89},
    }
    out = trust_methods.confidence_margin_trust(preds, probs, metrics)
    assert out["meta"]["method"] == "confidence_margin_trust"
    assert out["predictions"][0] == 1


def test_best_accuracy_selector_chooses_highest_validation_accuracy():
    preds = {"good": np.array([1, 0]), "better": np.array([1, 1])}
    metrics = {
        "good": {"test_accuracy": 0.91},
        "better": {"test_accuracy": 0.97},
    }
    out = trust_methods.best_accuracy_selector(preds, metrics)
    assert out["meta"]["selected_model"] == "better"
    assert np.array_equal(out["predictions"], preds["better"])


def test_stacking_meta_trust_signature_avoids_test_labels():
    signature = inspect.signature(trust_methods.stacking_meta_trust)
    assert "test_labels" not in signature.parameters
