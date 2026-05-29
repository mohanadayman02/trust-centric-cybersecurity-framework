import numpy as np
from pipeline import trust_methods


def test_majority_voting_simple():
    preds = {
        "m1": np.array([1, 0, 1, 1]),
        "m2": np.array([1, 0, 0, 1]),
        "m3": np.array([0, 0, 1, 1]),
    }
    out = trust_methods.majority_voting(preds)
    assert out["meta"]["method"] == "majority_voting"
    assert np.array_equal(out["predictions"], np.array([1, 0, 1, 1]))


def test_accuracy_based_trust_prefers_best():
    preds = {
        "a": np.array([1, 1, 0, 0]),
        "b": np.array([1, 0, 0, 0]),
    }
    metrics = {"a": {"test_accuracy": 0.6}, "b": {"test_accuracy": 0.9}}
    out = trust_methods.accuracy_based_trust(preds, metrics)
    assert out["meta"]["method"] == "accuracy_based_trust"
    # model b should dominate
    assert np.array_equal(out["predictions"], preds["b"]) or out["predictions"].sum() >= preds["b"].sum()


def test_error_aware_punishes_fnr():
    preds = {"m1": np.array([0, 1, 0, 1]), "m2": np.array([1, 1, 1, 1])}
    metrics = {"m1": {"test_accuracy": 0.9, "fpr": 0.1, "fnr": 0.5}, "m2": {"test_accuracy": 0.85, "fpr": 0.05, "fnr": 0.1}}
    out = trust_methods.error_aware_trust(preds, metrics, alpha=0.3, beta=0.7)
    assert out["meta"]["method"] == "error_aware_trust"


def test_confidence_based_trust_handles_missing_probs():
    preds = {"m1": np.array([1, 0]), "m2": np.array([1, 1])}
    # no probabilities provided
    out = trust_methods.confidence_based_trust(preds, None, {"m1": {"test_f1": 0.5}, "m2": {"test_f1": 0.8}})
    assert out["meta"]["method"] == "confidence_based_trust"
    assert out["predictions"].shape[0] == 2


def test_class_specific_trust_returns_array():
    preds = {"m1": np.array([0, 1, 1]), "m2": np.array([1, 1, 0])}
    metrics = {"m1": {"test_f1": 0.6}, "m2": {"test_f1": 0.7}}
    out = trust_methods.class_specific_trust(preds, metrics)
    assert out["meta"]["method"] == "class_specific_trust"
    assert out["predictions"].shape[0] == 3


def test_hybrid_trust_outputs_weights():
    preds = {"m1": np.array([0, 1]), "m2": np.array([1, 1])}
    probs = {"m1": None, "m2": None}
    metrics = {"m1": {"test_f1": 0.4, "test_recall": 0.5, "specificity": 0.6, "fnr": 0.2}, "m2": {"test_f1": 0.8, "test_recall": 0.9, "specificity": 0.8, "fnr": 0.1}}
    out = trust_methods.hybrid_trust(preds, probs, metrics)
    assert out["meta"]["method"] == "hybrid_trust"
    assert out["predictions"].shape[0] == 2


def test_attack_recall_trust_prefers_high_recall_model():
    preds = {"high_recall": np.array([1, 1, 1]), "high_precision": np.array([0, 0, 1])}
    metrics = {
        "high_recall": {"test_recall": 0.95, "fnr": 0.05, "test_f1": 0.78, "test_precision": 0.62},
        "high_precision": {"test_recall": 0.55, "fnr": 0.45, "test_f1": 0.80, "test_precision": 0.95},
    }
    out = trust_methods.attack_recall_trust(preds, metrics)
    assert out["meta"]["method"] == "attack_recall_trust"
    assert out["meta"]["trust_scores"]["high_recall"] > out["meta"]["trust_scores"]["high_precision"]


def test_fnr_penalty_trust_penalizes_high_fnr():
    preds = {"safe": np.array([0, 0, 1]), "risky": np.array([1, 1, 1])}
    metrics = {
        "safe": {"test_f1": 0.82, "fnr": 0.10, "specificity": 0.80},
        "risky": {"test_f1": 0.83, "fnr": 0.50, "specificity": 0.70},
    }
    out = trust_methods.fnr_penalty_trust(preds, metrics, beta=0.8, alpha=0.2)
    assert out["meta"]["method"] == "fnr_penalty_trust"
    assert out["meta"]["trust_scores"]["safe"] > out["meta"]["trust_scores"]["risky"]


def test_attack_override_trust_chooses_attack_for_confident_trusted_model():
    preds = {"trusted_attack": np.array([1, 0]), "other": np.array([0, 0])}
    probs = {
        "trusted_attack": np.array([[0.05, 0.95], [0.80, 0.20]]),
        "other": np.array([[0.60, 0.40], [0.70, 0.30]]),
    }
    metrics = {
        "trusted_attack": {"test_f1": 0.92},
        "other": {"test_f1": 0.60},
    }
    out = trust_methods.attack_override_trust(
        preds,
        probs,
        metrics,
        attack_label="auto",
        attack_confidence_threshold=0.65,
        min_attack_model_trust=0.80,
    )
    assert out["meta"]["method"] == "attack_override_trust"
    assert out["predictions"][0] == 1


def test_best_safe_model_selector_picks_highest_safety_score():
    preds = {"safe_model": np.array([0, 1, 0]), "less_safe_model": np.array([1, 1, 1])}
    metrics = {
        "safe_model": {"test_recall": 0.90, "test_f1": 0.88, "fnr": 0.05, "test_accuracy": 0.91},
        "less_safe_model": {"test_recall": 0.70, "test_f1": 0.91, "fnr": 0.25, "test_accuracy": 0.93},
    }
    out = trust_methods.best_safe_model_selector(preds, metrics)
    assert out["meta"]["method"] == "best_safe_model_selector"
    assert out["meta"]["selected_model"] == "safe_model"
    assert np.array_equal(out["predictions"], preds["safe_model"])


def test_role_aware_trust_voting_handles_disagreement_cases():
    preds = {
        "General Traffic Agent": np.array([0, 0, 0]),
        "Attack Recall Agent": np.array([1, 0, 1]),
        "Normal Behavior Agent": np.array([0, 0, 0]),
        "Hard-Case Agent": np.array([1, 1, 1]),
    }
    probs = {
        "General Traffic Agent": np.array([0.30, 0.20, 0.25]),
        "Attack Recall Agent": np.array([0.90, 0.35, 0.92]),
        "Normal Behavior Agent": np.array([0.20, 0.15, 0.25]),
        "Hard-Case Agent": np.array([0.80, 0.85, 0.70]),
    }
    metrics = {
        "General Traffic Agent": {"test_f1": 0.86},
        "Attack Recall Agent": {"test_f1": 0.82},
        "Normal Behavior Agent": {"test_f1": 0.81},
        "Hard-Case Agent": {"test_f1": 0.79},
    }
    roles = {
        "General Traffic Agent": "general",
        "Attack Recall Agent": "attack_recall",
        "Normal Behavior Agent": "normal_behavior",
        "Hard-Case Agent": "hard_case",
    }
    out = trust_methods.role_aware_trust_voting(preds, probs, metrics, roles)
    assert out["meta"]["method"] == "role_aware_trust_voting"
    assert out["predictions"].shape[0] == 3
    assert out["predictions"][0] == 1


def test_trust_agent_selector_returns_prediction_per_sample():
    preds = {
        "General Traffic Agent": np.array([0, 1, 0, 1]),
        "Attack Recall Agent": np.array([1, 1, 0, 1]),
        "Normal Behavior Agent": np.array([0, 0, 0, 0]),
        "Hard-Case Agent": np.array([1, 0, 1, 1]),
    }
    probs = {
        "General Traffic Agent": np.array([0.20, 0.65, 0.30, 0.70]),
        "Attack Recall Agent": np.array([0.85, 0.90, 0.40, 0.88]),
        "Normal Behavior Agent": np.array([0.10, 0.25, 0.20, 0.30]),
        "Hard-Case Agent": np.array([0.75, 0.45, 0.60, 0.80]),
    }
    metrics = {
        "General Traffic Agent": {"test_f1": 0.84, "test_recall": 0.82, "specificity": 0.87},
        "Attack Recall Agent": {"test_f1": 0.81, "test_recall": 0.94, "specificity": 0.72},
        "Normal Behavior Agent": {"test_f1": 0.80, "test_recall": 0.72, "specificity": 0.95},
        "Hard-Case Agent": {"test_f1": 0.78, "test_recall": 0.79, "specificity": 0.81},
    }
    roles = {
        "General Traffic Agent": "general",
        "Attack Recall Agent": "attack_recall",
        "Normal Behavior Agent": "normal_behavior",
        "Hard-Case Agent": "hard_case",
    }
    x_val = np.array([[0.0, 0.0], [0.1, 0.2], [0.9, 0.8], [1.0, 1.0]])
    y_val = np.array([0, 0, 1, 1])
    val_preds = {
        "General Traffic Agent": np.array([0, 0, 1, 1]),
        "Attack Recall Agent": np.array([1, 0, 1, 1]),
        "Normal Behavior Agent": np.array([0, 0, 0, 1]),
        "Hard-Case Agent": np.array([0, 1, 1, 1]),
    }
    x_test = np.array([[0.05, 0.10], [0.95, 0.90], [0.40, 0.45], [0.80, 0.75]])
    out = trust_methods.trust_agent_selector(preds, probs, metrics, roles, x_val, y_val, val_preds, x_test)
    assert out["meta"]["method"] == "trust_agent_selector"
    assert out["predictions"].shape[0] == x_test.shape[0]


def test_trust_agent_selector_uses_disagreement_bonus_for_hard_case():
    preds = {
        "General Traffic Agent": np.array([0, 0]),
        "Attack Recall Agent": np.array([1, 0]),
        "Normal Behavior Agent": np.array([0, 0]),
        "Hard-Case Agent": np.array([1, 1]),
    }
    probs = {
        "General Traffic Agent": np.array([0.30, 0.20]),
        "Attack Recall Agent": np.array([0.80, 0.45]),
        "Normal Behavior Agent": np.array([0.20, 0.25]),
        "Hard-Case Agent": np.array([0.70, 0.75]),
    }
    metrics = {
        "General Traffic Agent": {"test_f1": 0.86, "test_recall": 0.84, "specificity": 0.88},
        "Attack Recall Agent": {"test_f1": 0.82, "test_recall": 0.91, "specificity": 0.75},
        "Normal Behavior Agent": {"test_f1": 0.80, "test_recall": 0.73, "specificity": 0.94},
        "Hard-Case Agent": {"test_f1": 0.79, "test_recall": 0.80, "specificity": 0.79},
    }
    roles = {
        "General Traffic Agent": "general",
        "Attack Recall Agent": "attack_recall",
        "Normal Behavior Agent": "normal_behavior",
        "Hard-Case Agent": "hard_case",
    }
    x_val = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1], [0.9, 0.9]])
    y_val = np.array([0, 1, 0, 1])
    val_preds = {
        "General Traffic Agent": np.array([0, 1, 0, 1]),
        "Attack Recall Agent": np.array([1, 1, 0, 1]),
        "Normal Behavior Agent": np.array([0, 0, 0, 1]),
        "Hard-Case Agent": np.array([1, 1, 1, 1]),
    }
    x_test = np.array([[0.05, 0.05], [0.95, 0.95]])
    out = trust_methods.trust_agent_selector(
        preds,
        probs,
        metrics,
        roles,
        x_val,
        y_val,
        val_preds,
        x_test,
        disagreement_bonus=0.40,
        attack_role_bonus=0.0,
        normal_role_bonus=0.0,
    )
    assert out["meta"]["selected_agents"][0] == "Hard-Case Agent"
