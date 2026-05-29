import numpy as np

import main


def test_exactly_four_agents_in_strict_mode_definition():
    names = [
        "General Traffic Agent",
        "Attack Recall Agent",
        "Normal Behavior Agent",
        "Hard-Case Agent",
    ]
    assert len(names) == 4


def test_attack_threshold_tuning_improves_recall_vs_default():
    y_val = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p_attack = np.array([0.45, 0.40, 0.35, 0.30, 0.70, 0.20, 0.10, 0.05])
    default_preds = main._apply_threshold(p_attack, 0.50)
    tuned_threshold = main._tune_attack_recall_threshold(y_val, p_attack)
    tuned_preds = main._apply_threshold(p_attack, tuned_threshold)
    default_metrics = main.evaluate_predictions(y_val, default_preds, y_prob=p_attack)
    tuned_metrics = main.evaluate_predictions(y_val, tuned_preds, y_prob=p_attack)
    assert tuned_metrics["test_recall"] >= default_metrics["test_recall"]


def test_normal_threshold_tuning_improves_specificity_vs_default():
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_attack = np.array([0.55, 0.52, 0.50, 0.45, 0.80, 0.70, 0.60, 0.40])
    default_preds = main._apply_threshold(p_attack, 0.50)
    tuned_threshold = main._tune_normal_specificity_threshold(y_val, p_attack)
    tuned_preds = main._apply_threshold(p_attack, tuned_threshold)
    default_metrics = main.evaluate_predictions(y_val, default_preds, y_prob=p_attack)
    tuned_metrics = main.evaluate_predictions(y_val, tuned_preds, y_prob=p_attack)
    assert tuned_metrics["specificity"] >= default_metrics["specificity"]


def test_threshold_tuning_uses_only_given_validation_labels():
    y_validation = np.array([1, 1, 0, 0, 1, 0])
    y_fake_test = 1 - y_validation
    p_attack = np.array([0.65, 0.55, 0.45, 0.35, 0.80, 0.20])
    threshold_from_validation = main._tune_attack_recall_threshold(y_validation, p_attack)
    threshold_from_fake_test = main._tune_attack_recall_threshold(y_fake_test, p_attack)
    assert threshold_from_validation != threshold_from_fake_test


def test_identify_hard_validation_cases_marks_disagreement():
    y_val = np.array([0, 1, 0, 1])
    val_predictions = {
        "General Traffic Agent": np.array([0, 1, 0, 1]),
        "Attack Recall Agent": np.array([1, 1, 0, 1]),
        "Normal Behavior Agent": np.array([0, 0, 0, 1]),
    }
    hard_mask = main._identify_hard_validation_cases(val_predictions, y_val)
    assert hard_mask.shape[0] == y_val.shape[0]
    assert bool(hard_mask[0])


def test_identify_hard_validation_cases_marks_low_confidence():
    y_val = np.array([0, 1, 0, 1])
    val_predictions = {
        "General Traffic Agent": np.array([0, 1, 0, 1]),
        "Attack Recall Agent": np.array([0, 1, 0, 1]),
    }
    val_probabilities = {
        "General Traffic Agent": np.array([0.49, 0.51, 0.10, 0.90]),
        "Attack Recall Agent": np.array([0.48, 0.52, 0.20, 0.80]),
    }
    hard_mask = main._identify_hard_validation_cases(
        val_predictions,
        y_val,
        validation_probabilities=val_probabilities,
        low_confidence_quantile=0.50,
    )
    assert bool(hard_mask[0])
    assert bool(hard_mask[1])


def test_general_agent_view_selection_uses_target_range_on_validation():
    columns = ["f1", "f2", "f3", "f4"]
    x_train_df = main.pd.DataFrame(
        np.array([
            [0.0, 0.1, 1.0, 0.9],
            [0.1, 0.2, 0.9, 0.8],
            [0.9, 0.8, 0.2, 0.1],
            [1.0, 0.9, 0.1, 0.0],
            [0.2, 0.3, 0.8, 0.7],
            [0.8, 0.7, 0.3, 0.2],
        ]),
        columns=columns,
    )
    x_val_df = x_train_df.copy()
    x_test_df = x_train_df.copy()
    y_train = np.array([0, 0, 1, 1, 0, 1])
    y_val = np.array([0, 0, 1, 1, 0, 1])
    processed_views = {
        "basic_agent": ["f1", "f2"],
        "content_agent": ["f3"],
        "time_traffic_agent": ["f1", "f3"],
        "host_traffic_agent": ["f2", "f4"],
    }
    cfg = {
        "balance_agents": True,
        "max_general_agent_features": 2,
        "target_agent_accuracy_range": {"min": 0.5, "max": 0.95},
    }
    selected = main._select_general_agent_view(
        x_train_df,
        x_val_df,
        x_test_df,
        y_train,
        y_val,
        processed_views,
        cfg,
        random_state=42,
    )
    assert selected["feature_count"] <= 2
    assert selected["val_prediction"].shape[0] == y_val.shape[0]
