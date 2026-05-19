import numpy as np
import tempfile
from pathlib import Path

import main
from pipeline.poisoning import (
    FULL_POISON_OUTPUT_COLUMNS,
    ROBUSTNESS_COLUMNS,
    build_poisoned_agent_prediction_sets,
    get_available_poison_trust_method_names,
    poison_predictions,
    run_poisoned_agent_experiments,
    save_poisoned_comparison_outputs,
)


def _sample_context():
    model_preds = {
        "General Traffic Agent": np.array([0, 1, 0, 1, 0, 1], dtype=int),
        "Attack Recall Agent": np.array([1, 1, 0, 1, 1, 0], dtype=int),
        "Normal Behavior Agent": np.array([0, 0, 0, 0, 0, 1], dtype=int),
        "Hard-Case Agent": np.array([1, 1, 1, 1, 0, 1], dtype=int),
    }
    model_probs = {name: arr.astype(float) for name, arr in model_preds.items()}
    validation_model_metrics = {
        "General Traffic Agent": {
            "test_f1": 0.80,
            "test_accuracy": 0.78,
            "test_recall": 0.76,
            "specificity": 0.80,
            "fnr": 0.24,
        },
        "Attack Recall Agent": {
            "test_f1": 0.82,
            "test_accuracy": 0.79,
            "test_recall": 0.90,
            "specificity": 0.65,
            "fnr": 0.10,
        },
        "Normal Behavior Agent": {
            "test_f1": 0.76,
            "test_accuracy": 0.77,
            "test_recall": 0.70,
            "specificity": 0.90,
            "fnr": 0.30,
        },
        "Hard-Case Agent": {
            "test_f1": 0.81,
            "test_accuracy": 0.80,
            "test_recall": 0.79,
            "specificity": 0.81,
            "fnr": 0.21,
        },
    }
    roles = {
        "General Traffic Agent": "general",
        "Attack Recall Agent": "attack_recall",
        "Normal Behavior Agent": "normal_behavior",
        "Hard-Case Agent": "hard_case",
    }
    x_val = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [0.8, 0.9],
            [0.9, 0.8],
            [0.2, 0.2],
            [0.7, 0.7],
        ]
    )
    y_val = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    x_test = x_val.copy()
    y_test = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    val_preds = {name: arr.copy() for name, arr in model_preds.items()}
    selector_params = {
        "neighbor_k": 3,
        "validation_role_weight": 0.25,
        "confidence_weight": 0.20,
        "margin_weight": 0.15,
        "local_accuracy_weight": 0.10,
        "disagreement_bonus": 0.10,
        "attack_role_bonus": 0.08,
        "normal_role_bonus": 0.08,
        "attack_confidence_threshold": 0.60,
        "normal_confidence_threshold": 0.65,
    }
    return {
        "model_preds": model_preds,
        "model_probs": model_probs,
        "validation_model_metrics": validation_model_metrics,
        "roles": roles,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "val_preds": val_preds,
        "selector_params": selector_params,
        "role_aware_cfg": {"attack_threshold": 0.60, "normal_threshold": 0.65},
    }


def test_poison_predictions_does_not_mutate_input():
    original = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    snapshot = original.copy()
    poisoned = poison_predictions(original, poison_rate=0.5, mode="flip", random_state=42)
    assert np.array_equal(original, snapshot)
    assert not np.shares_memory(original, poisoned)


def test_flip_mode_is_deterministic():
    arr = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    out_a = poison_predictions(arr, poison_rate=0.375, mode="flip", random_state=123)
    out_b = poison_predictions(arr, poison_rate=0.375, mode="flip", random_state=123)
    assert np.array_equal(out_a, out_b)


def test_normal_bias_forces_selected_predictions_to_zero():
    arr = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=int)
    poisoned = poison_predictions(arr, poison_rate=0.3, mode="normal_bias", random_state=7)
    n = arr.shape[0]
    poison_idx = np.random.default_rng(7).choice(n, size=int(n * 0.3), replace=False)
    assert np.all(poisoned[poison_idx] == 0)
    clean_idx = np.setdiff1d(np.arange(n), poison_idx)
    assert np.array_equal(poisoned[clean_idx], arr[clean_idx])


def test_attack_bias_forces_selected_predictions_to_one():
    arr = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=int)
    poisoned = poison_predictions(arr, poison_rate=0.4, mode="attack_bias", random_state=11)
    n = arr.shape[0]
    poison_idx = np.random.default_rng(11).choice(n, size=int(n * 0.4), replace=False)
    assert np.all(poisoned[poison_idx] == 1)
    clean_idx = np.setdiff1d(np.arange(n), poison_idx)
    assert np.array_equal(poisoned[clean_idx], arr[clean_idx])


def test_poisoned_experiment_poisons_exactly_one_agent_at_a_time():
    preds = {
        "Agent A": np.array([0, 1, 0, 1, 0, 1], dtype=int),
        "Agent B": np.array([1, 1, 0, 0, 1, 0], dtype=int),
        "Agent C": np.array([0, 0, 1, 1, 0, 1], dtype=int),
    }
    poisoned_sets = build_poisoned_agent_prediction_sets(
        preds,
        poison_rate=0.5,
        poison_mode="flip",
        poison_random_state=42,
    )
    for poisoned_agent, scenario_preds in poisoned_sets.items():
        changed_agents = [
            agent_name
            for agent_name in preds
            if not np.array_equal(
                np.asarray(preds[agent_name], dtype=int),
                np.asarray(scenario_preds[agent_name], dtype=int),
            )
        ]
        assert changed_agents == [poisoned_agent]


def test_poisoned_output_has_full_metric_columns():
    ctx = _sample_context()
    result_df = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
    )
    for col in FULL_POISON_OUTPUT_COLUMNS:
        assert col in result_df.columns


def test_original_and_poisoned_rows_both_exist_per_agent():
    ctx = _sample_context()
    result_df = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
    )
    for agent_name in ctx["model_preds"]:
        subset = result_df[result_df["Poisoned Agent"] == agent_name]
        assert (subset["Evaluation Type"] == "Original Agent").any()
        assert (subset["Evaluation Type"] == "Poisoned Agent").any()


def test_all_trust_methods_are_included():
    ctx = _sample_context()
    expected_methods = set(
        get_available_poison_trust_method_names(
            model_probs=ctx["model_probs"],
            validation_model_metrics=ctx["validation_model_metrics"],
            roles=ctx["roles"],
            x_val_full=ctx["x_val"],
            y_val=ctx["y_val"],
            validation_predictions=ctx["val_preds"],
            x_test_full=ctx["x_test"],
            role_aware_cfg=ctx["role_aware_cfg"],
            selector_params=ctx["selector_params"],
        )
    )
    result_df = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
    )
    eval_types = set(result_df["Evaluation Type"].tolist())
    assert expected_methods.issubset(eval_types)


def test_recovery_metrics_compute_correctly():
    ctx = _sample_context()
    result_df = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
    )
    for agent_name in ctx["model_preds"]:
        subset = result_df[result_df["Poisoned Agent"] == agent_name]
        majority = subset[subset["Evaluation Type"] == "Majority Vote"].iloc[0]
        trust_rows = subset[
            ~subset["Evaluation Type"].isin(["Original Agent", "Poisoned Agent", "Majority Vote"])
        ]
        if trust_rows.empty:
            continue
        row = trust_rows.iloc[0]
        assert np.isclose(
            float(row["Accuracy Recovery"]),
            float(row["Accuracy"] - majority["Accuracy"]),
        )
        assert np.isclose(
            float(row["F1 Recovery"]),
            float(row["F1"] - majority["F1"]),
        )
        assert np.isclose(
            float(row["FNR Reduction"]),
            float(majority["FNR"] - row["FNR"]),
        )


def test_majority_vote_metrics_are_preserved():
    ctx = _sample_context()
    result_df = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
    )
    poisoned_sets = build_poisoned_agent_prediction_sets(
        ctx["model_preds"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
    )
    for agent_name, poisoned_preds in poisoned_sets.items():
        majority_pred = main.trust_methods.majority_voting(poisoned_preds)["predictions"]
        majority_metrics = main.evaluate_predictions(ctx["y_test"], majority_pred)
        row = result_df[
            (result_df["Poisoned Agent"] == agent_name)
            & (result_df["Evaluation Type"] == "Majority Vote")
        ].iloc[0]
        assert np.isclose(float(row["Accuracy"]), float(majority_metrics["test_accuracy"]))
        assert np.isclose(float(row["F1"]), float(majority_metrics["test_f1"]))
        assert np.isclose(float(row["FNR"]), float(majority_metrics["fnr"]))


def test_clean_and_poisoned_rows_exist_in_robustness_report():
    ctx = _sample_context()
    _, artifacts = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
        return_artifacts=True,
    )
    for agent_name, df in artifacts["robustness_by_agent"].items():
        assert not df[df["Scenario"] == "Clean"].empty
        assert not df[df["Scenario"] == "Poisoned"].empty
        for col in ROBUSTNESS_COLUMNS:
            assert col in df.columns


def test_degradation_metrics_compute_correctly():
    ctx = _sample_context()
    _, artifacts = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
        return_artifacts=True,
    )
    for _, df in artifacts["robustness_by_agent"].items():
        for method_name in df["Trust Method"].unique():
            clean_row = df[(df["Trust Method"] == method_name) & (df["Scenario"] == "Clean")].iloc[0]
            poisoned_row = df[(df["Trust Method"] == method_name) & (df["Scenario"] == "Poisoned")].iloc[0]
            assert np.isclose(float(clean_row["Accuracy Drop"]), 0.0)
            assert np.isclose(float(clean_row["F1 Drop"]), 0.0)
            assert np.isclose(float(clean_row["FNR Increase"]), 0.0)
            assert np.isclose(float(clean_row["FPR Increase"]), 0.0)
            assert np.isclose(float(poisoned_row["Accuracy Drop"]), float(clean_row["Accuracy"] - poisoned_row["Accuracy"]))
            assert np.isclose(float(poisoned_row["F1 Drop"]), float(clean_row["F1"] - poisoned_row["F1"]))
            assert np.isclose(float(poisoned_row["FNR Increase"]), float(poisoned_row["FNR"] - clean_row["FNR"]))
            assert np.isclose(float(poisoned_row["FPR Increase"]), float(poisoned_row["FPR"] - clean_row["FPR"]))


def test_separate_per_agent_folders_and_summary_are_created_and_legacy_outputs_preserved():
    ctx = _sample_context()
    full_df, artifacts = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=ctx["y_test"],
        model_preds=ctx["model_preds"],
        model_probs=ctx["model_probs"],
        validation_model_metrics=ctx["validation_model_metrics"],
        roles=ctx["roles"],
        x_val_full=ctx["x_val"],
        y_val=ctx["y_val"],
        validation_predictions=ctx["val_preds"],
        x_test_full=ctx["x_test"],
        role_aware_cfg=ctx["role_aware_cfg"],
        selector_params=ctx["selector_params"],
        poison_rate=0.3,
        poison_mode="flip",
        poison_random_state=42,
        return_artifacts=True,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        save_poisoned_comparison_outputs(
            full_df,
            out_dir,
            dataset_file_stem="dummy_ds",
            robustness_artifacts=artifacts,
        )

        # legacy outputs must still exist
        assert (out_dir / "dummy_ds_poisoned_agent_comparison.csv").exists()
        assert (out_dir / "dummy_ds_poisoned_agent_comparison.md").exists()
        # full outputs must exist
        assert (out_dir / "dummy_ds_poisoned_full_comparison.csv").exists()
        assert (out_dir / "dummy_ds_poisoned_full_comparison.md").exists()
        # robustness tree
        poisoning_root = out_dir / "poisoning" / "dummy_ds"
        assert poisoning_root.exists()
        for agent_name in ctx["model_preds"]:
            slug = (
                agent_name.strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
                .replace("__", "_")
            )
            agent_dir = poisoning_root / slug
            assert (agent_dir / "full_comparison.csv").exists()
            assert (agent_dir / "full_comparison.md").exists()
            assert (agent_dir / "robustness_report.csv").exists()
            assert (agent_dir / "robustness_report.md").exists()
            assert (agent_dir / "summary.md").exists()
        assert (poisoning_root / "overall_robustness_summary.md").exists()


def test_running_without_run_poisoned_experiments_preserves_existing_behavior():
    args = main._parse_cli_args([])
    options = main._build_poison_experiment_options(args)
    assert options["run_poisoned_experiments"] is False
