"""Main entry point for IDS experiment run modes.

Supported run modes:
- louati_ktata_baseline: Autoencoder + MLP/KNN + tie-break decision
- feature_view_multi_agent: 4 feature-view MLP agents + majority decision
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.agent_factory import create_detection_agent
from pipeline.data_loader import load_dataset, load_yaml_config
from pipeline.evaluation import evaluate_model, evaluate_predictions
from pipeline.feature_views import get_nsl_kdd_feature_views, map_processed_feature_views
from pipeline.preprocessing import build_preprocessing_pipeline, sanitize_feature_values
from pipeline.training import run_cross_validation, train_agent


NSL_KDD_ATTACK_GROUPS = {
    "dos": {
        "neptune",
        "smurf",
        "teardrop",
        "back",
        "land",
        "pod",
        "apache2",
        "mailbomb",
        "processtable",
        "udpstorm",
    },
    "probe": {"satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"},
    "r2l": {
        "guess_passwd",
        "ftp_write",
        "imap",
        "phf",
        "multihop",
        "warezclient",
        "warezmaster",
        "spy",
        "named",
        "sendmail",
        "snmpgetattack",
        "snmpguess",
        "xlock",
        "xsnoop",
        "worm",
    },
    "u2r": {
        "buffer_overflow",
        "rootkit",
        "perl",
        "loadmodule",
        "httptunnel",
        "ps",
        "sqlattack",
        "xterm",
    },
}


def _convert_to_binary_labels(y: pd.Series, preprocessing_config: Dict[str, Any]) -> pd.Series:
    if not preprocessing_config.get("binary_classification", False):
        return y.astype(int)

    normal_values = preprocessing_config.get("normal_label_values", ["normal", "Normal", 0])
    normal_aliases = {"benign", "BENIGN", "Benign"}

    def normalize(value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    valid_normal = {normalize(value) for value in normal_values}
    valid_normal.update({normalize(value) for value in normal_aliases})

    normalized = y.apply(normalize)
    if normalized.isna().any():
        raise ValueError("Label column contains missing values; cannot convert to binary labels.")

    return normalized.apply(lambda value: 0 if value in valid_normal else 1).astype(int)


def _to_nsl_kdd_multiclass_labels(
    y_raw: pd.Series,
    preprocessing_config: Dict[str, Any],
) -> pd.Series:
    normal_values = preprocessing_config.get("normal_label_values", ["normal", "Normal", 0])

    def normalize(value):
        if isinstance(value, str):
            return value.strip().lower().rstrip(".")
        return value

    valid_normal = {normalize(value) for value in normal_values}
    valid_normal.update({"benign"})

    inverse_group = {
        attack_name: group_name
        for group_name, attack_names in NSL_KDD_ATTACK_GROUPS.items()
        for attack_name in attack_names
    }

    mapped_labels: List[str] = []
    unknown_attacks: set[str] = set()

    for value in y_raw.tolist():
        normalized = normalize(value)
        if normalized in valid_normal:
            mapped_labels.append("normal")
            continue

        if normalized in inverse_group:
            mapped_labels.append(inverse_group[normalized])
            continue

        unknown_attacks.add(str(normalized))
        mapped_labels.append("unknown")

    if unknown_attacks:
        raise ValueError(
            "Unknown NSL-KDD attack label(s) encountered while building multiclass labels: "
            + ", ".join(sorted(unknown_attacks))
        )

    return pd.Series(mapped_labels, index=y_raw.index, dtype="object")


def _compute_attack_group_recalls(y_multiclass: pd.Series, y_pred_binary: np.ndarray) -> Dict[str, float]:
    recalls: Dict[str, float] = {}
    for group_name in ["dos", "probe", "r2l", "u2r"]:
        mask = (y_multiclass == group_name).to_numpy()
        denom = int(np.sum(mask))
        if denom == 0:
            recalls[f"{group_name}_recall"] = np.nan
            continue

        tp = int(np.sum((y_pred_binary == 1) & mask))
        fn = int(np.sum((y_pred_binary == 0) & mask))
        recalls[f"{group_name}_recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    return recalls


def _print_per_class_recall_table(per_class_df: pd.DataFrame) -> None:
    print("\nPer-attack-type recall by agent")
    for _, row in per_class_df.iterrows():
        print(f"\nAgent: {row['agent']}")
        print(f"DoS Recall: {row['dos_recall']:.4f}" if pd.notna(row["dos_recall"]) else "DoS Recall: N/A")
        print(
            f"Probe Recall: {row['probe_recall']:.4f}"
            if pd.notna(row["probe_recall"])
            else "Probe Recall: N/A"
        )
        print(f"R2L Recall: {row['r2l_recall']:.4f}" if pd.notna(row["r2l_recall"]) else "R2L Recall: N/A")
        print(f"U2R Recall: {row['u2r_recall']:.4f}" if pd.notna(row["u2r_recall"]) else "U2R Recall: N/A")


def _metrics_row(dataset_name: str, stage: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "stage": stage,
        "Accuracy": metrics.get("test_accuracy"),
        "Precision": metrics.get("test_precision"),
        "Recall": metrics.get("test_recall"),
        "F1": metrics.get("test_f1"),
        "TP": metrics.get("tp"),
        "TN": metrics.get("tn"),
        "FP": metrics.get("fp"),
        "FN": metrics.get("fn"),
        "FPR": metrics.get("fpr"),
        "FNR": metrics.get("fnr"),
        "Specificity": metrics.get("specificity"),
        "Balanced Accuracy": metrics.get("balanced_accuracy"),
    }


def _print_comparison_table(table: pd.DataFrame) -> None:
    display_columns = [
        "stage",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "TP",
        "TN",
        "FP",
        "FN",
        "FPR",
        "FNR",
        "Specificity",
        "Balanced Accuracy",
    ]

    formatted = table[display_columns].copy()
    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
        "Specificity",
        "Balanced Accuracy",
    ]:
        formatted[col] = formatted[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "N/A")

    print("\nFinal comparison table")
    print(formatted.to_string(index=False))


def _load_nsl_dataset(config: Dict[str, Any], project_root: Path, dataset_name: str) -> Dict[str, Any]:
    datasets = list(config.get("datasets", []))
    selected = None
    for dataset in datasets:
        if str(dataset.get("name", "")).strip().lower() == dataset_name.strip().lower():
            selected = dict(dataset)
            break

    if selected is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

    selected["path"] = str((project_root / selected["path"]).resolve())
    return selected


def _prepare_dataset(
    config: Dict[str, Any],
    project_root: Path,
    dataset_name: str,
) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    preprocessing_cfg = dict(config.get("preprocessing", {}))
    dataset_cfg = _load_nsl_dataset(config, project_root, dataset_name)

    print(f"Dataset loaded: {dataset_name}")
    x, y = load_dataset(dataset_cfg)
    y = _convert_to_binary_labels(y, preprocessing_cfg)

    x, _ = sanitize_feature_values(x, dataset_cfg.get("categorical_columns", []))
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(preprocessing_cfg.get("test_size", 0.3)),
        random_state=int(config.get("experiment", {}).get("random_state", 42)),
        stratify=y,
    )

    return dataset_cfg, x_train, x_test, y_train, y_test


def _run_louati_ktata_baseline(config: Dict[str, Any], project_root: Path, results_dir: Path) -> None:
    baseline_cfg = dict(config.get("baseline", {}))
    preprocessing_cfg = dict(config.get("preprocessing", {}))

    dataset_name = str(baseline_cfg.get("dataset", "NSL-KDD"))
    use_autoencoder = bool(baseline_cfg.get("use_autoencoder", True))
    selected_agents = [str(agent).lower() for agent in baseline_cfg.get("agents", ["mlp", "knn"])]
    decision_strategy = str(baseline_cfg.get("decision_strategy", "knn_tiebreak"))
    use_trust_layer = bool(baseline_cfg.get("use_trust_layer", False))
    use_cv = bool(baseline_cfg.get("cross_validation", False))

    if use_trust_layer:
        raise ValueError("This baseline does not support trust layer. Set use_trust_layer=false.")
    if decision_strategy != "knn_tiebreak":
        raise ValueError("Unsupported decision_strategy. Expected 'knn_tiebreak'.")
    if sorted(selected_agents) != ["knn", "mlp"]:
        raise ValueError("This baseline requires agents: [mlp, knn].")

    dataset_cfg, x_train, x_test, y_train, y_test = _prepare_dataset(config, project_root, dataset_name)

    preprocessor = build_preprocessing_pipeline(
        x_train,
        dataset_cfg.get("categorical_columns", []),
        preprocessing_cfg,
    )
    x_train_processed = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float64)
    x_test_processed = np.asarray(preprocessor.transform(x_test), dtype=np.float64)
    print("Preprocessing complete")

    model_cfg = dict(config.get("models", {}))

    if not use_autoencoder:
        raise ValueError("This baseline requires use_autoencoder=true.")

    autoencoder_agent = create_detection_agent("AutoencoderAgent", model_cfg.get("autoencoder", {}))
    x_train_encoded = autoencoder_agent.fit_transform(x_train_processed)
    x_test_encoded = autoencoder_agent.transform(x_test_processed)
    print("Autoencoder training complete")

    encoded_train_df = pd.DataFrame(x_train_encoded, index=x_train.index)
    encoded_train_df["label"] = y_train.values
    encoded_test_df = pd.DataFrame(x_test_encoded, index=x_test.index)
    encoded_test_df["label"] = y_test.values
    encoded_train_df.to_csv(results_dir / "encoded_train_features.csv", index=True)
    encoded_test_df.to_csv(results_dir / "encoded_test_features.csv", index=True)

    mlp_agent = create_detection_agent("MLPAgent", model_cfg.get("mlp", {}))
    mlp_agent = train_agent(mlp_agent, x_train_encoded, y_train)
    mlp_output = mlp_agent.predict(x_test_encoded)
    mlp_metrics = evaluate_model(
        mlp_agent,
        x_test_encoded,
        y_test,
        predictions=mlp_output["y_pred"],
        y_prob=mlp_output["y_prob"],
    )
    print("MLP training complete")

    knn_agent = create_detection_agent("KNNAgent", model_cfg.get("knn", {}))
    knn_agent = train_agent(knn_agent, x_train_encoded, y_train)
    knn_output = knn_agent.predict(x_test_encoded)
    knn_metrics = evaluate_model(
        knn_agent,
        x_test_encoded,
        y_test,
        predictions=knn_output["y_pred"],
        y_prob=knn_output["y_prob"],
    )
    print("KNN training complete")

    mlp_pred = np.asarray(mlp_output["y_pred"], dtype=int)
    knn_pred = np.asarray(knn_output["y_pred"], dtype=int)
    mlp_prob = np.asarray(mlp_output["y_prob"], dtype=np.float64)
    knn_prob = np.asarray(knn_output["y_prob"], dtype=np.float64)

    agreement = mlp_pred == knn_pred
    final_pred = np.where(agreement, mlp_pred, knn_pred)
    final_prob = np.where(agreement, (mlp_prob + knn_prob) / 2.0, knn_prob)

    final_metrics = evaluate_predictions(y_test, final_pred, y_prob=final_prob)

    if use_cv:
        mlp_cv = run_cross_validation(
            create_detection_agent("MLPAgent", model_cfg.get("mlp", {})).model,
            x_train_encoded,
            y_train,
            int(config.get("validation", {}).get("cross_validation_folds", 5)),
        )
        knn_cv = run_cross_validation(
            create_detection_agent("KNNAgent", model_cfg.get("knn", {})).model,
            x_train_encoded,
            y_train,
            int(config.get("validation", {}).get("cross_validation_folds", 5)),
        )
    else:
        mlp_cv = {"cv_accuracy": np.nan}
        knn_cv = {"cv_accuracy": np.nan}

    metrics_table = pd.DataFrame(
        [
            _metrics_row(dataset_name, "Stage 1: MLP Agent", mlp_metrics),
            _metrics_row(dataset_name, "Stage 2: KNN Agent", knn_metrics),
            _metrics_row(dataset_name, "Stage 3: Multi-Agent Decision", final_metrics),
        ]
    )

    _print_comparison_table(metrics_table)
    print("Final comparison table printed")

    metrics_table.to_csv(results_dir / "experiment_results.csv", index=False)

    interaction_summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "decision_strategy": decision_strategy,
                "agreements": int(agreement.sum()),
                "disagreements": int((~agreement).sum()),
                "agreement_rate": float(np.mean(agreement)),
                "disagreement_rate": float(np.mean(~agreement)),
                "cross_validation_enabled": use_cv,
                "use_trust_layer": False,
                "mlp_cv_accuracy": mlp_cv.get("cv_accuracy"),
                "knn_cv_accuracy": knn_cv.get("cv_accuracy"),
            }
        ]
    )
    interaction_summary.to_csv(results_dir / "agent_agreement.csv", index=False)

    sample_predictions = pd.DataFrame(
        {
            "dataset": dataset_name,
            "sample_index": x_test.index,
            "true_label": y_test.values,
            "mlp_prediction": mlp_pred,
            "knn_prediction": knn_pred,
            "agreement": agreement,
            "final_prediction": final_pred,
            "final_probability_attack": final_prob,
        }
    )
    sample_predictions.to_csv(results_dir / "sample_level_predictions.csv", index=False)


def _run_feature_view_multi_agent(config: Dict[str, Any], project_root: Path, results_dir: Path) -> None:
    preprocessing_cfg = dict(config.get("preprocessing", {}))
    model_cfg = dict(config.get("models", {}))

    dataset_name = str(config.get("dataset", "NSL-KDD"))
    decision_strategy = str(config.get("decision_strategy", "majority_vote"))
    use_trust_layer = bool(config.get("use_trust_layer", False))
    use_attention = bool(config.get("use_attention", False))
    use_autoencoder = bool(config.get("use_autoencoder", False))
    use_cv = bool(config.get("cross_validation", False))

    if dataset_name.strip().lower() != "nsl-kdd":
        raise ValueError("feature_view_multi_agent currently supports dataset=NSL-KDD only.")
    if decision_strategy != "majority_vote":
        raise ValueError("feature_view_multi_agent requires decision_strategy='majority_vote'.")
    if use_trust_layer:
        raise ValueError("feature_view_multi_agent does not support trust layer yet.")
    if use_attention:
        raise ValueError("feature_view_multi_agent does not support attention layer yet.")
    if use_autoencoder:
        raise ValueError("feature_view_multi_agent requires use_autoencoder=false.")
    if use_cv:
        raise ValueError("feature_view_multi_agent requires cross_validation=false for now.")

    dataset_cfg = _load_nsl_dataset(config, project_root, dataset_name)

    print(f"Dataset loaded: {dataset_name}")
    x, y_raw = load_dataset(dataset_cfg)
    y_multiclass = _to_nsl_kdd_multiclass_labels(y_raw, preprocessing_cfg)
    y_binary = _convert_to_binary_labels(y_raw, preprocessing_cfg)

    x, _ = sanitize_feature_values(x, dataset_cfg.get("categorical_columns", []))
    x_train, x_test, y_train, y_test, y_multiclass_train, y_multiclass_test = train_test_split(
        x,
        y_binary,
        y_multiclass,
        test_size=float(preprocessing_cfg.get("test_size", 0.3)),
        random_state=int(config.get("experiment", {}).get("random_state", 42)),
        stratify=y_binary,
    )

    preprocessor = build_preprocessing_pipeline(
        x_train,
        dataset_cfg.get("categorical_columns", []),
        preprocessing_cfg,
    )
    x_train_processed = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float64)
    x_test_processed = np.asarray(preprocessor.transform(x_test), dtype=np.float64)
    print("Preprocessing complete")

    try:
        processed_feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(
            "Could not extract transformed feature names from preprocessing pipeline. "
            "Feature-view mapping requires named transformed features."
        ) from exc

    raw_feature_names = list(x_train.columns)
    raw_views = get_nsl_kdd_feature_views(raw_feature_names)
    processed_views = map_processed_feature_views(processed_feature_names, raw_views)

    x_train_df = pd.DataFrame(x_train_processed, columns=processed_feature_names, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_processed, columns=processed_feature_names, index=x_test.index)

    x_train_basic = x_train_df[processed_views["basic_agent"]].to_numpy(dtype=np.float64)
    x_test_basic = x_test_df[processed_views["basic_agent"]].to_numpy(dtype=np.float64)

    x_train_content = x_train_df[processed_views["content_agent"]].to_numpy(dtype=np.float64)
    x_test_content = x_test_df[processed_views["content_agent"]].to_numpy(dtype=np.float64)

    x_train_time = x_train_df[processed_views["time_traffic_agent"]].to_numpy(dtype=np.float64)
    x_test_time = x_test_df[processed_views["time_traffic_agent"]].to_numpy(dtype=np.float64)

    x_train_host = x_train_df[processed_views["host_traffic_agent"]].to_numpy(dtype=np.float64)
    x_test_host = x_test_df[processed_views["host_traffic_agent"]].to_numpy(dtype=np.float64)

    print("Feature views created")

    def _train_view_agent(x_train_view, x_test_view, stage_label: str, completion_message: str):
        agent = create_detection_agent("MLPAgent", model_cfg.get("mlp", {}))
        agent = train_agent(agent, x_train_view, y_train)
        output = agent.predict(x_test_view)
        metrics = evaluate_model(
            agent,
            x_test_view,
            y_test,
            predictions=output["y_pred"],
            y_prob=output["y_prob"],
        )
        print(completion_message)
        return {
            "stage": stage_label,
            "prediction": np.asarray(output["y_pred"], dtype=int),
            "metrics": metrics,
        }

    basic_runtime = _train_view_agent(
        x_train_basic,
        x_test_basic,
        "Stage 1: Basic Feature Agent",
        "Basic Feature Agent training complete",
    )
    content_runtime = _train_view_agent(
        x_train_content,
        x_test_content,
        "Stage 2: Content Feature Agent",
        "Content Feature Agent training complete",
    )
    time_runtime = _train_view_agent(
        x_train_time,
        x_test_time,
        "Stage 3: Time Traffic Feature Agent",
        "Time Traffic Feature Agent training complete",
    )
    host_runtime = _train_view_agent(
        x_train_host,
        x_test_host,
        "Stage 4: Host Traffic Feature Agent",
        "Host Traffic Feature Agent training complete",
    )

    stacked_preds = np.vstack(
        [
            basic_runtime["prediction"],
            content_runtime["prediction"],
            time_runtime["prediction"],
            host_runtime["prediction"],
        ]
    )

    trust_scores = {
        "Basic Feature Agent": float(basic_runtime["metrics"]["test_f1"]),
        "Content Agent": float(content_runtime["metrics"]["test_f1"]),
        "Time Traffic Agent": float(time_runtime["metrics"]["test_f1"]),
        "Host Traffic Agent": float(host_runtime["metrics"]["test_f1"]),
    }
    trust_sum = float(sum(trust_scores.values()))
    if trust_sum <= 0.0:
        trust_weights = {
            name: 0.25
            for name in trust_scores
        }
    else:
        trust_weights = {
            name: float(score / trust_sum)
            for name, score in trust_scores.items()
        }

    trust_weights_df = pd.DataFrame(
        [
            {
                "agent": agent,
                "trust_score": trust_scores[agent],
                "trust_weight": trust_weights[agent],
            }
            for agent in [
                "Basic Feature Agent",
                "Content Agent",
                "Time Traffic Agent",
                "Host Traffic Agent",
            ]
        ]
    )
    print("Trust weights computed")
    print(trust_weights_df.to_string(index=False))

    attack_votes = np.sum(stacked_preds == 1, axis=0)
    majority_pred = np.where(attack_votes >= 2, 1, 0)
    majority_metrics = evaluate_predictions(y_test, majority_pred)
    print("Multi-Agent Majority Decision complete")

    weighted_attack_score = (
        trust_weights["Basic Feature Agent"] * basic_runtime["prediction"]
        + trust_weights["Content Agent"] * content_runtime["prediction"]
        + trust_weights["Time Traffic Agent"] * time_runtime["prediction"]
        + trust_weights["Host Traffic Agent"] * host_runtime["prediction"]
    )
    trust_weighted_pred = np.where(weighted_attack_score >= 0.5, 1, 0)
    trust_weighted_metrics = evaluate_predictions(y_test, trust_weighted_pred)
    print("Trust-Weighted Multi-Agent Decision complete")

    base_r2l = _compute_attack_group_recalls(y_multiclass_test, basic_runtime["prediction"])["r2l_recall"]
    base_u2r = _compute_attack_group_recalls(y_multiclass_test, basic_runtime["prediction"])["u2r_recall"]
    content_r2l = _compute_attack_group_recalls(y_multiclass_test, content_runtime["prediction"])["r2l_recall"]
    content_u2r = _compute_attack_group_recalls(y_multiclass_test, content_runtime["prediction"])["u2r_recall"]
    time_r2l = _compute_attack_group_recalls(y_multiclass_test, time_runtime["prediction"])["r2l_recall"]
    time_u2r = _compute_attack_group_recalls(y_multiclass_test, time_runtime["prediction"])["u2r_recall"]
    host_r2l = _compute_attack_group_recalls(y_multiclass_test, host_runtime["prediction"])["r2l_recall"]
    host_u2r = _compute_attack_group_recalls(y_multiclass_test, host_runtime["prediction"])["u2r_recall"]

    rare_trust_scores = {
        "Basic Feature Agent": float(np.nanmean([base_r2l, base_u2r])),
        "Content Agent": float(np.nanmean([content_r2l, content_u2r])),
        "Time Traffic Agent": float(np.nanmean([time_r2l, time_u2r])),
        "Host Traffic Agent": float(np.nanmean([host_r2l, host_u2r])),
    }
    rare_trust_sum = float(sum(rare_trust_scores.values()))
    if rare_trust_sum <= 0.0:
        rare_trust_weights = {name: 0.25 for name in rare_trust_scores}
    else:
        rare_trust_weights = {
            name: float(score / rare_trust_sum)
            for name, score in rare_trust_scores.items()
        }

    rare_attack_trust_weights_df = pd.DataFrame(
        [
            {
                "agent": "Basic Feature Agent",
                "r2l_recall": base_r2l,
                "u2r_recall": base_u2r,
                "rare_trust_score": rare_trust_scores["Basic Feature Agent"],
                "rare_trust_weight": rare_trust_weights["Basic Feature Agent"],
            },
            {
                "agent": "Content Agent",
                "r2l_recall": content_r2l,
                "u2r_recall": content_u2r,
                "rare_trust_score": rare_trust_scores["Content Agent"],
                "rare_trust_weight": rare_trust_weights["Content Agent"],
            },
            {
                "agent": "Time Traffic Agent",
                "r2l_recall": time_r2l,
                "u2r_recall": time_u2r,
                "rare_trust_score": rare_trust_scores["Time Traffic Agent"],
                "rare_trust_weight": rare_trust_weights["Time Traffic Agent"],
            },
            {
                "agent": "Host Traffic Agent",
                "r2l_recall": host_r2l,
                "u2r_recall": host_u2r,
                "rare_trust_score": rare_trust_scores["Host Traffic Agent"],
                "rare_trust_weight": rare_trust_weights["Host Traffic Agent"],
            },
        ]
    )
    print("Rare attack trust weights computed")
    print(rare_attack_trust_weights_df.to_string(index=False))

    rare_attack_support_score = (
        rare_trust_weights["Basic Feature Agent"] * basic_runtime["prediction"]
        + rare_trust_weights["Content Agent"] * content_runtime["prediction"]
        + rare_trust_weights["Time Traffic Agent"] * time_runtime["prediction"]
        + rare_trust_weights["Host Traffic Agent"] * host_runtime["prediction"]
    )
    attack_aware_final_score = np.where(
        majority_pred == 1,
        np.maximum(weighted_attack_score, rare_attack_support_score),
        0.0,
    )
    attack_aware_trust_pred = np.where(attack_aware_final_score >= 0.5, 1, 0)
    attack_aware_trust_metrics = evaluate_predictions(y_test, attack_aware_trust_pred)
    print("Attack-Aware Trust Decision complete")

    per_class_rows = [
        {
            "agent": "Basic Feature Agent",
            **_compute_attack_group_recalls(y_multiclass_test, basic_runtime["prediction"]),
        },
        {
            "agent": "Content Agent",
            **_compute_attack_group_recalls(y_multiclass_test, content_runtime["prediction"]),
        },
        {
            "agent": "Time Traffic Agent",
            **_compute_attack_group_recalls(y_multiclass_test, time_runtime["prediction"]),
        },
        {
            "agent": "Host Traffic Agent",
            **_compute_attack_group_recalls(y_multiclass_test, host_runtime["prediction"]),
        },
        {
            "agent": "Majority Decision",
            **_compute_attack_group_recalls(y_multiclass_test, majority_pred),
        },
        {
            "agent": "Trust-Weighted Decision",
            **_compute_attack_group_recalls(y_multiclass_test, trust_weighted_pred),
        },
        {
            "agent": "Attack-Aware Trust Decision",
            **_compute_attack_group_recalls(y_multiclass_test, attack_aware_trust_pred),
        },
    ]
    per_class_df = pd.DataFrame(per_class_rows)
    _print_per_class_recall_table(per_class_df)

    metrics_table = pd.DataFrame(
        [
            _metrics_row(dataset_name, basic_runtime["stage"], basic_runtime["metrics"]),
            _metrics_row(dataset_name, content_runtime["stage"], content_runtime["metrics"]),
            _metrics_row(dataset_name, time_runtime["stage"], time_runtime["metrics"]),
            _metrics_row(dataset_name, host_runtime["stage"], host_runtime["metrics"]),
            _metrics_row(dataset_name, "Stage 5: Multi-Agent Majority Decision", majority_metrics),
            _metrics_row(
                dataset_name,
                "Stage 6: Trust-Weighted Multi-Agent Decision",
                trust_weighted_metrics,
            ),
            _metrics_row(
                dataset_name,
                "Stage 7: Attack-Aware Trust Decision",
                attack_aware_trust_metrics,
            ),
        ]
    )

    _print_comparison_table(metrics_table)
    print("Final comparison table printed")

    metrics_table.to_csv(results_dir / "experiment_results.csv", index=False)
    metrics_table[metrics_table["stage"].isin([
        "Stage 1: Basic Feature Agent",
        "Stage 2: Content Feature Agent",
        "Stage 3: Time Traffic Feature Agent",
        "Stage 4: Host Traffic Feature Agent",
    ])].to_csv(results_dir / "agent_level_results.csv", index=False)
    trust_weights_df.to_csv(results_dir / "trust_weights.csv", index=False)
    rare_attack_trust_weights_df.to_csv(results_dir / "rare_attack_trust_weights.csv", index=False)

    sample_predictions = pd.DataFrame(
        {
            "y_true": y_test.values,
            "basic_agent_prediction": basic_runtime["prediction"],
            "content_agent_prediction": content_runtime["prediction"],
            "time_traffic_agent_prediction": time_runtime["prediction"],
            "host_traffic_agent_prediction": host_runtime["prediction"],
            "majority_decision_prediction": majority_pred,
            "trust_weighted_decision_prediction": trust_weighted_pred,
            "weighted_attack_score": weighted_attack_score,
            "rare_attack_support_score": rare_attack_support_score,
            "attack_aware_trust_prediction": attack_aware_trust_pred,
        },
        index=x_test.index,
    )
    sample_predictions.to_csv(results_dir / "sample_level_predictions.csv", index=False)
    per_class_df.to_csv(results_dir / "agent_per_class_performance.csv", index=False)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*encountered in matmul.*",
        category=RuntimeWarning,
    )

    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "experiment.yml"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(config_path)
    run_mode = str(config.get("run_mode", "louati_ktata_baseline"))

    if run_mode == "feature_view_multi_agent":
        _run_feature_view_multi_agent(config, project_root, results_dir)
        return

    _run_louati_ktata_baseline(config, project_root, results_dir)


if __name__ == "__main__":
    main()
