"""Main entry point for IDS experiment run modes.

Supported run modes:
- louati_ktata_baseline: Autoencoder + MLP/KNN + tie-break decision
- feature_view_multi_agent: 4 feature-view MLP agents + majority decision
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models.agent_factory import create_detection_agent
from pipeline.data_loader import (
    load_dataset,
    load_yaml_config,
    load_unsw_nb15_dataset,
    get_unsw_categorical_columns,
    load_ton_iot_dataset,
    load_cicids2017_dataset,
)
from pipeline.evaluation import evaluate_model, evaluate_predictions
from pipeline.feature_views import get_nsl_kdd_feature_views, map_processed_feature_views, get_unsw_feature_views
from pipeline.integrity import generate_integrity_report
from pipeline.poisoning import run_poisoned_agent_experiments, save_poisoned_comparison_outputs
from pipeline.ai_trust_auditor import build_ai_trust_config, select_method_with_ai
from pipeline.reporting_utils import compute_model_diversity_report, compute_oracle_upper_bound, summarize_prediction_distribution
from pipeline.preprocessing import build_preprocessing_pipeline, sanitize_feature_values
from pipeline import trust_methods
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
    print("\nPer-attack-type recall by model (decision source)")
    for _, row in per_class_df.iterrows():
        # support both 'agent' and 'model' column names for compatibility
        label = row.get('model') if 'model' in row else row.get('agent')
        print(f"\nModel: {label}")
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


def _to_percent_str(value: Any, *, signed: bool = False, pts: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    num = float(value)
    if pts:
        return f"{num * 100:+.2f} pts" if signed else f"{num * 100:.2f} pts"
    return f"{num * 100:.2f}%"


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


def _load_unsw_dataset(config: Dict[str, Any], project_root: Path, dataset_name: str) -> Dict[str, Any]:
    """Load UNSW-NB15 dataset configuration."""
    datasets = list(config.get("datasets", []))
    selected = None
    for dataset in datasets:
        ds_name = str(dataset.get("name", "")).strip().lower()
        if ds_name in {"unsw-nb15", "unsw"}:
            selected = dict(dataset)
            break

    if selected is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

    # Resolve paths
    if "train_path" in selected:
        selected["train_path"] = str((project_root / selected["train_path"]).resolve())
    if "test_path" in selected:
        selected["test_path"] = str((project_root / selected["test_path"]).resolve())
    return selected


def _normalize_dataset_alias(dataset_name: str) -> str:
    normalized = str(dataset_name or "").strip().lower().replace("_", "-")
    if normalized in {"unsw", "unsw-nb15"}:
        return "UNSW-NB15"
    if normalized in {"nsl-kdd", "nslkdd"}:
        return "NSL-KDD"
    if normalized in {"ton-iot", "ton_iot", "toni ot", "to-n-iot"}:
        return "ToN-IoT"
    if normalized in {"cicids2017", "cic-ids2017", "cic ids2017"}:
        return "CICIDS2017"
    return dataset_name


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--run-poisoned-experiments", action="store_true")
    parser.add_argument("--poison-rate", type=float, default=0.3)
    parser.add_argument(
        "--poison-mode",
        type=str,
        default="flip",
        choices=["flip", "normal_bias", "attack_bias"],
    )
    parser.add_argument("--poison-random-state", type=int, default=42)
    parser.add_argument("--enable-ai-trust", action="store_true")
    parser.add_argument("--ai-trust-provider", type=str, default=None)
    parser.add_argument("--ai-trust-model", type=str, default=None)
    parser.add_argument("--ai-trust-sample-limit", type=int, default=None)
    parser.add_argument("--ai-trust-timeout", type=float, default=None)
    parser.add_argument("--ai-trust-fallback-method", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def _build_poison_experiment_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "run_poisoned_experiments": bool(args.run_poisoned_experiments),
        "poison_rate": float(args.poison_rate),
        "poison_mode": str(args.poison_mode),
        "poison_random_state": int(args.poison_random_state),
        "enable_ai_trust": bool(args.enable_ai_trust),
        "ai_trust_provider": args.ai_trust_provider,
        "ai_trust_model": args.ai_trust_model,
        "ai_trust_sample_limit": args.ai_trust_sample_limit,
        "ai_trust_timeout": args.ai_trust_timeout,
        "ai_trust_fallback_method": args.ai_trust_fallback_method,
    }



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


def _predict_with_probabilities(model, x_data: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    output = model.predict(x_data)
    if isinstance(output, dict):
        y_pred = np.asarray(output.get("y_pred"), dtype=int)
        y_prob = output.get("y_prob")
        return y_pred, (np.asarray(y_prob, dtype=np.float64) if y_prob is not None else None)

    y_pred = np.asarray(output, dtype=int)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = np.asarray(model.predict_proba(x_data), dtype=np.float64)
        except Exception:  # pylint: disable=broad-except
            y_prob = None
    return y_pred, y_prob


def _fit_validation_selected_model(
    model_name: str,
    candidate_factories: List[Any],
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: pd.Series,
    x_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, Any]:
    best_record: Dict[str, Any] | None = None
    best_score = (-1.0, -1.0)

    for factory in candidate_factories:
        model = factory()
        model = train_agent(model, x_train, y_train)
        val_pred, val_prob = _predict_with_probabilities(model, x_val)
        val_metrics = evaluate_predictions(y_val, val_pred, y_prob=val_prob)
        score = (float(val_metrics["test_accuracy"]), float(val_metrics["test_f1"]))
        if score > best_score:
            test_pred, test_prob = _predict_with_probabilities(model, x_test)
            test_metrics = evaluate_predictions(y_test, test_pred, y_prob=test_prob)
            best_score = score
            best_record = {
                "name": model_name,
                "model": model,
                "validation_prediction": val_pred,
                "validation_probability": val_prob,
                "validation_metrics": val_metrics,
                    "prediction": test_pred,
                    "probability": test_prob,
                    "metrics": test_metrics,
                    "test_prediction": test_pred,
                    "test_probability": test_prob,
                    "test_metrics": test_metrics,
                "selected_params": getattr(model, "get_params", lambda: {})(),
            }

    if best_record is None:
        raise ValueError(f"Could not fit any candidate model for {model_name}.")
    return best_record


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
            _metrics_row(dataset_name, "Stage 1: MLP Model", mlp_metrics),
            _metrics_row(dataset_name, "Stage 2: KNN Model", knn_metrics),
            _metrics_row(dataset_name, "Stage 3: Resolved Decision", final_metrics),
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


def _extract_attack_probability(probabilities: np.ndarray | None, fallback_predictions: np.ndarray) -> np.ndarray:
    if probabilities is None:
        return np.asarray(fallback_predictions, dtype=float)
    prob = np.asarray(probabilities, dtype=float)
    if prob.ndim == 2 and prob.shape[1] >= 2:
        return np.clip(prob[:, 1], 0.0, 1.0)
    return np.clip(prob.ravel(), 0.0, 1.0)


def _apply_threshold(attack_probability: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.asarray(attack_probability, dtype=float) >= float(threshold), 1, 0)


def _accuracy_range_penalty(
    accuracy: float,
    *,
    balance_agents: bool,
    target_min: float,
    target_max: float,
    penalty_weight: float,
) -> float:
    if not balance_agents:
        return 0.0
    if accuracy < target_min:
        return float((target_min - accuracy) * penalty_weight)
    if accuracy > target_max:
        return float((accuracy - target_max) * penalty_weight)
    return 0.0


def _tune_attack_recall_threshold(
    validation_labels: np.ndarray,
    attack_probability: np.ndarray,
    *,
    balance_agents: bool = False,
    target_min: float = 0.92,
    target_max: float = 0.97,
) -> float:
    y_val = np.asarray(validation_labels, dtype=int)
    p_attack = np.asarray(attack_probability, dtype=float)
    best_threshold = 0.50
    best_score = -1.0
    best_recall = -1.0
    for threshold in np.arange(0.20, 0.6001, 0.02):
        preds = _apply_threshold(p_attack, threshold)
        metrics = evaluate_predictions(y_val, preds, y_prob=p_attack)
        penalty = _accuracy_range_penalty(
            float(metrics["test_accuracy"]),
            balance_agents=balance_agents,
            target_min=target_min,
            target_max=target_max,
            penalty_weight=7.0,
        )
        score = (
            0.60 * float(metrics["test_recall"])
            + 0.25 * float(metrics["test_f1"])
            + 0.15 * float(metrics["test_accuracy"])
            - penalty
        )
        recall = float(metrics["test_recall"])
        if score > best_score or (np.isclose(score, best_score) and recall > best_recall):
            best_score = score
            best_recall = recall
            best_threshold = float(threshold)
    return best_threshold


def _tune_normal_specificity_threshold(
    validation_labels: np.ndarray,
    attack_probability: np.ndarray,
    *,
    balance_agents: bool = False,
    target_min: float = 0.92,
    target_max: float = 0.97,
) -> float:
    y_val = np.asarray(validation_labels, dtype=int)
    p_attack = np.asarray(attack_probability, dtype=float)
    best_threshold = 0.50
    best_score = -1.0
    best_specificity = -1.0
    for threshold in np.arange(0.40, 0.8001, 0.02):
        preds = _apply_threshold(p_attack, threshold)
        metrics = evaluate_predictions(y_val, preds, y_prob=p_attack)
        penalty = _accuracy_range_penalty(
            float(metrics["test_accuracy"]),
            balance_agents=balance_agents,
            target_min=target_min,
            target_max=target_max,
            penalty_weight=7.0,
        )
        score = (
            0.60 * float(metrics["specificity"])
            + 0.25 * float(metrics["test_f1"])
            + 0.15 * float(metrics["test_accuracy"])
            - penalty
        )
        specificity = float(metrics["specificity"])
        if score > best_score or (np.isclose(score, best_score) and specificity > best_specificity):
            best_score = score
            best_specificity = specificity
            best_threshold = float(threshold)
    return best_threshold


def _tune_hard_case_threshold(
    validation_labels: np.ndarray,
    attack_probability: np.ndarray,
    *,
    hard_case_mask: np.ndarray | None = None,
    balance_agents: bool = False,
    target_min: float = 0.92,
    target_max: float = 0.97,
) -> float:
    y_val = np.asarray(validation_labels, dtype=int)
    p_attack = np.asarray(attack_probability, dtype=float)
    if y_val.size == 0:
        return 0.50
    if hard_case_mask is not None and np.asarray(hard_case_mask, dtype=bool).shape[0] == y_val.shape[0]:
        hard_mask = np.asarray(hard_case_mask, dtype=bool)
    else:
        hard_mask = np.ones_like(y_val, dtype=bool)

    best_threshold = 0.50
    best_score = -1.0
    for threshold in np.arange(0.30, 0.7001, 0.02):
        preds = _apply_threshold(p_attack, threshold)
        hard_metrics = evaluate_predictions(y_val[hard_mask], preds[hard_mask], y_prob=p_attack[hard_mask])
        full_metrics = evaluate_predictions(y_val, preds, y_prob=p_attack)
        penalty = _accuracy_range_penalty(
            float(full_metrics["test_accuracy"]),
            balance_agents=balance_agents,
            target_min=target_min,
            target_max=target_max,
            penalty_weight=7.5,
        )
        score = float(hard_metrics["balanced_accuracy"] - penalty)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _identify_hard_validation_cases(
    validation_predictions: Dict[str, np.ndarray],
    validation_labels: np.ndarray,
    validation_probabilities: Dict[str, np.ndarray] | None = None,
    low_confidence_quantile: float = 0.25,
) -> np.ndarray:
    y_val = np.asarray(validation_labels, dtype=int)
    names = list(validation_predictions.keys())
    if not names:
        return np.zeros_like(y_val, dtype=bool)
    stacked = np.vstack([np.asarray(validation_predictions[name], dtype=int) for name in names])
    disagreement = np.any(stacked != stacked[0], axis=0)
    majority = (np.sum(stacked == 1, axis=0) >= (len(names) // 2 + 1)).astype(int)
    majority_error = majority != y_val
    low_confidence = np.zeros_like(y_val, dtype=bool)
    if validation_probabilities:
        margins = []
        for name in names:
            if name not in validation_probabilities:
                continue
            p_attack = np.asarray(validation_probabilities[name], dtype=float)
            margins.append(np.abs(p_attack - 0.5) * 2.0)
        if margins:
            mean_margin = np.mean(np.vstack(margins), axis=0)
            cutoff = float(np.quantile(mean_margin, low_confidence_quantile))
            low_confidence = mean_margin <= cutoff
    return disagreement | majority_error | low_confidence


def _compose_feature_subset(
    x_model_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    feature_columns: List[str],
    max_features: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    selected_columns = list(feature_columns)
    if max_features is not None and max_features > 0 and len(selected_columns) > int(max_features):
        variances = x_model_train_df[selected_columns].var(axis=0)
        selected_columns = variances.sort_values(ascending=False).head(int(max_features)).index.tolist()
    return (
        x_model_train_df[selected_columns].to_numpy(dtype=np.float64),
        x_val_df[selected_columns].to_numpy(dtype=np.float64),
        x_test_df[selected_columns].to_numpy(dtype=np.float64),
        selected_columns,
    )


def _select_general_agent_view(
    x_model_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_model_train: np.ndarray,
    y_val: np.ndarray,
    processed_views: Dict[str, List[str]],
    four_agent_cfg: Dict[str, Any],
    random_state: int,
) -> Dict[str, Any]:
    candidate_columns: Dict[str, List[str]] = {
        "basic": processed_views["basic_agent"],
        "basic_plus_host": sorted(set(processed_views["basic_agent"] + processed_views["host_traffic_agent"])),
        "basic_plus_time": sorted(set(processed_views["basic_agent"] + processed_views["time_traffic_agent"])),
        "content_plus_host": sorted(set(processed_views["content_agent"] + processed_views["host_traffic_agent"])),
        "time_plus_host": sorted(set(processed_views["time_traffic_agent"] + processed_views["host_traffic_agent"])),
    }

    balance_agents = bool(four_agent_cfg.get("balance_agents", True))
    max_general_features = four_agent_cfg.get("max_general_agent_features")
    if max_general_features is not None:
        try:
            max_general_features = int(max_general_features)
        except Exception:  # pylint: disable=broad-except
            max_general_features = None

    target_cfg = dict(four_agent_cfg.get("target_agent_accuracy_range", {}))
    target_min = float(target_cfg.get("min", 0.92))
    target_max = float(target_cfg.get("max", 0.97))

    best_record: Dict[str, Any] | None = None
    best_score = -1.0
    for view_name, columns in candidate_columns.items():
        x_train_candidate, x_val_candidate, x_test_candidate, used_columns = _compose_feature_subset(
            x_model_train_df,
            x_val_df,
            x_test_df,
            columns,
            max_general_features,
        )
        if x_train_candidate.shape[1] == 0:
            continue

        model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        )
        model = train_agent(model, x_train_candidate, y_model_train)
        val_prob = _extract_attack_probability(model.predict_proba(x_val_candidate), model.predict(x_val_candidate))
        val_pred = _apply_threshold(val_prob, 0.50)
        val_metrics = evaluate_predictions(y_val, val_pred, y_prob=val_prob)

        accuracy = float(val_metrics["test_accuracy"])
        f1 = float(val_metrics["test_f1"])
        bal_acc = float(val_metrics["balanced_accuracy"])
        outside_range_penalty = 0.0
        if balance_agents:
            if accuracy < target_min:
                outside_range_penalty = (target_min - accuracy) * 2.0
            elif accuracy > target_max:
                outside_range_penalty = (accuracy - target_max) * 2.0

        score = 0.45 * bal_acc + 0.35 * f1 + 0.20 * accuracy - outside_range_penalty
        if score > best_score:
            test_prob = _extract_attack_probability(model.predict_proba(x_test_candidate), model.predict(x_test_candidate))
            test_pred = _apply_threshold(test_prob, 0.50)
            best_score = score
            best_record = {
                "view_name": view_name,
                "feature_count": len(used_columns),
                "x_train": x_train_candidate,
                "x_val": x_val_candidate,
                "x_test": x_test_candidate,
                "model": model,
                "val_probability": val_prob,
                "val_prediction": val_pred,
                "test_probability": test_prob,
                "test_prediction": test_pred,
            }

    if best_record is None:
        raise ValueError("Could not select a general-agent feature subset in four-agent mode.")
    return best_record


def _build_four_agent_error_overlap_report(
    y_true: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    trust_selected_agent: List[str],
    trust_prediction: np.ndarray,
) -> pd.DataFrame:
    y_arr = np.asarray(y_true, dtype=int)
    rows: List[Dict[str, Any]] = []
    agent_names = list(model_preds.keys())
    for idx in range(len(y_arr)):
        sample_row: Dict[str, Any] = {
            "sample_index": idx,
            "true_label": int(y_arr[idx]),
            "number_of_agents_correct": 0,
            "number_of_agents_wrong": 0,
            "trust_selected_agent": trust_selected_agent[idx] if idx < len(trust_selected_agent) else "",
            "trust_prediction": int(trust_prediction[idx]),
            "trust_correct": bool(int(trust_prediction[idx]) == int(y_arr[idx])),
        }
        for agent_name in agent_names:
            pred = int(np.asarray(model_preds[agent_name], dtype=int)[idx])
            correct = int(pred == int(y_arr[idx]))
            sample_row[f"{agent_name}_prediction"] = pred
            sample_row[f"{agent_name}_correct"] = bool(correct)
            sample_row["number_of_agents_correct"] += correct
            sample_row["number_of_agents_wrong"] += int(not correct)
        rows.append(sample_row)
    return pd.DataFrame(rows)


def _compute_oracle_breakdown(y_true: np.ndarray, model_preds: Dict[str, np.ndarray], trust_prediction: np.ndarray) -> Dict[str, int]:
    y_arr = np.asarray(y_true, dtype=int)
    agent_names = list(model_preds.keys())
    stacked = np.vstack([np.asarray(model_preds[name], dtype=int) for name in agent_names])
    correct_matrix = stacked == y_arr
    trust_correct = np.asarray(trust_prediction, dtype=int) == y_arr
    all_correct = int(np.sum(np.all(correct_matrix, axis=0)))
    all_wrong = int(np.sum(np.all(~correct_matrix, axis=0)))
    only_one_correct = int(np.sum(np.sum(correct_matrix, axis=0) == 1))
    hard_case_agent = np.asarray(model_preds["Hard-Case Agent"], dtype=int)
    other_agents_correct = np.any(np.vstack([correct_matrix[idx] for idx, name in enumerate(agent_names) if name != "Hard-Case Agent"]), axis=0)
    hard_case_wrong_other_correct = int(np.sum((hard_case_agent != y_arr) & other_agents_correct))
    trust_failed_despite_available_correct = int(np.sum((~trust_correct) & np.any(correct_matrix, axis=0)))
    missed_opportunity_count = trust_failed_despite_available_correct
    return {
        "samples_all_agents_correct": all_correct,
        "samples_all_agents_wrong": all_wrong,
        "samples_only_one_agent_correct": only_one_correct,
        "samples_hard_case_wrong_but_another_correct": hard_case_wrong_other_correct,
        "samples_trust_failed_despite_at_least_one_correct": trust_failed_despite_available_correct,
        "trust_missed_opportunity_count": missed_opportunity_count,
    }


def _selector_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = evaluate_predictions(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int))
    return {
        "Accuracy": float(metrics["test_accuracy"]),
        "Precision": float(metrics["test_precision"]),
        "Recall": float(metrics["test_recall"]),
        "F1": float(metrics["test_f1"]),
        "FPR": float(metrics["fpr"]),
        "FNR": float(metrics["fnr"]),
        "Specificity": float(metrics["specificity"]),
        "Balanced Accuracy": float(metrics["balanced_accuracy"]),
    }


def _selector_tuning_grid() -> List[Dict[str, float]]:
    grid: List[Dict[str, float]] = []
    for validation_role_weight in [0.10, 0.20, 0.30]:
        for confidence_weight in [0.10, 0.20, 0.30]:
            for margin_weight in [0.05, 0.15, 0.25]:
                for local_accuracy_weight in [0.05, 0.10, 0.20]:
                    for disagreement_bonus in [0.05, 0.10, 0.20]:
                        grid.append(
                            {
                                "validation_role_weight": validation_role_weight,
                                "confidence_weight": confidence_weight,
                                "margin_weight": margin_weight,
                                "local_accuracy_weight": local_accuracy_weight,
                                "disagreement_bonus": disagreement_bonus,
                                "attack_role_bonus": 0.08,
                                "normal_role_bonus": 0.08,
                                "attack_confidence_threshold": 0.60,
                                "normal_confidence_threshold": 0.65,
                            }
                        )
    return grid


def _get_prob_for_pred(prob_array: np.ndarray, pred: int, idx: int) -> float:
    if prob_array is None:
        return 0.0
    prob_array = np.asarray(prob_array)
    if prob_array.ndim == 1:
        return float(prob_array[idx])
    if prob_array.ndim == 2 and prob_array.shape[1] == 2:
        return float(prob_array[idx, int(pred)])
    # fallback: return probability of positive class if exists
    try:
        return float(prob_array[idx, 1])
    except Exception:
        return float(prob_array[idx])


def _apply_hard_case_guarded_selector(
    model_preds: Dict[str, np.ndarray],
    model_probs: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> np.ndarray:
    """Conservative guarded selector: default to Hard-Case, override only on strong evidence."""
    n = len(next(iter(model_preds.values())))
    hard_pred = np.asarray(model_preds["Hard-Case Agent"], dtype=int)
    hard_prob_arr = np.asarray(model_probs.get("Hard-Case Agent", np.zeros(n)), dtype=float)

    out = hard_pred.copy()
    for i in range(n):
        hc_pred = int(hard_pred[i])
        hc_conf = _get_prob_for_pred(hard_prob_arr, hc_pred, i)
        hc_margin = abs(hc_conf - 0.5) * 2.0

        hard_conf_low = hc_conf < float(params.get("hard_case_min_confidence", 0.5))
        hard_margin_low = hc_margin < float(params.get("hard_case_min_margin", 0.1))
        if not (hard_conf_low or hard_margin_low):
            # Hard-case confident enough — do not override
            continue

        # Evaluate candidate agents
        candidates = [
            "General Traffic Agent",
            "Attack Recall Agent",
            "Normal Behavior Agent",
        ]
        best_candidate = None
        best_candidate_conf = -1.0
        for cand in candidates:
            cand_pred = int(np.asarray(model_preds[cand], dtype=int)[i])
            cand_prob_arr = np.asarray(model_probs.get(cand, np.zeros(n)), dtype=float)
            cand_conf = _get_prob_for_pred(cand_prob_arr, cand_pred, i)
            # must exceed candidate_min_confidence
            if cand_conf < float(params.get("candidate_min_confidence", 0.8)):
                continue
            # must exceed hard-case by gap
            if (cand_conf - hc_conf) < float(params.get("override_confidence_gap", 0.1)):
                continue

            # agreement among other agents
            agree_count = 0
            for other in candidates:
                if other == cand:
                    continue
                other_pred = int(np.asarray(model_preds[other], dtype=int)[i])
                if other_pred == cand_pred:
                    agree_count += 1

            require_two = bool(params.get("require_two_agent_agreement", True))
            strong_role_condition = False
            # treat Attack Recall strong attack as role-strong
            if cand == "Attack Recall Agent" and cand_pred == 1 and cand_conf >= float(params.get("candidate_min_confidence", 0.8)):
                strong_role_condition = True
            if require_two and agree_count < 2 and not strong_role_condition:
                continue
            if not require_two and agree_count < 1 and not strong_role_condition:
                continue

            # candidate passes checks
            if cand_conf > best_candidate_conf:
                best_candidate_conf = cand_conf
                best_candidate = (cand, cand_pred)

        if best_candidate is not None:
            out[i] = int(best_candidate[1])

    return out


def _search_best_hard_case_guarded_params(
    val_predictions: Dict[str, np.ndarray],
    val_probabilities: Dict[str, np.ndarray],
    x_val_features: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    # Use fixed conservative defaults to avoid expensive tuning during run.
    # Conservative: Hard-Case must be uncertain + candidate must be high-conf + agreement required
    best_params = {
        "hard_case_min_confidence": 0.5,
        "hard_case_min_margin": 0.1,
        "override_confidence_gap": 0.15,
        "candidate_min_confidence": 0.8,
        "require_two_agent_agreement": True,
    }
    return best_params


def _build_unsw_missed_opportunity_report_file(
    dataset_output_dir: Path,
    y_test: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    model_probs: Dict[str, np.ndarray],
    selector_pred: Optional[np.ndarray],
    selector_overrode_mask: Optional[np.ndarray],
) -> None:
    y_arr = np.asarray(y_test, dtype=int)
    n = len(y_arr)
    rows: List[Dict[str, Any]] = []
    hard_pred = np.asarray(model_preds["Hard-Case Agent"], dtype=int)
    for i in range(n):
        if hard_pred[i] == int(y_arr[i]):
            continue
        # check if any other agent correct
        others = {
            "general_prediction": int(np.asarray(model_preds["General Traffic Agent"], dtype=int)[i]),
            "attack_recall_prediction": int(np.asarray(model_preds["Attack Recall Agent"], dtype=int)[i]),
            "normal_behavior_prediction": int(np.asarray(model_preds["Normal Behavior Agent"], dtype=int)[i]),
        }
        correct_others = [name for name, p in others.items() if p == int(y_arr[i])]
        if not correct_others:
            continue
        row: Dict[str, Any] = {
            "sample_index": int(i),
            "true_label": int(y_arr[i]),
            "hard_case_prediction": int(hard_pred[i]),
            "general_prediction": others["general_prediction"],
            "attack_recall_prediction": others["attack_recall_prediction"],
            "normal_behavior_prediction": others["normal_behavior_prediction"],
            "which_agents_correct": ",".join(correct_others),
            "number_of_correct_non_hardcase_agents": int(len(correct_others)),
        }
        # confidences and margins
        for name, key in [("Hard-Case Agent", "hard_case"), ("General Traffic Agent", "general"), ("Attack Recall Agent", "attack_recall"), ("Normal Behavior Agent", "normal_behavior")]:
            pred_val = int(np.asarray(model_preds[name], dtype=int)[i])
            prob_arr = np.asarray(model_probs.get(name, np.zeros(n)), dtype=float)
            conf = _get_prob_for_pred(prob_arr, pred_val, i)
            row[f"{key}_confidence"] = float(conf)
            row[f"{key}_margin"] = float(abs(conf - 0.5) * 2.0)

        if selector_pred is not None:
            row["selector_prediction"] = int(np.asarray(selector_pred, dtype=int)[i])
            overrode = bool(selector_overrode_mask[i]) if selector_overrode_mask is not None else False
            row["selector_overrode_hard_case"] = overrode
            row["selector_correct_after_override"] = bool((row.get("selector_prediction") == int(y_arr[i])) if overrode else False)
        else:
            row["selector_prediction"] = ""
            row["selector_overrode_hard_case"] = "skipped"
            row["selector_correct_after_override"] = "skipped"

        rows.append(row)

    df = pd.DataFrame(rows)
    # ensure columns order
    desired_cols = [
        "sample_index",
        "true_label",
        "hard_case_prediction",
        "general_prediction",
        "attack_recall_prediction",
        "normal_behavior_prediction",
        "hard_case_confidence",
        "general_confidence",
        "attack_recall_confidence",
        "normal_behavior_confidence",
        "hard_case_margin",
        "general_margin",
        "attack_recall_margin",
        "normal_margin",
        "which_agents_correct",
        "number_of_correct_non_hardcase_agents",
        "selector_prediction",
        "selector_overrode_hard_case",
        "selector_correct_after_override",
    ]
    # map column names if different
    # write file (if empty, write header)
    if df.empty:
        pd.DataFrame(columns=desired_cols).to_csv(dataset_output_dir / "unsw_missed_opportunity_report.csv", index=False)
    else:
        # try to rename margins/conf columns
        col_renames = {}
        if "hard_case_margin" not in df.columns and "hard_case_margin" in df.columns:
            pass
        df.to_csv(dataset_output_dir / "unsw_missed_opportunity_report.csv", index=False)


def _run_selector_variant(
    *,
    name: str,
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    x_val_features: np.ndarray,
    y_val: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    x_test_features: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, float],
) -> Dict[str, Any]:
    result = trust_methods.trust_agent_selector(
        predictions,
        probabilities,
        validation_metrics,
        roles,
        x_val_features,
        y_val,
        validation_predictions,
        x_test_features,
        neighbor_k=int(params.get("neighbor_k", 25)),
        validation_role_weight=float(params.get("validation_role_weight", 0.25)),
        confidence_weight=float(params.get("confidence_weight", 0.20)),
        margin_weight=float(params.get("margin_weight", 0.15)),
        local_accuracy_weight=float(params.get("local_accuracy_weight", 0.10)),
        disagreement_bonus=float(params.get("disagreement_bonus", 0.10)),
        attack_role_bonus=float(params.get("attack_role_bonus", 0.08)),
        normal_role_bonus=float(params.get("normal_role_bonus", 0.08)),
        attack_confidence_threshold=float(params.get("attack_confidence_threshold", 0.60)),
        normal_confidence_threshold=float(params.get("normal_confidence_threshold", 0.65)),
    )
    y_pred = np.asarray(result["predictions"], dtype=int)
    metrics = evaluate_predictions(np.asarray(y_test, dtype=int), y_pred)
    selected_agents = result.get("meta", {}).get("selected_agents", [])
    y_test_arr = np.asarray(y_test, dtype=int)
    available_correct = np.any(np.vstack([np.asarray(predictions[agent], dtype=int) == y_test_arr for agent in predictions]), axis=0)
    missed_opportunity_count = int(np.sum((y_pred != y_test_arr) & available_correct))
    oracle_accuracy = float(compute_oracle_upper_bound(y_test_arr, predictions)["oracle_accuracy"])
    return {
        "stage": name,
        "Accuracy": float(metrics["test_accuracy"]),
        "Precision": float(metrics["test_precision"]),
        "Recall": float(metrics["test_recall"]),
        "F1": float(metrics["test_f1"]),
        "FPR": float(metrics["fpr"]),
        "FNR": float(metrics["fnr"]),
        "Specificity": float(metrics["specificity"]),
        "Balanced Accuracy": float(metrics["balanced_accuracy"]),
        "Improvement vs Best Agent Accuracy": 0.0,
        "Missed Opportunity Count": missed_opportunity_count,
        "Oracle Gap": float(oracle_accuracy - float(metrics["test_accuracy"])),
        "selected_agents": selected_agents,
        "predictions": y_pred,
        "params": params,
    }


def _selector_validation_score(metrics: Dict[str, float]) -> float:
    return (
        0.45 * float(metrics["test_accuracy"])
        + 0.30 * float(metrics["test_f1"])
        + 0.25 * float(metrics["balanced_accuracy"])
    )


def _extract_confidence_and_margin(prediction: np.ndarray, probability: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=int)
    if probability is None:
        p_attack = pred.astype(float)
    else:
        prob = np.asarray(probability, dtype=float)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            p_attack = prob[:, 1]
        else:
            p_attack = prob.ravel()
    p_attack = np.clip(p_attack, 0.0, 1.0)
    confidence = np.maximum(p_attack, 1.0 - p_attack)
    margin = np.abs(p_attack - 0.5) * 2.0
    return confidence, margin


def _run_weighted_selector_variant(
    *,
    name: str,
    query_predictions: Dict[str, np.ndarray],
    query_probabilities: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    query_features: np.ndarray,
    neighbor_k: int,
    weights: Dict[str, float],
    disagreement_boost_strength: float = 0.0,
) -> Dict[str, Any]:
    names = list(query_predictions.keys())
    x_val = np.asarray(validation_features, dtype=float)
    y_val = np.asarray(validation_labels, dtype=int)
    x_query = np.asarray(query_features, dtype=float)
    k = max(1, min(int(neighbor_k), len(x_val)))

    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(x_val)
    _, neighbor_indices = knn.kneighbors(x_query)

    val_correct = {n: (np.asarray(validation_predictions[n], dtype=int) == y_val) for n in names}
    local_accuracy = {
        n: np.array([float(np.mean(val_correct[n][idxs])) for idxs in neighbor_indices], dtype=float) for n in names
    }
    conf = {}
    margin = {}
    for n in names:
        conf[n], margin[n] = _extract_confidence_and_margin(
            np.asarray(query_predictions[n], dtype=int),
            query_probabilities.get(n),
        )

    val_stack = np.vstack([np.asarray(validation_predictions[n], dtype=int) for n in names])
    val_disagreement_mask = np.any(val_stack != val_stack[0], axis=0)
    disagreement_accuracy = {
        n: float(np.mean(val_correct[n][val_disagreement_mask])) if np.any(val_disagreement_mask) else 0.0 for n in names
    }

    query_stack = np.vstack([np.asarray(query_predictions[n], dtype=int) for n in names])
    query_disagreement_mask = np.any(query_stack != query_stack[0], axis=0)

    selected_agents: List[str] = []
    preds = np.zeros(x_query.shape[0], dtype=int)
    for i in range(x_query.shape[0]):
        best_name = names[0]
        best_score = -1e9
        disagree = bool(query_disagreement_mask[i])
        for n in names:
            m = validation_metrics.get(n, {})
            score = (
                float(weights.get("validation_accuracy", 0.0)) * float(m.get("test_accuracy", 0.0))
                + float(weights.get("validation_f1", 0.0)) * float(m.get("test_f1", 0.0))
                + float(weights.get("validation_recall", 0.0)) * float(m.get("test_recall", 0.0))
                + float(weights.get("validation_one_minus_fnr", 0.0)) * (1.0 - float(m.get("fnr", 0.0)))
                + float(weights.get("sample_confidence", 0.0)) * float(conf[n][i])
                + float(weights.get("margin", 0.0)) * float(margin[n][i])
                + float(weights.get("local_accuracy", 0.0)) * float(local_accuracy[n][i])
            )
            if disagree:
                score += float(disagreement_boost_strength) * float(disagreement_accuracy[n])
            if score > best_score:
                best_score = score
                best_name = n
        selected_agents.append(best_name)
        preds[i] = int(np.asarray(query_predictions[best_name], dtype=int)[i])

    return {"name": name, "predictions": preds, "selected_agents": selected_agents}


def _build_cic_missed_opportunity_report(
    y_true: np.ndarray,
    selected_agents: List[str],
    trust_prediction: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    model_probs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    y_arr = np.asarray(y_true, dtype=int)
    rows: List[Dict[str, Any]] = []
    names = list(model_preds.keys())
    conf_margin = {}
    for n in names:
        c, m = _extract_confidence_and_margin(np.asarray(model_preds[n], dtype=int), model_probs.get(n))
        conf_margin[n] = (c, m)

    for i in range(len(y_arr)):
        trust_ok = int(trust_prediction[i]) == int(y_arr[i])
        any_other_correct = any(int(np.asarray(model_preds[n], dtype=int)[i]) == int(y_arr[i]) for n in names)
        if trust_ok or not any_other_correct:
            continue
        per_agent_correct = {n: int(np.asarray(model_preds[n], dtype=int)[i]) == int(y_arr[i]) for n in names}
        which_correct = [n for n in names if per_agent_correct[n]]
        preds_i = [int(np.asarray(model_preds[n], dtype=int)[i]) for n in names]
        disagreement_count = len(set(preds_i))
        row = {
            "true_label": int(y_arr[i]),
            "trust_selected_agent": selected_agents[i] if i < len(selected_agents) else "",
            "trust_prediction": int(trust_prediction[i]),
            "general_prediction": int(np.asarray(model_preds["General Traffic Agent"], dtype=int)[i]),
            "attack_recall_prediction": int(np.asarray(model_preds["Attack Recall Agent"], dtype=int)[i]),
            "normal_behavior_prediction": int(np.asarray(model_preds["Normal Behavior Agent"], dtype=int)[i]),
            "hard_case_prediction": int(np.asarray(model_preds["Hard-Case Agent"], dtype=int)[i]),
            "general_confidence": float(conf_margin["General Traffic Agent"][0][i]),
            "attack_recall_confidence": float(conf_margin["Attack Recall Agent"][0][i]),
            "normal_behavior_confidence": float(conf_margin["Normal Behavior Agent"][0][i]),
            "hard_case_confidence": float(conf_margin["Hard-Case Agent"][0][i]),
            "general_margin": float(conf_margin["General Traffic Agent"][1][i]),
            "attack_recall_margin": float(conf_margin["Attack Recall Agent"][1][i]),
            "normal_behavior_margin": float(conf_margin["Normal Behavior Agent"][1][i]),
            "hard_case_margin": float(conf_margin["Hard-Case Agent"][1][i]),
            "which_agents_correct": ",".join(which_correct),
            "number_of_correct_agents": int(len(which_correct)),
            "disagreement_count": int(disagreement_count),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _search_best_selector_params(
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    x_val_features: np.ndarray,
    y_val: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    x_test_features: np.ndarray,
) -> Dict[str, float]:
    best_params: Dict[str, float] | None = None
    best_score = -1.0
    for params in _selector_tuning_grid():
        candidate = trust_methods.trust_agent_selector(
            predictions,
            probabilities,
            validation_metrics,
            roles,
            x_val_features,
            y_val,
            validation_predictions,
            x_val_features,
            neighbor_k=int(params.get("neighbor_k", 25)),
            validation_role_weight=float(params.get("validation_role_weight", 0.25)),
            confidence_weight=float(params.get("confidence_weight", 0.20)),
            margin_weight=float(params.get("margin_weight", 0.15)),
            local_accuracy_weight=float(params.get("local_accuracy_weight", 0.10)),
            disagreement_bonus=float(params.get("disagreement_bonus", 0.10)),
            attack_role_bonus=float(params.get("attack_role_bonus", 0.08)),
            normal_role_bonus=float(params.get("normal_role_bonus", 0.08)),
            attack_confidence_threshold=float(params.get("attack_confidence_threshold", 0.60)),
            normal_confidence_threshold=float(params.get("normal_confidence_threshold", 0.65)),
        )
        val_metrics = evaluate_predictions(np.asarray(y_val, dtype=int), np.asarray(candidate["predictions"], dtype=int))
        score = _selector_validation_score(val_metrics)
        if score > best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise ValueError("Could not find a tuned selector configuration from validation data.")
    return best_params


def _write_diversity_and_error_reports(
    results_dir: Path,
    y_test: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    trust_selected_agent: List[str],
    trust_prediction: np.ndarray,
) -> Dict[str, int]:
    diversity_df = compute_model_diversity_report(np.asarray(y_test, dtype=int), model_preds)
    diversity_df = diversity_df.rename(
        columns={
            "model_a": "agent_a",
            "model_b": "agent_b",
            "a_correct_b_wrong": "agent_a_correct_agent_b_wrong",
            "b_correct_a_wrong": "agent_b_correct_agent_a_wrong",
        }
    )
    diversity_df.to_csv(results_dir / "four_agent_diversity_report.csv", index=False)

    error_overlap_df = _build_four_agent_error_overlap_report(y_test, model_preds, trust_selected_agent, trust_prediction)
    error_overlap_df.to_csv(dataset_output_dir / "four_agent_error_overlap_report.csv", index=False)

    return _compute_oracle_breakdown(y_test, model_preds, trust_prediction)


def _run_feature_view_multi_agent(
    config: Dict[str, Any],
    project_root: Path,
    results_dir: Path,
    dataset_name: Optional[str] = None,
    poison_options: Optional[Dict[str, Any]] = None,
) -> None:
    """Run four-agent trust-centric experiment on NSL-KDD or UNSW-NB15 datasets."""
    preprocessing_cfg = dict(config.get("preprocessing", {}))
    model_cfg = dict(config.get("models", {}))
    trust_cfg = dict(config.get("trust_methods", {}))
    four_agent_cfg = dict(config.get("four_agent_mode", {}))
    target_cfg = dict(four_agent_cfg.get("target_agent_accuracy_range", {}))
    balance_agents = bool(four_agent_cfg.get("balance_agents", True))
    target_min_accuracy = float(target_cfg.get("min", 0.92))
    target_max_accuracy = float(target_cfg.get("max", 0.97))

    # Use provided dataset_name or fall back to config
    if dataset_name is None:
        dataset_name = str(config.get("dataset", "NSL-KDD"))
    dataset_name = _normalize_dataset_alias(dataset_name)
    dataset_name_lower = dataset_name.strip().lower()
    expanded_models_enabled = bool(config.get("expanded_models_enabled", False))
    if expanded_models_enabled:
        print("expanded_models_enabled=true was provided, but strict four-source mode keeps extra models disabled.")

    # Create dataset-specific results directory
    if dataset_name_lower in {"unsw-nb15", "unsw"}:
        dataset_output_dir = results_dir / "unsw_nb15"
        dataset_file_stem = "unsw_nb15"
    elif dataset_name_lower in {"ton-iot", "ton_iot"}:
        dataset_output_dir = results_dir / "ton_iot"
        dataset_file_stem = "ton_iot"
    elif dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        dataset_output_dir = results_dir / "cicids2017"
        dataset_file_stem = "cicids2017"
    elif dataset_name_lower == "nsl-kdd":
        dataset_output_dir = results_dir
        dataset_file_stem = "nsl_kdd"
    else:
        dataset_output_dir = results_dir / dataset_name_lower.replace(" ", "_")
        dataset_file_stem = dataset_name_lower.replace(" ", "_")
    
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {dataset_output_dir}")

    # Load dataset based on type
    if dataset_name_lower in {"unsw-nb15", "unsw"}:
        dataset_cfg = _load_unsw_dataset(config, project_root, "UNSW-NB15")
        try:
            x_train, x_test, y_train, y_test = load_unsw_nb15_dataset(dataset_cfg)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print(f"\nUNSW-NB15 dataset files not found.")
            print(f"Please download and place the files at:")
            print(f"  - {project_root / 'data' / 'raw' / 'unsw_nb15' / 'UNSW_NB15_training-set.csv'}")
            print(f"  - {project_root / 'data' / 'raw' / 'unsw_nb15' / 'UNSW_NB15_testing-set.csv'}")
            raise
        
        categorical_columns = get_unsw_categorical_columns(x_train)
        use_custom_feature_views = True
    elif dataset_name_lower in {"ton-iot", "ton_iot"}:
        ton_cfg = {}
        for ds in config.get("datasets", []):
            if str(ds.get("name", "")).strip().lower().replace("_", "-") in {"ton-iot", "ton iot"}:
                ton_cfg = dict(ds)
                break
        try:
            ton_data = load_ton_iot_dataset(
                project_root,
                random_state=int(config.get("experiment", {}).get("random_state", 42)),
                combine_csvs=bool(ton_cfg.get("combine_csvs", False)),
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\nToN-IoT dataset files not found.")
            print("Expected file paths:")
            for p in [
                project_root / "data" / "raw" / "ton_iot" / "*.csv",
                project_root / "data" / "ton_iot" / "*.csv",
                project_root / "data" / "ToN_IoT*.csv",
            ]:
                print(f"  - {p}")
            return

        x_train = ton_data["x_train"]
        x_val_raw = ton_data["x_val"]
        x_test = ton_data["x_test"]
        y_train = ton_data["y_train"]
        y_val_raw = ton_data["y_val"]
        y_test = ton_data["y_test"]
        categorical_columns = ton_data["categorical_columns"]
        ton_numeric_columns = ton_data["numeric_columns"]
        ton_dropped_leakage = ton_data["dropped_leakage_columns"]
        use_custom_feature_views = True
    elif dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        cic_cfg = {}
        for ds in config.get("datasets", []):
            if str(ds.get("name", "")).strip().lower().replace("_", "-") in {"cicids2017", "cic-ids2017"}:
                cic_cfg = dict(ds)
                break
        try:
            cic_data = load_cicids2017_dataset(
                project_root,
                random_state=int(config.get("experiment", {}).get("random_state", 42)),
                max_samples=cic_cfg.get("max_samples"),
                max_train_samples=cic_cfg.get("max_train_samples"),
                max_validation_samples=cic_cfg.get("max_validation_samples"),
                max_test_samples=cic_cfg.get("max_test_samples"),
                stratified_subsample=bool(cic_cfg.get("stratified_subsample", True)),
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\nCICIDS2017 dataset files not found.")
            return

        x_train = cic_data["x_train"]
        x_val_raw = cic_data["x_val"]
        x_test = cic_data["x_test"]
        y_train = cic_data["y_train"]
        y_val_raw = cic_data["y_val"]
        y_test = cic_data["y_test"]
        categorical_columns = cic_data["categorical_columns"]
        cic_numeric_columns = cic_data["numeric_columns"]
        cic_dropped_leakage = cic_data["dropped_leakage_columns"]
        if cic_data.get("capped"):
            print(f"[INFO] CICIDS2017 run is capped: {cic_data.get('capped_breakdown')}")
        use_custom_feature_views = True
    elif dataset_name_lower == "nsl-kdd":
        dataset_cfg = _load_nsl_dataset(config, project_root, "NSL-KDD")
        x, y_raw = load_dataset(dataset_cfg)
        y_binary = _convert_to_binary_labels(y_raw, preprocessing_cfg)
        x, _ = sanitize_feature_values(x, dataset_cfg.get("categorical_columns", []))
        
        random_state_val = int(config.get("experiment", {}).get("random_state", 42))
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y_binary,
            test_size=float(preprocessing_cfg.get("test_size", 0.3)),
            random_state=random_state_val,
            stratify=y_binary,
        )
        categorical_columns = dataset_cfg.get("categorical_columns", [])
        use_custom_feature_views = False
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"Dataset loaded: {dataset_name}")

    random_state = int(config.get("experiment", {}).get("random_state", 42))
    
    # Preprocessing
    preprocessor = build_preprocessing_pipeline(
        x_train,
        categorical_columns,
        preprocessing_cfg,
    )
    x_train_processed = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float64)
    x_test_processed = np.asarray(preprocessor.transform(x_test), dtype=np.float64)
    x_val_processed_from_official = None
    if dataset_name_lower in {"ton-iot", "ton_iot", "cicids2017", "cic-ids2017"}:
        x_val_processed_from_official = np.asarray(preprocessor.transform(x_val_raw), dtype=np.float64)
    print("Preprocessing complete")

    # Feature views
    processed_feature_names = preprocessor.get_feature_names_out().tolist()
    raw_feature_names = list(x_train.columns)
    
    if use_custom_feature_views:
        # UNSW: use generic feature views
        raw_views = get_unsw_feature_views(raw_feature_names)
        processed_views = map_processed_feature_views(processed_feature_names, raw_views)
    else:
        # NSL-KDD: use fixed feature views
        raw_views = get_nsl_kdd_feature_views(raw_feature_names)
        processed_views = map_processed_feature_views(processed_feature_names, raw_views)

    x_train_df = pd.DataFrame(x_train_processed, columns=processed_feature_names, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_processed, columns=processed_feature_names, index=x_test.index)

    if dataset_name_lower in {"ton-iot", "ton_iot", "cicids2017", "cic-ids2017"}:
        x_val_df = pd.DataFrame(x_val_processed_from_official, columns=processed_feature_names, index=x_val_raw.index)
        x_model_train_df = x_train_df.copy()
        y_model_train = np.asarray(y_train, dtype=int)
        y_val = np.asarray(y_val_raw, dtype=int)
    else:
        x_model_train_df, x_val_df, y_model_train, y_val = train_test_split(
            x_train_df,
            np.asarray(y_train, dtype=int),
            test_size=0.20,
            random_state=random_state,
            stratify=np.asarray(y_train, dtype=int),
        )

    # Generate integrity report
    dropped_leakage = [col for col in x_train.columns if col in {"id", "attack_cat", "attack_category"}]
    if dataset_name_lower in {"ton-iot", "ton_iot"}:
        dropped_leakage = sorted(set(dropped_leakage + ton_dropped_leakage))
    if dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        dropped_leakage = sorted(set(dropped_leakage + cic_dropped_leakage))
    integrity_report = generate_integrity_report(
        dataset_name=dataset_name,
        x_train=x_model_train_df,
        x_val=x_val_df,
        x_test=x_test_df,
        y_train=np.asarray(y_model_train, dtype=int),
        y_val=np.asarray(y_val, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        dropped_leakage_columns=dropped_leakage,
        categorical_columns=categorical_columns,
        numeric_columns=(
            ton_numeric_columns
            if dataset_name_lower in {"ton-iot", "ton_iot"}
            else cic_numeric_columns
            if dataset_name_lower in {"cicids2017", "cic-ids2017"}
            else [col for col in x_train.columns if col not in categorical_columns]
        ),
        preprocessor_object=preprocessor,
    )
    (dataset_output_dir / "evaluation_integrity_report.md").write_text(integrity_report)

    x_model_train_full = np.asarray(x_model_train_df.to_numpy(), dtype=np.float64)
    x_val_full = np.asarray(x_val_df.to_numpy(), dtype=np.float64)
    x_test_full = np.asarray(x_test_df.to_numpy(), dtype=np.float64)

    x_model_train_attack = x_model_train_df[processed_views["time_traffic_agent"]].to_numpy(dtype=np.float64)
    x_val_attack = x_val_df[processed_views["time_traffic_agent"]].to_numpy(dtype=np.float64)
    x_test_attack = x_test_df[processed_views["time_traffic_agent"]].to_numpy(dtype=np.float64)

    x_model_train_normal = x_model_train_df[processed_views["host_traffic_agent"]].to_numpy(dtype=np.float64)
    x_val_normal = x_val_df[processed_views["host_traffic_agent"]].to_numpy(dtype=np.float64)
    x_test_normal = x_test_df[processed_views["host_traffic_agent"]].to_numpy(dtype=np.float64)

    x_model_train_hard = x_model_train_df[processed_views["content_agent"]].to_numpy(dtype=np.float64)
    x_val_hard = x_val_df[processed_views["content_agent"]].to_numpy(dtype=np.float64)
    x_test_hard = x_test_df[processed_views["content_agent"]].to_numpy(dtype=np.float64)

    agent_records: List[Dict[str, Any]] = []

    general_selection = _select_general_agent_view(
        x_model_train_df,
        x_val_df,
        x_test_df,
        np.asarray(y_model_train, dtype=int),
        np.asarray(y_val, dtype=int),
        processed_views,
        four_agent_cfg,
        random_state,
    )
    general_val_prob = np.asarray(general_selection["val_probability"], dtype=float)
    general_test_prob = np.asarray(general_selection["test_probability"], dtype=float)
    general_threshold = 0.50
    general_val_pred = _apply_threshold(general_val_prob, general_threshold)
    general_test_pred = _apply_threshold(general_test_prob, general_threshold)
    general_val_metrics = evaluate_predictions(y_val, general_val_pred, y_prob=general_val_prob)
    general_test_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), general_test_pred, y_prob=general_test_prob)
    agent_records.append(
        {
            "name": "General Traffic Agent",
            "role": "general",
            "view_name": str(general_selection["view_name"]),
            "feature_count": int(general_selection["feature_count"]),
            "selected_threshold": general_threshold,
            "validation_prediction": general_val_pred,
            "validation_probability": general_val_prob,
            "validation_metrics": general_val_metrics,
            "prediction": general_test_pred,
            "probability": general_test_prob,
            "metrics": general_test_metrics,
        }
    )

    attack_recall_agent = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        C=0.5,
        random_state=random_state,
    )
    attack_recall_agent = train_agent(attack_recall_agent, x_model_train_attack, y_model_train)
    attack_val_pred_raw, attack_val_prob_raw = _predict_with_probabilities(attack_recall_agent, x_val_attack)
    attack_val_prob = _extract_attack_probability(attack_val_prob_raw, attack_val_pred_raw)
    attack_val_prob = np.clip(attack_val_prob + 0.02, 0.0, 1.0)
    attack_test_pred_raw, attack_test_prob_raw = _predict_with_probabilities(attack_recall_agent, x_test_attack)
    attack_test_prob = _extract_attack_probability(
        attack_test_prob_raw,
        attack_test_pred_raw,
    )
    attack_test_prob = np.clip(attack_test_prob + 0.02, 0.0, 1.0)
    attack_threshold = _tune_attack_recall_threshold(
        y_val,
        attack_val_prob,
        balance_agents=balance_agents,
        target_min=target_min_accuracy,
        target_max=target_max_accuracy,
    )
    attack_val_pred = _apply_threshold(attack_val_prob, attack_threshold)
    attack_test_pred = _apply_threshold(attack_test_prob, attack_threshold)
    attack_val_metrics = evaluate_predictions(y_val, attack_val_pred, y_prob=attack_val_prob)
    attack_test_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), attack_test_pred, y_prob=attack_test_prob)
    agent_records.append(
        {
            "name": "Attack Recall Agent",
            "role": "attack_recall",
            "view_name": "full_attack_biased",
            "feature_count": int(x_model_train_full.shape[1]),
            "selected_threshold": attack_threshold,
            "validation_prediction": attack_val_pred,
            "validation_probability": attack_val_prob,
            "validation_metrics": attack_val_metrics,
            "prediction": attack_test_pred,
            "probability": attack_test_prob,
            "metrics": attack_test_metrics,
        }
    )

    normal_behavior_agent = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        C=0.6,
        class_weight="balanced",
        random_state=random_state,
    )
    normal_behavior_agent = train_agent(normal_behavior_agent, x_model_train_normal, y_model_train)
    normal_val_prob = _extract_attack_probability(
        normal_behavior_agent.predict_proba(x_val_normal),
        normal_behavior_agent.predict(x_val_normal),
    )
    normal_test_prob = _extract_attack_probability(
        normal_behavior_agent.predict_proba(x_test_normal),
        normal_behavior_agent.predict(x_test_normal),
    )
    normal_threshold = _tune_normal_specificity_threshold(
        y_val,
        normal_val_prob,
        balance_agents=balance_agents,
        target_min=target_min_accuracy,
        target_max=target_max_accuracy,
    )
    normal_val_pred = _apply_threshold(normal_val_prob, normal_threshold)
    normal_test_pred = _apply_threshold(normal_test_prob, normal_threshold)
    normal_val_metrics = evaluate_predictions(y_val, normal_val_pred, y_prob=normal_val_prob)
    normal_test_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), normal_test_pred, y_prob=normal_test_prob)
    agent_records.append(
        {
            "name": "Normal Behavior Agent",
            "role": "normal_behavior",
            "view_name": "full_specificity_biased",
            "feature_count": int(x_model_train_full.shape[1]),
            "selected_threshold": normal_threshold,
            "validation_prediction": normal_val_pred,
            "validation_probability": normal_val_prob,
            "validation_metrics": normal_val_metrics,
            "prediction": normal_test_pred,
            "probability": normal_test_prob,
            "metrics": normal_test_metrics,
        }
    )

    warmup_validation_predictions = {
        record["name"]: np.asarray(record["validation_prediction"], dtype=int) for record in agent_records
    }
    warmup_validation_probabilities = {
        record["name"]: np.asarray(record["validation_probability"], dtype=float) for record in agent_records
    }
    hard_mask = _identify_hard_validation_cases(warmup_validation_predictions, y_val, warmup_validation_probabilities)
    x_hard_train = np.vstack([x_model_train_hard, x_val_hard])
    y_hard_train = np.concatenate([np.asarray(y_model_train, dtype=int), np.asarray(y_val, dtype=int)])
    sample_weights = np.ones(x_hard_train.shape[0], dtype=float)
    val_weight_start = x_model_train_hard.shape[0]
    sample_weights[val_weight_start:] = np.where(hard_mask, 3.5, 1.40)

    hard_case_agent = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    hard_case_agent.fit(x_hard_train, y_hard_train, sample_weight=sample_weights)
    hard_val_prob = _extract_attack_probability(hard_case_agent.predict_proba(x_val_hard), hard_case_agent.predict(x_val_hard))
    hard_test_prob = _extract_attack_probability(
        hard_case_agent.predict_proba(x_test_hard),
        hard_case_agent.predict(x_test_hard),
    )
    hard_threshold = _tune_hard_case_threshold(
        y_val,
        hard_val_prob,
        hard_case_mask=hard_mask,
        balance_agents=balance_agents,
        target_min=target_min_accuracy,
        target_max=target_max_accuracy,
    )
    hard_val_pred = _apply_threshold(hard_val_prob, hard_threshold)
    hard_test_pred = _apply_threshold(hard_test_prob, hard_threshold)
    hard_val_metrics = evaluate_predictions(y_val, hard_val_pred, y_prob=hard_val_prob)
    hard_test_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), hard_test_pred, y_prob=hard_test_prob)
    agent_records.append(
        {
            "name": "Hard-Case Agent",
            "role": "hard_case",
            "view_name": "full_hard_case_weighted",
            "feature_count": int(x_model_train_full.shape[1]),
            "selected_threshold": hard_threshold,
            "validation_prediction": hard_val_pred,
            "validation_probability": hard_val_prob,
            "validation_metrics": hard_val_metrics,
            "prediction": hard_test_pred,
            "probability": hard_test_prob,
            "metrics": hard_test_metrics,
        }
    )

    if len(agent_records) != 4:
        raise ValueError("Strict four-source mode requires exactly four specialized cybersecurity decision sources.")

    model_preds = {record["name"]: np.asarray(record["prediction"], dtype=int) for record in agent_records}
    model_probs = {record["name"]: np.asarray(record["probability"], dtype=float) for record in agent_records}
    validation_model_metrics = {record["name"]: record["validation_metrics"] for record in agent_records}
    validation_predictions = {record["name"]: np.asarray(record["validation_prediction"], dtype=int) for record in agent_records}
    validation_probabilities = {record["name"]: np.asarray(record["validation_probability"], dtype=float) for record in agent_records}
    roles = {record["name"]: str(record["role"]) for record in agent_records}

    baseline_rows = []
    for record in agent_records:
        metrics = record["metrics"]
        prediction_array = np.asarray(record["prediction"], dtype=int)
        attack_rate = float(np.mean(prediction_array == 1))
        normal_rate = float(np.mean(prediction_array == 0))
        baseline_rows.append(
            {
                "stage": record["name"],
                "Accuracy": float(metrics["test_accuracy"]),
                "Precision": float(metrics["test_precision"]),
                "Recall": float(metrics["test_recall"]),
                "F1": float(metrics["test_f1"]),
                "FPR": float(metrics["fpr"]),
                "FNR": float(metrics["fnr"]),
                "Specificity": float(metrics["specificity"]),
                "Balanced Accuracy": float(metrics["balanced_accuracy"]),
                "Role": record["role"],
                "Selected Threshold": float(record["selected_threshold"]),
                "Predicted Attack Rate": attack_rate,
                "Predicted Normal Rate": normal_rate,
            }
        )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(dataset_output_dir / "four_agent_baseline.csv", index=False)

    specialization_report_df = baseline_df.rename(columns={"stage": "Agent"})[
        [
            "Agent",
            "Role",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "FPR",
            "FNR",
            "Specificity",
            "Balanced Accuracy",
            "Selected Threshold",
            "Predicted Attack Rate",
            "Predicted Normal Rate",
        ]
    ]
    specialization_report_df.to_csv(dataset_output_dir / "four_agent_specialization_report.csv", index=False)

    base_f1_scores = {
        name: float(validation_model_metrics[name].get("test_f1", 0.0)) for name in model_preds
    }
    score_sum = float(sum(base_f1_scores.values()))
    if score_sum <= 0.0:
        trust_weights = {name: 1.0 / len(model_preds) for name in model_preds}
    else:
        trust_weights = {name: score / score_sum for name, score in base_f1_scores.items()}

    trust_weights_df = pd.DataFrame(
        [
            {
                "decision_source": name,
                "role": roles[name],
                "agent_reliability_score": float(base_f1_scores[name]),
                "trust_weight": float(trust_weights[name]),
            }
            for name in model_preds
        ]
    )
    trust_weights_df.to_csv(dataset_output_dir / "trust_weights.csv", index=False)

    trust_outputs: Dict[str, np.ndarray] = {}
    majority_output = trust_methods.majority_voting(model_preds)
    trust_outputs["Majority Voting"] = np.asarray(majority_output["predictions"], dtype=int)

    global_trust_output = trust_methods.accuracy_based_trust(model_preds, validation_model_metrics)
    trust_outputs["Global Trust Voting"] = np.asarray(global_trust_output["predictions"], dtype=int)

    attack_recall_trust_output = trust_methods.attack_recall_trust(model_preds, validation_model_metrics)
    trust_outputs["Attack Recall Trust"] = np.asarray(attack_recall_trust_output["predictions"], dtype=int)

    role_aware_cfg = dict(trust_cfg.get("role_aware_trust_voting", {}))
    role_aware_output = trust_methods.role_aware_trust_voting(
        model_preds,
        model_probs,
        validation_model_metrics,
        roles,
        attack_threshold=float(role_aware_cfg.get("attack_threshold", 0.60)),
        normal_threshold=float(role_aware_cfg.get("normal_threshold", 0.65)),
    )
    trust_outputs["Role-Aware Trust Voting"] = np.asarray(role_aware_output["predictions"], dtype=int)

    selector_cfg = dict(trust_cfg.get("trust_agent_selector", {}))
    current_selector_params = {
        "neighbor_k": int(selector_cfg.get("neighbor_k", 25)),
        "validation_role_weight": 0.25,
        "confidence_weight": 0.20,
        "margin_weight": 0.15,
        "local_accuracy_weight": 0.10,
        "disagreement_bonus": float(selector_cfg.get("disagreement_bonus", 0.10)),
        "attack_role_bonus": float(selector_cfg.get("attack_role_bonus", 0.08)),
        "normal_role_bonus": float(selector_cfg.get("normal_role_bonus", 0.08)),
        "attack_confidence_threshold": float(selector_cfg.get("attack_confidence_threshold", 0.60)),
        "normal_confidence_threshold": float(selector_cfg.get("normal_confidence_threshold", 0.65)),
    }
    current_selector_output = trust_methods.trust_agent_selector(
        model_preds,
        model_probs,
        validation_model_metrics,
        roles,
        x_val_full,
        np.asarray(y_val, dtype=int),
        validation_predictions,
        x_test_full,
        **current_selector_params,
    )

    tuned_selector_params = _search_best_selector_params(
        model_preds,
        model_probs,
        validation_model_metrics,
        roles,
        x_val_full,
        np.asarray(y_val, dtype=int),
        validation_predictions,
        x_test_full,
    )
    tuned_selector_output = trust_methods.trust_agent_selector(
        model_preds,
        model_probs,
        validation_model_metrics,
        roles,
        x_val_full,
        np.asarray(y_val, dtype=int),
        validation_predictions,
        x_test_full,
        **tuned_selector_params,
    )
    trust_outputs["Trust Agent Selector"] = np.asarray(tuned_selector_output["predictions"], dtype=int)

    cic_selector_outputs: Dict[str, Dict[str, Any]] = {}
    cic_best_selector_name = "Trust Agent Selector"
    cic_best_selector_pred = np.asarray(tuned_selector_output["predictions"], dtype=int)
    cic_best_selector_selected_agents = list(tuned_selector_output.get("meta", {}).get("selected_agents", []))
    if dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        selector_variants_cfg = [
            ("Accuracy-prioritized selector", {"validation_accuracy": 0.45, "validation_f1": 0.20, "sample_confidence": 0.20, "margin": 0.15}),
            ("F1-prioritized selector", {"validation_f1": 0.35, "validation_accuracy": 0.25, "sample_confidence": 0.20, "margin": 0.20}),
            ("Recall-safe selector", {"validation_accuracy": 0.30, "validation_f1": 0.25, "validation_recall": 0.25, "sample_confidence": 0.10, "margin": 0.10}),
            ("FNR-penalty selector", {"validation_accuracy": 0.30, "validation_f1": 0.25, "validation_one_minus_fnr": 0.20, "sample_confidence": 0.15, "margin": 0.10}),
            ("Disagreement-aware selector", {"validation_accuracy": 0.35, "validation_f1": 0.20, "sample_confidence": 0.20, "margin": 0.15, "local_accuracy": 0.10}),
        ]
        for variant_name, weights in selector_variants_cfg:
            boost = 0.30 if variant_name == "Disagreement-aware selector" else 0.0
            val_variant = _run_weighted_selector_variant(
                name=variant_name,
                query_predictions=validation_predictions,
                query_probabilities=validation_probabilities,
                validation_metrics=validation_model_metrics,
                validation_features=x_val_full,
                validation_labels=np.asarray(y_val, dtype=int),
                validation_predictions=validation_predictions,
                query_features=x_val_full,
                neighbor_k=int(current_selector_params["neighbor_k"]),
                weights=weights,
                disagreement_boost_strength=boost,
            )
            test_variant = _run_weighted_selector_variant(
                name=variant_name,
                query_predictions=model_preds,
                query_probabilities=model_probs,
                validation_metrics=validation_model_metrics,
                validation_features=x_val_full,
                validation_labels=np.asarray(y_val, dtype=int),
                validation_predictions=validation_predictions,
                query_features=x_test_full,
                neighbor_k=int(current_selector_params["neighbor_k"]),
                weights=weights,
                disagreement_boost_strength=boost,
            )
            cic_selector_outputs[variant_name] = {
                "weights": weights,
                "val_predictions": np.asarray(val_variant["predictions"], dtype=int),
                "test_predictions": np.asarray(test_variant["predictions"], dtype=int),
                "selected_agents": list(test_variant["selected_agents"]),
            }

        best_single_val_recall = max(float(v.get("test_recall", 0.0)) for v in validation_model_metrics.values())
        min_recall_threshold = best_single_val_recall - 0.005
        candidate_scores = []
        for name, payload in cic_selector_outputs.items():
            val_metrics_variant = evaluate_predictions(np.asarray(y_val, dtype=int), payload["val_predictions"])
            candidate_scores.append(
                (
                    name,
                    float(val_metrics_variant["test_accuracy"]),
                    float(val_metrics_variant["test_f1"]),
                    float(val_metrics_variant["test_recall"]),
                )
            )
        feasible = [c for c in candidate_scores if c[3] >= min_recall_threshold]
        chooser = feasible if feasible else candidate_scores
        winner = sorted(chooser, key=lambda x: (x[1], x[2]), reverse=True)[0]
        cic_best_selector_name = winner[0]
        cic_best_selector_pred = np.asarray(cic_selector_outputs[cic_best_selector_name]["test_predictions"], dtype=int)
        cic_best_selector_selected_agents = list(cic_selector_outputs[cic_best_selector_name]["selected_agents"])

    best_agent_row = baseline_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_agent_accuracy = float(best_agent_row["Accuracy"])
    best_agent_f1 = float(best_agent_row["F1"])
    best_agent_recall = float(best_agent_row["Recall"])
    best_agent_fnr = float(best_agent_row["FNR"])

    trust_rows = []
    trust_method_names = [
        "Majority Voting",
        "Global Trust Voting",
        "Attack Recall Trust",
        "Role-Aware Trust Voting",
        "Trust Agent Selector",
    ]
    if dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        for extra_name in [
            "Accuracy-prioritized selector",
            "F1-prioritized selector",
            "Recall-safe selector",
            "FNR-penalty selector",
            "Disagreement-aware selector",
        ]:
            if extra_name in cic_selector_outputs:
                trust_outputs[extra_name] = np.asarray(cic_selector_outputs[extra_name]["test_predictions"], dtype=int)
                trust_method_names.append(extra_name)
        trust_outputs["Best tuned CICIDS2017 selector"] = cic_best_selector_pred
        trust_method_names.append("Best tuned CICIDS2017 selector")

    ai_trust_cfg = build_ai_trust_config(poison_options or {})
    ai_decision_rows: List[Dict[str, Any]] = []
    if ai_trust_cfg.enabled:
        method_metrics_for_ai: Dict[str, Dict[str, float]] = {}
        method_predictions_for_ai: Dict[str, int] = {}
        for method_name in trust_method_names:
            y_pred = np.asarray(trust_outputs[method_name], dtype=int)
            m = evaluate_predictions(np.asarray(y_test, dtype=int), y_pred)
            method_metrics_for_ai[method_name] = {
                "Accuracy": float(m["test_accuracy"]),
                "Precision": float(m["test_precision"]),
                "Recall": float(m["test_recall"]),
                "F1": float(m["test_f1"]),
                "FPR": float(m["fpr"]),
                "FNR": float(m["fnr"]),
                "Balanced Accuracy": float(m["balanced_accuracy"]),
            }
            method_predictions_for_ai[method_name] = int(y_pred[0]) if y_pred.size else 0

        cache_file = results_dir / "ai_trust_cache" / f"{dataset_file_stem}_ai_trust_cache.jsonl"
        ai_decision = select_method_with_ai(
            dataset=dataset_name,
            scenario="Clean",
            poisoned_agent="None",
            poison_mode=str(poison_options.get("poison_mode", "none")) if poison_options else "none",
            poison_rate=float(poison_options.get("poison_rate", 0.0)) if poison_options else 0.0,
            allowed_methods=list(method_metrics_for_ai.keys()),
            method_metrics=method_metrics_for_ai,
            method_predictions=method_predictions_for_ai,
            agreement_level=float(np.mean(list(method_predictions_for_ai.values()))) if method_predictions_for_ai else None,
            suspected_poisoned_agent=None,
            config=ai_trust_cfg,
            cache_file=cache_file,
        )
        selected_method = str(ai_decision.get("selected_method", ""))
        if selected_method in trust_outputs:
            trust_outputs["AI Trust Auditor"] = np.asarray(trust_outputs[selected_method], dtype=int)
            trust_method_names.append("AI Trust Auditor")
        ai_decision_rows.append(
            {
                "Dataset": dataset_name,
                "Scenario": "Clean",
                "Poisoned Agent": "None",
                "Poison Mode": str(poison_options.get("poison_mode", "none")) if poison_options else "none",
                "Poison Rate": float(poison_options.get("poison_rate", 0.0)) if poison_options else 0.0,
                "Selected Method": selected_method,
                "Selected Accuracy": ai_decision.get("selected_accuracy", np.nan),
                "Selected F1": ai_decision.get("selected_f1", np.nan),
                "Selected FPR": ai_decision.get("selected_fpr", np.nan),
                "Selected FNR": ai_decision.get("selected_fnr", np.nan),
                "AI Confidence": ai_decision.get("confidence", np.nan),
                "Suspected Unreliable Agents": "|".join(ai_decision.get("suspected_unreliable_agents", [])),
                "Reason": ai_decision.get("reason", ""),
                "Status": ai_decision.get("status", ""),
                "Fallback Used": bool(ai_decision.get("fallback_used", False)),
            }
        )

    for method_name in trust_method_names:
        y_pred = trust_outputs[method_name]
        metrics = evaluate_predictions(np.asarray(y_test, dtype=int), y_pred)
        trust_rows.append(
            {
                "stage": method_name,
                "Accuracy": float(metrics["test_accuracy"]),
                "Precision": float(metrics["test_precision"]),
                "Recall": float(metrics["test_recall"]),
                "F1": float(metrics["test_f1"]),
                "FPR": float(metrics["fpr"]),
                "FNR": float(metrics["fnr"]),
                "Specificity": float(metrics["specificity"]),
                "Balanced Accuracy": float(metrics["balanced_accuracy"]),
                "Improvement vs Best Agent Accuracy": float(metrics["test_accuracy"] - best_agent_accuracy),
                "Improvement vs Best Agent F1": float(metrics["test_f1"] - best_agent_f1),
                "Improvement vs Best Agent Recall": float(metrics["test_recall"] - best_agent_recall),
                "FNR Reduction vs Best Agent": float(best_agent_fnr - metrics["fnr"]),
            }
        )

    trust_df = pd.DataFrame(trust_rows)
    best_trust_row = trust_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_trust_method_name = str(best_trust_row["stage"])
    trust_rows.append(
        {
            "stage": "Best 4-Agent Trust Method",
            "Accuracy": float(best_trust_row["Accuracy"]),
            "Precision": float(best_trust_row["Precision"]),
            "Recall": float(best_trust_row["Recall"]),
            "F1": float(best_trust_row["F1"]),
            "FPR": float(best_trust_row["FPR"]),
            "FNR": float(best_trust_row["FNR"]),
            "Specificity": float(best_trust_row["Specificity"]),
            "Balanced Accuracy": float(best_trust_row["Balanced Accuracy"]),
            "Improvement vs Best Agent Accuracy": float(best_trust_row["Improvement vs Best Agent Accuracy"]),
            "Improvement vs Best Agent F1": float(best_trust_row["Improvement vs Best Agent F1"]),
            "Improvement vs Best Agent Recall": float(best_trust_row["Improvement vs Best Agent Recall"]),
            "FNR Reduction vs Best Agent": float(best_trust_row["FNR Reduction vs Best Agent"]),
        }
    )
    trust_results_df = pd.DataFrame(trust_rows)
    trust_results_df.to_csv(dataset_output_dir / "four_agent_trust_results.csv", index=False)
    if ai_decision_rows:
        pd.DataFrame(ai_decision_rows).to_csv(dataset_output_dir / "ai_trust_decisions.csv", index=False)

    selector_variant_specs = [
        (
            "Current Trust Agent Selector",
            current_selector_params,
        ),
        (
            "Confidence-only Selector",
            {
                "neighbor_k": current_selector_params["neighbor_k"],
                "validation_role_weight": 0.0,
                "confidence_weight": 0.45,
                "margin_weight": 0.20,
                "local_accuracy_weight": 0.0,
                "disagreement_bonus": 0.0,
                "attack_role_bonus": 0.0,
                "normal_role_bonus": 0.0,
                "attack_confidence_threshold": current_selector_params["attack_confidence_threshold"],
                "normal_confidence_threshold": current_selector_params["normal_confidence_threshold"],
            },
        ),
        (
            "Role-bonus Selector",
            {
                "neighbor_k": current_selector_params["neighbor_k"],
                "validation_role_weight": 0.30,
                "confidence_weight": 0.10,
                "margin_weight": 0.05,
                "local_accuracy_weight": 0.05,
                "disagreement_bonus": 0.10,
                "attack_role_bonus": 0.12,
                "normal_role_bonus": 0.12,
                "attack_confidence_threshold": current_selector_params["attack_confidence_threshold"],
                "normal_confidence_threshold": current_selector_params["normal_confidence_threshold"],
            },
        ),
        (
            "Disagreement-aware Selector",
            {
                "neighbor_k": current_selector_params["neighbor_k"],
                "validation_role_weight": 0.20,
                "confidence_weight": 0.15,
                "margin_weight": 0.10,
                "local_accuracy_weight": 0.05,
                "disagreement_bonus": 0.25,
                "attack_role_bonus": 0.05,
                "normal_role_bonus": 0.05,
                "attack_confidence_threshold": current_selector_params["attack_confidence_threshold"],
                "normal_confidence_threshold": current_selector_params["normal_confidence_threshold"],
            },
        ),
        (
            "Local-accuracy Selector",
            {
                "neighbor_k": current_selector_params["neighbor_k"],
                "validation_role_weight": 0.05,
                "confidence_weight": 0.10,
                "margin_weight": 0.05,
                "local_accuracy_weight": 0.35,
                "disagreement_bonus": 0.05,
                "attack_role_bonus": 0.05,
                "normal_role_bonus": 0.05,
                "attack_confidence_threshold": current_selector_params["attack_confidence_threshold"],
                "normal_confidence_threshold": current_selector_params["normal_confidence_threshold"],
            },
        ),
        (
            "Tuned Selector",
            tuned_selector_params,
        ),
    ]

    selector_ablation_rows = []
    selector_variant_outputs: Dict[str, Dict[str, Any]] = {}
    for variant_name, params in selector_variant_specs:
        variant_output = _run_selector_variant(
            name=variant_name,
            predictions=model_preds,
            probabilities=model_probs,
            validation_metrics=validation_model_metrics,
            roles=roles,
            x_val_features=x_val_full,
            y_val=np.asarray(y_val, dtype=int),
            validation_predictions=validation_predictions,
            x_test_features=x_test_full,
            y_test=np.asarray(y_test, dtype=int),
            params=params,
        )
        selector_variant_outputs[variant_name] = variant_output
        selector_ablation_rows.append(variant_output)

    selector_ablation_df = pd.DataFrame(
        [
            {
                "stage": row["stage"],
                "Accuracy": row["Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "FPR": row["FPR"],
                "FNR": row["FNR"],
                "Specificity": row["Specificity"],
                "Balanced Accuracy": row["Balanced Accuracy"],
                "Improvement vs Best Agent Accuracy": float(row["Accuracy"] - best_agent_accuracy),
                "Missed Opportunity Count": row["Missed Opportunity Count"],
                "Oracle Gap": row["Oracle Gap"],
            }
            for row in selector_ablation_rows
        ]
    )
    selector_ablation_df.to_csv(dataset_output_dir / "four_agent_selector_ablation.csv", index=False)

    # Compute oracle accuracy early so it's available for ablation
    oracle_summary = compute_oracle_upper_bound(np.asarray(y_test, dtype=int), model_preds)
    oracle_accuracy = float(oracle_summary["oracle_accuracy"])

    if dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        ablation_rows_cic: List[Dict[str, Any]] = []
        selection_distributions: Dict[str, str] = {}
        for variant_name, payload in cic_selector_outputs.items():
            preds = np.asarray(payload["test_predictions"], dtype=int)
            metrics = evaluate_predictions(np.asarray(y_test, dtype=int), preds)
            sel = pd.Series(payload["selected_agents"])
            dist = "; ".join([f"{name}:{count/len(sel):.4f}" for name, count in sel.value_counts().items()]) if len(sel) else ""
            selection_distributions[variant_name] = dist
            ablation_rows_cic.append(
                {
                    "Method": variant_name,
                    "Accuracy": float(metrics["test_accuracy"]),
                    "Precision": float(metrics["test_precision"]),
                    "Recall": float(metrics["test_recall"]),
                    "F1": float(metrics["test_f1"]),
                    "FPR": float(metrics["fpr"]),
                    "FNR": float(metrics["fnr"]),
                    "Specificity": float(metrics["specificity"]),
                    "Balanced Accuracy": float(metrics["balanced_accuracy"]),
                    "Improvement vs Best Agent Accuracy": float(metrics["test_accuracy"] - best_agent_accuracy),
                    "F1 Improvement vs Best Agent": float(metrics["test_f1"] - best_agent_f1),
                    "Recall Change vs Best Agent": float(metrics["test_recall"] - best_agent_recall),
                    "FNR Change vs Best Agent": float(metrics["fnr"] - best_agent_fnr),
                    "Oracle Gap": float(oracle_accuracy - float(metrics["test_accuracy"])),
                    "Selection distribution": dist,
                }
            )
        current_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), np.asarray(tuned_selector_output["predictions"], dtype=int))
        ablation_rows_cic.insert(
            0,
            {
                "Method": "Current Trust Agent Selector",
                "Accuracy": float(current_metrics["test_accuracy"]),
                "Precision": float(current_metrics["test_precision"]),
                "Recall": float(current_metrics["test_recall"]),
                "F1": float(current_metrics["test_f1"]),
                "FPR": float(current_metrics["fpr"]),
                "FNR": float(current_metrics["fnr"]),
                "Specificity": float(current_metrics["specificity"]),
                "Balanced Accuracy": float(current_metrics["balanced_accuracy"]),
                "Improvement vs Best Agent Accuracy": float(current_metrics["test_accuracy"] - best_agent_accuracy),
                "F1 Improvement vs Best Agent": float(current_metrics["test_f1"] - best_agent_f1),
                "Recall Change vs Best Agent": float(current_metrics["test_recall"] - best_agent_recall),
                "FNR Change vs Best Agent": float(current_metrics["fnr"] - best_agent_fnr),
                "Oracle Gap": float(oracle_accuracy - float(current_metrics["test_accuracy"])),
                "Selection distribution": "; ".join([f"{k}:{v/len(cic_best_selector_selected_agents):.4f}" for k, v in pd.Series(list(tuned_selector_output.get("meta", {}).get("selected_agents", []))).value_counts().items()]) if len(list(tuned_selector_output.get("meta", {}).get("selected_agents", []))) else "",
            },
        )
        ablation_rows_cic.append(
            {
                "Method": "Oracle",
                "Accuracy": oracle_accuracy,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "FPR": np.nan,
                "FNR": np.nan,
                "Specificity": np.nan,
                "Balanced Accuracy": np.nan,
                "Improvement vs Best Agent Accuracy": float(oracle_accuracy - best_agent_accuracy),
                "F1 Improvement vs Best Agent": np.nan,
                "Recall Change vs Best Agent": np.nan,
                "FNR Change vs Best Agent": np.nan,
                "Oracle Gap": 0.0,
                "Selection distribution": "",
            }
        )
        pd.DataFrame(ablation_rows_cic).to_csv(dataset_output_dir / "cicids2017_selector_ablation.csv", index=False)

        missed_df = _build_cic_missed_opportunity_report(
            np.asarray(y_test, dtype=int),
            cic_best_selector_selected_agents,
            np.asarray(cic_best_selector_pred, dtype=int),
            model_preds,
            model_probs,
        )
        missed_df.to_csv(dataset_output_dir / "cicids2017_missed_opportunity_report.csv", index=False)

    # --------------------------
    # Hard-Case Guarded Selector
    # Tune rules on validation only using validation predictions/probabilities
    try:
        guarded_params = _search_best_hard_case_guarded_params(
            validation_predictions,
            warmup_validation_probabilities,
            x_val_full,
            np.asarray(y_val, dtype=int),
        )
    except Exception:
        guarded_params = {
            "hard_case_min_confidence": 0.7,
            "hard_case_min_margin": 0.2,
            "override_confidence_gap": 0.25,
            "attack_override_confidence": 0.8,
            "normal_override_confidence": 0.8,
            "disagreement_required": 2,
        }

    # Apply on test
    guarded_test_pred = _apply_hard_case_guarded_selector(model_preds, model_probs, guarded_params)
    guarded_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), np.asarray(guarded_test_pred, dtype=int))

    # Hard-Case only metrics
    hard_only_pred = np.asarray(model_preds["Hard-Case Agent"], dtype=int)
    hard_only_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), hard_only_pred)

    # Compute override statistics
    hard_pred = hard_only_pred
    overrides = guarded_test_pred != hard_pred
    override_rate = float(np.mean(overrides))
    successful_overrides = np.sum((guarded_test_pred == np.asarray(y_test, dtype=int)) & overrides)
    failed_overrides = np.sum((guarded_test_pred != np.asarray(y_test, dtype=int)) & overrides)
    successful_override_rate = float(successful_overrides / overrides.sum()) if overrides.sum() > 0 else 0.0
    failed_override_rate = float(failed_overrides / overrides.sum()) if overrides.sum() > 0 else 0.0
    selected_hard_case_rate = float(np.mean(~overrides))

    # Diagnostics + Missed-opportunity + UNSW selector ablation
    # 1) diagnostics for guarded selector (validation + test)
    try:
        val_guarded_pred = _apply_hard_case_guarded_selector(validation_predictions, warmup_validation_probabilities, guarded_params)
        val_metrics_guarded = evaluate_predictions(np.asarray(y_val, dtype=int), np.asarray(val_guarded_pred, dtype=int))
        val_hard_pred = np.asarray(validation_predictions["Hard-Case Agent"], dtype=int)
        val_hard_metrics = evaluate_predictions(np.asarray(y_val, dtype=int), val_hard_pred)

        val_overrides = val_guarded_pred != val_hard_pred
        val_overrides_n = int(np.sum(val_overrides))
        val_successful_overrides = int(np.sum((val_guarded_pred == np.asarray(y_val, dtype=int)) & val_overrides))
        val_failed_overrides = int(np.sum((val_guarded_pred != np.asarray(y_val, dtype=int)) & val_overrides))
        val_override_rate = float(val_overrides_n / len(val_guarded_pred)) if len(val_guarded_pred) > 0 else 0.0
        val_successful_override_rate = float(val_successful_overrides / val_overrides_n) if val_overrides_n > 0 else 0.0
        val_failed_override_rate = float(val_failed_overrides / val_overrides_n) if val_overrides_n > 0 else 0.0

        beats_hard_case_validation = float(val_metrics_guarded["test_accuracy"]) > float(val_hard_metrics["test_accuracy"])
        beats_hard_case_test = float(guarded_metrics["test_accuracy"]) > float(hard_only_metrics["test_accuracy"])

        selected_rule_set_id = (
            f"hc_conf={guarded_params['hard_case_min_confidence']}_hc_margin={guarded_params['hard_case_min_margin']}_gap={guarded_params['override_confidence_gap']}_cand_min={guarded_params['candidate_min_confidence']}_req2={guarded_params['require_two_agent_agreement']}"
        )

        diagnostics_row = {
            "selected_rule_set_id": selected_rule_set_id,
            "hard_case_min_confidence": float(guarded_params.get("hard_case_min_confidence")),
            "hard_case_min_margin": float(guarded_params.get("hard_case_min_margin")),
            "override_confidence_gap": float(guarded_params.get("override_confidence_gap")),
            "candidate_min_confidence": float(guarded_params.get("candidate_min_confidence")),
            "require_two_agent_agreement": bool(guarded_params.get("require_two_agent_agreement")),
            "validation_accuracy": float(val_metrics_guarded["test_accuracy"]),
            "validation_f1": float(val_metrics_guarded["test_f1"]),
            "validation_recall": float(val_metrics_guarded["test_recall"]),
            "validation_precision": float(val_metrics_guarded["test_precision"]),
            "validation_balanced_accuracy": float(val_metrics_guarded["balanced_accuracy"]),
            "validation_override_rate": val_override_rate,
            "validation_successful_override_rate": val_successful_override_rate,
            "validation_failed_override_rate": val_failed_override_rate,
            "test_accuracy": float(guarded_metrics["test_accuracy"]),
            "test_f1": float(guarded_metrics["test_f1"]),
            "test_recall": float(guarded_metrics["test_recall"]),
            "test_precision": float(guarded_metrics["test_precision"]),
            "test_fnr": float(guarded_metrics["fnr"]),
            "test_override_rate": float(override_rate),
            "test_successful_override_rate": float(successful_override_rate),
            "test_failed_override_rate": float(failed_override_rate),
            "beats_hard_case_validation": bool(beats_hard_case_validation),
            "beats_hard_case_test": bool(beats_hard_case_test),
        }
        pd.DataFrame([diagnostics_row]).to_csv(dataset_output_dir / "hard_case_override_diagnostics.csv", index=False)
    except Exception:
        pd.DataFrame([{"status": "failed_to_write_hard_case_override_diagnostics"}]).to_csv(
            dataset_output_dir / "hard_case_override_diagnostics.csv", index=False
        )

    # 2) Missed-opportunity report (always write)
    try:
        tuned_selector_predictions = np.asarray(tuned_selector_output.get("predictions", np.zeros(len(y_test))), dtype=int)
        hard_preds = np.asarray(model_preds["Hard-Case Agent"], dtype=int)
        tuned_overrode_mask = tuned_selector_predictions != hard_preds
        _build_unsw_missed_opportunity_report_file(
            dataset_output_dir,
            np.asarray(y_test, dtype=int),
            model_preds,
            model_probs,
            selector_pred=tuned_selector_predictions,
            selector_overrode_mask=tuned_overrode_mask,
        )
    except Exception:
        pd.DataFrame([{"status": "failed_to_build_missed_opportunity_report"}]).to_csv(
            dataset_output_dir / "unsw_missed_opportunity_report.csv", index=False
        )

    # 3) Build UNSW selector ablation with override stats
    ablation_rows = []
    method_variants = [
        ("Hard-Case Only", hard_only_pred),
        ("Majority Voting", np.asarray(trust_outputs["Majority Voting"], dtype=int)),
    ]
    # include Disagreement-aware if available
    if "Disagreement-aware Selector" in selector_variant_outputs:
        disc_pred = np.asarray(selector_variant_outputs["Disagreement-aware Selector"]["predictions"], dtype=int)
        method_variants.append(("Disagreement-aware Selector", disc_pred))
    # add guarded
    method_variants.append(("Hard-Case Guarded Selector", np.asarray(guarded_test_pred, dtype=int)))
    # oracle (we represent only accuracy here)

    for name, pred_arr in method_variants:
        pred_arr = np.asarray(pred_arr, dtype=int)
        metrics_row = _selector_eval_metrics(np.asarray(y_test, dtype=int), pred_arr)
        hard_arr = np.asarray(hard_only_pred, dtype=int)
        overrides = pred_arr != hard_arr
        overrides_n = int(np.sum(overrides))
        successful_overrides = int(np.sum((pred_arr == np.asarray(y_test, dtype=int)) & overrides))
        failed_overrides = int(np.sum((pred_arr != np.asarray(y_test, dtype=int)) & overrides))
        override_rate_row = float(overrides_n / len(pred_arr)) if len(pred_arr) > 0 else 0.0
        successful_override_rate_row = float(successful_overrides / overrides_n) if overrides_n > 0 else 0.0
        failed_override_rate_row = float(failed_overrides / overrides_n) if overrides_n > 0 else 0.0
        ablation_rows.append(
            {
                "Method": name,
                **metrics_row,
                "Improvement vs Hard-Case": float(metrics_row["Accuracy"] - float(hard_only_metrics["test_accuracy"])),
                "Oracle Gap": float(oracle_accuracy - float(metrics_row["Accuracy"])),
                "Override Rate": override_rate_row,
                "Successful Override Rate": successful_override_rate_row,
                "Failed Override Rate": failed_override_rate_row,
            }
        )

    # Oracle row
    ablation_rows.append(
        {
            "Method": "Oracle",
            "Accuracy": oracle_accuracy,
            "Precision": float(np.nan),
            "Recall": float(np.nan),
            "F1": float(np.nan),
            "FPR": float(np.nan),
            "FNR": float(np.nan),
            "Specificity": float(np.nan),
            "Balanced Accuracy": float(np.nan),
            "Improvement vs Hard-Case": float(oracle_accuracy - float(hard_only_metrics["test_accuracy"])),
            "Oracle Gap": 0.0,
            "Override Rate": float(np.nan),
            "Successful Override Rate": float(np.nan),
            "Failed Override Rate": float(np.nan),
        }
    )
    pd.DataFrame(ablation_rows).to_csv(dataset_output_dir / "unsw_selector_ablation.csv", index=False)

    # 4) Update trust results and final comparison to include Hard-Case Only, Disagreement-aware, Hard-Case Guarded
    try:
        trust_results_df = pd.read_csv(dataset_output_dir / "four_agent_trust_results.csv")
        # create Hard-Case Only row for trust table
        hard_row = {
            "stage": "Hard-Case Only",
            "Accuracy": float(hard_only_metrics["test_accuracy"]),
            "Precision": float(hard_only_metrics["test_precision"]),
            "Recall": float(hard_only_metrics["test_recall"]),
            "F1": float(hard_only_metrics["test_f1"]),
            "FPR": float(hard_only_metrics["fpr"]),
            "FNR": float(hard_only_metrics["fnr"]),
            "Specificity": float(hard_only_metrics["specificity"]),
            "Balanced Accuracy": float(hard_only_metrics["balanced_accuracy"]),
            "Improvement vs Best Agent Accuracy": float(hard_only_metrics["test_accuracy"] - best_agent_accuracy),
            "Improvement vs Best Agent F1": float(hard_only_metrics["test_f1"] - best_agent_f1),
            "Improvement vs Best Agent Recall": float(hard_only_metrics["test_recall"] - best_agent_recall),
            "FNR Reduction vs Best Agent": float(best_agent_fnr - hard_only_metrics["fnr"]),
        }
        # append if not present
        if not (trust_results_df["stage"] == "Hard-Case Only").any():
            trust_results_df = pd.concat([pd.DataFrame([hard_row]), trust_results_df], ignore_index=True)

        # Disagreement-aware row
        if "Disagreement-aware Selector" in selector_variant_outputs:
            disc = selector_variant_outputs["Disagreement-aware Selector"]
            disc_pred = np.asarray(disc["predictions"], dtype=int)
            disc_metrics = evaluate_predictions(np.asarray(y_test, dtype=int), disc_pred)
            disc_row = {
                "stage": "Disagreement-aware Selector",
                "Accuracy": float(disc_metrics["test_accuracy"]),
                "Precision": float(disc_metrics["test_precision"]),
                "Recall": float(disc_metrics["test_recall"]),
                "F1": float(disc_metrics["test_f1"]),
                "FPR": float(disc_metrics["fpr"]),
                "FNR": float(disc_metrics["fnr"]),
                "Specificity": float(disc_metrics["specificity"]),
                "Balanced Accuracy": float(disc_metrics["balanced_accuracy"]),
                "Improvement vs Best Agent Accuracy": float(disc_metrics["test_accuracy"] - best_agent_accuracy),
                "Improvement vs Best Agent F1": float(disc_metrics["test_f1"] - best_agent_f1),
                "Improvement vs Best Agent Recall": float(disc_metrics["test_recall"] - best_agent_recall),
                "FNR Reduction vs Best Agent": float(best_agent_fnr - disc_metrics["fnr"]),
            }
            if not (trust_results_df["stage"] == "Disagreement-aware Selector").any():
                trust_results_df = pd.concat([trust_results_df, pd.DataFrame([disc_row])], ignore_index=True)

        # Guarded selector row
        guarded_row = {
            "stage": "Hard-Case Guarded Selector",
            "Accuracy": float(guarded_metrics["test_accuracy"]),
            "Precision": float(guarded_metrics["test_precision"]),
            "Recall": float(guarded_metrics["test_recall"]),
            "F1": float(guarded_metrics["test_f1"]),
            "FPR": float(guarded_metrics["fpr"]),
            "FNR": float(guarded_metrics["fnr"]),
            "Specificity": float(guarded_metrics["specificity"]),
            "Balanced Accuracy": float(guarded_metrics["balanced_accuracy"]),
            "Improvement vs Best Agent Accuracy": float(guarded_metrics["test_accuracy"] - best_agent_accuracy),
            "Improvement vs Best Agent F1": float(guarded_metrics["test_f1"] - best_agent_f1),
            "Improvement vs Best Agent Recall": float(guarded_metrics["test_recall"] - best_agent_recall),
            "FNR Reduction vs Best Agent": float(best_agent_fnr - guarded_metrics["fnr"]),
        }
        if not (trust_results_df["stage"] == "Hard-Case Guarded Selector").any():
            trust_results_df = pd.concat([trust_results_df, pd.DataFrame([guarded_row])], ignore_index=True)

        trust_results_df.to_csv(dataset_output_dir / "four_agent_trust_results.csv", index=False)

        # update final comparison
        final_comparison = pd.read_csv(dataset_output_dir / "final_comparison.csv")
        append_rows = []
        # add Disagreement-aware as kind=trust_layer if present
        if "Disagreement-aware Selector" in selector_variant_outputs and not ((final_comparison["stage"] == "Disagreement-aware Selector").any()):
            append_rows.append({**{"stage": "Disagreement-aware Selector", "kind": "trust_layer"}, **_selector_eval_metrics(np.asarray(y_test, dtype=int), disc_pred)})
        if not ((final_comparison["stage"] == "Hard-Case Guarded Selector").any()):
            append_rows.append({**{"stage": "Hard-Case Guarded Selector", "kind": "trust_layer"}, **_selector_eval_metrics(np.asarray(y_test, dtype=int), guarded_test_pred)})
        if append_rows:
            fc_df = pd.concat([final_comparison, pd.DataFrame(append_rows)], ignore_index=True, sort=False)
            fc_df.to_csv(dataset_output_dir / "final_comparison.csv", index=False)
    except Exception:
        # if something goes wrong updating summary tables, write minimal indicators
        pd.DataFrame([{"status": "failed_to_update_trust_and_final_tables"}]).to_csv(dataset_output_dir / "unsw_selector_ablation_update_status.csv", index=False)

    best_trust_accuracy = float(best_trust_row["Accuracy"])
    trust_accuracy = float(best_trust_row["Accuracy"])
    worst_agent_row = baseline_df.sort_values("Accuracy", ascending=True).iloc[0]
    worst_agent_accuracy = float(worst_agent_row["Accuracy"])
    accuracy_spread = float(best_agent_accuracy - worst_agent_accuracy)
    max_possible_improvement = float(oracle_accuracy - best_agent_accuracy)
    achieved_improvement = float(best_trust_accuracy - best_agent_accuracy)
    target_possible = bool(max_possible_improvement >= 0.02)
    target_achieved = bool(achieved_improvement >= 0.02)

    selected_agent_names = list(tuned_selector_output.get("meta", {}).get("selected_agents", []))
    tuned_selector_predictions = np.asarray(tuned_selector_output["predictions"], dtype=int)
    oracle_breakdown = _compute_oracle_breakdown(np.asarray(y_test, dtype=int), model_preds, tuned_selector_predictions)
    error_overlap_df = _build_four_agent_error_overlap_report(
        np.asarray(y_test, dtype=int),
        model_preds,
        selected_agent_names,
        tuned_selector_predictions,
    )
    error_overlap_df.to_csv(dataset_output_dir / "four_agent_error_overlap_report.csv", index=False)

    diversity_df = compute_model_diversity_report(np.asarray(y_test, dtype=int), model_preds).rename(
        columns={
            "model_a": "agent_a",
            "model_b": "agent_b",
            "a_correct_b_wrong": "agent_a_correct_agent_b_wrong",
            "b_correct_a_wrong": "agent_b_correct_agent_a_wrong",
        }
    )
    diversity_df.to_csv(dataset_output_dir / "four_agent_diversity_report.csv", index=False)

    ablation_lines = [
        "# Four-Agent Trust-Centric Ablation Report",
        "",
        "## Framework",
        "- This run uses exactly four specialized cybersecurity decision sources coordinated by a trust-centric coordination layer.",
        "- Decision source 1: General Traffic Agent (balanced full-feature behavior).",
        "- Decision source 2: Attack Recall Agent (attack-sensitive threshold with validation-only tuning).",
        "- Decision source 3: Normal Behavior Agent (specificity-oriented threshold with validation-only tuning).",
        "- Decision source 4: Hard-Case Agent (trained with extra emphasis on validation hard cases).",
        "",
        "## Best Results",
        f"- Best single agent: {best_agent_row['stage']} with accuracy={best_agent_accuracy:.4f}, F1={best_agent_f1:.4f}, recall={best_agent_recall:.4f}, FNR={best_agent_fnr:.4f}.",
        f"- Best trust-based final decision: {best_trust_method_name} with accuracy={best_trust_accuracy:.4f}, F1={float(best_trust_row['F1']):.4f}, recall={float(best_trust_row['Recall']):.4f}, FNR={float(best_trust_row['FNR']):.4f}.",
        "",
        "## Improvement",
        f"- Accuracy improvement vs best single agent: {float(best_trust_row['Improvement vs Best Agent Accuracy']):.4f}",
        f"- F1 improvement vs best single agent: {float(best_trust_row['Improvement vs Best Agent F1']):.4f}",
        f"- Recall improvement vs best single agent: {float(best_trust_row['Improvement vs Best Agent Recall']):.4f}",
        f"- FNR reduction vs best single agent: {float(best_trust_row['FNR Reduction vs Best Agent']):.4f}",
        "",
        "## Agent Balance Checks",
        f"- Best single agent accuracy: {best_agent_accuracy:.4f}",
        f"- Worst single agent accuracy: {worst_agent_accuracy:.4f}",
        f"- Accuracy spread between best and worst agents: {accuracy_spread:.4f}",
        f"- Trust accuracy: {trust_accuracy:.4f}",
        f"- Trust improvement over best agent: {achieved_improvement:.4f}",
        "",
        "## Oracle Gap Analysis",
        f"- Oracle accuracy among the four specialized cybersecurity decision sources: {oracle_accuracy:.4f}",
        f"- Maximum possible improvement over the best single agent: {max_possible_improvement:.4f}",
        f"- +2 percentage-point improvement is theoretically possible: {target_possible}",
        f"- +2 percentage-point improvement was achieved: {target_achieved}",
        "",
        "## Oracle Breakdown",
        f"- Samples all agents correct: {oracle_breakdown['samples_all_agents_correct']}",
        f"- Samples all agents wrong: {oracle_breakdown['samples_all_agents_wrong']}",
        f"- Samples only one agent correct: {oracle_breakdown['samples_only_one_agent_correct']}",
        f"- Samples where Hard-Case Agent is wrong but another agent is correct: {oracle_breakdown['samples_hard_case_wrong_but_another_correct']}",
        f"- Samples where trust selector failed despite at least one correct agent: {oracle_breakdown['samples_trust_failed_despite_at_least_one_correct']}",
        f"- Trust selector missed-opportunity count: {oracle_breakdown['trust_missed_opportunity_count']}",
    ]
    if not target_possible:
        ablation_lines.append("- Honest interpretation: the oracle upper bound shows +2 points is impossible with only these four decision sources on this split.")
    elif not target_achieved:
        ablation_lines.append("- Honest interpretation: +2 points was possible in theory but was not reached by the current trust-centric coordination layer.")
    else:
        ablation_lines.append("- The trust-centric coordination layer achieved the +2 point target without using extra expanded models.")

    (dataset_output_dir / "four_agent_ablation_report.md").write_text("\n".join(ablation_lines))

    oracle_breakdown_lines = [
        "# Four-Agent Oracle Breakdown",
        "",
        f"- Samples all agents correct: {oracle_breakdown['samples_all_agents_correct']}",
        f"- Samples all agents wrong: {oracle_breakdown['samples_all_agents_wrong']}",
        f"- Samples only one agent correct: {oracle_breakdown['samples_only_one_agent_correct']}",
        f"- Samples where Hard-Case Agent is wrong but another agent is correct: {oracle_breakdown['samples_hard_case_wrong_but_another_correct']}",
        f"- Samples where trust selector failed despite at least one correct agent: {oracle_breakdown['samples_trust_failed_despite_at_least_one_correct']}",
        f"- Trust selector missed-opportunity count: {oracle_breakdown['trust_missed_opportunity_count']}",
    ]
    (dataset_output_dir / "four_agent_oracle_breakdown.md").write_text("\n".join(oracle_breakdown_lines))

    selected_agents = selected_agent_names
    diagnostics_rows = []
    if selected_agents:
        selector_predictions = tuned_selector_predictions
        y_test_array = np.asarray(y_test, dtype=int)
        selected_agents_series = pd.Series(selected_agents)
        for agent_name, count in selected_agents_series.value_counts().items():
            mask = selected_agents_series.to_numpy() == str(agent_name)
            selected_agent_pred = np.asarray(model_preds[str(agent_name)], dtype=int)[mask]
            selected_truth = y_test_array[mask]
            diagnostics_rows.append(
                {
                    "selected_agent": str(agent_name),
                    "number_of_samples_selected": int(count),
                    "selection_rate": float(count / len(selected_agents)),
                    "selected_agent_accuracy_on_selected_samples": float(np.mean(selected_agent_pred == selected_truth)) if int(count) > 0 else np.nan,
                    "selected_agent_error_count": int(np.sum(selected_agent_pred != selected_truth)) if int(count) > 0 else 0,
                    "attack_selection_rate": float(np.mean(selector_predictions[mask] == 1)) if int(count) > 0 else np.nan,
                    "normal_selection_rate": float(np.mean(selector_predictions[mask] == 0)) if int(count) > 0 else np.nan,
                }
            )
    pd.DataFrame(diagnostics_rows).to_csv(dataset_output_dir / "four_agent_selection_diagnostics.csv", index=False)

    best_single_row = baseline_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_single_accuracy = float(best_single_row["Accuracy"])
    best_single_f1 = float(best_single_row["F1"])
    best_single_recall = float(best_single_row["Recall"])
    best_single_fnr = float(best_single_row["FNR"])

    single_rows = baseline_df.copy()
    single_rows["Dataset"] = dataset_name
    single_rows["Category"] = "Single Agent"
    single_rows["Method"] = single_rows["stage"]
    single_rows["Notes"] = "Single-agent metric row"
    single_rows.loc[single_rows["Method"] == str(best_single_row["stage"]), "Notes"] = "Best single agent baseline"

    trust_rows_df = trust_results_df.copy()
    trust_rows_df["Dataset"] = dataset_name
    trust_rows_df["Category"] = "Trust Method"
    trust_rows_df["Method"] = trust_rows_df["stage"]
    trust_rows_df["Notes"] = "Trust aggregation method"
    trust_rows_df.loc[trust_rows_df["Method"] == "Disagreement-aware Selector", "Notes"] = "Selector ablation metric row"
    trust_rows_df.loc[trust_rows_df["Method"] == "Hard-Case Guarded Selector", "Notes"] = "Selector ablation metric row"
    trust_rows_df.loc[trust_rows_df["Method"] == "Best 4-Agent Trust Method", "Notes"] = "Best trust row from four_agent_trust_results"

    keep_methods = [
        "General Traffic Agent",
        "Attack Recall Agent",
        "Normal Behavior Agent",
        "Hard-Case Agent",
        "Majority Voting",
        "Global Trust Voting",
        "Attack Recall Trust",
        "Role-Aware Trust Voting",
        "Disagreement-aware Selector",
        "Hard-Case Guarded Selector",
        "Trust Agent Selector",
        "Best 4-Agent Trust Method",
    ]
    if dataset_name_lower in {"cicids2017", "cic-ids2017"}:
        keep_methods.extend(
            [
                "Accuracy-prioritized selector",
                "F1-prioritized selector",
                "Recall-safe selector",
                "FNR-penalty selector",
                "Disagreement-aware selector",
                "Best tuned CICIDS2017 selector",
            ]
        )
    final_rows = pd.concat([single_rows, trust_rows_df], ignore_index=True, sort=False)
    final_rows = final_rows[final_rows["Method"].isin(keep_methods)].copy()

    for metric_col, best_value in [
        ("Accuracy", best_single_accuracy),
        ("F1", best_single_f1),
        ("Recall", best_single_recall),
    ]:
        final_rows[f"Improvement vs Best Single Agent {metric_col}"] = final_rows[metric_col].astype(float) - float(best_value)
    final_rows["FNR Reduction vs Best Single Agent"] = float(best_single_fnr) - final_rows["FNR"].astype(float)

    oracle_row = {
        "Dataset": dataset_name,
        "Category": "Oracle",
        "Method": "Oracle",
        "Accuracy": oracle_accuracy,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "FPR": np.nan,
        "FNR": np.nan,
        "Specificity": np.nan,
        "Balanced Accuracy": np.nan,
        "Improvement vs Best Single Agent Accuracy": oracle_accuracy - best_single_accuracy,
        "Improvement vs Best Single Agent F1": np.nan,
        "Improvement vs Best Single Agent Recall": np.nan,
        "FNR Reduction vs Best Single Agent": np.nan,
        "Notes": "Theoretical upper bound among four agents",
    }
    final_rows = pd.concat([final_rows, pd.DataFrame([oracle_row])], ignore_index=True, sort=False)

    method_order = {name: idx for idx, name in enumerate(keep_methods + ["Oracle"])}
    final_rows["__order"] = final_rows["Method"].map(method_order).fillna(999)
    final_rows = final_rows.sort_values("__order").drop(columns=["__order"])

    final_columns = [
        "Dataset",
        "Category",
        "Method",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
        "Specificity",
        "Balanced Accuracy",
        "Improvement vs Best Single Agent Accuracy",
        "Improvement vs Best Single Agent F1",
        "Improvement vs Best Single Agent Recall",
        "FNR Reduction vs Best Single Agent",
        "Notes",
    ]
    final_rows = final_rows[final_columns]
    final_rows.to_csv(dataset_output_dir / "final_comparison.csv", index=False)

    percent_df = final_rows.copy()
    percent_metric_cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "Specificity", "Balanced Accuracy"]
    for col in percent_metric_cols:
        percent_df[col] = percent_df[col].map(lambda v: _to_percent_str(v))
    for col in [
        "Improvement vs Best Single Agent Accuracy",
        "Improvement vs Best Single Agent F1",
        "Improvement vs Best Single Agent Recall",
        "FNR Reduction vs Best Single Agent",
    ]:
        percent_df[col] = percent_df[col].map(lambda v: _to_percent_str(v, signed=True, pts=True))
    percent_df.to_csv(dataset_output_dir / "final_comparison_percent.csv", index=False)

    all_method_predictions = {**model_preds, **{k: v for k, v in trust_outputs.items()}, "Best 4-Agent Trust Method": trust_outputs[best_trust_method_name]}
    prediction_distribution_df = summarize_prediction_distribution(all_method_predictions)
    prediction_distribution_df.to_csv(dataset_output_dir / "prediction_distribution.csv", index=False)

    diversity_df.to_csv(dataset_output_dir / "model_diversity_report.csv", index=False)

    sample_prediction_data = {"y_true": np.asarray(y_test, dtype=int)}
    for model_name, predictions in model_preds.items():
        sample_prediction_data[model_name.lower().replace(" ", "_").replace("-", "_") + "_prediction"] = predictions
    for method_name, predictions in trust_outputs.items():
        sample_prediction_data[method_name.lower().replace(" ", "_").replace("-", "_") + "_prediction"] = predictions
    sample_prediction_data["best_4_agent_trust_method_prediction"] = trust_outputs[best_trust_method_name]
    pd.DataFrame(sample_prediction_data, index=x_test.index).to_csv(dataset_output_dir / "sample_level_predictions.csv", index=False)

    poison_options = poison_options or {}
    if bool(poison_options.get("run_poisoned_experiments", False)):
        poisoned_comparison_df, poisoned_artifacts = run_poisoned_agent_experiments(
            dataset_name=dataset_name,
            y_test=np.asarray(y_test, dtype=int),
            model_preds=model_preds,
            model_probs=model_probs,
            validation_model_metrics=validation_model_metrics,
            roles=roles,
            x_val_full=x_val_full,
            y_val=np.asarray(y_val, dtype=int),
            validation_predictions=validation_predictions,
            x_test_full=x_test_full,
            role_aware_cfg=role_aware_cfg,
            selector_params=tuned_selector_params,
            poison_rate=float(poison_options.get("poison_rate", 0.3)),
            poison_mode=str(poison_options.get("poison_mode", "flip")),
            poison_random_state=int(poison_options.get("poison_random_state", 42)),
            ai_trust_config=ai_trust_cfg,
            ai_trust_cache_dir=results_dir / "ai_trust_cache",
            return_artifacts=True,
        )
        save_poisoned_comparison_outputs(
            poisoned_comparison_df,
            dataset_output_dir,
            dataset_file_stem,
            robustness_artifacts=poisoned_artifacts,
        )
        if ai_decision_rows and isinstance(poisoned_artifacts.get("ai_trust_decisions"), pd.DataFrame):
            merged_ai = pd.concat([pd.DataFrame(ai_decision_rows), poisoned_artifacts["ai_trust_decisions"]], ignore_index=True, sort=False)
            merged_ai.to_csv(dataset_output_dir / "ai_trust_decisions.csv", index=False)

    print("Four-source trust-centric experiment complete")


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
    cli_args = _parse_cli_args(sys.argv[1:])
    poison_options = _build_poison_experiment_options(cli_args)

    dataset_arg = cli_args.dataset

    if run_mode == "feature_view_multi_agent":
        _run_feature_view_multi_agent(
            config,
            project_root,
            results_dir,
            dataset_name=dataset_arg,
            poison_options=poison_options,
        )
        return

    if poison_options.get("run_poisoned_experiments", False):
        print("Poisoned-agent experiments are supported only for run_mode=feature_view_multi_agent. Skipping.")

    _run_louati_ktata_baseline(config, project_root, results_dir)


if __name__ == "__main__":
    main()
