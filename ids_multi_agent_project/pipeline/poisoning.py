"""Utilities for optional poisoned-agent experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from pipeline import trust_methods
from pipeline.ai_trust_auditor import AITrustConfig, select_method_with_ai
from pipeline.evaluation import evaluate_predictions

REQUIRED_POISON_COMPARISON_COLUMNS = [
    "Poisoned Agent",
    "Evaluation Type",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FPR",
    "FNR",
    "TPR",
    "TNR",
    "Specificity",
    "Balanced Accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
]

FULL_POISON_OUTPUT_COLUMNS = [
    "Dataset",
    "Poison Mode",
    "Poison Rate",
    *REQUIRED_POISON_COMPARISON_COLUMNS,
    "ROC AUC",
    "PR AUC",
    "Accuracy Recovery",
    "F1 Recovery",
    "FNR Reduction",
]

ROBUSTNESS_COLUMNS = [
    "Trust Method",
    "Scenario",
    "Poisoned Agent",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "FPR",
    "FNR",
    "TPR",
    "TNR",
    "Specificity",
    "Balanced Accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
    "Accuracy Drop",
    "F1 Drop",
    "FNR Increase",
    "FPR Increase",
]


def _slugify(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def poison_predictions(
    y_pred,
    poison_rate: float = 0.3,
    mode: str = "flip",
    random_state: int = 42,
    y_true=None,
):
    """Return a poisoned copy of binary predictions."""
    _ = y_true  # reserved for future poisoning policies
    if not 0.0 <= float(poison_rate) <= 1.0:
        raise ValueError("poison_rate must be between 0.0 and 1.0.")
    valid_modes = {"flip", "normal_bias", "attack_bias"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported poison mode '{mode}'. Expected one of {sorted(valid_modes)}.")

    original = np.asarray(y_pred, dtype=int)
    poisoned = original.copy()
    n_samples = int(poisoned.shape[0])
    n_poison = int(n_samples * float(poison_rate))
    if n_poison <= 0 or n_samples == 0:
        return poisoned

    rng = np.random.default_rng(int(random_state))
    poison_idx = rng.choice(n_samples, size=n_poison, replace=False)

    if mode == "flip":
        poisoned[poison_idx] = 1 - poisoned[poison_idx]
    elif mode == "normal_bias":
        poisoned[poison_idx] = 0
    elif mode == "attack_bias":
        poisoned[poison_idx] = 1
    return poisoned


def build_poisoned_agent_prediction_sets(
    model_preds: Dict[str, np.ndarray],
    poison_rate: float,
    poison_mode: str,
    poison_random_state: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create one poisoned prediction dict per agent (one poisoned agent at a time)."""
    poisoned_sets: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, poisoned_agent in enumerate(model_preds.keys()):
        cloned = {name: np.asarray(pred, dtype=int).copy() for name, pred in model_preds.items()}
        cloned[poisoned_agent] = poison_predictions(
            cloned[poisoned_agent],
            poison_rate=poison_rate,
            mode=poison_mode,
            random_state=int(poison_random_state) + idx,
        )
        poisoned_sets[poisoned_agent] = cloned
    return poisoned_sets


def _to_metric_row(
    *,
    dataset_name: str,
    poisoned_agent: str,
    evaluation_type: str,
    poison_mode: str,
    poison_rate: float,
    metrics: Dict[str, Any],
    accuracy_recovery: float | None = None,
    f1_recovery: float | None = None,
    fnr_reduction: float | None = None,
) -> Dict[str, Any]:
    return {
        "Dataset": dataset_name,
        "Poison Mode": poison_mode,
        "Poison Rate": float(poison_rate),
        "Poisoned Agent": poisoned_agent,
        "Evaluation Type": evaluation_type,
        "Accuracy": float(metrics.get("test_accuracy", np.nan)),
        "Precision": float(metrics.get("test_precision", np.nan)),
        "Recall": float(metrics.get("test_recall", np.nan)),
        "F1": float(metrics.get("test_f1", np.nan)),
        "FPR": float(metrics.get("fpr", np.nan)),
        "FNR": float(metrics.get("fnr", np.nan)),
        "TPR": float(metrics.get("tpr", np.nan)),
        "TNR": float(metrics.get("tnr", np.nan)),
        "Specificity": float(metrics.get("specificity", np.nan)),
        "Balanced Accuracy": float(metrics.get("balanced_accuracy", np.nan)),
        "TP": int(metrics.get("tp", 0)),
        "TN": int(metrics.get("tn", 0)),
        "FP": int(metrics.get("fp", 0)),
        "FN": int(metrics.get("fn", 0)),
        "ROC AUC": float(metrics.get("roc_auc", np.nan)) if metrics.get("roc_auc", np.nan) is not None else np.nan,
        "PR AUC": float(metrics.get("pr_auc", np.nan)) if metrics.get("pr_auc", np.nan) is not None else np.nan,
        "Accuracy Recovery": accuracy_recovery if accuracy_recovery is not None else np.nan,
        "F1 Recovery": f1_recovery if f1_recovery is not None else np.nan,
        "FNR Reduction": fnr_reduction if fnr_reduction is not None else np.nan,
    }


def _to_robustness_row(
    *,
    trust_method: str,
    scenario: str,
    poisoned_agent: str,
    metrics: Dict[str, Any],
    accuracy_drop: float,
    f1_drop: float,
    fnr_increase: float,
    fpr_increase: float,
) -> Dict[str, Any]:
    return {
        "Trust Method": trust_method,
        "Scenario": scenario,
        "Poisoned Agent": poisoned_agent,
        "Accuracy": float(metrics.get("test_accuracy", np.nan)),
        "Precision": float(metrics.get("test_precision", np.nan)),
        "Recall": float(metrics.get("test_recall", np.nan)),
        "F1": float(metrics.get("test_f1", np.nan)),
        "FPR": float(metrics.get("fpr", np.nan)),
        "FNR": float(metrics.get("fnr", np.nan)),
        "TPR": float(metrics.get("tpr", np.nan)),
        "TNR": float(metrics.get("tnr", np.nan)),
        "Specificity": float(metrics.get("specificity", np.nan)),
        "Balanced Accuracy": float(metrics.get("balanced_accuracy", np.nan)),
        "TP": int(metrics.get("tp", 0)),
        "TN": int(metrics.get("tn", 0)),
        "FP": int(metrics.get("fp", 0)),
        "FN": int(metrics.get("fn", 0)),
        "Accuracy Drop": float(accuracy_drop),
        "F1 Drop": float(f1_drop),
        "FNR Increase": float(fnr_increase),
        "FPR Increase": float(fpr_increase),
    }


def _build_trust_method_registry(
    *,
    model_probs: Dict[str, np.ndarray],
    validation_model_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    x_val_full: np.ndarray,
    y_val: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    x_test_full: np.ndarray,
    role_aware_cfg: Dict[str, Any],
    selector_params: Dict[str, Any],
) -> List[Tuple[str, Any]]:
    return [
        ("Majority Vote", lambda preds: trust_methods.majority_voting(preds)),
        ("Global Trust Voting", lambda preds: trust_methods.accuracy_based_trust(preds, validation_model_metrics)),
        ("F1 Trust Voting", lambda preds: trust_methods.f1_based_trust(preds, validation_model_metrics)),
        ("Error-Aware Trust", lambda preds: trust_methods.error_aware_trust(preds, validation_model_metrics)),
        ("Attack Recall Trust", lambda preds: trust_methods.attack_recall_trust(preds, validation_model_metrics)),
        ("FNR Penalty Trust", lambda preds: trust_methods.fnr_penalty_trust(preds, validation_model_metrics)),
        ("Best Safe Model Selector", lambda preds: trust_methods.best_safe_model_selector(preds, validation_model_metrics)),
        ("Best Accuracy Selector", lambda preds: trust_methods.best_accuracy_selector(preds, validation_model_metrics)),
        ("Class-Specific Trust", lambda preds: trust_methods.class_specific_trust(preds, validation_model_metrics)),
        ("Dynamic Trust", lambda preds: trust_methods.dynamic_trust(preds, validation_model_metrics)),
        ("Confidence-Based Trust", lambda preds: trust_methods.confidence_based_trust(preds, model_probs, validation_model_metrics)),
        ("Hybrid Trust", lambda preds: trust_methods.hybrid_trust(preds, model_probs, validation_model_metrics)),
        ("Attack Override Trust", lambda preds: trust_methods.attack_override_trust(preds, model_probs, validation_model_metrics)),
        (
            "Role-Aware Trust Voting",
            lambda preds: trust_methods.role_aware_trust_voting(
                preds,
                model_probs,
                validation_model_metrics,
                roles,
                attack_threshold=float(role_aware_cfg.get("attack_threshold", 0.60)),
                normal_threshold=float(role_aware_cfg.get("normal_threshold", 0.65)),
            ),
        ),
        (
            "Trust Agent Selector",
            lambda preds: trust_methods.trust_agent_selector(
                preds,
                model_probs,
                validation_model_metrics,
                roles,
                x_val_full,
                np.asarray(y_val, dtype=int),
                validation_predictions,
                x_test_full,
                **selector_params,
            ),
        ),
        (
            "Confidence-Margin Trust",
            lambda preds: trust_methods.confidence_margin_trust(preds, model_probs, validation_model_metrics),
        ),
        (
            "Local Accuracy Trust",
            lambda preds: trust_methods.local_accuracy_trust(
                x_val_full,
                np.asarray(y_val, dtype=int),
                validation_predictions,
                x_test_full,
                preds,
            ),
        ),
    ]


def get_available_poison_trust_method_names(
    *,
    model_probs: Dict[str, np.ndarray],
    validation_model_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    x_val_full: np.ndarray,
    y_val: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    x_test_full: np.ndarray,
    role_aware_cfg: Dict[str, Any],
    selector_params: Dict[str, Any],
) -> List[str]:
    registry = _build_trust_method_registry(
        model_probs=model_probs,
        validation_model_metrics=validation_model_metrics,
        roles=roles,
        x_val_full=x_val_full,
        y_val=y_val,
        validation_predictions=validation_predictions,
        x_test_full=x_test_full,
        role_aware_cfg=role_aware_cfg,
        selector_params=selector_params,
    )
    return [name for name, _ in registry]


def _run_all_trust_methods(
    *,
    predictions: Dict[str, np.ndarray],
    registry: List[Tuple[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    outputs: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    for method_name, method_fn in registry:
        try:
            output = method_fn(predictions)
            y_pred = np.asarray(output.get("predictions", []), dtype=int)
            if y_pred.size == 0:
                skipped.append(method_name)
                continue
            outputs[method_name] = {"predictions": y_pred, "meta": output.get("meta", {})}
        except Exception:
            skipped.append(method_name)
    return outputs, skipped


def _build_agent_summary_markdown(poisoned_agent: str, robustness_df: pd.DataFrame) -> str:
    lines = [f"# {poisoned_agent} Robustness Summary", ""]
    poisoned_rows = robustness_df[robustness_df["Scenario"] == "Poisoned"].copy()
    if poisoned_rows.empty:
        lines.append("No poisoned robustness rows available.")
        return "\n".join(lines) + "\n"
    top = poisoned_rows.sort_values(["Accuracy Drop", "FNR Increase"], ascending=[True, True]).iloc[0]
    lines.append(f"- Most resilient method: {top['Trust Method']}")
    lines.append(f"- Accuracy drop: {float(top['Accuracy Drop']):.4f}")
    lines.append(f"- F1 drop: {float(top['F1 Drop']):.4f}")
    lines.append(f"- FNR increase: {float(top['FNR Increase']):.4f}")
    lines.append(f"- FPR increase: {float(top['FPR Increase']):.4f}")
    return "\n".join(lines) + "\n"


def _build_overall_summary_markdown(dataset_name: str, overall_df: pd.DataFrame) -> str:
    poisoned = overall_df[overall_df["Scenario"] == "Poisoned"].copy()
    lines = [f"# Overall Robustness Summary - {dataset_name}", ""]
    if poisoned.empty:
        lines.append("No poisoned rows available.")
        return "\n".join(lines) + "\n"

    grouped = poisoned.groupby("Trust Method", as_index=False).agg(
        avg_accuracy_drop=("Accuracy Drop", "mean"),
        avg_f1_drop=("F1 Drop", "mean"),
        avg_fnr_increase=("FNR Increase", "mean"),
        avg_fpr_increase=("FPR Increase", "mean"),
    )
    grouped["robustness_score"] = (
        grouped["avg_accuracy_drop"] + grouped["avg_f1_drop"] + grouped["avg_fnr_increase"] + grouped["avg_fpr_increase"]
    )
    best = grouped.sort_values(["robustness_score", "avg_fnr_increase"], ascending=[True, True]).iloc[0]
    lowest_fnr = grouped.sort_values("avg_fnr_increase", ascending=True).iloc[0]
    lowest_acc_drop = grouped.sort_values("avg_accuracy_drop", ascending=True).iloc[0]
    lines.append(f"- Best average robustness: {best['Trust Method']}")
    lines.append(f"- Most resilient trust method: {best['Trust Method']}")
    lines.append(f"- Lowest average FNR increase: {lowest_fnr['Trust Method']}")
    lines.append(f"- Smallest average accuracy degradation: {lowest_acc_drop['Trust Method']}")
    lines.append("")
    lines.append("## Method Averages")
    for _, row in grouped.sort_values("robustness_score", ascending=True).iterrows():
        lines.append(
            f"- {row['Trust Method']}: "
            f"acc_drop={float(row['avg_accuracy_drop']):.4f}, "
            f"f1_drop={float(row['avg_f1_drop']):.4f}, "
            f"fnr_inc={float(row['avg_fnr_increase']):.4f}, "
            f"fpr_inc={float(row['avg_fpr_increase']):.4f}"
        )
    return "\n".join(lines) + "\n"


def run_poisoned_agent_experiments(
    *,
    dataset_name: str,
    y_test: np.ndarray,
    model_preds: Dict[str, np.ndarray],
    model_probs: Dict[str, np.ndarray],
    validation_model_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    x_val_full: np.ndarray,
    y_val: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    x_test_full: np.ndarray,
    role_aware_cfg: Dict[str, Any],
    selector_params: Dict[str, Any],
    poison_rate: float = 0.3,
    poison_mode: str = "flip",
    poison_random_state: int = 42,
    ai_trust_config: AITrustConfig | None = None,
    ai_trust_cache_dir: str | Path | None = None,
    return_artifacts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, Any]]:
    """Run one-poisoned-agent-at-a-time scientific comparison table."""
    print("[POISON_COMPARISON_START] " f"dataset={dataset_name} mode={poison_mode} rate={poison_rate:.4f}")
    print("[ROBUSTNESS_COMPARISON_START] " f"dataset={dataset_name} mode={poison_mode} rate={poison_rate:.4f}")
    y_true = np.asarray(y_test, dtype=int)

    poisoned_sets = build_poisoned_agent_prediction_sets(
        model_preds=model_preds,
        poison_rate=poison_rate,
        poison_mode=poison_mode,
        poison_random_state=poison_random_state,
    )
    trust_registry = _build_trust_method_registry(
        model_probs=model_probs,
        validation_model_metrics=validation_model_metrics,
        roles=roles,
        x_val_full=x_val_full,
        y_val=y_val,
        validation_predictions=validation_predictions,
        x_test_full=x_test_full,
        role_aware_cfg=role_aware_cfg,
        selector_params=selector_params,
    )

    clean_method_outputs, clean_skipped = _run_all_trust_methods(predictions=model_preds, registry=trust_registry)
    for skipped_name in clean_skipped:
        print(
            "[ROBUSTNESS_METHOD_BASELINE] "
            f"dataset={dataset_name} trust_method={skipped_name} status=skipped "
            f"mode={poison_mode} rate={poison_rate:.4f}"
        )
    clean_method_metrics: Dict[str, Dict[str, Any]] = {}
    ai_decisions: List[Dict[str, Any]] = []
    for method_name, payload in clean_method_outputs.items():
        clean_metrics = evaluate_predictions(y_true, np.asarray(payload["predictions"], dtype=int))
        clean_method_metrics[method_name] = clean_metrics
        print(
            "[ROBUSTNESS_METHOD_BASELINE] "
            f"dataset={dataset_name} trust_method={method_name} status=ok "
            f"mode={poison_mode} rate={poison_rate:.4f}"
        )

    rows: List[Dict[str, Any]] = []
    robustness_by_agent: Dict[str, pd.DataFrame] = {}
    ai_cfg = ai_trust_config or AITrustConfig(enabled=False)
    cache_root = Path(ai_trust_cache_dir) if ai_trust_cache_dir is not None else Path("results/ai_trust_cache")
    cache_file = cache_root / f"{_slugify(dataset_name)}_ai_trust_cache.jsonl"
    for poisoned_agent, poisoned_preds in poisoned_sets.items():
        original_agent_pred = np.asarray(model_preds[poisoned_agent], dtype=int)
        poisoned_agent_pred = np.asarray(poisoned_preds[poisoned_agent], dtype=int)

        original_metrics = evaluate_predictions(
            y_true,
            original_agent_pred,
            y_prob=model_probs.get(poisoned_agent),
        )
        print(
            "[POISON_AGENT_BASELINE] "
            f"dataset={dataset_name} agent={poisoned_agent} mode={poison_mode} rate={poison_rate:.4f}"
        )
        rows.append(
            _to_metric_row(
                dataset_name=dataset_name,
                poisoned_agent=poisoned_agent,
                evaluation_type="Original Agent",
                poison_mode=poison_mode,
                poison_rate=poison_rate,
                metrics=original_metrics,
            )
        )

        poisoned_metrics = evaluate_predictions(y_true, poisoned_agent_pred)
        print(
            "[POISON_AGENT_POISONED] "
            f"dataset={dataset_name} agent={poisoned_agent} mode={poison_mode} rate={poison_rate:.4f}"
        )
        rows.append(
            _to_metric_row(
                dataset_name=dataset_name,
                poisoned_agent=poisoned_agent,
                evaluation_type="Poisoned Agent",
                poison_mode=poison_mode,
                poison_rate=poison_rate,
                metrics=poisoned_metrics,
            )
        )

        trust_outputs, skipped_poisoned = _run_all_trust_methods(predictions=poisoned_preds, registry=trust_registry)
        for skipped_name in skipped_poisoned:
            print(
                "[ROBUSTNESS_METHOD_POISONED] "
                f"dataset={dataset_name} poisoned_agent={poisoned_agent} trust_method={skipped_name} "
                f"status=skipped mode={poison_mode} rate={poison_rate:.4f}"
            )
        if "Majority Vote" not in trust_outputs:
            majority_pred = np.asarray(trust_methods.majority_voting(poisoned_preds)["predictions"], dtype=int)
            trust_outputs["Majority Vote"] = {"predictions": majority_pred, "meta": {}}

        majority_metrics = evaluate_predictions(y_true, np.asarray(trust_outputs["Majority Vote"]["predictions"], dtype=int))
        rows.append(
            _to_metric_row(
                dataset_name=dataset_name,
                poisoned_agent=poisoned_agent,
                evaluation_type="Majority Vote",
                poison_mode=poison_mode,
                poison_rate=poison_rate,
                metrics=majority_metrics,
            )
        )

        robustness_rows: List[Dict[str, Any]] = []
        for method_name in clean_method_metrics:
            clean_metrics = clean_method_metrics[method_name]
            robustness_rows.append(
                _to_robustness_row(
                    trust_method=method_name,
                    scenario="Clean",
                    poisoned_agent="None",
                    metrics=clean_metrics,
                    accuracy_drop=0.0,
                    f1_drop=0.0,
                    fnr_increase=0.0,
                    fpr_increase=0.0,
                )
            )

            poisoned_output = trust_outputs.get(method_name)
            if poisoned_output is None:
                print(
                    "[ROBUSTNESS_METHOD_POISONED] "
                    f"dataset={dataset_name} poisoned_agent={poisoned_agent} trust_method={method_name} "
                    f"status=skipped mode={poison_mode} rate={poison_rate:.4f}"
                )
                continue

            method_metrics = evaluate_predictions(y_true, np.asarray(poisoned_output["predictions"], dtype=int))
            print(
                "[POISON_METHOD_EVAL] "
                f"dataset={dataset_name} agent={poisoned_agent} trust_method={method_name} "
                f"mode={poison_mode} rate={poison_rate:.4f}"
            )
            print(
                "[ROBUSTNESS_METHOD_POISONED] "
                f"dataset={dataset_name} poisoned_agent={poisoned_agent} trust_method={method_name} "
                f"status=ok mode={poison_mode} rate={poison_rate:.4f}"
            )

            rows.append(
                _to_metric_row(
                    dataset_name=dataset_name,
                    poisoned_agent=poisoned_agent,
                    evaluation_type=method_name,
                    poison_mode=poison_mode,
                    poison_rate=poison_rate,
                    metrics=method_metrics,
                    accuracy_recovery=float(method_metrics["test_accuracy"] - majority_metrics["test_accuracy"]),
                    f1_recovery=float(method_metrics["test_f1"] - majority_metrics["test_f1"]),
                    fnr_reduction=float(majority_metrics["fnr"] - method_metrics["fnr"]),
                )
            )

            robustness_rows.append(
                _to_robustness_row(
                    trust_method=method_name,
                    scenario="Poisoned",
                    poisoned_agent=poisoned_agent,
                    metrics=method_metrics,
                    accuracy_drop=float(clean_metrics["test_accuracy"] - method_metrics["test_accuracy"]),
                    f1_drop=float(clean_metrics["test_f1"] - method_metrics["test_f1"]),
                    fnr_increase=float(method_metrics["fnr"] - clean_metrics["fnr"]),
                    fpr_increase=float(method_metrics["fpr"] - clean_metrics["fpr"]),
                )
            )

        if ai_cfg.enabled:
            method_metrics_for_ai: Dict[str, Dict[str, float]] = {}
            method_predictions_for_ai: Dict[str, int] = {}
            for method_name in clean_method_metrics:
                poisoned_output = trust_outputs.get(method_name)
                if poisoned_output is None:
                    continue
                method_metrics = evaluate_predictions(y_true, np.asarray(poisoned_output["predictions"], dtype=int))
                method_metrics_for_ai[method_name] = {
                    "Accuracy": float(method_metrics["test_accuracy"]),
                    "Precision": float(method_metrics["test_precision"]),
                    "Recall": float(method_metrics["test_recall"]),
                    "F1": float(method_metrics["test_f1"]),
                    "FPR": float(method_metrics["fpr"]),
                    "FNR": float(method_metrics["fnr"]),
                    "Balanced Accuracy": float(method_metrics["balanced_accuracy"]),
                }
                method_predictions_for_ai[method_name] = int(np.asarray(poisoned_output["predictions"], dtype=int)[0])

            allowed_methods = [m for m in clean_method_metrics.keys() if m in method_metrics_for_ai]
            ai_decision = select_method_with_ai(
                dataset=dataset_name,
                scenario="Poisoned",
                poisoned_agent=poisoned_agent,
                poison_mode=poison_mode,
                poison_rate=float(poison_rate),
                allowed_methods=allowed_methods,
                method_metrics=method_metrics_for_ai,
                method_predictions=method_predictions_for_ai,
                agreement_level=float(np.mean(list(method_predictions_for_ai.values()))) if method_predictions_for_ai else None,
                suspected_poisoned_agent=poisoned_agent,
                config=ai_cfg,
                cache_file=cache_file,
            )
            ai_decisions.append(
                {
                    "Dataset": dataset_name,
                    "Scenario": "Poisoned",
                    "Poisoned Agent": poisoned_agent,
                    "Poison Mode": poison_mode,
                    "Poison Rate": float(poison_rate),
                    "Selected Method": ai_decision.get("selected_method", ""),
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
            if ai_decision.get("selected_method") in trust_outputs:
                selected_name = str(ai_decision["selected_method"])
                ai_pred = np.asarray(trust_outputs[selected_name]["predictions"], dtype=int)
                ai_metrics = evaluate_predictions(y_true, ai_pred)
            rows.append(
                _to_metric_row(
                    dataset_name=dataset_name,
                    poisoned_agent=poisoned_agent,
                    evaluation_type="AI Trust Auditor",
                    poison_mode=poison_mode,
                    poison_rate=poison_rate,
                    metrics=ai_metrics,
                    accuracy_recovery=float(ai_metrics["test_accuracy"] - majority_metrics["test_accuracy"]),
                    f1_recovery=float(ai_metrics["test_f1"] - majority_metrics["test_f1"]),
                    fnr_reduction=float(majority_metrics["fnr"] - ai_metrics["fnr"]),
                )
            )
            clean_for_selected = clean_method_metrics.get(selected_name, ai_metrics)
            robustness_rows.append(
                _to_robustness_row(
                    trust_method="AI Trust Auditor",
                    scenario="Clean",
                    poisoned_agent="None",
                    metrics=clean_for_selected,
                    accuracy_drop=0.0,
                    f1_drop=0.0,
                    fnr_increase=0.0,
                    fpr_increase=0.0,
                )
            )
            robustness_rows.append(
                _to_robustness_row(
                    trust_method="AI Trust Auditor",
                    scenario="Poisoned",
                    poisoned_agent=poisoned_agent,
                    metrics=ai_metrics,
                    accuracy_drop=float(clean_for_selected["test_accuracy"] - ai_metrics["test_accuracy"]),
                    f1_drop=float(clean_for_selected["test_f1"] - ai_metrics["test_f1"]),
                    fnr_increase=float(ai_metrics["fnr"] - clean_for_selected["fnr"]),
                    fpr_increase=float(ai_metrics["fpr"] - clean_for_selected["fpr"]),
                )
            )

        robustness_df = pd.DataFrame(robustness_rows)
        for col in ROBUSTNESS_COLUMNS:
            if col not in robustness_df.columns:
                robustness_df[col] = np.nan
        robustness_by_agent[poisoned_agent] = robustness_df[ROBUSTNESS_COLUMNS].copy()

    table = pd.DataFrame(rows)
    for col in FULL_POISON_OUTPUT_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[FULL_POISON_OUTPUT_COLUMNS].copy()
    overall_robustness_df = (
        pd.concat(list(robustness_by_agent.values()), ignore_index=True, sort=False)
        if robustness_by_agent
        else pd.DataFrame(columns=ROBUSTNESS_COLUMNS)
    )

    print("[POISON_COMPARISON_DONE] " f"dataset={dataset_name} rows={len(table)}")
    print("[ROBUSTNESS_SUMMARY_DONE] " f"dataset={dataset_name} rows={len(overall_robustness_df)}")

    if not return_artifacts:
        return table

    artifacts = {
        "dataset_name": dataset_name,
        "dataset_slug": _slugify(dataset_name),
        "poison_mode": poison_mode,
        "poison_rate": float(poison_rate),
        "poisoned_agent_names": list(poisoned_sets.keys()),
        "robustness_by_agent": robustness_by_agent,
        "overall_robustness_df": overall_robustness_df,
        "ai_trust_decisions": pd.DataFrame(ai_decisions),
    }
    return table, artifacts


def _to_markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in table.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def save_poisoned_comparison_outputs(
    comparison_df: pd.DataFrame,
    output_dir,
    dataset_file_stem: str,
    robustness_artifacts: Dict[str, Any] | None = None,
) -> tuple[str, str]:
    output_dir = Path(output_dir)
    # Keep existing output names for backward compatibility.
    legacy_csv_path = output_dir / f"{dataset_file_stem}_poisoned_agent_comparison.csv"
    legacy_md_path = output_dir / f"{dataset_file_stem}_poisoned_agent_comparison.md"

    full_csv_path = output_dir / f"{dataset_file_stem}_poisoned_full_comparison.csv"
    full_md_path = output_dir / f"{dataset_file_stem}_poisoned_full_comparison.md"
    summary_md_path = output_dir / f"{dataset_file_stem}_poisoned_summary.md"

    comparison_df.to_csv(legacy_csv_path, index=False)
    legacy_md_path.write_text(_to_markdown_table(comparison_df), encoding="utf-8")

    comparison_df.to_csv(full_csv_path, index=False)
    full_md_path.write_text(_to_markdown_table(comparison_df), encoding="utf-8")
    summary_md_path.write_text("# Poisoned-Agent Summary\n\nSee full comparison table.\n", encoding="utf-8")

    if robustness_artifacts:
        dataset_slug = _slugify(dataset_file_stem)
        poisoning_root = output_dir / "poisoning" / dataset_slug
        poisoning_root.mkdir(parents=True, exist_ok=True)
        ai_decisions_df = robustness_artifacts.get("ai_trust_decisions")
        if isinstance(ai_decisions_df, pd.DataFrame) and not ai_decisions_df.empty:
            ai_decisions_df.to_csv(output_dir / "ai_trust_decisions.csv", index=False)

        robustness_by_agent = robustness_artifacts.get("robustness_by_agent", {})
        for poisoned_agent, robustness_df in robustness_by_agent.items():
            agent_slug = _slugify(poisoned_agent)
            agent_dir = poisoning_root / agent_slug
            agent_dir.mkdir(parents=True, exist_ok=True)

            per_agent_full = comparison_df[comparison_df["Poisoned Agent"] == poisoned_agent].copy()
            (agent_dir / "full_comparison.csv").write_text(per_agent_full.to_csv(index=False), encoding="utf-8")
            (agent_dir / "full_comparison.md").write_text(_to_markdown_table(per_agent_full), encoding="utf-8")
            (agent_dir / "robustness_report.csv").write_text(robustness_df.to_csv(index=False), encoding="utf-8")
            (agent_dir / "robustness_report.md").write_text(_to_markdown_table(robustness_df), encoding="utf-8")
            (agent_dir / "summary.md").write_text(
                _build_agent_summary_markdown(poisoned_agent, robustness_df),
                encoding="utf-8",
            )
            print(f"[ROBUSTNESS_AGENT_SAVE] dataset={dataset_slug} poisoned_agent={agent_slug}")

        overall_df = robustness_artifacts.get("overall_robustness_df", pd.DataFrame(columns=ROBUSTNESS_COLUMNS))
        (poisoning_root / "overall_robustness_summary.md").write_text(
            _build_overall_summary_markdown(str(robustness_artifacts.get("dataset_name", dataset_file_stem)), overall_df),
            encoding="utf-8",
        )
        print(f"[ROBUSTNESS_SUMMARY_DONE] dataset={dataset_slug} file=overall_robustness_summary.md")

    return str(full_csv_path), str(full_md_path)
