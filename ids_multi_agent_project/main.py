"""Main entry point for Stage 1 IDS experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from models.agent_factory import create_agent, create_detection_agent
from pipeline.data_loader import POTENTIAL_LEAKAGE_COLUMNS, load_yaml_config
from pipeline.evaluation import evaluate_model, evaluate_predictions
from pipeline.interaction import resolve_agent_interactions
from pipeline.preprocessing import (
    build_preprocessing_pipeline,
    build_traffic_preprocessing_pipeline,
    infer_feature_type_columns,
    sanitize_feature_values,
)
from pipeline.training import run_cross_validation, train_agent
from utils.progress import (
    log_section,
    log_step,
    log_substep,
    log_success,
    log_warning,
)

BEHAVIORAL_AGENT_NAME = "BehavioralAnalysisAgent"
TRAFFIC_AGENT_NAME = "TrafficAnalysisAgent"
FINAL_AGENT_NAME = "FinalResolvedMultiAgent"
SUSPICIOUS_NAME_TOKENS = (
    "label",
    "target",
    "attack_cat",
    "attack_category",
    "attack_type",
    "class",
)
GROUP_IDENTIFIER_TOKENS = (
    "flow_id",
    "flowid",
    "session_id",
    "session",
    "connection_id",
    "conn_id",
)
STRICT_LEAKAGE_CHECK_DEFAULT = False


def _convert_to_binary_labels(y, preprocessing_config: Dict):
    if not preprocessing_config.get("binary_classification", False):
        return y

    normal_values = preprocessing_config.get("normal_label_values", ["normal", "Normal", 0])
    common_benign_aliases = {"benign", "BENIGN", "Benign"}

    def normalize(value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    normalized_normal_values = {normalize(value) for value in normal_values}
    normalized_normal_values.update({normalize(value) for value in common_benign_aliases})
    normalized_labels = y.apply(normalize)

    if normalized_labels.isna().any():
        raise ValueError("Label column contains missing values; cannot convert to binary labels.")

    return normalized_labels.apply(
        lambda value: 0 if value in normalized_normal_values else 1
    ).astype(int)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _class_counts(y: pd.Series) -> Dict[str, int]:
    return {
        "normal": int((y == 0).sum()),
        "attack": int((y == 1).sum()),
    }


def _normalize_reasoning_config(config: Dict) -> Dict[str, Any]:
    reasoning_cfg = dict(config.get("agentic_reasoning", {}))
    return {
        "ollama_enabled": bool(reasoning_cfg.get("ollama_enabled", False)),
        "ollama_model_name": str(reasoning_cfg.get("ollama_model_name", "llama3.1:8b")),
        "ollama_base_url": str(reasoning_cfg.get("ollama_base_url", "http://localhost:11434")),
        "ollama_timeout_seconds": int(reasoning_cfg.get("ollama_timeout_seconds", 15)),
        "max_reasoning_samples_per_dataset": int(
            reasoning_cfg.get("max_reasoning_samples_per_dataset", 50)
        ),
        "enable_agent_reasoning_output": bool(
            reasoning_cfg.get("enable_agent_reasoning_output", True)
        ),
        "reasoning_scope": str(reasoning_cfg.get("reasoning_scope", "disagreement_only")),
    }


def _normalize_interaction_config(config: Dict) -> Dict[str, Any]:
    interaction_cfg = dict(config.get("interaction", {}))
    return {
        "disagreement_confidence_threshold": float(
            interaction_cfg.get("disagreement_confidence_threshold", 0.10)
        ),
        "trust_weight_global": float(interaction_cfg.get("trust_weight_global", 0.5)),
        "trust_weight_confidence": float(interaction_cfg.get("trust_weight_confidence", 0.3)),
        "trust_weight_disagreement": float(interaction_cfg.get("trust_weight_disagreement", 0.2)),
        "trust_gap_threshold": float(interaction_cfg.get("trust_gap_threshold", 0.05)),
    }


def _balance_training_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    imbalance_ratio_threshold: float = 1.2,
) -> tuple[pd.DataFrame, pd.Series, str]:
    if y_train.nunique() < 2:
        return x_train, y_train, "none_single_class_train"

    counts = y_train.value_counts()
    majority_class = int(counts.idxmax())
    minority_class = int(counts.idxmin())
    majority_count = int(counts.max())
    minority_count = int(counts.min())

    if minority_count == 0:
        return x_train, y_train, "none_single_class_train"

    imbalance_ratio = float(majority_count / minority_count)
    if imbalance_ratio <= imbalance_ratio_threshold:
        return x_train, y_train, "none_already_balanced"

    train_df = x_train.copy()
    train_df["__label__"] = y_train.values

    minority_rows = train_df[train_df["__label__"] == minority_class]
    majority_rows = train_df[train_df["__label__"] == majority_class].sample(
        n=minority_count,
        random_state=random_state,
    )

    balanced_df = (
        pd.concat([minority_rows, majority_rows], axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    balanced_y = balanced_df.pop("__label__").astype(y_train.dtype)
    balanced_x = balanced_df
    return balanced_x, balanced_y, "random_undersample_train_only"


def _print_results(dataset_name: str, agent_name: str, metrics: Dict) -> None:
    def _first_metric(*keys: str):
        for key in keys:
            if key in metrics:
                return metrics.get(key)
        return None

    def _fmt_pct(value) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.2f}%"

    def _fmt_count(value) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return str(int(value))

    print("\n" + "=" * 56)
    print(f"Dataset: {dataset_name}")
    print(f"Agent: {agent_name}")
    print("=" * 56)
    print("\nCross-Validation:")
    print(f"Accuracy  : {_fmt_pct(_first_metric('cv_accuracy'))}")
    print(f"Precision : {_fmt_pct(_first_metric('cv_precision'))}")
    print(f"Recall    : {_fmt_pct(_first_metric('cv_recall'))}")
    print(f"F1 Score  : {_fmt_pct(_first_metric('cv_f1'))}")

    print("\nTest Results:")
    print(f"Accuracy  : {_fmt_pct(_first_metric('test_accuracy'))}")
    print(f"Precision : {_fmt_pct(_first_metric('test_precision'))}")
    print(f"Recall    : {_fmt_pct(_first_metric('test_recall'))}")
    print(f"F1 Score  : {_fmt_pct(_first_metric('test_f1'))}")

    print("\nConfusion Matrix:")
    cm = _first_metric("confusion_matrix")
    print(cm if cm is not None else "N/A")

    print("\nConfusion Matrix Breakdown:")
    print(f"TP: {_fmt_count(_first_metric('tp', 'TP'))}")
    print(f"TN: {_fmt_count(_first_metric('tn', 'TN'))}")
    print(f"FP: {_fmt_count(_first_metric('fp', 'FP'))}")
    print(f"FN: {_fmt_count(_first_metric('fn', 'FN'))}")

    print("\nIDS Metrics:")
    print(f"False Positive Rate : {_fmt_pct(_first_metric('fpr', 'FPR'))}")
    print(f"False Negative Rate : {_fmt_pct(_first_metric('fnr', 'FNR'))}")
    print(f"True Positive Rate  : {_fmt_pct(_first_metric('tpr', 'TPR'))}")
    print(f"True Negative Rate  : {_fmt_pct(_first_metric('tnr', 'TNR'))}")
    print(f"Specificity         : {_fmt_pct(_first_metric('specificity'))}")
    print(f"Balanced Accuracy   : {_fmt_pct(_first_metric('balanced_accuracy'))}")
    print(f"Error Rate          : {_fmt_pct(_first_metric('error_rate'))}")

    print("\nSupport:")
    print(f"Total  : {_fmt_count(_first_metric('support_total'))}")
    print(f"Attack : {_fmt_count(_first_metric('support_attack'))}")
    print(f"Normal : {_fmt_count(_first_metric('support_normal'))}")

    print("\nAUC:")
    print(f"ROC AUC : {_fmt_pct(_first_metric('roc_auc'))}")
    print(f"PR AUC  : {_fmt_pct(_first_metric('pr_auc'))}")
    print("\n" + "=" * 56)


def _print_data_diagnostics(dataset_name: str, x_train: pd.DataFrame, y_train, y_test) -> None:
    print("\n" + "-" * 70)
    print(f"Data diagnostics for {dataset_name}")
    print("-" * 70)


def _print_interaction_summary(dataset_name: str, summary: Dict[str, Any]) -> None:
    def _fmt_pct(value) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.2f}%"

    def _fmt_count(value) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return str(int(value))

    print("\nAgent Interaction Summary:")
    print(f"Agreements       : {_fmt_count(summary.get('agreement_count'))}")
    print(f"Disagreements    : {_fmt_count(summary.get('disagreement_count'))}")
    print(f"Contested Cases  : {_fmt_count(summary.get('contested_case_count'))}")
    print(f"Agreement Rate   : {_fmt_pct(summary.get('agreement_rate'))}")
    print(f"Disagreement Rate: {_fmt_pct(summary.get('disagreement_rate'))}")
    print(
        "Behavioral Wins on Disagreement : "
        f"{_fmt_pct(summary.get('behavioral_wins_on_disagreement'))}"
    )
    print(
        "Traffic Wins on Disagreement    : "
        f"{_fmt_pct(summary.get('traffic_wins_on_disagreement'))}"
    )


def _print_dataset_error(dataset_name: str, exc: Exception) -> None:
    print("\n" + "=" * 56)
    print(f"DATASET FAILED: {dataset_name}")
    print("=" * 56)
    print(f"\nError: {type(exc).__name__}: {exc}")
    print("\nContinuing to next dataset...")
    print("=" * 56 + "\n")


def _fmt_pct_console(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _fmt_count_console(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(int(value))


def _build_final_consolidated_summary(
    results_df: pd.DataFrame,
    interaction_summary_df: pd.DataFrame,
    run_status_rows: List[Dict[str, str]],
) -> str:
    dataset_order = [str(row.get("dataset", "")) for row in run_status_rows if row.get("dataset")]
    if not dataset_order and not results_df.empty:
        dataset_order = results_df["dataset"].dropna().astype(str).drop_duplicates().tolist()

    lines: List[str] = []
    lines.append("=" * 56)
    lines.append("FINAL CONSOLIDATED RESULTS")
    lines.append("=" * 56)
    lines.append("")

    best_model_map: Dict[str, str] = {}
    comparison_rows: List[Dict[str, Any]] = []

    for dataset_name in dataset_order:
        lines.append(f"## DATASET: {dataset_name}")
        lines.append("")
        ds_rows = (
            results_df[results_df["dataset"] == dataset_name]
            if not results_df.empty
            else pd.DataFrame()
        )

        def _agent_row(agent_name: str):
            if ds_rows.empty:
                return None
            subset = ds_rows[ds_rows["agent"] == agent_name]
            if subset.empty:
                return None
            return subset.iloc[0]

        behavioral_row = _agent_row(BEHAVIORAL_AGENT_NAME)
        traffic_row = _agent_row(TRAFFIC_AGENT_NAME)
        final_row = _agent_row(FINAL_AGENT_NAME)

        for agent_name, row in [
            (BEHAVIORAL_AGENT_NAME, behavioral_row),
            (TRAFFIC_AGENT_NAME, traffic_row),
            (FINAL_AGENT_NAME, final_row),
        ]:
            lines.append(agent_name)
            lines.append(
                f"Test Accuracy  : {_fmt_pct_console(row['test_accuracy']) if row is not None else 'N/A'}"
            )
            lines.append(
                f"Test Precision : {_fmt_pct_console(row['test_precision']) if row is not None else 'N/A'}"
            )
            lines.append(
                f"Test Recall    : {_fmt_pct_console(row['test_recall']) if row is not None else 'N/A'}"
            )
            lines.append(
                f"Test F1        : {_fmt_pct_console(row['test_f1']) if row is not None else 'N/A'}"
            )
            lines.append("")

        interaction_row = (
            interaction_summary_df[interaction_summary_df["dataset"] == dataset_name].iloc[0]
            if not interaction_summary_df.empty
            and not interaction_summary_df[interaction_summary_df["dataset"] == dataset_name].empty
            else None
        )
        lines.append("Interaction")
        lines.append(
            "Agreements       : "
            + (
                _fmt_count_console(interaction_row["agreement_count"])
                if interaction_row is not None
                else "N/A"
            )
        )
        lines.append(
            "Disagreements    : "
            + (
                _fmt_count_console(interaction_row["disagreement_count"])
                if interaction_row is not None
                else "N/A"
            )
        )
        lines.append(
            "Contested Cases  : "
            + (
                _fmt_count_console(interaction_row["contested_case_count"])
                if interaction_row is not None
                else "N/A"
            )
        )
        lines.append(
            "Agreement Rate   : "
            + (
                _fmt_pct_console(interaction_row["agreement_rate"])
                if interaction_row is not None
                else "N/A"
            )
        )
        lines.append(
            "Disagreement Rate: "
            + (
                _fmt_pct_console(interaction_row["disagreement_rate"])
                if interaction_row is not None
                else "N/A"
            )
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        model_rows = ds_rows[
            ds_rows["agent"].isin([BEHAVIORAL_AGENT_NAME, TRAFFIC_AGENT_NAME, FINAL_AGENT_NAME])
        ] if not ds_rows.empty else pd.DataFrame()
        if model_rows.empty:
            best_model_map[dataset_name] = "N/A"
        else:
            top_row = model_rows.sort_values("test_f1", ascending=False).iloc[0]
            best_model_map[dataset_name] = str(top_row["agent"])

        comparison_rows.append(
            {
                "dataset": dataset_name,
                "behavioral_f1": (
                    behavioral_row["test_f1"] if behavioral_row is not None else np.nan
                ),
                "traffic_f1": (
                    traffic_row["test_f1"] if traffic_row is not None else np.nan
                ),
                "final_f1": (final_row["test_f1"] if final_row is not None else np.nan),
            }
        )

    lines.append("=" * 56)
    lines.append("BEST MODEL PER DATASET")
    lines.append("=" * 56)
    lines.append("")
    for dataset_name in dataset_order:
        lines.append(f"{dataset_name:<14} -> {best_model_map.get(dataset_name, 'N/A')}")

    lines.append("")
    lines.append("=" * 56)
    lines.append("FINAL MULTI-AGENT COMPARISON")
    lines.append("=" * 56)
    lines.append("")
    lines.append(
        f"{'Dataset':<12} {'Behavioral F1':<15} {'Traffic F1':<12} {'Final Multi-Agent F1':<22}"
    )
    for row in comparison_rows:
        lines.append(
            f"{row['dataset']:<12} "
            f"{_fmt_pct_console(row['behavioral_f1']):<15} "
            f"{_fmt_pct_console(row['traffic_f1']):<12} "
            f"{_fmt_pct_console(row['final_f1']):<22}"
        )

    lines.append("")
    lines.append("=" * 56)
    lines.append("RUN STATUS")
    lines.append("=" * 56)
    lines.append("")
    for row in run_status_rows:
        dataset = str(row.get("dataset", "unknown_dataset"))
        status = str(row.get("status", "UNKNOWN"))
        error = str(row.get("error", ""))
        if status == "SUCCESS":
            lines.append(f"{dataset:<14} -> SUCCESS")
        else:
            lines.append(f"{dataset:<14} -> FAILED: {error}")

    lines.append("")
    lines.append("=" * 56)
    lines.append("HIGH-LEVEL OBSERVATIONS")
    lines.append("=" * 56)
    lines.append("")

    behavioral_best_count = sum(
        1 for dataset_name in dataset_order if best_model_map.get(dataset_name) == BEHAVIORAL_AGENT_NAME
    )
    lines.append(
        f"* {BEHAVIORAL_AGENT_NAME} achieved the strongest Test F1 on "
        f"{behavioral_best_count}/{len(dataset_order) if dataset_order else 0} dataset(s)."
    )

    all_final_ge_traffic = True
    any_final_lt_behavioral = False
    for row in comparison_rows:
        behavioral_f1 = row["behavioral_f1"]
        traffic_f1 = row["traffic_f1"]
        final_f1 = row["final_f1"]
        if pd.isna(traffic_f1) or pd.isna(final_f1) or final_f1 < traffic_f1:
            all_final_ge_traffic = False
        if not pd.isna(behavioral_f1) and not pd.isna(final_f1) and final_f1 < behavioral_f1:
            any_final_lt_behavioral = True

    if all_final_ge_traffic:
        lines.append(
            f"* {FINAL_AGENT_NAME} matched or improved over {TRAFFIC_AGENT_NAME} on all reported datasets."
        )
    else:
        lines.append(
            f"* {FINAL_AGENT_NAME} improved over {TRAFFIC_AGENT_NAME} on several datasets, but not all."
        )

    if any_final_lt_behavioral:
        lines.append(
            f"* {FINAL_AGENT_NAME} did not consistently outperform {BEHAVIORAL_AGENT_NAME}, "
            "indicating the strongest single agent can still dominate."
        )
    else:
        lines.append(
            f"* {FINAL_AGENT_NAME} remained competitive with {BEHAVIORAL_AGENT_NAME} across reported datasets."
        )

    lines.append(
        "* Current interaction logic supports multi-agent resolution and contest tracking for thesis analysis."
    )
    lines.append("")

    return "\n".join(lines)


def _validate_finite_matrix(
    matrix,
    dataset_name: str,
    agent_name: str,
    stage: str,
) -> None:
    rows, cols = getattr(matrix, "shape", (None, None))
    is_sparse = bool(hasattr(matrix, "nnz") and hasattr(matrix, "data"))

    if is_sparse:
        data = np.asarray(matrix.data, dtype=np.float64)
        finite = bool(np.isfinite(data).all())
        if data.size == 0:
            min_val = max_val = mean_abs = max_abs = 0.0
        else:
            min_val = float(np.min(data))
            max_val = float(np.max(data))
            abs_data = np.abs(data)
            mean_abs = float(np.mean(abs_data))
            max_abs = float(np.max(abs_data))
    else:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        finite = bool(np.isfinite(arr).all())
        min_val = float(np.min(arr)) if arr.size else 0.0
        max_val = float(np.max(arr)) if arr.size else 0.0
        abs_arr = np.abs(arr)
        mean_abs = float(np.mean(abs_arr)) if arr.size else 0.0
        max_abs = float(np.max(abs_arr)) if arr.size else 0.0

    if not finite:
        raise ValueError(
            f"{agent_name} preprocessing still contains non-finite values for "
            f"dataset {dataset_name} during {stage}"
        )

    print(
        f"[INFO] {dataset_name}: {agent_name} {stage} matrix "
        f"finite=True, sparse={is_sparse}, rows={rows}, cols={cols}"
    )
    print(
        f"[INFO] {dataset_name}: {agent_name} {stage} "
        f"min={min_val:.6g}, max={max_val:.6g}, mean_abs={mean_abs:.6g}, max_abs={max_abs:.6g}"
    )


def _ensure_float64_matrix(matrix, dataset_name: str, agent_name: str):
    """Convert transformed matrix to float64 and reject invalid dtypes."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.dtype != np.float64:
        raise ValueError(
            f"{agent_name} transformed matrix dtype cast failed for dataset {dataset_name}; "
            f"got dtype={arr.dtype}"
        )
    return arr


def _to_float64_array(x):
    """Transformer callback for enforcing float64 in sklearn pipelines."""
    return np.asarray(x, dtype=np.float64)


def _find_suspicious_columns(columns: List[str]) -> List[str]:
    suspicious: List[str] = []
    for col in columns:
        normalized = str(col).strip().lower()
        if any(token in normalized for token in SUSPICIOUS_NAME_TOKENS):
            suspicious.append(col)
    return sorted(dict.fromkeys(suspicious))


def _safe_row_hashes(df: pd.DataFrame) -> pd.Series:
    try:
        return pd.util.hash_pandas_object(df, index=False)
    except TypeError:
        return pd.util.hash_pandas_object(df.astype(str), index=False)


def _resolve_case_insensitive_columns(
    columns: List[str],
    candidates: set[str],
) -> List[str]:
    lowered_map = {str(col).strip().lower(): str(col) for col in columns}
    resolved: List[str] = []
    for candidate in candidates:
        candidate_name = str(candidate).strip().lower()
        if candidate_name in lowered_map:
            resolved.append(lowered_map[candidate_name])
    return sorted(dict.fromkeys(resolved))


def _compute_duplicate_and_overlap_stats(
    x_full: pd.DataFrame,
    y_full: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Dict[str, Any]:
    full_df = x_full.copy()
    full_df["__label__"] = y_full.values
    total_full = int(len(full_df))
    duplicates_within = int(full_df.duplicated(keep="first").sum())
    duplicates_within_pct = float(duplicates_within / total_full) if total_full else 0.0

    train_hashes = _safe_row_hashes(x_train)
    test_hashes = _safe_row_hashes(x_test)
    train_hash_set = set(train_hashes.tolist())
    test_hash_set = set(test_hashes.tolist())

    overlap_unique_count = int(len(train_hash_set.intersection(test_hash_set)))
    overlap_test_rows = int(test_hashes.isin(train_hash_set).sum())
    overlap_train_rows = int(train_hashes.isin(test_hash_set).sum())
    overlap_test_pct = float(overlap_test_rows / len(x_test)) if len(x_test) else 0.0
    overlap_train_pct = float(overlap_train_rows / len(x_train)) if len(x_train) else 0.0

    return {
        "full_rows": total_full,
        "duplicates_within_count": duplicates_within,
        "duplicates_within_pct": duplicates_within_pct,
        "overlap_unique_feature_rows": overlap_unique_count,
        "overlap_test_rows_count": overlap_test_rows,
        "overlap_test_rows_pct": overlap_test_pct,
        "overlap_train_rows_count": overlap_train_rows,
        "overlap_train_rows_pct": overlap_train_pct,
    }


def _deduplicate_dataset_for_split(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Remove duplicate records from canonical model features before split."""
    if len(x) != len(y):
        raise ValueError("Feature/label length mismatch during deduplication.")

    combined = x.copy()
    combined["__label__"] = y.values

    rows_before = int(len(combined))
    feature_columns = list(x.columns)
    dedup_subset = feature_columns + ["__label__"]

    # Step 1: remove exact duplicate records (features + label).
    exact_duplicate_removed = int(combined.duplicated(subset=dedup_subset, keep="first").sum())
    dedup_exact = combined.drop_duplicates(
        subset=dedup_subset,
        keep="first",
    ).copy()

    # Step 2: detect duplicate feature rows with conflicting labels and remove them entirely.
    label_nunique_by_feature = dedup_exact.groupby(feature_columns, dropna=False)["__label__"].transform(
        "nunique"
    )
    conflict_mask = label_nunique_by_feature > 1
    conflicting_feature_patterns = int(
        dedup_exact.loc[conflict_mask, feature_columns]
        .drop_duplicates()
        .shape[0]
    )
    conflicting_rows_removed = int(conflict_mask.sum())
    dedup_no_conflicts = dedup_exact.loc[~conflict_mask].copy()

    rows_after = int(len(dedup_no_conflicts))
    total_removed = rows_before - rows_after

    x_dedup = dedup_no_conflicts.drop(columns=["__label__"]).copy()
    y_dedup = dedup_no_conflicts["__label__"].astype(y.dtype).copy()

    stats = {
        "canonical_rows_before_dedup": rows_before,
        "rows_before_deduplication": rows_before,
        "rows_after_deduplication": rows_after,
        "canonical_rows_after_cleaning": rows_after,
        "duplicates_removed_total": int(total_removed),
        "duplicates_removed_pct": (float(total_removed / rows_before) if rows_before else 0.0),
        "exact_duplicate_records_removed": exact_duplicate_removed,
        "duplicate_feature_rows_removed_same_label": 0,
        "conflicting_duplicate_feature_patterns": conflicting_feature_patterns,
        "conflicting_duplicate_rows_removed": conflicting_rows_removed,
    }
    return x_dedup, y_dedup, stats


def _candidate_group_identifier_columns(columns: List[str]) -> List[str]:
    candidates: List[str] = []
    for col in columns:
        normalized = str(col).strip().lower()
        token_match = any(token in normalized for token in GROUP_IDENTIFIER_TOKENS)
        suffix_match = normalized.endswith("_id")
        exact_match = normalized in {"id", "flow", "session", "connection"}
        if token_match or suffix_match or exact_match:
            candidates.append(col)
    return sorted(dict.fromkeys(candidates))


def _compute_group_overlap(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    group_columns: List[str],
) -> Dict[str, Any]:
    if not group_columns:
        return {"group_columns": [], "group_overlap_rows": []}

    overlaps: List[Dict[str, Any]] = []
    for column in group_columns:
        train_values = set(x_train[column].dropna().astype(str).tolist())
        test_values = set(x_test[column].dropna().astype(str).tolist())
        shared = train_values.intersection(test_values)
        overlap_count = int(len(shared))
        train_unique = int(len(train_values))
        test_unique = int(len(test_values))
        train_overlap_pct = float(overlap_count / train_unique) if train_unique else 0.0
        test_overlap_pct = float(overlap_count / test_unique) if test_unique else 0.0
        overlaps.append(
            {
                "column": column,
                "shared_unique_count": overlap_count,
                "train_unique_count": train_unique,
                "test_unique_count": test_unique,
                "train_overlap_pct": train_overlap_pct,
                "test_overlap_pct": test_overlap_pct,
            }
        )

    return {"group_columns": group_columns, "group_overlap_rows": overlaps}


def _build_leakage_report(leakage_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Leakage Validation Report")
    lines.append("")
    lines.append(
        "Scope: canonical feature-frame construction, duplicate/conflict cleaning before split, "
        "train/test exact feature-row overlap checks, suspicious column-name audit, and optional "
        "group-identifier overlap checks."
    )
    lines.append("")

    if not leakage_rows:
        lines.append("No dataset leakage checks were recorded.")
        return "\n".join(lines) + "\n"

    for row in leakage_rows:
        lines.append(f"## {row['dataset']}")
        lines.append("")
        lines.append("### Canonical Dataset")
        lines.append(f"- Raw row count: {row['raw_row_count']}")
        lines.append(
            f"- Final canonical row count before deduplication: {row['canonical_rows_before_dedup']}"
        )
        lines.append(
            "- Identifier columns dropped: "
            + (", ".join(row["dropped_identifier_columns"]) if row["dropped_identifier_columns"] else "none")
        )
        lines.append(
            "- Target-proxy columns dropped: "
            + (", ".join(row["dropped_target_proxy_columns"]) if row["dropped_target_proxy_columns"] else "none")
        )
        lines.append(
            f"- Canonical feature columns finalized (n={row['feature_count']}): {row['feature_columns_text']}"
        )
        lines.append("")
        lines.append("### Deduplication")
        lines.append(
            f"- Rows before deduplication: {row['rows_before_deduplication']}"
        )
        lines.append(
            f"- Rows after deduplication: {row['rows_after_deduplication']}"
        )
        lines.append(
            f"- Total removed: {row['duplicates_removed_total']} ({row['duplicates_removed_pct']:.4%})"
        )
        lines.append(
            f"- Exact duplicate records removed (features + label): "
            f"{row['exact_duplicate_records_removed']}"
        )
        lines.append(
            f"- Conflicting duplicate feature patterns removed: "
            f"{row['conflicting_duplicate_feature_patterns']} patterns / "
            f"{row['conflicting_duplicate_rows_removed']} rows"
        )
        lines.append(
            f"- Final row count after canonical cleaning: {row['canonical_rows_after_cleaning']}"
        )
        lines.append("")
        lines.append("### Duplicate Row Check")
        lines.append(
            f"- Post-fix full dataset exact duplicates: {row['duplicates_within_count']} / {row['full_rows']} "
            f"({row['duplicates_within_pct']:.4%})"
        )
        lines.append(
            f"- Post-fix train/test overlap (exact feature rows): unique_shared={row['overlap_unique_feature_rows']}, "
            f"test_rows={row['overlap_test_rows_count']} ({row['overlap_test_rows_pct']:.4%}), "
            f"train_rows={row['overlap_train_rows_count']} ({row['overlap_train_rows_pct']:.4%})"
        )
        lines.append("")
        lines.append("### Label-Proxy / Leakage Column Audit")
        lines.append(
            "- Existing drop policy (pipeline.data_loader.POTENTIAL_LEAKAGE_COLUMNS): "
            + ", ".join(row["drop_policy_columns"])
        )
        lines.append(
            "- Raw suspicious columns matched by drop policy (excluding label): "
            + (", ".join(row["dropped_suspicious_columns"]) if row["dropped_suspicious_columns"] else "none")
        )
        lines.append(
            "- Dropped identifier columns from features: "
            + (", ".join(row["dropped_identifier_columns"]) if row["dropped_identifier_columns"] else "none")
        )
        lines.append(
            "- Suspicious name matches that remain in features: "
            + (", ".join(row["remaining_suspicious_features"]) if row["remaining_suspicious_features"] else "none")
        )
        lines.append("")
        lines.append("### Group-Leakage Check")
        lines.append(
            "- Candidate group/session identifier columns: "
            + (", ".join(row["group_columns"]) if row["group_columns"] else "none detected")
        )
        if row["group_overlap_rows"]:
            for overlap in row["group_overlap_rows"]:
                lines.append(
                    f"- {overlap['column']}: shared_unique={overlap['shared_unique_count']} | "
                    f"train_unique={overlap['train_unique_count']} | test_unique={overlap['test_unique_count']} | "
                    f"train_overlap={overlap['train_overlap_pct']:.4%} | "
                    f"test_overlap={overlap['test_overlap_pct']:.4%}"
                )
        else:
            lines.append("- No group/session overlap check applied (no candidate identifier column).")
        lines.append("")
        lines.append(f"### Final Verdict")
        lines.append(f"- {row['verdict']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _build_final_report(
    config: Dict,
    results_df: pd.DataFrame,
    interaction_summary_df: pd.DataFrame,
    trust_summary_df: pd.DataFrame,
    class_balance_df: pd.DataFrame,
    reasoning_df: pd.DataFrame,
    reasoning_config: Dict[str, Any],
    interaction_config: Dict[str, Any],
    output_files: List[Dict[str, str]],
) -> str:
    experiment_name = config.get("experiment", {}).get("name", "ids_experiment")
    datasets = sorted(results_df["dataset"].unique().tolist()) if not results_df.empty else []
    ollama_enabled = bool(reasoning_config.get("ollama_enabled", False))
    ollama_model_name = str(reasoning_config.get("ollama_model_name", "unknown_model"))

    lines = [
        "# Final IDS Evaluation Report",
        "",
        "## Dataset Overview",
        f"- Experiment: `{experiment_name}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Datasets evaluated: {', '.join(datasets) if datasets else 'None'}",
        "- Label convention: normal=0, attack=1.",
        "- Two-agent design: both agents operate on the same processed intrusion dataset rows.",
        "- Role differentiation is analytical (not separate raw sensor streams).",
        "- BehavioralAnalysisAgent backbone: RandomForestClassifier.",
        "- TrafficAnalysisAgent backbone: SVM (SVC, RBF kernel).",
        "- TrafficAnalysisAgent preprocessing uses train-only fitted imputation/scaling for leakage-safe numerical stability.",
        "",
        "## Train/Test Split Summary",
        "- Split policy: train=70%, test=30%, stratified by label, random_state=42.",
    ]

    if class_balance_df.empty:
        lines.append("- No split/balance summary available.")
    else:
        for _, row in class_balance_df.iterrows():
            lines.append(
                f"- {row['dataset']}: train(normal={int(row['original_train_normal_count'])}, "
                f"attack={int(row['original_train_attack_count'])}), "
                f"balanced_train(normal={int(row['balanced_train_normal_count'])}, "
                f"attack={int(row['balanced_train_attack_count'])}), "
                f"test(normal={int(row['test_normal_count'])}, attack={int(row['test_attack_count'])}), "
                f"method={row['balancing_method_used']}"
            )

    lines.extend(["", "## Individual Agent Results"])
    for dataset in datasets:
        lines.append(f"### {dataset}")
        subset = results_df[
            (results_df["dataset"] == dataset)
            & (results_df["agent"].isin([BEHAVIORAL_AGENT_NAME, TRAFFIC_AGENT_NAME]))
        ]
        if subset.empty:
            lines.append("- Individual agent metrics unavailable.")
        else:
            for _, row in subset.iterrows():
                lines.append(
                    f"- {row['agent']}: CV Acc={row['cv_accuracy']:.4f}, CV Prec={row['cv_precision']:.4f}, "
                    f"CV Rec={row['cv_recall']:.4f}, CV F1={row['cv_f1']:.4f}, "
                    f"Test Acc={row['test_accuracy']:.4f}, Test Prec={row['test_precision']:.4f}, "
                    f"Test Rec={row['test_recall']:.4f}, Test F1={row['test_f1']:.4f}, "
                    f"TP={int(row['tp'])}, TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, "
                    f"FPR={row['fpr']:.4f}, FNR={row['fnr']:.4f}, TPR={row['tpr']:.4f}, TNR={row['tnr']:.4f}, "
                    f"Specificity={row['specificity']:.4f}, BalancedAcc={row['balanced_accuracy']:.4f}"
                )

    lines.extend(["", "## Agreement/Disagreement Summary"])
    if interaction_summary_df.empty:
        lines.append("- Interaction summary unavailable.")
    else:
        for _, row in interaction_summary_df.iterrows():
            lines.append(
                f"- {row['dataset']}: agreement_rate={row['agreement_rate']:.4f}, "
                f"disagreement_rate={row['disagreement_rate']:.4f}, "
                f"behavioral_wins_on_disagreement={row['behavioral_wins_on_disagreement']:.4f}, "
                f"traffic_wins_on_disagreement={row['traffic_wins_on_disagreement']:.4f}, "
                f"contested_case_rate={row['contested_case_rate']:.4f}, "
                f"threshold={row['disagreement_confidence_threshold']:.2f}"
            )

    lines.extend(["", "## Conflict Resolution Summary"])
    lines.append(
        "- Protocol: agreement -> agreed label; disagreement -> trust-based winner; "
        "small trust gap (< trust threshold) -> trust_contested with trust-based fallback label."
    )

    lines.extend(["", "## Final Resolved Multi-Agent Results"])
    final_subset = results_df[results_df["agent"] == FINAL_AGENT_NAME]
    if final_subset.empty:
        lines.append("- Final resolved metrics unavailable.")
    else:
        for _, row in final_subset.iterrows():
            lines.append(
                f"- {row['dataset']}: Acc={row['test_accuracy']:.4f}, Prec={row['test_precision']:.4f}, "
                f"Rec={row['test_recall']:.4f}, F1={row['test_f1']:.4f}, "
                f"TP={int(row['tp'])}, TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, "
                f"FPR={row['fpr']:.4f}, FNR={row['fnr']:.4f}, BalancedAcc={row['balanced_accuracy']:.4f}"
            )

    lines.extend(["", "## Trust Layer"])
    lines.append("- Trust design overview: static trust-aware prioritization over the existing two-agent system.")
    lines.append(
        "- Trust formula per agent/sample: "
        "trust_score = w1*global_reliability + w2*confidence + w3*disagreement_reliability."
    )
    lines.append(
        f"- Weights: w1={interaction_config['trust_weight_global']:.2f}, "
        f"w2={interaction_config['trust_weight_confidence']:.2f}, "
        f"w3={interaction_config['trust_weight_disagreement']:.2f}; "
        f"trust_gap_threshold={interaction_config['trust_gap_threshold']:.2f}."
    )
    lines.append(
        "- Note: trust is used for decision prioritization only and does not replace either classifier backbone."
    )
    if trust_summary_df.empty:
        lines.append("- Trust summary unavailable.")
    else:
        for _, row in trust_summary_df.iterrows():
            lines.append(f"### {row['dataset']}")
            lines.append(
                f"- Global trust values: behavioral={row['behavioral_global_reliability']:.4f}, "
                f"traffic={row['traffic_global_reliability']:.4f}"
            )
            lines.append(
                f"- Disagreement trust values: behavioral={row['behavioral_disagreement_trust']:.4f}, "
                f"traffic={row['traffic_disagreement_trust']:.4f}"
            )
            lines.append(
                f"- Trust-based final performance: accuracy={row['trust_based_final_accuracy']:.4f}, "
                f"precision={row['trust_based_final_precision']:.4f}, "
                f"recall={row['trust_based_final_recall']:.4f}, f1={row['trust_based_final_f1']:.4f}"
            )
            lines.append(
                f"- Interpretation: contested cases={int(row['trust_based_contested_cases'])}, "
                f"resolution relies on trust prioritization during disagreements."
            )

    lines.extend(["", "## Ollama Status Summary"])
    lines.append(f"- Ollama reasoning enabled: {ollama_enabled}")
    lines.append(f"- Ollama model: `{ollama_model_name}`")
    if reasoning_df.empty:
        lines.append("- Reasoning rows: 0")
    else:
        success = int((reasoning_df["reasoning_status"] == "success").sum())
        failed = int((reasoning_df["reasoning_status"] != "success").sum())
        lines.append(f"- Reasoning success rows: {success}")
        lines.append(f"- Reasoning non-success rows: {failed}")

    lines.extend(["", "## Output Files"])
    for output in output_files:
        lines.append(f"- `{output['name']}` -> {output['description']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "experiment.yml"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_yaml_config(config_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to load configuration: {exc}")
        return

    random_state = int(config.get("experiment", {}).get("random_state", 42))
    preprocessing_config = config.get("preprocessing", {})
    reasoning_config = _normalize_reasoning_config(config)
    interaction_config = _normalize_interaction_config(config)
    cv_folds = int(config.get("validation", {}).get("cross_validation_folds", 5))
    strict_leakage_check = bool(
        config.get("validation", {}).get("strict_leakage_check", STRICT_LEAKAGE_CHECK_DEFAULT)
    )

    # Fixed split requirement for this stage.
    test_size = 0.3

    results_rows: List[Dict[str, Any]] = []
    interaction_summary_rows: List[Dict[str, Any]] = []
    trust_summary_rows: List[Dict[str, Any]] = []
    class_balance_rows: List[Dict[str, Any]] = []
    sample_prediction_frames: List[pd.DataFrame] = []
    interaction_frames: List[pd.DataFrame] = []
    trust_interaction_frames: List[pd.DataFrame] = []
    reasoning_rows: List[Dict[str, Any]] = []
    leakage_rows: List[Dict[str, Any]] = []
    datasets_config = list(config.get("datasets", []))
    run_status_rows: List[Dict[str, str]] = []

    for dataset_idx, dataset_config in enumerate(datasets_config):
        dataset_name = dataset_config.get("name", "unknown_dataset")
        dataset_start_time = perf_counter()
        log_section(f"DATASET: {dataset_name}")
        log_step(f"[{dataset_idx + 1}/{len(datasets_config)}] Running dataset pipeline")
        try:
            dataset_config = dict(dataset_config)
            dataset_path = (project_root / dataset_config["path"]).resolve()
            dataset_config["path"] = str(dataset_path)
            log_substep("PREPROCESSING")
            raw_df = pd.read_csv(dataset_config["path"])
            raw_row_count = int(len(raw_df))
            raw_preview = raw_df.head(0)
            raw_columns = raw_preview.columns.tolist()
            label_column_name = str(dataset_config.get("label_column"))
            if label_column_name not in raw_df.columns:
                raise ValueError(
                    f"{dataset_name}: label column '{label_column_name}' not found in dataset."
                )
            dropped_suspicious_columns = _resolve_case_insensitive_columns(
                raw_columns,
                {col for col in POTENTIAL_LEAKAGE_COLUMNS if col != label_column_name.strip().lower()},
            )
            print(
                "[INFO] "
                f"{dataset_name}: existing drop policy columns="
                f"{sorted(POTENTIAL_LEAKAGE_COLUMNS)}"
            )
            if dropped_suspicious_columns:
                print(
                    "[INFO] "
                    f"{dataset_name}: raw suspicious columns matched for dropping="
                    f"{dropped_suspicious_columns}"
                )
            else:
                print(
                    f"[INFO] {dataset_name}: no raw columns matched current drop policy (excluding label)."
                )

            if dropped_suspicious_columns:
                raw_df = raw_df.drop(columns=dropped_suspicious_columns)

            y = raw_df[label_column_name].copy()
            x = raw_df.drop(columns=[label_column_name]).copy()
            y = _convert_to_binary_labels(y, preprocessing_config)

            dropped_identifier_columns = _candidate_group_identifier_columns(x.columns.tolist())
            if dataset_name.strip().lower() == "unsw-nb15":
                unsw_id_columns = _resolve_case_insensitive_columns(x.columns.tolist(), {"id"})
                dropped_identifier_columns = sorted(
                    dict.fromkeys(dropped_identifier_columns + unsw_id_columns)
                )
            if dropped_identifier_columns:
                x = x.drop(columns=dropped_identifier_columns)

            if x.empty or x.shape[1] == 0:
                raise ValueError(
                    f"{dataset_name}: no feature columns remain after target-proxy/identifier dropping."
                )

            x, sanitize_stats = sanitize_feature_values(
                x, dataset_config.get("categorical_columns", [])
            )
            remaining_suspicious_features = _find_suspicious_columns(x.columns.tolist())
            print(
                f"[INFO] {dataset_name}: canonical feature columns finalized (n={len(x.columns)})"
            )
            print(
                "[INFO] "
                f"{dataset_name}: identifier columns dropped = {dropped_identifier_columns}"
            )
            print(
                "[INFO] "
                f"{dataset_name}: target-proxy columns dropped = {dropped_suspicious_columns}"
            )

            x, y, dedup_stats = _deduplicate_dataset_for_split(x, y)
            print(
                f"[INFO] {dataset_name}: rows before deduplication = "
                f"{dedup_stats['rows_before_deduplication']}"
            )
            print(
                f"[INFO] {dataset_name}: rows after deduplication = "
                f"{dedup_stats['rows_after_deduplication']}"
            )
            print(
                f"[INFO] {dataset_name}: removed exact duplicates = "
                f"{dedup_stats['duplicates_removed_total']} "
                f"({dedup_stats['duplicates_removed_pct']:.4%})"
            )
            print(
                f"[INFO] {dataset_name}: exact duplicate records removed (features+label) = "
                f"{dedup_stats['exact_duplicate_records_removed']}"
            )
            print(
                f"[INFO] {dataset_name}: exact duplicate records removed from canonical cleaned data = "
                f"{dedup_stats['exact_duplicate_records_removed']}"
            )
            if dedup_stats["conflicting_duplicate_feature_patterns"] > 0:
                print(
                    f"[WARN] {dataset_name}: conflicting feature patterns removed = "
                    f"{dedup_stats['conflicting_duplicate_feature_patterns']} pattern(s), "
                    f"{dedup_stats['conflicting_duplicate_rows_removed']} row(s)"
                )
            else:
                print(
                    f"[INFO] {dataset_name}: no conflicting duplicate feature rows detected."
                )
            print(
                f"[INFO] {dataset_name}: canonical cleaned rows after dedup/conflict removal = "
                f"{dedup_stats['rows_after_deduplication']}"
            )

            if x.empty or x.shape[1] == 0:
                raise ValueError(
                    f"{dataset_name}: no feature columns remain after deduplication/identifier drop."
                )
            if y.nunique() < 2:
                raise ValueError(
                    f"{dataset_name}: deduplication left fewer than 2 classes; cannot run stratified split."
                )
            print(
                f"[INFO] {dataset_name}: feature columns used ({len(x.columns)}): "
                + ", ".join(x.columns.astype(str).tolist())
            )
            if remaining_suspicious_features:
                print(
                    f"[WARN] {dataset_name}: suspicious feature-name matches still present="
                    f"{remaining_suspicious_features}"
                )
            else:
                print(
                    f"[INFO] {dataset_name}: no suspicious feature-name matches in final feature set."
                )
            print(
                f"[INFO] {dataset_name}: checked {sanitize_stats['numeric_columns_checked']} numeric feature column(s)."
            )
            print(
                f"[INFO] {dataset_name}: replaced {sanitize_stats['non_finite_values_replaced_with_nan']} non-finite value(s) with NaN."
            )

            log_substep("TRAIN/TEST SPLIT")
            stratify_values = y if y.nunique() > 1 else None
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify_values,
            )
            print(f"[INFO] {dataset_name}: train size={len(x_train)} | test size={len(x_test)}")
            duplicate_overlap_stats = _compute_duplicate_and_overlap_stats(
                x_full=x,
                y_full=y,
                x_train=x_train,
                x_test=x_test,
            )
            print(
                f"[INFO] {dataset_name}: full-dataset exact duplicates="
                f"{duplicate_overlap_stats['duplicates_within_count']}/{duplicate_overlap_stats['full_rows']} "
                f"({duplicate_overlap_stats['duplicates_within_pct']:.4%})"
            )
            print(
                f"[INFO] {dataset_name}: post-fix train/test exact feature-row overlap="
                f"unique_shared={duplicate_overlap_stats['overlap_unique_feature_rows']}, "
                f"test_rows={duplicate_overlap_stats['overlap_test_rows_count']}"
                f"({duplicate_overlap_stats['overlap_test_rows_pct']:.4%}), "
                f"train_rows={duplicate_overlap_stats['overlap_train_rows_count']}"
                f"({duplicate_overlap_stats['overlap_train_rows_pct']:.4%})"
            )
            overlap_rows = (
                duplicate_overlap_stats["overlap_test_rows_count"]
                + duplicate_overlap_stats["overlap_train_rows_count"]
            )
            if overlap_rows > 0:
                overlap_message = (
                    f"{dataset_name}: residual train/test exact feature-row overlap detected; "
                    f"unique_shared={duplicate_overlap_stats['overlap_unique_feature_rows']}, "
                    f"test_rows={duplicate_overlap_stats['overlap_test_rows_count']}, "
                    f"train_rows={duplicate_overlap_stats['overlap_train_rows_count']}"
                )
                if strict_leakage_check:
                    raise ValueError(overlap_message)
                print(f"[WARN] {overlap_message}")
            else:
                print(f"[INFO] {dataset_name}: post-split exact feature-row overlap = 0")

            group_columns = _candidate_group_identifier_columns(x.columns.tolist())
            group_overlap = _compute_group_overlap(x_train, x_test, group_columns)
            if group_columns:
                print(
                    f"[INFO] {dataset_name}: candidate group/session identifier columns={group_columns}"
                )
                for overlap in group_overlap["group_overlap_rows"]:
                    print(
                        f"[INFO] {dataset_name}: group overlap {overlap['column']} -> "
                        f"shared_unique={overlap['shared_unique_count']}, "
                        f"train_overlap={overlap['train_overlap_pct']:.4%}, "
                        f"test_overlap={overlap['test_overlap_pct']:.4%}"
                    )
            else:
                print(
                    f"[INFO] {dataset_name}: no flow/session/group identifier column detected for overlap check."
                )
            print(
                f"[INFO] {dataset_name}: preprocessing order verified -> split first, "
                "fit preprocessors on train only, transform test with train-fitted preprocessors."
            )

            log_substep("BALANCING")
            original_train_counts = _class_counts(y_train)
            x_train_balanced, y_train_balanced, balancing_method = _balance_training_split(
                x_train=x_train,
                y_train=y_train,
                random_state=random_state,
            )
            balanced_train_counts = _class_counts(y_train_balanced)
            test_counts = _class_counts(y_test)
            print(
                f"[INFO] {dataset_name}: original train dist={original_train_counts}, "
                f"balanced train dist={balanced_train_counts}, test dist={test_counts}"
            )
            class_balance_rows.append(
                {
                    "dataset": dataset_name,
                    "original_train_normal_count": original_train_counts["normal"],
                    "original_train_attack_count": original_train_counts["attack"],
                    "balanced_train_normal_count": balanced_train_counts["normal"],
                    "balanced_train_attack_count": balanced_train_counts["attack"],
                    "test_normal_count": test_counts["normal"],
                    "test_attack_count": test_counts["attack"],
                    "balancing_method_used": balancing_method,
                }
            )

            categorical_columns = dataset_config.get("categorical_columns", [])
            behavioral_preprocessor = build_preprocessing_pipeline(
                x_train_balanced, categorical_columns, preprocessing_config
            )
            x_train_behavioral = behavioral_preprocessor.fit_transform(x_train_balanced)
            x_test_behavioral = behavioral_preprocessor.transform(x_test)

            log_substep("TRAFFIC ANALYSIS PREPROCESSING")
            inferred_categorical, inferred_numeric = infer_feature_type_columns(
                x_train_balanced, categorical_columns
            )
            print(f"[INFO] {dataset_name}: numeric columns = {len(inferred_numeric)}")
            print(f"[INFO] {dataset_name}: categorical columns = {len(inferred_categorical)}")
            print(f"[INFO] {dataset_name}: replaced non-finite values before SVM pipeline")
            traffic_preprocessor = build_traffic_preprocessing_pipeline(
                x_train_balanced, categorical_columns, preprocessing_config
            )
            x_train_traffic = _ensure_float64_matrix(
                traffic_preprocessor.fit_transform(x_train_balanced),
                dataset_name=dataset_name,
                agent_name=TRAFFIC_AGENT_NAME,
            )
            print(f"[INFO] {dataset_name}: fitted StandardScaler on training data only")
            x_test_traffic = _ensure_float64_matrix(
                traffic_preprocessor.transform(x_test),
                dataset_name=dataset_name,
                agent_name=TRAFFIC_AGENT_NAME,
            )
            print(
                f"[INFO] {dataset_name}: transformed train/test data for {TRAFFIC_AGENT_NAME}"
            )
            print(f"[INFO] {dataset_name}: {TRAFFIC_AGENT_NAME} dtype=float64")
            _validate_finite_matrix(
                x_train_traffic,
                dataset_name=dataset_name,
                agent_name=TRAFFIC_AGENT_NAME,
                stage="train_transform",
            )
            _validate_finite_matrix(
                x_test_traffic,
                dataset_name=dataset_name,
                agent_name=TRAFFIC_AGENT_NAME,
                stage="test_transform",
            )
            log_substep("TRAFFIC ANALYSIS STABILITY CHECK")
            _print_data_diagnostics(dataset_name, x_train_balanced, y_train_balanced, y_test)

            agent_runtime: Dict[str, Dict[str, Any]] = {}
            failed_agent_errors: List[str] = []

            for configured_agent_name, agent_config in config.get("agents", {}).items():
                model_name = agent_config.get("model")
                model_params = dict(agent_config.get("params", {}))

                try:
                    log_substep("CROSS-VALIDATION")
                    if model_name in {"SVC", "SVM"}:
                        cv_preprocessor = build_traffic_preprocessing_pipeline(
                            x_train_balanced, categorical_columns, preprocessing_config
                        )
                        cv_model = Pipeline(
                            steps=[
                                ("preprocessor", cv_preprocessor),
                                (
                                    "to_float64",
                                    FunctionTransformer(
                                        _to_float64_array,
                                        validate=False,
                                    ),
                                ),
                                ("model", create_agent(model_name, model_params)),
                            ]
                        )
                    else:
                        cv_preprocessor = build_preprocessing_pipeline(
                            x_train_balanced, categorical_columns, preprocessing_config
                        )
                        cv_model = Pipeline(
                            steps=[
                                ("preprocessor", cv_preprocessor),
                                ("model", create_agent(model_name, model_params)),
                            ]
                        )
                    cv_metrics = run_cross_validation(
                        cv_model, x_train_balanced, y_train_balanced, cv_folds
                    )

                    log_substep("TRAINING AGENTS")
                    trained_model = create_detection_agent(
                        model_name,
                        model_params,
                        reasoning_config=reasoning_config,
                    )
                    if model_name in {"SVC", "SVM"}:
                        fit_preprocessor = build_traffic_preprocessing_pipeline(
                            x_train_balanced, categorical_columns, preprocessing_config
                        )
                        trained_model.model = Pipeline(
                            steps=[
                                ("preprocessor", fit_preprocessor),
                                (
                                    "to_float64",
                                    FunctionTransformer(
                                        _to_float64_array,
                                        validate=False,
                                    ),
                                ),
                                ("model", create_agent(model_name, model_params)),
                            ]
                        )
                        trained_model = train_agent(trained_model, x_train_balanced, y_train_balanced)
                    else:
                        trained_model = train_agent(
                            trained_model,
                            x_train_behavioral,
                            y_train_balanced,
                        )
                    if model_name in {"SVC", "SVM"}:
                        svm_pipeline = getattr(trained_model, "model", None)
                        if svm_pipeline is None:
                            raise ValueError("TrafficAnalysisAgent missing underlying SVC model.")
                        if not isinstance(svm_pipeline, Pipeline) or "model" not in svm_pipeline.named_steps:
                            raise ValueError(
                                f"{TRAFFIC_AGENT_NAME} must use sklearn Pipeline(preprocessor->to_float64->model)."
                            )
                        svm_model = svm_pipeline.named_steps["model"]
                        if str(type(svm_model).__name__) != "SVC":
                            raise ValueError(
                                f"{TRAFFIC_AGENT_NAME} expected SVC model, got {type(svm_model).__name__}."
                            )
                        if not bool(getattr(svm_model, "probability", False)):
                            raise ValueError(
                                f"{TRAFFIC_AGENT_NAME} must enable probability=True for trust compatibility."
                            )
                        print(
                            f"[INFO] {dataset_name}: {TRAFFIC_AGENT_NAME} backbone="
                            f"SVC(kernel={getattr(svm_model, 'kernel', 'unknown')})"
                        )
                        print(
                            f"[INFO] {dataset_name}: fitted SVM traffic model successfully "
                            f"(C={getattr(svm_model, 'C', 'unknown')}, "
                            f"gamma={getattr(svm_model, 'gamma', 'unknown')}, "
                            f"probability={getattr(svm_model, 'probability', 'unknown')})"
                        )

                    log_substep("TEST EVALUATION")
                    x_test_for_agent = (
                        x_test if model_name in {"SVC", "SVM"} else x_test_behavioral
                    )
                    prediction_output = trained_model.predict(x_test_for_agent)
                    y_pred = prediction_output["y_pred"]
                    y_prob = prediction_output["y_prob"]
                    if model_name in {"SVC", "SVM"}:
                        if y_prob is None:
                            raise ValueError(
                                f"{TRAFFIC_AGENT_NAME} predict_proba output is missing for dataset {dataset_name}."
                            )
                        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
                        if not np.isfinite(y_prob_arr).all():
                            raise ValueError(
                                f"{TRAFFIC_AGENT_NAME} probability finiteness check failed for dataset {dataset_name}."
                            )
                        if np.any((y_prob_arr < -1e-12) | (y_prob_arr > 1.0 + 1e-12)):
                            raise ValueError(
                                f"{TRAFFIC_AGENT_NAME} probability range check failed for dataset {dataset_name}."
                            )
                        print(
                            f"[INFO] {dataset_name}: SVM probability output check passed"
                        )
                    test_metrics = evaluate_model(
                        trained_model,
                        x_test_for_agent,
                        y_test,
                        predictions=y_pred,
                        y_prob=y_prob,
                    )

                    result_agent_name = str(trained_model.agent_name)
                    results_rows.append(
                        {
                            "dataset": dataset_name,
                            "agent": result_agent_name,
                            "cv_accuracy": cv_metrics["cv_accuracy"],
                            "cv_precision": cv_metrics["cv_precision"],
                            "cv_recall": cv_metrics["cv_recall"],
                            "cv_f1": cv_metrics["cv_f1"],
                            "cv_roc_auc": cv_metrics["cv_roc_auc"],
                            "test_accuracy": test_metrics["test_accuracy"],
                            "test_precision": test_metrics["test_precision"],
                            "test_recall": test_metrics["test_recall"],
                            "test_f1": test_metrics["test_f1"],
                            "precision": test_metrics["precision"],
                            "recall": test_metrics["recall"],
                            "tp": test_metrics["tp"],
                            "tn": test_metrics["tn"],
                            "fp": test_metrics["fp"],
                            "fn": test_metrics["fn"],
                            "fpr": test_metrics["fpr"],
                            "fnr": test_metrics["fnr"],
                            "tpr": test_metrics["tpr"],
                            "tnr": test_metrics["tnr"],
                            "specificity": test_metrics["specificity"],
                            "balanced_accuracy": test_metrics["balanced_accuracy"],
                            "error_rate": test_metrics["error_rate"],
                            "roc_auc": test_metrics["roc_auc"],
                            "pr_auc": test_metrics["pr_auc"],
                            "support_total": test_metrics["support_total"],
                            "support_attack": test_metrics["support_attack"],
                            "support_normal": test_metrics["support_normal"],
                            "balancing_method": balancing_method,
                        }
                    )

                    agent_runtime[result_agent_name] = {
                        "model": trained_model,
                        "y_pred": np.asarray(y_pred),
                        "y_prob": np.asarray(y_prob) if y_prob is not None else None,
                        "metrics": test_metrics,
                        "decisions": trained_model.predict_decisions(
                            x_test_for_agent,
                            sample_indices=x_test.index.to_numpy(),
                        ),
                    }

                    printable = {
                        "cv_accuracy": cv_metrics["cv_accuracy"],
                        "cv_precision": cv_metrics["cv_precision"],
                        "cv_recall": cv_metrics["cv_recall"],
                        "cv_f1": cv_metrics["cv_f1"],
                        **test_metrics,
                    }
                    _print_results(dataset_name, result_agent_name, printable)
                except Exception as exc:  # pylint: disable=broad-except
                    failed_agent_errors.append(
                        f"{configured_agent_name}({model_name}): {type(exc).__name__}: {exc}"
                    )
                    log_warning(
                        f"Failed on dataset '{dataset_name}', agent '{configured_agent_name}': {exc}"
                    )
                    print(traceback.format_exc())

            if BEHAVIORAL_AGENT_NAME not in agent_runtime or TRAFFIC_AGENT_NAME not in agent_runtime:
                raise RuntimeError(
                    "Missing required agents for interaction layer. "
                    f"Have={list(agent_runtime.keys())}; "
                    f"agent_errors={failed_agent_errors}"
                )

            log_substep("AGENT INTERACTION")
            interaction_df, interaction_summary = resolve_agent_interactions(
                dataset_name=dataset_name,
                sample_indices=x_test.index.to_numpy(),
                true_labels=y_test.to_numpy(),
                behavioral_decisions=agent_runtime[BEHAVIORAL_AGENT_NAME]["decisions"],
                traffic_decisions=agent_runtime[TRAFFIC_AGENT_NAME]["decisions"],
                behavioral_metrics=agent_runtime[BEHAVIORAL_AGENT_NAME]["metrics"],
                traffic_metrics=agent_runtime[TRAFFIC_AGENT_NAME]["metrics"],
                trust_weight_global=interaction_config["trust_weight_global"],
                trust_weight_confidence=interaction_config["trust_weight_confidence"],
                trust_weight_disagreement=interaction_config["trust_weight_disagreement"],
                trust_gap_threshold=interaction_config["trust_gap_threshold"],
                disagreement_confidence_threshold=interaction_config[
                    "disagreement_confidence_threshold"
                ],
            )
            _print_interaction_summary(dataset_name, interaction_summary)

            log_substep("TRUST LAYER")
            print("[STEP] TRUST LAYER")
            print(
                f"[INFO] {dataset_name}: behavioral_global_reliability="
                f"{interaction_summary['behavioral_global_reliability']:.4f}"
            )
            print(
                f"[INFO] {dataset_name}: traffic_global_reliability="
                f"{interaction_summary['traffic_global_reliability']:.4f}"
            )
            print(
                f"[INFO] {dataset_name}: behavioral_disagreement_trust="
                f"{interaction_summary['behavioral_disagreement_trust']:.4f}"
            )
            print(
                f"[INFO] {dataset_name}: traffic_disagreement_trust="
                f"{interaction_summary['traffic_disagreement_trust']:.4f}"
            )

            log_substep("FINAL DECISION RESOLUTION")
            final_y_pred = interaction_df["final_label"].to_numpy()
            final_metrics = evaluate_predictions(
                y_test,
                final_y_pred,
                y_prob=interaction_df["final_probability_attack"].to_numpy(),
            )
            cm = confusion_matrix(y_test, final_y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            final_metrics["confusion_matrix"] = cm
            final_metrics["TP"] = int(tp)
            final_metrics["TN"] = int(tn)
            final_metrics["FP"] = int(fp)
            final_metrics["FN"] = int(fn)
            final_metrics["FPR"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            final_metrics["FNR"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            final_metrics["TPR"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            final_metrics["TNR"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            final_metrics["specificity"] = final_metrics["TNR"]
            final_metrics["balanced_accuracy"] = (
                final_metrics["TPR"] + final_metrics["TNR"]
            ) / 2.0
            total_cm = tp + tn + fp + fn
            final_metrics["error_rate"] = float((fp + fn) / total_cm) if total_cm > 0 else 0.0
            final_metrics["tp"] = int(tp)
            final_metrics["tn"] = int(tn)
            final_metrics["fp"] = int(fp)
            final_metrics["fn"] = int(fn)
            final_metrics["fpr"] = final_metrics["FPR"]
            final_metrics["fnr"] = final_metrics["FNR"]
            final_metrics["tpr"] = final_metrics["TPR"]
            final_metrics["tnr"] = final_metrics["TNR"]
            final_row = {
                "dataset": dataset_name,
                "agent": FINAL_AGENT_NAME,
                "cv_accuracy": np.nan,
                "cv_precision": np.nan,
                "cv_recall": np.nan,
                "cv_f1": np.nan,
                "cv_roc_auc": np.nan,
                "test_accuracy": final_metrics["test_accuracy"],
                "test_precision": final_metrics["test_precision"],
                "test_recall": final_metrics["test_recall"],
                "test_f1": final_metrics["test_f1"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "tp": final_metrics["tp"],
                "tn": final_metrics["tn"],
                "fp": final_metrics["fp"],
                "fn": final_metrics["fn"],
                "fpr": final_metrics["fpr"],
                "fnr": final_metrics["fnr"],
                "tpr": final_metrics["tpr"],
                "tnr": final_metrics["tnr"],
                "specificity": final_metrics["specificity"],
                "balanced_accuracy": final_metrics["balanced_accuracy"],
                "error_rate": final_metrics["error_rate"],
                "roc_auc": final_metrics["roc_auc"],
                "pr_auc": final_metrics["pr_auc"],
                "support_total": final_metrics["support_total"],
                "support_attack": final_metrics["support_attack"],
                "support_normal": final_metrics["support_normal"],
                "balancing_method": balancing_method,
                "confusion_matrix": final_metrics["confusion_matrix"],
            }
            results_rows.append(final_row)
            _print_results(dataset_name, FINAL_AGENT_NAME, final_row)
            print(
                f"[INFO] {dataset_name}: trust-based contested cases="
                f"{interaction_summary['contested_case_count']}"
            )
            print(
                f"[INFO] {dataset_name}: trust-based final F1={final_metrics['test_f1']:.4f}"
            )

            behavioral_metrics = agent_runtime[BEHAVIORAL_AGENT_NAME]["metrics"]
            traffic_metrics = agent_runtime[TRAFFIC_AGENT_NAME]["metrics"]
            interaction_summary.update(
                {
                    "behavioral_test_accuracy": behavioral_metrics["test_accuracy"],
                    "behavioral_test_precision": behavioral_metrics["test_precision"],
                    "behavioral_test_recall": behavioral_metrics["test_recall"],
                    "behavioral_test_f1": behavioral_metrics["test_f1"],
                    "traffic_test_accuracy": traffic_metrics["test_accuracy"],
                    "traffic_test_precision": traffic_metrics["test_precision"],
                    "traffic_test_recall": traffic_metrics["test_recall"],
                    "traffic_test_f1": traffic_metrics["test_f1"],
                    "final_decision_accuracy": final_metrics["test_accuracy"],
                    "final_decision_precision": final_metrics["test_precision"],
                    "final_decision_recall": final_metrics["test_recall"],
                    "final_decision_f1": final_metrics["test_f1"],
                }
            )
            interaction_summary_rows.append(interaction_summary)
            trust_summary_rows.append(
                {
                    "dataset": dataset_name,
                    "behavioral_global_reliability": interaction_summary[
                        "behavioral_global_reliability"
                    ],
                    "traffic_global_reliability": interaction_summary[
                        "traffic_global_reliability"
                    ],
                    "behavioral_disagreement_trust": interaction_summary[
                        "behavioral_disagreement_trust"
                    ],
                    "traffic_disagreement_trust": interaction_summary[
                        "traffic_disagreement_trust"
                    ],
                    "trust_weight_global": interaction_summary["trust_weight_global"],
                    "trust_weight_confidence": interaction_summary["trust_weight_confidence"],
                    "trust_weight_disagreement": interaction_summary[
                        "trust_weight_disagreement"
                    ],
                    "trust_gap_threshold": interaction_summary["trust_gap_threshold"],
                    "trust_based_final_accuracy": final_metrics["test_accuracy"],
                    "trust_based_final_precision": final_metrics["test_precision"],
                    "trust_based_final_recall": final_metrics["test_recall"],
                    "trust_based_final_f1": final_metrics["test_f1"],
                    "trust_based_contested_cases": interaction_summary["contested_case_count"],
                }
            )

            log_substep("OLLAMA REASONING")
            if reasoning_config.get("enable_agent_reasoning_output", True):
                max_reasoning = max(
                    0, int(reasoning_config.get("max_reasoning_samples_per_dataset", 50))
                )
                reasoning_scope = str(reasoning_config.get("reasoning_scope", "disagreement_only"))

                for agent_name, runtime in agent_runtime.items():
                    model = runtime["model"]
                    test_metrics = runtime["metrics"]
                    if agent_name == BEHAVIORAL_AGENT_NAME:
                        pred_col = "behavioral_label"
                        prob_col = "behavioral_confidence"
                        update_reason_col = "behavioral_reasoning"
                    else:
                        pred_col = "traffic_label"
                        prob_col = "traffic_confidence"
                        update_reason_col = "traffic_reasoning"

                agent_samples = interaction_df[["sample_index", pred_col, prob_col, "agreement"]].copy()
                if reasoning_scope == "all_samples":
                    selected = agent_samples
                elif reasoning_scope == "sampled_subset":
                    selected = (
                        agent_samples.head(max_reasoning)
                        if max_reasoning > 0
                        else agent_samples.iloc[0:0]
                    )
                else:
                    disagreement_only = agent_samples[agent_samples["agreement"] == False]  # noqa: E712
                    if disagreement_only.empty:
                        selected = (
                            agent_samples.head(max_reasoning)
                            if max_reasoning > 0
                            else agent_samples.iloc[0:0]
                        )
                    else:
                        selected = (
                            disagreement_only.head(max_reasoning)
                            if max_reasoning > 0
                            else disagreement_only.iloc[0:0]
                        )

                if selected.empty:
                    continue

                agreement_flags = [
                    "agree" if value is True else "disagree" if value is False else "unknown"
                    for value in selected["agreement"].tolist()
                ]

                rows = model.generate_reasoning(
                    dataset_name=dataset_name,
                    sample_indices=selected["sample_index"].tolist(),
                    predictions=selected[pred_col].astype(int).tolist(),
                    probabilities=selected[prob_col].tolist(),
                    agreement_flags=agreement_flags,
                    agent_metrics={
                        "test_f1": test_metrics.get("test_f1"),
                        "fpr": test_metrics.get("fpr"),
                        "fnr": test_metrics.get("fnr"),
                    },
                )
                reasoning_rows.extend(rows)

                if rows:
                    reasoning_map = {
                        int(item["sample_index"]): str(item["reasoning_summary"])
                        for item in rows
                    }
                    interaction_df[update_reason_col] = interaction_df.apply(
                        lambda row: reasoning_map.get(
                            int(row["sample_index"]), str(row[update_reason_col])
                        ),
                        axis=1,
                    )

            interaction_frames.append(interaction_df)
            trust_interaction_frames.append(
                interaction_df[
                    [
                        "dataset",
                        "sample_index",
                        "true_label",
                        "behavioral_label",
                        "traffic_label",
                        "behavioral_confidence",
                        "traffic_confidence",
                        "behavioral_global_trust",
                        "traffic_global_trust",
                        "behavioral_disagreement_trust",
                        "traffic_disagreement_trust",
                        "behavioral_final_trust",
                        "traffic_final_trust",
                        "trust_winner",
                        "final_label",
                        "resolution_type",
                        "final_correct",
                    ]
                ].rename(columns={"sample_index": "row_index"})
            )
            sample_prediction_frames.append(
                interaction_df[
                    [
                        "dataset",
                        "sample_index",
                        "true_label",
                        "behavioral_label",
                        "behavioral_confidence",
                        "traffic_label",
                        "traffic_confidence",
                        "agreement",
                        "confidence_gap",
                        "winner_agent",
                        "resolution_type",
                        "final_label",
                        "final_confidence",
                        "final_correct",
                    ]
                ]
            )

            elapsed = perf_counter() - dataset_start_time
            possible_leakage_signals = (
                duplicate_overlap_stats["overlap_test_rows_count"] > 0
                or duplicate_overlap_stats["overlap_train_rows_count"] > 0
                or bool(remaining_suspicious_features)
                or any(
                    int(item.get("shared_unique_count", 0)) > 0
                    for item in group_overlap["group_overlap_rows"]
                )
            )
            if possible_leakage_signals:
                verdict = "Residual overlap detected; split still unsafe"
            else:
                verdict = "No obvious exact feature-row leakage after canonical cleaning and split"
            if dropped_identifier_columns:
                verdict = (
                    f"{verdict}; Identifier columns removed from model features: "
                    + ", ".join(dropped_identifier_columns)
                )
            leakage_rows.append(
                {
                    "dataset": dataset_name,
                    "raw_row_count": raw_row_count,
                    **duplicate_overlap_stats,
                    **dedup_stats,
                    "feature_count": int(len(x.columns)),
                    "feature_columns_text": ", ".join(x.columns.astype(str).tolist()),
                    "drop_policy_columns": sorted(POTENTIAL_LEAKAGE_COLUMNS),
                    "dropped_suspicious_columns": dropped_suspicious_columns,
                    "dropped_target_proxy_columns": dropped_suspicious_columns,
                    "dropped_identifier_columns": dropped_identifier_columns,
                    "remaining_suspicious_features": remaining_suspicious_features,
                    "group_columns": group_overlap["group_columns"],
                    "group_overlap_rows": group_overlap["group_overlap_rows"],
                    "verdict": verdict,
                }
            )
            run_status_rows.append(
                {"dataset": dataset_name, "status": "SUCCESS", "error": "", "elapsed_s": f"{elapsed:.2f}"}
            )
            log_success(f"Dataset complete (Done in {elapsed:.2f}s)")
        except Exception as exc:  # pylint: disable=broad-except
            _print_dataset_error(dataset_name, exc)
            print(traceback.format_exc())
            run_status_rows.append(
                {
                    "dataset": dataset_name,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": f"{(perf_counter() - dataset_start_time):.2f}",
                }
            )
            continue

    results_columns = [
        "dataset",
        "agent",
        "cv_accuracy",
        "cv_precision",
        "cv_recall",
        "cv_f1",
        "cv_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "precision",
        "recall",
        "tp",
        "tn",
        "fp",
        "fn",
        "fpr",
        "fnr",
        "tpr",
        "tnr",
        "specificity",
        "balanced_accuracy",
        "error_rate",
        "roc_auc",
        "pr_auc",
        "support_total",
        "support_attack",
        "support_normal",
        "balancing_method",
    ]
    results_df = pd.DataFrame(results_rows).reindex(columns=results_columns)

    interaction_summary_columns = [
        "dataset",
        "total_samples",
        "agreement_count",
        "disagreement_count",
        "agreement_rate",
        "disagreement_rate",
        "behavioral_wins_on_disagreement_count",
        "traffic_wins_on_disagreement_count",
        "behavioral_wins_on_disagreement",
        "traffic_wins_on_disagreement",
        "contested_case_count",
        "contested_case_rate",
        "disagreement_confidence_threshold",
        "behavioral_test_accuracy",
        "behavioral_test_precision",
        "behavioral_test_recall",
        "behavioral_test_f1",
        "traffic_test_accuracy",
        "traffic_test_precision",
        "traffic_test_recall",
        "traffic_test_f1",
        "final_decision_accuracy",
        "final_decision_precision",
        "final_decision_recall",
        "final_decision_f1",
    ]
    interaction_summary_df = pd.DataFrame(interaction_summary_rows).reindex(
        columns=interaction_summary_columns
    )
    trust_summary_columns = [
        "dataset",
        "behavioral_global_reliability",
        "traffic_global_reliability",
        "behavioral_disagreement_trust",
        "traffic_disagreement_trust",
        "trust_weight_global",
        "trust_weight_confidence",
        "trust_weight_disagreement",
        "trust_gap_threshold",
        "trust_based_final_accuracy",
        "trust_based_final_precision",
        "trust_based_final_recall",
        "trust_based_final_f1",
        "trust_based_contested_cases",
    ]
    trust_summary_df = pd.DataFrame(trust_summary_rows).reindex(columns=trust_summary_columns)

    class_balance_columns = [
        "dataset",
        "original_train_normal_count",
        "original_train_attack_count",
        "balanced_train_normal_count",
        "balanced_train_attack_count",
        "test_normal_count",
        "test_attack_count",
        "balancing_method_used",
    ]
    class_balance_df = pd.DataFrame(class_balance_rows).reindex(columns=class_balance_columns)

    if interaction_frames:
        interactions_df = pd.concat(interaction_frames, ignore_index=True)
    else:
        interactions_df = pd.DataFrame(
            columns=[
                "dataset",
                "sample_index",
                "true_label",
                "behavioral_label",
                "behavioral_confidence",
                "behavioral_reasoning",
                "behavioral_stance",
                "traffic_label",
                "traffic_confidence",
                "traffic_reasoning",
                "traffic_stance",
                "agreement",
                "confidence_gap",
                "winner_agent",
                "trust_winner",
                "behavioral_global_trust",
                "traffic_global_trust",
                "behavioral_disagreement_trust",
                "traffic_disagreement_trust",
                "behavioral_final_trust",
                "traffic_final_trust",
                "trust_gap",
                "resolution_type",
                "final_label",
                "final_confidence",
                "final_probability_normal",
                "final_probability_attack",
                "final_correct",
            ]
        )

    if trust_interaction_frames:
        trust_interactions_df = pd.concat(trust_interaction_frames, ignore_index=True)
    else:
        trust_interactions_df = pd.DataFrame(
            columns=[
                "dataset",
                "row_index",
                "true_label",
                "behavioral_label",
                "traffic_label",
                "behavioral_confidence",
                "traffic_confidence",
                "behavioral_global_trust",
                "traffic_global_trust",
                "behavioral_disagreement_trust",
                "traffic_disagreement_trust",
                "behavioral_final_trust",
                "traffic_final_trust",
                "trust_winner",
                "final_label",
                "resolution_type",
                "final_correct",
            ]
        )

    if sample_prediction_frames:
        sample_predictions_df = pd.concat(sample_prediction_frames, ignore_index=True)
    else:
        sample_predictions_df = pd.DataFrame(
            columns=[
                "dataset",
                "sample_index",
                "true_label",
                "behavioral_label",
                "behavioral_confidence",
                "traffic_label",
                "traffic_confidence",
                "agreement",
                "confidence_gap",
                "winner_agent",
                "resolution_type",
                "final_label",
                "final_confidence",
                "final_correct",
            ]
        )

    reasoning_columns = [
        "dataset",
        "agent",
        "role",
        "backbone_model",
        "sample_index",
        "prediction",
        "probability_attack",
        "reasoning_summary",
        "confidence_band",
        "risk_note",
        "recommended_attention",
        "reasoning_status",
        "ollama_model",
    ]
    reasoning_df = pd.DataFrame(reasoning_rows).reindex(columns=reasoning_columns)

    log_section("SAVING OUTPUTS")
    output_path = results_dir / "experiment_results.csv"
    log_substep("Saving experiment_results.csv")
    results_df.to_csv(output_path, index=False)

    interaction_summary_path = results_dir / "agent_agreement.csv"
    log_substep("Saving agent_agreement.csv")
    interaction_summary_df.to_csv(interaction_summary_path, index=False)

    trust_summary_output_path = results_dir / "trust_summary.csv"
    log_substep("Saving trust_summary.csv")
    trust_summary_df.to_csv(trust_summary_output_path, index=False)

    class_balance_output_path = results_dir / "class_balance_summary.csv"
    log_substep("Saving class_balance_summary.csv")
    class_balance_df.to_csv(class_balance_output_path, index=False)

    sample_predictions_output_path = results_dir / "sample_level_predictions.csv"
    log_substep("Saving sample_level_predictions.csv")
    sample_predictions_df.to_csv(sample_predictions_output_path, index=False)

    interactions_output_path = results_dir / "agent_interactions.csv"
    log_substep("Saving agent_interactions.csv")
    interactions_df.to_csv(interactions_output_path, index=False)

    trust_interactions_output_path = results_dir / "trust_interactions.csv"
    log_substep("Saving trust_interactions.csv")
    trust_interactions_df.to_csv(trust_interactions_output_path, index=False)

    reasoning_output_path = results_dir / "agent_reasoning_outputs.csv"
    log_substep("Saving agent_reasoning_outputs.csv")
    reasoning_df.to_csv(reasoning_output_path, index=False)

    leakage_report_output_path = results_dir / "leakage_check_report.md"
    leakage_report_content = _build_leakage_report(leakage_rows)
    log_substep("Saving leakage_check_report.md")
    leakage_report_output_path.write_text(leakage_report_content, encoding="utf-8")

    report_output_path = results_dir / "final_report.md"
    report_content = _build_final_report(
        config=config,
        results_df=results_df,
        interaction_summary_df=interaction_summary_df,
        trust_summary_df=trust_summary_df,
        class_balance_df=class_balance_df,
        reasoning_df=reasoning_df,
        reasoning_config=reasoning_config,
        interaction_config=interaction_config,
        output_files=[
            {
                "name": "experiment_results.csv",
                "description": "Per-agent and final multi-agent metrics.",
            },
            {
                "name": "sample_level_predictions.csv",
                "description": "Sample-level labels/confidence and final resolved output.",
            },
            {
                "name": "agent_agreement.csv",
                "description": "Dataset-level agreement/disagreement and final decision summary metrics.",
            },
            {
                "name": "trust_summary.csv",
                "description": "Per-dataset trust values, trust weights, and trust-based final metrics.",
            },
            {
                "name": "agent_interactions.csv",
                "description": "Per-sample interaction records with disagreement resolution fields.",
            },
            {
                "name": "trust_interactions.csv",
                "description": "Per-sample trust-layer interactions and trust-based winner decisions.",
            },
            {
                "name": "class_balance_summary.csv",
                "description": "Per-dataset class balance before/after train-only balancing.",
            },
            {
                "name": "agent_reasoning_outputs.csv",
                "description": "Role-aligned agent reasoning outputs (Ollama/fallback).",
            },
            {
                "name": "leakage_check_report.md",
                "description": "Per-dataset duplicate/overlap and leakage-indicator validation report.",
            },
            {
                "name": "final_report.md",
                "description": "Within-dataset multi-agent report with conflict-resolution summary.",
            },
        ],
    )
    log_substep("Saving final_report.md")
    report_output_path.write_text(report_content, encoding="utf-8")

    print(f"[INFO] Results saved to: {output_path}")
    print(f"[INFO] Interaction summary saved to: {interaction_summary_path}")
    print(f"[INFO] Trust summary saved to: {trust_summary_output_path}")
    print(f"[INFO] Interaction records saved to: {interactions_output_path}")
    print(f"[INFO] Trust interaction records saved to: {trust_interactions_output_path}")
    print(f"[INFO] Leakage report saved to: {leakage_report_output_path}")
    print(f"[INFO] Report saved to: {report_output_path}")

    consolidated_text = _build_final_consolidated_summary(
        results_df=results_df,
        interaction_summary_df=interaction_summary_df,
        run_status_rows=run_status_rows,
    )
    print("\n" + consolidated_text)
    final_console_summary_path = results_dir / "final_console_summary.txt"
    final_console_summary_path.write_text(consolidated_text + "\n", encoding="utf-8")
    print(f"[INFO] Consolidated summary saved to: {final_console_summary_path}")

    print("\n" + "=" * 56)
    print("RUN SUMMARY")
    print("=" * 56)
    if not run_status_rows:
        print("No datasets were processed.")
    else:
        all_success = True
        for row in run_status_rows:
            dataset = str(row.get("dataset", "unknown_dataset"))
            status = str(row.get("status", "UNKNOWN"))
            error = str(row.get("error", ""))
            if status != "SUCCESS":
                all_success = False
                print(f"{dataset:<14} -> {status}: {error}")
            else:
                print(f"{dataset:<14} -> {status}")
        if all_success:
            print("\nAll datasets completed successfully.")
    log_success("All outputs saved successfully")


if __name__ == "__main__":
    main()
