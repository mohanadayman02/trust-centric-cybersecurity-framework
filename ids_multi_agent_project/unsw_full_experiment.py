import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_DATA_PATH = Path("data/unsw_nb15.csv")
DEFAULT_OUTPUT_PATH = Path("unsw_experiment_results.csv")


class BehavioralAgent:
    def __init__(self) -> None:
        self.model = SVC(probability=True, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class TrafficAgent:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def load_data(data_path: Path, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(data_path)

    if target_col is None:
        for candidate in ["label", "Label", "target", "Target", "class", "Class"]:
            if candidate in df.columns:
                target_col = candidate
                break

    if target_col is None or target_col not in df.columns:
        raise ValueError("Target column not found. Pass --target-col explicitly.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.lower().map(
            {
                "normal": 0,
                "benign": 0,
                "0": 0,
                "attack": 1,
                "malicious": 1,
                "anomaly": 1,
                "1": 1,
            }
        ).fillna(y)

    if not pd.api.types.is_numeric_dtype(y):
        uniq = pd.Series(y).astype(str).unique().tolist()
        if len(uniq) != 2:
            raise ValueError("Target must be binary for this experiment.")
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y = pd.Series(y).astype(str).map(mapping)

    y = y.astype(int)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise ValueError("Target values must be binary {0, 1}.")

    return X, y, target_col


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )


def leakage_audit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_col: str,
) -> Dict[str, List[str]]:
    suspicious_pattern = re.compile(r"(label|class|target|attack)", re.IGNORECASE)

    suspicious_name_cols = [c for c in X_train.columns if suspicious_pattern.search(c)]

    id_like_cols = [
        c
        for c in X_train.columns
        if c.strip().lower() in {"id", "flow_id", "record_id", "sample_id", "index"}
    ]

    duplicate_mask = X_train.T.duplicated(keep="first")
    duplicate_cols = X_train.columns[duplicate_mask].tolist()

    deterministic_cols: List[str] = []
    y_train_arr = y_train.to_numpy()
    for col in X_train.columns:
        s = X_train[col]
        nunique = s.nunique(dropna=False)
        if nunique <= 1 or nunique > 100:
            continue

        s_key = s.astype(str).fillna("<NA>")
        grouped = pd.DataFrame({"k": s_key, "y": y_train_arr}).groupby("k")["y"].nunique()
        if grouped.max() != 1:
            continue

        mapping = pd.DataFrame({"k": s_key, "y": y_train_arr}).groupby("k")["y"].first()
        pred = s_key.map(mapping).to_numpy()
        if np.array_equal(pred, y_train_arr):
            deterministic_cols.append(col)

    target_like_dup_cols = [c for c in X_train.columns if c.strip().lower() == target_col.strip().lower()]

    dropped_cols = sorted(set(id_like_cols + duplicate_cols + deterministic_cols + target_like_dup_cols))

    issues: List[str] = []
    for col in deterministic_cols:
        issues.append(f"LEAKAGE SOURCE FOUND: column '{col}' has a deterministic 1-to-1 mapping to the target on train split; dropped.")
    for col in target_like_dup_cols:
        issues.append(f"LEAKAGE SOURCE FOUND: column '{col}' duplicates target semantics by name; dropped.")
    for col in id_like_cols:
        issues.append(f"LEAKAGE SOURCE FOUND: column '{col}' is an ID-like metadata field; dropped.")
    for col in duplicate_cols:
        issues.append(f"LEAKAGE SOURCE FOUND: column '{col}' is a duplicate feature column; dropped.")

    return {
        "suspicious_name_cols": suspicious_name_cols,
        "id_like_cols": id_like_cols,
        "duplicate_cols": duplicate_cols,
        "deterministic_cols": deterministic_cols,
        "target_like_dup_cols": target_like_dup_cols,
        "dropped_cols": dropped_cols,
        "issues": issues,
    }


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "tpr": recall,
        "fnr": fnr,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    return metrics


def run_stage_svm(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, evaluate(y_test, y_pred, y_prob)


def run_stage_lr(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, evaluate(y_test, y_pred, y_prob)


def run_behavioral_agent(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    agent = BehavioralAgent()
    agent.fit(X_train, y_train)
    y_pred = agent.predict(X_test)
    y_prob = agent.predict_proba(X_test)
    return agent, evaluate(y_test, y_pred, y_prob)


def run_traffic_agent(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    agent = TrafficAgent()
    agent.fit(X_train, y_train)
    y_pred = agent.predict(X_test)
    y_prob = agent.predict_proba(X_test)
    return agent, evaluate(y_test, y_pred, y_prob)


def run_multi_agent(behavioral_agent: BehavioralAgent, traffic_agent: TrafficAgent, X_test: np.ndarray, y_test: np.ndarray):
    pred_behavioral = behavioral_agent.predict(X_test)
    pred_traffic = traffic_agent.predict(X_test)
    prob_behavioral = behavioral_agent.predict_proba(X_test)
    prob_traffic = traffic_agent.predict_proba(X_test)

    avg_prob = (prob_behavioral + prob_traffic) / 2.0
    final_pred = (avg_prob >= 0.5).astype(int)

    agreement_rate = float(np.mean(pred_behavioral == pred_traffic))
    disagreement_rate = 1.0 - agreement_rate

    metrics = evaluate(y_test, final_pred, avg_prob)
    metrics["agreement_rate"] = agreement_rate
    metrics["disagreement_rate"] = disagreement_rate
    return metrics


def run_trust_layer(
    behavioral_agent: BehavioralAgent,
    traffic_agent: TrafficAgent,
    X_test: np.ndarray,
    y_test: np.ndarray,
    trust_behavioral: float,
    trust_traffic: float,
):
    pred_behavioral = behavioral_agent.predict(X_test)
    pred_traffic = traffic_agent.predict(X_test)
    prob_behavioral = behavioral_agent.predict_proba(X_test)
    prob_traffic = traffic_agent.predict_proba(X_test)

    denom = trust_behavioral + trust_traffic
    if denom == 0:
        weighted_score = (prob_behavioral + prob_traffic) / 2.0
    else:
        weighted_score = (prob_behavioral * trust_behavioral + prob_traffic * trust_traffic) / denom

    final_pred = (weighted_score >= 0.5).astype(int)

    agreement_rate = float(np.mean(pred_behavioral == pred_traffic))
    disagreement_rate = 1.0 - agreement_rate

    metrics = evaluate(y_test, final_pred, weighted_score)
    metrics["agreement_rate"] = agreement_rate
    metrics["disagreement_rate"] = disagreement_rate
    return metrics


def _class_distribution(y: pd.Series) -> Dict[int, int]:
    return y.value_counts().sort_index().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    print("[START] Loading data...", flush=True)
    X, y, target_col = load_data(args.data_path, args.target_col)
    print(f"[OK] Data loaded: {len(X)} rows, {X.shape[1]} raw features", flush=True)

    print("[START] Single split (70/30, stratify, random_state=42)...", flush=True)
    X_train_raw, X_test_raw, y_train_s, y_test_s = split_data(X, y)
    print("[OK] Split complete", flush=True)

    audit = leakage_audit(X_train_raw, y_train_s, target_col)
    for msg in audit["issues"]:
        print(msg, flush=True)

    dropped_cols = audit["dropped_cols"]
    if dropped_cols:
        X_train_raw = X_train_raw.drop(columns=dropped_cols, errors="ignore")
        X_test_raw = X_test_raw.drop(columns=dropped_cols, errors="ignore")

    print("[START] Preprocessing fit on train only (impute/encode/scale)...", flush=True)
    X_train, X_test, _ = preprocess(X_train_raw, X_test_raw)
    y_train = y_train_s.to_numpy()
    y_test = y_test_s.to_numpy()
    print("[OK] Preprocessing complete", flush=True)

    print("[DEBUG] target column used:", target_col, flush=True)
    print("[DEBUG] final feature count:", X_train.shape[1], flush=True)
    print("[DEBUG] dropped columns:", dropped_cols if dropped_cols else "None", flush=True)
    print("[DEBUG] suspicious-name columns:", audit["suspicious_name_cols"] if audit["suspicious_name_cols"] else "None", flush=True)
    print("[DEBUG] train class distribution:", _class_distribution(y_train_s), flush=True)
    print("[DEBUG] test class distribution:", _class_distribution(y_test_s), flush=True)

    results: List[Dict[str, float]] = []

    print("[STAGE 1/6] Training SVM...", flush=True)
    _, svm_metrics = run_stage_svm(X_train, X_test, y_train, y_test)
    results.append({"stage": "Stage 1 - SVM", **svm_metrics})
    print("[STAGE 1/6] Done", flush=True)

    print("[STAGE 2/6] Training Logistic Regression...", flush=True)
    _, lr_metrics = run_stage_lr(X_train, X_test, y_train, y_test)
    results.append({"stage": "Stage 2 - Logistic Regression", **lr_metrics})
    print("[STAGE 2/6] Done", flush=True)

    print("[STAGE 3/6] Running Behavioral Agent...", flush=True)
    behavioral_agent, behavioral_metrics = run_behavioral_agent(X_train, X_test, y_train, y_test)
    results.append({"stage": "Stage 3 - Behavioral Agent", **behavioral_metrics})
    print("[STAGE 3/6] Done", flush=True)

    print("[STAGE 4/6] Running Traffic Agent...", flush=True)
    traffic_agent, traffic_metrics = run_traffic_agent(X_train, X_test, y_train, y_test)
    results.append({"stage": "Stage 4 - Traffic Agent", **traffic_metrics})
    print("[STAGE 4/6] Done", flush=True)

    print("[STAGE 5/6] Running Multi-Agent without trust...", flush=True)
    multi_agent_metrics = run_multi_agent(behavioral_agent, traffic_agent, X_test, y_test)
    results.append({"stage": "Stage 5 - Multi-Agent (No Trust)", **multi_agent_metrics})
    print("[STAGE 5/6] Done", flush=True)

    print("[STAGE 6/6] Running Multi-Agent with trust...", flush=True)
    trust_behavioral = behavioral_metrics["f1"]
    trust_traffic = traffic_metrics["f1"]
    trust_metrics = run_trust_layer(
        behavioral_agent,
        traffic_agent,
        X_test,
        y_test,
        trust_behavioral=trust_behavioral,
        trust_traffic=trust_traffic,
    )
    results.append({"stage": "Stage 6 - Multi-Agent (With Trust)", **trust_metrics})
    print("[STAGE 6/6] Done", flush=True)

    results_df = pd.DataFrame(results)
    print("[DONE] Final results table:", flush=True)
    print(results_df.to_string(index=False))

    suspicious_perfect = results_df[
        (results_df["accuracy"] >= 0.9999)
        & (results_df["roc_auc"].notna())
        & (np.isclose(results_df["roc_auc"], 1.0))
    ]
    if not suspicious_perfect.empty:
        for stage_name in suspicious_perfect["stage"].tolist():
            print(
                f"[WARNING] Suspiciously perfect metrics detected in '{stage_name}' (accuracy >= 0.9999 and roc_auc == 1.0).",
                flush=True,
            )

    print(f"[SAVE] Writing CSV to {args.output_path} ...", flush=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_path, index=False)
    print("[OK] CSV saved", flush=True)


if __name__ == "__main__":
    main()
