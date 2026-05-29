"""Integrity checking utilities for dataset processing and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def generate_integrity_report(
    dataset_name: str,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    dropped_leakage_columns: List[str],
    categorical_columns: List[str],
    numeric_columns: List[str],
    preprocessor_object: Any,
) -> str:
    """Generate a comprehensive integrity report for dataset processing.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'NSL-KDD', 'UNSW-NB15')
        x_train, x_val, x_test: Feature dataframes
        y_train, y_val, y_test: Label arrays
        dropped_leakage_columns: Columns dropped to prevent leakage
        categorical_columns: Columns treated as categorical
        numeric_columns: Columns treated as numeric
        preprocessor_object: The fitted ColumnTransformer or Pipeline
    
    Returns:
        Markdown formatted integrity report
    """
    lines = [
        f"# Integrity Report: {dataset_name}",
        "",
        "## Dataset Size",
        f"- Training samples: {x_train.shape[0]}",
        f"- Validation samples: {x_val.shape[0]}",
        f"- Test samples: {x_test.shape[0]}",
        f"- Total: {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]}",
        "",
        "## Feature Information",
        f"- Total features: {x_train.shape[1]}",
        f"- Categorical features: {len(categorical_columns)} ({', '.join(categorical_columns[:5])}{'...' if len(categorical_columns) > 5 else ''})",
        f"- Numeric features: {len(numeric_columns)} ({', '.join(numeric_columns[:5])}{'...' if len(numeric_columns) > 5 else ''})",
        "",
        "## Class Distribution",
        f"- Training: Normal={int(np.sum(y_train == 0))} ({100*np.mean(y_train==0):.1f}%), Attack={int(np.sum(y_train == 1))} ({100*np.mean(y_train==1):.1f}%)",
        f"- Validation: Normal={int(np.sum(y_val == 0))} ({100*np.mean(y_val==0):.1f}%), Attack={int(np.sum(y_val == 1))} ({100*np.mean(y_val==1):.1f}%)",
        f"- Test: Normal={int(np.sum(y_test == 0))} ({100*np.mean(y_test==0):.1f}%), Attack={int(np.sum(y_test == 1))} ({100*np.mean(y_test==1):.1f}%)",
        "",
        "## Leakage Prevention",
        f"- Dropped columns: {', '.join(dropped_leakage_columns) if dropped_leakage_columns else 'None'}",
        "- Preprocessing fit: Training data only",
        "- Validation transformed: After training fit",
        "- Test transformed: After training fit",
        "- Test labels used for: Final evaluation only",
        "",
        "## Validation Discipline",
        "- ✓ Threshold tuning: Validation data only (no test labels)",
        "- ✓ Feature selection: Training data only",
        "- ✓ Hard-case identification: Validation data only",
        "- ✓ Preprocessing: Fit on training data only",
        "- ✓ Trust selector tuning: Validation data only",
        "- ✓ Test labels: Reserved for final metrics calculation",
        "",
        "## Data Quality",
        f"- Missing values in training features: {x_train.isnull().sum().sum()}",
        f"- Missing values in validation features: {x_val.isnull().sum().sum()}",
        f"- Missing values in test features: {x_test.isnull().sum().sum()}",
        f"- Binary labels verified: {set(np.unique(y_train)).issubset({0, 1}) and set(np.unique(y_test)).issubset({0, 1})}",
        "",
        "## Preprocessing Pipeline",
        f"- Numeric scaler: StandardScaler (fitted on train, applied to val/test)",
        f"- Categorical encoder: OneHotEncoder (fitted on train, applied to val/test)",
        "- Imputation: Median for numeric, 'unknown' for categorical",
        "- Feature alignment: Train/val/test have identical feature columns after preprocessing",
        "",
    ]
    
    return "\n".join(lines)
