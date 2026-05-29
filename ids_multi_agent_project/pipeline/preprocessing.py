"""Preprocessing utilities for IDS datasets."""

from __future__ import annotations

from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Train-derived quantile clipper for numeric stability."""

    def __init__(
        self,
        enabled: bool = True,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.enabled = bool(enabled)
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, x, y=None):
        if not self.enabled:
            return self

        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        lq = min(max(self.lower_quantile, 0.0), 0.5)
        uq = max(min(self.upper_quantile, 1.0), 0.5)
        if lq >= uq:
            lq, uq = 0.01, 0.99

        with np.errstate(invalid="ignore"):
            self.lower_bounds_ = np.nanquantile(arr, lq, axis=0)
            self.upper_bounds_ = np.nanquantile(arr, uq, axis=0)
        return self

    def transform(self, x):
        if not self.enabled or self.lower_bounds_ is None or self.upper_bounds_ is None:
            return x

        arr = np.asarray(x, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            finite_mask = np.isfinite(col)
            if not finite_mask.any():
                continue
            col_min = self.lower_bounds_[col_idx]
            col_max = self.upper_bounds_[col_idx]
            col[finite_mask] = np.clip(col[finite_mask], col_min, col_max)
            arr[:, col_idx] = col
        return arr


def infer_feature_type_columns(
    x: pd.DataFrame, categorical_columns: List[str]
) -> Tuple[List[str], List[str]]:
    """Infer categorical and numeric feature columns."""
    configured_categorical = list(categorical_columns or [])
    missing_categorical = [col for col in configured_categorical if col not in x.columns]
    if missing_categorical:
        warnings.warn(
            "Configured categorical column(s) not found and will be ignored: "
            + ", ".join(missing_categorical),
            UserWarning,
        )

    valid_configured = [col for col in configured_categorical if col in x.columns]
    auto_detected = x.select_dtypes(include=["object", "category"]).columns.tolist()

    valid_categorical: List[str] = []
    for col in valid_configured + auto_detected:
        if col not in valid_categorical:
            valid_categorical.append(col)

    numeric_columns = [col for col in x.columns if col not in valid_categorical]
    return valid_categorical, numeric_columns


def build_preprocessing_pipeline(
    x: pd.DataFrame, categorical_columns: List[str], config: Dict
) -> ColumnTransformer:
    """Build a preprocessing pipeline using ColumnTransformer.

    Behavior:
    - Listed categorical columns are treated as categorical.
    - If listed categorical columns are missing, a clear warning is emitted.
    - Object/category dtype columns are auto-detected as categorical.
    - Remaining columns are treated as numeric.
    """
    handle_missing = config.get("handle_missing_values", True)
    encode_categorical = config.get("encode_categorical_features", True)
    scale_numeric = config.get("scale_numeric_features", True)

    valid_categorical, numeric_columns = infer_feature_type_columns(x, categorical_columns)

    transformers = []

    if numeric_columns:
        numeric_steps = []
        if handle_missing:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        if numeric_steps:
            numeric_pipeline = Pipeline(steps=numeric_steps)
            transformers.append(("num", numeric_pipeline, numeric_columns))
        else:
            transformers.append(("num", "passthrough", numeric_columns))

    if valid_categorical:
        categorical_steps = []
        if handle_missing:
            categorical_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="unknown")))

        if encode_categorical:
            categorical_steps.append(
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            )
            categorical_pipeline = Pipeline(steps=categorical_steps)
            transformers.append(("cat", categorical_pipeline, valid_categorical))
        else:
            transformers.append(("cat", "passthrough", valid_categorical))

    if not transformers:
        raise ValueError("No valid columns available for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_traffic_preprocessing_pipeline(
    x: pd.DataFrame, categorical_columns: List[str], config: Dict
) -> ColumnTransformer:
    """Build a numerically stable preprocessing pipeline for TrafficAnalysisAgent."""
    clip_enabled = bool(config.get("traffic_numeric_clipping_enabled", True))
    clip_lower = float(config.get("traffic_clip_lower_quantile", 0.01))
    clip_upper = float(config.get("traffic_clip_upper_quantile", 0.99))

    valid_categorical, numeric_columns = infer_feature_type_columns(x, categorical_columns)
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "clipper",
                    QuantileClipper(
                        enabled=clip_enabled,
                        lower_quantile=clip_lower,
                        upper_quantile=clip_upper,
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if valid_categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, valid_categorical))

    if not transformers:
        raise ValueError("No valid columns available for TrafficAnalysisAgent preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def sanitize_feature_values(
    x: pd.DataFrame, categorical_columns: List[str]
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Sanitize feature values before preprocessing.

    Steps:
    - Determine numeric columns using configured categorical columns + dtype inference.
    - Coerce numeric columns to numeric dtype where possible.
    - Replace inf/-inf and any non-finite numeric values with NaN.

    Returns:
        Tuple of (sanitized dataframe, stats dictionary).
    """
    sanitized = x.copy()

    configured_categorical = list(categorical_columns or [])
    valid_configured = [col for col in configured_categorical if col in sanitized.columns]
    auto_detected = sanitized.select_dtypes(include=["object", "category"]).columns.tolist()

    final_categorical = []
    for col in valid_configured + auto_detected:
        if col not in final_categorical:
            final_categorical.append(col)

    numeric_columns = [col for col in sanitized.columns if col not in final_categorical]

    inf_count_before = 0
    non_finite_replaced = 0

    for col in numeric_columns:
        series = pd.to_numeric(sanitized[col], errors="coerce")
        arr = series.to_numpy(dtype=np.float64, copy=False)

        finite_mask = np.isfinite(arr)
        inf_mask = np.isinf(arr)

        inf_count_before += int(np.sum(inf_mask))
        non_finite_replaced += int(np.sum(~finite_mask))

        # Replace all non-finite values (inf, -inf, NaN) with NaN for imputation.
        series[~finite_mask] = np.nan
        sanitized[col] = series

    stats = {
        "numeric_columns_checked": len(numeric_columns),
        "inf_values_found": inf_count_before,
        "non_finite_values_replaced_with_nan": non_finite_replaced,
    }
    return sanitized, stats
