"""Data loading helpers for IDS experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml


POTENTIAL_LEAKAGE_COLUMNS = {
    "attack_cat",
    "attack_category",
    "class",
    "target",
    "binary_label",
    "label_binary",
}


def load_yaml_config(config_path: str | Path) -> Dict:
    """Load YAML configuration from disk."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a dictionary.")

    return config


def load_dataset(dataset_config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from CSV and split into features (X) and labels (y)."""
    dataset_path = dataset_config.get("path")
    label_column = dataset_config.get("label_column")

    if not dataset_path:
        raise ValueError("Dataset config is missing 'path'.")
    if not label_column:
        raise ValueError("Dataset config is missing 'label_column'.")

    csv_path = Path(dataset_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Could not read CSV '{csv_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in dataset '{csv_path}'."
        )

    # Drop additional target-like columns that can leak label information.
    lowered_to_original = {column.lower(): column for column in df.columns}
    leakage_cols = []
    for candidate in POTENTIAL_LEAKAGE_COLUMNS:
        if candidate == label_column.lower():
            continue
        if candidate in lowered_to_original:
            leakage_cols.append(lowered_to_original[candidate])

    if leakage_cols:
        print(
            "[WARN] Dropping potential target-leakage columns: "
            + ", ".join(leakage_cols)
        )
        df = df.drop(columns=leakage_cols)

    # Verify we don't still have duplicate label-like columns.
    label_like = [column for column in df.columns if column.strip().lower() == "label"]
    if len(label_like) > 1:
        raise ValueError(
            f"Dataset '{csv_path}' contains duplicate label columns: {label_like}"
        )

    x = df.drop(columns=[label_column]).copy()
    y = df[label_column].copy()

    if x.shape[1] == 0:
        raise ValueError(
            f"Dataset '{csv_path}' has no feature columns after removing label '{label_column}'."
        )
    if y.empty:
        raise ValueError(f"Label column '{label_column}' is empty in dataset '{csv_path}'.")

    return x, y
