"""Data loading helpers for IDS experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
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

TON_LABEL_CANDIDATES = ["label", "Label", "type", "attack", "attack_cat", "normal", "class"]
CIC_LABEL_CANDIDATES = ["Label", "label", "Attack", "Class", "class"]


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


def load_unsw_nb15_dataset(
    dataset_config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load UNSW-NB15 dataset with official train/test split.
    
    Returns:
        Tuple of (x_train, x_test, y_train, y_test)
    """
    train_path = dataset_config.get("train_path")
    test_path = dataset_config.get("test_path")
    label_column = dataset_config.get("label_column", "label")

    if not train_path or not test_path:
        raise ValueError("UNSW loader requires both 'train_path' and 'test_path' in config.")

    train_path_obj = Path(train_path)
    test_path_obj = Path(test_path)

    if not train_path_obj.exists():
        raise FileNotFoundError(
            f"UNSW training set not found: {train_path_obj}\n"
            f"Expected path: data/raw/unsw_nb15/UNSW_NB15_training-set.csv"
        )
    if not test_path_obj.exists():
        raise FileNotFoundError(
            f"UNSW testing set not found: {test_path_obj}\n"
            f"Expected path: data/raw/unsw_nb15/UNSW_NB15_testing-set.csv"
        )

    try:
        df_train = pd.read_csv(train_path_obj)
        df_test = pd.read_csv(test_path_obj)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Could not read UNSW CSV files: {exc}") from exc

    if df_train.empty or df_test.empty:
        raise ValueError("UNSW training or testing set is empty.")

    if label_column not in df_train.columns or label_column not in df_test.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in UNSW dataset."
        )

    # Drop leakage columns: id, attack_cat, duplicates
    leakage_candidates = {"id", "attack_cat", "attack_category", "class", "target"}
    cols_to_drop = [col for col in leakage_candidates if col in df_train.columns]
    
    if cols_to_drop:
        print(f"[INFO] Dropping UNSW leakage columns from train/test: {cols_to_drop}")
        df_train = df_train.drop(columns=cols_to_drop)
        df_test = df_test.drop(columns=cols_to_drop)

    # Extract labels
    y_train = df_train[label_column].copy()
    y_test = df_test[label_column].copy()

    # Ensure binary labels (0 and 1)
    if not set(y_train.unique()).issubset({0, 1}) or not set(y_test.unique()).issubset({0, 1}):
        raise ValueError(
            f"UNSW labels must be binary (0/1). Found unique values in train: {y_train.unique()}, test: {y_test.unique()}"
        )

    # Extract features
    x_train = df_train.drop(columns=[label_column]).copy()
    x_test = df_test.drop(columns=[label_column]).copy()

    if x_train.shape[1] == 0 or x_test.shape[1] == 0:
        raise ValueError("UNSW dataset has no feature columns after removing labels and leakage columns.")

    # Ensure same columns between train and test
    common_cols = sorted(set(x_train.columns) & set(x_test.columns))
    if len(common_cols) < min(x_train.shape[1], x_test.shape[1]):
        print(f"[WARN] Train/test have different columns. Using common columns only.")
        x_train = x_train[common_cols]
        x_test = x_test[common_cols]

    return x_train, x_test, y_train, y_test


def get_unsw_categorical_columns(x_df: pd.DataFrame) -> list[str]:
    """Auto-detect categorical columns in UNSW dataset."""
    known_categorical = {"proto", "service", "state"}
    present_categorical = [col for col in known_categorical if col in x_df.columns]
    auto_detected = x_df.select_dtypes(include=["object", "category"]).columns.tolist()
    result = list(set(present_categorical + auto_detected))
    return sorted(result)


def _normalize_binary_label_value(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("ToN-IoT label contains missing values.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0 if float(value) == 0.0 else 1
    normalized = str(value).strip().lower()
    if normalized in {"0", "normal", "benign", "false", "no"}:
        return 0
    if normalized in {"1", "attack", "malicious", "true", "yes"}:
        return 1
    return 0 if "normal" in normalized or "benign" in normalized else 1


def _detect_ton_label_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in TON_LABEL_CANDIDATES:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise ValueError(
        "Could not detect ToN-IoT label column. Expected one of: "
        + ", ".join(TON_LABEL_CANDIDATES)
    )


def _drop_ton_leakage_columns(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    output = df.copy()
    lower_to_original = {c.lower(): c for c in output.columns}
    leakage_candidates = {"label", "attack", "attack_cat", "type", "class"}
    for candidate in leakage_candidates:
        if candidate in lower_to_original:
            original = lower_to_original[candidate]
            if original != label_column and original in output.columns:
                output = output.drop(columns=[original])
                dropped.append(original)

    identifier_like = [c for c in output.columns if c.lower() in {"id", "idx", "flow_id"} or c.lower().endswith("_id")]
    for col in identifier_like:
        if col in output.columns:
            output = output.drop(columns=[col])
            dropped.append(col)

    timestamp_like = [
        c
        for c in output.columns
        if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower() or c.lower() == "ts"
    ]
    for col in timestamp_like:
        if col in output.columns:
            output = output.drop(columns=[col])
            dropped.append(col)

    return output, sorted(set(dropped))


def _find_ton_iot_files(project_root: Path, combine_csvs: bool) -> dict[str, Any]:
    expected_paths = [
        project_root / "data" / "raw" / "ton_iot" / "*.csv",
        project_root / "data" / "ton_iot" / "*.csv",
        project_root / "data" / "ToN_IoT*.csv",
    ]
    raw_candidates = sorted((project_root / "data" / "raw" / "ton_iot").glob("*.csv"))
    flat_candidates = sorted((project_root / "data" / "ton_iot").glob("*.csv"))
    root_candidates = sorted((project_root / "data").glob("ToN_IoT*.csv"))
    all_candidates = raw_candidates + flat_candidates + root_candidates

    if not all_candidates:
        return {"found": False, "expected_paths": expected_paths}

    named = {p.name.lower(): p for p in all_candidates}
    train_file = None
    test_file = None
    for name, path in named.items():
        if "train" in name:
            train_file = path
        if "test" in name:
            test_file = path
    if train_file is not None and test_file is not None:
        return {
            "found": True,
            "mode": "official_split",
            "train_path": train_file,
            "test_path": test_file,
            "expected_paths": expected_paths,
        }

    if combine_csvs:
        return {
            "found": True,
            "mode": "single_or_combined",
            "csv_paths": all_candidates,
            "expected_paths": expected_paths,
        }

    network_like = [p for p in all_candidates if any(key in p.name.lower() for key in ["network", "netflow", "flow"])]
    chosen = network_like[0] if network_like else all_candidates[0]
    return {
        "found": True,
        "mode": "single_or_combined",
        "csv_paths": [chosen],
        "expected_paths": expected_paths,
    }


def load_ton_iot_dataset(
    project_root: str | Path,
    *,
    random_state: int = 42,
    val_size_from_train: float = 0.20,
    test_size_single_csv: float = 0.20,
    val_size_single_csv: float = 0.20,
    combine_csvs: bool = False,
) -> Dict[str, Any]:
    """Load ToN-IoT dataset with flexible file handling and binary labels."""
    from sklearn.model_selection import train_test_split

    root = Path(project_root)
    discovery = _find_ton_iot_files(root, combine_csvs=combine_csvs)
    if not discovery.get("found", False):
        expected = "\n".join([f"  - {p}" for p in discovery["expected_paths"]])
        raise FileNotFoundError(
            "ToN-IoT dataset files not found.\nExpected one of:\n" + expected
        )

    dropped_leakage_columns: list[str] = []

    if discovery["mode"] == "official_split":
        df_train = pd.read_csv(discovery["train_path"])
        df_test = pd.read_csv(discovery["test_path"])
        label_col_train = _detect_ton_label_column(df_train)
        label_col_test = _detect_ton_label_column(df_test)

        y_train_full = df_train[label_col_train].map(_normalize_binary_label_value).astype(int)
        y_test = df_test[label_col_test].map(_normalize_binary_label_value).astype(int)

        x_train_full, dropped_train = _drop_ton_leakage_columns(df_train, label_col_train)
        x_test, dropped_test = _drop_ton_leakage_columns(df_test, label_col_test)
        dropped_leakage_columns = sorted(set(dropped_train + dropped_test + [label_col_train, label_col_test]))
        x_train_full = x_train_full.drop(columns=[label_col_train], errors="ignore")
        x_test = x_test.drop(columns=[label_col_test], errors="ignore")

        common_cols = sorted(set(x_train_full.columns) & set(x_test.columns))
        x_train_full = x_train_full[common_cols].copy()
        x_test = x_test[common_cols].copy()

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=float(val_size_from_train),
            random_state=int(random_state),
            stratify=y_train_full,
        )
    else:
        csv_paths = discovery["csv_paths"]
        frames = [pd.read_csv(path) for path in csv_paths]
        combined = pd.concat(frames, ignore_index=True)
        label_col = _detect_ton_label_column(combined)
        y_all = combined[label_col].map(_normalize_binary_label_value).astype(int)
        x_all, dropped = _drop_ton_leakage_columns(combined, label_col)
        dropped_leakage_columns = sorted(set(dropped + [label_col]))
        x_all = x_all.drop(columns=[label_col], errors="ignore")

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_all,
            y_all,
            test_size=float(test_size_single_csv),
            random_state=int(random_state),
            stratify=y_all,
        )
        val_ratio_on_train_val = float(val_size_single_csv) / max(1e-8, (1.0 - float(test_size_single_csv)))
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=val_ratio_on_train_val,
            random_state=int(random_state),
            stratify=y_train_val,
        )

    categorical_columns = sorted(x_train.select_dtypes(include=["object", "category"]).columns.tolist())
    numeric_columns = [c for c in x_train.columns if c not in categorical_columns]

    return {
        "x_train": x_train.copy(),
        "x_val": x_val.copy(),
        "x_test": x_test.copy(),
        "y_train": y_train.astype(int).copy(),
        "y_val": y_val.astype(int).copy(),
        "y_test": y_test.astype(int).copy(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "dropped_leakage_columns": dropped_leakage_columns,
        "source_mode": discovery["mode"],
        "source_files": [str(p) for p in discovery.get("csv_paths", [])]
        if discovery["mode"] != "official_split"
        else [str(discovery["train_path"]), str(discovery["test_path"])],
        "expected_paths": [str(p) for p in discovery["expected_paths"]],
    }


def _detect_cic_label_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in CIC_LABEL_CANDIDATES:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise ValueError(
        "Could not detect CICIDS2017 label column. Expected one of: "
        + ", ".join(CIC_LABEL_CANDIDATES)
    )


def _map_cic_to_binary(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("CICIDS2017 label contains missing values.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0 if float(value) == 0.0 else 1
    normalized = str(value).strip().lower()
    return 0 if normalized in {"benign", "normal", "0"} else 1


def _drop_cic_leakage_and_bad_columns(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    dropped: list[str] = []
    cols_lower = {c.lower(): c for c in output.columns}

    explicit_drop = {
        "flow id",
        "flow_id",
        "source ip",
        "src ip",
        "destination ip",
        "dst ip",
        "timestamp",
        "label",
        "class",
        "attack",
    }
    for key in explicit_drop:
        if key in cols_lower:
            col = cols_lower[key]
            if col != label_column and col in output.columns:
                output = output.drop(columns=[col])
                dropped.append(col)

    for col in list(output.columns):
        series = output[col]
        if series.isna().all():
            output = output.drop(columns=[col])
            dropped.append(col)
            continue
        if series.nunique(dropna=False) <= 1 and col != label_column:
            output = output.drop(columns=[col])
            dropped.append(col)

    return output, sorted(set(dropped))


def _find_cic_csv_files(project_root: Path) -> dict[str, Any]:
    expected_paths = [
        project_root / "data" / "raw" / "cicids2017" / "*.csv",
        project_root / "data" / "cicids2017.csv",
        project_root / "data" / "processed" / "cicids2017" / "*.csv",
    ]

    processed_dir = project_root / "data" / "processed" / "cicids2017"
    processed_train = processed_dir / "train.csv"
    processed_test = processed_dir / "test.csv"
    processed = sorted(processed_dir.glob("*.csv"))
    single_candidates = [project_root / "data" / "cicids2017.csv"]
    single_candidates = [p for p in single_candidates if p.exists()]
    raw = sorted((project_root / "data" / "raw" / "cicids2017").glob("*.csv"))

    if processed_train.exists() and processed_test.exists():
        return {
            "found": True,
            "mode": "official_split",
            "train_path": processed_train,
            "test_path": processed_test,
            "expected_paths": expected_paths,
        }
    if single_candidates:
        return {"found": True, "mode": "single", "csv_paths": single_candidates, "expected_paths": expected_paths}
    if processed:
        return {"found": True, "mode": "multi", "csv_paths": processed, "expected_paths": expected_paths}
    if raw:
        return {"found": True, "mode": "multi", "csv_paths": raw, "expected_paths": expected_paths}
    return {"found": False, "expected_paths": expected_paths}


def _apply_optional_stratified_cap(
    x_df: pd.DataFrame,
    y_ser: pd.Series,
    max_samples: Optional[int],
    random_state: int,
    stratified_subsample: bool,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    if max_samples is None or int(max_samples) <= 0 or len(x_df) <= int(max_samples):
        return x_df, y_ser, False
    n = int(max_samples)
    if stratified_subsample:
        from sklearn.model_selection import train_test_split

        x_cap, _, y_cap, _ = train_test_split(
            x_df,
            y_ser,
            train_size=n,
            random_state=int(random_state),
            stratify=y_ser,
        )
        return x_cap, y_cap, True
    sampled_idx = y_ser.sample(n=n, random_state=int(random_state)).index
    return x_df.loc[sampled_idx], y_ser.loc[sampled_idx], True


def load_cicids2017_dataset(
    project_root: str | Path,
    *,
    random_state: int = 42,
    test_size_single_csv: float = 0.20,
    val_size_single_csv: float = 0.20,
    max_samples: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    stratified_subsample: bool = True,
) -> Dict[str, Any]:
    """Load CICIDS2017 from single or multi-CSV sources with binary mapping and clean splits."""
    from sklearn.model_selection import train_test_split

    root = Path(project_root)
    discovery = _find_cic_csv_files(root)
    if not discovery.get("found", False):
        expected = "\n".join([f"  - {p}" for p in discovery["expected_paths"]])
        raise FileNotFoundError("CICIDS2017 dataset files not found.\nExpected one of:\n" + expected)

    capped_total = False
    if discovery["mode"] == "official_split":
        train_df = pd.read_csv(discovery["train_path"], low_memory=False)
        test_df = pd.read_csv(discovery["test_path"], low_memory=False)
        label_col_train = _detect_cic_label_column(train_df)
        label_col_test = _detect_cic_label_column(test_df)

        y_train_full = train_df[label_col_train].map(_map_cic_to_binary).astype(int)
        y_test = test_df[label_col_test].map(_map_cic_to_binary).astype(int)

        x_train_full, dropped_train = _drop_cic_leakage_and_bad_columns(train_df, label_col_train)
        x_test, dropped_test = _drop_cic_leakage_and_bad_columns(test_df, label_col_test)
        dropped_leakage_columns = sorted(set(dropped_train + dropped_test + [label_col_train, label_col_test]))
        x_train_full = x_train_full.drop(columns=[label_col_train], errors="ignore")
        x_test = x_test.drop(columns=[label_col_test], errors="ignore")

        common_cols = sorted(set(x_train_full.columns) & set(x_test.columns))
        x_train_full = x_train_full[common_cols].copy()
        x_test = x_test[common_cols].copy()

        x_train_full = x_train_full.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan)
        x_test = x_test.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan)
        for col in x_train_full.columns:
            if x_train_full[col].dtype == object:
                x_train_full[col] = pd.to_numeric(x_train_full[col], errors="ignore")
            if x_test[col].dtype == object:
                x_test[col] = pd.to_numeric(x_test[col], errors="ignore")

        x_train_full, y_train_full, capped_total = _apply_optional_stratified_cap(
            x_train_full, y_train_full, max_samples, random_state, stratified_subsample
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=float(val_size_single_csv),
            random_state=int(random_state),
            stratify=y_train_full,
        )
        label_col = label_col_train
    else:
        frames = [pd.read_csv(path, low_memory=False) for path in discovery["csv_paths"]]
        combined = pd.concat(frames, ignore_index=True)
        label_col = _detect_cic_label_column(combined)
        y_all = combined[label_col].map(_map_cic_to_binary).astype(int)

        x_all, dropped = _drop_cic_leakage_and_bad_columns(combined, label_col)
        dropped_leakage_columns = sorted(set(dropped + [label_col]))
        x_all = x_all.drop(columns=[label_col], errors="ignore")

        # replace inf/-inf then coerce numeric columns
        x_all = x_all.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan)
        for col in x_all.columns:
            if x_all[col].dtype == object:
                coerced = pd.to_numeric(x_all[col], errors="ignore")
                x_all[col] = coerced

        # optional global cap before splitting
        x_all, y_all, capped_total = _apply_optional_stratified_cap(
            x_all, y_all, max_samples, random_state, stratified_subsample
        )

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_all,
            y_all,
            test_size=float(test_size_single_csv),
            random_state=int(random_state),
            stratify=y_all,
        )
        val_ratio_on_train_val = float(val_size_single_csv) / max(1e-8, (1.0 - float(test_size_single_csv)))
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=val_ratio_on_train_val,
            random_state=int(random_state),
            stratify=y_train_val,
        )

    # per-split optional caps
    x_train, y_train, capped_train = _apply_optional_stratified_cap(
        x_train, y_train, max_train_samples, random_state, stratified_subsample
    )
    x_val, y_val, capped_val = _apply_optional_stratified_cap(
        x_val, y_val, max_validation_samples, random_state, stratified_subsample
    )
    x_test, y_test, capped_test = _apply_optional_stratified_cap(
        x_test, y_test, max_test_samples, random_state, stratified_subsample
    )

    categorical_columns = sorted(x_train.select_dtypes(include=["object", "category"]).columns.tolist())
    numeric_columns = [c for c in x_train.columns if c not in categorical_columns]

    return {
        "x_train": x_train.copy(),
        "x_val": x_val.copy(),
        "x_test": x_test.copy(),
        "y_train": y_train.astype(int).copy(),
        "y_val": y_val.astype(int).copy(),
        "y_test": y_test.astype(int).copy(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "dropped_leakage_columns": dropped_leakage_columns,
        "label_column": label_col,
        "source_mode": discovery["mode"],
        "source_files": [str(p) for p in discovery["csv_paths"]]
        if "csv_paths" in discovery
        else [str(discovery["train_path"]), str(discovery["test_path"])],
        "expected_paths": [str(p) for p in discovery["expected_paths"]],
        "capped": bool(capped_total or capped_train or capped_val or capped_test),
        "capped_breakdown": {
            "max_samples": max_samples,
            "max_train_samples": max_train_samples,
            "max_validation_samples": max_validation_samples,
            "max_test_samples": max_test_samples,
            "capped_total": capped_total,
            "capped_train": capped_train,
            "capped_validation": capped_val,
            "capped_test": capped_test,
        },
    }
