"""Tests for ToN-IoT dataset loading and preprocessing behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import main
from pipeline import data_loader, preprocessing


def _synthetic_ton_df() -> pd.DataFrame:
    rows = []
    for i in range(20):
        label = 0 if i < 10 else 1
        rows.append(
            {
                "id": i + 1,
                "ts": 1000 + i,
                "proto": ["tcp", "udp", "icmp", "tcp"][i % 4],
                "service": ["http", "dns", "ntp", "ssh", "https", "smtp"][i % 6],
                "value": float(i) / 20.0,
                "attack_cat": "Normal" if label == 0 else ["DoS", "Recon", "Worm"][i % 3],
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def test_ton_loader_detects_binary_label_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ton_dir = root / "data" / "raw" / "ton_iot"
        ton_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_ton_df().to_csv(ton_dir / "ton.csv", index=False)
        result = data_loader.load_ton_iot_dataset(root, random_state=7)
        assert set(np.unique(result["y_train"])).issubset({0, 1})
        assert set(np.unique(result["y_val"])).issubset({0, 1})
        assert set(np.unique(result["y_test"])).issubset({0, 1})


def test_ton_loader_drops_leakage_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ton_dir = root / "data" / "raw" / "ton_iot"
        ton_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_ton_df().to_csv(ton_dir / "ton.csv", index=False)
        result = data_loader.load_ton_iot_dataset(root, random_state=7)
        for leakage in ["id", "attack_cat", "label", "ts"]:
            assert leakage in result["dropped_leakage_columns"]
        assert "id" not in result["x_train"].columns
        assert "attack_cat" not in result["x_train"].columns


def test_ton_loader_categorical_columns_encodeable():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ton_dir = root / "data" / "raw" / "ton_iot"
        ton_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_ton_df().to_csv(ton_dir / "ton.csv", index=False)
        result = data_loader.load_ton_iot_dataset(root, random_state=7)
        pre = preprocessing.build_preprocessing_pipeline(result["x_train"], result["categorical_columns"], {})
        x_train_processed = pre.fit_transform(result["x_train"])
        x_val_processed = pre.transform(result["x_val"])
        assert x_train_processed.shape[0] == result["x_train"].shape[0]
        assert x_val_processed.shape[1] == x_train_processed.shape[1]


def test_ton_loader_split_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ton_dir = root / "data" / "raw" / "ton_iot"
        ton_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_ton_df().to_csv(ton_dir / "ton.csv", index=False)
        result = data_loader.load_ton_iot_dataset(root, random_state=7)
        total = result["x_train"].shape[0] + result["x_val"].shape[0] + result["x_test"].shape[0]
        assert total == 20
        assert result["x_train"].shape[0] > 0
        assert result["x_val"].shape[0] > 0
        assert result["x_test"].shape[0] > 0


def test_ton_dataset_aliases_supported():
    assert main._normalize_dataset_alias("ToN-IoT") == "ToN-IoT"
    assert main._normalize_dataset_alias("ton_iot") == "ToN-IoT"
    assert main._normalize_dataset_alias("TON-IOT") == "ToN-IoT"
