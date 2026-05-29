"""Tests for CICIDS2017 dataset loading behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import main
from pipeline import data_loader


def _cic_df() -> pd.DataFrame:
    rows = []
    for i in range(30):
        benign = i < 15
        rows.append(
            {
                "Flow ID": f"f{i}",
                "Source IP": f"10.0.0.{i%5}",
                "Destination IP": f"192.168.0.{i%7}",
                "Timestamp": f"2017-07-0{(i%7)+1} 10:00:0{i%10}",
                "Flow Duration": np.inf if i == 2 else (1000 + i),
                "Tot Fwd Pkts": i * 3,
                "Label": "BENIGN" if benign else "DoS Hulk",
            }
        )
    return pd.DataFrame(rows)


def test_cic_loader_detects_label_and_maps_binary():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data").mkdir(parents=True, exist_ok=True)
        _cic_df().to_csv(root / "data" / "cicids2017.csv", index=False)
        out = data_loader.load_cicids2017_dataset(root, random_state=7)
        assert set(np.unique(out["y_train"])).issubset({0, 1})
        assert set(np.unique(out["y_val"])).issubset({0, 1})
        assert set(np.unique(out["y_test"])).issubset({0, 1})


def test_cic_loader_drops_leakage_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data").mkdir(parents=True, exist_ok=True)
        _cic_df().to_csv(root / "data" / "cicids2017.csv", index=False)
        out = data_loader.load_cicids2017_dataset(root, random_state=7)
        for col in ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"]:
            assert col in out["dropped_leakage_columns"]
        for col in ["Flow ID", "Source IP", "Destination IP", "Timestamp"]:
            assert col not in out["x_train"].columns


def test_cic_loader_handles_inf_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data").mkdir(parents=True, exist_ok=True)
        _cic_df().to_csv(root / "data" / "cicids2017.csv", index=False)
        out = data_loader.load_cicids2017_dataset(root, random_state=7)
        assert np.isfinite(out["x_train"].select_dtypes(include=[np.number]).to_numpy(dtype=float)).all() or np.isnan(
            out["x_train"].select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).any()


def test_cic_loader_prefers_processed_train_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed_dir = root / "data" / "processed" / "cicids2017"
        processed_dir.mkdir(parents=True, exist_ok=True)
        df = _cic_df()
        df.iloc[:20].to_csv(processed_dir / "train.csv", index=False)
        df.iloc[20:].to_csv(processed_dir / "test.csv", index=False)
        out = data_loader.load_cicids2017_dataset(root, random_state=7)
        assert out["source_mode"] == "official_split"
        assert len(out["x_test"]) == 10


def test_cic_loader_single_csv_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data").mkdir(parents=True, exist_ok=True)
        _cic_df().to_csv(root / "data" / "cicids2017.csv", index=False)
        out = data_loader.load_cicids2017_dataset(root, random_state=7)
        assert out["source_mode"] == "single"


def test_cic_alias_and_output_directory_name():
    assert main._normalize_dataset_alias("CICIDS2017") == "CICIDS2017"
    assert main._normalize_dataset_alias("cicids2017") == "CICIDS2017"
    assert main._normalize_dataset_alias("CIC-IDS2017") == "CICIDS2017"
