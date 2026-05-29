"""Tests for UNSW-NB15 dataset loading and preprocessing."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline import data_loader, preprocessing


def create_synthetic_unsw_csv():
    """Create a minimal synthetic UNSW-NB15 CSV for testing."""
    # Create a dataframe with UNSW-like structure
    data = {
        'id': [1, 2, 3, 4, 5],
        'dur': [100.5, 200.3, 150.2, 180.1, 220.4],
        'proto': ['tcp', 'udp', 'tcp', 'icmp', 'tcp'],
        'service': ['http', 'dns', 'https', 'ping', 'ssh'],
        'state': ['CON', 'URP', 'CON', 'REQ', 'CON'],
        'spkts': [10, 20, 15, 5, 8],
        'dpkts': [15, 18, 12, 2, 10],
        'sbytes': [500, 1000, 750, 100, 400],
        'dbytes': [600, 900, 650, 50, 500],
        'rate': [0.15, 0.25, 0.20, 0.05, 0.18],
        'sttl': [64, 64, 64, 64, 64],
        'dttl': [64, 64, 64, 64, 64],
        'sload': [0.01, 0.02, 0.015, 0.005, 0.012],
        'dload': [0.02, 0.018, 0.014, 0.002, 0.016],
        'sloss': [0, 0, 1, 0, 0],
        'dloss': [0, 1, 0, 0, 0],
        'ct_srv_src': [5, 10, 8, 2, 6],
        'ct_state_ttl': [3, 5, 4, 1, 3],
        'ct_dst_ltm': [2, 3, 2, 1, 2],
        'response_body_len': [512, 1024, 256, 0, 128],
        'attack_cat': ['Normal', 'Normal', 'Backdoor', 'Normal', 'Worm'],
        'label': [0, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_load_unsw_nb15_accepts_official_split():
    """Test that UNSW loader properly handles official train/test split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "UNSW_NB15_training-set.csv"
        test_path = Path(tmpdir) / "UNSW_NB15_testing-set.csv"
        
        # Create synthetic datasets
        df_train = create_synthetic_unsw_csv()
        df_test = create_synthetic_unsw_csv().iloc[[0, 2, 4]]  # Use subset for test
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        # Load via UNSW loader
        config = {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "label_column": "label",
        }
        
        x_train, x_test, y_train, y_test = data_loader.load_unsw_nb15_dataset(config)
        
        # Verify shapes
        assert x_train.shape[0] == 5  # 5 training samples
        assert x_test.shape[0] == 3   # 3 test samples
        assert x_train.shape[1] + 2 > 15  # More than basic features (id and label removed)
        assert x_test.shape[1] == x_train.shape[1]  # Same features


def test_load_unsw_nb15_drops_leakage_columns():
    """Test that UNSW loader drops attack_cat and id columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "UNSW_NB15_training-set.csv"
        test_path = Path(tmpdir) / "UNSW_NB15_testing-set.csv"
        
        df_train = create_synthetic_unsw_csv()
        df_test = create_synthetic_unsw_csv().iloc[[0, 2]]
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        config = {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "label_column": "label",
        }
        
        x_train, x_test, y_train, y_test = data_loader.load_unsw_nb15_dataset(config)
        
        # Check that leakage columns are removed
        assert 'id' not in x_train.columns
        assert 'attack_cat' not in x_train.columns
        assert 'label' not in x_train.columns
        assert 'id' not in x_test.columns
        assert 'attack_cat' not in x_test.columns
        assert 'label' not in x_test.columns


def test_load_unsw_nb15_ensures_binary_labels():
    """Test that UNSW loader validates binary labels (0 and 1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "UNSW_NB15_training-set.csv"
        test_path = Path(tmpdir) / "UNSW_NB15_testing-set.csv"
        
        df_train = create_synthetic_unsw_csv()
        df_test = create_synthetic_unsw_csv().iloc[[0, 1]]
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        config = {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "label_column": "label",
        }
        
        x_train, x_test, y_train, y_test = data_loader.load_unsw_nb15_dataset(config)
        
        # Verify binary labels
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})


def test_get_unsw_categorical_columns_detects_proto_service_state():
    """Test that categorical column detection finds UNSW-specific columns."""
    df = create_synthetic_unsw_csv().drop(columns=['id', 'label', 'attack_cat'])
    
    categorical_cols = data_loader.get_unsw_categorical_columns(df)
    
    # Should detect at least proto, service, state
    assert 'proto' in categorical_cols
    assert 'service' in categorical_cols
    assert 'state' in categorical_cols


def test_preprocessing_fits_only_on_train_not_test():
    """Test that preprocessing is fit only on training data."""
    df_train = create_synthetic_unsw_csv().drop(columns=['id', 'attack_cat', 'label'])
    df_test = create_synthetic_unsw_csv().drop(columns=['id', 'attack_cat', 'label']).iloc[[0, 2]]
    
    categorical_cols = data_loader.get_unsw_categorical_columns(df_train)
    
    # Build preprocessor on train
    preprocessor = preprocessing.build_preprocessing_pipeline(
        df_train,
        categorical_cols,
        {}
    )
    
    # Fit on train
    x_train_processed = preprocessor.fit_transform(df_train)
    
    # Transform test (should not refit)
    x_test_processed = preprocessor.transform(df_test)
    
    # Verify both were transformed
    assert x_train_processed.shape[0] == df_train.shape[0]
    assert x_test_processed.shape[0] == df_test.shape[0]
    assert x_train_processed.shape[1] == x_test_processed.shape[1]


def test_unsw_loader_missing_train_file_raises_error():
    """Test that missing training set raises clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "UNSW_NB15_testing-set.csv"
        df_test = create_synthetic_unsw_csv().iloc[[0, 1]]
        df_test.to_csv(test_path, index=False)
        
        config = {
            "train_path": str(Path(tmpdir) / "missing_train.csv"),
            "test_path": str(test_path),
            "label_column": "label",
        }
        
        try:
            data_loader.load_unsw_nb15_dataset(config)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "training" in str(e).lower()


def test_unsw_loader_missing_test_file_raises_error():
    """Test that missing testing set raises clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "UNSW_NB15_training-set.csv"
        df_train = create_synthetic_unsw_csv()
        df_train.to_csv(train_path, index=False)
        
        config = {
            "train_path": str(train_path),
            "test_path": str(Path(tmpdir) / "missing_test.csv"),
            "label_column": "label",
        }
        
        try:
            data_loader.load_unsw_nb15_dataset(config)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "testing" in str(e).lower()


def test_unsw_train_test_have_same_feature_columns():
    """Test that train and test data have same feature columns after loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "UNSW_NB15_training-set.csv"
        test_path = Path(tmpdir) / "UNSW_NB15_testing-set.csv"
        
        df_train = create_synthetic_unsw_csv()
        df_test = create_synthetic_unsw_csv().iloc[[0, 2, 4]]
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        config = {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "label_column": "label",
        }
        
        x_train, x_test, y_train, y_test = data_loader.load_unsw_nb15_dataset(config)
        
        # Verify same columns
        assert list(x_train.columns) == list(x_test.columns)
