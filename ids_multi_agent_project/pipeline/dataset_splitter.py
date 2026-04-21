"""Dataset splitting utilities for processed train/test exports."""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save(df, dataset_name, output_dir="data/processed"):
    """Split a cleaned dataframe into stratified train/test CSV files."""
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in dataset '{dataset_name}'.")

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    display_name = dataset_name.replace("_", "-").upper()
    print(f"[INFO] Splitting dataset: {display_name}")

    original_dist = df["label"].value_counts(dropna=False).sort_index().to_dict()
    print(f"[INFO] Original distribution: {original_dist}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42,
    )

    print(f"[INFO] Train size: {len(train_df)} | Test size: {len(test_df)}")
    train_dist = train_df["label"].value_counts(dropna=False).sort_index().to_dict()
    test_dist = test_df["label"].value_counts(dropna=False).sort_index().to_dict()
    print(f"[INFO] Train distribution: {train_dist}")
    print(f"[INFO] Test distribution: {test_dist}")

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")
    train_df.to_csv(train_path, index=False, chunksize=100_000)
    test_df.to_csv(test_path, index=False, chunksize=100_000)
    print(f"[INFO] Saved to: {dataset_dir}/")

