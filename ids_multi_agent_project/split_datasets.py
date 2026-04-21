"""Generate stratified 70/30 train/test splits for all project datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.dataset_splitter import split_and_save


def main() -> None:
    """Load cleaned datasets and save processed train/test splits."""
    project_root = Path(__file__).resolve().parent

    df_nsl_kdd = pd.read_csv(project_root / "data" / "nsl_kdd.csv")
    df_unsw_nb15 = pd.read_csv(project_root / "data" / "unsw_nb15.csv")
    df_cicids2017 = pd.read_csv(project_root / "data" / "cicids2017.csv")

    output_dir = project_root / "data" / "processed"
    split_and_save(df_nsl_kdd, "nsl_kdd", output_dir=str(output_dir))
    split_and_save(df_unsw_nb15, "unsw_nb15", output_dir=str(output_dir))
    split_and_save(df_cicids2017, "cicids2017", output_dir=str(output_dir))


if __name__ == "__main__":
    main()

