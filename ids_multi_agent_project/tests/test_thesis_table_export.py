from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.thesis_table_export import REQUIRED_METHODS, export_thesis_tables


DATASETS = ["nsl_kdd", "unsw_nb15", "cicids2017"]


def test_thesis_export_creates_folder_and_expected_filenames():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    written, _, _ = export_thesis_tables(results_dir)
    names = {p.name for p in written}

    expected = set()
    for ds in DATASETS:
        expected.add(f"{ds}_single_agent_performance.csv")
        expected.add(f"{ds}_clean_trust_methods.csv")
        expected.add(f"{ds}_poisoning_general_traffic_agent.csv")
        expected.add(f"{ds}_poisoning_attack_recall_agent.csv")
        expected.add(f"{ds}_poisoning_normal_behavior_agent.csv")
        expected.add(f"{ds}_poisoning_hard_case_agent.csv")
        expected.add(f"{ds}_combined_robustness.csv")
        expected.add(f"{ds}_overall_robustness_ranking.csv")
    expected.add("all_datasets_overall_robustness_summary.csv")

    assert expected.issubset(names)
    assert (results_dir / "thesis_tables").exists()


def test_required_columns_and_methods_and_non_empty_outputs():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    export_thesis_tables(results_dir)
    tdir = results_dir / "thesis_tables"

    single_cols = [
        "Dataset", "Agent", "Accuracy", "Precision", "Recall", "F1-Score", "FPR", "FNR",
        "Specificity", "Balanced Accuracy", "TP", "TN", "FP", "FN",
    ]
    clean_cols = [
        "Dataset", "Trust Method", "Accuracy", "Precision", "Recall", "F1-Score", "FPR", "FNR",
        "Specificity", "Balanced Accuracy", "TP", "TN", "FP", "FN", "Status",
    ]

    for ds in DATASETS:
        sdf = pd.read_csv(tdir / f"{ds}_single_agent_performance.csv")
        cdf = pd.read_csv(tdir / f"{ds}_clean_trust_methods.csv")
        comb = pd.read_csv(tdir / f"{ds}_combined_robustness.csv")

        assert list(sdf.columns) == single_cols
        assert list(cdf.columns) == clean_cols
        assert not sdf.empty
        assert not cdf.empty
        assert not comb.empty

        clean_methods = set(cdf["Trust Method"].astype(str))
        comb_methods = set(comb["Trust Method"].astype(str))
        for method in REQUIRED_METHODS:
            assert method in clean_methods
            assert method in comb_methods


def test_missing_method_handled_gracefully():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    _, method_logs, _ = export_thesis_tables(results_dir)
    assert isinstance(method_logs, list)
