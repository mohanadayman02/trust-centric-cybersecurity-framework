from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REQUIRED_METHODS = [
    "Majority Vote",
    "Global Trust Voting",
    "Dynamic Trust",
    "Trust Agent Selector",
]
OPTIONAL_AI_METHOD = "AI Trust Auditor"

POISON_AGENT_SLUGS = [
    "general_traffic_agent",
    "attack_recall_agent",
    "normal_behavior_agent",
    "hard_case_agent",
]

POISON_AGENT_NAMES = {
    "general_traffic_agent": "General Traffic Agent",
    "attack_recall_agent": "Attack Recall Agent",
    "normal_behavior_agent": "Normal Behavior Agent",
    "hard_case_agent": "Hard-Case Agent",
}

DATASETS = [
    ("NSL-KDD", "nsl_kdd"),
    ("UNSW-NB15", "unsw_nb15"),
    ("CICIDS2017", "cicids2017"),
]


@dataclass
class DatasetPaths:
    dataset_name: str
    dataset_slug: str
    base_dir: Path
    final_comparison: Path
    poisoned_full_comparison: Path
    poisoning_root: Path
    overall_summary_md: Path


def _normalize_method_name(value: str) -> str:
    v = str(value).strip()
    if v == "Majority Voting":
        return "Majority Vote"
    return v


def _dataset_base_dir(results_dir: Path, slug: str) -> Path:
    if (results_dir / slug).exists():
        return results_dir / slug
    return results_dir


def resolve_dataset_paths(results_dir: Path, dataset_name: str, slug: str) -> DatasetPaths:
    base = _dataset_base_dir(results_dir, slug)
    poisoned_csv = base / f"{slug}_poisoned_full_comparison.csv"
    final_csv = base / "final_comparison.csv"

    nested_poison_root = base / "poisoning" / slug
    legacy_poison_root = results_dir / "poisoning" / slug
    poison_root = nested_poison_root if nested_poison_root.exists() else legacy_poison_root

    return DatasetPaths(
        dataset_name=dataset_name,
        dataset_slug=slug,
        base_dir=base,
        final_comparison=final_csv,
        poisoned_full_comparison=poisoned_csv,
        poisoning_root=poison_root,
        overall_summary_md=poison_root / "overall_robustness_summary.md",
    )


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _single_agent_table(paths: DatasetPaths) -> pd.DataFrame:
    _require(paths.final_comparison)
    df = pd.read_csv(paths.final_comparison)
    sa = df[df["Category"] == "Single Agent"].copy()
    sa = sa.rename(
        columns={
            "Method": "Agent",
            "F1": "F1-Score",
        }
    )
    sa["Dataset"] = paths.dataset_name
    cols = [
        "Dataset",
        "Agent",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "FPR",
        "FNR",
        "Specificity",
        "Balanced Accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    # enrich confusion counts from poisoned full comparison original rows
    _require(paths.poisoned_full_comparison)
    pf = pd.read_csv(paths.poisoned_full_comparison)
    originals = pf[pf["Evaluation Type"] == "Original Agent"][
        ["Poisoned Agent", "TP", "TN", "FP", "FN"]
    ].drop_duplicates(subset=["Poisoned Agent"])
    sa = sa.merge(originals, how="left", left_on="Agent", right_on="Poisoned Agent")
    sa = sa.drop(columns=["Poisoned Agent"], errors="ignore")

    for c in ["TP", "TN", "FP", "FN"]:
        sa[c] = sa[c].astype("Int64")

    return sa[cols]


def _read_robustness(path: Path) -> pd.DataFrame:
    _require(path)
    df = pd.read_csv(path)
    df["Trust Method"] = df["Trust Method"].map(_normalize_method_name)
    return df


def _status_row(dataset: str, method: str, reason: str, extra: Dict[str, object] | None = None) -> Dict[str, object]:
    row = {
        "Dataset": dataset,
        "Trust Method": method,
        "Accuracy": pd.NA,
        "Precision": pd.NA,
        "Recall": pd.NA,
        "F1-Score": pd.NA,
        "FPR": pd.NA,
        "FNR": pd.NA,
        "Specificity": pd.NA,
        "Balanced Accuracy": pd.NA,
        "TP": pd.NA,
        "TN": pd.NA,
        "FP": pd.NA,
        "FN": pd.NA,
        "Status": f"unavailable: {reason}",
    }
    if extra:
        row.update(extra)
    return row


def _clean_trust_table(paths: DatasetPaths, method_logs: List[str]) -> pd.DataFrame:
    robust = _read_robustness(paths.poisoning_root / "general_traffic_agent" / "robustness_report.csv")
    clean = robust[robust["Scenario"] == "Clean"].copy()

    methods_to_include = list(REQUIRED_METHODS)
    if (clean["Trust Method"] == OPTIONAL_AI_METHOD).any():
        methods_to_include.append(OPTIONAL_AI_METHOD)

    rows: List[Dict[str, object]] = []
    for method in methods_to_include:
        hit = clean[clean["Trust Method"] == method]
        if hit.empty:
            method_logs.append(f"{paths.dataset_name}: {method} unavailable in clean trust table")
            rows.append(_status_row(paths.dataset_name, method, "missing in robustness_report clean scenario"))
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "Dataset": paths.dataset_name,
                "Trust Method": method,
                "Accuracy": r["Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1-Score": r["F1"],
                "FPR": r["FPR"],
                "FNR": r["FNR"],
                "Specificity": r["Specificity"],
                "Balanced Accuracy": r["Balanced Accuracy"],
                "TP": r["TP"],
                "TN": r["TN"],
                "FP": r["FP"],
                "FN": r["FN"],
                "Status": "available",
            }
        )

    return pd.DataFrame(rows)


def _observation(eval_type: str, row: pd.Series, mv_row: pd.Series | None, best_acc_eval: str, best_f1_eval: str, lowest_fnr_eval: str) -> str:
    if eval_type == "Poisoned Agent" and pd.notna(row.get("Accuracy")) and float(row["Accuracy"]) < 0.75:
        return "Severe degradation after poisoning"
    if eval_type in (best_acc_eval, best_f1_eval):
        return "Best overall recovery"
    if eval_type == lowest_fnr_eval:
        return "Strongest attack detection preservation"
    if pd.notna(row.get("FNR")) and pd.notna(row.get("FPR")) and float(row["FNR"]) < 0.1 and float(row["FPR"]) > 0.2:
        return "Maintains attack recall but increases false positives"
    if mv_row is not None and eval_type in REQUIRED_METHODS and pd.notna(row.get("Accuracy")) and pd.notna(mv_row.get("Accuracy")):
        if float(row["Accuracy"]) < float(mv_row["Accuracy"]):
            return "Lower robustness than majority voting"
    return "Stable performance profile"


def _per_agent_poison_table(paths: DatasetPaths, agent_slug: str, method_logs: List[str]) -> pd.DataFrame:
    _require(paths.poisoned_full_comparison)
    df = pd.read_csv(paths.poisoned_full_comparison)
    agent_name = POISON_AGENT_NAMES[agent_slug]
    sub = df[df["Poisoned Agent"] == agent_name].copy()
    sub = sub.drop_duplicates(subset=["Evaluation Type"], keep="first")

    eval_types = ["Original Agent", "Poisoned Agent", *REQUIRED_METHODS]
    if (sub["Evaluation Type"] == OPTIONAL_AI_METHOD).any():
        eval_types.append(OPTIONAL_AI_METHOD)
    rows: List[Dict[str, object]] = []

    candidate = sub[sub["Evaluation Type"].isin(REQUIRED_METHODS)]
    best_acc_eval = candidate.sort_values("Accuracy", ascending=False)["Evaluation Type"].iloc[0] if not candidate.empty else ""
    best_f1_eval = candidate.sort_values("F1", ascending=False)["Evaluation Type"].iloc[0] if not candidate.empty else ""
    lowest_fnr_eval = candidate.sort_values("FNR", ascending=True)["Evaluation Type"].iloc[0] if not candidate.empty else ""
    mv_hit = sub[sub["Evaluation Type"] == "Majority Vote"]
    mv_row = mv_hit.iloc[0] if not mv_hit.empty else None

    for et in eval_types:
        hit = sub[sub["Evaluation Type"] == et]
        if hit.empty:
            reason = "missing in poisoned_full_comparison"
            if et in REQUIRED_METHODS:
                method_logs.append(f"{paths.dataset_name} / {agent_name}: {et} unavailable")
            rows.append(
                {
                    "Dataset": paths.dataset_name,
                    "Poisoned Agent": agent_name,
                    "Evaluation Type": et,
                    "Accuracy": pd.NA,
                    "Precision": pd.NA,
                    "Recall": pd.NA,
                    "F1-Score": pd.NA,
                    "FPR": pd.NA,
                    "FNR": pd.NA,
                    "Specificity": pd.NA,
                    "Balanced Accuracy": pd.NA,
                    "TP": pd.NA,
                    "TN": pd.NA,
                    "FP": pd.NA,
                    "FN": pd.NA,
                    "Observation": f"Unavailable: {reason}",
                }
            )
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "Dataset": paths.dataset_name,
                "Poisoned Agent": agent_name,
                "Evaluation Type": et,
                "Accuracy": r["Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1-Score": r["F1"],
                "FPR": r["FPR"],
                "FNR": r["FNR"],
                "Specificity": r["Specificity"],
                "Balanced Accuracy": r["Balanced Accuracy"],
                "TP": r["TP"],
                "TN": r["TN"],
                "FP": r["FP"],
                "FN": r["FN"],
                "Observation": _observation(et, r, mv_row, best_acc_eval, best_f1_eval, lowest_fnr_eval),
            }
        )

    return pd.DataFrame(rows)


def _combined_robustness(paths: DatasetPaths, method_logs: List[str]) -> pd.DataFrame:
    frames = []
    for agent_slug in POISON_AGENT_SLUGS:
        rpath = paths.poisoning_root / agent_slug / "robustness_report.csv"
        robust = _read_robustness(rpath)
        robust["Poisoned Agent"] = robust["Poisoned Agent"].replace({"None": "None"})
        frames.append(robust)
    all_robust = pd.concat(frames, ignore_index=True)

    methods_to_include = list(REQUIRED_METHODS)
    if (all_robust["Trust Method"] == OPTIONAL_AI_METHOD).any():
        methods_to_include.append(OPTIONAL_AI_METHOD)

    rows: List[Dict[str, object]] = []
    for method in methods_to_include:
        clean_hit = all_robust[(all_robust["Trust Method"] == method) & (all_robust["Scenario"] == "Clean")]
        if clean_hit.empty:
            method_logs.append(f"{paths.dataset_name}: {method} unavailable in combined robustness clean")
            rows.append(
                {
                    "Dataset": paths.dataset_name,
                    "Trust Method": method,
                    "Scenario": "Clean",
                    "Poisoned Agent": "None",
                    "Accuracy": pd.NA,
                    "F1-Score": pd.NA,
                    "FPR": pd.NA,
                    "FNR": pd.NA,
                    "Accuracy Drop": pd.NA,
                    "F1 Drop": pd.NA,
                    "FPR Increase": pd.NA,
                    "FNR Increase": pd.NA,
                    "Status": "unavailable: missing clean scenario",
                }
            )
        else:
            c = clean_hit.iloc[0]
            rows.append(
                {
                    "Dataset": paths.dataset_name,
                    "Trust Method": method,
                    "Scenario": "Clean",
                    "Poisoned Agent": "None",
                    "Accuracy": c["Accuracy"],
                    "F1-Score": c["F1"],
                    "FPR": c["FPR"],
                    "FNR": c["FNR"],
                    "Accuracy Drop": c["Accuracy Drop"],
                    "F1 Drop": c["F1 Drop"],
                    "FPR Increase": c["FPR Increase"],
                    "FNR Increase": c["FNR Increase"],
                    "Status": "available",
                }
            )

        for agent_slug in POISON_AGENT_SLUGS:
            agent_name = POISON_AGENT_NAMES[agent_slug]
            hit = all_robust[
                (all_robust["Trust Method"] == method)
                & (all_robust["Scenario"] == "Poisoned")
                & (all_robust["Poisoned Agent"] == agent_name)
            ]
            if hit.empty:
                method_logs.append(f"{paths.dataset_name} / {agent_name}: {method} unavailable in poisoned scenario")
                rows.append(
                    {
                        "Dataset": paths.dataset_name,
                        "Trust Method": method,
                        "Scenario": "Poisoned",
                        "Poisoned Agent": agent_name,
                        "Accuracy": pd.NA,
                        "F1-Score": pd.NA,
                        "FPR": pd.NA,
                        "FNR": pd.NA,
                        "Accuracy Drop": pd.NA,
                        "F1 Drop": pd.NA,
                        "FPR Increase": pd.NA,
                        "FNR Increase": pd.NA,
                        "Status": "unavailable: missing poisoned scenario",
                    }
                )
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "Dataset": paths.dataset_name,
                    "Trust Method": method,
                    "Scenario": "Poisoned",
                    "Poisoned Agent": agent_name,
                    "Accuracy": r["Accuracy"],
                    "F1-Score": r["F1"],
                    "FPR": r["FPR"],
                    "FNR": r["FNR"],
                    "Accuracy Drop": r["Accuracy Drop"],
                    "F1 Drop": r["F1 Drop"],
                    "FPR Increase": r["FPR Increase"],
                    "FNR Increase": r["FNR Increase"],
                    "Status": "available",
                }
            )

    return pd.DataFrame(rows)


def _ranking_table(paths: DatasetPaths, combined: pd.DataFrame) -> pd.DataFrame:
    p = combined[(combined["Scenario"] == "Poisoned") & (combined["Status"] == "available")].copy()
    grouped = (
        p.groupby("Trust Method", as_index=False)[["Accuracy Drop", "F1 Drop", "FPR Increase", "FNR Increase"]]
        .mean(numeric_only=True)
        .rename(
            columns={
                "Accuracy Drop": "Avg Accuracy Drop",
                "F1 Drop": "Avg F1 Drop",
                "FPR Increase": "Avg FPR Increase",
                "FNR Increase": "Avg FNR Increase",
            }
        )
    )
    grouped = grouped.sort_values(
        by=["Avg Accuracy Drop", "Avg F1 Drop", "Avg FNR Increase", "Avg FPR Increase"],
        ascending=True,
    ).reset_index(drop=True)
    grouped.insert(0, "Dataset", paths.dataset_name)
    grouped["Robustness Rank"] = grouped.index + 1

    notes = []
    if not grouped.empty:
        best_method = grouped.iloc[0]["Trust Method"]
    else:
        best_method = ""
    for _, r in grouped.iterrows():
        n = ""
        if r["Trust Method"] == best_method:
            n = "Best average robustness"
        elif r["Avg FNR Increase"] == grouped["Avg FNR Increase"].min():
            n = "Strongest attack detection preservation"
        else:
            n = "Robustness trade-off"
        notes.append(n)
    grouped["Notes"] = notes
    return grouped


def _summary_from_md(paths: DatasetPaths) -> Dict[str, str]:
    _require(paths.overall_summary_md)
    text = paths.overall_summary_md.read_text(encoding="utf-8")

    def pull(label: str) -> str:
        m = re.search(rf"-\s*{re.escape(label)}:\s*(.+)", text)
        return m.group(1).strip() if m else "Unknown"

    best = pull("Best average robustness")
    low_fnr = pull("Lowest average FNR increase")
    low_acc = pull("Smallest average accuracy degradation")
    key = f"{best} leads average robustness; {low_fnr} best preserves detection."
    return {
        "Dataset": paths.dataset_name,
        "Best Average Robustness Method": best,
        "Lowest FNR Increase Method": low_fnr,
        "Smallest Accuracy Degradation Method": low_acc,
        "Key Finding": key,
    }


def _ensure_non_empty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"Generated table is empty: {name}")


def export_thesis_tables(results_dir: Path) -> Tuple[List[Path], List[str], Dict[str, int]]:
    thesis_dir = results_dir / "thesis_tables"
    thesis_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    method_logs: List[str] = []
    row_counts: Dict[str, int] = {}
    summary_rows: List[Dict[str, str]] = []

    for dataset_name, slug in DATASETS:
        paths = resolve_dataset_paths(results_dir, dataset_name, slug)

        single_df = _single_agent_table(paths)
        _ensure_non_empty(single_df, f"{slug}_single_agent_performance.csv")
        single_path = thesis_dir / f"{slug}_single_agent_performance.csv"
        single_df.to_csv(single_path, index=False)
        written.append(single_path)
        row_counts[single_path.name] = len(single_df)

        clean_df = _clean_trust_table(paths, method_logs)
        _ensure_non_empty(clean_df, f"{slug}_clean_trust_methods.csv")
        clean_path = thesis_dir / f"{slug}_clean_trust_methods.csv"
        clean_df.to_csv(clean_path, index=False)
        written.append(clean_path)
        row_counts[clean_path.name] = len(clean_df)

        for agent_slug in POISON_AGENT_SLUGS:
            per_df = _per_agent_poison_table(paths, agent_slug, method_logs)
            _ensure_non_empty(per_df, f"{slug}_poisoning_{agent_slug}.csv")
            per_path = thesis_dir / f"{slug}_poisoning_{agent_slug}.csv"
            per_df.to_csv(per_path, index=False)
            written.append(per_path)
            row_counts[per_path.name] = len(per_df)

        combined_df = _combined_robustness(paths, method_logs)
        _ensure_non_empty(combined_df, f"{slug}_combined_robustness.csv")
        combined_path = thesis_dir / f"{slug}_combined_robustness.csv"
        combined_df.to_csv(combined_path, index=False)
        written.append(combined_path)
        row_counts[combined_path.name] = len(combined_df)

        ranking_df = _ranking_table(paths, combined_df)
        _ensure_non_empty(ranking_df, f"{slug}_overall_robustness_ranking.csv")
        ranking_path = thesis_dir / f"{slug}_overall_robustness_ranking.csv"
        ranking_df.to_csv(ranking_path, index=False)
        written.append(ranking_path)
        row_counts[ranking_path.name] = len(ranking_df)

        summary_rows.append(_summary_from_md(paths))

    summary_df = pd.DataFrame(summary_rows)
    _ensure_non_empty(summary_df, "all_datasets_overall_robustness_summary.csv")
    summary_path = thesis_dir / "all_datasets_overall_robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    written.append(summary_path)
    row_counts[summary_path.name] = len(summary_df)

    return written, method_logs, row_counts


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    written, method_logs, row_counts = export_thesis_tables(results_dir)

    print("[THESIS_EXPORT] Generated files:")
    for p in sorted(written):
        print(f"- {p}")

    print("[THESIS_EXPORT] Row counts:")
    for name in sorted(row_counts):
        print(f"- {name}: {row_counts[name]}")

    if method_logs:
        print("[THESIS_EXPORT] Missing/unavailable methods:")
        for m in method_logs:
            print(f"- {m}")
    else:
        print("[THESIS_EXPORT] Missing/unavailable methods: none")


if __name__ == "__main__":
    main()
