"""Download and prepare IDS datasets for Stage 1 experiments.

This script prepares three datasets and saves final CSV files in data/:
- data/nsl_kdd.csv
- data/unsw_nb15.csv
- data/cicids2017.csv

Notes:
- It skips a dataset if the prepared CSV already exists.
- It performs light cleaning only (column and label normalization).
- It does not split data, preprocess features for modeling, or train models.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]


def download_file(url: str, destination: Path, timeout: int = 90) -> bool:
    """Download a file from URL to destination.

    Returns:
        True if download succeeded, False otherwise.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url}")

    try:
        with urlopen(url, timeout=timeout) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        print(f"[OK] Saved to {destination}")
        return True
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[WARN] Download failed for {url}: {exc}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract ZIP or TAR archives.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
            print(f"[OK] Extracted ZIP: {archive_path}")
            return True

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(extract_to)
            print(f"[OK] Extracted TAR: {archive_path}")
            return True

        print(f"[WARN] Unsupported archive format: {archive_path}")
        return False
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        print(f"[WARN] Failed to extract archive {archive_path}: {exc}")
        return False


def validate_zip_archive(archive_path: Path) -> bool:
    """Validate ZIP archive integrity and delete corrupted files."""
    if not archive_path.exists():
        return False

    try:
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            bad_file = zip_file.testzip()
            if bad_file is not None:
                print(
                    f"[WARN] Archive corrupted or not a real ZIP file: {archive_path} "
                    f"(first bad entry: {bad_file})"
                )
                archive_path.unlink(missing_ok=True)
                print(f"[INFO] Deleted invalid archive: {archive_path}")
                return False
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"[WARN] Archive corrupted or not a real ZIP file: {archive_path}")
        print(f"[WARN] ZIP validation error: {exc}")
        archive_path.unlink(missing_ok=True)
        print(f"[INFO] Deleted invalid archive: {archive_path}")
        return False

    return True


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by stripping surrounding whitespace and BOM."""
    cleaned = df.copy()
    cleaned.columns = [str(col).replace("\ufeff", "").strip() for col in cleaned.columns]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    return cleaned


def _clean_label_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string label values."""
    cleaned = df.copy()
    cleaned["label"] = cleaned["label"].apply(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    return cleaned


def _print_dataset_report(dataset_name: str, df: pd.DataFrame) -> None:
    """Print dataset summary including label distribution."""
    print(f"\n[DATASET] {dataset_name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print("Label distribution:")
    print(df["label"].value_counts(dropna=False).to_string())


def _report_existing_output(dataset_name: str, output_path: Path) -> None:
    """Print report for an already-prepared dataset file."""
    try:
        existing_df = pd.read_csv(output_path, low_memory=False)
        existing_df = _clean_columns(existing_df)
        if "label" in existing_df.columns:
            _print_dataset_report(dataset_name, existing_df)
        else:
            print(
                f"[WARN] {dataset_name}: existing file has no 'label' column: {output_path}"
            )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Could not read existing prepared file {output_path}: {exc}")


def _finalize_and_save(df: pd.DataFrame, output_path: Path, dataset_name: str) -> bool:
    """Apply final cleaning rules and save CSV output."""
    final_df = _clean_columns(df)

    if "label" not in final_df.columns:
        print(f"[ERROR] {dataset_name}: missing required 'label' column.")
        return False

    # Ensure exactly one target column named 'label'.
    label_like_columns = [col for col in final_df.columns if col.strip().lower() == "label"]
    if len(label_like_columns) != 1:
        print(
            f"[ERROR] {dataset_name}: expected exactly one label column, found {len(label_like_columns)}."
        )
        return False

    final_df = _clean_label_values(final_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[OK] Saved prepared dataset: {output_path}")

    _print_dataset_report(dataset_name, final_df)
    return True


def _find_label_column(columns: List[str]) -> Optional[str]:
    """Find a likely label column from a list of column names."""
    possible = [
        "label",
        "Label",
        "class",
        "Class",
        "target",
        "Target",
        "attack",
        "Attack",
        "y",
        "Y",
        "binary_label",
        "Binary Label",
    ]

    for name in possible:
        if name in columns:
            return name

    # Common UNSW/CIC alternative target.
    if "attack_cat" in columns:
        return "attack_cat"

    return None


def find_csv_files_recursive(folder_path: Path) -> List[Path]:
    """Find all CSV files recursively under a folder and print discoveries."""
    csv_files = sorted(
        [
            path
            for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() == ".csv"
        ]
    )
    print(f"[INFO] Searching recursively for CSV files in: {folder_path.resolve()}")
    print(f"[INFO] Found {len(csv_files)} CSV file(s).")
    if csv_files:
        print("[INFO] Discovered CSV paths:")
        for csv_path in csv_files:
            print(f"  - {csv_path}")
    return csv_files


def _print_folder_debug_contents(folder_path: Path, max_items: int = 100) -> None:
    """Print folder contents to debug discovery issues without crashing."""
    if not folder_path.exists():
        print(f"[DEBUG] Folder does not exist: {folder_path.resolve()}")
        return

    print(f"[DEBUG] Folder exists: {folder_path.resolve()}")
    items = sorted(folder_path.rglob("*"))
    print(f"[DEBUG] Total nested items found: {len(items)}")
    if not items:
        print("[DEBUG] Folder is empty.")
        return

    print(f"[DEBUG] Showing up to {max_items} item(s):")
    for item in items[:max_items]:
        marker = "/" if item.is_dir() else ""
        print(f"  - {item}{marker}")


def _load_and_clean_cicids_files(csv_files: List[Path], prefix: str = "") -> List[pd.DataFrame]:
    """Load CICIDS CSV files safely and normalize label column per file."""
    parts: List[pd.DataFrame] = []
    loaded_file_count = 0

    for csv_path in csv_files:
        try:
            print(f"[INFO] Loading {prefix}file: {csv_path}")
            part = pd.read_csv(csv_path, low_memory=False)
            print(f"[INFO] Rows loaded from {csv_path.name}: {len(part)}")
            part = _clean_columns(part)

            target_col = _find_label_column(list(part.columns))
            if target_col is None:
                print(
                    f"[WARN] Skipping file without detectable label column: {csv_path}"
                )
                continue

            if target_col != "label":
                part = part.rename(columns={target_col: "label"})

            parts.append(part)
            loaded_file_count += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skipping unreadable CSV file '{csv_path}': {exc}")

    print(f"[INFO] Successfully loaded {loaded_file_count} CSV file(s).")
    return parts


def prepare_nsl_kdd(project_root: Path) -> Tuple[bool, Path, str]:
    """Download and prepare NSL-KDD into data/nsl_kdd.csv."""
    dataset_name = "NSL-KDD"
    output_path = project_root / "data" / "nsl_kdd.csv"

    if output_path.exists():
        print(f"[SKIP] {dataset_name}: prepared file already exists at {output_path}")
        return True, output_path, "already exists"

    raw_dir = project_root / "data" / "raw" / "nsl_kdd"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_urls = [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    ]
    test_urls = [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
    ]

    train_file = raw_dir / "KDDTrain+.txt"
    test_file = raw_dir / "KDDTest+.txt"

    if not train_file.exists():
        train_ok = any(download_file(url, train_file) for url in train_urls)
    else:
        print(f"[INFO] Using existing raw file: {train_file}")
        train_ok = True

    if not test_file.exists():
        test_ok = any(download_file(url, test_file) for url in test_urls)
    else:
        print(f"[INFO] Using existing raw file: {test_file}")
        test_ok = True

    if not train_ok or not test_ok:
        return False, output_path, "download failed"

    try:
        train_df = pd.read_csv(train_file, header=None, names=NSL_KDD_COLUMNS)
        test_df = pd.read_csv(test_file, header=None, names=NSL_KDD_COLUMNS)
        combined = pd.concat([train_df, test_df], ignore_index=True)
        return _finalize_and_save(combined, output_path, dataset_name), output_path, "prepared"
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {dataset_name}: failed to parse/prepare files: {exc}")
        return False, output_path, "parse failed"


def prepare_unsw_nb15(project_root: Path) -> Tuple[bool, Path, str]:
    """Download and prepare UNSW-NB15 into data/unsw_nb15.csv."""
    dataset_name = "UNSW-NB15"
    output_path = project_root / "data" / "unsw_nb15.csv"

    if output_path.exists():
        print(f"[SKIP] {dataset_name}: prepared file already exists at {output_path}")
        _report_existing_output(dataset_name, output_path)
        return True, output_path, "already exists"

    raw_dir = project_root / "data" / "raw" / "unsw_nb15"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_file = raw_dir / "UNSW_NB15_training-set.csv"
    test_file = raw_dir / "UNSW_NB15_testing-set.csv"

    # Mirrors ordered by likelihood/stability.
    train_urls = [
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/yanickdepotter/UNSW-NB15/master/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/abhinav-bhardwaj/UNSW-NB15/master/UNSW_NB15_training-set.csv",
    ]
    test_urls = [
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/yanickdepotter/UNSW-NB15/master/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/abhinav-bhardwaj/UNSW-NB15/master/UNSW_NB15_testing-set.csv",
    ]

    if not train_file.exists():
        train_ok = any(download_file(url, train_file) for url in train_urls)
    else:
        print(f"[INFO] Using existing raw file: {train_file}")
        train_ok = True

    if not test_file.exists():
        test_ok = any(download_file(url, test_file) for url in test_urls)
    else:
        print(f"[INFO] Using existing raw file: {test_file}")
        test_ok = True

    if not train_ok or not test_ok:
        local_csvs = sorted(raw_dir.glob("*.csv"))
        if not local_csvs:
            print(
                "[ERROR] UNSW-NB15 download failed and no local CSV files were found in "
                f"{raw_dir}"
            )
            return False, output_path, "download/local missing"
        print("[INFO] Falling back to local UNSW-NB15 CSV files.")
        source_files = local_csvs
    else:
        source_files = [train_file, test_file]

    try:
        parts = [pd.read_csv(path, low_memory=False) for path in source_files]
        combined = pd.concat(parts, ignore_index=True)
        combined = _clean_columns(combined)

        target_col = _find_label_column(list(combined.columns))
        if target_col is None:
            print("[ERROR] UNSW-NB15: could not detect target column to rename as 'label'.")
            return False, output_path, "label not found"

        if target_col != "label":
            combined = combined.rename(columns={target_col: "label"})

        return _finalize_and_save(combined, output_path, dataset_name), output_path, "prepared"
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {dataset_name}: failed to parse/prepare files: {exc}")
        return False, output_path, "parse failed"


def prepare_cicids2017(project_root: Path) -> Tuple[bool, Path, str]:
    """Prepare CICIDS2017 into data/cicids2017.csv.

    This dataset is often large and distributed as many CSV files.
    The script first attempts to use local extracted CSV files under:
    data/raw/cicids2017/
    """
    dataset_name = "CICIDS2017"
    output_path = project_root / "data" / "cicids2017.csv"

    if output_path.exists():
        print(f"[SKIP] {dataset_name}: prepared file already exists at {output_path}")
        _report_existing_output(dataset_name, output_path)
        return True, output_path, "already exists"

    raw_dir = project_root / "data" / "raw" / "cicids2017"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Priority path: merge user-provided local raw CSV files first.
    local_csv_files = find_csv_files_recursive(raw_dir)
    if local_csv_files:
        parts = _load_and_clean_cicids_files(local_csv_files, prefix="CICIDS2017 local ")

        if parts:
            combined = pd.concat(parts, ignore_index=True)
            combined = _clean_columns(combined)
            print(f"[INFO] CICIDS2017 merged shape (local): {combined.shape}")

            target_col = _find_label_column(list(combined.columns))
            if target_col is None:
                print("[ERROR] CICIDS2017: could not detect target column to rename as 'label'.")
                return False, output_path, "label not found"

            if target_col != "label":
                combined = combined.rename(columns={target_col: "label"})

            return _finalize_and_save(combined, output_path, dataset_name), output_path, "prepared"

        print("[WARN] No readable local CICIDS2017 CSV files were found.")
        print("[INFO] Trying download/mirror fallback for CICIDS2017...")

    # Try direct combined mirror first.
    combined_mirror_urls = [
        "https://raw.githubusercontent.com/Th1nhNg0/CICIDS2017/master/cicids2017.csv",
    ]
    downloaded_combined = raw_dir / "cicids2017_combined.csv"

    if not downloaded_combined.exists():
        _ = any(download_file(url, downloaded_combined) for url in combined_mirror_urls)

    if downloaded_combined.exists():
        try:
            combined_df = pd.read_csv(downloaded_combined, low_memory=False)
            combined_df = _clean_columns(combined_df)
            target_col = _find_label_column(list(combined_df.columns))
            if target_col is None:
                print("[WARN] CICIDS2017 mirror file lacks target column; trying local raw CSVs.")
            else:
                if target_col != "label":
                    combined_df = combined_df.rename(columns={target_col: "label"})
                return (
                    _finalize_and_save(combined_df, output_path, dataset_name),
                    output_path,
                    "prepared",
                )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed reading CICIDS2017 mirror file: {exc}")

    # Try downloading official/mirror archive and extract CSV files.
    archive_urls = [
        # Official dataset page (informational). Included for logging/troubleshooting.
        "http://www.unb.ca/cic/datasets/ids-2017.html",
        # Official CIC dataset host direct archive path.
        "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
        # Fallback mirrors.
        "https://raw.githubusercontent.com/rokibulroni/CIC-IDS-2017-Dataset/main/MachineLearningCSV.zip",
        "https://github.com/rokibulroni/CIC-IDS-2017-Dataset/raw/main/MachineLearningCSV.zip",
    ]
    archive_path = raw_dir / "MachineLearningCSV.zip"

    # Validate existing archive and remove if corrupted.
    if archive_path.exists():
        _ = validate_zip_archive(archive_path)

    # If no CSV files are present, ensure we have a valid archive by downloading it.
    if not any(path.is_file() and path.suffix.lower() == ".csv" for path in raw_dir.rglob("*")):
        if not archive_path.exists():
            downloaded = False
            for url in archive_urls:
                # Skip non-archive links for actual file write.
                if not url.lower().endswith(".zip"):
                    print(f"[INFO] CICIDS2017 reference URL: {url}")
                    continue
                if download_file(url, archive_path):
                    if validate_zip_archive(archive_path):
                        downloaded = True
                        break
            if not downloaded and not archive_path.exists():
                print("[WARN] Could not download a valid CICIDS2017 archive.")

    if archive_path.exists():
        if validate_zip_archive(archive_path):
            if not extract_archive(archive_path, raw_dir):
                print(f"[WARN] Failed to extract archive: {archive_path}")

    csv_files = find_csv_files_recursive(raw_dir)
    csv_files = [path for path in csv_files if path.name != downloaded_combined.name]

    if not csv_files:
        print("[ERROR] CICIDS2017 raw CSV files not found.")
        print(f"[ERROR] Checked folder (recursive): {raw_dir}")
        _print_folder_debug_contents(raw_dir)
        print("Place extracted CICIDS2017 CSV files under this folder.")
        print("If you extracted a ZIP, the CSVs may be inside an extra nested folder.")
        print("Example nested path pattern:")
        print(f"  {raw_dir}/<some_subfolder>/*.csv")
        print("Suggested archive name:")
        print("  MachineLearningCSV.zip")
        print("Then run this script again.")
        return False, output_path, "raw files missing"

    parts = _load_and_clean_cicids_files(csv_files, prefix="CICIDS2017 ")

    if not parts:
        print("[ERROR] CICIDS2017: no readable CSV files found after scanning.")
        return False, output_path, "no readable csv files"

    combined = pd.concat(parts, ignore_index=True)
    combined = _clean_columns(combined)
    print(f"[INFO] CICIDS2017 merged shape: {combined.shape}")

    target_col = _find_label_column(list(combined.columns))
    if target_col is None:
        print("[ERROR] CICIDS2017: could not detect target column to rename as 'label'.")
        return False, output_path, "label not found"

    if target_col != "label":
        combined = combined.rename(columns={target_col: "label"})

    return _finalize_and_save(combined, output_path, dataset_name), output_path, "prepared"


def _ensure_directories(project_root: Path) -> None:
    """Create needed data folders for downloads and raw inputs."""
    required = [
        project_root / "data",
        project_root / "data" / "raw",
        project_root / "data" / "raw" / "nsl_kdd",
        project_root / "data" / "raw" / "unsw_nb15",
        project_root / "data" / "raw" / "cicids2017",
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Prepare NSL-KDD, UNSW-NB15, and CICIDS2017 datasets in order."""
    project_root = Path(__file__).resolve().parent
    _ensure_directories(project_root)

    print("Starting dataset download/preparation for Stage 1 IDS experiments...")

    results: List[Dict[str, str]] = []

    for dataset_name, prepare_fn in [
        ("NSL-KDD", prepare_nsl_kdd),
        ("UNSW-NB15", prepare_unsw_nb15),
        ("CICIDS2017", prepare_cicids2017),
    ]:
        print("\n" + "=" * 80)
        print(f"Preparing {dataset_name}")
        print("=" * 80)

        success, output_path, note = prepare_fn(project_root)
        results.append(
            {
                "dataset": dataset_name,
                "status": "success" if success else "failed",
                "output_path": str(output_path),
                "note": note,
            }
        )

    print("\n" + "#" * 80)
    print("Preparation Summary")
    print("#" * 80)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
