from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import sha256_file


class DatasetValidationError(ValueError):
    pass


def _validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"{dataset_name}: missing required columns: {missing}")


def _validate_missing_values(df: pd.DataFrame, dataset_name: str) -> None:
    missing_total = int(df.isna().sum().sum())
    if missing_total > 0:
        raise DatasetValidationError(f"{dataset_name}: found {missing_total} missing values; please clean the dataset explicitly.")


def _validate_target_classes(
    y: pd.Series,
    expected_classes: list[str],
    dataset_name: str,
    strict: bool,
) -> None:
    labels = set(map(str, y.astype(str).unique()))
    expected = set(expected_classes)

    missing_expected = expected - labels
    unknown_found = labels - expected

    if strict and (missing_expected or unknown_found):
        raise DatasetValidationError(f"{dataset_name}: class label mismatch; missing_expected={sorted(missing_expected)}, " f"unknown_found={sorted(unknown_found)}")


def _validate_dtypes(df: pd.DataFrame, categorical_columns: list[str], dataset_name: str) -> None:
    for col in categorical_columns:
        if col not in df.columns:
            raise DatasetValidationError(f"{dataset_name}: categorical column not found: {col}")
        if pd.api.types.is_numeric_dtype(df[col]):
            raise DatasetValidationError(f"{dataset_name}: categorical column '{col}' appears numeric; explicit categorical/string values required")


def _warn_duplicate_rows(df: pd.DataFrame, dataset_name: str, logger: Any) -> None:
    dup_count = int(df.duplicated().sum())
    if dup_count > 0 and logger is not None:
        logger.warning("%s: found %d duplicate rows", dataset_name, dup_count)


def validate_dataset(df: pd.DataFrame, config: dict[str, Any], dataset_name: str, logger: Any = None) -> None:
    dataset_cfg = config["dataset"]
    target = dataset_cfg["target_column"]
    excluded = dataset_cfg["excluded_columns"]
    categorical = dataset_cfg["categorical_columns"]

    _validate_columns(df, [target, *categorical], dataset_name)
    _validate_missing_values(df, dataset_name)
    _validate_dtypes(df, categorical, dataset_name)
    _validate_target_classes(
        y=df[target],
        expected_classes=dataset_cfg["expected_classes"],
        dataset_name=dataset_name,
        strict=bool(config["experiment"].get("strict_dataset_validation", True)),
    )
    _warn_duplicate_rows(df, dataset_name, logger)

    for col in excluded:
        if col not in df.columns and logger is not None:
            logger.warning(
                "%s: excluded column '%s' not found (will continue)",
                dataset_name,
                col,
            )


def load_datasets(config: dict[str, Any], logger: Any = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train_df = load_training_dataset(config, logger=logger)
    test_df = load_test_dataset(config, logger=logger)

    train_path = Path(config["dataset"]["train_path"])
    test_path = Path(config["dataset"]["test_path"])

    hashes = {
        "train_file": str(train_path.name),
        "test_file": str(test_path.name),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
    }
    return train_df, test_df, hashes


def load_training_dataset(config: dict[str, Any], logger: Any = None) -> pd.DataFrame:
    train_path = Path(config["dataset"]["train_path"])

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    train_df = pd.read_csv(train_path)

    validate_dataset(train_df, config, dataset_name="train", logger=logger)

    return train_df


def load_test_dataset(config: dict[str, Any], logger: Any = None) -> pd.DataFrame:
    test_path = Path(config["dataset"]["test_path"])

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    test_df = pd.read_csv(test_path)
    validate_dataset(test_df, config, dataset_name="test", logger=logger)
    return test_df


def split_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    dataset_cfg = config["dataset"]
    target = dataset_cfg["target_column"]
    excluded = set(dataset_cfg["excluded_columns"])

    feature_cols = [c for c in df.columns if c != target and c not in excluded]
    X = df[feature_cols].copy()
    y = df[target].astype(str).copy()
    return X, y
