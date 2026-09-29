from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import DatasetValidationError, load_datasets, split_xy, validate_dataset


def test_validate_dataset_missing_column(synthetic_config, synthetic_dataset):
    df = synthetic_dataset.drop(columns=["attack_cat"])
    with pytest.raises(DatasetValidationError):
        validate_dataset(df, synthetic_config, dataset_name="train")


def test_validate_dataset_missing_values(synthetic_config, synthetic_dataset):
    df = synthetic_dataset.copy()
    df.loc[0, "f0"] = None
    with pytest.raises(DatasetValidationError):
        validate_dataset(df, synthetic_config, dataset_name="train")


def test_load_datasets_and_split_xy(tmp_path, synthetic_config, synthetic_dataset):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    synthetic_dataset.to_csv(train_path, index=False)
    synthetic_dataset.to_csv(test_path, index=False)

    synthetic_config["dataset"]["train_path"] = str(train_path)
    synthetic_config["dataset"]["test_path"] = str(test_path)

    train_df, test_df, hashes = load_datasets(synthetic_config)
    assert len(train_df) == len(synthetic_dataset)
    assert len(test_df) == len(synthetic_dataset)
    assert "train_sha256" in hashes

    X, y = split_xy(train_df, synthetic_config)
    assert "attack_cat" not in X.columns
    assert "id" not in X.columns
    assert "label" not in X.columns
    assert y.name == "attack_cat"
