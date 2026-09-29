from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_config(tmp_path):
    return {
        "experiment": {"base_seed": 42, "runs_root": str(tmp_path / "runs"), "save_models": False},
        "dataset": {
            "train_path": str(tmp_path / "train.csv"),
            "test_path": str(tmp_path / "test.csv"),
            "target_column": "attack_cat",
            "excluded_columns": ["id", "label"],
            "categorical_columns": ["proto", "service", "state"],
            "expected_classes": [
                "Normal",
                "Generic",
                "Exploits",
                "Fuzzers",
                "DoS",
                "Reconnaissance",
                "Analysis",
                "Backdoor",
                "Shellcode",
                "Worms",
            ],
        },
        "cv": {"outer_folds": 2, "inner_folds": 2, "shuffle": True},
        "feature_selection": {
            "n_estimators": 10,
            "criterion": "gini",
            "top_k": 5,
            "n_jobs": -1,
        },
        "smotenc": {"k_neighbors": 1, "target_count": 8},
        "models": {
            "enabled": ["DT", "RF", "SVM", "LR", "XGBoost"],
            "search_spaces": {
                "DT": {
                    "max_depth": {"type": "int", "low": 3, "high": 30},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                    "criterion": {"type": "categorical", "values": ["gini", "entropy"]},
                },
                "RF": {
                    "n_estimators": {"type": "int", "low": 100, "high": 200},
                    "max_depth": {"type": "int", "low": 5, "high": 10},
                    "min_samples_split": {"type": "int", "low": 2, "high": 5},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 3},
                    "max_features": {"type": "categorical", "values": ["sqrt", "log2", 0.5]},
                },
                "SVM": {
                    "C": {"type": "log_float", "low": -2, "high": 2},
                    "gamma": {"type": "log_float", "low": -4, "high": 0},
                },
                "LR": {
                    "C": {"type": "log_float", "low": -3, "high": 2},
                },
                "XGBoost": {
                    "n_estimators": {"type": "int", "low": 100, "high": 150},
                    "max_depth": {"type": "int", "low": 3, "high": 6},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                    "min_child_weight": {"type": "int", "low": 1, "high": 4},
                    "gamma": {"type": "float", "low": 0.0, "high": 2.0},
                },
            },
        },
        "gwo": {"population_size": 3, "iterations": 2},
        "logging": {"level": "INFO"},
        "reports": {"dpi": 100, "style": "whitegrid"},
        "smoke_test": {
            "rows_per_class_train": 6,
            "rows_per_class_test": 3,
            "outer_folds": 2,
            "inner_folds": 2,
            "gwo_population_size": 3,
            "gwo_iterations": 2,
        },
    }


@pytest.fixture
def synthetic_dataset(synthetic_config):
    rng = np.random.default_rng(42)
    classes = synthetic_config["dataset"]["expected_classes"]

    rows = []
    idx = 0
    for cls in classes:
        for _ in range(12):
            row = {
                "id": idx,
                "label": 0,
                "attack_cat": cls,
                "proto": rng.choice(["tcp", "udp", "icmp"]),
                "service": rng.choice(["http", "dns", "smtp"]),
                "state": rng.choice(["FIN", "CON", "INT"]),
            }
            for i in range(39):
                row[f"f{i}"] = float(rng.normal())
            rows.append(row)
            idx += 1

    df = pd.DataFrame(rows)
    return df
