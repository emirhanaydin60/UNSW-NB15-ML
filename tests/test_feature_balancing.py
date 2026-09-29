from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from src.balancing import apply_smotenc, build_sampling_strategy
from src.feature_selection import select_top_features
from src.preprocessing import FoldPreprocessor, get_categorical_indices, get_selected_categorical_columns
from src.checkpoint import CheckpointManager
from src.nested_cv import NestedCVExperiment, RunContext


def test_feature_selection_top_k(synthetic_config, synthetic_dataset):
    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"])
    y = synthetic_dataset["attack_cat"]

    pre = FoldPreprocessor(categorical_columns=synthetic_config["dataset"]["categorical_columns"])
    pre.fit(X)
    X_enc = pre.transform(X)

    selected, rank_df = select_top_features(X_enc, y, synthetic_config, random_state=42)
    assert len(selected) == synthetic_config["feature_selection"]["top_k"]
    assert len(rank_df) == X_enc.shape[1]


def test_sampling_strategy_and_smotenc(synthetic_config, synthetic_dataset):
    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"]).copy()
    y = synthetic_dataset["attack_cat"].copy()

    pre = FoldPreprocessor(categorical_columns=synthetic_config["dataset"]["categorical_columns"])
    pre.fit(X)
    X_enc = pre.transform(X)

    selected = list(X_enc.columns[:8])
    X_sel = X_enc[selected]
    selected_cat = get_selected_categorical_columns(selected, synthetic_config["dataset"]["categorical_columns"])
    cat_idx = get_categorical_indices(X_sel.columns, selected_cat)

    strategy = build_sampling_strategy(y, target_count=20)
    assert all(v == 20 for v in strategy.values())

    X_res, y_res, before, after, _ = apply_smotenc(
        X_sel,
        y,
        categorical_feature_indices=cat_idx,
        smote_config={"k_neighbors": 1, "target_count": 20},
        random_state=42,
    )
    assert len(X_res) == len(y_res)
    assert len(y_res) > len(y)
    assert set(before).issubset(set(after))


def test_feature_selection_uses_inner_training_only(monkeypatch, tmp_path, synthetic_config, synthetic_dataset):
    config = copy.deepcopy(synthetic_config)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = type("Logger", (), {"info": lambda self, *args, **kwargs: None, "warning": lambda self, *args, **kwargs: None, "exception": lambda self, *args, **kwargs: None})()
    exp = NestedCVExperiment(config=config, run_context=RunContext(run_id="unit", run_dir=run_dir, smoke_test=False), logger=logger, checkpoint_manager=CheckpointManager(run_dir))

    seen_rows: list[int] = []

    def fake_select_top_features(X_train_encoded, y_train, config, random_state):
        seen_rows.append(len(X_train_encoded))
        selected = list(X_train_encoded.columns[: config["feature_selection"]["top_k"]])
        rank_df = pd.DataFrame({"feature": list(X_train_encoded.columns), "importance": np.linspace(1.0, 0.0, num=len(X_train_encoded.columns))})
        rank_df["rank"] = range(1, len(rank_df) + 1)
        rank_df["selected"] = rank_df["feature"].isin(selected)
        return selected, rank_df

    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    monkeypatch.setattr("src.nested_cv.select_top_features", fake_select_top_features)
    monkeypatch.setattr("src.nested_cv.build_model", lambda **kwargs: DummyModel())

    X_outer = synthetic_dataset.drop(columns=["attack_cat", "id", "label"]).reset_index(drop=True)
    y_outer = synthetic_dataset["attack_cat"].reset_index(drop=True)
    fitness_fn = exp._build_fitness_fn("RF", 0, 1, X_outer.iloc[:60], y_outer.iloc[:60])
    result = fitness_fn({}, 0, 0)

    assert result["fitness"] >= 0.0
    assert seen_rows == [30, 30]


def test_smote_fallback_has_no_auxiliary_feature_injection(tmp_path, synthetic_config, synthetic_dataset, caplog):
    config = copy.deepcopy(synthetic_config)
    config["smotenc"]["target_count"] = 20
    config["smotenc"]["k_neighbors"] = 1
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"test-smote-fallback-{run_dir.name}")
    exp = NestedCVExperiment(config=config, run_context=RunContext(run_id="unit", run_dir=run_dir, smoke_test=False), logger=logger, checkpoint_manager=CheckpointManager(run_dir))

    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"])
    y = synthetic_dataset["attack_cat"]
    numeric_only = X[[c for c in X.columns if c not in config["dataset"]["categorical_columns"]]]

    caplog.set_level("INFO")
    X_bal, y_bal, before, after, _strategy, method = exp._apply_balancing_with_policy(
        numeric_only,
        y,
        categorical_indices=[],
        stage_random_state=42,
        log_context={"stage": "unit_test"},
    )

    assert method == "smote"
    assert list(X_bal.columns) == list(numeric_only.columns)
    assert before == y.value_counts().sort_index().to_dict()
    assert after == y_bal.value_counts().sort_index().to_dict()
    assert any("SMOTE_USED_BECAUSE_NO_SELECTED_CATEGORICAL_FEATURES" in record.message for record in caplog.records)
