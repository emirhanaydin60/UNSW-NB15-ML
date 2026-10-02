from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.checkpoint import CheckpointManager
from src.nested_cv import NestedCVExperiment, RunContext
from src.utils import upsert_csv_rows


def _sample_balanced_real_subset(df: pd.DataFrame, target: str, rows_per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for cls in sorted(df[target].astype(str).unique()):
        cls_df = df[df[target].astype(str) == cls]
        parts.append(cls_df.sample(n=rows_per_class, random_state=seed))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _make_experiment(tmp_path: Path, config: dict) -> NestedCVExperiment:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = type("Logger", (), {"info": lambda self, *args, **kwargs: None, "warning": lambda self, *args, **kwargs: None, "exception": lambda self, *args, **kwargs: None})()
    return NestedCVExperiment(
        config=config,
        run_context=RunContext(run_id="test_run", run_dir=run_dir, smoke_test=False),
        logger=logger,
        checkpoint_manager=CheckpointManager(run_dir),
    )


def test_outer_cv_does_not_load_official_test_dataset(monkeypatch, tmp_path, real_config, real_train_df):
    config = copy.deepcopy(real_config)
    config["cv"]["outer_folds"] = 5
    # revised methodology: inner CV removed; HPO split used instead

    subset = _sample_balanced_real_subset(real_train_df, config["dataset"]["target_column"], rows_per_class=10, seed=42)
    exp = _make_experiment(tmp_path, config)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("official test loader was called during outer CV")

    monkeypatch.setattr("src.nested_cv.load_test_dataset", fail_if_called)
    monkeypatch.setattr(NestedCVExperiment, "_run_single_outer_model", lambda *args, **kwargs: None)

    exp.run_outer_cv(subset)


def test_hpo_feature_selection_uses_hpo_training_only(monkeypatch, tmp_path, synthetic_config, synthetic_dataset):
    config = copy.deepcopy(synthetic_config)
    exp = _make_experiment(tmp_path, config)

    captured_rows: list[int] = []

    def fake_select_top_features(X_train_encoded, y_train, config, random_state):
        captured_rows.append(len(X_train_encoded))
        selected = list(X_train_encoded.columns[: config["feature_selection"]["top_k"]])
        rank_df = pd.DataFrame(
            {
                "feature": list(X_train_encoded.columns),
                "importance": np.linspace(1.0, 0.0, num=len(X_train_encoded.columns)),
            }
        )
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

    X_outer, y_outer = synthetic_dataset.drop(columns=["attack_cat", "id", "label"]), synthetic_dataset["attack_cat"]
    fitness_fn = exp._build_fitness_fn(
        model_name="RF",
        model_idx=0,
        outer_fold=1,
        X_outer_train_raw=X_outer.iloc[:60].reset_index(drop=True),
        y_outer_train=y_outer.iloc[:60].reset_index(drop=True),
    )
    result = fitness_fn({}, iteration=0, wolf_idx=0)

    assert result["fitness"] >= 0.0
    # feature selection must be called exactly once on the HPO-train partition
    assert len(captured_rows) == 1
    assert 0 < captured_rows[0] < 60


def test_smote_fallback_without_selected_categorical_features(tmp_path, synthetic_config, synthetic_dataset, caplog):
    config = copy.deepcopy(synthetic_config)
    config["smotenc"]["target_count"] = 20
    config["smotenc"]["k_neighbors"] = 1
    exp = _make_experiment(tmp_path, config)
    exp.logger = logging.getLogger(f"test-real-smote-fallback-{tmp_path.name}")

    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"])
    y = synthetic_dataset["attack_cat"]
    numeric_only = X[[c for c in X.columns if c not in config["dataset"]["categorical_columns"]]].copy()

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


def test_idempotent_csv_upsert(tmp_path):
    path = tmp_path / "outer_fold_metrics.csv"
    rows = [
        {"phase": "outer_cv", "model": "RF", "outer_fold": 1, "balanced_accuracy": 0.8},
        {"phase": "outer_cv", "model": "RF", "outer_fold": 1, "balanced_accuracy": 0.8},
    ]
    upsert_csv_rows(path, rows, ["phase", "model", "outer_fold", "balanced_accuracy"], ["phase", "model", "outer_fold"])
    df = pd.read_csv(path)
    assert len(df) == 1
    upsert_csv_rows(path, rows, ["phase", "model", "outer_fold", "balanced_accuracy"], ["phase", "model", "outer_fold"])
    df = pd.read_csv(path)
    assert len(df) == 1


@pytest.mark.usefixtures("real_config")
def test_real_small_end_to_end_pipeline(monkeypatch, tmp_path, real_config, real_train_df, real_test_df):
    config = copy.deepcopy(real_config)
    config["experiment"]["runs_root"] = str(tmp_path / "runs")
    config["experiment"]["save_models"] = False
    config["cv"]["outer_folds"] = 5
    config["smotenc"]["k_neighbors"] = 1
    config["smotenc"]["target_count"] = 4

    train_subset = _sample_balanced_real_subset(real_train_df, config["dataset"]["target_column"], rows_per_class=10, seed=7)
    test_subset = _sample_balanced_real_subset(real_test_df, config["dataset"]["target_column"], rows_per_class=5, seed=8)

    train_path = tmp_path / "train_subset.csv"
    test_path = tmp_path / "test_subset.csv"
    train_subset.to_csv(train_path, index=False)
    test_subset.to_csv(test_path, index=False)
    config["dataset"]["train_path"] = str(train_path)
    config["dataset"]["test_path"] = str(test_path)

    def fake_load_test_dataset(_config, logger=None):
        return pd.read_csv(test_path)

    def fake_optimize(self, fitness_fn, checkpoint_callback=None, resume_state=None, logger=None):
        from src.gwo import decode_position

        position = np.full(len(self.dimensions), 0.5, dtype=float)
        params = decode_position(position, self.dimensions)
        eval_result = fitness_fn(params, 0, 0)
        state = resume_state if resume_state is not None else self._initial_state()
        state.update(
            {
                "iteration": self.iterations,
                "next_wolf_index": 0,
                "positions": [position.tolist()] * self.population_size,
                "fitness_scores": [eval_result["fitness"]] * self.population_size,
                "alpha_position": position.tolist(),
                "beta_position": position.tolist(),
                "delta_position": position.tolist(),
                "alpha_score": float(eval_result["fitness"]),
                "beta_score": float(eval_result["fitness"]),
                "delta_score": float(eval_result["fitness"]),
                "best_params": params,
                "best_fitness": float(eval_result["fitness"]),
                "convergence_history": [
                    {
                        "iteration": 1,
                        "best_fitness": float(eval_result["fitness"]),
                        "alpha_score": float(eval_result["fitness"]),
                        "beta_score": float(eval_result["fitness"]),
                        "delta_score": float(eval_result["fitness"]),
                    }
                ],
                "evaluations": [
                    {
                        "iteration": 1,
                        "wolf_index": 0,
                        "params": params,
                        "fitness": float(eval_result["fitness"]),
                        "mean_inner_balanced_accuracy": float(eval_result["mean_balanced_accuracy"]),
                        "mean_inner_macro_f1": float(eval_result["mean_macro_f1"]),
                        "mean_balanced_accuracy": float(eval_result["mean_balanced_accuracy"]),
                        "mean_macro_f1": float(eval_result["mean_macro_f1"]),
                        "best_fitness_so_far": float(eval_result["fitness"]),
                        "became_alpha": True,
                        "failed": False,
                        "error": "",
                    }
                ],
                "rng_state": state.get("rng_state", {}),
            }
        )
        if checkpoint_callback is not None:
            checkpoint_callback("evaluation", state, state["evaluations"][0])
            checkpoint_callback("iteration_complete", state, None)
        return {
            "best_params": params,
            "best_fitness": float(eval_result["fitness"]),
            "convergence_history": state["convergence_history"],
            "evaluations": state["evaluations"],
            "state": state,
        }

    class DummyModel:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            if not hasattr(self, "classes_") or len(self.classes_) == 0:
                return np.zeros(X.shape[0], dtype=int)
            return np.full(X.shape[0], int(self.classes_[0]), dtype=int)

    monkeypatch.setattr("src.nested_cv.load_test_dataset", fake_load_test_dataset)
    monkeypatch.setattr("src.gwo.GreyWolfOptimizer.optimize", fake_optimize)
    monkeypatch.setattr("src.nested_cv.build_model", lambda **kwargs: DummyModel())

    exp = _make_experiment(tmp_path, config)
    exp.execute_full_pipeline(train_subset)

    results_dir = tmp_path / "run" / "results"
    assert (results_dir / "outer_fold_metrics.csv").exists()
    assert (results_dir / "outer_fold_predictions.csv").exists()
    assert (results_dir / "statistical_results.csv").exists()
    assert (results_dir / "final_test_metrics.csv").exists()
    assert (results_dir / "final_test_predictions.csv").exists()
    assert exp.state["outer_cv_complete"] is True
    assert exp.state["final_training_complete"] is True
    assert exp.state["reports_generated"] is True
