from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.checkpoint import CheckpointManager
from src.gwo import GreyWolfOptimizer
from src.utils import upsert_csv_rows


def test_checkpoint_save_load(tmp_path):
    run_dir = tmp_path / "run"
    ckpt = CheckpointManager(run_dir)
    state = {"a": 1, "b": {"c": 2}}
    ckpt.save_state(state)
    loaded = ckpt.load_state()
    assert loaded == state


def test_gwo_resume_behavior(tmp_path):
    run_dir = tmp_path / "run"
    ckpt = CheckpointManager(run_dir)
    gwo_key = "unit_test"

    opt = GreyWolfOptimizer(
        model_name="toy",
        search_space_cfg={"x": {"type": "float", "low": -1.0, "high": 1.0}},
        population_size=3,
        iterations=2,
        seed=42,
    )

    eval_counter = {"n": 0}

    def fitness_fn(params, _it, _wolf):
        eval_counter["n"] += 1
        return {
            "fitness": -abs(params["x"]),
            "mean_balanced_accuracy": 0.5,
            "mean_macro_f1": 0.5,
        }

    def cb(_event, state, _row):
        ckpt.save_gwo_state(gwo_key, state)

    partial = opt.optimize(fitness_fn=fitness_fn, checkpoint_callback=cb)
    assert eval_counter["n"] > 0

    saved = ckpt.load_gwo_state(gwo_key)
    assert saved is not None

    opt2 = GreyWolfOptimizer(
        model_name="toy",
        search_space_cfg={"x": {"type": "float", "low": -1.0, "high": 1.0}},
        population_size=3,
        iterations=2,
        seed=42,
    )
    resumed = opt2.optimize(fitness_fn=fitness_fn, resume_state=saved)
    assert "best_fitness" in resumed
    assert len(resumed["convergence_history"]) == 2
    assert len(partial["evaluations"]) <= len(resumed["evaluations"])


def test_csv_upsert_is_idempotent(tmp_path):
    path = tmp_path / "outer_fold_predictions.csv"
    rows = [
        {"phase": "outer_cv", "model": "RF", "outer_fold": 1, "row_in_fold": 0, "y_true": "Normal", "y_pred": "Normal"},
        {"phase": "outer_cv", "model": "RF", "outer_fold": 1, "row_in_fold": 0, "y_true": "Normal", "y_pred": "Normal"},
    ]
    fieldnames = ["phase", "model", "outer_fold", "row_in_fold", "y_true", "y_pred"]
    key_fields = ["phase", "model", "outer_fold", "row_in_fold"]

    upsert_csv_rows(path, rows, fieldnames, key_fields)
    upsert_csv_rows(path, rows, fieldnames, key_fields)

    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "RF"
