from __future__ import annotations

import numpy as np

from src.gwo import GreyWolfOptimizer, build_search_dimensions, decode_position


def test_decode_position_boundaries():
    dims = build_search_dimensions(
        {
            "a": {"type": "int", "low": 1, "high": 10},
            "b": {"type": "float", "low": 0.1, "high": 0.9},
            "c": {"type": "log_float", "low": -2, "high": 2},
            "d": {"type": "categorical", "values": ["x", "y", "z"]},
        }
    )
    low = decode_position(np.array([0.0, 0.0, 0.0, 0.0]), dims)
    high = decode_position(np.array([1.0, 1.0, 1.0, 1.0]), dims)

    assert low["a"] == 1
    assert high["a"] == 10
    assert 0.1 <= low["b"] <= 0.9
    assert 0.1 <= high["b"] <= 0.9
    assert low["d"] == "x"
    assert high["d"] == "z"


def test_gwo_runs_and_returns_best():
    space = {
        "x": {"type": "float", "low": -5.0, "high": 5.0},
    }
    opt = GreyWolfOptimizer(
        model_name="toy",
        search_space_cfg=space,
        population_size=4,
        iterations=3,
        seed=42,
    )

    def fitness_fn(params, _it, _wolf):
        x = params["x"]
        score = -abs(x)
        return {
            "fitness": score,
            "mean_balanced_accuracy": score,
            "mean_macro_f1": score,
        }

    out = opt.optimize(fitness_fn)
    assert "best_params" in out
    assert "best_fitness" in out
    assert len(out["convergence_history"]) == 3
