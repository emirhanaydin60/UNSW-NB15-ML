from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class SearchDimension:
    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = None


def build_search_dimensions(space_cfg: dict[str, Any]) -> list[SearchDimension]:
    dims: list[SearchDimension] = []
    for param_name, cfg in space_cfg.items():
        kind = str(cfg["type"])
        if kind == "categorical":
            dims.append(
                SearchDimension(
                    name=param_name,
                    kind=kind,
                    values=list(cfg["values"]),
                )
            )
        else:
            dims.append(
                SearchDimension(
                    name=param_name,
                    kind=kind,
                    low=float(cfg["low"]),
                    high=float(cfg["high"]),
                )
            )
    return dims


def decode_position(position: np.ndarray, dimensions: list[SearchDimension]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for i, dim in enumerate(dimensions):
        x = float(np.clip(position[i], 0.0, 1.0))
        if dim.kind == "int":
            assert dim.low is not None and dim.high is not None
            raw = dim.low + x * (dim.high - dim.low)
            params[dim.name] = int(np.clip(round(raw), dim.low, dim.high))
        elif dim.kind == "float":
            assert dim.low is not None and dim.high is not None
            params[dim.name] = float(dim.low + x * (dim.high - dim.low))
        elif dim.kind == "log_float":
            assert dim.low is not None and dim.high is not None
            exponent = dim.low + x * (dim.high - dim.low)
            params[dim.name] = float(10**exponent)
        elif dim.kind == "categorical":
            assert dim.values is not None and len(dim.values) > 0
            idx = int(np.clip(round(x * (len(dim.values) - 1)), 0, len(dim.values) - 1))
            params[dim.name] = dim.values[idx]
        else:
            raise ValueError(f"Unsupported search dimension type: {dim.kind}")
    return params


FitnessFn = Callable[[dict[str, Any], int, int], dict[str, float]]
CheckpointCallback = Callable[[str, dict[str, Any], dict[str, Any] | None], None]


class GreyWolfOptimizer:
    def __init__(
        self,
        model_name: str,
        search_space_cfg: dict[str, Any],
        population_size: int,
        iterations: int,
        seed: int,
        poor_fitness_value: float = -1.0,
    ):
        self.model_name = model_name
        self.dimensions = build_search_dimensions(search_space_cfg)
        self.population_size = int(population_size)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.poor_fitness_value = float(poor_fitness_value)

    def _initial_state(self) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        positions = rng.random((self.population_size, len(self.dimensions)))
        return {
            "model_name": self.model_name,
            "population_size": self.population_size,
            "iterations": self.iterations,
            "iteration": 0,
            "next_wolf_index": 0,
            "positions": positions.tolist(),
            "fitness_scores": [None] * self.population_size,
            "alpha_position": None,
            "beta_position": None,
            "delta_position": None,
            "alpha_score": float("-inf"),
            "beta_score": float("-inf"),
            "delta_score": float("-inf"),
            "best_params": None,
            "best_fitness": float("-inf"),
            "convergence_history": [],
            "evaluations": [],
            "rng_state": rng.bit_generator.state,
            "failed_evaluations": 0,
        }

    def _promote_wolf(self, state: dict[str, Any], position: np.ndarray, fitness: float, params: dict[str, Any]) -> None:
        if fitness > state["alpha_score"]:
            state["delta_score"] = state["beta_score"]
            state["delta_position"] = state["beta_position"]
            state["beta_score"] = state["alpha_score"]
            state["beta_position"] = state["alpha_position"]
            state["alpha_score"] = float(fitness)
            state["alpha_position"] = position.tolist()
        elif fitness > state["beta_score"]:
            state["delta_score"] = state["beta_score"]
            state["delta_position"] = state["beta_position"]
            state["beta_score"] = float(fitness)
            state["beta_position"] = position.tolist()
        elif fitness > state["delta_score"]:
            state["delta_score"] = float(fitness)
            state["delta_position"] = position.tolist()

        if fitness > state["best_fitness"]:
            state["best_fitness"] = float(fitness)
            state["best_params"] = params

    def optimize(
        self,
        fitness_fn: FitnessFn,
        checkpoint_callback: CheckpointCallback | None = None,
        resume_state: dict[str, Any] | None = None,
        logger: Any = None,
    ) -> dict[str, Any]:
        state = resume_state if resume_state is not None else self._initial_state()

        rng = np.random.default_rng(self.seed)
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]

        while int(state["iteration"]) < self.iterations:
            iteration = int(state["iteration"])
            positions = np.asarray(state["positions"], dtype=float)
            fitness_scores = list(state["fitness_scores"])
            start_wolf = int(state.get("next_wolf_index", 0))

            for wolf_idx in range(start_wolf, self.population_size):
                position = positions[wolf_idx]
                params = decode_position(position, self.dimensions)
                eval_result: dict[str, float]
                failed = False
                error_message = ""

                try:
                    eval_result = fitness_fn(params, iteration, wolf_idx)
                    fitness = float(eval_result["fitness"])
                except Exception:
                    fitness = self.poor_fitness_value
                    eval_result = {
                        "fitness": fitness,
                        "mean_balanced_accuracy": self.poor_fitness_value,
                        "mean_macro_f1": self.poor_fitness_value,
                    }
                    failed = True
                    error_message = traceback.format_exc(limit=3)
                    state["failed_evaluations"] = int(state.get("failed_evaluations", 0)) + 1

                fitness_scores[wolf_idx] = fitness
                self._promote_wolf(state, position=position, fitness=fitness, params=params)

                row = {
                    "iteration": iteration + 1,
                    "wolf_index": wolf_idx,
                    "params": params,
                    "fitness": float(eval_result["fitness"]),
                    "mean_inner_balanced_accuracy": float(eval_result["mean_balanced_accuracy"]),
                    "mean_inner_macro_f1": float(eval_result["mean_macro_f1"]),
                    "best_fitness_so_far": float(state["best_fitness"]),
                    "became_alpha": bool(abs(fitness - state["alpha_score"]) < 1e-12),
                    "failed": failed,
                    "error": error_message,
                }
                state["evaluations"].append(row)
                state["fitness_scores"] = fitness_scores
                state["next_wolf_index"] = wolf_idx + 1
                state["positions"] = positions.tolist()
                state["rng_state"] = rng.bit_generator.state

                if logger is not None:
                    logger.info(
                        "GWO fitness evaluated",
                        extra={
                            "gwo_iter": f"{iteration + 1}/{self.iterations}",
                            "wolf": wolf_idx,
                            "best_fitness": f"{state['best_fitness']:.6f}",
                        },
                    )

                if checkpoint_callback is not None:
                    checkpoint_callback("evaluation", state, row)

            state["convergence_history"].append(
                {
                    "iteration": iteration + 1,
                    "best_fitness": float(state["best_fitness"]),
                    "alpha_score": float(state["alpha_score"]),
                    "beta_score": float(state["beta_score"]),
                    "delta_score": float(state["delta_score"]),
                }
            )

            if int(state["iteration"]) == self.iterations - 1:
                state["iteration"] = self.iterations
                state["next_wolf_index"] = 0
                state["fitness_scores"] = [None] * self.population_size
                state["rng_state"] = rng.bit_generator.state
                if checkpoint_callback is not None:
                    checkpoint_callback("iteration_complete", state, None)
                break

            a = 2 - (2 * iteration / max(1, self.iterations - 1))
            alpha = np.asarray(state["alpha_position"], dtype=float)
            beta = np.asarray(state["beta_position"], dtype=float)
            delta = np.asarray(state["delta_position"], dtype=float)

            for i in range(self.population_size):
                for d in range(len(self.dimensions)):
                    r1, r2 = rng.random(), rng.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[d] - positions[i, d])
                    X1 = alpha[d] - A1 * D_alpha

                    r1, r2 = rng.random(), rng.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[d] - positions[i, d])
                    X2 = beta[d] - A2 * D_beta

                    r1, r2 = rng.random(), rng.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[d] - positions[i, d])
                    X3 = delta[d] - A3 * D_delta

                    positions[i, d] = np.clip((X1 + X2 + X3) / 3.0, 0.0, 1.0)

            state["positions"] = positions.tolist()
            state["fitness_scores"] = [None] * self.population_size
            state["iteration"] = iteration + 1
            state["next_wolf_index"] = 0
            state["rng_state"] = rng.bit_generator.state

            if checkpoint_callback is not None:
                checkpoint_callback("iteration_complete", state, None)

        return {
            "best_params": state["best_params"],
            "best_fitness": float(state["best_fitness"]),
            "convergence_history": state["convergence_history"],
            "evaluations": state["evaluations"],
            "state": state,
        }
