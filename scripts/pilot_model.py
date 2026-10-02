from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path.cwd()))

from src.configuration import load_config
from src.data_loader import load_training_dataset
from src.nested_cv import NestedCVExperiment, RunContext
from src.checkpoint import CheckpointManager
from src.gwo import GreyWolfOptimizer, decode_position
from src import models as models_module
from src.utils import derive_seed, utc_now_iso, ensure_dir


def optionally_set_system_env_threads(n_threads: int | None):
    if n_threads is None:
        return
    import os

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)


def run_pilot_for_model(model_name: str, config_path: str | Path, workers: int = 4, model_threads: int = 4) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg = dict(cfg)
    cfg["models"] = dict(cfg.get("models", {}))
    cfg["models"]["enabled"] = [model_name]
    cfg["gwo"] = dict(cfg.get("gwo", {}))
    cfg["gwo"]["population_size"] = 5
    cfg["gwo"]["iterations"] = 1
    cfg.setdefault("experiment", {})["workers"] = workers
    cfg.setdefault("experiment", {})["threads"] = model_threads

    run_id = f"pilot_{model_name.lower()}_{int(time.time())}"
    run_dir = Path.cwd() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(run_id=run_id, run_dir=run_dir, smoke_test=False)

    # lightweight logger
    class SimpleLogger:
        def info(self, *a, **k):
            print("INFO:", *a)

        def warning(self, *a, **k):
            print("WARN:", *a)

        def exception(self, *a, **k):
            print("EXC:", *a)

    logger = SimpleLogger()
    exp = NestedCVExperiment(config=cfg, run_context=ctx, logger=logger, checkpoint_manager=CheckpointManager(run_dir))

    train_df = load_training_dataset(cfg)
    target_col = cfg["dataset"]["target_column"]

    outer_folds = int(cfg["cv"]["outer_folds"])
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=bool(cfg["cv"].get("shuffle", True)), random_state=derive_seed(int(cfg["experiment"]["base_seed"]), "outer_cv"))

    results: dict[str, Any] = {"run_id": run_id, "start": utc_now_iso(), "outer_folds": []}

    fold_idx = 0
    total_candidate_count = 0
    total_success = 0
    total_failed = 0
    candidate_details = []

    for tr_idx, val_idx in skf.split(train_df, train_df[target_col].astype(str)):
        fold_idx += 1
        X_outer_train_raw = train_df.iloc[tr_idx].reset_index(drop=True).drop(columns=[target_col])
        y_outer_train = train_df[target_col].iloc[tr_idx].astype(str).reset_index(drop=True)

        fitness_fn = exp._build_fitness_fn(model_name=model_name, model_idx=0, outer_fold=fold_idx, X_outer_train_raw=X_outer_train_raw, y_outer_train=y_outer_train)

        optimizer = GreyWolfOptimizer(model_name=model_name, search_space_cfg=cfg["models"]["search_spaces"][model_name], population_size=5, iterations=1, seed=derive_seed(int(cfg["experiment"]["base_seed"]), "gwo", model_name, fold_idx))
        init = optimizer._initial_state()
        positions = init["positions"]

        fold_start = time.perf_counter()
        results_per_fold = []
        errors = []

        optionally_set_system_env_threads(model_threads)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for wolf_idx, pos in enumerate(positions):
                params = decode_position(np.array(pos, dtype=float), optimizer.dimensions)
                fut = ex.submit(fitness_fn, params, 0, wolf_idx)
                futures[fut] = (wolf_idx, params)

            for fut in as_completed(futures):
                wolf_idx, params = futures[fut]
                total_candidate_count += 1
                try:
                    res = fut.result()
                    results_per_fold.append({"wolf": wolf_idx, "params": params, "result": res})
                    total_success += 1
                    candidate_details.append({"outer_fold": fold_idx, "wolf": wolf_idx, "status": "success"})
                except Exception as exc:
                    errors.append({"wolf": wolf_idx, "params": params, "error": str(exc)})
                    total_failed += 1
                    candidate_details.append({"outer_fold": fold_idx, "wolf": wolf_idx, "status": "failed", "error": str(exc)})

        fold_elapsed = time.perf_counter() - fold_start
        results["outer_folds"].append({"outer_fold": fold_idx, "elapsed_seconds": fold_elapsed, "num_success": len(results_per_fold), "num_failed": len(errors)})

    results["end"] = utc_now_iso()
    results["total_elapsed_seconds"] = sum(f["elapsed_seconds"] for f in results["outer_folds"]) if results["outer_folds"] else 0.0
    results["total_candidates"] = total_candidate_count
    results["total_success"] = total_success
    results["total_failed"] = total_failed
    results["candidate_details"] = candidate_details

    metadata_dir = Path(exp.paths["metadata"])
    runtime_events_file = metadata_dir / "runtime_events.jsonl"
    runtime_events = []
    if runtime_events_file.exists():
        with runtime_events_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    runtime_events.append(json.loads(line))
                except Exception:
                    continue

    results["runtime_events_count"] = len(runtime_events)
    results["runtime_events_sample"] = runtime_events[:2]
    results["runtime_events_file"] = str(runtime_events_file)

    stage_totals = {}
    for ev in runtime_events:
        sd = ev.get("stage_durations", {}) or {}
        for k, v in sd.items():
            stage_totals[k] = stage_totals.get(k, 0.0) + float(v or 0.0)

    results["stage_totals"] = stage_totals

    summary_path = run_dir / "pilot_summary.json"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/pilot_model.py <MODEL_NAME>")
        print("Model names: DT, RF, SVM, LR, XGBoost")
        sys.exit(1)
    model = sys.argv[1]
    res = run_pilot_for_model(model, Path.cwd() / "config.yaml", workers=4, model_threads=4)
    print(json.dumps(res, indent=2))
