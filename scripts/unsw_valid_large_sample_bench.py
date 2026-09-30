"""Validate and benchmark on a real large stratified UNSW-NB15 sample.

This script follows the user's scientific constraints:
- does NOT modify `k_neighbors` or SMOTE behavior
- does NOT load the official test set
- performs fold-local viability checks for SMOTENC(k=5)
- samples 25k rows (increases to 50k if needed)
- runs one representative inner-CV fitness evaluation per model
- runs one 10-wolf GWO iteration for prioritized XGBoost configs
- writes incremental JSON to Temp/unsw_combined_parallel_bench_results.json

Run with the project's conda env (pytorch-v1).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nested_cv import NestedCVExperiment, RunContext
from src.gwo import GreyWolfOptimizer, decode_position
from src.data_loader import load_training_dataset
from src.checkpoint import CheckpointManager
from src.logger import setup_logger
from src.utils import utc_now_iso, get_system_metadata
import src.models as models_module

JSON_PATH = Path.cwd() / "Temp" / "unsw_combined_parallel_bench_results.json"

SAMPLE_TRIES = [25000, 50000]
XGB_CONFIGS = [(4, 4), (6, 2), (8, 1), (10, 1)]
OTHER_MODELS = ["DT", "RF", "SVM", "LR"]
OTHER_WORKER_THREADS = {
    "DT": (4, 1),
    "RF": (4, 4),
    "SVM": (4, 4),
    "LR": (4, 4),
}


def ensure_temp_json():
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JSON_PATH.exists():
        base = {
            "benchmark": "unsw_valid_large_sample_bench",
            "timestamp": utc_now_iso(),
            "environment": {},
            "dataset": {},
            "measurements": [],
        }
        with JSON_PATH.open("w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)
            f.flush()
            os.fsync(f.fileno())


def load_json():
    with JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict[str, Any]):
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def measurement_key(entry: dict[str, Any]) -> str:
    return f"{entry['model']}|{entry.get('workers')}|{entry.get('model_threads')}|{entry.get('sample_size')}"


def stratified_sample(train_df: pd.DataFrame, n: int, target_col: str) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedShuffleSplit

    y = train_df[target_col].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=42)
    for train_idx, _ in sss.split(train_df, y):
        return train_df.iloc[train_idx].reset_index(drop=True)
    return train_df.sample(n=min(n, len(train_df)), random_state=42).reset_index(drop=True)


def check_sample_viability(sample_df: pd.DataFrame, config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check that for every inner-fold training split, any class that will be upsampled has > k_neighbors samples.

    Returns (viable, details)
    """
    target_col = config["dataset"]["target_column"]
    inner_folds = int(config["cv"]["inner_folds"])
    skf_inner = __import__("sklearn").model_selection.StratifiedKFold(n_splits=inner_folds, shuffle=bool(config["cv"].get("shuffle", True)), random_state=42)

    details = {
        "sample_rows": len(sample_df),
        "class_counts": sample_df[target_col].astype(str).value_counts().to_dict(),
        "inner_fold_issues": [],
    }

    target_count = int(config["smotenc"]["target_count"])
    k_neighbors = int(config["smotenc"]["k_neighbors"])

    X = sample_df.drop(columns=[target_col])
    y = sample_df[target_col].astype(str)

    for fold_idx, (train_idx, val_idx) in enumerate(skf_inner.split(X, y), start=1):
        y_train = y.iloc[train_idx]
        counts = y_train.value_counts().to_dict()
        for cls, cnt in counts.items():
            if int(cnt) < target_count:
                # SMOTE/SMOTENC will target this class; ensure cnt > k_neighbors
                if int(cnt) <= k_neighbors:
                    details["inner_fold_issues"].append(
                        {
                            "inner_fold": int(fold_idx),
                            "class": cls,
                            "count_in_train": int(cnt),
                            "required_min": int(k_neighbors) + 1,
                        }
                    )

    viable = len(details["inner_fold_issues"]) == 0
    return viable, details


def build_middle_params_for_model(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    # use midpoint (0.5) in each search dimension
    search_space = config["models"]["search_spaces"][model_name]
    optimizer = GreyWolfOptimizer(model_name=model_name, search_space_cfg=search_space, population_size=10, iterations=1, seed=42)
    dims = optimizer.dimensions
    pos = np.full((len(dims),), 0.5)
    params = decode_position(pos, dims)
    return params


def run_fitness_once(experiment: NestedCVExperiment, model_name: str, sample_df: pd.DataFrame) -> dict[str, Any]:
    target_col = experiment.config["dataset"]["target_column"]
    X_outer_train_raw = sample_df.drop(columns=[target_col])
    y_outer_train = sample_df[target_col].astype(str)

    fitness_fn = experiment._build_fitness_fn(model_name=model_name, model_idx=0, outer_fold=1, X_outer_train_raw=X_outer_train_raw, y_outer_train=y_outer_train)
    params = build_middle_params_for_model(experiment.config, model_name)

    t0 = time.perf_counter()
    try:
        res = fitness_fn(params, 0, 0)
        elapsed = time.perf_counter() - t0
        return {"status": "success", "elapsed_seconds": elapsed, "result": res}
    except Exception as exc:
        return {"status": "failed", "elapsed_seconds": None, "error": str(exc)}


def run_gwo_one_iteration_concurrent(experiment: NestedCVExperiment, model_name: str, sample_df: pd.DataFrame, workers: int, model_threads: int) -> dict[str, Any]:
    # build fitness fn
    target_col = experiment.config["dataset"]["target_column"]
    X_outer_train_raw = sample_df.drop(columns=[target_col])
    y_outer_train = sample_df[target_col].astype(str)
    fitness_fn = experiment._build_fitness_fn(model_name=model_name, model_idx=0, outer_fold=1, X_outer_train_raw=X_outer_train_raw, y_outer_train=y_outer_train)

    # prepare optimizer positions
    search_space = experiment.config["models"]["search_spaces"][model_name]
    optimizer = GreyWolfOptimizer(model_name=model_name, search_space_cfg=search_space, population_size=10, iterations=1, seed=42)
    init = optimizer._initial_state()
    positions = init["positions"]

    # monkeypatch model builder to set threads for XGBoost/RF locally
    _ORIG_BUILD = models_module.build_model

    def build_model_override(name: str, params: dict[str, Any], random_state: int, num_classes: int):
        if name == "XGBoost":
            from xgboost import XGBClassifier

            nj = model_threads
            return XGBClassifier(objective="multi:softprob", num_class=num_classes, random_state=random_state, n_jobs=int(nj), verbosity=0, tree_method="hist")
        if name == "RF":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(random_state=random_state, n_jobs=int(model_threads))
        return _ORIG_BUILD(name, params, random_state, num_classes)

    models_module.build_model = build_model_override

    t0 = time.perf_counter()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for idx, pos in enumerate(positions):
            params = decode_position(np.array(pos, dtype=float), optimizer.dimensions)
            fut = ex.submit(fitness_fn, params, 0, idx)
            futures[fut] = (idx, params)

        for fut in as_completed(futures):
            idx, params = futures[fut]
            try:
                res = fut.result()
                results.append({"wolf": idx, "params": params, "result": res})
            except Exception as exc:
                errors.append({"wolf": idx, "params": params, "error": str(exc)})

    elapsed = time.perf_counter() - t0
    models_module.build_model = _ORIG_BUILD

    return {
        "elapsed_seconds": elapsed,
        "num_success": len(results),
        "num_failed": len(errors),
        "errors": errors,
    }


def upsert_measurement(entry: dict[str, Any]):
    ensure_temp_json()
    base = load_json()
    measurements = base.get("measurements", [])
    # remove duplicates
    measurements = [m for m in measurements if measurement_key(m) != measurement_key(entry)]
    measurements.append(entry)
    base["measurements"] = measurements
    base["timestamp"] = utc_now_iso()
    base.setdefault("dataset", {})["last_sample_rows"] = entry.get("sample_size")
    write_json(base)


def main():
    ensure_temp_json()

    # metadata
    try:
        sys_meta = get_system_metadata()
    except Exception:
        sys_meta = {}

    base = load_json()
    base["environment"] = {"python": sys.executable, "system": sys_meta}
    write_json(base)

    # load config and data
    import yaml

    cfg_path = Path.cwd() / "config" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path.cwd() / "config" / "config.example.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_df = load_training_dataset(config)

    # try sample sizes in order
    chosen_sample = None
    chosen_details = None
    for n in SAMPLE_TRIES:
        if len(train_df) < n:
            continue
        sample_df = stratified_sample(train_df, n, config["dataset"]["target_column"])
        viable, details = check_sample_viability(sample_df, config)
        details["attempted_sample_size"] = n
        upsert_measurement({
            "model": "SAMPLE_VALIDATION",
            "sample_size": int(n),
            "status": "success" if viable else "invalid",
            "details": details,
            "timestamp": utc_now_iso(),
        })
        if viable:
            chosen_sample = sample_df
            chosen_details = details
            break

    if chosen_sample is None:
        print("No viable sample found (25k and 50k insufficient). Exiting.")
        return

    # prepare experiment skeleton but do NOT run full outer CV
    run_dir = Path.cwd() / "runs" / f"valid_bench_{int(time.time())}"
    ckpt = CheckpointManager(run_dir)
    logger = setup_logger(run_dir / "bench.log")
    ctx = RunContext(run_id=f"validbench_{int(time.time())}", run_dir=run_dir, smoke_test=False)
    experiment = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)

    # run representative fitness for each model
    for model_name in experiment.model_names:
        print("Running representative fitness for", model_name)
        res = run_fitness_once(experiment, model_name, chosen_sample)
        entry = {
            "model": model_name,
            "sample_size": int(len(chosen_sample)),
            "workers": None,
            "model_threads": None,
            "elapsed_seconds": res.get("elapsed_seconds"),
            "status": res.get("status"),
            "error": res.get("error") if res.get("status") != "success" else None,
            "timestamp": utc_now_iso(),
        }
        upsert_measurement(entry)

    # XGBoost prioritized GWO one-iteration tests
    for workers, threads in XGB_CONFIGS:
        print(f"Running XGBoost GWO one-iteration: workers={workers} threads={threads}")
        try:
            res = run_gwo_one_iteration_concurrent(experiment, "XGBoost", chosen_sample, workers=workers, model_threads=threads)
            status = "success" if res["num_failed"] == 0 else "invalid"
            entry = {
                "model": "XGBoost",
                "sample_size": int(len(chosen_sample)),
                "workers": int(workers),
                "model_threads": int(threads),
                "elapsed_seconds": float(res.get("elapsed_seconds", 0.0)),
                "status": status,
                "error": res.get("errors") if res.get("num_failed", 0) > 0 else None,
                "number_of_successful_wolves": int(res.get("num_success", 0)),
                "number_of_inner_folds": int(config["cv"]["inner_folds"]),
                "timestamp": utc_now_iso(),
            }
        except Exception as exc:
            entry = {
                "model": "XGBoost",
                "sample_size": int(len(chosen_sample)),
                "workers": int(workers),
                "model_threads": int(threads),
                "elapsed_seconds": None,
                "status": "failed",
                "error": str(exc),
                "number_of_successful_wolves": 0,
                "number_of_inner_folds": int(config["cv"]["inner_folds"]),
                "timestamp": utc_now_iso(),
            }
        upsert_measurement(entry)

    # Other models: run one 10-wolf iteration with chosen worker/thread hints
    for model_name in OTHER_MODELS:
        workers, threads = OTHER_WORKER_THREADS.get(model_name, (4, 4))
        print(f"Running {model_name} GWO one-iteration: workers={workers} threads={threads}")
        try:
            res = run_gwo_one_iteration_concurrent(experiment, model_name, chosen_sample, workers=workers, model_threads=threads)
            status = "success" if res["num_failed"] == 0 else "invalid"
            entry = {
                "model": model_name,
                "sample_size": int(len(chosen_sample)),
                "workers": int(workers),
                "model_threads": int(threads),
                "elapsed_seconds": float(res.get("elapsed_seconds", 0.0)),
                "status": status,
                "error": res.get("errors") if res.get("num_failed", 0) > 0 else None,
                "number_of_successful_wolves": int(res.get("num_success", 0)),
                "number_of_inner_folds": int(config["cv"]["inner_folds"]),
                "timestamp": utc_now_iso(),
            }
        except Exception as exc:
            entry = {
                "model": model_name,
                "sample_size": int(len(chosen_sample)),
                "workers": int(workers),
                "model_threads": int(threads),
                "elapsed_seconds": None,
                "status": "failed",
                "error": str(exc),
                "number_of_successful_wolves": 0,
                "number_of_inner_folds": int(config["cv"]["inner_folds"]),
                "timestamp": utc_now_iso(),
            }
        upsert_measurement(entry)

    print("Benchmark complete. Results appended to", str(JSON_PATH))


if __name__ == "__main__":
    main()
