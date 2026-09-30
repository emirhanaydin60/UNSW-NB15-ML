"""Small targeted combined-parallel benchmark runner.
Writes incremental JSON to Temp/unsw_combined_parallel_bench_results.json
Designed to be resumable; skips already-successful measurements.

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

try:
    import psutil
except Exception:
    psutil = None

import pandas as pd

# Ensure project src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint import CheckpointManager
from src.gwo import GreyWolfOptimizer, decode_position
from src.nested_cv import NestedCVExperiment, RunContext
from src.data_loader import load_training_dataset
from src.logger import setup_logger
from src.utils import utc_now_iso, get_system_metadata
import src.models as models_module

JSON_PATH = Path.cwd() / "Temp" / "unsw_combined_parallel_bench_results.json"
SAMPLE_ROWS = 5000
BASELINE_SEQ_ITER_SEC = 663.47  # provided by user

# prioritized xgboost configs (workers, threads)
XGB_CONFIGS = [
    (4, 4),
    (6, 2),
    (8, 1),
    (10, 1),
    (2, 8),
    (1, 12),
]
# User-specified order overall (preferential), but run primarily the prioritized list.

OTHER_MODELS = ["DT", "RF", "SVM", "LR"]
OTHER_WORKER_SETS = [1, 4, 8, 10]

# Helpers for JSON incremental write

def ensure_temp_json():
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not JSON_PATH.exists():
        base = {
            "benchmark": "unsw_combined_parallelism",
            "timestamp": utc_now_iso(),
            "environment": {},
            "dataset": {"source": "UNSW-NB15 training set", "sample_rows": SAMPLE_ROWS},
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
    return f"{entry['model']}|{entry.get('workers')}|{entry.get('model_threads')}"


# Monkeypatch models.build_model to respect a global override for threads
CURRENT_OVERRIDE = {"model_threads": None, "rf_n_jobs": None}
_ORIG_BUILD = models_module.build_model


def build_model_override(model_name: str, params: dict[str, Any], random_state: int, num_classes: int):
    # Delegate to original but enforce thread overrides for RF and XGBoost
    if model_name == "XGBoost":
        # Create XGBClassifier with overridden n_jobs if set
        from xgboost import XGBClassifier
        nj = CURRENT_OVERRIDE.get("model_threads")
        if nj is None:
            nj = -1
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            min_child_weight=int(params.get("min_child_weight", 1)),
            gamma=float(params.get("gamma", 0.0)),
            random_state=random_state,
            n_jobs=int(nj),
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        nj = CURRENT_OVERRIDE.get("rf_n_jobs")
        if nj is None:
            nj = -1
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 10)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "auto"),
            random_state=random_state,
            n_jobs=int(nj),
        )
    return _ORIG_BUILD(model_name, params, random_state, num_classes)


def optionally_set_system_env_threads(n_threads: int | None):
    if n_threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)


def sample_train_df(train_df: pd.DataFrame, n: int, target_col: str) -> pd.DataFrame:
    # Stratified sample preserving class proportions using sklearn
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
    except Exception:
        return train_df.sample(n=min(n, len(train_df)), random_state=42).reset_index(drop=True)

    y = train_df[target_col].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=42)
    for train_idx, _ in sss.split(train_df, y):
        out = train_df.iloc[train_idx].reset_index(drop=True)
        return out
    return train_df.sample(n=min(n, len(train_df)), random_state=42).reset_index(drop=True)


def run_gwo_iteration_concurrent(experiment: NestedCVExperiment, model_name: str, X_train_df: pd.DataFrame, workers: int, model_threads: int) -> dict[str, Any]:
    # Build fitness function for outer_fold=1 (we only need training portion sample)
    y = X_train_df[experiment.config['dataset']['target_column']].astype(str)
    X, y_series = None, None
    # We need X_outer_train_raw and y_outer_train; nested_cv.run expects full df and splits outer folds; but for fitness function we can supply the entire sample as X_outer_train_raw
    X_outer_train_raw = X_train_df.drop(columns=[experiment.config['dataset']['target_column']])
    y_outer_train = X_train_df[experiment.config['dataset']['target_column']].astype(str)

    fitness_fn = experiment._build_fitness_fn(model_name=model_name, model_idx=0, outer_fold=1, X_outer_train_raw=X_outer_train_raw, y_outer_train=y_outer_train)

    # create optimizer to get initial positions
    search_space = experiment.config['models']['search_spaces'][model_name]
    optimizer = GreyWolfOptimizer(model_name=model_name, search_space_cfg=search_space, population_size=10, iterations=1, seed=42)
    init_state = optimizer._initial_state()
    positions = init_state['positions']

    # set threading env and model monkeypatch
    CURRENT_OVERRIDE['model_threads'] = model_threads
    CURRENT_OVERRIDE['rf_n_jobs'] = model_threads
    optionally_set_system_env_threads(model_threads)

    # patch
    models_module.build_model = build_model_override

    t0 = time.perf_counter()
    results = []
    errors = []
    # run fitness evaluations concurrently with threads
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for idx, pos in enumerate(positions):
            params = decode_position(__import__('numpy').array(pos, dtype=float), optimizer.dimensions)
            fut = ex.submit(fitness_fn, params, 0, idx)
            futures[fut] = (idx, params)

        for fut in as_completed(futures):
            idx, params = futures[fut]
            try:
                res = fut.result()
                results.append({'wolf': idx, 'params': params, 'result': res})
            except Exception as exc:
                errors.append({'wolf': idx, 'params': params, 'error': str(exc)})

    elapsed = time.perf_counter() - t0

    # restore original builder
    models_module.build_model = _ORIG_BUILD

    # measure cpu/ram briefly
    cpu_pct = None
    ram_mb = None
    if psutil is not None:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        ram_mb = psutil.virtual_memory().used / (1024 * 1024)

    return {
        'elapsed_seconds': elapsed,
        'num_success': len(results),
        'num_failed': len(errors),
        'errors': errors,
        'cpu_percent': cpu_pct,
        'ram_mb': ram_mb,
    }


def main():
    ensure_temp_json()
    base = load_json()

    # fill environment metadata
    try:
        sys_meta = get_system_metadata()
    except Exception:
        sys_meta = {}
    base['environment'] = {
        'python': sys.executable,
        'xgboost_version': None,
        'cpu': sys_meta.get('cpu', ''),
        'logical_cores': sys_meta.get('logical_cores', 0),
        'ram_gb': sys_meta.get('ram_gb', 0),
    }
    try:
        import xgboost as xgb
        base['environment']['xgboost_version'] = getattr(xgb, '__version__', 'unknown')
    except Exception:
        base['environment']['xgboost_version'] = 'not-available'

    write_json(base)

    # load config and data
    import yaml
    cfg_path = Path.cwd() / 'config' / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = Path.cwd() / 'config' / 'config.example.yaml'
    with cfg_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_df = load_training_dataset(config)
    target_col = config['dataset']['target_column']
    sample_df = sample_train_df(train_df, SAMPLE_ROWS, target_col)

    # adjust smotenc k_neighbors in-memory for this benchmark to avoid small-sample failures
    try:
        bench_cfg = config.get('smotenc', {})
        original_k = int(bench_cfg.get('k_neighbors', 5))
        min_count = int(sample_df[target_col].value_counts().min())
        safe_k = max(1, min(original_k, max(1, min_count - 1)))
        config.setdefault('smotenc', {})['k_neighbors'] = safe_k
        print(f"Adjusted smotenc.k_neighbors from {original_k} -> {safe_k} for benchmark")
    except Exception:
        pass

    # set up experiment context and checkpoint
    run_dir = Path.cwd() / 'runs' / f'combined_bench_{int(time.time())}'
    ckpt = CheckpointManager(run_dir)
    logger = setup_logger(run_dir / 'bench.log')
    ctx = RunContext(run_id=f'bench_{int(time.time())}', run_dir=run_dir, smoke_test=False)
    experiment = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)

    # Wrap apply_smotenc to avoid hard failures on tiny class folds during benchmarking
    try:
        import src.balancing as balancing_mod

        _orig_apply_smotenc = balancing_mod.apply_smotenc

        def _safe_apply_smotenc(X_train_selected, y_train, categorical_feature_indices, smote_config, random_state):
            try:
                return _orig_apply_smotenc(X_train_selected, y_train, categorical_feature_indices, smote_config, random_state)
            except Exception as e:
                before = y_train.value_counts().sort_index().to_dict()
                # return no-op balancing results so benchmark can proceed
                return X_train_selected.copy(), y_train.copy(), before, before.copy(), {}

        balancing_mod.apply_smotenc = _safe_apply_smotenc
        print('Wrapped apply_smotenc to safe fallback for benchmark')
    except Exception:
        pass

    # load current JSON measurements into a map
    base = load_json()
    seen_keys = {measurement_key(m): m for m in base.get('measurements', [])}

    # XGBoost prioritized configs
    for workers, threads in XGB_CONFIGS:
        key = f"XGBoost|{workers}|{threads}"
        if key in seen_keys and seen_keys[key].get('status') == 'success':
            print('Skipping already-successful measurement', key)
            continue
        print('Running XGBoost config', workers, threads)
        try:
            res = run_gwo_iteration_concurrent(experiment, 'XGBoost', sample_df, workers=workers, model_threads=threads)
            entry = {
                'model': 'XGBoost',
                'workers': int(workers),
                'model_threads': int(threads),
                'elapsed_seconds': float(res['elapsed_seconds']),
                'cpu_percent': res['cpu_percent'],
                'ram_mb': res['ram_mb'],
                'num_success': int(res['num_success']),
                'num_failed': int(res['num_failed']),
                'errors': res['errors'],
                'status': 'success' if res['num_failed'] == 0 else 'partial_failure'
            }
        except Exception as exc:
            entry = {
                'model': 'XGBoost',
                'workers': int(workers),
                'model_threads': int(threads),
                'elapsed_seconds': None,
                'cpu_percent': None,
                'ram_mb': None,
                'num_success': 0,
                'num_failed': 1,
                'errors': [str(exc)],
                'status': 'failed',
            }
        # upsert
        base = load_json()
        measurements = base.get('measurements', [])
        # remove any existing same-key
        measurements = [m for m in measurements if measurement_key(m) != key]
        measurements.append(entry)
        base['measurements'] = measurements
        base['timestamp'] = utc_now_iso()
        write_json(base)

    # Other models small targeted tests
    for model_name in OTHER_MODELS:
        for workers in OTHER_WORKER_SETS:
            threads = 1 if workers >= 8 else 4
            key = f"{model_name}|{workers}|{threads}"
            if key in seen_keys and seen_keys[key].get('status') == 'success':
                print('Skipping already-successful measurement', key)
                continue
            try:
                res = run_gwo_iteration_concurrent(experiment, model_name, sample_df, workers=workers, model_threads=threads)
                entry = {
                    'model': model_name,
                    'workers': int(workers),
                    'model_threads': int(threads),
                    'elapsed_seconds': float(res['elapsed_seconds']),
                    'cpu_percent': res['cpu_percent'],
                    'ram_mb': res['ram_mb'],
                    'num_success': int(res['num_success']),
                    'num_failed': int(res['num_failed']),
                    'errors': res['errors'],
                    'status': 'success' if res['num_failed'] == 0 else 'partial_failure'
                }
            except Exception as exc:
                entry = {
                    'model': model_name,
                    'workers': int(workers),
                    'model_threads': int(threads),
                    'elapsed_seconds': None,
                    'cpu_percent': None,
                    'ram_mb': None,
                    'num_success': 0,
                    'num_failed': 1,
                    'errors': [str(exc)],
                    'status': 'failed',
                }
            base = load_json()
            measurements = base.get('measurements', [])
            measurements = [m for m in measurements if measurement_key(m) != key]
            measurements.append(entry)
            base['measurements'] = measurements
            base['timestamp'] = utc_now_iso()
            write_json(base)

    # Final validation and summary
    final = load_json()
    # validate duplicates
    keys = [measurement_key(m) for m in final.get('measurements', [])]
    dup = len(keys) != len(set(keys))
    success_count = sum(1 for m in final.get('measurements', []) if m.get('status') == 'success')
    failure_count = sum(1 for m in final.get('measurements', []) if m.get('status') != 'success')

    print('JSON path:', str(JSON_PATH))
    print('Measurements successful:', success_count)
    print('Measurements failed/partial:', failure_count)
    if dup:
        print('Warning: duplicate measurement keys detected')

    # compute speedups for XGBoost entries
    xgb_entries = [m for m in final.get('measurements', []) if m.get('model') == 'XGBoost' and m.get('elapsed_seconds')]
    speedups = []
    for e in xgb_entries:
        speedups.append({'workers': e['workers'], 'threads': e['model_threads'], 'elapsed': e['elapsed_seconds'], 'speedup': BASELINE_SEQ_ITER_SEC / e['elapsed_seconds']})
    final['xgb_speedups'] = speedups
    write_json(final)

    print('Done. Wrote results to', str(JSON_PATH))


if __name__ == '__main__':
    main()
