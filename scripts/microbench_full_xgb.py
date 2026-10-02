from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import yaml

# ensure project src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_training_dataset
from scripts.unsw_valid_large_sample_bench import run_gwo_one_iteration_concurrent
from src.checkpoint import CheckpointManager
from src.logger import setup_logger
from src.nested_cv import RunContext, NestedCVExperiment


def main():
    cfg_path = Path.cwd() / "config" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path.cwd() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # load full training dataset (no sampling)
    train_df = load_training_dataset(config)
    n_rows = len(train_df)
    target_col = config["dataset"]["target_column"]

    # compute predictor count before feature selection
    predictors = [c for c in train_df.columns if c != target_col and c not in config.get("dataset", {}).get("excluded_columns", [])]
    num_predictors = len(predictors)

    # print effective configuration and checks
    print(f"training rows = {n_rows}")
    print(f"target = {target_col}")
    print(f"predictor_features_before_fs = {num_predictors}")
    print(f"outer_folds = {config['cv']['outer_folds']}")
    print(f"inner_folds = {config['cv']['inner_folds']}")
    print(f"population = 10")
    print(f"iterations = 1")
    print(f"workers = 4")
    print(f"threads = 4")
    print(f"k_neighbors = {config['smotenc']['k_neighbors']}")
    print(f"target_count = {config['smotenc']['target_count']}")

    if n_rows != 175341:
        print("Warning: training rows != 175341; aborting to avoid unintended sample changes.")
        print(f"Found {n_rows} rows")
        return

    # prepare run context
    run_dir = Path.cwd() / "runs" / f"microbench_xgb_full_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(run_dir)
    logger = setup_logger(run_dir / "bench.log")
    ctx = RunContext(run_id=f"microbench_full_{int(time.time())}", run_dir=run_dir, smoke_test=False)
    experiment = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)

    # run the one-iteration GWO with XGBoost, workers=4, threads=4
    t0 = time.perf_counter()
    res = run_gwo_one_iteration_concurrent(experiment, "XGBoost", train_df, workers=4, model_threads=4)
    elapsed = time.perf_counter() - t0

    diag = run_dir / "metadata" / "diagnostics" / "balancing_events.jsonl"
    eff = run_dir / "metadata" / "effective_run_metadata.json"

    # collect balancing diagnostic errors if any
    balancing_errors = []
    if diag.exists():
        try:
            with diag.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("exception"):
                        balancing_errors.append(obj)
        except Exception:
            pass

    # reporting
    print("\nRun result summary:")
    print(f"total_elapsed_seconds: {elapsed:.6f}")
    print(f"successful_wolves: {res.get('num_success', 0)}")
    print(f"failed_wolves: {res.get('num_failed', 0)}")
    print(f"smotenc_diagnostic_errors: {len(balancing_errors)}")
    print(f"effective_metadata: {eff}")
    print(f"diagnostic_jsonl: {diag}")
    print("full_training_class_distribution:")
    print(json.dumps(train_df[target_col].astype(str).value_counts().to_dict(), indent=2))

    # if any balancing error, print last one and stop
    if balancing_errors:
        print("\nSMOTENC failure detected — reporting last diagnostic entry:")
        print(json.dumps(balancing_errors[-1], indent=2))
        return

    # compute empirical ratios against earlier measured times
    ref_25k = 597.5007705
    ref_50k = 719.2256153
    ref_75k = 746.8376649
    ref_100k = 800.9573896

    print("\nEmpirical ratios:")
    print(f"full_time / 25k_time = {elapsed / ref_25k:.6f}")
    print(f"full_time / 50k_time = {elapsed / ref_50k:.6f}")
    print(f"full_time / 75k_time = {elapsed / ref_75k:.6f}")
    print(f"full_time / 100k_time = {elapsed / ref_100k:.6f}")


if __name__ == "__main__":
    main()
