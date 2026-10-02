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
from scripts.unsw_valid_large_sample_bench import stratified_sample, check_sample_viability, run_gwo_one_iteration_concurrent
from src.nested_cv import NestedCVExperiment, RunContext
from src.checkpoint import CheckpointManager
from src.logger import setup_logger


def run_config(workers: int, threads: int, sample_df, config: dict, timestamp_tag: str) -> dict:
    run_dir = Path.cwd() / "runs" / f"microbench_xgb_{workers}w_{threads}t_{timestamp_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(run_dir)
    logger = setup_logger(run_dir / "bench.log")
    ctx = RunContext(run_id=f"microbench_{workers}w_{threads}t_{timestamp_tag}", run_dir=run_dir, smoke_test=False)
    experiment = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)

    t0 = time.perf_counter()
    res = run_gwo_one_iteration_concurrent(experiment, "XGBoost", sample_df, workers=workers, model_threads=threads)
    elapsed = time.perf_counter() - t0

    # locate diagnostics
    diag = run_dir / "metadata" / "diagnostics" / "balancing_events.jsonl"
    eff = run_dir / "metadata" / "effective_run_metadata.json"

    # read diagnostics to count balancing errors
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

    return {
        "workers": workers,
        "threads": threads,
        "total_elapsed_seconds": float(elapsed),
        "num_success": int(res.get("num_success", 0)),
        "num_failed": int(res.get("num_failed", 0)),
        "errors": res.get("errors", []),
        "balancing_errors": balancing_errors,
        "diag_path": str(diag),
        "effective_metadata": str(eff),
        "raw_result": res,
    }


def main():
    cfg_path = Path.cwd() / "config" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path.cwd() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_df = load_training_dataset(config)
    sample_df = stratified_sample(train_df, 25000, config["dataset"]["target_column"])  # deterministic seed=42 inside

    viable, details = check_sample_viability(sample_df, config)
    if not viable:
        print("Sample not viable according to pre-flight check; aborting.")
        print(json.dumps(details, indent=2))
        return

    configs = [(4, 4), (6, 2), (8, 1), (10, 1)]
    timestamp_tag = str(int(time.time()))
    results = []

    for workers, threads in configs:
        print(f"Running workers={workers} threads={threads} ...")
        out = run_config(workers, threads, sample_df, config, timestamp_tag)
        results.append(out)

        # if any balancing exception occurred, stop this configuration series
        if out["balancing_errors"]:
            print("SMOTENC failure detected. Stopping further runs.")
            print("Diagnostic entry:")
            print(json.dumps(out["balancing_errors"][-1], indent=2))
            break

    # print table and JSON summary
    print("\nRun summary:")
    for r in results:
        print(json.dumps({
            "workers": r["workers"],
            "threads": r["threads"],
            "total_elapsed_seconds": r["total_elapsed_seconds"],
            "num_success": r["num_success"],
            "num_failed": r["num_failed"],
            "diag_path": r["diag_path"],
            "effective_metadata": r["effective_metadata"],
            "balancing_errors_count": len(r["balancing_errors"]),
        }, indent=2))

    # compute speedups only for completed runs (exclude interrupted due to failures)
    baseline = 811.9216
    print("\nPerformance table:")
    print("Workers\tThreads\tTime(s)\tSpeedup\tRelativeRuntime\tSuccessfulWolves")
    for r in results:
        t = r["total_elapsed_seconds"]
        speedup = baseline / t if t > 0 else None
        rel = t / baseline if t > 0 else None
        print(f"{r['workers']}\t{r['threads']}\t{t:.3f}\t{speedup:.4f}\t{rel:.4f}\t{r['num_success']}")


if __name__ == "__main__":
    main()
