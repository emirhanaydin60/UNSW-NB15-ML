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


def run_for_size(n: int, config: dict, timestamp_tag: str) -> dict:
    train_df = load_training_dataset(config)
    if len(train_df) < n:
        raise RuntimeError(f"Training dataset has only {len(train_df)} rows, requested {n}")

    sample_df = stratified_sample(train_df, n, config["dataset"]["target_column"])  # deterministic sampling seed=42
    viable, details = check_sample_viability(sample_df, config)

    run_dir = Path.cwd() / "runs" / f"microbench_xgb_{n}rows_4w4t_{timestamp_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(run_dir)
    logger = setup_logger(run_dir / "bench.log")
    ctx = RunContext(run_id=f"microbench_{n}_{timestamp_tag}", run_dir=run_dir, smoke_test=False)
    experiment = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)

    result = {
        "sample_size": n,
        "class_counts": details.get("class_counts"),
        "viable": viable,
        "elapsed_seconds": None,
        "num_success": 0,
        "num_failed": 0,
        "errors": None,
        "balancing_errors": [],
        "diag_path": str(run_dir / "metadata" / "diagnostics" / "balancing_events.jsonl"),
        "effective_metadata": str(run_dir / "metadata" / "effective_run_metadata.json"),
    }

    if not viable:
        return result

    t0 = time.perf_counter()
    res = run_gwo_one_iteration_concurrent(experiment, "XGBoost", sample_df, workers=4, model_threads=4)
    elapsed = time.perf_counter() - t0

    result["elapsed_seconds"] = float(elapsed)
    result["num_success"] = int(res.get("num_success", 0))
    result["num_failed"] = int(res.get("num_failed", 0))
    result["errors"] = res.get("errors", [])

    # read balancing diagnostics for exceptions
    diag = Path(result["diag_path"])
    if diag.exists():
        try:
            with diag.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("exception"):
                        result["balancing_errors"].append(obj)
        except Exception:
            pass

    return result


def main():
    cfg_path = Path.cwd() / "config" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path.cwd() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sizes = [50000, 75000, 100000]
    timestamp_tag = str(int(time.time()))
    outputs = []

    for n in sizes:
        print(f"Running sample size {n} with workers=4 threads=4 ...")
        out = run_for_size(n, config, timestamp_tag)
        outputs.append(out)

        # if SMOTENC failure occurred, stop further runs
        if out.get("balancing_errors"):
            print("SMOTENC failure detected for size", n)
            print(json.dumps(out["balancing_errors"][-1], indent=2))
            break

    # print results and table
    baseline = 597.5007705
    print("\nResults:")
    for o in outputs:
        rel = None
        if o.get("elapsed_seconds"):
            rel = o["elapsed_seconds"] / baseline
        print(json.dumps({
            "sample_size": o["sample_size"],
            "class_counts": o["class_counts"],
            "viable": o["viable"],
            "elapsed_seconds": o["elapsed_seconds"],
            "relative_to_25k": rel,
            "num_success": o["num_success"],
            "num_failed": o["num_failed"],
            "balancing_errors_count": len(o.get("balancing_errors", [])),
            "diag_path": o["diag_path"],
            "effective_metadata": o["effective_metadata"],
        }, indent=2))

    # Print summary table header
    print("\nSample Size\tTime(s)\tRelative to 25k\tSuccessful Wolves\tSMOTENC Errors")
    for o in outputs:
        t = o.get("elapsed_seconds")
        rel = (t / baseline) if t else None
        print(f"{o['sample_size']}\t{t if t else 'N/A'}\t{rel:.4f if rel else 'N/A'}\t{o.get('num_success')}\t{len(o.get('balancing_errors', []))}")


if __name__ == "__main__":
    main()
