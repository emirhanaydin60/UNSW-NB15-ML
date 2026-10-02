from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
import sys
from pathlib import Path

# ensure project src is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_training_dataset
from scripts.unsw_valid_large_sample_bench import stratified_sample, check_sample_viability, run_gwo_one_iteration_concurrent
from src.nested_cv import NestedCVExperiment, RunContext
from src.checkpoint import CheckpointManager
from src.logger import setup_logger
from src.utils import utc_now_iso


def main():
    root = Path.cwd()
    cfg_path = root / "config" / "config.yaml"
    if not cfg_path.exists():
        # fall back to project root config.yaml
        cfg_path = root / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # verify effective parameters
    base_seed = int(config.get("experiment", {}).get("base_seed", 42))
    k_neighbors = int(config["smotenc"]["k_neighbors"])
    target_count = int(config["smotenc"]["target_count"])
    outer_folds = int(config["cv"]["outer_folds"])
    inner_folds = int(config["cv"]["inner_folds"])
    gwo_population = 10
    gwo_iterations = 1

    print("Effective parameters:")
    print(f" sample_size=25000")
    print(f" sampling_seed=42")
    print(f" base_seed={base_seed}")
    print(f" config_path={cfg_path}")
    print(f" k_neighbors={k_neighbors}")
    print(f" target_count={target_count}")
    print(f" outer_folds={outer_folds}")
    print(f" inner_folds={inner_folds}")
    print(f" gwo_population={gwo_population}")
    print(f" gwo_iterations={gwo_iterations}")
    print(f" model=XGBoost")
    print(f" workers=1")
    print(f" model_threads=1")

    train_df = load_training_dataset(config)
    sample_df = stratified_sample(train_df, 25000, config["dataset"]["target_column"])  # deterministic seed=42 inside

    viable, details = check_sample_viability(sample_df, config)
    print("Sample viability:", viable)
    print(json.dumps(details, indent=2))

    run_dir = Path.cwd() / "runs" / f"microbench_xgb_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(run_dir)
    logger = setup_logger(run_dir / "bench.log")
    ctx = RunContext(run_id=f"microbench_{int(time.time())}", run_dir=run_dir, smoke_test=False)
    experiment = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)

    t0 = time.perf_counter()
    res = run_gwo_one_iteration_concurrent(experiment, "XGBoost", sample_df, workers=1, model_threads=1)
    elapsed = time.perf_counter() - t0

    print("GWO one-iteration result:")
    print(json.dumps(res, indent=2))
    print(f"total_elapsed_seconds: {elapsed:.3f}")

    # show diagnostic paths
    diag = run_dir / "metadata" / "diagnostics" / "balancing_events.jsonl"
    eff = run_dir / "metadata" / "effective_run_metadata.json"
    print("diagnostic_jsonl=", str(diag))
    print("effective_metadata=", str(eff))


if __name__ == "__main__":
    main()
