from __future__ import annotations
import sys
import time
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError

sys.path.insert(0, str(Path.cwd()))

from src.configuration import load_config
from src.data_loader import load_training_dataset
from src.nested_cv import NestedCVExperiment, RunContext
from src.checkpoint import CheckpointManager
from src.gwo import GreyWolfOptimizer, decode_position
from src.utils import utc_now_iso, ensure_dir, derive_seed


class SimpleLogger:
    def info(self, *a, **k):
        print("INFO:", *a)

    def warning(self, *a, **k):
        print("WARN:", *a)

    def exception(self, *a, **k):
        print("EXC:", *a)


def optionally_set_system_env_threads(n_threads: int | None):
    if n_threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)


def run_single_candidate(timeout_seconds: int = 1800):
    cfg = load_config(Path.cwd() / "config.yaml")
    cfg = dict(cfg)
    cfg.setdefault("models", {})
    cfg["models"]["enabled"] = ["SVM"]
    cfg.setdefault("gwo", {})["population_size"] = 5
    cfg.setdefault("gwo", {})["iterations"] = 1
    cfg.setdefault("experiment", {})["workers"] = 1
    cfg.setdefault("experiment", {})["threads"] = 1

    run_id = f"single_candidate_svm_diag_{int(time.time())}"
    run_dir = Path.cwd() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(run_id=run_id, run_dir=run_dir, smoke_test=False)
    logger = SimpleLogger()
    exp = NestedCVExperiment(config=cfg, run_context=ctx, logger=logger, checkpoint_manager=CheckpointManager(run_dir))

    train_df = load_training_dataset(cfg)
    target_col = cfg["dataset"]["target_column"]

    # Outer fold 1 as diagnostic
    from sklearn.model_selection import StratifiedKFold
    outer_folds = int(cfg["cv"]["outer_folds"]) if cfg.get("cv") else 5
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=bool(cfg.get("cv", {}).get("shuffle", True)), random_state=derive_seed(int(cfg["experiment"]["base_seed"]), "outer_cv"))

    tr_idx, val_idx = next(iter(skf.split(train_df, train_df[target_col].astype(str))))
    X_outer_train_raw = train_df.iloc[tr_idx].reset_index(drop=True).drop(columns=[target_col])
    y_outer_train = train_df[target_col].iloc[tr_idx].astype(str).reset_index(drop=True)

    fitness_fn = exp._build_fitness_fn(model_name="SVM", model_idx=0, outer_fold=1, X_outer_train_raw=X_outer_train_raw, y_outer_train=y_outer_train)

    optimizer = GreyWolfOptimizer(model_name="SVM", search_space_cfg=cfg["models"]["search_spaces"]["SVM"], population_size=5, iterations=1, seed=derive_seed(int(cfg["experiment"]["base_seed"]), "gwo", "SVM", 1))
    init = optimizer._initial_state()
    positions = init["positions"]
    # pick first wolf position
    pos = positions[0]
    params = decode_position(pos, optimizer.dimensions)

    optionally_set_system_env_threads(1)

    print("Starting single candidate evaluation at", utc_now_iso())
    print("Params:", params)

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fitness_fn, params, 0, 0)
        try:
            res = fut.result(timeout=timeout_seconds)
            print("Candidate completed; result summary keys:", list(res.keys()))
        except TimeoutError:
            print("TIMEOUT: candidate did not finish within", timeout_seconds, "seconds")
            return {"status": "timeout", "timeout_seconds": timeout_seconds}
        except Exception as exc:
            print("EXCEPTION during candidate:", exc)
            return {"status": "exception", "error": str(exc)}

    # collect runtime event(s)
    runtime_events_file = Path(exp.paths["metadata"]) / "runtime_events.jsonl"
    events = []
    if runtime_events_file.exists():
        with runtime_events_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    out = {
        "status": "completed",
        "result": res,
        "runtime_events_count": len(events),
        "runtime_events": events,
    }
    # write summary
    summary_path = run_dir / "single_candidate_summary.json"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Done at", utc_now_iso())
    return out


if __name__ == "__main__":
    out = run_single_candidate(timeout_seconds=1800)
    print(json.dumps({"status": out.get("status"), "runtime_events_count": out.get("runtime_events_count")}, indent=2))
