from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.checkpoint import CheckpointManager
from src.configuration import apply_smoke_overrides, load_config, save_effective_config
from src.data_loader import load_datasets
from src.logger import setup_logger
from src.nested_cv import NestedCVExperiment, RunContext
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UNSW-NB15 multiclass intrusion benchmark with nested CV + GWO")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--run-id", default=None, help="Existing run ID for resume/status/report")
    parser.add_argument("--smoke-test", action="store_true", help="Run lightweight smoke experiment")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="Start a new experiment run")
    action.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    action.add_argument("--restart", action="store_true", help="Start a new run as restart")
    action.add_argument("--status", action="store_true", help="Show run status")
    action.add_argument("--generate-reports", action="store_true", help="Generate reports from saved results")

    return parser.parse_args()


def make_run_id(smoke: bool) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_smoke" if smoke else ts


def list_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir()])


def resolve_existing_run_dir(runs_root: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    runs = list_runs(runs_root)
    if not runs:
        raise FileNotFoundError("No runs found to resume/status/report")
    return runs[-1]


def prepare_smoke_subsets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = config["dataset"]["target_column"]
    expected = config["dataset"]["expected_classes"]
    smoke_cfg = config["smoke_test"]

    train_n = int(smoke_cfg.get("rows_per_class_train", 60))
    test_n = int(smoke_cfg.get("rows_per_class_test", 30))
    seed = int(config["experiment"]["base_seed"])

    train_parts = []
    test_parts = []
    for cls in expected:
        cls_train = train_df[train_df[target].astype(str) == cls]
        cls_test = test_df[test_df[target].astype(str) == cls]
        if cls_train.empty or cls_test.empty:
            raise ValueError(f"Smoke subset cannot be built; class missing: {cls}")
        train_parts.append(cls_train.sample(n=min(train_n, len(cls_train)), random_state=seed))
        test_parts.append(cls_test.sample(n=min(test_n, len(cls_test)), random_state=seed))

    train_out = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_out = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_out, test_out


def build_experiment(config: dict, run_dir: Path, smoke_test: bool):
    log_file = run_dir / "logs" / "run.log"
    logger = setup_logger(log_file, level=str(config["logging"].get("level", "INFO")))
    run_id = run_dir.name
    ckpt = CheckpointManager(run_dir)
    ctx = RunContext(run_id=run_id, run_dir=run_dir, smoke_test=smoke_test)
    exp = NestedCVExperiment(config=config, run_context=ctx, logger=logger, checkpoint_manager=ckpt)
    return exp, logger


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.smoke_test:
        cfg = apply_smoke_overrides(cfg)

    runs_root = ensure_dir(Path(cfg["experiment"]["runs_root"]))

    if args.run:
        run_id = make_run_id(smoke=args.smoke_test)
        run_dir = ensure_dir(runs_root / run_id)
    elif args.restart:
        run_id = make_run_id(smoke=args.smoke_test)
        run_dir = ensure_dir(runs_root / f"{run_id}_restart")
    else:
        run_dir = resolve_existing_run_dir(runs_root, args.run_id)

    saved_cfg_path = run_dir / "config" / "config_used.yaml"
    if saved_cfg_path.exists() and (args.resume or args.status or args.generate_reports):
        cfg = load_config(saved_cfg_path)

    inferred_smoke = bool(args.smoke_test or run_dir.name.endswith("_smoke"))
    exp, logger = build_experiment(cfg, run_dir, smoke_test=inferred_smoke)

    cfg_out = run_dir / "config" / "config_used.yaml"
    if not cfg_out.exists() or args.run or args.restart:
        save_effective_config(cfg, cfg_out)

    if args.status:
        print(json.dumps(exp.status(), indent=2))
        return

    if args.generate_reports:
        exp.run_reports()
        print(f"Reports generated in: {run_dir}")
        return

    train_df, test_df, hashes = load_datasets(cfg, logger=logger)
    exp.write_metadata(hashes)

    if args.smoke_test:
        train_df, test_df = prepare_smoke_subsets(train_df, test_df, cfg)

    if args.resume:
        if not exp.ckpt.validate_state():
            raise RuntimeError("Checkpoint integrity check failed")

    exp.execute_full_pipeline(train_df=train_df)
    print(f"Experiment finished successfully: {run_dir}")


if __name__ == "__main__":
    main()
