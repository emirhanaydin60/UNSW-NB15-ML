"""Fold-replicating viability checker for SMOTENC(k_neighbors=5).

This script deterministically reproduces the production nested-CV
outer->inner splitting and the fold-local preprocessing + feature-selection
decisions used by `NestedCVExperiment._build_fitness_fn` to determine
whether any inner-train fold would cause imblearn.SMOTENC to fail
because a targeted class has too few examples (must be > k_neighbors).

Usage (example):
  python scripts/fold_replicating_viability_check.py --sample-size 25000

Options:
  --sample-size N    Run the checker on a single sample size (default: required)
  --all-sizes        If set, run the sequence 25000,50000,75000,100000
  --self-test        Run a small synthetic unit test and exit

Important: this script does NOT launch any benchmarks or GWO runs.
It reads production config and data-loading logic and attempts to
mirror the pipeline exactly; any non-exact matches are reported as WARNINGs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_training_dataset, split_xy
from src.preprocessing import FoldPreprocessor, get_selected_categorical_columns, get_categorical_indices
from src.feature_selection import select_top_features
from src.utils import derive_seed

SAMPLE_OPTIONS = [25000, 50000, 75000, 100000]


def stratified_sample(train_df: pd.DataFrame, n: int, target_col: str) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedShuffleSplit

    y = train_df[target_col].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=int(derive_seed(42, "sampling")))
    for train_idx, _ in sss.split(train_df, y):
        return train_df.iloc[train_idx].reset_index(drop=True)
    return train_df.sample(n=min(n, len(train_df)), random_state=42).reset_index(drop=True)


def check_sample_fold_replicating(sample_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run the fold-replicating viability check on a single sampled dataframe.

    Returns a detailed report dict and overall pass boolean under report['overall_pass'].
    """
    dataset_cfg = config["dataset"]
    target_col = dataset_cfg["target_column"]
    base_seed = int(config["experiment"]["base_seed"])
    outer_folds = int(config["cv"]["outer_folds"])
    inner_folds = int(config["cv"]["inner_folds"])
    k_neighbors = int(config["smotenc"]["k_neighbors"])
    target_count = int(config["smotenc"]["target_count"])

    X_all, y_all = split_xy(sample_df, config)

    from sklearn.model_selection import StratifiedKFold

    skf_outer = StratifiedKFold(n_splits=outer_folds, shuffle=bool(config["cv"].get("shuffle", True)), random_state=base_seed)

    report: Dict[str, Any] = {
        "sample_rows": len(sample_df),
        "sample_class_counts": sample_df[target_col].astype(str).value_counts().to_dict(),
        "details": {},
        "warnings": [],
        "any_targeted_classes": False,
        # legacy name removed from top-level summary; use explicit failure flag
        "failure_detected_in_targeted_class": False,
        # aggregated neighbor-safety across all folds
        "imblearn_boundary_pass": True,
        "strict_safety_pass": True,
        "target_count": target_count,
    }

    overall_pass = True
    min_overall = None
    limiting_info = None
    any_targeted = False
    min_targeted_count_all = None
    overall_imblearn_pass = True
    overall_strict_pass = True

    enabled_models = list(config["models"]["enabled"])

    for outer_idx, (outer_tr_idx, outer_val_idx) in enumerate(skf_outer.split(X_all, y_all), start=1):
        X_outer_train_raw = sample_df.iloc[outer_tr_idx].reset_index(drop=True)
        y_outer_train = X_outer_train_raw[dataset_cfg["target_column"]].astype(str)

        outer_key = f"outer_fold_{outer_idx}"
        report["details"][outer_key] = {}

        # For each model (production creates a per-model inner seed)
        for model_name in enabled_models:
            inner_seed = derive_seed(base_seed, model_name, outer_idx, "inner_cv")
            skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=bool(config["cv"].get("shuffle", True)), random_state=int(inner_seed))

            report["details"][outer_key][model_name] = {"inner_folds": {}}

            for inner_idx, (itr, ival) in enumerate(skf_inner.split(X_outer_train_raw.drop(columns=[dataset_cfg["target_column"]]), y_outer_train), start=1):
                X_inner_train_raw = X_outer_train_raw.iloc[itr].reset_index(drop=True)
                y_inner_train = X_inner_train_raw[dataset_cfg["target_column"]].astype(str)

                # Preprocessing: fit FoldPreprocessor on inner-train and transform
                preproc = FoldPreprocessor(categorical_columns=dataset_cfg["categorical_columns"])
                preproc.fit(X_inner_train_raw.drop(columns=[dataset_cfg["target_column"]]))
                X_inner_train_encoded = preproc.transform(X_inner_train_raw.drop(columns=[dataset_cfg["target_column"]]))

                # Feature selection: use same seed as production (42)
                try:
                    selected_features, rank_df = select_top_features(X_inner_train_encoded, y_inner_train, config=config, random_state=42)
                except Exception as exc:  # pragma: no cover - defensive
                    report["warnings"].append(f"feature_selection_failed_outer{outer_idx}_model{model_name}_inner{inner_idx}: {exc}")
                    selected_features = list(X_inner_train_encoded.columns)[: int(config["feature_selection"]["top_k"])]

                selected_categorical_columns = get_selected_categorical_columns(selected_features, dataset_cfg["categorical_columns"]) if selected_features else []
                cat_indices = get_categorical_indices(selected_features, selected_categorical_columns)

                counts = y_inner_train.value_counts().to_dict()
                targeted_map = {str(cls): int(cnt) for cls, cnt in counts.items() if int(cnt) < target_count}

                inner_key = f"inner_fold_{inner_idx}"
                fold_info: Dict[str, Any] = {
                    "class_counts": counts,
                    "target_count": target_count,
                    "targeted_classes": targeted_map,
                    "min_targeted_class_count": min(targeted_map.values()) if targeted_map else None,
                    "rarest_targeted_class": min(targeted_map, key=targeted_map.get) if targeted_map else None,
                    "categorical_selected": bool(selected_categorical_columns),
                    "categorical_columns": selected_categorical_columns,
                    "categorical_indices": cat_indices,
                    "smote_variant": "SMOTENC" if bool(cat_indices) else "SMOTE",
                    "checks": [],
                }

                # For each targeted class report both imblearn boundary and strict safety
                fold_pass_strict = True
                fold_pass_imblearn = True
                if targeted_map:
                    any_targeted = True
                    # update global min for any targeted class counts
                    m = int(min(targeted_map.values()))
                    if min_targeted_count_all is None or m < min_targeted_count_all:
                        min_targeted_count_all = m

                for cls, cnt in targeted_map.items():
                    imblearn_ok = int(cnt) >= k_neighbors
                    strict_ok = int(cnt) > k_neighbors
                    fold_info["checks"].append(
                        {
                            "class": cls,
                            "count_in_inner_train": int(cnt),
                            "required_min_imblearn": int(k_neighbors),
                            "required_min_strict": int(k_neighbors) + 1,
                            "imblearn_boundary_check": bool(imblearn_ok),
                            "strict_safety_check": bool(strict_ok),
                        }
                    )
                    if not strict_ok:
                        fold_pass_strict = False
                        overall_pass = False
                        if min_overall is None or int(cnt) < min_overall:
                            min_overall = int(cnt)
                            limiting_info = {
                                "limiting_class": cls,
                                "limiting_outer_fold": outer_idx,
                                "limiting_inner_fold": inner_idx,
                                "model": model_name,
                                "count": int(cnt),
                            }
                    if not imblearn_ok:
                        fold_pass_imblearn = False
                    # aggregate
                    overall_imblearn_pass = overall_imblearn_pass and bool(imblearn_ok)
                    overall_strict_pass = overall_strict_pass and bool(strict_ok)

                fold_info["strict_safety_pass"] = fold_pass_strict
                fold_info["imblearn_boundary_pass"] = fold_pass_imblearn
                fold_info["pass"] = fold_pass_strict
                report["details"][outer_key][model_name]["inner_folds"][inner_key] = fold_info

    report["overall_pass"] = overall_pass
    report["any_targeted_classes"] = any_targeted
    report["failure_detected_in_targeted_class"] = limiting_info is not None
    report["minimum_targeted_class_count"] = min_targeted_count_all
    if limiting_info is not None:
        report.update(
            {
                "minimum_inner_train_count_across_all_folds": min_overall,
                "limiting_class": limiting_info["limiting_class"],
                "limiting_outer_fold": limiting_info["limiting_outer_fold"],
                "limiting_inner_fold": limiting_info["limiting_inner_fold"],
                "limiting_model": limiting_info.get("model"),
            }
        )
    else:
        report.update(
            {
                "minimum_inner_train_count_across_all_folds": None,
                "limiting_class": None,
                "limiting_outer_fold": None,
                "limiting_inner_fold": None,
                "limiting_model": None,
            }
        )

    # aggregated neighbor-safety results
    report["imblearn_boundary_pass"] = overall_imblearn_pass
    report["strict_safety_pass"] = overall_strict_pass

    return report


def pretty_print_report(report: dict[str, Any]) -> None:
    print(f"SAMPLE_ROWS: {report['sample_rows']}")
    print("STATUS:", "PASS" if report["overall_pass"] else "FAIL")
    print()

    for outer_key, outer_val in report.get("details", {}).items():
        print(outer_key)
        for model_name, mval in outer_val.items():
            print(f"  model: {model_name}")
            for inner_key, fold in mval.get("inner_folds", {}).items():
                print(f"    {inner_key}")
                print(f"      class_counts: {fold['class_counts']}")
                print(f"      target_count: {fold.get('target_count')}")
                print(f"      targeted_classes: {fold.get('targeted_classes')}")
                print(f"      min_targeted_class_count: {fold.get('min_targeted_class_count')}")
                print(f"      rarest_targeted_class: {fold.get('rarest_targeted_class')}")
                print(f"      smote_variant: {fold['smote_variant']}")
                for chk in fold.get("checks", []):
                    imblearn_status = "OK" if chk.get("imblearn_boundary_check") else "FAIL"
                    strict_status = "OK" if chk.get("strict_safety_check") else "FAIL"
                    print(f"        - class={chk['class']} count={chk['count_in_inner_train']} required_min_imblearn={chk['required_min_imblearn']} imblearn_check={imblearn_status} required_min_strict={chk['required_min_strict']} strict_check={strict_status}")
                print(f"      PASS (strict): {fold['pass']}")
                print(f"      imblearn_boundary_pass: {fold.get('imblearn_boundary_pass')}")
                print(f"      strict_safety_pass: {fold.get('strict_safety_pass')}")
            print()

    print("SUMMARY:")
    print("  sample_size:", report.get("sample_rows"))
    print("  any_targeted_classes:", report.get("any_targeted_classes"))
    print("  failure_detected_in_targeted_class:", report.get("failure_detected_in_targeted_class"))
    print("  minimum_targeted_class_count:", report.get("minimum_targeted_class_count"))
    print("  minimum_inner_train_count_across_all_folds:", report.get("minimum_inner_train_count_across_all_folds"))
    print("  limiting_class:", report.get("limiting_class"))
    print("  limiting_outer_fold:", report.get("limiting_outer_fold"))
    print("  limiting_inner_fold:", report.get("limiting_inner_fold"))
    print("  imblearn_boundary_pass:", report.get("imblearn_boundary_pass"))
    print("  strict_safety_pass:", report.get("strict_safety_pass"))
    if report.get("warnings"):
        print("\nWARNINGS:")
        for w in report.get("warnings", []):
            print(" ", w)


def run_self_test() -> int:
    """Run a tiny deterministic self-test to validate logic (no dataset file needed)."""
    # Build a tiny synthetic dataset designed to FAIL: class 'rare' will be too small
    rows = []
    for i in range(100):
        rows.append({"f1": i, "proto": "tcp", "service": "s", "state": "S0", "attack_cat": "Normal"})
    for i in range(5):
        rows.append({"f1": i + 1000, "proto": "udp", "service": "s", "state": "S0", "attack_cat": "Worms"})
    df = pd.DataFrame(rows)

    # Minimal config baseline from example
    import yaml

    cfg_path = Path.cwd() / "config" / "config.example.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Use smaller folds for self-test to be quick
    config["cv"]["outer_folds"] = 2
    config["cv"]["inner_folds"] = 2
    config["smotenc"]["target_count"] = 10

    report = check_sample_fold_replicating(df, config)
    pretty_print_report(report)
    # Expect a FAIL because Worms count small
    return 0 if not report["overall_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, help="Sample size to test (e.g. 25000)")
    parser.add_argument("--all-sizes", action="store_true", help="Run all predefined sample sizes in order")
    parser.add_argument("--self-test", action="store_true", help="Run internal self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.sample_size and not args.all_sizes:
        print("Error: --sample-size required unless --all-sizes is used", file=sys.stderr)
        return 2

    import yaml

    cfg_path = Path.cwd() / "config" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path.cwd() / "config" / "config.example.yaml"

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_df = load_training_dataset(config)

    sizes = SAMPLE_OPTIONS if args.all_sizes else [args.sample_size]
    for n in sizes:
        if n > len(train_df):
            print(f"Requested sample {n} larger than available training rows {len(train_df)}; skipping")
            continue
        sample_df = stratified_sample(train_df, n, config["dataset"]["target_column"])
        print(f"\n=== Running fold-replicating check for sample_size={n} ===\n")
        report = check_sample_fold_replicating(sample_df, config)
        pretty_print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
