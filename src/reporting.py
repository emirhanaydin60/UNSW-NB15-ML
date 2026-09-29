from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.statistics import run_statistical_analysis, summarize_cv_performance
from src.utils import atomic_write_json, ensure_dir
from src.visualization import generate_all_figures


def generate_tables(run_dir: str | Path, alpha: float = 0.05) -> None:
    run_path = Path(run_dir)
    results = run_path / "results"
    tables = ensure_dir(run_path / "tables")

    outer_metrics_file = results / "outer_fold_metrics.csv"
    if outer_metrics_file.exists():
        outer = pd.read_csv(outer_metrics_file)
        summary = summarize_cv_performance(outer)
        summary.to_csv(tables / "table_a_cross_validation_performance.csv", index=False)

        friedman_df, pairwise_df = run_statistical_analysis(outer, alpha=alpha)
        stat_rows = []
        for _, row in friedman_df.iterrows():
            stat_rows.append(
                {
                    "test_type": "friedman",
                    "metric": row["metric"],
                    "model_a": "",
                    "model_b": "",
                    "friedman_statistic": row["friedman_statistic"],
                    "raw_p_value": row["p_value"],
                    "holm_adjusted_p_value": "",
                    "significant": row["significant"],
                }
            )
        for _, row in pairwise_df.iterrows():
            stat_rows.append(
                {
                    "test_type": "wilcoxon_holm",
                    "metric": row["metric"],
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "friedman_statistic": "",
                    "raw_p_value": row["raw_p_value"],
                    "holm_adjusted_p_value": row["holm_adjusted_p_value"],
                    "significant": row["significant"],
                }
            )
        pd.DataFrame(stat_rows).to_csv(results / "statistical_results.csv", index=False)
        friedman_df.to_csv(tables / "table_d_friedman_results.csv", index=False)
        pairwise_df.to_csv(tables / "table_e_wilcoxon_pairwise.csv", index=False)

    final_hparams_file = results / "final_hyperparameters.csv"
    if final_hparams_file.exists():
        hparams = pd.read_csv(final_hparams_file)
        long_rows: list[dict[str, Any]] = []
        for _, row in hparams.iterrows():
            params = json.loads(row["hyperparameters_json"])
            for k, v in params.items():
                long_rows.append(
                    {
                        "model": row["model"],
                        "hyperparameter": k,
                        "selected_value": v,
                    }
                )
        pd.DataFrame(long_rows).to_csv(tables / "table_b_final_gwo_hyperparameters.csv", index=False)

    selected_features_file = results / "selected_features.csv"
    if selected_features_file.exists():
        selected = pd.read_csv(selected_features_file)
        final_fs = selected[selected["phase"] == "final_train"].copy()
        if not final_fs.empty:
            final_fs = final_fs.sort_values("rank")
            final_fs[["rank", "feature", "importance"]].to_csv(tables / "table_c_selected_features.csv", index=False)

    final_test_file = results / "final_test_metrics.csv"
    if final_test_file.exists():
        final_test = pd.read_csv(final_test_file)
        final_test[["model", "balanced_accuracy", "macro_f1"]].to_csv(tables / "table_f_final_official_test_performance.csv", index=False)


def generate_reports(run_dir: str | Path, config: dict[str, Any]) -> None:
    generate_tables(run_dir)
    generate_all_figures(run_dir, config)

    metadata_file = Path(run_dir) / "metadata" / "report_metadata.json"
    atomic_write_json(
        metadata_file,
        {
            "generated": True,
            "run_dir": str(run_dir),
        },
    )
