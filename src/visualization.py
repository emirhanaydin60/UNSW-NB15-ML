from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import ensure_dir


def _save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    ensure_dir(out_base.parent)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def configure_style(style: str = "whitegrid") -> None:
    sns.set_theme(style=style, context="paper")


def plot_class_distribution(class_distribution_df: pd.DataFrame, out_dir: str | Path, dpi: int = 300) -> None:
    before = class_distribution_df[class_distribution_df["stage"] == "before"]
    after = class_distribution_df[class_distribution_df["stage"] == "after"]

    merged = before.merge(
        after,
        on=["phase", "model", "outer_fold", "class"],
        suffixes=("_before", "_after"),
        how="outer",
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    melted = merged.melt(
        id_vars=["class"],
        value_vars=["count_before", "count_after"],
        var_name="distribution",
        value_name="count",
    )
    sns.barplot(data=melted, x="class", y="count", hue="distribution", ax=ax)
    ax.set_title("Class Distribution Before vs After SMOTENC")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    _save_figure(fig, Path(out_dir) / "class_distribution_before_after_smotenc", dpi)


def plot_gwo_convergence(gwo_convergence_df: pd.DataFrame, out_dir: str | Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=gwo_convergence_df,
        x="iteration",
        y="best_fitness",
        hue="model",
        style="phase",
        markers=False,
        dashes=True,
        ax=ax,
    )
    ax.set_title("GWO Convergence Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    _save_figure(fig, Path(out_dir) / "gwo_convergence_curves", dpi)


def plot_performance_bars(
    outer_metrics_df: pd.DataFrame,
    metric: str,
    out_dir: str | Path,
    dpi: int = 300,
) -> None:
    summary = outer_metrics_df.groupby("model", as_index=False)[metric].agg(["mean", "std"]).reset_index().rename(columns={"mean": "metric_mean", "std": "metric_std"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["model"], summary["metric_mean"], yerr=summary["metric_std"], capsize=4)
    ax.set_title(f"Model Comparison: {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    _save_figure(fig, Path(out_dir) / f"model_comparison_{metric}", dpi)


def plot_confusion_matrices(
    confusion_dir: str | Path,
    out_dir: str | Path,
    dpi: int = 300,
) -> None:
    confusion_path = Path(confusion_dir)
    for csv_file in sorted(confusion_path.glob("*.csv")):
        cm = pd.read_csv(csv_file, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix: {csv_file.stem}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        _save_figure(fig, Path(out_dir) / f"confusion_matrix_{csv_file.stem}", dpi)


def generate_all_figures(run_dir: str | Path, config: dict[str, Any]) -> None:
    run_path = Path(run_dir)
    results_dir = run_path / "results"
    figures_dir = run_path / "figures"

    configure_style(config.get("reports", {}).get("style", "whitegrid"))
    dpi = int(config.get("reports", {}).get("dpi", 300))

    class_dist_file = results_dir / "class_distributions.csv"
    if class_dist_file.exists():
        plot_class_distribution(pd.read_csv(class_dist_file), figures_dir, dpi=dpi)

    gwo_conv_file = results_dir / "gwo_convergence.csv"
    if gwo_conv_file.exists():
        plot_gwo_convergence(pd.read_csv(gwo_conv_file), figures_dir, dpi=dpi)

    outer_metrics_file = results_dir / "outer_fold_metrics.csv"
    if outer_metrics_file.exists():
        outer = pd.read_csv(outer_metrics_file)
        plot_performance_bars(outer, "balanced_accuracy", figures_dir, dpi=dpi)
        plot_performance_bars(outer, "macro_f1", figures_dir, dpi=dpi)

    plot_confusion_matrices(results_dir / "confusion_matrices", figures_dir, dpi=dpi)
