from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def holm_correction(p_values: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    significant = [False] * m

    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * p
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)

    for i, p_adj in enumerate(adjusted):
        significant[i] = p_adj < alpha
    return adjusted, significant


def run_statistical_analysis(
    outer_fold_metrics: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_folds = {1, 2, 3, 4, 5}

    if outer_fold_metrics.empty:
        raise ValueError("Statistical analysis requires results for exactly five paired outer folds and five models")

    if outer_fold_metrics.duplicated(subset=["model", "outer_fold"]).any():
        raise ValueError("Statistical analysis requires exactly one row per model and outer fold")

    observed_folds = set(map(int, outer_fold_metrics["outer_fold"].unique()))
    if observed_folds != required_folds:
        raise ValueError(f"Statistical analysis requires outer folds {sorted(required_folds)}; got {sorted(observed_folds)}")

    model_names = sorted(map(str, outer_fold_metrics["model"].unique()))
    if len(model_names) != 5:
        raise ValueError(f"Statistical analysis requires results for exactly five models; got {len(model_names)}")

    for model_name in model_names:
        model_folds = set(map(int, outer_fold_metrics.loc[outer_fold_metrics["model"] == model_name, "outer_fold"].tolist()))
        if model_folds != required_folds:
            raise ValueError(f"Model {model_name} must have exactly one observation for folds 1-5; got {sorted(model_folds)}")

    friedman_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    pairwise_columns = [
        "model_a",
        "model_b",
        "metric",
        "raw_p_value",
        "holm_adjusted_p_value",
        "significant",
    ]

    for metric in ["balanced_accuracy", "macro_f1"]:
        pivot = outer_fold_metrics.pivot_table(
            index="outer_fold",
            columns="model",
            values=metric,
            aggfunc="mean",
        )
        models = list(pivot.columns)
        if len(models) < 2 or pivot.shape[0] < 2:
            friedman_rows.append(
                {
                    "metric": metric,
                    "friedman_statistic": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                }
            )
            continue

        try:
            stat, p_value = friedmanchisquare(*[pivot[m].values for m in models])
        except Exception:
            stat, p_value = float("nan"), 1.0
        is_sig = p_value < alpha
        friedman_rows.append(
            {
                "metric": metric,
                "friedman_statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(is_sig),
            }
        )

        if is_sig:
            raw_ps: list[float] = []
            meta: list[tuple[str, str]] = []
            for m1, m2 in combinations(models, 2):
                try:
                    _, w_p = wilcoxon(pivot[m1].values, pivot[m2].values, zero_method="wilcox")
                except Exception:
                    w_p = 1.0
                raw_ps.append(float(w_p))
                meta.append((m1, m2))

            adj_ps, sig_flags = holm_correction(raw_ps, alpha=alpha)
            for (m1, m2), raw_p, adj_p, sig in zip(meta, raw_ps, adj_ps, sig_flags):
                pairwise_rows.append(
                    {
                        "model_a": m1,
                        "model_b": m2,
                        "metric": metric,
                        "raw_p_value": raw_p,
                        "holm_adjusted_p_value": adj_p,
                        "significant": bool(sig),
                    }
                )

    return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows, columns=pairwise_columns)


def summarize_cv_performance(outer_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = outer_fold_metrics.groupby("model", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    return grouped
