from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import compute_metrics_dict
from src.statistics import run_statistical_analysis, summarize_cv_performance


def test_fitness_calculation():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    out = compute_metrics_dict(y_true, y_pred)
    assert "balanced_accuracy" in out
    assert "macro_f1" in out
    assert out["fitness"] == (out["balanced_accuracy"] + out["macro_f1"]) / 2.0


def test_statistical_analysis_and_summary():
    rows = []
    for fold in range(1, 6):
        for model, base in [("DT", 0.60), ("RF", 0.70), ("SVM", 0.68), ("LR", 0.66), ("XGBoost", 0.72)]:
            rows.append(
                {
                    "outer_fold": fold,
                    "model": model,
                    "balanced_accuracy": base + 0.01 * fold,
                    "macro_f1": base - 0.02 + 0.01 * fold,
                }
            )
    df = pd.DataFrame(rows)

    friedman_df, pairwise_df = run_statistical_analysis(df)
    summary = summarize_cv_performance(df)

    assert not friedman_df.empty
    assert {"metric", "friedman_statistic", "p_value", "significant"}.issubset(friedman_df.columns)
    assert not summary.empty
    assert {"model", "balanced_accuracy_mean", "macro_f1_mean"}.issubset(summary.columns)

    # Pairwise may be empty if Friedman not significant in synthetic data; type check only.
    assert isinstance(pairwise_df, pd.DataFrame)


def test_statistical_analysis_rejects_invalid_fold_structure():
    rows = []
    for fold in range(1, 6):
        for model in ["DT", "RF", "SVM", "LR", "XGBoost"]:
            rows.append(
                {
                    "outer_fold": fold,
                    "model": model,
                    "balanced_accuracy": 0.5 + 0.01 * fold,
                    "macro_f1": 0.4 + 0.01 * fold,
                }
            )
    bad_df = pd.DataFrame(rows[:-1])

    with pytest.raises(ValueError, match="exactly five models|exactly one observation|folds 1-5"):
        run_statistical_analysis(bad_df)
