from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_balanced_accuracy_macro_f1(y_true, y_pred) -> tuple[float, float]:
    ba = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(ba), float(macro_f1)


def compute_metrics_dict(y_true, y_pred) -> dict[str, float]:
    ba, macro_f1 = compute_balanced_accuracy_macro_f1(y_true, y_pred)
    return {
        "balanced_accuracy": ba,
        "macro_f1": macro_f1,
        "fitness": (ba + macro_f1) / 2.0,
    }


def confusion_matrix_df(y_true, y_pred, class_order: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    return pd.DataFrame(cm, index=class_order, columns=class_order)


def classification_report_dict(y_true, y_pred, class_order: list[str]) -> dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        labels=class_order,
        target_names=class_order,
        output_dict=True,
        zero_division=0,
    )
