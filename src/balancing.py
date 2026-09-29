from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC


def build_sampling_strategy(y: pd.Series, target_count: int) -> dict[str, int]:
    counts = y.value_counts()
    strategy: dict[str, int] = {}
    for cls, cnt in counts.items():
        if int(cnt) < target_count:
            strategy[str(cls)] = int(target_count)
    return strategy


def apply_smotenc(
    X_train_selected: pd.DataFrame,
    y_train: pd.Series,
    categorical_feature_indices: list[int],
    smote_config: dict[str, Any],
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int], dict[str, int], dict[str, int]]:
    before = y_train.value_counts().sort_index().to_dict()
    target_count = int(smote_config["target_count"])
    strategy = build_sampling_strategy(y_train, target_count=target_count)

    if not strategy:
        after = before.copy()
        return X_train_selected.copy(), y_train.copy(), before, after, strategy

    k_neighbors = int(smote_config["k_neighbors"])
    smote = SMOTENC(
        categorical_features=categorical_feature_indices,
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)

    # Keep ordinal categorical features integer-valued after synthetic generation.
    if categorical_feature_indices:
        X_resampled = np.asarray(X_resampled)
        X_resampled[:, categorical_feature_indices] = np.round(X_resampled[:, categorical_feature_indices])
        X_resampled = pd.DataFrame(X_resampled, columns=X_train_selected.columns)
    else:
        X_resampled = pd.DataFrame(X_resampled, columns=X_train_selected.columns)

    y_resampled = pd.Series(y_resampled, name=y_train.name)
    after = y_resampled.value_counts().sort_index().to_dict()
    return X_resampled, y_resampled, before, after, strategy


def prepare_smotenc_frame(
    X_selected: pd.DataFrame,
    X_reference: pd.DataFrame,
    selected_categorical_columns: list[str],
    all_categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """
    Build the dataframe used by SMOTENC.

    If selected features contain no categorical columns, add temporary auxiliary
    categorical columns from the fold-encoded reference dataframe. These
    auxiliary columns are dropped after resampling before model fitting.
    """
    smote_df = X_selected.copy()
    cat_cols = [c for c in selected_categorical_columns if c in smote_df.columns]

    auxiliary_cols: list[str] = []
    if not cat_cols:
        for col in all_categorical_columns:
            if col in X_reference.columns and col not in smote_df.columns:
                smote_df[col] = X_reference[col].values
                auxiliary_cols.append(col)
        cat_cols = [c for c in auxiliary_cols if c in smote_df.columns]

    cat_indices = [smote_df.columns.get_loc(c) for c in cat_cols]
    return smote_df, cat_indices, auxiliary_cols
