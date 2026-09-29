from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass
class FoldPreprocessor:
    categorical_columns: list[str]

    def __post_init__(self) -> None:
        self.feature_columns_: list[str] | None = None
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    def fit(self, X_train: pd.DataFrame) -> "FoldPreprocessor":
        self.feature_columns_ = list(X_train.columns)
        self.encoder_.fit(X_train[self.categorical_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        X_aligned = X[self.feature_columns_].copy()
        X_aligned[self.categorical_columns] = self.encoder_.transform(X_aligned[self.categorical_columns])
        return X_aligned


@dataclass
class ModelTransformer:
    categorical_columns: list[str]
    scale: bool

    def __post_init__(self) -> None:
        self.column_transformer_: ColumnTransformer | None = None
        self.scaler_: StandardScaler | None = None

    def fit_transform(self, X_train: pd.DataFrame) -> sparse.spmatrix | np.ndarray:
        cat_cols = [c for c in self.categorical_columns if c in X_train.columns]
        num_cols = [c for c in X_train.columns if c not in cat_cols]

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    cat_cols,
                ),
                ("num", "passthrough", num_cols),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

        Xt = self.column_transformer_.fit_transform(X_train)

        if self.scale:
            self.scaler_ = StandardScaler(with_mean=False)
            Xt = self.scaler_.fit_transform(Xt)
        return Xt

    def transform(self, X: pd.DataFrame) -> sparse.spmatrix | np.ndarray:
        if self.column_transformer_ is None:
            raise RuntimeError("ModelTransformer must be fitted before transform")
        Xt = self.column_transformer_.transform(X)
        if self.scale and self.scaler_ is not None:
            Xt = self.scaler_.transform(Xt)
        return Xt


def get_selected_categorical_columns(
    selected_features: Sequence[str],
    original_categorical: Sequence[str],
) -> list[str]:
    selected = set(selected_features)
    return [c for c in original_categorical if c in selected]


def get_categorical_indices(feature_names: Sequence[str], categorical_columns: Sequence[str]) -> list[int]:
    cat_set = set(categorical_columns)
    return [i for i, f in enumerate(feature_names) if f in cat_set]
