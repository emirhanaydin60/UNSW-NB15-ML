from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def select_top_features(
    X_train_encoded: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    fs_cfg = config["feature_selection"]
    model = RandomForestClassifier(
        n_estimators=int(fs_cfg["n_estimators"]),
        criterion=str(fs_cfg["criterion"]),
        random_state=random_state,
        n_jobs=int(fs_cfg.get("n_jobs", -1)),
    )
    model.fit(X_train_encoded, y_train)

    importances = pd.Series(model.feature_importances_, index=X_train_encoded.columns)
    rank_df = importances.sort_values(ascending=False).reset_index().rename(columns={"index": "feature", 0: "importance"})
    rank_df["rank"] = range(1, len(rank_df) + 1)
    k = int(fs_cfg["top_k"])
    selected = rank_df.head(k)["feature"].tolist()
    rank_df["selected"] = rank_df["feature"].isin(selected)
    return selected, rank_df
