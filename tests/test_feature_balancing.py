from __future__ import annotations

from src.balancing import apply_smotenc, build_sampling_strategy
from src.feature_selection import select_top_features
from src.preprocessing import FoldPreprocessor, get_categorical_indices, get_selected_categorical_columns


def test_feature_selection_top_k(synthetic_config, synthetic_dataset):
    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"])
    y = synthetic_dataset["attack_cat"]

    pre = FoldPreprocessor(categorical_columns=synthetic_config["dataset"]["categorical_columns"])
    pre.fit(X)
    X_enc = pre.transform(X)

    selected, rank_df = select_top_features(X_enc, y, synthetic_config, random_state=42)
    assert len(selected) == synthetic_config["feature_selection"]["top_k"]
    assert len(rank_df) == X_enc.shape[1]


def test_sampling_strategy_and_smotenc(synthetic_config, synthetic_dataset):
    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"]).copy()
    y = synthetic_dataset["attack_cat"].copy()

    pre = FoldPreprocessor(categorical_columns=synthetic_config["dataset"]["categorical_columns"])
    pre.fit(X)
    X_enc = pre.transform(X)

    selected = list(X_enc.columns[:8])
    X_sel = X_enc[selected]
    selected_cat = get_selected_categorical_columns(selected, synthetic_config["dataset"]["categorical_columns"])
    cat_idx = get_categorical_indices(X_sel.columns, selected_cat)

    strategy = build_sampling_strategy(y, target_count=20)
    assert all(v == 20 for v in strategy.values())

    X_res, y_res, before, after, _ = apply_smotenc(
        X_sel,
        y,
        categorical_feature_indices=cat_idx,
        smote_config={"k_neighbors": 1, "target_count": 20},
        random_state=42,
    )
    assert len(X_res) == len(y_res)
    assert len(y_res) > len(y)
    assert set(before).issubset(set(after))
