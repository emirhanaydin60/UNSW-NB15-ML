from __future__ import annotations

from main import prepare_smoke_subsets


def test_prepare_smoke_subsets(synthetic_config, synthetic_dataset):
    train, test = prepare_smoke_subsets(synthetic_dataset, synthetic_config)
    expected_classes = synthetic_config["dataset"]["expected_classes"]
    assert set(train["attack_cat"].unique()) == set(expected_classes)
    assert set(test["attack_cat"].unique()) == set(expected_classes)
