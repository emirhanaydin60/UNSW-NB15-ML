from __future__ import annotations

from src.preprocessing import FoldPreprocessor, ModelTransformer


def test_ordinal_encoding_unknown_value(synthetic_dataset):
    train = synthetic_dataset.iloc[:50].copy()
    test = synthetic_dataset.iloc[50:60].copy()
    test.loc[:, "proto"] = "new_proto"

    pre = FoldPreprocessor(categorical_columns=["proto", "service", "state"])
    pre.fit(train.drop(columns=["attack_cat", "id", "label"]))

    Xt = pre.transform(test.drop(columns=["attack_cat", "id", "label"]))
    assert (Xt["proto"] == -1).any()


def test_model_transformer_shapes(synthetic_dataset):
    X = synthetic_dataset.drop(columns=["attack_cat", "id", "label"]).copy()
    train = X.iloc[:80]
    val = X.iloc[80:100]

    pre = FoldPreprocessor(categorical_columns=["proto", "service", "state"])
    pre.fit(train)
    train_enc = pre.transform(train)
    val_enc = pre.transform(val)

    tr = ModelTransformer(categorical_columns=["proto", "service", "state"], scale=True)
    Xtr = tr.fit_transform(train_enc)
    Xval = tr.transform(val_enc)

    assert Xtr.shape[0] == len(train)
    assert Xval.shape[0] == len(val)
