from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_ORDER = ["DT", "RF", "SVM", "LR", "XGBoost"]


def model_requires_scaling(model_name: str) -> bool:
    return model_name in {"SVM", "LR"}


def build_model(model_name: str, params: dict[str, Any], random_state: int, num_classes: int):
    if model_name == "DT":
        return DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=random_state,
        )

    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "SVM":
        return SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            decision_function_shape="ovr",
        )

    if model_name == "LR":
        return LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=int(params["min_child_weight"]),
            gamma=float(params["gamma"]),
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    raise ValueError(f"Unsupported model name: {model_name}")
