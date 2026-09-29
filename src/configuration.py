from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.utils import atomic_write_text, ensure_dir

REQUIRED_TOP_LEVEL_KEYS = [
    "experiment",
    "dataset",
    "cv",
    "feature_selection",
    "smotenc",
    "models",
    "gwo",
    "logging",
    "reports",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            raise ValueError(f"Missing required top-level config key: {key}")

    dataset = config["dataset"]
    for key in ["train_path", "test_path", "target_column", "categorical_columns", "excluded_columns"]:
        if key not in dataset:
            raise ValueError(f"Missing dataset config key: {key}")

    cv = config["cv"]
    if int(cv["outer_folds"]) < 2 or int(cv["inner_folds"]) < 2:
        raise ValueError("Both outer_folds and inner_folds must be >= 2")

    fs = config["feature_selection"]
    if int(fs["top_k"]) <= 0:
        raise ValueError("feature_selection.top_k must be positive")

    gwo = config["gwo"]
    if int(gwo["population_size"]) <= 0 or int(gwo["iterations"]) <= 0:
        raise ValueError("gwo.population_size and gwo.iterations must be positive")


def apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    smoke = out.get("smoke_test", {})
    out["cv"]["outer_folds"] = int(smoke.get("outer_folds", 2))
    out["cv"]["inner_folds"] = int(smoke.get("inner_folds", 2))
    out["gwo"]["population_size"] = int(smoke.get("gwo_population_size", 3))
    out["gwo"]["iterations"] = int(smoke.get("gwo_iterations", 3))
    if "smotenc_target_count" in smoke:
        out["smotenc"]["target_count"] = int(smoke["smotenc_target_count"])
    if "smotenc_k_neighbors" in smoke:
        out["smotenc"]["k_neighbors"] = int(smoke["smotenc_k_neighbors"])
    return out


def save_effective_config(config: dict[str, Any], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    payload = yaml.safe_dump(config, sort_keys=False)
    atomic_write_text(output_path, payload)
