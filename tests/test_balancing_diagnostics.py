from __future__ import annotations

import json
import logging

import pandas as pd

from src.checkpoint import CheckpointManager
from src.nested_cv import NestedCVExperiment, RunContext


def test_balancing_diagnostics_written_on_exception(tmp_path, synthetic_config):
    # Prepare a tiny dataset where one class has only 3 examples
    cfg = dict(synthetic_config)
    cfg["smotenc"]["k_neighbors"] = 4
    cfg["smotenc"]["target_count"] = 10

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("test-balancing-diag")
    exp = NestedCVExperiment(config=cfg, run_context=RunContext(run_id="unit", run_dir=run_dir, smoke_test=False), logger=logger, checkpoint_manager=CheckpointManager(run_dir))

    # Build X_selected with one categorical column and two numeric columns
    X = pd.DataFrame(
        {
            "proto": ["tcp", "tcp", "tcp", "udp", "udp", "udp", "icmp", "icmp", "icmp", "tcp"],
            "f1": [0.1 * i for i in range(10)],
            "f2": [1.0 * i for i in range(10)],
        }
    )
    # y: class 'Rare' only 3 samples, others 7 samples
    y = pd.Series(["Rare", "Rare", "Rare", "Common", "Common", "Common", "Common", "Common", "Common", "Common"], name="attack_cat")

    # categorical_indices points to 'proto' at index 0
    cat_idx = [0]

    # Call balancing expecting SMOTENC to raise (k_neighbors=4 > n_samples_fit for 'Rare')
    diag_file = run_dir / "metadata" / "diagnostics" / "balancing_events.jsonl"
    try:
        exp._apply_balancing_with_policy(
            X_selected=X,
            y_selected=y,
            categorical_indices=cat_idx,
            stage_random_state=42,
            log_context={"model": "UNIT", "outer_fold": 1, "inner_fold": 1, "selected_features": list(X.columns), "selected_categorical_columns": ["proto"]},
        )
        raised = False
    except Exception:
        raised = True

    assert raised, "Expected SMOTENC to raise for too-few-samples"
    # Diagnostics file must exist and contain a JSON entry with exception
    assert diag_file.exists(), "Diagnostics file not written"
    with diag_file.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry.get("exception") is not None
    assert entry.get("balancing_method") in ("smotenc", "smote")
