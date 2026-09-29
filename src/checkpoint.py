from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.utils import atomic_write_json, ensure_dir


class CheckpointManager:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = ensure_dir(self.run_dir / "checkpoints")
        self.gwo_dir = ensure_dir(self.checkpoints_dir / "gwo")
        self.models_dir = ensure_dir(self.checkpoints_dir / "models")
        self.state_file = self.checkpoints_dir / "experiment_state.json"

    def save_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_file, state)

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_file.exists():
            return None
        with self.state_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def validate_state(self) -> bool:
        try:
            _ = self.load_state()
            return True
        except Exception:
            return False

    def save_gwo_state(self, key: str, state: dict[str, Any]) -> Path:
        path = self.gwo_dir / f"{key}.json"
        atomic_write_json(path, state)
        return path

    def load_gwo_state(self, key: str) -> dict[str, Any] | None:
        path = self.gwo_dir / f"{key}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_model_artifact(self, model_name: str, artifact_name: str, obj: Any) -> Path:
        model_dir = ensure_dir(self.models_dir / model_name)
        out = model_dir / artifact_name
        tmp = out.with_suffix(out.suffix + ".tmp")
        joblib.dump(obj, tmp)
        tmp.replace(out)
        return out

    def clear_state(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
