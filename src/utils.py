from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *[str(p) for p in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def atomic_write_json(path: str | Path, data: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    payload = json.dumps(_to_serializable(data), indent=2, sort_keys=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path_obj.parent),
        delete=False,
        prefix=path_obj.name,
        suffix=".tmp",
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    tmp_path.replace(path_obj)


def atomic_write_text(path: str | Path, content: str) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path_obj.parent),
        delete=False,
        prefix=path_obj.name,
        suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    tmp_path.replace(path_obj)


def append_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_exists = path_obj.exists()

    with path_obj.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not path_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_serializable(row.get(k)) for k in fieldnames})


def upsert_csv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
    key_fields: Sequence[str],
) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    new_df = pd.DataFrame(list(rows), columns=list(fieldnames))
    if path_obj.exists():
        existing_df = pd.read_csv(path_obj)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    if key_fields:
        combined = combined.drop_duplicates(subset=list(key_fields), keep="last")

    combined = combined.reindex(columns=list(fieldnames))

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path_obj.parent),
        delete=False,
        prefix=path_obj.name,
        suffix=".tmp",
    ) as tmp:
        combined.to_csv(tmp, index=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    tmp_path.replace(path_obj)


def get_system_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def json_hash(data: Any) -> str:
    encoded = json.dumps(_to_serializable(data), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
