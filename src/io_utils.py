from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def create_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"feynman_ballistic_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"feynman_ballistic_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **to_builtin(payload),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
