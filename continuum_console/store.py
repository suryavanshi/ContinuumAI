from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .commands import build_command
from .catalog import ALGORITHM_BY_ID
from .sample_data import sample_run, sample_traces


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"runs": [sample_run()]})

    def _read(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._read()["runs"], key=lambda run: run["created_at"], reverse=True)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return next((run for run in self.list_runs() if run["id"] == run_id), None)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = validate_config(payload)
        now = datetime.now(timezone.utc).isoformat()
        run = {
            "id": "run_" + uuid.uuid4().hex[:12],
            "name": payload.get("name") or f"{config['algorithm']}_{config['dataset']}",
            "status": "draft",
            "created_at": now,
            "updated_at": now,
            "config": config,
            "command": build_command(config),
            "metrics": [],
            "traces": [],
            "logs": ["Run configured. Launch is gated until CONTINUUM_ENABLE_LAUNCH=1."],
        }
        with self._lock:
            data = self._read()
            data["runs"].append(run)
            self._write(data)
        return run

    def update(self, run_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            for run in data["runs"]:
                if run["id"] == run_id:
                    run.update(changes)
                    run["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._write(data)
                    return run
        return None


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    algorithm = str(payload.get("algorithm", "sdpo"))
    if algorithm not in {"sdpo", "opd", "harvey-sdpo", "kimi-sdpo"}:
        raise ValueError("algorithm must be sdpo, opd, harvey-sdpo, or kimi-sdpo")
    model = str(payload.get("model", "")).strip()
    dataset = str(payload.get("dataset", "")).strip()
    if not model or not dataset:
        raise ValueError("model and dataset are required")
    allowed_models = {item for algorithm_config in ALGORITHM_BY_ID.values() for item in algorithm_config["models"]}
    if model not in allowed_models:
        raise ValueError("model is not in the approved training catalog")
    steps = int(payload.get("steps", 10))
    max_steps = int(os.environ.get("CONTINUUM_MAX_STEPS", "100"))
    if not 1 <= steps <= max_steps:
        raise ValueError(f"steps must be between 1 and {max_steps}")
    train_rows = int(payload.get("train_rows", 64))
    val_rows = int(payload.get("val_rows", 16))
    if not 1 <= train_rows <= int(os.environ.get("CONTINUUM_MAX_TRAIN_ROWS", "2048")):
        raise ValueError("train_rows exceeds the configured safety limit")
    if not 1 <= val_rows <= int(os.environ.get("CONTINUUM_MAX_VAL_ROWS", "512")):
        raise ValueError("val_rows exceeds the configured safety limit")
    return {
        "algorithm": algorithm,
        "model": model,
        "teacher_model": str(payload.get("teacher_model", model)),
        "dataset": dataset,
        "steps": steps,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "hint_placement": str(payload.get("hint_placement", "before-mistake")),
        "hint": str(payload.get("hint", "")),
        "topology": str(payload.get("topology", "single-gpu")),
        "smoke_gpu": bool(payload.get("smoke_gpu", False)),
    }
