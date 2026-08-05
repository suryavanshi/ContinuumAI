from __future__ import annotations

import os
from typing import Any

from .catalog import ALGORITHM_BY_ID


def _option(name: str, value: Any) -> list[str]:
    if value is None or value == "":
        return []
    return [f"--{name.replace('_', '-')}", str(value)]


def build_command(config: dict[str, Any]) -> list[str]:
    """Build an argv-safe Modal command using the repository's launchers."""
    algorithm = str(config.get("algorithm", "sdpo"))
    if algorithm not in ALGORITHM_BY_ID:
        raise ValueError(f"unknown algorithm: {algorithm}")

    script = ALGORITHM_BY_ID[algorithm]["script"]
    if algorithm == "sdpo":
        script += "::main"
    elif algorithm == "harvey-sdpo":
        script += "::harvey_main"
    command = [os.environ.get("CONTINUUM_MODAL_BIN", "modal"), "run", "--detach", script]
    if algorithm == "harvey-sdpo":
        command += _option("train_tasks", config.get("train_rows", 8))
        command += _option("eval_preview_tasks", config.get("val_rows", 2))
    else:
        command += _option("dataset", config.get("dataset"))
        command += _option("train_rows", config.get("train_rows"))
        command += _option("val_rows", config.get("val_rows"))
        command += _option("total_training_steps", config.get("steps"))
    command += _option("model", config.get("model"))

    if algorithm == "opd":
        command += _option("teacher_model", config.get("teacher_model"))
    elif algorithm == "kimi-sdpo":
        command += _option("mode", config.get("topology", "h200-lora"))
    elif algorithm == "sdpo":
        command += _option("static_feedback", config.get("hint"))
        if config.get("smoke_gpu"):
            command.append("--smoke-gpu")
    return command
