from __future__ import annotations

ALGORITHMS = [
    {
        "id": "sdpo",
        "name": "SDPO",
        "description": "Feedback-conditioned on-policy self-distillation with a frozen self-teacher.",
        "script": "SDPO/modal_verl_sdpo.py",
        "models": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-35B-A3B"],
        "defaults": {"dataset": "gsm8k_sdpo", "steps": 20, "train_rows": 64, "val_rows": 16},
    },
    {
        "id": "opd",
        "name": "OPD",
        "description": "On-policy distillation from a separate teacher using Verl's k1 objective.",
        "script": "OPD/modal_verl_qwen35_opd.py",
        "models": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"],
        "defaults": {"dataset": "gsm8k", "steps": 20, "train_rows": 64, "val_rows": 16},
    },
    {
        "id": "harvey-sdpo",
        "name": "Harvey LAB SDPO",
        "description": "Rubric-feedback self-distillation for long-form legal agent trajectories.",
        "script": "SDPO/modal_verl_sdpo.py",
        "models": ["Qwen/Qwen3.5-35B-A3B"],
        "defaults": {"dataset": "harvey_lab_sdpo", "steps": 10, "train_rows": 8, "val_rows": 2},
    },
    {
        "id": "kimi-sdpo",
        "name": "Kimi K2.6 SDPO",
        "description": "Multi-GPU SDPO topologies for Kimi K2.6 full-parameter or LoRA training.",
        "script": "SDPO/modal_verl_kimi_k26_sdpo.py",
        "models": ["moonshotai/Kimi-K2.5"],
        "defaults": {"dataset": "kimi_sdpo", "steps": 10, "train_rows": 32, "val_rows": 8},
    },
]

ALGORITHM_BY_ID = {item["id"]: item for item in ALGORITHMS}


def catalog() -> dict:
    return {
        "algorithms": ALGORITHMS,
        "hint_placements": [
            {"id": "before-mistake", "name": "Before the mistake"},
            {"id": "after-trace", "name": "After the full trace"},
            {"id": "system", "name": "Teacher system prompt"},
        ],
        "topologies": ["single-gpu", "h200-lora", "h200-full", "b200-full"],
    }
