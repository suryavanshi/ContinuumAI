from __future__ import annotations

from datetime import datetime, timezone


def _tokens() -> list[dict]:
    words = (
        "I should first retrieve the user's reservations with get_user_details, then ask them to choose "
        "the reservation they want to cancel before calling cancel_reservation."
    ).split(" ")
    deltas = [-0.08, 0.04, 0.13, 0.24, 0.35, -0.18, -0.31, 0.08, 0.17, 0.42, 0.29,
              -0.07, 0.11, 0.26, -0.14, 0.06, 0.33, 0.19, -0.05, 0.21, 0.09, 0.28]
    return [
        {
            "text": word + (" " if index < len(words) - 1 else ""),
            "delta": deltas[index % len(deltas)],
            "student_logprob": round(-1.7 + deltas[index % len(deltas)], 3),
            "teacher_logprob": round(-1.7 + 2 * deltas[index % len(deltas)], 3),
            "selected": abs(deltas[index % len(deltas)]) >= 0.25,
        }
        for index, word in enumerate(words)
    ]


def sample_traces() -> list[dict]:
    base_prompt = "Cancel all upcoming flights for user amelia_davis_8890. The user does not know their reservation IDs."
    hint = "Use get_user_details with the user_id to locate reservation IDs before requesting reservation details or cancellation."
    variants = [
        ("airline-0182", "The tool requires reservation_id; I will call get_reservation_details with user_id.", 0.0, 0.31),
        ("airline-0239", "I need to list the user's reservations before cancelling any booking.", 1.0, 0.18),
        ("airline-0304", "I'll ask the user for a reservation ID even though a lookup tool is available.", 0.0, 0.27),
        ("airline-0411", "First call get_user_details, then confirm each eligible cancellation.", 1.0, 0.12),
        ("airline-0527", "Proceed directly to cancel_reservation using the supplied user id.", 0.0, 0.38),
    ]
    return [
        {
            "id": trace_id,
            "prompt": base_prompt,
            "teacher_prompt": base_prompt + "\n\nPrivileged feedback: " + hint,
            "hint": hint,
            "response": response,
            "reward": reward,
            "kl": kl,
            "judge_selected": sum(1 for token in _tokens() if token["selected"]),
            "tokens": _tokens(),
        }
        for trace_id, response, reward, kl in variants
    ]


def sample_run() -> dict:
    losses = [0.318, 0.282, 0.251, 0.227, 0.198, 0.173, 0.151, 0.137, 0.122, 0.114]
    rewards = [0.31, 0.34, 0.39, 0.44, 0.52, 0.58, 0.63, 0.69, 0.73, 0.78]
    return {
        "id": "run_sdpo_airline_001",
        "name": "airline_tool_recovery",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "algorithm": "sdpo",
            "model": "Qwen/Qwen3.5-0.8B",
            "teacher_model": "Qwen/Qwen3.5-0.8B",
            "dataset": "airline_production_traces",
            "steps": 10,
            "train_rows": 64,
            "val_rows": 16,
            "hint_placement": "before-mistake",
            "hint": "Use get_user_details before reservation-specific tools.",
            "topology": "single-gpu",
        },
        "command": ["python3", "-m", "modal", "run", "--detach", "SDPO/modal_verl_sdpo.py"],
        "metrics": [
            {"step": index + 1, "distillation_loss": loss, "reward": rewards[index], "teacher_kl": round(loss * 0.71, 3)}
            for index, loss in enumerate(losses)
        ],
        "traces": sample_traces(),
        "logs": [
            "Prepared 64 feedback-conditioned trajectories",
            "Loaded frozen self-teacher Qwen/Qwen3.5-0.8B",
            "step:10 - distillation/loss:0.114 - critic/rewards/mean:0.78 - training/global_step:10",
            "Saving checkpoint to /cache/checkpoints/airline_tool_recovery/global_step_10",
        ],
    }
