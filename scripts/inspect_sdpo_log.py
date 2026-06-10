from __future__ import annotations

import argparse
import re
from pathlib import Path


FAILURE_PATTERNS = (
    "Traceback (most recent call last)",
    "subprocess.CalledProcessError",
    "RuntimeError:",
    "ValueError:",
    "IndexError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "CUDA out of memory",
    "CUDA error",
)

COMPLETION_PATTERNS = (
    "training/global_step",
    "train/critic/rewards/mean",
    "critic/rewards/mean",
    "distillation/loss",
    "actor/pg_loss",
    "Saving checkpoint",
    "local_global_step_folder",
)

RESUME_PATTERNS = (
    "Found checkpoint",
    "Load from checkpoint",
    "Resuming from",
    "Setting global step",
)


def _last_matching_line(lines: list[str], patterns: tuple[str, ...]) -> str | None:
    for line in reversed(lines):
        if any(pattern in line for pattern in patterns):
            return line.strip()
    return None


def _last_matching_index(lines: list[str], patterns: tuple[str, ...]) -> int | None:
    for idx in range(len(lines) - 1, -1, -1):
        if any(pattern in lines[idx] for pattern in patterns):
            return idx
    return None


def _last_fresh_step_line(lines: list[str], expect_steps: int) -> str | None:
    step_patterns = (
        f"training/global_step:{expect_steps}",
        f"training/global_step: {expect_steps}",
        f"step:{expect_steps} -",
        f"step: {expect_steps} -",
        f"global_step_{expect_steps}",
        f"'global_step': {expect_steps}",
        f'"global_step": {expect_steps}',
    )
    for line in reversed(lines):
        if any(resume_pattern in line for resume_pattern in RESUME_PATTERNS):
            continue
        if any(step_pattern in line for step_pattern in step_patterns):
            return line.strip()
    return None


def inspect_log(path: Path, expect_steps: int) -> str:
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    configured_steps = None
    for match in re.finditer(r"Total training steps:\s*(\d+)", text):
        configured_steps = int(match.group(1))

    failure_line = _last_matching_line(lines, FAILURE_PATTERNS)
    failure_idx = _last_matching_index(lines, FAILURE_PATTERNS)
    completion_line = _last_matching_line(lines, COMPLETION_PATTERNS)
    hard_step_line = None
    hard_step_idx = None
    if expect_steps > 0:
        hard_step_line = _last_fresh_step_line(lines, expect_steps)
        hard_step_idx = _last_matching_index(lines, (hard_step_line,)) if hard_step_line else None

    if failure_line and hard_step_line and hard_step_idx is not None and failure_idx is not None and hard_step_idx > failure_idx:
        status = "completed-with-warning"
    elif failure_line:
        status = "failed"
    elif hard_step_line:
        status = "completed"
    elif completion_line:
        status = "progress-evidence-seen"
    else:
        status = "incomplete"

    parts = [
        f"status={status}",
        f"log={path}",
        f"lines={len(lines)}",
        f"configured_steps={configured_steps}",
        f"expected_steps={expect_steps}",
    ]
    if failure_line:
        parts.append(f"failure_line={failure_line}")
    if hard_step_line:
        parts.append(f"step_evidence={hard_step_line}")
    elif completion_line:
        parts.append(f"metric_or_checkpoint_evidence={completion_line}")
    else:
        parts.append("metric_or_checkpoint_evidence=<none>")

    tail = "\n".join(lines[-8:]) if lines else ""
    return "\n".join(parts) + "\n\nlast_lines:\n" + tail


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a Verl SDPO smoke log.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--expect-steps", type=int, default=1)
    args = parser.parse_args()
    print(inspect_log(args.log, args.expect_steps))


if __name__ == "__main__":
    main()
