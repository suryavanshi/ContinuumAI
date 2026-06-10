from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.inspect_sdpo_log import inspect_log


class InspectSdpoLogTest(unittest.TestCase):
    def inspect_text(self, text: str, expect_steps: int = 1) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sdpo.log"
            path.write_text(text)
            return inspect_log(path, expect_steps)

    def test_model_load_only_log_is_incomplete(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "Total training steps: 1",
                    "Qwen3_5ForConditionalGeneration contains 852.99M parameters",
                    "Loading safetensors checkpoint shards: 100% Completed",
                ]
            )
        )

        self.assertIn("status=incomplete", result)
        self.assertIn("configured_steps=1", result)
        self.assertIn("metric_or_checkpoint_evidence=<none>", result)

    def test_expected_global_step_is_completed(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "Total training steps: 1",
                    "Saving checkpoint to /cache/checkpoints/run/global_step_1",
                ]
            )
        )

        self.assertIn("status=completed", result)
        self.assertIn("step_evidence=Saving checkpoint", result)

    def test_metric_without_expected_step_is_progress_only(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "Total training steps: 2",
                    "critic/rewards/mean=0.25 actor/pg_loss=0.1",
                ]
            ),
            expect_steps=2,
        )

        self.assertIn("status=progress-evidence-seen", result)
        self.assertIn("metric_or_checkpoint_evidence=critic/rewards/mean=0.25", result)

    def test_resume_from_expected_checkpoint_is_not_fresh_completion(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "Total training steps: 1",
                    "Found checkpoint: /cache/checkpoints/run/global_step_1",
                    "Load from checkpoint folder: /cache/checkpoints/run/global_step_1",
                    "Resuming from /cache/checkpoints/run/global_step_1",
                    "Training Progress: 100%|██████████| 1/1",
                ]
            )
        )

        self.assertIn("status=incomplete", result)
        self.assertIn("metric_or_checkpoint_evidence=<none>", result)

    def test_failure_takes_precedence_over_completion(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "Saving checkpoint to /cache/checkpoints/run/global_step_1",
                    "Traceback (most recent call last)",
                ]
            )
        )

        self.assertIn("status=failed", result)
        self.assertIn("failure_line=Traceback", result)

    def test_step_metric_after_shutdown_warning_is_completed_with_warning(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "Total training steps: 1",
                    "RuntimeError: DataLoader worker (pid 123) is killed by signal: Killed.",
                    "step:1 - actor/loss:-1.0 - training/global_step:1 - critic/rewards/mean:1.0",
                    "'Final validation metrics: None'",
                ]
            )
        )

        self.assertIn("status=completed-with-warning", result)
        self.assertIn("step_evidence=step:1 -", result)

    def test_import_index_error_is_failed(self) -> None:
        result = self.inspect_text(
            "\n".join(
                [
                    "File /root/modal_verl_sdpo.py, line 15, in <module>",
                    "IndexError: 2",
                    "Runner failed with exception: IndexError(2)",
                ]
            )
        )

        self.assertIn("status=failed", result)
        self.assertIn("failure_line=IndexError: 2", result)


if __name__ == "__main__":
    unittest.main()
