from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("VERL_UPLOAD_LOCAL", "0")

from scripts import modal_verl_sdpo


class ModalVerlSdpoArgsTest(unittest.TestCase):
    def test_image_only_mode_uses_latest_public_verl_ref(self) -> None:
        self.assertFalse(modal_verl_sdpo.VERL_UPLOAD_LOCAL)
        self.assertEqual("v0.8.0", modal_verl_sdpo.VERL_GIT_REF)

    def test_default_image_is_modelscope_fallback(self) -> None:
        self.assertIn("modelscope-registry.us-west-1.cr.aliyuncs.com", modal_verl_sdpo.VERL_IMAGE_TAG)
        self.assertIn("cuda12.9.1", modal_verl_sdpo.VERL_IMAGE_TAG)

    def test_model_cache_path_is_modal_volume_path(self) -> None:
        self.assertEqual(
            Path("/cache/models/hf/Qwen_Qwen3_5-0_8B"),
            modal_verl_sdpo._model_cache_path("Qwen/Qwen3.5-0.8B"),
        )

    def test_dataset_prepare_writes_plain_parquet_without_hf_metadata(self) -> None:
        script = modal_verl_sdpo._prepare_sdpo_dataset_script(
            out_dir=Path("/cache/data/gsm8k_sdpo"),
            hf_dataset="openai/gsm8k",
            hf_config="main",
            train_split="train",
            val_split="test",
            prompt_column="question",
            answer_column="answer",
            feedback_column=None,
            previous_attempt_column=None,
            data_source="openai/gsm8k",
            ability="math",
            train_rows=2,
            val_rows=1,
            instruction_suffix="",
            static_feedback="",
            reprompt_template="{prompt}",
        )

        self.assertIn("pa.Table.from_pylist(records)", script)
        self.assertIn("pq.write_table(table, path)", script)
        self.assertNotIn(".to_parquet(", script)
        self.assertNotIn(".map(function=", script)

    def test_reward_extra_info_does_not_shadow_batch_data_source(self) -> None:
        self.assertNotIn('"data_source": data_source', modal_verl_sdpo.SDPO_REWARD_CODE)
        self.assertIn('"sdpo_feedback_available": bool(feedback)', modal_verl_sdpo.SDPO_REWARD_CODE)

    def build_args(self, **overrides: object) -> list[str]:
        params = {
            "model": "/cache/models/hf/Qwen_Qwen3_5-0_8B",
            "teacher_model": "/cache/models/hf/Qwen_Qwen3_5-0_8B",
            "teacher_key": "openai/gsm8k",
            "train_path": Path("/cache/data/gsm8k_sdpo/train.parquet"),
            "val_path": Path("/cache/data/gsm8k_sdpo/test.parquet"),
            "reward_path": Path("/cache/runtime/sdpo_reward.py"),
            "run_name": "gsm8k_sdpo-sdpo-test",
            "total_training_steps": 1,
            "train_batch_size": 2,
            "ppo_mini_batch_size": 2,
            "max_prompt_length": 512,
            "max_response_length": 128,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 2048,
            "ppo_clip_ratio": 0.2,
            "distillation_clip_ratio": 0.2,
            "distillation_topk": 32,
            "distillation_loss_coef": 1.0,
            "distillation_loss_max_clamp": 10.0,
            "save_hf_checkpoint": True,
        }
        params.update(overrides)
        return modal_verl_sdpo._build_verl_args(**params)

    def test_smoke_args_include_scaling_sdpo_clip_and_group_size_one(self) -> None:
        args = self.build_args()

        self.assertIn("actor_rollout_ref.rollout.n=1", args)
        self.assertIn("actor_rollout_ref.actor.clip_ratio=0.2", args)
        self.assertIn("actor_rollout_ref.actor.clip_ratio_low=0.2", args)
        self.assertIn("actor_rollout_ref.actor.clip_ratio_high=0.2", args)
        self.assertIn("distillation.distillation_loss.clip_ratio=0.2", args)
        self.assertIn("distillation.distillation_loss.clip_ratio_low=0.2", args)
        self.assertIn("distillation.distillation_loss.clip_ratio_high=0.2", args)

    def test_smoke_args_keep_vllm_limits_small(self) -> None:
        args = self.build_args()

        self.assertIn("actor_rollout_ref.rollout.max_num_seqs=8", args)
        self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=2048", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.max_num_seqs=8", args)
        self.assertIn(
            "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=2048",
            args,
        )

    def test_smoke_args_use_single_agent_loop_worker_for_tiny_batch(self) -> None:
        args = self.build_args()

        self.assertIn("actor_rollout_ref.rollout.agent.num_workers=1", args)

    def test_smoke_args_save_hf_checkpoint_at_final_step(self) -> None:
        args = self.build_args(total_training_steps=1, save_hf_checkpoint=True)

        self.assertIn("trainer.save_freq=1", args)
        self.assertIn("trainer.max_actor_ckpt_to_keep=1", args)
        self.assertIn("actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model']", args)
        self.assertIn("actor_rollout_ref.actor.checkpoint.load_contents=['model']", args)

    def test_checkpoint_save_can_be_disabled(self) -> None:
        args = self.build_args(save_hf_checkpoint=False)

        self.assertIn("trainer.save_freq=-1", args)

    def test_distillation_uses_k1_policy_gradient_with_task_rewards(self) -> None:
        args = self.build_args()

        self.assertIn("actor_rollout_ref.model.path=/cache/models/hf/Qwen_Qwen3_5-0_8B", args)
        self.assertIn(
            "distillation.teacher_models.teacher_model.model_path=/cache/models/hf/Qwen_Qwen3_5-0_8B",
            args,
        )
        self.assertIn("distillation.distillation_loss.loss_mode=k1", args)
        self.assertIn("distillation.distillation_loss.use_task_rewards=True", args)
        self.assertIn("distillation.distillation_loss.use_policy_gradient=True", args)
        self.assertIn("distillation.distillation_loss.loss_max_clamp=10.0", args)


if __name__ == "__main__":
    unittest.main()
