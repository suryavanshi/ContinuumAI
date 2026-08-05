from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("VERL_UPLOAD_LOCAL", "0")

from SDPO import modal_verl_sdpo


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

    def test_hf_cli_downloader_is_available(self) -> None:
        self.assertTrue(hasattr(modal_verl_sdpo.download_model_hf_cli, "remote"))

    def test_small_sdpo_smoke_has_a_dedicated_executor(self) -> None:
        self.assertTrue(hasattr(modal_verl_sdpo.train_sdpo_smoke, "remote"))

    def test_local_entrypoint_forces_bfloat16_for_smoke(self) -> None:
        source = Path(modal_verl_sdpo.__file__).read_text(encoding="utf-8")
        self.assertIn('fsdp_model_dtype="bfloat16" if smoke_gpu else fsdp_model_dtype', source)

    def test_dataset_prepare_writes_plain_parquet_without_hf_metadata(self) -> None:
        script = modal_verl_sdpo._prepare_sdpo_dataset_script(
            out_dir=Path("/cache/data/gsm8k_sdpo"),
            hf_dataset="openai/gsm8k",
            hf_config="main",
            hf_train_file=None,
            hf_val_file=None,
            train_split="train",
            val_split="test",
            dataset_format="qa",
            conversations_column="conversations",
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

    def test_sharegpt_prepare_uses_trajectory_as_feedback_hint(self) -> None:
        script = modal_verl_sdpo._prepare_sdpo_dataset_script(
            out_dir=Path("/cache/data/eto_sdpo"),
            hf_dataset="agent-eto/eto-sft-trajectory",
            hf_config=None,
            hf_train_file="data/webshop_sft.json",
            hf_val_file="data/sciworld_sft.json",
            train_split="train",
            val_split="test",
            dataset_format="sharegpt",
            conversations_column="conversations",
            prompt_column="question",
            answer_column="answer",
            feedback_column=None,
            previous_attempt_column=None,
            data_source="agent-eto/eto-sft-trajectory",
            ability="agent",
            train_rows=2,
            val_rows=1,
            instruction_suffix="",
            static_feedback="",
            reprompt_template="{prompt}\n\n{feedback}",
        )

        self.assertIn('"train"] = hf_train_file', script)
        self.assertIn('dataset_format == "sharegpt"', script)
        self.assertIn("Successful reference trajectory:", script)
        self.assertIn("conversation_to_fields(example)", script)

    def test_reward_extra_info_does_not_shadow_batch_data_source(self) -> None:
        self.assertNotIn('"data_source": data_source', modal_verl_sdpo.SDPO_REWARD_CODE)
        self.assertIn('"sdpo_feedback_available": bool(feedback)', modal_verl_sdpo.SDPO_REWARD_CODE)

    def test_harvey_reward_branch_is_opt_in_by_data_source(self) -> None:
        self.assertIn('str(data_source or "") == "harvey/lab"', modal_verl_sdpo.SDPO_REWARD_CODE)
        self.assertIn('"harvey_reward_hits": hits', modal_verl_sdpo.SDPO_REWARD_CODE)
        self.assertIn('"reward_terms"', modal_verl_sdpo.SDPO_REWARD_CODE)

    def test_harvey_modal_pipeline_is_available(self) -> None:
        self.assertTrue(hasattr(modal_verl_sdpo.prepare_harvey_sdpo_dataset, "remote"))
        self.assertTrue(callable(modal_verl_sdpo.harvey_main))
        self.assertEqual("/opt/continuum/run_qwen36_harvey_sdpo_fsdp.sh", modal_verl_sdpo.REMOTE_HARVEY_SDPO_SHELL)
        self.assertEqual("/opt/harvey-labs", modal_verl_sdpo.REMOTE_HARVEY_LABS_DIR)

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
            "total_epochs": 1,
            "train_batch_size": 2,
            "ppo_mini_batch_size": 2,
            "max_prompt_length": 512,
            "max_response_length": 128,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 2048,
            "n_gpus_per_node": 1,
            "tensor_model_parallel_size": 1,
            "rollout_expert_parallel_size": 1,
            "distillation_n_gpus_per_node": None,
            "distillation_tensor_model_parallel_size": None,
            "distillation_expert_parallel_size": None,
            "rollout_gpu_memory_utilization": 0.35,
            "distillation_gpu_memory_utilization": 0.45,
            "enable_activation_offload": False,
            "enable_param_offload": False,
            "enable_optimizer_offload": False,
            "fsdp_model_dtype": "fp32",
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
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=1", args)
        self.assertIn("actor_rollout_ref.rollout.expert_parallel_size=1", args)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.35", args)
        self.assertIn("actor_rollout_ref.model.enable_activation_offload=False", args)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.param_offload=False", args)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.optimizer_offload=False", args)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.model_dtype=fp32", args)
        self.assertIn("trainer.n_gpus_per_node=1", args)
        self.assertIn("trainer.total_epochs=1", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.max_num_seqs=8", args)
        self.assertIn(
            "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=2048",
            args,
        )
        self.assertIn("distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.45", args)

    def test_topology_args_can_use_two_gpu_tensor_and_expert_parallel(self) -> None:
        args = self.build_args(
            n_gpus_per_node=2,
            tensor_model_parallel_size=2,
            rollout_expert_parallel_size=2,
        )

        self.assertIn("trainer.n_gpus_per_node=2", args)
        self.assertIn("distillation.n_gpus_per_node=2", args)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=2", args)
        self.assertIn("actor_rollout_ref.rollout.expert_parallel_size=2", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.expert_parallel_size=2", args)

    def test_teacher_topology_can_be_smaller_than_actor(self) -> None:
        args = self.build_args(
            n_gpus_per_node=2,
            tensor_model_parallel_size=2,
            rollout_expert_parallel_size=2,
            distillation_n_gpus_per_node=1,
            distillation_tensor_model_parallel_size=1,
            distillation_expert_parallel_size=1,
        )

        self.assertIn("trainer.n_gpus_per_node=2", args)
        self.assertIn("distillation.n_gpus_per_node=1", args)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=2", args)
        self.assertIn("actor_rollout_ref.rollout.expert_parallel_size=2", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1", args)
        self.assertIn("distillation.teacher_models.teacher_model.inference.expert_parallel_size=1", args)

    def test_cpu_offload_and_bfloat16_dtype_can_be_enabled(self) -> None:
        args = self.build_args(enable_activation_offload=True, fsdp_model_dtype="bfloat16")

        self.assertIn("actor_rollout_ref.model.enable_activation_offload=True", args)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16", args)

    def test_total_epochs_can_be_overridden(self) -> None:
        args = self.build_args(total_epochs=5, total_training_steps=5)

        self.assertIn("trainer.total_epochs=5", args)
        self.assertIn("trainer.total_training_steps=5", args)

    def test_vllm_memory_utilization_can_be_tuned_per_role(self) -> None:
        args = self.build_args(
            rollout_gpu_memory_utilization=0.5,
            distillation_gpu_memory_utilization=0.9,
        )

        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.5", args)
        self.assertIn(
            "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.9",
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

    def test_shell_env_can_route_to_harvey_dataset(self) -> None:
        env = modal_verl_sdpo._build_shell_env(
            model="/cache/models/Qwen_Qwen3_6-35B-A3B",
            teacher_model="/cache/models/Qwen_Qwen3_6-35B-A3B",
            teacher_key="harvey/lab",
            train_path=Path("/cache/data/harvey_lab_sdpo/train.parquet"),
            val_path=Path("/cache/data/harvey_lab_sdpo/test.parquet"),
            reward_path=Path("/cache/runtime/sdpo_reward.py"),
            run_name="harvey_lab_sdpo-shell-test",
            total_training_steps=1,
            total_epochs=1,
            train_batch_size=1,
            ppo_mini_batch_size=1,
            max_prompt_length=4096,
            max_response_length=1024,
            max_num_seqs=1,
            max_num_batched_tokens=6144,
            n_gpus_per_node=4,
            tensor_model_parallel_size=4,
            rollout_expert_parallel_size=4,
            distillation_n_gpus_per_node=1,
            distillation_tensor_model_parallel_size=1,
            distillation_expert_parallel_size=1,
            rollout_gpu_memory_utilization=0.35,
            distillation_gpu_memory_utilization=0.9,
            enable_activation_offload=True,
            fsdp_model_dtype="bfloat16",
            ppo_clip_ratio=0.2,
            distillation_clip_ratio=0.2,
            distillation_topk=32,
            distillation_loss_coef=1.0,
            distillation_loss_max_clamp=10.0,
            save_hf_checkpoint=False,
        )

        self.assertEqual("harvey/lab", env["TEACHER_KEY"])
        self.assertEqual("/cache/data/harvey_lab_sdpo/train.parquet", env["TRAIN_FILE"])
        self.assertEqual("4", env["NGPUS_PER_NODE"])
        self.assertEqual("4096", env["MAX_PROMPT_LENGTH"])
        self.assertEqual("1024", env["MAX_RESPONSE_LENGTH"])

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
