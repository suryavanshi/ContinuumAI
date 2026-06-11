#!/usr/bin/env bash
# SDPO-style self-distillation | single teacher | vLLM rollout | FSDP training | NVIDIA GPUs
#
# This mirrors Verl's OPD shell examples, but routes one self-teacher through the
# SDPO bridge settings used in this repository: task rewards stay enabled and
# the teacher is selected from each sample's data_source.

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/cache/models/Qwen_Qwen3_6-35B-A3B}
TEACHER_MODEL=${TEACHER_MODEL:-$STUDENT_MODEL}
TEACHER_KEY=${TEACHER_KEY:-agent-eto/eto-sft-trajectory}

TRAIN_FILE=${TRAIN_FILE:-/cache/data/eto_sdpo/train.parquet}
VAL_FILE=${VAL_FILE:-/cache/data/eto_sdpo/test.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-/cache/runtime/sdpo_reward.py}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

# TRAIN_SIZE_BSZ is kept as the primary override because it is easy to grep in
# run logs. TRAIN_BATCH_SIZE remains accepted for compatibility with Verl examples.
TRAIN_SIZE_BSZ=${TRAIN_SIZE_BSZ:-${TRAIN_BATCH_SIZE:-2}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_SIZE_BSZ}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-0}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_CLIP_RATIO=${PPO_CLIP_RATIO:-0.2}
FSDP_MODEL_DTYPE=${FSDP_MODEL_DTYPE:-bfloat16}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
OPTIMIZER_OVERRIDE_CONFIG=${OPTIMIZER_OVERRIDE_CONFIG:-}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-True}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_EP=${ROLLOUT_EP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}

TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NUM_REPLICAS=${TEACHER_NUM_REPLICAS:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_EP=${TEACHER_EP:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.9}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-$(( TEACHER_NUM_REPLICAS * TEACHER_TP ))}

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}
DISTILLATION_CLIP_RATIO=${DISTILLATION_CLIP_RATIO:-0.2}
DISTILLATION_LOSS_MAX_CLAMP=${DISTILLATION_LOSS_MAX_CLAMP:-10.0}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-True}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}

PROJECT_NAME=${PROJECT_NAME:-continuum_verl_sdpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen_sdpo_mopd_fsdp}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$HOME/checkpoints/$EXPERIMENT_NAME}
LOGGER=${LOGGER:-console}
BALANCE_BATCH=${BALANCE_BATCH:-False}
SHUFFLE=${SHUFFLE:-False}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}
# ---- end user-adjustable ----

max_num_tokens=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
if [[ "${PPO_MAX_TOKEN_LEN_PER_GPU}" == "0" ]]; then
    PPO_MAX_TOKEN_LEN_PER_GPU=$(( max_num_tokens * TRAIN_SIZE_BSZ ))
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=${TRAIN_SIZE_BSZ}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=${SHUFFLE}
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD}
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=${USE_TORCH_COMPILE}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.clip_ratio=${PPO_CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${PPO_CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${PPO_CLIP_RATIO}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}
)
if [[ -n "${OPTIMIZER_OVERRIDE_CONFIG}" ]]; then
    ACTOR+=(actor_rollout_ref.actor.optim.override_optimizer_config="${OPTIMIZER_OVERRIDE_CONFIG}")
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REWARD=(
    reward.custom_reward_function.name=compute_score
    reward.custom_reward_function.path="$REWARD_FN_PATH"
)

TRAINER=(
    trainer.balance_batch=${BALANCE_BATCH}
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.max_actor_ckpt_to_keep=1
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR"
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model']"
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    distillation.teacher_models.teacher_model.key="$TEACHER_KEY"
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.num_replicas=${TEACHER_NUM_REPLICAS}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.expert_parallel_size=${TEACHER_EP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.teacher_models.teacher_model.inference.max_num_seqs=${MAX_NUM_SEQS}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS}
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.clip_ratio=${DISTILLATION_CLIP_RATIO}
    distillation.distillation_loss.clip_ratio_low=${DISTILLATION_CLIP_RATIO}
    distillation.distillation_loss.clip_ratio_high=${DISTILLATION_CLIP_RATIO}
    distillation.distillation_loss.distillation_loss_coef=${DISTILLATION_LOSS_COEF}
    distillation.distillation_loss.loss_max_clamp=${DISTILLATION_LOSS_MAX_CLAMP}
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${DISTILLATION[@]}" \
    "$@"
