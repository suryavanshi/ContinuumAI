#!/usr/bin/env bash
# Harvey LAB SDPO-style self-distillation defaults for Qwen3.6-35B-A3B.
#
# This script runs inside Modal. It reuses the generic ContinuumAI Verl SDPO
# shell but pins the dataset, teacher key, and longer legal-work-product lengths.

set -euo pipefail

export STUDENT_MODEL="${STUDENT_MODEL:-/cache/models/Qwen_Qwen3_6-35B-A3B}"
export TEACHER_MODEL="${TEACHER_MODEL:-$STUDENT_MODEL}"
export TEACHER_KEY="${TEACHER_KEY:-harvey/lab}"

export TRAIN_FILE="${TRAIN_FILE:-/cache/data/harvey_lab_sdpo/train.parquet}"
export VAL_FILE="${VAL_FILE:-/cache/data/harvey_lab_sdpo/test.parquet}"
export REWARD_FN_PATH="${REWARD_FN_PATH:-/cache/runtime/sdpo_reward.py}"

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export TRAIN_SIZE_BSZ="${TRAIN_SIZE_BSZ:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-6144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

export FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-bfloat16}"
export ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-True}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"
export ROLLOUT_EP="${ROLLOUT_EP:-4}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.35}"

export TEACHER_WORLD_SIZE="${TEACHER_WORLD_SIZE:-1}"
export TEACHER_TP="${TEACHER_TP:-1}"
export TEACHER_EP="${TEACHER_EP:-1}"
export TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.9}"

export PROJECT_NAME="${PROJECT_NAME:-continuum_harvey_sdpo}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen36_35b_a3b_harvey_sdpo}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"

exec bash /opt/continuum/run_qwen_sdpo_mopd_fsdp.sh "$@"
