from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import modal

try:
    import modal.experimental as modal_experimental
except Exception:  # pragma: no cover - depends on Modal client version.
    modal_experimental = None


APP_NAME = "continuum-verl-kimi-k26-sdpo"
VERL_IMAGE_TAG = os.environ.get("VERL_IMAGE_TAG", "verlai/verl:vllm020.dev1")
THIS_FILE = pathlib.Path(__file__).resolve()
DEFAULT_VERL_LOCAL_DIR = THIS_FILE.parents[2] / "verl" if len(THIS_FILE.parents) > 2 else pathlib.Path("/opt/verl")
VERL_LOCAL_DIR = pathlib.Path(
    os.environ.get("VERL_LOCAL_DIR", str(DEFAULT_VERL_LOCAL_DIR))
).expanduser().resolve()
REMOTE_VERL_DIR = "/opt/verl"
CACHE_DIR = pathlib.Path("/cache")
DATA_ROOT = CACHE_DIR / "data"
RUNTIME_DIR = CACHE_DIR / "runtime"
LOG_DIR = CACHE_DIR / "logs"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("continuum-verl-kimi-k26-sdpo-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(VERL_IMAGE_TAG)
    .add_local_dir(
        VERL_LOCAL_DIR,
        remote_path=REMOTE_VERL_DIR,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "wandb",
            "checkpoints",
            "outputs",
            "logs",
        ],
    )
    .run_commands(
        f"cd {REMOTE_VERL_DIR} && python -m pip install --no-deps -e .",
    )
    .pip_install(
        "hf-transfer",
        "pydantic>=2.12,<3",
        "fastapi[standard]>=0.115.0",
        "aiohttp>=3.13.3",
        "typer>=0.20.0",
        "rich>=13.7.1",
        "importlib-metadata>=6,<8.8",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_VERL_DIR,
            "HF_HOME": "/cache/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
    )
)


KIMI_SDPO_REWARD_CODE = r'''
import re


def _normalize(value):
    value = str(value or "").strip()
    value = value.replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value.lower()


def _extract_answer(text):
    text = str(text or "")
    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        return marker[-1].strip()
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged[-1].strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    number = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if number:
        return number[-1]
    return text.strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    extra_info = extra_info or {}
    extracted = _extract_answer(solution_str)
    normalized_prediction = _normalize(extracted)
    normalized_truth = _normalize(ground_truth)
    exact = bool(normalized_truth) and normalized_prediction == normalized_truth
    contains = bool(normalized_truth) and normalized_truth in _normalize(solution_str)
    score = 1.0 if exact else 0.5 if contains else 0.0
    feedback = extra_info.get("feedback_raw", "")
    return {
        "score": score,
        "extracted_answer": extracted,
        "exact": exact,
        "contains_ground_truth": contains,
        "feedback_raw": feedback,
        "sdpo_feedback_available": bool(feedback),
        "data_source": data_source,
    }
'''


@dataclass(frozen=True)
class KimiTopology:
    name: str
    cluster_nodes: int
    gpus_per_node: int
    actor_tp: int
    actor_pp: int
    actor_ep: int
    actor_etp: int
    actor_cp: int
    rollout_tp: int
    rollout_dp: int
    rollout_ep: int
    teacher_tp: int
    lora_rank: int = 0
    lora_alpha: int = 64


H200_FULL = KimiTopology(
    name="h200-full",
    cluster_nodes=16,
    gpus_per_node=8,
    actor_tp=8,
    actor_pp=2,
    actor_ep=8,
    actor_etp=1,
    actor_cp=1,
    rollout_tp=8,
    rollout_dp=1,
    rollout_ep=8,
    teacher_tp=8,
)

B200_FULL = KimiTopology(
    name="b200-full",
    cluster_nodes=8,
    gpus_per_node=8,
    actor_tp=8,
    actor_pp=1,
    actor_ep=8,
    actor_etp=1,
    actor_cp=1,
    rollout_tp=8,
    rollout_dp=1,
    rollout_ep=8,
    teacher_tp=8,
)

H200_LORA = KimiTopology(
    name="h200-lora",
    cluster_nodes=2,
    gpus_per_node=8,
    actor_tp=4,
    actor_pp=1,
    actor_ep=4,
    actor_etp=1,
    actor_cp=1,
    rollout_tp=8,
    rollout_dp=1,
    rollout_ep=8,
    teacher_tp=8,
    lora_rank=32,
    lora_alpha=64,
)


def _clustered(size: int, rdma: bool = True):
    if modal_experimental is None or not hasattr(modal_experimental, "clustered"):
        def passthrough(fn):
            return fn

        return passthrough
    return modal_experimental.clustered(size=size, rdma=rdma)


def _get_cluster_info():
    if modal_experimental is not None and hasattr(modal_experimental, "get_cluster_info"):
        return modal_experimental.get_cluster_info()

    class SingleNodeInfo:
        rank = 0
        container_ips = ["127.0.0.1"]

    return SingleNodeInfo()


def _run(cmd: list[str], cwd: str = REMOTE_VERL_DIR, env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(CACHE_DIR),
            "HF_HOME": "/cache/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": REMOTE_VERL_DIR,
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "0"),
            "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "^lo,docker0"),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
            "RAY_DEDUP_LOGS": "0",
        }
    )
    return env


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _dataset_dir(dataset: str) -> pathlib.Path:
    return DATA_ROOT / _safe_name(dataset)


def _dataset_paths(dataset: str) -> tuple[pathlib.Path, pathlib.Path]:
    data_dir = _dataset_dir(dataset)
    return data_dir / "train.parquet", data_dir / "test.parquet"


def _write_reward_module() -> pathlib.Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    reward_path = RUNTIME_DIR / "kimi_k26_sdpo_reward.py"
    reward_path.write_text(KIMI_SDPO_REWARD_CODE)
    return reward_path


def _prepare_sdpo_dataset_script(
    *,
    out_dir: pathlib.Path,
    hf_dataset: str,
    hf_config: str | None,
    train_split: str,
    val_split: str,
    prompt_column: str,
    answer_column: str,
    feedback_column: str | None,
    previous_attempt_column: str | None,
    data_source: str,
    ability: str,
    train_rows: int,
    val_rows: int,
    instruction_suffix: str,
    static_feedback: str,
    reprompt_template: str,
) -> str:
    return f"""
import pathlib
import datasets

out_dir = pathlib.Path({str(out_dir)!r})
out_dir.mkdir(parents=True, exist_ok=True)

hf_dataset = {hf_dataset!r}
hf_config = {hf_config!r}
loaded = datasets.load_dataset(hf_dataset, hf_config) if hf_config else datasets.load_dataset(hf_dataset)
train = loaded[{train_split!r}]
val = loaded[{val_split!r}]

def limit_rows(split, rows):
    rows = int(rows)
    if rows <= 0:
        return split
    return split.select(range(min(rows, len(split))))

train = limit_rows(train, {int(train_rows)})
val = limit_rows(val, {int(val_rows)})

prompt_column = {prompt_column!r}
answer_column = {answer_column!r}
feedback_column = {feedback_column!r}
previous_attempt_column = {previous_attempt_column!r}
instruction_suffix = {instruction_suffix!r}
static_feedback = {static_feedback!r}
reprompt_template = {reprompt_template!r}

def read_column(example, column, default=""):
    if not column:
        return default
    value = example.get(column, default)
    return "" if value is None else str(value)

def make_prompt(base_prompt, previous_attempt, feedback):
    base_prompt = base_prompt.strip()
    if instruction_suffix:
        base_prompt = base_prompt + " " + instruction_suffix
    if previous_attempt or feedback:
        return reprompt_template.format(
            prompt=base_prompt,
            solution=previous_attempt,
            feedback=feedback,
        )
    return base_prompt

def make_map_fn(split_name):
    def process_fn(example, idx):
        prompt_raw = read_column(example, prompt_column)
        answer_raw = read_column(example, answer_column)
        feedback_raw = read_column(example, feedback_column, static_feedback) or static_feedback
        previous_attempt = read_column(example, previous_attempt_column)
        prompt = make_prompt(prompt_raw, previous_attempt, feedback_raw)
        return {{
            "data_source": {data_source!r},
            "prompt": [{{"role": "user", "content": prompt}}],
            "ability": {ability!r},
            "reward_model": {{"style": "rule", "ground_truth": answer_raw}},
            "extra_info": {{
                "split": split_name,
                "index": idx,
                "original_prompt": prompt_raw,
                "answer": answer_raw,
                "feedback_raw": feedback_raw,
                "previous_attempt": previous_attempt,
                "sdpo_bridge": True,
                "target_model_family": "kimi-k2.6",
            }},
        }}
    return process_fn

train = train.map(function=make_map_fn("train"), with_indices=True)
val = val.map(function=make_map_fn("test"), with_indices=True)
train.to_parquet(out_dir / "train.parquet")
val.to_parquet(out_dir / "test.parquet")
print("prepared", len(train), "train rows and", len(val), "val rows from", hf_dataset)
"""


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    timeout=1800,
    startup_timeout=1800,
)
def prepare_kimi_sdpo_dataset(
    dataset: str = "gsm8k_kimi_k26_sdpo",
    hf_dataset: str = "openai/gsm8k",
    hf_config: str | None = "main",
    train_split: str = "train",
    val_split: str = "test",
    prompt_column: str = "question",
    answer_column: str = "answer",
    feedback_column: str | None = None,
    previous_attempt_column: str | None = None,
    data_source: str = "openai/gsm8k",
    ability: str = "math",
    train_rows: int = 8,
    val_rows: int = 4,
    instruction_suffix: str = 'Think carefully and put the final answer after "####".',
    static_feedback: str = "",
    reprompt_template: str = (
        "{prompt}\\n\\n"
        "Previous attempt:\\n{solution}\\n\\n"
        "Environment feedback:\\n{feedback}\\n\\n"
        "Use the feedback to produce a corrected solution. Keep only useful reasoning."
    ),
) -> str:
    data_dir = _dataset_dir(dataset)
    script = _prepare_sdpo_dataset_script(
        out_dir=data_dir,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        train_split=train_split,
        val_split=val_split,
        prompt_column=prompt_column,
        answer_column=answer_column,
        feedback_column=feedback_column,
        previous_attempt_column=previous_attempt_column,
        data_source=data_source,
        ability=ability,
        train_rows=train_rows,
        val_rows=val_rows,
        instruction_suffix=instruction_suffix,
        static_feedback=static_feedback,
        reprompt_template=reprompt_template,
    )
    _run([sys.executable, "-c", script], env=_base_env())
    cache_volume.commit()
    train_path, val_path = _dataset_paths(dataset)
    return f"{train_path} and {val_path}"


def _start_ray_for_cluster(expected_nodes: int, gpus_per_node: int) -> tuple[object, dict[str, str]]:
    if expected_nodes > 1 and (
        modal_experimental is None
        or not hasattr(modal_experimental, "clustered")
        or not hasattr(modal_experimental, "get_cluster_info")
    ):
        raise RuntimeError(
            "This Kimi topology requires Modal clustered functions. Upgrade the "
            "Modal client or use a single-node/smaller script."
        )

    info = _get_cluster_info()
    rank = int(info.rank)
    ips = list(info.container_ips)
    head_ip = ips[0]
    env = _base_env()

    print(f"cluster rank={rank} expected_nodes={expected_nodes} ips={ips}", flush=True)
    if len(ips) < expected_nodes:
        print(
            f"Modal cluster currently reports {len(ips)} nodes; waiting for {expected_nodes}.",
            flush=True,
        )

    if rank == 0:
        _run(
            [
                "ray",
                "start",
                "--head",
                "--node-ip-address",
                head_ip,
                "--port",
                "6379",
                "--dashboard-host",
                "0.0.0.0",
                "--num-gpus",
                str(gpus_per_node),
                "--disable-usage-stats",
            ],
            env=env,
        )
        env["RAY_ADDRESS"] = "auto"
        time.sleep(20)
    else:
        _run(
            [
                "ray",
                "start",
                "--address",
                f"{head_ip}:6379",
                "--node-ip-address",
                ips[rank],
                "--num-gpus",
                str(gpus_per_node),
                "--disable-usage-stats",
                "--block",
            ],
            env=env,
        )
    return info, env


def _append_common_verl_args(
    *,
    args: list[str],
    topology: KimiTopology,
    model: str,
    self_teacher_model: str,
    mcore_model_path: str | None,
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    teacher_key: str,
    reward_path: pathlib.Path,
    run_name: str,
    total_training_steps: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    rollout_n: int,
    max_num_batched_tokens: int,
    distillation_topk: int,
    distillation_loss_coef: float,
    learning_rate: str,
    all_offload: bool,
    lora_rank: int,
    lora_alpha: int,
) -> None:
    max_num_tokens = max_prompt_length + max_response_length + 1
    ppo_max_token_len_per_gpu = max_num_tokens
    offload_value = str(bool(all_offload))

    args.extend(
        [
            "algorithm.adv_estimator=grpo",
            "algorithm.use_kl_in_reward=False",
            f"data.train_files={train_path}",
            f"data.val_files={val_path}",
            f"data.train_batch_size={train_batch_size}",
            f"data.max_prompt_length={max_prompt_length}",
            f"data.max_response_length={max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            "data.shuffle=False",
            f"actor_rollout_ref.model.path={model}",
            "actor_rollout_ref.model.trust_remote_code=True",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.model.use_fused_kernels=True",
            "actor_rollout_ref.actor.use_torch_compile=False",
            f"actor_rollout_ref.actor.optim.lr={learning_rate}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "actor_rollout_ref.actor.entropy_coeff=0",
            "actor_rollout_ref.actor.megatron.use_mbridge=True",
            "actor_rollout_ref.actor.megatron.vanilla_mbridge=False",
            f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={topology.actor_tp}",
            f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={topology.actor_pp}",
            f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={topology.actor_ep}",
            f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={topology.actor_etp}",
            f"actor_rollout_ref.actor.megatron.context_parallel_size={topology.actor_cp}",
            f"actor_rollout_ref.actor.megatron.param_offload={offload_value}",
            f"actor_rollout_ref.actor.megatron.optimizer_offload={offload_value}",
            f"actor_rollout_ref.actor.megatron.grad_offload={offload_value}",
            "+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True",
            "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform",
            "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full",
            "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1",
            "+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={topology.rollout_tp}",
            f"actor_rollout_ref.rollout.data_parallel_size={topology.rollout_dp}",
            f"actor_rollout_ref.rollout.expert_parallel_size={topology.rollout_ep}",
            "actor_rollout_ref.rollout.moe_tensor_parallel_size=1",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
            "actor_rollout_ref.rollout.enforce_eager=True",
            "actor_rollout_ref.rollout.enable_chunked_prefill=True",
            "actor_rollout_ref.rollout.enable_prefix_caching=True",
            "actor_rollout_ref.rollout.free_cache_engine=True",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            f"actor_rollout_ref.rollout.max_model_len={max_num_tokens}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens}",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
            f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={topology.actor_tp}",
            f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={topology.actor_pp}",
            f"actor_rollout_ref.ref.megatron.expert_model_parallel_size={topology.actor_ep}",
            f"actor_rollout_ref.ref.megatron.expert_tensor_parallel_size={topology.actor_etp}",
            f"actor_rollout_ref.ref.megatron.context_parallel_size={topology.actor_cp}",
            f"actor_rollout_ref.ref.megatron.param_offload={offload_value}",
            "actor_rollout_ref.ref.megatron.use_mbridge=True",
            "reward.custom_reward_function.name=compute_score",
            f"reward.custom_reward_function.path={reward_path}",
            "trainer.balance_batch=True",
            "trainer.logger=console",
            "trainer.project_name=continuum_verl_kimi_k26_sdpo",
            f"trainer.experiment_name={run_name}",
            f"trainer.n_gpus_per_node={topology.gpus_per_node}",
            f"trainer.nnodes={topology.cluster_nodes}",
            "trainer.val_before_train=False",
            "trainer.save_freq=-1",
            "trainer.test_freq=-1",
            "trainer.total_epochs=1",
            f"trainer.total_training_steps={total_training_steps}",
            f"trainer.default_local_dir={CACHE_DIR / 'checkpoints' / run_name}",
            "ray_kwargs.ray_init.address=auto",
            "model_engine=megatron",
            "distillation.enabled=True",
            f"distillation.n_gpus_per_node={topology.gpus_per_node}",
            f"distillation.nnodes={topology.cluster_nodes}",
            "distillation.teacher_key=data_source",
            f"distillation.teacher_models.teacher_model.key={teacher_key}",
            f"distillation.teacher_models.teacher_model.model_path={self_teacher_model}",
            "distillation.teacher_models.teacher_model.num_replicas=1",
            "distillation.teacher_models.teacher_model.inference.name=vllm",
            f"distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size={topology.teacher_tp}",
            "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.30",
            f"distillation.teacher_models.teacher_model.inference.max_model_len={max_num_tokens}",
            "distillation.distillation_loss.loss_mode=k1",
            f"distillation.distillation_loss.topk={distillation_topk}",
            "distillation.distillation_loss.use_task_rewards=True",
            "distillation.distillation_loss.use_policy_gradient=True",
            f"distillation.distillation_loss.distillation_loss_coef={distillation_loss_coef}",
            "distillation.distillation_loss.loss_max_clamp=10.0",
            "distillation.distillation_loss.log_prob_min_clamp=-10.0",
        ]
    )

    if mcore_model_path:
        args.extend(
            [
                "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True",
                f"actor_rollout_ref.actor.megatron.dist_checkpointing_path={mcore_model_path}",
                "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True",
                f"actor_rollout_ref.ref.megatron.dist_checkpointing_path={mcore_model_path}",
            ]
        )

    if lora_rank > 0:
        args.extend(
            [
                f"actor_rollout_ref.model.lora.rank={lora_rank}",
                f"actor_rollout_ref.model.lora.alpha={lora_alpha}",
                "actor_rollout_ref.model.lora.lora_A_init_method=kaiming",
            ]
        )


def _train_kimi_sdpo(
    *,
    topology: KimiTopology,
    model: str,
    self_teacher_model: str | None,
    dataset: str,
    train_files: str | None,
    val_files: str | None,
    teacher_key: str,
    total_training_steps: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    rollout_n: int,
    max_num_batched_tokens: int,
    distillation_topk: int,
    distillation_loss_coef: float,
    learning_rate: str,
    all_offload: bool,
    mcore_model_path: str | None,
) -> str:
    info, env = _start_ray_for_cluster(topology.cluster_nodes, topology.gpus_per_node)
    if int(info.rank) != 0:
        return "worker joined ray cluster"

    default_train_path, default_val_path = _dataset_paths(dataset)
    train_path = pathlib.Path(train_files) if train_files else default_train_path
    val_path = pathlib.Path(val_files) if val_files else default_val_path
    teacher_model = self_teacher_model or model

    if not train_path.exists() or not val_path.exists():
        raise RuntimeError(
            "Missing train/validation parquet files. Run prepare_kimi_sdpo_dataset first, "
            "or pass --skip-prepare --train-files <path> --val-files <path>."
        )

    reward_path = _write_reward_module()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{_safe_name(dataset)}-{topology.name}-sdpo-{stamp}"
    log_path = LOG_DIR / f"{run_name}.log"

    args: list[str] = []
    _append_common_verl_args(
        args=args,
        topology=topology,
        model=model,
        self_teacher_model=teacher_model,
        mcore_model_path=mcore_model_path,
        train_path=train_path,
        val_path=val_path,
        teacher_key=teacher_key,
        reward_path=reward_path,
        run_name=run_name,
        total_training_steps=total_training_steps,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        rollout_n=rollout_n,
        max_num_batched_tokens=max_num_batched_tokens,
        distillation_topk=distillation_topk,
        distillation_loss_coef=distillation_loss_coef,
        learning_rate=learning_rate,
        all_offload=all_offload,
        lora_rank=topology.lora_rank,
        lora_alpha=topology.lora_alpha,
    )

    print("Kimi K2.6 SDPO configuration:", flush=True)
    print(f"  topology={topology}", flush=True)
    print(f"  model={model}", flush=True)
    print(f"  self_teacher_model={teacher_model}", flush=True)
    print(f"  train_files={train_path}", flush=True)
    print(f"  val_files={val_path}", flush=True)
    print(f"  mcore_model_path={mcore_model_path}", flush=True)
    print(f"  log_path={log_path}", flush=True)

    _run(
        [
            sys.executable,
            "-c",
            (
                "import torch, transformers, vllm, verl; "
                "print('torch', torch.__version__, 'cuda', torch.version.cuda, "
                "'available', torch.cuda.is_available()); "
                "print('transformers', transformers.__version__); "
                "print('vllm', vllm.__version__)"
            ),
        ],
        env=env,
    )

    cmd = (
        "set -euo pipefail; "
        f"{shlex.quote(sys.executable)} -m verl.trainer.main_ppo "
        f"{' '.join(shlex.quote(arg) for arg in args)} 2>&1 | tee {shlex.quote(str(log_path))}"
    )
    _run(["bash", "-lc", cmd], env=env)
    cache_volume.commit()
    return str(log_path)


@app.function(
    image=image,
    gpu="H200:8",
    volumes={"/cache": cache_volume},
    timeout=60 * 60 * 24,
    startup_timeout=3600,
    ephemeral_disk=1_000_000,
)
@_clustered(size=16, rdma=True)
def train_kimi_sdpo_h200_full(**kwargs) -> str:
    return _train_kimi_sdpo(topology=H200_FULL, **kwargs)


@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/cache": cache_volume},
    timeout=60 * 60 * 24,
    startup_timeout=3600,
    ephemeral_disk=1_000_000,
)
@_clustered(size=8, rdma=True)
def train_kimi_sdpo_b200_full(**kwargs) -> str:
    return _train_kimi_sdpo(topology=B200_FULL, **kwargs)


@app.function(
    image=image,
    gpu="H200:8",
    volumes={"/cache": cache_volume},
    timeout=60 * 60 * 24,
    startup_timeout=3600,
    ephemeral_disk=1_000_000,
)
@_clustered(size=2, rdma=True)
def train_kimi_sdpo_h200_lora(**kwargs) -> str:
    return _train_kimi_sdpo(topology=H200_LORA, **kwargs)


@app.local_entrypoint()
def main(
    mode: str = "h200-lora",
    model: str = "moonshotai/Kimi-K2.6",
    self_teacher_model: str | None = None,
    mcore_model_path: str | None = None,
    dataset: str = "gsm8k_kimi_k26_sdpo",
    train_files: str | None = None,
    val_files: str | None = None,
    teacher_key: str = "openai/gsm8k",
    hf_dataset: str = "openai/gsm8k",
    hf_config: str | None = "main",
    train_split: str = "train",
    val_split: str = "test",
    prompt_column: str = "question",
    answer_column: str = "answer",
    feedback_column: str | None = None,
    previous_attempt_column: str | None = None,
    data_source: str = "openai/gsm8k",
    ability: str = "math",
    train_rows: int = 8,
    val_rows: int = 4,
    instruction_suffix: str = 'Think carefully and put the final answer after "####".',
    static_feedback: str = "",
    total_training_steps: int = 1,
    train_batch_size: int = 8,
    ppo_mini_batch_size: int = 8,
    max_prompt_length: int = 2048,
    max_response_length: int = 2048,
    rollout_n: int = 1,
    max_num_batched_tokens: int = 8192,
    distillation_topk: int = 64,
    distillation_loss_coef: float = 1.0,
    learning_rate: str = "1e-6",
    all_offload: bool = True,
    skip_prepare: bool = False,
) -> None:
    print(f"Modal app: {APP_NAME}")
    print(f"Mode: {mode}")
    if not skip_prepare:
        prepared = prepare_kimi_sdpo_dataset.remote(
            dataset=dataset,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            train_split=train_split,
            val_split=val_split,
            prompt_column=prompt_column,
            answer_column=answer_column,
            feedback_column=feedback_column,
            previous_attempt_column=previous_attempt_column,
            data_source=data_source,
            ability=ability,
            train_rows=train_rows,
            val_rows=val_rows,
            instruction_suffix=instruction_suffix,
            static_feedback=static_feedback,
        )
        print(f"Prepared data: {prepared}")

    train_kwargs = dict(
        model=model,
        self_teacher_model=self_teacher_model,
        mcore_model_path=mcore_model_path,
        dataset=dataset,
        train_files=train_files,
        val_files=val_files,
        teacher_key=teacher_key,
        total_training_steps=total_training_steps,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        rollout_n=rollout_n,
        max_num_batched_tokens=max_num_batched_tokens,
        distillation_topk=distillation_topk,
        distillation_loss_coef=distillation_loss_coef,
        learning_rate=learning_rate,
        all_offload=all_offload,
    )

    if mode == "h200-full":
        log_path = train_kimi_sdpo_h200_full.remote(**train_kwargs)
    elif mode == "b200-full":
        log_path = train_kimi_sdpo_b200_full.remote(**train_kwargs)
    elif mode == "h200-lora":
        log_path = train_kimi_sdpo_h200_lora.remote(**train_kwargs)
    else:
        raise ValueError("mode must be one of: h200-lora, h200-full, b200-full")

    print(f"Training log committed to Modal volume: {log_path}")
