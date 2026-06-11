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
REMOTE_SDPO_SHELL = "/opt/continuum/run_qwen_sdpo_mopd_fsdp.sh"

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
    .add_local_file(
        THIS_FILE.parent / "run_qwen_sdpo_mopd_fsdp.sh",
        remote_path=REMOTE_SDPO_SHELL,
        copy=True,
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


def _build_shell_env(
    *,
    topology: KimiTopology,
    student_model: str,
    teacher_model: str,
    teacher_key: str,
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    reward_path: pathlib.Path,
    run_name: str,
    total_training_steps: int,
    total_epochs: int,
    train_size_bsz: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size_per_gpu: int,
    log_prob_micro_batch_size_per_gpu: int,
    max_prompt_length: int,
    max_response_length: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    actor_lr: str,
    ppo_clip_ratio: float,
    fsdp_model_dtype: str,
    param_offload: bool,
    optimizer_offload: bool,
    optimizer_override_config: str,
    enable_activation_offload: bool,
    use_torch_compile: bool,
    rollout_gpu_mem_util: float,
    teacher_num_replicas: int,
    teacher_gpu_mem_util: float,
    distillation_loss_mode: str,
    distillation_topk: int,
    distillation_loss_coef: float,
    distillation_clip_ratio: float,
    distillation_loss_max_clamp: float,
    use_task_rewards: bool,
    use_policy_gradient: bool,
    balance_batch: bool,
    shuffle: bool,
    save_freq: int,
    test_freq: int,
    val_before_train: bool,
    trust_remote_code: bool,
) -> dict[str, str]:
    teacher_world_size = teacher_num_replicas * topology.teacher_tp
    return {
        "STUDENT_MODEL": student_model,
        "TEACHER_MODEL": teacher_model,
        "TEACHER_KEY": teacher_key,
        "TRAIN_FILE": str(train_path),
        "VAL_FILE": str(val_path),
        "REWARD_FN_PATH": str(reward_path),
        "NNODES": str(topology.cluster_nodes),
        "NGPUS_PER_NODE": str(topology.gpus_per_node),
        "TRAIN_SIZE_BSZ": str(train_size_bsz),
        "PPO_MINI_BATCH_SIZE": str(ppo_mini_batch_size),
        "PPO_MICRO_BATCH_SIZE_PER_GPU": str(ppo_micro_batch_size_per_gpu),
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": str(log_prob_micro_batch_size_per_gpu),
        "MAX_PROMPT_LENGTH": str(max_prompt_length),
        "MAX_RESPONSE_LENGTH": str(max_response_length),
        "MAX_NUM_BATCHED_TOKENS": str(max_num_batched_tokens),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "ACTOR_LR": actor_lr,
        "PPO_CLIP_RATIO": str(ppo_clip_ratio),
        "FSDP_MODEL_DTYPE": fsdp_model_dtype,
        "PARAM_OFFLOAD": str(bool(param_offload)),
        "OPTIMIZER_OFFLOAD": str(bool(optimizer_offload)),
        "OPTIMIZER_OVERRIDE_CONFIG": optimizer_override_config,
        "ENABLE_ACTIVATION_OFFLOAD": str(bool(enable_activation_offload)),
        "USE_TORCH_COMPILE": str(bool(use_torch_compile)),
        "ROLLOUT_TP": str(topology.rollout_tp),
        "ROLLOUT_EP": str(topology.rollout_ep),
        "ROLLOUT_GPU_MEM_UTIL": str(rollout_gpu_mem_util),
        "TEACHER_NNODES": str(topology.cluster_nodes),
        "TEACHER_NUM_REPLICAS": str(teacher_num_replicas),
        "TEACHER_TP": str(topology.teacher_tp),
        "TEACHER_EP": str(topology.rollout_ep),
        "TEACHER_GPU_MEM_UTIL": str(teacher_gpu_mem_util),
        "TEACHER_WORLD_SIZE": str(teacher_world_size),
        "DISTILLATION_LOSS_MODE": distillation_loss_mode,
        "DISTILLATION_TOPK": str(distillation_topk),
        "DISTILLATION_LOSS_COEF": str(distillation_loss_coef),
        "DISTILLATION_CLIP_RATIO": str(distillation_clip_ratio),
        "DISTILLATION_LOSS_MAX_CLAMP": str(distillation_loss_max_clamp),
        "USE_TASK_REWARDS": str(bool(use_task_rewards)),
        "USE_POLICY_GRADIENT": str(bool(use_policy_gradient)),
        "PROJECT_NAME": "continuum_verl_kimi_k26_sdpo",
        "EXPERIMENT_NAME": run_name,
        "DEFAULT_LOCAL_DIR": str(CACHE_DIR / "checkpoints" / run_name),
        "LOGGER": "console",
        "BALANCE_BATCH": str(bool(balance_batch)),
        "SHUFFLE": str(bool(shuffle)),
        "SAVE_FREQ": str(save_freq),
        "TEST_FREQ": str(test_freq),
        "TOTAL_EPOCHS": str(total_epochs),
        "TOTAL_TRAINING_STEPS": str(total_training_steps),
        "VAL_BEFORE_TRAIN": str(bool(val_before_train)),
        "TRUST_REMOTE_CODE": str(bool(trust_remote_code)),
    }


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
    total_epochs: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size_per_gpu: int,
    log_prob_micro_batch_size_per_gpu: int,
    max_prompt_length: int,
    max_response_length: int,
    rollout_n: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    ppo_clip_ratio: float,
    distillation_topk: int,
    distillation_loss_coef: float,
    distillation_clip_ratio: float,
    distillation_loss_max_clamp: float,
    learning_rate: str,
    fsdp_model_dtype: str,
    param_offload: bool,
    optimizer_offload: bool,
    optimizer_override_config: str,
    enable_activation_offload: bool,
    use_torch_compile: bool,
    rollout_gpu_mem_util: float,
    teacher_num_replicas: int,
    teacher_gpu_mem_util: float,
    use_task_rewards: bool,
    use_policy_gradient: bool,
    balance_batch: bool,
    shuffle: bool,
    save_hf_checkpoint: bool,
    mcore_model_path: str | None,
) -> str:
    if mcore_model_path:
        raise RuntimeError(
            "mcore_model_path is not used by the FSDP shell recipe. Pass an HF "
            "student model path via --model / STUDENT_MODEL instead."
        )

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
    tmp_log_path = pathlib.Path("/tmp") / f"{run_name}.log"
    save_freq = total_training_steps if save_hf_checkpoint else -1
    env["VERL_LOGGING_LEVEL"] = "INFO"
    shell_env = _build_shell_env(
        topology=topology,
        student_model=model,
        teacher_model=teacher_model,
        teacher_key=teacher_key,
        train_path=train_path,
        val_path=val_path,
        reward_path=reward_path,
        run_name=run_name,
        total_training_steps=total_training_steps,
        total_epochs=total_epochs,
        train_size_bsz=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        actor_lr=learning_rate,
        ppo_clip_ratio=ppo_clip_ratio,
        fsdp_model_dtype=fsdp_model_dtype,
        param_offload=param_offload,
        optimizer_offload=optimizer_offload,
        optimizer_override_config=optimizer_override_config,
        enable_activation_offload=enable_activation_offload,
        use_torch_compile=use_torch_compile,
        rollout_gpu_mem_util=rollout_gpu_mem_util,
        teacher_num_replicas=teacher_num_replicas,
        teacher_gpu_mem_util=teacher_gpu_mem_util,
        distillation_loss_mode="k1",
        distillation_topk=distillation_topk,
        distillation_loss_coef=distillation_loss_coef,
        distillation_clip_ratio=distillation_clip_ratio,
        distillation_loss_max_clamp=distillation_loss_max_clamp,
        use_task_rewards=use_task_rewards,
        use_policy_gradient=use_policy_gradient,
        balance_batch=balance_batch,
        shuffle=shuffle,
        save_freq=save_freq,
        test_freq=-1,
        val_before_train=False,
        trust_remote_code=True,
    )
    env.update(shell_env)

    extra_args: list[str] = []
    if rollout_n != 1:
        extra_args.append(f"actor_rollout_ref.rollout.n={rollout_n}")
    if topology.lora_rank > 0:
        extra_args.extend(
            [
                f"actor_rollout_ref.model.lora.rank={topology.lora_rank}",
                f"actor_rollout_ref.model.lora.alpha={topology.lora_alpha}",
                "actor_rollout_ref.model.lora.lora_A_init_method=kaiming",
            ]
        )

    print("Kimi K2.6 SDPO configuration:", flush=True)
    print(f"  topology={topology}", flush=True)
    for key in sorted(shell_env):
        print(f"  {key}={env[key]}", flush=True)
    if extra_args:
        print(f"  extra_args={' '.join(extra_args)}", flush=True)
    print(f"  shell_script={REMOTE_SDPO_SHELL}", flush=True)
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
        f"rm -f {shlex.quote(str(tmp_log_path))}; "
        "persist_log() { "
        f"if [ -f {shlex.quote(str(tmp_log_path))} ]; then "
        f"cp {shlex.quote(str(tmp_log_path))} {shlex.quote(str(log_path))}; "
        f"sync {shlex.quote(str(log_path))}; "
        "fi; "
        "}; "
        "trap persist_log EXIT; "
        "set +e; "
        f"bash {shlex.quote(REMOTE_SDPO_SHELL)} "
        f"{' '.join(shlex.quote(arg) for arg in extra_args)} "
        f"2>&1 | tee {shlex.quote(str(tmp_log_path))}; "
        "status=${PIPESTATUS[0]}; "
        'exit "${status}"'
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
    total_epochs: int = 1,
    train_batch_size: int = 8,
    ppo_mini_batch_size: int = 8,
    ppo_micro_batch_size_per_gpu: int = 1,
    log_prob_micro_batch_size_per_gpu: int = 1,
    max_prompt_length: int = 2048,
    max_response_length: int = 2048,
    rollout_n: int = 1,
    max_num_batched_tokens: int = 8192,
    max_num_seqs: int = 1,
    ppo_clip_ratio: float = 0.2,
    distillation_topk: int = 64,
    distillation_loss_coef: float = 1.0,
    distillation_clip_ratio: float = 0.2,
    distillation_loss_max_clamp: float = 10.0,
    learning_rate: str = "1e-6",
    all_offload: bool = True,
    fsdp_model_dtype: str = "bfloat16",
    optimizer_override_config: str = "",
    enable_activation_offload: bool = True,
    use_torch_compile: bool = False,
    rollout_gpu_mem_util: float = 0.35,
    teacher_num_replicas: int = 1,
    teacher_gpu_mem_util: float = 0.9,
    use_task_rewards: bool = True,
    use_policy_gradient: bool = True,
    balance_batch: bool = False,
    shuffle: bool = False,
    save_hf_checkpoint: bool = False,
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
        total_epochs=total_epochs,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        rollout_n=rollout_n,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        ppo_clip_ratio=ppo_clip_ratio,
        distillation_topk=distillation_topk,
        distillation_loss_coef=distillation_loss_coef,
        distillation_clip_ratio=distillation_clip_ratio,
        distillation_loss_max_clamp=distillation_loss_max_clamp,
        learning_rate=learning_rate,
        fsdp_model_dtype=fsdp_model_dtype,
        param_offload=all_offload,
        optimizer_offload=all_offload,
        optimizer_override_config=optimizer_override_config,
        enable_activation_offload=enable_activation_offload,
        use_torch_compile=use_torch_compile,
        rollout_gpu_mem_util=rollout_gpu_mem_util,
        teacher_num_replicas=teacher_num_replicas,
        teacher_gpu_mem_util=teacher_gpu_mem_util,
        use_task_rewards=use_task_rewards,
        use_policy_gradient=use_policy_gradient,
        balance_batch=balance_batch,
        shuffle=shuffle,
        save_hf_checkpoint=save_hf_checkpoint,
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
