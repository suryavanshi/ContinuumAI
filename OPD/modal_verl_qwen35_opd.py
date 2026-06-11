from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone

import modal


APP_NAME = "continuum-verl-opd"
VERL_IMAGE_TAG = os.environ.get("VERL_IMAGE_TAG", "verlai/verl:vllm020.dev1")
THIS_FILE = pathlib.Path(__file__).resolve()
DEFAULT_VERL_LOCAL_DIR = THIS_FILE.parents[2] / "verl" if len(THIS_FILE.parents) > 2 else pathlib.Path("/opt/verl")
VERL_LOCAL_DIR = pathlib.Path(
    os.environ.get("VERL_LOCAL_DIR", str(DEFAULT_VERL_LOCAL_DIR))
).expanduser().resolve()
REMOTE_VERL_DIR = "/opt/verl"
CACHE_DIR = pathlib.Path("/cache")
DATA_ROOT = CACHE_DIR / "data"
LOG_DIR = CACHE_DIR / "logs"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("continuum-verl-opd-cache", create_if_missing=True)

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


def _run(cmd: list[str], cwd: str = REMOTE_VERL_DIR, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
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


def _prepare_hf_dataset_script(
    *,
    out_dir: pathlib.Path,
    hf_dataset: str,
    hf_config: str | None,
    train_split: str,
    val_split: str,
    prompt_column: str,
    answer_column: str,
    data_source: str,
    ability: str,
    instruction_suffix: str,
    train_rows: int,
    val_rows: int,
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

def make_map_fn(split_name):
    def process_fn(example, idx):
        prompt_raw = str(example.get({prompt_column!r}, ""))
        answer_raw = str(example.get({answer_column!r}, "")) if {answer_column!r} else ""
        suffix = {instruction_suffix!r}
        content = prompt_raw if not suffix else prompt_raw + " " + suffix
        return {{
            "data_source": {data_source!r},
            "prompt": [{{"role": "user", "content": content}}],
            "ability": {ability!r},
            "reward_model": {{"style": "rule", "ground_truth": answer_raw}},
            "extra_info": {{
                "split": split_name,
                "index": idx,
                "answer": answer_raw,
                "prompt": prompt_raw,
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
def prepare_dataset(
    dataset: str = "gsm8k",
    train_rows: int = 8,
    val_rows: int = 4,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    train_split: str = "train",
    val_split: str = "test",
    prompt_column: str = "question",
    answer_column: str = "answer",
    data_source: str | None = None,
    ability: str = "general",
    instruction_suffix: str = "",
) -> str:
    """Prepare a tiny dataset split for a Modal smoke test."""
    data_dir = _dataset_dir(dataset)

    if hf_dataset:
        script = _prepare_hf_dataset_script(
            out_dir=data_dir,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            train_split=train_split,
            val_split=val_split,
            prompt_column=prompt_column,
            answer_column=answer_column,
            data_source=data_source or hf_dataset,
            ability=ability,
            instruction_suffix=instruction_suffix,
            train_rows=train_rows,
            val_rows=val_rows,
        )
        _run([sys.executable, "-c", script], env=_base_env())
        cache_volume.commit()
        train_path, val_path = _dataset_paths(dataset)
        return f"{train_path} and {val_path}"

    if dataset != "gsm8k":
        raise ValueError(
            "Only dataset='gsm8k' has built-in preprocessing without --hf-dataset. "
            "For other datasets, either pass --hf-dataset with prompt/answer "
            "columns or pass --skip-prepare --train-files <path> --val-files <path>."
        )

    full_dir = CACHE_DIR / "data" / "gsm8k_full"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (full_dir / "train.parquet").exists() or not (full_dir / "test.parquet").exists():
        _run(
            [
                sys.executable,
                "examples/data_preprocess/gsm8k.py",
                "--local_save_dir",
                str(full_dir),
            ],
            env=_base_env(),
        )

    subset_script = f"""
import pathlib
import pandas as pd

full_dir = pathlib.Path({str(full_dir)!r})
out_dir = pathlib.Path({str(data_dir)!r})
out_dir.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(full_dir / "train.parquet").head({int(train_rows)})
test = pd.read_parquet(full_dir / "test.parquet").head({int(val_rows)})
train.to_parquet(out_dir / "train.parquet")
test.to_parquet(out_dir / "test.parquet")
print("prepared", len(train), "train rows and", len(test), "val rows")
"""
    _run([sys.executable, "-c", subset_script], env=_base_env())
    cache_volume.commit()
    train_path, val_path = _dataset_paths(dataset)
    return f"{train_path} and {val_path}"


@app.function(
    image=image,
    gpu="H100:2",
    volumes={"/cache": cache_volume},
    timeout=7200,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def train_opd(
    student_model: str = "Qwen/Qwen3.5-0.8B",
    teacher_model: str = "Qwen/Qwen3.5-4B",
    dataset: str = "gsm8k",
    train_files: str | None = None,
    val_files: str | None = None,
    teacher_key: str = "openai/gsm8k",
    total_training_steps: int = 1,
    train_batch_size: int = 2,
    ppo_mini_batch_size: int = 2,
    max_prompt_length: int = 256,
    max_response_length: int = 128,
) -> str:
    """Launch a tiny Verl on-policy distillation run using k1 loss."""
    default_train_path, default_val_path = _dataset_paths(dataset)
    train_path = pathlib.Path(train_files) if train_files else default_train_path
    val_path = pathlib.Path(val_files) if val_files else default_val_path

    if not train_path.exists() or not val_path.exists():
        raise RuntimeError(
            "Missing train/validation parquet files. Run prepare_dataset first for "
            "built-in datasets, or pass --skip-prepare --train-files <path> "
            "--val-files <path> for already prepared Verl-compatible data."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{_safe_name(dataset)}-opd-k1-{stamp}"
    log_path = LOG_DIR / f"{run_name}.log"
    max_num_tokens = max_prompt_length + max_response_length + 1
    ppo_max_token_len_per_gpu = max_num_tokens * max(train_batch_size, 1)

    env = _base_env()
    env["VERL_LOGGING_LEVEL"] = "INFO"

    print("Smoke run configuration:", flush=True)
    print(f"  dataset={dataset}", flush=True)
    print(f"  train_files={train_path}", flush=True)
    print(f"  val_files={val_path}", flush=True)
    print(f"  student_model={student_model}", flush=True)
    print(f"  teacher_model={teacher_model}", flush=True)
    print("  distillation_loss.loss_mode=k1", flush=True)
    print("  distillation_loss.use_policy_gradient=True", flush=True)
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

    args = [
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
        f"actor_rollout_ref.model.path={student_model}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.max_model_len={max_num_tokens}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "trainer.balance_batch=False",
        "trainer.logger=console",
        "trainer.project_name=continuum_verl_opd",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_training_steps}",
        f"trainer.default_local_dir={CACHE_DIR / 'checkpoints' / run_name}",
        "distillation.enabled=True",
        "distillation.n_gpus_per_node=1",
        "distillation.nnodes=1",
        f"distillation.teacher_models.teacher_model.key={teacher_key}",
        f"distillation.teacher_models.teacher_model.model_path={teacher_model}",
        "distillation.teacher_models.teacher_model.num_replicas=1",
        "distillation.teacher_models.teacher_model.inference.name=vllm",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.45",
        f"distillation.teacher_models.teacher_model.inference.max_model_len={max_num_tokens}",
        "distillation.distillation_loss.loss_mode=k1",
        "distillation.distillation_loss.topk=32",
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.use_policy_gradient=True",
        "distillation.distillation_loss.loss_max_clamp=10.0",
        "distillation.distillation_loss.log_prob_min_clamp=-10.0",
    ]

    cmd = (
        f"set -euo pipefail; "
        f"{shlex.quote(sys.executable)} -m verl.trainer.main_ppo "
        f"{' '.join(shlex.quote(arg) for arg in args)} 2>&1 | tee {shlex.quote(str(log_path))}"
    )
    _run(["bash", "-lc", cmd], env=env)
    cache_volume.commit()
    return str(log_path)


@app.local_entrypoint()
def main(
    student_model: str = "Qwen/Qwen3.5-0.8B",
    teacher_model: str = "Qwen/Qwen3.5-4B",
    dataset: str = "gsm8k",
    train_files: str | None = None,
    val_files: str | None = None,
    teacher_key: str = "openai/gsm8k",
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    train_split: str = "train",
    val_split: str = "test",
    prompt_column: str = "question",
    answer_column: str = "answer",
    data_source: str | None = None,
    ability: str = "general",
    instruction_suffix: str = "",
    train_rows: int = 8,
    val_rows: int = 4,
    total_training_steps: int = 1,
    skip_prepare: bool = False,
) -> None:
    print(f"Modal app: {APP_NAME}")
    if not skip_prepare:
        prepared = prepare_dataset.remote(
            dataset=dataset,
            train_rows=train_rows,
            val_rows=val_rows,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            train_split=train_split,
            val_split=val_split,
            prompt_column=prompt_column,
            answer_column=answer_column,
            data_source=data_source,
            ability=ability,
            instruction_suffix=instruction_suffix,
        )
        print(f"Prepared data: {prepared}")
    log_path = train_opd.remote(
        student_model=student_model,
        teacher_model=teacher_model,
        dataset=dataset,
        train_files=train_files,
        val_files=val_files,
        teacher_key=teacher_key,
        total_training_steps=total_training_steps,
    )
    print(f"Training log committed to Modal volume: {log_path}")
