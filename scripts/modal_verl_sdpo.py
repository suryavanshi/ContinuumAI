from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone

import modal


APP_NAME = "continuum-verl-sdpo"
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
cache_volume = modal.Volume.from_name("continuum-verl-sdpo-cache", create_if_missing=True)

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


SDPO_REWARD_CODE = r'''
import re


def _normalize(value):
    value = str(value or "").strip()
    value = value.replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


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
    letter = re.findall(r"\b([A-D])\b", text.upper())
    if letter:
        return letter[-1]
    return text.strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    extra_info = extra_info or {}
    extracted = _extract_answer(solution_str)
    score = 1.0 if _normalize(extracted) == _normalize(ground_truth) else 0.0
    feedback = extra_info.get("feedback_raw", "")
    return {
        "score": score,
        "extracted_answer": extracted,
        "feedback_raw": feedback,
        "sdpo_feedback_available": bool(feedback),
        "data_source": data_source,
    }
'''


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


def _write_reward_module() -> pathlib.Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    reward_path = RUNTIME_DIR / "sdpo_reward.py"
    reward_path.write_text(SDPO_REWARD_CODE)
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
def prepare_sdpo_dataset(
    dataset: str = "gsm8k_sdpo",
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
    instruction_suffix: str = 'Let us think step by step and output the final answer after "####".',
    static_feedback: str = "",
    reprompt_template: str = (
        "{prompt}\\n\\n"
        "Previous attempt:\\n{solution}\\n\\n"
        "Environment feedback:\\n{feedback}\\n\\n"
        "Use the feedback to produce a corrected solution."
    ),
) -> str:
    """Prepare feedback-conditioned, Verl-compatible parquet data for SDPO-style training."""
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


@app.function(
    image=image,
    gpu="H100:2",
    volumes={"/cache": cache_volume},
    timeout=7200,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def train_sdpo(
    model: str = "Qwen/Qwen3.5-0.8B",
    self_teacher_model: str | None = None,
    dataset: str = "gsm8k_sdpo",
    train_files: str | None = None,
    val_files: str | None = None,
    teacher_key: str = "openai/gsm8k",
    total_training_steps: int = 1,
    train_batch_size: int = 2,
    ppo_mini_batch_size: int = 2,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    distillation_topk: int = 32,
    distillation_loss_coef: float = 1.0,
) -> str:
    """Run SDPO-style training by composing Verl GRPO, custom rewards, and OPD."""
    default_train_path, default_val_path = _dataset_paths(dataset)
    train_path = pathlib.Path(train_files) if train_files else default_train_path
    val_path = pathlib.Path(val_files) if val_files else default_val_path
    teacher_model = self_teacher_model or model

    if not train_path.exists() or not val_path.exists():
        raise RuntimeError(
            "Missing train/validation parquet files. Run prepare_sdpo_dataset first, "
            "or pass --skip-prepare --train-files <path> --val-files <path>."
        )

    reward_path = _write_reward_module()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{_safe_name(dataset)}-sdpo-{stamp}"
    log_path = LOG_DIR / f"{run_name}.log"
    max_num_tokens = max_prompt_length + max_response_length + 1
    ppo_max_token_len_per_gpu = max_num_tokens * max(train_batch_size, 1)

    env = _base_env()
    env["VERL_LOGGING_LEVEL"] = "INFO"

    print("SDPO bridge configuration:", flush=True)
    print("  mode=feedback-conditioned self-teacher via Verl OPD", flush=True)
    print(f"  model={model}", flush=True)
    print(f"  self_teacher_model={teacher_model}", flush=True)
    print(f"  dataset={dataset}", flush=True)
    print(f"  train_files={train_path}", flush=True)
    print(f"  val_files={val_path}", flush=True)
    print("  distillation_loss.loss_mode=k1", flush=True)
    print("  distillation_loss.use_task_rewards=True", flush=True)
    print(f"  reward_path={reward_path}", flush=True)
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
        f"actor_rollout_ref.model.path={model}",
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
        "reward.custom_reward_function.name=compute_score",
        f"reward.custom_reward_function.path={reward_path}",
        "trainer.balance_batch=False",
        "trainer.logger=console",
        "trainer.project_name=continuum_verl_sdpo",
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
        "distillation.teacher_key=data_source",
        f"distillation.teacher_models.teacher_model.key={teacher_key}",
        f"distillation.teacher_models.teacher_model.model_path={teacher_model}",
        "distillation.teacher_models.teacher_model.num_replicas=1",
        "distillation.teacher_models.teacher_model.inference.name=vllm",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.45",
        f"distillation.teacher_models.teacher_model.inference.max_model_len={max_num_tokens}",
        "distillation.distillation_loss.loss_mode=k1",
        f"distillation.distillation_loss.topk={distillation_topk}",
        "distillation.distillation_loss.use_task_rewards=True",
        "distillation.distillation_loss.use_policy_gradient=True",
        f"distillation.distillation_loss.distillation_loss_coef={distillation_loss_coef}",
        "distillation.distillation_loss.loss_max_clamp=10.0",
        "distillation.distillation_loss.log_prob_min_clamp=-10.0",
    ]

    cmd = (
        "set -euo pipefail; "
        f"{shlex.quote(sys.executable)} -m verl.trainer.main_ppo "
        f"{' '.join(shlex.quote(arg) for arg in args)} 2>&1 | tee {shlex.quote(str(log_path))}"
    )
    _run(["bash", "-lc", cmd], env=env)
    cache_volume.commit()
    return str(log_path)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-0.8B",
    self_teacher_model: str | None = None,
    dataset: str = "gsm8k_sdpo",
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
    instruction_suffix: str = 'Let us think step by step and output the final answer after "####".',
    static_feedback: str = "",
    total_training_steps: int = 1,
    skip_prepare: bool = False,
) -> None:
    print(f"Modal app: {APP_NAME}")
    if not skip_prepare:
        prepared = prepare_sdpo_dataset.remote(
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
    log_path = train_sdpo.remote(
        model=model,
        self_teacher_model=self_teacher_model,
        dataset=dataset,
        train_files=train_files,
        val_files=val_files,
        teacher_key=teacher_key,
        total_training_steps=total_training_steps,
    )
    print(f"Training log committed to Modal volume: {log_path}")
