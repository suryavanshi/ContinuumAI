from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import modal


APP_NAME = "continuum-verl-sdpo"
DEFAULT_VERL_IMAGE_TAG = (
    "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:"
    "ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.1-modelscope1.35.4-swift4.1.3"
)
VERL_IMAGE_TAG = os.environ.get("VERL_IMAGE_TAG", DEFAULT_VERL_IMAGE_TAG)
VERL_UPLOAD_LOCAL = os.environ.get("VERL_UPLOAD_LOCAL", "0").lower() not in {
    "0",
    "false",
    "no",
}
VERL_GIT_REF = os.environ.get("VERL_GIT_REF", "v0.8.0")
THIS_FILE = pathlib.Path(__file__).resolve()
DEFAULT_VERL_LOCAL_DIR = THIS_FILE.parents[2] / "verl" if len(THIS_FILE.parents) > 2 else pathlib.Path("/opt/verl")
VERL_LOCAL_DIR = pathlib.Path(
    os.environ.get("VERL_LOCAL_DIR", str(DEFAULT_VERL_LOCAL_DIR))
).expanduser().resolve()
REMOTE_VERL_DIR = "/opt/verl"
CACHE_DIR = pathlib.Path("/cache")
DATA_ROOT = CACHE_DIR / "data"
MODEL_ROOT = CACHE_DIR / "models" / "hf"
RUNTIME_DIR = CACHE_DIR / "runtime"
LOG_DIR = CACHE_DIR / "logs"
REMOTE_SDPO_SHELL = "/opt/continuum/run_qwen_sdpo_mopd_fsdp.sh"
REMOTE_HARVEY_SDPO_SHELL = "/opt/continuum/run_qwen36_harvey_sdpo_fsdp.sh"
REMOTE_HARVEY_PIPELINE = "/opt/continuum/harvey_lab_sdpo_pipeline.py"
REMOTE_HARVEY_LABS_DIR = "/opt/harvey-labs"
HARVEY_LABS_LOCAL_DIR = pathlib.Path(
    os.environ.get("HARVEY_LABS_LOCAL_DIR", "/Users/kb/Documents/proj/git_projs/harvey-labs")
).expanduser()
HARVEY_JUDGE_SECRET_NAME = os.environ.get("HARVEY_JUDGE_SECRET_NAME", "")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("continuum-verl-sdpo-cache", create_if_missing=True)
harvey_judge_secrets = (
    [modal.Secret.from_name(HARVEY_JUDGE_SECRET_NAME)] if HARVEY_JUDGE_SECRET_NAME else []
)

image = modal.Image.from_registry(VERL_IMAGE_TAG)

if VERL_UPLOAD_LOCAL:
    image = (
        image.add_local_dir(
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
            f"cd {REMOTE_VERL_DIR} && python -m pip install -r requirements.txt && python -m pip install --no-deps -e .",
        )
    )
else:
    image = image.run_commands(
        "git --version || (apt-get update && apt-get install -y git)",
        (
            f"rm -rf {REMOTE_VERL_DIR} && "
            f"git clone --depth 1 --branch {shlex.quote(VERL_GIT_REF)} "
            f"https://github.com/verl-project/verl {REMOTE_VERL_DIR} && "
            f"cd {REMOTE_VERL_DIR} && python -m pip install -r requirements.txt && python -m pip install --no-deps -e ."
        ),
    )

image = (
    image.pip_install(
        "hf-transfer",
        "huggingface_hub>=0.36.0",
        "pydantic>=2.12,<3",
        "fastapi[standard]>=0.115.0",
        "aiohttp>=3.13.3",
        "typer>=0.20.0",
        "rich>=13.7.1",
        "importlib-metadata>=6,<8.8",
        "tilelang",
        "anthropic>=0.40.0",
        "openai>=1.0.0",
        "google-genai",
        "mistralai",
        "pandas",
        "pdfplumber",
        "markitdown",
        "python-docx",
        "openpyxl",
        "python-pptx",
    )
    .run_commands("apt-get update && apt-get install -y pandoc && rm -rf /var/lib/apt/lists/*")
    .add_local_file(
        THIS_FILE.parent / "run_qwen_sdpo_mopd_fsdp.sh",
        remote_path=REMOTE_SDPO_SHELL,
        copy=True,
    )
    .add_local_file(
        THIS_FILE.parent / "run_qwen36_harvey_sdpo_fsdp.sh",
        remote_path=REMOTE_HARVEY_SDPO_SHELL,
        copy=True,
    )
    .add_local_file(
        THIS_FILE.parent / "harvey_lab_sdpo_pipeline.py",
        remote_path=REMOTE_HARVEY_PIPELINE,
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

if HARVEY_LABS_LOCAL_DIR.exists():
    image = image.add_local_dir(
        HARVEY_LABS_LOCAL_DIR,
        remote_path=REMOTE_HARVEY_LABS_DIR,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "results",
        ],
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
    if str(data_source or "") == "harvey/lab":
        text = _normalize(solution_str)
        terms = extra_info.get("reward_terms", []) or []
        if isinstance(terms, str):
            terms = [terms]
        cleaned_terms = [_normalize(term) for term in terms if _normalize(term)]
        hits = sum(1 for term in cleaned_terms if term in text)
        coverage = hits / max(len(cleaned_terms), 1)
        length_bonus = 0.2 if len(str(solution_str or "").split()) >= 80 else 0.0
        refusal_penalty = 0.3 if "cannot provide legal advice" in text or "i can't" in text else 0.0
        score = max(0.0, min(1.0, coverage + length_bonus - refusal_penalty))
        return {
            "score": score,
            "harvey_reward_terms": len(cleaned_terms),
            "harvey_reward_hits": hits,
            "sdpo_feedback_available": bool(extra_info.get("feedback_raw", "")),
        }
    extracted = _extract_answer(solution_str)
    score = 1.0 if _normalize(extracted) == _normalize(ground_truth) else 0.0
    feedback = extra_info.get("feedback_raw", "")
    return {
        "score": score,
        "extracted_answer": extracted,
        "feedback_raw": feedback,
        "sdpo_feedback_available": bool(feedback),
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


def _is_local_model_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("file:")


def _model_cache_path(model: str) -> pathlib.Path:
    return MODEL_ROOT / _safe_name(model)


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
    hf_train_file: str | None,
    hf_val_file: str | None,
    train_split: str,
    val_split: str,
    dataset_format: str,
    conversations_column: str,
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
import json
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

out_dir = pathlib.Path({str(out_dir)!r})
out_dir.mkdir(parents=True, exist_ok=True)

hf_dataset = {hf_dataset!r}
hf_config = {hf_config!r}
hf_train_file = {hf_train_file!r}
hf_val_file = {hf_val_file!r}
data_files = {{}}
if hf_train_file:
    data_files["train"] = hf_train_file
if hf_val_file:
    data_files["test"] = hf_val_file
dataset_format = {dataset_format!r}
if data_files and dataset_format == "sharegpt":
    def load_raw_json(repo_id, filename):
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as handle:
            if filename.endswith(".jsonl"):
                return [json.loads(line) for line in handle if line.strip()]
            loaded_json = json.load(handle)
        if isinstance(loaded_json, list):
            return loaded_json
        if isinstance(loaded_json, dict):
            for value in loaded_json.values():
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported raw JSON dataset shape in {{filename}}")
    loaded = {{
        "train": load_raw_json(hf_dataset, hf_train_file),
        "test": load_raw_json(hf_dataset, hf_val_file or hf_train_file),
    }}
elif data_files:
    loaded = datasets.load_dataset(hf_dataset, hf_config, data_files=data_files) if hf_config else datasets.load_dataset(hf_dataset, data_files=data_files)
else:
    loaded = datasets.load_dataset(hf_dataset, hf_config) if hf_config else datasets.load_dataset(hf_dataset)
train = loaded[{train_split!r}]
val = loaded[{val_split!r}]

def limit_rows(split, rows):
    rows = int(rows)
    if rows <= 0:
        return split
    if hasattr(split, "select"):
        return split.select(range(min(rows, len(split))))
    return split[: min(rows, len(split))]

train = limit_rows(train, {int(train_rows)})
val = limit_rows(val, {int(val_rows)})

prompt_column = {prompt_column!r}
answer_column = {answer_column!r}
feedback_column = {feedback_column!r}
previous_attempt_column = {previous_attempt_column!r}
conversations_column = {conversations_column!r}
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

def normalize_turn(turn):
    if not isinstance(turn, dict):
        return "", str(turn)
    role = str(turn.get("from") or turn.get("role") or turn.get("speaker") or "").lower()
    value = turn.get("value", turn.get("content", turn.get("message", "")))
    return role, "" if value is None else str(value)

def conversation_to_fields(example):
    turns = example.get(conversations_column) or []
    if isinstance(turns, str):
        return turns, turns, ""
    normalized = [normalize_turn(turn) for turn in turns]
    prompt_parts = []
    answer = ""
    for role, value in normalized:
        if role in ("human", "user") and not prompt_parts:
            prompt_parts.append(value)
        if role in ("gpt", "assistant", "agent"):
            answer = value
    if not prompt_parts and normalized:
        prompt_parts.append(normalized[0][1])
    trace_lines = []
    for role, value in normalized:
        label = role or "turn"
        trace_lines.append(f"{{label}}: {{value}}")
    return "\\n\\n".join(prompt_parts), answer, "\\n".join(trace_lines)

def make_record(example, idx, split_name):
    if dataset_format == "sharegpt":
        prompt_raw, answer_raw, trajectory = conversation_to_fields(example)
        feedback_raw = read_column(example, feedback_column, static_feedback) or static_feedback
        if not feedback_raw:
            feedback_raw = "Successful reference trajectory:\\n" + trajectory
        previous_attempt = read_column(example, previous_attempt_column)
        if not previous_attempt:
            previous_attempt = trajectory
    else:
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

def write_plain_parquet(split, split_name, path):
    records = [make_record(example, idx, split_name) for idx, example in enumerate(split)]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)
    return len(records)

train_count = write_plain_parquet(train, "train", out_dir / "train.parquet")
val_count = write_plain_parquet(val, "test", out_dir / "test.parquet")
print("prepared", train_count, "train rows and", val_count, "val rows from", hf_dataset)
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
    hf_train_file: str | None = None,
    hf_val_file: str | None = None,
    train_split: str = "train",
    val_split: str = "test",
    dataset_format: str = "qa",
    conversations_column: str = "conversations",
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
        hf_train_file=hf_train_file,
        hf_val_file=hf_val_file,
        train_split=train_split,
        val_split=val_split,
        dataset_format=dataset_format,
        conversations_column=conversations_column,
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
    gpu="H200:5",
    volumes={"/cache": cache_volume},
    secrets=harvey_judge_secrets,
    timeout=8 * 60 * 60,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def prepare_harvey_sdpo_dataset(
    dataset: str = "harvey_lab_sdpo",
    model: str = "/cache/models/Qwen_Qwen3_6-35B-A3B",
    train_tasks: int = 8,
    eval_preview_tasks: int = 2,
    seed: int = 17,
    doc_chars: int = 1200,
    max_prompt_chars: int = 9000,
    tensor_parallel_size: int = 4,
    max_model_len: int = 12288,
    max_new_tokens: int = 1024,
    judge_model: str = "claude-sonnet-4-6",
    judge_parallel: int = 2,
    skip_generation: bool = False,
    skip_judge_eval: bool = False,
) -> str:
    """Generate Harvey LAB attempts, run Harvey eval, and emit SDPO parquet."""
    bench_root = pathlib.Path(REMOTE_HARVEY_LABS_DIR)
    if not bench_root.exists():
        raise RuntimeError(
            f"Harvey LAB repo was not uploaded to {REMOTE_HARVEY_LABS_DIR}. "
            "Set HARVEY_LABS_LOCAL_DIR before running Modal if the local clone moved."
        )

    data_dir = _dataset_dir(dataset)
    cmd = [
        sys.executable,
        REMOTE_HARVEY_PIPELINE,
        "--bench-root",
        str(bench_root),
        "--out-dir",
        str(data_dir),
        "--model",
        model,
        "--train-tasks",
        str(train_tasks),
        "--eval-preview-tasks",
        str(eval_preview_tasks),
        "--seed",
        str(seed),
        "--doc-chars",
        str(doc_chars),
        "--max-prompt-chars",
        str(max_prompt_chars),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-new-tokens",
        str(max_new_tokens),
        "--judge-model",
        judge_model,
        "--judge-parallel",
        str(judge_parallel),
    ]
    if skip_generation:
        cmd.append("--skip-generation")
    if skip_judge_eval:
        cmd.append("--skip-judge-eval")
    _run(cmd, cwd="/cache", env=_base_env())
    cache_volume.commit()
    manifest_path = data_dir / "manifest.json"
    return manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else str(data_dir)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    timeout=3600,
    startup_timeout=1800,
)
def download_model(model: str, force: bool = False) -> str:
    """Download a HF model once into the Modal volume and return the local path."""
    if _is_local_model_path(model):
        return model

    from huggingface_hub import snapshot_download

    target_dir = _model_cache_path(model)
    marker = target_dir / ".snapshot_complete"
    if marker.exists() and not force:
        print(f"Reusing cached model {model}: {target_dir}", flush=True)
        return str(target_dir)

    if force and target_dir.exists():
        print(f"Refreshing cached model {model}: {target_dir}", flush=True)
        shutil.rmtree(target_dir)
    else:
        print(f"Downloading model {model}: {target_dir}", flush=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=not force,
    )
    marker.write_text(datetime.now(timezone.utc).isoformat())
    cache_volume.commit()
    print(f"Model ready: {target_dir}", flush=True)
    return str(target_dir)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    timeout=12 * 60 * 60,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def download_model_hf_cli(model: str, local_dir: str | None = None, force: bool = False) -> str:
    """Download a HF model with the HF CLI into the Modal volume."""
    if _is_local_model_path(model):
        return model

    target_dir = pathlib.Path(local_dir).expanduser() if local_dir else _model_cache_path(model)
    marker = target_dir / ".hf_cli_download_complete"
    if marker.exists() and not force:
        print(f"Reusing HF CLI cached model {model}: {target_dir}", flush=True)
        return str(target_dir)

    if force and target_dir.exists():
        print(f"Refreshing HF CLI cached model {model}: {target_dir}", flush=True)
        shutil.rmtree(target_dir)
    else:
        print(f"Downloading model with HF CLI {model}: {target_dir}", flush=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if not cli:
        raise RuntimeError("Neither `hf` nor `huggingface-cli` is available in the image.")

    cmd = [
        cli,
        "download",
        model,
        "--local-dir",
        str(target_dir),
    ]
    if cli.endswith("huggingface-cli"):
        cmd.append("--resume-download")
    _run(cmd, cwd="/cache", env=env)
    marker.write_text(datetime.now(timezone.utc).isoformat())
    cache_volume.commit()
    print(f"HF CLI model ready: {target_dir}", flush=True)
    return str(target_dir)


def _build_verl_args(
    *,
    model: str,
    teacher_model: str,
    teacher_key: str,
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    reward_path: pathlib.Path,
    run_name: str,
    total_training_steps: int,
    total_epochs: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    n_gpus_per_node: int,
    tensor_model_parallel_size: int,
    rollout_expert_parallel_size: int,
    distillation_n_gpus_per_node: int | None,
    distillation_tensor_model_parallel_size: int | None,
    distillation_expert_parallel_size: int | None,
    rollout_gpu_memory_utilization: float,
    distillation_gpu_memory_utilization: float,
    enable_activation_offload: bool,
    enable_param_offload: bool,
    enable_optimizer_offload: bool,
    fsdp_model_dtype: str,
    ppo_clip_ratio: float,
    distillation_clip_ratio: float,
    distillation_topk: int,
    distillation_loss_coef: float,
    distillation_loss_max_clamp: float,
    save_hf_checkpoint: bool,
) -> list[str]:
    max_num_tokens = max_prompt_length + max_response_length + 1
    ppo_max_token_len_per_gpu = max_num_tokens * max(train_batch_size, 1)
    teacher_n_gpus_per_node = distillation_n_gpus_per_node or n_gpus_per_node
    teacher_tensor_model_parallel_size = (
        distillation_tensor_model_parallel_size or tensor_model_parallel_size
    )
    teacher_expert_parallel_size = distillation_expert_parallel_size or rollout_expert_parallel_size

    return [
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
        f"actor_rollout_ref.model.enable_activation_offload={enable_activation_offload}",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.clip_ratio={ppo_clip_ratio}",
        f"actor_rollout_ref.actor.clip_ratio_low={ppo_clip_ratio}",
        f"actor_rollout_ref.actor.clip_ratio_high={ppo_clip_ratio}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={enable_param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={enable_optimizer_offload}",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={fsdp_model_dtype}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.expert_parallel_size={rollout_expert_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.max_model_len={max_num_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={max_num_seqs}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens}",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "reward.custom_reward_function.name=compute_score",
        f"reward.custom_reward_function.path={reward_path}",
        "trainer.balance_batch=False",
        "trainer.logger=console",
        "trainer.project_name=continuum_verl_sdpo",
        f"trainer.experiment_name={run_name}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        f"trainer.save_freq={total_training_steps if save_hf_checkpoint else -1}",
        "trainer.test_freq=-1",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.total_training_steps={total_training_steps}",
        "trainer.max_actor_ckpt_to_keep=1",
        f"trainer.default_local_dir={CACHE_DIR / 'checkpoints' / run_name}",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model']",
        "actor_rollout_ref.actor.checkpoint.load_contents=['model']",
        "distillation.enabled=True",
        f"distillation.n_gpus_per_node={teacher_n_gpus_per_node}",
        "distillation.nnodes=1",
        "distillation.teacher_key=data_source",
        f"distillation.teacher_models.teacher_model.key={teacher_key}",
        f"distillation.teacher_models.teacher_model.model_path={teacher_model}",
        "distillation.teacher_models.teacher_model.num_replicas=1",
        "distillation.teacher_models.teacher_model.inference.name=vllm",
        f"distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size={teacher_tensor_model_parallel_size}",
        f"distillation.teacher_models.teacher_model.inference.expert_parallel_size={teacher_expert_parallel_size}",
        f"distillation.teacher_models.teacher_model.inference.gpu_memory_utilization={distillation_gpu_memory_utilization}",
        f"distillation.teacher_models.teacher_model.inference.max_model_len={max_num_tokens}",
        f"distillation.teacher_models.teacher_model.inference.max_num_seqs={max_num_seqs}",
        f"distillation.teacher_models.teacher_model.inference.max_num_batched_tokens={max_num_batched_tokens}",
        "distillation.distillation_loss.loss_mode=k1",
        f"distillation.distillation_loss.topk={distillation_topk}",
        "distillation.distillation_loss.use_task_rewards=True",
        "distillation.distillation_loss.use_policy_gradient=True",
        f"distillation.distillation_loss.clip_ratio={distillation_clip_ratio}",
        f"distillation.distillation_loss.clip_ratio_low={distillation_clip_ratio}",
        f"distillation.distillation_loss.clip_ratio_high={distillation_clip_ratio}",
        f"distillation.distillation_loss.distillation_loss_coef={distillation_loss_coef}",
        f"distillation.distillation_loss.loss_max_clamp={distillation_loss_max_clamp}",
        "distillation.distillation_loss.log_prob_min_clamp=-10.0",
    ]


def _train_sdpo_impl(
    model: str = "Qwen/Qwen3.5-0.8B",
    self_teacher_model: str | None = None,
    dataset: str = "gsm8k_sdpo",
    train_files: str | None = None,
    val_files: str | None = None,
    teacher_key: str = "openai/gsm8k",
    total_training_steps: int = 1,
    total_epochs: int = 1,
    train_batch_size: int = 2,
    ppo_mini_batch_size: int = 2,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 2048,
    n_gpus_per_node: int = 1,
    tensor_model_parallel_size: int = 1,
    rollout_expert_parallel_size: int = 1,
    distillation_n_gpus_per_node: int | None = None,
    distillation_tensor_model_parallel_size: int | None = None,
    distillation_expert_parallel_size: int | None = None,
    rollout_gpu_memory_utilization: float = 0.35,
    distillation_gpu_memory_utilization: float = 0.45,
    enable_activation_offload: bool = False,
    enable_param_offload: bool = True,
    enable_optimizer_offload: bool = True,
    fsdp_model_dtype: str = "fp32",
    ppo_clip_ratio: float = 0.2,
    distillation_clip_ratio: float = 0.2,
    distillation_topk: int = 32,
    distillation_loss_coef: float = 1.0,
    distillation_loss_max_clamp: float = 10.0,
    save_hf_checkpoint: bool = True,
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
    tmp_log_path = pathlib.Path("/tmp") / f"{run_name}.log"

    env = _base_env()
    env["VERL_LOGGING_LEVEL"] = "INFO"

    print("SDPO bridge configuration:", flush=True)
    print("  mode=feedback-conditioned self-teacher via Verl OPD", flush=True)
    print(f"  image={VERL_IMAGE_TAG}", flush=True)
    print(f"  upload_local_verl={VERL_UPLOAD_LOCAL}", flush=True)
    if VERL_UPLOAD_LOCAL:
        print(f"  verl_local_dir={VERL_LOCAL_DIR}", flush=True)
    else:
        print(f"  verl_git_ref={VERL_GIT_REF}", flush=True)
    print(f"  model={model}", flush=True)
    print(f"  self_teacher_model={teacher_model}", flush=True)
    print(f"  dataset={dataset}", flush=True)
    print(f"  train_files={train_path}", flush=True)
    print(f"  val_files={val_path}", flush=True)
    print(f"  total_training_steps={total_training_steps}", flush=True)
    print(f"  train_batch_size={train_batch_size}", flush=True)
    print(f"  ppo_mini_batch_size={ppo_mini_batch_size}", flush=True)
    print(f"  max_num_seqs={max_num_seqs}", flush=True)
    print(f"  max_num_batched_tokens={max_num_batched_tokens}", flush=True)
    print(f"  n_gpus_per_node={n_gpus_per_node}", flush=True)
    print(f"  tensor_model_parallel_size={tensor_model_parallel_size}", flush=True)
    print(f"  rollout_expert_parallel_size={rollout_expert_parallel_size}", flush=True)
    print(f"  rollout_gpu_memory_utilization={rollout_gpu_memory_utilization}", flush=True)
    print(
        f"  distillation_n_gpus_per_node={distillation_n_gpus_per_node or n_gpus_per_node}",
        flush=True,
    )
    print(
        "  distillation_tensor_model_parallel_size="
        f"{distillation_tensor_model_parallel_size or tensor_model_parallel_size}",
        flush=True,
    )
    print(
        "  distillation_expert_parallel_size="
        f"{distillation_expert_parallel_size or rollout_expert_parallel_size}",
        flush=True,
    )
    print(
        f"  distillation_gpu_memory_utilization={distillation_gpu_memory_utilization}",
        flush=True,
    )
    print(f"  enable_activation_offload={enable_activation_offload}", flush=True)
    print(f"  enable_param_offload={enable_param_offload}", flush=True)
    print(f"  enable_optimizer_offload={enable_optimizer_offload}", flush=True)
    print(f"  fsdp_model_dtype={fsdp_model_dtype}", flush=True)
    print(f"  ppo_clip_ratio={ppo_clip_ratio}", flush=True)
    print("  distillation_loss.loss_mode=k1", flush=True)
    print("  distillation_loss.use_task_rewards=True", flush=True)
    print(f"  distillation_clip_ratio={distillation_clip_ratio}", flush=True)
    print(f"  distillation_loss.loss_max_clamp={distillation_loss_max_clamp}", flush=True)
    print(f"  save_hf_checkpoint={save_hf_checkpoint}", flush=True)
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

    args = _build_verl_args(
        model=model,
        teacher_model=teacher_model,
        teacher_key=teacher_key,
        train_path=train_path,
        val_path=val_path,
        reward_path=reward_path,
        run_name=run_name,
        total_training_steps=total_training_steps,
        total_epochs=total_epochs,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        n_gpus_per_node=n_gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rollout_expert_parallel_size=rollout_expert_parallel_size,
        distillation_n_gpus_per_node=distillation_n_gpus_per_node,
        distillation_tensor_model_parallel_size=distillation_tensor_model_parallel_size,
        distillation_expert_parallel_size=distillation_expert_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        distillation_gpu_memory_utilization=distillation_gpu_memory_utilization,
        enable_activation_offload=enable_activation_offload,
        enable_param_offload=enable_param_offload,
        enable_optimizer_offload=enable_optimizer_offload,
        fsdp_model_dtype=fsdp_model_dtype,
        ppo_clip_ratio=ppo_clip_ratio,
        distillation_clip_ratio=distillation_clip_ratio,
        distillation_topk=distillation_topk,
        distillation_loss_coef=distillation_loss_coef,
        distillation_loss_max_clamp=distillation_loss_max_clamp,
        save_hf_checkpoint=save_hf_checkpoint,
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
        f"{shlex.quote(sys.executable)} -m verl.trainer.main_ppo "
        f"{' '.join(shlex.quote(arg) for arg in args)} 2>&1 | tee {shlex.quote(str(tmp_log_path))}; "
        "status=${PIPESTATUS[0]}; "
        'exit "${status}"'
    )
    _run(["bash", "-lc", cmd], env=env)
    cache_volume.commit()
    return str(log_path)


@app.function(
    image=image,
    gpu="H200:5",
    volumes={"/cache": cache_volume},
    timeout=7200,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def train_sdpo(**kwargs) -> str:
    """Run the standard multi-GPU SDPO executor."""
    return _train_sdpo_impl(**kwargs)


@app.function(
    image=image,
    gpu="H100:2",
    volumes={"/cache": cache_volume},
    timeout=7200,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def train_sdpo_smoke(**kwargs) -> str:
    """Run the 0.8B smoke gate with one actor GPU and one teacher GPU."""
    return _train_sdpo_impl(
        enable_param_offload=False,
        enable_optimizer_offload=False,
        **kwargs,
    )


def _build_shell_env(
    *,
    model: str,
    teacher_model: str,
    teacher_key: str,
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    reward_path: pathlib.Path,
    run_name: str,
    total_training_steps: int,
    total_epochs: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    n_gpus_per_node: int,
    tensor_model_parallel_size: int,
    rollout_expert_parallel_size: int,
    distillation_n_gpus_per_node: int | None,
    distillation_tensor_model_parallel_size: int | None,
    distillation_expert_parallel_size: int | None,
    rollout_gpu_memory_utilization: float,
    distillation_gpu_memory_utilization: float,
    enable_activation_offload: bool,
    fsdp_model_dtype: str,
    ppo_clip_ratio: float,
    distillation_clip_ratio: float,
    distillation_topk: int,
    distillation_loss_coef: float,
    distillation_loss_max_clamp: float,
    save_hf_checkpoint: bool,
) -> dict[str, str]:
    teacher_tp = distillation_tensor_model_parallel_size or 1
    teacher_world_size = distillation_n_gpus_per_node or teacher_tp
    save_freq = total_training_steps if save_hf_checkpoint else -1
    return {
        "STUDENT_MODEL": model,
        "TEACHER_MODEL": teacher_model,
        "TEACHER_KEY": teacher_key,
        "TRAIN_FILE": str(train_path),
        "VAL_FILE": str(val_path),
        "REWARD_FN_PATH": str(reward_path),
        "NNODES": "1",
        "NGPUS_PER_NODE": str(n_gpus_per_node),
        "TRAIN_SIZE_BSZ": str(train_batch_size),
        "PPO_MINI_BATCH_SIZE": str(ppo_mini_batch_size),
        "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": "1",
        "MAX_PROMPT_LENGTH": str(max_prompt_length),
        "MAX_RESPONSE_LENGTH": str(max_response_length),
        "MAX_NUM_BATCHED_TOKENS": str(max_num_batched_tokens),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "FSDP_MODEL_DTYPE": fsdp_model_dtype,
        "PARAM_OFFLOAD": "True",
        "OPTIMIZER_OFFLOAD": "True",
        "OPTIMIZER_OVERRIDE_CONFIG": "{foreach: False}",
        "ENABLE_ACTIVATION_OFFLOAD": str(bool(enable_activation_offload)),
        "USE_TORCH_COMPILE": "False",
        "ROLLOUT_TP": str(tensor_model_parallel_size),
        "ROLLOUT_EP": str(rollout_expert_parallel_size),
        "ROLLOUT_GPU_MEM_UTIL": str(rollout_gpu_memory_utilization),
        "TEACHER_NNODES": "1",
        "TEACHER_NUM_REPLICAS": "1",
        "TEACHER_TP": str(teacher_tp),
        "TEACHER_EP": str(distillation_expert_parallel_size or 1),
        "TEACHER_WORLD_SIZE": str(teacher_world_size),
        "TEACHER_GPU_MEM_UTIL": str(distillation_gpu_memory_utilization),
        "DISTILLATION_LOSS_MODE": "k1",
        "DISTILLATION_TOPK": str(distillation_topk),
        "DISTILLATION_LOSS_COEF": str(distillation_loss_coef),
        "DISTILLATION_CLIP_RATIO": str(distillation_clip_ratio),
        "DISTILLATION_LOSS_MAX_CLAMP": str(distillation_loss_max_clamp),
        "USE_TASK_REWARDS": "True",
        "USE_POLICY_GRADIENT": "True",
        "PROJECT_NAME": "continuum_verl_sdpo",
        "EXPERIMENT_NAME": run_name,
        "DEFAULT_LOCAL_DIR": str(CACHE_DIR / "checkpoints" / run_name),
        "LOGGER": "console",
        "BALANCE_BATCH": "False",
        "SHUFFLE": "False",
        "SAVE_FREQ": str(save_freq),
        "TEST_FREQ": "-1",
        "TOTAL_EPOCHS": str(total_epochs),
        "TOTAL_TRAINING_STEPS": str(total_training_steps),
        "VAL_BEFORE_TRAIN": "False",
        "TRUST_REMOTE_CODE": "True",
    }


@app.function(
    image=image,
    gpu="H200:5",
    volumes={"/cache": cache_volume},
    timeout=7200,
    startup_timeout=1800,
    ephemeral_disk=600_000,
)
def train_sdpo_shell(
    model: str = "/cache/models/Qwen_Qwen3_6-35B-A3B",
    self_teacher_model: str | None = None,
    dataset: str = "eto_sdpo",
    train_files: str | None = None,
    val_files: str | None = None,
    teacher_key: str = "agent-eto/eto-sft-trajectory",
    total_training_steps: int = 5,
    total_epochs: int = 1,
    train_batch_size: int = 2,
    ppo_mini_batch_size: int = 2,
    max_prompt_length: int = 2048,
    max_response_length: int = 128,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 2048,
    n_gpus_per_node: int = 4,
    tensor_model_parallel_size: int = 4,
    rollout_expert_parallel_size: int = 4,
    distillation_n_gpus_per_node: int | None = 1,
    distillation_tensor_model_parallel_size: int | None = 1,
    distillation_expert_parallel_size: int | None = 1,
    rollout_gpu_memory_utilization: float = 0.35,
    distillation_gpu_memory_utilization: float = 0.9,
    enable_activation_offload: bool = True,
    fsdp_model_dtype: str = "bfloat16",
    ppo_clip_ratio: float = 0.2,
    distillation_clip_ratio: float = 0.2,
    distillation_topk: int = 32,
    distillation_loss_coef: float = 1.0,
    distillation_loss_max_clamp: float = 10.0,
    save_hf_checkpoint: bool = False,
    remote_shell_script: str = REMOTE_SDPO_SHELL,
) -> str:
    """Run the OPD-style SDPO shell recipe on Modal."""
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
    run_name = f"{_safe_name(dataset)}-sdpo-shell-{stamp}"
    log_path = LOG_DIR / f"{run_name}.log"
    tmp_log_path = pathlib.Path("/tmp") / f"{run_name}.log"

    env = _base_env()
    env["VERL_LOGGING_LEVEL"] = "INFO"
    shell_env = _build_shell_env(
        model=model,
        teacher_model=teacher_model,
        teacher_key=teacher_key,
        train_path=train_path,
        val_path=val_path,
        reward_path=reward_path,
        run_name=run_name,
        total_training_steps=total_training_steps,
        total_epochs=total_epochs,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        n_gpus_per_node=n_gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rollout_expert_parallel_size=rollout_expert_parallel_size,
        distillation_n_gpus_per_node=distillation_n_gpus_per_node,
        distillation_tensor_model_parallel_size=distillation_tensor_model_parallel_size,
        distillation_expert_parallel_size=distillation_expert_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        distillation_gpu_memory_utilization=distillation_gpu_memory_utilization,
        enable_activation_offload=enable_activation_offload,
        fsdp_model_dtype=fsdp_model_dtype,
        ppo_clip_ratio=ppo_clip_ratio,
        distillation_clip_ratio=distillation_clip_ratio,
        distillation_topk=distillation_topk,
        distillation_loss_coef=distillation_loss_coef,
        distillation_loss_max_clamp=distillation_loss_max_clamp,
        save_hf_checkpoint=save_hf_checkpoint,
    )
    env.update(shell_env)

    print("SDPO shell configuration:", flush=True)
    for key in sorted(shell_env):
        print(f"  {key}={env[key]}", flush=True)
    print(f"  shell_script={remote_shell_script}", flush=True)
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
        f"bash {shlex.quote(remote_shell_script)} 2>&1 | tee {shlex.quote(str(tmp_log_path))}; "
        "status=${PIPESTATUS[0]}; "
        'exit "${status}"'
    )
    _run(["bash", "-lc", cmd], env=env)
    cache_volume.commit()
    return str(log_path)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-0.8B",
    self_teacher_model: Optional[str] = None,
    dataset: str = "gsm8k_sdpo",
    train_files: Optional[str] = None,
    val_files: Optional[str] = None,
    teacher_key: str = "openai/gsm8k",
    hf_dataset: str = "openai/gsm8k",
    hf_config: Optional[str] = "main",
    hf_train_file: Optional[str] = None,
    hf_val_file: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "test",
    dataset_format: str = "qa",
    conversations_column: str = "conversations",
    prompt_column: str = "question",
    answer_column: str = "answer",
    feedback_column: Optional[str] = None,
    previous_attempt_column: Optional[str] = None,
    data_source: str = "openai/gsm8k",
    ability: str = "math",
    train_rows: int = 8,
    val_rows: int = 4,
    instruction_suffix: str = 'Let us think step by step and output the final answer after "####".',
    static_feedback: str = "",
    total_training_steps: int = 1,
    total_epochs: int = 1,
    train_batch_size: int = 2,
    ppo_mini_batch_size: int = 2,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 2048,
    n_gpus_per_node: int = 1,
    tensor_model_parallel_size: int = 1,
    rollout_expert_parallel_size: int = 1,
    distillation_n_gpus_per_node: Optional[int] = None,
    distillation_tensor_model_parallel_size: Optional[int] = None,
    distillation_expert_parallel_size: Optional[int] = None,
    rollout_gpu_memory_utilization: float = 0.35,
    distillation_gpu_memory_utilization: float = 0.45,
    enable_activation_offload: bool = False,
    fsdp_model_dtype: str = "fp32",
    ppo_clip_ratio: float = 0.2,
    distillation_clip_ratio: float = 0.2,
    distillation_loss_max_clamp: float = 10.0,
    save_hf_checkpoint: bool = True,
    skip_model_prefetch: bool = False,
    force_model_download: bool = False,
    skip_prepare: bool = False,
    run_shell_script: bool = False,
    smoke_gpu: bool = False,
) -> None:
    print(f"Modal app: {APP_NAME}")
    student_model_path = model
    teacher_model_path = self_teacher_model
    if not skip_model_prefetch:
        student_model_path = download_model.remote(model=model, force=force_model_download)
        print(f"Student model path: {student_model_path}")
        if self_teacher_model:
            if self_teacher_model == model:
                teacher_model_path = student_model_path
            else:
                teacher_model_path = download_model.remote(
                    model=self_teacher_model,
                    force=force_model_download,
                )
        else:
            teacher_model_path = student_model_path
        print(f"Self-teacher model path: {teacher_model_path}")

    if not skip_prepare:
        prepared = prepare_sdpo_dataset.remote(
            dataset=dataset,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            hf_train_file=hf_train_file,
            hf_val_file=hf_val_file,
            train_split=train_split,
            val_split=val_split,
            dataset_format=dataset_format,
            conversations_column=conversations_column,
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
    train_func = train_sdpo_shell if run_shell_script else (train_sdpo_smoke if smoke_gpu else train_sdpo)
    log_path = train_func.remote(
        model=student_model_path,
        self_teacher_model=teacher_model_path,
        dataset=dataset,
        train_files=train_files,
        val_files=val_files,
        teacher_key=teacher_key,
        total_training_steps=total_training_steps,
        total_epochs=total_epochs,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        n_gpus_per_node=n_gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rollout_expert_parallel_size=rollout_expert_parallel_size,
        distillation_n_gpus_per_node=distillation_n_gpus_per_node,
        distillation_tensor_model_parallel_size=distillation_tensor_model_parallel_size,
        distillation_expert_parallel_size=distillation_expert_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        distillation_gpu_memory_utilization=distillation_gpu_memory_utilization,
        enable_activation_offload=enable_activation_offload,
        # Qwen3.5's recurrent-attention kernels are not safe in the FP32 path
        # used by this Torch/vLLM image. The tiny H100 smoke preset therefore
        # follows the repository's production recipes and uses BF16.
        fsdp_model_dtype="bfloat16" if smoke_gpu else fsdp_model_dtype,
        ppo_clip_ratio=ppo_clip_ratio,
        distillation_clip_ratio=distillation_clip_ratio,
        distillation_loss_max_clamp=distillation_loss_max_clamp,
        save_hf_checkpoint=save_hf_checkpoint,
    )
    print(f"Training log committed to Modal volume: {log_path}")


@app.local_entrypoint()
def harvey_main(
    model: str = "/cache/models/Qwen_Qwen3_6-35B-A3B",
    self_teacher_model: Optional[str] = None,
    dataset: str = "harvey_lab_sdpo",
    train_tasks: int = 8,
    eval_preview_tasks: int = 2,
    seed: int = 17,
    doc_chars: int = 1200,
    max_prompt_chars: int = 9000,
    prepare_tensor_parallel_size: int = 4,
    prepare_max_model_len: int = 12288,
    prepare_max_new_tokens: int = 1024,
    judge_model: str = "claude-sonnet-4-6",
    judge_parallel: int = 2,
    skip_generation: bool = False,
    skip_judge_eval: bool = False,
    skip_prepare: bool = False,
    run_training: bool = True,
    skip_model_prefetch: bool = True,
    force_model_download: bool = False,
    total_training_steps: int = 1,
    total_epochs: int = 1,
    train_batch_size: int = 4,
    ppo_mini_batch_size: int = 4,
    max_prompt_length: int = 4096,
    max_response_length: int = 1024,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int = 6144,
    n_gpus_per_node: int = 4,
    tensor_model_parallel_size: int = 4,
    rollout_expert_parallel_size: int = 4,
    distillation_n_gpus_per_node: Optional[int] = 1,
    distillation_tensor_model_parallel_size: Optional[int] = 1,
    distillation_expert_parallel_size: Optional[int] = 1,
    rollout_gpu_memory_utilization: float = 0.35,
    distillation_gpu_memory_utilization: float = 0.9,
    enable_activation_offload: bool = True,
    fsdp_model_dtype: str = "bfloat16",
    ppo_clip_ratio: float = 0.2,
    distillation_clip_ratio: float = 0.2,
    distillation_loss_max_clamp: float = 10.0,
    save_hf_checkpoint: bool = False,
    spawn_training: bool = True,
) -> None:
    """Prepare Harvey LAB SDPO data and train Qwen3.6-35B-A3B on Modal."""
    print(f"Modal app: {APP_NAME}")
    student_model_path = model
    teacher_model_path = self_teacher_model
    if not skip_model_prefetch:
        student_model_path = download_model_hf_cli.remote(
            model=model,
            force=force_model_download,
        )
        print(f"Student model path: {student_model_path}")
        if self_teacher_model:
            teacher_model_path = (
                student_model_path
                if self_teacher_model == model
                else download_model_hf_cli.remote(model=self_teacher_model, force=force_model_download)
            )
        else:
            teacher_model_path = student_model_path
        print(f"Self-teacher model path: {teacher_model_path}")

    if not teacher_model_path:
        teacher_model_path = student_model_path

    if not skip_prepare:
        manifest = prepare_harvey_sdpo_dataset.remote(
            dataset=dataset,
            model=student_model_path,
            train_tasks=train_tasks,
            eval_preview_tasks=eval_preview_tasks,
            seed=seed,
            doc_chars=doc_chars,
            max_prompt_chars=max_prompt_chars,
            tensor_parallel_size=prepare_tensor_parallel_size,
            max_model_len=prepare_max_model_len,
            max_new_tokens=prepare_max_new_tokens,
            judge_model=judge_model,
            judge_parallel=judge_parallel,
            skip_generation=skip_generation,
            skip_judge_eval=skip_judge_eval,
        )
        print(f"Prepared Harvey data manifest:\n{manifest}")

    if not run_training:
        print("Skipping training because --run-training=false.")
        return

    train_kwargs = dict(
        model=student_model_path,
        self_teacher_model=teacher_model_path,
        dataset=dataset,
        teacher_key="harvey/lab",
        total_training_steps=total_training_steps,
        total_epochs=total_epochs,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        n_gpus_per_node=n_gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        rollout_expert_parallel_size=rollout_expert_parallel_size,
        distillation_n_gpus_per_node=distillation_n_gpus_per_node,
        distillation_tensor_model_parallel_size=distillation_tensor_model_parallel_size,
        distillation_expert_parallel_size=distillation_expert_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        distillation_gpu_memory_utilization=distillation_gpu_memory_utilization,
        enable_activation_offload=enable_activation_offload,
        fsdp_model_dtype=fsdp_model_dtype,
        ppo_clip_ratio=ppo_clip_ratio,
        distillation_clip_ratio=distillation_clip_ratio,
        distillation_loss_max_clamp=distillation_loss_max_clamp,
        save_hf_checkpoint=save_hf_checkpoint,
        remote_shell_script=REMOTE_HARVEY_SDPO_SHELL,
    )
    if spawn_training:
        call = train_sdpo_shell.spawn(**train_kwargs)
        print(f"Spawned Harvey SDPO training call: {call.object_id or call.local_uuid}")
        dashboard_url = call.get_dashboard_url()
        if dashboard_url:
            print(f"Training call dashboard: {dashboard_url}")
    else:
        log_path = train_sdpo_shell.remote(**train_kwargs)
        print(f"Harvey SDPO training log committed to Modal volume: {log_path}")
