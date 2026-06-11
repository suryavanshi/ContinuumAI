# OPD Learnings

Notes from building and smoke-testing on-policy distillation with Verl and
Modal.

## Goal

Run on-policy distillation for a small Qwen3.5 student using Verl on Modal,
without modifying the Verl codebase. The smoke run used:

- student: `Qwen/Qwen3.5-0.8B`
- teacher: `Qwen/Qwen3.5-4B`
- dataset: tiny GSM8K parquet split
- loss: `distillation.distillation_loss.loss_mode=k1`
- policy-gradient distillation:
  `distillation.distillation_loss.use_policy_gradient=True`

## Working Script

The launcher is:

```bash
python3 -m modal run --detach OPD/modal_verl_qwen35_opd.py \
  --train-rows 2 \
  --val-rows 1 \
  --total-training-steps 1
```

The script now avoids machine-specific paths. It defaults to a sibling `../verl`
checkout locally, and falls back safely to `/opt/verl` when Modal imports the
mounted script inside the container.

## What Worked

- Modal can run the newer Verl image `verlai/verl:vllm020.dev1` on H100.
- The ARM-specific `verlai/verl:vllm020.aarch64.dev1` tag is not appropriate
  for Modal's x86_64 GPU workers.
- Mounting the local Verl checkout into `/opt/verl` and installing it with
  `pip install --no-deps -e .` works and keeps framework code unchanged.
- A Modal Volume works well for HF cache, datasets, checkpoints, and logs.
- Verl accepted the OPD config with:
  - `distillation.enabled=True`
  - `distillation.teacher_models.teacher_model.model_path=<teacher>`
  - `distillation.distillation_loss.loss_mode=k1`
  - `distillation.distillation_loss.use_policy_gradient=True`
- `use_policy_gradient=True` is required for `k1`; `k1` with
  `use_policy_gradient=False` is invalid.
- The smoke run got through image build, GSM8K preprocessing, Ray startup,
  config validation, student model load, FSDP setup, and teacher vLLM startup.

## Modal Image Lessons

The older local Modal CLI injected runtime dependencies that conflicted with
the Verl/vLLM image. The most visible failure was:

```text
ImportError: cannot import name 'AliasChoices' from 'pydantic'
```

The fix was to use the newer Modal client:

```bash
python3 -m modal --version
```

and then restore the training stack dependencies after Modal's runtime layer:

```text
pydantic>=2.12,<3
fastapi[standard]>=0.115.0
aiohttp>=3.13.3
typer>=0.20.0
rich>=13.7.1
importlib-metadata>=6,<8.8
```

This fixed the pydantic v1/v2 mismatch and let Qwen3.5 model loading proceed.

## Modal Resource Lessons

- `ephemeral_disk=300_000` MiB was rejected by Modal. The script now uses
  `ephemeral_disk=600_000`.
- Detached runs are useful for long image/model startup, but always check app
  state with:

```bash
python3 -m modal app list --json
python3 -m modal app logs <app-id> --tail 500
```

- Local launcher processes can be terminated after the remote Modal app is
  detached. The remote app continues running.

## Qwen3.5 / vLLM Notes

The logs include warnings such as:

```text
Unrecognized keys in `rope_parameters` for 'rope_type'='default'
```

and:

```text
Only support config type ... but got qwen3_5. MFU will always be zero.
```

These were warnings, not immediate blockers. Student and teacher model loading
continued.

## Current Caveats

- The OPD smoke reached teacher vLLM startup, but a full completed training-step
  log was not captured before moving on.
- Hugging Face access was unauthenticated, so downloads emitted rate-limit
  warnings. For serious runs, use a Modal Secret for HF auth.
- Qwen3.5 support is still noisy in the current container stack. Expect
  compatibility warnings from Transformers/vLLM until the stack stabilizes.

## Next Improvements

- Add a smaller default teacher for faster smoke tests, then scale to larger
  teachers after the one-step path reliably completes.
- Add a Modal Secret hook for `HF_TOKEN`.
- Add a small run-status helper that fetches app logs and extracts:
  config validation, dataset length, model load, vLLM readiness, first step,
  reward metrics, and distillation metrics.
- Keep custom datasets as explicit Verl-compatible parquet inputs unless the
  dataset has a simple prompt/answer schema.
