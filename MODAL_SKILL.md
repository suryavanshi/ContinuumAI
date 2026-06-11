---
name: modal-training
description: Launch, monitor, and debug Modal GPU training jobs without storing secrets in repository files.
---

# Modal Training Skill

Use this runbook when launching Modal jobs, checking app state, inspecting logs,
or debugging GPU training workflows.

## Secret Hygiene

- Never commit Modal token IDs, Modal token secrets, Hugging Face tokens, W&B
  API keys, SSH keys, or bearer tokens.
- If a local note contains a `modal token set ...` command, treat it only as a
  reminder that auth may already exist. Do not copy the command or values.
- Prefer local Modal auth state and Modal Secrets. In code, reference secrets by
  name only, for example `modal.Secret.from_name("huggingface-secret")`.
- Keep tracked docs focused on workflow, not credentials.

## Local CLI

Prefer the upgraded Python module invocation:

```bash
python3 -m modal --version
```

An older Modal CLI may also live outside `PATH`:

```bash
/Users/kb/Library/Python/3.9/bin/modal --version
```

Use the older full path only if the newer module invocation is unavailable. For
real training, prefer detached runs:

```bash
python3 -m modal run --detach path/to/app.py::main
```

For short smoke tests, attached runs are useful because failures stream directly
to the terminal.

## Launch Pattern

1. Prove prerequisites first: Modal auth, image import, dataset prep, and model
   access.
2. Split CPU/data prep from GPU training when possible.
3. Use a Modal Volume for Hugging Face cache, datasets, logs, and checkpoints.
4. Start with the smallest dataset and shortest run that exercises the full
   stack.
5. Keep topology fixed while changing one model, loss, or hyperparameter at a
   time.

## Monitoring

Useful commands:

```bash
python3 -m modal app list --json
python3 -m modal app logs <app-id> --tail 500
python3 -m modal app logs <app-id> --search "error"
python3 -m modal app logs <app-id> --search "loss"
python3 -m modal container list
python3 -m modal volume ls <volume-name> / --env main
```

Large jobs may sit in scheduling for several minutes. Empty logs usually mean
the image is still building or workers have not started.

## Debugging

- If a GPU job fails immediately, verify the data files exist in the mounted
  volume before relaunching.
- If logs are noisy, filter for `error`, `exception`, `CUDA`, `OOM`, `NCCL`,
  `ray`, `loss`, and `global_step`.
- If a job appears stuck, compare app state, logs, container state, and committed
  volume files. Live container files can differ from committed volume state.
- Stop stale apps before relaunching a replacement that reuses the same cache
  volume.

## Verl OPD Notes

- Verl on-policy distillation uses:
  - `distillation.enabled=True`
  - `distillation.teacher_models.*.model_path=<teacher>`
  - `distillation.distillation_loss.loss_mode=k1`
  - `distillation.distillation_loss.use_policy_gradient=True`
- `k1` is a policy-gradient OPD loss. Do not set
  `use_policy_gradient=False` with `loss_mode=k1`.
- Use same-family teacher/student models first to avoid tokenizer and vocab
  alignment problems.
- For a smoke run, use tiny batches, short prompt/response lengths,
  `trainer.total_training_steps=1`, `trainer.logger=console`, and W&B disabled.

## Verl Image Selection

- The Modal launchers default to `verlai/verl:vllm020.dev1`, which is an
  x86_64/amd64-friendly Verl image for Modal GPU workers.
- Override the image per run with `VERL_IMAGE_TAG` when testing a newer
  compatible Verl image:

  ```bash
  VERL_IMAGE_TAG=verlai/verl:<compatible-amd64-tag> \
    python3 -m modal run --detach OPD/modal_verl_qwen35_opd.py
  ```

- Avoid ARM-only tags such as `verlai/verl:vllm020.aarch64.dev1` on normal
  Modal GPU workers. Use `.aarch64.*` images only when the target runtime is
  explicitly ARM, otherwise the image can fail before Verl or Ray starts.
