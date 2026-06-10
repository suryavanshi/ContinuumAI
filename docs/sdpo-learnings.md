# SDPO Learnings

Notes from implementing a Self-Distilled Policy Optimization style run with
Modal and upstream Verl, using the original SDPO repository only as reference.

## Goal

Create a Modal + Verl launcher for SDPO without modifying the Verl codebase.
The reference SDPO implementation is a Verl fork with internal actor-loss and
self-distillation changes. This repo instead builds a bridge from public Verl
hooks.

## What Exact SDPO Adds In The Reference Repo

The reference implementation adds framework-level support for:

- `actor_rollout_ref.actor.policy_loss.loss_mode=sdpo`
- `actor_rollout_ref.actor.self_distillation.*`
- self-distillation batches built inside the trainer
- feedback-conditioned reprompting inside the trainer loop
- full-logit or top-k self-distillation
- EMA or trust-region self-teacher updates
- metrics such as feedback-used fraction and self-distillation mask coverage

Those features require changes inside Verl's trainer and actor code. Since the
goal here is not to modify Verl, the current script does not claim to reproduce
that exact internal loss path.

## Implemented Bridge

The new launcher is:

```bash
python3 -m modal run --detach scripts/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --total-training-steps 1
```

It composes upstream Verl features:

- prepares feedback-conditioned prompts into Verl-compatible parquet
- uses a custom reward function written into the Modal volume at runtime
- runs GRPO with Verl's existing reward manager
- enables Verl's OPD teacher loop
- defaults the self-teacher model to the same base model as the student
- sets `distillation.distillation_loss.use_task_rewards=True` so task reward and
  dense distillation are combined

This is best understood as an SDPO bridge: feedback-conditioned self-teacher
distillation through OPD, not the SDPO fork's custom `loss_mode=sdpo`.

## Working Script

The launcher is:

[scripts/modal_verl_sdpo.py](../scripts/modal_verl_sdpo.py)

The default smoke run uses:

- model: `Qwen/Qwen3.5-0.8B`
- self-teacher: same model by default
- dataset: `openai/gsm8k`, converted into `gsm8k_sdpo`
- image:
  `modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.1-modelscope1.35.4-swift4.1.3`
- loss: `distillation.distillation_loss.loss_mode=k1`
- reward: custom answer extractor and verifier

By default the launcher starts from the ModelScope fallback image and clones
public Verl from `VERL_GIT_REF` into `/opt/verl`. Set `VERL_UPLOAD_LOCAL=1` only
when intentionally testing the sibling local Verl checkout. Public image probes
showed that `verlai/verl:vllm020.dev1` does not contain an importable `verl`
package by itself, so no-local-upload smoke tests should use the public clone
path:

```bash
VERL_GIT_REF=v0.8.0 \
python3 -m modal run scripts/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --total-training-steps 1
```

That tests the base image plus a public Verl release without uploading the local
`/Users/.../verl` checkout.

For rich-feedback datasets, pass:

```bash
python3 -m modal run --detach scripts/modal_verl_sdpo.py \
  --dataset code_feedback_smoke \
  --hf-dataset your_org/your_feedback_dataset \
  --prompt-column prompt \
  --answer-column answer \
  --feedback-column feedback \
  --previous-attempt-column failed_solution \
  --data-source code_feedback \
  --ability code \
  --train-rows 8 \
  --val-rows 4 \
  --total-training-steps 1
```

## Modal Test Results

The first Modal launch exposed a real portability bug:

```text
IndexError: 2
DEFAULT_VERL_LOCAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "verl"
```

Locally, the script path has enough parents. Inside Modal, the mounted script was
imported as `/root/modal_verl_sdpo.py`, so `parents[2]` did not exist.

The fix was to use a guarded default:

```python
THIS_FILE = pathlib.Path(__file__).resolve()
DEFAULT_VERL_LOCAL_DIR = (
    THIS_FILE.parents[2] / "verl"
    if len(THIS_FILE.parents) > 2
    else pathlib.Path("/opt/verl")
)
```

The same fix was applied to the OPD launcher.

After the fix, the Modal SDPO smoke run:

- prepared the HF dataset
- wrote tiny train/test parquet files
- started the GPU training function
- confirmed CUDA availability
- launched Verl `main_ppo`
- started Ray
- validated the Verl config
- loaded the Qwen3.5 student
- initialized FSDP
- started reward workers
- started the vLLM self-teacher server
- loaded the self-teacher checkpoint

The active run at the time of testing was:

```text
ap-lNwBGNx21kqYd4dypXj1Kw
```

It had one active task after self-teacher checkpoint load.

Follow-up evidence from the committed Modal volume log:

```bash
python3 -m modal volume get continuum-verl-sdpo-cache \
  /logs/gsm8k_sdpo-sdpo-20260602T073330Z.log \
  /tmp/gsm8k_sdpo-20260602T073330Z.log --env main

python3 scripts/inspect_sdpo_log.py \
  /tmp/gsm8k_sdpo-20260602T073330Z.log --expect-steps 1
```

The inspector reports `status=incomplete`: the log configured
`Total training steps: 1`, loaded the self-teacher checkpoint, and then ended
without rollout/training metrics, `global_step_1`, checkpoint saves, or a
traceback. A read-only Modal volume check also found no `/checkpoints`
directory. Treat this prior run as not proven to have completed one training
step.

A later read-only app inventory also found `ap-hka0FbzvRrWT3wfXePkFTp`, a
detached `continuum-verl-sdpo` app still retrying stale pre-fix code. Its logs
fail during import with the original `parents[2]` `IndexError`, before dataset
prep or training. Do not count it as training evidence.

The no-workspace-upload ModelScope smoke later passed the one-step gate with a
fresh timestamped checkpoint directory:

```text
/logs/shell_sdpo_smoke_plain_20260604T075811Z.log
/checkpoints/shell_sdpo_smoke_plain_20260604T075811Z/global_step_1
```

The downloaded log classifies as:

```text
status=completed-with-warning
configured_steps=1
expected_steps=1
step_evidence=step:1 ... training/global_step:1 ...
```

The warning is a Ray DataLoader worker shutdown traceback. The process exited
0, the `step:1` metrics line appears after the warning, and the checkpoint
exists, so this proves the image plus public-Verl path can complete one tiny
training step.

## Useful Config Details

The bridge relies on:

```text
reward.custom_reward_function.path=/cache/runtime/sdpo_reward.py
reward.custom_reward_function.name=compute_score
distillation.enabled=True
distillation.teacher_key=data_source
distillation.teacher_models.teacher_model.key=<data_source>
distillation.teacher_models.teacher_model.model_path=<self_teacher_model>
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.use_task_rewards=True
```

The important difference from the earlier OPD smoke is
`use_task_rewards=True`. That keeps the verifier reward in the policy update
instead of using only the distillation objective.

## Dataset Lessons

For upstream Verl, the simplest data path is still parquet with:

- `data_source`
- `prompt` as chat messages
- `ability`
- `reward_model.ground_truth`
- `extra_info`

The SDPO bridge adds these fields into `extra_info` when available:

- `feedback_raw`
- `previous_attempt`
- `original_prompt`
- `sdpo_bridge=True`

If a dataset already includes failed attempts and feedback, the script can
reprompt directly. If it only has prompt/answer pairs, the run behaves more like
self-teacher OPD plus verifier reward than rich-feedback SDPO.

## Current Caveats

- This is not exact SDPO from the reference fork. Exact SDPO needs internal Verl
  support for `loss_mode=sdpo`, self-distillation masks, reprompted batches, and
  EMA/trust-region teacher updates.
- The committed log and volume artifacts now confirm this remains incomplete:
  no training metrics, no `global_step_1`, and no checkpoint directory were
  present for the prior run.
- A later attempt to rerun the attached one-step smoke was blocked before launch
  because uploading the local ContinuumAI launcher plus sibling Verl checkout to
  Modal was considered an external-disclosure risk. The launcher now supports
  `VERL_UPLOAD_LOCAL=0` for a narrower public-Verl smoke path.
- Public `modal shell --image verlai/verl:vllm020.dev1` inspection found no
  importable `verl` package and no `/opt/verl` directory. The ModelScope
  fallback image also lacked `verl`, and Modal shell's runtime dependency
  injection downgraded packages such as `pydantic`, `fastapi`, and `aiohttp`.
  Use the patched launcher image build, not raw `modal shell`, for training
  smoke tests.
- The successful shell smoke still prints a Ray DataLoader worker shutdown
  traceback after the step completes. Keep classifying logs with
  `scripts/inspect_sdpo_log.py` and watch whether that warning disappears in
  the packaged Modal launcher.
- The default GSM8K smoke does not include real environment feedback. It is
  useful for infrastructure validation, not for demonstrating the full benefit
  of rich-feedback SDPO.
- Unauthenticated HF downloads produce warnings and may be slower.
- Qwen3.5 still emits vLLM/rope metadata warnings in the current stack.

## Next Improvements

- Add a run mode that starts from a rich-feedback dataset, not plain GSM8K.
- Add a first-pass generator that creates failed attempts and feedback, then
  writes SDPO-ready parquet before training.
- Add Modal Sandboxes for code tasks so feedback can come from tests and runtime
  errors.
- Add a metrics scraper for reward score, feedback availability, distillation
  loss, and first training step.
- Consider a small local plugin layer that registers a custom distillation loss
  at runtime if Verl exposes enough extension hooks, still without editing the
  upstream Verl repo.
