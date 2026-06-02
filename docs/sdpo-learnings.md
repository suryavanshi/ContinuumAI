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
- loss: `distillation.distillation_loss.loss_mode=k1`
- reward: custom answer extractor and verifier

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
- The smoke had reached self-teacher checkpoint load, but a completed one-step
  metrics line was not captured before this note was written.
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
