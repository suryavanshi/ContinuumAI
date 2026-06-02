# ContinuumAI

[View the on-policy distillation infographic](https://suryavanshi.github.io/ContinuumAI/)

ContinuumAI is a small orchestration workspace for post-training open models on
elastic GPU infrastructure. The current focus is continual learning with
on-policy distillation, on-policy self-distillation, and RLVR-style recipes for
models such as Qwen3.5, Qwen3.x, Kimi-K2, and future open-weight reasoning
models.

The repository is intentionally thin: it should coordinate frameworks like
Verl, Slime, and Training Gym without modifying those framework repos.

## What This Repo Contains

- [POST_TRAINING_PLAN.md](POST_TRAINING_PLAN.md): the overall implementation
  plan for open-model post-training on Modal.
- [MODAL_SKILL.md](MODAL_SKILL.md): a secret-free Modal training runbook for
  launching, monitoring, and debugging GPU jobs.
- [docs/opd-learnings.md](docs/opd-learnings.md): practical notes from the
  Modal + Verl OPD smoke tests.
- [docs/sdpo-learnings.md](docs/sdpo-learnings.md): practical notes from the
  Modal + Verl SDPO bridge smoke tests.
- [scripts/modal_verl_qwen35_opd.py](scripts/modal_verl_qwen35_opd.py): a Modal
  + Verl smoke launcher for Qwen3.5 on-policy distillation with `k1` loss.
- [scripts/modal_verl_sdpo.py](scripts/modal_verl_sdpo.py): a Modal + Verl
  SDPO bridge that uses feedback-conditioned prompts, a custom reward function,
  and Verl's existing on-policy distillation path without modifying Verl.
- [docs/index.html](docs/index.html): a GitHub Pages-ready infographic for
  on-policy distillation and self-distillation as continual learning.

## Core Idea

Continual learning for LLMs needs more than another fine-tuning script. A useful
system has to keep a model improving on new domains while protecting previous
skills and keeping GPU time busy.

ContinuumAI treats post-training as a loop:

1. Sample from the current policy on live or curated tasks.
2. Score the trajectory with verifiers, tools, tests, judges, or environments.
3. Add dense guidance from a teacher, a frozen checkpoint, or a feedback-aware
   self-teacher.
4. Update the policy with RL and distillation losses.
5. Evaluate drift, regressions, cost, and task improvement before continuing.

## Training Patterns

**On-policy distillation**  
The student samples trajectories from its current policy. A stronger teacher
scores those exact trajectories with token-level log probabilities. The trainer
combines dense teacher guidance, for example `k1`, with optional scalar rewards.

**On-policy self-distillation**  
The model generates an attempt, receives environment feedback, then produces a
feedback-conditioned target distribution for its own trajectory. This is useful
when tests, tool traces, judge comments, or verifier explanations provide
learning signal beyond a pass/fail reward.

**Recovery distillation for continual learning**  
After domain updates, a frozen earlier checkpoint or stronger general teacher
can pull the model back toward broad instruction-following and reasoning
behavior, reducing regression while the model absorbs new skills.

## References

- [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
  from Thinking Machines.
- [Self-Distilled Policy Optimization](https://github.com/lasgroup/SDPO) from
  `lasgroup/SDPO`.

## Modal + Verl Smoke Run

The current smoke launcher uses a local copy of Verl mounted into a Modal image
and runs one tiny Qwen3.5 OPD job on GSM8K. By default it looks for a sibling
`../verl` checkout; set `VERL_LOCAL_DIR` if your Verl repo lives elsewhere.

```bash
python3 -m modal run --detach scripts/modal_verl_qwen35_opd.py \
  --train-rows 2 \
  --val-rows 1 \
  --total-training-steps 1
```

The script defaults to:

- student: `Qwen/Qwen3.5-0.8B`
- teacher: `Qwen/Qwen3.5-4B`
- image: `verlai/verl:vllm020.dev1`
- loss: `distillation.distillation_loss.loss_mode=k1`
- policy-gradient OPD: `distillation.distillation_loss.use_policy_gradient=True`

For another simple Hugging Face dataset, provide the split and column mapping.
The script will create Verl-compatible parquet files in the Modal volume:

```bash
python3 -m modal run --detach scripts/modal_verl_qwen35_opd.py \
  --dataset my_math_smoke \
  --hf-dataset openai/gsm8k \
  --hf-config main \
  --prompt-column question \
  --answer-column answer \
  --data-source openai/gsm8k \
  --ability math \
  --train-rows 8 \
  --val-rows 4 \
  --total-training-steps 1
```

For a dataset that needs custom preprocessing, prepare Verl-compatible parquet
files in the Modal volume and point the trainer at them:

```bash
python3 -m modal run --detach scripts/modal_verl_qwen35_opd.py \
  --skip-prepare \
  --dataset my_dataset \
  --train-files /cache/data/my_dataset/train.parquet \
  --val-files /cache/data/my_dataset/test.parquet \
  --teacher-key my_dataset/source \
  --total-training-steps 1
```

Use Modal app logs to monitor a detached run:

```bash
python3 -m modal app list --json
python3 -m modal app logs <app-id> --tail 500
```

## Modal + Verl SDPO Bridge

The SDPO launcher composes upstream Verl features rather than patching Verl's
actor loss. It prepares feedback-conditioned prompts, uses a custom verifier
reward, and runs on-policy distillation with a frozen self-teacher. By default,
the self-teacher is the same base model as the student.

```bash
python3 -m modal run --detach scripts/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --total-training-steps 1
```

For rich-feedback datasets, map the feedback and previous-attempt columns into
the reprompted training examples:

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

Exact SDPO from the reference implementation adds `loss_mode=sdpo`,
self-distillation batches, and EMA/trust-region self-teacher updates inside the
trainer. This repo keeps Verl unchanged, so the script implements the closest
composition available through public Verl hooks.

## Roadmap

- Add declarative experiment specs for model, dataset, algorithm, reward,
  framework, compute, and eval settings.
- Build adapters that generate Training Gym, Slime, and Verl launches from the
  same spec.
- Add run manifests that capture model versions, framework SHAs, Modal app IDs,
  checkpoints, evals, and failure notes.
- Prototype SDPO-style feedback-conditioned self-distillation on code and math
  tasks.
- Scale from Qwen3.5 smoke runs to larger Qwen and Kimi recipes once tokenizer,
  architecture, and rollout support are validated.

## Secret Hygiene

Do not commit Modal tokens, Hugging Face tokens, W&B keys, SSH keys, or bearer
tokens. Use local auth state or provider-native secrets, and keep repository
files limited to workflow, configuration, and public model references.
