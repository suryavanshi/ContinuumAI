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
- [docs/platform-roadmap.md](docs/platform-roadmap.md): the prioritized plan for
  production security, multi-user access, durable jobs, compute management,
  evaluations, checkpoints, and Verl GRPO-family support.
- [docs/platform-handoff.md](docs/platform-handoff.md): the current platform
  implementation, deployment, smoke-run results, operations, and known gaps.
- [MODAL_SKILL.md](MODAL_SKILL.md): a secret-free Modal training runbook for
  launching, monitoring, and debugging GPU jobs.
- [docs/opd-learnings.md](docs/opd-learnings.md): practical notes from the
  Modal + Verl OPD smoke tests.
- [docs/sdpo-learnings.md](docs/sdpo-learnings.md): practical notes from the
  Modal + Verl SDPO bridge smoke tests.
- [docs/scaling-sdpo-replication.md](docs/scaling-sdpo-replication.md): the
  current smoke gate and run ladder for replicating Trajectory's Scaling SDPO
  recipe.
- [docs/kimi-k26-sdpo-modal.md](docs/kimi-k26-sdpo-modal.md): Kimi K2.6 SDPO
  sizing and topology notes for Modal H200/B200 runs.
- [docs/kimi-k26-gpu-layout.html](docs/kimi-k26-gpu-layout.html): a
  GitHub Pages-ready infographic for Kimi K2.6 GPU and distributed training
  layouts.
- [docs/qwen-35b-a3b-sdpo.html](docs/qwen-35b-a3b-sdpo.html): a
  GitHub Pages-ready infographic for the Qwen 35B-A3B SDPO smoke run on Modal.
- [OPD/modal_verl_qwen35_opd.py](OPD/modal_verl_qwen35_opd.py): a Modal
  + Verl smoke launcher for Qwen3.5 on-policy distillation with `k1` loss.
- [SDPO/modal_verl_sdpo.py](SDPO/modal_verl_sdpo.py): a Modal + Verl
  SDPO bridge that uses feedback-conditioned prompts, a custom reward function,
  and Verl's existing on-policy distillation path without modifying Verl.
- [SDPO/modal_verl_kimi_k26_sdpo.py](SDPO/modal_verl_kimi_k26_sdpo.py): a
  larger-model SDPO bridge with Kimi K2.6 H200, B200, and H200 LoRA topologies.
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

## Experiment Console

The repository includes a local front- and backend for inspecting SDPO/OPD
runs, modeled on the workflow needed to productionize self-distillation:

- configure SDPO, OPD, Harvey LAB, and Kimi experiments from the existing
  Modal launchers;
- inspect loss and reward curves, sampled trajectories, privileged hints, and
  token-level teacher/student preference deltas;
- ingest metrics, traces, and logs through a small JSON API; and
- preview the exact argv-safe Modal command before launch.

Start it with Python only (no frontend build step or extra packages):

```bash
python3 -m continuum_console.api --port 8787
```

Then open `http://127.0.0.1:8787`. Runs are stored in
`.continuum/runs.json`. Remote launch is deliberately gated; set
`CONTINUUM_ENABLE_LAUNCH=1` only in an environment where submitting detached
Modal jobs is intended.

Authentication is mandatory for a public deployment. Local development can
enable the same signed, HTTP-only session flow with:

```bash
CONTINUUM_REQUIRE_AUTH=1 \
CONTINUUM_ADMIN_PASSWORD='<12+ character password>' \
CONTINUUM_SESSION_SECRET='<32+ random characters>' \
python3 -m continuum_console.api --port 8787
```

For Modal, create the authentication secret without adding values to the
repository, then deploy the included web-server wrapper:

```bash
modal secret create continuum-console-auth \
  CONTINUUM_ADMIN_USER=admin \
  CONTINUUM_ADMIN_PASSWORD='<strong password>' \
  CONTINUUM_SESSION_SECRET='<random 32+ character value>'

modal deploy continuum_console/modal_app.py
```

The deployment uses a single autoscaling control-plane container and the
`continuum-console-data` Modal Volume for run state. The app binds to
`0.0.0.0`, uses secure signed cookies, rejects cross-origin mutations, caps
training steps and dataset sizes, allowlists models, requires an exact run-ID
launch confirmation, and uses Modal's built-in workload identity for backend
GPU submissions. Modal intentionally ignores account-token environment
variables inside containers, so no account token is copied into the image or
attached as an application Secret.

GUI smoke runs use the repository's x86-compatible `verlai/verl:vllm020.dev1`
image by default to remain compatible with workspaces still on Modal's legacy
Image Builder. Override it with `CONTINUUM_SMOKE_VERL_IMAGE_TAG` after upgrading
the workspace Image Builder and validating a newer image.

Useful API endpoints are `GET /api/catalog`, `GET/POST /api/runs`,
`GET /api/runs/:id`, `POST /api/runs/:id/ingest`, and
`POST /api/runs/:id/launch`.

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

The OPD smoke launcher uses a local copy of Verl mounted into a Modal image and
runs one tiny Qwen3.5 OPD job on GSM8K. By default it looks for a sibling
`../verl` checkout; set `VERL_LOCAL_DIR` if your Verl repo lives elsewhere.

```bash
python3 -m modal run --detach OPD/modal_verl_qwen35_opd.py \
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

Set `VERL_IMAGE_TAG` to try a newer compatible amd64 Verl image. Avoid
`.aarch64.*` tags on standard Modal GPU workers unless the target runtime is
explicitly ARM.

For another simple Hugging Face dataset, provide the split and column mapping.
The script will create Verl-compatible parquet files in the Modal volume:

```bash
python3 -m modal run --detach OPD/modal_verl_qwen35_opd.py \
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
python3 -m modal run --detach OPD/modal_verl_qwen35_opd.py \
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
the self-teacher is the same base model as the student. It now defaults to the
ModelScope CUDA 12.9 fallback image from the AutoAgentTrain runbook.

The SDPO launcher prefetches models into `/cache/models/hf/<safe-name>` by
default and passes that local path to Verl, vLLM, and the self-teacher. Use
`--skip-model-prefetch` only when intentionally testing raw model resolver
behavior.

```bash
python3 -m modal run --detach SDPO/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --total-training-steps 1
```

For the smallest SDPO smoke, keep the run attached. The default clones public
Verl from `VERL_GIT_REF` into `/opt/verl` inside the ModelScope image and skips
uploading the local Verl checkout:

```bash
VERL_GIT_REF=v0.8.0 \
python3 -m modal run SDPO/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --total-training-steps 1
```

Download the committed log and classify it before launching anything longer:

```bash
python3 scripts/inspect_sdpo_log.py /path/to/sdpo.log --expect-steps 1
```

The one-step gate is: training metrics or `global_step_1` must appear, and a
checkpoint should exist when `--save-hf-checkpoint` is enabled.

For rich-feedback datasets, map the feedback and previous-attempt columns into
the reprompted training examples:

```bash
python3 -m modal run --detach SDPO/modal_verl_sdpo.py \
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

## Kimi K2.6 SDPO Topologies

The Kimi launcher keeps large-model experiments separate from the small smoke
scripts:

```bash
python3 -m modal run --detach SDPO/modal_verl_kimi_k26_sdpo.py \
  --mode h200-lora \
  --train-rows 8 \
  --val-rows 4 \
  --total-training-steps 1
```

Use `--mode h200-full` or `--mode b200-full` only after validating model access,
checkpoint format, dataset parquet files, and reward behavior. See
[docs/kimi-k26-sdpo-modal.md](docs/kimi-k26-sdpo-modal.md) for GPU sizing, TP,
EP, NCCL, H200, B200, and LoRA guidance.

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
