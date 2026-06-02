# Open-Model Post-Training Plan

This plan treats `ContinuumAI` as the orchestration layer and leaves the
framework repositories (`verl`, `slime`, `training-gym`) unchanged. The goal is
to make RL post-training and dense self-distillation runs easy to launch,
compare, debug, and repeat on Modal for models such as Qwen3.x and Kimi-K2.x.

## Starting Point

Local code already contains most of the infrastructure primitives:

- `training-gym` provides `TrainConfig`, `DeploymentConfig`, `SlimeRecipe`,
  model presets, Modal cluster launchers, checkpoint volumes, eval helpers, and
  a dashboard.
- `training-gym/tutorials/rl/003_on_policy_distillation` demonstrates
  on-policy distillation from Qwen3-8B teacher to Qwen3-4B student.
- `training-gym/tutorials/rl/004_qwen35b` demonstrates Qwen3.6-35B-A3B GRPO on
  DAPO math with Slime on 1 x 8 H100.
- `training-gym/modal_training_gym/tools/convert_kimi_int4_to_bf16.py` is an
  early Kimi support clue, but Kimi is not yet a complete first-class training
  preset in Training Gym.
- `slime` is the best immediate framework for Modal-based large-model RL:
  Megatron training, SGLang rollouts, delta weight sync, async rollout paths,
  custom reward/generate hooks, and explicit data-buffer flow.
- `verl` is the broader algorithm framework: GRPO, PPO, DAPO, GSPO, RLOO,
  ReMax, LoRA RL, multi-turn/tool use, FSDP/FSDP2, Megatron, vLLM, and SGLang.

The Modal infrastructure post frames the scaling problem as three coupled
systems: trainer, rollout engine, and isolated environments. It also calls out
the recurring failures to avoid: glue code sprawl, cluster queue/bring-up time,
and GPU under-utilization from slow environments or weight sync.

## Target Shape

Build a small post-training control plane in `ContinuumAI`:

1. A declarative experiment spec for model, dataset, algorithm, framework,
   reward, eval, cluster shape, checkpoints, and observability.
2. A launcher that converts the spec into a Training Gym run or, later, a Verl
   run.
3. A library of presets for Qwen3/Qwen3.5/Qwen3.6 and Kimi models.
4. A library of algorithm recipes: GRPO/RLVR, on-policy distillation, SDPO-style
   self-distillation, SFT warmup, LoRA/full fine-tuning, and eval-only runs.
5. A run registry that records spec, git SHAs, model/dataset versions,
   checkpoint paths, eval deltas, cost estimates, and failure notes.

The user-facing target should be:

```bash
continuum train configs/qwen3_4b_opd_math.yaml
continuum eval runs/<run_id>/trained_checkpoint.yaml
continuum compare runs/qwen-opd-* --metric aime24
```

## Architecture

### Layer 1: Experiment Specs

Create YAML or TOML specs with strict schemas:

- `model`: base HF repo, architecture preset, tokenizer policy, trust-remote-code
  setting, conversion requirements, and checkpoint source.
- `teacher`: optional teacher deployment for OPD, or feedback-conditioned
  self-teacher for SDPO.
- `dataset`: HF dataset or local dataset, split, row limits, prompt/label
  columns, chat template settings, and preprocessing.
- `algorithm`: `grpo`, `opd`, `sdpo`, `sft`, `dapo`, etc.
- `reward`: built-in reward type, Python reward callback, environment feedback,
  verifier, sandbox config, and reward post-processing.
- `framework`: `slime` first, `verl` later.
- `compute`: Modal GPU type, node count, GPUs per node, tensor parallelism,
  expert parallelism, rollout engines, RDMA requirement, colocated vs
  disaggregated.
- `observability`: W&B project/group, dashboard tags, eval interval, sample
  logging, trace/profiling toggles.
- `safety`: max cost, max runtime, checkpoint interval, stop conditions.

### Layer 2: Framework Adapters

Implement adapters in `ContinuumAI`, not by changing framework code.

First adapter: `TrainingGymSlimeAdapter`

- Maps specs into `TrainConfig`, `DeploymentConfig`, `SlimeRecipe`, and existing
  model/dataset classes.
- Supports inline custom reward and generate functions through Training Gym's
  callable serialization.
- Materializes run-local Python modules for custom reward/generate/post-process
  functions when the framework requires import paths.
- Applies model defaults, then spec overrides.
- Writes a normalized run manifest before launch.

Second adapter: `VerlAdapter`

- Starts as a config generator plus command launcher for known Verl examples.
- Later maps specs to Verl Hydra configs for GRPO/PPO/DAPO/GSPO.
- Useful when algorithm exploration matters more than Modal integration.

### Layer 3: Model Presets

Prioritize presets in this order:

1. Qwen3-4B and Qwen3-8B for fast OPD and RL smoke tests.
2. Qwen3.6-35B-A3B for the first serious MoE run, because Training Gym already
   has a local model preset and tutorial.
3. Qwen3-30B-A3B / Qwen3-32B for scaling and teacher experiments.
4. Kimi-K2.x after conversion, tokenizer, architecture, and SGLang/Megatron
   support are validated.

Each preset should include:

- minimum viable Modal topology;
- rollout topology;
- recommended max response length and dynamic batching;
- LoRA/full fine-tuning support status;
- checkpoint conversion requirements;
- known failure modes.

### Layer 4: Algorithm Recipes

#### GRPO/RLVR

Use this as the baseline for math/code/verifiable tasks.

- Start from `training-gym/tutorials/rl/004_qwen35b`.
- Use built-in Slime `rm_type="deepscaler"` for math.
- Add custom rewards for code, tool use, and multi-turn tasks.
- Track sparse reward stability, response length, pass@k, and eval deltas.

#### On-Policy Distillation

Use this when a stronger teacher can provide token-level dense supervision.

- Start from `training-gym/tutorials/rl/003_on_policy_distillation`.
- Student samples trajectories.
- Teacher SGLang endpoint returns logprobs on student trajectories.
- Slime applies reverse-KL-style per-token penalty plus optional scalar task
  reward.
- Use same-family teachers first to avoid tokenizer alignment issues.
- Later test cross-family teachers, such as Kimi -> Qwen, only with explicit
  token/character alignment checks.

#### SDPO-Style Self-Distillation

Use this when the environment returns textual feedback such as test failures,
compiler errors, judge comments, or verifier explanations.

Spec behavior:

- Student samples a rollout.
- Environment returns scalar outcome plus rich feedback text.
- The same model is queried again with feedback-conditioned context to produce
  self-teacher next-token distributions.
- The policy learns from dense token-level feedback-conditioned targets.
- If only scalar feedback exists, successful rollouts can become implicit
  feedback for failed attempts.

Implementation path:

1. Build this first as a Slime custom reward/generate block using SGLang logprob
   calls.
2. Keep the reward term and dense self-distillation term separately logged.
3. For code tasks, use Modal Sandboxes for feedback from tests and runtime
   errors.
4. Add gating or coefficient schedules to prevent bad feedback-conditioned
   targets from overpowering RL reward.

#### SFT Warmup And Recovery Distillation

For continual learning:

- SFT or midtrain on domain data.
- Evaluate task skill and general behavior.
- Run OPD from a frozen earlier checkpoint or stronger teacher on instruction
  following/chat/reasoning prompts to recover behavior.
- Alternate domain updates and recovery distillation.

## Modal Operating Model

Use Modal for the infrastructure pieces that are expensive to hand-roll:

- clustered multi-node GPU functions with RDMA where available;
- shared model/data/checkpoint volumes;
- SGLang teacher and rollout deployments;
- Modal Sandboxes for code/tool environments;
- dashboard, W&B, logs, GPU metrics, and retries.

Initial sizing rules:

- Keep first runs single-node and short: Qwen3-4B/8B, 100-500 prompts, frequent
  checkpoint/eval.
- Move to 1 x 8 H100 for Qwen3.6-35B-A3B.
- Use colocated trainer/rollout initially to reduce sync complexity.
- Use delta weight sync or disaggregated rollout only after a baseline is stable.
- Size sandbox pools to at least the number of active episodes/rollouts; tune by
  measuring GPU wait time and environment latency.

## Implementation Phases

### Phase 0: Reproducibility Baseline

Deliverables:

- Run manifest schema.
- Local CLI skeleton.
- Importable wrappers for Training Gym launch/eval.
- Smoke configs for Qwen3-4B GRPO and Qwen3-8B teacher deployment.

Exit criteria:

- A run manifest captures exact model, dataset, recipe, framework commit,
  checkpoint path, W&B info, and eval result.
- A tiny no-GPU/local validation command checks schemas and import paths.

### Phase 1: Training Gym + Slime MVP

Deliverables:

- `qwen3_4b_opd_math.yaml` based on tutorial 003.
- `qwen3_6_35b_grpo_math.yaml` based on tutorial 004.
- Reward/eval module for DAPO math.
- Dashboard/run-registry integration.

Exit criteria:

- Launch OPD teacher/student training from one config.
- Launch Qwen3.6-35B GRPO from one config.
- Compare base and trained evals with a single command.

### Phase 2: Environment And Feedback Tasks

Deliverables:

- Code RL task spec using Modal Sandboxes.
- Custom generate/reward flow for runtime/test feedback.
- Sandbox buffer/concurrency telemetry.
- Eval suite for pass@1/pass@k and failure category tracking.

Exit criteria:

- GPU utilization is not blocked by environment startup in steady state.
- Failed rollouts store enough feedback to debug reward and SDPO behavior.

### Phase 3: SDPO Prototype

Deliverables:

- Feedback-conditioned self-teacher module.
- Dense token-level SDPO loss path through Slime OPD-style hooks.
- Coefficient schedules and ablations: GRPO only, SDPO only, GRPO + SDPO.
- Code and math tasks with rich feedback and scalar-only feedback variants.

Exit criteria:

- SDPO produces measurable sample-efficiency improvement over GRPO on at least
  one small task.
- Logs separate scalar reward, self-distillation loss, response length, and eval
  accuracy.

### Phase 4: Larger Models And Kimi

Deliverables:

- Qwen3-30B/32B and Qwen3.6-35B tuned presets.
- Kimi conversion workflow and architecture/preset validation.
- LoRA vs full fine-tuning comparison.
- Delta weight sync or disaggregated rollout benchmarks.

Exit criteria:

- One successful Kimi serve/eval run.
- One successful Kimi training smoke test, even if small.
- Clear cost/performance recommendation per model family.

### Phase 5: Verl Adapter

Deliverables:

- Spec-to-Verl config generator.
- Known-good Verl GRPO/DAPO/Qwen configs.
- Comparison harness: same dataset/eval, Slime vs Verl.

Exit criteria:

- Framework choice is data-driven: Slime for Modal-first large-model runs,
  Verl for algorithm breadth and FSDP/FSDP2/vLLM experiments.

## Metrics

Track every run with:

- eval accuracy/pass@k;
- reward mean/std and invalid-output rate;
- response length distribution;
- teacher KL or self-distillation loss;
- tokens/sec for training and rollout;
- GPU utilization and GPU wait time;
- environment latency and sandbox failure rate;
- checkpoint size and weight-sync time;
- wall-clock time and estimated Modal cost.

## Main Risks

- Tokenizer mismatch in cross-family OPD or Kimi-to-Qwen distillation.
- Sparse reward hacking or overly strict answer parsers.
- SDPO self-teacher reinforcing bad retrospection.
- GPU under-utilization from sandbox/environment latency.
- Weight sync becoming dominant as model size grows.
- Generated tutorial files in Training Gym must not be edited directly.
- Kimi support likely requires model architecture, conversion, and serving
  validation before full RL is realistic.

## Recommended First Week

1. Build the spec schema and CLI skeleton in `ContinuumAI`.
2. Add a Training Gym Slime adapter for existing Qwen models.
3. Port the OPD math tutorial into a config-driven run.
4. Port the Qwen3.6-35B GRPO tutorial into a config-driven run.
5. Add run manifests and eval comparison.
6. Do one tiny Qwen3-4B/8B OPD run.
7. Do one Qwen3.6-35B preflight or short GRPO run if Modal multi-node/H100
   access is available.

## Decision

Start with Training Gym + Slime. It already matches the infrastructure shape:
Modal launch, SGLang rollout, Megatron training, custom reward/generate hooks,
checkpointing, and dashboarding. Add Verl later as a second backend for broader
algorithm experiments once the experiment spec and evaluation harness are
stable.
