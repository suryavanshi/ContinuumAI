# Scaling SDPO Replication Runbook

This runbook tracks the path from infrastructure smoke tests to a closer
replication of Trajectory's Scaling SDPO result with Verl on Modal.

## Target Recipe

From Trajectory's June 2, 2026 field note:

- Train with group size 1 and retain failed rollouts.
- Handle off-policy data with staleness around `K=3`.
- Use PPO ratio clipping at `epsilon=0.2`.
- Add per-token advantage clipping at `3x` to reduce seed variance.
- Validate the recipe at small scale before moving toward larger, longer
  agentic workloads.

The current upstream-Verl launcher in this repo is an SDPO-style bridge, not
the exact SDPO fork from the post. It uses feedback-conditioned prompts,
task rewards, and Verl's `k1` policy-gradient distillation path. Upstream Verl
exposes PPO clipping for this distillation policy loss, but does not expose the
post's exact running-mean `3x` advantage clipping without framework changes.

Public image probes showed that neither `verlai/verl:vllm020.dev1` nor the
ModelScope fallback image contains an importable `verl` package by itself. The
previous Modal launchers made Verl available by copying and installing the
sibling local checkout. In no-local-upload mode, the launcher now clones
`VERL_GIT_REF` from public GitHub into `/opt/verl` during image build.

## Current Evidence

The previous committed SDPO smoke log in Modal is:

```text
/logs/gsm8k_sdpo-sdpo-20260602T073330Z.log
```

It configured `Total training steps: 1`, loaded the Qwen3.5 student, initialized
FSDP, started the self-teacher vLLM server, and loaded the self-teacher
checkpoint. It did not show training metrics, `global_step_1`, checkpoint save,
or a traceback. The Modal volume also had no `/checkpoints` directory.

Classification:

```text
status=incomplete
configured_steps=1
metric_or_checkpoint_evidence=<none>
```

So the prior SDPO smoke has not proven that the Docker/image path completes one
training step.

A second detached `continuum-verl-sdpo` app was found during read-only Modal
inventory:

```text
ap-hka0FbzvRrWT3wfXePkFTp
```

Its logs repeatedly fail during module import with:

```text
DEFAULT_VERL_LOCAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "verl"
IndexError: 2
```

That app is stale pre-fix code and never reaches dataset prep or training. It
should be stopped before relaunching a replacement to keep Modal state clean.
It was stopped with:

```bash
python3 -m modal app stop ap-hka0FbzvRrWT3wfXePkFTp --yes
```

The tiny student/self-teacher model was prefetched without uploading workspace
code by using `modal shell` with the ModelScope image and mounting
`continuum-verl-sdpo-cache`. The snapshot is now present in the Modal volume:

```text
/cache/models/hf/Qwen_Qwen3_5-0_8B
```

The existing tiny dataset files are also present:

```text
/cache/data/gsm8k_sdpo/train.parquet
/cache/data/gsm8k_sdpo/test.parquet
```

The corrected one-step training launch is ready, but it still uploads the local
`ContinuumAI` launcher code to Modal and therefore requires explicit approval:

```bash
VERL_GIT_REF=v0.8.0 python3 -m modal run SDPO/modal_verl_sdpo.py \
  --skip-model-prefetch \
  --skip-prepare \
  --model /cache/models/hf/Qwen_Qwen3_5-0_8B \
  --self-teacher-model /cache/models/hf/Qwen_Qwen3_5-0_8B \
  --train-rows 2 \
  --val-rows 1 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --total-training-steps 1
```

A no-workspace-upload alternative was prepared by uploading only a generated
shell script to the Modal volume:

```text
/cache/runtime/run_sdpo_shell_smoke.sh
```

It runs the ModelScope image, restores the package versions that Modal shell
runtime injection downgrades, clones public Verl `v0.8.0`, and executes the
same one-step Verl config against the cached local model and dataset paths. The
attempted H100 shell app was:

```text
ap-gnWcxS5u5zZuKmm2DnZolV
```

It stayed at zero tasks and no active containers, so the local Modal shell
process was terminated. Treat that as a shell startup/scheduling non-result, not
as a training failure.

The next no-workspace-upload shell smoke used an A100 shell app:

```text
ap-PBQxmlEF0oCh73yqmDxN46
```

That run proved the ModelScope image can restore dependencies, clone public
Verl `v0.8.0`, import `verl`, import CUDA `torch`, load the cached local model
path, and pass Verl config validation. It did not complete a training step. The
failure moved to dataset loading: the existing tiny parquet files were written
with HuggingFace feature metadata using feature type `List`, while the
`datasets` version in the current Verl/ModelScope environment expects feature
types such as `Sequence` or `LargeList`.

The checked-in dataset preparation now writes plain PyArrow parquet records
instead of calling HuggingFace `Dataset.to_parquet`, so regenerated smoke data
should avoid that stale metadata incompatibility.

The saved log is:

```text
/logs/shell-sdpo-smoke-20260604T040718Z.log
```

Local classification after downloading it to `/private/tmp`:

```text
status=failed
configured_steps=None
failure_line=ValueError: Feature type 'List' not found.
metric_or_checkpoint_evidence=<none>
```

The final no-workspace-upload smoke was run with a timestamped checkpoint
directory to avoid accidental resume:

```text
/logs/shell_sdpo_smoke_plain_20260604T075811Z.log
/checkpoints/shell_sdpo_smoke_plain_20260604T075811Z/global_step_1
```

It used the ModelScope fallback image, public Verl `v0.8.0`, the cached local
model path `/cache/models/hf/Qwen_Qwen3_5-0_8B`, plain PyArrow smoke parquet,
rollout group size 1, PPO/distillation clipping at `0.2`, and `k1`
policy-gradient distillation with task rewards. The run trained from scratch,
completed the one configured training step, wrote the actor checkpoint, and
exited with shell status 0.

Downloaded log classification:

```text
status=completed-with-warning
configured_steps=1
expected_steps=1
step_evidence=step:1 ... training/global_step:1 ... critic/rewards/mean:1.0 ...
```

Known warning: Ray prints a `RuntimeError: DataLoader worker ... is killed by
signal: Killed` during/after shutdown, but the fresh `step:1` metrics and
`Final validation metrics: None` appear after that line, the shell exits 0, and
the `global_step_1` checkpoint exists. Treat this as a cleanup warning for the
tiny smoke, not as failure to complete one training step.

## One-Step Gate

Before any longer run, require all of:

- Model prefetch writes the student and self-teacher into
  `/cache/models/hf/<safe-name>`, unless the model arguments already point to
  local Modal volume paths.
- Dataset prep writes tiny train/test parquet files.
- Verl config validation passes.
- Student and self-teacher load.
- Training reaches step 1 with metrics such as reward, actor, or distillation
  loss.
- If `save_hf_checkpoint=True`, `/cache/checkpoints/<run>/global_step_1` exists
  and contains an actor HF export.
- The log is written through `/tmp` and then copied/synced into `/cache/logs`;
  direct tee into a Modal volume path can leave an empty downloaded log.
- Timestamp the smoke run/checkpoint directory so a later verification does not
  resume from an old `global_step_1` checkpoint.

Use the local log inspector after downloading a committed Modal log:

```bash
python3 scripts/inspect_sdpo_log.py /path/to/sdpo.log --expect-steps 1
```

Local preflight before uploading a new Modal app:

```bash
python3 -m unittest tests.test_inspect_sdpo_log tests.test_modal_verl_sdpo_args
python3 -m py_compile SDPO/modal_verl_sdpo.py scripts/inspect_sdpo_log.py
```

The arg tests assert that the smoke config uses rollout group size 1, PPO
clipping at `0.2`, distillation-policy clipping at `0.2`, small vLLM scheduling
limits, `k1` policy-gradient distillation with task rewards, and final HF actor
checkpoint export. They also assert that model paths passed to Verl use local
Modal volume paths.

## Smoke Ladder

1. No-local-Verl-upload smoke:

```bash
VERL_UPLOAD_LOCAL=0 VERL_GIT_REF=v0.8.0 \
python3 -m modal run SDPO/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --total-training-steps 1
```

Model prefetch is enabled by default. Use `--force-model-download` to refresh a
cached snapshot, and use `--skip-model-prefetch` only for intentional raw
resolver tests.

2. Local-Verl smoke, only after approving code upload:

The AutoAgentTrain Verl/Modal skill recommends using a released Verl tag rather
than moving `main` for reproducible smoke runs. Refresh
<https://github.com/verl-project/verl/releases> before launching; as checked on
June 4, 2026, GitHub showed `v0.8.0` as the latest release. The local sibling
checkout was `main` at `v0.8.0-11-gcb821109`, so treat that as an intentional
main-branch experiment if used.

```bash
python3 -m modal run SDPO/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --total-training-steps 1
```

3. Fallback base-image smoke if the default image stalls again:

```bash
VERL_IMAGE_TAG=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.1-modelscope1.35.4-swift4.1.3 \
VERL_UPLOAD_LOCAL=0 VERL_GIT_REF=v0.8.0 \
python3 -m modal run SDPO/modal_verl_sdpo.py \
  --train-rows 2 \
  --val-rows 1 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --total-training-steps 1
```

4. Rich-feedback smoke:

Use a dataset with `prompt`, `answer`, `feedback`, and `failed_solution`
columns so the run tests feedback-conditioned self-distillation rather than
plain GSM8K self-teacher OPD.

5. Short stability run:

After the one-step gate passes, run 5 to 10 steps with the same image and
dataset. Keep `ppo_clip_ratio=0.2`, `distillation_clip_ratio=0.2`, and log
distillation loss range before trying longer runs or off-policy staleness.

## Improvement Path Toward SDPO++

- Add a real feedback dataset or generator so failures carry actionable hints.
- Add an exact advantage-clipping implementation in Verl if the public hooks are
  insufficient for the `3x` running-mean stabilizer.
- Use Verl's async/off-policy support to test staleness once the synchronous
  one-step and short runs are stable.
- Track seed variance, reward/pass rate, distillation loss range, clipping
  fractions, response length, wall-clock time, and checkpoint quality.
