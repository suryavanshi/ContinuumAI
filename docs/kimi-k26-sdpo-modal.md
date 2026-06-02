# Kimi K2.6 SDPO on Modal

This note is a planning and launch guide for training a large MoE model such as
`moonshotai/Kimi-K2.6` with the Modal + Verl SDPO bridge in
[`scripts/modal_verl_kimi_k26_sdpo.py`](../scripts/modal_verl_kimi_k26_sdpo.py).

The script does not modify Verl. It composes existing Verl pieces:

- GRPO-style on-policy rollout and reward learning.
- Feedback-conditioned prompts for SDPO-style self-distillation.
- Verl on-policy distillation with `k1` policy-gradient loss.
- Megatron actor/ref config for MoE-scale training.
- Modal clustered GPU functions with RDMA enabled.

## Sizing Assumptions

Kimi K2.6 should be treated as a trillion-parameter MoE, not as a dense 32B
model. The active compute per token is much smaller than the full parameter
count, but full training still has to move and store a very large expert pool.

Modal's RL infrastructure blog gives the most useful anchor numbers for
planning:

- Kimi K2.6 full weight update: `595.2 GB` with INT4 MoE weights and BF16
  attention.
- Kimi K2.6 reference topology: `16x8 H200`.
- Kimi K2.6 LoRA shared-outer update, rank 32: `9.4 GB`.
- Kimi K2.6 LoRA per-expert update, rank 32: `41.0 GB`.

The GPU count is driven by more than raw model weight size. Budget for:

- actor weights, gradients, optimizer state, and activations;
- frozen reference/self-teacher log-prob workers;
- rollout engine KV cache;
- long-context prompts and responses;
- Ray worker placement overhead;
- checkpoint and weight-sync buffers.

## Recommended Starting Topologies

| Mode | Modal GPUs | Total GPUs | Actor TP | Actor PP | Actor EP | Actor ETP | Actor CP | Rollout TP | Rollout EP | Use case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `h200-full` | `16 x H200:8` | 128 | 8 | 2 | 8 | 1 | 1 | 8 | 8 | Full Kimi K2.6 pilot |
| `b200-full` | `8 x B200:8` | 64 | 8 | 1 | 8 | 1 | 1 | 8 | 8 | B200 full-tune probe |
| `h200-lora` | `2 x H200:8` | 16 | 4 | 1 | 4 | 1 | 1 | 8 | 8 | Rank-32 LoRA pilot |

Interpretation:

- `TP` is tensor parallelism. Increase it when each rank cannot fit attention
  and dense blocks.
- `PP` is pipeline parallelism. Use it to spread layers across more GPUs when
  the per-layer memory pressure is high.
- `EP` is expert model parallelism. For MoE models, this is the main lever for
  distributing experts.
- `ETP` is expert tensor parallelism. Start at `1`; raise only if individual
  expert MLPs are still too large or too slow.
- `CP` is context parallelism. Increase it for very long context runs.
- Rollout `TP` and rollout `EP` need to match the serving engine constraints.
  In current Verl/vLLM configs, keep `rollout_ep == rollout_tp *
  rollout_dp` when expert parallelism is enabled.

## Modal NCCL And RDMA

For Kimi-scale RL, weight synchronization is the difference between a useful
training loop and idle GPUs. Modal's RL infrastructure blog gives the practical
numbers:

| Kimi K2.6 update | TCP transfer | RDMA transfer |
| --- | ---: | ---: |
| Full update, `595.2 GB` | `95.23 s` | `1.49 s` |
| LoRA shared-outer rank 32, `9.4 GB` | `1.50 s` | `23.5 ms` |
| LoRA per-expert rank 32, `41.0 GB` | `6.56 s` | `102.5 ms` |

Use Modal clustered functions with `rdma=True` for colocated trainer, rollout,
reference, and self-teacher workers:

```python
@app.function(gpu="H200:8", timeout=60 * 60 * 24)
@modal.experimental.clustered(size=16, rdma=True)
def train():
    ...
```

The new script starts a Ray cluster across the Modal cluster and passes
`ray_kwargs.ray_init.address=auto` into Verl. It also sets conservative NCCL
defaults:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_DEBUG=WARN
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^lo,docker0
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

Do not overfit the NCCL config before the first run. On Modal, `rdma=True` is
the main switch. Tune only after collecting failure logs for NCCL timeouts,
interface selection, or placement issues.

## Training On H200

H200 is the safer first full-training target because Hopper kernels and
framework wheels tend to be more mature than Blackwell. Modal H200 GPUs have
large HBM capacity, and Modal's GPU docs list H200 as a supported GPU type.

Full Kimi K2.6 pilot:

```bash
python3 -m modal run --detach scripts/modal_verl_kimi_k26_sdpo.py \
  --mode h200-full \
  --dataset kimi_math_feedback \
  --hf-dataset your_org/your_feedback_dataset \
  --prompt-column prompt \
  --answer-column answer \
  --feedback-column feedback \
  --previous-attempt-column failed_solution \
  --train-rows 128 \
  --val-rows 32 \
  --total-training-steps 10
```

If using a converted Megatron/MCore checkpoint in the Modal volume:

```bash
python3 -m modal run --detach scripts/modal_verl_kimi_k26_sdpo.py \
  --mode h200-full \
  --skip-prepare \
  --train-files /cache/data/kimi_feedback/train.parquet \
  --val-files /cache/data/kimi_feedback/test.parquet \
  --mcore-model-path /cache/checkpoints/kimi-k26-mcore
```

Scale sequence length slowly. Start around `2048 + 2048`; once placement and
loss are healthy, increase `--max-prompt-length`, `--max-response-length`, and
the topology's `actor_cp` value if needed.

## Training On B200

B200 has more memory and bandwidth than H200, and Modal supports `B200` and
`B200+` GPU requests. B200 can be attractive for rollout-heavy and memory-bound
MoE workloads, but Blackwell software support may require newer vLLM, CUDA, and
kernel stacks.

B200 full-tune probe:

```bash
VERL_IMAGE_TAG=verlai/verl:<compatible-b200-amd64-tag> \
python3 -m modal run --detach scripts/modal_verl_kimi_k26_sdpo.py \
  --mode b200-full \
  --dataset kimi_math_feedback \
  --hf-dataset your_org/your_feedback_dataset \
  --prompt-column prompt \
  --answer-column answer \
  --feedback-column feedback \
  --previous-attempt-column failed_solution \
  --train-rows 128 \
  --val-rows 32 \
  --total-training-steps 10
```

Use `B200+` only after checking CUDA compatibility. Modal docs note that B200+
can place on B200 or B300, and B300 requires CUDA 13.0+.

If the 8-node B200 topology does not fit:

- increase the clustered size to `16`;
- set actor `PP=2`;
- lower prompt/response length;
- reduce rollout `max_num_batched_tokens`;
- keep `rollout_n=1` until the first stable run.

## LoRA Plus H200

LoRA is the best first Kimi K2.6 training experiment. It keeps the base model
mostly frozen and reduces the weight update from hundreds of GB to GB-scale
adapter updates. Modal's blog shows why this matters: with RDMA, Kimi K2.6
rank-32 shared-outer LoRA sync is in tens of milliseconds rather than seconds.

Rank-32 H200 LoRA pilot:

```bash
python3 -m modal run --detach scripts/modal_verl_kimi_k26_sdpo.py \
  --mode h200-lora \
  --dataset kimi_code_feedback \
  --hf-dataset your_org/your_feedback_dataset \
  --prompt-column prompt \
  --answer-column answer \
  --feedback-column feedback \
  --previous-attempt-column failed_solution \
  --ability code \
  --train-rows 256 \
  --val-rows 64 \
  --total-training-steps 25
```

Use LoRA for:

- algorithm validation;
- reward and feedback debugging;
- rollout/environment throughput tests;
- continual-learning experiments where adapter promotion is acceptable;
- fast regression checks before full fine-tuning.

Move from LoRA to full fine-tuning only after the SDPO reward, feedback
template, rollout quality, and evaluation gates are stable.

## Script Notes

The script defaults to `mode=h200-lora` to avoid accidental 128-GPU launches.
Available modes:

- `h200-lora`: 2 Modal H200 nodes, 16 GPUs total, rank-32 Megatron LoRA.
- `h200-full`: 16 Modal H200 nodes, 128 GPUs total.
- `b200-full`: 8 Modal B200 nodes, 64 GPUs total.

The script accepts generic Hugging Face datasets through column mapping, or
prebuilt Verl parquet files via `--skip-prepare --train-files ... --val-files
...`.

Kimi K2.6 full training may require a validated Megatron-Bridge/MCore
conversion. The launcher accepts `--mcore-model-path` for that path, but the
conversion step is intentionally outside this repo so framework code remains
unchanged.

## References

- Modal, [Reinforcement learning is an infrastructure problem](https://modal.com/blog/reinforcement-learning-infrastructure-problem)
- Modal, [GPU acceleration docs](https://modal.com/docs/guide/gpu)
- Modal, [Introducing B200s and H200s on Modal](https://modal.com/blog/introducing-b200-h200)
- Moonshot AI, [Kimi K2.6 model page](https://www.kimi.com/ai-models/kimi-k2-6)
- Hugging Face, [`moonshotai/Kimi-K2.6`](https://huggingface.co/moonshotai/Kimi-K2.6)
