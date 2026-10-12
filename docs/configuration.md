# Configuration

Everything is set on the command line of `riley serve`. `riley serve --help`
prints the complete list; this page explains what matters.

## Capacity

| Option | Default | Meaning |
|---|---|---|
| `--model PATH` | required | Checkpoint directory |
| `--model-id ID` | source model name | Name clients put in `"model"` |
| `--bind ADDRESS` | `127.0.0.1:8080` | Listener address |
| `--device ORDINAL` | 0 | CUDA device |
| `--max-active-sequences N` | 8 | Requests generating at the same time |
| `--max-waiting-requests N` | 64 | Queue length; further requests get 429 |
| `--max-sequence-tokens N` | checkpoint maximum | Prompt plus output tokens per request |
| `--max-output-tokens N` | min(1024, context − 1) | Upper bound for `max_tokens` |
| `--kv-blocks N` | enough for every active sequence | KV-cache size, in 16-token blocks |
| `--max-weight-bytes N` | 2 GiB (8 GiB with the D128 backend) | Refuse checkpoints larger than this |

GPU memory is reserved at start-up from these bounds and does not grow with
load. Lower `--max-sequence-tokens` and `--max-active-sequences` to what you
actually serve: with Qwen2.5-3B, 32 sequences of 1,024 tokens use about 8.5 GiB.

## Step size

Riley runs one *step* at a time. A step contains one row per decoding request
plus as many prompt tokens as fit.

| Option | Default | Meaning |
|---|---|---|
| `--batch-token-budget N` | 512 | Rows per step |
| `--prefill-chunk-tokens N` | 512 | Prompt tokens one request may add to a step |
| `--batch-shape-policy` | `fixed-max` | `power-of-two` runs each step at the smallest listed shape that fits |
| `--batch-shape-buckets LIST` | powers of two | The shapes used by `power-of-two` |
| `--prefill-plateau-rows LIST` | off | See below |

A larger budget finishes prompt bursts in fewer steps (better tail
time-to-first-token) but makes every step that carries prompt tokens slower for
the requests already decoding. For Qwen2.5-3B, 256 was the best trade-off.

`--prefill-plateau-rows` exists because step cost is not smooth in the row
count: with Qwen2.5-3B on an RTX 4090 a step costs about 1 ms more from row 129
and again from row 193. With `--prefill-plateau-rows 128,192`, a step that
would end just past such a boundary leaves those few prompt tokens for the next
step. A request waits at most once and its output does not change. The
boundaries depend on model and GPU, so measure before using other values.

## Execution options

Most remaining options select between an original implementation and a faster
one. They are grouped here by what they do to the output.

**Bit-identical to the default path** — same tokens, same logits:

| Option | Fast value | What it does |
|---|---|---|
| `--execution-completion` | `iteration-batch` (default) | Wait for the GPU once per step, not once per kernel |
| `--mlp-activation` | `fused-silu-multiply-v1` | SiLU and gate multiply in one kernel |
| `--query-key-rope` | `fused-split-v1` | RoPE inside the Q/K/V split kernel |
| `--kv-cache-write` | `fused-split-v1` | KV-cache write inside the same kernel |
| `--lm-head-rows` | `output-rows-v1` | Compute logits only for rows that emit a token |
| `--gemm-algorithm-selection` | `bit-equivalent-table-v1` | Per row count, use the fastest GEMM algorithm that reproduces the reference algorithm's bytes |
| `--decode-step-graph` | `capture-v1` | Record steps as CUDA graphs and replay them |
| `--sampling-backend` | `gpu-greedy` | Greedy token selection on the GPU |

**Different rounding, still independent of batching** — a request's output is
the same whatever else is in the batch, but may differ from the default path
in the last bits:

| Option | Value | What it does |
|---|---|---|
| `--residual-rmsnorm` | `fused` | Residual add and RMSNorm in one kernel |
| `--mlp-residual-norm` | `fused-next-norm-v1` | MLP residual add fused with the next norm |
| `--projection-bias-backend` | `merged-qkv-strict-staged-v1` | One GEMM for Q, K and V |
| `--gemm-reduction-policy` | `first-heuristic-split-k-v1` | A faster cuBLASLt reduction order, fixed for all row counts |
| `--decode-attention-backend` | `native-bf16-paged-split-gqa-d128-two-stage` | Paged attention for Qwen2.5-3B (16 query heads, 2 KV heads, head size 128) |
| `--decode-attention-merge` | `global-max-v1` | Merge attention pages in one kernel |
| `--decode-attention-producer` | `tensor-core-v1` | Compute attention pages with tensor-core arithmetic |

The D128 backend accepts only the Qwen2.5-3B attention geometry and a step
budget of at most 256. Other models use the default backend.

## Recommended configuration for Qwen2.5-3B

This is the configuration behind [performance.md](performance.md):

```bash
riley serve --model /models/qwen2.5-3b-instruct --model-id qwen3b \
  --max-active-sequences 32 --max-sequence-tokens 1024 --max-output-tokens 128 \
  --max-weight-bytes 8589934592 \
  --batch-token-budget 256 --prefill-chunk-tokens 256 \
  --batch-shape-policy power-of-two \
  --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 \
  --prefill-plateau-rows 128,192 \
  --decode-attention-backend native-bf16-paged-split-gqa-d128-two-stage \
  --decode-attention-merge global-max-v1 \
  --decode-attention-producer tensor-core-v1 \
  --projection-bias-backend merged-qkv-strict-staged-v1 \
  --gemm-reduction-policy first-heuristic-split-k-v1 \
  --gemm-algorithm-selection bit-equivalent-table-v1 \
  --residual-rmsnorm fused --mlp-activation fused-silu-multiply-v1 \
  --mlp-residual-norm fused-next-norm-v1 \
  --query-key-rope fused-split-v1 --kv-cache-write fused-split-v1 \
  --lm-head-rows output-rows-v1 --decode-step-graph capture-v1 \
  --sampling-backend gpu-greedy
```

With this configuration, each prompt produced exactly one output across
concurrency 1 to 32 in every measured run, and 11 of 15 test prompts produced
the same tokens as Hugging Face Transformers; the other four diverge at a
position where the reference's two best logits are within 0.25 of each other.

## The reference-exact route

A build with the `hf-exact-probe` feature adds
`--numerics-profile hf-eager-qwen-p2048-o128-probe-v1`. It serves one request
at a time (2,048-token prompt, up to 128 greedy output tokens) and reproduces
the Hugging Face Transformers BF16 output byte for byte, logits included. It is
a correctness anchor, not a serving mode: it is several times slower.
