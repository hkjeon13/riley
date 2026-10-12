# Performance

All numbers on this page were measured on one RTX 4090 (24 GB), Ubuntu 22.04,
CUDA 12.8.1, with BF16 weights and greedy decoding, through the HTTP API with
streaming. The comparison engine is vLLM 0.27.1 with prefix caching off.

## Qwen2.5-3B-Instruct against vLLM

Measured on 2026-10-12 with the
[recommended configuration](configuration.md#recommended-configuration-for-qwen25-3b).

Workloads: *fixed* uses 12 prompts of about 126 tokens with 8 to 32 output
tokens; *natural* uses three prompts of 16, 128 and 398 tokens with 32, 64 and
128 output tokens. Each is run at 1, 8, 16 and 32 concurrent clients.

| Workload | Throughput, tok/s (Riley / vLLM) | Ratio | Time per output token p50, ms (Riley / vLLM) | Ratio | Time to first token p50, ms (Riley / vLLM) | Ratio |
|---|---|---:|---|---:|---|---:|
| 1 client, fixed | 129.1 / 129.7 | 0.995 | 7.60 / 7.40 | 1.027 | 10.2 / 13.0 | 0.78 |
| 1 client, natural | 129.3 / 132.0 | 0.979 | 7.61 / 7.44 | 1.022 | 10.4 / 13.4 | 0.78 |
| 8 clients, fixed | 836.2 / 762.6 | 1.096 | 8.57 / 8.98 | 0.955 | 20.2 / 31.9 | 0.63 |
| 8 clients, natural | 908.1 / 835.8 | 1.086 | 8.31 / 8.95 | 0.928 | 18.9 / 37.3 | 0.51 |
| 16 clients, fixed | 1425.5 / 1247.1 | 1.143 | 9.53 / 10.29 | 0.926 | 27.1 / 37.1 | 0.73 |
| 16 clients, natural | 1612.6 / 1494.5 | 1.079 | 8.98 / 9.63 | 0.933 | 28.8 / 40.8 | 0.71 |
| 32 clients, fixed | 2040.6 / 1874.0 | 1.089 | 12.42 / 13.52 | 0.919 | 38.6 / 53.8 | 0.72 |
| 32 clients, natural | 2581.8 / 2488.2 | 1.038 | 10.54 / 10.76 | 0.980 | 37.1 / 63.8 | 0.58 |

Ratios are Riley divided by vLLM: above 1 is better for throughput, below 1 is
better for latency.

- **8 to 32 clients:** Riley delivers 4 to 14% more tokens per second and 2 to
  8% lower median time per token.
- **1 client:** the two engines are within 3% of each other, with vLLM slightly
  ahead. A single-request step is dominated by reading the 6 GB of weights
  once per token, which both engines do at the same speed.
- **Time to first token:** the median is 22 to 49% lower with Riley in every
  workload.
- **GPU memory:** 7.2 to 8.5 GiB for Riley, 15.2 to 15.3 GiB for vLLM (started
  with `--gpu-memory-utilization 0.6`).

### Where vLLM is still ahead

Tail time to first token. When 32 clients start at the same instant, Riley
works through the burst of prompts in more, smaller steps:

| Workload | TTFT p95, ms (Riley / vLLM) | TTFT p99, ms (Riley / vLLM) |
|---|---|---|
| 1 client, natural | 28.8 / 25.1 | 29.3 / 25.8 |
| 8 clients, fixed | 31.2 / 38.2 | 56.5 / 50.8 |
| 8 clients, natural | 37.8 / 45.5 | 65.4 / 63.2 |
| 16 clients, natural | 62.3 / 103.0 | 158.2 / 123.0 |
| 32 clients, fixed | 222.2 / 176.1 | 247.1 / 176.6 |
| 32 clients, natural | 240.2 / 241.4 | 324.9 / 242.1 |

Larger steps shorten these tails by 10 to 25% but cost 1.5 to 7.4% of
throughput and push the median time per token above vLLM's, so the default
keeps the smaller step. See [configuration.md](configuration.md#step-size).

### Output consistency

- In every run, each prompt produced exactly one output regardless of
  concurrency, batch composition or repetition (15 of 15 prompts). vLLM's
  outputs varied with batching for more than half of the same prompts.
- Against Hugging Face Transformers (BF16, greedy), 11 of 15 prompts match
  token for token. The other four diverge where the reference's two best
  logits are within 0.25. This count moves by a few prompts between
  numerically equivalent implementations, so read it as "agrees except at
  near-ties", not as a quality score.

### Stability

Five minutes of continuous load per workload, with 60 cancelled requests each:
no errors, no output mismatches, clean shutdown, and all KV-cache blocks
returned. A 30-minute soak was not run on this exact build.

## SmolLM2-135M against vLLM

The small-model results come from a separate kernel branch that is **not**
part of `main`; its source is kept as a git bundle in
[`archive/smollm2-batch14-source`](../archive/smollm2-batch14-source). Measured
on 2026-10-11 on the same machine and workloads:

| | Riley / vLLM ratio across the 8 workloads |
|---|---|
| Throughput | 1.21 to 1.88 |
| Time per output token p50 | 0.64 to 0.90 |
| Time to first token p50 | 0.19 to 0.57 |

For example, one client on the fixed workload: 1,110 against 769 tok/s, and
0.79 against 0.89 ms per token. GPU memory was 1.2 to 1.9 GiB against about
8 GiB. A 30-minute soak (329,472 requests) finished with every output
identical to its reference.

## How the measurements were made

- Both engines served the same checkpoint directory over HTTP on the same GPU,
  one after the other, in the same run. Engine order alternated between
  repetitions.
- Each workload ran twice: 32 warm-up requests, then 192 measured requests.
- The plan, including the pass criteria, was written before the run. No
  request was excluded afterwards; runs disturbed by other load on the machine
  were kept under their own name and not used.
- Time per output token is measured from the client side between streamed
  tokens.

The raw request logs, driver scripts and the full lab notebook were removed
from the tree to keep the repository small. They remain in the git history at
commit `f90683ec` under `benchmarks/results/`.
