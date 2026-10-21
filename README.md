# Riley

**A Rust-native LLM inference server for a single NVIDIA GPU.**

Riley (Rust Inference LLM Engine) serves Llama- and Qwen-style models over an
OpenAI-compatible completions API. The server, scheduler and KV-cache manager
are Rust; the arithmetic runs in ahead-of-time compiled CUDA kernels and
cuBLASLt. There is no Python in the serving process.

> Status: pre-release (0.1.0 release candidates). BF16 measurements include
> separate RTX 4090 and A100 qualification campaigns; results are model- and
> workload-specific.

## Highlights

- **Continuous batching** with prompt and decode work sharing each GPU step.
- **Paged KV cache** sized at start-up: memory does not grow under load.
- **Batch-invariant output**: a request produces the same tokens whatever else
  is in the batch.
- **Verified checkpoints**: Hugging Face safetensors are loaded directly, and
  only if every file matches a hashed manifest.
- **Streaming, cancellation and bounded queues** built into the server.

## Performance

### A100 · Qwen3-4B · latest qualified checkpoint

Measured 2026-10-21 on the same physical A100 (GPU 7), sequentially, with
five fresh processes per engine and concurrency. Values are the medians of
the five run-level metrics. Riley uses the qualified v13B implementation;
vLLM is stock 0.27.1 with FlashAttention 2 and FULL_AND_PIECEWISE graphs.

| Clients | Riley output tok/s | vLLM output tok/s | Riley / vLLM | TTFT p95 ms, Riley / vLLM | TPOT p95 ms, Riley / vLLM | E2E p95 ms, Riley / vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 717.594 | 1149.627 | 62.42% | 88.177 / 83.390 | 20.994 / 12.380 | 689.143 / 414.511 |
| 32 | 933.101 | 1674.932 | 55.71% | 270.162 / 163.738 | 33.801 / 17.755 | 1063.540 / 559.684 |

Both engines use BF16, greedy sampling, token budget 576, maximum sequence
length 1024 and prefix caching disabled. Each arm has 64 warm-up and 192
measured requests: 12 fixed prompts of 115–130 tokens and balanced generation
limits of 8/16/24/32 tokens. TPOT is the request-average output-token latency,
not a token-level inter-token percentile. Riley additionally preserves literal
full-vocabulary BF16 equality to the original native/Hugging Face reference;
stock vLLM's numerical path is not Riley's correctness reference.

The v14 structural candidate was **rejected**: throughput changed by +0.76%
at C16 and −0.27% at C32, with three paired tail regressions above 2%.
These results do not establish throughput parity with vLLM. Method, repeated
tails, correctness and resource receipts:
[v14 comparison](docs/qwen-mixed-attention-structural-v14-20261021.md),
[retained v13 design](docs/qwen-piecewise-graph-merged-gateup-v13-20261020.md),
[evidence index](benchmarks/evidence/20261021-structural-v14/README.md).

### RTX 4090 · Qwen2.5-3B-Instruct · historical campaign

Qwen2.5-3B-Instruct against vLLM 0.27.1 on the same RTX 4090, same run, greedy
decoding (Riley / vLLM):

| Concurrent clients | Throughput, tok/s | Time per token p50, ms | Time to first token p50, ms | GPU memory |
|---|---|---|---|---|
| 1 | 129 / 130 – 132 | 7.6 / 7.4 | 10 / 13 | 7.2 / 15.2 GiB |
| 8 | 836 – 908 / 763 – 836 | 8.3 – 8.6 / 9.0 | 19 – 20 / 32 – 37 | |
| 16 | 1426 – 1613 / 1247 – 1495 | 9.0 – 9.5 / 9.6 – 10.3 | 27 – 29 / 37 – 41 | |
| 32 | 2041 – 2582 / 1874 – 2488 | 10.5 – 12.4 / 10.8 – 13.5 | 37 – 39 / 54 – 64 | 8.5 / 15.3 GiB |

At 8 to 32 clients Riley is 4 to 14% ahead on throughput; at one client the two
are within 3%. vLLM has the better tail time to first token when many clients
start at once. Details, method and caveats: [docs/performance.md](docs/performance.md).

## Quick start

Requirements: Linux x86_64, an NVIDIA GPU, CUDA Toolkit 12.8, CMake 3.22+,
Rust 1.85.

```bash
export CUDA_HOME=/usr/local/cuda
cargo build --locked --release --features cuda,server
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

./target/release/riley serve --model /models/qwen2.5-3b-instruct --model-id qwen3b

curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3b", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

`--model` points to a checkpoint directory with a manifest; see
[Getting started](docs/getting-started.md#prepare-a-checkpoint).

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Build, prepare a checkpoint, run, send a request |
| [Configuration](docs/configuration.md) | Server options and the recommended fast configuration |
| [HTTP API](docs/api.md) | Endpoints, request fields, streaming, errors |
| [Architecture](docs/architecture.md) | Crates, request flow, KV cache, numerical guarantees |
| [Performance](docs/performance.md) | Measurements against vLLM and how they were taken |
| [History](docs/HISTORY.md) | How Riley was developed and what improved along the way |

## Supported

| | |
|---|---|
| Models | `LlamaForCausalLM` (SmolLM2), `Qwen2ForCausalLM` (Qwen2.5) |
| Precision | BF16 |
| Hardware | One NVIDIA GPU; kernels built for `sm_89` by default |
| API | `POST /v1/completions` with streaming; models, health and metrics endpoints |

Not supported yet: chat-completions endpoint, quantised weights, multiple GPUs.

## License

[MIT](LICENSE)
