# Riley

**A Rust-native LLM inference server for a single NVIDIA GPU.**

Riley (Rust Inference LLM Engine) serves Llama- and Qwen-style models over an
OpenAI-compatible completions API. The server, scheduler and KV-cache manager
are Rust; the arithmetic runs in ahead-of-time compiled CUDA kernels and
cuBLASLt. There is no Python in the serving process.

> Status: pre-release (0.1.0 release candidates). Developed and measured on one
> RTX 4090 with BF16 weights.

## Highlights

- **Continuous batching** with prompt and decode work sharing each GPU step.
- **Paged KV cache** sized at start-up: memory does not grow under load.
- **Batch-invariant output**: a request produces the same tokens whatever else
  is in the batch.
- **Verified checkpoints**: Hugging Face safetensors are loaded directly, and
  only if every file matches a hashed manifest.
- **Streaming, cancellation and bounded queues** built into the server.

## Performance

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
