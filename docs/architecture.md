# Architecture

Riley is a Rust workspace on top of one native library of ahead-of-time
compiled CUDA kernels. The server process contains no Python.

```text
HTTP client
    |
riley-server      HTTP/1.1, OpenAI completions, streaming, shutdown
    |
riley-scheduler   admission, continuous batching, step planning
    |
riley-runtime     model execution: prefill, decode, paged KV cache, sampling
    |         \
riley-model    riley-tensor    checkpoint loading / shapes, views, buffers
                   |
               riley-cuda      safe Rust edge of the C ABI
                   |
               kernels/        CUDA C++ (nvcc), cuBLASLt
```

## Crates

| Crate | Owns | Must not know about |
|---|---|---|
| `riley-core` | Shared error and value types | CUDA, models, scheduling, HTTP |
| `riley-cuda` | The Rust side of the native C ABI: devices, streams, buffers, kernels, CUDA graphs | Tensors, models, the server |
| `riley-tensor` | Shapes, layouts, views and buffer ownership | Model architecture, HTTP |
| `riley-model` | Strict loading of config, tokenizer and safetensors; the canonical model description | Scheduling, HTTP |
| `riley-runtime` | Running a model: prefill, decode, KV cache, sampling | Scheduling policy, HTTP |
| `riley-scheduler` | Bounded admission and continuous batching | HTTP representation |
| `riley-server` | The `riley` binary, the HTTP layer, and wiring everything together | Native ABI details |

`riley-native` is a development-only crate and is not part of the server.
Every crate has empty default features; `cuda` and `server` are switched on at
the top (`cargo build --features cuda,server`). `unsafe` is denied
workspace-wide and confined to the FFI boundary.

## The native boundary

`kernels/` is CUDA C++ behind a single C header,
[`kernels/include/riley_cuda.h`](../kernels/include/riley_cuda.h). The ABI is
versioned and only grows: structs carry their size, handles are opaque and
closed exactly once, every call returns a status with the failing stage, and
no exception or panic crosses the boundary. Matrix multiplication goes to
cuBLASLt; normalisation, RoPE, attention, cache writes and token selection are
Riley's own kernels.

Device buffers are owned Rust values. An asynchronous copy keeps its stream
and both buffers alive until it completes, so memory cannot be freed under a
running kernel.

## Loading a model

`riley-model` reads Hugging Face `config.json`, `tokenizer.json` and
safetensors weights with its own bounded parsers. Every file must be listed
with its size and SHA-256 in `riley-checkpoint.json`; anything else is refused.
The config is turned into a small model description (norm type, attention
geometry, MLP shape, RoPE parameters). Llama-style and Qwen2-style models go
through the same execution code.

## How a request runs

1. **Admission.** The HTTP thread validates the request, tokenises the prompt
   and submits it. If the waiting queue is full the client gets 429.
2. **Planning.** The scheduler builds one *step* at a time: one row for every
   request that is decoding, then prompt tokens from waiting requests until the
   row budget is used. Prompt and decode rows share a step.
3. **Execution.** The runtime runs the transformer once over all rows of the
   step. Attention reads each request's history from the paged KV cache and
   appends the new keys and values.
4. **Sampling.** Rows that finish a prompt or decode a token select the next
   token, on the GPU for greedy requests.
5. **Streaming.** Tokens are detokenised and written to each client as
   server-sent events. A finished, cancelled or disconnected request returns
   its cache blocks.

### Paged KV cache

Keys and values live in fixed 16-token blocks drawn from a pool that is sized
at start-up. A request holds a list of blocks, so requests of different
lengths share memory without copying or fragmentation, and GPU memory does not
grow under load.

### Attention

For Qwen2.5-3B the decode attention is a two-stage kernel: each page of a
request's history is reduced to a small partial result (maximum score, weight
sum, weighted values), and a second kernel merges the pages. Pages are
processed in parallel across the GPU, and with grouped-query attention each
key/value page is loaded once for all the query heads that share it.

### Steps as CUDA graphs

A step with a given number of rows always launches the same kernels, so Riley
records it once as a CUDA graph and replays it, with row counts and addresses
supplied from device memory. This removes most of the launch overhead between
kernels.

## Numerical guarantees

Riley treats three properties separately:

- **Batch invariance.** A request's output does not depend on which other
  requests share its step, or on how its prompt was split across steps. Kernels
  compute each row from that row's data only, and a GEMM algorithm is used for
  a row count only if a start-up probe shows it produces the same bytes as the
  reference algorithm.
- **Reference agreement.** On the fast path, results follow Hugging Face
  Transformers closely but not bit for bit: the rounding order of BF16
  arithmetic differs, which can flip a choice between two nearly tied tokens.
- **Reference exactness.** A separate, slower route reproduces the reference
  output byte for byte and anchors the correctness tests (see
  [configuration.md](configuration.md#the-reference-exact-route)).

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | The Rust workspace |
| `kernels/` | CUDA sources, the C header and native tests |
| `tools/riley-checkpoint/` | Checkpoint preparation and verification |
| `tools/python/reference/` | Python tools that produce reference outputs; never used by the server |
| `benchmarks/`, `ci/` | Measurement and validation tooling |
| `archive/` | Source of the small-model kernel branch, as a git bundle |
| `docs/` | This documentation |
