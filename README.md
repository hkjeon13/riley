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
five run-level metrics. The adopted v15C prefill/host implementation is compared
with the previous v13B and stock vLLM 0.27.1 (FA2, FULL_AND_PIECEWISE).

| Clients | Previous Riley tok/s | Latest Riley tok/s | Change | vLLM tok/s | Latest Riley / vLLM |
|---:|---:|---:|---:|---:|---:|
| 16 | 717.829 | 747.513 | +4.14% | 1138.307 | 65.67% |
| 32 | 932.064 | 980.137 | +5.16% | 1683.754 | 58.21% |

Latency, milliseconds (latest Riley / stock vLLM):

| Clients | TTFT p95 | TTFT p99 | TPOT p95 | TPOT p99 | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 82.409 / 83.158 | 164.515 / 110.557 | 20.028 / 12.706 | 21.469 / 13.315 | 660.452 / 424.361 |
| 32 | 245.456 / 162.982 | 335.009 / 216.597 | 31.542 / 17.574 | 38.458 / 22.718 | 1008.210 / 554.416 |

Both engines use BF16, greedy sampling, token budget 576, maximum sequence
length 1024 and prefix caching disabled. Each arm has 64 warm-up and 192
measured requests: 12 fixed prompts of 115–130 tokens and balanced generation
limits of 8/16/24/32 tokens. TPOT is the request-average output-token latency,
not a token-level inter-token percentile. The Riley recipe explicitly selects
`owned-hf-eager-bf16-v1`, `native-prefill-sm80-v5` and `segments-v1`; these results
do not cover every model or the generic default settings.

Riley preserves literal full-vocabulary BF16 equality to the original
native/Hugging Face reference: 4,480 BF16 files, 524 CPU tests and five actual
GPU contract tests passed, including 120 alignment/view cases. Stock vLLM's
numerical path is not Riley's correctness reference. No paired TTFT/TPOT/E2E
p95/p99 regression above 2% occurred in this final five-repeat comparison.
The throughput gap remains substantial; the +10% improvement target is unmet.

[Latest implementation, all repeats and caveats](docs/qwen-prefill-host-v15c-20261021.md) ·
[Audited evidence](benchmarks/evidence/20261021-prefill-host-v15c/README.md).
Earlier experiments remain separately documented: [v14 rejected](docs/qwen-mixed-attention-structural-v14-20261021.md)
(+0.76% / −0.27%), [v15A screen](docs/qwen-prefill-host-v15a-20261021.md), and
[v15B unadopted checkpoint](docs/qwen-prefill-host-v15b-20261021.md)
(+3.06% / +5.45%, one C32 TPOT p99 pair +2.40%). Those samples were retained;
the final alignment guard is a compatibility fix, not a proven cause of the
different tail outcome.

[v16A structural research](docs/qwen-mlp-decode-v16a-20261021.md) retains unsuccessful
Gate/Up, Down and KV redesigns and selects a native-order QK/PV candidate for
whole-model qualification. Its component timings do not replace the server
measurements above.

Completed server experiments, **all unadopted**. Each row uses its
own fresh v15C and stock controls, with five completed repetitions per engine:

| Experiment | Clients | v15C control tok/s | Candidate tok/s | Change | vLLM tok/s | Candidate / vLLM |
|---|---:|---:|---:|---:|---:|---:|
| v16B | 16 | 729.296 | 738.407 | +1.25% | 1130.970 | 65.29% |
| v16B | 32 | 973.861 | 986.968 | +1.35% | 1672.321 | 59.02% |
| v16C | 16 | 738.804 | 760.250 | +2.90% | 1146.853 | 66.29% |
| v16C | 32 | 975.260 | 991.783 | +1.69% | 1676.802 | 59.15% |
| v16G | 16 | 735.892 | 747.695 | +1.60% | 1143.677 | 65.38% |
| v16G | 32 | 979.990 | 998.974 | +1.94% | 1690.953 | 59.08% |

v16B changes QK/PV execution; v16C adds a separate merged prefill Gate/Up
plan. Both pass the original 4,480-file native/HF BF16 gate. v16C also passes
six GPU contract tests, including 36 new raw merged cases. v16C retains a
C16 TTFT p95 pair +3.48% and C32 TPOT p99 pairs up to +4.38%. The large
throughput improvement objective remains unmet. Completed normal samples
were all kept. C16's vLLM startup/continuation interruption is documented.

[v16B report and diagnosis](docs/qwen-mlp-decode-v16b-20261021.md) ·
[v16C implementation, all repeats and caveats](docs/qwen-mlp-prefill-v16c-20261021.md) ·
[v16C audited evidence](benchmarks/evidence/20261021-mlp-prefill-v16c/README.md).

[v16E small-tile mixed Gate/Up](docs/qwen-mlp-small-v16e-20261021.md) also
preserves the tested BF16 bytes but is slower across many shapes than the
optimized merged-prefill control. It remains a component experiment and is
excluded from qualified server throughput claims.

[v16D coalesced prefill Gate/Up](docs/qwen-mlp-batch-v16d-20261021.md) passes
2,676 layer0 BF16 cases and reduces actual-route GEMM time in a component
screen. Its whole-model successor G completed qualification but was rejected
on paired tails (see below).

[v16F serving-capacity qualification](docs/qwen-mlp-batch-v16f-20261021.md)
passes the original numerical suite but fails an added cold scratch-constructor
check before logits. It is unadopted and has no normal throughput measurements.

[v16G coalesced Gate/Up + QK/PV](docs/qwen-mlp-batch-v16g-20261021.md)
extends the merged prefill path to whole eligible batches (65–576 rows) and
fixes the cold scratch guard found in F. It passes the original 4,480-file
native/HF BF16 gate plus 2,048 serving-capacity files, 524 CPU tests and six
GPU contract tests (120 alignment/view and 66 raw merged cases). The C16
TTFT p95 r0 pair is +2.42%; C32 has TTFT p95/p99 pairs up to +5.66%
and TPOT p99 pairs up to +12.37%. It remains unadopted. All 30 normal arms and
four separate profile arms are retained. The current runtime remains v15C;
the large throughput objective is unmet. [Audited G evidence](benchmarks/evidence/20261021-mlp-batch-v16g/README.md).

<!-- v17 independent experiments -->
### A100 · Qwen3-4B · independent v17 experiments

Decode projection and mixed execution/committed-event overlap are separate source candidates from v15C. Each row uses its own fresh same-GPU controls and five processes per engine/concurrency. No completed normal arm is excluded.

| Experiment | Clients | v15C tok/s | Candidate tok/s | Change | vLLM tok/s | Candidate / vLLM | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| v17A Gate/Up | 16 | 735.627 | 743.473 | +1.07% | 1155.362 | 64.35% | Rejected |
| v17A Gate/Up | 32 | 978.131 | 981.322 | +0.33% | 1691.051 | 58.03% | Rejected |

The original literal native/HF BF16 references and paired TTFT/TPOT p95/p99
2% tail gates are retained. “Criteria pass” in an experiment receipt is
qualification; application to main is recorded separately. The projection candidate is rejected on tails and will not be combined with B4. The independent mixed/overlap comparison is running. Main remains v15C.

[projection-v17a report](docs/qwen-projection-v17a-20261021.md)

<!-- end v17 independent experiments -->

Dated A100 matrices for other model families and their original numerical
contracts are in [model coverage](docs/model-coverage-a100.md). Separate native
preset screens are documented in [native HF performance](docs/hf-native-performance.md),
including their modified-baseline and acceptance status.

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
| Models | `LlamaForCausalLM` (SmolLM2), `Qwen2ForCausalLM` (Qwen2.5), dense `Qwen3ForCausalLM` (Qwen3) |
| Precision | BF16 |
| Hardware | One NVIDIA GPU; kernels built for `sm_89` by default |
| API | `POST /v1/completions` with streaming; models, health and metrics endpoints |

Not supported yet: chat-completions endpoint, quantised weights, multiple GPUs.

## License

[MIT](LICENSE)
