# Serving validation

What was added for agent and RAG workloads on 2026-11-02, how it behaves under
load, and where it still trails vLLM. Everything here ran on A100-SXM4-40GB
GPUs of `aws-a100` (physical 6 and 7) with vLLM 0.27.1 as the baseline.
Evidence is in `benchmarks/evidence/20261102-gap/`; the scripts are in
`benchmarks/live/`.

The tables in the [README](../README.md#performance) use closed-loop clients
and short prompts. This page adds what those do not show: open-loop arrivals,
bursts, long prompts, long outputs, and requests that need their logits on the
CPU.

## Method

Unless noted the model is Qwen2.5-7B-Instruct in BF16, prefix caching is off on
both engines, and Riley uses `--inference-profile native-bf16-varlen-v1`.
Each engine ran twice, alternating (Riley, vLLM, vLLM, Riley), on the same GPU;
tables show the mean of the two. Cells of 192 requests follow 32 warm-up
requests. The later grammar check affects only requests with a grammar, so the
load tables were not repeated after it.

| Name | Prompt tokens | Output tokens | Load |
|---|---:|---:|---|
| closed-cN | 72 | 64 | N clients, each waiting for its reply |
| open-rN | 72 | 64 | Poisson arrivals, N requests/s |
| burst-rN | 72 | 64 | Mean rate N, arriving 1 s on, 3 s off |
| mid-* | 397 | 64 | As above |
| rag-* | mean 2,640 (1,500–3,000 words) | 128 | 16 to 64 requests |
| reason-cN | 72 | up to 1,024 (mean 837) | N clients |

Short cells ran with `--max-sequence-tokens 1024`; rag and reason with
`--max-sequence-tokens 8192`.

## Load

Riley / vLLM, times in ms:

| Cell | Throughput tok/s | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 |
|---|---:|---:|---:|---:|---:|
| closed-c8 | **650** / 621 | **32.5** / 51.2 | **38.2** / 56.0 | **38.5** / 58.7 | **12.0** / 12.2 |
| closed-c32 | **2246** / 2090 | **40.0** / 111.5 | **41.0** / 176.8 | **59.2** / 183.0 | 13.6 / 13.6 |
| open-r6 | 387 / 386 | **21.7** / 36.6 | **27.6** / 41.9 | **34.8** / 43.7 | **11.9** / 12.3 |
| open-r16 | 996 / 994 | **22.8** / 39.3 | **30.9** / 47.2 | **36.7** / 52.3 | **12.5** / 12.9 |
| open-r32 | 1887 / 1879 | **27.6** / 49.1 | **230.8** / 287.4 | **295.7** / 350.6 | **13.5** / 14.1 |
| burst-r6 | 389 / 389 | **22.8** / 39.4 | **31.3** / 46.8 | **40.6** / 50.3 | **12.5** / 12.9 |
| burst-r16 | 1086 / 1083 | **373.7** / 426.4 | **919.4** / 989.5 | **1022.8** / 1098.0 | **13.4** / 13.9 |
| mid-r8 | 484 / 483 | **54.0** / 59.5 | 85.8 / **82.0** | 106.9 / **95.7** | **13.4** / 14.6 |
| mid-r16 | 940 / 939 | **64.3** / 76.0 | 153.9 / **130.3** | 203.5 / **158.8** | **15.7** / 19.2 |
| mid-c32 | **1369** / 1297 | **102.3** / 357.9 | 708.3 / **360.6** | 748.4 / **362.0** | **18.7** / 19.2 |
| rag-c1 | 67 / **74** | 320.8 / **233.1** | 332.4 / **242.3** | 332.4 / **242.3** | 14.0 / **12.3** |
| rag-c8 | 320 / **348** | 638.4 / **415.2** | 1285.1 / **702.2** | 1472.9 / **834.4** | **18.5** / 19.9 |
| rag-r1 | 112 / 113 | 323.5 / **237.8** | 468.4 / **261.7** | 536.3 / **362.7** | 15.0 / **13.9** |
| rag-r3 | 314 / **319** | 439.6 / **257.4** | 884.4 / **445.0** | 1080.0 / **511.8** | **19.4** / 21.6 |
| reason-c8 | 580 / **594** | **26.9** / 40.4 | **27.3** / 40.8 | **27.9** / 40.8 | 12.4 / 12.4 |
| reason-c32 | **2098** / 2059 | **28.9** / 46.6 | **30.6** / 73.2 | **32.7** / 74.4 | **13.6** / 13.7 |

No request failed in any cell. Riley's inter-token p99 stays lower under long
prompts (rag-c8 23.8 against 128.5 ms), because prompt work is cut into steps
that never stall decoding.

- **Short prompts:** Riley leads on every TTFT percentile, closed or open loop.
  The older note that tail TTFT at 32 clients trails vLLM does not hold on this
  workload.
- **Prompts of 400 tokens:** median TTFT leads, tail TTFT does not.
- **Prompts of 2,000 tokens or more:** this is the remaining gap. vLLM reaches
  the first token about 1.4× sooner at one client and 2× sooner at the p95
  under load, and at one client it also decodes faster at this context length
  (TPOT 12.3 against 14.0 ms).

Raising `--batch-token-budget` to 512 narrows the RAG gap (rag-c1 TTFT 296,
rag-r3 316 ms) but raises short-prompt TTFT (closed-c32 p50 40 to 72, closed-c8
32 to 46 ms). The default stays 256; use 512 only for a server that mostly
receives long prompts.

## Long context

`native-bf16-varlen-v1` used to stop at 1,024 tokens and needed the slower
`page-v1` attention beyond that. It now reaches 8,192 tokens and a step budget
of 512. Compared with `page-v1` on the same cells:

| Cell | `page-v1` | Now |
|---|---:|---:|
| rag-c1 TTFT p50 | 618 | 321 |
| rag-c8 tok/s | 196 | 320 |
| rag-r3 TTFT p50 | 6041 | 440 |
| reason-c32 tok/s | 1877 | 2098 |

Output correctness, 18 prompts of 670 to 6,080 tokens, 32 output tokens:

| | Same tokens at C8 | Same tokens at C32 | Same text as HF BF16 | Same first token as HF |
|---|---:|---:|---:|---:|
| Riley, Qwen2.5-7B | **54/54** | **54/54** | 10/18 | 17/18 |
| vLLM, Qwen2.5-7B | 40/54 | 39/54 | **16/18** | 17/18 |
| Riley, Qwen3-4B | **54/54** | **54/54** | 10/18 | 17/18 |

Riley's output does not depend on the batch; vLLM's changed in about a quarter
of these runs. Riley's agreement with HF on long prompts is below vLLM's on
Qwen2.5-7B. The sample is small and the cause is not established, so treat this
as open.

KV memory grows with the context: see
[configuration](configuration.md#capacity). Qwen3-4B at 32 sequences of 8,192
tokens does not fit a 40 GB GPU without `--kv-blocks`; the row above ran with
`--kv-blocks 6000`.

## Structured output and log probabilities

Requests with a grammar need no logits unless the GPU's choice breaks the
grammar. The token picked on the GPU is checked against the grammar first, and
only a rejected token costs a row download. Requests with `logprobs` always
download their row, and the full distribution is computed on the CPU.

Throughput tok/s (TPOT p50 ms), 64 output tokens:

| Cell | Riley before the check | Riley | vLLM |
|---|---:|---:|---:|
| plain, 8 clients | 651 (11.97) | **651** (11.97) | 621 (12.21) |
| schema, 8 clients | 513 (15.09) | **624** (12.34) | 605 (12.26) |
| logprobs, 8 clients | 360 (21.77) | 369 (21.27) | **610** (12.48) |
| plain, 32 clients | 2245 (13.58) | **2252** (13.54) | 2095 (13.55) |
| schema, 32 clients | 1194 (25.74) | 2048 (14.73) | **2087** (13.52) |
| 1 schema + 31 plain | 1970 (13.97) | 2025 (13.58) | not measured |

Schema at 32 clients is 2 % behind vLLM on throughput and ahead on TTFT p95
(65.5 against about 173 ms). One schema client among plain clients costs the
others almost nothing (TPOT 13.58 against 13.54 ms).

The check changes speed, not output: 32 schema requests at 8 clients, half with
`logprobs`, returned identical text with and without it.

`logprobs` remain 1.7× slower than vLLM at 8 clients: the row download and
CPU work add about 1.2 ms per row per step. Use them for evaluation rather than
for production traffic.

Accuracy of `logprobs` against Hugging Face:

| Model | Absolute error p50 | p99 | Top-5 overlap |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 0.0041 | 0.119 | 98 % |
| Qwen3-4B | 0.0004 | 0.157 | 99 % |

## API checks

`benchmarks/live/api_check.py` runs 40 checks against a live server: chat and
completions, streaming and `include_usage`, stop handling, `logprobs`,
`json_object`, schemas (including refused keywords), every `tool_choice`, a
streamed tool call, and a tool result round trip.

| Model | Tool style | Result |
|---|---|---|
| Qwen2.5-7B | Hermes | 40/40 |
| Qwen3-4B | Hermes | 40/40 |
| Qwen2.5-32B (two GPUs) | Hermes | 40/40 |
| Llama-3.2-1B | Llama JSON | 39/40 |

The Llama failure is the round trip: the model calls the tool again rather than
answering. Hugging Face generates the same call from the same rendered prompt,
so this is the model, not the server.

## LoRA

`riley-checkpoint merge-lora` with a test adapter on Qwen2.5-0.5B: 290 tensors, 494 million
values, of which 1,006 differ from a PEFT `merge_and_unload()` by at most 2.4e-4.
Served text matched PEFT on 4 of 6 prompts, against 5 of 6 for the base model
and Hugging Face, so the merge adds no visible error on this small sample. Only
static merging exists; adapters cannot be switched at run time.

## Stability

30 minutes of mixed traffic at 8 requests/s on Qwen2.5-7B with
`--max-sequence-tokens 8192`: 14,378 requests of plain, schema, tool, logprob,
long-prompt and long-output kinds, plus 758 clients that disconnected mid-stream.

- No failed or malformed response, and no overload.
- Resident memory went from 15,140 to 15,164 MiB and did not grow in the second
  half.
- At the end no request was active and no KV block stayed allocated.

A further 10 minutes on the final build (4,806 requests, 250 disconnects) gave
the same result.

## Two GPUs

Qwen2.5-32B-Instruct, `--tensor-parallel-devices 0,1`, 72-token prompts, 64
output tokens, 96 requests, two runs each. Riley / vLLM:

| Clients | Throughput tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| 1 | **36.1** / 35.3 | **36.9** / 41.3 | **40.0** / 43.4 | **27.6** / 28.1 |
| 8 | **270.8** / 265.4 | **82.9** / 141.7 | **94.9** / 160.7 | 28.6 / **28.3** |
| 32 | **914.3** / 873.3 | **98.6** / 318.7 | **99.6** / 421.0 | 33.0 / **31.7** |

Throughput and TTFT lead at every concurrency; TPOT trails at 8 and 32 clients.

## What is not covered

- Four and eight GPUs, and more than one host: only two GPUs were available.
- Mixture-of-experts models, vision input, quantized weights and speculative
  decoding. These need new kernels, not serving changes, and remain unsupported.
- Tool calling for models outside the Hermes and Llama JSON formats.
- A production trace. All traffic here is synthetic.
