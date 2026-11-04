# Serving validation

What was added for agent and RAG workloads on 2026-11-02, how it behaves under
load, and where it still trails vLLM. Everything here ran on A100-SXM4-40GB
GPUs of `aws-a100` with vLLM 0.27.1 as the baseline. vLLM runs on GPUs 4 and 5
of this host are capped near 20 tok/s for reasons not found, so every vLLM
number here comes from GPUs 6 and 7.
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
| rag-* | about 1,780 and 3,500 (1,500 and 3,000 words, alternating) | 128 | 16 to 64 requests |
| rag6k-c1 | about 3,500 and 5,800 (3,000 and 5,000 words, alternating) | 128 | 1 client, 16 requests |
| reason-cN | 72 | up to 1,024 (mean 837) | N clients |

Short cells ran with `--max-sequence-tokens 1024`; rag and reason with
`--max-sequence-tokens 8192`. Because prompt lengths alternate, the TTFT p50 of
rag-c1 and rag6k-c1 is the longer prompt.

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
| rag-c1 | 71 / **74** | 326.7 / **231.6** | 335.0 / **234.2** | 335.0 / **234.2** | 12.3 / 12.3 |
| rag-c8 | 339 / **348** | 590.6 / **375.1** | 1240.9 / **639.0** | 1454.0 / **771.0** | **17.6** / 20.0 |
| rag-r1 | 112 / 113 | 329.3 / **234.6** | 478.6 / **260.5** | 526.0 / **355.0** | **13.4** / 13.8 |
| rag-r3 | 317 / **319** | 416.8 / **260.3** | 845.9 / **449.7** | 1033.4 / **518.9** | **18.1** / 21.6 |
| rag6k-c1 | 64 / **68** | 560.1 / **395.0** | 565.3 / **402.5** | 565.3 / **402.5** | **12.4** / 12.5 |
| reason-c8 | 566 / **594** | **27.4** / 40.2 | **27.7** / 40.6 | **34.7** / 40.8 | 12.7 / **12.3** |
| reason-c32 | 2080 / 2078 | **29.2** / 46.5 | **33.2** / 67.8 | **35.0** / 68.8 | **13.6** / 13.7 |

No request failed in any cell. Riley's inter-token p99 stays lower under long
prompts (rag-c8 24.1 against 129.5 ms), because prompt work is cut into steps
that never stall decoding.

- **Short prompts:** Riley leads on every TTFT percentile, closed or open loop.
  The older note that tail TTFT at 32 clients trails vLLM does not hold on this
  workload.
- **Prompts of 400 tokens:** median TTFT leads, tail TTFT does not.
- **Prompts of 2,000 tokens or more:** this is the remaining gap. vLLM reaches
  the first token about 1.4× sooner at one client and 1.9× sooner at the p95
  under load; after the changes under Prompt speed, 1.25× and 1.8×. Decoding is no longer behind: TPOT is level at one client and
  lower under load.

### Prompt speed

TTFT at one client grows with prompt length, on GPU 6, median of 6 requests:

| Prompt tokens | 110 | 512 | 971 | 1,780 | 2,581 | 3,497 | 5,802 | Cost per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Riley, default (256 per step) | **16.1** | 57.6 | 88.5 | 161.2 | 249.0 | 331.6 | 566.0 | 96 µs |
| Riley, 128 per step | **16.0** | 75.9 | 128.2 | 231.7 | 348.4 | 469.7 | 794.4 | 136 µs |
| Riley, 512 per step | 16.3 | 53.4 | 82.2 | 152.6 | 233.9 | 310.3 | 535.1 | 90 µs |
| Riley, grid and merge (below) | 15.9 | 55.4 | 87.5 | 156.1 | 241.5 | 322.0 | 549.5 | 93 µs |
| Riley, vectorized kernels (below) | **15.4** | 52.6 | 82.5 | 148.6 | 231.0 | 307.0 | 526.0 | 89 µs |
| Riley, no LM head on prompt chunks (below) | **15.4** | 50.6 | 79.2 | 142.2 | 219.9 | 293.0 | 501.2 | 85 µs |
| vLLM | 20.9 | **44.1** | **70.2** | **115.2** | **176.6** | **234.5** | **395.9** | **66 µs** |

- The fit has an intercept near zero, so no fixed cost per request explains the
  gap. Riley leads only below about 200 tokens.
- Splitting attention does not help TTFT: rows of a prompt read the context
  written so far, and rag-c1 stayed at 324–350 ms for every split size tried.
- More tokens per step help little, and even 512 stays well above vLLM. The
  cost is in the forward pass per token, not in the number of steps.

Nsight Systems 2026.5.1 on one 2,629-token request with 4 output tokens
(`benchmarks/evidence/20261102-gap/prof/`) shows GPU kernel time of 271 ms for
Riley against 199 ms for vLLM. The GPU is idle for 7 and 5 ms, so host work does
not explain the gap. Riley runs 11 prompt steps of 256 rows; the vLLM server
default on A100 runs 2 of up to 2,048.

| Part | Riley ms | vLLM ms | Why Riley is slower |
|---|---:|---:|---|
| Projections (GEMM) | 213 | 173 | 256 rows fill the GPU poorly. Per layer at 256 rows: QKV 151 and o_proj 129 TFLOP/s, against 244 and 194 at 2,048 rows |
| Attention and merge | 34 | 14 | The attention costs 38 µs at a 256-token context, where the work is a few µs. The merge costs 8 µs per layer when every row has one part and 37 µs with two |
| RMSNorm, SiLU, RoPE and KV write | 25 | 12 | Scalar BF16 loads. Per 256 rows RMSNorm takes 13.5 µs (vLLM 5.7), SiLU 38.9 µs (21). Fixed below: now 12 ms |

Two changes to this attention followed, both writing the same bytes as before,
which a GPU test checks against FlashInfer's own launch and merge:

- **Strided grid.** The attention launched one block per possible work item:
  9,088 blocks for 256 rows on an 8,192-token server, most of which exit at
  once. It now launches at most the blocks that fit on the GPU, and they stride
  over the planned items. Attention went from 25.8 to 21.0 ms per request.
- **Merge.** FlashInfer's merge gives each row and head one block of 128
  threads; it now uses one thread per 8 values and replays the same fold. The
  merge went from 7.9 to 2.6 ms per request (20 to 6.7 µs a call).

Two runs each on GPUs 6 and 7, Riley only:

| Cell | Before | Strided grid | And merge |
|---|---:|---:|---:|
| rag-c1 TTFT p50 ms | 327.6 | 323.3 | **317.0** |
| rag6k-c1 TTFT p50 ms | 561.3 | 556.9 | **543.4** |
| rag-r3 TTFT p50 ms | 417.7 | 405.3 | **391.2** |
| rag-c8 TPOT p50 ms | 17.6 | 17.5 | **17.1** |
| reason-c8 tok/s | 565.7 | 565.4 | 565.7 |
| closed-c32 tok/s | 2256 | 2251 | 2256 |

TTFT at 3,497 tokens and one client went from 331.6 to 322.0 ms (vLLM 234.5),
or 96 to 93 µs per token. Each build matched the one before on 32 of 32
requests, passed the API checks 40/40 and gave the same 54/54 and 12/18 in the
long check. The short-row cost on a long-context server (reason-c8) did not
move with either change, so its cause is still unknown.

A third change rewrote the three element-wise kernels of each layer, each
keeping its arithmetic and reduction order, so bytes stay the same. GPU tests
compare every output against the kernels they replace, over 17 hidden sizes
for the norm and every BF16 gate pattern for SiLU:

- **RMSNorm.** The wide kernel used 1,024 threads per row, so only two blocks
  fit on an SM and 256 rows took two waves, and each warp loaded 64 bytes at a
  time. It now uses 256 threads with 16-byte loads, stages the row in shared
  memory per column and replays the same per-lane sums and tree. Hidden sizes
  that are not a multiple of 8, or unaligned rows, keep the wide kernel.
- **SiLU.** One thread now forms 8 outputs with 16-byte loads and stores.
- **RoPE and KV write.** One work item now forms two adjacent rotary pairs or
  two value elements with BF16x2 loads and 32-bit index arithmetic, and looks
  up the cache slot once.

Per call RMSNorm went from 11.6 to 5.8 µs, SiLU from 29.4 to 12.7 µs and
RoPE with the KV write from 10.8 to 6.7 µs, about vLLM's cost for the same
work. GPU kernel time per 2,629-token request went from 267.0 to 250.5 ms.
Two runs each on GPUs 6 and 7, Riley only:

| Cell | Grid and merge | Vectorized |
|---|---:|---:|
| rag-c1 TTFT p50 ms | 317.8 | **301.8** |
| rag6k-c1 TTFT p50 ms | 542.9 | **517.5** |
| rag-r3 TTFT p50 ms | 378.5 | **350.3** |
| rag-c8 tok/s | 344.6 | **358.8** |
| rag-c8 TPOT p50 ms | 17.2 | **16.6** |
| mid-r16 TTFT p50 ms | 63.9 | **59.5** |
| closed-c8 tok/s | 650.4 | **657.9** |
| closed-c32 tok/s | 2256 | **2313** |
| reason-c8 tok/s | 565.8 | **571.5** |
| reason-c32 tok/s | 2078 | **2109** |

The norm also runs on every decode step, so decode cells gained too. The build
matched the one before on 32 of 32 requests, passed the API checks 40/40 and
gave the same 54/54 and 12/18 in the long check (`vec/`).

A step that carries only part of a prompt emits no token, yet it ran the LM
head on all 256 rows: 1.07 ms per step for Qwen2.5-7B's 152,064-token
vocabulary, 10.7 ms of a 2,629-token request. Such a step now skips it; nothing
read those logits. GPU time per request went from 249.8 to 239.6 ms. Two runs
each on GPUs 6 and 7 (`lm/`):

| Cell | Before | No LM head on prompt chunks |
|---|---:|---:|
| rag-c1 TTFT p50 ms | 303.5 | **288.3** |
| rag6k-c1 TTFT p50 ms | 520.2 | **493.6** |
| closed-c32 tok/s | 2309 | 2311 |
| reason-c8 tok/s | 571.7 | 571.5 |

Outputs matched on 32 of 32 requests, the API checks passed 40/40 and the long
check gave the same 54/54 and 12/18 with the same common prefixes.

What remains is mostly the projections at 256 rows (203 ms of GEMM per request
against vLLM's 173) and attention (about 10 ms behind). Two ways to change
their numerics were measured and dropped:

- **Other GEMM anchors.** `benchmarks/profiling/gemm_anchor_search.cu`
  enumerates every cuBLASLt configuration Riley's recording contract accepts,
  groups row-count invariant ones by output bytes and finds, per row count,
  the fastest member of each group (`search/`). For gate|up, Q|K|V and down the
  current anchor's group already holds the fastest kernels at 256 rows. Only
  o_proj had a faster group on the probe, a split-K one, and anchoring on it
  made decoding slower in the server (TPOT 12.2 to 12.6 ms, closed-c32 2307 to
  2254 tok/s; `gemm/`). The probe's random data also misranks kernels: it put
  the gate|up anchor at 443 µs and another tiling at 302, while in the server
  they take 288 and 294. Search results need a server profile before use.
- **Smaller attention chunks.** `--attention-kv-chunk-tokens` 512 and 256 were
  slower than 1,024 (TTFT at 3,497 tokens 292.6 and 294.0 against 290.7 ms), and
  256 lowered agreement with HF (11/18, first token 17/18; `kc/`).

Larger prompt steps remain the lever for the GEMM part: 512 rows recover about
4 % for a lone long prompt (below), and a step of 1,024 rows fails on the GEMM
equivalence table (`NotSupported`), so it needs new verified GEMM shapes first.

A step of 512 needs `--batch-token-budget 512 --prefill-chunk-tokens 512` and
matching `--batch-shape-buckets`, and it raises short-prompt TTFT (closed-c32
p50 40 to 72, closed-c8 32 to 46 ms). The default stays 256; use 512 only for a
server that mostly receives long prompts.

Opening 512 only while nothing is decoding (`--prefill-burst-steady-budget 256
--prefill-burst-max-decode-rows 0`) still raised closed-c8 TTFT p50 from 31 to
44 ms: closed-loop clients with equal output lengths finish together, and the
next short prompts then share a 512-row step. `--prefill-burst-min-prompt-tokens
512` adds a second condition, a ready prompt with at least 512 tokens left. Two
runs each on GPUs 6 and 7, outputs identical on 32 of 32 requests (`b0/`, `b1/`):

| Cell | Steps of 256 | Only while idle | And a long prompt |
|---|---:|---:|---:|
| rag-c1 TTFT p50 ms | 302.7 | 288.1 | 291.1 |
| rag6k-c1 TTFT p50 ms | 519.3 | 498.9 | 496.7 |
| rag-c8 TPOT p50 ms | 16.7 | 16.7 | 16.8 |
| closed-c8 TTFT p50 ms | 31.1 | 44.2 | 32.0 |
| closed-c32 TTFT p95 ms | 38.4 | 56.5 | 38.4 |
| reason-c8 TTFT p50 ms | 26.9 | 27.5 | 27.5 |

With both conditions a long prompt alone gains about 4 % and short prompts lose
about 0.6 ms, which the 512-row shapes cost even when steps stay at 256. The
default stays as it is; the options suit a server whose traffic is mostly long
prompts from few clients.

## Long context

`native-bf16-varlen-v1` used to stop at 1,024 tokens and needed the slower
`page-v1` attention beyond that. It now reaches 8,192 tokens and a step budget
of 512. Compared with `page-v1` on the same cells:

| Cell | `page-v1` | Now |
|---|---:|---:|
| rag-c1 TTFT p50 | 618 | 327 |
| rag-c8 tok/s | 196 | 339 |
| rag-r3 TTFT p50 | 6041 | 417 |
| reason-c32 tok/s | 1877 | 2080 |

Without a split, every attention call scans the whole context, so decoding
slowed down as context grew (TPOT 14.05 ms at rag-c1 against vLLM's 12.3). The
context of each row is now cut at fixed token positions and the parts are merged
in order. The cuts depend only on the row, so output stays independent of the
batch.

`--attention-kv-chunk-tokens` sets the size. A split was rejected earlier
because it slows short rows, which those screens tested; on long context it
pays for itself. Qwen2.5-7B, two runs each, 8,192-token server:

| Split size | No split | 256 | 512 | 1,024 |
|---|---:|---:|---:|---:|
| rag-c1 TPOT ms | 14.05 | **11.99** | 12.01 | 12.33 |
| rag-c8 tok/s | 319 | 323 | 336 | **336** |
| rag-r3 TPOT ms | 19.55 | 19.23 | 18.26 | **18.09** |
| rag-r3 TTFT p50 ms | 452 | 472 | 441 | **412** |
| reason-c32 tok/s | 2097 | **2108** | 2094 | 2078 |

The default is 1,024:

- A row of up to 1,024 tokens is one part and keeps the same bytes, confirmed by
  a GPU test across query and KV head counts. A server with a context of 1,024
  or less skips the split path, so the speed tables of the model matrix still
  apply.
- 32 mixed requests matched the unsplit build, 32/32, at 8,192 tokens.
- On an 8,192-token server, requests under 1,024 tokens still pay about 1 %
  (reason-c32) to 2.4 % (reason-c8), because the step keeps room for rows with
  more parts. Use a smaller server context for short-only traffic.

Output correctness, 18 prompts of 670 to 6,080 tokens, 32 output tokens:

| | Same tokens at C8 | Same tokens at C32 | Same text as HF BF16 | Same first token as HF |
|---|---:|---:|---:|---:|
| Riley, Qwen2.5-7B | **54/54** | **54/54** | 12/18 | **18/18** |
| vLLM, Qwen2.5-7B | 40/54 | 39/54 | **16/18** | 17/18 |
| Riley, Qwen3-4B | **54/54** | **54/54** | 12/18 | 17/18 |

Riley's output does not depend on the batch; vLLM's changed in about a quarter
of these runs. The split moved agreement with HF from 10/18 to 12/18 on both
models, but it stays below vLLM's on Qwen2.5-7B. The sample is small and the
cause is not established, so treat this as open.

With two GPUs (`--tensor-parallel-devices`) the same check on Qwen2.5-7B gave
54/54 at both concurrencies and 13/18 identical text.

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

A further 10 minutes on the build with the KV split (4,806 requests, 250
disconnects) gave the same result.

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
