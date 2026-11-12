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
4 % for a lone long prompt (below). A step of 1,024 rows used to fail with
`NotSupported`; that is fixed now (next section).

#### Prompt steps of 1,024 rows

Two things stopped steps above 512 rows:

- **Workspace.** Qwen2.5-7B's down projection anchors on a split-K algorithm
  whose partial sums grow with the rows: 22 MB at 1,024 rows, over the 16 MiB
  workspace cap. Plans above 256 rows now get a cap scaled by
  ⌈rows / 256⌉. The anchor is still selected at 128 rows under 16 MiB, so no
  algorithm choice changes.
- **Attention.** `fixed-tile-unsplit-v1` refused budgets above 512. Its planner
  is one block of 1,024 threads, one per row, so the limit is now 1,024.

`gemm_anchor_search K N wide` (`benchmarks/profiling/`) finds the
configurations whose 512-, 1,024- and 2,048-row products reproduce the anchor's
256-row bytes on every row. The input repeats its first 256 rows, so a match
also shows that a row's bytes do not depend on its position. For Qwen2.5-7B,
102 configurations match per projection, and the fastest run at about 250
TFLOP/s at 2,048 rows, against 150–170 at 256 rows. The same search ran over
the 51 other projection shapes of the qualified models (`search/wide/`). The
fastest member per row range went into the A100 equivalence table. As before,
each entry is probed against the anchor's bytes when the plan is prepared.

Steps above 256 rows are no longer recorded as step graphs. At 1,024 rows the
kernels take about 70 ms, so a graph saves nothing. Recording them used up the
32-graph budget during start-up warm-up, and small decode shapes then ran
eagerly: rag-c1 TPOT rose from 12.24 to 12.63 ms until this was changed.

A budget of 1,024 on every step costs short prompts, as 512 did. The burst
options give long prompts the wide steps only:

```bash
--batch-token-budget 1024 --prefill-chunk-tokens 1024 \
--batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 \
--prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 \
--prefill-burst-min-prompt-tokens 512
```

Qwen2.5-7B on a server with `--max-sequence-tokens 8192`, two runs each on
GPU 7 (Riley; `burst5/`). vLLM is the same day's baseline on GPU 6:

| Cell | Steps of 256 | Burst to 1,024 | vLLM |
|---|---:|---:|---:|
| rag-c1 TTFT p50 ms | 286.0 | **254.8** | 235.5 |
| rag6k-c1 TTFT p50 ms | 491.1 | **445.2** | 395.5 |
| rag-c8 TTFT p95 ms | 1,110 | **876** | 640 |
| rag-r3 TTFT p95 ms | 709 | **522** | 457 |
| rag-c1 TPOT p50 ms | 12.24 | 12.24 | 12.3 |
| rag-c8 TPOT p50 ms | **16.6** | 19.6 | 20.0 |
| rag-c8 tok/s | **358** | 345 | 347 |
| reason-c8 tok/s | 572 | 572 | – |
| reason-c32 tok/s | 2,110 | 2,110 | – |

Long prompts reach their first token 11–26 % sooner. Under load, prompt chunks
of up to 1,024 rows share steps with the requests already decoding, so their
TPOT rises: rag-c8 goes from 16.6 to 19.6 ms (vLLM 20.0) and its throughput
drops 3.7 %, a tie with vLLM. The default therefore stays at 256. On a server
with `--max-sequence-tokens 1024` the same options cost short prompts
(mid-r16 TTFT p50 59.7 to 62.4 ms, closed-c32 37.3 to 38.3 ms), so use them on
long-context servers only.

Smaller shapes are anchored on the widest plan's algorithm. With a widest plan
of 1,024 rows that is a configuration chosen for 1,024 rows, and a decode
shape without its own table entry fell back to it. The smallest models lost
0.2 ms per token this way (Qwen2.5-0.5B rag-c1 TPOT 2.33 to 2.53 ms). A plan
above 256 rows now passes on the algorithm it was itself anchored to, and the
loss is gone (2.33 ms). Shapes of a widest plan up to 256 rows keep the
previous rule.

Six models on GPU 7, two runs each, Riley with steps of 256 and with the
burst; vLLM from the baseline on GPU 6. Ratios are Riley / vLLM for TTFT and
TPOT, vLLM / Riley for throughput (`models6/`, `models7/`):

| Model | rag-c1 TTFT | rag6k-c1 TTFT | rag-c8 TTFT p95 | rag-r3 TTFT p95 | rag-c8 tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.65 → 1.32 | 1.76 → 1.39 | 2.18 → 1.39 | 1.63 → 1.27 | 1.11 → 1.06 |
| Qwen3-0.6B | 1.72 → 1.37 | 1.75 → 1.38 | 1.99 → 1.26 | 2.07 → 1.39 | 1.08 → 1.03 |
| Llama-3.2-3B | 1.26 → 1.04 | 1.29 → 1.09 | 1.83 → 1.03 | 1.82 → 1.25 | 1.02 → 1.02 |
| Qwen3-4B | 1.35 → 1.22 | 1.38 → 1.29 | 1.59 → 1.26 | 1.79 → 1.34 | 1.01 → 1.06 |
| Qwen2.5-7B | 1.21 → 1.08 | 1.24 → 1.13 | 1.73 → 1.37 | 1.55 → 1.14 | 0.97 → 1.01 |
| Llama-3.1-8B | 1.22 → 1.08 | 1.22 → 1.11 | 1.94 → 1.36 | 1.83 → 1.25 | 1.00 → 1.04 |

Every model and setting returned the same text and log probabilities on 80 of
80 mixed requests. Models of 3B and more come within 1.04–1.29× at one client.
The smallest models stay furthest behind. For Qwen3-0.6B a 2,608-token request
spends 32.4 ms on the GPU: GEMM 16.1 ms, attention 8.8 ms (about 90 TFLOP/s,
half of what FlashAttention-2 reaches on an A100), and 4.0 ms in the Qwen3
head norm with RoPE and the KV write.

That head-norm kernel used one block of 128 threads and seven barriers per row
and head. It now uses one warp per head with four columns per lane. The lane
adds its columns in the block's tree order (offsets 64 and 32), and shuffles
finish from 16 down, so the bytes are the same; the GPU test against the
separate slice, norm, RoPE and cache-write launches passes. Qwen3-0.6B rag-c1
TTFT went from 64.6 to 62.5 ms and rag6k-c1 from 114.5 to 110.6 ms, and
outputs matched the previous build on 80 of 80 requests (`hn/`).

The attention partial-state workspace was sized for Riley's two-stage
attention, [rows, pages, heads, 130] FP32: 10.9 GB for a 40-head model at
1,024 rows and 8,192 tokens. `fixed-tile-unsplit-v1` uses it only for the
FlashInfer planner, so in that mode it is now sized to the planner (the opt-in
segment-decode and direct-prompt modes keep the full size). A fixed-tile step
that cannot plan now fails instead of running the other attention kernels,
whose bytes differ. With the burst options, GPU memory at start went from 37.5
to 30.3 GB for Qwen2.5-7B, 39.9 to 31.7 GB for Qwen3-4B and 8.7 to 5.1 GB for
Qwen2.5-0.5B, with 80 of 80 requests unchanged. All four 14B models now start
with the burst and the KV share vLLM gets at 0.9 utilization; before, they ran
out of memory even with a smaller cache (`lp10/`).

Outputs do not change with the step size: 32 of 32 structured requests and 80
of 80 logprob requests matched the steps-of-256 build. The long check gave
54/54 invariant at 8 and 32 clients and 12/18 equal to HF for both settings,
and the two settings agreed on 18 of 18 (`long2/`).

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

`logprobs` used to be 1.7× slower than vLLM at 8 clients: each requesting row
was downloaded whole (about 300 KB) and passed over on the CPU, about 1.2 ms
per row per step. A GPU kernel now summarizes every output row in one launch
after the step: the row maximum, the 16 partial sums of the normalizer, the
selected token's logit and the top tokens. The CPU finishes the log-softmax from
these few words. The kernel replays the CPU's arithmetic (the same exponential
approximation, operations and order per partial sum, no contracted multiply-add),
so the values are byte-identical. A GPU test compares every summary word with
the CPU over four shapes, including the 151,936-column row, and two servers
returned the same text and log probabilities on 80 of 80 mixed requests
(completions and chat, `top_logprobs` 0, 1, 5 and 20, with requests without
`logprobs` in the same batches; `lp/`). Requests with a grammar keep the
full-row path.

Qwen2.5-7B, 64 output tokens, two runs each on GPU 7, vLLM from the same day's
baseline on GPU 6 (tok/s, TPOT p50 ms):

| Cell | Riley before | Riley | vLLM |
|---|---:|---:|---:|
| plain, 8 clients | 659 (11.83) | 659 (11.82) | – |
| logprobs, 8 clients | 358 (21.9) | **626** (12.45) | 605 (12.47) |
| logprobs, 32 clients | 588 (53.3) | **2186** (13.98) | 2029 (13.73) |

Accuracy of `logprobs` against Hugging Face:

| Model | Absolute error p50 | p99 | Top-5 overlap |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 0.0041 | 0.119 | 98 % |
| Qwen3-4B | 0.0004 | 0.157 | 99 % |

## All 24 models

Each model on GPU 6, 2026-11-04/05: vLLM 0.27.1 and Riley `h6e7a` alternated
twice (`base0/`), then Riley with the GPU log-probability summaries, wider
prompt steps and the burst options twice more (`stage1/`; 14B models on the
build with the smaller attention workspace). The table in
`benchmarks/evidence/20261104-goal/stage1-table.md` has every model. Ranges,
Riley / vLLM (throughput vLLM / Riley), before → after:

| Cell | Before | After |
|---|---:|---:|
| rag-c1 TTFT p50 | 1.14–1.73 | 0.93–1.32 |
| rag6k-c1 TTFT p50 | 1.17–1.76 | 0.97–1.39 |
| rag-c8 TTFT p95 | 1.28–2.25 | 0.66–1.43 |
| rag-r3 TTFT p95 | 1.03–2.15 | 0.91–1.39 |
| rag-c8 TPOT p50 | 0.83–1.01 | 0.93–1.10 |
| rag-c8 tok/s | 0.97–1.12 | 0.88–1.06 |
| logprobs, 8 clients, tok/s | 1.37–3.68 | 0.62–1.00 |
| logprobs, 32 clients, tok/s | 2.24–8.64 | 0.45–0.98 |

Qwen2.5-14B rag-c8 TTFT p95 stays at about 1.4× (1,751 ms, vLLM 1,240). Letting
two prompts share a 1,024-row step (`--prefill-chunk-tokens 512`) left it at
1,746 ms and cost 7 % throughput (`p95/`), so with the burst itself this is two
levers under 2 % and the cell stays at the current structure's limit; the
remaining gap is the prompt cost per token. At 3 requests/s both engines are
overloaded with 14B (TTFT p95 above 13 s), so rag-r3 says little there.

Llama models and the 7B–14B Qwen models reach the first token within
1.09–1.12× of vLLM at one client; the small Qwen models (0.5B–3B) remain at
1.19–1.32×. Under load (rag-c8) the burst trades TPOT for TTFT: TPOT rises to
1.04–1.10× of vLLM's on the small models.

#### Tokenization

A Qwen3-0.6B profile of a 2,608-token prompt showed 31.6 ms of GPU work in a
request the client timed at 44.2 ms; Qwen2.5-1.5B showed 54.0 against 67.8.
The 13 ms outside the GPU was the tokenizer: a prompt over the context limit,
rejected right after tokenization, took 41 ms for 9,000 words and 92 ms for
20,000, about 4.5 µs a token. Riley's byte-level BPE merged strings looked up
in nested ordered maps and allocated a string per symbol. It now merges token
IDs through one hashed table of merge pairs, with the same order (lowest rank,
then leftmost pair), so the IDs are the same: identical on code, prose, Korean,
mixed bytes and a 3,000-letter run for the Qwen2.5, Qwen3, Llama 3 and SmolLM2
tokenizers, and a unit test compares it with the string engine on every
string of up to ten letters over a tie-heavy merge table. The same request
over the limit now takes 6.3 ms, the 2,600-token request 33.3 ms instead of
43.5, and served text is byte-identical (build `lp13`, `lp13/`).

Llama 3 tokenizers were already fast on words, since `ignore_merges` finds a
whole word in the vocabulary, which is why the Llama models led on long prompts
while the Qwen models of the same size did not. All 24 models again with the
burst options, Riley `lp13` against the vLLM runs of `base0/` (`stage1t/`;
every model in `benchmarks/evidence/20261104-goal/stage1t-table.md`):

| Cell | Burst before | With the tokenizer |
|---|---:|---:|
| rag-c1 TTFT p50 | 0.93–1.32 | 0.56–1.11 |
| rag6k-c1 TTFT p50 | 0.97–1.39 | 0.65–1.20 |
| rag-c8 TTFT p95 | 0.66–1.43 | 0.47–1.42 |
| rag-r3 TTFT p95 | 0.91–1.39 | 0.56–1.26 |
| rag-c8 TPOT p50 | 0.93–1.10 | 0.93–1.15 |
| rag-c8 tok/s | 0.88–1.06 | 0.83–1.05 |

The small Qwen models moved most: rag-c1 from 1.19–1.32 to 0.89–1.06. The
slowest first tokens are now Qwen3-4B (1.11), Qwen3-8B (1.10), Qwen3-14B and
DeepSeek-R1-Distill-Llama-8B (1.08). With requests admitted sooner, the small
models' rag-c8 TPOT rose by up to 0.09 (Qwen2.5-0.5B 1.15) as prompt steps
arrive closer together; their throughput is within 3 % of vLLM's.

What remains on the larger models is GPU work. A 2,608-token prompt keeps the
GPU busy for 221 ms on Qwen3-8B and 191 ms on Qwen2.5-7B, idle under 3 ms
(`prof12/`). Qwen3-8B's projections run at about 233 TFLOP/s against 248 for
Qwen2.5-7B: their anchors are split-K algorithms, and the 1,024-row
configurations that keep the anchors' bytes are split-K too, with a reduction
of about 100 µs per layer and step. Prompt attention takes 27 ms per request
(178 µs a layer at 1,024 rows, about 146 TFLOP/s). Neither changes without
changing outputs.

Under load the median first token leads and the tail does not: Qwen2.5-7B
rag-c8 TTFT p50 351 ms against vLLM's 407, p95 804 against 640. With 128
output tokens the 8 clients fall into step: a profile shows runs of 12–19
prompt steps, each about 67 ms (63 ms alone, so the decode rows they carry cost
little), then 105–127 decode steps (`profc8/`, `tail/`). The tail is the last
prompt of a wave, which waits for every prompt before it, so it follows the
prompt cost per token above. Two step-shape changes that keep outputs were
measured on Qwen2.5-7B and Qwen3-8B, Riley only, two runs each:

- Finer buckets above 256 rows (384, 640, 768 and 896 added; every model
  prepares them): a last chunk of 700 rows runs in 52 instead of 62 ms, but
  the cells moved by −0.4 to +0.3 % on rag-c1 and −3.0 / −4.6 % on rag6k-c1;
  rag-c8 p95 did not improve (`fineb13/`).
- Keeping the burst open for the last chunk of a long prompt that has started
  (it used to close once fewer than `--prefill-burst-min-prompt-tokens` were
  left, so the chunk ran in steps of 256), on Qwen2.5-7B, Qwen3-4B, Qwen3-8B
  and Llama-3.1-8B: rag-c1 −0.4 to −1.2 %, the other cells within ±1 % except
  Qwen3-8B rag-r3 p95 −2.0 %; output bytes the same (`ab14/`, build `lp14`).

Both stay under 2 % on the cells they target.

Short prompts, closed loop with 1, 8, 16 and 32 clients, 30-word prompts and 64
output tokens, Riley and vLLM alternated twice per model on GPU 7 (`short/`):
TPOT p50 is at most 1.00× of vLLM's at one client on every model, at most
1.03× at 8–32 clients on all cells but one (Qwen2.5-Coder-14B at 32 clients,
1.04×), and throughput at 32 clients is 1.04–1.72× vLLM's. For Qwen2.5-Coder-14B at 32
clients smaller steps trade throughput for TPOT: 26.49 / 26.14 ms with steps of
256 / 128 (vLLM 25.46), at 1,153 / 1,118 tok/s (vLLM 1,073) and TTFT p50 76 /
83 ms (vLLM 289); 192 was slower on every cell (`budget14/`). Neither lever
moved the cell by 2 %, so it stays as the current structure's limit.

### Decode at long context and the KV chunk

`fixed-tile-unsplit-v1` splits each row's context at fixed 1,024-token
positions. A decoding row is then planned as one or a few work items per KV
head, too few to fill the GPU, and decode at long context runs slower than
vLLM's: Qwen2.5-7B reason-c8 reached 571 tok/s against vLLM's 593. The
existing `--attention-kv-chunk-tokens` changes the split. Same GPU, Riley with
1,024 / 512 / 256 against vLLM (`chunk/`):

| Cell | 1,024 | 512 | 256 | vLLM |
|---|---:|---:|---:|---:|
| Qwen2.5-7B reason-c8 tok/s | 571 | 594 | **604** | 593 |
| Qwen2.5-7B reason-c32 tok/s | 2,112 | 2,137 | **2,163** | 2,104 |
| Qwen2.5-7B rag-c1 TPOT ms | 12.23 | **11.94** | 11.99 | 12.33 |
| Qwen2.5-0.5B rag-c1 TPOT ms | 2.33 | 2.12 | **2.05** | 2.08 |
| Qwen2.5-0.5B reason-c8 tok/s | 3,319 | 3,388 | **3,526** | 2,855 |

Three more models with 1,024 / 512 against vLLM:

| Cell | 1,024 | 512 | vLLM |
|---|---:|---:|---:|
| Qwen3-4B reason-c8 tok/s | 901 | **918** | 916 |
| Qwen3-4B reason-c32 tok/s | 2,853 | 2,902 | **2,971** |
| Llama-3.1-8B reason-c8 tok/s | 528 | 544 | **560** |
| Llama-3.1-8B reason-c32 tok/s | 1,860 | **1,878** | 1,794 |
| Qwen2.5-1.5B reason-c8 tok/s | 1,659 | 1,675 | **1,698** |
| Qwen2.5-1.5B rag-c1 TPOT ms | 4.41 | **4.05** | 4.53 |

A different split changes the attention arithmetic, so outputs change. The
long check stays invariant (54/54 at 8 and 32 clients) with 512 on all four
models tried. HF agreement moves by at most one continuation either way:
identical 12 → 13 (Qwen2.5-7B), 6 → 6 (Qwen2.5-0.5B), 12 → 11 (Qwen3-4B),
14 → 13 (Llama-3.1-8B) of 18 (`long512/`). A split of 256 adds little over 512 on Llama-3.1-8B
(reason-c8 543 tok/s, vLLM 560) and Qwen3-4B (reason-c32 2,922, vLLM 2,971),
costs 1–3 % TTFT at one client, and agrees with HF on 11 of 18 for both, so of
the two, 512 is the candidate. The rest of those two cells' gap is not in the
attention split. The default became 512 on
2026-11-05, with the user's approval; the re-validation of every model is below.

Re-validation with the 512 default (build `lp12`), every model on GPU 6, the
long check (`val512/`): all 24 models are invariant on 54 of 54 at 8 and at 32
clients. The first token matches HF on 17 or 18 of 18 prompts for every model,
and whole 32-token continuations on 6–15 of 18, depending on the model.

Speed with the 512 default, reason-c8 and reason-c32 against vLLM, alternated
twice on GPU 6 (`val512/speed/`, every model in
`benchmarks/evidence/20261104-goal/val512-speed-table.md`). Throughput
Riley / vLLM is 0.94–1.18 at 8 clients and 0.97–1.10 at 32 (14B models
with the KV share below); TPOT p50 is
0.94–1.04× of vLLM's at 8 clients and 0.91–1.06× at 32. Two cells stay more
than 3 % behind: Qwen2.5-Coder-3B (0.940) and Qwen2.5-3B (0.965) at 8 clients.
Both have two KV heads, so a decoding row has even fewer work items; a split
of 256 takes them to 1.041 and 1.022 (32 clients 1.070 and 1.048; `c256/`).

Since 2026-11-07, with the user's approval, models with two KV heads or fewer
split at 256: Qwen2.5-0.5B, -1.5B and -3B, Qwen2.5-Coder-1.5B and -3B, and
DeepSeek-R1-Distill-Qwen-1.5B. Splitting every row at 256 (build `lp15`,
`val256/`) also cost the prompt side: on the same build, alternating the two
chunks (`ab256/`), rag-c1 TTFT rose 3.9–4.1 %, rag6k-c1 5.5–6.4 % and rag-c8
TTFT p95 3.8–4.5 % on Qwen2.5-3B and Qwen2.5-Coder-1.5B. A prompt row has its
step's other query rows for parallel work, so more parts only add merge work.
Only generated rows now split at 256 and prompt rows keep 512 (build `lp16`):
the planner takes each request's chunk from the row kinds that the
prompt-row producer already receives, where a prompt row at position 256 or
later is marked whatever step it arrives in. A prompt row before position 256
lies within one part of either chunk, so a request that spans that position
has the same bytes whichever chunk it takes. The GPU test runs every
invariance case of `flashinfer_attention_gpu` with prompt rows at 512 and the
other rows at 256, and checks that each row has the bytes of its own chunk
alone.

Re-validation of `lp16` (`val16/`; GPU 7 was shared with another tenant's
training during the long check and GSM8K, which changes no bytes): all six are
invariant on 54 of 54 at 8 and at 32 clients, the first token matches HF on 17
or 18 of 18 prompts, and whole continuations:

| Model | HF identical, 512 | 256 everywhere (`lp15`) | 256 for generated rows (`lp16`) |
|---|---:|---:|---:|
| Qwen2.5-3B | 8 | 7 | 8 |
| Qwen2.5-Coder-3B | 9 | 10 | 11 |
| Qwen2.5-1.5B | 15 | 11 | 13 |
| Qwen2.5-Coder-1.5B | 8 | 6 | 9 |
| DeepSeek-R1-Distill-Qwen-1.5B | 14 | 13 | 13 |
| Qwen2.5-0.5B | 6 | 8 | 5 |

Throughput on `lp16` (`val16/speed`, order rvvr,
`benchmarks/evidence/20261104-goal/val16-speed-table.md`; 512 in brackets):

| Model | reason-c8 | reason-c32 | c8 TPOT | c32 TPOT |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 1.026 [0.965] | 1.042 [1.020] | 0.979 | 0.975 |
| Qwen2.5-Coder-3B | 1.039 [0.940] | 1.047 [1.067] | 0.986 | 0.970 |
| Qwen2.5-1.5B | 0.972 [0.997] | 1.055 [1.026] | 0.977 | 0.948 |
| Qwen2.5-Coder-1.5B | 1.087 [0.989] | 1.084 [1.036] | 0.986 | 0.968 |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.029 [0.995] | 1.008 [0.998] | 0.982 | 1.002 |
| Qwen2.5-0.5B | 1.214 [1.180] | 1.171 [1.085] | 0.906 | 0.888 |

Qwen2.5-1.5B at 8 clients is within the run-to-run spread: Riley's two runs
(1,644 and 1,641 tok/s) equal vLLM's second (1,645), its first was higher, and
Riley's TPOT is lower; the two engines also stop at different lengths.

The long-prompt cells with the burst options are at or below the `lp13`
values (`stage1t16/`, `benchmarks/evidence/20261104-goal/stage1t16-table.md`):
rag-c1 0.87–1.04, rag6k-c1 0.96–1.08, rag-c8 TPOT 1.06–1.15 as before.

The three 14B models with 48 layers first measured 0.80–0.87 at 32 clients
with a lower TPOT than vLLM's (0.95–0.96) and TTFT p95 of 13–27 s: requests
waited for KV space. The benchmark helper had given Riley 1,859 blocks (29,744
tokens), keeping 2.5 GiB free, where vLLM at 0.9 of the GPU holds 41,040
tokens; 32 requests of 30 + 1,024 tokens need 2,144 blocks. With 2,576 blocks,
about vLLM's tokens, Riley peaks at 36.9 GB against vLLM's 37.1 GB and
reason-c32 is 1.013 (Qwen2.5-14B), 1.031 (DeepSeek-R1-Distill-Qwen-14B) and
1.037 (Qwen2.5-Coder-14B; `kv2600/`). Qwen3-14B has 40 layers, fits the same requests in its blocks, and
was already 1.048. A server sized with `--kv-blocks` serves no more requests at
once than the blocks hold; Riley does not size the pool from free memory.

## Task accuracy

GSM8K on the current build (`benchmarks/live/gsm8k.py`: the 1,319 test
questions, the first four train examples as a 4-shot prompt, greedy, at most 256
tokens, stop at `Question:`, 32 requests at a time), Riley and vLLM 0.27.1 on
the same GPU, `--max-sequence-tokens 4096` (`gsm8k/`):

| Model | Riley | vLLM | Riley − vLLM |
|---|---:|---:|---:|
| Qwen2.5-7B | 84.46 % | 84.69 % | −0.23 |
| Llama-3.2-3B | 65.96 % | 65.43 % | +0.53 |
| Qwen3-4B | 85.44 % | 84.91 % | +0.53 |

With the KV chunk of 512 (the default since 2026-11-05): Qwen2.5-7B 84.99 %,
Llama-3.2-3B 65.66 %, Qwen3-4B 85.90 %, each at or above vLLM's (`gsm8k512/`).

On the models with two KV heads, generated rows split at 256 (`lp16`,
`val16/`; 256 everywhere, `lp15`, in brackets, `val256/`):

| Model | Riley | vLLM | Riley − vLLM |
|---|---:|---:|---:|
| Qwen2.5-3B | 69.83 % [70.05] | 70.58 % | −0.75 |
| Qwen2.5-Coder-3B | 67.32 % [67.25] | 67.70 % | −0.38 |
| Qwen2.5-1.5B | 59.67 % [60.96] | 60.80 % | −1.13 |
| Qwen2.5-Coder-1.5B | 47.92 % [48.22] | 47.76 % | +0.16 |
| DeepSeek-R1-Distill-Qwen-1.5B | 62.85 % [62.02] | 62.40 % | +0.45 |
| Qwen2.5-0.5B | 33.59 % [32.68] | 33.21 % | +0.38 |

DeepSeek-R1-Distill-Qwen-1.5B first measured against vLLM's 67.70 %. Riley
puts the begin-of-sentence token first, as `LlamaTokenizerFast` does for the
checkpoint's `add_bos_token: true`; the Transformers release in the vLLM
environment loads this tokenizer as `TokenizersBackend` and adds none (619
prompt tokens against Riley's 620 on all of 20 prompts). With the token written
at the front of each prompt, vLLM scores 62.40 %, the row above; Riley with
512 everywhere scores 62.55 %.

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

The same 30 minutes on the build with GPU log-probability summaries, wider
prompt steps and the 1,024-row burst options (14,378 requests, 758
disconnects, 1,120 with `logprobs`) also had no problem: resident memory grew
0.7 MiB over the second half, and no request or KV block was left at the end
(`stage3/`). The API checks passed 40/40 with and without the burst options.

## Two GPUs

Qwen2.5-32B-Instruct, `--tensor-parallel-devices 0,1`, 72-token prompts, 64
output tokens, 96 requests, two runs each. Riley / vLLM:

| Clients | Throughput tok/s | TTFT p50 ms | TTFT p95 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| 1 | **36.1** / 35.3 | **36.9** / 41.3 | **40.0** / 43.4 | **27.6** / 28.1 |
| 8 | **270.8** / 265.4 | **82.9** / 141.7 | **94.9** / 160.7 | 28.6 / **28.3** |
| 32 | **914.3** / 873.3 | **98.6** / 318.7 | **99.6** / 421.0 | 33.0 / **31.7** |

Throughput and TTFT lead at every concurrency; TPOT trails at 8 and 32 clients.

Again on 2026-11-05 with the current build (`compare.py --tp 2`, 30-word
prompts, 64 output tokens, 192 requests, Riley and vLLM alternated twice;
`tp32/`):

| Clients | Throughput tok/s | TTFT p50 ms | TPOT p50 ms | TPOT Riley / vLLM |
|---|---:|---:|---:|---:|
| 1 | **35.9** / 35.4 | **36.4** / 41.4 | **27.35** / 27.98 | 0.978 |
| 8 | **274.1** / 266.0 | **80.1** / 140.9 | 28.33 / **28.27** | 1.002 |
| 16 | **523.4** / 491.1 | **82.3** / 239.0 | 29.53 / **29.28** | 1.009 |
| 32 | **945.7** / 877.5 | **93.8** / 283.6 | **32.31** / 32.37 | 0.998 |

TPOT is within 1 % at every concurrency and ahead at 1 and 32 clients.

## What is not covered

- Four and eight GPUs, and more than one host: only two GPUs were available.
- Mixture-of-experts models, vision input, quantized weights and speculative
  decoding. These need new kernels, not serving changes, and remain unsupported.
- Tool calling for models outside the Hermes and Llama JSON formats.
- A production trace. All traffic here is synthetic.
