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

### Qualified native attention · 2026-10-24

Qwen3-4B BF16 greedy on A100: **90 fresh-process arms**, five repeats per
engine in six cells, 64 warm-up + 192 measured requests. Each cell compares
page-native, fixed-tile native and stock vLLM 0.27.1 sequentially on the same
physical GPU. Values are run-level medians; change is the median paired
change against page-native. Native/stock token budgets are 256/576, context
is 1024, and prefix caching is disabled. All preregistered G1–G5 gates pass.

| Workload | Clients | Page-native tok/s | Fixed-tile native tok/s | vLLM tok/s | Native / vLLM | Paired change |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 1 | 123.8 | 124.8 | 115.6 | 1.079x | +0.56% |
| Fixed | 8 | 716.9 | 767.2 | 710.5 | 1.080x | +7.02% |
| Fixed | 16 | 1155.7 | 1285.2 | 1129.8 | 1.138x | +11.45% |
| Fixed | 32 | 1696.1 | 1940.3 | 1686.9 | 1.150x | +14.27% |
| Natural | 8 | 817.9 | 847.7 | 830.5 | 1.021x | +3.63% |
| Natural | 32 | 2118.3 | 2370.9 | 2303.3 | 1.029x | +11.98% |

All 30 candidate arms match their own canonical outputs for **5760/5760**
measured requests, with zero cross-concurrency mismatches. HF-reference
correctness is 12/15 (36/45 with concurrent repeats). This is a separate
numerics mode: measured natural-workload HF token-exact agreement changes
from 128/192 to 64/192; HF-literal validation uses a separate profile.

High-concurrency TTFT still trails vLLM, despite better throughput and TPOT:

| C32 workload | Native TPOT p50 ms | vLLM TPOT p50 ms | Native TTFT p95 ms | vLLM TTFT p95 ms |
|---|---:|---:|---:|---:|
| Fixed | 13.35 | 15.75 | 226.2 | 162.8 |
| Natural | 11.59 | 11.99 | 248.0 | 223.3 |

Select the optimized route with `--inference-profile native-bf16-varlen-v1`
(head dimension 128, context ≤1024, batch budget ≤256, active sequences ≤32).
It fixes the query tile at 64, uses no KV split, and disables scalar FMA
contraction in the FlashInfer translation unit to preserve batch invariance.
`native-bf16-v1` explicitly selects the page path. Both profiles include the
published benchmark defaults; explicit options remain subject to validation.
The ordinary server invocation retains its existing defaults.

[Implementation, all gates, latency, accuracy and evidence](docs/phase2-flashinfer-resume-20261024.md).

### Native BF16 vs vLLM · 2026-10-22 baseline

Qwen3-4B on one A100 GPU7, BF16 greedy, three fresh processes per
engine/cell, 64 warm-up + 192 measured requests; run-level median tok/s.
Native BF16 preserves batch-invariant tokens; HF-literal is a separate,
slower validation profile. These numbers require the explicit
[native flags](benchmarks/evidence/20261022-phase0/native-extra.json),
not the default server invocation. vLLM is stock 0.27.1 with its normal
compile/CUDA Graph path. Native/stock token budgets are 256/576.

| Workload | Clients | Riley native tok/s | vLLM tok/s | Riley / vLLM |
|---|---:|---:|---:|---:|
| Fixed output | 1 | 124.0 | 116.3 | 1.066x |
| Fixed output | 8 | 717.8 | 719.3 | 0.998x |
| Fixed output | 16 | 1153.2 | 1148.2 | 1.004x |
| Fixed output | 32 | 1698.5 | 1683.5 | 1.009x |
| Natural output | 8 | 818.2 | 828.5 | 0.988x |
| Natural output | 32 | 2106.7 | 2304.7 | 0.914x |

At this baseline, the native gap was long-context/mixed attention and burst TTFT.
The concurrency-scaled prefill burst policy passed invariance and throughput
non-inferiority checks, but missed its preregistered fixed-C32 TTFT goal
(-5.28% vs the required -10%); it remains opt-in. Fused, segment, direct and
hybrid attention experiments also missed their overall adoption gates.
[Baseline, latency and accuracy](docs/phase0-contract-cost-and-aa-20261022.md) ·
[Burst and attention qualification](docs/phase1-native-attention-and-burst-20261022.md).

### Weight-layout operator experiment · 2026-10-22

A cold K-major Gate/Up weight layout reduces operator latency by **4.28–5.07%**
for R2–16 across actual Qwen layers 0/17/35; all 3,348 numerical cases preserve
literal BF16 outputs. Down packing is slower, and none of 70 transposed-weight
cuBLASLt candidates is both exact and faster. Additional tile/stage variants do
not beat the selected layout. These are operator results. The server experiments
are complete: v19A passes 7,760 literal BF16 files; capacity-aware v19B passes
10,096. Fresh matched results and the unchanged tail-gate decision are below. [Implementation, full operator table and evidence](docs/qwen-weight-layout-v19-screen-20261022.md).

A separate lossless BF16 format reduces Gate/Up/Down stored weight bytes by
17.58%/16.41%, but decoding makes Gate/Up 31–35% slower and Down about twice
as slow. All 360 operator comparisons are exact; this candidate is rejected.
[Lossless format implementation and results](docs/qwen-lossless-projection-v19-20261022.md).

### Latest capacity-aware comparison · 2026-10-22

v19B preserves accuracy but does **not** meet the preregistered adoption gates;
the accepted runtime remains v15C. Its exact qualified patch is retained.
C16 enables packed Gate/Up with 2.96875 GiB extra weights; C32 uses the original
projection path with no packed allocation or launches. Each engine/concurrency
has five fresh processes on the same physical GPU7, 64 warm-up + 192 measured requests; values are
run-level medians. No completed normal arm is excluded or repeated.

| C | Matched v15C tok/s | v19B tok/s | Change | vLLM tok/s | v19B/vLLM |
|---|---:|---:|---:|---:|---:|
| 16 | 749.718 | 758.596 | +1.18% | 1147.495 | 66.11% |
| 32 | 980.833 | 976.130 | -0.48% | 1688.063 | 57.83% |

| Clients | Metric ms | v15C | v19B | vLLM |
|---|---|---:|---:|---:|
| 16 | TTFT p95 / p99 | 79.808 / 163.501 | 81.099 / 162.441 | 84.171 / 111.642 |
| 16 | TPOT p95 / p99 | 19.999 / 21.457 | 19.774 / 21.082 | 12.671 / 13.328 |
| 32 | TTFT p95 / p99 | 250.153 / 334.932 | 253.244 / 340.307 | 165.031 / 218.328 |
| 32 | TPOT p95 / p99 | 31.664 / 38.560 | 31.715 / 38.285 | 17.341 / 22.687 |

Tail flags (>2% paired TTFT/TPOT p95/p99): C16=4,
C32=6. The original accuracy and tail
thresholds are unchanged. 524 CPU tests and 10,096 literal BF16 files pass.
C32 differences are baseline-path variation, not a packed-kernel speedup.
The large throughput improvement target remains unmet.
[Implementation, every pair and evidence](docs/qwen-weight-layout-v19b-20261022.md).

### Latest structural experiments · 2026-10-22

The candidates below passed the original accuracy gates but were **not adopted**.
Each row uses its own fresh five-run matched v15C/candidate/stock campaign on
GPU7; values are run-level medians. These are experimental results, with the
accepted v15C runtime unchanged.

| Experiment | Clients | Matched v15C tok/s | Candidate tok/s | Change | vLLM tok/s | Candidate / vLLM |
|---|---:|---:|---:|---:|---:|---:|
| Multi-prefill Graph v18A | 16 | 748.663 | 739.910 | -1.17% | 1136.356 | 65.11% |
| Multi-prefill Graph v18A | 32 | 981.234 | 981.079 | -0.02% | 1684.366 | 58.25% |
| CPU–GPU overlap v18B4 | 16 | 756.053 | 753.877 | -0.29% | 1153.558 | 65.35% |
| CPU–GPU overlap v18B4 | 32 | 992.120 | 989.579 | -0.26% | 1680.772 | 58.88% |
| Packed Gate/Up v19A | 16 | 732.961 | 751.208 | +2.49% | 1140.859 | 65.85% |
| Packed Gate/Up v19A | 32 | 979.213 | 977.470 | -0.18% | 1667.614 | 58.61% |

v18A supports multi-prefill/variable-R MLP Graph pieces and passes 9,600 literal
BF16 files. v18B4 overlaps real result processing/planning with GPU execution
and passes 7,552 files plus 530 CPU tests. Actual hidden CPU result work is only
2.815/1.003 ms in the separate C16/C32 profile windows. Both retain paired tail
regressions; neither closes the throughput gap. All completed normal arms are
retained; supplemental inspected profiles are separate from timing results.

[Graph implementation and evidence](docs/qwen-piecewise-v18a-20261021.md) ·
[Overlap implementation, all repeats and caveats](docs/qwen-overlap-v18b4-20261021.md).
[Packed backend implementation, memory cost and all pairs](docs/qwen-weight-layout-v19a-20261022.md).

### A100 · Qwen3-4B · accepted v15C adoption comparison · 2026-10-21

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
| v17B4 mixed/overlap | 16 | 746.287 | 748.243 | +0.26% | 1145.795 | 65.30% | Rejected |
| v17B4 mixed/overlap | 32 | 975.412 | 985.618 | +1.05% | 1679.361 | 58.69% | Rejected |
| v17B5 fixed metadata | 16 | 738.330 | 744.243 | +0.80% | 1148.787 | 64.79% | Rejected |
| v17B5 fixed metadata | 32 | 980.503 | 986.012 | +0.56% | 1680.887 | 58.66% | Rejected |

The original literal native/HF BF16 references and paired TTFT/TPOT p95/p99
2% tail gates are retained. “Criteria pass” in an experiment receipt is
qualification; application to main is recorded separately. The projection candidate and both mixed/overlap revisions fail the original paired tail gate, so the candidates are not combined. Main remains v15C; a large throughput gain is not established.

B5 executes mixed full-body Graphs, but only 4/1 complete launches were observed in the separate C16/C32 profiles (5.55%/1.84% of kernel sums). Committed-event delivery overlap does not implement a whole one-batch-ahead scheduler. These profile fractions are not normal-run causal speedups.

A separate host-counter audit found multi-prefill in 48/89 (C16) and 41/77 (C32) mixed steps; only 8/89 and 5/77 selected the prepared cache. These post-startup whole-process counts include warmup/recovery and are not normal-run hit rates. The diagnostic binary passed224 original native/SSE value checks; its timings are excluded. [Shape audit and scope](docs/qwen-overlap-v17b5-20261021.md#추가-shape-계수-진단)

[projection-v17a report](docs/qwen-projection-v17a-20261021.md) · [overlap-v17b4 report](docs/qwen-overlap-v17b4-20261021.md) · [overlap-v17b5 report](docs/qwen-overlap-v17b5-20261021.md)

<!-- end v17 independent experiments -->

### A100 · model coverage against vLLM 0.27.1 · 2026-10-28

Eight cells (C1/C8/C16/C32 × fixed/natural), warm-up 32 and 192 measured
requests, Riley and vLLM interleaved twice in the same run on one A100-SXM4-40GB.
Pass = C8–C32 throughput ≥ vLLM and TPOT p50 ≤ vLLM, C1 within ±3 %, TTFT p50
≤ vLLM in every cell. Every model below serves over `/v1/completions` and gives
the same tokens for all 15 fixture prompts at C1, C8 and C32.

Two Riley configurations are used. *Fast* is the HF-literal `qwen3b-fast`
flag set plus the D1 flags. *Native* is the server defaults plus
`--inference-profile native-bf16-varlen-v1` (FlashInfer fixed-tile attention);
it changes the numerics, so its token checks were rerun. The native results
below are head dim 128; head dim 64 support is new and not yet qualified.

| Model | Config | Result | Throughput vs vLLM (8 cells) | Cells that miss (TPOT p50 vs vLLM) | 300 s stability |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | fast | **pass** | – | – | pass 8/8 |
| Qwen2.5-3B-Instruct | fast | **pass** | – | – | pass 8/8 |
| Qwen2.5-7B-Instruct | fast | **pass** | – | – | pass 8/8 |
| SmolLM2-135M | fast | **pass** | – | – | pass 8/8 |
| Qwen3-4B | native | **pass** | 1.01–1.15× | – | pass 8/8 |
| Qwen3-8B | native | **pass** | 0.99–1.13× (C1 tie) | – | pass 8/8 |
| Llama-3.1-8B-Instruct | native | **pass** | 0.98–1.15× (C1 tie) | – | pass 8/8 |
| Qwen2.5-Coder-7B-Instruct | native | **pass** | 1.05–1.28× | – | pass 8/8 |
| Qwen2.5-14B-Instruct | native | **pass** (vLLM at memory 0.9, a separate condition) | 1.00–1.11× | – | pass 8/8 |
| Qwen3-1.7B | native | miss | 0.99–1.51× | C1 fixed 1.06× | – |
| Llama-3.2-3B-Instruct | native | miss | 1.00–1.09× | C32 natural 1.002–1.014× (three repeats) | – |
| Qwen3-0.6B | native (screen) | miss | 1.00–2.12× | C1 fixed 1.11×, C1 natural 1.04× | – |
| Qwen2.5-0.5B-Instruct | fast | miss | – | C1 fixed 1.13× | pass 8/8 |
| SmolLM2-360M | fast | miss | – | C1 fixed 1.12×, C32 fixed 1.03× | – |
| Llama-3.2-1B-Instruct | fast | miss | – | C32 natural 1.06× | – |
| SmolLM2-1.7B | fast | miss | – | C16 natural 1.02×, C32 natural 1.11× | – |

TTFT p50 is 0.3–0.9× vLLM in every cell of the native runs; TTFT p95/p99 at
C32 stays above vLLM. FP32 HF greedy agreement of the native runs is no lower
than the earlier Riley configuration for every model (Qwen3-1.7B and Qwen3-8B
are one prompt below vLLM or HF BF16).

Qwen2.5-14B went from miss to pass after the wide RMSNorm path was extended
from hidden size 4096 to 8192: its residual RMSNorm took 2.05 ms per decode
step against vLLM's 0.47 ms. The change is bit-identical to the old kernel.

What is left: one-row (C1) kernel latency of the smallest models, closed with
evidence after two levers under 1 %; Llama-3.2-3B C32 natural, where no real
lever has been tried yet; and long-context decode attention for the
head-dim-64 models. The FlashInfer fixed-tile attention now also takes head
dim 64 and passes its GPU test (batch-invariant on five head-dim-64
geometries). A C32-natural screen of Llama-3.2-1B and SmolLM2-1.7B with it was
still running at the time of this commit. Full tables, flags,
evidence names and the rejected levers are in
[model coverage](docs/model-coverage-a100.md).

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
