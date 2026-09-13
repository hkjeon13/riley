# FFN pipeline serving batch — 2026-09-13

**Model/serving integration and comparison completed; experimental backend not promoted.** The FFN pipeline preserves measured outputs and improves throughput by2.20–3.64% over V7 in this screen. It does not meet the complete vLLM serving goal. C32 natural throughput remains9.22% below vLLM, with21.21% longer median TPOT.

## Implementation and numerical evidence

Both gate/up/SwiGLU and down kernels are connected through one explicit Rust → C/C++ → CUDA path. Select `--graph-numerics variable-smol-v7 --execution-graph-policy require --ffn-backend pipeline-experimental-v1`. CLI selection is loopback-only and rejects FlashInfer combination. Existing V7 remains default. The native recorder retains existing parent/stream/extent checks, requires363 packed weights, and selects the two kernels together for pure decode. Mixed/prefill uses the original arithmetic. A distinct graph fingerprint hashes the FFN pipeline source and profile marker. Shared memory is kernel-local, without new device allocations.

- Full-model free generation: eight synthetic boundary prompts repeated across32 requests, lengths1/15/16/17/127/128/129/511,32 generated tokens each. **1,024/1,024 tokens equal**, repeated-prompt invariance, zero retained allocations.
- Natural teacher-forced full logits: eight fixed passages,32 target positions each,49,152 vocabulary entries. **12,582,912 BF16 values bitwise equal**, each dump25,165,824 bytes. Hashes are in final-verification JSON; original dumps remain under `/tmp/riley-opt-260912/ffn-natural-v1`.
- Full-model native memcheck: **0 errors**; native standalone racecheck/memcheck evidence remains in [the preceding batch](../20260913-ffn-pipeline-native/README.md). A full-model racecheck was not run.
-31 CLI tests and4 configuration-profile tests pass. CUDA release build passes.442 local/remote source/build/test files match. CPU tests do not establish GPU correctness; the GPU evidence above is separate.

These are measured bounded fixtures, not general task-quality certification. The previous FlashInfer numerical failures are neither modified nor waived by this independent exact-order backend.

## Same-condition serving protocol

SmolLM2-135M BF16, RTX4090, driver580.173.02, CUDA13.0 toolchain, vLLM0.27.1. Model/tokenizer file hashes are verified by the archived Round16 launch plan; fresh vLLM invocations and all argv are retained. GUI remains running; Blender remains stopped. Each process starts only after no foreign compute process is reported and the GPU is at most48°C. This is not a controlled CPU/graphics laboratory or a matched peak-memory claim.

The same frozen Riley executable, SHA256 `1002f9d8428ca1b0ac0eeb67602de28b6c1b5bb8fe574784787046139a1df74f`, runs both V7 and FFN; only the FFN flag changes. Source baseline before this patch is `8fa73afa`; full source hashes accompany the patch. C16/C32 admission matches client concurrency; token budget512, KV block counts matched between Riley modes, chunk128 for fixed and512 for natural, GPU greedy. vLLM uses CUDA graphs, BF16 and disabled prefix caching; its memory utilization setting is0.3. All engines serve the same prompt/output distributions. Natural lengths16/128/398 use32/64/128 output tokens; fixed uses the frozen diverse P128 corpus.

For each condition, run V7 → FFN → vLLM then reverse order. Each process has192 excluded warmups and768 retained streaming requests. Each table pools1,536 retained requests per engine. There are24 processes and23,040 total requests including warmups: **zero transport failures**, both Riley modes match the frozen reference on all15,360 of their requests. Output token totals agree across all three engines for every condition. vLLM token/text differences from the Riley reference are observations, not a vLLM quality-failure verdict.

Throughput divides summed successful output tokens by summed active measurement wall time. TTFT/TPOT/E2E use request-level token-arrival timestamps, with median and nearest-rank P95/P99. Initialization, warmup, gaps between processes and profiling are excluded. Two process repetitions do not establish statistical significance or long-duration tail stability. Per-run min–max throughput is reported alongside pooled figures; raw per-run summaries remain available.

## c16-fixed

| Metric | Previous V7 | FFN candidate | vLLM |
| --- | ---: | ---: | ---: |
| Output tokens/s | 6,125.71 | 6,273.78 | 5,005.57 |
| Requests/s | 306.29 | 313.69 | 250.28 |
| TTFT P50 / P95 / P99 (ms) | 5.893 / 8.036 / 20.458 | 6.229 / 8.576 / 18.296 | 18.125 / 29.488 / 35.618 |
| TPOT P50 / P95 / P99 (ms) | 2.399 / 2.599 / 2.714 | 2.321 / 2.542 / 2.677 | 2.278 / 3.012 / 3.547 |
| E2E P50 / P95 / P99 (ms) | 49.033 / 83.280 / 86.798 | 45.518 / 81.740 / 84.016 | 62.546 / 101.358 / 111.057 |
| Failed retained requests | 0 / 1,536 | 0 / 1,536 | 0 / 1,536 |
| Exact frozen Riley reference matches | 1536 / 1,536 | 1536 / 1,536 | 1430 / 1,536 |
| Peak GPU memory | Not measured | Not measured | Not measured |
| Per-run tokens/s min–max | 6,085.82–6,166.13 | 6,207.98–6,340.99 | 4,942.31–5,070.47 |

Candidate throughput: **+2.42% vs V7**, **+25.34% vs vLLM**. Median TPOT: -3.27% vs V7, +1.85% vs vLLM.

## c16-natural

| Metric | Previous V7 | FFN candidate | vLLM |
| --- | ---: | ---: | ---: |
| Output tokens/s | 7,904.87 | 8,079.13 | 7,696.61 |
| Requests/s | 105.87 | 108.20 | 103.08 |
| TTFT P50 / P95 / P99 (ms) | 6.894 / 11.057 / 30.784 | 6.847 / 11.365 / 23.674 | 13.211 / 22.928 / 35.291 |
| TPOT P50 / P95 / P99 (ms) | 1.923 / 2.008 / 2.062 | 1.883 / 1.952 / 1.981 | 1.855 / 2.147 / 2.403 |
| E2E P50 / P95 / P99 (ms) | 127.164 / 258.814 / 264.890 | 124.357 / 253.070 / 256.706 | 131.716 / 269.827 / 287.950 |
| Failed retained requests | 0 / 1,536 | 0 / 1,536 | 0 / 1,536 |
| Exact frozen Riley reference matches | 1536 / 1,536 | 1536 / 1,536 | 1154 / 1,536 |
| Peak GPU memory | Not measured | Not measured | Not measured |
| Per-run tokens/s min–max | 7,859.41–7,950.86 | 8,066.93–8,091.37 | 7,690.65–7,702.59 |

Candidate throughput: **+2.20% vs V7**, **+4.97% vs vLLM**. Median TPOT: -2.05% vs V7, +1.52% vs vLLM.

## c32-fixed

| Metric | Previous V7 | FFN candidate | vLLM |
| --- | ---: | ---: | ---: |
| Output tokens/s | 8,331.38 | 8,634.28 | 6,731.35 |
| Requests/s | 416.57 | 431.71 | 336.57 |
| TTFT P50 / P95 / P99 (ms) | 8.893 / 19.847 / 47.324 | 9.277 / 20.562 / 45.610 | 29.653 / 56.620 / 69.212 |
| TPOT P50 / P95 / P99 (ms) | 3.428 / 3.830 / 4.051 | 3.256 / 3.657 / 3.927 | 3.248 / 5.194 / 6.669 |
| E2E P50 / P95 / P99 (ms) | 72.397 / 122.276 / 133.102 | 70.847 / 118.140 / 124.536 | 88.356 / 160.204 / 186.320 |
| Failed retained requests | 0 / 1,536 | 0 / 1,536 | 0 / 1,536 |
| Exact frozen Riley reference matches | 1536 / 1,536 | 1536 / 1,536 | 1449 / 1,536 |
| Peak GPU memory | Not measured | Not measured | Not measured |
| Per-run tokens/s min–max | 8,221.45–8,444.28 | 8,614.93–8,653.71 | 6,634.43–6,831.15 |

Candidate throughput: **+3.64% vs V7**, **+28.27% vs vLLM**. Median TPOT: -5.01% vs V7, +0.26% vs vLLM.

## c32-natural

| Metric | Previous V7 | FFN candidate | vLLM |
| --- | ---: | ---: | ---: |
| Output tokens/s | 10,331.95 | 10,574.34 | 11,648.12 |
| Requests/s | 138.37 | 141.62 | 156.00 |
| TTFT P50 / P95 / P99 (ms) | 10.319 / 18.478 / 69.902 | 10.323 / 17.855 / 65.006 | 22.389 / 39.383 / 72.327 |
| TPOT P50 / P95 / P99 (ms) | 2.951 / 3.083 / 3.138 | 2.876 / 3.033 / 3.089 | 2.373 / 2.785 / 3.054 |
| E2E P50 / P95 / P99 (ms) | 196.133 / 397.293 / 404.222 | 191.897 / 390.719 / 399.599 | 174.081 / 355.666 / 373.527 |
| Failed retained requests | 0 / 1,536 | 0 / 1,536 | 0 / 1,536 |
| Exact frozen Riley reference matches | 1536 / 1,536 | 1536 / 1,536 | 1141 / 1,536 |
| Peak GPU memory | Not measured | Not measured | Not measured |
| Per-run tokens/s min–max | 10,304.45–10,359.60 | 10,563.14–10,585.56 | 11,472.52–11,829.18 |

Candidate throughput: **+2.35% vs V7**, **-9.22% vs vLLM**. Median TPOT: -2.54% vs V7, +21.21% vs vLLM.

## Trace and next decision

A separate96-request natural C32 node trace confirms the new `riley_ffn_pipeline::gate_up` and `down_parts` kernels execute. The graph-node census covers the middle80% of launches and includes profiler overhead. In that window, pure-decode FFN is29.39% of kernel time and attention35.27%; mixed/prefill attention is35.25%. Unlike the previous V7 trace, graph counts and input scheduling differ, so subtracting aggregate durations would not be a valid serving-speedup calculation. Full kernel names/durations and SQLite hashes are in `trace/areas.json`.

The pipeline changes pure-decode FFN only; attention, mixed/prefill and host gaps remain. That scope, together with the modest real-serving gains, does not justify repeated small pipeline tweaks as the next main track. Keep this version opt-in. Next prioritize the prefill/mixed attention and resource-policy work in PR03/PR04, starting from this trace and the existing research contracts. Do not call existing chunking a new feature or accept an arithmetic-changing backend without its numerical gate. Whole-layer persistent execution remains a separate broader roadmap item.

The final objective remains unachieved: natural C32 throughput and median TPOT fail the vLLM targets; multi-GPU/Hopper/Blackwell serving, burst/open-loop goodput, long-duration stability and broader model/workload coverage remain unverified. No pipeline default promotion is made. Rollback is `--ffn-backend existing` or omitting the flag.

## Evidence and reproduction

`comparison-*.json`, launch/summary/accounting/exit JSON, logs and `compact/` contain reconciled request-level data. The exporter removes repeated frame/text/request bodies but records their hashes; every source rows-file SHA256 is in `raw-manifest.json`. Full original rows remain at `/tmp/riley-opt-260912/ffn-serving-screen-v1` on `ai-assistant`. A first export attempted the old six-run exporter before the new version finished copying and was rejected; no benchmark was rerun or discarded. The corrected24-run export and all23,040 records were subsequently validated.

Reproduce with `benchmarks/analysis/ffn_pipeline_serving_screen.py ROOT FROZEN_BINARY NEW_OUTPUT_DIR 192 768`, using its archived launch plan/fixtures/client dependencies. Export with `export_flashinfer_serving_screen.py SOURCE DESTINATION 24`; summarize each condition with `summarize_flashinfer_serving_screen.py EXPORTED_DIRECTORY c32-natural ffn` (and the other three condition names). Final verification records no GPU compute processes after cleanup. vLLM shutdown warnings are preserved in logs; zero HTTP failures is not a host-resource leak certification. No Blender restore or system driver change occurred.
