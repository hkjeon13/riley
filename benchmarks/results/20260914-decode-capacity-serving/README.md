# C8/C16 serving baseline and decode scope — 2026-09-14

The current prefill-FFN + paired candidate exceeds vLLM output throughput by **9.10% at C8** and **3.11% at C16** in this screen. C16 TPOT and P99 remain worse. This does not replace the prior C32/C64 failures or establish the overall serving goal.

## Matched conditions

RTX 4090, GUI retained, Blender down, no concurrent GPU compute or build/profiler during timed lanes. SmolLM2-135M BF16, frozen natural input16/128/398 and output32/64/128. Client concurrency and active capacity both 8 or both 16; token budget/chunk512, context1024, KV2048 pages. Each fresh lane: 192 warmup + 768 retained requests. Order: single, paired baseline, prefill-FFN paired, vLLM; then reverse. No wall-time-budget experiment. vLLM 0.27.1 uses the existing pinned environment and workload argv.

All Riley lanes use the same binary from `cf6df5e8`, SHA `3abb1af2281811909ca58ed306b163f676bd7fa0f57b9f4bfb32ec7b29ce72ff`. The `prior_paired` controller label denotes the **same binary without the FFN option**, not an older revision. This isolates option effects and preserves the current production code. FA3 is compiled but unselected; the new adaptive native candidate below is not in this server.

Throughput and P50 are medians of the two repeats; P95/P99 use the worse repeat. Full percentiles, launch argv, model/controller hashes and individual retained/warmup observations are in `evidence/c8` and `evidence/c16`. This is a closed-loop shared-host screen, not clock-locked/open-loop/soak qualification.

## C8

| Path | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| Single | 4,933.17 | 6.715 | 1.515 | 206.428 | 282.519 |
| Paired baseline | 4,979.37 | 7.412 | 1.486 | 204.484 | 224.608 |
| Prefill-FFN paired | 4,971.90 | 6.951 | 1.493 | 205.374 | 220.168 |
| vLLM | 4,557.01 | 11.109 | 1.547 | 229.013 | 286.442 |

Candidate versus vLLM: throughput +9.10%, TTFT −37.43%, TPOT −3.48%. Versus paired baseline throughput is −0.15%; no low-concurrency FFN improvement is established. [Aggregate](comparison-c8.json).

## C16

| Path | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| Single | 7,804.99 | 7.298 | 1.911 | 260.786 | 312.341 |
| Paired baseline | 7,757.91 | 8.256 | 1.877 | 260.923 | 363.491 |
| Prefill-FFN paired | 7,890.48 | 8.393 | 1.846 | 253.507 | 358.635 |
| vLLM | 7,652.80 | 13.689 | 1.801 | 280.141 | 348.503 |

Candidate versus vLLM: throughput +3.11%, TTFT −38.68%, TPOT +2.51%, P99 +2.91%. Versus paired baseline throughput +1.71%, but TTFT regresses. The candidate is not an all-metric promotion. [Aggregate](comparison-c16.json).

Across both screens: 12,288 retained requests, zero protocol errors; every lane exited 0. Riley warmup+retained: 11,520/11,520 frozen-reference matches. vLLM matches the Riley retained reference in 1,181/1,536 requests at each concurrency; cross-engine numerical/token differences are not hidden. These results are not a general model-quality gate. Raw frames remain remotely under `/data/riley-serving-260913-recovery/decode-capacity-c{8,16}-v1`; `evidence/raw-manifest.json` hashes all original files. Compressed request records retain token/text/usage/timestamps/checks and replace frames with a canonical hash.

## Separate C8 profile and next optimization batch

The bounded Nsight trace used 192 requests per lane, active8/client8, the same binary and frozen workload. The paired lane enables the actual prefill-FFN option. Both lanes matched all references, exited0 and left no owned processes. Peak paired profiler tree RSS was 1.56GB, below the existing 8GiB limit. The controller now accepts explicit active capacity and prefill-FFN selection; classification recognizes the prefill-FFN down kernel. Existing trace analysis tests: 3 passed. The first offline analysis failed because a helper was absent from the recovery tree; copying the existing helper and rerunning succeeded without rerunning the trace.

In the paired trace's middle80% by graph-launch count, decode has 1,373 launches and 1,758.435ms graph span; prefill/mixed has 145 launches and 582.634ms. Decode accounts for **75.11% of these graph spans**. Three decode projection groups consume 472.262ms, **29.04% of decode kernel duration**: QKV145.479ms, attention-output projection122.112ms, FFN-down204.671ms. See `trace/paired-areas.json` and `trace/paired-analysis.json`. Profiler overhead/client pacing remain; this is not an unprofiled speedup forecast or HBM measurement.

`shared32_projection_compute` calculates two 16-row MMA tiles even with ≤16 active rows. QKV, output projection and FFN down share that helper, so all three belong in one adaptive-row batch. Gate/up already skips inactive warps and is excluded. Keep K recurrence, partial BF16 rounding and 32-row scratch stride; use one tile for ≤16 active rows and preserve the two-tile path otherwise. Device-side dispatch avoids host metadata readback and extra graph identities per row count.

The [isolated native candidate](../20260914-decode-adaptive-native/README.md) passes exact partial/sanitizer checks and reduces these three operations' native time. Next connect it to ordinary and paired/future decode, bind the source/profile identity, verify full-model free generation and logits, then compare C8/C16 and C32/C64 against this baseline and vLLM. Do not promote it on the native result alone. Larger-model, long-context, high-concurrency tail and multi-GPU goals remain open.
