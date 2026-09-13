# Adaptive decode projection model/serving integration — 2026-09-14

Model integration and four serving screens are complete. Throughput improves about4% versus the previous candidate at C8/C16, but C32 is only+0.64% and C64 is−0.06%. High-concurrency P99 is worse. The candidate stays opt-in; the overall vLLM goal is not achieved.

## Implemented batch

- QKV, attention-output projection and FFN down use one 16-row MMA tile when the device packet has ≤16 active rows, otherwise the existing two-tile arithmetic. This is the [previously validated native batch](../20260914-decode-adaptive-native/README.md), now connected to all 30 layers of pure decode. Mixed/prefill retains its existing implementation.
- Retained C ABI recorder and Rust session select the candidate explicitly. Paired/future decode clones the same compact decode graph, so it uses the selected kernels too. Existing parent retention, packet patching, settlement and cancellation contracts remain in effect. There is no additional workspace, host readback or weight/KV repacking.
- `--decode-projection adaptive-rows-experimental-v1` selects independent loopback V7 with required graphs. Existing prefill-FFN and paired decode may be combined. FA3/FlashInfer, decode FFN and wall-time-policy combinations are excluded. The default is `existing`. The graph catalog hashes the candidate source and a distinct profile marker.
- Benchmark/trace controllers can select the new option. When measuring this batch, `prior_paired` runs the saved previous binary with prefill-FFN + paired, and `paired` adds adaptive decode to that same configuration. This avoids accidentally attributing the older FFN improvement to the new batch.

Runtime remains Rust → C ABI → CUDA. Python runs only as an offline build/measurement tool.

## Correctness and build evidence

| Gate | Result |
|---|---|
| CPU numerical/config selection | 9 tests passed |
| CPU CLI/server | 36 tests passed |
| Native/Rust release model and server build | Passed |
| Independent synthetic free generation | 3,616 tokens; zero baseline differences; repeated-prompt invariance |
| Actual pure-decode row counts | Includes1,8,16,17,20,22,26,28,32 and partial drain shapes |
| Natural teacher-forced logits | 8 passages ×32 targets ×49,152 vocabulary: 12,582,912 BF16 values bitwise equal |
| Paired dependent model execution | Serial equivalence, first-step terminal/cancel handling, page reclamation and zero allocations passed |
| C32 serving smoke | Both lanes: 192 warmup +768 retained exact; stop96/cancel32 checks; normal exit |

Natural baseline/candidate dump SHA-256 both `492a46578581d131ab67c8c1cdb2d70f36a6539c868530e36269f1484f19f939`. The first test version passed active8/16 but requested unsupported scheduler capacity17. v2 instead admits17 requests under capacity32 and passes; the original failure log is retained. No numerical gate was relaxed. These fixtures are not a general model-quality corpus.

Prior native SM89 graph102/memcheck/racecheck and SM90a/SM100a compilation are documented in the linked native report. This integration ran on4090; Hopper/Blackwell/multi-GPU runtime remains unverified.

## Serving comparison

RTX4090, frozen SmolLM2-135M BF16 natural input16/128/398 and output32/64/128; context1024, KV2048 pages, token budget/chunk512. C8/C16/C32 use matching active capacity; clientC64 uses active32 on both engines. Each fresh lane has192 warmup +768 retained requests. Order is single → previous prefill-FFN paired → adaptive prefill-FFN paired → vLLM, then reverse. Throughput/P50 below are two-repeat medians; P95/P99 are the worse repeat. All percentiles and the single lane are in `evidence/{c8,c16,c32,c64}/comparison.json`.

| C / active | Path | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| 8 /8 | Previous candidate | 4,991.73 | 7.161 | 1.480 | 203.768 | 240.619 |
| 8 /8 | Adaptive | 5,187.38 | 7.002 | 1.426 | 196.721 | 219.879 |
| 8 /8 | vLLM | 4,563.70 | 11.129 | 1.543 | 234.131 | 292.508 |
| 16 /16 | Previous candidate | 7,818.86 | 8.487 | 1.851 | 261.665 | 372.105 |
| 16 /16 | Adaptive | 8,131.81 | 8.538 | 1.791 | 246.974 | 333.429 |
| 16 /16 | vLLM | 7,602.60 | 13.355 | 1.799 | 272.514 | 362.255 |
| 32 /32 | Previous candidate | 10,912.82 | 10.531 | 2.680 | 374.784 | 475.198 |
| 32 /32 | Adaptive | 10,982.46 | 10.594 | 2.656 | 374.783 | 489.113 |
| 32 /32 | vLLM | 11,330.53 | 22.085 | 2.356 | 389.460 | 453.361 |
| 64 /32 | Previous candidate | 11,128.61 | 216.933 | 2.724 | 599.825 | 719.133 |
| 64 /32 | Adaptive | 11,121.43 | 215.845 | 2.723 | 605.239 | 736.446 |
| 64 /32 | vLLM | 11,466.29 | 206.804 | 2.566 | 618.029 | 670.094 |

Candidate/previous throughput changes: C8+3.92% (orders+3.17/+4.67%), C16+4.00% (+2.83/+5.20%), C32+0.64% (+0.86/+0.42%), C64−0.06% (−1.60/+1.51%). C64 has no established throughput improvement. Candidate/vLLM throughput changes are +13.67%, +6.96%, −3.07%, −3.01%. C8 TPOT is about7.6% below vLLM; C16 is nearly equal. C32/C64 TPOT remains higher, and their P99 is higher than both previous candidate and vLLM. No broad promotion is supported by these two repeats.

All32 timed lanes exited0: 24,576 retained requests with zero protocol errors. All23,040 Riley warmup+retained requests match frozen references. vLLM retained reference matches are C8 1,197/1,536; C16 1,169/1,536; C32 1,118/1,536; C64 1,092/1,536. These cross-engine differences are retained and not treated as Riley bitwise parity. Every retained response, including vLLM and smoke, has its specified32/64/128 output tokens. The offline exporter recomputed every retained summary from the original client token/timestamp records and matched both completion and per-lane summaries before export.

GPU compute, build and profiler jobs did not overlap the timed runs. GUI was retained and Blender remains down. This is a closed-loop shared-host screen, not CPU/clock-isolated, open-loop or soak qualification. vLLM0.27.1 uses `--no-enable-prefix-caching`; Riley also has no prefix reuse here. Future cache-enabled comparisons must enable matching engine settings and remain separate from this cache-off baseline.

## Actual dispatch and provenance

The separate C8 trace observed472 ordinary and430 future decode graphs. Each executed exactly90 adaptive kernels (30 layers × QKV/output/down). Both traced lanes returned96/96 reference-exact responses, exited0 and left no owned process; peak profiler tree RSS was about1.53GB. `trace/adaptive-kernel-receipt.json` records kernel counts and the original SQLite hash. This proves dispatch, not a profiled throughput claim.

Candidate server SHA-256: `d85368b3bc3af73593f28cac2ea7bcb01f09c45f2888e6a5f4fb6be5c9101bab`; previous server: `3abb1af2281811909ca58ed306b163f676bd7fa0f57b9f4bfb32ec7b29ce72ff`. The source is `b5cb082c` plus this implementation; the remote recovery tree contains mirrored source files, not a clean git checkout. `evidence/source-hashes.json` records changed source hashes; the final receipt verifies they match the remote tree, native runtime dependencies resolve without Python/Torch, and no GPU compute process remains. Per-lane launch and preparation files record argv, controller/model hashes, GPU and workload conditions. Original full SSE responses/logs remain at `/data/riley-serving-260913-recovery/adaptive-decode-{smoke,c8,c16,c32,c64}-v1`; each exported folder includes a raw-file manifest. Compressed records retain token/text/usage/timestamps/checks and replace full frames with a canonical hash.

## Decision and next area

Retain the explicit experimental option; remove `--decode-projection adaptive-rows-experimental-v1` to return to the previous execution path. Do not pursue more tiny variants of this same projection family based on C32/C64 noise. The next structural area is PR10 KV prefix ownership and local transfer, with separate cache-hit/cache-miss workloads and matched vLLM caching settings. This does not resolve or discard the existing cache-off gap. The overall vLLM target, broader models/workloads, open-loop SLO, soak and multi-GPU validation remain open.
