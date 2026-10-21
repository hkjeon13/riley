# Multi-prefill / variable-R piecewise Graph v18A

Completed 2026-10-22 on aws-a100. **Rejected for default integration**: measured throughput did not improve and ten paired TTFT/TPOT tail checks exceeded +2%. The accepted v15C runtime remains unchanged. All 30 completed normal arms are retained.

| Clients | v15C tok/s | Candidate tok/s | Change | Stock vLLM tok/s | Candidate / stock |
|---:|---:|---:|---:|---:|---:|
| 16 | 748.663 | 739.910 | -1.17% | 1136.356 | 65.11% |
| 32 | 981.234 | 981.079 | -0.02% | 1684.366 | 58.25% |

Five fresh processes per engine and concurrency, same physical GPU7 sequentially. Each arm has 64 warm and 192 measured requests, 12 fixed115–130-token prompts, balanced8/16/24/32 output limits,3840 output tokens. BF16 greedy, token budget576, sequence limit1024, prefix off. Riley selects owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1; stock vLLM 0.27.1 uses FA2 FULL_AND_PIECEWISE/version 2. Changes are ratios of five-run medians; paired tail flags remain separately recorded. TPOT is request-average output-token latency, not token-level ITL.

## Implemented candidate

The archived ten-file patch adds fixed-address prefill MLP pieces. A device U32 decode-prefix offset lets the same graph accept changing R without rebuilding for each request partition. Original prefill groups64–130 are copied into stable scratch; eligible combined prefill161–576 uses the existing projection arithmetic. Exact-P graphs cover all eligible P before service: 17,388 captures across 36 layers, sealed before traffic. Attention remains request-scoped eager execution; this is not a whole mixed-body Graph.

Original Gate/Up/Down selection and BF16 SiLU boundaries are preserved. Gather/scatter checks owners, context, alignment, aliasing and capacity; malicious device offsets produce no writes. Graph destruction precedes workspace/plan destruction. The serving cache misses equal startup construction misses in the inspected logs; invoked pieces were actual cache hits.

Source baseline `57502e7a0fafd165966071e88b3e8a56cdea0e3d`; candidate binary SHA256 `cc1e4ed77d79c51fe658f641b18305cdfbb7bbc63d8df9dedc06b40ff1aad2e6`. The patch reconstructs all 938 qualified source hashes, including the new prefill_piece.rs file.

## Verification and observed limitations

526 CPU tests passed (four original C02 fixtures rerun only in explicitly authorized private /tmp),40 ignored. Original five GPU contract functions and 120 alignment/view cases passed. 9,600 full-vocabulary BF16 files match the original native/Hugging Face reference literally; neither the oracle nor tolerance changed. A separate native ABI probe verified33 R=0..32 replays of one graph and 75 rejected invalid/busy calls, hostile device offsets and zero final allocations.

Separate Nsight before/after captures prove1,152 actual piece launches at C16 and 1,008 at C32, including P>130 multi-prefill replay. These pieces account for17.71% and 20.56% of candidate kernel-duration sums. Graph replay reduced host kernel-launch calls, but does not fuse the arithmetic: total profiled kernel sums remained roughly unchanged (C16 1664.94→1668.31 ms; C32 1323.48→1319.85 ms). Gate/Up plus Down still account for about44%/42% of candidate kernel time. Stable scratch adds copies. This identifies limited scope and unchanged arithmetic cost; it does not isolate a causal cost for each copy.

Profile GPU spans decreased while normal HTTP throughput did not improve. Profiler overhead differs with API/capture structure, so those spans are not evidence of end-to-end acceleration. CUDA synchronization API duration includes GPU waiting and is not isolated CPU execution time. No stock operator profile was collected; no direct stock-stage cost claim is made.

The supplementary first-C cohort audit observes initial TTFT p95 of 163.61→166.18 ms at C16 and 335.77→337.05 ms at C32; stock121.45/215.90 ms. The nearest overall TTFT-p95 request belongs to the initial cohort in all five runs for each engine/concurrency. These are SSE observations including scheduling/publication/network, not isolated GPU prefill latency.

All34 normal/profile arms passed the independent SSE/resource audit (6,490 stored SSE structures). Normal/profile whole-device peak21,089 MiB stayed below22,528 MiB. Original GPU6/7 holder PIDs, start ticks and CUDA contexts remained intact; idle holds retain20 GiB. Stock/native numerical agreements are reported per arm separately and are not an accuracy ranking.

Ten paired tail flags: six at C16 and four at C32, fully retained in final-statistics.json. C32 repeat 2 TPOT p99 increased9.74%. The plan disallows automatic adoption above 2%; negative throughput also makes further confirmation unjustified here. CPU–GPU overlap is an independent next experiment; combining this rejected path requires new evidence.

## References and evidence

[vLLM model runner v2](https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md) describes persistent batch metadata and asynchronous lifetime concerns. [SGLang overlap scheduling](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) motivates separately measuring actual overlap; this experiment imports neither runtime.

[Evidence index](../benchmarks/evidence/20261021-piecewise-v18a/README.md) · [Reconstructable candidate patch](../benchmarks/candidates/piecewise-v18a/candidate.patch).
