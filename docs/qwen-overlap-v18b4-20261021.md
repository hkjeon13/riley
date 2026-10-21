# Actual CPU–GPU overlap v18B4

Completed 2026-10-22 on aws-a100. **Not adopted:** the new execution structure overlaps real CPU work with GPU kernels, but throughput is lower at both C16/C32 and six paired TTFT/TPOT tail checks exceed +2%. The accepted v15C runtime remains unchanged. The large throughput improvement objective remains unmet. The rejected v18A prefill MLP Graph was not combined with this candidate.

| Clients | Matched v15C tok/s | Candidate tok/s | Change | Stock vLLM tok/s | Candidate / stock |
|---:|---:|---:|---:|---:|---:|
| 16 | 756.053 | 753.877 | -0.29% | 1153.558 | 65.35% |
| 32 | 992.120 | 989.579 | -0.26% | 1680.772 | 58.88% |

Five fresh processes per engine/concurrency on the same physical GPU7, sequentially. Each normal arm has64 warm and 192 measured requests,3840 output tokens,12 fixed115–130-token prompts and balanced8/16/24/32 output limits. BF16 greedy, budget576, maximum sequence1024, prefix off. Riley selects owned-hf-eager-bf16-v1/native-prefill-sm80-v5/segments-v1; stock vLLM 0.27.1 uses FA2 FULL_AND_PIECEWISE/version 2. Changes are ratios of five-run medians, not medians of paired changes or pooled quantiles. Each experimental campaign has its own fresh matched baseline/stock samples.

Candidate / stock latency, milliseconds; medians of five per-run quantiles. TPOT is request-average output-token latency, not token-level ITL.

| Clients | TTFT p95 | TTFT p99 | TPOT p95 | TPOT p99 |
|---:|---:|---:|---:|---:|
| 16 | 80.665 / 82.308 | 161.642 / 109.747 | 19.917 / 12.473 | 21.387 / 13.171 |
| 32 | 247.449 / 162.142 | 335.223 / 215.653 | 31.417 / 17.490 | 37.804 / 22.634 |

## Implemented execution structure

After iteration N finishes its GPU execution and greedy-token download, the CPU constructs a conservative N+1 decode forecast using those exact token IDs. It excludes known finished/canceled rows, requires already-owned physical KV pages, checks context bounds and restricts the preview to the qualified native mode with configured maximum 2–32 rows. Partial prefill, stop strings, known pending work and queued admissions disable the preview. Active R=1 within a C16/C32 process is supported.

The affine CUDA executor, stream and timer move into local owners. A scoped callback runs after N+1 GPU submission and before completion waiting. It performs ordinary generation acceptance, result construction, authoritative commit of N, bounded publication of committed events and authoritative next planning. CPU planning uses the existing scheduler; CUDA owners remain alive throughout. No new inference thread or unbounded event queue is added.

The worker processes commands at its ordinary boundary. Completed preview outputs are reused only when the entire real IterationPlan equals the forecast: iteration/request IDs, order, tokens, positions, block IDs, valid lengths and output slots. A mismatch discards the preview and executes the real plan. Extra writes remain inside already-exclusive KV pages; another GPU batch cannot overwrite/reuse physical storage before the current completion guard returns. A queue/lifecycle hint is advisory and cannot replace exact validation.

CPU error paths do not assert device quiescence while the preview is pending. The enclosing GPU completion guard is observed before stored CPU errors propagate; shutdown then settles the reservation. Publication occurs only after current GPU work and authoritative commit. Unit tests exercise real queued-admission visibility, early publication before later completion, event order, no duplicate events and failure handling. Streaming cancellation/reuse is also tested on the actual server.

This is a bounded one-step forecast using downloaded exact tokens. It does not implement SGLang's GPU future-token map, persistent GPU-resident input metadata, or overlap all pre-submit packing/dispatch. New admissions still enter at worker boundaries. Source arithmetic and native projection/attention kernels remain the v15C paths.

## Actual overlap and remaining cost

The original four separate 64-request Nsight arms use native CPU NVTX ranges. Intersections with real CUPTI GPU-kernel intervals prove actual overlap at both concurrencies:

| Clients | Result ranges | CPU result range ms | Overlapped ms | Result range overlap | GPU kernel sum ms |
|---:|---:|---:|---:|---:|---:|
| 16 | 37 | 2.846 | 2.815 | 98.90% | 1651.868 |
| 32 | 18 | 1.022 | 1.003 | 98.11% | 1327.339 |

Authoritative planning is nested inside the result range: 0.648 ms overlaps GPU work at C16 and 0.209 ms at C32. Do not add these to the outer result times. The callback overlaps a small CPU phase: 2.815/1.003 ms within these diagnostic windows. This is consistent with negligible end-to-end benefit; it does not isolate the causal cost of each forecasting operation. Counters cover correctness/warm/recovery as well as timing, so they are not measured-window hit rates or proof of acceleration.

The original profiles requested external GEMM inspection, but the setup omitted two LD_PRELOAD libraries. Native CPU NVTX and overlap proof remained valid; eager GEMM classification was incomplete. Those four traces and loader errors are retained. Four **additional, separately audited** inspected profiles restore the exact libraries previously used in v18A. Their table below is diagnostic only; these samples never enter the 30 normal-arm throughput statistics.

Supplemental inspected candidate kernel-duration sums, milliseconds:

| Clients | Gate/Up | Down | KV projection | QK/PV | Gate/Up + Down / all kernels |
|---:|---:|---:|---:|---:|---:|
| 16 | 490.77 | 258.42 | 189.52 | 159.47 | 44.79% |
| 32 | 358.59 | 197.36 | 170.25 | 138.95 | 42.17% |

GPU arithmetic remains a substantial cost. This candidate leaves projection/attention kernels and pre-submit input preparation in place. The small callback budget, unchanged arithmetic and retained serial preparation explain the limited scope of the attempted improvement; no isolated ablation establishes a single sole cause of the small negative normal result. No stock operator trace was collected, so these are Riley costs, not a direct per-stage speed ratio against vLLM. API synchronization duration includes GPU waiting. Kernel-duration sums, GPU spans and busy fractions are not normal HTTP throughput or bandwidth-saturation evidence. Nsight's EFS temporary-path warnings are retained.

The first-C measured-request cohort after warm-up has initial TTFT p95 of 162.873→161.764 ms at C16 and 334.220→335.405 ms at C32; stock120.668/215.675 ms. These SSE observations include scheduling/publication/network and do not isolate GPU prefill. They do not establish that the initial-prefill gap is fixed.

Paired tail flags, zero-based repeat IDs:

- C16, repeat 0: ttft_ms_p95 +5.30%.
- C16, repeat 0: ttft_ms_p99 +2.75%.
- C16, repeat 3: ttft_ms_p99 +7.06%.
- C16, repeat 4: tpot_ms_p99 +2.32%.
- C32, repeat 0: tpot_ms_p99 +3.00%.
- C32, repeat 1: tpot_ms_p99 +11.52%.

The comprehensive plan forbids automatic adoption with any paired tail above 2% and permits investigation/confirmation for a promising candidate. Negative median throughput at both concurrencies does not justify additional confirmation here. No threshold was loosened and no favorable subset was selected.

## Qualification, startup recovery and evidence

530 CPU tests passed,40 ignored. Only the four original C02 fixtures use explicitly authorized private /tmp; all model/build/benchmark files remain on EFS. The original five GPU contract functions and 120 alignment/view cases passed. 7,552 full-vocabulary BF16 files match the original native/Hugging Face references literally: 4480 original,2048 bounded576 and 1024 canonical-output-slot files. The numerical oracle and tolerances did not change; stock vLLM is not the numerical oracle.

A fresh candidate-only C16/C32 functional pilot independently verified224 stored SSE responses plus disconnect/cancel/retry/resource recovery on this exact binary. The main independent audit verifies6,490 stored SSE structures across 30 normal and 4 original profile arms. Supplemental inspection has its own four-arm audit. Riley tokens/text/prompt/finish/usage match the original native/HF serving reference; stock match counts are reported separately, not used as an accuracy ranking. Whole-device observed normal/original-profile peak is18,267 MiB under the 22,528 MiB cap; supplemental observations are audited separately. Original GPU6/7 holder PIDs/start ticks/contexts remain intact, with 20 GiB idle holds; final fresh hold receipts are separate from the archive snapshots.

An initial stock launch failed because the new EFS IPC parent directory was missing. This occurred before correctness/warm/measured requests. The two already completed C16 Riley arms were retained. The unchanged measurement runner executed the remaining 13 C16 arms, including the missing stock arm, yielding 15 total: two original plus 13 continuation arms. Original failed startup/process state and all raw logs remain, and aggregate completion is explicitly identified. No completed normal arm was repeated or excluded; all 15 C32 arms ran normally. The correction receipt records chunk arguments/result hashes and original paths remain accessible through symlinks. A resume preflight initially rejected the failed startup's monitoring file; that assertion failed before launching additional work and is also retained.

The supplemental inspector setup initially omitted the trusted process-monitor Python module. That failed before correctness/warm/measured requests. The exact existing helper was copied without changing admission policy, and the failed premeasurement profile is retained. No completed diagnostic arm was replaced.

B1/B2 are archived CPU-only prototypes; B3 completed 7552 numerical files and 224 functional SSE values but no normal comparison. B4 adds pending-GPU failure guards and queue/lifecycle-aware admission gates. Prototype patches and receipts are retained; none contributes normal timing samples.

Baseline commit `57502e7a0fafd165966071e88b3e8a56cdea0e3d`; candidate binary SHA256 `2f2751b896df84449b1500cfc291e1211e226c246c4e464113470bcf0eb1b277`. All937 qualified source hashes are stable. The ten-file patch reconstructs the qualified changed source exactly; main integration was rejected.

[Evidence index](../benchmarks/evidence/20261021-overlap-v18b4/README.md) · [Candidate patch](../benchmarks/candidates/overlap-v18b4/candidate.patch) · [Independent Graph experiment](qwen-piecewise-v18a-20261021.md).

[vLLM model runner v2](https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md) describes persistent batch metadata and asynchronous lifetime concerns. [SGLang overlap scheduling](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) describes one-step-ahead scheduling and trace validation. This is an independent Rust/native implementation; it imports neither library's kernels.

The next unmeasured structural work should target the large projection cost while preserving accumulation/BF16 boundaries and avoid rebuilding host/device input metadata each iteration. Weight-layout/backend changes and a GPU-resident token/input pipeline each need isolated component proof before another full comparison. Their benefit is not established by this experiment.
