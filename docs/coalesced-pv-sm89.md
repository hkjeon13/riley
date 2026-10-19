# Coalesced heterogeneous PV: SM89 qualification and serving decision

The private V199 candidate is not adopted. Actual164 remains the current source. V199 preserved numerical results, but its measured serving gains were below 2% and every latency gate had at least one regression. The broad goal of matching or beating stock vLLM remains open.

## Same-device serving comparison

RTX4090 on ai-assistant; BF16, 22 GiB process GPU cap, context1024, token budget512, KV block16 and64 blocks per concurrent sequence, prefix caching off, original EOS and greedy sampling. Each row is one fresh before/after/stock triple with96 natural requests per arm,45 correctness requests, disconnect/retry checks and all native Graph bucket widths warmed. This pilot is not five independent repeats. Native CUDA12.8/cuBLAS120804 and stock vLLM0.27.1/Torch CUDA13 are disclosed separately.

| Model | Concurrency | Before164 tok/s | Private199 tok/s | Gain | Stock vLLM tok/s |
|---|---:|---:|---:|---:|---:|
| qwen4b | 16 | 597.23 | 602.82 | +0.94% | 1125.32 |
| qwen4b | 32 | 745.89 | 757.99 | +1.62% | 1691.45 |
| mistral | 32 | 542.35 | 546.00 | +0.67% | 1058.84 |

| Model / concurrency | TTFT p95 before → after (ms) | TPOT p99 before → after (ms) |
|---|---:|---:|
| qwen4b / 16 | 188.018 → 194.745 | 25.117 → 25.283 |
| qwen4b / 32 | 607.848 → 613.391 | 40.526 → 39.971 |
| mistral / 32 | 1074.926 → 1069.059 | 59.896 → 60.273 |

TPOT percentiles are percentiles of each response's mean inter-token interval, not individual token ITL. All nine latency quantiles remain in the raw audit. All arms produced7168 measured output tokens.3248 complete native response values (tokens/text/prompt/finish/usage) were independently audited; GPU ownership/budget/recovery and zero native Graph captures during measurement passed. Stock output equivalence to native/HF is not asserted, and exact-match rates are not quality scores.

## Numerical evidence

- CAPI:18944 full conditions covering T1..512, actualR2..32 and six mixed shapes per T; BF16 context/probability and scaled F32 equal.18944 captured Graphs,37888 zero/restored-input states and zero hot allocations.629 smoke conditions were separate.
- Qwen3-4B and Mistral-7B:104 pinned same-device HF full-vocabulary rows/model, C1 eager, fixed Graph and changingR1..32 exact. Actual Driver Graph row keys2304/2048 and first-replay coverage passed. Actual new PV kernel launches were observed.
- Long trajectory:256 full-vocabulary rows/model compared with actual164 native eager, including both fixed32 and all-count paths. Qwen38,895,616 and Mistral8,388,608 BF16 values per after arm matched. Each case now independently visitsR1..32, and new PV launchesR2..32 were observed. This additional trajectory is a native reference comparison, not newly regenerated HF.
- Legacy modes2/3/4, CLI33 early denial, Smol HF104 and native Qwen2.5 regressions passed. The pre-existing historical Qwen2.5 HF16 difference is retained separately.

## Preserved diagnostic failures

Original long all-count scheduling consumed its32 decreasing counts in the first case. Mistral's actual mid prompt has142 tokens, so T143..269 contains no V199 eligible PV length; its second long prompt starts at415 with only one active row. Missing PV trace was therefore a coverage hole. A first attempted cyclic schedule wrongly reactivated retired copies without their intervening KV history and correctly failed strict replica equality at step97. The final diagnostic resets to32 only after a new full prompt prefill for each case and decreases monotonically within that case. It reuses all valid completed original before/fixed/all-count GPU runs and adds two new coverage runs. Original failures, raw arrays and receipts remain unchanged. A CPU coordinator path error was retained before any GPU launch. No equality criterion was weakened.

## Why the component win was insufficient

V198 measured PV-only speedups2.31..3.50x atR8 and3.27..5.49x atR32 with warm caches (smaller post-flush gains), but it covers only106 additional fallback lengths. These event timings omit gather, QK, softmax, projections, prefill, scheduling and HTTP. The full-serving pilot establishes only0.67..1.62% aggregate gains. It does not prove the complete attention stage or linear projections became that much faster.

## Next finite experiment

V208 tested real Gate AND Up weights with the unchanged accepted164 FP32 FFMA order. One merged column grid improved the pair by approximately6.5..8.0% after flushing caches for Qwen and2.8..3.7% for Mistral.192 conditions and64 Graphs/128 changed-input states were exact. Ktile224 raised actual Qwen occupancy from3 to4 blocks/SM, yet slowed the pair, so that staging change is rejected. Mistral's256-thread variant remained at2 blocks/SM. Source adoption and whole-model improvement are not established by this component test.

The next prototype overlaps weight prefetch with computation using two shared buffers and cp.async while retaining the native FFMA sequence. Its first private launcher incorrectly allocated one buffer and failed with illegal memory access. The corrected prototype shares one explicit26,624-byte two-buffer size between launch and occupancy queries and asserts the device layout. V210 then passed192 numerical conditions and64 Graphs/128 changed-input states. After a512MiB write, paired event measurements improved the Gate/Up pair8.95..10.78% for Qwen and5.98..7.23% for Mistral; these are component results atR25/32, not whole-serving gains. V211 uses identical CUDA kernel bodies in a smaller host harness: memcheck has passed with zero errors; racecheck is still running. Actual-batch coverage must be widened and full-model/serving tests must pass before source adoption.

Evidence: artifacts/multimodel-parity-20261017/v205-verified-performance-table.json, v205-pilot-independent-audit.json, v199-independent-CAPI-audit.json, v199-independent-live-Graph-audit.json, v204d-controller-result.json, v208-controller-result.json and preserved v209 receipts. No deployment, commit, push or foreign job termination.
