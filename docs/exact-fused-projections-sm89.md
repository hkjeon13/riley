# Exact fused projections and Attention graphs on RTX4090

The existing explicit `owned-hf-eager-bf16-v1` / `iteration-batch` / `lt-mlp-v2` / `segments-v1` route now uses the qualified V104 implementation. Default serving options remain unchanged. This is an incremental exactness-preserving optimization; stock-vLLM parity and all-tail nonregression are not achieved.

## Implementation

- SmolLM2-135M Q/K/V run in one kernel launch with separate weights and outputs, preserving the original FP32 FMA chains and reduction tree. Shared input and weight tiles are reused across decode rows.
- Gate/Up and SiLU multiplication run together, with the original BF16 rounding points retained after both projections, SiLU, and multiplication.
- Smol Attention participates in full-body reusable decode graphs at actual active counts 2..32. Dynamic context/page metadata and separate legacy/actual-row graph keys avoid stale lengths and key aliasing. Existing supported Attention arithmetic is retained.
- Qwen3-4B selects the previously qualified paired-column kernel at actual rows 9..16 only for Query, Gate/Up and LM head. Other projection shapes retain their existing routes.
- The strict environment gate is RTX4090 SM89/128 SMs, CUDA runtime12080/cuBLAS120804, the qualified compiled architecture, and no workspace/MPS overrides. Unsupported environments and shapes retain existing behavior. No A100 performance or exactness claim is made for these new recipes.

Installed stock vLLM0.27.1 source was inspected: `QKVParallelLinear`, `MergedColumnParallelLinear` and `SiluAndMul` provide the structural references. Riley retains its own reduction/rounding contract. Evidence is in `v104-installed-vllm-linear-source-evidence.txt`.

The V103 Attention gather geometry passed exactness but lost 2.73% C32-natural throughput and increased request-mean TPOT p99 by 7.308% in its pilot. It is excluded. The earlier Tensor-MMA candidate changed values and is also excluded. Failed sources and raw receipts remain preserved.

## Direct before/after/stock comparison

Five fresh process triples per condition; 150 server arms total, alternating before/after/stock order. Original V58 is the genuine before binary; all gains below are direct comparisons rather than products of pilot gains. Units are output tokens/s; numbers are medians. C denotes concurrent requests.

| Model | C | Workload | Before V58 | After V104 | Change | Stock vLLM | After/stock |
|---|---:|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 1 | mixed caps | 654.6 | 715.1 | +9.2% | 762.5 | 93.8% |
| SmolLM2-135M | 8 | mixed caps | 1780.5 | 2500.9 | +40.5% | 3012.5 | 83.0% |
| SmolLM2-135M | 16 | mixed caps | 2184.8 | 3197.3 | +46.3% | 4786.3 | 66.8% |
| SmolLM2-135M | 32 | mixed caps | 2405.8 | 3676.5 | +52.8% | 6717.1 | 54.7% |
| SmolLM2-135M | 1 | original EOS | 672.8 | 738.5 | +9.8% | 897.9 | 82.2% |
| SmolLM2-135M | 8 | original EOS | 2119.2 | 3206.4 | +51.3% | 4258.8 | 75.3% |
| SmolLM2-135M | 16 | original EOS | 2595.8 | 4508.0 | +73.7% | 6952.2 | 64.8% |
| SmolLM2-135M | 32 | original EOS | 2910.7 | 5501.1 | +89.0% | 10396.7 | 52.9% |
| Qwen3-4B | 16 | mixed caps | 469.3 | 489.2 | +4.2% | 874.7 | 55.9% |
| Qwen3-4B | 16 | original EOS | 522.4 | 543.5 | +4.0% | 1017.2 | 53.4% |

Same RTX4090, BF16, context1024, scheduler token budget512, KV blocks64×C, 22GiB process memory cap, prefix cache off, neutral greedy/original EOS. Mixed caps means per-prompt8/16/24/32 (mean20), not identical20-token requests. Natural caps are32/64/128. Native uses CUDA12.8.1; installed stock uses Torch2.13+CUDA13.0. Stock agreement with HF is recorded independently and is not an exactness claim.

## Latency review

TPOT p99 is a percentile of per-request mean first-to-last-token intervals, not token ITL p99. TPOT and E2E p95/p99 medians improve in all ten conditions. Seven conditions pass both ratio-of-medians and paired-median nonregression for all six reviewed tails. The following positive TTFT changes remain unresolved; acceptance labels and equality were not relaxed.

| Condition | Metric | Before ms | After ms | Change | Paired median change |
|---|---|---:|---:|---:|---:|
| smol C1 fixed | ttft_ms_p95 | 3.757136 | 3.776340 | +0.511% | -0.305% |
| smol C1 natural | ttft_ms_p95 | 5.950691 | 5.998399 | +0.802% | +0.774% |
| smol C1 natural | ttft_ms_p99 | 5.995892 | 6.069598 | +1.229% | +1.229% |
| qwen4b C16 fixed | ttft_ms_p95 | 224.324393 | 225.043554 | +0.321% | +0.048% |
| qwen4b C16 fixed | ttft_ms_p99 | 242.191828 | 242.220159 | +0.012% | +0.012% |

These data do not establish that the increases are random noise or prove their cause. The explicit optimization route is integrated with the above limitation recorded; it is not an all-tail acceptance or production deployment.

## Validation and provenance

- Independent final audit checked 22,060 complete native response values against genuine V58 eager references, verified owned GPU start ticks, empty admission, the22GiB cap, capture-free measured native windows, and20 first-repeat recovery arms (two disconnects/three retries each).
- Fused QKV and Gate/Up each passed2,460 safe-CAPI conditions,1,920 changed-input graphs, fallback/halo/ownership checks and zero hot allocations. Qwen3-4B passed7,848 conditions/graphs and15,696 changed graph states across all109 selected weight cases.
- Full-model native equality and changing active counts1..32 passed. Fresh HF comparisons passed for Smol, Qwen3-4B and Mistral. Qwen2.5 retained its pre-existing cross-runtime HF difference while preserving before/after native equality.
- Actual integrated CPU regression passed484 tests;40 ordinary GPU tests were explicitly ignored and are not counted as GPU evidence. The finite model/server checks below are the GPU evidence.
- The actual integrated checkout was rebuilt separately, including all932 private candidate source hashes plus the two existing local documentation files. The first pre-build manifest assertion rejected the two extra docs before compilation; v2 verified the named extras and every source hash. Original failure receipt remains.
- Actual integrated model checks passed for four models, including Smol/Qwen3-4B C1 and all-count graph shrink. Two fresh integrated server runs independently checked 348 complete values, ownership, memory, capture-free windows and recovery.
- Timed private V104 binary SHA256: `1045415c67237d60d18c96826a308942a78237824bd1f096f7fd3803ff1a7e45`.
- Actual integrated server binary SHA256: `6b3ca7a9aaab5fe38b85b490c199d20bde73b9afeb4aeccc8f92c4b5db2724c8`. The benchmark above names the timed private build; source equivalence and post-integration runtime checks are separately recorded.
- Main changes are limited to seven reviewed runtime/CUDA/server files; unrelated dirty work and raw candidates are preserved. No deployment, commit or push occurred.

Primary receipts under `artifacts/multimodel-parity-20261017`: `v104-final-confirm5-audit.json`, `v104-adoption-receipt.json`, `candidate-v104-integrated-manifest.json`, `v104-integrated-model-result.json`, `v104-integrated-serving-result.json`, and `v104-integrated-build-v1-failed.json`. The broader multi-model stock-vLLM target remains open.
