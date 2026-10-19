# Exact four-column projections on RTX4090

V107 extends the existing explicit `owned-hf-eager-bf16-v1` / `iteration-batch` / `lt-mlp-v2` / `segments-v1` route. Default options are unchanged. The source change is limited to `kernels/src/cublas_gemm_probe.cu`. Stock-vLLM parity remains unachieved.

## Change and reason

Four independent output columns reuse one packed BF16 input load and conversion. Each output retains its original serial FP32 FMA sequence, lane-strided recipe, half-tree reduction and final BF16 rounding. Only 22 measured shape/count intervals are selected; every row count in each selected interval had at least 1.05× speedup in the initial component screen. Other cases retain V104 routes. Mistral Query/Output share a geometry; the Output component benefit was inferred from shape and subsequently checked through all real Output weights and whole-model execution.

Qwen3-4B uses qualified intervals for Gate/Up, Down, Query, output and LM head. Mistral7B uses qualified intervals for the same roles. C1/R2, Key/Value, Smol, Qwen2.5 and unqualified hardware/runtime retain their existing routes. The physical gate remains SM89 / 128 SMs, CUDA runtime 12080 / cuBLAS 120804, qualified compiled architecture, 16-byte vector spans and no workspace/MPS overrides. Dispatch is based on GEMM geometry and environment, not a model-name allowlist. Another model sharing a selected geometry can use the same route; the numerical/performance evidence here covers the named models only. No new SM80/A100 recipe is claimed.

A current V104 Qwen3-4B diagnostic found approximately 90% GPU busy time. Its largest exposed kernel family was the R16/C8 paired projection: 929.74 ms in a 4807.92 ms instrumented window, including Gate 666.78 ms, Query 149.64 ms and LM 113.33 ms. Normal Graph bodies are opaque, so their time is not attributed to operators. Profiled throughput is excluded from the performance table.

Offline HTTP first-to-last-token interval accounting did not support fewer simultaneously streaming native requests as the simple gap explanation: native stream-active averages were above stock in these four windows. HTTP intervals are not actual GPU scheduled row counts and do not prove scheduler efficiency. No scheduler rewrite is justified by that observation alone.

## Performance

Same RTX4090/BF16, context 1024, token budget 512, KV blocks64×concurrency, 22 GiB process cap, prefix cache off, neutral greedy and original EOS. Natural output caps 32/64/128. Native uses CUDA 12.8.1; stock vLLM 0.27.1 uses Torch CUDA 13.0. Native before is the genuine integrated V104 binary; after is private V107. Gains are direct, not products of earlier campaigns.

Each confirmed row contains three fresh native process pairs (one matched pilot plus two extra pairs, last pair order reversed). Stock has one fresh matched pilot process per row. Qwen3-4B C16 was flat and was not repeated unchanged. These are not five fresh triples or broad workload parity results.

| Model / concurrency | Native fresh pairs | Before104 token/s | V107 token/s | Gain | Stock vLLM single-pilot token/s |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B C32 natural | 3 | 622.6 | 654.1 | +5.05% | 1400.7 |
| Mistral7B C16 natural | 3 | 345.6 | 369.2 | +6.84% | 560.6 |
| Mistral7B C32 natural | 3 | 412.5 | 473.2 | +14.71% | 868.3 |
| Qwen3-4B C16 natural | 1 | 499.1 | 499.6 | +0.11% | 896.7 |

## Latency and limits

The three confirmed cells pass both ratio-of-medians and paired-median nonregression for TTFT/TPOT/E2E p95/p99. Qwen3-4B C16 remains a single flat pilot with TTFT p95 +0.10%; this is not an all-workload or all-tail acceptance. Fixed-cap-mixture performance and the full multi-model stock target remain open. TPOT means each request's first-to-last-token mean interval percentile, not token ITL p99.

| Condition | TTFT p95 change | TPOT p99 change | E2E p99 change |
|---|---:|---:|---:|
| qwen4b C32 | -4.16% | -4.64% | -5.81% |
| mistral C16 | -5.06% | -13.79% | -9.37% |
| mistral C32 | -5.03% | -12.81% | -15.15% |

## Validation and provenance

- Component screen: 1,804 conditions and 44 changed-input Graphs across Qwen3-4B/Mistral first-layer/LM shapes, with every actual row count 1..32 timed. All four variants preserve original native M1 values. Compiler output showed zero spills, 96 registers for R8 and 111 for R16.
- Actual safe C API: 342 selected real weight cases, 24,624 full conditions and 49,248 changed-input Graph states, plus 648 smoke conditions. Full BF16 buffers and halos match independent native calls, including base offsets 0/256, counts 1..32/33/64, mixed prefill/singleton groups, stream leases and zero hot allocation.
- Full-model fresh HF 104 comparisons passed for both wide models at C1, C16 eager/Graph and shrinking1..32 counts. Driver API introspection proves actual quad kernels inside captured Graphs. Smol and Qwen2.5 native regressions passed; the pre-existing Qwen2.5 cross-runtime HF difference is retained.
- Independent pilot/confirmation audit checked 8,928 complete native response values, owned PID start ticks, empty GPU admission, 22 GiB cap, capture-free measured windows and disconnect/retry recovery.
- The actual integrated tree was rebuilt separately:all 932 private source hashes match, plus three named existing local documentation files. CPU 484 tests passed, 40 ordinary GPU tests ignored. Dedicated four-model GPU tests were run separately.
- Actual integrated server checks validated 1,248 complete values across fresh Qwen3-4B/Mistral C32 servers, including all-row warmup, recovery and GPU-resource checks. Only these post-integration correctness runs use the V108 shortest-full-reference warm anchor (at least 32 tokens); comparison/confirmation tables retain their original V96 protocol. No token cap or equality check is truncated.
- Timed private V107 SHA256: `8231e821599cefb7ad1c4574f6f74d4e329a132c95a846a384a9f78f61913274`.
- Actual integrated binary SHA256: `798f7bd79beb6bb41a454d8b462f1e5517d2692f8dcefc0a93a4ad27c474ef22`. The table labels the timed private build; source equivalence and actual integrated runtime checks are distinct evidence.
- Existing dirty work, private candidates and failed raw records are preserved. No deployment, commit or push occurred.

Receipts under `artifacts/multimodel-parity-20261017`: `v107-native-confirm-audit.json`, `v107-serving-pilot-audit.json`, `v107-CAPI-controller-result.json`, `v107-existing-model-regression-result.json`, `v107-adoption-receipt.json`, `v107-integrated-model-result.json`, `v107-integrated-serving-result.json`, `v107-bottleneck-interpretation.json`, and `v107-http-activity-diagnostic.json`.
