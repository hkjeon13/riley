# BF16 residual-prefill experiment and serving result

The combined FP32 softmax denominator and BF16 probability-residual candidate passes the existing fixed natural NLL/KL screen, but **regresses serving performance** and still fails strict free-running greedy equivalence. Do not promote this implementation. Default Riley remains unchanged.

## Implementation and isolation

The pinned FlashInfer prefill kernel sums its BF16-rounded probability fragments for the denominator. Existing V7 keeps FP32 probabilities for its denominator. We evaluated that difference directly, rather than assuming FP16 conversion was required:

- FP32 denominator alone slightly increases synthetic error at all three tested Q/K scales.
- The combined candidate computes `p_hi = BF16(p)`, `p_lo = BF16(p - float(p_hi))`, sums the original FP32 probabilities for normalization, and accumulates both `p_hi × V` and `p_lo × V`. Q/K/V and output storage stay BF16; FP16 overflow/subnormal restrictions are not introduced.
- This is a Riley numerical modification to the FlashInfer library kernel in a separate verified header copy, not an unmodified upstream feature. Original installed headers and the main serving build are untouched.
- `experimental_prefill_residual.py` verifies the warp-synchronized input header and final transformed header hashes. `prepare_prefill_residual_model.py` creates a separate source tree, binds a diagnostic graph identity, and gives its loopback-only CLI the explicit name `flashinfer-prefill-residual-diagnostic-v1`. `isolated-source-verification.json` proves the generated source changes reproduce the tested remote files.
- The model retains the previous prefill workspace and existing decode arithmetic. All model execution remains Rust → native CUDA; Python is used for build/reference/benchmark orchestration only.

## Numerical and memory results

The synthetic same-input screen covers 868,032 BF16 outputs per lane at Q/K scales 0.125, 1, 4. Inputs are bitwise equal as float32 dumps. Weighted RMSE against the FP64 oracle:

| Q/K scale | FI BF16 | FP32 denominator only | Denominator + BF16 residual |
|---|---:|---:|---:|
| 0.125 | 2.39778e-5 | 2.41968e-5 | 1.98484e-5 |
| 1 | 2.85481e-5 | 2.87700e-5 | 2.09566e-5 |
| 4 | 1.65473e-4 | 1.65972e-4 | 1.60219e-4 |

The synthetic maximum-error limit remains 0.01. Residual primitive memcheck and racecheck both pass with zero errors/warnings. This does not establish model quality or a speedup.

The unchanged eight-passage, 256-target full-model FP32-reference screen gives:

| Measure | Existing V7 | Residual candidate |
|---|---:|---:|
| NLL | 2.95928943 | 2.95714459 |
| KL(FP32 reference || engine) | 0.0007591519 | 0.0007462197 |
| FP32 argmax matches / 256 | 244 | 250 |

Both predeclared NLL and KL conditions pass. **General quality remains unaccepted.** Independent synthetic free generation differs in 12/32 sequences and 148/1,024 tokens; strict equivalence exits 101. Repeated-prompt invariance passes. Whole-model native memcheck reports 0 errors; baseline/candidate logits under memcheck match the ordinary dumps bitwise. Session teardown reports zero remaining allocations. No full-model racecheck was performed.

## Matched serving comparison

RTX 4090, the same BF16 SmolLM2-135M checkpoint and C32 natural corpus (16/128/398 prompt tokens), budget/chunk 512, context 1024, synchronous metadata, required graphs and GPU greedy. Both Riley lanes use the same frozen diagnostic binary; the candidate flag selects residual prefill. FFN pipeline is not selected. Fresh vLLM uses the archived matched launch configuration.

Two orders: V7 → candidate → vLLM, then vLLM → candidate → V7. Each process has 192 warmup and 768 retained requests: 5,760 requests total, 1,536 retained per engine. All retained output-token counts are equal (114,688 per engine). All transport runs complete with 0 failures. Percentiles use nearest rank over pooled retained requests; throughput reconciles tokens and elapsed windows.

| Metric | Existing V7 | Residual candidate | vLLM |
|---|---:|---:|---:|
| Output tokens/s ↑ | 10,420.8 | 9,637.2 | 11,802.2 |
| TTFT P50 (ms) ↓ | 10.401 | 11.852 | 18.109 |
| TTFT P95 (ms) ↓ | 17.180 | 19.893 | 37.379 |
| TTFT P99 (ms) ↓ | 65.943 | 74.482 | 64.361 |
| TPOT P50 (ms) ↓ | 2.928 | 3.170 | 2.385 |
| TPOT P95 (ms) ↓ | 3.040 | 3.287 | 2.777 |
| TPOT P99 (ms) ↓ | 3.098 | 3.355 | 3.312 |
| E2E P50 (ms) ↓ | 194.522 | 210.597 | 169.291 |
| E2E P95 (ms) ↓ | 393.354 | 425.284 | 350.901 |
| E2E P99 (ms) ↓ | 399.601 | 432.350 | 375.173 |

Candidate throughput is **7.52% below V7 and 18.34% below vLLM**. Both orderings reproduce the regression. Candidate TTFT P50 is lower than vLLM, but TPOT and E2E are worse; the serving objective is not met. Retained exact frozen-reference matches are V7 1,536/1,536, candidate 1,532/1,536, and vLLM 1,118/1,536. These numerical comparisons are separate from the zero transport-failure count.

This is one exploratory workload/concurrency screen, not a release qualification or a high-concurrency stability campaign. GPU clocks were not locked; before/after telemetry and launch arguments are archived. No unmeasured hardware result is claimed for Hopper, Blackwell or multi-GPU.

## Cost evidence and next batch

The graph-node primitive trace identifies the active kernel (CTA Q128, block 32×4×1): 239 registers/thread, 32,768 bytes dynamic shared memory, 0 local memory/thread, six observed launches. Corresponding uncompensated code uses 233 registers. ptxas reports heavy spills in another compiled variant, but that variant is **not** selected by this trace. Do not attribute the serving regression to those unused spills. The trace is primitive resource evidence, not a serving time decomposition.

The combined implementation adds a second P×V MMA and retains extra fragments, while keeping the large Q128 geometry. The serving regression establishes that this combination is not worthwhile as implemented; it does not isolate each cost. The next attention batch should co-design smaller query tiles/work distribution and shorter residual-fragment lifetimes, preserve the improved normalization contract, then rerun model gates and a matched serving screen. Do not continue optimizing the synthetic error metric alone or replace BF16 KV storage globally.

## Evidence and reproducibility

- `primitive-comparison.json`: per-case precision and binary/header/input/output hashes.
- `natural-metrics.json`, `natural-final.log`, `free-generation.log`: exact fixed corpus and independent-generation outcomes.
- `model-memcheck.log`, `memcheck-verification.json`: final native full-model memory evidence.
- `active-kernel-resources.json`, `bf16-resource.log`, `residual-resource.log`: active launch and compiler resource information. Raw traces remain remote.
- `serving/`: complete launch/accounting/summary records, compact per-request timestamps/token IDs and raw-frame hashes. Raw HTTP rows remain in `/tmp/riley-opt-260912/prefill-residual-serving-v1`.
- Frozen binary SHA256: `230a372994253cce30a1005b6cf0e79688cb1dd3fdbcde5e526b45a8bc8ed86f`; final verification confirms it is unchanged and GPU compute processes are empty.

Preparation errors were corrected before measuring: initial script execution preceded upload completion; the first model-overlay extraction altered literal whitespace and was rejected by context checks; the first server build omitted the required `server` feature; and the first summary call used the exporter directory without its required `compact/` layout. Completed artifacts correspond to corrected runs. No failed numerical check was relabeled as a pass.

The diagnostic source copy is `/tmp/riley-opt-260912/prefill-residual-model-source-v1`; its build target is separate from the regular one. The repository's optional build does not import the new residual transform. This commit supplies reproducible experiments and results, not a production backend switch.
