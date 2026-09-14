# Projection CTA serving screen — C32

RTX 4090, SmolLM2-135M, BF16, same prompt fixtures, 32 output tokens, client concurrency/active capacity 32, 754,974,720-byte KV payload per engine and prefix caching. Each lane uses 64 warmup and 512 retained requests; shared/unique workloads each run in two reversed engine orders. Riley runtime uses Rust → C ABI → CUDA. vLLM is version 0.29.0.

| Workload / engine | Tokens/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| shared / prior | 10212.66 | 14.517 | 2.668 | 117.085 | 132.074 |
| shared / candidate | 10730.27 | 13.769 | 2.538 | 110.671 | 128.306 |
| shared / vllm029 | 9723.89 | 36.144 | 1.712 | 324.196 | 328.079 |
| unique / prior | 3863.98 | 93.645 | 5.591 | 325.113 | 345.337 |
| unique / candidate | 3909.66 | 77.028 | 5.949 | 315.420 | 341.853 |
| unique / vllm029 | 3790.14 | 60.439 | 7.456 | 376.275 | 528.891 |

Values are medians of two run-level estimates, not pooled percentiles or confidence intervals. All 6,912 warmup/retained responses were revalidated from SSE artifacts; Riley responses match prior greedy reference. Stop outputs match between Riley engines, cancellation closes before output budget and recovery requests pass. The vLLM preflight and all 12 lane processes exited successfully. VLLM token agreement is reported separately from Riley's strict gate. Model and native gates are in adjacent result directories.

Shared throughput improves 5.07% over prior, with positive paired differences in both orders. Unique improves only 1.18% in the two-run aggregate, with +6.04% in the first order but -4.09% in the reverse order; unique TPOT is 6.40% worse. Shared candidate TPOT remains slower than vLLM, while unique candidate TTFT remains slower than vLLM. Therefore the full throughput/latency goal is not achieved.

VLLM unique throughput varies from 4,917 to 2,663 tokens/s across runs. System-wide I/O/CPU pressure is recorded in `assessment.json`; these are host-wide measurements and do not isolate the serving process or establish the cause of variation. Do not claim candidate superiority over vLLM from the aggregate throughput alone or compare this aggregate directly with an earlier day's absolute rates. No default promotion: keep this an explicit opt-in while testing low concurrency and repeating under more stable host conditions.

Source/experiment: `RILEY_EXPERIMENT_PROJECTION_CTAS=1` selects separate CTA projection only in the candidate; prior is the frozen dense-wire baseline. Binary hashes, controller/client hashes, launch settings, build test results and source snapshots are verified by `benchmarks/analysis/verify_projection_cta_serving.py`. The release model gate covers active capacities 16/32 with dependent decode, termination/cancellation and allocation cleanup. Runtime ticket tests pass (11 tests).

The first attempt failed the unchanged 16 GiB memory preflight. Moving 1.15 GB of verified historical FFN evidence out of tmpfs changed the environment and allowed this attempt. No build, archive export or extra GPU experiment was run during timed serving. Export happened after all lanes and Blender restoration. Raw trace/SQLite artifacts are not included.
