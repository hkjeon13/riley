# Persistent FFN phase-DAG feasibility

A finite cooperative kernel now executes V7 gate/up/SwiGLU → five-way down projection → residual/RMSNorm. The unchanged MMA order, BF16 rounding boundaries and row reductions are shared with the operator path. Native and full-model equality pass. Serving evidence is recorded below; this is not completion of PR07/PR19 or the overall goal.

## Implementation contract

`persistent_ffn.cuh` assigns gate and down tiles to warp workers, uses grid-wide barriers between stages, then assigns norm rows to complete 256-thread CTAs. Every thread reaches both grid barriers. Invalid row counts0 or33 return uniformly. A CTA barrier protects reuse of the eight-float norm reduction buffer when a block processes another row. Metadata stays immutable during launch.

The launch plan verifies cooperative-launch support and uses occupancy/device SM count to cap the grid at48 blocks. Enqueue rechecks device identity and occupancy before cooperative launch. Invalid pointer, device and oversized-plan rejection are covered. The raw internal helper relies on its caller for disjoint extents and lifetime; it is not a public safe buffer-owner API. The isolated full-model path reuses the existing validated Rust/native recorder ownership.

The original gate body gained an explicit logical output-column argument with its original default. The norm body gained an explicit row/shared-sums parameter; the existing global wrapper preserves its launch and arithmetic. No production dispatch selects the persistent kernel. Python prepares an isolated diagnostic source tree and runs offline tools; serving remains Rust → C/C++ → CUDA.

This is an FFN stage DAG, not the full attention+MLP layer executor. Global activation and down-partial storage remain; there is no claim that these DRAM transfers disappeared. There is no unbounded work queue or custom spinning grid barrier. See [NVIDIA cooperative launch documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf) for grid synchronization requirements.

## Native verification

24 replay cases: rows0/1/15/16/17/31/32/33, three changing-input iterations. Gate activations, down partials, residual and normalized outputs are bitwise equal to the operator path. Inactive norm/residual destinations retain sentinels. Memcheck:0 errors. Racecheck:0 errors/0 warnings. The sanitizer harness precedes the extra timing and host-rejection checks and is archived separately as `sanitizer-probe.cu`; both binaries use identical kernel sources.

The executed SM89 kernel reports63 registers/thread,32 bytes shared memory,0 local memory and48 CTAs. SM90a and SM100a compilation succeeded; runtime tests for those architectures are **skipped because those GPUs are unavailable**. This is not multi-GPU implementation or validation.

The native timing harness captures separate three-operator and one-persistent-node graphs, performs50 warmups and500 repeats, and measures CUDA-event windows in both orders. The input, residual and weights are reused across repeats; these hot repeated FFN windows do not model30 distinct layers or HTTP serving.

| Rows | Operators, two runs (µs) | Persistent, two runs (µs) |
|---|---:|---:|
| 1 | 10.598 /10.598 | 8.985 /8.983 |
| 16 | 13.013 /12.999 | 11.594 /11.581 |
| 32 | 15.446 /15.456 | 15.008 /15.018 |

The approximately3–15% lower graph window motivates full-model measurement. It does not establish a serving speedup or explain the contribution of launch gaps versus task scheduling.

## Full-model integration and validation

`prepare_persistent_ffn_model.py` creates a separate source copy. It replaces the pure-decode FFN branch in all30 layers with the cooperative phase DAG, retains original mixed/prefill execution, uses an independent graph identity hashing the kernel and reused bodies, and renames the loopback diagnostic CLI to `--ffn-backend persistent-ffn-diagnostic-v1`. The source receipt reproduces all six modified files byte-for-byte. The experiment uses a separate Cargo target.

- Independent generation:32 requests,1,024/1,024 tokens identical, repeated-prompt invariance passes.
- Eight-passage natural teacher-forced test:12,582,912 BF16 logits identical. Baseline and candidate SHA256 are `492a46578581d131ab67c8c1cdb2d70f36a6539c868530e36269f1484f19f939`, also matching the previous accepted baseline dump.
- Whole-model native memcheck:0 errors; logits match ordinary runs bitwise and test teardown reports zero allocations. Full-model racecheck was not run.
- CUDA server release build completed. Model tests retain their original `ffn_pipeline_*` harness names because the isolated generator replaces that selected implementation; they now exercise the persistent candidate. This is not a claim that the previous cp.async backend was combined with this kernel.

The fixtures establish bounded equality, not broad model quality or long-soak stability. Full attention DAG, dynamic prefill persistence, ticket/async integration and a supported production selector remain outstanding.

## Reproduction

Build `benchmarks/analysis/persistent_ffn_probe.cu` with the pinned nvcc13.0.88 toolchain, `-std=c++17 -O3 -lineinfo -arch=sm_89`. SM90a/SM100a checks use `-c -arch=sm_90a` and `-c -arch=sm_100a`. Native logs, object/binary/source hashes and full-model command script are adjacent. Remote artifacts are under `/tmp/riley-opt-260912/persistent-ffn-v1` and `persistent-ffn-model-v1`. Installed dependencies and Blender state were unchanged.

## Matched serving decision

Same frozen binary V7/persistent FFN, fresh vLLM0.27.1, BF16 SmolLM2-135M, RTX4090, C32 natural16/128/398 prompts and32/64/128 output tokens. Budget/chunk512, context1024, GPU greedy, synchronous metadata and required graphs. Existing mixed/prefill kernels are retained. Blender stays stopped; GUI remains active, GPU clocks are not locked.

Two orders: V7 → persistent FFN → vLLM, then reverse. Each process has192 warmups and768 retained requests:5,760 requests total,1,536 retained and114,688 output tokens per engine. Both Riley modes match all retained reference outputs; vLLM matches1,110/1,536, a separate numerical observation. All runs have0 transport failures. P50 is median, P95/P99 nearest rank over pooled requests, throughput uses summed retained tokens divided by summed wall windows.

| Metric | Existing V7 | Persistent FFN | vLLM |
|---|---:|---:|---:|
| Output tokens/s ↑ | 10,440.7 | 10,715.1 | 11,793.7 |
| Requests/s ↑ | 139.831 | 143.506 | 157.951 |
| TTFT P50 (ms) ↓ | 10.294 | 10.336 | 18.624 |
| TTFT P95 (ms) ↓ | 16.433 | 17.548 | 40.003 |
| TTFT P99 (ms) ↓ | 67.339 | 66.435 | 59.904 |
| TPOT P50 (ms) ↓ | 2.920 | 2.850 | 2.381 |
| TPOT P95 (ms) ↓ | 3.024 | 2.970 | 2.818 |
| TPOT P99 (ms) ↓ | 3.066 | 3.015 | 3.106 |
| E2E P50 (ms) ↓ | 193.224 | 189.536 | 170.022 |
| E2E P95 (ms) ↓ | 391.135 | 384.305 | 354.562 |
| E2E P99 (ms) ↓ | 409.743 | 392.617 | 376.709 |

Persistent throughput is **+2.63% versus V7 and -9.15% versus vLLM**. Median TPOT is19.67% longer than vLLM. Per-run tokens/s: V710420.8/10460.7, persistent10789.6/10641.6, vLLM11749.7/11837.9. Both orders show a throughput gain over V7, but TTFT P95 worsens and the complete objective is unmet. Two repetitions of a short C32 screen do not establish high-concurrency stability, peak memory or statistical significance.

The earlier cp.async FFN candidate was not included in this matched run, so this screen does not establish superiority over that optional backend.

Retain as a diagnostic candidate, without default promotion. Next examine extending the finite task DAG across attention/projection boundaries with explicit scratch lifetimes and grid participation, while preserving accepted numerical operations. FFN-only results do not prove a full-layer executor will be faster: synchronization, register/resource constraints and remaining global intermediates must be measured. PR07/PR19 remain incomplete.

Raw HTTP frames are remote under `/tmp/riley-opt-260912/persistent-ffn-serving-v1`; `serving/compact` retains reconciled token IDs/timestamps and raw hashes. Frozen binary SHA256 is `5e9b7215ff06773af4368a6210fb38775c53c66ccd55d8493982ac55101e190c`. Final verification confirms it is unchanged and GPU compute processes are empty. No Python process participates in Riley inference.
