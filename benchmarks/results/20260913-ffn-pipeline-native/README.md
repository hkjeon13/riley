# FFN asynchronous pipeline native batch — 2026-09-13

Status: **two candidate kernels implemented; native graph numerical/memory checks passed; model/runtime/serving integration not yet implemented.** This is not a serving speedup or completion of PR05. The previous unprofiled serving comparison remains authoritative.

## Implemented batch

`kernels/optional/ffn_pipeline.cuh` implements gate/up/SwiGLU and five-way down projection together. Two shared-memory stages alternate global→shared `cp.async` loads with existing m16n8k16 BF16-input/FP32-accumulator MMA operations. Gate/up weights are shared by the two row warps. All CTA threads participate in commit/wait and the shared reuse barrier, including threads assigned inactive rows. Copy sources are in bounds even for predicated zero-fill.

The candidate retains the baseline gate/up accumulation order, BF16 gate/up rounding before SwiGLU, down320-wide split (final256), BF16 partial rounding and subsequent FP32 partial storage. Full-model equivalence still needs measurement. The pipeline is our CUDA transport implementation using the CUDA/PTX primitive; it is not a claim that the whole GEMM is a CUTLASS library implementation. No Python serving dependency is introduced.

Raw launch contract: gate grid192, block64; down grid72×5, block32; BF16 input/packed weights must be16-byte aligned and have the same full extents/layouts as V7. Active is a stable device scalar during replay. Caller owns stream/context and every parent until completion. The probe enforces these extents/launches, but a safe model recorder is not connected yet. These raw kernels are not exposed as an accepted runtime backend.

## Executed checks

- Native CUDA graph captures baseline gate/down and candidate gate/down. Three successive input seeds cycle active rows0/1/15/16/17/31/32/33 on retained pointers. Valid rows cover both sides of the16-row warp boundary. Zero and33 exercise uniform rejection; sentinel-filled inactive outputs are compared too.
- All24 replay cases had **zero bit mismatches** in all32×1536 gate outputs and all5×32×576 FP32 down partials. Weights are one deterministic synthetic fixture; this is not a real-model quality evaluation.
- Native-only compute-sanitizer memcheck: **0 errors**. Racecheck: **0 hazards, 0 errors, 0 warnings**.
- SM89 compiled resource attributes: gate12,288 shared bytes /38 registers; down10,240 shared bytes /30 registers. Both report0 local bytes; ptxas reports no spill stores/loads. These figures do not prove occupancy or speedup.
- SM90a and SM100a AOT object build passed. Object SHA256 `acdcbc71f8e559845d494bf735f75eaac6379ba5d1da254c5cc3884b4c562f57`. Runtime tests for these architectures are **SKIP: hardware unavailable**, not passed.

## Build / evidence

The source/binary hashes and complete build/probe/sanitizer logs are retained here. Candidate and reference source hashes were reconciled between local and remote checkout. Remote binary: `/tmp/riley-opt-260912/ffn-pipeline-probe-v1`; source overlay: `/tmp/riley-opt-260912/hardware-validation-source` on `ai-assistant`.

Build with task-local nvcc13.0.88:

```sh
nvcc -O3 -std=c++17 -arch=sm_89 --fmad=false -lineinfo -Xptxas=-v benchmarks/analysis/ffn_pipeline_probe.cu -o ffn-pipeline-probe
compute-sanitizer --tool memcheck --error-exitcode 86 ./ffn-pipeline-probe
compute-sanitizer --tool racecheck --error-exitcode 87 ./ffn-pipeline-probe
```

The actual run used compute-sanitizer from `/data/cuda-12.8.1/bin` and the task-local580.173 driver library with `/data/riley-g04-cuda13/lib`. Cross-build used `-gencode arch=compute_90a,code=sm_90a -gencode arch=compute_100a,code=sm_100a -c`, the same O3/C++17/fmad flags, and the same source. No toolchain or system driver was changed.

## Next integration gate

Connect both kernels under one explicit FFN backend selector and graph identity, keeping existing V7 default. Test pure/mixed stage transitions, full-model logits and actual greedy generation before comparing fixed/natural C16/C32 serving against frozen V7 and fresh vLLM. Keep the existing unaccepted FlashInfer branch separate to attribute precision and performance effects. No throughput, TTFT, TPOT or tail gain has been measured for this FFN batch yet.
