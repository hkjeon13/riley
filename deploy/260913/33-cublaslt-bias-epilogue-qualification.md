# PR-N02B — cuBLASLt BF16 bias epilogue qualification

상태: P0/V2b가 Qwen2.5-3B P2048 Q/K/V에서 Riley의 strict staged bias를 exact하게 검증했다. HF actual module output은 다른 fused rounding contract를 보인다. 이 PR은 그 fused contract를 **별도 native candidate**로 qualification한다. 기본 serving selector, CUDA Graph integration, HTTP scheduler는 이 PR 범위 밖이다.

## 목표

Q/K/V projection의 `GEMM → row_bias_add` 두 GPU launch와 BF16 output read/write를 cuBLASLt `CUBLASLT_EPILOGUE_BIAS` candidate 하나로 줄일 수 있는지 확인한다. cuBLASLt BIAS는 D의 packed row-length bias를 broadcast하고 final postprocessing 전 적용한다. Riley의 row-major logical output은 기존 column-major TN mapping으로 cuBLASLt D rows가 projection output width가 되므로, bias 방향을 바꾸지 않는다. [cuBLASLt epilogue API](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltepilogue-t), [bias pointer contract](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltmatmuldescattributes-t)를 따른다.

이 연산은 strict `BF16 GEMM → BF16-to-FP32 add → BF16 RNE`와 다른 rounding을 할 수 있다. strict path를 교체하거나 fused result를 strict-equivalent라고 표시하지 않는다.

## 변경 묶음

1. native C ABI에 기존 `RileyCudaGemmPlan`과 분리된 `RileyCudaBiasGemmPlan`을 추가한다. create는 `epilogue=BIAS`만 받아 cold phase에서 descriptor, deterministic heuristic, `cublasLtMatmulAlgoCheck`, bounded workspace를 준비한다. current strict GEMM ABI는 `epilogue=NONE`만 계속 허용한다.
2. Rust에 `CudaPreparedBiasEpilogueGemm`과 `BiasGemmParams { input, weight, bias, output, workspace }`를 추가한다. hot path에서는 allocation, descriptor creation, heuristic selection을 하지 않는다. foreign context, short span, overlap, changed shape/dtype, poisoned plan을 fail closed 한다. execute failure 뒤 strict fallback을 같은 request에서 시도하지 않는다.
3. Qwen representative shape GPU qualifier를 추가한다. Q `[M, 2048, 2048]`, K/V `[M, 256, 2048]`, `M=1/8/32/2048`를 사용한다. actual Qwen weight/bias와 V2b artifact의 HF actual output을 이용한 layer gate는 native plan이 준비된 뒤 별도 ignored GPU test로 둔다.
4. native event AB receipt를 strict pair와 fused candidate에 대해 같은 stream, same workspace cap, warmup/repeat/order로 기록한다. 이는 operator receipt이며 serving 성능 결과가 아니다.

## 수치와 lifecycle gate

- V2/V2b strict artifact hash·metrics와 existing row-bias tests는 변하지 않아야 한다.
- fused candidate의 Q/K/V output은 사전에 선언한 HF **actual module output** BF16 gate를 통과해야 한다. 결과를 본 뒤 tolerance를 완화하지 않는다. 실패하면 fused profile은 보류한다.
- each shape/bias pointer에서 repeated BF16 hash, deterministic algorithm identity, zero allocation growth over hot repeats, plan/context close 뒤 zero accounting을 확인한다.
- input/weight/bias/output/workspace의 overlap, bad bias length, invalid dtype, foreign context, unsupported heuristic, `AlgoCheck` rejection은 fail closed 한다. cold prepare의 unsupported case만 strict caller fallback이 가능하고, execute failure는 plan poison/error다.
- run receipt에는 GPU, compute capability, CUDA/cuBLASLt version, descriptor fields, selected algorithm identity, workspace cap, shape, alignment, and fallback reason을 적는다.

## hardware scope

Ada SM89에서 first qualification을 실행한다. Hopper, Blackwell, multi-GPU는 static architecture allow-list로 자동 enable하지 않는다. device·toolkit·cuBLASLt version·descriptor·shape·alignment·workspace 별 heuristic과 `AlgoCheck` receipt가 있을 때만 candidate가 준비된다. CUDA 12.8.1 release notes의 Blackwell small-`M` fixed issue를 고려해 Blackwell decode `M=1`은 12.8.1 미만에서 skip하고, 지원 toolchain에서도 same artifact gate를 다시 실행한다. [CUDA 12.8.1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).

## 다음 PR과 롤백

이 PR이 native quality와 operator AB gate를 통과해도 serving default를 바꾸지 않는다. 다음 PR만 `LlamaProjectionBiasMode::{StrictStagedV1, CublasLtBiasEpilogueExperimentalV1}`로 Q/K/V prefill·decode dispatch에 제한적으로 연결하고, graph capture는 또 다른 PR로 분리한다. 동일 모델·revision·workload·concurrency로 ABBA serving 반복을 수행해 throughput, TTFT, TPOT, P95/P99, failure rate, quality를 vLLM과 함께 기록한 뒤에만 승격한다.

fused gate 또는 operator AB가 실패하면 plan/feature를 비활성으로 남긴다. strict path는 무변경이므로 rollback은 selector 연결을 하지 않는 것으로 끝난다. strict arithmetic을 보존한 launch/host overhead 후보는 GEMM→row-bias composite CUDA Graph 경로로 새 PR에서 평가한다.
