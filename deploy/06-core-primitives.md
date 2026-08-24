# PR 06 — 핵심 GPU Primitive와 GEMM Adapter

**상태:** Planned  
**선행 조건:** [PR 05](05-model-loading-and-ir.md)  
**다음:** [PR 07 — Llama 기준 Forward](07-llama-reference-forward.md)

[← 이전](05-model-loading-and-ir.md) | [목차](README.md) | [다음 →](07-llama-reference-forward.md)

## 목적

full model을 조립하기 전에 자주 쓰는 연산을 작은 독립 테스트로 검증한다. 이 단계에서 production GPU implementation의 기술 선택 순서를 고정한다.

## Production 구현 선택

| 연산 종류 | 초기 production 구현 | 후속 최적화 조건 |
|---|---|---|
| Dense GEMM | cuBLASLt | 필요한 fusion/dtype/layout이 없을 때 CUTLASS 검토 |
| Embedding gather | CUDA C++ | profiler 기반 vectorization/layout 개선 |
| RMSNorm | CUDA C++ | residual+norm fusion은 PR 15 |
| Residual/elementwise | CUDA C++ | GEMM epilogue 또는 fusion은 별도 PR |
| SiLU/gated multiply | CUDA C++ | CUTLASS epilogue는 측정 후 |
| RoPE | CUDA C++ | RoPE+KV write fusion은 PR 15 |
| DType cast | CUDA C++ 또는 backend epilogue | 불필요한 cast 제거 우선 |

다음은 production 기본 경로가 아니다.

- Python/PyTorch custom op
- Triton Python JIT
- 직접 작성한 universal GEMM
- profiler 없이 도입한 CUTLASS template stack

Triton은 `experiments/triton/`의 prototype과 비교 구현으로 사용할 수 있다. 성과가 확인되면 CUDA C++/CUTLASS로 이식하거나 별도 production 승인 절차를 거친다.

## 연산 범위

1. host→device weight upload
2. embedding gather
3. RMSNorm
4. elementwise residual add
5. SiLU
6. gated multiply
7. RoPE apply
8. dtype cast가 필요한 경우의 명시적 kernel
9. cuBLASLt GEMM adapter

## GEMM 원칙

직접 universal GEMM을 작성하지 않는다.

기본 순서:

```text
cuBLASLt
→ algorithm/epilogue tuning
→ CUTLASS가 필요한지 profiler와 capability로 판단
→ custom GEMM은 최후 수단
```

Adapter가 표현할 항목:

- M/N/K
- transpose
- input/weight/output dtype
- accumulator dtype
- bias/epilogue capability
- workspace
- stream
- selected algorithm metadata
- deterministic requirement
- backend implementation ID

처음에는 deterministic한 알고리즘을 선택하고 auto-tuning은 뒤로 미룬다.

## CUDA C++와 C ABI

각 production primitive는 다음 경계를 사용한다.

```text
Rust safe wrapper
→ Rust FFI validation
→ C ABI
→ CUDA C++ kernel/wrapper
```

C ABI header에는 Rust/C++ 양쪽에서 검증 가능한 dtype, shape, byte length, stream과 status만 노출한다. PyTorch tensor 또는 Python object를 받지 않는다.

## Reference 경로

각 operation은 CPU 또는 Python/PyTorch reference 결과를 가질 수 있다. Reference 생성 환경과 production 실행 환경은 분리한다.

테스트 shape:

- 실제 target model shape
- 작은 odd shape
- batch 1
- 여러 token
- zero/near-zero 입력
- 큰 magnitude 입력

RMSNorm과 RoPE는 FP32 accumulator 차이를 명시적으로 비교한다.

Python reference가 만든 fixture는 JSON/safetensors/명시적 binary 또는 checksum으로 저장하고 production test는 Python 없이 읽는다.

## Kernel registry 최소형

```rust
OpId
KernelCapability
KernelKey
KernelImplementation
```

`KernelImplementation`은 다음과 같은 origin을 기록한다.

```text
CudaCpp
CuBlasLt
Cutlass
ExternalNative
ExperimentalTriton
ReferenceCpu
```

`ExperimentalTriton`은 production planner가 기본 선택할 수 없다.

아직 복잡한 planner를 만들지 않고 reference/optimized implementation 선택만 가능하게 한다.

## CUTLASS 도입 Gate

PR 06에서는 CUTLASS kernel 구현을 필수로 하지 않는다. 이후 도입하려면 다음 중 하나를 입증한다.

- cuBLASLt가 필요한 epilogue/fusion을 표현하지 못함
- quantized 또는 grouped GEMM이 target workload의 병목
- 특수 layout conversion이 end-to-end 비용을 유발
- target shape에서 반복 가능한 latency/throughput 이점

CUTLASS 추가는 독립 PR로 수행하고 cuBLASLt fallback을 유지한다.

## 성능 기록

microbenchmark는 결과를 판단하기 위한 보조 자료다.

- latency median/p95
- temporary bytes
- kernel launch 수
- achieved bandwidth 또는 GEMM throughput
- implementation ID
- native/Python-free 실행 여부

microbenchmark 우위가 end-to-end 우위를 의미한다고 주장하지 않는다.

## 비범위

- attention
- KV cache
- full decoder layer
- fused residual+norm
- CUTLASS operation 구현
- Triton production integration
- NVRTC
- auto-tuning database

## 완료 기준

- [ ] 모든 primitive가 target dtype에서 reference tolerance 통과
- [ ] GEMM 기본 경로가 cuBLASLt로 target shapes를 실행
- [ ] custom primitive는 CUDA C++와 C ABI 뒤에 구현
- [ ] stream이 호출 chain 전체에 명시적으로 전달됨
- [ ] allocation-free 반복 실행 경로가 존재
- [ ] kernel implementation을 runtime flag로 선택 가능
- [ ] Python 없는 환경에서 primitive integration test 통과
- [ ] CUTLASS/Triton/NVRTC가 초기 runtime 필수 dependency가 아님

[← 이전](05-model-loading-and-ir.md) | [목차](README.md) | [다음 →](07-llama-reference-forward.md)
