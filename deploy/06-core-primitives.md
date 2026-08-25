# PR 06 — 핵심 GPU Primitive와 GEMM Adapter

**상태:** Active
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

## 구현 결과

### 문제와 범위

PR 05까지는 checkpoint를 canonical weight view로 읽을 수 있었지만 GPU에 올리거나
decoder가 재사용할 production 연산 경계가 없었다. 이 PR은 다음 review commit으로 그
경계를 추가했다.

1. `182e204` — FP32 reference primitive, exact-key kernel registry, production 선택 정책
2. `e466dfd` — GEMM accumulator dtype까지 포함하는 registry key 보강
3. `3815891` — CUDA C++ primitive/C ABI, safe Rust wrapper, weight upload, cuBLASLt GEMM
4. `bbee345`, `3497d53`, `8384765` — CUDA feature lint, Rust 1.85, integration-test 호환성

구현 snapshot은 `8384765a1782149c0810412edbff6ae9478dc724`다. Attention, KV cache,
decoder layer 조립과 fusion은 계속 비범위다.

### Production 경계와 실패 계약

- Embedding, RMSNorm, residual add, SiLU, gated multiply, standard non-interleaved Llama
  RoPE, BF16↔FP32 cast를 CUDA C++로 구현했다. 모든 호출은 typed byte span과 명시적
  stream을 받는 C11 ABI 뒤에 있으며 Python/PyTorch object를 받지 않는다.
- Rust wrapper는 dtype, alignment, half-open byte range, shape 곱셈 overflow, context와
  stream ownership을 native call 전에 검증한다. mutable buffer는 active-use 동안 다시
  빌리거나 해제할 수 없고 실패 시 poison/fail-closed한다.
- Embedding의 잘못된 token ID는 output을 쓰기 전에 structured OOB report로 실패한다.
  RMSNorm과 RoPE는 FP32 accumulator를 사용하고 BF16 변환은 round-to-nearest-even 및
  canonical NaN `0x7fff` 계약을 reference/native 양쪽에서 고정한다.
- `CudaUploadedWeights`는 `(shard_path, tensor_name)` exact identity만 deduplicate하고,
  deterministic physical order와 reusable pinned staging buffer를 사용한다. tied token
  embedding/LM head는 하나의 device allocation을 공유하며 explicit `close`로 모두
  회수한다.
- GEMM은 `Y[M,N] = X[M,K] × W[N,K]^T`의 BF16 input/weight/output, FP32 accumulator
  계약을 cuBLASLt row-major adapter로 구현했다. Prepare 단계가 deterministic algorithm,
  workspace와 implementation metadata를 고정하고 Execute 단계는 allocation-free다.
  split-K는 1 이하, reduction scheme은 none으로 제한한다.
- `KernelRegistry`는 runtime `KernelPreference::{Reference, Optimized}`로 exact capability를
  선택한다. Optimized production 선택은 deterministic native implementation만 허용하고
  `ExperimentalTriton`을 기본 선택할 수 없다.
- Raw C handle의 close/call-start 동시성은 caller가 외부 동기화해야 한다고 ABI에
  명시했다. Safe Rust API에서는 ownership과 exclusive borrow가 이 조건을 강제한다.

### 검증과 결과

실제 model/tokenizer/CUDA/GPU 실행은 로컬에서 하지 않았다. 사용자 요청에 따라 clean
`git archive` snapshot을 `server-4096`에 전송하고, RTX 4090에서 `--gpus all`과
`--network none`으로만 다음을 검증했다.

```text
workspace --all-features tests: 100 passed, 0 failed (GPU/model tests는 별도 실행)
remote CUDA tests:              runtime 7 + memory 5 + primitive 3 + GEMM 1 passed
GEMM cases inside one test:     odd/SmolLM2 16 shapes, 모두 reference/determinism 통과
real checkpoint weight upload:  1 passed, 273 logical slots / 272 physical tensors
uploaded device bytes:          269,030,016 / pinned staging 4,194,304
strict Clippy:                  workspace/all-targets/all-features `-D warnings` 통과
Compute Sanitizer primitives:   3 passed / 0 errors / 0 bytes leaked
Compute Sanitizer GEMM:         16 shapes / 0 errors / 0 bytes leaked
```

Primitive fixture는 odd shape, batch 1과 여러 token, zero/near-zero, 큰 magnitude, 긴 RoPE
position을 포함한다. GEMM은 sampled FP32 reference 대비 BF16 RNE 결과, 같은 입력의 반복
output byte equality, 실행 전후 allocation counter와 algorithm metadata를 함께 검사한다.
실제 pinned SmolLM2 upload test는 모든 273개 slot의 source/shape/dtype/byte length/index,
foreign-context 실패 cleanup, tied alias, 실제 token-0 embedding payload까지 검증한다.

대표 FFI execute+sync microbenchmark는 다음과 같다. 이는 end-to-end 성능 주장이 아니라
현재 adapter의 재현 기준선이다.

| Case `(M,N,K)` | median ms | p95 ms | effective TFLOPS | temporary bytes |
|---|---:|---:|---:|---:|
| Q/O `(128,576,576)` | 0.009399 | 0.009779 | 9.037 | 0 |
| gate/up `(128,1536,576)` | 0.011184 | 0.011397 | 20.251 | 0 |
| down `(128,576,1536)` | 0.014185 | 0.019033 | 15.967 | 0 |
| LM head `(1,49152,576)` | 0.030528 | 0.031250 | 1.855 | 0 |
| LM head `(7,49152,576)` | 0.025829 | 0.026361 | 15.346 | 0 |

검증 환경은 RTX 4090(compute capability 8.9), driver `580.173.02`, CUDA runtime
`12.8`, cuBLASLt `12.8.4`, Rust `1.85.0`, Compute Sanitizer `2025.1.0.0`이다.
Container image는
`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`이며
SM89 AOT, shared cuBLASLt/cudart와 CUDA driver linkage를 사용했다.

외부 evidence는 다음 append-only 위치에 보존했다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr06/8384765a1782149c0810412edbff6ae9478dc724/
SHA256SUMS sha256: 2755501ce9a569f4b9a5464f09540aca00010cfa77ad653cbf8832eea289e04d
workspace-tests.log
clippy-all-features.log
cuda-gpu-tests.log
cuda-weight-upload.log
compute-sanitizer-primitives.log
compute-sanitizer-gemm.log
gpu.txt / image-id.txt / compute-sanitizer-version.txt
```

### 롤백과 PR 크기 예외

PR 05 validation commit `46727cf` 대비 구현 snapshot은 25개 파일
`+8,675/-42`줄이다. 권장 production diff를 넘지만 ABI+safe wrapper+reference가 각
primitive마다 같은 계약을 독립적으로 표현하고, GEMM/weight upload의 원격 GPU 검증을
동시에 닫아야 하는 첫 native execution boundary다. Review는 registry/reference
(`182e204`, `e466dfd`)와 native implementation(`3815891`)으로 나눌 수 있다. 전체
runtime forward는 아직 이 경로를 호출하지 않으므로 문제가 있으면 이 commit들을 역순
revert해 PR 05의 Python-free model loader 상태로 돌아갈 수 있다.

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

- [x] 모든 primitive가 target dtype에서 reference tolerance 통과
- [x] GEMM 기본 경로가 cuBLASLt로 target shapes를 실행
- [x] custom primitive는 CUDA C++와 C ABI 뒤에 구현
- [x] stream이 호출 chain 전체에 명시적으로 전달됨
- [x] allocation-free 반복 실행 경로가 존재
- [x] kernel implementation을 runtime flag로 선택 가능
- [x] Python 없는 환경에서 primitive integration test 통과
- [x] CUTLASS/Triton/NVRTC가 초기 runtime 필수 dependency가 아님

구현 gate는 통과했다. 이 문서는 선행 PR과 함께 merge되기 전까지 `Active`, merge
후 `Complete`로 전환한다.

[← 이전](05-model-loading-and-ir.md) | [목차](README.md) | [다음 →](07-llama-reference-forward.md)
