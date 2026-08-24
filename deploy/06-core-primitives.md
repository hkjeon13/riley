# PR 06 — 핵심 GPU Primitive와 GEMM Adapter

**상태:** Planned  
**선행 조건:** [PR 05](05-model-loading-and-ir.md)  
**다음:** [PR 07 — Llama 기준 Forward](07-llama-reference-forward.md)

[← 이전](05-model-loading-and-ir.md) | [목차](README.md) | [다음 →](07-llama-reference-forward.md)

## 목적

full model을 조립하기 전에 자주 쓰는 연산을 작은 독립 테스트로 검증한다.

## 연산 범위

1. host→device weight upload
2. embedding gather
3. RMSNorm
4. elementwise residual add
5. SiLU
6. gated multiply
7. RoPE apply
8. dtype cast가 필요한 경우의 명시적 kernel
9. cuBLASLt 또는 선택한 vendor GEMM adapter

## GEMM 원칙

직접 universal GEMM을 작성하지 않는다.

Adapter가 표현할 항목:

- M/N/K
- transpose
- input/weight/output dtype
- accumulator dtype
- bias/epilogue capability
- workspace
- stream
- selected algorithm metadata

처음에는 deterministic한 알고리즘을 선택하고 auto-tuning은 뒤로 미룬다.

## Reference 경로

각 operation은 CPU 또는 PyTorch reference 결과를 갖는다.

테스트 shape:

- 실제 target model shape
- 작은 odd shape
- batch 1
- 여러 token
- zero/near-zero 입력
- 큰 magnitude 입력

RMSNorm과 RoPE는 FP32 accumulator 차이를 명시적으로 비교한다.

## Kernel registry 최소형

```rust
OpId
KernelCapability
KernelKey
KernelImplementation
```

아직 복잡한 planner를 만들지 않고 reference/optimized implementation 선택만 가능하게 한다.

## 성능 기록

microbenchmark는 결과를 판단하기 위한 보조 자료다.

- latency median/p95
- temporary bytes
- kernel launch 수
- achieved bandwidth 또는 GEMM throughput

microbenchmark 우위가 end-to-end 우위를 의미한다고 주장하지 않는다.

## 비범위

- attention
- KV cache
- full decoder layer
- fused residual+norm
- auto-tuning database

## 완료 기준

- [ ] 모든 primitive가 target dtype에서 reference tolerance 통과
- [ ] GEMM adapter가 target shapes를 실행
- [ ] stream이 호출 chain 전체에 명시적으로 전달됨
- [ ] allocation-free 반복 실행 경로가 존재
- [ ] kernel implementation을 runtime flag로 선택 가능

[← 이전](05-model-loading-and-ir.md) | [목차](README.md) | [다음 →](07-llama-reference-forward.md)
