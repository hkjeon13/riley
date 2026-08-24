# PR 04 — Tensor 표현과 GPU 메모리 수명

**상태:** Planned  
**선행 조건:** [PR 03](03-cuda-host-runtime.md)  
**다음:** [PR 05 — 모델 로딩과 IR](05-model-loading-and-ir.md)

[← 이전](03-cuda-host-runtime.md) | [목차](README.md) | [다음 →](05-model-loading-and-ir.md)

## 목적

모델 연산 전에 device memory, shape, stride, dtype, view와 async lifetime을 명확히 표현한다.

## 핵심 타입

```rust
DType
Shape
Strides
Layout
DeviceBuffer<T 또는 untyped>
PinnedHostBuffer
TensorView<'a>
TensorViewMut<'a>
Workspace
```

## 필수 invariant

- logical element count와 byte length가 일치한다.
- view는 소유 buffer보다 오래 살 수 없다.
- mutable alias를 만들지 않는다.
- kernel 실행이 끝나기 전에 backing storage가 해제되지 않는다.
- reshape는 element order가 유지될 때만 zero-copy다.
- transpose와 contiguous materialization을 구분한다.
- dtype cast가 암시적으로 발생하지 않는다.

## Allocation 단계

이번 PR에서는 단순하고 검증 가능한 allocator를 사용한다.

1. direct device allocation/free wrapper
2. 선택적 pinned host allocation
3. workspace 명시 전달
4. allocation accounting

Caching allocator와 paged KV pool은 아직 구현하지 않는다.

## Layout 표준

초기 canonical layout을 문서로 고정한다.

예:

```text
Hidden: [batch, sequence, hidden]
Q/K/V: [batch, heads, sequence, head_dim]
Weights: checkpoint 원본과 execution packed layout 구분
```

각 kernel은 지원 stride/layout을 capability로 선언한다. 지원하지 않는 layout을 조용히 copy하지 말고 planner가 명시적으로 변환한다.

## Async lifetime 전략

가능한 방법 중 하나를 명시적으로 선택한다.

- stream-ordered allocator
- event를 붙인 deferred free queue
- execution step이 buffer를 보유

중요한 것은 Rust borrow가 host lifetime만 보장하고 GPU 실행 완료까지 자동 보장하지 않는다는 점이다.

## 테스트

- allocation/free accounting 0으로 복귀
- overflow와 zero-sized tensor
- view slicing과 offset
- reshape 가능/불가능 케이스
- non-contiguous transpose 표현
- host↔device copy round trip
- 서로 다른 stream에서 잘못된 조기 free 방지

## 비범위

- model-specific tensor
- weight packing
- memory pool 최적화
- KV block allocator
- unified memory

## 완료 기준

- [ ] ownership과 async lifetime invariant 문서화
- [ ] tensor metadata unit test 완비
- [ ] GPU copy round trip 정확성
- [ ] allocation 통계 조회 가능
- [ ] implicit contiguous/cast가 없음

[← 이전](03-cuda-host-runtime.md) | [목차](README.md) | [다음 →](05-model-loading-and-ir.md)
