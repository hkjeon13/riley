# PR 04 — Tensor 표현과 GPU 메모리 수명

**상태:** Active
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

이번 PR은 **execution step에 해당하는 pending copy token이 stream과 두 buffer를
완료까지 보유**하는 전략을 선택한다.

- safe Rust `CudaPendingH2D`/`CudaPendingD2H`가 originating stream,
  device buffer, pinned host buffer를 실제 `&mut`로 borrow한다.
- native copy token도 세 resource에 persistent active-use reservation을 둔다.
- 정상 query/synchronize와 context-stack 복원이 모두 성공한 뒤에만 reservation을
  해제한다.
- pending 값을 `mem::forget`하거나 완료 확인이 실패하면 token, accounting,
  context child lease를 남겨 reuse/free를 영구 거부한다.
- 한 resource에는 동시에 copy 하나만 허용하며 완료 뒤에만 다른 stream으로
  명시적으로 handoff한다.

중요한 것은 Rust borrow가 host lifetime만 보장하고 GPU 실행 완료까지 자동 보장하지 않는다는 점이다.

상세 계약은 [Tensor와 CUDA memory invariant](../docs/tensor-memory-invariants.md)와
[CUDA C ABI v1](../docs/cuda-abi-v1.md)에 고정한다. Stream-ordered allocator,
deferred-free queue와 caching pool은 도입하지 않는다.

## 테스트

- allocation/free accounting 0으로 복귀
- overflow와 zero-sized tensor
- view slicing과 offset
- reshape 가능/불가능 케이스
- non-contiguous transpose 표현
- host↔device copy round trip
- 서로 다른 stream에서 잘못된 조기 free 방지

정상·validation 경로와 Compute Sanitizer는 이번 PR에서 검증한다. 반면 rollback
`cudaFree*`, deferred copy 오류, context restoration 실패를 강제로 만드는 test-only
fault backend는 공유 primary context와 leak gate를 오염시키지 않도록
[PR 16 오류 격리 gate](16-reliability-and-release.md#오류-격리)에서 별도 process로
구현한다. 해당 분기는 이번 PR에서 fail-closed source audit 대상으로 고정한다.

## 비범위

- model-specific tensor
- weight packing
- memory pool 최적화
- KV block allocator
- unified memory

## PR 계약 요약

- **문제:** PR 03의 diagnostic smoke buffer만으로는 model tensor의 dtype/shape/stride,
  backing lifetime, 범용 device/pinned allocation과 async copy 완료를 표현할 수 없었다.
- **범위:** checked tensor metadata와 borrowed view, 명시적 workspace, opaque untyped
  device/pinned buffer, context별 coherent accounting, stream-ordered H2D/D2H completion
  token을 구현한다.
- **의미 보존 등급:** `N/A`. Model operation이나 수학적 최적화를 구현하지 않으며
  copy round trip은 입력 byte를 그대로 보존해야 한다.
- **설계 결정:** reshape는 contiguous zero-copy만, transpose는 metadata-only만
  허용한다. mutable layout은 non-overlap을 보수적으로 증명해야 한다. async 안전은
  Rust borrow와 native active-use token을 함께 사용하고 모호한 실패에서는 leak을
  선택해 fail closed한다.
- **롤백:** `rustinfer-tensor`의 `cuda` feature를 끄면 metadata는 host storage만으로
  동작하고 native CUDA를 build/link하지 않는다. PR 04 전체는 PR 03 evidence commit
  `9428af8` 위의 독립 snapshot으로 되돌릴 수 있다.

## 완료 기준

- [ ] ownership과 async lifetime invariant 문서화
- [ ] tensor metadata unit test 완비
- [ ] GPU copy round trip 정확성
- [ ] allocation 통계 조회 가능
- [ ] implicit contiguous/cast가 없음

[← 이전](03-cuda-host-runtime.md) | [목차](README.md) | [다음 →](05-model-loading-and-ir.md)
