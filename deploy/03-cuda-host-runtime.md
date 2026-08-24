# PR 03 — CUDA Host Runtime과 FFI 경계

**상태:** Planned  
**선행 조건:** [PR 02](02-workspace-and-ci.md)  
**다음:** [PR 04 — Tensor와 메모리](04-tensor-and-memory.md)

[← 이전](02-workspace-and-ci.md) | [목차](README.md) | [다음 →](04-tensor-and-memory.md)

## 목적

Rust에서 CUDA device, context, stream, event와 kernel launch를 예측 가능하게 제어하는 최소 host runtime을 만든다.

## 설계 원칙

- CUDA C/C++와 Rust 사이에는 좁은 `extern "C"` ABI를 둔다.
- raw CUDA error code는 즉시 Rust error로 변환한다.
- global implicit context보다 명시적 device/context ownership을 선호한다.
- default stream 의미에 의존하지 않는다.
- destructor에서 실패를 숨길 수 있는 작업은 explicit close/synchronize API도 제공한다.

## 최소 타입

```rust
CudaDevice
CudaContext
CudaStream
CudaEvent
CudaModule 또는 KernelHandle
CudaError
DeviceProperties
```

각 타입은 다음을 문서화한다.

- ownership
- thread 이동 가능 여부
- `Send`/`Sync` 여부
- drop 시 동작
- async 작업과의 lifetime 관계

## 구현 범위

1. device enumerate와 property 조회
2. target device 선택
3. stream 생성·동기화
4. event record/wait/elapsed time
5. host callback 또는 동등한 완료 확인 수단
6. 아주 작은 smoke kernel launch
7. kernel launch 직후와 sync 시점 오류 구분

Smoke kernel은 성능 목적이 아니다. vector add 또는 buffer fill 정도로 제한한다.

## FFI safety checklist

- [ ] null pointer 처리
- [ ] length overflow 검사
- [ ] host/device pointer 혼동 방지
- [ ] stream handle lifetime 보장
- [ ] launch parameter 범위 검사
- [ ] CUDA error 문자열 포함
- [ ] panic이 FFI 경계를 넘어가지 않음

## 테스트

### CPU-only

- CUDA 비활성 build
- 오류 enum/format
- invalid device index

### GPU

- device 정보 조회
- 두 stream의 event ordering
- async kernel 후 명시적 sync
- 잘못된 launch가 오류로 전달되는지 확인
- 반복 생성/drop 시 resource leak smoke

## 비범위

- 범용 tensor
- allocator pool
- cuBLASLt
- model operation
- CUDA Graph

## 완료 기준

- [ ] Rust에서 device/stream/event lifecycle이 safe API로 노출됨
- [ ] `unsafe`가 FFI module 밖으로 새지 않음
- [ ] smoke kernel 결과가 정확함
- [ ] compute capability와 memory 정보가 benchmark metadata로 출력됨
- [ ] CUDA 미설치 환경 오류가 명확함

[← 이전](02-workspace-and-ci.md) | [목차](README.md) | [다음 →](04-tensor-and-memory.md)
