# PR 03 — CUDA Host Runtime과 FFI 경계

**상태:** Active
**선행 조건:** [PR 02](02-workspace-and-ci.md)  
**다음:** [PR 04 — Tensor와 메모리](04-tensor-and-memory.md)

[← 이전](02-workspace-and-ci.md) | [목차](README.md) | [다음 →](04-tensor-and-memory.md)

## 목적

Rust에서 CUDA device, context, stream, event와 CUDA C++ kernel launch를 예측 가능하게 제어하는 최소 host runtime을 만든다.

## 언어 경계

```text
Rust
→ ownership, validation, device/context/stream/event lifetime

extern "C" ABI
→ 고정 폭 type과 status code를 사용하는 연결 규약

CUDA C++
→ 실제 smoke kernel과 향후 production GPU operation
```

`extern "C"`는 ABI를 의미하며 kernel 구현 언어가 C라는 뜻이 아니다. 이 단계의 native source는 CUDA C++로 작성한다.

Python, PyTorch extension, Triton JIT를 통해 CUDA를 호출하는 우회 경로는 만들지 않는다.

## 설계 원칙

- CUDA C++와 Rust 사이에는 좁은 `extern "C"` ABI를 둔다.
- raw CUDA error code는 즉시 Rust error로 변환한다.
- global implicit context보다 명시적 device/context ownership을 선호한다.
- default stream 의미에 의존하지 않는다.
- destructor에서 실패를 숨길 수 있는 작업은 explicit close/synchronize API도 제공한다.
- C++ exception이 ABI를 넘어오지 않게 한다.
- Rust panic이 ABI를 넘어가지 않게 한다.
- FFI function은 Python object나 PyTorch tensor를 받지 않는다.

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

## C ABI 기본 규칙

- opaque handle 또는 CUDA native handle의 소유권을 명시
- `void*`는 device pointer인지 host pointer인지 함수명·문서로 구분
- shape와 byte length는 overflow를 검사할 수 있는 고정 폭 정수 사용
- return은 status code, 상세 오류는 별도 조회 또는 caller buffer 사용
- ABI version과 library build metadata 제공
- struct padding에 의존하는 복잡한 C++ type 노출 금지

예:

```cpp
extern "C" RustInferStatus rustinfer_fill_f32(
    float* device_output,
    std::uint64_t element_count,
    float value,
    cudaStream_t stream
);
```

## 구현 범위

1. device enumerate와 property 조회
2. target device 선택
3. stream 생성·동기화
4. event record/wait/elapsed time
5. host callback 또는 동등한 완료 확인 수단
6. CUDA C++ smoke kernel launch
7. kernel launch 직후와 sync 시점 오류 구분
8. native library ABI/build version 조회

Smoke kernel은 성능 목적이 아니다. vector add 또는 buffer fill 정도로 제한한다.

## FFI safety checklist

- [ ] null pointer 처리
- [ ] length overflow 검사
- [ ] host/device pointer 혼동 방지
- [ ] stream handle lifetime 보장
- [ ] launch parameter 범위 검사
- [ ] CUDA error 문자열 포함
- [ ] C++ exception이 ABI를 넘지 않음
- [ ] panic이 FFI 경계를 넘어가지 않음
- [ ] Python 또는 PyTorch object가 API에 없음

## 테스트

### CPU-only

- CUDA 비활성 build
- 오류 enum/format
- invalid device index
- native library가 없을 때 명확한 오류

### GPU

- device 정보 조회
- 두 stream의 event ordering
- async kernel 후 명시적 sync
- 잘못된 launch가 오류로 전달되는지 확인
- 반복 생성/drop 시 resource leak smoke
- Python이 없는 환경에서 동일 테스트 실행

의도적인 illegal-access/assert kernel로 late device fault를 만드는 검사는 이
단계의 7-test smoke에 섞지 않는다. 그런 fault는 CUDA context를 poison하고
`compute-sanitizer`의 zero-error leak gate와 같은 프로세스의 후속 lifecycle
검증을 오염시킨다. PR 03은 정상 fill의 명시적 synchronize와 non-poisoning
invalid launch를 통해 `LAUNCH`/`SYNCHRONIZE` stage 경계를 고정한다. 실제
device-fault injection은 별도 프로세스 격리와 복구 정책을 함께 검증하는
[PR 16 오류 격리 gate](16-reliability-and-release.md#오류-격리)에서 수행한다.

## 비범위

- 범용 tensor
- allocator pool
- cuBLASLt
- CUTLASS
- model operation
- CUDA Graph
- Triton
- NVRTC

## 완료 기준

- [ ] Rust에서 device/stream/event lifecycle이 safe API로 노출됨
- [ ] `unsafe`가 FFI module 밖으로 새지 않음
- [ ] CUDA C++ smoke kernel 결과가 정확함
- [ ] C ABI와 ABI version이 문서화됨
- [ ] compute capability와 memory 정보가 benchmark metadata로 출력됨
- [ ] Python 없는 환경에서 host runtime test 통과
- [ ] CUDA 미설치 환경 오류가 명확함

[← 이전](02-workspace-and-ci.md) | [목차](README.md) | [다음 →](04-tensor-and-memory.md)
