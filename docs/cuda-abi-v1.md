# rustinfer CUDA C ABI v1

이 문서는 `kernels/include/rustinfer_cuda.h`가 노출하는 ABI version 1의
binary contract를 정의한다. 이 경계는 Rust ownership API와 CUDA C++ 구현을
연결하기 위한 것이며 Python, PyTorch object, model tensor를 받지 않는다.
ABI metadata 함수 이외의 호출은 CUDA driver 또는 runtime을 초기화할 수 있다.

## 호환성과 type 규칙

- `rustinfer_cuda_abi_version()`은 `RUSTINFER_CUDA_ABI_VERSION`, 현재 `1`을
  반환한다. Rust wrapper는 native library의 값이 자신이 기대하는 값과 같은지
  확인한 뒤 다른 symbol을 사용한다.
- 모든 exported function은 C linkage와 `noexcept`를 사용한다. C++ exception은
  ABI 밖으로 전파되지 않아야 한다.
- ABI의 integer는 `<stdint.h>`의 고정 폭 type만 사용한다. status는 `int32_t`,
  byte/count는 `uint64_t`, 완료 여부는 `uint8_t`의 `0` 또는 `1`이다.
- v1에 field나 function을 추가할 때는 기존 field의 순서, 폭, offset, 의미를
  바꾸지 않는다. 새 struct field는 끝의 reserved 영역을 소비하거나 새
  `struct_size`-gated version을 사용하고, 새 symbol은 기존 symbol을 유지한 채
  추가한다. 기존 layout, numeric value, ownership 또는 함수 의미를 바꾸는
  변경은 ABI version을 올린다.
- v1 caller는 output struct의 `struct_size`를 자신의 `sizeof(struct)`로
  초기화한다. `RustInferCudaDeviceProperties`가 v1 크기보다 작으면
  `INVALID_ARGUMENT/VALIDATION`이며 output을 신뢰하면 안 된다. error buffer는
  선택 사항이므로 `RustInferCudaErrorInfo`가 `NULL`이거나 너무 작으면 상세
  진단만 쓰지 않고 operation status는 그대로 반환한다. reserved field는
  입력에서 `0`으로 두고 출력에서도 의미를 부여하지 않는다.

## 고정 layout

v1은 CUDA가 지원되는 64-bit host C ABI를 대상으로 하며 native build의
`static_assert`가 다음 크기와 핵심 offset을 검증한다.

### `RustInferCudaErrorInfo`

| offset | C type | field | 의미 |
| ---: | --- | --- | --- |
| 0 | `uint32_t` | `struct_size` | caller가 제공한 buffer contract |
| 4 | `int32_t` | `native_code` | 원래 CUDA code, validation/internal 오류는 `0` |
| 8 | `uint32_t` | `domain` | 오류를 만든 subsystem |
| 12 | `uint32_t` | `stage` | 오류를 관측한 lifecycle 지점 |
| 16 | `char[256]` | `message` | NUL-terminated best-effort 진단 |

전체 크기는 272 bytes다. message는 operation과 CUDA 오류 문자열을 포함하되
capacity에 맞게 잘릴 수 있다. status와 숫자 field가 기계 판정의 기준이며
message 문자열은 stable parsing interface가 아니다. caller가 error pointer를
`NULL`로 전달해도 operation의 status 반환은 동일하다.

### `RustInferCudaDeviceProperties`

| offset | C type | field |
| ---: | --- | --- |
| 0 | `uint32_t` | `struct_size` |
| 4 | `int32_t` | `ordinal` |
| 8 | `uint64_t` | `total_memory_bytes` |
| 16 | `uint32_t` | `compute_capability_major` |
| 20 | `uint32_t` | `compute_capability_minor` |
| 24 | `uint32_t` | `multiprocessor_count` |
| 28 | `uint32_t` | `warp_size` |
| 32 | `uint32_t` | `max_threads_per_block` |
| 36 | `int32_t` | `driver_version` |
| 40 | `int32_t` | `runtime_version` |
| 44 | `uint32_t[5]` | `reserved` |
| 64 | `char[256]` | `name` |

전체 크기는 320 bytes다. `name`은 NUL-terminated device name이다. 조회가
실패하면 partially populated output을 사용하지 않는다.

## status, domain, stage

모든 operation은 `RustInferCudaStatus`를 반환한다. `SUCCESS (0)`만 성공이고
나머지는 다음 stable 분류다.

| 값 | status | 의미 |
| ---: | --- | --- |
| 1 | `INVALID_ARGUMENT` | null, 잘못된 크기 또는 CUDA가 거부한 인자 |
| 2 | `INVALID_DEVICE` | 존재하지 않거나 음수인 device ordinal |
| 3 | `OUT_OF_RANGE` | count/size/launch 범위 초과 |
| 4 | `NOT_READY` | 비차단 query 대상이 아직 완료되지 않음 |
| 5 | `OUT_OF_MEMORY` | host 또는 device allocation 실패 |
| 6 | `DRIVER_ERROR` | 별도 mapping이 없는 CUDA Driver API 오류 |
| 7 | `RUNTIME_ERROR` | 별도 mapping이 없는 CUDA Runtime API 오류 |
| 8 | `INVALID_STATE` | handle ownership 또는 lifecycle 불일치 |
| 9 | `INTERNAL_ERROR` | native invariant 위반 |

error domain은 `NONE (0)`, `VALIDATION (1)`, `DRIVER (2)`, `RUNTIME (3)`,
`INTERNAL (4)`이다. stage는 `INITIALIZE (1)`, `VALIDATION (2)`, `CREATE (3)`,
`LAUNCH (4)`, `SYNCHRONIZE (5)`, `QUERY (6)`, `RECORD (7)`, `COPY (8)`,
`CLOSE (9)`이다. 하나의 CUDA native code가 여러 operation에서 나올 수
있으므로 Rust 오류는 status뿐 아니라 domain, stage, native code, operation과
owned message를 함께 보존한다.

`stream_query`와 `event_query`가 아직 완료되지 않았으면 `NOT_READY`를
반환하고 `out_complete`는 `0`이다. 이는 성공적인 `false`가 아니라 호출자가
재시도할 수 있는 비차단 상태다. query가 `SUCCESS`이면 `out_complete`는 `1`이다.
단, query 자체가 `NOT_READY`였더라도 호출 thread의 context-stack 복원에
실패하면 복원 오류가 우선한다. Rust wrapper는 정상적으로 복원된
`NOT_READY`만 `Ok(false)`로 변환하며 context poison을 완료 미상태로 숨기지 않는다.

## handle ownership과 context

`RustInferCudaContext`, `RustInferCudaStream`, `RustInferCudaEvent`,
`RustInferCudaSmokeBuffer`는 incomplete C type인 opaque handle이다. caller는
그 주소의 내부 layout을 읽거나 복사하지 않는다.

- `*_create` 성공은 caller에게 handle 하나의 소유권을 넘긴다. 실패 시 output
  handle은 `NULL`이다.
- `*_close`는 handle의 주소를 받는다. `*handle == NULL`인 close는 성공하는
  idempotent no-op이고, 바깥 pointer 자체가 `NULL`이면
  `INVALID_ARGUMENT/CLOSE`다. context의 poison/live-child 검사, stream/event의
  context 진입, smoke buffer의 context 진입과 in-flight synchronize처럼
  destructive CUDA API를 호출하기 전의 검사가 실패하면 handle은 non-`NULL`로
  유지된다.
- `cuDevicePrimaryCtxRelease`, `cudaStreamDestroy`, `cudaEventDestroy` 또는
  `cudaFree`를 한 번 시도한 뒤에는 그 결과 status와 관계없이 opaque handle을
  single-shot으로 소비하고 `NULL`로 바꾼다. CUDA API가 실제 side effect 뒤에
  앞선 asynchronous 오류를 반환했는지, genuine native failure로 resource가
  남았는지를 ABI에서 안전하게 구별할 수 없기 때문이다. 전자는 재시도하면
  double release/destroy/free가 될 수 있고, 후자는 native resource 또는 lease를
  안전하게 누수시키더라도 재시도하지 않는다. 파괴 뒤 context-stack 복원이나
  child-counter 검사가 실패한 경우도 non-success와 `NULL`이 함께 반환될 수 있다.
- close가 non-success일 때는 갱신된 handle도 반드시 확인한다. handle이 그대로
  non-`NULL`인 precheck 실패에만 retry 또는 상위 cleanup 정책을 적용하며,
  `NULL`이면 절대 재시도하지 않는다. 모든 명시적 close 오류는 처리하거나
  기록해야 한다. Rust `Drop`은 unwind하지 않는 best-effort fallback일 뿐,
  close/synchronize 오류 관측을 대신하지 않는다.
- stream, event, smoke buffer는 생성에 사용한 context를 참조한다. 모든 child를
  close하고 비동기 작업을 완료하기 전에는 context를 close하지 않는다.
  native context는 live child 수를 추적하며 child가 남은 context close를
  `INVALID_STATE/CLOSE`로 거부한다. 서로 다른 context가 소유한
  stream/event/buffer의 조합도 `INVALID_STATE`다.
- v1 native handle 자체는 임의 alias를 통한 concurrent mutation을 보장하지
  않는다. safe Rust wrapper가 ownership과 공유 수명을 보존하며, raw handle을
  꺼내거나 임의의 `Send`/`Sync`를 가정하지 않는다.

context 생성은 target device의 CUDA **primary context에 대한 공유 lease**를
`cuDevicePrimaryCtxRetain`으로 얻는다. 프로세스에 독점 context를 만들거나
primary context를 reset하지 않는다. close는 해당 lease의 release를 정확히 한
번만 시도한다. 같은 ordinal에서 context를 여러 번 생성하면 underlying
`CUcontext`는 공유하지만 각 retained lease와 opaque owner identity는 별개다.
따라서 서로 다른 owner에서 만든 stream/event/buffer 조합은 underlying
`CUcontext`가 같아도 `INVALID_STATE`로 거부된다.

context-dependent call은 먼저 `cuCtxGetCurrent`로 호출 thread의 current context를
snapshot한다. target primary context가 이미 current이면 caller의 stack을
건드리지 않고 그 context를 잠시 borrow한다. 그렇지 않으면 target을 push하고
호출 종료 전에 자신이 만든 pop debt를 pop하여 이전 current context를 복원한다.
process-global implicit current context나 `cudaSetDevice` 상태를 ownership 근거로
사용하지 않는다.

snapshot/push/pop의 결과나 pop된 context identity가 모호하면 context owner에
monotonic restoration-failed poison을 기록한다. push가 오류를 반환했더라도
target이 current가 된 것이 확인되면 pop debt를 끝까지 복원한다. 복원을 확정할
수 없으면 이후 context 진입과 primary-lease close를 `INVALID_STATE`로 거부하고
wrapper와 lease를 보존한다. 이 catastrophic path는 불확실한 current context를
release해 stale context를 만들기보다 fail-closed leak을 선택한다.

primary context가 공유되므로 `CudaContext::synchronize`의
`cudaDeviceSynchronize`는 같은 device의 다른 retained lease나 다른 library가
enqueue한 작업에도 영향을 줄 수 있다. unfinished `CudaPendingFill`의 `Drop`도
먼저 launch stream을 synchronize하고, buffer fallback close가 필요하면
context-wide device synchronize까지 수행할 수 있다. 독립적인 정상 완료 관측은
가능하면 명시적 stream/event synchronize와 `CudaPendingFill::finish`를 사용한다.

safe Rust wrapper의 thread/lifetime 계약은 다음과 같다.

- `CudaRuntime`, `CudaDevice`, `DeviceProperties`는 owned/cached host metadata다.
- `CudaContext`는 `Send + Sync`인 primary-context lease다. child는 내부 `Arc`를
  보유하므로 context native handle이 먼저 파괴될 수 없다. `CudaContext::close`
  는 자신을 소비하며 child 또는 공유 reference가 남아 있으면 Rust-side
  `InvalidState/Validation`을 반환한다.
- `CudaStream`과 `CudaEvent`는 host thread 사이로 이동할 수 있는 `Send`지만
  의도적으로 `!Sync`다. ordering/state를 바꾸는 operation은 `&mut self`를
  요구한다. C ABI의 `NOT_READY`는 두 safe `query` method에서 오류가 아닌
  `Ok(false)`로 변환되고, 완료는 `Ok(true)`다.
- `CudaKernel`은 context에 묶인 immutable AOT smoke-kernel handle이며
  `Send + Sync`다. 범용 module/kernel loader가 아니다.
- `CudaPendingFill`은 launch stream을 완료까지 exclusive borrow한다. `finish`가
  synchronize, host copy, explicit buffer close의 오류를 반환하는 정상 경로다.
  unfinished value의 `Drop`은 stream synchronize와 buffer close를 best-effort로
  수행하지만 그 오류를 호출자에게 보고할 수 없다. borrowed `CudaStream`과 함께
  host thread 사이로 이동할 수 있는 `Send`지만 shared mutation을 허용하는
  `Sync`는 아니다.

## stream, event와 비동기 완료

- 모든 stream은 `cudaStreamNonBlocking`으로 명시 생성한다. null/default stream을
  API로 노출하지 않으며 legacy default-stream ordering에 의존하지 않는다.
- event는 timing-enabled 상태로 생성된다. ordering은 `event_record`와
  `stream_wait_event`로 명시하고 elapsed time은 같은 context의 두 event에만
  요청한다.
- launch 함수의 성공은 enqueue와 즉시 launch 검사가 성공했다는 뜻이지 device
  실행 완료를 뜻하지 않는다. 실행 중 발생한 late/asynchronous 오류는
  `stream_synchronize`, `event_synchronize` 또는 `context_synchronize`에서
  `SYNCHRONIZE` stage로 관측될 수 있다.
- `smoke_copy_to_host`는 host output capacity를 element 단위로 검증하고, launch에
  사용한 non-default stream의 완료를 먼저 확인한 뒤 synchronous device-to-host
  copy를 수행한다. 따라서 caller-owned host pointer를 ABI 반환 뒤 보유하지 않으며
  완료된 결과만 반환한다. 범용 allocation/copy API가 아니라 PR 03 진단용
  buffer에만 적용된다.
- resource를 닫기 전 사용 중인 stream을 동기화하고, event/buffer/stream을
  context보다 먼저 닫는다. destructor가 암묵적으로 모든 비동기 오류를
  보고한다고 가정하지 않는다.

PR 03의 7-test GPU smoke는 정상 fill의 명시적 완료와 context를 poison하지 않는
invalid-launch 진단만 사용한다. illegal access나 device assert 같은 late fault는
공유 primary context를 poison하여 다른 lease/library의 후속 synchronize와 close,
resource-leak 관측, compute-sanitizer zero-error gate까지 오염시킬 수 있다. 그런
fault injection은 같은 프로세스의 lifecycle smoke에 섞지 않고, 별도 process와
복구 정책을 갖춘 [PR 16 오류 격리 gate](../deploy/16-reliability-and-release.md#오류-격리)에서
검증한다.

## 오류와 언어 경계

native 함수는 null pointer, struct size, device ordinal, element-count 산술과
context ownership을 사용 전에 검사한다. CUDA API 오류는 즉시 stable status로
mapping되고 원래 code와 CUDA 문자열은 caller-owned error buffer에 복사된다.
kernel enqueue 직후의 launch 오류는 `LAUNCH`, 완료 시 드러난 오류는
`SYNCHRONIZE`로 구분한다. C++ exception과 Rust panic은 이 ABI를 통과하지
않는다.

Rust API는 native error buffer를 호출 중에만 빌려 쓰고 반환 전에 owned
`CudaError`로 복사한다. `CudaErrorKind`는 stable high-level 분류이며
`CudaErrorDomain`, `CudaErrorStage`, `native_code`, `operation`, `message`가
진단 context를 보존한다. `cuda` feature가 꺼진 build에서 초기화 시도는 native
symbol을 찾거나 dynamic loading을 시도하지 않고 `Unavailable/Rust/Initialize`
오류와 `cuda` feature를 켜라는 진단을 반환한다.

## PR 03 경계

v1의 allocation과 kernel은 lifecycle 및 error propagation을 검증하는
`RustInferCudaSmokeBuffer`와 fill smoke operation으로 제한한다. 범용 tensor,
allocator/pool, arbitrary device pointer, model loading/operation, cuBLASLt,
CUTLASS, NVRTC, Triton, CUDA Graph는 ABI v1의 PR 03 surface가 아니다. 범용
device allocation과 tensor ownership은 PR 04에서 이 ABI의 ownership과
stream-ordering 원칙 위에 별도로 정의한다.
