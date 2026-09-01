# C05 — CUDA Graph Ownership ABI

**상태:** In progress — C05-4는 실제 thread-local capture owner와 abort/recovery를 닫는다. graph end·instantiate·replay는 C05-5로 분리한다.
**의미 등급:** `E0` infrastructure  
**한 가지 목적:** CUDA Graph capture·instantiate·replay·close를 안전하게 소유하는 additive native C ABI와 Rust wrapper를 구현한다.

[이전: C04](04-llama-executor-refactor.md) | [목차](README.md) | [다음: C06](06-graph-signature-dispatcher.md)

## 1. 배경

Riley는 non-default stream, event, command batch와 resource ledger를 이미 갖고 있지만 CUDA Graph lifecycle은 production ABI에 없다. Graph를 Llama executor 안에서 바로 구현하면 native handle, static address, stream capture, failure recovery가 model code와 섞인다.

이 PR은 실제 Llama decode graph를 만들지 않는다. model-independent graph ownership과 오류 계약만 닫는다.

### C05-0 — graph contract foundation (CPU/ABI-only)

`RileyCudaErrorInfo`의 272-byte v1 layout, 기존 status/domain/generic stage는 불변으로 유지한다.
대신 header는 opaque graph/capture/exec handle, thread-local capture mode, `unknown | unsupported |
supported` closed capture-capability vocabulary, 상세 graph stage와 56-byte
`RileyCudaGraphErrorInfo` companion record를 additive하게 선언한다. `unknown`은 zero-initialized
default이며 반드시 거부한다. 이 record는 capture/exec ID, submission/completion/resource-release
certainty, poison 상태를 보존하며 기존 error buffer의 tail extension이 아니다. C11/C++/Rust layout test는
size, alignment, all critical offset과 enum numeric contract를 고정한다.

Rust의 `graph` module은 lifecycle transition을 CPU에서도 fail-closed로 검증한다. feature-off
`CudaStream::begin_graph_capture`는 fake graph나 eager fallback 없이 actionable unavailable error를
반환한다. 실제 native
capture/instantiate/replay/close symbol, resource lease, operation-specific capture capability query와 GPU
lifecycle/fault/performance test는 후속 vertical slice로 남긴다. 따라서 이 slice는 GPU parity나 launch
overhead 개선을 주장하지 않는다.

### C05-1 — native capture-begin fail-closed vertical slice (CPU/ABI-only)

`riley_cuda_graph_capture_begin` 하나를 static native archive와 Rust FFI에 연결한다. 유효한
`out_capture`은 항상 먼저 null로 만들고, optional `RileyCudaGraphErrorInfo`는 forward-compatible
`struct_size`를 보존한 채 `CAPTURE_BEGIN`과 zero ID/flag로 초기화한다. malformed companion record는
generic validation error로 거부하고 기록을 쓰지 않는다.

이 slice는 `cudaStreamBeginCapture`, current-context 전환, stream dereference, command-batch/resource
lease 변경, graph allocation을 수행하지 않는다. 대신 thread-local mode와 non-null stream을 검증한 뒤
native `NOT_SUPPORTED`을 반환한다. 따라서 CUDA-enabled Rust build도 Rust-only placeholder가 아닌 native
ABI 결과를 받지만, 성공 capture handle은 만들지 않는다. C11 source ABI test와 CPU-only source contract는
새 symbol/wiring과 이 fail-closed 제한을 고정하며 GPU 실행·성능 개선은 주장하지 않는다.

### C05-2 — canonical graph-failure companion decoder (CPU/ABI-only)

C05-2는 graph module에 하나의 private Rust C-layout mirror와 canonical decoder를 둔다. decoder는
v1 required prefix보다 큰 forward-compatible record를 허용하지만, reserved field, 네 ABI boolean,
stage와 zero/nonzero capture·exec ID를 모두 엄격히 해석한다. unknown stage는 success로 바꾸지 않고
unknown value로 보존하며 malformed companion은 Internal/Validation error로 fail-closed한다.

C05-1 capture-begin stub은 generic decoder보다 더 좁게 exact v1 record size, CaptureBegin stage,
zero ID와 zero lifecycle flags를 계속 요구한다. valid native NOT_SUPPORTED result는 원래 CudaError
경로를 그대로 유지한다. 이 slice는 native capture 시작, non-null capture handle, stream/context 또는
command-batch lease 변화, abort/end/instantiate, graph allocation, replay, GPU execution과 성능 주장을
추가하지 않는다. 실제 owner는 valid handle의 consume-on-error, abort/recovery, context와 resource lease
정책을 함께 닫는 후속 slice에서만 도입한다.

### C05-3 — type-level exclusive stream lease (CPU-only)

`GraphCapture<'stream>`은 기존 mutable stream borrow와 `!Send + !Sync` marker를 compile-fail
doctest로 고정한다. live capture 동안 같은 stream에 query, synchronize, command-batch begin,
close를 호출할 수 없고, `GraphCapture<'static>`을 다른 thread로 보내거나 공유할 수도 없다.

이 slice는 type-only ownership contract다. 성공 capture를 만들지 않으며 native handle, C ABI,
FFI, stream/context mutation, command-batch/resource lease, abort/end/instantiate/replay를 추가하지
않는다. 따라서 실제 capture owner는 non-null native handle의 consume-on-error, invalidation abort와
recovery, resource lifetime을 함께 닫는 다음 native slice에서만 도입한다.

### C05-4 — native capture owner and abort recovery (CUDA)

`riley_cuda_graph_capture_begin`은 이제 `cudaStreamBeginCapture`를 실제 호출하고, 성공한 경우에만
non-null `RileyCudaGraphCapture` owner와 non-zero capture ID를 돌려준다. owner는 시작 stream/context,
thread-local capture의 시작 host-thread token, 그리고 stream exclusive-use lease를 함께 보관한다.
따라서 **같은 host thread**에서 nested capture, active command batch, CUDA query/synchronize/close 및 다른 일반
CUDA stream 작업은 CUDA 호출 전에 거부된다. 이 gate는 `ThreadLocal` capture owner와 정확히 결합되며, CUDA가
정의한 ThreadLocal semantics처럼 다른 host thread의 independent stream work를 전역으로 serialize하지 않는다.
단, `cudaDeviceSynchronize`·`cudaMemGetInfo`·primary-context retain/release처럼 capture stream을 포함하는
**context-wide control**은 다른 host thread에서도 안전한 독립 작업이 아니다. primary-context별 active-capture
domain의 짧은 admission lock이 control lease와 capture begin의 check-and-mark를 단일 전이로 직렬화한다.
따라서 다른 wrapper/thread가 그런 control을 CUDA에 넘기기 전에 거부되며, 관측 순서 경쟁으로 둘 다 CUDA에
들어갈 수 없다. capture가 끝난 뒤에는 그 lease도 해제되어 independent work는 정상적으로 재개된다.

진단용 `CudaPendingFill`과 async H2D/D2H copy는 예외다. 이 값들의 finish/Drop은 allocation,
completion 또는 release를 수행하므로, 모든 smoke buffer create(0-element 포함)는 `cudaMalloc` **전**,
copy는 `cudaMemcpyAsync`와 token allocation **전** 같은 global+primary-context admission lock에 pending
lease를 게시하고 native owner가 확실히 소비될 때까지 유지한다. capture begin은 어느 device의 pending
lease라도 있으면 거부하고, capture가 먼저 시작된 경우에는 다른 host thread의 `CudaKernel::launch_fill`과
copy submission도 CUDA 호출 전에 거부한다. 하나의 native smoke buffer는 재launch되어도 create-time
lease 하나만 유지한다. 따라서 abort 뒤 재시도 가능한 `InvalidState` 때문에 Rust Drop이 native child
owner를 잃는 경로를 만들지 않는다. 이는 diagnostic pending lifecycle만 보수적으로 직렬화하며, C05-4가
일반 capture enqueue/replay를 지원한다는 의미는 아니다.

새 additive `riley_cuda_graph_capture_abort(RileyCudaGraphCapture** ...)`는 one-shot owner를 받는다.
abort는 시작 thread에서만 `cudaStreamEndCapture`를 호출하고, 반환된 (empty 또는 invalidated) graph는
즉시 destroy한다. end/destroy/context restoration이 모두 확실할 때만 capture child와 stream lease를
해제한다. 이 시점에만 capture 중 Drop/close된 foreign 또는 same-context stream, event, context,
device/pinned buffer, cuBLASLt/HF plan의 embedded deferred-close FIFO를 순서대로 drain한다. 어느 한
단계나 deferred close가 불명확하면 raw handle은 재시도하지 않도록 consume하고, native owner와 stream
lease를 의도적으로 retained/poisoned 상태로 남긴다. 성공하지 않은 begin이 실제 capture를 시작했을
가능성이 있으면 output owner를 통해 Rust boundary가 같은 abort recovery를 먼저 시도한다.

Rust `GraphCapture<'stream>`은 이제 phantom marker가 아니라 실제 FFI owner와 mutable `CudaStream`
borrow를 함께 소유한다. `abort(self)`와 Drop은 정확히 한 번만 native abort를 시도한다. 이 slice는
capture 안에 enqueue 가능한 public operation, `capture_end -> CapturedGraph`, instantiate, launch,
replay 또는 성능 향상을 추가하지 않는다. GPU test는 begin → abort/Drop → 동일 stream eager fill의
recovery와 repeated lifecycle close, 그리고 같은 primary context의 다른 host thread에서 context-wide control이
거부되고 abort 뒤 다시 허용되는지를 검증한다.

### C05-5 — admitted operation, graph end, and replay (planned)

C05-5는 C05-4의 recovery-proven capture owner 위에서만 시작한다. 사전 할당된 capture-safe custom
fill을 명시 whitelist로 넣고, `CapturedGraph`/`GraphExec`/`GraphLaunch`의 end·instantiate·replay·completion
owner를 추가한다. graph exec가 retained resource lease를 소유하고 launch completion 전 close를 막는
계약, multi-replay parity 및 foreign-stream rejection을 이 slice에서 검증한다. H2D/D2H chain,
cuBLASLt capture, node update, fault-injection 및 1000-replay performance claim은 그 이후 slice로 남긴다.

## 2. 범위

### 포함

- stream capture begin/end/abort
- captured graph instantiate
- graph executable launch
- 제한된 node parameter update capability
- graph/graph-exec destroy
- resource retention과 close ordering
- capture/instantiate/launch/update 오류 분리
- public C ABI layout 및 Rust safe wrapper
- feature-off stub와 source contract
- 실제 GPU lifecycle/fault/sanitizer test

### 비범위

- Llama/Qwen graph signature
- decode bucket capture
- graph cache/dispatcher
- production default 변경
- kernel fusion
- runtime JIT/NVRTC

## 3. Native 상태 모델

```text
Uninitialized
  -> Capturing
  -> Captured
  -> Instantiated
  -> Launching
  -> Instantiated
  -> Closed

모든 상태
  -> Poisoned   # completion 또는 resource 상태가 불명확
```

허용되지 않는 상태 전이는 CUDA 호출 전에 fail-closed한다.

예:

- Capturing 중 nested capture 거부
- Instantiated 전 launch 거부
- Launching 또는 completion 미확정 상태의 close 거부
- Closed handle 재사용 거부
- Poisoned handle update/launch 거부

## 4. 제안 C ABI

실제 이름은 ABI v1 naming 규칙을 따른다.

```c
typedef struct RileyCudaGraph RileyCudaGraph;
typedef struct RileyCudaGraphExec RileyCudaGraphExec;

RileyStatus riley_cuda_graph_capture_begin(
    RileyCudaStream* stream,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture);

RileyStatus riley_cuda_graph_capture_end(
    RileyCudaGraphCapture* capture,
    RileyCudaGraph** out_graph);

RileyStatus riley_cuda_graph_capture_abort(
    RileyCudaGraphCapture* capture);

RileyStatus riley_cuda_graph_instantiate(
    RileyCudaGraph* graph,
    RileyCudaGraphExec** out_exec);

RileyStatus riley_cuda_graph_exec_launch(
    RileyCudaGraphExec* exec,
    RileyCudaStream* stream,
    RileyCudaCompletionToken** out_completion);

RileyStatus riley_cuda_graph_exec_close(RileyCudaGraphExec* exec);
RileyStatus riley_cuda_graph_close(RileyCudaGraph* graph);
```

Node update는 arbitrary raw pointer API로 열지 않는다. C06에서 필요한 closed update descriptor가 정해지기 전에는 capability query와 최소 memcpy/kernel parameter update만 추가한다.

## 5. Resource ownership

Graph capture가 참조한 다음 resource는 graph exec보다 오래 살아야 한다.

- device/pinned buffers
- GEMM plan/handle
- kernel parameter storage
- module/function handle
- stream/context owner

기존 command-batch resource ledger와 별개로 graph exec는 deduplicated immutable resource set을 소유한다. Graph exec close가 성공하기 전 resource의 explicit close는 `busy`로 실패해야 한다.

Graph launch마다 resource refcount를 증가시키지 않도록 cold instantiate 시 고정 lease를 잡고, in-flight launch는 별도 bounded counter/completion token으로 관리한다.

## 6. Rust safe API

```rust
GraphCapture<'stream>
CapturedGraph
GraphExec
GraphLaunch<'exec, 'stream>
```

원칙:

- `GraphCapture`는 capture stream을 mutable borrow한다.
- capture가 끝나기 전 stream을 일반 command에 사용할 수 없다.
- `GraphExec`는 retained resource owner를 내부에 보관한다.
- `GraphLaunch` completion 전 `GraphExec`와 참조 buffer를 close할 수 없다.
- native handle은 `Send/Sync`를 자동 구현하지 않고 실제 thread/stream 계약에 맞춰 명시한다.
- C ABI를 가로질러 unwind하지 않는다.

compile-fail doctest로 early drop, double mutable borrow, launch 중 close 시도를 고정한다.

## 7. Capture-safe whitelist

C05 시점에는 다음 operation만 graph capture admission 대상으로 선언한다.

- validated fixed-size H2D/D2H memcpy
- 기존 custom CUDA kernel launch
- capture 지원이 확인된 cuBLASLt matmul plan
- event-free same-stream dependency

다음은 기본 거부한다.

- host callback
- dynamic allocation/free
- stream/context create/destroy
- capture 중 synchronize/query
- unsupported library call
- pointer/lifetime이 검증되지 않은 external backend

operation별 capture capability는 bool이 아니라 `supported | unsupported | unknown` closed enum으로 제공한다. `unknown`은 거부다.

## 8. 오류 계약

오류에는 최소 다음 metadata를 보존한다.

```text
CUDA domain/status
stage: begin | enqueue | end | instantiate | update | launch | completion | close
capture ID / exec ID
submission_started
completion_known
resource_release_known
poisoned
```

capture 실패가 stream 전체를 항상 poison하지는 않는다. CUDA가 capture invalidation 상태를 보고하면 명시적 abort/stream recovery를 수행하고, recovery가 확인된 경우에만 eager fallback에 stream을 재사용한다.

launch 후 completion 미확정 오류는 graph exec와 resource lease를 유지하고 상위 runtime에 보수 오류를 반환한다.

## 9. 예상 파일 변경

```text
kernels/include/riley_cuda.h
kernels/src/graph.cu
kernels/src/ffi_internal.hpp
kernels/src/smoke_fill.cu
kernels/CMakeLists.txt
kernels/tests/abi_layout.c
crates/riley-cuda/src/graph.rs
crates/riley-cuda/src/ffi.rs
crates/riley-cuda/src/lib.rs
crates/riley-cuda/tests/graph_cpu.rs
crates/riley-cuda/tests/graph_gpu.rs
crates/riley-cuda/tests/graph_compile_fail.rs
```

C ABI는 기존 symbol/layout을 깨지 않는 additive 변경이어야 한다.

## 10. 테스트

### Feature-off/CPU

- header/source symbol inventory
- enum/layout/size/alignment C11 compile
- invalid state transition
- null/foreign/stale handle rejection
- Rust compile-fail lifetime tests
- graph module이 model/runtime/server를 import하지 않는 boundary test

### GPU

- fill kernel 2~3개 capture/instantiate/1000 replay
- fixed memcpy+kernel chain parity
- 두 stream/context handle 혼용 거부
- launch completion 전 resource close 거부
- capture abort 후 stream 재사용
- instantiate/launch fault injection
- explicit close 후 live native/Rust allocation 0

### Sanitizer

- memcheck
- racecheck 가능 범위
- repeated create/replay/close leak smoke

## 11. Performance scope

이 PR은 LLM latency 개선을 주장하지 않는다. graph wrapper 자체의 launch overhead microbenchmark만 기록한다.

- direct kernel chain submission
- captured graph replay
- wrapper/native API overhead

결과는 C07의 가설 입력이며 production promotion 근거가 아니다.

## 12. 승인 기준

- additive ABI와 layout test 통과
- 1000회 replay output exact
- capture abort/recovery 후 stream 정상 사용
- invalid transition이 device mutation 전 거부
- in-flight close/use-after-free/double-close 0
- completion 미확정 fault가 성공으로 보고되지 않음
- close 후 Riley-owned native/Rust allocation 0
- production model/runtime behavior와 default 변화 없음

## 13. 롤백

header, native implementation, Rust wrapper, test를 같은 ABI addition 범위로 revert한다. C06 이후 consumer가 생기기 전에는 완전 revert가 가능하다. ABI가 release된 이후에는 symbol 삭제 대신 deprecated stub와 version transition 문서가 필요하다.

## 14. 완료 정의

model code가 raw CUDA Graph handle이나 unsafe pointer를 직접 다루지 않고도 safe Rust owner를 통해 capture/replay/close할 수 있으며, 모든 실패 경로의 completion과 resource 상태가 명확할 때 완료다.
