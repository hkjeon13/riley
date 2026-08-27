# C05 — CUDA Graph Ownership ABI

**상태:** Planned  
**의미 등급:** `E0` infrastructure  
**한 가지 목적:** CUDA Graph capture·instantiate·replay·close를 안전하게 소유하는 additive native C ABI와 Rust wrapper를 구현한다.

[이전: C04](04-llama-executor-refactor.md) | [목차](README.md) | [다음: C06](06-graph-signature-dispatcher.md)

## 1. 배경

Riley는 non-default stream, event, command batch와 resource ledger를 이미 갖고 있지만 CUDA Graph lifecycle은 production ABI에 없다. Graph를 Llama executor 안에서 바로 구현하면 native handle, static address, stream capture, failure recovery가 model code와 섞인다.

이 PR은 실제 Llama decode graph를 만들지 않는다. model-independent graph ownership과 오류 계약만 닫는다.

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
