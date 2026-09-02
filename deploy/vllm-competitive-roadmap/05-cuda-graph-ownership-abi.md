# C05 — CUDA Graph Ownership ABI

**상태:** In progress — C05-18까지 C07 pure-decode의 indexed BF16 RoPE와 multi-row paged `KvWrite` 한 kernel씩을 independent fixed-address CUDA Graph lifecycle/parity로 닫았다. C05-16의 raw result receipt는 계속 C07 `GpuGreedy`/`CompletionBoundary`나 executor integration을 뜻하지 않으며, C05-17/18도 full decode graph 또는 성능 향상 근거는 아니다.
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

### C05-5 — admitted fill operation, graph end, and replay (in progress)

C05-5는 C05-4의 recovery-proven capture owner 위에서만 동작한다. 유일한 whitelist는 caller가
사전 할당한 하나의 `CudaDeviceBuffer`에 대해 같은 fixed shape의 f32 fill kernel을 한 번 이상 순차
enqueue하는 것이다. begin은 buffer active-use lease와 최종 graph의 host storage를
`cudaStreamBeginCapture` **전**에 확보하고, enqueue는 value만 받으며 host allocation/free를 수행하지
않는다. 적어도 하나의 성공한 enqueue가 있어야 end가 가능하다.

`GraphFillCapture<'stream, 'buffer>` → `CapturedGraph<'stream, 'buffer>` →
`GraphExec<'stream, 'buffer>` → `GraphLaunch<'exec, 'stream, 'buffer>` safe owner는 capture stream과
buffer를 계속 mutable-borrow한다. native graph/exec도 capture context child, **정확히 그** stream과
buffer의 active-use lease를 유지한다. launch는 해당 capture stream 외의 raw stream pointer를 CUDA 호출 전에
거부하고, `GraphLaunch::finish`가 completion 경계다. completion 전 exec close는 거부되며 close가 성공한
뒤에만 stream/buffer와 D2H를 재사용할 수 있다.

GPU regression은 fill node 3개를 capture한 뒤 instantiate하고 1,000회 `launch → finish`를 순차 실행한다.
마지막 fill 값의 bit-exact D2H 결과와 모든 explicit close 뒤 allocation statistics 0을 확인한다. 이 1,000회는
correctness/lifetime regression이지 성능 향상 주장이 아니다. fixed H2D/D2H capture chain, cuBLASLt capture,
node update, fault injection과 latency microbenchmark는 후속 slice로 남긴다.

### C05-14 — fixed-address BF16 row-gather capture (CUDA; completed)

C05-13의 deterministic BF16 argmax는 decode 출력의 마지막 selection kernel만 닫았으며, 실제
eager output path는 먼저 dense logits에서 output-row permutation을 gather한다. C05-14는 그 선행
primitive를 독립 one-node graph로 검증한다. 이것은 C07 executor나 GPU-greedy capability를
승격하는 작업이 아니다.

capture begin은 같은 context의 세 **서로 다른** 고정 device allocation과 immutable geometry를 받는다.
입력은 BF16 `[input_row_count, column_count]`, index는 U32 `[output_row_count]`, 출력은 BF16
`[output_row_count, column_count]`다. `input_row_count`, `output_row_count`, `column_count`는 모두
nonzero이며 모든 element/byte product와 allocation capacity를 CUDA capture 전에 checked arithmetic으로
검증한다. safe begin은 eager `row_gather`와 같은 temporary `row_indices_host` mirror를 받아
길이·in-range·unique validation을 native entry 전에 끝내지만 그 host slice를 graph lifetime에 보관하지
않는다. index H2D staging은 capture 밖의 기존 owner가 계속 소유한다. graph node는 이미 고정 주소에 있는
device index bytes만 읽으며, raw C caller의 malformed device index에 대한 eager row-gather의 sentinel/NaN behavior도
바꾸지 않는다. safe API는 duplicate 또는 out-of-range host mirror를 admit하지 않는다.

safe owner는 stream/input/index/output을 capture → graph → exec → launch completion → close까지
by-value로 유지한다. native capture/graph/exec은 같은 exclusive-use lease를 보유하고, enqueue는
정확히 한 번의 BF16 row-gather kernel만 allocation·synchronize·node update 없이 기록한다. fresh
inputs, spans, offsets, aliasing, H2D/D2H, argmax, host result handling, sampling, C07 evidence,
dispatcher/executor wiring은 모두 범위 밖이다. 따라서 기존 C05-13 `Supported` 또는 이 slice의
`Supported` 어느 쪽도 단독으로 C07 `GpuGreedy`/`CompletionBoundary`를 지원한다고 해석할 수 없다.

GPU acceptance는 eager와 graph의 exact BF16 output bytes를 valid unique permutation에서 비교하고, 최소
64회 sequential replay, second-enqueue rejection, duplicate/out-of-range host-mirror preflight rejection,
preflight failure 후 untouched-resource recovery, abort recovery, explicit close 뒤 allocation 0을 확인했다.
추가로 raw C/device-index OOB는 eager와 동일한 BF16 NaN sentinel bytes를 내는지 private CUDA test로
확인했다. 이 결과는 one-node row-gather lifecycle/parity의 근거일 뿐 C07 승격 근거는 아니다.

### C05-15 — fixed-address BF16 row-gather → argmax capture (CUDA; completed)

C05-15는 C05-14의 output-row gather와 C05-13의 deterministic BF16 argmax를 **하나의** fixed-address
CUDA Graph capture 안에 순서대로 기록하는 두-node, device-only vertical slice다. 기존 C05-14 exec를
finish한 뒤 C05-13 exec를 launch하는 것은 두 독립 replay일 뿐 graph 내부 dependency와 하나의
four-resource lifecycle을 증명하지 못하므로 이 slice의 구현으로 허용하지 않는다.

capture begin은 같은 context의 네 **서로 다른** 고정 device allocation과 immutable geometry를 받는다:
BF16 input `[input_row_count, vocabulary_size]`, U32 row-index `[output_row_count]`, gathered BF16
`[output_row_count, vocabulary_size]`, `RileyCudaBf16ArgmaxResult` `[output_row_count]`다.
`input_row_count`, `output_row_count`, `vocabulary_size`는 nonzero이며 `vocabulary_size <= u32::MAX`다.
모든 element/byte product, result-byte product, allocation capacity와 네-way alias는 CUDA capture 전에
checked arithmetic으로 검증한다. safe begin은 C05-14와 동일한 temporary `row_indices_host` mirror를
기존 eager validator로 길이·in-range·unique 검사한 뒤, `output_row_count`도 그 mirror 길이에서만
유도한다. 그 host slice는 즉시 버리고 capture/graph/exec/replay owner에 보관하지 않는다. index H2D
staging은 capture 밖의 기존 owner가 계속 소유한다.

enqueue는 allocation, synchronize, node update 없이 같은 capture stream에 정확히 row-gather kernel 한 번,
그 gathered BF16 allocation을 logits로 쓰는 deterministic argmax kernel 한 번을 기록한다. native
capture/graph/exec과 safe owner는 stream/input/index/gathered/result 네 allocation의 exclusive-use lease를
capture → graph → exec → launch completion → close까지 by-value로 유지한다. raw C caller의 malformed
device index는 existing eager row-gather와 같은 BF16 NaN sentinel을 gathered buffer에 쓰며, 뒤 argmax의
non-finite status/result bytes도 eager `row_gather → deterministic_bf16_argmax` chain과 정확히 일치해야
한다.

새 additive capability/operation은 named `Bf16RowGatherArgmax = 9`이고, atomic composite의 begin/enqueue
C ABI만 새로 둔다. C05-13의 `Bf16Argmax = 7` 또는 C05-14의 `Bf16RowGather = 8` `Supported`를 조합해
이 chain을 admit하거나 capability 9를 대신할 수 없다. 두 kernel launch가 모두 CUDA success를 반환한
뒤에만 capture enqueue-count를 complete로 기록한다. 첫 node가 기록된 뒤 둘째 launch 또는 post-launch가
실패하면 capture는 partial-capture terminal 상태가 된다. 같은 owner의 re-enqueue, end, instantiate는
CUDA 호출 전에 거부하고 one-shot abort만 허용한다. abort/end-capture/destroy/context restoration과 네 lease
release가 모두 known일 때만 stream/resources를 회복해 돌려주며, 하나라도 불명확하면 C05-4와 같이 owner와
lease를 poisoned-retained로 남긴다. `gathered`는 이 두 node 사이에서만 의도적으로 공유되는 allocation이고,
두 독립 graph owner/lease로 이를 표현하지 않는다.

이 slice의 `GraphLaunch::finish`는 네 device allocation을 재사용·close할 수 있게 하는 graph lifecycle
completion일 뿐 token/status D2H, host result validation, scheduler commit 또는 consumer-visible
`CompletionBoundary`를 뜻하지 않는다. H2D input/index staging, D2H token/status transfer, spans/offsets,
sampling, C07 capability/evidence, graph identity, dispatcher/executor wiring은 모두 범위 밖이다. 따라서
새 `Supported` capability도 단독으로 C07 `GpuGreedy`나 `CompletionBoundary`를 지원한다고 해석할 수 없다.

GPU acceptance는 valid unique permutation에서 eager two-kernel chain과 gathered BF16 및 result-record의
exact bytes를 비교하고, 최소 64회 sequential replay, second-enqueue rejection, 네-way alias/geometry/
capacity/foreign-context/duplicate/out-of-range host-mirror preflight rejection, preflight failure 뒤
untouched-resource recovery, abort·explicit-close 뒤 allocation 0을 확인한다. 별도 private CUDA test는
raw device-index OOB가 eager와 같은 gathered NaN 및 argmax
`INVALID_TOKEN_ID`/non-finite result bytes를 내는지 확인한다. source-contract/fault path는 첫 node 뒤
둘째 node failure에서 end가 거부되고 known abort recovery만 stream/resources를 돌려주는지도 확인한다.
원격 RTX 4090/CUDA 12.8에서 native ABI link, CUDA library, 전체 graph GPU 회귀를 통과했고, valid
unique permutation의 64회 replay는 gathered BF16/result-record 모두 eager와 byte-exact였다. preflight와
abort recovery, raw device-index OOB의 gathered NaN과 `INVALID_TOKEN_ID`/non-finite result-record bytes도
별도 GPU test로 닫았다. 이 결과는 device-only two-node graph의 근거일 뿐 C07 승격 근거는 아니다.

### C05-16 — fixed-address BF16 row-gather → argmax → result D2H capture (CUDA; completed)

C05-16은 C05-15의 동일한 output boundary를 한 capture 안에서 **세** fixed-address node로 좁힌다:
BF16 row-gather, gathered logits의 deterministic BF16 argmax, 그리고 `RileyCudaBf16ArgmaxResult`의 exact
pinned-host D2H다. C05-15 exec가 끝난 뒤 일반 D2H token을 따로 submit하는 것은 두 독립 lifecycle일 뿐
graph 내부 dependency, fixed host destination 및 하나의 completion receipt를 증명하지 못하므로 이 slice의
구현으로 허용하지 않는다.

capture begin은 같은 primary context의 stream, 네 **서로 다른** device allocation(BF16 input, U32
row-index, gathered BF16, device result records), 그리고 한 pinned-host result allocation을 by-value로 받는다.
geometry는 C05-15와 같고, pinned result byte length는 `output_row_count * sizeof(RileyCudaBf16ArgmaxResult)`와
정확히 같아야 한다. 모든 element/byte product, device/pinned capacity, four-way device alias, context와 idle
lease를 CUDA capture 전에 checked arithmetic으로 검사한다. safe begin은 C05-15와 같은 temporary
`row_indices_host` mirror로 output row count와 eager-safe unique/in-range validation만 유도하며, host mirror와
host consumer slice는 capture lifetime에 보관하지 않는다. input/index H2D staging도 계속 capture 밖의 owner가
소유한다.

enqueue는 allocation, synchronize, node update 없이 같은 capture stream에 gather kernel 한 번, argmax kernel
한 번, fixed device-result-to-pinned-host `cudaMemcpyAsync` 한 번을 그 순서로 기록한다. native
capture/graph/exec과 safe owner는 stream 및 네 device/pinned allocation의 exclusive-use lease를 capture → graph
→ exec → launch completion → close까지 보유한다. 세 node 중 어느 submission 또는 post-launch context
restoration이 실패해도 complete enqueue-count를 기록하지 않는다. recorded prefix가 있을 수 있으므로 re-enqueue,
end, instantiate는 CUDA 호출 전에 거부하고 one-shot abort만 허용한다. end-capture/destroy/context restoration과
다섯 allocation lease release가 모두 known일 때만 resources를 회복하며, 불명확하면 owner와 lease를
poisoned-retained로 남긴다.

성공한 `GraphLaunch::finish` 뒤에만 op-specific completion receipt/view가 pinned result allocation에서
caller-provided byte slice로 exact record bytes를 읽을 수 있게 한다. finish 전에는 pinned read/write/close를
거부하고, receipt는 token/status를 해석·검증하거나 scheduler를 commit하지 않는다. 따라서 이 completion은 raw
D2H visibility/lifecycle fact일 뿐 consumer-visible `CompletionBoundary`가 아니다. raw malformed device index는
C05-15와 같은 gathered NaN과 argmax non-finite result record를 만들고, D2H bytes도 eager chain의 bytes와
정확히 같아야 한다.

새 additive capability/operation은 `Bf16RowGatherArgmaxD2H = 10`이며 C05-9의 native vocabulary만 확장한다.
capability 10의 `Supported`는 C05-15 capability 9 또는 existing H2D capability를 조합해 대체할 수 없다.
또한 C07 `GpuGreedy`와 `CompletionBoundary`는 계속 `Unknown`이다. 이 slice는 C06 registry/dispatch,
C07 inventory mapping, executor wiring, graph identity, metrics, host result validation, scheduler commit, sampling,
fresh input/node update를 만들거나 허용하지 않는다.

GPU acceptance는 eager gather → argmax → D2H와 captured gathered/result-pinned bytes의 exact parity, 최소 64회
sequential replay, second-enqueue rejection, foreign context·four-way device alias·pinned length/busy·geometry/capacity·
duplicate/out-of-range host-mirror preflight rejection, finish 전 pinned CPU access rejection, finish 뒤 exact read,
abort/explicit close 뒤 allocation statistics 0을 확인한다. private CUDA test는 raw OOB device index의 gathered
NaN 및 `INVALID_TOKEN_ID`/non-finite record D2H bytes parity를 확인하고, source/fault contract는 둘째 또는 셋째
node failure가 abort-only terminal state가 됨을 고정한다.

원격 RTX 4090/CUDA 12.8에서 native ABI link, CUDA feature library, source-contract, 그리고 전체 ignored graph GPU
suite 37개를 통과했다. 정상 valid-permutation은 64회 replay에서 eager와 gathered BF16 및 raw D2H
result-record bytes가 정확히 일치했고, exact pinned-size preflight/abort recovery와 raw device-index OOB의
NaN·non-finite result bytes도 별도 GPU regression으로 닫았다. 이 결과는 세-node raw result lifecycle/parity의
근거일 뿐 host token/status validation, scheduler commit, C07 inventory 승격 또는 성능 향상 주장은 아니다.

### C05-17 — fixed-address BF16 indexed-RoPE capture (CUDA; completed)

C05-17은 C07 V1 pure-decode chain의 per-row-position RoPE primitive만 one-node CUDA Graph로 좁힌다. 이것은
기존 eager `indexed_rope`의 non-interleaved Llama BF16 semantics와 raw device-position OOB의 BF16-NaN
sentinel을 보존하는 ownership/parity slice이며, embedding, Q/KV projection, attention, KV mutation 또는 full
layer loop를 capture하지 않는다.

capture begin은 같은 primary context의 stream과 다섯 **서로 다른** fixed device allocation을 by-value로 받는다:
BF16 input `[active_row_count, head_count, head_size]`, F32 cosine table
`[table_position_count, rotary_dimension / 2]`, 같은 크기의 F32 sine table, U32 device positions
`[active_row_count]`, BF16 output이다. `active_row_count`, `head_count`, `head_size`, `rotary_dimension`,
`table_position_count`는 모두 nonzero이고 `rotary_dimension`은 even이며 `head_size`를 넘지 않는다. 모든
element/byte product, five-way alias, context와 idle lease를 CUDA capture 전에 checked arithmetic으로 검사한다.
safe begin은 temporary `positions_host` mirror가 `active_row_count`와 정확히 같은 길이이고 모든 값이
`table_position_count`보다 작은지만 existing eager validator와 같은 의미로 확인한다. mirror는 native begin
validation 뒤 보관하거나 graph node에 capture하지 않으며, 이미 staged된 device positions의 byte identity를
주장하지 않는다.

enqueue는 allocation, synchronize, node update 또는 host report 없이 같은 capture stream에 fixed-address indexed
BF16 RoPE kernel을 정확히 한 번 기록한다. native capture/graph/exec과 safe owner는 stream 및 다섯 allocation의
exclusive-use lease를 capture → graph → exec → launch completion → close까지 유지한다. enqueue 또는 context
restoration이 불명확해진 경우 re-enqueue/end/instantiate는 CUDA 호출 전에 거부하고 one-shot abort만 허용한다.
abort/end-capture/destroy/context restoration과 모든 lease release가 known일 때만 resource bundle을 회복하며,
그 외에는 owner와 raw addresses를 poisoned-retained로 남긴다.

새 additive capability/operation은 `IndexedRopeBf16 = 11`이다. capability 11의 `Supported`는 canonical
RMSNorm, existing generic RoPE, H2D 또는 다른 C05 operation을 조합해 대체할 수 없다. 이 slice는 H2D/D2H,
completion receipt, final norm, C06 dispatch/registry, C07 executor wiring/identity/metrics, node update, fresh
inputs, sampling, scheduler commit 또는 performance promotion을 만들지 않는다. 후속 C07 evidence adapter만 이
정확한 operation을 `Rope` slot에 매핑할 수 있고, inventory aggregate는 나머지 operation이 명시적으로
review되기 전까지 `Unknown`이다.

GPU acceptance criteria는 valid host mirror와 eager indexed-RoPE의 exact BF16 output bytes, 최소 64회 sequential replay,
second-enqueue rejection, foreign context·five-way alias·geometry/capacity·busy·host-mirror preflight rejection 뒤
untouched-resource recovery, abort/explicit close 뒤 allocation statistics 0을 확인한다. private CUDA test는 raw
device-position OOB가 eager와 동일한 BF16-NaN sentinel bytes를 내는지 확인한다. 이 결과는 one-node RoPE
lifecycle/parity의 근거일 뿐 full decode graph 또는 성능 향상 근거는 아니다.

**완료 검증 (2026-09-02):** 원격 RTX 4090/CUDA 12.8에서 native ABI link, CUDA feature library, source-contract
CPU graph 27개와 전체 ignored graph GPU suite 40개를 통과했다. 정상 valid host mirror는 eager BF16 output과
byte-exact하고, input/cos/sin/device-position bytes가 그대로인 채 64회 sequential replay하며 second enqueue를
거부했다. raw device-position OOB는 eager와 같은 rotary BF16-NaN sentinel 및 non-rotary tail-copy bytes를
보존했다. out-of-range host mirror·short output preflight와 begin 뒤 abort recovery는 resource bundle 및
allocation statistics 0을 확인했다. 이는 one-node RoPE lifecycle/parity의 실제 근거일 뿐 full decode,
C07 executor integration 또는 end-to-end 성능 향상을 입증하지 않는다.

### C05-18 — fixed-address BF16 ragged paged-KV write capture (CUDA; completed)

C05-18은 C07 V1 pure-decode의 multi-row paged K/V write primitive만 one-node CUDA Graph로 좁힌다. 이것은
기존 eager `ragged_paged_kv_cache_write`의 BF16 source-to-pool bit preservation과 packed device metadata의
raw bounds-invalid-row no-op semantics를 보존하는 ownership/parity slice이며, K/V projection, metadata H2D,
attention, scheduler logical commit, full layer loop 또는 executor integration을 capture하지 않는다.

capture begin은 같은 primary context의 stream과 아홉 **서로 다른** fixed device allocation을 by-value로 받는다:
BF16 key source와 value source `[active_row_count, KVH, D]`, BF16 key pool과 value pool
`[physical_block_count, KVH, 16, D]`, 그리고 U32 sequence-block offsets `[sequence_count + 1]`, U32
block IDs `[block_count]`, U16 valid tokens `[block_count]`, U32 row sequence slots
`[active_row_count]`, U32 row positions `[active_row_count]`이다. `sequence_count`, `block_count`,
`active_row_count`, `physical_block_count`, `KVH`, `D`는 nonzero이며 packed format version 1과 block size 16,
checked source/pool/metadata byte capacities, nine-way nonalias, same context와 idle lease를 CUDA capture 전에
검사한다.

safe begin은 existing `PackedBatchHostV1`/`PackedBatchV1`의 validated host mirror를 admission evidence로만
사용한다. CSR monotonicity·nonempty sequence range·canonical valid-token count·physical ID uniqueness/range·row
slot/position·duplicate logical row address 검증 뒤 host slices는 graph owner에 보관하지 않으며, capture 뒤
device metadata bytes가 host mirror와 계속 일치한다고 주장하지 않는다.

enqueue는 allocation, synchronize, node update, H2D/D2H 또는 host report 없이 같은 capture stream에
fixed-address ragged paged-KV write kernel을 정확히 한 번 기록한다. native capture/graph/exec과 safe owner는
stream 및 아홉 allocation의 exclusive-use lease를 capture → graph → exec → launch completion → close까지
유지한다. raw device metadata가 bounds-invalid row를 만들면 kernel의 existing bounds guard가 해당 row의 K/V
pool write만 no-op으로 남겨 eager와 같은 의미를 보존한다. enqueue 또는 context restoration이 불명확해지면 re-enqueue,
end/instantiate는 CUDA 호출 전에 거부하고 one-shot abort만 허용한다.

새 additive capability/operation은 `RaggedPagedKvCacheWriteBf16 = 12`이다. capability 12의 `Supported`는
contiguous K/V write, generic scatter, RoPE, H2D 또는 다른 C05 operation을 조합해 대체할 수 없다. 이 slice는
metadata upload/update, attention read, projection, scheduler commit, C06 dispatch/registry, C07 executor
wiring/identity/metrics, node update, fresh allocations, sampling 또는 performance promotion을 만들지 않는다.
후속 C07 evidence adapter만 이 정확한 operation을 `KvWrite` slot에 매핑할 수 있고, inventory aggregate는
나머지 operation이 명시적으로 review되기 전까지 `Unknown`이다.

GPU acceptance criteria는 multi-sequence·page-16 boundary·shuffled physical block ID valid host mirror에서 eager
pool BF16 bytes와 exact parity, source/metadata bytes 불변, 최소 64회 sequential replay, second-enqueue rejection,
foreign context·nine-way alias·geometry/capacity·busy·host-mirror preflight rejection 뒤 untouched-resource
recovery, abort/explicit close 뒤 allocation statistics 0을 확인한다. private CUDA test는 raw device metadata의
bounds-invalid row가 eager와 같이 그 row의 key/value pool write만 no-op으로 남기는지를 확인한다. 이 결과는 one-node
`KvWrite` lifecycle/parity의 근거일 뿐 full decode graph 또는 성능 향상 근거는 아니다.

**완료 검증 (2026-09-02):** 원격 RTX 4090/CUDA 12.8에서 native ABI link, CUDA feature library(53 passed, 3 ignored),
CPU graph source-contract 28개, 기존 ignored graph GPU suite 40개, 그리고 C05-18 전용 GPU 회귀 3개를 통과했다.
정상 fixture는 두 sequence, page-16 boundary, shuffled physical block ID에서 eager와 key/value pool BF16 bytes가
정확히 일치했고, source와 다섯 device-metadata allocation이 그대로인 채 64회 sequential replay하며 second enqueue를
거부했다. 별도 raw test는 host admission witness는 유효하게 둔 채 device row slot만 sequence-count 밖으로 바꾸어,
그 행의 K/V pool write가 eager와 같은 bounds-invalid no-op임을 확인했다. short pool preflight 실패는 모든 untouched
resource를 되돌렸고 begin 뒤 abort도 bundle close 및 allocation statistics 0으로 회복했다. 이는 one-node `KvWrite`
lifecycle/parity의 실제 근거일 뿐 C07 executor integration, full decode graph 또는 end-to-end 성능 향상을 입증하지
않는다.

## 2. 범위

### 포함

- stream capture begin/end/abort
- captured graph instantiate
- 사전 할당된 f32 fill node의 graph executable launch/completion
- graph/graph-exec destroy
- resource retention과 close ordering
- capture/enqueue/end/instantiate/launch/completion/close 오류 분리
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
typedef struct RileyCudaGraphCapture RileyCudaGraphCapture;
typedef struct RileyCudaGraph RileyCudaGraph;
typedef struct RileyCudaGraphExec RileyCudaGraphExec;
typedef struct RileyCudaGraphLaunch RileyCudaGraphLaunch;

RileyCudaStatus riley_cuda_graph_capture_begin_fill_f32(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* buffer,
    uint64_t element_count,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);

RileyCudaStatus riley_cuda_graph_capture_enqueue_fill_f32(
    RileyCudaGraphCapture* capture,
    float value,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);

RileyCudaStatus riley_cuda_graph_capture_end(
    RileyCudaGraphCapture** capture,
    RileyCudaGraph** out_graph,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);

RileyCudaStatus riley_cuda_graph_instantiate(
    RileyCudaGraph** graph,
    RileyCudaGraphExec** out_exec,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);

RileyCudaStatus riley_cuda_graph_exec_launch(
    RileyCudaGraphExec* exec,
    RileyCudaStream* stream,
    RileyCudaGraphLaunch** out_launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);

RileyCudaStatus riley_cuda_graph_launch_complete(
    RileyCudaGraphLaunch** launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);
RileyCudaStatus riley_cuda_graph_close(
    RileyCudaGraph** graph,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);
RileyCudaStatus riley_cuda_graph_exec_close(
    RileyCudaGraphExec** exec,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error);
```

C05-4의 generic `riley_cuda_graph_capture_begin`/`abort`는 recovery-only path로 그대로 유지한다.
C05-5는 arbitrary node update, memcpy 또는 raw pointer API를 열지 않는다.

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

C05-5 시점에는 다음 operation만 graph capture admission 대상으로 선언한다.

- one caller-owned, preallocated device buffer에 대한 fixed-shape f32 fill kernel

H2D/D2H memcpy, 다른 custom kernel, cuBLASLt matmul과 event dependency는 아직 admission 대상이 아니다.

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
- 마지막 fill의 D2H bit-exact parity
- native exact-stream launch rejection과 safe Rust lifetime lease
- launch completion 전 resource close 거부
- fill capture abort 후 stream/buffer 재사용
- explicit close 후 live native/Rust allocation 0

### Sanitizer

- memcheck
- racecheck 가능 범위
- repeated create/replay/close leak smoke

## 11. Performance scope

이 slice는 LLM latency나 graph launch overhead 개선을 주장하지 않는다. 1,000회 replay는 output,
completion, resource-close ordering regression만 검증한다. 별도 microbenchmark와 C07의 promotion 근거는
capture-safe operation set이 넓어진 뒤 기록한다.

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
