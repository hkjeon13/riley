# C04 — Llama Batch Executor 동작 보존 분리

**상태:** In progress — C04-1은 executor shape 관측/히트 value type을 별도 metrics module로
분리했고, C04-2는 error/resource/result vocabulary를 executor facade로 이동했으며, C04-3은
shape policy·bucket·history를, C04-4는 raw batch-input buffer의 cold allocation/cleanup을,
C04-5는 packed metadata의 checked layout descriptor를, C04-6은 seven-source host slab packer를
각각 분리했고, C04-7은 borrowed CUDA metadata view binding을, C04-8은 anchored GEMM
shape-variant의 cold inventory preparation을, C04-9는 greedy output record의 host validation과
canonical token map을, C04-10은 typed error poison routing과 command-submission disposition을
분리했고, C04-11은 prepared dense-row variant selection을 scalar shape helper로, C04-12는
gathered logits의 checked BF16 span byte length를 output helper로, C04-13은 packed batch의
host preflight validation을 metadata helper로, C04-14는 cleanup 중 첫 CUDA 오류 보존을 error
facade로, C04-15는 typed CUDA error construction을 error facade로, C04-16은 greedy output
record의 checked byte length를 output helper로, C04-17은 absolute RoPE host-table byte builder를
rope helper로, C04-18은 borrowed output primitive dispatch를 dispatch helper로, C04-19는 absolute
RoPE position-count scalar arithmetic을 rope helper로, C04-20은 borrowed iteration command-batch
completion guard를 dispatch helper로, C04-21은 checked zeroed host-byte allocator를 host helper로
C04-22는 typed checked byte-length arithmetic을 error facade로, C04-23은 sequence-block-offset
count를 metadata helper로, C04-24는 typed `usize`-to-`u64` conversion을 error facade로 분리했다.
C04-25는 cold output capacity sizing을 output helper로 연결했고, C04-26은 typed packed-region
validation을 metadata helper로 공유했으며, C04-27은 output/RoPE scalar conversion을 error facade로
일관화했고, C04-28은 packed CUDA slab capacity 검증을 metadata helper로 공유했으며, C04-29는
absolute RoPE cold table shape preflight를 rope helper로 공유했다. C04-30은 host-only prepared
executor configuration을 config helper로 분리했고, C04-31은 cold allocation-accounting을
allocation helper로 분리한다. CUDA owner, KV,
buffer orchestration, pinned-memory
write/metadata transport, dispatch, output public API와 production default는 유지한다.
**의미 등급:** `reference`  
**한 가지 목적:** CUDA Graph와 fusion을 안전하게 추가할 수 있도록 거대한 Llama executor의 ownership·shape·metadata·output 경계를 모듈로 분리한다.

[이전: C03](03-scheduler-output-routing-fuzz.md) | [목차](README.md) | [다음: C05](05-cuda-graph-ownership-abi.md)

## 1. 배경

현재 `crates/riley-runtime/src/llama/batch_executor.rs`는 weight/KV ownership, shape bucket, GEMM plan, metadata transport, execution, output download, metric을 한 파일에서 다룬다. 여기에 graph capture와 여러 fused path를 직접 추가하면 lifetime 검토, rollback, test 영향 범위가 지나치게 커진다.

이 PR은 성능 개선이나 production default 변경을 하지 않는다. public API, C ABI, CLI, output, allocation behavior를 유지한 채 내부 구조만 분리한다.

## 2. 목표 구조

```text
crates/riley-runtime/src/llama/executor/
  mod.rs                # public composition and stable facade
  config.rs             # host-only prepared executor configuration and validation
  allocation.rs         # cold scalar allocation-accounting/report
  owner.rs              # weights/KV/stream/workspace lifetime
  buffers.rs            # cold-reserved tensor/device/host buffers
  shape.rs              # active-row bucket and shape variant
  gemm_plan.rs           # anchored plan inventory
  host.rs                # checked zeroed host bytes
  metadata.rs            # synchronous/packed transport
  dispatch.rs            # exact backend, output primitives, and command completion guard
  output.rs              # logits/token status and canonical output map
  rope.rs                # cold host-side RoPE table bytes and scalar shape
  metrics.rs             # allocation-free counters/snapshots
  poison.rs              # post-dispatch failure state
```

향후 C05/C06용 예약 module은 interface만 둘 수 있다.

```text
  graph.rs               # feature-off placeholder/interface only
```

실제 CUDA Graph 구현은 C05 이후에만 추가한다.

## 3. Freeze할 외부 계약

- `PreparedLlamaBatchExecutor` 생성/실행/close API
- scheduler runtime adapter가 사용하는 trait
- CUDA C ABI symbol과 layout
- server CLI 및 default
- model load와 weight hash validation
- exact fallback 조합
- metric name/meaning
- error category와 poison semantics
- hot path allocation count

공개 타입 rename이 필요하면 type alias와 deprecation 없이 내부 `pub(crate)` 경계에서 해결한다.

## 4. Ownership 원칙

### 단일 실체 owner

다음은 executor owner 하나만 소유한다.

- uploaded weights
- paged KV arena
- RoPE table
- maximum-capacity forward/output workspace
- pinned/device metadata slab
- CUDA stream/context lease

shape variant는 descriptor와 plan만 소유하며 weight/KV/buffer를 복제하지 않는다.

### Drop에 의존하지 않는 close

기존 explicit close 결과와 오류 전달을 유지한다. module 분리 후에도 `Drop`은 best-effort 보조일 뿐 qualification의 정상 회수 경계가 아니다.

### Poison state

post-dispatch mutation이 불명확한 오류는 executor 전체 또는 명시된 sub-owner를 poison한다. module 간 오류 변환에서 CUDA stage/status를 잃지 않는다.

## 5. 분리 순서

1. 기존 behavior를 characterizing test로 고정한다.
2. metric과 작은 value type부터 `metrics.rs`, `shape.rs`로 이동한다.
3. buffer descriptor와 cold allocation을 `buffers.rs`로 이동한다.
4. metadata packing/transport를 `metadata.rs`로 이동한다.
5. GEMM shape plan을 `gemm_plan.rs`로 이동한다.
6. output download/status/canonical map을 `output.rs`로 이동한다.
7. execute orchestration을 `dispatch.rs`로 축소한다.
8. facade `mod.rs`가 기존 API를 동일하게 노출한다.
9. 원래 파일은 제거하거나 얇은 compatibility module로 끝낸다.

각 단계는 컴파일 가능한 commit으로 유지하되 최종 merge는 하나의 behavior-preserving PR이다.

### C04-1 — shape metrics value-type extraction

첫 source slice는 `LlamaBatchShapeObservation`과 `LlamaBatchShapeBucketHit`의 derive, field
order, getter 및 `riley_runtime::llama::*` public reexport를 보존한 채
`llama/executor/metrics.rs`로 이동한다. batch owner의 history는 같은 nominal type을 계속 쓰되,
CUDA buffer/weight/KV/stream/dispatch/close/poison/output code는 이동하거나 변경하지 않는다.
CPU source boundary test는 이 value-only module이 scheduler/server와 runtime resource owner를
import하지 않음을 고정한다. 이는 GPU parity나 performance non-regression을 대신하지 않으며,
그 검증은 C02-qualified candidate가 준비된 뒤 후속 C04 slice에서 수행한다.

### C04-2 — error/resource facade extraction

`LlamaBatchExecutorResource`, `LlamaBatchExecutorError`, result alias와 `Display`/`Error`/`From`
계약은 `llama/executor/error.rs`에 둔다. 기존 `batch_executor` facade가 같은 nominal type을
`pub use`하여 `riley_runtime::llama::*` public path와 error text/category를 보존한다. 이 error
vocabulary는 batch metadata, forward failure, CUDA site, paged-KV error를 기술할 수 있지만
prepared owner·buffer·weight·KV·stream을 소유하지 않는다. C04-3의 `shape.rs`는 이 facade만
참조하여 이전 batch-owner module로 역의존하지 않는다.

### C04-3 — shape policy and history extraction

`LlamaBatchShapePolicy`, fixed-capacity bucket validation, prepared-bucket selection과 hit/history
accounting은 `llama/executor/shape.rs`에 둔다. 기존 `batch_executor` facade가 policy와 maximum
bucket constant를 같은 nominal type으로 `pub use`하여 public path를 보존한다. shape history는
prepared owner type을 받지 않고 dense-row predicate만 받아 unsupported smaller plan을 제거하되
maximum rollback bucket을 항상 유지한다. 따라서 shape module은 error/metrics vocabulary만
참조하고 CUDA context·buffer·stream, model/forward/GEMM/KV/plan owner, scheduler/server를 알지
않는다.

### C04-4 — raw batch-input buffer extraction

Per-operation metadata buffer, packed metadata slab, their host workspace와 cold allocation/cleanup
helper는 `llama/executor/buffers.rs`에 둔다. transport policy match와 allocation-report aggregation은
batch owner에 남기므로 새 module은 `BatchMetadataTransport`, model/forward/GEMM/KV/RoPE/output,
stream/dispatch를 참조하지 않는다. owner close는 기존 순서대로 모든 resource close를 시도하고,
input helper에서 받은 첫 cleanup error만 기존 전체 close result에 병합한다. public API와
allocation count/byte contract는 바꾸지 않는다.

### C04-5 — checked packed-metadata layout extraction

`ByteRegion`, `PackedIterationLayout`과 alignment/region 계산은
`llama/executor/metadata.rs`에 둔다. 이 module은 packed input slab의 offset, byte length,
cold capacity와 overflow/invalid-capacity error만 계산하며 device allocation, pinned-memory
write, H2D copy, device view binding, command-batch lifecycle이나 `BatchMetadataTransport` policy를
소유하지 않는다. batch owner는 기존 pack/transport/dispatch 순서와 public nominal type을 그대로
유지한다. 따라서 이번 source-only slice는 packed layout의 alignment·capacity semantics를
변경하지 않으며 GPU parity와 performance non-regression을 주장하지 않는다.

### C04-6 — pure packed-metadata host-packing extraction

seven source array를 preallocated byte slab에 native-endian으로 encode하는
`pack_iteration_input`과 region/bounds helper는 `llama/executor/metadata.rs`에 둔다. capacity와
active-row preflight는 zero-fill 이전에 실행하고, 기존 seven-source 순서와 zero-padding을
그대로 유지한다. synchronous path가 공유하는 U16/U32 byte encoder도 같은 pure helper를 쓴다.
이 module은 allocation, pinned-memory write, H2D copy, device span/view, stream,
`BatchMetadataTransport` policy와 command-batch lifecycle을 소유하지 않는다. 따라서 이 역시
GPU parity나 performance non-regression을 주장하지 않는 source-only slice다.

### C04-7 — borrowed CUDA metadata-view extraction

separate cold buffer와 packed slab을 `PackedBatchV1` 및 optional token/output span으로 bind하는
descriptor는 `llama/executor/device_views.rs`에 둔다. 이 component는 preallocated
`CudaDeviceBuffer`를 빌려 range/dtype/error mapping만 수행하며 allocation, pinned-memory,
upload/copy, stream/command batch, transport policy, forward/KV/model ownership을 갖지 않는다.
batch owner는 host descriptor 생성, H2D enqueue, preflight/dispatch ordering과 completion/poison
semantics를 그대로 보유한다. 이 source-only slice도 GPU parity나 performance non-regression을
주장하지 않는다.

### C04-8 — anchored GEMM shape-variant inventory extraction

`llama/executor/gemm_plan.rs`는 shared maximum forward owner 아래의 optional exact dense-row
plan과 anchored GEMM handle inventory를 cold prepare한다. FixedMaximum은 bucket validation이나
host reservation 전에 빈 inventory를 유지하고, active-row mode는 configured maximum 아래의
buckets만 exact anchor로 준비한다. anchored CUDA GEMM plan이 정확히 NotSupported일 때만
해당 smaller variant를 생략해 다음 prepared bucket 또는 maximum owner로 fallback하며, 다른
failure는 이미 만든 variant를 close한 뒤 원 error를 보존한다.

batch owner는 variant field, history, workspace maximum fold와 한 번의 shared-workspace
reconciliation, runtime selection/dispatch, poison 및 close precedence를 계속 소유한다. 새
module은 weight/KV/buffer/stream/metadata transport/output 또는 forward workspace owner를
복제하지 않는다. 이 source-only slice는 GPU parity나 performance non-regression을 주장하지
않는다.

### C04-9 — greedy output record validation extraction

`llama/executor/output.rs`는 device가 생성한 fixed-width `{token_id,status}` greedy record를
host에서 검증한 뒤 canonical dense output-slot token map으로 쓰는 순수 helper를 소유한다. 모든
record를 먼저 검증하므로 invalid/non-finite 결과는 caller destination의 partial publication을
막는다. non-finite logits는 기존의 non-poisoning typed error를 유지하며, invalid native
record가 executor를 poison하는 판단은 batch owner에 그대로 남긴다.

CUDA download, preallocated host/device result storage, output-ready/mode lifecycle, allocation
report, dispatch와 public download API는 batch owner가 계속 소유한다. 이 source-only slice는
GPU parity나 performance non-regression을 주장하지 않는다.

### C04-10 — typed poison routing extraction

`llama/executor/poison.rs`는 command batch가 시작되어 partial device mutation이 가능해진
disposition과 typed batch error의 established poison mapping을 보유한다. CUDA validation failure의
non-poisoning rule, nested forward failure routing, invalid configuration/overflow의 fail-closed
rule은 그대로 유지한다. forward GEMM poison 상태는 owner가 error routing 뒤 지연 callback으로
제공하므로 새 module은 forward owner나 CUDA resource를 소유하거나 보관하지 않는다.

iteration-batch completion에서 submission-started 여부를 보고 executor 전체를 poison하는
outer decision, shape variant GEMM poison fold, KV/buffer/stream lifecycle과 explicit close는 batch
owner가 계속 소유한다. 이 source-only slice는 GPU parity나 performance non-regression을
주장하지 않는다.

### C04-11 — prepared dense-row selection extraction

`llama/executor/shape.rs`는 successfully prepared optional dense-row variant의 scalar row count를
받아 active batch에 맞는 가장 작은 available bucket을 선택하고, 없으면 exact maximum owner로
fallback한다. batch owner는 기존 configuration/policy validation을 selection 이전에 수행하므로
empty/over-capacity error category와 public `select_dense_rows` contract는 그대로 유지한다.

prepared variant의 CUDA plan/GEMM handle, mutable dispatch lookup, shape-history update와 output/KV/
buffer lifecycle은 batch owner가 계속 소유한다. 이 source-only slice는 GPU parity나 performance
non-regression을 주장하지 않는다.

### C04-12 — gathered logits byte-length extraction

`llama/executor/output.rs`는 dense `[O,V]` BF16 gathered-logits span의 checked `u64` byte length와
overflow error mapping을 계산한다. batch owner는 기존의 row-gather/argmax CUDA primitive binding에
그 scalar byte length만 사용하므로 output buffer allocation, download API, public `usize` query와
output lifecycle은 바꾸지 않는다.

이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-13 — packed batch host-preflight extraction

`llama/executor/metadata.rs`는 packed schema, cold metadata capacity, token vocabulary와 row
position을 pure host preflight로 검증한다. owner는 model maximum position과 profile-derived
position cap만 scalar로 주입하며, validation 순서와 typed error category를 그대로 유지한다.
Fixed37의 `8192` logical-token cap도 metadata validation 안에서 model position bound보다 먼저
평가한다.

metadata packing/allocation/H2D transport, prepared CUDA owner, shape selection, dispatch, poison,
output lifecycle와 public API는 batch owner가 계속 소유한다. 이 source-only slice는 GPU parity나
performance non-regression을 주장하지 않는다.

### C04-14 — first cleanup-error routing extraction

`llama/executor/error.rs`는 CUDA resource close가 실패할 때 first cleanup error만 보존하는
`record_close` helper를 소유한다. batch owner와 metadata-input buffer helper는 기존 close 순서를
그대로 유지하고, 이후 close도 계속 시도하며, cleanup error category와 첫 오류 우선순위도
변경하지 않는다.

forward/KV/output/stream의 owner lifecycle과 close 순서 결정은 각 owner에 남는다. 이
source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-15 — typed CUDA error construction extraction

`llama/executor/error.rs`는 `ExecutionSite`와 `CudaError`를 stable typed
`LlamaBatchExecutorError::Cuda`로 묶는 `cuda_error` helper를 소유한다. batch owner,
metadata buffer allocation, borrowed metadata-view binding은 이 helper를 기존 local mapper
name으로 import하여 모든 `map_err` call site와 poison/dispatch ordering을 그대로 유지한다.

CUDA primitive invocation, buffer/stream/KV ownership과 poison decision은 각 existing owner에
남는다. 이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-16 — greedy output record byte-length extraction

`llama/executor/output.rs`는 fixed-width `{token_id,status}` greedy record의 checked byte
length와 `GreedyResults` overflow error mapping을 소유한다. batch owner는 cold prepared bound
검사, device buffer allocation, argmax output span binding, host download, output-ready lifecycle과
invalid native result의 poison 판단을 그대로 보유한다.

이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-17 — absolute RoPE host-table builder extraction

`llama/executor/rope.rs`는 selected cold path가 쓰는 absolute position-major RoPE angle bytes와
CPU cosine/sine table bytes를 native-endian으로 materialize한다. element/offset overflow는 기존
`RopeCos` category를, temporary host-byte sizing/allocation failure는 기존 `HostWorkspace` category를
그대로 유지한다.

batch owner는 RoPE profile 선택, device allocation, upload, CUDA table launch, stream/close와
allocation-report ownership을 계속 보유한다. forward/decode의 유사 builder는 다른 error/resource
contract이므로 이번 slice에서 공유하거나 변경하지 않는다. 이 source-only slice는 GPU parity나
performance non-regression을 주장하지 않는다.

### C04-18 — borrowed output primitive dispatch extraction

`llama/executor/dispatch.rs`는 fixed graph가 logits를 만든 뒤, pre-bound output index/logit/result
buffers를 row-gather와 optional deterministic greedy argmax CUDA primitive에 bind한다. non-empty
output의 index-buffer 확인, gathered-logit 확인, row gather, 그 뒤에만 greedy result 확인/argmax를
수행하는 기존 error order와 `Forward`/typed CUDA error mapping을 보존한다.

batch owner는 metadata transport, shape/plan 선택, fixed graph 실행, output-ready state, poison,
allocation 및 close ownership을 계속 보유한다. 이 source-only slice는 GPU parity나 performance
non-regression을 주장하지 않는다.

### C04-19 — absolute RoPE position-count scalar extraction

`llama/executor/rope.rs`는 cold-prepared table byte length와 head dimension에서 absolute RoPE
position count를 계산한다. 기존 half-width의 `u64` conversion, F32 row-byte checked multiply,
그리고 table byte length의 floor division을 그대로 유지해 `RopeCos` overflow category와 table
shape semantics를 보존한다.

batch owner는 CUDA RoPE buffer의 ownership과 `.byte_len()` 조회, table allocation/upload/kernel
launch, profile 선택 및 lifecycle을 계속 보유한다. 이 source-only slice는 GPU parity나 performance
non-regression을 주장하지 않는다.

### C04-20 — borrowed iteration command-batch completion guard

`llama/executor/dispatch.rs`는 iteration-batch mode의 shared command-batch lifecycle을 보유하지
않는 borrowed guard로 수행한다. native begin 성공 뒤에만 mutation-unknown disposition을 기록하고,
non-replaceable command proxy로 body를 한 번 실행한 뒤 body 성공/실패와 무관하게 finish를 호출한다.
finish error가 body error보다 우선하는 기존 completion contract를 그대로 유지한다.

batch owner는 synchronous/packed metadata preflight, packed H2D와 view rebind, fixed graph body,
output state 및 실제 poison decision을 계속 보유한다. 이 source-only slice는 GPU parity나
performance non-regression을 주장하지 않는다.

### C04-21 — checked zeroed host-byte allocator extraction

`llama/executor/host.rs`는 cold host workspace가 쓰는 checked element-byte multiplication, exact
reserve, zero fill 및 boxed-byte conversion을 한 곳에 둔다. overflow와 reserve failure는 기존처럼
각각 `HostWorkspace` arithmetic/host-allocation error로 매핑한다.

batch owner, metadata buffer helper, RoPE builder는 각자의 semantic preflight와 resource ownership을
계속 보유하고 shared allocator에는 element count와 byte width만 전달한다. 이 source-only slice는 GPU
parity나 performance non-regression을 주장하지 않는다.

### C04-22 — typed checked byte-length facade extraction

`llama/executor/error.rs`는 executor resource를 명시한 checked element-byte multiplication과
`ArithmeticOverflow` mapping을 한 곳에 둔다. batch owner, raw metadata buffer helper, packed-metadata
layout/encoder는 기존 resource와 호출 순서를 유지한 채 shared scalar helper만 사용한다.

allocation, upload, packing cursor advance, CUDA span/primitive invocation과 error precedence는 각
caller에 그대로 남는다. 이 source-only slice는 GPU parity나 performance non-regression을 주장하지
않는다.

### C04-23 — sequence-block-offset count extraction

`llama/executor/metadata.rs`는 bounded batch의 CSR block-row offset count(`max_rows + 1`)와
`SequenceBlockOffsets` overflow mapping을 보유한다. batch allocation report, synchronous host/device
metadata allocation, packed-layout capacity가 같은 scalar helper를 사용한다.

각 caller의 allocation/report/layout 순서와 byte sizing, CUDA resource ownership은 변경하지 않는다.
이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-24 — typed `usize`-to-`u64` conversion extraction

`llama/executor/error.rs`는 CUDA ABI가 받는 native scalar의 checked `usize`-to-`u64` conversion과
resource-preserving `ArithmeticOverflow` mapping을 보유한다. batch owner, raw metadata buffer,
borrowed metadata views, output dispatch가 기존 call-site 순서와 resource를 유지해 이를 공유한다.

CUDA span/primitive invocation, allocation, dispatch/poison ordering과 owner lifecycle은 각 caller에
그대로 남는다. 이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-25 — cold output-capacity sizing reuse

`llama/executor/output.rs`는 dense gathered-logits와 greedy result의 canonical checked capacity
calculation을 보유한다. greedy CUDA ABI capacity helper는 native-width conversion을 multiplication보다
먼저 수행해 기존 `GreedyResults` overflow precedence를 유지한다.

batch owner는 cold output-buffer allocation, allocation report, dispatch, download/poison lifecycle을
계속 소유하며 shared sizing helper만 호출한다. 이 source-only slice는 GPU parity나 performance
non-regression을 주장하지 않는다.

### C04-26 — packed-region validation extraction

`llama/executor/metadata.rs`는 typed U32/U16 host encoder가 공통으로 수행하는 source byte-length
check, layout length mismatch, destination range validation을 shared helper로 보유한다. typed encoder는
각자의 native-endian encoding과 기존 mismatch reason을 계속 명시한다.

metadata packing order, destination ownership, CUDA upload/dispatch와 allocation lifecycle은 변경하지
않는다. 이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-27 — typed CUDA-scalar conversion reuse

`llama/executor/error.rs`의 typed `usize`-to-`u64` conversion은 output capacity와 RoPE table-row
scalar에도 같은 resource-preserving overflow mapping을 제공한다. 각 output/RoPE helper의 conversion과
checked multiplication 순서는 유지한다.

CUDA 호출, buffer allocation, host/device upload, dispatch, owner lifecycle은 변경하지 않는다. 이
source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-28 — packed CUDA-slab capacity validation reuse

`PackedIterationLayout`은 CUDA ABI의 `u64` slab byte capacity를 native capacity로 checked conversion한
뒤 layout capacity를 검증한다. packed host write 전의 batch owner와 borrowed device span bind 전의
device view가 같은 pure helper를 사용하며, `PackedIterationInput` overflow mapping을 유지한다.

host write/H2D copy, device span 생성, CUDA invocation, allocation과 owner lifecycle은 각 caller에
그대로 남는다. 이 source-only slice는 GPU parity나 performance non-regression을 주장하지 않는다.

### C04-29 — absolute RoPE cold table-shape preflight reuse

`llama/executor/rope.rs`는 absolute RoPE table의 half-width와 checked element count를 두 cold
builder가 공유하도록 계산한다. allocation 전의 `RopeCos` overflow mapping, odd head dimension의
floor semantics, zero-element table은 유지한다.

F32 byte sizing, angles의 단일 allocation과 CPU cosine/sine의 순차 allocation, native-byte write loop,
CUDA upload/launch와 owner lifecycle은 각 caller에 그대로 남는다. 이 source-only slice는 GPU parity나
performance non-regression을 주장하지 않는다.

### C04-30 — host-only prepared-executor configuration extraction

selection enum/ID helper, `PreparedLlamaBatchExecutorConfig` builder/validation 및
`normalize_prepared_config`는 `llama/executor/config.rs`로 이동한다. `BatchOutputMode`, mutable
shape-history와 `shape_history_for_config`는 resource-owning batch owner에 그대로 남는다. config
component는 scalar configuration과 validation만 다루며 CUDA context/buffer/stream, model weight, KV
arena, metadata transport storage, allocation, dispatch, output 또는 poison owner를 소유하거나 호출하지
않는다.

기존 `riley_runtime::llama::*` public path와 nominal type은 `batch_executor` facade의 reexport로
그대로 유지한다. batch owner는 normalized config의 기존 field/validation 순서와 cold prepare,
resource ownership, cleanup/poison/dispatch 순서를 그대로 유지한다. source-boundary test는 config
component가 scheduler/server와 runtime resource owner를 import하지 않음을 고정한다. 이
behavior-preserving CPU-only slice는 CUDA parity, allocation evidence 또는 performance
non-regression을 주장하지 않는다.

### C04-31 — cold allocation-accounting extraction

`PreparedLlamaBatchAllocationReport`와 `build_batch_allocation_report`는
`llama/executor/allocation.rs`로 이동한다. helper는 forward allocation report, immutable batch
bounds/transport, KV byte scalar 및 RoPE/output capacity scalar만 받는다. CUDA context/buffer/stream,
KV layout, host-workspace owner, model/forward executor owner를 받거나 소유하지 않는다.

batch owner는 모든 cold resource allocation이 성공한 기존 지점에서 helper를 호출하고, helper는 같은
bounds와 transport로 host/pinned capacity를 다시 계산한다. 따라서 allocation/cleanup/poison 순서와
resource ownership은 batch owner에 그대로 남으며, 기존 `riley_runtime::llama::*` public path와
nominal type은 `batch_executor` facade reexport로 유지한다. source-boundary 및 nominal-type test만
추가하는 CPU-only slice이며, CUDA allocation parity, GPU evidence 또는 performance improvement를
주장하지 않는다.

## 6. Allocation 검증

다음 경로의 allocation snapshot을 refactor 전/후 비교한다.

- cold prepare
- c1 decode 100 iterations
- c8 mixed prefill/decode
- packed metadata fallback
- GPU greedy success/failure
- cancellation/commit failure
- explicit close

hot iteration의 Rust heap, pinned/device allocation delta는 0이어야 한다. cold allocation 수가 달라지면 byte/count와 이유를 문서화하고 VRAM/KV capacity가 변하지 않아야 한다.

## 7. Correctness 검증

- canonical 31-case
- SmolLM2/Qwen 32-step greedy
- fixed-max vs active-row
- metadata synchronous vs packed
- CPU logits vs GPU token
- mixed output slot mapping
- KV `15→16→17`
- invalid metadata와 unsupported shape fallback
- poison/close lifecycle

가능한 경로는 raw BF16 bytes와 token hash를 refactor 전 accepted executable과 비교한다.

## 8. Performance non-regression

이 PR은 개선을 주장하지 않는다. 같은 GPU에서 before/after를 5쌍 실행한다.

```text
TTFT p95 ratio <= 1.03
TPOT p95 ratio <= 1.03
throughput ratio >= 0.97
peak VRAM ratio <= 1.00
kernel launch inventory unchanged
H2D/D2H bytes unchanged
```

3%를 넘는 변화가 있으면 원인을 찾기 전 병합하지 않는다. compiler layout 차이로 미세 변화가 있어도 threshold를 결과에 맞춰 늘리지 않는다.

## 9. 예상 파일 변경

```text
crates/riley-runtime/src/llama/mod.rs
crates/riley-runtime/src/llama/batch_executor.rs
crates/riley-runtime/src/llama/executor/*.rs
crates/riley-runtime/tests/llama_batch_gpu.rs
crates/riley-runtime/tests/architecture_boundary.rs
```

`riley-cuda`, scheduler, server 변경은 import/path 적응을 제외하고 금지한다.

## 10. Architecture boundary test

- executor module이 scheduler/server를 import하지 않음
- graph placeholder가 CUDA native symbol을 호출하지 않음
- shape plan이 weight/KV storage를 소유하지 않음
- output module이 request scheduling policy를 알지 않음
- metadata module이 model architecture decision을 알지 않음

dependency 방향을 source scan과 compile test로 고정한다.

## 11. Review 단위

리뷰 시 rename/move와 semantic change를 분리해 볼 수 있도록 다음을 제공한다.

- old→new symbol map
- 각 module 책임 표
- public API diff가 비어 있음을 확인한 report
- before/after call graph
- allocation/benchmark comparison

대규모 format-only 변경을 함께 넣지 않는다.

## 12. 승인 기준

- 모든 기존 CPU/CUDA test 통과
- public Rust API/C ABI/CLI contract 변화 없음
- canonical output/token exact
- hot allocation 0 유지
- KV capacity/peak VRAM 회귀 없음
- performance non-regression gate 통과
- 원래 executor가 facade 또는 제거 가능한 크기로 축소
- C05가 graph owner를 별도 module에 추가할 수 있는 명시적 extension point 확보

## 13. 롤백

module 분리는 한 commit 범위로 함께 revert한다. 부분적으로 old/new module을 섞어 rollback하지 않는다. rollback 후 C03 corpus와 기존 GPU parity를 다시 실행한다.

## 14. 완료 정의

한 파일을 수정하지 않고도 graph ownership, shape dispatch, output fast path를 각각 독립 module에서 구현·리뷰할 수 있고, refactor 전후 observable behavior와 성능이 계약 범위에서 동일할 때 완료다.
