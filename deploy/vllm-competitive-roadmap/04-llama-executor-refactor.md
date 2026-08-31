# C04 — Llama Batch Executor 동작 보존 분리

**상태:** In progress — C04-1은 executor shape 관측/히트 value type을 별도 metrics module로
분리했고, C04-2는 error/resource/result vocabulary를 executor facade로 이동했으며, C04-3은
shape policy·bucket·history를, C04-4는 raw batch-input buffer의 cold allocation/cleanup을,
C04-5는 packed metadata의 checked layout descriptor를, C04-6은 seven-source host slab packer를
각각 분리했고, C04-7은 borrowed CUDA metadata view binding을, C04-8은 anchored GEMM
shape-variant의 cold inventory preparation을 분리했다. CUDA owner, KV, buffer orchestration,
pinned-memory write/metadata transport, dispatch, output, public API와 production default는
유지한다.
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
  owner.rs              # weights/KV/stream/workspace lifetime
  buffers.rs            # cold-reserved tensor/device/host buffers
  shape.rs              # active-row bucket and shape variant
  gemm_plan.rs           # anchored plan inventory
  metadata.rs            # synchronous/packed transport
  dispatch.rs            # exact backend selection
  output.rs              # logits/token status and canonical output map
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
