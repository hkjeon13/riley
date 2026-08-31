# C07 — Pure-decode CUDA Graph Buckets

**상태:** In progress — C07-16은 same-layout pinned/device binding의 exact slab H2D enqueue를 caller-owned command batch로 좁힌다.
**의미 등급:** `E0`  
**한 가지 목적:** pure-decode `M={1,2,4,8,16,32}`의 stable-address GPU chain을 capture/replay하여 M2 성능 gate를 판정한다.

[이전: C06](06-graph-signature-dispatcher.md) | [목차](README.md) | [다음: C08](08-executable-pattern-registry.md)

### C07-0 — fixed pure-decode bucket catalog (CPU-only)

초기 catalog는 active row `1,2,4,8,16,32`만 허용하고, `3→4`, `5→8`, `9→16`, `17→32`처럼
가장 작은 상계 bucket으로만 padding한다. zero 또는 `32` 초과 row는 maximum executor shape로
대체하지 않고 bucket 없음으로 남겨 exact eager를 유지한다.

이 slice는 catalog selection이 graph prepared/capture-safe/launch-complete를 뜻하지 않음을 명시한다.
C06 eligibility/registry/signature, executor shape policy, metadata packing, CLI/default, CUDA capture와
performance 판정에는 연결하지 않는다. future owner는 base batch validation 뒤 checked row conversion으로만
catalog를 사용할 수 있다.

### C07-1 — fixed pure-decode metadata layout descriptor (CPU-only)

C07-1은 exact bucket `M∈{1,2,4,8,16,32}`와 cold `block_entry_capacity B`에서 schema v1의
상대 offset, byte size, alignment, 최종 정렬된 총 byte 수만 계산한다. `M`은 catalog의 정확한 원소여야 하며
`B >= M`이어야 한다. `3` 같은 active-row count를 이 descriptor에서 `4`로 올림 처리하지 않는다.

canonical field order는 `header`, token IDs, position IDs, row sequence slots, sequence block offsets,
physical block IDs, valid-token counts, output-token indices, control/status다. `header`와 control/status는
0 byte를 허용하지 않는 opaque region이며 v1에서 모두 4-byte alignment를 요구한다. schema의 capacity,
offset, byte size, alignment, total은 host word size와 무관한 `u64`이고, 각 산술과 정렬은 overflow 시
닫힌 typed error로 실패한다. 이 schema version은 기존 `LLAMA_BATCH_METADATA_V1_VERSION`과 별개다.

이 단계는 relative geometry 계약일 뿐 실제 slab allocation·주소 안정성·byte packing·padding sentinel·동적
batch validation·현재 V1 packed metadata ABI의 adapter·C06 registry lookup·CUDA capture/replay를 수행하지
않는다. 현재 V1의 CSR/KV ownership은 동적이므로, future executor owner가 production-valid padding과
ownership을 별도로 증명하기 전에는 이 layout을 runtime graph input으로 사용할 수 없다.

### C07-2 — canonical fixed-layout digest (CPU-only)

C07-2는 C07-1 layout에서 domain-separated SHA-256 value digest를 만든다. raw Rust struct bytes를 hash하지 않고
little-endian으로 schema version, canonical field count, exact `M`, cold `B`, required base alignment, canonical field
순서의 field tag·offset·byte size·alignment, final total byte를 streaming 순서로 hash한다. 따라서 어느 cold
geometry input이 달라도 identity가 달라진다.

`header` 또는 control/status의 의미가 geometry를 바꾸지 않고 달라져도 schema version을 올려야 한다. 같은
geometry digest를 과거 의미와 재사용하는 것은 허용하지 않는다.

digest에는 allocation address, pointer, payload byte, active-row count, actual block count, padding sentinel, dynamic
batch value가 들어가지 않는다. 이 type은 C07-local value identity일 뿐 C06 `GraphMetadataLayoutSignature` 변환,
registry key, graph prepared/capture/replay 권한이나 current V1 packed metadata ABI adapter를 만들지 않는다. future
cold owner는 production ownership·padding·exact equality를 별도로 검증한 뒤에만 이를 더 큰 graph identity에 넣을 수 있다.

### C07-3 — trailing pure-decode padding plan (CPU-only)

C07-3은 base pure-decode validation 뒤의 active-row count `A`만 받아 C07-0 selector의 exact bucket `M`과
`P = M - A`를 계산한다. supported `A=1..32`에서 active lane은 prefix `[0,A)`, trailing placeholder lane은
`[A,M)`이며, `0` 또는 `32` 초과는 `None`으로 남아 maximum bucket으로 대체하지 않는다.

이 plan은 metadata lane topology일 뿐 request/output/KV block 또는 caller row mapping이 아니다. sentinel/zero-fill,
row transformation, block-table padding, actual metadata packing과 kernel mask는 future owner가 별도로 증명해야 한다.
`A`와 `P`는 dynamic iteration fact이므로 C07-1 cold layout identity나 C07-2 geometry digest에 넣지 않는다.

### C07-4 — exact layout/padding binding (CPU-only)

C07-4는 검증된 cold metadata layout과 C07-3 padding plan을 결합할 때 두 exact bucket `M`이 같아야 한다는
계약을 typed error로 닫는다. mismatch는 다른 bucket 재선택이나 maximum fallback으로 복구하지 않는다. binding은
caller-provided digest를 받지 않고 bound layout에서 C07-2 geometry digest를 직접 계산·보관한다.

이 binding은 bucket equality와 layout-derived identity만 증명한다. metadata content, address stability, row/request/output/KV
mapping, sentinel/packing, allocation, C06 signature/registry, capture/replay 권한을 만들지 않는다. cold owner는 이 binding을
재사용할지와 어느 lifecycle에서 만들지를 별도로 검토해야 하며, 이후 실제 buffer와 kernel contract를 검증해야 한다.

### C07-5 — checked fixed metadata slab writer (CPU-only)

C07-5는 C07-4 binding과 caller-owned fixed-length source slices를 받아 caller-owned byte slab prefix에 canonical little-endian으로
쓴다. input은 header/control의 exact region byte length, M-sized token/position/row-slot/output fields, M+1-sized block-offset field,
B-sized physical-block/valid-token fields를 모두 포함해야 하며 trailing placeholder lane도 caller가 명시적으로 제공한다. 모든
destination/source length와 region addressability를 먼저 검사해 error에는 destination을 바꾸지 않으며, destination은 required slab byte
이상이면 된다. 성공 시 required prefix만 alignment gap을 포함해 zero-fill한 뒤 C07-1 canonical region에 각 field를 기록하고, extra tail은
보존한다. header/control은 여전히 opaque bytes다.

이 writer는 current V1 batch ABI adapter, request/output/KV mapping, padding sentinel/validity, allocation ownership, slab base address
alignment/stability validation, host-to-device copy, C06 registry/signature, CUDA capture/replay를 수행하지 않는다. trailing placeholder
lane의 실제 byte value와 kernel-mask 의미는 caller가 future production contract에서 증명해야 하며, 단순히 이 writer가 성공했다고
graph-safe가 되지 않는다.

### C07-6 — V1 pure-decode eligibility preflight (CPU-only)

C07-6는 이미 base validation을 통과한 `LlamaPackedBatchMetadata`를 read-only로 보고 C07 candidate인지 닫힌 결과로 분류한다.
candidate는 prefill row/token이 없고, decode row count가 전체 row count와 같으며, decode token count와 total input token count가
모두 row count와 같고, output count도 row count와 같아야 한다. 즉 각 row는 exactly one decode input과 one output을 가진다.
row count는 checked `u32` conversion 뒤 C07-3 padding planner에만 전달하며 `A=1..32`의 exact/padded bucket만 candidate가 된다.
`A=0`, `A>32`, host-width conversion failure, prefill 또는 mixed shape는 typed ineligible로 남고 maximum bucket으로 대체하지 않는다.

성공값은 C07-3 `PureDecodeGraphPaddingPlan`뿐이다. 이 preflight는 fixed layout/binding/packer 호출, field materialization,
padding sentinel/header/control 생성, allocation/address ownership, host-to-device copy, C06 registry/signature/dispatch, graph
capture/replay, executor mutation, CLI/default 변경을 수행하지 않는다. 따라서 candidate는 향후 graph-safe metadata 또는 runnable
graph를 뜻하지 않고, 계속 exact eager 상태에서 다음 ownership·field-mapping contract를 검토하기 위한 read-only fact다.

### C07-7 — V1 candidate/cold-layout exact binding (CPU-only)

C07-7은 C07-6의 closed V1 candidate result와 caller-supplied C07-1 cold layout을 값으로만 결합한다.
이미 ineligible인 결과는 layout을 검사하거나 binding을 만들지 않고 같은 typed reason으로 반환한다. eligible result만 C07-4의
exact bucket binder에 전달하므로 `M` 불일치는 기존 `LayoutPaddingBucketMismatch` typed error로 남고, bucket 재선택이나
maximum fallback은 없다. 성공값은 C07-4 binding과 layout-derived C07-2 geometry digest를 보관한다.

이 단계는 V1 raw field/CSR/KV block capacity 검증이나 fixed-slab source materialization, trailing padding sentinel/header/control,
allocation/address ownership, host-to-device copy, C06 signature/registry/dispatch, CUDA capture/replay, executor mutation을 하지 않는다.
따라서 bound result도 자체로 graph-safe metadata나 runnable graph가 아니며, C07-8의 strict no-tail projection 밖의
field/capacity/padding case는 계속 exact eager를 유지한다.

### C07-8 — strict exact-shape V1 native-field projection (CPU-only)

C07-8은 C07-6 preflight와 C07-7 layout binding을 한 호출에서 순서대로 적용한 뒤, `A=M`이고 V1 flattened block entry count
`K`가 cold layout capacity `B`와 정확히 같을 때만 seven native fields를 borrowed view로 반환한다. token IDs, position IDs,
row sequence slots, block CSR offsets, physical block IDs, valid-token counts, output-token indices의 길이는 각각 `M, M, M, M+1,
B, B, M`이어야 한다. eligible batch의 `M` mismatch는 기존 C07-4 `LayoutPaddingBucketMismatch` error로 유지한다.

`A<M`의 row padding 또는 `K!=B`의 block tail은 typed ineligible이며 maximum fallback이나 zero/sentinel 대입으로 복구하지 않는다.
header/control bytes는 V1 native field가 아니므로 이 view에 없고, C07-5 packer 호출이나 fixed slab source creation도 하지 않는다.
따라서 projection은 copy, allocation/address ownership, host-to-device transfer, C06 signature/registry/dispatch, CUDA capture/replay,
executor mutation 권한을 만들지 않는다. future tail materialization은 kernel-mask와 ownership semantics를 별도 증명한 이후에만 가능하며,
그 전까지 exact eager를 유지한다.

### C07-9 — exact opaque-region length binding (CPU-only)

C07-9는 C07-8의 already-projected seven native fields와 caller-owned header/control-status byte slices를 두 독립 borrow lifetime으로
결합한다. header와 control/status는 각각 bound C07-1 layout의 corresponding opaque region byte capacity와 exact equality여야 하며,
header mismatch를 control/status보다 먼저 typed error로 닫는다. C07-8이 이미 A=M, K=B 및 seven native field capacity를 닫았으므로,
이 단계는 remaining two opaque input shapes만 확인한다.

opaque byte value, schema interpretation, sentinel, kernel-mask meaning은 검증하지 않는다. 성공값은 native slices와 opaque byte slices의
borrowed grouping일 뿐 C07-5 field-source creation/packer invocation, fixed slab write, padding-tail materialization, allocation/address
ownership, host-to-device copy, C06 signature/registry/dispatch, CUDA capture/replay, executor mutation을 수행하거나 graph-safe 권한을
부여하지 않는다. 그 전까지 모든 path는 exact eager를 유지한다.

### C07-10 — exact nine-field source composition (CPU-only)

C07-10은 successful C07-9 grouping을 shortest reborrow lifetime으로 C07-5의 canonical nine-field source order에만 재구성한다.
same value는 originating C07 binding도 함께 보관하므로, future caller가 exact fields와 unrelated cold layout binding을 무심코 섞지
않는다. order는 header, token IDs, position IDs, row sequence slots, block CSR offsets, physical block IDs, valid-token counts,
output-token indices, control/status다.

C07-8이 seven native capacity와 no-tail shape를, C07-9가 two opaque region byte length를 이미 닫았으므로 이 bridge는 Result/error,
length 재검사 또는 error translation을 만들지 않는다. C07-5 field-source construction 외에는 packer invocation, fixed slab write/zero-fill,
opaque byte semantics, padding-tail materialization, allocation/address ownership, host-to-device copy, C06 signature/registry/dispatch,
CUDA capture/replay, executor mutation을 수행하거나 graph-safe 권한을 부여하지 않는다. 그 전까지 모든 path는 exact eager를 유지한다.

### C07-11 — exact V1 caller-owned slab write (CPU-only)

C07-11은 C07-8 projection, C07-9 opaque-region binding, C07-10 nine-field composition을 순서대로 적용한 뒤 C07-5 canonical
little-endian writer에 caller-owned destination을 한 번 전달한다. exact success는 Written이고, C07-8 ineligible은 opaque bytes나
destination을 검사하기 전에 unchanged typed reason으로 반환한다. eligible layout mismatch, opaque header/control mismatch, C07-5
destination/source error는 각각 original typed error family로 보존한다. header는 control/status보다 먼저 검사되며 C07-9 opaque error는
뒤의 C07-5 destination error보다 우선한다.

all ineligible/error path는 C07-5 write 전 또는 C07-5의 pre-write validation에서 끝나므로 destination을 변경하지 않는다. success에서는
C07-5가 required slab prefix와 alignment gap만 기록하고 caller tail을 보존한다. 이 wrapper는 slab allocation, base-address alignment or
stability, persistent ownership, opaque-byte semantics, padding-tail materialization, host-to-device copy, C06 signature/registry/dispatch,
CUDA capture/replay, executor mutation을 수행하거나 graph-safe/runnable authority를 부여하지 않는다. 그 전까지 모든 path는 exact eager를
유지한다.

### C07-12 — cold exact V1 aligned host-slab owner (CPU-only)

C07-12는 C07-11의 caller-owned destination을 하나의 cold-owned fixed host allocation으로 좁힌다. layout의 exact `total_bytes`에
`required_base_alignment - 1` byte alignment window를 더해 한 번만 fallible allocation하고, `Box<[u8]>` 자체의 1-byte alignment에
의존하지 않고 safe `align_offset`으로 interior payload base를 선택·검증한다. public payload view는 exact `total_bytes` 길이이고 layout의
base alignment를 만족하며, backing/offset은 private이므로 owner를 move하거나 repeated write해도 owner lifetime 동안 payload allocation
address는 바뀌지 않는다. cold prepare 후 write는 stored layout과 exact source를 C07-11에 한 번만 전달해 기존 Written/Ineligible/error
identity와 failure precedence를 그대로 보존하고 hot path에서 allocation을 하지 않는다.

이 owner는 CPU pageable host storage만 가진다. pinned allocation, raw pointer export, mutable byte view, H2D/device slab, C06
signature/registry/dispatch, padding-tail materialization, opaque-byte semantics, CUDA capture/replay, executor integration은 만들지 않는다.
따라서 aligned stable host address는 준비하지만 graph-safe metadata 또는 runnable graph authority는 아직 부여하지 않으며, 그 전까지
모든 path는 exact eager를 유지한다.

### C07-13 — successful exact V1 host-slab lease (CPU-only)

C07-13은 C07-12의 successful exact write만 same-owner read-only lease로 바꾼다. lease는 방금 기록한 exact payload slice와 그 owner의
cold layout·geometry digest를 한 value로 보관하므로, future caller가 bytes를 다른 cold geometry와 섞지 않는다. lease가 살아 있는 동안
`&mut` owner borrow가 유지되어 Rust가 re-write, owner move, drop을 막고, lease가 끝난 뒤에만 같은 stable payload를 다시 기록할 수 있다.
leased write는 C07-12에 정확히 한 번 위임한다. 따라서 Written만 lease가 되고, Ineligible 및 모든 typed error의 identity·precedence와
no-mutation behavior는 그대로 보존되며 hot path allocation도 없다.

lease는 successful host write의 freshness/ownership fact일 뿐 opaque-byte semantics, pinned host storage, H2D completion, device contents,
C06 signature/registry/dispatch, CUDA capture/replay 또는 graph-safe/runnable authority를 만들지 않는다. 현재 V1 row/block tail·kernel mask
semantics가 닫히기 전까지 모든 path는 exact eager를 유지한다.

### C07-14 — cold exact V1 pinned-host staging lease (CUDA feature; no GPU execution test)

C07-14는 C07-13의 successful exact host-slab lease만 받아 same cold layout의 one-shot pinned-host staging write로 옮긴다. pinned owner는
`layout.total_bytes()`와 정확히 같은 길이의 pinned allocation을 cold prepare에서 한 번만 만들고 layout·geometry digest와 함께 보관한다.
staging 전에는 digest가 아니라 complete layout equality를 확인하므로 서로 다른 cold geometry가 섞이지 않으며, mismatch는 pinned/source digest를
진단값으로 가진 typed error로 끝나고 pinned write를 하지 않는다. matching layout은 host lease의 exact bytes를 offset zero에 동기 write한 뒤에만
same-owner read-only pinned lease를 반환한다. lease가 살아 있는 동안 Rust borrow가 re-stage, owner move, explicit close를 막고, lease가 끝난
뒤에만 같은 pinned allocation을 다시 stage하거나 close할 수 있다.

native pinned write failure에는 lease를 반환하지 않으며 pinned contents의 transactional rollback은 주장하지 않는다. 이 단계는 device slab,
stream/command, host-to-device copy, copy completion, CUDA graph capture/replay, C06 signature/registry/dispatch, executor integration 또는
graph-safe/runnable authority를 만들지 않는다. CPU-only source contract가 allocation/write count와 feature gating을 확인하고 CUDA feature는
compile-only로 확인한다. actual GPU execution, transfer measurement, and graph replay는 이후 단계에 남기며 그 전까지 모든 path는 exact eager를
유지한다.

### C07-15 — cold exact V1 device-slab geometry binding (CUDA feature; no GPU execution test)

C07-15는 `layout.total_bytes()`와 정확히 같은 길이의 opaque device allocation을 cold prepare에서 한 번 만들고 layout·geometry digest와 함께
owner에 보관한다. C07-14의 successful pinned lease만 받을 수 있는 binding API는 digest가 아니라 complete layout equality를 먼저 확인한다.
mismatch는 device/pinned geometry digest를 가진 typed error이며 allocation·copy·native command를 만들지 않는다. matching result는 same cold
layout의 `&CudaDeviceBuffer`와 `&CudaPinnedHostBuffer`를 independent lifetime으로 함께 보관할 뿐 raw pointer나 byte view를 노출하지 않는다.
device owner의 mutable borrow와 pinned lease의 retained borrow가 binding이 살아 있는 동안 rebind, owner move, explicit close를 막는다.

이 binding은 device allocation ownership과 geometry equality만 증명한다. H2D submission, stream/command batch, copy completion, device content
freshness, CUDA graph capture/replay, C06 signature/registry/dispatch, executor integration 또는 graph-safe/runnable authority는 만들지 않는다.
CPU-only source contract가 exact cold allocation과 no-copy boundary를 검증하고 CUDA feature는 toolkit이 존재하는 host에서만 compile-only로 확인한다.
actual GPU execution, transfer measurement, and graph replay는 이후 단계에 남기며 그 전까지 모든 path는 exact eager를 유지한다.

### C07-16 — exact V1 H2D command-batch enqueue (CUDA feature; no GPU execution test)

C07-16은 C07-15의 successful pinned/device two-lifetime binding을 caller-owned active command-batch proxy로 `&'stream` borrow한다.
adapter는 exact device offset `0`, pinned offset `0`, `binding.layout().total_bytes()` 길이의 H2D를 단 한 번 enqueue하고 native
`CudaResult<()>`를 번역 없이 반환한다. 이 stream-lifetime borrow는 batch가 끝날 때까지 pinned/device owner rebind, re-stage, move,
explicit close를 Rust level에서 막는다. `Ok(())`는 enqueue 성공만 뜻하며 device contents, copy completion, freshness, graph-ready 또는
execution authority를 뜻하지 않는다.

adapter는 stream 또는 command batch를 만들거나 finish하지 않는다. completion과 late error는 caller가 existing command-batch finish boundary에서
명시적으로 관찰해야 하며, C07-16은 async token, allocation, host read/write, D2H, CUDA graph capture/replay, C06 signature/registry/dispatch,
executor integration을 만들지 않는다. CPU-only source contract가 one-copy/full-length/lifetime/no-completion boundary를 검증하고 CUDA feature는
toolkit이 존재하는 host에서만 library compile-only로 확인한다. actual GPU execution, transfer measurement, and graph replay는 이후 단계에
남기며 그 전까지 모든 path는 exact eager를 유지한다.

## 1. 배경과 가설

M1 이후 대표 candidate의 host execute와 CUDA stream span 차이가 매우 작았다는 evidence가 있어 CUDA Graph가 단독으로 큰 TPOT 개선을 만들지 못할 가능성도 있다. 따라서 이 PR은 `Graph가 반드시 빠르다`를 전제로 하지 않는다.

사전 가설은 다음과 같다.

> stable-address pure decode에서 반복 kernel submission과 host jitter를 graph replay로 줄이면 representative TPOT p50/p95가 사전 threshold 이상 개선된다.

실패하면 graph path를 default로 승격하지 않고 evidence와 함께 opt-in 또는 rejected 상태로 남긴다.

## 2. 대상 경로

초기 graph는 다음 조건을 모두 만족할 때만 사용한다.

- pure decode iteration
- active-row bucket `1,2,4,8,16,32`
- prepared model/weight/KV owner
- fixed metadata layout signature
- capture-safe attention/GEMM/kernel inventory
- temperature 0, repetition penalty 1, finish-token masking 없음
- GPU greedy output
- iteration-batch completion

하나라도 다르면 exact eager다.

## 3. Stable address 설계

Graph는 pointer가 매 iteration 바뀌지 않아야 한다. executor owner가 다음 maximum-capacity storage를 cold prepare한다.

```text
fixed token input buffer
fixed packed metadata host slab
fixed packed metadata device slab
fixed descriptor/control buffer
shared forward workspace
shared output token/status buffer
shared weights/KV/RoPE
bucket-specific GEMM descriptors
```

iteration별 값은 address를 바꾸지 않고 buffer contents만 갱신한다.

## 4. Metadata layout

packed slab의 offset이 실제 길이에 따라 움직이지 않도록 bucket별 fixed-offset layout을 사용한다.

```text
header
bucket-sized token IDs
bucket-sized position IDs
fixed-capacity request row metadata
fixed-capacity block-table descriptors
output slot metadata
control/status
```

사용하지 않는 tail은 deterministic zero/sentinel로 채운다. layout signature에는 field offset, size, alignment, schema version이 포함된다.

H2D는 다음 두 후보를 독립 비교한다.

1. graph 안의 fixed-size memcpy node
2. graph launch 전 same-stream async copy + graph replay

한 PR에서 결과를 본 뒤 유리한 후보만 숨기지 않고 두 candidate의 evidence를 보존한다.

## 5. Capture boundary

host validation과 scheduler plan packing은 capture 밖에서 완료한다.

Graph 내부 후보:

```text
fixed metadata H2D 또는 dependency
embedding
layer loop:
  norms/projections/RoPE/KV write/attention/MLP/residual
final norm
LM head
GPU greedy
small token/status D2H 또는 completion dependency
```

embedding token range validation처럼 host report/synchronize가 필요한 operation은 capture-safe prevalidation 또는 device status aggregation으로 분리한다. capture를 위해 안전 검사를 제거하지 않는다.

## 6. Graph prepare

- startup 또는 model load의 cold phase에서 bucket별 capture
- sample metadata는 production-valid sentinel을 사용
- instantiate 후 one-shot parity 확인
- graph resource bytes와 capture time 기록
- 실패 bucket은 startup receipt에 제외 이유를 기록하고 eager fallback

request hot path에서 새로운 bucket을 capture하지 않는다.

## 7. Launch sequence

```text
validate immutable IterationPlan
select bucket M
pack fixed-layout metadata
copy/update control data
lookup exact GraphSignature
launch graph
wait/query one completion boundary
download/validate token status
scheduler commit
publish stream event
```

commit 실패 시 기존 abort/ownership 계약을 유지한다. graph를 재실행해 commit을 보상하지 않는다.

## 8. Correctness matrix

- bucket `1,2,4,8,16,32`
- active rows exactly bucket 및 padding case
- position/KV page boundary
- request/output slot permutation
- repeated decode 1, 2, 32, 128 steps
- graph/eager alternating history
- graph hit 후 unsupported sampling eager fallback
- capture/launch failure
- cancellation before launch, in-flight, post-result/pre-commit
- SmolLM2와 Qwen representative geometry

검증 항목:

- generated token hash exact
- final token/top-1 mismatch 0
- KV continuation parity
- inactive row zero/sentinel integrity
- graph replay determinism
- owner close allocation 0

## 9. Performance campaign

### Primary

- RTX 4090
- current accepted clean candidate
- SmolLM2 diagnostic `c1/p128/o32/greedy`
- exact eager candidate vs graph candidate
- independent process 5쌍 AB/BA

### Required regression

- `c2,c4,c8/p128/o32`
- `c1/p1024/o128`
- `c1/p4096/o128`
- mixed prefill/decode는 eager로 fallback하며 이전 대비 회귀가 없어야 함

### Metric

- request mean TPOT p50/p95
- TTFT p50/p95
- E2E, output tok/s
- host execute, CUDA API time
- stream span, GPU idle gap
- kernel launch submission count와 graph launch count
- peak VRAM, usable KV blocks
- graph hit rate

## 10. 사전 promotion gate

- token/failure/dropped trace 0
- primary paired TPOT improvement `>= 15%`
- M2 absolute target `TPOT p50 <= 3.58 ms`
- TTFT p95 ratio `<= 1.05`
- c8 throughput ratio `>= 0.95`
- long/mixed fallback workload p95 ratio `<= 1.05`
- peak VRAM 증가 `<= 5%`
- usable KV block 감소 없음 또는 사전 선언한 1% 미만 reservation
- hot allocation 0

15% 미만이면 default로 승격하지 않는다. lifecycle 안정성만 유의미하면 experimental/benchmark-only로 유지할 수 있다.

## 11. Configuration

C06의 CLI를 사용한다.

```text
--execution-graph-policy disabled|auto|require
--execution-graph-buckets 1,2,4,8,16,32
```

성능/correctness gate 전까지 default는 `disabled`다. 승격 시에도 `disabled` rollback이 항상 남아야 한다.

## 12. 예상 파일 변경

```text
crates/riley-runtime/src/llama/executor/graph.rs
crates/riley-runtime/src/llama/executor/metadata.rs
crates/riley-runtime/src/llama/executor/dispatch.rs
crates/riley-runtime/src/llama/executor/output.rs
crates/riley-runtime/tests/llama_graph_gpu.rs
crates/riley-scheduler/tests/llama_iteration_gpu.rs
crates/riley-server/src/benchmark.rs
benchmarks/results/<campaign>/...
docs/decode-performance-implementation-report.md
```

## 13. 실패와 recovery

- prepare/capture failure: 해당 bucket 제외, eager 유지
- launch 전 mismatch: eager
- launch completion 확인 + graph failure: graph poison, safe하면 이후 iteration eager
- completion 미확정: executor poison, request failure, 자동 재실행 금지
- repeated graph failure: circuit breaker로 graph policy disabled 상태 전환 가능하되 metric/receipt에 기록

## 14. 롤백

운영 즉시 rollback은 `--execution-graph-policy disabled`다. 코드 rollback은 bucket registration과 Llama graph implementation을 revert하되 C05/C06의 generic ABI/dispatcher는 독립적으로 유지할 수 있다.

## 15. 완료 정의

모든 준비 bucket에서 graph/eager 의미가 일치하고 사전 15%·M2 gate를 통과하여 `auto` default 승격 여부를 closed report로 판정할 수 있을 때 완료다. gate 미달도 정상적인 완료 결과이며 `not-promoted`로 기록한다.
