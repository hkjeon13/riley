# PR 10 — Paged KV Block Manager

**상태:** Implemented — `b0f45eb0ac6b3fd73d198f09eef58375a85f2349` 원격 GPU 검증 완료<br>
**선행 조건:** [PR 09](09-single-request-decode.md)  
**다음:** [PR 11 — Sampling과 Generation](11-sampling-and-generation.md)

[← 이전](09-single-request-decode.md) | [목차](README.md) | [다음 →](11-sampling-and-generation.md)

## 목적

연속 요청별 cache를 고정 크기 block pool과 block table 기반으로 바꾸되, 아직 scheduler나 prefix cache는 구현하지 않는다. block table ABI는 이후 exact paged decode와 선택적인 query-aware page 통계를 추가할 수 있도록 versioned metadata 확장 지점을 제공한다.

## 구현 결과

의미 보존 등급은 수학적 최적화가 아닌 exact systems layout 변경이므로 `reference`다.
실행 및 검증 기준 revision은 `b0f45eb0ac6b3fd73d198f09eef58375a85f2349`다.

- `5faced1` — exact paged CUDA scatter, materialized reference, D64 online decode
- `60a2fd2` — generation-safe host block pool과 lifecycle
- `03808c5` — default paged Llama runtime 통합과 GPU model test
- `43b6a0f`, `1975dcf`, `b0f45eb` — ABI narrowing, panic-free metric, all-feature lint gate 보강

기본 decode cache 정책을 exact paged cache로 변경했다. PR 09 contiguous cache는
`with_contiguous_kv_cache()`로 명시적으로 선택하는 reference/rollback 정책으로 남겼다.
Paged 오류가 발생했을 때 contiguous로 자동 재실행하지는 않는다.

물리 block 크기는 16 token이다. Separate BF16 K/V pool layout은
`[layer, physical_block, kv_head, token_in_block, head_dim]`이며 prepare 시 전체 pool을
한 번 할당한다. Device ID/valid-token 배열, pinned upload staging, host encoding,
duplicate-validation scratch와 attention workspace도 prepare 시 확보한다. RoPE 이후 K/V
scatter가 logical token을 V1 table의 physical block/offset으로 변환한다.

Materialized paged reference는 PR 09의 staged-BF16 reference 계산 지점을 유지한다. D64
optimized producer는 logical block마다 FP32 `DecodePartialState(m,l,n[D])` 하나를 만들고
PR 09 reducer가 logical order에서 merge한 뒤 한 번만 normalize한다. 두 경로 모두 모든
committed page를 읽는다. 여기서 exact는 page pruning/근사를 하지 않는다는 뜻이며,
reduction 순서가 다른 optimized 결과와 contiguous model logits가 bit-exact라는 뜻은 아니다.

## 데이터 구조

```rust
KvBlockPool
BlockId
BlockTable
SequenceState
KvLayout
AllocationError
```

필수 metadata:

- free/allocated 상태
- owner 또는 reference count
- logical token range
- physical block index
- block 내 유효 token 수
- layer stride/layout
- generation/cookie로 stale handle 방지
- block table format version

## Versioned block table

초기 실행 경로에는 address translation에 필요한 정보만 둔다.

```rust
struct BlockTableV1 {
    block_ids: DeviceOrHostView<BlockId>,
    valid_tokens: DeviceOrHostView<u16>,
    logical_length: u32,
}
```

향후 page selection 또는 offload 기능의 metadata를 block address와 강하게 결합하지 않는다. 선택적 sidecar를 별도 capability로 연결한다.

```rust
struct OptionalBlockMetadata {
    version: u16,
    kind: MetadataKind,
    device_view: OpaqueDeviceView,
}
```

가능한 후속 metadata 예:

- key dimension별 min/max
- value norm upper bound
- quantization scale
- offload tier와 residency
- prefix hash 또는 reuse metadata

이 PR에서는 이러한 통계를 계산하거나 사용하는 코드를 구현하지 않는다. sidecar가 없을 때 exact paged path가 추가 branch 없이 동작하는 것을 우선한다.

구현된 V1 exact address table은 numeric version `1`과 block size `16`을 고정한다.

- logical-block 순서의 distinct U32 physical IDs
- U16 valid-token counts
- U32 logical length
- `block_count == ceil(logical_length / 16)`
- 앞 block의 valid count는 16, 마지막 block만 `1..=16`
- 모든 physical ID는 prepared pool 범위 안에 존재

Safe Rust 경계는 version, 배열 길이, valid count, duplicate와 out-of-pool ID를 CUDA 실행
전에 검사한다. 물리 block당 1 byte duplicate scratch를 prepare 시 확보해 검사는
O(physical blocks + logical blocks)이고 per-token heap preparation이 없다. Runtime/CUDA의
version과 block-size 상수는 compile-time assertion으로 연결하며 C11/CUDA C++/Rust ABI
layout assertion도 유지한다.

Device V1 payload에는 U32 physical index만 있고 pool cookie나 generation은 없다. 따라서
low-level CUDA table 자체가 stale generation을 검출한다고 주장하지 않는다. Runtime의
`BlockId(pool_cookie, physical_index, generation)`와 owner 검증이 table upload 전에
foreign/stale/double-free/wrong-owner를 구분하며 reservation lifetime 동안 reuse를 막는다.

Optional sidecar는 host manager에서 별도 version으로 관리하고 현재 block generation에
묶는다. Content 변경, truncate, free 또는 reuse 시 invalidate된다. CUDA V1 exact ABI는
sidecar 없음만 받으므로 기본 실행에는 metadata-dependent branch가 없다.

## 정책

- 고정 block size 하나로 시작
- deterministic free list 우선
- 한 요청만 사용해도 paged address translation 검증
- allocation과 execution을 분리
- OOM 시 부분 할당 rollback
- metadata sidecar는 optional·versioned·lifetime-bound
- block reuse 시 이전 sidecar도 함께 invalidate

## Allocation transaction과 lifecycle

- `reserve_to`는 필요한 block 수를 먼저 확인하고 tentative block만 확보한다. Committed
  logical length는 바꾸지 않으며 OOM은 model/device mutation 전에 반환한다.
- Reserved table upload와 모든 layer 실행이 성공해야 `commit`이 logical length를
  마지막으로 publish한다.
- Device content를 변경하지 않은 실패는 explicit rollback으로 tentative block을 반납한다.
- Reservation 이후 CUDA/model mutation 실패는 fail-closed로 sequence를 poison하고
  tentative block을 회수하며 관련 sidecar를 invalidate한다.
- Reservation token을 잃어도 성공/실패를 추측하지 않는다. Abandoned recovery가 rollback과
  poison을 함께 수행한다.
- `reset`은 block을 모두 반환하고 preallocated host/device storage는 유지한다. 같은 physical
  index가 다시 선택될 수 있지만 generation은 단조 증가한다.
- Cancellation ownership transfer는 non-cloneable reclaim token을 사용한다.
- Explicit close는 오류를 관찰하는 정상 cleanup 경로이고 Drop은 best-effort fallback이다.

## GPU 경로

- RoPE 적용 후 K/V를 지정 block/offset에 기록
- block table을 decode attention에 전달
- PR 09의 `DecodePartialState`를 block별 또는 여러 block 묶음별로 생성 가능
- 필요하면 작은 metadata buffer를 pinned host에서 async copy
- 매 token마다 전체 block table을 재할당하지 않음
- optional sidecar가 없는 기본 경로의 성능을 우선

## 테스트

- 여러 길이의 allocate/free
- block 경계 직전/직후 token
- random allocation sequence property test
- OOM과 rollback
- double free/stale handle 검출
- request cancellation simulation
- block reuse 시 이전 데이터가 결과에 영향을 주지 않음
- contiguous cache 결과와 parity
- sidecar 없음/빈 sidecar/unknown version 처리
- block 해제 시 sidecar lifetime 종료
- page 처리 순서를 바꾼 exact partial-state merge

## 메모리 지표

- `usable_kv_bytes`: 현재 사용량이 아니라 preallocated K+V physical pool 전체 용량
- `block_table_device_bytes`: preallocated U32 ID와 U16 valid-token device 배열 용량
- `block_table_host_bytes`: generation-bound handles, U32/U16 backing, host encoding,
  duplicate scratch, pool slot/free-list metadata의 선언된 capacity
- `cache_unused_capacity_bytes`: maximum logical length를 넘는 정적 pool 용량. Block rounding과
  explicit overprovisioning을 포함하며 현재 tail fragmentation과는 별도
- `paged_internal_fragmentation_bytes`: 현재 committed 마지막 block의 미사용 token slot bytes.
  Empty sequence는 0
- `sidecar_device_bytes`: `OpaqueDeviceView.byte_len`의 합. Shared/overlapping view도
  descriptor마다 합산하므로 unique physical allocation bytes가 아님
- `free_block_count`, `allocated_block_count`, `high_water_mark`: 물리 lease accounting
- `lifetime_allocation_count`: 성공한 physical-block lease 수
- Allocation latency: 성공한 free-list pop, generation 증가, owner bind만 측정. OOM preflight,
  실패 allocation, table encoding/upload, CUDA allocation과 model execution은 제외

SmolLM2 shape의 K+V는 token당 23,040 bytes, block당 368,640 bytes다. 8,192-token pool은
512 blocks와 188,743,680 usable KV bytes를 preallocate한다. 최대 device V1 table은
3,072 bytes이고 pinned staging은 2,048 bytes다. Paged cache 자체는 K/V와 두 table을 위한
device allocation 4개, table staging을 위한 pinned allocation 1개를 소유한다.

## 후속 error-bounded page selection을 위한 제약

PR 17에서 query-aware page selection을 검토할 수 있지만, PR 10의 완료 조건은 exact paged cache다.

후속 기능이 min/max summary를 사용하더라도 다음을 지켜야 한다.

- summary는 실제 block content와 동일 generation에 속함
- partial block의 유효 token만 summary에 포함
- K가 갱신되면 summary도 invalid 또는 갱신
- mask와 head/group layout을 summary 계산이 정확히 반영
- sidecar 미지원 backend는 exact full-page scan으로 fallback

## 비범위

- prefix block sharing
- CPU/SSD offload
- eviction policy
- multi-GPU
- scheduler priority
- key min/max 또는 value norm summary 계산
- query-aware page pruning
- approximate attention
- paged 오류의 자동 contiguous retry/fallback
- device V1 table 자체의 generation 검증
- sidecar 통계 계산 또는 실행 경로에서의 소비
- D64 optimized capability 이외 shape의 paged 성능 최적화

## 검증 결과

검증은 exact source snapshot `b0f45eb0…`을 read-only mount한 `server-4096`에서 수행했다.
RTX 4090(sm89, 24,564 MiB), driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93,
Rust 1.85.0 환경이고 container network는 비활성화했다. Image는
`rustinfer-native-cuda:pr04-c6c93e2`
(`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`)다.
모든 Cargo command는 locked/offline으로 실행했고 로컬에서는 model/CUDA/GPU inference를
실행하지 않았다.

### Correctness와 boundary

- Low-level paged GPU test는 logical length `1/15/16/17/31/32/33/128/129`와 shuffled
  physical IDs를 통과했다. Paged materialized reference는 contiguous reference와 exact였고,
  실제 paged workspace를 descending order로 다시 reduce한 결과도 gate를 통과했다.
- SmolLM2 contiguous↔paged 32 decode는 33 rows에서 top-1 mismatch 0, prefill byte-exact였다.
  Worst diagnostic은 cosine `0.999664329560`, max abs `0.3125`, mean abs `0.136817740`이다.
- 128 decode는 129 rows에서 top-1 mismatch 0, prefill byte-exact였다. Worst diagnostic은
  cosine `0.998526788174`, max abs `1.625`, mean abs `0.501330264`이다.
- Numeric 값은 연산 순서 차이를 포함한 diagnostic이며 E0 numeric 재인증이나 bit-exact
  optimized model decode 주장으로 해석하지 않는다.
- PR 09 contiguous GPU regression 3개가 모두 통과했다.

### Lifecycle, OOM, sanitizer

- 같은 physical block을 generation `1 → 2 → 3`으로 재사용했다. Same-prompt replay와
  different-prompt fresh 비교가 byte-exact였고 contamination은 없었다.
- Invalid prompt는 reservation/table upload 전에 실패했고 owner를 poison하지 않았다.
- 1-block pool은 logical length 16에서 다음 block을 pre-mutation OOM으로 거절했다. Table,
  logits, pool/CUDA accounting은 유지됐고 reset 뒤 byte-exact replay가 가능했다.
- 8,064-token prefill + 128 decode로 8,192 tokens/512 blocks에 도달했다. 다음 call은
  table/logits/pool mutation 전 capacity error를 반환했다.
- Reset은 block을 전부 반환했고 explicit close와 implicit paged Drop 뒤 CUDA allocation
  accounting은 0이었다.
- Lifecycle run은 high-water mark 1, 성공 lease 3회, lease bookkeeping latency 합 1,533 ns,
  최대 599 ns였다. Clock-controlled allocator benchmark가 아니므로 성능 수치로 일반화하지
  않는다.
- Compute Sanitizer low-level/model memcheck는 `0 errors, 0 bytes leaked`, low-level
  racecheck는 `0 hazards, 0 errors, 0 warnings`였다.
- Workspace all-features는 `125 passed, 0 failed, 46 ignored`, doctest는
  `13 passed, 0 failed`였다. Strict all-feature Clippy와 independent C11 ABI 검사도 통과했다.

Nsight Compute는 shape 129 producer/reducer와 near-limit 첫 decode(logical 8,065)의 launch
geometry를 보존했다. Producer grid는 각각 `(9,9,1)`과 `(505,9,1)`, reducer는 `(1,1,1)`이다.
모든 CSV에 `No metrics to collect found in sections` 경고가 있으므로 이는 page-scaled launch
구조 증거일 뿐 bandwidth, occupancy 또는 성능 우열 자료가 아니다.

Successful prefill/decode의 Rust library hot path는 prepare 때 확보한 table backing,
encoding, duplicate scratch와 workspace를 재사용한다. Source-contract test는 hot region에
`Vec`, `Box`, `vec!`, `collect`, `String`, `format!`, device/pinned allocation 호출 등이
들어오는 것을 막는다. 이는 library source/structure 수준의 보장이며 CUDA driver/cuBLASLt
내부, test harness, logits destination 또는 호출자 callback까지 포함한 whole-process
zero-allocation 측정은 아니다.

전체 evidence는
[`20260825T091408Z-rustinfer-paged-kv-pr10-run001`](../benchmarks/results/20260825T091408Z-rustinfer-paged-kv-pr10-run001/README.md)에 있다.
원격 append-only root와 checksum은 다음과 같다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr10/b0f45eb/full
source tar sha256=ffadca38bc96170e4858b36f97b0d6ff9d403af29b7d8ca5a56a61aa7d3e2a88
SHA256SUMS sha256=868ea7d12278f576965c1651a294dbc92e672459063f3d9301e2f6b2eea73150
```

앞선 `03808c5`, `43b6a0f`, `1975dcf` remote roots는 strict Clippy 실패 원인 추적용이고
통과 evidence가 아니다.

## 완료 기준

- [x] contiguous cache와 동일 token 결과
- [x] block 경계 테스트 통과
- [x] allocation/free accounting 정확
- [x] OOM 후 pool 상태가 일관됨
- [x] successful Rust library decode hot path가 preallocated storage를 재사용하고 source guard 통과
- [x] block table format과 versioning이 문서화됨
- [x] optional sidecar 없이 exact path가 동작
- [x] runtime host manager가 stale block/metadata를 generation/cookie로 검출

[← 이전](09-single-request-decode.md) | [목차](README.md) | [다음 →](11-sampling-and-generation.md)
