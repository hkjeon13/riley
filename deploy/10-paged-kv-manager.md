# PR 10 — Paged KV Block Manager

**상태:** Planned  
**선행 조건:** [PR 09](09-single-request-decode.md)  
**다음:** [PR 11 — Sampling과 Generation](11-sampling-and-generation.md)

[← 이전](09-single-request-decode.md) | [목차](README.md) | [다음 →](11-sampling-and-generation.md)

## 목적

연속 요청별 cache를 고정 크기 block pool과 block table 기반으로 바꾸되, 아직 scheduler나 prefix cache는 구현하지 않는다. block table ABI는 이후 exact paged decode와 선택적인 query-aware page 통계를 추가할 수 있도록 versioned metadata 확장 지점을 제공한다.

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

## 정책

- 고정 block size 하나로 시작
- deterministic free list 우선
- 한 요청만 사용해도 paged address translation 검증
- allocation과 execution을 분리
- OOM 시 부분 할당 rollback
- metadata sidecar는 optional·versioned·lifetime-bound
- block reuse 시 이전 sidecar도 함께 invalidate

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

- usable KV bytes
- block table metadata overhead
- optional sidecar bytes
- internal fragmentation
- free block count
- high-water mark
- allocation latency

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

## 완료 기준

- [ ] contiguous cache와 동일 token 결과
- [ ] block 경계 테스트 통과
- [ ] allocation/free accounting 정확
- [ ] OOM 후 pool 상태가 일관됨
- [ ] decode step에서 host heap allocation 없음
- [ ] block table format과 versioning이 문서화됨
- [ ] optional sidecar 없이 exact fast path가 동작
- [ ] stale block과 stale metadata를 generation/cookie로 검출

[← 이전](09-single-request-decode.md) | [목차](README.md) | [다음 →](11-sampling-and-generation.md)