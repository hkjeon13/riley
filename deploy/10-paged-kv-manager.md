# PR 10 — Paged KV Block Manager

**상태:** Planned  
**선행 조건:** [PR 09](09-single-request-decode.md)  
**다음:** [PR 11 — Sampling과 Generation](11-sampling-and-generation.md)

[← 이전](09-single-request-decode.md) | [목차](README.md) | [다음 →](11-sampling-and-generation.md)

## 목적

연속 요청별 cache를 고정 크기 block pool과 block table 기반으로 바꾸되, 아직 scheduler나 prefix cache는 구현하지 않는다.

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
- layer stride/layout
- generation/cookie로 stale handle 방지

## 정책

- 고정 block size 하나로 시작
- deterministic free list 우선
- 한 요청만 사용해도 paged address translation 검증
- allocation과 execution을 분리
- OOM 시 부분 할당 rollback

## GPU 경로

- RoPE 적용 후 K/V를 지정 block/offset에 기록
- block table을 decode attention에 전달
- 필요하면 작은 metadata buffer를 pinned host에서 async copy
- 매 token마다 전체 block table을 재할당하지 않음

## 테스트

- 여러 길이의 allocate/free
- block 경계 직전/직후 token
- random allocation sequence property test
- OOM과 rollback
- double free/stale handle 검출
- request cancellation simulation
- block reuse 시 이전 데이터가 결과에 영향을 주지 않음
- contiguous cache 결과와 parity

## 메모리 지표

- usable KV bytes
- metadata overhead
- internal fragmentation
- free block count
- high-water mark
- allocation latency

## 비범위

- prefix block sharing
- CPU/SSD offload
- eviction policy
- multi-GPU
- scheduler priority

## 완료 기준

- [ ] contiguous cache와 동일 token 결과
- [ ] block 경계 테스트 통과
- [ ] allocation/free accounting 정확
- [ ] OOM 후 pool 상태가 일관됨
- [ ] decode step에서 host heap allocation 없음
- [ ] block table format이 문서화됨

[← 이전](09-single-request-decode.md) | [목차](README.md) | [다음 →](11-sampling-and-generation.md)
