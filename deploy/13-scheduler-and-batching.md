# PR 13 — Rust Scheduler와 Continuous Batching

**상태:** Planned  
**선행 조건:** [PR 12](12-qwen-compatibility.md)  
**다음:** [PR 14 — API와 Streaming](14-api-and-streaming.md)

[← 이전](12-qwen-compatibility.md) | [목차](README.md) | [다음 →](14-api-and-streaming.md)

## 목적

여러 요청의 prefill/decode 작업을 한 GPU iteration으로 구성하는 Rust-native scheduler를 구현한다.

## 첫 정책

처음에는 단순하고 설명 가능한 정책을 사용한다.

- FCFS
- starvation 방지용 waiting-time aging
- iteration별 token budget
- prefill chunk 상한
- decode request 우선 또는 명시적 비율
- KV block 부족 시 admission 거절

cache-aware priority, speculative decoding, offload는 제외한다.

## 상태 모델

```rust
Waiting
Admitted
Prefilling
Decoding
Finished
Cancelled
Failed
```

모든 전이는 한 곳에서 검증한다. request가 어떤 실패 경로에서도 두 번 free되지 않아야 한다.

## Batch plan

Scheduler는 GPU tensor를 직접 계산하지 않고 immutable한 실행 계획을 만든다.

```rust
IterationPlan {
  prefill_items,
  decode_items,
  token_count,
  block_tables,
  output_slots,
}
```

Runtime은 계획을 실행하고 결과를 scheduler에 반환한다.

## Backpressure

- 최대 waiting requests
- 최대 active sequences
- 최대 total tokens/KV blocks
- admission timeout 또는 즉시 reject 정책
- cancellation 우선 처리

## 테스트

- deterministic scheduler simulation
- mixed prompt lengths
- decode starvation 방지
- prefill chunking
- KV OOM admission
- request cancellation at every state
- output slot mapping
- random event property test
- long-running allocation accounting

## 지표

- queue wait
- scheduler CPU time
- iteration batch size
- batched tokens
- prefill/decode 비율
- KV utilization
- rejection/cancellation
- GPU idle gap

## 비범위

- HTTP server
- prefix caching
- priority tenant/SLA
- multi-GPU routing
- offload

## 완료 기준

- [ ] concurrency 1 결과가 단일 요청 path와 동일
- [ ] 여러 요청 결과가 독립 실행과 일치
- [ ] cancellation에서 block 누수 없음
- [ ] scheduler simulation이 재현 가능
- [ ] p95 queue/iteration 지표 수집
- [ ] overload 시 bounded memory 유지

[← 이전](12-qwen-compatibility.md) | [목차](README.md) | [다음 →](14-api-and-streaming.md)
