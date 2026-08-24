# PR 11 — Logits Processing, Sampling, Generation Loop

**상태:** Planned  
**선행 조건:** [PR 10](10-paged-kv-manager.md)  
**다음:** [PR 12 — Qwen 호환성](12-qwen-compatibility.md)

[← 이전](10-paged-kv-manager.md) | [목차](README.md) | [다음 →](12-qwen-compatibility.md)

## 목적

prefill, paged KV decode와 token 선택을 연결해 단일 요청의 완전한 autoregressive generation을 만든다.

이 단계가 Gate C다.

## 정책 범위

초기 구현 순서:

1. greedy
2. temperature
3. top-k
4. top-p
5. repetition penalty
6. min/max new tokens
7. EOS/stop token

frequency/presence penalty와 복잡한 grammar constraint는 뒤로 미룬다.

## CPU와 GPU 경계

첫 버전은 정확성을 위해 CPU sampling을 허용할 수 있다. 단, logits 전체 copy 비용을 측정한다.

그 후 같은 PR에서 무리하게 GPU sampler를 추가하지 않는다. GPU sampling이 필요하면 작은 후속 commit 또는 PR 15 최적화 대상으로 둔다.

## RNG 계약

- explicit seed
- request별 RNG state
- batch 순서 변화가 다른 request RNG를 오염시키지 않음
- deterministic mode와 high-performance mode 구분 가능

## Generation state

```rust
GenerationRequest
GenerationState
SamplingParams
StopState
GeneratedToken
FinishReason
```

Finish reason:

- eos
- stop token/string
- length
- cancelled
- error

## 검증

- greedy golden token exact match
- fixed seed 반복 결과
- top-k/top-p boundary
- temperature 0 처리
- EOS 즉시 종료
- max token 종료
- UTF-8 decode boundary
- cancellation 시 cache 회수
- callback/consumer 오류 시 cleanup

## 비범위

- continuous batching
- HTTP/SSE
- beam search
- speculative decoding
- structured output

## 완료 기준

- [ ] 단일 요청이 token stream을 생성
- [ ] greedy golden sequence 일치
- [ ] fixed seed sampling 재현
- [ ] 모든 종료 경로에서 KV block 회수
- [ ] 생성 상태가 model runtime과 분리됨
- [ ] per-token CPU/GPU timing이 수집됨

[← 이전](10-paged-kv-manager.md) | [목차](README.md) | [다음 →](12-qwen-compatibility.md)
