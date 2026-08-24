# PR 11 — Logits Processing, Sampling, Generation Loop

**상태:** Planned  
**선행 조건:** [PR 10](10-paged-kv-manager.md)  
**다음:** [PR 12 — Qwen 호환성](12-qwen-compatibility.md)

[← 이전](10-paged-kv-manager.md) | [목차](README.md) | [다음 →](12-qwen-compatibility.md)

## 목적

prefill, paged KV decode와 token 선택을 연결해 단일 요청의 완전한 autoregressive generation을 만든다. 이후 rejection-sampling 기반 speculative decoding을 정확하게 추가할 수 있도록 **request-local RNG와 sampling 의미 계약**을 먼저 고정한다.

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

그 후 같은 PR에서 무리하게 GPU sampler를 추가하지 않는다. GPU sampling이 필요하면 작은 후속 commit 또는 PR 15 최적화 대상으로 둔다. CPU와 GPU sampler는 동일한 logits processing 순서와 RNG contract를 구현해야 한다.

## Sampling pipeline 순서

동일한 parameter 집합이라도 처리 순서가 달라지면 분포가 달라질 수 있으므로 순서를 명시한다.

```text
raw logits
→ token constraints
→ repetition or other penalties
→ temperature
→ top-k
→ top-p
→ normalization or equivalent sampling
→ categorical sample
```

각 단계는 입력·출력 dtype과 masking 값의 의미를 문서화한다.

## RNG 계약

- explicit seed
- request별 RNG state
- batch 순서 변화가 다른 request RNG를 오염시키지 않음
- deterministic mode와 high-performance mode 구분 가능
- RNG algorithm과 version을 결과 metadata에 기록
- snapshot, restore, fork 기능을 정의
- rejected 또는 cancelled branch의 RNG 소비 정책을 명시

권장 interface:

```rust
trait RequestRng {
    type Snapshot;

    fn snapshot(&self) -> Self::Snapshot;
    fn restore(&mut self, snapshot: &Self::Snapshot);
    fn fork(&self, domain: RngDomain) -> Self;
    fn next_uniform(&mut self) -> f32;
}
```

`fork`는 draft, target correction, user-visible sampling처럼 서로 다른 확률 경로가 같은 request 안에서도 독립적인 stream을 갖도록 한다.

## Sampling 결과 계약

```rust
struct SamplingResult {
    token_id: u32,
    token_logprob: Option<f32>,
    finish_reason: Option<FinishReason>,
}
```

추후 speculative verifier가 필요로 하는 기능을 위해 sampler 내부에는 다음 primitive를 분리한다.

- processed logits 생성
- 특정 token의 probability 또는 log-probability 조회
- categorical sample
- 두 분포의 positive residual에서 sample할 수 있는 확장 지점

PR 11에서는 residual distribution sampler를 구현하지 않는다. API가 ordinary sampling과 강하게 결합되지 않도록 경계만 정한다.

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
- snapshot 후 sample, restore 후 재실행 결과 일치
- request RNG fork 간 독립성
- batch 처리 순서 변경 시 request별 결과 계약 확인
- top-k/top-p boundary
- temperature 0 처리
- all-masked 또는 degenerate distribution 오류
- EOS 즉시 종료
- max token 종료
- UTF-8 decode boundary
- cancellation 시 cache와 RNG state 회수
- callback/consumer 오류 시 cleanup

## Gumbel 계열에 대한 범위 결정

Gumbel-Softmax는 categorical variable의 학습용 연속 relaxation이며, 일반적인 inference sampling의 연산량을 직접 줄이는 기본 경로로 사용하지 않는다. 따라서 PR 11 범위에 포함하지 않는다.

향후 candidate-tree 또는 특수 stochastic decoding에서 Gumbel-Max/Top-k가 필요해지면 ordinary sampler와 분리된 연구 PR로 검토한다.

## 비범위

- continuous batching
- HTTP/SSE
- beam search
- speculative decoding
- residual distribution correction
- Gumbel-Softmax
- structured output

## 완료 기준

- [ ] 단일 요청이 token stream을 생성
- [ ] greedy golden sequence 일치
- [ ] fixed seed sampling 재현
- [ ] RNG snapshot/restore/fork unit test 통과
- [ ] sampling transform 순서와 알고리즘 version이 문서화됨
- [ ] 모든 종료 경로에서 KV block과 RNG state 회수
- [ ] 생성 상태가 model runtime과 분리됨
- [ ] per-token CPU/GPU timing이 수집됨

[← 이전](10-paged-kv-manager.md) | [목차](README.md) | [다음 →](12-qwen-compatibility.md)