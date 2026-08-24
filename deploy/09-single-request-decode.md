# PR 09 — 연속 KV Cache 기반 단일 요청 Decode

**상태:** Planned  
**선행 조건:** [PR 08](08-prefill-attention.md)  
**다음:** [PR 10 — Paged KV Manager](10-paged-kv-manager.md)

[← 이전](08-prefill-attention.md) | [목차](README.md) | [다음 →](10-paged-kv-manager.md)

## 목적

한 요청에 대해 prefill 후 token 하나씩 생성할 수 있는 contiguous/static KV cache와 decode attention을 구현한다. decode attention은 긴 KV 범위를 작은 chunk로 처리하고 PR 08의 `(m, l, n)` 부분합을 병합할 수 있는 exact 경로를 사용한다.

## 이번 단계의 cache

단순성을 위해 요청별 연속 cache를 사용한다.

```text
[layer][K or V][batch][kv_head][max_seq][head_dim]
```

최대 길이는 요청 시작 시 고정한다. Paged allocation은 PR 10으로 미룬다.

## 필수 동작

- cache 사전 할당
- prefill K/V write
- decode position에 K/V append
- 현재 logical length 관리
- boundary 검사
- request reset/drop
- cache on/off parity

## Decode attention

- query length 1 우선
- GQA/MQA head mapping
- cached K/V read
- causal mask의 implicit 처리 가능
- output projection 전 layout 규약 고정
- KV range별 online softmax partial state
- 여러 range의 associative merge

reference decode path와 optimized path를 분리한다.

## `DecodePartialState` 계약

KV 범위 하나의 출력은 정규화된 attention output이 아니라 다음 부분합이다.

```rust
struct DecodePartialState {
    max_score: Accumulator,
    exp_sum: Accumulator,
    weighted_value_sum: VectorAccumulator,
}
```

merge는 PR 08의 online softmax 공식과 동일하다. 이 구조를 contiguous KV에서 먼저 검증하면 PR 10의 paged block과 이후 split-K decode에도 같은 reducer를 재사용할 수 있다.

필수 규칙:

- logical KV 순서와 physical chunk 순서를 구분
- empty chunk와 fully masked chunk 표현
- accumulator dtype 고정
- merge 결과를 마지막 한 번만 normalize
- 부분 output을 먼저 normalize한 뒤 평균내는 잘못된 구현 금지

## 구현 순서

1. 전체 contiguous KV를 한 번에 읽는 reference decode
2. 동일 KV를 고정 chunk로 나누는 partial-state reference
3. CPU 또는 단순 CUDA merge 검증
4. optimized decode backend 연결
5. chunk 수와 CTA partition을 runtime capability로 선택
6. one-pass와 split-range 결과 비교

## 검증

- full forward의 마지막 token logits와 prefill+decode logits 비교
- cache on/off greedy token 일치
- one-range와 multi-range decode 비교
- chunk boundary를 1, head-dependent 값, 임의 값으로 변경
- merge 순서 변경에 대한 dtype별 tolerance
- 1, 2, 32, 128 step decode
- max sequence boundary
- EOS 직전 cache length
- reset 후 재사용
- 서로 다른 prompt를 순차 실행했을 때 데이터 오염 없음

## 성능

- per-token latency
- kernel launch count
- K/V read bandwidth
- partial-state bytes와 merge 비용
- CPU launch overhead
- GPU idle gap
- allocation 없는 steady-state 여부

긴 context에서 decode는 계산량보다 KV memory traffic이 지배할 수 있으므로 FLOPs뿐 아니라 실제 읽은 KV bytes를 기록한다.

## 비범위

- 여러 요청
- block pool
- prefix sharing
- offload
- KV page pruning 또는 근사 attention
- sampling 정책
- API streaming

## 완료 기준

- [ ] 단일 요청에서 prefill+N decode가 정확함
- [ ] cache on/off greedy sequence 일치
- [ ] one-range와 multi-range 부분합 결과가 허용 오차 내 일치
- [ ] decode hot path에 device allocation 없음
- [ ] 최대 길이 초과가 안전한 오류로 종료
- [ ] request drop 후 VRAM accounting 복귀
- [ ] partial-state ABI가 PR 10 paged block에서도 재사용 가능하게 문서화됨

[← 이전](08-prefill-attention.md) | [목차](README.md) | [다음 →](10-paged-kv-manager.md)