# PR 09 — 연속 KV Cache 기반 단일 요청 Decode

**상태:** Planned  
**선행 조건:** [PR 08](08-prefill-attention.md)  
**다음:** [PR 10 — Paged KV Manager](10-paged-kv-manager.md)

[← 이전](08-prefill-attention.md) | [목차](README.md) | [다음 →](10-paged-kv-manager.md)

## 목적

한 요청에 대해 prefill 후 token 하나씩 생성할 수 있는 contiguous/static KV cache와 decode attention을 구현한다.

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

reference decode path와 optimized path를 분리한다.

## 검증

- full forward의 마지막 token logits와 prefill+decode logits 비교
- cache on/off greedy token 일치
- 1, 2, 32, 128 step decode
- max sequence boundary
- EOS 직전 cache length
- reset 후 재사용
- 서로 다른 prompt를 순차 실행했을 때 데이터 오염 없음

## 성능

- per-token latency
- kernel launch count
- K/V read bandwidth
- CPU launch overhead
- allocation 없는 steady-state 여부

## 비범위

- 여러 요청
- block pool
- prefix sharing
- offload
- sampling 정책
- API streaming

## 완료 기준

- [ ] 단일 요청에서 prefill+N decode가 정확함
- [ ] cache on/off greedy sequence 일치
- [ ] decode hot path에 device allocation 없음
- [ ] 최대 길이 초과가 안전한 오류로 종료
- [ ] request drop 후 VRAM accounting 복귀

[← 이전](08-prefill-attention.md) | [목차](README.md) | [다음 →](10-paged-kv-manager.md)
