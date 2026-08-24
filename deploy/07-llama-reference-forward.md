# PR 07 — Llama-compatible 기준 Forward

**상태:** Planned  
**선행 조건:** [PR 06](06-core-primitives.md)  
**다음:** [PR 08 — Prefill Attention](08-prefill-attention.md)

[← 이전](06-core-primitives.md) | [목차](README.md) | [다음 →](08-prefill-attention.md)

## 목적

최적화보다 정확성을 우선하여 한 Llama-compatible checkpoint의 **cache 없는 full-sequence forward와 logits**를 완성한다.

이 단계가 Gate B다.

## 실행 그래프

```text
Token IDs
→ Embedding
→ N × Decoder Block
   → RMSNorm
   → Q/K/V projection
   → correctness-first attention
   → output projection
   → residual
   → RMSNorm
   → gated MLP
   → residual
→ Final RMSNorm
→ LM Head
→ Logits
```

## Attention reference

이 PR에서는 느린 분리 구현을 허용한다.

- QK matmul
- scale
- causal mask
- softmax
- AV matmul

목적은 attention interface와 tensor layout의 정확성을 검증하는 것이다. FlashAttention 대체는 PR 08에서 수행한다.

## Weight와 execution plan

- model load 후 immutable execution plan 생성
- layer별 weight binding 검증
- workspace 크기 사전 계산
- forward 중 weight name lookup 금지
- forward hot path에서 JSON/hash map 접근 금지

## 검증 레벨

1. embedding output
2. 첫 decoder block 주요 checkpoint
3. 중간 layer checksum
4. final hidden state
5. logits slice와 통계
6. top-k tokens

가능하면 PyTorch hook으로 생성한 golden fixture와 비교한다.

## 메모리 검증

- 반복 forward에서 allocation count 증가 없음
- peak VRAM 기록
- weight alias/tied head 확인
- 오류 발생 시 intermediate buffer 회수

## 비범위

- KV cache
- token generation loop
- optimized attention
- batching
- Qwen
- server

## 완료 기준

- [ ] 동일 token IDs에 대해 logits가 허용 오차 내 일치
- [ ] greedy next token이 golden result와 일치
- [ ] 여러 sequence length에서 causal mask 정확
- [ ] 100회 반복 forward에 메모리 증가 없음
- [ ] 오류 발생 시 어떤 layer/op에서 실패했는지 표시

**중단 조건:** token 결과가 일치하지 않으면 tolerance를 확대하지 말고 최초 divergence layer를 찾는다.

[← 이전](06-core-primitives.md) | [목차](README.md) | [다음 →](08-prefill-attention.md)
