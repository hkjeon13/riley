# PR 08 — Attention Backend와 Prefill 경로

**상태:** Planned  
**선행 조건:** [PR 07](07-llama-reference-forward.md)  
**다음:** [PR 09 — 단일 요청 Decode](09-single-request-decode.md)

[← 이전](07-llama-reference-forward.md) | [목차](README.md) | [다음 →](09-single-request-decode.md)

## 목적

correctness-first attention을 교체 가능한 backend interface로 만들고, full-sequence prefill에서 검증된 고성능 backend를 사용한다.

## Interface 요구사항

```rust
AttentionMode::Prefill
Q/K/V views
head counts
head dimension
causal/local mask
scale
workspace
stream
output view
```

backend는 capability를 선언한다.

- supported GPU arch
- dtype
- head dimension
- causal 여부
- variable sequence 지원
- non-contiguous layout 지원 여부
- CUDA Graph capture 가능 여부

## 구현 순서

1. 기존 reference backend를 interface 뒤로 이동
2. 검증된 외부 또는 vendor attention backend 연결
3. unsupported shape는 reference로 fallback
4. backend 선택 이유를 trace에 기록
5. output parity 검증

처음부터 universal custom FlashAttention을 작성하지 않는다.

## Shape matrix

- sequence 1, 128, 1K, 4K
- batch 1, 2, 4
- MHA와 target GQA shape
- padding 없는 packed 또는 dense input 중 선택한 형식
- target head dimension

## 성능 판단

- attention kernel latency
- end-to-end prefill latency
- TTFT 영향
- workspace와 peak VRAM
- layout conversion 비용
- fallback 비율

backend 자체는 빨라도 앞뒤 transpose/copy 때문에 전체가 느려질 수 있으므로 trace로 확인한다.

## 비범위

- decode attention
- paged cache
- variable-request continuous batching
- custom fusion

## 완료 기준

- [ ] reference와 optimized backend parity
- [ ] target prefill shapes 모두 실행
- [ ] unsupported 조합이 안전하게 fallback 또는 명시적 실패
- [ ] baseline 대비 prefill 수치가 raw result로 보존
- [ ] hidden copy/contiguous 비용이 profiler에 표시됨

[← 이전](07-llama-reference-forward.md) | [목차](README.md) | [다음 →](09-single-request-decode.md)
