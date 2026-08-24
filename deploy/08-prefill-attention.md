# PR 08 — Attention Backend와 Prefill 경로

**상태:** Planned  
**선행 조건:** [PR 07](07-llama-reference-forward.md)  
**다음:** [PR 09 — 단일 요청 Decode](09-single-request-decode.md)

[← 이전](07-llama-reference-forward.md) | [목차](README.md) | [다음 →](09-single-request-decode.md)

## 목적

correctness-first attention을 교체 가능한 backend interface로 만들고, full-sequence prefill에서 검증된 고성능 backend를 사용한다. 최적화된 경로는 score matrix 전체를 HBM에 materialize하지 않는 **online softmax 기반 `E0` 변환**을 우선한다.

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
- online reduction 지원 여부
- split-K 또는 partial-state merge 지원 여부

## Online softmax의 정확한 부분합

한 query row의 score 구간 `C`를 처리할 때 다음 상태만 유지한다.

```text
m_C = max score in C
l_C = sum exp(score - m_C)
n_C = sum exp(score - m_C) * value
```

두 구간 `A`, `B`는 전체 score를 다시 보지 않고 병합할 수 있다.

```text
m = max(m_A, m_B)
l = exp(m_A - m) * l_A + exp(m_B - m) * l_B
n = exp(m_A - m) * n_A + exp(m_B - m) * n_B
output = n / l
```

이 재구성은 실수 연산에서 일반 softmax attention과 동일하다. 실제 GPU에서는 reduction 순서가 달라져 rounding 차이가 생길 수 있으므로 `E0` tolerance와 token-level 회귀를 적용한다.

## 구현 순서

1. 기존 score-matrix reference backend를 interface 뒤로 이동
2. `OnlineSoftmaxState`의 reference CPU 또는 단순 CUDA 구현 작성
3. 두 부분합의 merge unit test 작성
4. 검증된 외부 또는 vendor attention backend 연결
5. backend가 online/tiled softmax를 사용하는지 capability와 문서로 확인
6. unsupported shape는 reference로 fallback
7. backend 선택 이유와 score materialization 여부를 trace에 기록
8. output parity 검증

처음부터 universal custom FlashAttention을 작성하지 않는다. 외부 backend가 target shape를 충분히 지원하면 해당 구현을 사용하고, custom kernel은 profiler가 증명한 공백에만 작성한다.

## Split-K와 병렬 merge

긴 sequence에서 K/V 범위를 여러 CTA 또는 partition으로 나눌 수 있다. 각 partition은 `(m, l, n)`을 만들고 최종 reducer가 결합한다.

필수 조건:

- partition 수와 순서가 결과 의미를 바꾸지 않음
- empty 또는 fully masked partition 처리
- `-inf` score와 all-masked row 처리
- accumulator dtype 명시
- workspace 없이 가능한 경로와 workspace 경로 구분

PR 08에서 split-K 최적화 자체는 필수가 아니지만, backend interface는 partial-state merge를 막지 않아야 한다.

## Shape matrix

- sequence 1, 128, 1K, 4K
- batch 1, 2, 4
- MHA와 target GQA shape
- padding 없는 packed 또는 dense input 중 선택한 형식
- target head dimension
- 극단적으로 큰 양수·음수 score
- causal mask의 첫·중간·마지막 row

## 검증

- score-matrix reference와 online softmax parity
- 하나의 구간과 여러 구간 결과 비교
- partition merge 순서를 바꾼 결과 비교
- fully masked row의 정의된 동작
- FP32 accumulator와 target output dtype 확인
- logits와 greedy next token 회귀

## 성능 판단

- attention kernel latency
- end-to-end prefill latency
- TTFT 영향
- workspace와 peak VRAM
- layout conversion 비용
- score matrix materialization bytes
- estimated/observed HBM traffic
- partial-state merge overhead
- fallback 비율

backend 자체는 빨라도 앞뒤 transpose/copy 때문에 전체가 느려질 수 있으므로 trace로 확인한다.

## 비범위

- decode attention
- paged cache
- variable-request continuous batching
- query-aware page pruning
- random-feature 또는 Nyström 근사 attention
- custom universal FlashAttention
- 모델 재학습이 필요한 attention 교체

## 완료 기준

- [ ] reference와 online/optimized backend parity
- [ ] `OnlineSoftmaxState` merge test 통과
- [ ] target prefill shapes 모두 실행
- [ ] optimized path에서 score matrix 전체를 HBM에 materialize하는지 여부가 측정됨
- [ ] unsupported 조합이 안전하게 fallback 또는 명시적 실패
- [ ] baseline 대비 prefill 수치가 raw result로 보존
- [ ] hidden copy/contiguous 비용이 profiler에 표시됨

[← 이전](07-llama-reference-forward.md) | [목차](README.md) | [다음 →](09-single-request-decode.md)