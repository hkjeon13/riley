# PR 12 — Qwen-compatible 모델 호환성

**상태:** Planned  
**선행 조건:** [PR 11](11-sampling-and-generation.md)  
**다음:** [PR 13 — Scheduler와 Batching](13-scheduler-and-batching.md)

[← 이전](11-sampling-and-generation.md) | [목차](README.md) | [다음 →](13-scheduler-and-batching.md)

## 목적

두 번째 모델 family를 지원하여 canonical IR과 kernel registry가 실제로 모델별 코드 복제를 줄이는지 검증한다.

## 원칙

Qwen 전용 전체 runtime을 복사하지 않는다. 차이는 config adapter와 명시적인 IR parameter로 표현한다.

검토할 차이:

- attention projection bias
- head dimension과 KV head 수
- norm epsilon
- RoPE theta/scaling
- activation과 MLP packing
- tied embeddings
- tokenizer/chat template
- sliding/local attention 여부

## 작업 순서

1. Qwen config를 canonical IR로 변환
2. checkpoint weight mapping 추가
3. 기존 primitive capability로 표현 가능한지 확인
4. 부족한 의미만 IR에 추가
5. Llama path에 regression이 없는지 확인
6. golden logits와 generation 비교

## Architecture leak 검사

다음 패턴이 생기면 설계를 재검토한다.

```rust
if model_name.contains("qwen") { ... }
```

허용되는 분기는 parser/adapter 단계이며, hot execution path는 semantic spec에 따라 분기해야 한다.

## 테스트

- IR snapshot Llama vs Qwen 비교
- weight map completeness
- logits parity
- greedy generation parity
- 두 모델을 같은 process에서 순차 load/unload
- tokenizer 특수 token
- unsupported variant 명시적 오류

## 비범위

- Qwen-VL
- MoE variant
- quantized checkpoint
- remote code
- 모든 Qwen 세대 지원

## 완료 기준

- [ ] 두 model family가 동일 execution modules를 재사용
- [ ] 모델명 기반 hot-path 분기 없음
- [ ] Qwen golden logits/token 일치
- [ ] Llama benchmark와 correctness 회귀 없음
- [ ] 추가된 IR field가 일반적인 의미로 문서화됨

[← 이전](11-sampling-and-generation.md) | [목차](README.md) | [다음 →](13-scheduler-and-batching.md)
