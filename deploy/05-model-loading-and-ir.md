# PR 05 — Hugging Face 모델 로딩과 Canonical IR

**상태:** Planned  
**선행 조건:** [PR 04](04-tensor-and-memory.md)  
**다음:** [PR 06 — 핵심 Primitive](06-core-primitives.md)

[← 이전](04-tensor-and-memory.md) | [목차](README.md) | [다음 →](06-core-primitives.md)

## 목적

HF `config.json`, tokenizer metadata와 `safetensors`를 읽어 **모델별 Python class가 아닌 canonical execution description**으로 변환한다.

이번 PR은 GPU forward를 하지 않는다.

## 초기 지원 범위

- Llama-compatible decoder-only config 한 종류
- dense model
- single GPU
- FP16 또는 BF16
- standard RoPE
- MHA/MQA/GQA
- gated MLP

## 구성 요소

### Config parser

검증할 항목:

- hidden size
- layer count
- attention heads와 KV heads
- head dimension
- intermediate size
- vocabulary size
- norm epsilon
- activation
- RoPE theta/scaling
- tie word embeddings
- bias 유무

알 수 없는 필드는 보존하거나 경고하되, 의미에 영향을 주는 unsupported 값은 즉시 실패한다.

### Canonical IR

최소 구조:

```rust
ModelSpec
EmbeddingSpec
DecoderBlockSpec
NormSpec
AttentionSpec
RopeSpec
GatedMlpSpec
LmHeadSpec
WeightBinding
```

모델 이름은 diagnostic metadata로 남기고 실행 분기 기준으로 사용하지 않는다.

### Weight loader

- safetensors header 검증
- tensor name → canonical weight slot mapping
- shape/dtype 검증
- shard index 지원 여부 결정
- memory-mapped host view 또는 bounded read
- tied weight alias 표현

### Tokenizer boundary

Tokenizer 구현체를 runtime core와 분리하는 trait를 정의한다.

```rust
trait Tokenizer {
    fn encode(...);
    fn decode(...);
}
```

이 PR에서는 한 tokenizer backend만 연결해도 되지만 engine 내부 타입과 직접 결합하지 않는다.

## Snapshot 테스트

- config → IR snapshot
- weight name mapping snapshot
- missing/extra/mismatched tensor 오류
- tied embeddings
- unsupported RoPE variant 실패
- tokenizer round trip

## 비범위

- Qwen-specific mapping
- quantization
- remote model download
- arbitrary `trust_remote_code`
- multimodal
- GPU upload 최적화

## 완료 기준

- [ ] 한 checkpoint를 deterministic IR로 변환
- [ ] 모든 필수 weight가 정확한 shape로 binding됨
- [ ] unsupported architecture가 조용히 오동작하지 않고 실패
- [ ] tokenizer round trip fixture 통과
- [ ] 모델 parsing에 Python runtime이 필요하지 않음

[← 이전](04-tensor-and-memory.md) | [목차](README.md) | [다음 →](06-core-primitives.md)
