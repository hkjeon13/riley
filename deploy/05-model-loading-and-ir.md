# PR 05 — Hugging Face 모델 로딩과 Canonical IR

**상태:** Planned  
**선행 조건:** [PR 04](04-tensor-and-memory.md)  
**다음:** [PR 06 — 핵심 Primitive](06-core-primitives.md)

[← 이전](04-tensor-and-memory.md) | [목차](README.md) | [다음 →](06-core-primitives.md)

## 목적

HF `config.json`, `tokenizer.json`과 `safetensors`를 **Rust가 직접 읽어**, 모델별 Python class가 아닌 canonical execution description으로 변환한다.

이번 PR은 GPU forward를 하지 않는다. Python/PyTorch/Transformers는 golden fixture 또는 optional offline converter에는 사용할 수 있지만 model loading runtime에는 포함하지 않는다.

## Runtime dependency 원칙

Production loader는 다음 입력을 Python 없이 처리한다.

```text
config.json
model.safetensors 또는 shard + index
serializer provenance manifest
Tokenizer-compatible JSON artifacts
```

금지:

- Python interpreter 실행
- `AutoModel.from_pretrained`를 server 시작 과정에서 호출
- `trust_remote_code`
- pickle과 Python class serialization
- loader 실패 시 Python Transformers fallback

Optional Python converter가 필요해도 출력은 Rust가 직접 검증할 수 있는 JSON/safetensors여야 한다.

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

Rust의 JSON parser와 명시적 typed config를 사용한다.

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

Rust-native loader가 다음을 담당한다.

- safetensors header 검증
- tensor name → canonical weight slot mapping
- shape/dtype 검증
- shard index 지원 여부 결정
- memory-mapped host view 또는 bounded read
- tied weight alias 표현
- source/transform provenance 확인
- duplicate/missing tensor 탐지

### Tokenizer boundary

Tokenizer 구현체를 runtime core와 분리하는 Rust trait를 정의한다.

```rust
trait Tokenizer {
    fn encode(...);
    fn decode(...);
}
```

첫 backend는 Rust-native tokenizer library를 사용한다. Python tokenizer service 또는 subprocess를 사용하지 않는다.

## Optional offline converter

HF 원본 checkpoint를 실행 layout으로 미리 변환하는 도구는 허용한다.

예:

- Q/K/V weight packing
- gate/up packing
- tensor rename
- weight transpose
- quantization/rotation/low-rank artifact 생성

Python으로 구현할 수 있지만 다음 계약을 지킨다.

1. runtime dependency가 아니다.
2. 원본 checkpoint도 가능한 범위에서 Rust가 직접 읽을 수 있어야 한다.
3. 변환 결과는 safetensors + JSON manifest다.
4. source model revision, converter revision, transform parameters를 기록한다.
5. 변환 artifact의 shape/dtype/checksum을 Rust가 재검증한다.

Manifest 예시:

```json
{
  "format": "rustinfer-checkpoint-v1",
  "source_model": "org/model",
  "source_revision": "immutable-revision",
  "converter_revision": "git-sha",
  "transforms": ["packed_qkv"],
  "dtype": "bf16"
}
```

## Snapshot 테스트

- config → IR snapshot
- weight name mapping snapshot
- missing/extra/mismatched tensor 오류
- tied embeddings
- unsupported RoPE variant 실패
- tokenizer round trip
- provenance manifest 검증
- Python converter가 만든 artifact를 Python 없는 test process에서 load
- Python executable이 PATH에 없어도 동일 결과

## 비범위

- Qwen-specific mapping
- quantization runtime
- remote model download
- arbitrary `trust_remote_code`
- multimodal
- GPU upload 최적화
- Python model execution fallback

## 완료 기준

- [ ] 한 checkpoint를 deterministic IR로 변환
- [ ] 모든 필수 weight가 정확한 shape로 binding됨
- [ ] Rust가 config, safetensors, tokenizer artifact를 직접 처리
- [ ] unsupported architecture가 조용히 오동작하지 않고 실패
- [ ] tokenizer round trip fixture 통과
- [ ] optional converted artifact에 provenance가 존재
- [ ] Python 없는 환경에서 parsing/loading test 통과
- [ ] 모델 parsing과 runtime loading에 Python/PyTorch/Transformers가 필요하지 않음

[← 이전](04-tensor-and-memory.md) | [목차](README.md) | [다음 →](06-core-primitives.md)
