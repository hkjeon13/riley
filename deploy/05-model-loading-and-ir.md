# PR 05 — Hugging Face 모델 로딩과 Canonical IR

**상태:** Active
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

## 구현 결과

### 문제와 범위

PR 04까지는 tensor와 memory ownership만 존재해, Hugging Face artifact를 실행
계획으로 바꾸는 신뢰 경계가 없었다. 이 PR은 다음 세 개의 독립적으로 롤백 가능한
커밋으로 그 경계를 완성했다.

1. `39fc065` — strict Llama config parser, dependency allowlist, versioned canonical IR
2. `31eebde` — bounded safetensors/shard-index parser, provenance, canonical weight binding
3. `f32ae98` — Rust-native tokenizer, aggregate verified session, Python-free CI gate

지원 범위는 pinned `HuggingFaceTB/SmolLM2-135M`이 사용하는 dense Llama BF16/FP16,
standard RoPE, MHA/MQA/GQA, gated SiLU MLP와 single/sharded safetensors다. Remote
download, GPU upload/forward, Qwen mapping, quantization과 arbitrary tokenizer pipeline은
계속 비범위다. 수학적 최적화가 아니므로 의미 보존 등급은 적용되지 않는다.

### 설계 결정

- 모든 JSON은 중첩 깊이까지 duplicate key를 거부하며, 실행 의미에 영향을 주는
  unknown/unsupported 값은 fail closed한다.
- `VerifiedArtifactSession`이 canonical root와 manifest를 한 번 고정하고 각 payload를
  정확히 한 번 `bounded read → length/SHA-256 → parse` 순서로 소비한다. 종료 시 실제
  소비 집합과 manifest 집합의 exact union을 한 번 검증한다.
- Weight loader는 single file과 shard index를 모두 지원하고, manifest를 layout
  authority로 사용한다. 모든 tensor name/shape/dtype를 canonical slot에 바인딩하며 tied
  LM head는 embedding storage의 borrow-anchored alias로 표현한다.
- Tokenizer는 범용 artifact를 묵시적으로 수용하지 않고 SmolLM2의 정확한
  `Sequence(Digits, ByteLevel) → BPE → ByteLevel decoder` 형상만 지원하는 first-party
  Rust backend다. 동작은 Hugging Face Tokenizers의 공식
  [ByteLevel](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs),
  [Digits](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/digits.rs),
  [BPE merge](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/word.rs)
  구현과 pinned reference token ID로 대조했다.
- Config dimension/layer, file/header/count/weight bytes, tokenizer vocab/merge/added-token과
  added-token trie node 수를 사전에 제한한다. 현재 portable reader는 trusted immutable
  checkpoint directory를 전제로 하며 최종 symlink를 거부한다.
- Production model crate에는 subprocess, Python, PyTorch, Transformers, Triton 또는
  runtime fallback이 없다. `ci/verify_python_free_model_loading.sh`가 isolated target에서
  binary dependency를 검사하고 빈 executable `PATH`로 직접 실행한다.

### 검증과 결과

로컬에서는 실제 model/tokenizer/CUDA/GPU를 실행하지 않았다. CPU-only synthetic
fixture와 static gate로 다음을 통과했다.

- `cargo fmt --all -- --check`
- `python3 ci/check_workspace_boundaries.py --locked`
- workspace all-target Clippy `-D warnings`
- workspace `--no-default-features` unit/integration/doctest 전체
- workspace documentation build
- Python-free isolated synthetic aggregate load

실제 artifact 검증은 사용자 요청에 따라 `server-4096`에서만 수행했다. 구현 snapshot
`f32ae980be61384a34c74dac27e6ad8f600cfa58`의 clean `git archive`를 Rust 1.85.0
container에서 빌드했고, pinned source revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`를 다음 고정 assertion으로 검증했다.

```text
source archive sha256: e272e1073a406c2ae30f8622a58c7c6899d9648a12269cbdf45f0e4a39a330f4
config.json:           704 bytes / 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer.json:        2,104,556 bytes / 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
model.safetensors:     269,060,552 bytes / 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
canonical model:       30 layers / BF16 / 272 physical tensors / tied LM-head alias
tokenizer parity:      English, Korean, number-symbol-code reference IDs exact
normal real load:      1 passed, 0 failed
empty-PATH real load:  1 passed, 0 failed
```

검증 host는 RTX 4090(compute capability 8.9), driver `580.173.02`이고 container image는
`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`다.
이번 PR은 GPU forward가 비범위라 GPU device는 사용하지 않았다. 최초 CUDA primitive
실행은 PR 06에서 `--gpus all`로 별도 검증한다.

외부 evidence는 다음 append-only 위치에 보존했다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr05/f32ae980be61384a34c74dac27e6ad8f600cfa58/
SHA256SUMS sha256: 380fd01f29e737b29c3be49e5204e807b378cda848486d73c090dc1ce99a0781
real-checkpoint.log:            3a9c88b6e7b4820b23bd0965c584830df0069d2e92c968d12d97ec8680819266
real-checkpoint-empty-path.log: 03027f840bc9da9ae8c01ea6f3f21585f1d1f3c6fadbbb14bf46cfb912f523ab
checkpoint-sha256.log:          0f4c2e1092ab618988df20c505b5836559727f819dc44d635cf46bd244148f9e
```

### 롤백

Runtime과 server는 아직 `LoadedModel`을 호출하지 않으므로 PR 04 base로 되돌려도 기존
실행 경로가 바뀌지 않는다. 세 구현 커밋은 dependency/config IR, physical checkpoint,
aggregate tokenizer 경계 순으로 분리되어 문제가 난 계층까지만 revert할 수 있다.

### PR 크기 예외

PR 04 base `9ba258f` 대비 검증 snapshot `f32ae98`은 `+7,913/-25`줄이다. 분류하면
production/build `+5,836/-2`, tests/fixtures `+1,178`, CI/docs `+725/-23`, lockfile
`+174`다. 권장 production diff를 넘지만 이 중 tokenizer 1,434줄에는 Unicode 16 Mark
generated table과 원문 license가 포함된다. 나머지도 하나의 질문—untrusted HF
artifact가 checksum-verified canonical execution description과 lifetime-safe weight
view로 바뀌었음을 어떻게 증명하는가—에 속하는 parser, provenance session, cross-file
validation과 Python-free gate다. 중간에 checksum 없는 parser나 tokenizer/weight가
서로 다른 manifest를 신뢰하는 상태를 남기지 않도록 세 review commit으로 나누되 한
interface 변화로 함께 롤백하는 PR 00 예외를 적용한다.

## 비범위

- Qwen-specific mapping
- quantization runtime
- remote model download
- arbitrary `trust_remote_code`
- multimodal
- GPU upload 최적화
- Python model execution fallback

## 완료 기준

- [x] 한 checkpoint를 deterministic IR로 변환
- [x] 모든 필수 weight가 정확한 shape로 binding됨
- [x] Rust가 config, safetensors, tokenizer artifact를 직접 처리
- [x] unsupported architecture가 조용히 오동작하지 않고 실패
- [x] tokenizer round trip fixture 통과
- [x] optional converted artifact에 provenance가 존재
- [x] Python 없는 환경에서 parsing/loading test 통과
- [x] 모델 parsing과 runtime loading에 Python/PyTorch/Transformers가 필요하지 않음

구현 gate는 통과했다. 이 문서는 선행 PR과 함께 merge되기 전까지 `Active`, merge
후 `Complete`로 전환한다.

[← 이전](04-tensor-and-memory.md) | [목차](README.md) | [다음 →](06-core-primitives.md)
