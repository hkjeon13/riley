# PR 12 — Qwen-compatible 모델 호환성

**상태:** Implemented

**선행 조건:** [PR 11](11-sampling-and-generation.md)

**다음:** [PR 13 — Scheduler와 Batching](13-scheduler-and-batching.md)

**검증된 실행 소스:** `842b1cdc68b46bc32ed2e72041c91e62c6e25fc7`

[← 이전](11-sampling-and-generation.md) | [목차](README.md) | [다음 →](13-scheduler-and-batching.md)

## 목적

두 번째 model family인 dense Qwen2를 지원하고, canonical IR과 기존 Llama
execution module을 재사용하여 모델별 runtime 복제가 필요하지 않음을 검증한다.
PR 12의 호환성 계약은 다음 고정 artifact를 기준으로 한다.

- 모델: `Qwen/Qwen2.5-0.5B-Instruct`
- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- dtype: BF16
- golden: `benchmarks/reference/qwen2.5-0.5b-instruct-bf16.json`
- golden SHA-256:
  `42cc7f3fd04098bc4d70836ee9d18dbf919f158a010da3da6fdaa3d9deeceab7`

## 구현 결과

### Cold model boundary

- `model_type=qwen2`와 `Qwen2ForCausalLM`은 bounded strict parser에서만
  선택한다. 알 수 없는 field, 불일치하는 차원, 지원하지 않는 execution
  semantic은 upload나 실행 전에 구조화된 오류로 거절한다.
- Qwen config를 `rustinfer-model-spec-v2` canonical IR로 변환한다. dense Qwen2는
  별도 architecture가 아니라 기존 `ModelArchitecture::Llama` topology를 사용한다.
- source weight 이름을 canonical slot으로 완전 매핑하고, tied LM head는 token
  embedding의 명시적 alias로 표현한다. Q/K/V bias는 각 projection의 optional
  canonical weight로 바인딩한다.
- checkpoint manifest의 config, tokenizer, tokenizer config, safetensors를 SHA-256으로
  검증하며, 고정 Qwen checkpoint는 명시적 1 GiB weight limit 안에서 로드한다.

### 일반화된 execution semantic

- `AttentionBiasSpec`은 query/key/value/output projection별 bias 존재 여부를
  일반 의미로 표현한다. Qwen은 Q/K/V만 `true`이며 Llama의 uniform bias 계약도
  같은 field를 사용한다.
- head dimension, query/KV head 수, RMSNorm epsilon, RoPE theta와 context bound,
  SiLU gated MLP, MLP bias 여부, tied embedding, special token을 canonical IR의
  model-independent field로 유지한다.
- Q/K/V projection 결과에는 공용 in-place BF16 row-bias CUDA primitive를 적용한다.
  prefill, decode, generation은 기존 Llama plan과 execution module을 그대로 재사용한다.
- tokenizer는 공용 ByteLevel BPE engine 위에서 고정 Qwen NFC +
  `Sequence(Split, ByteLevel)` profile을 엄격히 검증한다. 고정 chat template의
  byte 길이와 SHA-256을 확인하고, bounded no-tools renderer를 제공한다.
- 모델 vocabulary 151,936개 중 tokenizer가 주소 지정할 수 있는 151,665개만
  sampling domain에 포함하여 padded tail token이 생성되지 않게 한다.

### Architecture leak 방지

모델 family 분기는 config/tokenizer/weight adapter의 cold boundary에만 있다. hot
execution path는 `ModelSpec`, projection bias, tied-weight alias 같은 semantic spec만
사용하며 모델명이나 `qwen` 문자열로 분기하지 않는다. 이 경계는
`qwen_architecture_boundary` 테스트로 고정했다.

```rust
// 금지되는 hot-path 형태
if model_name.contains("qwen") { /* ... */ }
```

## Semantic/reference 정책

독립 reference fixture는 고정 Transformers eager/BF16 환경에서 영어, 한국어,
Rust 코드 chat 3개를 생성한다. 각 case는 rendered chat과 prompt token ID, masking
전 마지막 logits의 top-10 및 고정 probe, cache-on/cache-off가 일치하는 8개 greedy
token을 기록한다. Rust gate는 raw top-1과 top-10 token 집합을 정확히 비교하고,
sparse logit 값에는 기존 PR 01 최종-logit max-absolute-error bound를 적용한다.
generation token은 cache-on 및 full-prefix cache-off fixture와 모두 정확히 일치해야
하며 greedy 경로는 RNG word를 소비하지 않는다.

정확성 oracle은 performance policy와 의도적으로 분리한다. Transformers eager
fixture는 attention 중간값을 staged BF16으로 반올림하므로, PR 12의 exact gate는
prefill과 decode 모두 기존의 명시적 materialized reference-attention policy를
선택한다. optimized online attention은 허용 오차 기반 성능 경로이며 이 eager
oracle과 bit-exact임을 전제하지 않는다. 이 선택은 테스트가 전달하는 일반
attention preference이며 Qwen 전용 hot-path 분기가 아니다. SmolLM2 optimized
online prefill은 별도의 golden 및 100회 byte-determinism regression gate로 유지한다.

## 진단과 수정 내역

초기 Qwen prefill은 down projection shape(`M=40, N=896, K=4864`)에서
cuBLASLt deterministic plan을 찾지 못했다. 진단 결과 16 MiB workspace가 부족한
것이 아니라, heuristic이 반환한 split-K 후보를 그대로 비결정적 후보로 버리고
있었다.

plan selection은 이제 후보 algorithm의 사본을 `split_k=1`,
`reduction_scheme=NONE`으로 정규화하고 `cublasLtMatmulAlgoCheck`로 해당 layout에서
다시 검증한다. 그 후 재계산된 workspace, 256-byte alignment, numerical flags와
algorithm metadata를 검사한다. Qwen down-projection의 decode `M=1` 및 prefill
`M=30/40/46` shape가 production과 같은 16 MiB cap으로 GPU correctness gate를
통과했다. 결정성 계약을 완화하거나 workspace 기본값을 늘리지는 않았다.

후속 진단에서는 optimized online attention과 staged-BF16 eager oracle 사이의
허용 가능한 numerical tie가 exact token 비교에 섞여 있음을 분리했다. 이에 따라
exact compatibility gate만 위의 reference-attention policy를 명시적으로 선택했고,
일반 optimized 경로의 regression gate는 그대로 유지했다. 실패했던 진단 이력은
성공 로그로 덮어쓰지 않고 원격 provenance에 lineage로 기록했다.

## 검증 결과

모든 CUDA build, checkpoint load, inference 및 sanitizer 실행은 로컬이 아닌
`server-4096`의 RTX 4090에서 수행했다.

| Gate | 결과 |
|---|---|
| workspace all-features Clippy (`-D warnings`) | 통과 |
| workspace all-targets/all-features tests | 184 passed, 0 failed, 55 ignored |
| workspace all-features doctests | 13 passed, 0 failed |
| 고정 Qwen metadata/full checkpoint 및 Llama→Qwen 순차 load | 3 passed |
| BF16 row-bias GPU correctness/error contract | 2 passed |
| odd, SmolLM2, Qwen cuBLASLt GEMM shape | 20 passed |
| Qwen golden logits/decode/generation | 3 cases × 8 tokens; direct decode와 generation exact |
| 같은 process의 SmolLM2→Qwen CUDA owner lifecycle | 각 close 뒤 allocation accounting 0 |
| SmolLM2 generation regression | 31 cases exact |
| SmolLM2 reference forward regression | golden 통과 |
| SmolLM2 optimized online prefill regression | golden 통과, 100회 byte exact |
| row-bias Compute Sanitizer memcheck | 0 errors, 0 leaked bytes |
| row-bias Compute Sanitizer racecheck | 0 hazards, 0 errors, 0 warnings |

## 원격 provenance와 증거

- authoritative source commit:
  `842b1cdc68b46bc32ed2e72041c91e62c6e25fc7`
- 원격 evidence root:
  `/home/psyche/rustinfer-artifacts/pr12/842b1cd/full`
- read-only source archive:
  `/home/psyche/rustinfer-artifacts/pr12/842b1cd/full/source.tar.gz`
- source archive SHA-256:
  `cc3cd6afeb59d3303cb682b8ab1435a36675d6bfd2b5bcc7eff9569e89076a70`
- 원격 `SHA256SUMS` SHA-256:
  `863090e529e8f4a860ce1fad4008f56915f2097aef132044aa9305b284717b63`
- container: `rustinfer-native-cuda:pr04-c6c93e2`
  (`sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`)
- GPU/runtime: NVIDIA GeForce RTX 4090, compute capability 8.9, driver
  580.173.02, CUDA 12.8.1, nvcc 12.8.93
- 실행 격리: source read-only mount, `--network none`, Cargo `--locked --offline`
- version-controlled evidence index:
  [PR 12 Qwen compatibility evidence](../benchmarks/results/20260825T123333Z-rustinfer-qwen-pr12-run001/README.md)

## 비범위와 제한

- 지원 범위는 위에 고정한 dense text Qwen2.5-0.5B-Instruct artifact profile이다.
- Qwen-VL, MoE, Qwen3, quantized checkpoint, remote code, sliding/local attention,
  RoPE scaling 또는 multi-axis RoPE는 지원하지 않으며 cold parse boundary에서
  명시적으로 거절한다.
- tools/tool-call chat rendering은 구현 범위가 아니다. 고정 no-tools template만
  지원한다.
- exact eager gate의 reference-attention 선택은 correctness oracle용이다. optimized
  attention의 bit-exact 동등성을 주장하지 않으며 그 경로는 tolerance 및 결정성
  gate로 관리한다.
- 이 PR은 Qwen 성능 benchmark나 throughput 목표를 추가하지 않는다.

## Rollback

런타임 feature flag로 Qwen만 끄는 구조가 아니므로 rollback은 PR 12 commit 범위를
하나의 단위로 revert한다. Qwen adapter, canonical bias/alias semantic, row-bias
primitive, golden과 gate를 함께 되돌려 부분 상태를 남기지 않는다. rollback 후에는
PR 11의 고정 SmolLM2 generation, reference forward, online prefill gate를 다시
실행해 기존 Llama 경로가 복원됐는지 확인한다. Qwen artifact나 golden checksum
drift, 새 CUDA/cuBLASLt 환경에서 deterministic-plan 또는 sanitizer gate 실패가
재현되면 rollout을 중단하고 이 절차를 적용한다.

## 완료 기준

- [x] 두 model family가 동일 execution modules를 재사용
- [x] 모델명 기반 hot-path 분기 없음
- [x] Qwen golden logits/token 일치
- [x] Llama benchmark와 correctness 회귀 없음
- [x] 추가된 IR field가 일반적인 의미로 문서화됨

[← 이전](11-sampling-and-generation.md) | [목차](README.md) | [다음 →](13-scheduler-and-batching.md)
