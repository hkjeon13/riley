# PR 01 — 기준선과 재현성 계약

**상태:** Active

**선행 조건:** [PR 00](00-pr-contract.md)  
**다음:** [PR 02 — Workspace와 CI](02-workspace-and-ci.md)

[← 이전](00-pr-contract.md) | [목차](README.md) | [다음 →](02-workspace-and-ci.md)

## 목적

코드를 쓰기 전에 비교 대상, 하드웨어, 모델, 입력 corpus, 정확도 기준과 성능 지표를 고정한다. 이후 추가되는 exact 변환, 분포 보존 알고리즘과 근사 최적화가 같은 기준선에서 평가되도록 결과 schema도 먼저 정의한다.

이 단계에서는 Python/PyTorch/Transformers를 **외부 reference lane**으로 사용할 수 있다. 그러나 `rustinfer` benchmark 대상은 별도의 Python-free production lane에서 실행해 reference 환경과 운영 dependency를 혼동하지 않는다.

## 핵심 결정

### 1. Primary GPU 한 종류

첫 release까지 primary GPU는 하나만 둔다.

- Volta/V100이면 FP16을 기본으로 한다.
- Ampere 이상이면 BF16을 우선 검토한다.
- 다른 세대는 correctness smoke만 수행하고 성능 목표에서 제외한다.

결정 결과는 `benchmarks/environment.md`에 기록한다.

### 2. Primary checkpoint 한 개

처음에는 작은 Llama-compatible checkpoint를 선택한다.

요건:

- 공개적으로 재현 가능
- `safetensors` 제공
- 단일 GPU에 여유 있게 적재
- tokenizer와 config가 표준 HF 형식
- 라이선스가 개발·benchmark 목적에 적합

Qwen-compatible checkpoint는 PR 12에서 추가한다.

### 3. Baseline engine

최소 비교 대상:

- Hugging Face Transformers eager 또는 SDPA reference
- vLLM
- 가능하면 SGLang 또는 TensorRT-LLM 중 하나

동일 model, dtype, sampling 설정을 강제한다.

## Reference lane과 Production lane

### Python reference lane

허용 dependency:

- Python
- PyTorch
- Transformers
- NumPy/SciPy

목적:

- golden logits와 token 생성
- hidden-state fixture
- checkpoint와 tokenizer 분석
- baseline engine 실행

### rustinfer production lane

허용 dependency:

- Rust release binary/library
- native CUDA library
- CUDA driver/runtime와 명시된 NVIDIA library
- model/tokenizer artifact

금지:

- Python interpreter
- PyTorch/Transformers import
- Python subprocess fallback
- Triton Python JIT

두 lane은 환경 manifest와 명령을 분리하고 결과만 공통 schema로 비교한다.

## Benchmark matrix

| 차원 | 초기 값 |
|---|---|
| Concurrency | 1, 2, 4, 8 |
| Prompt tokens | 128, 1K, 4K |
| Output tokens | 32, 128 |
| Sampling | greedy 우선, 이후 고정 seed sampling |
| 상태 | cold load, warm steady-state 분리 |
| Approximation | 초기에는 `disabled` |
| Semantic class | reference 또는 `E0`만 초기 release 대상 |

기록 지표:

- model load time
- TTFT
- median/p95 TPOT 또는 ITL
- end-to-end latency
- tokens/s
- CPU utilization
- GPU utilization
- peak VRAM
- failure count

## 결과 schema의 공통 필드

모든 benchmark row 또는 run metadata에는 다음을 기록한다.

```yaml
semantic_class: reference | E0 | E1 | A1 | M1
implementation_id: string
reference_implementation: string
runtime_dependency_class: python-reference | native-production
approximation_enabled: bool
error_budget: null | number
seed: null | integer
warm_state: cold | warm
model_revision: string
engine_revision: string
```

최적화별 추가 필드는 nullable하게 둔다.

```yaml
speculative:
  draft_model: null
  lookahead: null
  acceptance_rate: null
  accepted_tokens_per_verify: null
  target_calls_per_output_token: null

sparse_attention:
  selected_pages: null
  total_pages: null
  omitted_mass_bound: null

quantization:
  weight_format: null
  activation_format: null
  kv_format: null
  calibration_revision: null
```

초기 baseline에서는 이 필드들이 비어 있어도 되지만 schema는 변경 이력을 남긴다.

## Golden correctness corpus

최소 다음 입력을 포함한다.

- 짧은 단문
- 한국어와 영어
- 숫자·기호·코드 조각
- 긴 반복 문맥
- 최대 길이 부근의 경계 입력
- 빈 문자열 또는 최소 token 입력
- EOS가 빠르게 발생하는 입력

저장할 reference:

- tokenized IDs
- 첫 layer 또는 선택 layer의 hidden-state checksum
- 최종 logits 일부와 통계
- greedy generated token IDs
- KV cache on/off 결과
- sampling에 사용하는 변환 후 logits 또는 log-probability fixture
- request별 RNG 초기 상태와 알고리즘 ID

전체 대형 tensor를 Git에 넣지 말고, 작은 fixture와 생성 스크립트를 둔다. Python object나 pickle이 아니라 JSON, safetensors, NumPy의 명시적 export 또는 checksum 형태를 사용한다.

## 의미 보존 등급별 정확도 기준

단일 scalar tolerance만 두지 않는다.

### Reference와 `E0`

- FP32 reference 대비 max/mean absolute error
- relative error
- cosine similarity
- top-k token set 일치
- greedy token exact match
- 여러 step 후 divergence 시점
- reduction partition 또는 merge 순서를 바꾼 결과

### `E1`

- greedy mode에서는 exact token match
- request별 RNG 격리와 snapshot/restore 재현
- acceptance probability와 residual sampling unit test
- 작은 categorical distribution의 exhaustive probability test
- 대규모 sampling의 frequency, total variation 또는 적절한 goodness-of-fit 통계
- output token당 draft/target 호출 수

고정 seed의 token sequence가 다른 구현과 반드시 같다는 뜻과 목표 분포가 같다는 뜻을 구분해서 보고한다.

### `A1`

- error budget 또는 approximation parameter 공개
- exact reference 대비 logits/token/quality 차이
- error-quality-latency curve
- exact fallback 결과
- approximation 사용률과 fallback 비율
- task 또는 corpus 품질 지표

KV page pruning처럼 omitted probability mass만 제한하는 경우, 그 수치를 attention output의 절대 오차와 동일하다고 표현하지 않는다. output norm bound가 필요하면 value norm에 대한 별도 상한을 포함한다.

허용 오차와 통계 threshold는 dtype·연산·등급별로 사전에 기록한다.

## 추가 최적화 지표

### Speculative decoding

- draft latency
- verification latency
- draft length
- acceptance rate
- accepted tokens per verification
- target model calls per output token
- rejected suffix 길이
- rollback count

### Query-aware page selection

- total/selected KV pages
- page metadata bytes
- page upper-bound 계산 시간
- omitted softmax mass bound
- exact fallback 비율
- long-context TPOT와 품질 변화

### 저정밀·등가변환

- transform-only full precision parity
- quantization 전후 error
- rotation/scaling runtime overhead
- weight/KV bytes
- GEMM throughput과 end-to-end latency

## 산출물

```text
benchmarks/
├── environment.md
├── matrix.yaml
├── prompts.jsonl
├── reference/
├── scripts/
├── schemas/
│   └── result.schema.json
└── results/.gitkeep

tools/python/reference/
└── Python reference 생성 도구
```

## 비범위

- rustinfer 구현
- custom CUDA kernel
- 최적화 주장
- multi-GPU
- quantization 실행
- speculative decoding 실행
- approximate attention 실행
- Python을 production runtime에 포함

## 완료 기준

- [ ] primary GPU, dtype, checkpoint 고정
- [ ] Python reference lane의 dependency와 실행 명령 고정
- [ ] Python-free rustinfer production lane의 dependency와 실행 명령 고정
- [ ] baseline 실행 명령 재현 가능
- [ ] benchmark matrix 파일화
- [ ] golden corpus와 reference 생성 절차 문서화
- [ ] 의미 보존 등급과 runtime dependency class를 포함한 raw result schema 정의
- [ ] RNG 알고리즘과 seed 기록 방식 정의
- [ ] 같은 환경에서 baseline 반복 실행 편차 확인

**중단 조건:** benchmark 결과가 반복 실행마다 크게 흔들리거나 두 lane의 model/dtype/sampling 조건이 다르면 다음 PR로 가지 않는다.

[← 이전](00-pr-contract.md) | [목차](README.md) | [다음 →](02-workspace-and-ci.md)
