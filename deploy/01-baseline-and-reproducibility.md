# PR 01 — 기준선과 재현성 계약

**상태:** Planned  
**선행 조건:** [PR 00](00-pr-contract.md)  
**다음:** [PR 02 — Workspace와 CI](02-workspace-and-ci.md)

[← 이전](00-pr-contract.md) | [목차](README.md) | [다음 →](02-workspace-and-ci.md)

## 목적

코드를 쓰기 전에 비교 대상, 하드웨어, 모델, 입력 corpus, 정확도 기준과 성능 지표를 고정한다.

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

## Benchmark matrix

| 차원 | 초기 값 |
|---|---|
| Concurrency | 1, 2, 4, 8 |
| Prompt tokens | 128, 1K, 4K |
| Output tokens | 32, 128 |
| Sampling | greedy 우선, 이후 고정 seed sampling |
| 상태 | cold load, warm steady-state 분리 |

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

전체 대형 tensor를 Git에 넣지 말고, 작은 fixture와 생성 스크립트를 둔다.

## 정확도 기준

단일 scalar tolerance만 두지 않는다.

- FP32 reference 대비 max/mean absolute error
- relative error
- cosine similarity
- top-k token set 일치
- greedy token exact match
- 여러 step 후 divergence 시점

허용 오차는 dtype과 연산별로 사전에 기록한다.

## 산출물

```text
benchmarks/
├── environment.md
├── matrix.yaml
├── prompts.jsonl
├── reference/
├── scripts/
└── results/.gitkeep
```

## 비범위

- rustinfer 구현
- custom CUDA kernel
- 최적화 주장
- multi-GPU
- quantization

## 완료 기준

- [ ] primary GPU, dtype, checkpoint 고정
- [ ] baseline 실행 명령 재현 가능
- [ ] benchmark matrix 파일화
- [ ] golden corpus와 reference 생성 절차 문서화
- [ ] raw 결과 schema 정의
- [ ] 같은 환경에서 baseline 반복 실행 편차 확인

**중단 조건:** benchmark 결과가 반복 실행마다 크게 흔들리면 다음 PR로 가지 않고 환경 안정화부터 수행한다.

[← 이전](00-pr-contract.md) | [목차](README.md) | [다음 →](02-workspace-and-ci.md)
