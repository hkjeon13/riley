# W4-A Qwen2.5-7B 장문 정확성 재현

2026-11-19 KST에 `aws-a100`의 물리 GPU 7에서 Qwen2.5-7B-Instruct로 기존 18개 프롬프트를 재현했다. Riley가 반환한 프롬프트 ID는 667–6,082개였고, Hugging Face와 vLLM에 그 ID를 그대로 공급했다. vLLM 응답에서도 동일 입력 ID를 확인했다. 요청별 생성 상한은 32토큰, Riley 서빙 문맥 상한은 8,192토큰이었다. [W4 계획](04-correctness-and-schema.md) · [과거 텍스트 비교](../serving-validation.md)

Riley는 코드 리비전 `f7c956d3c`의 CUDA 빌드와 `native-bf16-varlen-v1`을 사용했다. 같은 빌드에서 W4-B API 계약도 검증했다. Hugging Face는 Torch 2.13.0+cu129, Transformers 5.22.1, BF16 SDPA로 생성했다. vLLM 0.27.1은 BF16, FlashAttention 2 및 컴파일·CUDA Graph 경로에서 순차 요청 18건을 실행했다. [Riley 실행 receipt](../../benchmarks/evidence/20261119-w4-long/riley/receipt.json) · [HF 참조 결과](../../benchmarks/evidence/20261119-w4-long/hf/reference.json) · [vLLM 실행 receipt](../../benchmarks/evidence/20261119-w4-long/vllm/receipt.json) · [W4-B 검증](04-structured-output-result.md)

## 결과

| 검증 | 일치 |
|---|---:|
| HF tokenizer와 Riley의 프롬프트 ID | 18/18 |
| Riley C8에서 C1과 생성 ID·텍스트·종료 사유 일치 | 54/54 |
| Riley C32에서 C1과 생성 ID·텍스트·종료 사유 일치 | 54/54 |
| Riley와 HF SDPA의 첫 생성 ID | 17/18 |
| Riley와 HF SDPA의 전체 생성 ID | 13/18 |
| Riley와 HF SDPA의 텍스트 | 13/18 |
| vLLM과 HF SDPA의 첫 생성 ID | 17/18 |
| vLLM과 HF SDPA의 전체 생성 ID·텍스트 | 각각 16/18 |
| Riley와 vLLM의 전체 생성 ID·텍스트 | 각각 13/18 |

Riley의 각 동시성에서 같은 18개 프롬프트를 세 번 반복했다. 세 엔진에서 18건 모두 32토큰 상한에서 끝났다. 차이는 생성 중에 발생했으며 EOS 제거 방식은 이 결과에 영향을 주지 않았다. [케이스별 교차 검산](../../benchmarks/evidence/20261119-w4-long/summary.json)

## 최초 분기

Riley는 logprobs 요청에서도 18건 모두 일반 요청과 같은 ID를 생성했다. HF는 같은 프롬프트 ID와 최초 분기 직전까지의 공통 토큰을 캐시 디코딩으로 재생했고, 재생한 SDPA가 원래 생성 때의 점수와 토큰 선택을 재현했다. 표의 위치는 생성 시작을 0으로 센 값이다.

| 사례 | 입력 ID 수 | 분기 위치 | Riley ID | HF SDPA ID | Riley 상위 logprob 차이 | HF SDPA 상위 logit 차이 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 681 | 2 | 6109 | 56589 | 0 | 0.125 |
| 3 | 3,301 | 0 | 264 | 5257 | 0 | 0.125 |
| 6 | 667 | 14 | 1741 | 2670 | 0 | 0.125 |
| 12 | 669 | 9 | 34235 | 13638 | 0.125 | 0 |
| 14 | 2,196 | 9 | 11931 | 38973 | 0 | 0.125 |

사례 0의 동일 prefix에서 HF attention을 eager로 바꾸면 SDPA에서 0.125 차이였던 두 ID의 점수가 19.75로 같아졌다. Riley와 HF SDPA가 갈라진 다섯 위치 모두 각 경로의 상위 두 점수가 0.125 이내였다. Riley 쪽 수치는 공개 API가 보고한 logprob 간격이고 전체 vocabulary의 raw logit은 수집하지 않았다. 다섯 사례의 최초 분기 위치와 attention backend에 따른 점수 변화를 확인했다. 최초로 달라지는 Riley 내부 층·연산과 KV 위치·mask의 영향은 추가 비교가 필요하다.

다국어·코드·수치·경계 길이 holdout과 GSM8K 전수 검증은 이번 표에 포함되지 않았다. 이 결과로 참조 exact 범위를 확대하거나 수치 동작을 변경하지 않는다.

vLLM의 최초 참조 시도는 시작 중 컴파일러가 실행 환경에서 `ninja`를 찾지 못해 종료됐다. 기존 vLLM 가상환경의 `ninja`를 이번 실행의 PATH에 추가하고 별도 컴파일 캐시를 사용해 18건을 완료했다. 참조 비교는 vLLM의 강제 prefix 점수나 동시성별 불변성까지 검증하지 않았다. 각 단계 종료 후 GPU 7은 동일 홀더의 20GiB 대기 상태로 돌아갔다. [최초 실패 receipt](../../benchmarks/evidence/20261119-w4-long/vllm-initial/receipt.json) · [최종 예약 상태](../../benchmarks/evidence/20261119-w4-long/vllm/idle-after.json)

- [Riley의 원본 프롬프트 ID·생성 ID·logprobs 및 반복 결과 (gzip)](../../benchmarks/evidence/20261119-w4-long/riley/capture.json.gz)
- [HF의 요청별 토큰·최초 분기·SDPA/eager 점수](../../benchmarks/evidence/20261119-w4-long/hf/reference.json)
- [동일 입력 ID를 확인한 vLLM 원본 응답 (gzip)](../../benchmarks/evidence/20261119-w4-long/vllm/results.json.gz)
- [캡처 및 참조 도구](../../benchmarks/live/long_accuracy.py) · [GPU 예약 래퍼](../../benchmarks/evidence/20261119-w4-long/tools/run_accuracy.py)
