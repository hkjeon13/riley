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

다국어·코드·수치·경계 길이 holdout은 아래에 별도로 기록했다. GSM8K 전수 품질 대조는 하단에 기록했다.

vLLM의 최초 참조 시도는 시작 중 컴파일러가 실행 환경에서 `ninja`를 찾지 못해 종료됐다. 기존 vLLM 가상환경의 `ninja`를 이번 실행의 PATH에 추가하고 별도 컴파일 캐시를 사용해 18건을 완료했다. 참조 비교는 vLLM의 강제 prefix 점수나 동시성별 불변성까지 검증하지 않았다. 각 단계 종료 후 GPU 7은 동일 홀더의 20GiB 대기 상태로 돌아갔다. [최초 실패 receipt](../../benchmarks/evidence/20261119-w4-long/vllm-initial/receipt.json) · [최종 예약 상태](../../benchmarks/evidence/20261119-w4-long/vllm/idle-after.json)

- [Riley의 원본 프롬프트 ID·생성 ID·logprobs 및 반복 결과 (gzip)](../../benchmarks/evidence/20261119-w4-long/riley/capture.json.gz)
- [HF의 요청별 토큰·최초 분기·SDPA/eager 점수](../../benchmarks/evidence/20261119-w4-long/hf/reference.json)
- [동일 입력 ID를 확인한 vLLM 원본 응답 (gzip)](../../benchmarks/evidence/20261119-w4-long/vllm/results.json.gz)
- [캡처 및 참조 도구](../../benchmarks/live/long_accuracy.py) · [GPU 예약 래퍼](../../benchmarks/evidence/20261119-w4-long/tools/run_accuracy.py)

## 독립 프롬프트 11개 교차 비교

같은 모델과 Riley GPU 빌드에서 한국어·일본어·코드·수치·반복 입력을 검증했다. 입력 ID는 255–7,917개, 생성 상한은 64·128·256토큰이었다. Riley와 vLLM에 사용된 입력 ID를 각 고정 fixture와 대조했고 HF tokenizer도 11개 모두 같은 ID를 반환했다. HF는 같은 ID에서 BF16 SDPA를, vLLM 0.27.1은 BF16 FlashAttention 2를 사용했다. [입력 fixture](../../benchmarks/evidence/20261119-w4-holdout/holdout.json.gz) · [케이스별 집계](../../benchmarks/evidence/20261119-w4-holdout/summary.json)

| 비교 | 첫 출력 ID | 전체 출력 ID | 출력 텍스트 | 종료 사유 |
|---|---:|---:|---:|---:|
| Riley / HF SDPA | 11/11 | 9/11 | 9/11 | 11/11 |
| vLLM / HF SDPA | 11/11 | 7/11 | 7/11 | 11/11 |
| Riley / vLLM | 11/11 | 8/11 | 8/11 | 11/11 |

Riley의 C8·C32 반복에서는 각각 33/33의 생성 ID·텍스트·종료 사유가 C1과 같았다. 세 엔진 각각 6건은 `stop`, 5건은 `length`로 종료했다. 토큰 비교에서 `stop`으로 끝난 응답에 포함된 마지막 EOS ID를 제거했다.

| Riley와 HF가 갈라진 사례 | 생성 위치 (0부터) | Riley / HF ID | Riley 상위 logprob 차이 | HF SDPA / eager 결과 |
|---|---:|---|---:|---|
| 수치 1,023토큰 | 67 | 510 / 311 | 0.25 | 두 경로 모두 두 ID가 동점 |
| 한국어 1,024토큰 | 77 | 12802 / 42039 | 0 | SDPA는 42039가 0.125, eager는 12802가 0.50 높음 |

7,836토큰 코드 입력에서는 세 엔진의 출력이 일치했다. 7,917토큰 한국어 입력에서 Riley와 HF는 일치했고 vLLM은 생성 위치 7에서 갈라졌다. 길이 경계에 따라 일관되게 분기하는 패턴은 이 표에서 확인되지 않았다. vLLM 종료 로그에는 semaphore 정리 경고 한 건이 남았으며, 각 단계 종료 뒤 GPU 7은 동일 홀더의 20GiB 예약으로 복귀했다. [Riley 캡처](../../benchmarks/evidence/20261119-w4-holdout/riley/capture.json.gz) · [HF 참조](../../benchmarks/evidence/20261119-w4-holdout/hf/reference.json) · [vLLM 원본](../../benchmarks/evidence/20261119-w4-holdout/vllm/results.json.gz) · [vLLM 로그](../../benchmarks/evidence/20261119-w4-holdout/vllm/server.log.gz) · [GPU 7 예약](../../benchmarks/evidence/20261119-w4-holdout/vllm/idle-after.json)

## 첫 prefill 분기 추적

생성 첫 토큰부터 갈라진 사례 3(3,301토큰)과 같은 근방 길이에서 출력이 같았던 사례 9(3,330토큰)를 사용한다. 진단용 빌드는 서빙과 같은 packed prefill 연산에서 선택한 마지막 입력 행의 embedding, 28개 레이어 중간값, 최종 norm 및 전체 logits를 BF16으로 복사한다. 해당 iteration만 CUDA Graph 재생을 우회한다. GPU 커맨드 배치가 완료된 뒤 복사본을 읽으며 진단 빌드의 출력 ID·텍스트·종료 사유가 기존 실행과 같아야 비교를 진행한다.

Hugging Face SDPA에는 같은 입력 ID를 공급하고 원래 첫 출력 ID를 재확인한다. QKV bias 이전 행과 RoPE 이후 query는 참조의 입력·가중치로 별도 재계산한다. 다른 지점은 실제 순전파에서 수집한다. 첫 BF16 불일치 지점은 추적 결과이며, KV 저장·mask 또는 특정 커널이 출력 차이의 원인이라는 판정에는 추가 대조가 필요하다.

격리 소스에서 Rust 타입 검사와 합성 BF16 비교기 검산을 통과했다. 첫 Linux CUDA 빌드는 소스 묶음에서 `kernels/CMakeLists.txt`가 빠져 코드 101로 종료됐다. 서버에 이미 있던 W4 빌드의 커널 파일을 기준 커밋의 34개 파일 해시와 대조했다. 같은 소스 묶음과 진단 패치에 검증한 커널 파일을 보완하며, GPU 7의 W3 성능 측정이 끝나면 GPU 6에서 진단을 수행하도록 등록했다. [진단 패치 (gzip)](../../benchmarks/evidence/20261119-w4-layer-trace/final-prefill.patch.gz) · [첫 빌드 로그](../../benchmarks/evidence/20261119-w4-layer-trace/initial-build.log.gz) · [커널 검증 로그](../../benchmarks/evidence/20261119-w4-layer-trace/retry-admission-kernels.log) · [재실행 명령](../../benchmarks/evidence/20261119-w4-layer-trace/retry-campaign.sh)

두 번째 시도는 CUDA 빌드와 사례 3의 추론까지 완료했다. 첫 출력 ID 264를 포함해 생성 ID·텍스트·종료 사유가 원래 실행과 일치했다. 진단 훅은 펼친 토큰 인덱스로 시퀀스 행 종류 배열을 조회해 대상 행을 기록하지 못했다. 출력 슬롯에서 토큰 인덱스와 시퀀스 행 인덱스를 각각 읽도록 교정했고, 기존 CSR 출력 매핑 테스트를 통과했다. 교정한 추적의 CUDA 빌드도 완료됐고 GPU 6에서 재실행을 시작했다. [두 번째 실패 기록](../../benchmarks/evidence/20261119-w4-layer-trace/retry-capture-failure/native3/receipt.json) · [원래 출력 비교](../../benchmarks/evidence/20261119-w4-layer-trace/retry-capture-failure/native3/response.json.gz) · [교정 패치](../../benchmarks/evidence/20261119-w4-layer-trace/selector-retry/selector.patch.gz) · [실행 순서](../../benchmarks/evidence/20261119-w4-layer-trace/selector-retry/campaign.sh)

## GSM8K 전수 품질 대조

Qwen2.5-7B-Instruct의 Riley와 vLLM이 각각 1,319문제를 오류 없이 완료했다. 동일한 train·test 데이터 해시, 4-shot 입력, 온도 0, 출력 상한 256토큰, 동시 요청 32개, 마지막 숫자 채점 방식을 사용했다. 1,319개 모두에서 요청과 입력 토큰 ID 해시를 대조했다. [데이터셋 매니페스트](../../benchmarks/evidence/20261119-w4-gsm8k/dataset.json) · [검증 집계](../../benchmarks/evidence/20261119-w4-gsm8k/qwen7-comparison.json)

| 모델 | Riley 정답 | vLLM 정답 | Riley − vLLM | 판정 |
|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 1,121/1,319 (84.99%) | 1,114/1,319 (84.46%) | +0.53%p | −1.2%p 품질 기준 통과 |

Riley만 맞힌 문제는 35개, vLLM만 맞힌 문제는 28개였다. 출력 ID와 텍스트는 각각 578개에서 같았다. 결과는 고정 채점 기준의 정확도를 나타낸다. 각 엔진 종료 뒤 GPU 7의 20GiB 예약 복귀를 확인했다. Llama-3.2-3B-Instruct, Qwen3-4B, Qwen2.5-3B-Instruct 비교를 진행 중이다. [Riley 원본](../../benchmarks/evidence/20261119-w4-gsm8k/qwen7-riley/result.json.gz) · [vLLM 원본](../../benchmarks/evidence/20261119-w4-gsm8k/qwen7-vllm/result.json.gz) · [양쪽 실행 체크섬](../../benchmarks/evidence/20261119-w4-gsm8k/qwen7-SHA256SUMS)
