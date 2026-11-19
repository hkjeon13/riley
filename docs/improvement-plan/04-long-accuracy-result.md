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

사례 0의 동일 prefix에서 HF attention을 eager로 바꾸면 SDPA에서 0.125 차이였던 두 ID의 점수가 19.75로 같아졌다. Riley와 HF SDPA가 갈라진 다섯 위치 모두 각 경로의 상위 두 점수가 0.125 이내였다. 표의 Riley 수치는 공개 API가 보고한 logprob 간격이다. 아래 prefill 추적에서 사례 3과 9의 전체 vocabulary logits와 첫 BF16 불일치 지점을 수집했다.

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

`d3d8fe6e5`에서 재빌드한 이미지 `0a907408ec73`을 A100 6번에서 실행했다. 고정된 Riley 캡처와 대조한 결과, 기존 18개 입력의 144건과 홀드아웃 11개 입력의 88건이 모두 일치했다. C1 일반·진단은 각각 한 번, C8·C32는 각각 세 번 실행했다. 입력 ID, 생성 ID, 텍스트, 종료 사유와 usage를 확인했다. 이전 GPU API 검사와 이미지·바이너리·체크포인트 해시가 같았고, 서버는 코드 0으로 종료한 뒤 자원이 모두 회수됐다. [A100 배포 이미지 결과](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-package-validation.json) · [대조기](../../benchmarks/live/long_accuracy.py) · [배포 이미지 실행 방법](01-validation-result.md#배포-이미지-gpu-검사)

## 공통 접두부의 근소한 점수 차이

두 입력군 29건의 원본 해시와 프롬프트 ID를 확인하고, Riley와 HF SDPA가 같은 접두부를 유지하는 동안 선택한 토큰을 다시 대조했다. 전체 생성 ID가 같은 요청은 22건이다. 첫 분기 7건 중 prefill 직후가 1건, decode가 6건이었다. 첫 분기에서 Riley가 보고한 상위 두 logprob 차이는 모두 0.25 이내였고, HF SDPA에서 두 엔진이 각각 선택한 ID의 점수 차이는 최대 0.125였다. [요청별 집계와 원본 해시](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/result.json)

두 엔진이 같은 ID를 선택한 위치는 1,306곳이다. 공개 `top_logprobs`에 두 항목 이상 남은 1,300곳 중 109곳은 차이가 0.25 이내였고, 25곳은 동점이었다. 전체 출력이 같았던 22건 중 14건에도 보고된 동점이 있었다. 이 필드는 디코딩된 텍스트를 키로 쓰므로, 표시가 같은 다른 ID는 집계에서 사라질 수 있다. 작은 점수 차이는 분기 원인을 좁히는 단서로 사용하며, 동점 처리 규칙 변경의 효과를 판정하려면 전체 토큰 ID별 점수가 필요하다. [공통 접두부 검산기](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/audit_shared_prefix.py)

같은 7개 분기 위치의 HF eager 재생에서는 SDPA 선택이 단독 1위인 경우가 4건, Riley 선택이 단독 1위인 경우가 1건, 상위 동점이 2건이었다. vLLM이 같은 접두부까지 도달한 6건에서는 HF SDPA 선택과 4번, Riley 선택과 2번 일치했다. 나머지 1건은 더 앞에서 갈라졌다. 다음 수치 원인 대조는 현재 첫 토큰 추적에 아직 포함되지 않은 공통 접두부의 decode 결정을 우선한다. [엔진별 동일 접두부 대조](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/result.json)

여섯 decode 분기와 두 첫 토큰 대조점에서 실제 출력 슬롯의 BF16 점수 행을 수집하는 진단 패치를 준비했다. C1의 원래 샘플링 경로를 사용하고, 진단 클라이언트는 출력 ID·텍스트·종료 사유·usage가 보존된 응답과 일치할 때만 점수를 검산한다. 로컬 타입 검사와 저장된 두 prefill 행 재계산을 통과했다. AWS 호스트 원본의 체크섬을 대조했고 W5 단독 설정 시험이 끝나면 A100 7번에서 CUDA 빌드와 여덟 행 수집을 시작하도록 등록했다. [고정 입력과 검증 범위](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/preflight.json) · [점수 행 검사기](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/capture_decode_rows.py) · [실행 등록](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/registration.json)

### 실제 BF16 점수 행의 분기 검산

첫 실행은 서버 기동 후 고정 Python 도구가 gzip 입력을 UTF-8로 읽어 요청 전송 전에 실패했다. GPU 7의 예약은 같은 20GiB 홀더로 복귀했다. 압축 판독만 수정해 원본 8건을 CPU에서 다시 확인하고, 같은 CUDA 바이너리와 입력으로 GPU 7에서 재실행했다. 응답 8건의 토큰 ID·텍스트·종료 사유·usage가 기존 기록과 일치했고, 서버 종료 코드는 0이었다. 모든 캡처에서 GPU greedy가 선택한 ID와 실제 서빙 ID가 일치했다. [첫 실행 실패 영수증](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/initial-failure.json) · [같은 바이너리 재실행](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/registration.json) · [압축 판독 패치](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/restore-gzip-input.patch.gz)

전체 주소 가능 토큰의 BF16 행을 AWS에서 재계산했다. Riley와 HF SDPA의 첫 분기 7건 중 5건에서 두 엔진이 고른 ID가 Riley 행의 공동 최대였다. Decode 분기만 보면 6건 중 4건이었다. 나머지 두 분기에서 Riley ID는 단독 최대였고, HF가 고른 ID의 점수는 사례 12에서 0.125, `numbers-1023`에서 0.25 낮았다. 이미 일치하던 사례 9의 Riley 행에서도 ID 264와 5257이 공동 최대였다. [사례별 검산과 원본 행 해시](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/verified-rows.json) · [점수 행 재계산 코드](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/verify_decode_capture.py)

두 엔진이 선택한 ID의 점수를 BF16 비트로 비교했다. 첫 분기 7건 모두 두 ID 중 하나의 점수만 인접한 BF16 값으로 달라졌다. HF SDPA를 기준으로 Riley ID 점수가 한 단계 오른 경우가 4건, HF ID 점수가 한 단계 내린 경우가 3건이었다. 관측 범위는 공통 접두부에서의 최종 점수다. 차이가 시작된 연산을 확인하려면 사례 12의 생성 위치 9와 `numbers-1023`의 위치 67에서 같은 KV와 마지막 norm·출력 투영을 대조해야 한다. [BF16 간격 재계산](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/logit-spacing.json) · [비트 비교 코드](../../benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture/retry/compare_logit_spacing.py)

기록된 행에서 공동 최대 중 큰 ID를 고르는 규칙을 계산하면 분기 5건의 선택이 HF와 같아지며, 현재 일치하는 사례 9는 264에서 5257로 바뀐다. 단독 최대였던 두 분기는 이 규칙에서도 남는다. 이 대조는 각 분기 위치에서 저장된 BF16 행에 적용했다. 전역 동점 선택 규칙의 변경은 미채택한다.

## 첫 prefill 분기 추적

생성 첫 토큰부터 갈라진 사례 3(3,301토큰)과 같은 근방 길이에서 출력이 같았던 사례 9(3,330토큰)를 사용한다. 진단용 빌드는 서빙과 같은 packed prefill 연산에서 선택한 마지막 입력 행의 embedding, 28개 레이어 중간값, 최종 norm 및 전체 logits를 BF16으로 복사한다. 해당 iteration만 CUDA Graph 재생을 우회한다. GPU 커맨드 배치가 완료된 뒤 복사본을 읽으며 진단 빌드의 출력 ID·텍스트·종료 사유가 기존 실행과 같아야 비교를 진행한다.

Hugging Face SDPA에는 같은 입력 ID를 공급하고 원래 첫 출력 ID를 재확인한다. QKV bias 이전 행과 RoPE 이후 query는 참조의 입력·가중치로 별도 재계산한다. 다른 지점은 실제 순전파에서 수집한다. 첫 BF16 불일치 지점은 추적 결과이며, KV 저장·mask 또는 특정 커널이 출력 차이의 원인이라는 판정에는 추가 대조가 필요하다.

격리 소스에서 Rust 타입 검사와 합성 BF16 비교기 검산을 통과했다. 첫 Linux CUDA 빌드는 소스 묶음에서 `kernels/CMakeLists.txt`가 빠져 코드 101로 종료됐다. 서버에 이미 있던 W4 빌드의 커널 파일을 기준 커밋의 34개 파일 해시와 대조했다. 같은 소스 묶음과 진단 패치에 검증한 커널 파일을 보완하며, GPU 7의 W3 성능 측정이 끝나면 GPU 6에서 진단을 수행하도록 등록했다. [진단 패치 (gzip)](../../benchmarks/evidence/20261119-w4-layer-trace/final-prefill.patch.gz) · [첫 빌드 로그](../../benchmarks/evidence/20261119-w4-layer-trace/initial-build.log.gz) · [커널 검증 로그](../../benchmarks/evidence/20261119-w4-layer-trace/retry-admission-kernels.log) · [재실행 명령](../../benchmarks/evidence/20261119-w4-layer-trace/retry-campaign.sh)

두 번째 시도는 CUDA 빌드와 사례 3의 추론까지 완료했다. 첫 출력 ID 264를 포함해 생성 ID·텍스트·종료 사유가 원래 실행과 일치했다. 진단 훅은 펼친 토큰 인덱스로 시퀀스 행 종류 배열을 조회해 대상 행을 기록하지 못했다. 출력 슬롯에서 토큰 인덱스와 시퀀스 행 인덱스를 각각 읽도록 교정했고, 기존 CSR 출력 매핑 테스트를 통과했다. [두 번째 실패 기록](../../benchmarks/evidence/20261119-w4-layer-trace/retry-capture-failure/native3/receipt.json) · [교정 패치](../../benchmarks/evidence/20261119-w4-layer-trace/selector-retry/selector.patch.gz) · [실행 순서](../../benchmarks/evidence/20261119-w4-layer-trace/selector-retry/campaign.sh)

교정한 빌드로 GPU 6에서 두 사례의 Riley·HF 추적 네 건을 완료했다. Riley는 양쪽 사례에서 원래 생성 ID 전체와 텍스트·종료 사유를 재현했다. HF는 두 사례의 기존 첫 토큰을 재현했다. 각 실행 직후 같은 홀더의 20GiB 예약 복귀를 확인했다. [실행 로그](../../benchmarks/evidence/20261119-w4-layer-trace/trace-result/campaign.log) · [원본 추적과 실행 기록 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/trace-result-SHA256SUMS)

| 사례 | 첫 BF16 차이 | 다른 원소 | 최대 절대차 | Riley / HF 첫 ID |
|---:|---|---:|---:|---|
| 3 | 0번 레이어, RoPE 적용 직후 query | 944/3,584 | 0.25 | 264 / 5257 |
| 9 | 0번 레이어, RoPE 적용 직후 query | 950/3,584 | 0.25 | 264 / 264 |

두 사례 모두 마지막 프롬프트 행의 embedding, 첫 norm, bias 적용 전 QKV 투영 값이 비트 단위로 일치했다. Riley의 사례 3 raw logits에서 ID 264와 5257은 18.375로 같았다. HF에서는 264가 18.25, 5257이 18.375였다. [사례 3 전체 지점](../../benchmarks/evidence/20261119-w4-layer-trace/trace-result/case3-comparison.json) · [사례 9 전체 지점](../../benchmarks/evidence/20261119-w4-layer-trace/trace-result/case9-comparison.json)

같은 pre-bias 투영을 BF16으로 반올림한 상태에서 bias와 RoPE를 적용하는 CPU 재계산은 두 사례 모두 Riley의 query 3,584개를 정확히 재현했다. FP64 누산에 bias를 먼저 더한 뒤 BF16으로 반올림하는 재계산은 HF query와 각각 7개 원소에서 달랐다. 이 결과는 첫 관측 차이를 투영 반올림과 bias 적용 순서에 연결한다. [재계산 도구](../../benchmarks/evidence/20261119-w4-layer-trace/tools/replay_layer0_query.py) · [사례 3 재계산](../../benchmarks/evidence/20261119-w4-layer-trace/trace-result/case3-query-replay.json) · [사례 9 재계산](../../benchmarks/evidence/20261119-w4-layer-trace/trace-result/case9-query-replay.json)

GPU 6의 HF SDPA에서 첫 레이어 Q만, 첫 레이어 QKV, 전체 레이어 QKV의 bias를 각각 BF16 투영 후 더했다. 두 사례에서 세 조건의 전체 logits 해시는 모두 기준값과 달랐다. 첫 토큰은 각 조건에서 기준과 같았고, 사례 3에서는 ID 264와 5257의 점수도 18.25와 18.375로 유지됐다. 로짓 전체가 같은 기준 반복 결과도 확인했다. 이 HF 실행에서는 bias 적용 순서만으로 Riley의 사례 3 첫 토큰이 재현되지 않았다. [bias 대조 원본](../../benchmarks/evidence/20261119-w4-layer-trace/bias-order-control/result.json) · [도구와 실행 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/bias-order-SHA256SUMS)

같은 HF 모델에서 캐시를 유지한 채 prefill을 나눴다. 사례 3은 `3072+229`와 `1024+1024+1024+229`, 사례 9는 `3328+2`와 `1024+1024+1024+256+2`로 실행했다. 각 구간 뒤 캐시 길이는 소비한 입력 길이와 일치했다. 사례 3의 `3072+229` 조건에서는 ID 5257의 logit이 18.375에서 18.5로 올랐고 두 사례의 첫 토큰은 모든 분할에서 HF 기준과 같았다. [prefill 분할 원본](../../benchmarks/evidence/20261119-w4-layer-trace/prefill-partition-control/result.json) · [도구와 실행 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/prefill-partition-SHA256SUMS)

Riley 네이티브 빌드 `f7c956d3c`에서 기준을 `dense-v1`과 `fused-split-v1`로 명시했다. A100 GPU 6에서 prompt attention을 `shared-kv-v1`로 바꾸는 실행과 KV 쓰기를 `separate-v1`로 바꾸는 실행을 각각 완료했다. 사례 3과 9에 같은 토큰 ID를 공급했고 세 조건 모두 일반·logprobs·일반 반복 요청의 생성 ID·텍스트·종료 사유가 원래 Riley와 같았다. 첫 토큰의 상위 logprobs도 세 조건에서 일치했다. 사례 3의 Riley 첫 토큰은 세 조건 모두 264였고 HF SDPA는 5257이었다. 각 서버는 코드 0, 활성·대기 요청과 KV·native 할당 0으로 종료했고 GPU 6은 동일 홀더의 20GiB 예약으로 복귀했다. KV 주소와 causal mask의 내부 상태는 별도 검증 범위다. [세 조건 비교](../../benchmarks/evidence/20261119-w4-layer-trace/attention-controls/result/comparison.json) · [실행 원본과 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/attention-controls/result-SHA256SUMS)

### 마지막 norm과 첫 토큰 순위

같은 모델 체크포인트의 토큰 264·5257 출력 가중치와 두 엔진이 남긴 마지막 norm 행으로 내적을 FP64에서 독립 재계산했다. 저장된 추적과 입력의 해시를 검증했으며, 두 사례·두 엔진에서 BF16으로 반올림한 두 점수 모두 실제 logits와 일치했다. [원본·가중치 행 해시와 점수](../../benchmarks/evidence/20261119-w4-layer-trace/final-head-control/result.json) · [CPU 재계산기](../../benchmarks/evidence/20261119-w4-layer-trace/tools/replay_final_head.py)

| 사례 | 엔진 | FP64 점수 차이 5257 − 264 | 관측 BF16 점수 264 / 5257 | 첫 출력 ID |
|---:|---|---:|---:|---:|
| 3 | Riley | +0.020146 | 18.375 / 18.375 | 264 |
| 3 | HF | +0.085401 | 18.25 / 18.375 | 5257 |
| 9 | Riley | −0.061039 | 18.25 / 18.25 | 264 |
| 9 | HF | +0.008971 | 18.375 / 18.375 | 264 |

사례 3에서 Riley의 FP64 점수 차이는 BF16 반올림 뒤 동점이 된다. HF의 마지막 norm 행으로 계산한 차이는 반올림 뒤에도 남는다. 동일한 출력 가중치로 두 엔진의 관측값을 재현했으므로 이 두 토큰의 차이는 출력 투영에 들어온 마지막 norm 행에서 이미 설명된다. 앞단의 어느 연산 조합이 그 norm 차이를 만들었는지는 추가 확인 범위다.

#### Residual 경로와 최종 RMSNorm 재계산

두 사례의 마지막 입력 행에서 28개 레이어의 attention·MLP residual 덧셈을 각각 재계산했다. 사례·엔진당 BF16 원소 200,704개의 결과가 모두 캡처와 일치했다. 최종 RMSNorm에서 정규화 직후 BF16으로 반올림하고 가중치를 곱한 뒤 다시 반올림하는 순서를 적용하자 두 사례·두 엔진의 3,584개 원소를 각각 전부 재현했다. 마지막에 한 번만 반올림하는 대조는 각 행에서 2,587–2,691개가 일치했다. [잔차·정규화 재계산 원본](../../benchmarks/evidence/20261119-w4-layer-trace/residual-margin-control/result.json) · [계산 도구](../../benchmarks/evidence/20261119-w4-layer-trace/tools/replay_residual_margin.py)

토큰 `5257 − 264`의 점수 차이가 Riley에서 HF로 바뀐 양을 같은 출력 가중치로 재계산했다. 누적 residual을 토큰 점수 방향으로 투영한 값과 RMS 배율 변화, 최종 norm의 두 BF16 반올림 영향을 합하면 관측한 점수 차이 변화와 일치한다.

| 사례 | 관측 점수 차이 변화 | 누적 residual | RMS 배율 | 최종 norm 반올림 |
|---:|---:|---:|---:|---:|
| 3 | +0.065256 | +0.036596 | +0.000078 | +0.028581 |
| 9 | +0.070010 | +0.052576 | −0.000073 | +0.017506 |

사례 3에서는 21·22번 레이어가 같은 방향으로 기여했고 27번 레이어는 반대 방향으로 기여했다. 이 값은 기존 활성값에서 관측한 차이로, 특정 레이어를 바꿨을 때의 결과를 예측하지 않는다. 다음 원인 대조에서는 같은 입력을 각 엔진의 attention·MLP에 공급해 그 연산에서 추가되는 차이를 측정한다.

### Native KV와 causal mask 재계산

GPU 6에서 두 사례의 첫 레이어가 사용한 물리 KV 페이지 207개와 209개를 읽었다. 마지막 prefill 묶음의 첫 행과 끝 행에서 KV head 0을 공유하는 query head 7개씩 비교했다. Native Q/K/V와 논리 위치의 causal 범위를 사용한 독립 FP64 재계산의 RMS 차이는 네 행에서 0.000133–0.000197이었다. 마지막 visible 토큰을 제외하면 RMS 차이는 0.0875–0.1171이었고, 첫 행에 미래 토큰을 포함하면 0.00175–0.02668이었다. [사례 3](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/result/case3-replay.json) · [사례 9](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/result/case9-replay.json)

두 사례의 관측 행에서는 causal 범위가 마지막 visible 토큰과 미래 토큰 대조보다 GPU 출력에 가까웠다. 첫 페이지를 뺐을 때 네 행의 RMS가 0.000136–0.000197로 비슷해 해당 페이지의 포함 여부를 이 대조로 판별하기 어렵다. 두 사례의 생성 ID·텍스트·종료 사유는 기존 Riley와 모두 같았고 GPU 6은 같은 홀더의 20GiB 예약으로 복귀했다. 원본 덤프에서 두 사례의 CPU 재계산을 다시 실행했다. 나머지 레이어·KV head와 Riley/HF 간 Q/K/V 차이의 원인은 여전히 확인 범위에 남는다. [실행 순서](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/campaign.sh) · [진단 패치](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/kv-trace.patch.gz) · [CPU 재계산기](../../benchmarks/evidence/20261119-w4-layer-trace/tools/replay_attention_mask.py) · [원본 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/result/SHA256SUMS)

같은 두 입력에서 HF SDPA의 실제 KV 캐시를 읽어 Riley의 0번 레이어 KV head 0과 비교하는 후속 시험을 등록했다. HF의 전체 첫 토큰 logits가 기존 기록과 일치하면, 같은 두 입력 위치에서 Q·K·V를 두 엔진 값으로 조합한 FP64 재계산을 각 엔진의 관측 attention 출력과 대조한다. 교체하는 값과 출력 차이의 관계를 한 레이어에서 측정한다. 합성 입력으로 비교 도구의 8개 조합과 KV 원소 단위 판정을 확인했다. 실제 GPU 대조는 14B 스케줄러 시험 뒤 A100 6번에서 시작한다. [사전 검증](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/preflight.json) · [실행 순서](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/campaign.sh)

첫 시도는 이전 원격 파서 사본의 체크섬이 고정된 값과 달라 GPU 사용 전에 종료됐다. 재실행 파일에 고정된 파서를 포함했다. 새 사전검사에서 두 Riley 저장본의 207·209개 KV 페이지와 원본 해시를 확인하고, 같은 프롬프트의 HF 전체 logits 해시를 기록했다. 재실행은 이미 시작된 RSS 진단이 끝난 뒤 A100 6번을 사용한다. [실패·교정 기록](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/retry.json) · [두 저장본 사전검산](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/preflight-captures.jsonl)

재실행의 HF SDPA는 두 사례의 전체 첫 토큰 logits를 바이트 단위로 재현했다. 사례 3의 첫 ID는 Riley 264, HF 5257이었다. 사례 9는 두 엔진에서 264였다. 0번 레이어 KV head 0의 입력 전체에서 K 원소는 사례별로 73,534/422,528개와 73,828/426,240개, V 원소는 139,854/422,528개와 142,394/426,240개가 달랐다. 원본의 Q·K·V 8개 조합을 FP64로 다시 계산했고 A100 6번은 같은 홀더의 20GiB 예약으로 복귀했다. [GPU 결과와 독립 재계산](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/verified-gpu-result.json)

아래 수치는 모든 입력을 Riley에서 읽은 FP64 재계산과 비교한, 관측 HF attention 출력에 대한 RMS 오차 감소율이다. 각 위치는 KV head 0에 연결된 query head 일곱 개를 포함한다.

| 사례와 입력 위치 | Q를 HF 관측값으로 교체 | K·V를 HF 관측값으로 교체 |
|---|---:|---:|
| 사례 3, 3,072 | 6.57% | 73.89% |
| 사례 3, 3,300 | 9.49% | 46.30% |
| 사례 9, 3,328 | 9.63% | 60.89% |
| 사례 9, 3,329 | 4.78% | 64.46% |

네 행에서 K·V 교체의 효과가 더 컸다. 각 엔진의 Q·K·V를 그대로 사용한 재계산과 해당 엔진의 관측 출력 사이 RMS는 Riley 0.000133–0.000197, HF 0.000099–0.000155였다. 여기서 확인한 영향 범위는 두 사례의 첫 레이어 attention 출력이다. [재계산기](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/verify_hf_kv.py)

두 입력의 마지막 토큰에서 양쪽 엔진의 norm 입력과 bias 전 BF16 투영이 일치했다. 같은 가중치와 입력으로 K·V의 첫 head를 다시 계산했다. 투영을 BF16으로 반올림한 뒤 bias를 더하면 Riley의 K·V 128원소가 각 벡터에서 모두 재현됐다. FP64 누산에 bias를 더한 뒤 BF16으로 반올림하면 HF의 K·V 128원소가 각 벡터에서 모두 재현됐다. K는 두 계산에 같은 RoPE 절차를 적용했다. 마지막 입력 위치의 K·V 차이를 bias와 반올림 순서에서 재현했고, 이후 레이어를 거쳐 최종 점수에 전달되는 영향은 별도 대조 범위다. [가중치·벡터 해시와 재계산 결과](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/kv-projection-replay.json) · [재계산 코드](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/replay_hf_kv_projection.py)

같은 HF SDPA에서 두 변수의 조합을 등록했다. 모든 레이어의 QKV에 bias를 더하는 순서와 1,024토큰 단위 prefill 분할을 각각 켜고 끈 네 조건에서 첫 토큰 ID·전체 logits 해시와 `5257 − 264` 점수 차이를 기록한다. 기존 bias·분할 단독 대조는 두 프롬프트 해시와 기준군 전체 logits 해시가 일치했다. 조합 시험은 각 단독 조건의 logits를 먼저 검증하며, A100 6번의 주기적 메모리 반환 시험 뒤 이어서 실행한다. [교차 시험](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/bias-partition/hf_bias_partition_cross.py) · [입력 검산](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/bias-partition/preflight-result.json)

A100 6번에서 네 교차 조건이 모두 완료됐다. 원래 HF 기준을 한 번 반복해 전체 로짓이 바이트 단위로 일치함을 확인했다. 두 사례 모두 네 조건의 전체 로짓 해시는 각각 달랐다. 첫 토큰 ID는 사례 3에서 모든 조건이 5257, 사례 9에서 모든 조건이 264였다. Riley의 저장된 첫 토큰은 두 사례에서 264다. HF의 `5257 − 264` 점수 차이는 사례 3에서 0.125, 사례 9에서 0.0으로 네 조건 모두에서 같았다. 두 변수의 영향은 전체 로짓에 나타났으며 이 점수와 첫 토큰은 그대로였다. 종료 뒤 GPU 6의 동일한 20GiB 예약을 확인했다. [완료 결과와 원본 해시](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/bias-partition/result/verified-summary.json)

#### 같은 BF16 입력에서의 MLP 비교

잔차 분석에서 점수 차이 기여가 컸던 22번과 27번 레이어를 골랐다. 두 사례에서 같은 HF 모델과 프롬프트로 전체 prefill과 1,024토큰 분할의 기준 로짓을 각각 재현한다. 이어 지정한 한 레이어에서 마지막 prompt 행의 post-attention norm만 Riley에서 캡처한 BF16 벡터로 교체한다. 교체된 입력의 바이트 일치를 검사한 뒤 HF MLP 출력이 동일한 입력을 받은 Riley MLP 출력과 얼마나 가까운지 비교하고 최종 로짓과 첫 토큰도 기록한다. 두 입력 3,301·3,330토큰과 두 엔진의 레이어별 원본 해시를 CPU에서 확인했다. A100 6번에서는 RSS 후속 시험이 종료되면 실행한다. [고정 입력 검산](../../benchmarks/evidence/20261119-w4-layer-trace/same-mlp-input/preflight.json) · [CPU 훅 시험](../../benchmarks/evidence/20261119-w4-layer-trace/same-mlp-input/hook-smoke.json) · [실행 등록](../../benchmarks/evidence/20261119-w4-layer-trace/same-mlp-input/registration.json)

A100 6번에서 두 입력과 두 prefill 형상의 기준 로짓을 모두 재현했다. 원래 HF의 전체 prefill에서는 22·27번 레이어의 관측 norm·MLP 행도 바이트 단위로 일치했다. 지정한 행에 Riley의 BF16 입력을 넣자 8개 조건 모두에서 Riley MLP 출력과 일치하는 원소가 늘었고 RMS 오차가 42.94–89.40% 감소했다. 모든 조건에서 수치 차이가 남았다. 첫 토큰은 사례 3에서 모든 조건이 5257, 사례 9에서 모든 조건이 264였다. 사례 9의 분할 실행에서 27번 레이어를 바꾼 조건은 로짓 5257의 값이 18.375에서 18.25로 바뀌었다. 이 결과는 두 레이어의 마지막 입력 행에 적용된다. 종료 뒤 GPU 6의 동일한 20GiB 예약을 확인했다. [같은 입력에서의 수치 대조](../../benchmarks/evidence/20261119-w4-layer-trace/same-mlp-input/result/verified-summary.json)

#### 저장된 residual 업데이트 교체

같은 두 사례에서 28개 레이어의 마지막 입력 행을 사용했다. 지정한 attention 출력 또는 MLP 출력 행만 상대 엔진의 저장값으로 바꿔 시간 순서대로 BF16 residual 덧셈과 최종 RMSNorm을 재계산했다. 원래 입력과 출력을 그대로 쓴 대조군은 두 엔진의 모든 residual 덧셈과 최종 norm을 정확히 재현했다. 모든 업데이트를 상대 엔진 값으로 교체하면 그 엔진의 두 토큰 점수가 정확히 재현됐다. 세 가중치 행의 해시는 체크포인트의 이전 대조 원본과 일치한다. [CPU 재계산기](../../benchmarks/evidence/20261119-w4-layer-trace/residual-replacement-control/replay_residual_replacements.py) · [레이어별 계산 결과](../../benchmarks/evidence/20261119-w4-layer-trace/residual-replacement-control/result.json.gz)

사례 3의 Riley 저장값에서는 두 토큰이 18.375로 동점이었다. 전체 attention 교체 뒤에도 동점이었고, 전체 MLP를 HF 값으로 교체하면 264는 18.25, 5257은 18.375였다. 단독 MLP 교체에서는 7·14·21·22·23번 레이어가 동점을 풀었다. 역방향의 HF 저장값에서는 단일 레이어 교체로 두 토큰의 BF16 순위가 바뀐 사례가 없었고, 모든 MLP 출력을 Riley 값으로 교체하면 동점이 됐다. 사례 9는 양쪽에서 전체 MLP 교체 뒤에도 동점이었다. 이 계산은 저장된 후속 레이어 출력을 고정한 두 토큰의 민감도를 보여 준다. [원래 점수 및 역방향 대조](../../benchmarks/evidence/20261119-w4-layer-trace/residual-replacement-control/result.json.gz)

후속 GPU 시험은 HF SDPA에서 같은 마지막 입력 위치의 출력 행을 교체한 뒤 나머지 순전파를 실제로 계산한다. 두 사례의 전체 prefill과 1,024토큰 분할에서 22번 레이어 MLP, 전체 attention, 전체 MLP, 두 종류 전체 교체를 각각 대조한다. 기준 로짓 재현과 교체 행의 BF16 바이트를 검증하며 전체 로짓 해시와 선택 토큰을 기록한다. CPU 훅 시험에서 마지막 조각의 마지막 행만 바뀌는 것을 확인했다. A100 6번의 RSS 시험이 끝난 뒤 이 검증을 시작하도록 등록했다. [후속 실행과 사전 검증](../../benchmarks/evidence/20261119-w4-layer-trace/residual-replacement-control/registration.json)

A100 6번에서 두 사례, 두 prefill 형상의 4개 대조군과 16개 교체 실행을 완료했다. 각 조건의 원래 HF 전체 로짓 해시를 교체 전과 훅 제거 후 모두 재현했고, 교체된 1개·28개·56개 행은 해당 Riley 저장값과 BF16 바이트가 일치했다. 표는 최종 선택 토큰이다. [순전파 결과와 원본 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/residual-replacement-control/gpu-verified.json)

| 사례·prefill | 기존 HF | 22번 MLP | 전체 attention | 전체 MLP | 전체 attention·MLP |
|---|---:|---:|---:|---:|---:|
| 3 전체 | 5257 | 264 | 5257 | 264 | 264 |
| 3 분할 | 5257 | 5257 | 264 | 5257 | 264 |
| 9 전체 | 264 | 264 | 264 | 264 | 264 |
| 9 분할 | 264 | 264 | 264 | 264 | 264 |

사례 3에서 전체 prefill은 22번 MLP 행 하나를 바꿔도 264와 5257의 점수가 18.375로 동점이 됐다. 분할 prefill은 전체 attention을 교체한 조건에서 그 동점을 재현했다. 두 종류를 함께 교체하면 두 사례와 두 prefill 형상 4개 모두에서 Riley가 선택한 토큰과 비교한 두 BF16 점수를 재현했다. 전체 어휘 로짓 해시는 모든 교체 실행에서 Riley 추적본과 달랐다. 시험 종료 뒤 기존 GPU 6 홀더의 20GiB 예약을 복구했다. [두 점수와 전체 로짓 체크섬](../../benchmarks/evidence/20261119-w4-layer-trace/residual-replacement-control/gpu-verified.json)

같은 네 행에서 visible 페이지를 한 장씩 제외한 818개 대조를 계산했다. 전체 페이지를 사용한 기준이 746개에서 관측값에 더 가까웠다. 마지막 visible 페이지를 제외하면 오차가 543–1,113배였고 첫 페이지를 제외하면 0.999–1.039배였다. 사례 9의 첫 행에서는 페이지 제외 대조 209개 중 58개가 기준과 같거나 더 가까웠다. 앞쪽 페이지의 포함 여부를 판별하려면 해당 페이지를 제외했을 때의 변화가 FP64 재계산과 GPU 출력 사이의 기준 오차보다 큰 입력을 선정해 다시 검증해야 한다. [페이지별 대조 결과](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/page-omission-sweep.json.gz) · [대조 도구](../../benchmarks/evidence/20261119-w4-layer-trace/tools/sweep_visible_kv_pages.py)

두 사례의 마지막 입력 행에 대해 bias 이전의 BF16 투영값과 체크포인트의 K/V bias를 사용해 캐시 저장값을 독립 계산했다. 레이어 0, KV head 0에서 K의 RoPE 적용과 V의 bias 적용 결과가 기록된 페이지의 마지막 입력 슬롯과 각각 128/128개 일치했다. 두 사례 모두 직전 슬롯에서는 K 5/128개, V 0/128개만 일치했다. 이 대조는 추적된 두 마지막 입력 슬롯의 저장값을 확인한다. [KV 저장값 대조](../../benchmarks/evidence/20261119-w4-layer-trace/kv-mask-audit/final-slot-replay.json) · [재계산 도구](../../benchmarks/evidence/20261119-w4-layer-trace/tools/replay_final_kv_slot.py)

## GSM8K 전수 품질 대조

네 모델에서 Riley와 vLLM이 각각 1,319문제를 오류 없이 완료했다. 같은 train·test 데이터 해시, 4-shot 입력, 온도 0, 출력 상한 256토큰, 동시 요청 32개, 마지막 숫자 채점 방식을 사용했다. 같은 모델의 두 엔진에서 모든 문제의 요청과 입력 토큰 ID 해시가 일치했다. 점수 차이는 원본 정답 수에서 계산했다. [데이터셋 매니페스트](../../benchmarks/evidence/20261119-w4-gsm8k/dataset.json) · [전체 실행 기록](../../benchmarks/evidence/20261119-w4-gsm8k/campaign-result.log)

| 모델 | Riley 정답 | vLLM 정답 | Riley − vLLM | 판정 |
|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 1,121/1,319 (84.99%) | 1,114/1,319 (84.46%) | +0.53%p | −1.2%p 품질 기준 통과 |
| Llama-3.2-3B-Instruct | 866/1,319 (65.66%) | 873/1,319 (66.19%) | −0.53%p | −1.2%p 품질 기준 통과 |
| Qwen3-4B | 1,133/1,319 (85.90%) | 1,118/1,319 (84.76%) | +1.14%p | −1.2%p 품질 기준 통과 |
| Qwen2.5-3B-Instruct | 921/1,319 (69.83%) | 934/1,319 (70.81%) | −0.99%p | −1.2%p 품질 기준 통과 |

Qwen2.5-7B에서 Riley만 맞힌 문제는 35개, vLLM만 맞힌 문제는 28개였다. 출력 ID와 텍스트는 각각 578개에서 같았다. [Qwen2.5-7B 양쪽 실행 체크섬](../../benchmarks/evidence/20261119-w4-gsm8k/qwen7-SHA256SUMS)

Llama-3.2-3B에서도 두 엔진의 1,319개 요청 설정과 입력 ID 해시가 일치했고 오류는 0건이었다. Riley만 맞힌 문제는 19개, vLLM만 맞힌 문제는 26개, 출력 ID와 텍스트가 같은 문제는 각각 899개였다. [Llama 결과](../../benchmarks/evidence/20261119-w4-gsm8k/llama3-comparison.json) · [Llama 양쪽 실행 체크섬](../../benchmarks/evidence/20261119-w4-gsm8k/llama3-SHA256SUMS)

Qwen3-4B에서 Riley만 맞힌 문제는 32개, vLLM만 맞힌 문제는 17개였다. [Qwen3-4B 결과](../../benchmarks/evidence/20261119-w4-gsm8k/qwen4-comparison.json) · [Qwen3-4B 양쪽 실행 체크섬](../../benchmarks/evidence/20261119-w4-gsm8k/qwen4-SHA256SUMS)

KV head 2개인 Qwen2.5-3B-Instruct에서 Riley만 맞힌 문제는 42개, vLLM만 맞힌 문제는 55개였다. 네 모델 모두 두 엔진의 실행이 끝날 때마다 GPU 7의 동일 홀더가 20GiB 예약으로 복귀했다. [Qwen2.5-3B 결과](../../benchmarks/evidence/20261119-w4-gsm8k/qwen3-comparison.json) · [Qwen2.5-3B 양쪽 실행 체크섬](../../benchmarks/evidence/20261119-w4-gsm8k/qwen3-SHA256SUMS)
