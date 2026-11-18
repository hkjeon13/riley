# W3 혼합 부하 스케줄링 실측

2026-11-19 KST에 `aws-a100` GPU 7에서 Qwen2.5-0.5B-Instruct의 1,024행 prefill burst를 확인했다. 기존 0.1.0 `riley-lp16`과 모델을 고정하고, 기존 정책과 후보를 같은 seed에서 순서를 바꿔 실행했다. 셀은 `closed-c32`, `rag-c1`, `rag-c8`, `rag-r3`다. 각 실험 뒤 같은 홀더가 20GiB 대기 예약으로 복귀했다. [W3 계획](03-long-context-and-scheduling.md)

| 후보 | `rag-c8` TTFT p95, seed 1·2 | `rag-c8` TPOT p50, seed 1·2 | 짧은 요청 영향 |
|---|---:|---:|---|
| decode 행 최대 32 → 2에서만 burst 허용 | +102%, +136% | −14.3%, −13.1% | `closed-c32` TTFT p95 +0.3%, +1.1% |
| 4,096토큰 미만 prompt를 최대 3ms 묶기, decode가 있을 때 적용 | +13.3%, +3.0% | −0.7%, +1.0% | `closed-c32` TTFT p95 +646%, +228% |

burst를 일찍 제한하면 `rag-c8`의 토큰 간 지연 p99가 두 seed에서 약 58% 감소했다. 그 대신 첫 토큰 지연이 두 배 이상 늘어 기본값으로 채택하지 않았다. Prompt 묶기는 짧은 요청이 기다리게 했고, `rag-c8`에서도 이득이 반복되지 않아 채택하지 않았다.

후보별 672개 측정 요청에서 입력 지문, 출력 텍스트 지문, 종료 사유, prompt/output 토큰 수가 대조군과 일치했다. Open-loop 셀에서는 예정 도착도 일치했다. 측정·워밍업에서 요청 실패와 429는 0건이었다. 출력 지문은 사용자에게 전달된 UTF-8 텍스트의 SHA-256이다. 개별 토큰 ID 일치나 전체 모델 재검증은 별도로 확인해야 한다.

각 `rag-c8` 셀의 완료 요청은 64건이므로 p95는 후보 선별에 사용했다. 절대 기준의 달성 여부와 14B tail의 원인은 추가 측정이 필요하다.

## Qwen3-4B 단독 입력 길이 곡선

같은 GPU에서 Qwen3-4B로 C1 요청을 길이당 12건씩 실행했다. 1,024행 burst와 출력 상한 32토큰을 고정했고, 아래 입력 길이는 서버가 보고한 평균 토큰 수다.

| 입력 토큰 | TTFT p50 (ms) | TPOT p50 (ms) |
|---:|---:|---:|
| 50 | 9.5 | 7.68 |
| 1,984 | 93.5 | 8.50 |
| 4,056 | 205.5 | 8.64 |
| 6,064 | 337.5 | 8.78 |
| 7,847 | 481.1 | 9.01 |

60건 모두 성공했다. 동시 요청이 없는 상태에서도 문맥이 길어질수록 첫 토큰 지연 증가 폭이 커졌다. 다음에는 이 입력을 그대로 프로파일링해 prefill GEMM, attention 및 호스트 비용을 구분한다.

- [Burst 제한 결과와 실행 근거](../../benchmarks/evidence/20261119-w3-small-rag/summary.json)
- [Prompt 묶기 결과와 실행 근거](../../benchmarks/evidence/20261119-w3-coalescing/summary.json)
- [Qwen3-4B 입력 길이 곡선](../../benchmarks/evidence/20261119-w3-qwen3-lengths/summary.json)

## 14B 지연 재현과 Nsight 계측

Qwen2.5-14B-Instruct와 Qwen2.5-Coder-14B-Instruct는 같은 모델 안에서 Riley와 vLLM의 KV 용량을 2,576개 16토큰 블록에 맞춰 요청한다. vLLM 시작 로그의 실제 KV 토큰 수도 기록한다. 14B의 기존 `rag-c8` 입력 생성기와 부하 조건을 사용해 seed 1·2의 기본 지연을 확인한다. 별도 대조에서는 같은 8개 요청을 한 번에 보내거나 50ms 간격으로 보내고, 각 묶음이 끝난 뒤 다음 묶음을 보낸다. 두 엔진 모두의 요청 지문과 실제 prompt/output 토큰 수를 기록한다.

Riley는 Nsight를 켠 별도 실행을 둔다. Qwen3-4B는 기존 C1 다섯 길이를, 14B 두 모델은 C1·C8과 두 도착 조건을 계측한다. 합성 요청의 전송 시점부터 첫 콘텐츠까지의 구간에 해당 GPU 프로세스의 커널을 맞추고, 겹친 구간은 합집합으로 센다. C1은 한 요청의 구간을 본다. C8의 GPU 시간은 여러 요청이 공유하므로 요청별 대기나 커널 비용으로 해석할 수 없다.

Nsight CPU 표식 두 개의 시간축 정렬 편차는 29.41µs였다. 기존 CUDA trace의 PID 인코딩을 확인했고, 합성 trace로 겹친 커널 구간과 다른 PID의 제외를 검산했다. 실제 세 모델 계측은 앞선 소크·정확성 검증 뒤 실행한다. [사전 확인](../../benchmarks/evidence/20261119-w3-rag-profile/nsight-clock-preflight.json) · [실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/campaign.sh) · [계측 집계기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/profile_summary.py)

### Qwen2.5-14B-Instruct 일반 서빙

GPU 7에서 `f7c956d3c`의 Riley CUDA 바이너리와 vLLM을 순차 실행했다. 두 엔진의 KV 용량을 41,216토큰으로 맞췄고 vLLM 시작 로그에서도 같은 수를 확인했다. 엔진별 224개 측정 요청과 36개 워밍업이 오류·429 없이 종료했다. 모든 측정 요청의 본문 지문·프롬프트 토큰 수가 엔진 간 일치했고 각 요청은 128토큰을 생성했다. [Riley 실행 인자](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-riley/plan.json) · [vLLM 실행 및 실제 KV](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-vllm/receipt.json) · [요청 집계와 해시 대조](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-comparison.json)

| 부하 | 완료 건수 | TTFT p95 Riley / vLLM (ms) | TPOT p50 Riley / vLLM (ms) |
|---|---:|---:|---:|
| `rag-c1` | 16 | 468 / 451 | 24.01 / 23.95 |
| `rag-c8` seed 1 | 64 | 1,756 / 1,222 | 39.74 / 39.76 |
| `rag-c8` seed 2 | 64 | 1,488 / 1,080 | 38.20 / 38.73 |
| 8개 동시 전송 | 40 | 3,221 / 2,970 | 29.46 / 34.00 |
| 8개 50ms 간격 전송 | 40 | 3,000 / 2,645 | 30.33 / 32.87 |

8개 요청의 첫 소켓 전송부터 마지막 전송까지 중앙값은 동시 조건 0.18–0.19ms, 간격 조건 350.32–350.80ms였다. 간격을 둔 조건에서 두 엔진 모두 TTFT p95가 줄었고, 감소폭은 Riley 6.87%, vLLM 10.93%였다. Riley와 vLLM의 `rag-c8` TPOT p50은 두 seed에서 1.4% 이내였지만 TTFT p95는 Riley가 각각 43.7%와 37.8% 길었다.

출력 텍스트와 종료 사유는 엔진 간 38/224에서 일치했다. 현재 성능표에는 같은 입력과 출력 길이의 결과를 사용했고 생성값 차이도 집계에 보존했다. 두 실험 종료 뒤 GPU 7은 동일 홀더의 20GiB 예약으로 돌아왔다. [원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-SHA256SUMS)

### Qwen2.5-14B-Instruct 커널 계측

같은 Riley 바이너리와 입력으로 160건을 계측했다. 스케줄 설정을 유지했고 모든 응답의 출력 텍스트와 종료 사유가 계측하지 않은 실행과 일치했다. TTFT p95는 셀별로 +0.38–1.43% 바뀌었다. CPU 시계와 Nsight의 두 표식 사이 오프셋 변화는 각 셀에서 0.49–5.87µs였다. [계측 결과와 원본 위치·해시](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-profile-summary.json)

| 부하 | 전송 → 첫 토큰 p50 (ms) | GPU 커널 비중 p50 | 커널 밖 시간 p50 (ms) | GEMM p50 (ms) | Attention p50 (ms) |
|---|---:|---:|---:|---:|---:|
| `rag-c1` | 359 | 98.07% | 8.18 | 291 | 41 |
| `rag-c8` seed 1 | 818 | 98.71% | 11.54 | 665 | 109 |
| 8개 동시 전송 | 2,517 | 98.08% | 47.34 | 2,039 | 294 |
| 8개 50ms 간격 전송 | 2,230 | 98.63% | 33.38 | 1,815 | 262 |

짧은 커널 이름 `Kernel2`가 CUTLASS GEMM을 숨겼고, FlashInfer prefill 커널은 이름이 `strided_paged_kernel`과 `merge_parts_kernel`이었다. 동일 SQLite의 전체 심볼을 사용해 다시 분류했으며 160개 요청의 GPU 커널 구간 합집합이 첫 집계와 일치했다. 첫 토큰까지 GPU가 대부분의 시간을 사용했고 GEMM 구간이 가장 길었다. 동시 요청은 GPU 구간을 공유하며, 이 표의 판정 범위는 첫 토큰까지의 공유 GPU 구간과 커널 종류다. [전체 심볼 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-full-symbols.json) · [재분류 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/classify_full_symbols.py)

같은 실행의 host trace는 128스텝 평균을 44개 기록했다. 이 평균값들의 중앙값은 제출 1.57ms, 완료 대기 32.90ms였다. 명령 처리·제출·완료 대기를 제외한 나머지 호스트 단계 합은 구간당 스텝 평균의 중앙값이 0.179ms였고, 이 중 plan 55.7µs, pack 31.5µs, download 36.8µs를 기록했다. 이 표본은 여러 부하와 워밍업을 포함한다. 요청별 큐 대기와 GPU 계산 중 실제로 겹친 CPU 시간은 추가 계측 항목이다. [host trace 원본](../../benchmarks/evidence/20261119-w3-rag-profile/qwen14-profile/server.log.gz)

### Qwen2.5-Coder-14B-Instruct 일반 서빙

GPU 7에서 실행 순서를 vLLM → Riley로 바꿨다. KV 용량 41,216토큰과 엔진별 워밍업 36건·측정 224건을 확인했다. 두 엔진은 모든 측정 요청에서 본문 지문과 입력 토큰 수가 일치했고 각각 128토큰을 생성했다. 측정과 워밍업의 오류·429는 0건이었다. 출력 텍스트와 종료 사유는 50/224에서 일치했다. [Riley 실행 조건](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-riley/receipt.json) · [vLLM 실행 조건과 KV](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-vllm/receipt.json) · [케이스별 결과](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-comparison.json)

| 부하 | 완료 건수 | TTFT p95 Riley / vLLM (ms) | TPOT p50 Riley / vLLM (ms) |
|---|---:|---:|---:|
| `rag-c1` | 16 | 465 / 453 | 24.03 / 23.96 |
| `rag-c8` seed 1 | 64 | 1,760 / 1,224 | 39.77 / 39.82 |
| `rag-c8` seed 2 | 64 | 1,531 / 1,089 | 38.78 / 38.80 |
| 8개 동시 전송 | 40 | 3,246 / 2,975 | 29.41 / 34.05 |
| 8개 50ms 간격 전송 | 40 | 3,001 / 2,666 | 30.35 / 32.91 |

Instruct와 비슷하게 단일 요청의 Riley TTFT p95는 vLLM 대비 +2.7%였고 동시성 8에서는 +43.8%, +40.6%였다. 동시 전송을 50ms 간격으로 바꾸면 Riley의 TTFT p95는 7.54% 감소하고 TPOT p50은 3.20% 증가했다. vLLM은 각각 10.38%, 3.34% 감소했다. 실험은 입출력 길이와 KV 용량을 통제했고, 각 엔진의 기존 스케줄러 설정을 사용했다. 두 실행이 끝난 뒤 GPU 7의 20GiB 예약 복귀를 확인했다. [원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-SHA256SUMS)

### Qwen2.5-Coder-14B-Instruct 커널 계측

같은 실행 파일과 요청으로 160건을 계측했다. 모든 출력 텍스트와 종료 사유가 비계측 실행과 일치했다. 계측 실행의 TTFT p95 변화는 셀별 −0.03%에서 +2.03%였고, 시계 오프셋 변화는 0.72–4.38µs였다. 두 14B 모델 모두 첫 토큰까지 GPU가 오래 실행됐고, 전체 심볼로 식별한 GEMM의 시간이 가장 길었다. [계측 결과와 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-profile-summary.json) · [전체 심볼 검산](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-full-symbols.json)

| 부하 | 전송 → 첫 토큰 p50 (ms) | GPU 커널 비중 p50 | 커널 밖 시간 p50 (ms) | GEMM p50 (ms) | Attention p50 (ms) |
|---|---:|---:|---:|---:|---:|
| `rag-c1` | 360 | 98.03% | 8.75 | 292 | 41 |
| `rag-c8` seed 1 | 822 | 98.56% | 12.71 | 667 | 110 |
| 8개 동시 전송 | 2,527 | 97.97% | 47.98 | 2,046 | 296 |
| 8개 50ms 간격 전송 | 2,248 | 98.60% | 32.71 | 1,819 | 263 |

128스텝 평균을 44개 수집했다. 평균값들의 중앙값은 제출 1.58ms, 완료 대기 32.96ms, 그 둘과 명령 처리를 제외한 나머지 호스트 단계 합 0.187ms였다. 이 집계는 워밍업과 여러 부하를 포함한다. 동시성 구간에서 여러 요청은 동일한 GPU 커널 실행을 공유한다. [계측 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/coder14-profile-SHA256SUMS) · [검증·집계 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/summarize_profile_arm.py)

### Qwen3-4B 계측의 종료 오류와 재시도

다섯 길이에서 10건의 워밍업과 60건의 스트리밍 요청이 완료됐다. 시도의 입력 토큰 평균은 약 50·1,984·4,056·6,064·7,847개였다. 서버 종료 중 GPU 감시기가 기존 Riley PID를 외부 프로세스로 판정해 시도가 실패했다. 이 PID는 실패 기록과 서버 시작 파일에서 같았다. GPU 7의 홀더는 종료 후 20GiB 예약으로 돌아왔다. 응답 측정 원본은 보존했으며 해당 시도의 커널 데이터는 미수집 상태다. [실패 기록](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-initial/receipt.json) · [초기 요청과 원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-initial-and-retry-SHA256SUMS)

재시도는 실행 중 확인한 서버의 PID·시작 시각을 기록하고, 같은 서버가 종료된 상태에서만 남은 NVML 표본을 받아들인다. Nsight의 임시 경로도 자식 프로세스가 종료된 뒤 정리하도록 순서를 바꿨다. Linux에서 실제 자식 프로세스를 사용해 실행·좀비·회수 상태, 시작 시각 불일치, 외부 PID를 검증했다. GPU 계측 재실행은 기존 대기열 뒤에 등록했다. [재시도 패치](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-retry/runner.patch.gz) · [재시도 순서](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-retry/campaign.sh)

### 엔진별 배치 토큰 예산 대조

Riley 실행은 스텝당 1,024토큰을 명시했다. 서버에 설치된 vLLM 0.27.1의 소스는 이 A100과 OpenAI 서버 조건에서 chunked prefill 기본 배치 한도를 2,048토큰으로 선택한다. 기존 실행의 실제 스케줄러 한도는 설치 소스에서 추론한 값으로 기록한다. Qwen2.5-14B-Instruct에서 vLLM의 한도를 각각 1,024·2,048로 명시하고 같은 KV·요청에 적용하는 대조를 등록했다. 현재 대기열 뒤에 실행하며 토큰 한도 변화의 효과를 확인한다. [기본값과 소스 해시](../../benchmarks/evidence/20261119-w3-budget-control/batch-budget-observation.json) · [대조 실행 순서](../../benchmarks/evidence/20261119-w3-budget-control/campaign.sh) · [결과 검증 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/compare_vllm_budgets.py)

## W2 이벤트 발행과 묶음 정책의 상호작용

W4·W5에 쓴 `f7c956d3c` CUDA 바이너리와 Qwen2.5-0.5B에서 이벤트 발행 `immediate-v1`·`after-submit-v1`과 wide prefill 허용 상한 32·2 decode 행을 교차한다. 현행 프로파일의 이벤트 발행은 `after-submit-v1`이며, 이전 W3 측정과 같은 burst 설정과 네 부하 `closed-c32`, `rag-c1`, `rag-c8`, `rag-r3`를 사용한다. 두 seed는 실행 순서를 뒤집는다. 각 조건은 별도 서버에서 측정한 뒤 8개 동일 입력의 C1·C8 생성 토큰 ID, 텍스트, 종료 사유와 usage도 확인한다.

A100 GPU 6에서 실행하도록 등록했다. 성능 측정 중 다른 CUDA 작업의 간섭을 줄이기 위해 진행 중인 14B 측정·레이어 추적·GSM8K 비교가 종료된 뒤 시작한다. 집계기는 현재 wide 기준 `after-submit-v1`/32행 대비 각 조건의 지표와, 두 이벤트 발행 모드에서 2행 정책 효과가 달라지는 비율을 seed별로 기록한다. 완료 처리, 같은 요청, 응답 일치, 종료 자원을 검증하고 원본이 바뀌면 해시 검사에서 실패한다. [실행 순서](../../benchmarks/evidence/20261119-w23-matrix/campaign.sh) · [측정 도구](../../benchmarks/evidence/20261119-w23-matrix/tools/measure_matrix.py) · [결과 집계기](../../benchmarks/evidence/20261119-w23-matrix/tools/summarize_matrix.py)
