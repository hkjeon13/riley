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

60건 모두 성공했다. 동시 요청이 없는 상태에서도 문맥이 길어질수록 첫 토큰 지연 증가 폭이 커졌다. 아래 Qwen3-4B 계측에서는 같은 길이·요청 조건의 GPU 커널 시간을 분해했다.

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

재실행에서 60건과 워밍업 10건을 오류 없이 마쳤다. 앞선 시도와 60건의 요청, 입력·생성 토큰 수, 출력 텍스트, 종료 사유가 모두 같았다. 서버는 코드 0으로 종료했고 GPU 7은 동일 홀더의 20GiB 예약으로 돌아갔다. Nsight의 원본과 SQLite 해시를 검증했고 요청별 GPU 커널 합집합을 전체 커널 심볼로 검산했다. [재실행 결과](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-summary.json) · [두 계측 실행의 출력 비교](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-retry-parity.json) · [Nsight 원본 위치와 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile/trace-files.json)

| 평균 입력 토큰 | 전송→첫 토큰 p50 (ms) | GPU 커널 비중 p50 | GEMM p50 (ms) | Attention p50 (ms) | 커널 밖 p50 (ms) |
|---:|---:|---:|---:|---:|---:|
| 50 | 10.20 | 85.74% | 7.24 | 0.62 | 1.45 |
| 1,984 | 94.65 | 95.90% | 66.44 | 13.75 | 3.87 |
| 4,056 | 209.21 | 96.74% | 133.29 | 47.80 | 6.83 |
| 6,064 | 342.18 | 97.22% | 200.04 | 100.75 | 9.50 |
| 7,847 | 486.03 | 97.48% | 266.52 | 164.68 | 12.24 |

7,847토큰에서 attention은 164.68ms로 전송부터 첫 토큰까지 시간의 약 34%였다. 약 6K에서 7.8K로 늘렸을 때 GEMM은 66.5ms, attention은 63.9ms 증가했다. 위 값은 Nsight를 켠 동시성 1의 중앙값이며 호스트·스케줄러 개별 원인이나 실제 꼬리 지연을 직접 판정하려면 추가 대조가 필요하다. [커널 심볼별 분류](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-full-symbols.json) · [계측 원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-result-SHA256SUMS)

이전에 계측 없이 잰 길이 곡선은 `riley-lp16`으로 실행했다. 후속 대조는 Nsight 실행과 같은 `f7c956d3c` 바이너리·입력·KV·설정을 사용했고, 다섯 길이에서 60건을 오류 없이 완료했다. 모든 출력 텍스트와 종료 사유가 계측 실행과 일치했다. 계측 켠 조건의 TTFT p95는 계측 끈 조건 대비 약 50토큰에서 +13.24%, 이후 길이별로 +2.28%, +0.53%, −0.34%, +0.56%였다. 각 길이당 12건, 조건별 한 번의 실행에서 관측한 차이다. [동일 바이너리 대조](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-profile-paired-summary.json) · [비계측 결과 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qwen4-unprofiled-SHA256SUMS)

### 엔진별 배치 토큰 예산 대조

Riley 실행은 스텝당 1,024행을 명시했다. 설치된 vLLM 0.27.1 소스에서 이 A100의 OpenAI 서버용 기본 배치 한도는 2,048토큰으로 추론했다. 후속 실험에서 Qwen2.5-14B-Instruct의 vLLM 한도를 1,024와 2,048로 명시했다. 양쪽 API 시작 로그에서 각각의 값을 확인했고 1,024 조건은 스케줄러 로그에도 값이 있었다. 두 조건의 KV 41,216토큰과 FlashAttention 2 경로가 같았다. [기본값과 소스 해시](../../benchmarks/evidence/20261119-w3-budget-control/batch-budget-observation.json) · [실행 원본과 체크섬](../../benchmarks/evidence/20261119-w3-budget-control/result-SHA256SUMS)

두 조건은 GPU 7에서 1,024 → 2,048 순서로 각각 224건을 측정했고, 워밍업 36건을 포함해 오류·429가 없었다. 요청 본문과 실제 입력 토큰 수, 생성 토큰 수는 두 조건에서 모두 일치했다. 출력 텍스트와 종료 사유가 일치한 요청은 66/224였다. [결과 집계](../../benchmarks/evidence/20261119-w3-budget-control/result/comparison.json) · [검증 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/compare_vllm_budgets.py)

| 부하 | 1,024 TTFT p95 (ms) | 2,048 TTFT p95 (ms) | 1,024의 변화율 |
|---|---:|---:|---:|
| `rag-c1` | 480 | 451 | +6.4% |
| `rag-c8` seed 1 | 1,097 | 1,244 | −11.9% |
| `rag-c8` seed 2 | 1,078 | 1,090 | −1.1% |
| 8개 동시 전송 | 3,260 | 2,980 | +9.4% |
| 8개 50ms 간격 전송 | 2,902 | 2,662 | +9.0% |

두 한도의 차이는 부하에 따라 방향이 달랐다. 이전 Riley 1,024행 실행과 같은 요청 본문·입력 길이·KV로 교차 검산하면 Riley의 `rag-c8` TTFT p95는 vLLM 1,024토큰보다 두 seed에서 60.2%와 38.0% 길었다. 두 엔진의 수치 한도를 명시해도 이 측정의 지연 격차가 남았다. vLLM 한도당 한 번, Riley는 앞선 세션의 결과이므로 반복 정밀도와 엔진별 스케줄러의 차이는 추가 검토 범위다. [Riley 교차 비교](../../benchmarks/evidence/20261119-w3-budget-control/result/riley-cross-session-comparison.json)

### Riley 14B의 요청별 대기와 prefill 계측

같은 Qwen2.5-14B-Instruct, KV 41,216토큰, 스텝당 1,024행과 진단 바이너리로 A100 GPU 7에서 계측 off·on을 순서대로 실행했다. 다섯 부하에서 조건별 측정 224건과 워밍업 36건을 오류·429 없이 마쳤다. 모든 측정 요청의 본문 지문, 입력·생성 토큰 수, 응답 텍스트 지문과 종료 사유가 일치했다. 두 실행 모두 코드 0으로 종료했고 같은 홀더의 20GiB 예약으로 복귀했다. [실행 순서 및 입력 해시](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/campaign.sh) · [원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/result/SHA256SUMS)

| 부하 | 비계측 TTFT p95 (ms) | 계측 TTFT p95 (ms) | 계측 조건 변화율 |
|---|---:|---:|---:|
| `rag-c1` | 477 | 473 | −0.84% |
| `rag-c8` seed 1 | 1,776 | 1,773 | −0.20% |
| `rag-c8` seed 2 | 1,768 | 1,592 | −9.93% |
| 8개 동시 전송 | 3,234 | 3,240 | +0.19% |
| 8개 50ms 간격 전송 | 3,018 | 3,011 | −0.22% |

`rag-c8` seed 2는 이전 비계측 세션과 이번 비계측 조건 사이에도 +18.82% 차이가 있었다. 두 세션은 입력·응답이 같았고 바이너리와 측정 시점이 달랐다. 위 표의 계측 off·on은 각각 한 번씩 실행했으며 계측 비용을 추정하려면 반복이 필요하다. [이전 세션과 입력·출력 대조](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/result/cross-session-control.json)

SSE 요청 ID로 엔진 제출부터 첫 응답 조각 생성까지를 연결했다. 스텝의 시작에서 반환까지에는 호스트와 GPU 작업이 포함된다. 같은 스텝의 CUDA 이벤트 시간은 해당 요청들이 공유한다. 아래 값은 각 seed에서 클라이언트 TTFT p95 이상인 네 요청의 중앙값이다. [요청별 시각과 집계](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/result/summary.json) · [검증 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/summarize_request_stages.py)

| `rag-c8` | 엔진 명령 대기 | 첫 prefill 선택 대기 | 선택된 prefill 실행 | prefill 사이 대기 | prefill 사이 다른 요청의 prefill과 겹친 시간 |
|---|---:|---:|---:|---:|---:|
| seed 1 | 127ms | 336ms | 545ms | 1,008ms | 1,006ms |
| seed 2 | 124ms | 195ms | 620ms | 657ms | 655ms |

두 seed에서 prefill 사이 대기 대부분은 다른 요청의 prefill 구간과 겹쳤다. 이 구간은 토큰화나 스텝 자체의 실행 시간과 별도로 확인됐다. 같은 행 예산으로 조각 선택 순서를 비교할 때는 첫 토큰 꼬리와 다른 요청의 대기·decode TPOT를 함께 측정해야 한다.

같은 추적에서 prefill 사이 대기와 겹친 스텝을 실제 처리 행 수와 decode 행 수로 나눴다. 아래 값은 부하별 TTFT p95 이상 요청의 중앙값이며 일반 행 예산은 256행이었다. 일반 예산 이하 스텝과 겹친 시간의 중앙값은 표의 네 부하에서 0ms였다. 각 열의 중앙값은 독립적으로 계산했다. [스텝 형태별 분해](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/result/wait-shapes.json) · [재현 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/analyze_prefill_wait_shapes.py)

| 부하 | TTFT 꼬리 요청 수 | 256행 초과, decode 0행 | 256행 초과, decode 1–2행 | 256행 초과, decode 3행 이상 |
|---|---:|---:|---:|---:|
| `rag-c8` seed 1 | 4 | 0ms | 535ms | 403ms |
| `rag-c8` seed 2 | 4 | 0ms | 200ms | 401ms |
| 8개 동시 전송 | 2 | 636ms | 1,457ms | 0.08ms |
| 8개 50ms 간격 전송 | 2 | 837ms | 758ms | 358ms |

긴 prefill 스텝이 decode 1–2행과 겹친 시간도 두 seed에서 컸다. 후속 대조에서는 decode 행 개수와 혼합 스텝의 토큰 예산을 따로 바꿔 평가한다. 위 수치는 관측한 실행 구간의 중첩이며, 스텝을 제한했을 때의 지연 변화는 실제 대조가 필요하다.

#### 대기 중 처리된 prefill 요청의 입장 순서

같은 진단 원본에서 요청 308개의 prefill 기록을 756개 스텝으로 재구성했다. 겹친 기록은 배치 행 수·decode 행 수·CUDA 시간이 일치하고 개별 prompt 행의 합이 배치와 맞는 경우 같은 스텝으로 묶었다. 각 TTFT 꼬리 요청의 prefill 사이 대기 중 처리된 prompt를 해당 요청보다 엔진에 먼저 들어왔는지로 분류했다. [요청별 시간과 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/result/prefill-turns.json) · [재계산 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/analyze_prefill_turns.py)

| 부하 | 전체 겹침 중앙값 | 먼저 입장한 prompt만 | 나중에 입장한 prompt만 | 두 순서가 섞인 스텝 | 후발 prompt만 처리한 스텝 중 이전 조각 반환 100ms 뒤에도 decode가 있던 구간 |
|---|---:|---:|---:|---:|---:|
| `rag-c8` seed 1 | 1,006ms | 279ms | 394ms | 204ms | 199ms |
| `rag-c8` seed 2 | 655ms | 209ms | 319ms | 68ms | 190ms |
| 8개 동시 전송 | 2,093ms | 760ms | 1,198ms | 135ms | 412ms |
| 8개 50ms 간격 전송 | 1,952ms | 412ms | 1,327ms | 213ms | 404ms |

각 열은 요청별 측정의 독립 중앙값이다. 두 RAG seed에서 대기 중 남은 입력이 512토큰 이하였던 구간의 중앙값은 334ms와 63ms였다. 스케줄러는 조각이 끝날 때 `ready_since_ns`를 갱신하며, 다음 선택은 이 시각과 aging 조건을 사용한다. 입장 순서와 반환 후 경과 시간은 추적에 기록됐다. 실제 우선순위와 aging 적용 여부는 다음 선택 시점의 `ready_since_ns`와 선택 사유를 기록해 확인한다. [prefill 선택](../../crates/riley-scheduler/src/scheduler.rs#L1745) · [조각 완료 처리](../../crates/riley-scheduler/src/scheduler.rs#L2568)

`d3d8fe6e5`의 스케줄러를 사용한 CPU 재현에서 먼저 입장한 요청, 나중에 입장한 요청, 먼저 입장한 요청 순으로 prefill을 선택했다. 각 스텝은 decode 1행과 prefill 15행을 처리했다. 각 요청의 대기 시간은 aging 임계값 100ms 미만이었고, 먼저 입장한 요청은 자신의 첫 조각을 commit한 뒤 한 차례 양보하고 다음 조각을 이어갔다. 이 실행은 모델 없이 스케줄러의 조각 단위 순서를 확인한다. [재현 테스트](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/scheduler-turn-repro/prefill_turn_repro.rs) · [실행 기록](../../benchmarks/evidence/20261119-w3-rag-profile/request-stages/scheduler-turn-repro/receipt.json)

후속 대조에서는 100ms 미만 기다린 새 prompt보다 진행 중인 prefill을 먼저 고려했다. 100ms를 지난 요청은 기존 순서로 경쟁하며 decode 선택과 기존 aging override를 적용했다. 격리 패치의 스케줄러 CPU 테스트 146건과 CUDA 기능을 켠 Rust 타입 검사가 통과했다. A100 GPU 7에서 같은 진단 바이너리의 옵션만 바꿔 위 다섯 부하와 32개 동시 짧은 요청을 실행했다. 두 조건의 측정 320쌍에서 출력이 일치했고, 별도 장문 8개는 C1·C8과 두 조건 모두에서 입력·생성 토큰 ID가 일치했다. 두 서버는 코드 0으로 종료했고 같은 홀더의 20GiB 예약으로 복귀했다. [실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-continuation/campaign.sh) · [CPU 검사 원본](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-continuation/scheduler-tests.log.gz) · [GPU 원본과 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-continuation/result/SHA256SUMS)

| 부하 | TTFT p95 변화 | decode TPOT p50 변화 | 처리량 변화 |
|---|---:|---:|---:|
| `rag-c8` seed 1 | −0.28% | −0.01% | +0.16% |
| `rag-c8` seed 2 | +2.17% | +1.44% | +0.23% |
| 짧은 `C32` | +0.21% | +0.03% | −0.07% |

이 설정은 채택하지 않았다. `rag-c8` seed 2의 TTFT 중앙값은 25.41% 줄었으나 p95는 2.17% 늘었고, decode TPOT가 1% 회귀 기준을 넘었다. 두 seed의 결과는 각 조건을 한 번씩 실행한 근거다. [정확성·성능 대조](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-continuation/result/summary.json) · [집계 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/compare_prefill_continuation.py)

### 완료 가능한 prefill 우선 후보

요청별 `prefill_start`와 `prefill_return`을 다시 읽어 각 요청의 남은 토큰을 대조했다. 두 `rag-c8` seed에서 선두 요청은 이번 스텝에서 끝낼 수 없지만 다른 요청은 끝낼 수 있는 경우가 각각 180스텝 중 41개와 168스텝 중 37개 있었다. 모두 decode가 진행 중이었다. 계측 시각으로 readiness를 근사한 후보 선별 집계이며 실제 우선순위 변경 횟수는 별도 실행에서 확인한다. [추적 집계](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-finish/finishing-opportunities.json) · [집계 도구](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-finish/scan_finishing_opportunities.py)

이 후보는 기존 aging override 직후의 decode 우선 스텝에서만 동작한다. 기존 decode 행을 먼저 배정하고, 남은 예산에 들어오는 완료 가능한 요청 한 개를 앞당긴다. 다음 aging override는 기존 순서를 사용한다. 격리한 스케줄러 CPU 테스트 146건과 진단 표지를 포함한 최종 단위 테스트 37건을 통과했다. CUDA 기능 구성은 로컬 Rust 타입 검사까지 확인했다. [진단 패치·실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-finish/campaign.sh) · [로컬 검사 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-finish/local-checks.sha256)

CUDA 12.4 SM80 진단 바이너리로 A100 6번에서 같은 부하 여섯 개를 옵션 off·on 순으로 한 번씩 실행했다. 후보 설정 로그에서 실제 순서 변경을 확인했다. 오류·429는 0건이었고 측정 320쌍의 출력 텍스트·완료 토큰 수·종료 사유가 일치했다. 별도 장문 8개는 C1·C8과 두 설정 모두에서 토큰 ID가 일치했다. 두 서버가 코드 0으로 종료했고 동일 홀더가 각 실행 뒤 20GiB를 다시 예약했다. 원본은 A100 호스트에서 검증했고 집계에 바이너리 및 원본 파일의 체크섬을 기록했다. [GPU 비교 결과](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-finish/result/summary.json) · [검증기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/compare_prefill_finish.py)

| 부하 | TTFT p95 변화 | decode TPOT p50 변화 | ITL p99 변화 | 처리량 변화 |
|---|---:|---:|---:|---:|
| `rag-c8` seed 1 | −7.43% | +0.93% | +1.78% | −0.26% |
| `rag-c8` seed 2 | +20.45% | −1.99% | +2.50% | −0.47% |
| `wave-sync` | −0.88% | +10.76% | +87.80% | +0.09% |
| `wave-50ms` | −0.02% | +7.04% | +87.68% | +0.06% |
| 짧은 `C32` | −0.15% | −0.34% | −0.87% | +0.36% |

이 후보는 채택하지 않았다. 기존 자동 판정은 짧은 `C32`의 TTFT·처리량과 `C32`·두 `rag-c8`의 TPOT를 통과시켰다. 실제 긴 요청의 TTFT p95는 seed마다 방향이 달랐고, 두 동시 유입 조건에서 decode TPOT와 ITL 꼬리가 모두 늘었다. 단일 실행 쌍에서 관측한 변화이므로 후속 후보 검증은 실행 순서를 뒤집어 반복하고 두 동시 유입 조건의 decode 지연도 판정에 포함한다.

요청별 전송·첫 조각·마지막 조각 시각을 같은 요청끼리 비교했다. `wave-sync` 40개 중 24개, `wave-50ms` 40개 중 27개에서 TPOT가 늘었다. 묶음별 TPOT 변화 중앙값은 각각 5개 중 3개와 5개 전체에서 양수였다. 두 조건의 실행 간 실제 전송 간격 범위는 각각 0.171–0.231ms와 348.15–351.15ms였다. `rag-c8` seed 2에서 후보의 p95 이상 네 요청은 모두 같은 기준 요청보다 TTFT가 늘었고, 증가량 중앙값은 507.9ms였다. 요청별 원본에 토큰 간 개별 간격이 보존되지 않아 ITL p99 상승을 특정 스텝에 연결하려면 추가 계측이 필요하다. [요청 비교 집계](../../benchmarks/evidence/20261119-w3-rag-profile/prefill-finish/result/request-latency.json) · [재계산 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/analyze_prefill_latency.py)

## 14B 혼합 스텝의 행 예산

요청별 계측에서 decode 요청 한두 개와 큰 prefill 조각을 함께 실행한 구간이 긴 요청의 대기와 겹쳤다. 이를 확인하기 위해 Qwen2.5-14B-Instruct에서 decode가 있는 스텝의 총 행을 512개로 제한하는 후보를 만들었다. 512는 기존 배치 shape 구간 중 일반 예산 256과 최대 예산 1,024 사이에 있다. prefill 전용 스텝은 1,024행까지 사용할 수 있고, 기존 정책이 256행을 선택하면 그 예산을 적용한다. [원인 계측](#riley-14b의-요청별-대기와-prefill-계측) · [진단 패치](../../benchmarks/evidence/20261119-w3-rag-profile/mixed-step-budget/source.patch.gz)

`d3d8fe6e5` 아카이브에 패치를 적용한 CPU 스케줄러 시험에서 147건이 통과했다. 기본값과 제한 설정에서 prefill 전용 및 혼합 스텝을 비교했고, 오래 대기한 prefill의 aging 경로와 atomic prefill 충돌도 확인했다. 패치는 전용 진단 빌드에서 환경 변수 `RILEY_W3_MIXED_STEP_BUDGET`으로 활성화한다. [CPU 검증 기록](../../benchmarks/evidence/20261119-w3-rag-profile/mixed-step-budget/cpu-preflight.json)

배포 이미지의 30분 운영 시험 뒤 A100 7번에서 같은 진단 바이너리를 기준 → 후보 → 후보 → 기준 순으로 실행하도록 등록했다. 두 RAG seed, 8개 동시 전송과 50ms 간격 전송, 짧은 C32에서 첫 토큰 지연과 decode TPOT·ITL을 비교한다. 같은 입력의 출력 지문과 별도 C1·C8 장문 토큰 ID도 대조한다. [실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/mixed-step-budget/campaign.sh) · [결과 검사기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/compare_mixed_step_budget.py)

네 실행에서 각각 요청 320건을 오류·429 없이 완료했다. 두 순서로 비교한 출력 640쌍에서 텍스트·생성 토큰 수·종료 사유가 일치했다. 별도 장문 8개는 네 실행 모두에서 C1·C8 입력과 생성 토큰 ID가 일치했다. 네 서버가 정상 종료했고, 각 실행 뒤 A100 7번은 같은 홀더의 20GiB 예약으로 복귀했다. 서버에서 계산한 집계를 원본으로 다시 계산해 바이트 일치를 확인했다. [검산 결과](../../benchmarks/evidence/20261119-w3-rag-profile/mixed-step-budget/result/summary.json) · [네 실행 원본](../../benchmarks/evidence/20261119-w3-rag-profile/mixed-step-budget/result/raw-results.tar.gz)

각 셀은 같은 실행 쌍에서 기준 대비 512행 후보의 변화율이다. 표 안의 두 수치는 기준 → 후보 순서와 후보 → 기준 순서다.

| 부하 | TTFT p95 | decode TPOT p50 | ITL p99 | 처리량 |
|---|---:|---:|---:|---:|
| `rag-c8` seed 1 | +13.06% / +12.22% | −3.68% / −3.63% | −47.37% / −47.75% | +2.23% / +2.43% |
| `rag-c8` seed 2 | +9.80% / +12.30% | −6.04% / −6.53% | −46.20% / −45.64% | +2.48% / +1.92% |
| 8개 동시 전송 | +3.48% / +3.30% | +0.12% / +0.32% | −45.59% / −44.97% | −1.29% / −1.12% |
| 8개 50ms 간격 전송 | −1.59% / +3.13% | +4.67% / −1.43% | −45.89% / −44.08% | −1.42% / −1.51% |
| 짧은 `C32` | −0.13% / −0.11% | +0.18% / +0.26% | −0.45% / +0.10% | −0.23% / −0.08% |

512행 제한은 미채택으로 판정했다. 두 RAG seed의 첫 토큰 p95가 두 실행 순서 모두에서 악화됐고, 50ms 간격 유입 한 쌍이 decode TPOT 회귀 1% 조건을 넘었다. 같은 50ms 조건의 기준군끼리도 TPOT p50이 6.25% 달라 회귀 크기의 반복성은 제한적이다. RAG 두 seed에서 각 조건의 p95 이상 요청 네 개는 모두 3,471–3,528토큰의 긴 입력에 속했다. 첫 토큰 꼬리를 우선하는 후속 대조에서는 이 길이대 요청의 prefill 사이 대기를 스텝별로 연결해 확인한다. [요청별 집계](../../benchmarks/evidence/20261119-w3-rag-profile/mixed-step-budget/result/request-latency.json) · [입력 길이·유입 구간 재계산 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/analyze_mixed_step_requests.py)

#### 실제 스텝의 prefill 선택과 aging 검증

격리한 `d3d8fe6e5` 계측 빌드에서 prefill 대기 시작 시각과 순서, 토큰 예산, decode 행 수, aging 적용 대상, 실제 실행 계획의 prefill 행 수를 같은 iteration ID로 기록한다. 스케줄러 ID를 SSE 응답 ID와 연결하므로 장문 요청의 각 조각이 언제 선택됐고 어떤 요청이 그사이에 실행됐는지 확인할 수 있다. 검산기는 계측된 시각으로 선택을 재계산하고 스케줄러 기록, 엔진의 실행 직전 기록과 요청별 prefill 구간을 맞춘다. [진단 패치](../../benchmarks/evidence/20261119-w3-rag-profile/scheduler-choices/source.patch.gz) · [검산기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/summarize_scheduler_choices.py)

CPU에서는 기본 설정으로 스케줄러 테스트 146건이 통과했다. 계측을 켠 두 시험의 8개 계획에서 조각 완료 뒤 대기 순서 변경과 aging override 뒤 decode 우선 복귀를 재계산했다. A100 7번은 W5 RSS 시험과 W4 장문 비교가 끝난 뒤 같은 Qwen2.5-14B-Instruct 부하 다섯 개를 기준 실행과 계측 실행으로 대조하도록 등록했다. [CPU 검사와 GPU 실행 등록](../../benchmarks/evidence/20261119-w3-rag-profile/scheduler-choices/preflight.json)

A100 7번의 두 실행에서 같은 합성 요청 224쌍의 출력 텍스트 해시·생성 토큰 수·종료 사유가 일치했다. 두 실행 모두 오류·429는 0건이었고 같은 홀더의 20GiB 예약으로 복귀했다. 계측 실행의 759개 prefill 스텝에서 스케줄러 선택·실행 행·CUDA 시간을 대조했다. 이 중 188개 스텝에서 aging이 대기 요청을 먼저 배정했다. 계측을 켠 한 번의 비교에서 두 RAG seed의 TTFT p95는 각각 −15.29%와 +8.81%였다. 이 실행의 성능 수치는 선택 원인을 추적하는 데 사용한다. [정확성·종료 검산](../../benchmarks/evidence/20261119-w3-rag-profile/scheduler-choices/result/verified-summary.json)

TTFT가 p95 이상인 요청의 조각 사이 대기와 겹친 CUDA 시간을 집계했다. 나중에 입장한 요청의 대기 시작 시각이 더 앞서 선택된 경우를 실제 스케줄러 시각으로 확인했다. CUDA 시간은 함께 실행된 decode도 포함하며, 한 스텝을 기다린 요청별로 한 번씩 집계했다.

| 부하 | 전체 CUDA 시간 중앙값 | 나중에 입장하고 더 일찍 준비된 prefill이 실행된 시간 중앙값 | 남은 512토큰 이하가 기존 일반 prefill 예산에 들어가는 TTFT 꼬리 요청 |
|---|---:|---:|---:|
| `rag-c8` seed 1 | 604.84ms | 331.05ms | 1/4 |
| `rag-c8` seed 2 | 668.39ms | 529.35ms | 2/4 |
| 8개 동시 전송 | 1,962.10ms | 1,001.87ms | 0/2 |
| 8개 50ms 간격 전송 | 1,956.34ms | 1,543.52ms | 1/2 |

마지막 열은 기존 decode 행과 aging으로 우선 배정된 요청을 그대로 두고, 나중에 입장한 요청이 선택된 스텝의 일반 prefill 행을 재배정해 앞선 요청을 완료할 수 있는 경우다. 관측한 실행 계획에서 계산한 범위다. 두 RAG seed의 TTFT 꼬리 요청 8개 중 3개가 이 조건에 해당했다. [대기 시간·완료 가능 조건 집계](../../benchmarks/evidence/20261119-w3-rag-profile/scheduler-choices/result/wait-attribution.json) · [CUDA 시간 연결 도구](../../benchmarks/evidence/20261119-w3-rag-profile/tools/attribute_scheduler_wait.py)

#### 1,024행 스텝의 계산 비용

실제 prefill 스텝 759개 중 627개가 활성 토큰 1,024행을 채웠다. 이 스텝이 계측된 prefill CUDA 시간의 89.4%를 차지했다. TTFT 꼬리 요청이 조각 사이에 대기한 고유 스텝은 두 RAG seed에서 17개와 15개, 두 동시 유입 조건에서 각각 29개였다. 이 스텝은 모두 1,024행이었다. 16행 간격으로 버킷 하나를 추가하는 계산에서 784행이 전체 추적의 36스텝에 영향을 줄 수 있었고, 위 TTFT 꼬리 대기 구간에서 영향을 받는 스텝은 0개였다. 이 계산은 설정된 버킷 기준이며 CUDA에서 실제 준비에 성공한 형상과 지연은 별도 확인한다. [활성 행과 버킷 검산](../../benchmarks/evidence/20261119-w3-rag-profile/scheduler-choices/result/bucket-audit.json) · [검산기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/inspect_prefill_buckets.py)

다음 단일 변수는 A100의 Qwen2.5-14B QKV `M=1024, K=5120, N=7168` GEMM이다. 고정 소스의 128행 기준 알고리즘은 split-K 1이고 513–1,024행 후보표의 엔트리는 split-K 3이다. 실험은 첫 합성 입력을 서버의 준비 시점 BF16 바이트 검사와 같은 방식으로 생성한다. 별도 seed에서도 기준 결과가 일치하는 알고리즘만 남겨, 준비 검사와 같은 방식으로 정한 기준에 대해 세 번씩 순서를 교대로 측정한다. 실제 서빙 입력의 출력·TTFT·decode 회귀는 후속 대조에서 확인한다. CUDA 12.4 진단 바이너리를 빌드했고 기존 W4 시험이 끝나면 A100 7번에서 실행한다. 재현 시 `qkv1024.patch.gz`를 같은 디렉터리에서 `qkv1024.patch`로 풀고 `dispatch.sha256`을 검증한다. [고정 소스·빌드 검증](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/preflight.json) · [실행 등록](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/registration.json)

A100 7번의 합성 입력 두 개에서 기존 wide 후보표의 split-K 3 알고리즘은 128행 기준 알고리즘과 1,024행 BF16 결과가 달랐다. 두 입력 모두 기준과 같은 바이트를 만든 구성은 102개였다. 가장 일관되게 빠른 구성은 `{21, 20, 11, 1, 0}`이었다. 같은 입력과 GPU에서 세 차례 순서를 교대로 측정한 결과 QKV 시간 중앙값은 404.80µs에서 333.94µs로 줄었다. 두 실행 순서의 쌍별 비율 중앙값은 0.8251이었다. 이 수치는 단일 QKV GEMM의 합성 시험 결과다. 실제 서빙에서 같은 1,024행에만 적용해 응답 바이트·TTFT와 decode 지연을 대조할 후보로 선정했다. [구성별 바이트 판정과 시간](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/result/comparison.json)

격리한 서빙 빌드는 A100과 cuBLASLt 12.4.5에서 기존 기준에 결속된 `M=1024, K=5120, N=7168` GEMM을 검사한다. 런타임 BF16 바이트 동등성 검사에 통과한 경우 후보를 선택하고, 판정 로그가 기대와 다르면 실험 검산이 실패한다. A100 7번에 기준 → 후보 → 후보 → 기준 순서를 등록했다. 기존 C1·C8과 두 도착 조건, 짧은 C32의 요청별 응답을 비교하고 별도 장문 8개의 C1·C8 토큰 ID도 맞춘다. [서빙 대조 등록](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/registration.json) · [정확성·성능 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/compare_qkv_serving.py)

선택 조건은 모델명과 관계없이 GEMM 형상에 적용된다. 프로젝트 캐시에 있는 Coder-14B, DeepSeek-R1-Distill-Qwen-14B, Qwen3-14B, Qwen2.5-32B-Instruct의 단일 GPU 병합 QKV도 같은 `K=5120, N=7168` 형상이다. TP=2에서는 모델의 query·KV head가 두 rank에 나뉘므로 rank당 병합 QKV의 `N`은 3,584다. 이 조사는 모델 설정과 실행 경로에 근거한 영향 범위다. 실제 서빙 대조에 사용한 모델은 Qwen2.5-14B-Instruct다. [설정 해시와 형상 계산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/shape-scope.json) · [TP rank 구성](../../crates/riley-model/src/ir.rs#L616) · [병합 QKV 준비](../../crates/riley-runtime/src/llama/forward.rs#L6806)


첫 기준군은 오류·429 없이 320건을 완료했다. 이전 혼합 스텝 시험의 기준군과 입력·출력 텍스트 해시·생성 토큰 수·종료 사유가 일치했다. 장문 8개의 C1·C8 토큰 ID도 이전 기록과 같았고 같은 GPU 홀더의 20GiB 예약을 복구했다. [첫 기준군 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/first-control-verification.json)

첫 후보 서버가 기준군과 같은 바이너리·cuBLASLt 12.4.5를 적재했다. 런타임 바이트 동등성 검사에 통과해 1,024행 QKV 후보를 실제로 선택했다. [후보 시작 시점의 검증](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/first-candidate-startup.json)

두 실행 순서의 비교를 완료했다. 각 실행은 같은 바이너리, 체크포인트, KV 41,216토큰으로 320건을 처리했다. 4회 총 1,280건에 오류·429가 없었고 각 조건의 응답 텍스트 지문, 입력·생성 토큰 수, 종료 사유가 일치했다. 별도 장문 8개의 C1·C8 토큰 ID도 네 조건에서 모두 같았다. 후보군 두 번의 런타임 선택을 검증했고 각 실행 뒤 동일한 홀더의 20GiB 예약을 복구했다. [전체 지표·입력 식별 해시·원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/comparison.json)

| 부하 | TTFT p95 변화, 기준 먼저 / 후보 먼저 | Decode TPOT p50 변화, 기준 먼저 / 후보 먼저 |
|---|---:|---:|
| `rag-c1` | −0.83% / −1.50% | +0.08% / +0.05% |
| `rag-c8-s1` | −2.80% / −2.26% | −0.63% / −0.76% |
| `rag-c8-s2` | −6.07% / +2.68% | +0.13% / −2.39% |
| `wave-sync` | −2.49% / −3.32% | −2.93% / −0.29% |
| `wave-50ms` | +1.64% / +1.76% | −5.34% / −3.04% |
| `short-c32` | +0.28% / +0.01% | +0.23% / −0.15% |

지연 변화는 각 순서에서 인접한 Riley 기준군 대비 후보군으로 계산했다. Decode TPOT 악화 1% 미만, 짧은 요청 TTFT p95와 처리량 악화 2% 미만 조건은 모두 통과했다. 첫 seed와 동시 유입의 p95는 두 번 모두 줄었다. 두 번째 seed에서는 기준군끼리도 p95가 8.51% 달랐고, 50ms 간격 유입에서는 p95가 두 번 모두 늘었다. RAG p95 개선의 일관성이 확인될 때까지 후보는 격리 시험에만 둔다. 이 비교의 판정 범위는 A100·cuBLASLt 12.4.5에서의 Qwen2.5-14B-Instruct 여섯 부하다. [지표별 기준군 변동과 가드레일](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/comparison.json)

요청 인덱스를 맞춰 첫 토큰 지연을 다시 계산했다. RAG 첫 seed에서 후보가 빨랐던 요청은 각각 64개 중 62개와 63개였다. 두 번째 seed에서 후보 p95는 두 실행 간 0.24ms 차이였고 기준군의 p95는 136.52ms 차이였다. 각 기준군의 느린 요청 인덱스도 달랐다. 이 대조에서 후보와 기준의 지연 부호가 바뀐 배경을 보여 주며, 다음 비교에서는 기준군 변동의 재현 여부도 함께 확인해야 한다. [요청별 지연 순위와 비교](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/tail-audit.json)

50ms 간격 유입에서 후보가 빨랐던 요청은 각각 40개 중 35개와 37개였다. 다만 다섯 번째 유입 묶음의 두 번째 요청은 3,503토큰 입력에서 첫 토큰까지 530~533ms 더 걸렸다. 각 실행에서 같은 묶음의 첫 요청 이후 50~51ms에 전송됐고, 기준군과 후보군의 전체 전송 폭은 약 350ms였다. 다섯 번째 묶음을 제외한 민감도 계산에서 p95 변화는 두 순서 모두 감소로 바뀌었다. 전체 표본의 p95가 채택 판정에 적용되며, 요청별 대기와 GPU 스텝을 함께 계측할 경우 이 유입 묶음에서 확인을 시작한다. [전송 간격과 p95 민감도](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/tail-audit.json)

이전 계측 빌드의 `wave-50ms` 요청 40개를 QKV 서빙 대조의 네 실행과 요청·응답 해시로 맞췄다. 느려졌던 다섯 번째 묶음의 두 번째 요청은 이전 기준 계측에서 첫 토큰까지 2,583.30ms였고, 1,831.02ms가 자체 prefill 조각 사이에 있었다. 그중 1,829.16ms는 다른 prefill 스텝과 겹쳤고 1,550.59ms는 나중에 입장한 요청만 처리한 스텝이었다. 이어지는 세 번째 요청은 prefill 조각 사이에 2,151.84ms를 기다렸다. 이 수치는 이전 기준 빌드에서 관측한 대기 구성이며 후보의 추가 지연 위치는 아래 대조에서 확인한다. [같은 요청의 단계 대조](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/archived-wave-timing.json)

후속 대조는 같은 바이너리에서 요청 계측을 두 조건 모두 켜고 1,024행 QKV 선택만 교대로 바꾼다. `wave-50ms`와 `rag-c8-s2`에서 비계측 결과와 요청·응답이 맞는지 확인하고, 50ms 묶음의 두 요청 및 각 조건의 p95 꼬리 요청을 단계별로 비교한다. 현재 후보 소스에 이전 계측 패치를 적용할 수 있는지 확인했고, 기존 단계 기록으로 검산기의 요청 결속과 후보 선택 실패 처리를 검사했다. RSS와 장문 시험 이후 A100 7번에서 시작하도록 등록했다. [실행 순서와 입력 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/registration.json) · [단계 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/compare_traced_qkv.py)

#### 50ms 간격 유입의 추가 지연 위치

A100 7번에서 네 비교군을 완료했다. 이전 비계측 비교의 해당 입력과 대조한 416건에서 요청 해시, 출력 텍스트 해시, 토큰 수와 종료 사유가 일치했다. 두 후보 실행에서 알고리즘 선택 표지를 확인했고, 네 서버 모두 정상 종료 후 같은 홀더의 20GiB 예약을 복구했다. [실행 시각·출력 대조와 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/stage-attribution.json)

| 계측 항목 | 기준 먼저 | 후보 먼저 |
|---|---:|---:|
| `rag-c8-s2` TTFT p95 변화 | +5.40% | −2.99% |
| `wave-50ms` TTFT p95 변화 | +1.47% | +2.11% |
| `wave-50ms` 41번 요청 TTFT 변화 | +527.67ms | +541.04ms |
| 같은 요청의 prefill 조각 사이 대기 변화 | +380.57ms | +392.26ms |
| 같은 요청의 prefill 스텝 실행 구간 변화 | +154.60ms | +157.31ms |

3,503토큰인 41번 요청은 두 쌍 모두 자체 prefill이 4스텝에서 6스텝으로 늘었다. 자체 조각 사이의 다른 prefill은 14스텝에서 17스텝으로, 나중에 입장한 요청만 실행한 스텝은 12개에서 15개로 늘었다. 같은 요청의 첫 두 1,024행 스텝은 두 쌍에서 GPU 시간이 각각 −3.26·−1.14ms와 −3.22·+0.66ms 변했다. GPU 시간은 스텝 전체의 측정값이다. 개별 QKV 호출 시간은 별도 계측이 필요하다. `rag-c8-s2`는 순서마다 p95 변화 방향이 달랐다. 기본값 적용은 계속 보류한다. [요청별 실행 행·대기 구간](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/stage-attribution.json) · [스텝 구성 재검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/attribute_qkv_steps.py)

#### 다섯 번째 묶음의 입장 시점과 prefill 순서

같은 네 실행의 다섯 번째 묶음에서 세 번째 스텝은 모두 41번 요청의 첫 1,024행을 실행했다. 45번 요청의 엔진 명령 전송 시각은 기준군에서 이 스텝 시작보다 2.09–2.70ms 앞섰고, 후보군에서는 3.44–3.54ms 뒤였다. 기준군은 스텝 직전 45번을 입장시켰고, 후보군은 해당 스텝이 반환된 직후 입장시켰다. 일곱 번째 스텝까지 선택된 요청과 행은 네 실행에서 같았다. 여덟 번째·아홉 번째 스텝의 순서는 기준군에서 45→41, 후보군에서 41→45였다. [시각·순서와 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/wave-admission.json)

CUDA 빌드와 `scheduler.rs`·`config.rs` 해시가 같은 스케줄러로 첫 9개 스텝을 CPU에서 재생했다. 네 실행의 요청 순서와 선택된 토큰 범위가 정확히 일치했다. 각 실행에서 45번의 입장 시점만 세 번째 스텝 경계 반대편으로 옮기자 여덟 번째·아홉 번째 선택도 두 방향에서 뒤집혔다. 별도 GPU 세션 없이 기존 기록으로 확인한 원인 범위는 첫 선택 차이까지다. [스케줄러 재생 코드](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/replay_wave.rs) · [기록 대조 도구](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/compare_wave_admission.py)

뒤이은 41번의 세 번째 prefill은 기준군에서 1,022행을 받았다. 후보군의 해당 스텝은 44번의 완료 prefill 778행과 decode 1행을 함께 처리해 41번에 245행을 배정했다. QKV 후보의 채택은 입장 경계가 달라지는 부하에서도 TTFT p95와 decode 회귀 기준을 함께 만족하는지에 달려 있다.

#### 공통 명령 송신 시각에서의 스케줄러 대조

네 실행에서 첫 요청이 입장한 시점을 각각 0으로 맞췄다. 나머지 일곱 요청의 입장 기록과 9개 prefill 스텝의 시각을 유지하고, 45번 명령 송신 시각만 244–255ms 사이의 같은 61개 값으로 바꿨다. 45번은 송신 후 44번보다 늦게 입장하며, 실행 중인 스텝이 끝나면 다음 계획에서 처리된다고 가정했다. 기존 36개 계획을 정확히 재현했고 두 CPU 빌드에서 244개 대조의 결과가 일치했다. [시각별 대조와 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/boundary-sweep.json) · [재현 스크립트](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/run_wave_boundary_sweep.sh)

| 공통 명령 송신 시각 | 기준군의 8→9번째 prefill | 후보군의 8→9번째 prefill |
|---|---|---|
| 245ms | 45번 → 41번 | 45번 → 41번 |
| 249ms | 45번 → 41번 | 41번 → 45번 |
| 253ms | 41번 → 45번 | 41번 → 45번 |

두 군이 갈라지는 구간은 기준 먼저 실행에서 246.096–252.734ms, 후보 먼저 실행에서 246.105–251.969ms였다. 각각 약 6.64ms, 5.86ms 폭이다. 이 구간의 249ms에서는 같은 명령 시각으로도 측정된 스텝 경계에 따라 두 군의 순서가 달라졌다. 이 대조는 후속 GPU 시험에서 경계 안팎의 유입 조건을 정할 근거다. TTFT와 채택 여부는 실제 부하로 판정한다.

#### 실제 요청의 도착 시각 대조

A100 7번에서 같은 Qwen2.5-14B 추적 바이너리로 45번 요청만 기존 도착 시각에서 −12ms, 0ms, +12ms 이동하는 시험을 등록했다. 기준 → 후보 → 후보 → 기준 순서로 세 조건을 실행해 총 480건을 측정하고, 뒤의 두 실행에서는 조건 순서를 뒤집는다. CPU 사전 검증에서 각 조건의 48개 도착 일정과 측정 요청 40개의 동일한 입력을 확인했고, 기존 GPU 로그로 양쪽의 prefill 순서를 재현했다. [등록 및 사전 검증 영수증](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/phase-gpu/registration.json)

W5 배포 이미지 부하와 조건부 장문 검증 절차가 종료되고 GPU 6·7의 기존 20GiB 예약을 확인한 뒤 시작한다. 각 실행에서 GPU 7의 예약을 시험에 넘겼다가 복구한다. 집계기는 실제 엔진 송신과 입장 시각을 41번 요청의 첫 prefill에 맞춰 비교하고, 예정한 조기·후기 조건이 실제로 성립했는지와 TTFT p95·TPOT·ITL을 확인한다. [GPU 실행 스크립트](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/phase-gpu/campaign.sh) · [집계기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/phase-gpu/verify_phase_gpu.py)

네 실행에서 측정 요청 480건을 오류와 429 없이 완료했다. 이전 같은 요청의 입력·응답 텍스트 해시, 입력·생성 토큰 수와 종료 사유가 일치했다. 각 서버가 정상 종료했고 GPU 7은 동일 홀더의 20GiB 예약으로 복귀했다. GPU 6은 같은 예약을 유지했다. [실측 집계와 원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/phase-gpu/verified-comparison.json)

두 실행 순서에서 45번의 입장 시점이 41번의 첫 prefill 앞뒤로 어떻게 바뀌는지 확인했다. 지연 변화는 각 순서의 인접 기준군 대비 QKV 후보이며, 기준 먼저 / 후보 먼저 순이다.

| 45번 도착 오프셋 | 기준 / 후보의 8→9번째 prefill | TTFT p95 변화 | Decode TPOT p50 변화 |
|---|---|---:|---:|
| −12ms | 45→41 / 45→41 | −2.50% / −2.59% | −2.88% / −2.76% |
| 0ms | 45→41 / 41→45 | +2.15% / +1.77% | −5.50% / −2.81% |
| +12ms | 41→45 / 41→45 | −2.44% / −2.45% | −2.59% / +0.62% |

기존 시각에서 후보의 41번 첫 토큰 지연은 537–541ms 늘었다. 그중 389–391ms는 prefill 조각 사이의 대기 증가였다. 두 조건의 prefill 순서가 같았던 앞뒤 도착 시각에서는 해당 요청의 첫 토큰 지연도 줄었다. 이 결과는 한 seed의 50ms 간격 부하에서 입장 경계에 따라 QKV 후보의 p95 결과가 바뀐다는 점을 보여 준다. RAG p95의 일관성 기준이 남아 있어 기본값 적용은 보류한다.

#### 명령 송신 시각을 사용한 우선순위 재생

위 GPU 시험의 12개 조건에서 다섯 번째 묶음의 요청 인덱스·입력 길이·상대 송신 및 입장 시각을 추출했다. 기존 스케줄러와 별도 계측 빌드에서 각각 GPU 기록의 첫 9스텝, 총 108개 계획을 다시 선택했다. CPU 재생은 실제 입장 시각과 스텝 시간을 입력으로 쓰고, 즉시 입장한 요청의 첫 prefill 우선순위 시각에 보정을 적용한다. [시각 근거와 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/command-clock-cpu/cpu-ready-inputs.json) · [CPU 재현 스크립트](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/command-clock-cpu/run_ready_replay.sh)

명령 송신 시각을 그대로 쓰면 12개 조건 모두에서 두 번째 스텝의 선택이 40번 요청에서 41번 요청으로 바뀌었다. 스텝 2에서 처음 순서가 바뀌는 최소 보정 폭은 76.879µs였다. 이 경계보다 작은 보정 폭에서는 선택 순서가 달라질 수 있는 시점을 검산했으며, 원래 도착 시각의 두 쌍에서 기존 8→9번째 순서 차이가 그대로 남았다. 따라서 이 규칙은 앞선 스텝을 유지하면서 관측한 입장 경계만 해소하는 후보에서 제외했다. 판정 범위는 기록한 실행 창을 고정한 prefill 재생이다. 첫 계획이 달라진 뒤의 실제 지연은 GPU에서 별도로 측정해야 한다. [보정 폭별 재계산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/command-clock-cpu/command-clock-comparison.json)

#### 두 RAG seed의 고정 도착 재생

Qwen2.5-14B-Instruct의 닫힌 부하에서 기록한 기준군과 QKV 후보의 상대 송신 시각을 seed별로 가져왔다. 각 시각표에 요청 64건이 들어 있다. A100 7번에서 같은 바이너리와 KV 2,576블록으로 기준 → 후보 → 후보 → 기준 순서로 실행했다. 네 시각표를 매번 요청 완료와 독립적으로 재생하고, 뒤 두 실행에서는 seed별 시각표 순서도 뒤집었다. [실행 순서와 입력 검증](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/fixed-rag-arrivals/campaign.sh) · [도착 시각 재생 코드](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/fixed-rag-arrivals/fixed_w3.py)

측정 1,024건을 오류와 429 없이 완료했다. 입력·응답 텍스트 해시, 입력·출력 토큰 수, 종료 사유가 기존 요청과 모두 일치했다. 별도 C1·C8 생성 ID 검사도 이전 네 실행을 재현했다. 후보 두 실행에서 cuBLASLt `120405`의 1,024행 계획 선택을 확인했다. 실제 송신 오차 p95는 각 실행에서 최대 2.24ms였고, 기준과 후보의 송신 시각 차이 p95는 각 비교에서 최대 1.51ms였다. 네 서버가 코드 0으로 종료됐고 GPU 7의 동일한 20GiB 예약이 매번 복구됐다. [집계·선택 기록·체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/fixed-rag-arrivals/verified-summary.json)

아래는 기준 대비 후보의 변화율이다. 각 칸에 기준 먼저 / 후보 먼저 순서의 결과를 적었다.

| seed | 시각표 출처 | TTFT p95 변화 | TPOT p50 변화 |
|---:|---|---:|---:|
| 1 | 기준군 | −10.30% / −10.12% | −9.35% / −4.37% |
| 1 | QKV 후보 | −4.69% / −3.54% | −9.23% / −5.19% |
| 2 | 기준군 | −4.28% / −3.62% | −4.13% / −3.85% |
| 2 | QKV 후보 | −7.02% / −12.88% | −7.30% / −7.17% |

네 시각표와 두 실행 순서에서 TTFT p95 및 TPOT p50이 모두 줄었다. 이 결과의 대상은 두 seed에서 수집한 도착 시각표다. 기존 50ms 간격 시험에는 TTFT p95가 증가한 조건이 남아 있어, 대상 모델의 다른 도착 분포에서 재현성과 회귀 기준을 확인할 때까지 후보는 격리 시험에 둔다. [독립 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/fixed-rag-arrivals/verify_fixed.py) · [원격 집계 절차](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/fixed-rag-arrivals/fetch-summary.sh)

#### 같은 실행의 요청 대기와 CUDA 커널 대조

A100 7번에서 Qwen2.5-14B-Instruct, KV 2,576블록과 기존 추적 바이너리를 고정해 기준과 1,024행 QKV 후보의 요청 단계·호스트 단계·Nsight 커널을 함께 기록했다. RAG C8 seed 2와 50ms 유입에서 측정 208건의 요청 지문·출력·토큰 수·종료 사유가 앞선 요청 단계 단독 계측과 일치했다. 초기 후보군은 종료 중 감시기가 실험 서버의 PID를 외부 프로세스로 판정해 정상 종료 영수증이 없었다. 시작 시각을 확인한 동일 PID가 종료되는 구간만 처리하도록 감시기를 보완해 후보군만 재실행했다. 두 유효 실행은 정상 종료했고 GPU 6·7은 각각 기존 20GiB 예약으로 복귀했다. [실행 영수증과 원본 체크섬](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/joint-clock/verified-summary.json) · [재실행](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/joint-clock/retry/campaign.sh)

클라이언트 요청 시작·첫 콘텐츠 수신의 하한과 상한을 서버 요청 단계와 맞췄다. 클라이언트의 NVTX 표식 두 개로 Nsight 시간축을 연결했다. 결합한 오차 구간 폭은 네 측정에서 0.50–0.56ms였고, 여러 대기 구간의 경계에 이를 적용한 시간 범위는 p95 이상 표본에서 최대 3.45ms였다. 표는 p95 이상 요청이 자신의 prefill을 실행하지 않고 기다린 시간 중 GPU 커널과 겹친 시간의 중앙값이다. GEMM과 attention은 같은 GPU를 공유하는 모든 요청의 연산을 포함한다. KV 열은 커널 이름으로 확인한 RoPE·KV 쓰기 융합 커널 시간이다. [시간축과 커널 분류 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/joint-clock/verify_joint.py)

| 부하·실행 | 대기 중 GPU 전체 (ms) | GEMM (ms) | Attention (ms) | RoPE·KV 쓰기 (ms) |
|---|---:|---:|---:|---:|
| RAG C8 seed 2 기준 | 1,008 | 832 | 122 | 13.1 |
| RAG C8 seed 2 후보 | 1,232 | 1,001 | 162 | 16.4 |
| 50ms 유입 기준 | 2,364 | 1,923 | 313 | 30.7 |
| 50ms 유입 후보 | 2,387 | 1,931 | 323 | 31.8 |

50ms 유입의 동일한 41번 요청에서 첫 토큰 지연은 551.14ms, prefill 조각 사이 대기는 399.48ms, 자기 prefill 실행 구간은 160.98ms 늘었다. 이 요청이 대기하는 동안 실행된 GPU 시간은 386.70ms 늘었고, GEMM과 attention에 각각 270.01ms와 94.79ms가 더 겹쳤다. 같은 서버의 호스트 스텝 평균을 128스텝 단위로 17개씩 집계한 중앙값은 plan 55.3→55.5µs, pack 29.7→29.6µs, submit 1.50→1.41ms였다. 호스트 표본은 각 프로세스 전체의 측정이다. 이 비교는 요청 선택과 다른 스텝의 GPU 실행이 만드는 대기를 우선 조사할 근거다. 요청 단계 단독 계측 대비 p95는 부하별로 0.40–3.61% 변해 이 수치는 병목 진단에 적용했다. [41번 요청과 호스트 단계 원본 집계](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/joint-clock/verified-summary.json)

같은 두 로그에서 실행별 120건의 워밍업·측정 입력 길이를 확인했다. 시작 시 shape를 준비한 48건은 실제 요청이 오기 전에 끝났다. 최대 입력 3,523토큰과 최대 출력 128토큰을 적용해 16토큰 블록으로 올림하면 요청당 229블록이다. 두 부하의 최대 동시성 8에서 예약 상한은 1,832블록으로 설정한 2,576블록보다 744블록 적었다. 활성 시퀀스 한도도 32였다. 이 실행의 대기는 KV 예약 상한이나 활성 수용량이 부족해서 생긴 것이 아니다. 블록 수용량을 낮춰 실제 압력을 가하는 비교가 남아 있다. [두 로그와 연결한 상한 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/joint-clock/kv-headroom.json) · [스케줄러 입장 조건](../../crates/riley-scheduler/src/scheduler.rs#L2737)

#### 실제 KV 예약 압력과 회수

A100 7번에서 Qwen2.5-14B-Instruct의 같은 진단 바이너리와 요청을 사용해 KV 한도만 2,576→768블록으로 바꿨다. 두 실행의 최대 활성 시퀀스는 32, 부하의 동시성은 8이었다. 앞선 RAG C8 seed 2와 50ms 유입의 요청 지문 및 입력 길이를 확인했다. 실행마다 측정 104건과 워밍업 16건이 오류·429 없이 완료됐으며 비교한 출력 텍스트 지문·생성 토큰 수·종료 사유가 104/104건 일치했다. [프로세스 및 요청별 검산](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/verified-summary.json)

스케줄러의 제출, 대기열 선두의 입장 검사 실패, 입장, 반환을 요청별로 연결했다. 입장 검사 실패는 모든 경우에 `예약 중 블록 + 해당 요청의 예약량 > 설정 한도`였고 활성 시퀀스에는 여유가 있었다. 기준군의 측정 요청 중 대기는 0건이었다. 768블록 설정에서는 대기열에 들어갔다가 앞 요청의 반환 후 입장한 경우도 있어, 대기 건수와 선두에서 KV 부족이 직접 기록된 건수를 함께 집계했다.

| 부하 | 768블록군 대기 / KV 부족 확인 / 전체 | 대기 요청의 입장 대기 중앙값 | 첫 KV 부족 표식부터 입장까지 중앙값 | 첫 토큰 p95, 2,576 → 768블록 |
|---|---:|---:|---:|---:|
| RAG C8 seed 2 | 64 / 49 / 64건 | 4.307초 | 0.530초 | 1.766 → 8.673초 |
| 50ms 유입 | 22 / 19 / 40건 | 4.467초 | 0.257초 | 3.040 → 9.143초 |

두 로그의 예약·반환을 순서대로 다시 계산했다. 최대 예약은 기준군 1,604블록, 제한군 708블록이었다. 각 실행의 종료 전 활성·대기 요청과 실제 할당 KV가 모두 0이었고 두 서버가 코드 0으로 종료했다. GPU 6·7은 기존 홀더의 20GiB 예약으로 복귀했다. 수치는 이 한 쌍의 계측 실행에 해당한다. 이 부하에서는 입장 후 GPU 스텝을 기다리는 상황에 KV 예약 부족에 따른 입장 전 대기가 추가됐다. [입장·회수 검산 코드](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/verify_kv_pressure.py) · [실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/campaign.sh)

초기 시도에서 소스 교체 위치가 중복되고, 후속 실행에서는 GPU 6의 외부 사용과 실행 스크립트의 예약 검사 오류를 발견했다. 이 기록은 비교에서 제외했고, 검증한 같은 CUDA 바이너리와 명시적으로 실패를 반환하는 예약 확인으로 두 실행을 완료했다. 빌드 재현 시 `joint-clock`의 `reused.sha256`과 `joint-clock/retry`의 `fix_terminal_observation.py`를 빌드 스크립트 옆에 두고 체크섬을 확인한다. [제외한 실행과 바이너리 해시](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/verified-setup.json) · [격리 빌드](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/build/campaign.sh)

#### 장문 chunk와 KV 한도의 교차 대조

Qwen2.5-14B-Instruct와 위 KV 압력 시험에 쓴 CUDA 진단 바이너리로 W4의 고정 경계 입력 11개를 실행했다. 입력은 255–7,917토큰, 생성 상한은 64–256토큰이다. 각 설정에서 C1은 입력마다 한 번, C8·C32는 각각 세 번 실행하도록 했다. 응답의 입력·생성 ID, 텍스트, 종료 사유와 usage를 비교하며, 이 실험은 출력 정확성과 예약 동작을 확인한다. [고정 입력](../../benchmarks/evidence/20261119-w4-holdout/holdout.json.gz) · [실행기](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/campaign.sh) · [입력·응답 대조 코드](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/exact_batch.py)

기본 입장 제한 30초로 실행했을 때 chunk 1,024·KV 2,576의 C8과 C32는 각각 33/33건이 C1과 일치했다. chunk 512·KV 768에서는 C32의 한 요청이 HTTP 408을 받았다. 이 요청의 마지막 KV 부족 기록은 대기에 들어간 뒤 29,983.07ms였고, 655블록을 이미 예약한 상태에서 511블록이 더 필요했다. 시험 요청 76건 가운데 75건은 활성화 후 KV를 반환했고 해당 1건은 활성 단계에 들어가지 못했다. 검사가 전체 응답 대조를 마치기 전에 중단됐으며 두 GPU는 각각 20GiB 예약으로 돌아왔다. [중단 실행의 집계와 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/default-timeout.json) · [집계 재현](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/fetch-default-timeout.sh) · [입장 대기 만료 처리](../../crates/riley-scheduler/src/scheduler.rs#L1577)

재실행에서는 같은 GPU 7, 모델, 바이너리, 1,024 배치 토큰 예산과 활성 시퀀스 한도 32를 유지했다. 네 조합 모두 입장 제한을 90초로 고정하고 prefill chunk 상한과 KV 한도만 바꿨다. 조합당 C1 11건, C8 33건, C32 33건을 각각 완료했다. 각 조합의 C8·C32는 같은 조합의 C1과 일치했고, 같은 동시성의 실행 결과는 네 조합 사이에서도 모두 일치했다. 비교한 308건의 입력·생성 ID, 텍스트, 종료 사유와 usage를 확인했다. 네 원본 응답 파일의 SHA-256은 기본 30초 기준군의 원본과도 같았다. [전체 비교의 요청 수·해시](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/extended-timeout/verified-summary.json)

| Prefill chunk 상한 | KV 블록 | KV 부족 요청 C8 / C32 | C32 최대 입장 대기 | C32에서 입장 대기 30초 초과 |
|---:|---:|---:|---:|---:|
| 1,024 | 2,576 | 0 / 3 | 9.33초 | 0 |
| 512 | 768 | 4 / 15 | 36.73초 | 1 |
| 1,024 | 768 | 5 / 15 | 32.66초 | 6 |
| 512 | 2,576 | 0 / 3 | 10.05초 | 0 |

각 실행에서 설정한 길이의 prefill chunk를 실제로 처리했고 입력의 처음부터 끝까지 누락 없이 처리했음을 스케줄러 로그로 확인했다. 표는 서버가 시작하며 처리한 내부 warm-up 48건을 각 실행에서 제외했다. C1은 네 조합 모두 KV 대기 없이 완료했다. 네 서버가 활성·대기 요청과 실제 KV 할당을 0으로 회수한 뒤 코드 0으로 종료했고, GPU 6·7은 동일 홀더의 20GiB 예약으로 복귀했다. [검산기와 성공 기준](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/extended-timeout/verify_exact_batch.py) · [대기 집계 재현](../../benchmarks/evidence/20261119-w3-rag-profile/kv-pressure/chunk-exact/extended-timeout/fetch-summary.sh)

이 정확성 결과는 Qwen2.5-14B-Instruct, 고정 입력 11개와 시험한 CUDA 바이너리에 적용된다. 이 부하에서 KV 768블록은 두 chunk 크기 모두 기본 입장 제한을 넘는 대기가 관측됐다. 운영 용량은 입력·출력 길이와 동시에 예약할 요청 수를 함께 산정해야 한다. 90초는 격리 진단에 사용했다. 실제 서비스 분포와 SLO는 W5에서 확인할 항목이다. [W5 운영 검증 범위](05-documentation-and-operations.md#합성-부하의-확인-기준)

재현 시 실행기와 같은 경로에 `dispatch.sha256`에 적힌 `benchmarks/live/long_accuracy.py`와 상위 `kv-pressure/verify_kv_pressure.py`를 둔다. 90초 대조는 같은 `campaign.sh`와 `exact_batch.py`에 `extended-timeout`의 어댑터·검산기·체크섬을 함께 사용한다.

#### Coder-14B의 1,024행 후보 검증

앞선 Qwen2.5-14B 시험과 같은 바이너리 및 A100 7번에서 `Qwen2.5-Coder-14B-Instruct`를 비교했다. 이 모델의 병합 QKV도 `K=5120, N=7168`이며 1,024행 후보가 적용되는 형상이다. 모델 revision `aedcc2d42b622764e023cf882b6652e646b95671`과 KV 2,576블록을 고정하고 기준 → 후보 → 후보 → 기준 순서로 실행했다. 후보 두 실행에서 cuBLASLt 12.4.5의 후보 선택을 확인했다. [고정 모델 입력](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/model.sha256) · [실행 및 런타임 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/verified-summary.json)

측정 1,280건을 오류·429 없이 완료했다. 입력 식별, 출력 텍스트 해시, 입력·생성 토큰 수, 종료 사유가 네 실행에서 일치했고, 별도의 장문 8개에서도 C1·C8 생성 토큰 ID가 일치했다. 매번 서버가 코드 0으로 종료했고 같은 GPU 7 홀더의 20GiB 예약으로 복귀했다. 표의 변화율은 각 순서에서 인접한 기준군 대비 후보군으로 계산했다.

| 부하 | TTFT p95 변화, 기준 먼저 / 후보 먼저 | Decode TPOT p50 변화, 기준 먼저 / 후보 먼저 |
|---|---:|---:|
| `rag-c1` | −0.92% / −1.70% | +0.05% / +0.05% |
| `rag-c8-s1` | −2.58% / −1.78% | −0.90% / −0.81% |
| `rag-c8-s2` | −2.64% / +7.72% | −1.02% / −1.07% |
| `wave-sync` | −2.50% / −1.41% | −0.86% / +1.99% |
| `wave-50ms` | +1.85% / +2.06% | −2.93% / −5.46% |
| `short-c32` | −0.07% / +0.20% | −0.04% / +0.09% |

짧은 요청의 TTFT·처리량은 2% 회귀 기준을 충족했다. 후보 먼저 실행한 동시 유입의 decode TPOT은 1.99% 늘어 1% 기준을 넘었다. 50ms 유입에서는 40건 중 33건과 34건이 빨라졌고, 다섯 번째 묶음의 41번 요청은 각각 534.83ms와 539.14ms 느려졌다. 같은 40건의 요청 본문 지문과 입력 토큰 수를 앞선 Qwen2.5-14B 시험과 맞췄으며, 그 모델에서도 41번 요청에 같은 방향의 큰 지연이 있었다. 다섯 번째 묶음을 뺀 민감도 계산에서 p95의 변화 방향이 뒤집혔고, 채택 판단은 전체 40건을 기준으로 했다. [묶음별 꼬리 지연과 입력 교차 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/wave-summary.json)

Coder-14B의 decode 회귀와 두 모델의 50ms 유입 결과에 따라 이 QKV 후보의 형상 공통 적용을 기각했다. Coder-14B에서 유입 시각·응답 지연은 검증했으며, 요청이 선택되는 내부 순서에 대한 GPU 계측 근거는 앞선 Qwen2.5-14B에서 얻었다. 재현할 때는 `tools`의 공유 검사기 두 파일을 실행 위치에 복사해 `dispatch.sha256`을 확인한다. [캠페인](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/campaign.sh) · [공유 교차 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/compare_profile_arms.py) · [공유 설정 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/summarize_profile_arm.py)

##### 동시 유입의 decode 회귀 구간

기존 네 실행의 원본에서 첫 콘텐츠부터 마지막 콘텐츠까지 걸린 시간을 생성 토큰 간격 수로 나눠 요청별 TPOT을 계산했다. 두 시각 간격과 두 실행 순서에서 각각 40건의 출력·토큰 수·종료 사유를 대조했고, 요청별 계산으로 기록된 TPOT p50을 재현했다. 동시 유입의 후보 먼저 비교에서 전체 p50은 28.783→29.356ms로 증가했고, p50을 결정한 요청은 9번에서 41번으로 바뀌었다. 대응 요청끼리 뺀 TPOT 차이의 중앙값은 −0.032ms였으며 40건 중 17건이 느려졌다. [요청별 상대 변화와 원본 해시](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/wave-decode-summary.json)

같은 비교의 두 번째 8건 묶음에서는 TPOT p50이 27.674→31.205ms로 증가했다. 대응 요청의 지연 차이 중앙값은 +1.116ms였고 소켓 쓰기 간격 범위는 기준 0.211ms, 후보 0.175ms였다. 두 번째 묶음을 제외한 민감도 계산의 전체 p50 변화는 −0.17%였고, 세 번째·다섯 번째 묶음을 각각 제외해도 +0.12%와 +0.03%로 작아졌다. 회귀 판정은 전체 40건의 +1.99%를 적용한다. 후속 GPU 계측에서 두 번째 묶음의 decode 진행 구간과 동시 prefill 행을 우선 대조할 근거로 사용한다. [독립 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/analyze_coder_wave_decode.py)

##### Coder-14B 실측의 prefill 대기 구간

A100 7번에서 앞선 Qwen2.5-14B 요청 계측용 CUDA 바이너리를 그대로 사용했다. Coder-14B의 기존 모델·KV·서버 설정을 유지하고 동시 유입과 50ms 유입을 기준→후보→후보→기준 순서로 다시 실행했다. 두 부하의 측정 320건에서 입력·응답 해시, 종료 사유, 토큰 사용량이 해당 비계측 실행과 일치했고 오류·429는 0건이었다. 서버 요청 로그와 클라이언트의 시간축 오차 범위는 실행별로 0.446–0.459ms였다. 네 실행 모두 정상 종료했고, A100 6·7번의 기존 20GiB 예약을 확인했다. [원본 지문과 집계](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/traced-waves/verified-trace.json)

| 50ms 유입, 41번 요청 | 기준 먼저 | 후보 먼저 |
|---|---:|---:|
| 요청의 TTFT 증가 | 539.74ms | 528.05ms |
| 요청의 prefill 조각 사이 대기 증가 | 390.70ms | 381.63ms |
| 전체 40건의 TTFT p95 변화 | +1.96% | +1.48% |

동시 유입의 전체 TPOT p50은 기준 먼저 −0.24%, 후보 먼저 +2.73%였다. 두 번째 묶음의 TPOT p50은 두 비교에서 각각 +6.03%, +14.24%였다. 같은 요청끼리 비교한 decode 구간과 prefill 로그 구간의 겹침 시간 차이는 첫 비교에서 중앙값 −41.03~−40.13ms, 역순에서 +25.55~+26.01ms였다. 구간 겹침의 방향이 두 순서에서 갈렸다. 50ms 유입의 41번 요청은 Qwen2.5-14B와 Coder-14B 모두 prefill 조각 사이 대기가 주요 증가 구간으로 확인됐다. 기존 비계측의 1% decode 가드레일과 형상 공통 적용 기각을 유지한다. [Qwen2.5-14B 계측](#같은-실행의-요청-대기와-cuda-커널-대조) · [구간 겹침 검산기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/traced-waves/verify_trace.py)

재현 시 분석 디렉터리에 상위 디렉터리의 `analyze_coder_wave_decode.py`와 [공유 요청 단계 검사기](../../benchmarks/evidence/20261119-w3-rag-profile/tools/summarize_request_stages.py)를 복사해 `analysis.sha256`을 확인한다. [실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/traced-waves/campaign.sh) · [독립 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/traced-waves/fetch-summary.sh)

##### Coder-14B의 입장 경계 재현

이전 Coder-14B 계측의 다섯 번째 묶음에서 첫 9개 계획은 모두 decode가 시작되기 전의 1,024행 단일 prefill이었다. 두 기준 실행에서 45번은 41번의 첫 prefill 시작 39–40µs 전에 입장했다. 두 후보 실행에서는 같은 스텝 반환 85–156µs 후에 입장했다. 기준군의 8·9번째 선택은 45→41이었고 후보군은 41→45였다. [입장 시각·형상·원본 지문](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/admission-boundary/replay-inputs.json)

현재 `main`에서 빌드한 스케줄러로 네 실행의 36개 계획을 재현했다. 같은 스텝 시각을 유지한 채 45번의 입장만 41번의 첫 prefill 전후로 옮기면 네 경우 모두 처음 7개 계획을 보존하면서 8·9번째 선택을 반대편 기록과 동일하게 재현했다. 이 재생은 실제 시각을 고정한 CPU 대조다. [스케줄러 재생 결과](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/admission-boundary/verified-replay.json) · [스케줄러 빌드와 재생](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/admission-boundary/run_replay.sh)

같은 CUDA 바이너리와 A100 7번에서 45번의 송신 시각만 −12ms, 0ms, +12ms로 바꿨다. 기준→후보→후보→기준 순으로 네 서버를 실행했고 후반 쌍은 오프셋 순서도 뒤집었다. 각 조건에서 워밍업 8건과 측정 40건을 처리했다. 측정 480건의 입력·응답 지문, 토큰 사용량과 종료 사유가 해당 원본에 모두 일치했고 오류·429는 0건이었다. 서버 종료가 모두 성공했고 두 GPU의 기존 20GiB 예약을 확인했다. [GPU 교차 검증 결과](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/admission-boundary/verified-phase.json)

| 45번 송신 오프셋 | 기준 / 후보의 8→9번째 선택 | TTFT p95 변화, 기준 먼저 / 후보 먼저 | 41번 TTFT 변화, 기준 먼저 / 후보 먼저 |
|---:|---|---:|---:|
| −12ms | 45→41 / 45→41 | −4.11% / −2.02% | −94.19ms / −55.14ms |
| 0ms | 45→41 / 41→45 | +0.91% / +2.04% | +515.40ms / +544.17ms |
| +12ms | 41→45 / 41→45 | −2.98% / −2.14% | −94.48ms / −67.25ms |

기존 송신 시각에서 후보의 41번은 prefill 조각 사이 대기가 373–395ms 늘었다. 45번을 옮겨 두 군의 prefill 순서를 맞추면 두 오프셋과 두 실행 순서에서 같은 요청의 대기와 TTFT가 줄었다. 앞선 Qwen2.5-14B에서도 같은 입장 경계와 p95 변화 방향을 관측했다. 이번 재현 범위는 해당 50ms 유입이며, 동시 유입의 decode 회귀 기준에 따른 1,024행 QKV 형상 공통 적용 기각을 유지한다. [Qwen2.5-14B의 같은 송신 시각 대조](#실제-요청의-도착-시각-대조)

재현 시 공용 실행기 `prepare_phase_probe.py`를 복사해 `dispatch.sha256`을 확인한다. 분석에는 `analysis.sha256`에 기록한 공용 분석기와 앞선 Coder-14B의 `traced-waves/verified-trace.json`을 `source-trace-report.json`으로 사용한다. [공용 Rust 재생기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/replay_wave.rs) · [공용 실행기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/phase-gpu/prepare_phase_probe.py) · [A100 실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/admission-boundary/campaign.sh) · [독립 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/coder14/admission-boundary/fetch-summary.sh)

##### prefill 스텝 초반에 송신된 요청의 우선순위

입장 시각 자체는 유지하고 첫 prefill의 정렬에만 적용하는 규칙을 격리 스케줄러에서 시험했다. 직전 prefill 스텝의 처음 5% 이내에 송신되고 스텝 종료 후 250µs 이내에 즉시 입장한 요청에 해당 스텝 종료 시각보다 1ns 앞선 정렬 키를 준다. 기존 대기·aging 시각과 KV 입장 조건을 사용하며, decode가 준비된 스텝은 기존 정렬을 쓴다. 5%와 250µs는 관측한 입장 경계를 검증하기 위한 진단값이다. [격리 스케줄러 패치](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/early-prefill-priority.patch.gz)

두 모델의 송신 시각 교차 실험 24개 조건과 별도로 기록한 기존 시각 8개 조건을 재생했다. 원본 스케줄러와 규칙을 끈 진단 스케줄러는 기록된 첫 9스텝 전부, 총 288개 계획과 일치했다. 규칙을 켜면 기존 시각의 후보군 8개 조건에서 8·9번째 선택만 41→45에서 45→41로 바뀌었다. 두 모델의 앞·뒤 오프셋 조건과 나머지 기준군은 기존 계획을 재현했다. 경계값과 decode 우선순위를 확인하는 두 테스트를 포함해 진단 스케줄러의 단위 테스트 38개가 통과했다. [CPU 대조 결과](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/verified-replay.json) · [별도 계측 입력](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/confirmation-inputs.json)

CPU 재생은 기존 GPU 스텝의 시간을 고정했다. [재현 실행기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/run_replay.sh) · [공용 컴파일 도구](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/command-clock-cpu/compile_replay.py) · [공용 기준 재생기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/replay_wave.rs) · [Qwen2.5-14B 재생 입력](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/command-clock-cpu/cpu-ready-inputs.json)

같은 규칙에 서버의 명령 송신·스텝 시각을 전달하는 진단 바이너리를 빌드했다. 이전 요청 계측과 같은 `d3d8fe6e5` 기반 CUDA 12.4 SM80 격리 빌드다. 소스 두 파일과 바이너리의 SHA-256, Linux AMD64 동적 링크, `RILEY_W3_EARLY_PREFILL_PRIORITY` 옵션 문자열을 대조했다. 이 빌드에는 GPU를 할당하지 않았다. 재현할 때는 우선순위 패치와 서버 연결 패치의 압축을 풀어 `dispatch.sha256`을 확인한다. [빌드 실행기](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/cuda-build/build-priority-cuda.sh) · [빌드 영수증](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/cuda-build/verified-build.json)

##### Coder-14B 서빙의 우선순위 교차 대조

CUDA 12.4 SM80 진단 바이너리와 A100 7번에서 Coder-14B의 모델·서버 인자·KV 2,576블록을 고정했다. 네 실행 모두 1,024행 QKV와 요청 추적을 켰고, prefill 우선순위만 끔→켬→켬→끔으로 바꿨다. 실행 중인 서버의 옵션을 확인했고, 이전 검증과 같은 RAG C1, RAG C8 두 seed, 동시 유입, 50ms 유입, 짧은 C32를 사용했다.

측정 1,280건 전체의 입력·출력 지문, 토큰 사용량과 종료 사유가 이전 Coder-14B 결과와 일치했다. 별도 장문 입력 8개의 C1·C8 토큰 ID도 네 환경에서 같았다. 오류·429는 0건이었고 모든 서버가 정상 종료했다. A100 6·7번의 기존 20GiB 예약 복귀를 확인했다. [교차 대조 영수증](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/coder14-gpu/verified-abba.json)

| 부하 | TTFT p95 변화, 끔 먼저 / 켬 먼저 | Decode TPOT p50 변화, 끔 먼저 / 켬 먼저 |
|---|---:|---:|
| RAG C1 | −1.34% / −0.36% | +0.00% / +0.04% |
| RAG C8 seed 1 | −15.08% / +8.86% | +2.75% / −0.85% |
| RAG C8 seed 2 | +9.70% / +22.07% | −0.56% / −2.04% |
| 동시 유입 | +0.12% / −1.02% | +2.62% / +0.59% |
| 50ms 유입 | −4.45% / −4.23% | +5.57% / +5.68% |
| 짧은 C32 | +0.15% / +0.03% | −0.08% / +0.18% |

50ms 유입에서 8·9번째 prefill은 두 비교 모두 41→45에서 45→41로 바뀌었다. 41번의 prefill 조각 사이 대기는 433–435ms 줄었고 첫 콘텐츠는 602–603ms 빨리 도착했다. 이후 decode 구간이 다른 요청의 prefill과 약 599ms 겹쳤으며 마지막 콘텐츠까지의 시간은 101–106ms 줄었다. 두 시계의 정렬 불확실성 폭은 실행별 0.5ms 이하였다. [구간 대조 계산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/coder14-gpu/verify_priority_abba.py)

50ms 유입의 decode TPOT p50이 두 순서 모두 1% 회귀 기준을 넘었고 RAG C8 seed 2의 TTFT p95도 두 번 증가해 후보를 미채택했다. 50ms 유입에서 같은 요청끼리 계산한 TPOT 차이 중앙값은 −0.017ms와 +0.037ms였고 느려진 요청 수는 40개 중 8개와 32개였다. 서로 다른 요청이 전체 p50을 결정하는 두 실행을 포함하며, 판정에는 기존의 부하별 TPOT p50을 적용했다. 검증 범위는 1,024행 QKV를 활성화한 Coder-14B다. [실행 순서](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/coder14-gpu/campaign.sh) · [독립 검산](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/prefill-dispatch-priority/coder14-gpu/fetch-summary.sh)

## Coder-14B 반복 문서의 prefix cache

기존 RAG 비교의 제각기 다른 프롬프트와 별도로, 문서 네 개를 여러 요청이 공유하는 부하를 만들었다. Coder-14B 기준 이미지에서 `--kv-blocks 2576`을 공통 적용하고 캐시 활성 조건에만 `--prefix-cache-blocks 640`을 지정했다. 캐시 블록은 기존 KV 풀 안에서 사용한다. 양쪽 모두 비관리자 체크포인트 소유자 UID로 같은 파일을 읽기 전용 마운트했다. [실행 설정](../../benchmarks/evidence/20261121-w3-coder-rag-cache/run_arm.py)

실행마다 문서 네 개를 순차 요청한 다음 6초 간격으로 여덟 묶음을 재생했다. 묶음마다 8개 요청을 50ms 간격으로 전송했다. 출력 상한은 64토큰이다. 별도 짧은 C32의 96개 요청을 같은 서버에서 처리했다. SSE로 받은 입력·생성 ID와 가시 텍스트를 지문화했다. 캐시 효과는 응답의 `cached_tokens`로 측정하고, 전송 시각의 오차와 종료 자원 회수를 함께 검사했다. [입력 생성과 전송](../../benchmarks/evidence/20261121-w3-coder-rag-cache/workload.py)

역순의 첫 활성 실행은 전송 오차 p95가 5ms를 넘어 제외했다. 역순 재실행에서는 전송 시각 6ms 전에 타이머를 깨우고 예정 시각에 보내도록 교정했다. 같은 입력과 예정 시각을 유지하고 역순의 양쪽 조건에 동일하게 적용했다. [실행 순서와 사전 검사](../../benchmarks/evidence/20261121-w3-coder-rag-cache/campaign.sh)

A100 7번에서 기준 이미지 `b731f48cc6ec`로 유효한 실행 네 개를 완료했다. 각 비교 쌍의 164개 요청에서 입력·생성 토큰 ID와 개수, 출력 텍스트, 종료 사유가 일치했다. 캐시 활성 RAG의 128개 측정 요청이 모두 2,032–2,048토큰을 재사용했다. 2,158–2,175토큰 입력의 약 94%다. 네 문서를 처음 넣은 순차 요청에서는 재사용이 0이었다. [검산 결과와 원본 해시](../../benchmarks/evidence/20261121-w3-coder-rag-cache/verified.json)

| 지표 | 비활성 먼저, 비활성 → 활성 | 활성 먼저, 비활성 → 활성 |
|---|---:|---:|
| 반복 문서 RAG TTFT p95 | 2,124.70 → 84.74ms (−96.01%) | 2,104.31 → 80.67ms (−96.17%) |
| 반복 문서 RAG decode TPOT p50 | 26.36 → 26.05ms (−1.18%) | 26.38 → 26.07ms (−1.18%) |
| 짧은 C32 TTFT p50 | 75.21 → 74.79ms (−0.57%) | 75.21 → 77.48ms (+3.03%) |
| 짧은 C32 TTFT p95 | 346.99 → 271.38ms (−21.79%) | 347.20 → 283.22ms (−18.43%) |
| 짧은 C32 출력 처리량 | 1,123.43 → 1,166.38 tok/s (+3.82%) | 1,123.07 → 1,165.48 tok/s (+3.78%) |

짧은 C32의 역순 TTFT p50이 2% 회귀 한도를 넘어 혼합 서비스 설정으로 채택하지 않았다. 같은 역순을 먼저 보낸 32개와 후속 64개로 나누면 각 집단의 TTFT p50은 232.81→175.37ms, 74.80→71.63ms였다. 두 집단을 합친 96건의 지표를 채택 판정에 적용했다. 활성 조건에서 짧은 요청은 80건이 32토큰, 12건이 16토큰을 재사용했고 4건은 재사용이 없었다. 반복 문서 부하의 이득 범위는 네 문서가 미리 적재된 조건이다. 새로운 문서만 들어오는 RAG와 기존 vLLM 대비 목표는 별도 판정으로 유지한다.

유효한 네 실행의 전송 오차 p95는 0.50–4.99ms였다. 모두 종료 코드 0과 활성·대기 요청, 요청 KV, device·pinned 할당 0을 기록했다. A100 6·7번의 기존 홀더와 20GiB 예약을 각 실행 전후에 확인했다. [독립 검산](../../benchmarks/evidence/20261121-w3-coder-rag-cache/fetch-summary.sh)

### 128토큰 미만 요청의 캐시 제외

앞선 짧은 C32 입력은 70–77토큰이었고 반복 문서 RAG 입력은 2,158–2,175토큰이었다. 격리한 진단 바이너리에서만 `RILEY_W3_PREFIX_CACHE_MIN_128=1`로 128토큰 미만의 요청을 prefix cache 조회와 등록에서 제외했다. 경계 CPU 검사에서 127토큰 요청의 재사용은 112→0토큰이 됐고, 두 조건에서 128토큰 요청은 계속 112토큰을 재사용했다. 요청이 끝났을 때 짧은 요청이 남긴 캐시 블록도 없어졌다. [스케줄러 진단 패치 (gzip)](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/min-prompt.patch.gz) · [CPU 검사](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/cpu.json)

기준 이미지의 소스 리비전 `36aa1880`과 같은 Rust 1.85.0·CUDA 12.4를 사용했다. A100 SM80으로 빌드한 실행 파일을 기준 이미지에 읽기 전용으로 연결했다. 실행 파일 하나에서 캐시 비활성, 기존 캐시 640블록, 128토큰 미만 제외를 비교했다. KV 풀 2,576블록과 이전 반복 문서의 입력·도착 시각표, 짧은 C32 96개 요청을 고정했다. [격리 빌드](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/build-verified.json) · [실행 순서](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/campaign.sh)

재빌드 경로에서 `gzip -dc min-prompt.patch.gz > min-prompt.patch`로 압축을 풀면 체크섬 목록과 동일한 입력을 얻는다. [빌드 입력 체크섬](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/build-inputs.sha256)

비활성 → 기존 캐시 → 128토큰 기준 → 128토큰 기준 → 기존 캐시 → 비활성 순으로 실행했다. 총 984건에서 기존 배포 이미지의 입력·생성 토큰 ID와 개수, 출력 텍스트, 종료 사유를 재현했다. 전송 오차 p95는 0.50–0.59ms였다. 128토큰 기준의 장문 128건은 기존 캐시와 요청별로 같은 2,032–2,048토큰을 재사용했다. 기존 캐시에서 짧은 요청은 각 순서 92/96건이 적중했고, 128토큰 기준에서는 각 0/96건이었다. [요청 단위 검산](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/verified.json)

| 지표 | 캐시 비활성 대비 기존 캐시, 정순 / 역순 | 캐시 비활성 대비 128토큰 기준, 정순 / 역순 |
|---|---:|---:|
| 반복 문서 RAG TTFT p95 | −96.23% / −96.20% | −96.19% / −96.16% |
| 짧은 C32 TTFT p50 | +10.80% / +16.59% | +1.20% / +12.19% |
| 짧은 C32 TTFT p95 | −24.34% / −29.41% | −1.57% / +1.03% |
| 짧은 C32 decode TPOT p50 | −2.40% / −2.61% | +1.41% / +1.11% |
| 짧은 C32 출력 처리량 | +3.72% / +3.45% | −1.71% / −0.91% |

128토큰 기준은 짧은 C32의 decode TPOT 1% 한도를 양쪽 순서에서 넘었고 역순의 TTFT p50도 2% 한도를 넘었다. 혼합 서비스 후보로 채택하지 않았다. 비활성과 128토큰 기준에서 짧은 C32는 각각 204개 스텝을 실행했고 32행 버킷은 모두 163개였다. 이 버킷의 평균 GPU 시간 변화는 약 +0.11%와 −0.11%였다. 전체 GPU 스텝 시간 증가 94.84ms와 49.46ms의 대부분이 다른 행 버킷에서 관측됐다. 이 집계에는 prefill과 decode가 섞인 스텝이 포함된다.

여섯 실행에서 종료 코드 0과 활성·대기 요청, KV, device·pinned 할당 0을 확인했다. A100 6·7번의 기존 홀더와 20GiB 예약도 복구됐다. [독립 검산](../../benchmarks/evidence/20261121-w3-prefix-cache-min-prompt/fetch-summary.sh)

### 짧은 C32의 첫 실행과 반복 실행

보존된 네 원본에서 192행 버킷은 캐시 비활성의 RAG 이전·도중에 각각 2회 사용됐고 128토큰 필터에서는 사용되지 않았다. 짧은 C32에서 각 실행이 한 번씩 사용했으며, 해당 GPU 시간은 비활성과 필터 순서대로 34.37→100.85ms와 34.37→44.10ms였다. 짧은 요청 구간에 처음 사용된 형상이 관측 지연에 영향을 주는지 확인하기 위해 C32를 같은 프로세스에서 재실행했다. [과거 원본 지문과 재검산](../../benchmarks/evidence/20261121-w3-coder-short-warmup/verify_warm.py)

앞선 `36aa1880` 진단 빌드 `c5282eac85ac`와 Coder-14B 체크포인트, KV 풀 2,576블록을 재사용했다. GPU 7번은 캐시 비활성 → 필터, 6번은 필터 → 캐시 비활성 순서였다. 각 실행의 장문 요청과 도착 시각표를 유지하고, 짧은 C32의 같은 96개 요청을 즉시 한 번 더 보냈다. 수치는 각각 같은 GPU의 캐시 비활성 실행을 기준으로 비교했다. [실행 순서](../../benchmarks/evidence/20261121-w3-coder-short-warmup/campaign.sh) · [반복 부하](../../benchmarks/evidence/20261121-w3-coder-short-warmup/workload_warm.py)

| 지표 | GPU 7 첫 실행 | GPU 7 반복 | GPU 6 첫 실행 | GPU 6 반복 |
|---|---:|---:|---:|---:|
| 짧은 C32 TTFT p50 | +8.80% | −0.73% | +1.07% | +0.67% |
| 짧은 C32 TTFT p95 | +1.04% | +0.37% | +1.12% | +0.93% |
| 짧은 C32 decode TPOT p50 | +0.73% | +0.02% | +0.55% | −0.06% |
| 짧은 C32 출력 처리량 | −0.60% | −0.02% | −0.68% | −0.17% |

필터를 적용한 두 실행의 반복 문서 RAG TTFT p95는 캐시 비활성 대비 96.12–96.14% 줄었다. 두 실행 모두 RAG 64건에서 캐시가 적중했고, 짧은 요청은 첫 실행과 반복에서 모두 적중 0건이었다. 전송 오차 p95는 0.46–0.58ms였다. 네 서버의 총 1,040건에서 기존 이미지와 입력·출력 토큰 ID, 텍스트, 종료 사유가 일치했다. 종료 자원을 회수하고 GPU 6·7의 기존 20GiB 예약을 확인했다. [검산 결과와 원본 체크섬](../../benchmarks/evidence/20261121-w3-coder-short-warmup/verified.json)

모든 C32 실행에서 204개 스텝 중 163개가 32행이었다. 필터 실행의 192행 GPU 시간은 첫 부하에서 GPU 7번 46.70ms, 6번 43.82ms였고 반복에서는 각각 34.17ms와 34.64ms였다. 두 번째 C32에서 사용한 형상은 모두 앞선 구간에서 이미 사용한 이력이 있었다. 이 대조에서는 동일 부하를 한 번 실행한 뒤 두 GPU에서 지연·처리량 기준을 충족했다. 첫 실행에서는 GPU 7번의 TTFT p50이 2% 한도를 넘었다. 같은 부하를 반복하는 진단과 실제 유입의 차이가 남아 있어 필터를 혼합 서비스 기본값으로 채택하지 않는다. [독립 재검산](../../benchmarks/evidence/20261121-w3-coder-short-warmup/fetch-summary.sh)

### 시작 워밍업과 첫 C32 지연

서버의 기존 옵션 `--startup-warmup step-shapes-v1`이 RAG 뒤 첫 C32의 회귀를 줄이는지 확인했다. 앞선 `36aa1880` 진단 빌드 `c5282eac85ac`, Coder-14B 체크포인트, KV 풀과 요청 시각표를 재사용했다. GPU마다 새로운 서버 프로세스에서 캐시 비활성·128토큰 필터와 시작 워밍업 켬·끔의 네 조합을 실행했다. GPU 7번과 6번의 실행 순서를 뒤집었고 수치는 각 GPU 내부에서 비교했다. 준비 상태에서 외부 요청을 보내기 전에 `/metrics`의 배치 형상을 저장했다. [실행 순서](../../benchmarks/evidence/20261121-w3-coder-startup-warmup/campaign.sh) · [준비 상태 측정](../../benchmarks/evidence/20261121-w3-coder-startup-warmup/workload_startup.py) · [설정 설명](../configuration.md#execution-options)

| RAG 뒤 첫 C32 TTFT p50 변화 | A100 6번 | A100 7번 |
|---|---:|---:|
| 워밍업을 끈 상태에서 비활성 대비 필터 | −4.53% | −7.16% |
| 워밍업을 켠 상태에서 비활성 대비 필터 | +9.62% | +9.97% |
| 비활성·워밍업 끔 대비 필터·워밍업 켬 | +4.88% | +1.68% |

워밍업을 켠 상태에서 필터의 첫 C32 TTFT p95는 GPU 6·7번에서 각각 2.28%, 0.32% 줄었고 decode TPOT p50은 0.47%, 0.44% 늘었다. 같은 비교에서 RAG TTFT p95는 각각 96.21%, 96.16% 줄었다. 첫 C32 전체 96건의 TTFT p50은 두 GPU에서 모두 2% 한도를 넘었다. 첫 32개와 후속 64개를 따로 보면 각 집단의 p50 변화는 더 작았다. 채택 기준에는 전체 96건의 지표를 적용했다. [기준별 판정과 요청 집단](../../benchmarks/evidence/20261121-w3-coder-startup-warmup/verified.json)

서버 로그의 내부 워밍업은 2.08–2.77초였다. 캐시 비활성 실행은 준비 상태 전에 두 GPU에서 각각 18개 버킷과 192행 스텝 2개를 처리했다. 필터 실행은 14개 버킷을 처리했고 첫 C32에서 사용한 176·192·240·256행 버킷은 준비 상태 때 사용 이력이 없었다. 여덟 실행의 1,312건에서 기존 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유가 일치했다. 종료 자원을 회수했고 GPU 6·7의 기존 20GiB 예약도 복귀했다. [원본 지문](../../benchmarks/evidence/20261121-w3-coder-startup-warmup/result.sha256) · [독립 검산](../../benchmarks/evidence/20261121-w3-coder-startup-warmup/fetch-summary.sh)

기존 시작 워밍업 옵션을 켜는 것만으로 필터의 첫 C32 지연 기준을 충족하지 못했다. 준비 상태에서 실행한 형상은 캐시 설정에 따라 달랐다. 혼합 서비스 기본값 채택은 보류한다.

### 시작 워밍업의 prefix cache 격리

`3951d90c8` 배포 소스에 한정 진단 패치를 적용했다. 워밍업 요청의 스케줄러 입력에 캐시 사용 여부를 전달해, 같은 프롬프트와 샘플링 설정으로 캐시 공유와 격리를 선택할 수 있게 했다. 격리 요청은 캐시 블록을 조회하거나 등록하지 않는다. 일반 요청은 앞서 시험한 128토큰 필터를 적용한다. CUDA 12.4 SM80 빌드 한 개에서 캐시 비활성·워밍업 끔, 비활성·워밍업 켬, 필터·공유 워밍업, 필터·격리 워밍업을 GPU 6·7번에서 역순으로 대조했다. [진단 패치](../../benchmarks/evidence/20261121-w3-private-startup-warmup/private-warmup.patch.gz) · [실행 순서](../../benchmarks/evidence/20261121-w3-private-startup-warmup/campaign.sh)

첫 컴파일에서 스케줄러의 요청 복사 경로에 추가 필드가 빠져 `E0027`이 발생했다. 내부 복사에서도 캐시 사용 여부를 보존하도록 교정한 뒤 CPU 테스트 두 건이 통과했다. 일반 요청이 재사용하는 블록을 격리 요청은 읽지 않았고, 격리 요청으로 먼저 실행했을 때 일반 요청의 첫 재사용 토큰도 0이었다. 기존 128토큰 필터 경계도 통과했다. [CPU 결과 (gzip)](../../benchmarks/evidence/20261121-w3-private-startup-warmup/cpu-tests.log.gz) · [빌드 지문](../../benchmarks/evidence/20261121-w3-private-startup-warmup/build-result.sha256)

| 준비 상태의 관측값 | 캐시 비활성·워밍업 켬 | 필터·공유 워밍업 | 필터·격리 워밍업 |
|---|---:|---:|---:|
| 누적 캐시 적중 토큰 | 0 | 1,232 | 0 |
| 보관 캐시 블록 | 0 | 64 | 0 |
| 사용한 배치 버킷 | 18 | 14 | 18 |
| 192행 스텝 | 2 | 0 | 2 |

두 GPU에서 이 값이 같았다. 공유 워밍업에서는 이후 첫 C32가 사용한 176·192·240·256행에 준비 시점의 실행 이력이 없었다. 격리 워밍업에서는 이 중 256행만 실행 이력이 없었고 캐시 비활성·워밍업 켬에서도 256행은 같았다. 격리 실행에서 일반 RAG 64건은 모두 캐시에 적중했고 짧은 C32는 적중 0건이었다. 같은 바이너리에서 워밍업 요청의 캐시 사용 여부만 바꿔, 캐시 재사용이 시작 시 배치 형상에 미치는 영향을 확인했다. [준비 상태 및 요청별 검산](../../benchmarks/evidence/20261121-w3-private-startup-warmup/verified.json)

| 캐시 비활성·워밍업 켬 대비 필터·격리 워밍업 | A100 6번 | A100 7번 |
|---|---:|---:|
| 첫 C32 TTFT p50 | −0.30% | −1.37% |
| 첫 C32 TTFT p95 | +27.19% | +16.96% |
| 첫 C32 decode TPOT p50 | +0.40% | +0.43% |
| 첫 C32 출력 처리량 | −1.46% | −0.54% |
| 반복 문서 RAG TTFT p95 | −96.14% | −96.16% |

같은 비교에서 첫 32개 요청의 TTFT p95는 GPU 6번에서 404.00→448.79ms, 7번에서 399.64→407.60ms였다. 후속 64개의 p95는 각각 112.28→75.51ms, 112.23→74.83ms였다. 전체 96건의 p95는 두 GPU 모두 2% 한도를 넘었다. 여덟 실행의 1,312건은 기존 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유가 일치했다. 종료 자원을 회수하고 GPU 6·7의 기존 20GiB 예약으로 복귀했다. 격리 워밍업은 192행의 실행 이력을 복원했으며 혼합 서비스 기본값 채택은 보류한다. [종료·출력·성능 증거](../../benchmarks/evidence/20261121-w3-private-startup-warmup/verified.json) · [독립 재검산](../../benchmarks/evidence/20261121-w3-private-startup-warmup/fetch-summary.sh)

### 첫 C32의 고정 도착 재생

이전 닫힌 C32에서 관측한 두 클라이언트 write 시각표를 같은 요청 본문과 결속했다. GPU 6번은 이전 캐시 비활성·워밍업 켬 실행, 7번은 이전 필터·격리 워밍업 실행의 시각표를 사용했다. 두 시각표는 각각 96건이며 첫 32건을 모두 보내는 데 각각 0.506ms와 0.570ms가 걸렸다. 각 GPU에서 동일한 시각표로 네 조건을 교차하고, 먼저 연결한 첫 32개 소켓의 write 순서를 검사했다. RAG 시각표와 진단 바이너리, 모델·KV 설정은 앞선 대조와 같다. [시각표와 재생 도구](../../benchmarks/evidence/20261121-w3-c32-arrival-replay/replay.py) · [실행 순서](../../benchmarks/evidence/20261121-w3-c32-arrival-replay/campaign.sh)

여덟 실행에서 본문 전송 순서가 시각표와 일치했다. 시각표 대비 전송 오차 p95는 0.78–1.13ms, 대조군과 후보 사이 전송 오차 차이의 p95는 최대 0.60ms였다. 첫 32건의 전송 구간은 1.50–1.80ms로 원본보다 길었다. 재생은 클라이언트 write를 기준으로 비교했고, 아래 판정은 관측된 전송 구간에 적용된다. [시각표 원본](../../benchmarks/evidence/20261121-w3-c32-arrival-replay/schedules.json) · [전송 오차 검산](../../benchmarks/evidence/20261121-w3-c32-arrival-replay/verified.json)

| 캐시 비활성·워밍업 켬 대비 필터·격리 워밍업 | A100 6번 | A100 7번 |
|---|---:|---:|
| 첫 C32 TTFT p50 | +9.20% | +3.48% |
| 첫 C32 TTFT p95 | +0.37% | +0.06% |
| 첫 C32 decode TPOT p50 | +0.16% | +0.22% |
| 첫 C32 출력 처리량 | −0.15% | −0.05% |

직전 닫힌 부하의 첫 C32 p95 증가 27.19%와 16.96%는 이 고정 도착 대조에서 재현되지 않았다. 첫 C32 p50은 두 GPU에서 모두 2% 한도를 넘었다. 기본 캐시 비활성·워밍업 끔을 기준으로 삼은 고정 도착 비교에서는 지연·처리량 조건을 통과했고, 이전 닫힌 부하에서는 같은 비교의 p95가 27.88%, 14.36% 늘었다. 반복 문서 RAG는 필터·격리 워밍업에서 계속 64건 모두 적중했다. 여덟 실행의 1,312건은 기존 이미지와 출력이 일치했다. GPU 6·7의 기존 20GiB 예약 복귀를 확인했다. 혼합 서비스 기본값 채택은 보류한다. [출력·성능 판정](../../benchmarks/evidence/20261121-w3-c32-arrival-replay/verified.json) · [독립 재검산](../../benchmarks/evidence/20261121-w3-c32-arrival-replay/fetch-summary.sh)

### 고정 C32의 서버 구간별 지연

`3951d90c8`의 진단 빌드에서 캐시 비활성·워밍업 켬과 128토큰 필터·격리 워밍업을 대조했다. 같은 바이너리에서 계측을 켠 두 조건과 끈 두 조건을 A100 6·7번에서 역순으로 실행했다. 앞서 보존한 C32 도착 시각표와 반복 문서 RAG 입력을 재사용했다. 각 서버의 서빙 작업을 끝낸 뒤 계측 기록을 출력했다. 첫 빌드는 계측 시계의 Linux FFI가 Rust 린트에 걸렸다. 해당 함수로 예외를 한정한 재빌드가 통과했다. [실행 순서](../../benchmarks/evidence/20261122-w3-c32-stage-timing/campaign.sh) · [계측 패치](../../benchmarks/evidence/20261122-w3-c32-stage-timing/stage-trace.patch.gz) · [첫 빌드 로그](../../benchmarks/evidence/20261122-w3-c32-stage-timing/first-compiler.log.gz) · [재빌드 지문](../../benchmarks/evidence/20261122-w3-c32-stage-timing/build-result.sha256)

SSE 응답의 요청 ID로 클라이언트 write와 서버 HTTP 파싱, 엔진 큐 송수신, 입장, prefill, 첫 텍스트 전송 시각을 연결했다. 각 요청의 write부터 첫 텍스트 수신까지의 시간은 구간 합계와 일치해야 한다. 엔진 큐 대기는 다른 요청의 prefill 배치와 겹친 시간도 집계했다. CUDA 시간은 해당 요청이 포함된 배치 전체의 이벤트 시간으로 같은 스텝의 다른 prefill과 decode도 포함한다. [구간 판정](../../benchmarks/evidence/20261122-w3-c32-stage-timing/verify.py)

| 캐시 비활성·워밍업 켬 대비 필터·격리 워밍업 | 계측 끔 GPU 6번 | 계측 끔 GPU 7번 | 계측 켬 GPU 6번 | 계측 켬 GPU 7번 |
|---|---:|---:|---:|---:|
| 첫 C32 TTFT p50 | +14.55% | +18.29% | +3.29% | −15.19% |
| 첫 C32 TTFT p95 | +1.88% | +4.21% | +0.10% | −2.64% |
| 첫 C32 decode TPOT p50 | +0.21% | +0.22% | +0.10% | −0.04% |
| 반복 문서 RAG TTFT p95 | −96.15% | −96.11% | −96.15% | −96.07% |

계측을 끈 GPU 6번에서는 후속 64건 중 63건이 느려졌고, 같은 요청을 짝지은 TTFT 차이의 중앙값은 +8.71ms였다. GPU 7번에서는 후속 64건의 TTFT p50이 65.24ms에서 57.39ms로 줄었고 p75는 66.69ms에서 79.70ms로 늘었다. 첫 32건의 요청별 차이 중앙값은 GPU 6번에서 +7.00ms, 7번에서 +15.19ms였다. 전체 96건의 p50 증가가 두 GPU에서 반복됐으며, 후속 요청 분포가 움직인 방식은 달랐다. [집단별 클라이언트 지연](../../benchmarks/evidence/20261122-w3-c32-stage-timing/verified.json)

계측을 켠 후속 64건에서 요청별 구간 차이의 중앙값은 아래와 같다. 각 구간의 중앙값은 별도로 구했다.

| 필터·격리 워밍업에서 바뀐 구간 | A100 6번 | A100 7번 |
|---|---:|---:|
| write부터 첫 텍스트 수신 | +2.55ms | −10.28ms |
| 엔진 큐 송신부터 처리 시작 | +2.31ms | −10.26ms |
| 위 큐 대기 중 다른 prefill 배치와 겹친 구간 | +2.25ms | −10.22ms |
| 해당 요청이 포함된 prefill 배치의 CUDA 시간 | +0.25ms | −0.10ms |

두 GPU 모두 계측 실행에서 가장 큰 중앙값 변화가 엔진 큐 대기에 있었다. 필터·격리 워밍업의 계측 켬·끔 자체 대조에서 전체 TTFT p50은 GPU 6번에서 −3.70%, 7번에서 −27.11%였다. 요청 도착 시각표는 같았고 전송 오차 p95는 0.92–1.27ms, 조건 간 전송 오차 차이 p95는 최대 0.93ms였다. 계측 여부와 실행 순서가 함께 바뀌어 계측을 끈 실행의 회귀 원인은 미확정이다. 혼합 서비스의 기본값 채택을 보류한다. [구간별 검산](../../benchmarks/evidence/20261122-w3-c32-stage-timing/verified.json)

계측을 켠 네 실행의 C32 384건을 누락 없이 연결했다. 여덟 실행의 1,312건은 기존 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유가 일치했다. 필터·격리 워밍업의 일반 RAG는 네 실행 모두 64건이 적중했고 짧은 요청 적중은 0건이었다. 종료 자원을 회수했고 GPU 6·7은 동일 홀더의 기존 20GiB 예약으로 복귀했다. 재현용 `dispatch.sha256`의 `replay-timing.patch`는 저장소의 `replay-timing.patch.gz`를 풀어 준비한다. [원본 지문](../../benchmarks/evidence/20261122-w3-c32-stage-timing/result.sha256) · [독립 재검산](../../benchmarks/evidence/20261122-w3-c32-stage-timing/fetch-summary.sh)

### 기존 요청 관측값으로 대조한 C32 입장 지연

앞선 상세 계측을 끈 실행에서 전체 C32의 첫 토큰 중앙값 증가가 반복됐다. 같은 `3951d90c8` 배포 소스와 캐시 진단 설정에서, Riley가 원래 기록하는 요청별 관측값을 서버 종료 후 출력하는 코드만 추가했다. 요청을 처리하는 동안 새 타임스탬프나 잠금을 추가하지 않았다. 각 GPU에서 캐시 비활성·워밍업 켬과 128토큰 필터·격리 워밍업을 두 번 대조했다. 각 GPU의 두 번째 대조와 다른 GPU의 실행 순서를 뒤집었으며, 매 실행에서 새 서버를 띄우고 이전과 같은 합성 RAG 및 C32 도착 시각표를 재생했다. [종료 시 추출 패치](../../benchmarks/evidence/20261122-w3-c32-admission-observations/export-observations.patch.gz) · [실행 순서](../../benchmarks/evidence/20261122-w3-c32-admission-observations/campaign.sh)

SSE 요청 ID로 서버의 `queue_wait`, `time_to_first_token`을 클라이언트 측 TTFT에 연결했다. `queue_wait`는 HTTP 처리에서 백엔드 제출을 시작한 시점부터 입장이 반환되기까지의 시간이다. 이 구간에는 제출 스레드의 토큰화, 엔진 큐 대기, 스케줄러 입장이 모두 들어간다. `time_to_first_token`은 입장 뒤 요청 추적기를 만든 시점부터 서버가 첫 텍스트를 보낼 때까지다. 클라이언트 TTFT에서 두 서버 구간과 클라이언트 write 이전 시간을 빼면 HTTP 처리와 전송 등을 포함한 나머지 시간이 남는다. [관측값 연결과 계산](../../benchmarks/evidence/20261122-w3-c32-admission-observations/verify.py)

| 캐시 비활성·워밍업 켬 대비 필터·격리 워밍업 | GPU 6번 1차 | GPU 6번 2차 | GPU 7번 1차 | GPU 7번 2차 |
|---|---:|---:|---:|---:|
| 전체 C32 TTFT p50 | −6.73% | +7.18% | −22.79% | +13.17% |
| 전체 C32 TTFT p95 | −0.28% | +1.89% | −5.24% | −0.07% |
| 전체 C32 decode TPOT p50 | −0.07% | −0.10% | −0.57% | +0.44% |
| 반복 문서 RAG TTFT p95 | −96.15% | −96.16% | −96.19% | −96.15% |

두 GPU의 2차에서 전체 C32 TTFT p50이 2% 한도를 넘었고 후속 64건은 모두 느려졌다. 후속 64건을 입력별로 짝지은 차이의 중앙값은 아래와 같다. 네 대조에서 백엔드 제출부터 입장까지의 변화는 클라이언트 TTFT 변화와 같은 방향이었다. 입장 뒤 첫 텍스트 구간의 변화는 절댓값 0.27ms 이하였다.

| 후속 64건의 요청별 차이 중앙값 | GPU 6번 1차 | GPU 6번 2차 | GPU 7번 1차 | GPU 7번 2차 |
|---|---:|---:|---:|---:|
| 클라이언트 TTFT | −3.66ms | +5.36ms | −1.53ms | +8.09ms |
| 백엔드 제출부터 입장까지 | −3.34ms | +2.37ms | −1.28ms | +6.89ms |
| 입장 뒤 첫 텍스트까지 | +0.03ms | −0.03ms | −0.27ms | +0.10ms |

GPU 6번의 2차는 후속 64건의 write 뒤 나머지 구간도 약 3.04ms 늘었다. GPU 7번의 2차 첫 32건에서는 해당 구간의 p50이 80.73ms에서 0.43ms로 줄었고 write 이전 p50은 두 조건 모두 0.04ms였다. 첫 32건의 입장 뒤 첫 텍스트 중앙값은 같은 대조에서 148.89ms에서 184.06ms로 늘었다. 따라서 첫 유입과 후속 유입의 구간별 변화는 따로 확인해야 한다. 현재 관측으로는 write 뒤 나머지 구간을 HTTP 입구와 SSE 전달 시간으로 나눌 수 없다. [요청 집단별 지연](../../benchmarks/evidence/20261122-w3-c32-admission-observations/verified.json)

여덟 실행에서 첫 32건의 write는 1.37–1.79ms에 마쳤고, 대조군 간 전송 오차 차이 p95는 최대 1.09ms였다. 서버 관측값 1,312건을 누락 없이 응답과 연결했고, 기존 배포 이미지 기준의 입력·생성 토큰 ID, 텍스트, 종료 사유와 일치했다. 캐시 후보의 RAG는 실행마다 64건이 적중했으며 짧은 C32는 적중이 0건이었다. 종료 뒤 활성·대기 요청과 GPU 자원은 0으로 돌아왔고 GPU 6·7은 동일 홀더의 20GiB 예약으로 복귀했다. 후보의 기본값 적용은 보류한다. [반복별 검산](../../benchmarks/evidence/20261122-w3-c32-admission-observations/verified.json) · [원본 지문](../../benchmarks/evidence/20261122-w3-c32-admission-observations/result.sha256) · [독립 재검산](../../benchmarks/evidence/20261122-w3-c32-admission-observations/fetch-summary.sh)

### 고정 C32의 일반 prefill 행 예산

엔진 작업자는 실행 중인 스텝이 반환된 다음 새 요청 명령을 처리한다. 앞선 반복 시험에서 후속 64건의 지연 차이와 백엔드 제출부터 입장까지의 차이가 같은 방향이어서, 짧은 요청의 스텝을 줄이면 이 대기가 얼마나 변하는지 시험했다. 기존 Coder-14B closed C32의 행 예산 시험은 vLLM과 TPOT을 대조했다. 이번 시험은 필터·격리 워밍업과 앞선 반복 문서 RAG를 포함한 고정 도착 조건에서 첫 토큰을 측정했다. [기존 closed C32 판정](../goal.md#6-운영-규칙) · [직전 C32 입장 지연](#기존-요청-관측값으로-대조한-c32-입장-지연)

캐시 비활성·워밍업 켬의 256행과 필터·격리 워밍업의 256·192·128행을 같은 진단 바이너리에서 대조했다. GPU 6·7번의 실행 순서를 뒤집었고 모든 조건에 같은 도착 시각표, KV 크기, 1,024행 burst 예산과 512토큰 burst 문턱을 적용했다. 실제 형상별 스텝 수와 GPU 시간은 `/metrics`에서 검산했다. 서버의 기존 요청 관측값은 종료 뒤 추출해 SSE 요청 ID와 연결했다. [실행 순서](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/campaign.sh) · [바이너리 지문](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/compiled-inputs.sha256) · [출력·지연·범위 검산](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/verify.py)

아래 수치는 같은 GPU의 필터·격리 워밍업 256행 대비 변화율이다.

| 측정 지표 | 192행 GPU 6번 | 192행 GPU 7번 | 128행 GPU 6번 | 128행 GPU 7번 |
|---|---:|---:|---:|---:|
| C32 TTFT p50 | +109.33% | +64.31% | +252.19% | +219.82% |
| C32 TTFT p95 | +19.28% | +27.90% | +54.04% | +56.47% |
| C32 decode TPOT p50 | +0.56% | +0.07% | −2.26% | −2.30% |
| C32 출력 처리량 | −2.20% | −1.46% | −4.78% | −4.18% |
| 반복 문서 RAG TTFT p95 | −0.69% | −0.32% | +24.43% | +28.34% |

각 상한에 도달한 배치의 평균 GPU 시간은 256행에서 37.18–37.74ms, 192행에서 33.78–34.01ms, 128행에서 28.16ms였다. C32 전체의 GPU 스텝은 204–205개에서 각각 209개와 221–222개로 늘었다. 192·128행은 캐시 비활성 256행과 비교해도 두 GPU 모두 C32 TTFT 한도를 넘었다. 필터·격리 워밍업 256행은 이번 대조에서 두 GPU의 회귀 기준을 충족했으며, [앞선 반복](#기존-요청-관측값으로-대조한-c32-입장-지연)의 p50 회귀를 함께 반영해 혼합 서비스 기본값은 보류한다. [배치 및 회귀 판정](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/verified.json)

후속 64건이 본문을 보내기 직전, 이미 보낸 요청 중 마지막 텍스트를 아직 받지 못한 요청이 32건 이상이었던 횟수는 필터·격리 워밍업 256행에서 GPU 6·7번 각각 5회·0회였다. 192행에서는 58회·43회, 128행에서는 55회·54회였다. 같은 요청의 지연을 짝지었을 때 192행에서 write 이후 두 서버 관측 구간을 제외한 나머지 시간은 중앙값 기준 50.99ms·22.01ms 늘었다. 128행의 증가량은 120.71ms·81.94ms였다. 이 집계는 클라이언트가 아직 마지막 텍스트를 받지 못한 요청 수를 보여 준다. 서버의 HTTP 처리와 SSE 전달 중 어느 부분에 지연이 집중됐는지는 남아 있다. [요청 집단별 지연과 동시 미완료 요청](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/verified.json)

대조 조건 간 write 오차 차이의 p95는 최대 0.87ms였다. 여덟 실행의 1,312건은 기존 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유가 일치했다. 필터 조건의 RAG 64건은 매번 캐시에 적중했고 짧은 요청 적중은 0건이었다. GPU 자원을 회수했고 A100 6·7번은 기존 홀더의 20GiB 예약으로 복귀했다. [원본 지문](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/result.sha256) · [독립 재검산](../../benchmarks/evidence/20261122-w3-c32-steady-budgets/fetch-summary.sh)

### 고정 C32에서 HTTP 워커 수와 연결 큐 대기

Riley의 HTTP 워커는 스트리밍이 끝날 때까지 연결을 맡는다. 앞선 128행 시험에서 후속 요청이 쌓이고 클라이언트 write 이후의 미계측 지연이 늘었다. 이 구간에서 HTTP 연결 큐의 기여를 확인하기 위해, 캐시 비활성·워밍업 켬과 필터·격리 워밍업의 256행 설정에서 HTTP 워커를 32개와 64개로 대조했다. 먼저 관측된 연결 적체를 다시 만들기 위해 필터의 128행에서도 두 워커 수를 비교했다. GPU별 실행 순서를 서로 뒤집었고, 최대 활성 시퀀스 32와 연결 큐 용량 128, 동일 바이너리·모델·도착 시각표를 적용했다. [실행 순서](../../benchmarks/evidence/20261122-w3-c32-http-workers/campaign.sh) · [진단 패치](../../benchmarks/evidence/20261122-w3-c32-http-workers/http-queue.patch.gz)

리스너가 연결을 큐에 넣기 시작한 시점부터 워커가 받아 HTTP 처리를 시작하기까지를 기록했다. 종료 후 요청 ID로 관측을 SSE 응답과 연결했다. 소켓을 미리 열 수 있어 워커 대기의 일부는 클라이언트가 본문을 보내기 전에 발생할 수 있다. 해당 구간은 클라이언트 TTFT의 나머지 구간과 각각 비교했다. [관측과 출력 검산](../../benchmarks/evidence/20261122-w3-c32-http-workers/verify.py)

| 필터·격리 워밍업 128행, 워커 32→64개 | A100 6번 | A100 7번 |
|---|---:|---:|
| 후속 64건의 HTTP 연결 큐 p95 | 248.678→0.024ms | 224.446→0.020ms |
| 후속 64건의 입장 뒤 첫 텍스트 차이 중앙값 | +141.95ms | +85.75ms |
| 후속 64건의 클라이언트 TTFT 차이 중앙값 | +0.77ms | −1.27ms |
| 전체 C32 TTFT p95 | 530.33→531.35ms | 530.25→531.10ms |

후속 요청의 연결 큐 대기를 줄인 128행 대조에서는 첫 텍스트까지의 지연이 입장 이후 구간에 쌓였다. 백엔드의 입장 반환은 스케줄러가 요청을 큐에 등록했다는 뜻이다. 입장 뒤 첫 텍스트까지의 관측은 활성 슬롯 대기와 prefill을 모두 포함한다. 64워커의 128행 C32는 두 GPU에서 221개 스텝을 써, 같은 64워커의 필터 256행 204개보다 많았고 TTFT p95도 각각 54.81%, 48.26% 늘었다. [구간별·배치별 검산](../../benchmarks/evidence/20261122-w3-c32-http-workers/verified.json)

256행의 32워커에서는 GPU 6번 후속 요청의 연결 큐 p95가 캐시 비활성 53.673ms, 필터 8.588ms였다. 64워커에서 각각 0.021ms, 0.019ms로 줄었고 같은 캐시 설정 대비 전체 C32 p95는 4.70%, 3.87% 줄었다. GPU 7번에서는 연결 큐 p95가 네 조건에서 모두 0.027ms 미만이었다. GPU 7번의 필터 설정에서 64워커는 전체 C32 p95를 2.08% 늘렸다. 같은 GPU의 64워커 캐시 비활성 기준 대비 필터의 C32 p95는 4.06% 늘어 2% 회귀 기준을 넘었다. 측정한 두 GPU의 합성 부하를 기준으로 64워커와 필터의 공통 적용을 보류한다. [회귀 검산](../../benchmarks/evidence/20261122-w3-c32-http-workers/verified.json)

조건 간 클라이언트 write 오차 차이 p95는 최대 0.39ms였다. 12회 실행의 1,968건은 기존 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유가 일치했다. 필터 실행에서는 반복 문서 RAG 64건이 모두 적중했고 C32 적중은 0건이었다. 종료 자원을 회수했고 A100 6·7번은 동일 홀더의 20GiB 예약으로 복귀했다. [빌드 지문](../../benchmarks/evidence/20261122-w3-c32-http-workers/build-result.sha256) · [원본 지문](../../benchmarks/evidence/20261122-w3-c32-http-workers/result.sha256) · [독립 재검산](../../benchmarks/evidence/20261122-w3-c32-http-workers/fetch-summary.sh)

### 고정 C32의 활성 슬롯 대기와 첫 prefill

앞선 HTTP 워커 시험에서 연결 큐 대기가 줄어든 만큼 입장 뒤 첫 토큰 지연이 늘어난 구간을 다시 측정했다. Coder-14B의 128토큰 캐시 필터와 격리 워밍업을 고정하고 일반 prefill 예산 128·256행과 HTTP 워커 32·64개를 교차했다. 두 GPU의 실행 순서를 뒤집었다. 128행의 두 워커 조건은 같은 바이너리로 계측을 끄고도 실행했다. [실행 조건](../../benchmarks/evidence/20261123-w3-c32-first-prefill/campaign.sh)

서버가 입장 응답을 반환하기까지의 시간은 엔진이 스케줄러에 요청을 등록한 시점을 포함한다. 스케줄러 등록부터 KV 시퀀스를 할당해 요청을 활성화할 때까지, 활성화부터 첫 prefill 계획까지의 시간을 기존 스케줄러 시각으로 계산했다. 실행 시작과 완료, 첫 서버 응답과 클라이언트 첫 콘텐츠는 호스트 단조 시계로 대조했다. SSE 응답 ID로 각 요청의 지연을 연결했다. [시각 계측](../../benchmarks/evidence/20261123-w3-c32-first-prefill/prepare.py) · [응답과 시각 판정](../../benchmarks/evidence/20261123-w3-c32-first-prefill/verify.py)

| 128행, 후속 64건의 p95, 워커 32 → 64개 | A100 6번 | A100 7번 |
|---|---:|---:|
| HTTP 연결 큐 | 254.958 → 0.026ms | 173.800 → 0.020ms |
| 요청 등록부터 활성화까지 | 112.664 → 227.664ms | 57.051 → 200.135ms |
| prefill 조각 사이 | 85.066 → 84.976ms | 56.739 → 113.177ms |
| 클라이언트 TTFT | 401.269 → 341.467ms | 303.576 → 339.342ms |

두 GPU 모두 연결 큐 p95는 줄고 활성화 대기 p95는 늘었다. 활성화부터 첫 prefill 계획까지의 p95는 네 조건 모두 1ms 미만이었다. 클라이언트의 후속 요청 TTFT p95는 GPU에 따라 감소와 증가가 엇갈렸다. 분포별 p95는 개별 요청의 합산 지연을 의미하지 않는다.

직전 RAG 부하는 모든 실행에서 활성·대기 요청과 요청이 쓰던 KV를 회수했다. 짧은 입력 최대 77토큰, 출력 상한 256토큰, 16토큰 KV 블록과 활성 32개를 적용하면 약정 KV 상한은 672블록이다. 설정 한도는 2,576블록이었다. 이 부하에서는 KV 약정에 여유가 있어 등록 뒤 병목은 32개 활성 슬롯 대기였다. [실행별 한도와 종료 판정](../../benchmarks/evidence/20261123-w3-c32-first-prefill/verified.json)

계측을 끈 128행에서도 워커를 64개로 늘리면 두 GPU의 연결 큐 p95는 249ms·225ms에서 0.024ms 미만으로 줄었다. 전체 96건의 TTFT p95는 GPU 6번에서 13.51%, 7번에서 0.01% 증가했다. 계측을 켰을 때의 전체 지연 변화와 크기가 달라 계측 실행은 구간 비교에 사용했다. 계측을 켠 256행에서 후속 64건의 활성 슬롯 대기 p95는 두 워커 설정과 두 GPU 모두 0.04ms 미만이었다. 워커 수를 고정하고 256행을 128행으로 줄인 네 비교에서 전체 TTFT p95는 52.88–56.98% 늘었고 decode TPOT p50은 1.66–2.32% 줄었다. 128행과 64워커의 기본값 적용은 보류한다. [조건별 비교](../../benchmarks/evidence/20261123-w3-c32-first-prefill/verified.json)

12회 실행의 1,968건이 기존 배포 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유까지 일치했다. 조건 간 실제 write 오차 차이 p95는 최대 0.875ms였다. 모든 실행에서 종료 자원을 회수하고 A100 6·7번의 기존 20GiB 예약을 복구했다. [실행 로그](../../benchmarks/evidence/20261123-w3-c32-first-prefill/campaign.log) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-first-prefill/result.sha256) · [원본 재검산](../../benchmarks/evidence/20261123-w3-c32-first-prefill/fetch-summary.sh)

### 고정 C32에서 짧은 입력의 burst 문턱 조정

직전 시험에서 128행의 꼬리 지연은 활성 슬롯 대기에 집중됐다. 같은 `fixed-tile-unsplit-v1` 경로는 최대 활성 시퀀스 32개를 요구한다. 이 경로에서는 낮은 decode 부하에서 짧은 prefill을 한 스텝에 더 많이 처리하면 슬롯 대기가 줄어드는지 확인했다. Coder-14B, 활성 32개, HTTP 워커 64개, 입력 길이 128토큰 이상에서만 캐시 조회, 격리 워밍업과 기존 고정 도착 시각표를 사용했다. 계측용 prefill 추적은 끄고 기존 HTTP 관측만 종료 후 읽었다. [프로파일의 실행 제한](../../crates/riley-server/src/main.rs#L1614) · [실행 순서](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/campaign.sh)

같은 바이너리에 GPU를 노출하지 않고 활성 48개를 지정했다. `fixed-tile-unsplit-v1`은 32개 상한 오류를 반환했다. `page-v1`은 CLI 검사를 통과했고, 시험에서 일부러 제공하지 않은 체크포인트 경로를 여는 단계까지 진행했다. [CLI 대조 결과](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/active-cli.json) · [재현 명령](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/check-active-cli.sh) · [이후 GPU 대조](#page-attention의-활성-슬롯-32개와-48개)

기준 두 개는 일반 예산 128·256행에 512토큰 burst 문턱을 뒀다. 후보 두 개는 128행과 64토큰 문턱을 사용하며 decode가 각각 8행·16행 이하일 때 1,024행까지 허용했다. 그보다 decode가 많으면 128행을 사용했다. C32 입력은 70–77토큰이다. 실제로 128행을 넘는 스텝을 집계하고, 동일 GPU에서 각 후보를 128행 기준과 256행 기준 모두에 대조했다. [요청과 형상 검산](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/verify.py)

| C32 실행 | TTFT p95, GPU 6번 | TTFT p95, GPU 7번 | 128행 초과 GPU 스텝, 6번 / 7번 |
|---|---:|---:|---:|
| 256행 기준 | 347.35ms | 341.81ms | 31 / 31 |
| 128행 기준 | 532.87ms | 531.15ms | 0 / 0 |
| 128행 + decode 8행 이하에서 burst | 455.39ms | 425.64ms | 1 / 1 |
| 128행 + decode 16행 이하에서 burst | 380.40ms | 334.66ms | 3 / 5 |

두 후보는 128행 기준 대비 TTFT p95를 14.54–36.99% 줄였다. 256행 기준 대비 16행 후보는 GPU 6번에서 TTFT p95가 9.51% 늘고 GPU 7번에서 2.09% 줄었다. 256행 기준과 대조한 네 비교에서 TTFT p50은 191.35–241.19% 늘었고 출력 처리량은 3.77–4.69% 줄었다. Decode TPOT p50은 2.49–2.87% 줄었으며 반복 문서 RAG의 TTFT p95 변화는 −0.38%~+1.43%였다. 짧은 요청 중앙값과 처리량 기준을 충족하지 못해 두 후보를 기본값으로 채택하지 않았다. [조건별 수치와 회귀 기준](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/verified.json)

8회 실행의 1,312건은 기존 배포 이미지와 입력·생성 토큰 ID, 텍스트, 종료 사유가 일치했다. 조건 간 실제 write 오차 차이 p95는 최대 0.346ms였다. 모든 서버가 종료 자원을 회수했고 A100 6·7번의 기존 20GiB 예약이 복귀했다. [실행 로그](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/campaign.log) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/result.sha256) · [원본 재검산](../../benchmarks/evidence/20261123-w3-c32-burst-thresholds/fetch-summary.sh)

### Page attention의 활성 슬롯 32개와 48개

`page-v1`이 48개 설정을 받아들인 CLI 대조에 이어 실제 Coder-14B 출력을 검증했다. 두 GPU에서 `page-v1`의 활성 슬롯 32·48개를 각각 128행으로 실행했다. `fixed-tile-unsplit-v1` 32개도 128·256행에서 실행해 같은 예산의 attention 모드 차이와 256행 기준 대비 지연을 확인했다. 두 GPU의 실행 순서를 뒤집고 HTTP 워커 64개, KV 2,576블록, 캐시 필터와 격리 워밍업 모드, 도착 시각표를 맞췄다. [실행 순서](../../benchmarks/evidence/20261123-w3-c32-page-slots/campaign.sh)

워밍업이 끝난 뒤 HTTP 서비스를 시작하기 직전에 스케줄러 점유량 집계를 켰다. 실제로 커밋된 스텝에서 활성 수와 decode 행 수의 최댓값, 활성 요청이 32개를 넘은 스텝 수를 집계하고 종료 뒤 읽었다. 요청별 prefill 추적은 끈 상태로 기존 HTTP 관측, 스텝 형상, 입출력 지문을 대조했다. 활성 한도에 따라 내부 워밍업 요청 구성은 달라질 수 있어 기동 시간과 워밍업 건수를 실행별로 기록했다. [점유량 진단](../../benchmarks/evidence/20261123-w3-c32-page-slots/prepare-peak.py) · [응답과 스케줄러 점유량 검산](../../benchmarks/evidence/20261123-w3-c32-page-slots/verify.py)

8,192토큰 문맥과 최대 1,024행의 조합에서 `page-v1`은 attention 준비 시 행·페이지·head별 F32 부분 상태를 확보한다. Coder-14B의 40개 query head와 16토큰 페이지, head당 130개 F32 값으로 계산한 부분 상태 버퍼는 10.16GiB다. 같은 입력의 출력 상한까지 합친 최대 요청 길이는 2,239토큰이었다. 문맥 상한을 2,304토큰으로 낮춘 대조는 입력과 도착 시각표를 유지하고 해당 버퍼를 계산상 2.86GiB로 줄인다. 두 문맥 설정의 결과를 각각 판정한다. [작업 공간 준비 코드](../../crates/riley-runtime/src/llama/executor/owner.rs#L534) · [부분 상태 크기](../../crates/riley-cuda/src/batch.rs#L793) · [2,304토큰 대조 실행](../../benchmarks/evidence/20261123-w3-c32-page-slots/bounded-campaign.sh)

**8,192토큰 문맥.** 두 A100 40GiB 장비에서 `page-v1`의 32개·48개 조건 네 건 모두 readiness 전에 `ragged_paged_attention` CUDA 버퍼 할당에서 OOM으로 종료됐다. 같은 바이너리의 고정 타일 32개 기준 네 건은 기동했고, 준비 상태의 서버 관리 장치 할당은 각각 38,625,687,680바이트였다. 기준의 656건은 기존 이미지의 입력·생성 토큰 ID, 텍스트 및 종료 사유와 일치했다. 실패한 네 건을 포함해 각 실행 뒤 기존 GPU 예약으로 복귀했다. [8,192토큰 재검산](../../benchmarks/evidence/20261123-w3-c32-page-slots/initial-verified.json) · [8,192토큰 로그 지문](../../benchmarks/evidence/20261123-w3-c32-page-slots/initial-result.sha256)

**2,304토큰 문맥.** 최대 스텝 1,024행을 유지하면 `page-v1` 32개·48개 네 조건은 계속 같은 CUDA 버퍼 할당에서 실패했다. GPU 7번 고정 타일 기준 두 건은 328개 출력이 기존 이미지와 일치했다. GPU 6번은 고정 타일 시험 중 다른 프로세스가 관측된 실행 한 건을 제외했고, 후속 기준 한 건은 예약 검사에서 실행 전에 차단했다. [OOM 및 제외된 조건](../../benchmarks/evidence/20261123-w3-c32-page-slots/rejected-row-verified.json)

**스텝 상한 192행 대조.** 문맥 상한을 8,192토큰으로 유지하고 최대 스텝·prefill 조각·버킷을 192행으로 맞추면 같은 부분 상태 버퍼는 계산상 1.90GiB다. 첫 시도의 GPU 7번 세 조건은 프로파일에서 상속한 192행 plateau 경계가 예산과 같아 CLI에서 거부됐다. GPU 6번 세 조건은 예약 검사에서 차단됐다. 이후 고정 타일 32개와 `page-v1` 32·48개에 128행 plateau 경계만 명시해 CPU에서 CLI 검사를 통과했다. 수정한 비교는 일반 예산 128행과 입력·출력 한도, 도착 시각표를 유지했다. [CLI 거부 실행](../../benchmarks/evidence/20261123-w3-c32-page-slots/rejected-row-verified.json) · [수정한 CLI 검사](../../benchmarks/evidence/20261123-w3-c32-page-slots/row-cli-verified.json) · [수정한 실행 순서](../../benchmarks/evidence/20261123-w3-c32-page-slots/workspace-live-campaign.sh)

| GPU 7번, 최대 스텝 192행 | 실제 활성 / decode 최댓값 | C32 TTFT p95 | C32 TPOT p50 | 기존 이미지와 같은 출력 |
|---|---:|---:|---:|---:|
| 고정 타일 32개 | 32 / 32 | 532.37ms | 26.34ms | 164/164 |
| Page 32개 | 32 / 32 | 1,341.98ms | 32.16ms | 80/164 |
| Page 48개 | 48 / 48 | 858.99ms | 36.42ms | 80/164 |

Page 48개에서 활성 요청이 32개를 넘은 커밋 스텝은 77개였다. Page 32개 대비 TTFT p95는 35.99% 줄고 출력 처리량은 5.97% 늘었다. Decode TPOT p50은 13.22% 증가했다. 두 page 실행은 서로 164/164건의 입력·생성 토큰 ID, 텍스트와 종료 사유가 일치했다. 고정 타일과는 두 page 실행 모두 짧은 C32 42/96건, RAG 35/64건, 캐시를 채우는 요청 3/4건이 일치했다. Page 48개는 고정 타일 대비 TTFT p95가 61.35%, decode TPOT p50이 38.25% 늘어 정확성과 성능 기준을 모두 충족하지 못했다. [출력 지문과 지연](../../benchmarks/evidence/20261123-w3-c32-page-slots/verified.json)

수정한 비교는 다른 프로세스를 감지한 GPU 6번의 세 조건을 시작 전에 제외했다. GPU 7번의 세 실행 492건은 정상 완료하고 종료 자원을 회수했다. 쌍별 클라이언트 write 오차 차이 p95는 최대 0.549ms였다. 최종 스냅샷에서 GPU 6·7번의 동일 홀더가 20GiB씩 예약했고, 다른 GPU 프로세스는 관측되지 않았다. 문맥 상한은 8,192토큰이었으며 실제 요청에서 시험한 최대 길이는 출력 상한을 포함해 2,239토큰이었다. [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-page-slots/result.sha256) · [로그](../../benchmarks/evidence/20261123-w3-c32-page-slots/campaign.log) · [독립 재검산](../../benchmarks/evidence/20261123-w3-c32-page-slots/fetch-summary.sh)

### 고정 타일의 활성 슬롯 32개와 48개

`fixed-tile-unsplit-v1`을 사용한 이전 진단 빌드에서 CLI의 활성 슬롯 상한만 32개에서 48개로 높였다. 체크포인트가 없는 경로를 지정한 CPU 사전 검사에서 이전 바이너리는 48개를 거부했고, 진단 바이너리는 48개 설정으로 모델 경로를 확인하는 단계까지 진행했다. 진단 바이너리에서 49개와 문맥 상한 8,193토큰은 거부됐다. [진단 빌드 변경](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/prepare.py) · [CLI 검사](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/check-cli.py)

같은 진단 바이너리로 Coder-14B의 32·48슬롯과 일반 스텝 예산 128·256행을 교차했다. 문맥 상한 8,192토큰, 최대 스텝과 prefill 조각 1,024행, KV 2,576블록, HTTP 워커 64개, prefix 필터와 격리 워밍업을 맞췄다. GPU별로 보존된 짧은 요청 96개의 도착 시각표를 재생했고, 캐시를 채우는 4개 요청과 RAG 64건을 같은 순서로 실행했다. GPU 6·7은 실행 순서를 서로 뒤집었다. [실행 순서와 옵션](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/campaign.sh) · [비교 기준](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/verify.py)

| GPU | 일반 예산 | 48슬롯에서 실제 활성 / decode 최대 | C32 TTFT p95, 32 → 48슬롯 | C32 TTFT p50 변화 | C32 TPOT p50 변화 |
|---:|---:|---:|---:|---:|---:|
| 6 | 128행 | 45 / 32 | 533.10 → 532.04ms (−0.20%) | +6.28% | +0.04% |
| 7 | 128행 | 43 / 32 | 531.44 → 531.20ms (−0.05%) | +4.04% | −0.003% |
| 6 | 256행 | 32 / 32 | 344.35 → 345.84ms (+0.43%) | +10.87% | +0.54% |
| 7 | 256행 | 32 / 32 | 333.24 → 340.09ms (+2.06%) | +8.46% | +0.996% |

128행의 48슬롯 설정에서는 각 GPU에서 32개를 넘긴 커밋 스텝이 32개였다. 후속 64건의 HTTP 연결 큐 p95는 두 설정 모두 0.02–0.03ms였고, 서버 관측의 입장 이후 첫 텍스트 p95는 슬롯을 늘렸을 때 31.69ms·55.75ms 증가했다. 256행에서는 48슬롯을 설정해도 실제 활성 수가 32개에 머물렀다. 전체 실행에서 관측된 decode 최대는 32행이었다. [실제 점유량과 지연 구간](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/verified.json)

8개 실행의 1,312개 응답은 모두 기존 이미지와 입력·생성 토큰 ID, 텍스트 및 종료 사유가 일치했다. 48슬롯 설정의 준비 시점 장치 할당은 38,630,558,144바이트로 32슬롯보다 4,870,464바이트 늘었다. 쌍별 클라이언트 write 오차 차이 p95는 최대 0.593ms였다. 시험 요청의 최대 길이는 출력 상한을 포함해 2,239토큰이었다. 두 GPU는 각 실행 이후 활성·대기 요청과 장치 자원을 회수하고 기존 20GiB 예약으로 복귀했다. [CLI 결과](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/cli.json) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/result.sha256) · [독립 재검산](../../benchmarks/evidence/20261123-w3-c32-fixed-slots/fetch-summary.sh)

128행에서 활성 요청 수는 늘었고 C32 TTFT p95는 0.20% 이하로 줄었다. C32 TTFT p50은 네 비교 모두 2% 회귀 한도를 넘었고, 가장 빠른 256행·32슬롯 기준보다 128행·48슬롯의 p95가 두 GPU에서 54.51%·59.41% 높았다. 이 도착 시각표에서는 활성 슬롯을 늘리는 후보를 채택하지 않는다.

### 256행 예산을 고정한 짧은 입력 burst

앞선 짧은 입력 burst 후보는 128행 예산에서 시험했다. 256행 기준과 비교하면 일반 예산과 burst 조건이 함께 달라진다. 이번 비교는 앞선 burst 시험과 같은 Coder-14B 바이너리에서 일반 예산을 256행으로 고정하고, burst 진입 조건만 바꿨다. 기준은 최소 512토큰·decode 최대 32행, 후보는 최소 64토큰·decode 최대 8행 또는 16행이다. 최대 스텝과 prefill 조각은 1,024행이다. [이전 문턱 대조](#고정-c32에서-짧은-입력의-burst-문턱-조정) · [실행 조건](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/campaign.sh)

두 GPU에서 고정 타일 attention, 활성 슬롯 32개, HTTP 워커 64개, KV 2,576블록, prefix 필터와 격리 워밍업을 맞췄다. 각 GPU에서 저장된 짧은 C32 요청 96건의 도착 시각표와 캐시를 채우는 요청 4건, 반복 문서 RAG 64건을 사용하고 실행 순서를 서로 뒤집었다. 확인 기준은 기존 이미지와 출력 일치, 짧은 요청의 TTFT p50·p95와 처리량, C32·RAG TPOT p50 및 RAG TTFT p95다. [재생 인자](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/prepare-runner.py) · [같은 비교 기준의 검산](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/compose-verifier.py)

| GPU | decode 문턱 | C32 TTFT p95, 기준 → 후보 | C32 TTFT p50 변화 | C32 TPOT p50 변화 | C32 출력 처리량 변화 | 256행 초과 스텝 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8행 | 363.71 → 421.53ms (+15.90%) | +35.18% | +0.52% | −0.14% | 1 |
| 7 | 8행 | 340.18 → 298.13ms (−12.36%) | +13.38% | −0.39% | −0.08% | 2 |
| 6 | 16행 | 363.71 → 284.29ms (−21.84%) | +151.91% | −2.01% | −2.17% | 5 |
| 7 | 16행 | 340.18 → 316.91ms (−6.84%) | +150.92% | −1.11% | −1.78% | 4 |

16행 후보에서 첫 32건의 TTFT p95는 416.26 → 284.29ms와 392.39 → 316.91ms로 줄었고, 후속 64건의 p95는 105.94 → 274.46ms와 66.68 → 262.79ms로 늘었다. 이 후보의 RAG TTFT p95는 두 GPU에서 각각 1.36%, 0.45% 줄었다. 일반 예산을 256행으로 고정해도 두 버스트 후보는 모든 비교에서 짧은 요청 TTFT p50의 2% 회귀 한도를 넘었다. 16행 후보는 GPU 6의 처리량 기준도 넘겨 두 후보를 미채택으로 판정했다. [후속 요청의 지연과 가드레일](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/verified.json)

6개 실행에서 응답 984건이 기존 이미지와 입력·생성 ID, 텍스트, 종료 사유까지 일치했다. 쌍별 클라이언트 write 오차 차이 p95는 최대 0.727ms였다. 정상 종료 뒤 활성·대기 요청과 장치 자원이 회수됐고 GPU 6·7은 각각 기존 20GiB 예약으로 복귀했다. [실행 로그](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/campaign.log) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/result.sha256) · [독립 재검산](../../benchmarks/evidence/20261123-w3-c32-steady256-burst/fetch-summary.sh)

### 짧은 입력 전용 512행 버스트

앞선 256행 대조에서 1,024행 버스트로 첫 요청 묶음의 TTFT p95가 줄고, 후속 요청의 TTFT p95가 늘었다. 이번 진단 빌드는 기준의 일반 예산 256행과 최대 1,024행을 사용한다. 실행 가능한 prefill의 원래 프롬프트가 모두 512토큰 미만이고, 그중 하나 이상에 64토큰 이상이 남았으며, decode 행 수가 설정한 8행 또는 16행 이하면 스텝 전체 예산을 512행까지 허용한다. 512토큰 이상의 프롬프트가 prefill에 포함된 스텝은 기준의 긴 입력 규칙을 따른다. [진단 패치](../../benchmarks/evidence/20261123-w3-c32-short-burst/prepare.py)

CPU의 실제 스케줄러 계획에서 기준과 각 문턱, decode 9·17행, 64토큰 미만 입력과 짧고 긴 입력이 섞인 경우를 확인했다. 같은 진단 바이너리에서 정책 끔·8행·16행을 두 A100에 교차 배치했다. 각 GPU의 보존 도착 시각표를 재생했고, 고정 타일과 활성 슬롯 32개, HTTP 워커 64개, prefix 필터, 격리 워밍업과 서버 인자를 맞췄다. [스케줄러 검사](../../benchmarks/evidence/20261123-w3-c32-short-burst/short-burst-test.rs) · [실행 순서](../../benchmarks/evidence/20261123-w3-c32-short-burst/campaign.sh)

| GPU | decode 문턱 | C32 TTFT p95, 기준 → 후보 | C32 TTFT p50 변화 | C32 TPOT p50 변화 | 512행 스텝 |
|---:|---:|---:|---:|---:|---:|
| 6 | 8행 | 353.92 → 319.29ms (−9.78%) | +9.45% | −0.54% | 2 |
| 7 | 8행 | 343.62 → 325.94ms (−5.15%) | +0.59% | +0.11% | 2 |
| 6 | 16행 | 353.92 → 346.91ms (−1.98%) | +70.97% | +0.24% | 3 |
| 7 | 16행 | 343.62 → 350.64ms (+2.04%) | +43.17% | +1.24% | 3 |

8행 후보는 두 GPU에서 C32 TTFT p95를 줄였고, RAG TTFT p95도 각각 1.08%, 0.95% 줄였다. 첫 32건의 p95는 GPU 6에서 407.03 → 373.00ms, GPU 7에서 396.49 → 378.44ms였다. 후속 64건의 p95는 각각 99.22 → 109.44ms와 71.38 → 72.60ms로 늘었다. GPU 7에서는 기존 비교 기준을 모두 충족했고, GPU 6에서는 C32 TTFT p50만 2% 회귀 한도를 넘었다. 16행 후보의 중앙값은 양쪽에서 회귀했다. 이 범위에서 8행·16행 후보의 기본 적용을 보류한다. [지연·출력·회귀 검산](../../benchmarks/evidence/20261123-w3-c32-short-burst/verified.json)

8행 후보의 후속 64건을 같은 요청끼리 짝지었다. GPU 6·7의 TTFT 차이 중앙값은 각각 +7.69ms·+1.26ms였다. 백엔드 제출부터 입장까지의 차이는 +10.05ms·+3.38ms였고, 입장 뒤 서버의 첫 텍스트까지는 −0.41ms·−0.25ms였다. 제출 구간의 측정값에는 토큰화, 엔진 큐 대기와 스케줄러 입장이 함께 들어간다. [요청별 구간 집계](../../benchmarks/evidence/20261123-w3-c32-short-burst/admission.json) · [원본 재검산](../../benchmarks/evidence/20261123-w3-c32-short-burst/check-admission.sh)

6개 실행의 응답 984건은 기존 이미지와 입력·생성 ID, 텍스트 및 종료 사유까지 일치했다. 기준 대비 두 후보의 클라이언트 write 오차 차이 p95는 최대 1.090ms였다. 정상 종료 뒤 서버 자원이 회수됐고 GPU 6·7은 기존 20GiB 예약으로 복귀했다. GPU 측정과 별도로 진단 빌드에서 짧고 긴 입력의 CPU 스케줄러 테스트를 통과했다. [CPU 테스트 결과](../../benchmarks/evidence/20261123-w3-c32-short-burst/scheduler-test.log.gz) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-short-burst/result.sha256) · [재검산](../../benchmarks/evidence/20261123-w3-c32-short-burst/fetch-summary.sh)

### 512행 버스트의 엔진 입장 구간

짧은 입력 전용 512행·decode 8행 후보의 제출 지연을 나눠 기록했다. 타임스탬프는 제출, 엔진 큐 전송 직전, 워커 입장 처리 시작, 입장 반환 직전에 저장한다. 엔진 실행은 스텝당 호스트 시작·종료 시각과 실제 prompt·decode 행 수를 기록한다. 실행과 대기가 겹친 부분은 decode 전용, 256행 이하 prefill 포함, 256행 초과 prefill 포함으로 집계한다. [계측 위치](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/prepare.py) · [시각 구간 판정](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/trace-phases.py)

같은 바이너리로 버스트 정책과 계측을 각각 켜고 끈 네 조합을 A100 6·7번에서 역순으로 실행했다. 기존 도착 시각표와 Coder-14B 체크포인트, 캐시 필터, 워밍업 방식을 사용했다. 클라이언트 응답 ID로 두 기록을 연결하고, 제출부터 입장까지의 내부 구간이 기존 서버 지표의 전체 제출 시간 안에 들어가는지 대조했다. [실행 순서와 재생 옵션](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/campaign.sh)

| GPU | 계측 | 8행 후보의 C32 TTFT p50 변화 | TTFT p95 변화 | TPOT p50 변화 |
|---:|---|---:|---:|---:|
| 6 | 끔 | −6.62% | −13.90% | −0.53% |
| 7 | 끔 | −22.09% | −7.73% | −0.06% |
| 6 | 켬 | +2.52% | −4.87% | +0.23% |
| 7 | 켬 | −13.70% | −13.08% | −0.99% |

계측을 끈 두 비교에서는 기록된 가드레일을 충족했다. 계측을 켠 비교에서는 GPU 6의 C32 TTFT p50이 2% 한도를 넘었고 GPU 7의 RAG TTFT p95는 0.59% 늘었다. 앞선 계측 없는 실험에서도 GPU 6의 C32 TTFT p50이 9.45% 늘었다. 기본 적용은 보류하고 반복 간 변동을 기록한다. 현재 실행에서 클라이언트 write 오차의 쌍별 차이 p95는 최대 1.791ms였다. [조합과 가드레일](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/verified.json)

| 후속 64건의 요청별 차이 중앙값, 계측 켬 | GPU 6 | GPU 7 |
|---|---:|---:|
| 제출부터 엔진 큐 전송 직전까지 | −0.014ms | +0.001ms |
| 큐 전송 직전부터 워커 입장 처리 시작까지 | +2.83ms | +5.82ms |
| 워커 입장 처리 시작부터 입장 반환 직전까지 | 절댓값 0.001ms 미만 | 절댓값 0.001ms 미만 |
| 큐 대기 중 256행 이하 prefill 포함 실행과 겹친 시간 | +2.78ms | 0.00ms |

계측 후보의 첫 32건 중 GPU 6에서 19건, GPU 7에서 23건의 큐 대기가 256행 초과 prefill 포함 실행과 겹쳤다. 후속 64건에는 그 겹침이 양쪽 모두 0건이었다. GPU 7에서는 후속 요청의 전송→처리 차이 평균이 −1.19ms였으므로 중앙값과 방향이 달랐다. 계측을 끈 실행의 기존 제출 지표에서 후속 64건의 요청별 차이 중앙값은 GPU 6에서 +19.91ms, GPU 7에서 −11.01ms였다. 큐 실행과 겹친 시간은 계측한 조건의 관측값으로 사용한다. [세부 구간과 계측 없이 읽은 지표](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/verified.json)

8개 실행에서 기존 이미지와 1,312개 응답이 입력·생성 ID, 텍스트 및 종료 사유까지 일치했다. 계측한 네 실행에서 워밍업 48건씩을 시각과 ID로 확인한 뒤 측정 요청 656건을 연결했고 누락된 레코드는 0건이었다. 각 서버의 종료 자원이 회수됐고 GPU 6·7은 기존 20GiB 예약으로 복귀했다. [실행 로그](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/campaign.log) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/result.sha256) · [재검산](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/fetch-summary.sh)

#### 후속 64건과 실행 중인 스텝의 경계

보존한 계측 네 건에서 큐 대기를 전송 시점에 실행 중이던 스텝의 남은 시간, 전송 뒤 시작한 스텝의 실행 시간, 어느 스텝에도 속하지 않는 시간으로 나눴다. 실행 경계는 엔진 워커에서 기록한 호스트 시각이다.

| GPU | 실행 중 전송, 기준 → 후보 | 전송 뒤 시작한 스텝과 겹친 요청 | 스텝 실행 밖 대기 p95 | 실행 중이던 스텝의 남은 시간 변화 중앙값 |
|---:|---:|---:|---:|---:|
| 6 | 64 → 61 | 0 → 1 | 0.156 → 0.158ms | +2.86ms |
| 7 | 64 → 64 | 0 → 0 | 0.193 → 0.126ms | +5.80ms |

양쪽 설정에서 전송 당시 스텝이 실행 중이던 쌍은 GPU 6번 61쌍, 7번 64쌍이다. 후보의 스텝 전체 길이 변화 중앙값은 각각 −0.05ms, −0.22ms였고, 스텝 시작부터 명령 전송까지의 변화 중앙값은 −2.85ms, −6.31ms였다. 이 실행에서는 후보의 명령이 스텝 초반에 도착하는 쪽으로 중앙값이 이동해 전송 후 남은 시간이 길었다. 행 수와 decode 행 수까지 일치한 쌍은 각각 14개와 4개였다. 다른 부하나 비계측 실행에도 같은 관계가 성립하는지는 확인이 필요하다. [수치와 원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/queue-boundaries.json) · [CPU 재검산](../../benchmarks/evidence/20261123-w3-c32-short-burst-admission/check-queue-boundaries.sh)

현재 엔진은 스텝을 시작하기 전에 명령 큐를 비운다. 계측 네 번의 후속 요청 관측 256건 중 전송 뒤 시작한 스텝과 겹친 건은 1건이었다. 다음 대조에서는 스텝 시작 시각과 후속 요청 도착 시각의 관계가 반복에서도 나타나는지 확인한다. [엔진 명령 처리 순서](../../crates/riley-server/src/engine.rs)

#### 후속 64건의 도착을 12ms 늦춘 교차 실행

같은 Coder-14B 바이너리와 256행 기준에서 기존 도착 시각표와 후속 요청 64건을 12ms 늦춘 시각표를 비교했다. 첫 32건의 예정 시각과 각 집단 내부의 요청 간격은 동일했다. 두 시각표 각각에서 512행·decode 8행 후보와 기준을 계측 켬·끔으로 시험했다. GPU 6번은 원래 시각표부터, 7번은 12ms 지연 시각표부터 실행했고 GPU별 네 설정의 순서도 서로 뒤집었다. 후속 요청의 실측 전송 이동 중앙값은 11.96–12.05ms였다. 두 집단 전체에서 설정한 이동량과 실측 이동량의 오차 p95는 최대 1.09ms였다. [시각표 변환](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/schedules.py) · [교차 실행](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/campaign.sh)

| GPU | 입장 계측 | 원래 C32 TTFT p50 / p95 변화 | 12ms 지연 C32 TTFT p50 / p95 변화 |
|---:|---|---:|---:|
| 6 | 끔 | +0.73% / −6.18% | −9.70% / −6.36% |
| 6 | 켬 | +12.76% / −9.08% | −3.49% / −7.55% |
| 7 | 끔 | +10.63% / +32.20% | +26.76% / −7.98% |
| 7 | 켬 | −5.05% / −7.35% | −27.97% / −12.86% |

GPU 6번의 계측에서 후속 64건의 요청별 큐 대기 차이 중앙값은 +9.38ms에서 −2.56ms로 바뀌었다. 전송 당시 실행 중이던 스텝의 남은 시간 차이는 +9.38ms에서 −2.61ms로 바뀌었다. GPU 7번의 큐 대기 차이는 −3.07ms에서 −4.38ms로 줄었다. 계측 8회의 후속 요청 관측 512건 중 전송 뒤 시작한 스텝과 겹친 건은 1건이었다. 계측을 끈 GPU 7번에서는 시각표에 따라 p95 변화의 방향이 바뀌었고, 두 시각표 모두 p50이 2% 회귀 한도를 넘었다. 순서를 교차한 이번 실행의 결과로 후보의 기본 적용을 보류한다. [요청별 구간과 판정](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/verified.json)

16회 실행의 2,624개 응답은 기준 이미지와 입력·생성 토큰 ID, 텍스트 및 종료 사유가 모두 일치했다. 종료 자원을 회수하고 GPU 6·7번에서 기존 20GiB 예약의 복귀를 확인했다. [실행 로그](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/campaign.log) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/variant-results.sha256) · [CPU 재검산](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/fetch-summary.sh)

후보의 계측 네 번을 다시 확인했다. 각 실행의 짧은 C32 구간에서 256행을 넘긴 스텝은 정확히 두 개였고, 원본 메트릭의 커밋 스텝 수와 일치했다. 첫 스텝은 512행 중 decode 0행, 두 번째 스텝은 512행 중 decode 7행이었다. 첫 스텝은 첫 32건 가운데 0–22건의 큐 대기와 겹쳤다. 두 번째 스텝과 큐 대기가 겹친 요청은 0건이었다. 두 번째 스텝은 첫 후속 요청의 큐 전송보다 1.60–1.61초 먼저 끝났다. 후속 요청의 지연을 줄이는 후보를 고를 때에는 후속 도착 당시 진행 중인 일반 스텝의 길이와 시작 시각을 검증한다. [스텝 경계·커밋 수 대조](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/verified.json)

#### C32 첫 토큰 p95를 결정한 요청

두 도착 시각표, 두 GPU, 계측 켬·끔의 여덟 비교에서 짧은 요청 96건씩을 입력·생성 ID, 출력과 종료 사유로 짝지었다. 보존한 응답과 서버 관측값으로 p50·p95를 재현했다. 여덟 비교 모두 기준과 후보의 p95를 결정한 요청은 처음 32건에 속했다. 후속 64건의 입장 구간 분석과 첫 집단의 p95 분석에는 서로 다른 요청이 사용된다. [원본 시각표 대조](../../benchmarks/evidence/20261123-w3-c32-arrival-phase/verified.json) · [요청별 재검산](../../benchmarks/evidence/20261123-w3-c32-ttft-phases/verified.json)

GPU 7번의 기존 도착 시각표·입장 계측 끔에서 전체 p95는 **343.42 → 453.98ms**였다. 기준의 7번 요청에서 후보의 28번 요청으로 p95 순위가 바뀌었다. 28번을 같은 요청끼리 비교하면 엔진 제출부터 입장 반환까지는 **34.54ms 감소**, 입장 반환부터 첫 서버 텍스트까지는 **22.90ms 감소**했다. 클라이언트 write 이후 이 두 구간을 제외한 시간은 **0.49 → 207.25ms**였다. 해당 요청의 서버 HTTP 연결 큐 기록은 **0.186 → 0.204ms**였다. 연결 큐를 포함해 기록된 구간만으로는 207.25ms의 위치를 더 좁힐 수 없다. [분위수·요청·구간 검산](../../benchmarks/evidence/20261123-w3-c32-ttft-phases/check_tail.py) · [결과 체크섬](../../benchmarks/evidence/20261123-w3-c32-ttft-phases/result.sha256)

512행 후보의 기본값 적용 보류를 유지한다. 같은 증상이 다시 나타나면 요청 본문 처리 후 엔진 제출 전까지와 첫 서버 텍스트 준비 뒤 클라이언트에 도착하기까지를 각각 계측해 지연 위치를 확인한다.

#### C32의 HTTP·SSE 전송 시각 대조

기존 Coder-14B 진단 소스 `3951d90c8`에 서버의 본문 파싱, 엔진 제출·입장, 헤더·첫 텍스트 쓰기 시각을 추가했다. 기준과 512행·decode 8행 후보에 원래 도착 시각표를 재생했다. 같은 진단 바이너리에서 서버 전송 계측을 각각 켜고 끄고 A100 6·7번의 순서를 교차했다. 여덟 실행 모두 클라이언트 수신 시각을 기록했고, 실행당 짧은 요청 96건씩 입력·생성 토큰 ID와 텍스트·종료 사유가 일치했다. 도착 오차 p95는 최대 2.12ms였다. 종료 뒤 각 GPU의 기존 20GiB 예약이 복귀했다. [실행 순서](../../benchmarks/evidence/20261123-w3-c32-transport/campaign.log) · [응답·시각 검산](../../benchmarks/evidence/20261123-w3-c32-transport/verified.json)

| GPU | 서버 전송 계측 | C32 TTFT p50 변화 | C32 TTFT p95 변화 |
|---:|---|---:|---:|
| 6 | 끔 | +37.45% | −11.21% |
| 6 | 켬 | +3.24% | −6.21% |
| 7 | 끔 | +10.27% | −6.47% |
| 7 | 켬 | −2.54% | −6.13% |

이전 207ms 미계측 지연은 이번에 재현되지 않았다. 여덟 실행의 768건에서 클라이언트 write 이후 엔진 제출·입장 후 첫 텍스트 구간을 제외한 시간은 최대 2.30ms였다. 서버 시각을 추가한 네 실행에서 p95를 결정한 요청의 첫 텍스트 준비부터 클라이언트 수신까지는 0.67–1.06ms였다. [클라이언트·서버 구간 결과](../../benchmarks/evidence/20261123-w3-c32-transport/request-boundaries.json) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-transport/result.sha256)

서버 전송 계측을 끈 GPU 6번에서 후보의 p50을 결정한 79번을 같은 요청끼리 비교하면 엔진 제출 구간은 16.69ms 줄었고 입장 후 첫 서버 텍스트까지는 38.34ms 늘었다. GPU 7번에서는 기준과 후보의 p50을 같은 75번이 결정했고 엔진 제출 구간이 6.68ms 늘었다. 두 비교 모두 p50의 2% 회귀 한도를 넘었다. 512행 후보의 기본값 적용 보류를 유지한다.

#### Decode가 없는 짧은 prefill의 512행 버스트

앞선 512행 후보에서 관측된 두 wide 스텝 중 decode가 없는 조건만 남겼다. CPU 계획 검사에서 같은 짧은 요청으로 decode 0행이면 512행, decode 1행이면 256행을 선택했다. 기존 도착 시각표, 모델, 하드웨어와 서버 설정을 유지했고 서버 전송 계측은 껐다. GPU 6번은 후보부터, 7번은 기준부터 실행했다. 도착 오차 p95는 두 비교에서 0.41ms와 0.59ms였다. 측정한 출력 328쌍은 입력·생성 토큰 ID, 텍스트 및 종료 사유가 일치했다. 종료 뒤 두 GPU의 기존 20GiB 예약을 확인했다. [변경한 조건](../../benchmarks/evidence/20261123-w3-c32-prefill-only/prepare-prefill.py) · [실행 순서](../../benchmarks/evidence/20261123-w3-c32-prefill-only/campaign.log)

| GPU | C32 TTFT p50 변화 | C32 TTFT p95 변화 | C32 TPOT p50 변화 | RAG TTFT p95 변화 |
|---:|---:|---:|---:|---:|
| 6 | +15.03% | +7.33% | +1.40% | +1.74% |
| 7 | −23.81% | −3.14% | −1.50% | −0.77% |

GPU 7번 비교는 기존 기준을 모두 충족했다. GPU 6번 비교는 C32 p50·p95·TPOT와 RAG p95 기준을 충족하지 못했다. 이번 비교에서 GPU와 실행 순서가 함께 달라 두 요인의 영향은 별도로 확인할 수 없다. Decode가 있는 스텝을 wide 대상에서 제외한 조건도 기본값 적용을 보류한다. [C32·RAG 비교와 판정](../../benchmarks/evidence/20261123-w3-c32-prefill-only/verified.json) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-prefill-only/result.sha256)

#### Decode 0행 후보의 역순 대조

앞선 비교와 동일한 CUDA 실행 파일, Coder-14B 체크포인트, 서버 인자와 도착 시각표를 두 GPU에서 재사용하고 각 GPU의 실행 순서를 뒤집었다. 두 회차의 기준·후보 응답 656쌍은 입력·생성 토큰 ID, 텍스트, 종료 사유와 토큰 수가 일치했다. 회차 사이에도 동일 조건의 출력이 일치했고 C32의 짝별 write 오차 p95는 모두 0.60ms 미만이었다. 종료 뒤 각 GPU의 기존 홀더와 20GiB 예약을 확인했다. [실행 기록](../../benchmarks/evidence/20261123-w3-c32-prefill-order/campaign.log) · [교차 검산](../../benchmarks/evidence/20261123-w3-c32-prefill-order/order-comparison.json) · [원본 체크섬](../../benchmarks/evidence/20261123-w3-c32-prefill-order/result.sha256)

| GPU | 회차와 시작 조건 | C32 TTFT p50 변화 | C32 TTFT p95 변화 | RAG TTFT p95 변화 |
|---:|---|---:|---:|---:|
| 6 | 최초, 후보부터 | +15.03% | +7.33% | +1.74% |
| 6 | 역순, 기준부터 | +13.70% | −2.41% | +2.36% |
| 7 | 최초, 기준부터 | −23.81% | −3.14% | −0.77% |
| 7 | 역순, 후보부터 | −17.97% | −3.18% | +1.08% |

실행 순서를 바꾼 뒤에도 C32 TTFT p50의 변화 방향은 GPU별로 같았다. GPU 6번은 두 순서에서 p50 회귀 기준을 넘었고, 역순의 GPU 7번은 RAG p95 개선 조건을 충족하지 못했다. 각 GPU의 순서별 비교는 한 번씩이며, 실행 시점에 따른 변동의 영향은 남아 있다. 기본값은 256행으로 유지한다.

C32 구간의 배치 카운터에서 후보의 512행 버킷은 실행당 한 번 커밋됐고 기록된 GPU 시간은 59.59–60.81ms였다. 기준의 최대 커밋 버킷은 256행 이하였다. 후속 64건을 쓸 때 아직 완료되지 않은 앞선 요청 수의 중앙값은 GPU 6번에서 최초 30→34건, 역순 30→32.5건이었다. GPU 7번에서는 최초 32→29건, 역순 29→29건이었다. 이 수치는 클라이언트가 관측한 미완료 요청이며 서버 큐 대기와 처리 중 요청의 기여는 따로 판정할 수 없다. [배치 카운터와 요청 집계](../../benchmarks/evidence/20261123-w3-c32-prefill-order/batch-shapes.json)

## 배포 이미지의 Coder-14B 교차 비교

2026-11-24 KST에 `main`의 CUDA 12.4 배포 이미지 `c5ec7b391`과 vLLM `0.27.1+cu129`를 AWS A100 40GiB 6·7번에서 비교했다. GPU 6번의 실행 순서는 Riley → vLLM, 7번은 vLLM → Riley였다. 같은 Qwen2.5-Coder-14B-Instruct 체크포인트(`aedcc2d4`)를 읽기 전용으로 사용했다. 두 엔진은 BF16, 최대 활성 32개, 문맥 8,192토큰 및 KV 2,576블록(41,216토큰, 8,103,395,328바이트)을 사용했다. vLLM은 시작 로그의 실제 KV 토큰 수까지 확인했다. [실행 조건과 고정 입력 지문](../../benchmarks/evidence/20261124-w3-package-coder14/campaign.sh) · [비교 설정](../../benchmarks/evidence/20261124-w3-package-coder14/cells.json)

Riley는 앞선 RAG 대조의 명시적 프로파일을 사용했다. 배치·prefill 상한은 1,024행이고 정상 상태 예산은 256행이다. vLLM은 배치 토큰 2,048을 명시했다. 두 엔진 모두 prefix cache를 끄고 측정했다. 체크포인트 파일 권한 때문에 Riley 컨테이너는 체크포인트 소유자 UID로 실행했다. 패키지와 실행 중인 바이너리의 해시를 대조했고, 컨테이너 포트는 loopback에 바인딩했다. 첫 시도는 검증된 체크포인트를 사용했으나 Python 3.10의 검사기 import에서 측정 전에 종료됐다. 아래 결과는 Riley와 vLLM 모두 기존 호환 Python 환경으로 다시 실행한 값이다. [패키지 및 GPU 실행기](../../benchmarks/evidence/20261124-w3-package-coder14/arm.py) · [원본 체크섬](../../benchmarks/evidence/20261124-w3-package-coder14/result.sha256)

| 측정 지표 | 요청/엔진/GPU | GPU 6 Riley / vLLM (ms) | GPU 7 Riley / vLLM (ms) | Riley/vLLM | 기존 목표 |
|---|---:|---:|---:|---:|---:|
| 짧은 C32 TPOT p50 | 192 | 26.66 / 25.47 | 26.56 / 25.47 | 1.0470× / 1.0428× | ≤1.03× |
| RAG C1 TTFT p50 | 16 | 473.76 / 449.22 | 464.76 / 441.08 | 1.0546× / 1.0537× | ≤1.15× / 최종 ≤1.05× |
| RAG C8 seed 1 TTFT p95 | 64 | 1,785.42 / 1,245.13 | 2,046.66 / 1,224.25 | 1.4339× / 1.6718× | ≤1.40× / 최종 ≤1.15× |
| RAG C8 seed 2 TTFT p95 | 64 | 1,743.96 / 1,097.98 | 1,615.85 / 1,088.73 | 1.5883× / 1.4842× | ≤1.40× / 최종 ≤1.15× |

동시성 32의 TPOT은 두 GPU에서 목표를 넘었고 Riley의 출력 처리량은 5.93–7.19% 높았다. RAG C1은 두 실행에서 1단계 기준을 충족했고 최종 기준보다 0.0037–0.0046배 높았다. RAG C8 첫 토큰 p95는 두 GPU와 두 seed에서 1단계 기준 1.40배를 넘었다. C8 출력 처리량은 vLLM 대비 2.58–3.02% 낮았다. C8의 격차가 두 GPU에서 재현됐으므로 두 seed의 첫 토큰 꼬리 지연을 다음 후보의 주 지표로 둔다. 짧은 C32 TPOT과 출력 처리량도 함께 검산한다. [독립 집계 및 판정](../../benchmarks/evidence/20261124-w3-package-coder14/verified.json)

네 실행은 실측 1,344건과 워밍업 208건을 오류와 429 없이 완료했다. 같은 GPU에서 짝지은 672건의 요청 본문, 입력·생성 토큰 수, 종료 사유가 일치했다. 출력을 두 GPU 사이에서 비교하면 Riley의 336건은 모두 텍스트 해시가 같았고, vLLM은 265건이 같았다. 엔진 간 같은 텍스트는 GPU별 128/336건과 129/336건이었다. 각 측정은 합성 닫힌 부하이며 GPU와 실행 순서가 함께 바뀐다. 이 결과의 범위는 지정한 두 프로파일과 Coder-14B다.

Riley는 종료 코드 0으로 두 번 종료했고 활성·대기 요청과 KV·native 할당을 모두 회수했다. 교대 전과 마지막 종료 뒤 두 GPU에서 기존 홀더의 20GiB 예약을 확인했다. 원본 요청 기록을 새 디렉터리에 푼 뒤 `python3 -B verify.py <디렉터리>`로 동일한 집계를 재현할 수 있다. [요청별 측정 보관본](../../benchmarks/evidence/20261124-w3-package-coder14/raw-results.tar.gz) · [독립 검산기](../../benchmarks/evidence/20261124-w3-package-coder14/verify.py)

## Coder-14B 요청별 prefill 조각 크기

2026-11-24 KST에 앞선 Coder-14B 배포 이미지 `c5ec7b391`, 모델 및 KV 2,576블록을 다시 사용했다. 스텝 전체의 1,024행과 정상 상태 256행을 유지하고 `--prefill-chunk-tokens`만 1,024에서 512로 줄였다. 후보는 하나의 요청이 한 스텝에 쓰는 prefill 행을 줄여 다른 요청이 같은 스텝에 들어갈 여지를 만든다. GPU 6번은 512 → 1,024, 7번은 1,024 → 512 순으로 실행했다. 짧은 C32, RAG C1·C8 두 seed에 이어 8개 동시·50ms 간격 유입을 비교했다. [실행 조건](../../benchmarks/evidence/20261124-w3-chunk-pair/campaign.sh) · [부하 설정](../../benchmarks/evidence/20261124-w3-chunk-pair/cells.json)

첫 시도의 두 서버는 여섯 부하를 마쳤고 별도 토큰 ID 검사의 클라이언트 인자 오류로 교차 단계 전에 종료됐다. 두 GPU가 기존 예약으로 돌아온 뒤 성능 부하 도구를 그대로 두고, ID 검사에만 호환 버전의 기존 도구를 연결해 네 실행을 다시 수행했다. 아래 표는 재실행에서 같은 GPU의 1,024행 대비 512행 변화율이다. [첫 시도와 실행 원본 체크섬](../../benchmarks/evidence/20261124-w3-chunk-pair/result.sha256)

| 부하 | 첫 토큰 p95 GPU 6 / 7 | 토큰당 생성 시간 p50 GPU 6 / 7 | 출력 처리량 GPU 6 / 7 |
|---|---:|---:|---:|
| 짧은 C32 | −12.82% / +2.87% | +0.10% / −0.18% | −0.28% / −0.18% |
| RAG C1 | +3.79% / +5.78% | +0.02% / −0.39% | −0.03% / −0.04% |
| RAG C8 seed 1 | +6.56% / +14.39% | +8.98% / +1.00% | −5.62% / −3.12% |
| RAG C8 seed 2 | +5.54% / +2.74% | +6.86% / +3.48% | −2.73% / −3.32% |
| 8개 동시 유입 | −0.89% / +4.52% | +1.76% / +2.22% | +0.25% / −0.20% |
| 8개 50ms 간격 유입 | −0.17% / +0.24% | −3.31% / −0.27% | −0.80% / −0.62% |

512행 후보는 네 RAG C8 비교에서 첫 토큰 p95가 **2.74–14.39% 증가**했고, 출력 처리량은 2.73–5.62% 감소했다. RAG C8의 네 비교 모두 커밋된 전체 GPU 실행 시간 합이 2.83–5.95% 늘었고 1,024행 버킷 사용도 실행당 19–36회 늘었다. 배치 카운터의 집계 구간에는 워밍업과 측정 요청이 포함된다. 512행 후보도 같은 스텝에 여러 요청의 행을 배치할 수 있어 이 버킷을 계속 사용했다. 측정된 GPU 시간은 prefill과 decode를 함께 포함한다. 짧은 C32의 TTFT p95는 GPU 7번에서 +2.87%였고, 두 GPU 모두 일부 부하의 decode TPOT가 1% 회귀 기준을 넘었다. 512행 후보는 미채택했다. [요청과 배치 카운터 검산](../../benchmarks/evidence/20261124-w3-chunk-pair/verified.json)

네 실행의 측정 1,664건과 워밍업 272건은 오류와 429 없이 완료됐다. GPU 안에서 비교한 832쌍의 요청 본문, 출력 텍스트, 토큰 수와 종료 사유가 같았다. 별도 입력 8개의 스트리밍 응답은 네 실행에서 입력·생성 토큰 ID까지 같았다. 각 실행은 코드 0으로 종료했고 활성·대기 요청과 KV·native 할당은 0이었다. 두 A100에서 기존 홀더의 20GiB 예약이 복원됐다. 측정 보관본을 새 디렉터리에 푼 뒤 `python3 -B verify.py <디렉터리>`로 요청과 배치 지표를 재현할 수 있다. [측정 보관본](../../benchmarks/evidence/20261124-w3-chunk-pair/raw-results.tar.gz) · [검산기](../../benchmarks/evidence/20261124-w3-chunk-pair/verify.py)

## Coder-14B RAG 도착 시각 대조

앞선 Coder-14B의 닫힌 부하 원본에서 측정 요청 64개의 소켓 쓰기 시각을 같은 GPU·seed·요청 ID끼리 비교했다. 각 실행의 첫 측정 요청 전송을 시각 0으로 맞췄다. 요청 본문, 실제 입력·생성 토큰 수와 종료 사유는 네 비교에서 일치했다. 각 실행은 측정 요청 전송 시점에 앞선 측정 요청이 미완료인 수의 중앙값이 7건이었다. 실제로 같은 요청을 쓰기까지 걸린 시간은 다음과 같이 달랐다. [원본 요청 기록과 체크포인트 지문](../../benchmarks/evidence/20261124-w3-package-coder14/raw-results.tar.gz) · [시각 대조 결과](../../benchmarks/evidence/20261124-w3-coder-arrivals/preflight.json)

| GPU | RAG seed | 마지막 측정 전송 Riley / vLLM | 같은 요청의 절대 전송 시각차 p95 |
|---:|---:|---:|---:|
| 6 | 1 | 44.32초 / 43.24초 | 1.254초 |
| 6 | 2 | 42.45초 / 41.65초 | 1.140초 |
| 7 | 1 | 43.95초 / 42.96초 | 1.225초 |
| 7 | 2 | 42.36초 / 41.41초 | 1.078초 |

닫힌 부하에서는 응답이 끝나면 다음 요청을 보낸다. 전송 시각 차이가 관측된 첫 토큰 지연 격차에 얼마나 영향을 주는지 확인하기 위해 두 엔진의 도착 시각표를 각각 두 엔진에 재생한다. Riley `0511b9280`, vLLM `0.27.1+cu129`, 입력과 KV 용량 41,216토큰을 고정한다. 원본에서 추출한 8개 시각표의 512개 요청 본문 해시를 사전 검사했다. 재생 뒤에는 예정 시각 대비 실제 쓰기 오차와 두 엔진 사이의 차이를 p95 기준 3ms 이내로 확인하고, 입출력 토큰 수와 종료 사유가 원본과 맞는 비교에서 목표를 판정한다. [재현 코드](../../benchmarks/evidence/20261124-w3-coder-arrivals/freeze-arrivals.py) · [실행 순서](../../benchmarks/evidence/20261124-w3-coder-arrivals/campaign.sh) · [판정 코드](../../benchmarks/evidence/20261124-w3-coder-arrivals/verify-replay.py)

## W2 이벤트 발행과 묶음 정책의 상호작용

W4·W5에 쓴 `f7c956d3c` CUDA 바이너리와 Qwen2.5-0.5B에서 이벤트 발행 `immediate-v1`·`after-submit-v1`과 wide prefill 허용 상한 32·2 decode 행을 교차한다. 현행 프로파일의 이벤트 발행은 `after-submit-v1`이며, 이전 W3 측정과 같은 burst 설정과 네 부하 `closed-c32`, `rag-c1`, `rag-c8`, `rag-r3`를 사용한다. 두 seed는 실행 순서를 뒤집는다. 각 조건은 별도 서버에서 측정한 뒤 8개 동일 입력의 C1·C8 생성 토큰 ID, 텍스트, 종료 사유와 usage도 확인한다.

A100 GPU 6에서 네 조합을 두 seed로 실행했다. 8번의 실행에서 총 2,688개 측정 요청을 오류·429 없이 완료했다. 각 seed에서 네 조합의 요청·도착 일정과 응답 텍스트·종료 사유·토큰 사용량이 일치했다. 각 조합의 별도 C1·C8 검사에서는 8개 프롬프트의 생성 토큰 ID까지 같았다. 각 서버 종료 뒤 활성·대기 요청과 KV 및 native 할당은 0이었고 GPU 6의 동일 홀더가 20GiB 예약으로 복귀했다. [전체 비교](../../benchmarks/evidence/20261119-w23-matrix/result/comparison.json) · [요청별 원본과 체크섬](../../benchmarks/evidence/20261119-w23-matrix/result-SHA256SUMS)

아래 수치는 같은 seed의 `after-submit-v1`/32행 대비 변화율이다. 각 셀은 seed 1 / seed 2 순서다.

| 검증 조합 | closed-c32 처리량 | closed-c32 TPOT p50 | rag-c8 TTFT p95 | rag-c8 ITL p99 |
|---|---:|---:|---:|---:|
| `immediate-v1` / 32행 | −6.4% / −2.0% | +7.2% / +1.4% | +0.4% / +10.3% | +2.1% / +1.1% |
| `after-submit-v1` / 2행 | −0.6% / −0.1% | +0.2% / −0.6% | +65.1% / +174.3% | −58.6% / −58.5% |
| `immediate-v1` / 2행 | −3.5% / −1.4% | +2.3% / +1.5% | +67.1% / +128.8% | −58.6% / −58.5% |

2행 설정은 두 이벤트 모드와 두 seed에서 `rag-c8` ITL 꼬리를 줄였으며 첫 토큰 p95를 늘렸다. `after-submit-v1`의 `closed-c32` TTFT p95는 2행으로 바꿨을 때 −31.5% / +1.0%로 seed별 방향이 달랐다. `immediate-v1`의 32행과 2행은 두 seed 모두 `closed-c32` TPOT p50이 1% 이상 늘었다. 이 범위에서는 `after-submit-v1`과 wide 기준을 채택 비교의 기준으로 유지한다. 이번 성능 결론은 Qwen2.5-0.5B와 고정 부하에 적용된다. [실행 순서](../../benchmarks/evidence/20261119-w23-matrix/campaign.sh) · [측정 도구](../../benchmarks/evidence/20261119-w23-matrix/tools/measure_matrix.py) · [결과 집계기](../../benchmarks/evidence/20261119-w23-matrix/tools/summarize_matrix.py)
