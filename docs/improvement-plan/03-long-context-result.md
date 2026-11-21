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
