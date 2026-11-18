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

## W2 이벤트 발행과 묶음 정책의 상호작용

W4·W5에 쓴 `f7c956d3c` CUDA 바이너리와 Qwen2.5-0.5B에서 이벤트 발행 `immediate-v1`·`after-submit-v1`과 wide prefill 허용 상한 32·2 decode 행을 교차한다. 현행 프로파일의 이벤트 발행은 `after-submit-v1`이며, 이전 W3 측정과 같은 burst 설정과 네 부하 `closed-c32`, `rag-c1`, `rag-c8`, `rag-r3`를 사용한다. 두 seed는 실행 순서를 뒤집는다. 각 조건은 별도 서버에서 측정한 뒤 8개 동일 입력의 C1·C8 생성 토큰 ID, 텍스트, 종료 사유와 usage도 확인한다.

A100 GPU 6에서 실행하도록 등록했다. 성능 측정 중 다른 CUDA 작업의 간섭을 줄이기 위해 진행 중인 14B 측정·레이어 추적·GSM8K 비교가 종료된 뒤 시작한다. [실행 순서](../../benchmarks/evidence/20261119-w23-matrix/campaign.sh) · [측정 도구](../../benchmarks/evidence/20261119-w23-matrix/tools/measure_matrix.py)
