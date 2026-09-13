# Optimization batch serving 비교 보고

보고 시점: 의미 있는 영역의 구현/도입 묶음이 실제 serving에서 검증 가능해졌을 때. 작은 수정마다 보고하지 않는다. 이전 보고 이후 변경 영역과 이번 측정으로 판단할 가설을 먼저 적는다.

## 조건과 증거

- Batch/영역, 직전 Riley revision·binary hash, 새 Riley revision·binary hash, vLLM version/commit
- 동일 모델 revision·tokenizer·dtype·quantization 및 각 engine numerical profile
- GPU 모델·개수·driver/CUDA·CPU·메모리·전력 설정, 다른 GPU 작업 상태
- Workload/dataset hash, prompt/output 길이 분포·stop 조건, concurrency·arrival rate·request 수
- Warmup 제외 범위, 반복 횟수·AB/BA 순서, cache 정책, HTTP/streaming 측정 방식
- Raw 결과 경로 및 correctness 판정. 미측정·실패·장비 부재 skip을 구분

## 동일 workload별 결과

Concurrency와 workload가 다른 행을 하나의 평균으로 합쳐 결론내리지 않는다. 다음 표를 조건별로 작성한다.

| 지표 | 직전 Riley | 새 Riley | vLLM | 새/직전 변화 | 새/vLLM 변화 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output throughput (tokens/s) | 미측정 | 미측정 | 미측정 | — | — |
| Request throughput (requests/s) | 미측정 | 미측정 | 미측정 | — | — |
| TTFT P50 / P95 / P99 (ms) | 미측정 | 미측정 | 미측정 | — | — |
| TPOT P50 / P95 / P99 (ms) | 미측정 | 미측정 | 미측정 | — | — |
| E2E P95 / P99 (ms) | 미측정 | 미측정 | 미측정 | — | — |
| 오류·timeout 수 / 전체 요청 | 미측정 | 미측정 | 미측정 | — | — |
| Correctness / quality gate | 미검증 | 미검증 | 미검증 | — | — |
| Peak GPU memory (GiB) | 미측정 | 미측정 | 미측정 | — | — |

변화율은 (새 값 / 비교 값 − 1) × 100%; throughput은 양수, latency는 음수가 개선이다. 반복별 결과와 분산/신뢰구간을 함께 보존하고, latency percentile은 정의에 맞는 요청별 원자료에서 산출한다. 고 concurrency 안정성은 짧은 screen 결과만으로 통과시키지 않는다.

## 판정과 다음 작업

확인된 개선·회귀, 수치 계약 여부, 최종 목표와의 차이, 승격/보류/철회, 다음 optimization batch를 기록한다. 개선이 없으면 원인을 분석하고 다음 방향을 결정한다. 논문 배수나 microbenchmark를 serving 결과로 표기하지 않는다.
