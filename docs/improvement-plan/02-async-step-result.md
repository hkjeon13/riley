# W2 A100 기준선과 이벤트 발행 비교

2026-11-18 KST에 `aws-a100`의 물리 GPU 7에서 SmolLM2-1.7B-Instruct `reason-c32`를 재측정했다. GPU 6은 대기 예약을 유지했다. [W2 계획](02-async-step.md) · [사전 조사](02-async-step-preflight.md)

## 장비와 측정 조건

기존 결과를 재현하기 위해 0.1.0 `riley-lp16` 바이너리와 기존 모델을 사용했다. 바이너리 SHA-256은 `311c24eb5d9327c1a564c0435661a630104b9dfd2bd5c7128fad04d7ff94a763`이다. W1 변경을 포함한 새 바이너리의 검증은 별도다. 물리 GPU 7은 A100-SXM4-40GB이며 `native-bf16-varlen-v1`, KV 9,006블록, concurrency 32, 30단어 입력, 출력 상한 1,024토큰과 seed 1을 사용했다.

정상 부하와 host trace 실행은 워밍업 32건과 측정 64건으로 맞췄다. Nsight 실행은 과거 trace와 같은 별도 32건에 워밍업 0건을 사용했다. W1의 갱신된 `load.py`로 SSE 종료, usage, 측정·워밍업 요청 수, 429와 오류 여부를 검증했다. 이 비교는 Riley의 실행 조건을 분석했으며 vLLM은 다시 실행하지 않았다.

## 기준선과 trace

| 실행 | 측정 성공 | 처리량 (tok/s) | TPOT p50 (ms) |
|---|---:|---:|---:|
| 계측 끔 1 | 64/64 | 4,182.64 | 6.700 |
| host trace | 64/64 | 4,190.58 | 6.693 |
| Nsight 진단 | 32/32 | 3,815.37 | 6.657 |
| 계측 끔 2 | 64/64 | 4,125.71 | 6.698 |

Nsight SQLite에서 embedding gridX=256과 argmax gridX=32가 일치하는 graph launch 676개를 찾았다. 이들 중 인접한 671쌍의 커널 사이 공백 중앙값은 470.7µs였고, 과거 trace에 같은 기준을 적용하면 487.6µs였다. 스텝 커널 구간은 중앙값 6.262ms, 스텝 내부의 커널 공백 합은 5.7µs였다. 커널 구간 집계에서 DMA는 제외했다.

host trace의 연속 부하에 가까운 128스텝 구간 18개에서 submit과 wait을 제외한 호스트 구간 합은 중앙값 142µs였다. plan 47.6µs, pack 21.4µs, download 23.6µs, sample 25.1µs, commit 10.1µs, publish 6.6µs가 각각의 중앙값이다. 이 값에는 다음 입력이나 commit 순서에 의존하는 작업이 포함된다. 471µs의 커널 공백을 전부 숨길 수 있다고 가정하면 개선 폭을 과대평가한다.

첫 Nsight 실행은 GPU에 나타난 별도 프로세스 그룹의 PID를 확인할 수 없어 중단했다. 중단 시 GPU 7에는 홀더만 남았다. 재시도는 UID, 시작 시각, 실행 파일, 포트를 포함한 인자 전체가 일치하는 프로파일 대상을 확인했고, 로컬 임시 디렉터리에서 수집해 SQLite로 내보냈다. 첫 실행의 실패 로그와 receipt는 보관했다.

## GPU 제출 후 이벤트 발행 비교

`riley-lp16`의 기존 `--event-publication after-submit-v1` 옵션은 다음 GPU 제출 직후 이전에 commit된 이벤트를 발행한다. 계측 없이 즉시 발행 → 제출 후 발행 → 제출 후 발행 → 즉시 발행 순서로 실행했다. 각 실행의 워밍업 32건과 측정 64건이 모두 완료됐고, 429와 요청 오류는 0건이었다.

| 모드 | 처리량 (tok/s), 2회 평균 | TPOT p50 (ms), 2회 평균 | TTFT p95 (ms), 2회 평균 |
|---|---:|---:|---:|
| `immediate-v1` | 4,182.36 | 6.7094 | 15.470 |
| `after-submit-v1` | 4,113.24 | 6.7103 | 15.407 |
| 변화율 | −1.65% | +0.013% | −0.41% |

이 부하에서 제출 후 발행은 채택하지 않는다. 두 번의 제출 후 발행에서 모두 즉시 발행보다 처리량이 낮았다. TPOT 차이는 약 0.001ms였다. 기존 outbox의 이벤트 순서·종료·취소 CPU 테스트 3건과 scheduler의 in-flight 라우팅·시뮬레이션 37건은 통과했다. GPU 비교는 HTTP 요청 성공과 지연·처리량을 검증했다. 요청별 출력 바이트 비교나 새 실행 구조의 CUDA 오류 검증은 수행하지 않았다.

W2의 기준선 재현과 기존 옵션 검증을 완료했다. 이번 단일 모델의 결과를 근거로 scheduler의 다중 in-flight 계획이나 KV 소유권 변경을 시작하지 않는다. 추후 독립적인 호스트 작업이 지배적인 부하가 확인되면 그 부하에서 다시 판단한다.

## GPU 예약 방식

시작 전에 GPU 6·7에서 활성 compute process가 없고 각 메모리 사용이 4MiB인 것을 확인했다. 기존 `gpu_hold` 도구로 각 장치에 20GiB 대기 예약을 시작했다. 7번 실험마다 예약을 64MiB로 낮췄다가 자식 프로세스 종료 후 20GiB로 되돌렸다. 동일 홀더의 PID와 시작 시각을 확인하며 전환했다. [기존 예약 도구](../../benchmarks/tools/gpu_hold/README.md)

호스트 예약은 메모리 할당과 사용 상태 표시를 제공한다. 측정 도구는 대상 GPU에서 예상 밖의 compute process를 발견하면 해당 실행을 중단한다.

최종 확인에서 GPU 6·7 모두 각 20GiB 대기 예약으로 유지 중이었다. GPU context를 합친 NVML 사용량은 각각 20,903MiB, 연산 사용률은 0%였다. 두 장 모두 별도 compute process가 없었다.

- [실행별 측정 결과](../../benchmarks/evidence/20261118-w2-sm17-a100/summary.json)
- [Nsight 구간 집계](../../benchmarks/evidence/20261118-w2-sm17-a100/profile-summary.json)
- [최종 GPU 예약 상태와 원본 trace 해시](../../benchmarks/evidence/20261118-w2-sm17-a100/gpu-reservation.json)
