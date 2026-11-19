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

### 커널 사이 공백과 CUDA 호출 시간의 대조

같은 Nsight SQLite 원본에서 32행 decode graph 676개와 인접 구간 671개를 재현했다. CUDA 호출은 동일한 제출 스레드에서, DMA는 동일한 CUDA 컨텍스트에서 집계했다. 각 시간 구간을 이전 graph의 마지막 커널과 다음 graph의 첫 커널 사이로 잘라 합산했다. [재집계 결과와 원본 해시](../../benchmarks/evidence/20261118-w2-sm17-a100/nsys2/sync-gap-summary.json) · [검산기](../../benchmarks/evidence/20261118-w2-sm17-a100/tools/summarize_sync_gaps.py)

| 관측 항목 | 중앙값 | p95 |
|---|---:|---:|
| 커널 사이 전체 | 470.72µs | 516.69µs |
| 다음 `cudaGraphLaunch` 호출과 겹친 시간 | 264.91µs | 294.46µs |
| DMA와 CUDA 동기화 호출이 차지한 합집합 | 46.26µs | 53.28µs |
| DMA·동기화·다음 graph 호출을 뺀 구간 | 167.75µs | 196.56µs |
| `cudaEventSynchronize` 완료 뒤 다음 launch 전의 `cudaStreamSynchronize` | 2.53µs | 2.83µs |

671개 모두에서 event 동기화 뒤 stream 동기화가 관측됐다. 호출 순서로 각 stream의 의존성은 판정할 수 없다. 같은 모델의 별도 host trace에서 `SUBMIT` 구간 중앙값은 86.7µs였다. 이 구간은 제출 이후 이벤트 훅까지 포함하며 Nsight의 단일 launch 호출과 측정 범위가 다르다. 따라서 Nsight에서 본 공백 470.72µs를 개선 가능 시간으로 사용할 수 없다. 후속 실행 후보를 평가할 때는 제출 전 작업의 비용과 Nsight를 끈 동일 부하의 지연을 확인한다. [호스트 구간 원본](../../benchmarks/evidence/20261118-w2-sm17-a100/summary.json) · [제출 훅과 구간 경계](../../crates/riley-runtime/src/llama/executor/dispatch.rs)

현행 소스의 정상 다운로드 경로에서 계측을 켜면 다운로드에 전달한 stream에 종료 이벤트를 기록하고 기다린 뒤 같은 stream을 동기화한다. 계측을 끄면 마지막 stream 동기화가 다운로드의 완료를 기다리는 단계다. 기존 Nsight에서 마지막 호출의 중앙값 2.53µs는 별도 비계측 기준 실행의 TPOT 6.698ms에 견줘 약 0.04% 규모였다. 호출 제거는 실제 개선 여부와 비동기 오류 전달 계약을 새로 입증해야 하므로 GPU 시험 우선순위는 제출 경계의 작업 비용에 둔다. [다운로드·계측·동기화 경로](../../crates/riley-scheduler/src/execution.rs) · [Nsight 집계](../../benchmarks/evidence/20261118-w2-sm17-a100/nsys2/sync-gap-summary.json)

### GPU 제출 전 KV 메타데이터 검사

`be7bc4d9c`의 Rust 소스와 후보를 격리해서 macOS ARM64에서 release로 빌드했다. 실제 `PackedBatchHostV1::new`와 `Scheduler::plan_iteration`을 각각 호출했다. 세 후보마다 기준 → 후보 → 후보 → 기준 순서를 다섯 번 반복해 평균 호출 시간의 중앙값을 비교했다. 스케줄러는 32개 요청의 prefill을 먼저 완료하고 decode 계획 320개를 측정했다. 측정에 쓰인 각 KV 테이블의 물리 블록 번호는 엄격한 오름차순임을 확인했다. [원본 시료·바이너리 해시](../../benchmarks/evidence/20261119-w2-submit-prepare/local-comparison.json)

| 후보 | 측정 결과 | 후속 판정 |
|---|---|---|
| 순서가 확인된 행 주소에서 중복 쌍 검사 생략 | decode 32행의 생성자에서 0.14–0.32µs 단축. 역순 1,024행은 +0.01µs | GPU 제출 지연 후보 우선순위에서 제외 |
| 읽기 전용 블록을 뺀 쓰기 집합 용량 예약 | 129·1,025·4,097토큰으로 시작한 32행 decode의 계획 시간이 각각 0.63·2.47·8.31µs 증가 | 미채택 |
| 물리 블록 번호가 엄격히 증가하는 테이블의 해시 중복 검사 생략 | 같은 입력 길이에서 각각 12.72→7.31, 32.42→15.65, 100.46→47.03µs. 모든 길이에서 10쌍 모두 단축 | 실제 서빙 비교 대상으로 선택 |

세 번째 후보는 순서가 다른 테이블에도 기존 방식으로 중복을 검사한다. 정렬·비정렬 테이블 입력 순서를 보존했고 중복 번호와 비정규 `valid_tokens`는 계속 거부했다. 스케줄러 단위 테스트 36건도 통과했다. 이 수치는 재사용이나 prefix 공유가 많은 서빙 환경의 경로 선택 비율, GPU 유휴 구간, 처리량을 입증하지 않는다. A100 비교에서 짧은 `reason-c32`와 긴 decode 부하의 출력·지연 및 GPU 유휴를 확인한 뒤 채택을 판단한다. [후보 패치](../../benchmarks/evidence/20261119-w2-submit-prepare/ordered-physical-tables.patch) · [스케줄러 측정 코드](../../benchmarks/evidence/20261119-w2-submit-prepare/scheduler_probe.rs) · [행 주소 측정 코드](../../benchmarks/evidence/20261119-w2-submit-prepare/packed_metadata_probe.rs)

### 정렬된 KV 테이블의 A100 서빙 결과

2026-11-19 KST에 `aws-a100` 물리 GPU 7에서 승인본 `d3d8fe6e5`와 정렬된 KV 테이블 후보를 CUDA 12.4로 각각 빌드했다. SmolLM2-1.7B, KV 9,006블록, `native-bf16-varlen-v1`과 `after-submit-v1` 설정에서 기준 → 후보 → 후보 → 기준 순으로 실행했다. 짧은 부하의 프롬프트는 평균 71.84토큰, 긴 부하는 평균 1,046.39토큰이었다. 각 실행·부하에서 워밍업 32건과 측정 64건이 완료됐고, 429와 요청 오류는 0건이었다. 응답 256쌍의 텍스트 해시·종료 사유·토큰 사용량이 일치했다. 별도 입력 8개의 C1·C8 토큰 ID도 네 실행에서 모두 일치했다. [실행 등록과 입력 해시](../../benchmarks/evidence/20261119-w2-submit-prepare/gpu-follow-up/registration.json) · [실행 코드](../../benchmarks/evidence/20261119-w2-submit-prepare/gpu-follow-up/campaign.sh) · [검산 결과](../../benchmarks/evidence/20261119-w2-submit-prepare/gpu-follow-up/verified-comparison.json)

| 부하 | 기준과 후보의 비교 순서 | TTFT p95 | TPOT p50 | ITL p99 | 출력 처리량 |
|---|---|---:|---:|---:|---:|
| 짧은 `reason-c32` | 기준 → 후보 | +0.225% | +0.271% | +0.393% | −0.204% |
| 짧은 `reason-c32` | 후보 → 기준 | +0.156% | +0.296% | +1.274% | −0.356% |
| 긴 `long-c32` | 기준 → 후보 | +0.186% | +0.106% | −0.124% | −0.161% |
| 긴 `long-c32` | 후보 → 기준 | +0.140% | +0.158% | −0.113% | −0.107% |

변화율은 각각의 기준 대비 후보로 계산했다. 짧은 요청의 TTFT p95·처리량과 두 부하의 TPOT p50은 사전 회귀 한도를 충족했다. 네 비교 모두에서 처리량이 0.107–0.356% 낮고 TPOT p50이 0.106–0.296% 높아, 반복되는 서빙 개선이 확인되지 않은 이 후보는 미채택한다.

별도 Nsight 계측 쌍에서 32행 decode 그래프 676개씩을 확인했다. 연속 쌍 671개의 커널 간격 중앙값은 468.416→454.752µs, p95는 505.696→496.640µs였다. 호스트의 `plan` 단계 중앙값은 38.55→32.75µs였다. 후보 계측에서는 측정 32건과 리포트가 저장된 뒤 종료 중인 자체 서버 PID를 실행기가 외부 GPU 프로세스로 분류했다. 전체 실행의 종료 코드 1과 후보의 종료 영수증 누락을 그대로 기록했다. 저장된 리포트를 CPU에서 재내보내고 서버 PID와 요청 32건, NVTX 시계, 그래프 개수를 대조해 측정 구간을 복원했다. 각 실행이 끝난 뒤 GPU 7에서 같은 홀더가 20GiB로 복귀한 것을 확인했다. [원본 해시와 복원 구간](../../benchmarks/evidence/20261119-w2-submit-prepare/gpu-follow-up/verified-comparison.json) · [Nsight 복원 코드](../../benchmarks/evidence/20261119-w2-submit-prepare/gpu-follow-up/recover-nsight.sh) · [복원 결과 검산기](../../benchmarks/evidence/20261119-w2-submit-prepare/gpu-follow-up/verify_recovered_w2.py)

## 최초 이벤트 발행 실험의 설정 정정

`riley-lp16`의 `after-submit-v1`은 다음 GPU 제출 직후 이전에 commit된 이벤트를 발행한다. `native-bf16-varlen-v1` 프로파일이 이 값을 설정하므로, 최초 비교에서 플래그를 생략한 두 실행도 `after-submit-v1`이었다. [생략한 실행의 인자](../../benchmarks/evidence/20261118-w2-sm17-a100/paired-off1/receipt.json) · [명시한 실행의 인자](../../benchmarks/evidence/20261118-w2-sm17-a100/event1/receipt.json) · [프로파일 구현](../../crates/riley-server/src/inference_profile.rs)

각 실행에서 워밍업 32건과 측정 64건이 완료됐고, 429와 요청 오류는 0건이었다. 아래 수치는 같은 설정에서 관측한 실행 간 차이다. 이 비교에 근거한 채택 판단은 철회한다.

| 설정 방법 | 처리량 (tok/s), 2회 평균 | TPOT p50 (ms), 2회 평균 | TTFT p95 (ms), 2회 평균 |
|---|---:|---:|---:|
| 프로파일에서 `after-submit-v1` 적용 | 4,182.36 | 6.7094 | 15.470 |
| 플래그로 `after-submit-v1` 명시 | 4,113.24 | 6.7103 | 15.407 |
| 변화율 | −1.65% | +0.013% | −0.41% |

이벤트 순서·종료·취소 CPU 테스트 3건과 scheduler의 in-flight 라우팅·시뮬레이션 37건은 통과했다. 위 Nsight 기준선은 프로파일 기본 동작의 관측치로 유지한다.

## 두 모드를 명시한 재비교

2026-11-19에 같은 A100, 바이너리, 모델, KV 예산에서 `--event-publication immediate-v1`과 `--event-publication after-submit-v1`을 명시했다. seed 1은 즉시 발행부터, seed 2는 제출 후 발행부터 측정했다. 네 번 모두 워밍업 32건과 측정 64건을 완료했고 429·요청 오류는 0건이었다. 각 seed의 요청 본문 해시가 같았으며, 완료된 64건 모두 두 모드의 응답 텍스트 바이트 해시·종료 사유·토큰 사용량이 같았다.

| seed | 즉시 / 제출 후 처리량 (tok/s) | 처리량 변화 | TTFT p95 변화 | TTFT p99 변화 | TPOT p50 변화 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,153.17 / 4,178.27 | +0.60% | +0.68% | +2.48% | −0.97% |
| 2 | 4,264.29 / 4,313.32 | +1.15% | −2.12% | −3.01% | −0.98% |

이 부하에서 제출 후 발행은 소폭 처리량이 늘었고 첫 토큰 꼬리 지연은 seed별 방향이 달랐다. 명시적 두 모드를 비교한 실험은 한 모델·한 부하·정책당 두 번이며, 유휴 커널 구간 변화는 따로 계측하지 않았다. 현행 기본값을 유지한다. 여러 in-flight 스텝이나 KV 소유권 변경은 독립 작업의 비중과 안전성을 추가로 검증한 뒤 판단한다. [요청별 근거 및 재집계](../../benchmarks/evidence/20261119-w2-explicit-events/summary.json)

W4·W5와 같은 CUDA 바이너리에서 두 이벤트 발행 모드와 W3의 두 burst 허용 상한을 교차했다. 두 seed에서 출력 일치와 종료 자원 회수를 확인했다. 현재 기준 대비 `immediate-v1`은 두 행 상한과 두 seed에서 모두 `closed-c32` TPOT p50이 1% 넘게 늘었다. [네 조합 결과와 판정](03-long-context-result.md#w2-이벤트-발행과-묶음-정책의-상호작용)

## GPU 예약 방식

시작 전에 GPU 6·7에서 활성 compute process가 없고 각 메모리 사용이 4MiB인 것을 확인했다. 기존 `gpu_hold` 도구로 각 장치에 20GiB 대기 예약을 시작했다. 7번 실험마다 예약을 64MiB로 낮췄다가 자식 프로세스 종료 후 20GiB로 되돌렸다. 동일 홀더의 PID와 시작 시각을 확인하며 전환했다. [기존 예약 도구](../../benchmarks/tools/gpu_hold/README.md)

호스트 예약은 메모리 할당과 사용 상태 표시를 제공한다. 측정 도구는 대상 GPU에서 예상 밖의 compute process를 발견하면 해당 실행을 중단한다.

최종 확인에서 GPU 6·7 모두 각 20GiB 대기 예약으로 유지 중이었다. GPU context를 합친 NVML 사용량은 각각 20,903MiB, 연산 사용률은 0%였다. 두 장 모두 별도 compute process가 없었다.

- [실행별 측정 결과](../../benchmarks/evidence/20261118-w2-sm17-a100/summary.json)
- [Nsight 구간 집계](../../benchmarks/evidence/20261118-w2-sm17-a100/profile-summary.json)
- [최종 GPU 예약 상태와 원본 trace 해시](../../benchmarks/evidence/20261118-w2-sm17-a100/gpu-reservation.json)
