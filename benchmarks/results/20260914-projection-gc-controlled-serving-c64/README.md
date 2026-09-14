# Projection serving — 공통 GC 통제 및 tmpfs 기록

서버 바이너리를 고정하고 Python 벤치마크 클라이언트의 timed-phase GC를 비활성화했다. 모든 엔진의 원본 응답·로그는 tmpfs에 기록하고 GPU 측정 완료 후 디스크로 복사했다. Rust serving 경로와 수치 계약은 변경하지 않았다. 이전 GC-enabled/디스크 기록 결과와 통합하거나 latency를 보정하지 않는다.

## 비교 조건

RTX4090, SmolLM2-135M BF16, context1024, client C64/active32, batch/chunk512, prefix caching 및 각 엔진720MiB KV payload다. Riley는2048 physical KV pages와512 unique prefix-page 한도, rolling decode/prefill FFN/adaptive projection을 사용한다. GQA staging/query reuse는 비활성이고 projection 후보만 새 backend를 활성화한다. 모델 경로·실행 조건은 launch 기록에, workload source·frozen binary 해시는 preparation에 보존했다.

Shared는32개 변형을 재사용하고 unique는256 warmup+8192 retained의 초기16-token page가 모두 다르다. 각 lane256 warmup+8192 retained 요청, prior/control/projection/vLLM 및 역순의 전체16 lane이다. 각 통계는 두 실행의 run-level estimate 중앙값이며 pooled percentile이 아니다. HTTP SSE token arrival을 그대로 사용하며 token 시각을 보간하지 않는다.

| workload | engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 9752.330 | 118.616 | 2.826 | 228.531 | 238.202 | 11.437 | 13.563 |
| shared | control | 10137.931 | 115.011 | 2.753 | 208.244 | 212.790 | 10.911 | 11.832 |
| shared | projection | 10286.597 | 112.686 | 2.708 | 210.423 | 218.699 | 10.708 | 12.006 |
| shared | vllm | 13097.803 | 84.369 | 2.205 | 190.413 | 223.331 | 5.363 | 8.362 |
| unique | prior | 3465.385 | 363.876 | 7.244 | 628.142 | 649.362 | 10.536 | 11.970 |
| unique | control | 3479.950 | 362.776 | 7.224 | 624.005 | 645.954 | 10.345 | 11.786 |
| unique | projection | 3780.188 | 333.643 | 6.642 | 574.482 | 599.056 | 9.791 | 11.167 |
| unique | vllm | 4568.827 | 270.077 | 5.405 | 513.372 | 567.431 | 7.241 | 13.062 |

## 변화율과 판정

- shared: 후보 throughput은 frozen prior 대비+5.48%, 같은 binary control 대비+1.47%, vLLM 대비-21.46%.
  frozen prior 대비 보고된 latency 증가 항목: 없음.
- unique: 후보 throughput은 frozen prior 대비+9.08%, 같은 binary control 대비+8.63%, vLLM 대비-17.26%.
  frozen prior 대비 보고된 latency 증가 항목: 없음.

위 변화율은 이 고정 workload에서의 기술 통계이며 통계적 유의성이나 전체 목표 달성 판정이 아니다. Throughput만 개선되어도 TPOT·tail·안정성 조건을 만족하지 않으면 승격하지 않는다. 기본 backend 변경과 서버 성능에 대한 GC 이득 합산은 하지 않는다.

## 각 실행과 host pressure

| 실행 | tok/s | E2E P99 ms | phase 구간 s | host CPU some % | host IO some % |
|---|---:|---:|---:|---:|---:|
| shared-p0-prior | 9545.20 | 259.90 | 27.84 | 2.17 | 7.39 |
| shared-p0-control | 10146.34 | 208.77 | 26.23 | 1.46 | 4.44 |
| shared-p0-projection | 10327.91 | 217.92 | 25.78 | 1.50 | 7.62 |
| shared-p0-vllm | 13907.47 | 195.92 | 19.23 | 1.22 | 4.22 |
| shared-p1-vllm | 12288.13 | 250.74 | 21.72 | 1.11 | 12.34 |
| shared-p1-projection | 10245.29 | 219.47 | 26.00 | 3.89 | 4.62 |
| shared-p1-control | 10129.52 | 216.81 | 26.29 | 1.36 | 3.24 |
| shared-p1-prior | 9959.46 | 216.50 | 26.74 | 1.32 | 2.77 |
| unique-p0-prior | 3459.84 | 653.80 | 76.18 | 1.32 | 3.76 |
| unique-p0-control | 3474.80 | 646.65 | 75.86 | 1.27 | 2.78 |
| unique-p0-projection | 3790.61 | 592.05 | 69.58 | 1.00 | 3.28 |
| unique-p0-vllm | 4537.03 | 556.81 | 58.21 | 1.06 | 5.87 |
| unique-p1-vllm | 4600.62 | 578.05 | 57.43 | 1.15 | 7.16 |
| unique-p1-projection | 3769.76 | 606.06 | 69.96 | 1.35 | 7.33 |
| unique-p1-control | 3485.10 | 645.25 | 75.64 | 1.16 | 8.90 |
| unique-p1-prior | 3470.93 | 644.92 | 75.95 | 1.18 | 4.40 |

PSI는 host 전체의 누적 stall counter 차이며 engine별 원인이나 GPU idle의 인과를 증명하지 않는다. Tmpfs는 이번 측정기가 만드는 로그/응답 디스크 쓰기를 측정 밖으로 옮긴 것이며 외부 IO·CPU stall을 제거하지 않는다. Phase bounds는 클라이언트 bookkeeping을 포함할 수 있고 throughput은 실제 첫 request 시작부터 마지막 request 종료로 계산한다.

## 검증과 재현

131072 retained 요청/4194304 output tokens와4096 warmup 응답을 검증했다. Riley retained98304개는 frozen prior와 prompt/output token/text/finish가 일치한다. 저장된 checks flag만 신뢰하지 않고 실제 SSE frames를 재구성하여 fixture·usage·arrival 개수/순서·phase bounds와 대조했다. `[DONE]`은 frames에 없으므로 transport completion은 hash-bound client 검사에 의존한다. vLLM reference agreement는 별도 비교값이며 Riley와 동일 출력이라고 주장하지 않는다.

Stop/cancel/recovery는3개 Riley lane에서 각192건을 검증했다. 각 측정 server exit0, rolling step/drain 및 선택 backend 로그를 확인했다. 모든 warmup/retained phase에서 GC 비활성, GC event/counter 증가0, 종료 후 GC 활성 복구 및 client peak RSS<6GiB guard를 확인했다. 누적 high-water RSS로 장기 누수 부재를 주장하지 않는다.

Tmpfs와 디스크 복사본의 모든 파일 byte size/SHA256 일치 기록은 common archive의 materialization.json에 있다. Lifecycle 실행 및 Blender3개 복구 receipt도 보존했다. 웹 뷰어3개는 계속 유지했다. Evidence는17개 무손실 archive로 나누고 각 archive hash/size를 manifest에 기록한다. Client/controller snapshot과 preparation hash를 고정하고 실행 helper와 로컬 helper의 byte 동일성을 확인한다. 원본 profiler/환경변수 전체 dump는 포함하지 않는다.

## 다음 범위

이 결과는 단일 모델·C64·두 closed-loop 역순 반복의 제한된 비교다. 긴 open-loop/soak, 여러 모델 및 multi-GPU/Hopper/Blackwell runtime qualification은 미완료다. C8/C32/C64 비교를 완료했다. C64 shared는 동일 binary control 대비 E2E P95/P99·ITL P99가 증가했으므로 throughput 개선만으로 승격하지 않는다. [전체 concurrency 표](../20260914-projection-serving-matrix/README.md)를 기준으로 고정 candidate/control의 bounded C32 Nsight profile에서 prefill·decode·graph 대기 비중을 확인한다. 이 과정에서도 vLLM에 미달하는 영역은 남은 서버 병목으로 분리하며 이전에 회귀한 attention 미세 변형을 반복하는 근거로 사용하지 않는다.
