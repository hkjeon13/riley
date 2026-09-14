# Adaptive FFN serving — 공통 GC 통제 및 tmpfs 기록

서버 바이너리를 고정하고 Python 벤치마크 클라이언트의 timed-phase GC를 비활성화했다. 모든 엔진의 원본 응답·로그는 tmpfs에 기록하고 GPU 측정 완료 후 디스크로 복사했다. Python은 외부 측정 클라이언트에만 사용하며 기존 수치 계약을 유지한다. 이전 GC-enabled/디스크 기록 결과와 통합하거나 latency를 보정하지 않는다.

## 비교 조건

RTX4090, SmolLM2-135M BF16, context1024, client C32/active32, batch/chunk512, prefix caching 및 각 엔진720MiB KV payload다. Riley는2048 physical KV pages와512 unique prefix-page 한도, rolling decode/prefill FFN/adaptive projection을 사용한다. 모든 Riley lane은 projection pipeline을 사용한다. Adaptive 후보만 행 수192를 기준으로 선택하는 FFN backend를 활성화하며 GQA staging/query reuse는 비활성이다. 모델 경로·실행 조건은 launch 기록에, workload source·frozen binary 해시는 preparation에 보존했다.

Shared는32개 변형을 재사용하고 unique는256 warmup+8192 retained의 초기16-token page가 모두 다르다. 각 lane256 warmup+8192 retained 요청, prior/control/adaptive/vLLM 및 역순의 전체16 lane이다. 각 통계는 두 실행의 run-level estimate 중앙값이며 pooled percentile이 아니다. HTTP SSE token arrival을 그대로 사용하며 token 시각을 보간하지 않는다.

| workload | engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 10407.103 | 13.989 | 2.689 | 102.029 | 108.522 | 10.594 | 11.455 |
| shared | control | 10279.036 | 14.149 | 2.726 | 102.824 | 108.604 | 10.717 | 11.575 |
| shared | adaptive | 10196.610 | 14.042 | 2.765 | 103.648 | 108.530 | 10.650 | 11.530 |
| shared | vllm | 11178.774 | 31.153 | 1.835 | 118.762 | 142.988 | 4.298 | 7.616 |
| unique | prior | 3730.682 | 63.547 | 6.686 | 304.622 | 327.490 | 9.941 | 11.646 |
| unique | control | 3789.281 | 62.403 | 6.649 | 293.568 | 306.349 | 9.431 | 10.924 |
| unique | adaptive | 4034.200 | 59.325 | 6.237 | 275.283 | 286.999 | 9.096 | 10.444 |
| unique | vllm | 4741.277 | 48.313 | 5.120 | 275.913 | 329.480 | 7.172 | 12.552 |

## 변화율과 판정

- shared: 후보 throughput은 frozen prior 대비-2.02%, 같은 binary control 대비-0.80%, vLLM 대비-8.79%.
  frozen prior 대비 보고된 latency 증가 항목: ttft_0.5_ms, tpot_0.5_ms, e2e_0.95_ms, e2e_0.99_ms, itl_0.95_ms, itl_0.99_ms.
- unique: 후보 throughput은 frozen prior 대비+8.14%, 같은 binary control 대비+6.46%, vLLM 대비-14.91%.
  frozen prior 대비 보고된 latency 증가 항목: 없음.

위 변화율은 이 고정 workload에서의 기술 통계이며 통계적 유의성이나 전체 목표 달성 판정이 아니다. Throughput만 개선되어도 TPOT·tail·안정성 조건을 만족하지 않으면 승격하지 않는다. 기본 backend 변경과 서버 성능에 대한 GC 이득 합산은 하지 않는다.

Unique는 두 순서 모두 같은 binary control 대비 throughput과 보고 latency가 개선됐다. Shared는 순서별 throughput 방향이 달랐고 중앙값은 소폭 회귀했다. vLLM의 unique E2E P99는398.07/260.89ms로 흔들리므로 후보의 중앙값 P99 우위를 안정적인 tail 우위로 승격하지 않는다. Host IO pressure가 높은 구간도 있었지만 이를 engine 성능 차이의 단일 원인으로 단정하지 않는다. 전체 목표는 미달이며 기본값은 유지한다.

## 각 실행과 host pressure

| 실행 | tok/s | E2E P99 ms | phase 구간 s | host CPU some % | host IO some % |
|---|---:|---:|---:|---:|---:|
| shared-p0-prior | 10425.15 | 107.18 | 25.52 | 1.35 | 3.39 |
| shared-p0-control | 10258.71 | 106.67 | 25.94 | 1.32 | 2.82 |
| shared-p0-adaptive | 10014.79 | 111.49 | 26.58 | 1.39 | 3.55 |
| shared-p0-vllm | 10497.20 | 174.39 | 25.35 | 2.62 | 5.56 |
| shared-p1-vllm | 11860.35 | 111.58 | 22.50 | 1.43 | 3.65 |
| shared-p1-adaptive | 10378.43 | 105.57 | 25.66 | 1.27 | 3.71 |
| shared-p1-control | 10299.36 | 110.53 | 25.86 | 1.46 | 16.88 |
| shared-p1-prior | 10389.06 | 109.86 | 25.63 | 1.32 | 25.60 |
| unique-p0-prior | 3784.32 | 306.46 | 69.68 | 1.13 | 20.30 |
| unique-p0-control | 3793.79 | 306.21 | 69.51 | 1.02 | 25.92 |
| unique-p0-adaptive | 4028.15 | 285.99 | 65.48 | 0.98 | 28.24 |
| unique-p0-vllm | 4553.11 | 398.07 | 57.99 | 1.30 | 31.02 |
| unique-p1-vllm | 4929.44 | 260.89 | 53.60 | 1.08 | 25.25 |
| unique-p1-adaptive | 4040.25 | 288.01 | 65.29 | 1.10 | 22.08 |
| unique-p1-control | 3784.77 | 306.48 | 69.67 | 1.00 | 4.69 |
| unique-p1-prior | 3677.05 | 348.52 | 71.71 | 2.24 | 4.61 |

PSI는 host 전체의 누적 stall counter 차이며 engine별 원인이나 GPU idle의 인과를 증명하지 않는다. Tmpfs는 이번 측정기가 만드는 로그/응답 디스크 쓰기를 측정 밖으로 옮긴 것이며 외부 IO·CPU stall을 제거하지 않는다. Phase bounds는 클라이언트 bookkeeping을 포함할 수 있고 throughput은 실제 첫 request 시작부터 마지막 request 종료로 계산한다.

## 검증과 재현

131072 retained 요청/4194304 output tokens와4096 warmup 응답을 검증했다. Riley retained98304개는 frozen prior와 prompt/output token/text/finish가 일치한다. 저장된 checks flag만 신뢰하지 않고 실제 SSE frames를 재구성하여 fixture·usage·arrival 개수/순서·phase bounds와 대조했다. `[DONE]`은 frames에 없으므로 transport completion은 hash-bound client 검사에 의존한다. vLLM reference agreement는 별도 비교값이며 Riley와 동일 출력이라고 주장하지 않는다.

Stop/cancel/recovery는3개 Riley lane에서 각96건을 검증했다. 각 측정 server exit0, rolling step/drain 및 선택 backend 로그를 확인했다. 모든 warmup/retained phase에서 GC 비활성, GC event/counter 증가0, 종료 후 GC 활성 복구 및 client peak RSS<6GiB guard를 확인했다. 누적 high-water RSS로 장기 누수 부재를 주장하지 않는다.

Tmpfs와 디스크 복사본의 모든 파일 byte size/SHA256 일치 기록은 common archive의 materialization.json에 있다. Lifecycle 실행 및 Blender3개 복구 receipt도 보존했다. 웹 뷰어3개는 계속 유지했다. Evidence는17개 무손실 archive로 나누고 각 archive hash/size를 manifest에 기록한다. Client/controller snapshot과 preparation hash를 고정하고 실행 helper와 로컬 helper의 byte 동일성을 확인한다. 원본 profiler/환경변수 전체 dump는 포함하지 않는다.

## 다음 범위

이 결과는 단일 모델·C32·두 closed-loop 역순 반복의 제한된 비교다. 긴 open-loop/soak, 여러 모델 및 multi-GPU/Hopper/Blackwell runtime qualification은 미완료다. C8/C64 확대 전에 shared 회귀와 unique 개선의 차이를 확인한다. Unified FFN의 작은 행 수 회귀를 threshold 미세 조정으로 감추지 않고 실제 활성 행 분포 및 graph/backend 자원 분리 비용을 다음 판단 근거로 삼는다. 이 과정에서도 vLLM에 미달하는 영역은 남은 서버 병목으로 분리하며 이전에 회귀한 attention 미세 변형을 반복하는 근거로 사용하지 않는다.
