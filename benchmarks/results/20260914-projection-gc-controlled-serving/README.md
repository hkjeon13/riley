# Projection serving — 공통 GC 통제 및 tmpfs 기록

서버 바이너리를 고정하고 Python 벤치마크 클라이언트의 timed-phase GC를 비활성화했다. 모든 엔진의 원본 응답·로그는 tmpfs에 기록하고 GPU 측정 완료 후 디스크로 복사했다. Rust serving 경로와 수치 계약은 변경하지 않았다. 이전 GC-enabled/디스크 기록 결과와 통합하거나 latency를 보정하지 않는다.

## 비교 조건

RTX4090, SmolLM2-135M BF16, context1024, client C32/active32, batch/chunk512, prefix caching 및 각 엔진720MiB KV payload다. Riley는2048 physical KV pages와512 unique prefix-page 한도, rolling decode/prefill FFN/adaptive projection을 사용한다. GQA staging/query reuse는 비활성이고 projection 후보만 새 backend를 활성화한다. 모델 경로·실행 조건은 launch 기록에, workload source·frozen binary 해시는 preparation에 보존했다.

Shared는32개 변형을 재사용하고 unique는256 warmup+8192 retained의 초기16-token page가 모두 다르다. 각 lane256 warmup+8192 retained 요청, prior/control/projection/vLLM 및 역순의 전체16 lane이다. 각 통계는 두 실행의 run-level estimate 중앙값이며 pooled percentile이 아니다. HTTP SSE token arrival을 그대로 사용하며 token 시각을 보간하지 않는다.

| workload | engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 10003.567 | 14.737 | 2.799 | 106.246 | 109.148 | 11.043 | 11.887 |
| shared | control | 10090.086 | 14.436 | 2.787 | 104.983 | 108.615 | 10.756 | 11.603 |
| shared | projection | 10339.939 | 13.893 | 2.723 | 102.306 | 104.886 | 10.449 | 11.291 |
| shared | vllm | 11886.598 | 28.620 | 1.798 | 101.330 | 111.052 | 3.920 | 6.905 |
| unique | prior | 3485.584 | 67.693 | 7.226 | 319.705 | 334.289 | 10.207 | 11.598 |
| unique | control | 3501.338 | 66.946 | 7.195 | 318.488 | 332.870 | 9.999 | 11.486 |
| unique | projection | 3811.233 | 62.180 | 6.616 | 291.679 | 303.996 | 9.259 | 10.620 |
| unique | vllm | 5001.031 | 46.981 | 4.991 | 226.404 | 245.467 | 6.224 | 8.648 |

## 변화율과 판정

- shared: 후보 throughput은 frozen prior 대비+3.36%, 같은 binary control 대비+2.48%, vLLM 대비-13.01%.
  기존 baseline 대비 보고된 latency 증가 항목: 없음.
- unique: 후보 throughput은 frozen prior 대비+9.34%, 같은 binary control 대비+8.85%, vLLM 대비-23.79%.
  기존 baseline 대비 보고된 latency 증가 항목: 없음.

위 변화율은 이 고정 workload에서의 기술 통계이며 통계적 유의성이나 전체 목표 달성 판정이 아니다. Throughput만 개선되어도 TPOT·tail·안정성 조건을 만족하지 않으면 승격하지 않는다. 기본 backend 변경과 서버 성능에 대한 GC 이득 합산은 하지 않는다. 후보는 vLLM보다 shared throughput이13.01%, unique throughput이23.79% 낮다. Shared TTFT·E2E P99는 vLLM보다 낮지만 TPOT·E2E P95·ITL tail은 높고, unique의 표에 있는 latency는 모두 vLLM보다 높다. 전체 목표 미달로 기본값 승격을 보류한다.

## 각 실행과 host pressure

| 실행 | tok/s | E2E P99 ms | phase 구간 s | host CPU some % | host IO some % |
|---|---:|---:|---:|---:|---:|
| shared-p0-prior | 10052.20 | 108.14 | 26.45 | 1.19 | 2.37 |
| shared-p0-control | 9972.11 | 111.50 | 26.69 | 1.18 | 2.59 |
| shared-p0-projection | 10446.98 | 103.72 | 25.48 | 1.21 | 3.01 |
| shared-p0-vllm | 11992.02 | 110.38 | 22.25 | 1.09 | 2.96 |
| shared-p1-vllm | 11781.17 | 111.72 | 22.64 | 1.05 | 3.70 |
| shared-p1-projection | 10232.89 | 106.05 | 26.02 | 1.23 | 2.61 |
| shared-p1-control | 10208.06 | 105.73 | 26.08 | 1.20 | 4.73 |
| shared-p1-prior | 9954.93 | 110.16 | 26.74 | 1.47 | 3.79 |
| unique-p0-prior | 3485.08 | 332.90 | 75.63 | 1.00 | 5.48 |
| unique-p0-control | 3496.75 | 332.75 | 75.38 | 0.99 | 4.81 |
| unique-p0-projection | 3809.81 | 304.86 | 69.21 | 1.01 | 25.77 |
| unique-p0-vllm | 4982.97 | 252.36 | 53.02 | 1.19 | 19.38 |
| unique-p1-vllm | 5019.09 | 238.57 | 52.64 | 1.05 | 25.64 |
| unique-p1-projection | 3812.66 | 303.14 | 69.16 | 0.96 | 27.61 |
| unique-p1-control | 3505.93 | 332.99 | 75.17 | 1.04 | 17.56 |
| unique-p1-prior | 3486.09 | 335.67 | 75.61 | 1.07 | 4.76 |

PSI는 host 전체의 누적 stall counter 차이며 engine별 원인이나 GPU idle의 인과를 증명하지 않는다. Tmpfs는 이번 측정기가 만드는 로그/응답 디스크 쓰기를 측정 밖으로 옮긴 것이며 외부 IO·CPU stall을 제거하지 않는다. Phase bounds는 클라이언트 bookkeeping을 포함할 수 있고 throughput은 실제 첫 request 시작부터 마지막 request 종료로 계산한다.

## 검증과 재현

131072 retained 요청/4194304 output tokens와4096 warmup 응답을 검증했다. Riley retained98304개는 frozen prior와 prompt/output token/text/finish가 일치한다. 저장된 checks flag만 신뢰하지 않고 실제 SSE frames를 재구성하여 fixture·usage·arrival 개수/순서·phase bounds와 대조했다. `[DONE]`은 frames에 없으므로 transport completion은 hash-bound client 검사에 의존한다. vLLM reference agreement는 별도 비교값이며 Riley와 동일 출력이라고 주장하지 않는다.

Stop/cancel/recovery는3개 Riley lane에서 각96건을 검증했다. 각 측정 server exit0, rolling step/drain 및 선택 backend 로그를 확인했다. 모든 warmup/retained phase에서 GC 비활성, GC event/counter 증가0, 종료 후 GC 활성 복구 및 client peak RSS<6GiB guard를 확인했다. 누적 high-water RSS로 장기 누수 부재를 주장하지 않는다.

Tmpfs와 디스크 복사본의 모든 파일 byte size/SHA256 일치 기록은 common archive의 materialization.json에 있다. Lifecycle 실행 및 Blender3개 복구 receipt도 보존했다. 웹 뷰어3개는 계속 유지했다. Evidence는17개 무손실 archive로 나누고 각 archive hash/size를 manifest에 기록한다. Client/controller snapshot과 preparation hash를 고정하고 실행 helper와 로컬 helper의 byte 동일성을 확인한다. 로컬에서17개 archive와167개 파일(4,734,246,393 uncompressed bytes)을 검증했고 원본 파일 hash가 materialization 기록과 일치했다. 원본 profiler/환경변수 전체 dump는 포함하지 않는다.

## 다음 범위

이 결과는 단일 모델·C32·두 closed-loop 역순 반복의 제한된 비교다. 긴 open-loop/soak, 여러 모델 및 multi-GPU/Hopper/Blackwell runtime qualification은 미완료다. 같은 후보와 측정 조건으로 C8/C64의 shared/unique 비교를 수행하여 적용 범위와 회귀를 확인한다. 이 과정에서도 vLLM에 미달하는 영역은 남은 서버 병목으로 분리하며 이전에 회귀한 attention 미세 변형을 반복하는 근거로 사용하지 않는다.
