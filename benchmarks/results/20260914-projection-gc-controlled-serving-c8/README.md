# Projection serving — 공통 GC 통제 및 tmpfs 기록

서버 바이너리를 고정하고 Python 벤치마크 클라이언트의 timed-phase GC를 비활성화했다. 모든 엔진의 원본 응답·로그는 tmpfs에 기록하고 GPU 측정 완료 후 디스크로 복사했다. Rust serving 경로와 수치 계약은 변경하지 않았다. 이전 GC-enabled/디스크 기록 결과와 통합하거나 latency를 보정하지 않는다.

## 비교 조건

RTX4090, SmolLM2-135M BF16, context1024, client C8/active8, batch/chunk512, prefix caching 및 각 엔진720MiB KV payload다. Riley는2048 physical KV pages와512 unique prefix-page 한도, rolling decode/prefill FFN/adaptive projection을 사용한다. GQA staging/query reuse는 비활성이고 projection 후보만 새 backend를 활성화한다. 모델 경로·실행 조건은 launch 기록에, workload source·frozen binary 해시는 preparation에 보존했다.

Shared는32개 변형을 재사용하고 unique는256 warmup+8192 retained의 초기16-token page가 모두 다르다. 각 lane256 warmup+8192 retained 요청, prior/control/projection/vLLM 및 역순의 전체16 lane이다. 각 통계는 두 실행의 run-level estimate 중앙값이며 pooled percentile이 아니다. HTTP SSE token arrival을 그대로 사용하며 token 시각을 보간하지 않는다.

| workload | engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 5057.940 | 11.964 | 1.234 | 52.854 | 54.056 | 1.722 | 3.620 |
| shared | control | 5091.214 | 12.185 | 1.227 | 51.893 | 53.257 | 1.722 | 3.616 |
| shared | projection | 5141.116 | 11.758 | 1.223 | 51.321 | 52.916 | 1.731 | 3.338 |
| shared | vllm | 4512.614 | 14.291 | 1.324 | 63.823 | 71.775 | 1.879 | 4.409 |
| unique | prior | 2606.097 | 55.428 | 1.380 | 99.300 | 100.478 | 5.362 | 5.737 |
| unique | control | 2607.196 | 55.285 | 1.379 | 99.232 | 100.860 | 5.276 | 5.731 |
| unique | projection | 2762.334 | 50.220 | 1.364 | 93.630 | 95.439 | 4.809 | 5.162 |
| unique | vllm | 2888.275 | 29.409 | 1.861 | 102.026 | 115.540 | 5.045 | 5.898 |

## 변화율과 판정

- shared: 후보 throughput은 frozen prior 대비+1.64%, 같은 binary control 대비+0.98%, vLLM 대비+13.93%.
  기존 baseline 대비 보고된 latency 증가 항목: itl_0.95_ms.
- unique: 후보 throughput은 frozen prior 대비+6.00%, 같은 binary control 대비+5.95%, vLLM 대비-4.36%.
  기존 baseline 대비 보고된 latency 증가 항목: 없음.

위 변화율은 이 고정 workload에서의 기술 통계이며 통계적 유의성이나 전체 목표 달성 판정이 아니다. Throughput만 개선되어도 TPOT·tail·안정성 조건을 만족하지 않으면 승격하지 않는다. 기본 backend 변경과 서버 성능에 대한 GC 이득 합산은 하지 않는다. C8 shared에서는 vLLM 대비 throughput +13.93%이고 표의 TTFT/TPOT·E2E 및 ITL tail도 낮아 이 제한된 조건의 최소 지표를 만족했다. +15% throughput/−10% TPOT 목표폭에는 못 미쳤다. Unique는 throughput −4.36%, TTFT 약70.76% 증가로 실패하며, TPOT·E2E·ITL tail 개선과 분리해서 해석한다. C32 실패와 미완료 qualification을 덮어쓰지 않는다. Shared ITL P95는 frozen prior보다 약0.49% 높아 모든 기존 지표가 개선됐다고 주장하지 않는다.

## 각 실행과 host pressure

| 실행 | tok/s | E2E P99 ms | phase 구간 s | host CPU some % | host IO some % |
|---|---:|---:|---:|---:|---:|
| shared-p0-prior | 5073.72 | 54.24 | 52.00 | 0.96 | 3.08 |
| shared-p0-control | 5073.85 | 54.01 | 52.00 | 1.04 | 3.52 |
| shared-p0-projection | 5136.79 | 52.89 | 51.37 | 0.93 | 3.42 |
| shared-p0-vllm | 4482.01 | 73.74 | 58.83 | 1.06 | 5.38 |
| shared-p1-vllm | 4543.22 | 69.81 | 58.04 | 1.02 | 3.65 |
| shared-p1-projection | 5145.44 | 52.94 | 51.29 | 1.00 | 2.78 |
| shared-p1-control | 5108.58 | 52.51 | 51.67 | 1.12 | 2.58 |
| shared-p1-prior | 5042.16 | 53.87 | 52.35 | 1.02 | 3.52 |
| unique-p0-prior | 2609.78 | 100.58 | 100.82 | 0.82 | 6.01 |
| unique-p0-control | 2608.93 | 100.79 | 100.84 | 0.85 | 7.38 |
| unique-p0-projection | 2762.95 | 95.33 | 95.23 | 0.94 | 15.77 |
| unique-p0-vllm | 2921.93 | 111.80 | 90.08 | 0.94 | 3.91 |
| unique-p1-vllm | 2854.62 | 119.28 | 92.19 | 0.99 | 3.58 |
| unique-p1-projection | 2761.72 | 95.54 | 95.27 | 0.94 | 4.01 |
| unique-p1-control | 2605.47 | 100.93 | 100.97 | 0.86 | 9.92 |
| unique-p1-prior | 2602.42 | 100.38 | 101.08 | 0.93 | 13.84 |

PSI는 host 전체의 누적 stall counter 차이며 engine별 원인이나 GPU idle의 인과를 증명하지 않는다. Tmpfs는 이번 측정기가 만드는 로그/응답 디스크 쓰기를 측정 밖으로 옮긴 것이며 외부 IO·CPU stall을 제거하지 않는다. Phase bounds는 클라이언트 bookkeeping을 포함할 수 있고 throughput은 실제 첫 request 시작부터 마지막 request 종료로 계산한다.

## 검증과 재현

131072 retained 요청/4194304 output tokens와4096 warmup 응답을 검증했다. Riley retained98304개는 frozen prior와 prompt/output token/text/finish가 일치한다. 저장된 checks flag만 신뢰하지 않고 실제 SSE frames를 재구성하여 fixture·usage·arrival 개수/순서·phase bounds와 대조했다. `[DONE]`은 frames에 없으므로 transport completion은 hash-bound client 검사에 의존한다. vLLM reference agreement는 별도 비교값이며 Riley와 동일 출력이라고 주장하지 않는다.

Stop/cancel/recovery는3개 Riley lane에서 각24건을 검증했다. 각 측정 server exit0, rolling step/drain 및 선택 backend 로그를 확인했다. 모든 warmup/retained phase에서 GC 비활성, GC event/counter 증가0, 종료 후 GC 활성 복구 및 client peak RSS<6GiB guard를 확인했다. 누적 high-water RSS로 장기 누수 부재를 주장하지 않는다.

Tmpfs와 디스크 복사본의 모든 파일 byte size/SHA256 일치 기록은 common archive의 materialization.json에 있다. Lifecycle 실행 및 Blender3개 복구 receipt도 보존했다. 웹 뷰어3개는 계속 유지했다. Evidence는17개 무손실 archive로 나누고 각 archive hash/size를 manifest에 기록한다. Client/controller snapshot과 preparation hash를 고정하고 실행 helper와 로컬 helper의 byte 동일성을 확인한다. 로컬에서17개 archive 및167개 파일(4,736,660,927 uncompressed bytes)의 무결성과 materialization hash를 검증했다. 원본 profiler/환경변수 전체 dump는 포함하지 않는다.

## 다음 범위

이 결과는 단일 모델·C8·두 closed-loop 역순 반복의 제한된 비교다. 긴 open-loop/soak, 여러 모델 및 multi-GPU/Hopper/Blackwell runtime qualification은 미완료다. 같은 후보와 측정 조건으로 남은 C64(client64/active32)의 shared/unique 비교를 수행하여 적용 범위와 회귀를 확인한다. 이 과정에서도 vLLM에 미달하는 영역은 남은 서버 병목으로 분리하며 이전에 회귀한 attention 미세 변형을 반복하는 근거로 사용하지 않는다.
