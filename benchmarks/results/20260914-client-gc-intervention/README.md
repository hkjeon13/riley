# Client GC 대조 실험 — shared C32

서버와 workload를 고정하고 오프라인 Python 클라이언트의 timed-phase GC 정책만 auto/disabled로 변경했다. 두 조건 모두 phase 전 이전 응답을 해제하고 full collection을 수행했다. 따라서 이전 관측 진단의 수치와 합산하지 않는다. Riley Rust → C ABI → CUDA 실행 경로는 바뀌지 않았다.

## 조건

RTX4090, SmolLM2-135M BF16, context1024, shared32 prompt 집합, C32/active32, batch/chunk512, prefix caching, 동일720MiB KV payload다. Riley는 기존 projection-off control binary이며 rolling/prefill FFN/adaptive backend다. 매 lane256 warmup+8192 retained 요청을 사용했다. 순서는 control-auto/control-disabled/vLLM-auto/vLLM-disabled 및 그 역순이다. 실제 각 실행의 원시 통계는 comparison.json과 progress.json에 남아 있다.

다음 표는 두 실행의 run-level 통계 중앙값이다. 서로 다른 요청을 합쳐 percentile을 다시 산출한 값이 아니다.

| 엔진 | 클라이언트 GC | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control | auto | 9589.300 | 14.589 | 2.755 | 117.428 | 317.392 | 10.843 | 11.715 |
| control | disabled | 10145.705 | 14.565 | 2.760 | 103.384 | 106.343 | 10.860 | 11.618 |
| vllm | auto | 11184.160 | 29.116 | 1.780 | 105.326 | 331.288 | 3.762 | 6.798 |
| vllm | disabled | 10427.640 | 27.480 | 1.811 | 100.144 | 233.743 | 4.017 | 6.579 |

## 판정

- control: disabled에서 throughput +5.80%, E2E P99 -66.49% 변화.
- vllm: disabled에서 throughput -6.76%, E2E P99 -29.44% 변화.
- 동일 disabled 조건의 Riley throughput은 vLLM 대비 -2.70%. 전체 serving 목표 달성 여부는 throughput뿐 아니라 표의 모든 latency와 미완료 workload/hardware gate를 함께 판단한다.

GC 비활성 phase에 GC callback event와 collection counter 증가가 없음을 검증했다. 자동 GC phase의 이벤트는 계속 기록했다. 측정 후 GC를 finally에서 복구하며 사전 collection/결과 직렬화는 token timing 밖이다. GC disable은 CPU allocation, 참조계수 처리, OS scheduling 또는 Python GIL을 제거하지 않는다. Callback wall time을 latency에서 빼거나 기존 측정치를 수정하지 않았다.

### 반복별 변동과 host pressure

| 실행 | tok/s | E2E P99 ms | host IO some % |
|---|---:|---:|---:|
| shared-p0-control-auto | 9585.76 | 305.16 | 3.05 |
| shared-p0-control-disabled | 10128.78 | 106.68 | 4.50 |
| shared-p0-vllm-auto | 11159.33 | 320.77 | 2.96 |
| shared-p0-vllm-disabled | 12016.04 | 104.52 | 3.26 |
| shared-p1-control-auto | 9592.84 | 329.62 | 4.35 |
| shared-p1-control-disabled | 10162.63 | 106.01 | 4.38 |
| shared-p1-vllm-auto | 11208.99 | 341.80 | 2.86 |
| shared-p1-vllm-disabled | 8839.24 | 362.96 | 25.15 |

Riley는 두 순서 모두 GC 비활성 시 throughput 약5.7–5.9% 증가, E2E P99 약106ms를 보였다. 반면 vLLM disabled는12016→8839 tok/s, P99 104.52→362.96ms로 크게 흔들렸다. 느린 run의 host IO some은25.15%로 나머지2.86–4.50%보다 높지만 global PSI는 프로세스별 지연 원인의 증거가 아니다. GC 비활성으로 모든 tail이 제거된다는 가설은 지지되지 않는다. 두 disabled run의 중앙값만으로 Riley가 vLLM에 근접했다고 주장하지 않는다.

## 검증·한계

65536 retained 응답/2097152 output tokens 및2048 warmup 응답의 SSE frame과 fixture, prompt/output token, text, finish, usage, arrival 순서·phase 경계를 검증했다. Riley retained32768개는 frozen prior와 정확히 일치한다. vLLM agreement는 별도 통계이며 동일 출력으로 주장하지 않는다. Frame에 보존되지 않는 `[DONE]` 검증은 hash-bound client에 의존한다. GC count delta/이벤트 경계·비중첩, 정책 복구와 peak RSS<6GiB guard를 확인했다. RSS는 누적 high-water mark이며 lane 메모리 사용량이나 장기 누수 검사로 해석하지 않는다.

8개 측정 server exit0, lifecycle 성공과 Blender3개 scene-query 복구 receipt를 보존했다. 세 웹 뷰어는 계속 유지했다. Server 종료 경고는 원본 로그에 보존한다. Raw profiler/환경변수 전체 dump는 수집하지 않았다. 준비/controller/client hash, 원본 response 및 lifecycle은 evidence의9개 무손실 archive/manifest에 있다. 로컬 재분석에서 기록된 progress 통계와 일치함을 확인한다.

고정 요청 수, shared32, 단일 모델·C32와 두 순서의 제한된 진단이다. Open-loop, 지속적인 신규 prefix, 여러 모델/하드웨어 또는 장기 안정성을 검증하지 않는다. 이득을 서버 backend 개선으로 집계하지 않는다. GC 영향을 통제한 versioned client를 양쪽에 공통 적용해 projection 후보와 frozen prior/control/vLLM을 재비교하고 C8/C64 및 unique workload를 이어서 평가한다. 측정 클라이언트의 인위적인 time correction은 하지 않는다.

## 다음 비교의 측정 조건

GC 비활성은 양쪽 engine에 공통 적용하고, bench 자체의 per-request log/원본 JSON 디스크 쓰기와 다음 lane의 writeback overlap을 줄이기 위해 로그·응답을 `/dev/shm` tmpfs에 기록한다. GPU 측정 완료와 Blender 복구 후 결과를 기존 evidence root로 복사한다. 기록을 버리거나 요청별 증거를 줄이지 않는다. 실행 전 tmpfs 여유12GiB, host MemAvailable16GiB를 요구하고 클라이언트 RSS6GiB guard를 유지한다. 이 저장 경로 변경도 새 조건으로 명시하며 외부 IO 또는 CPU stall 제거를 주장하지 않는다. 다음 source/controller와 검증기를 준비했으며 이 보고서의 실험에는 적용되지 않았다.
