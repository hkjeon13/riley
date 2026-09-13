# Paired decode actual serving trace — retry completed

대상 runtime commit은 `7d5439713346d7fcd01e7488cd368c3f158e9ff9`다. 이전 I/O 압박 때 실패했던 Nsight 수집을 재시도하여 single·paired 모두 HTTP 96개, frozen reference exact, exit 0, 잔여 소유 process 0으로 완료했다. Serving runtime은 Rust→C ABI→CUDA이며 Python은 외부 측정/분석 도구다. 이번 변경은 분석기와 증거 기록이며 runtime 성능 변경은 없다.

## 조건과 재현

RTX 4090, SmolLM2-135M, C32, V7, GUI 유지·Blender 종료. 명령·binary/controller/client SHA는 각 `*-launch.json`, correctness/peak RSS/종료는 `*-receipt.json`에 있다. Binary SHA는 두 lane 모두 `b24a67903c86afd10b8bf7b4be4b1d05343fbd421733d76ae8c5d3eeaa151241`이다. Peak owned-tree RSS는 single 1,496,043,520 bytes, paired 1,521,946,624 bytes다. 8 GiB RSS watchdog은 샘플링 감시이며 cgroup hard limit은 아니다.

원본 SQLite·Nsight report·HTTP rows·로그는 SSH host `ai-assistant`의 `/data/riley-serving-260913-recovery/paired-trace-v3`에 보존한다. `raw-manifest.json`에 각 파일 크기와 SHA256이 있다. 분석은 `paired_decode_trace_analysis.py INPUT.sqlite OUTPUT.json`, `serving_area_census.py INPUT.sqlite OUTPUT.json`으로 재현한다. SQLite SHA는 분석 JSON에도 기록한다.

Paired capture의 일부 launch는 두 stream에 걸친 kernel event를 가진다. 분석기는 명시적인 `merge_graph_streams=True`에서 process/context/correlation별 내부 stream을 합치고 실제 stream 목록을 보존한다. GPU active 시간은 interval union으로 계산한다. 서로 다른 launch envelope의 겹침, 누락된 launch, 모호한 correlation은 계속 거부한다. CLI 기본은 기존 strict single-stream이다. 다중 stream 허용·중복 시간 제거·launch 간 겹침 거부를 포함한 분석 테스트 8개가 통과했다.

## 관측 결과

시작/종료를 정확히 잘라낸 steady state가 아닌 **launch 개수 기준 중간 80%**다. 두 trace의 batch 구성도 완전히 같지 않으므로 아래 시간은 unprofiled 성능 비교로 사용하지 않는다.

| Trace 항목 | single | paired |
|---|---:|---:|
| 선택 launch 수 | 242 | 241 |
| 선택 GPU window (ms) | 618.449 | 607.111 |
| Graph envelope 합 (ms) | 504.327 | 492.860 |
| Graph 사이 gap 합 (ms) | 114.122 | 114.251 |
| 그중 이전 GPU 종료→다음 CPU launch 시작 (ms) | 112.197 | 112.319 |
| Graph 내부 activity 없는 시간 (ms) | 37.947 | 28.435 |

| Paired 경계 | 개수 | gap 중앙값 (µs) | P95 (µs) | 합 (ms) |
|---|---:|---:|---:|---:|
| pair 내부 | 88 | 6.161 | 9.250 | 0.550 |
| pair 이후 | 88 | 716.528 | 1,143.706 | 59.995 |
| 기타 | 64 | 637.029 | 1,826.069 | 53.706 |

Pair 내부의 post-GPU/next-CPU-launch gap은 모두 0이다. 하지만 pair 이후 host gap 중앙값은 704.890 µs다. 이 결과는 두 graph 선제 제출이 내부 대기를 없애면서, 결과 처리/다음 준비 경계에는 큰 대기가 남는다는 해석을 지지한다. 각 host 세부 작업의 인과관계를 이 trace만으로 확정하지 않는다. Client pacing과 profiler 비용도 포함된다. 모든 graph 사이 gap을 제거한다는 낙관적 계산은 single 1.226×, paired 1.232×이지만 **실제 serving 예상 개선율이 아니다**.

Decode kernel duration 합의 구성은 single attention 33.28%, FFN 32.63%, 기타 projection/QKV/RoPE 20.61%, norm 8.05%다. Paired도 33.24%/32.50%/20.57%/8.03%로 비슷하다. 이는 kernel duration 합의 비율이며 wall time 비율이나 HBM bandwidth 측정이 아니다. 기존 attention/FFN negative experiment를 뒤집는 근거가 아니다.

## 직전 unprofiled serving 비교 — 이번 trace와 구분

[원 측정 및 조건](../20260913-prepared-window-batch/README.md). 아래는 재측정한 수치가 아니라 직전 milestone의 두 순서 측정을 요약한 것이다. 처리량 및 TTFT/TPOT는 두 run의 중앙값, E2E P99는 더 나쁜 run이다.

| Lane | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P99 ms |
|---|---:|---:|---:|---:|
| single | 10,338.63 | 10.142 | 2.893 | 492.067 |
| 직전 paired | 10,036.53 | 10.886 | 2.952 | 476.177 |
| 현재 paired | 10,347.99 | 10.874 | 2.834 | 509.770 |
| vLLM 0.27.1 | 11,599.06 | 20.096 | 2.312 | 474.229 |

현재 paired는 직전 paired보다 약 3.1% 빠르지만 vLLM보다 10.79% 낮다. Default는 single 유지. 높은 concurrency/장시간 안정성 및 최종 목표 달성을 주장하지 않는다.

## 다음 optimization batch

PR02의 실제 host/GPU overlap을 우선 검토한다. Pair 길이를 단순히 늘리는 작업은 이번 근거로 선택하지 않는다.

1. 완료된 결과의 검증·scheduler commit·HTTP publication과 다음 GPU 실행을 분리할 수 있는 소유권 경계를 설계한다. 실제 next launch 이후 이전 결과 처리 시간이 겹치는지 측정 가능하게 만든다.
2. 다음 descriptor 준비를 active GPU execution과 겹치되 EOS/cancel/page generation과 admission에 의존하는 부분은 별도로 확정한다. 현재 두-slot 수명 및 GPU 완료 전 재사용 금지를 보존한다.
3. Event query/drain과 publication 순서를 연결하고, overlap 구간 및 pair 이후 gap을 함께 확인한다. Future token을 사용자에게 먼저 공개하거나 correctness gate를 낮추지 않는다.

구현 전에 현재 scheduler/session borrow 및 commit 의존성을 확인해야 한다. 이 문서는 위 구조가 구현되었다는 증거가 아니다. 구현 완료 후 exact reference, stop/cancel/page 경계 회귀를 통과한 후보에만 같은 workload의 single/직전/후보/vLLM 두 순서 비교를 실행한다. 처리량과 TPOT뿐 아니라 P95/P99도 개선 여부를 판정하며, 효과가 없으면 기본값을 바꾸지 않는다. Hopper/Blackwell/multi-GPU 전용 검사는 장비 부재를 명시하되 4090에서 실행 가능한 실패를 skip하지 않는다.
