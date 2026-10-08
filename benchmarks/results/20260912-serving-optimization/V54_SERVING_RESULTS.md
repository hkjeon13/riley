# V54 serving 결과 — 채택 거절

V54는 packed V 직접 쓰기와 vector pair 읽기를 V7 mixed/decode 경로에 함께 적용했다. Correctness 검증은 통과했지만 자연 길이 serving throughput과 TPOT가 악화되어 기본 후보로 채택하지 않는다. 일반 비교 기준 V51과 실험 기준 V52의 frozen binary를 보존한다. 전체 목표는 미달이다.

## 같은 조건의 비교

RTX 4090, SmolLM2-135M BF16, V7 GPU greedy, token budget 512, admission C16/C32. Fixed는 chunk 128, natural은 chunk 512이며 vLLM 0.27.1과 동일 외부 workload 및 KV budget을 사용했다. V51/V52/V54/vLLM의 두 역순, 32 lanes, lane당 warmup 96/retained 384다. 총 12,288개 retained 요청 실패 0, Riley reference 9,216개 모두 일치했다. vLLM의 일부 출력은 Riley reference와 다르므로 엔진 간 모든 출력이 동일하다고 주장하지 않는다. 생성 token 총량은 lane별 fixed 7,680/natural 28,672로 일치했다.

아래 수치는 두 실행의 중앙값이다. 장시간 안정성 또는 통계적 유의성 검증으로 해석하지 않는다.

| Workload | V51 tok/s | V52 tok/s | V54 tok/s | vLLM tok/s | V54 TPOT ms | vLLM TPOT ms | V54 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 6136.266 | 6302.281 | 6237.151 | 5098.439 | 2.310 | 2.249 | 84.533 | 115.662 |
| c16-natural | 7756.540 | 7800.817 | 7316.180 | 7957.458 | 2.057 | 1.786 | 278.058 | 269.270 |
| c32-fixed | 8072.680 | 8166.518 | 8259.346 | 7455.701 | 3.489 | 2.862 | 141.175 | 158.317 |
| c32-natural | 10264.532 | 10344.122 | 10100.575 | 11204.474 | 2.965 | 2.366 | 424.967 | 415.656 |

V54 vs V52 throughput 변화는 C16 fixed −1.03%, C16 natural −6.21%, C32 fixed +1.14%, C32 natural −2.35%다. Natural TPOT는 각각 +6.92%, +1.95% 악화됐다. vLLM 대비 TPOT는 모든 조건에서 2.70–25.29% 높았다. TTFT는 낮지만 최종 성공 기준을 충족하지 못한다.

## 원인과 다음 실험

Nsight trace에서 C32 natural mapped attention은 V52 52.307→V54 45.015µs, decode independent_values는 9.730→15.267µs다. V 읽기 배치가 mixed 경로를 돕지만 decode를 손해 보게 하는 양상이다. 전체 graph의 prefill_or_mixed 구분은 pure prefill과 mixed를 합친 것이며, 실행 구성 차이 때문에 이 kernel 평균을 그대로 serving 효과로 계산하지 않는다. Profiler의 CUDA API 시간은 production latency가 아니다.

세 V55 prototype은 packed decode의 branchless bit mask, 완전한 K16 tile의 fast path, K16 순차 loop를 비교한다. V52 token-major와 V54 packed 기준을 모두 포함한다. 현재 primitive 검증 중이며 V55 application 변경/성능 채택은 없다.

## 검증 및 보존

V54 commit `636376e67fdcc508552371628808fc21d2378c44`, binary SHA256 `14f772200199121fb05127948129a5be163b4c97f610638a5b6adf0ddcbb6c52`. Full model 16 regressions, V7 memcheck 3, HTTP 37, CPU/GPU fallback 22응답 일치. Mixed primitive와 context4096 경계, writer 84 cases, decode 168 cases 및 memcheck/racecheck 통과.

- `raw/integration-v54`: 31파일, archive SHA256 `95edfbd83b9b8caa11527291bc2152312ce1f826fdfaffa04b71b35c3b3e7866`.
- `raw/packed-v54`: 70파일, archive SHA256 `46bc05a92b325baab39f7c27920f70e2883e10c10d397cda9b02a692e7d8bcfd`.
- `raw/serving-round60`: 267파일, archive SHA256 `b2d12d7512fc41d33f69b0a09ed0f3c85978befa1e7f216b8439ed04c1f767ba`.

각 파일의 size/SHA256를 local에서 검증했다. Remote controller stdout을 파일로 저장하지 않아 최초 exporter가 중단됐으며, 존재하지 않는 log 대신 실제 terminal 완료를 기록한 execution receipt를 포함하도록 수정했다. Benchmark는 반복하지 않았다. Blender의 기존 PID/port 부재도 exporter가 확인했다. Blender는 종료 상태를 유지한다.

V54 profile 35개 파일도 `raw/profile-v54`에서 size/SHA256를 검증했다. Archive SHA256 `24a3e177acfc9db6e5f3a6b34d494816fac747925ab1becf516e5dd79095338f`. 자연 길이 C16/C32와 fixed C32의 총 288개 요청 기준 출력이 일치했다.
