# Round14: 완료된 C1 비교와 Batch8 기각

**Batch8은 기각하고 Batch7 API baseline을 유지한다.** Candidate/baseline 다섯 쌍 모두 throughput이 낮고 median token TPOT가 높았다. Paired ratio 중앙값 기준 throughput **-5.799%**, TPOT **+7.305%**, E2E **+6.200%**이며 TTFT는 **-0.054%로 사실상 동일**했다. 수치 검증 성공과 serving 성능 개선은 별개다. 원인은 아직 확정하지 않았고 결합 구간 profiling으로 확인한다.

Baseline/vLLM 다섯 쌍의 throughput 비율 중앙값은 **1.037739 (+3.774%)**, token TTFT는 **0.701692 (-29.831%)**지만 median token TPOT는 **1.054239 (+5.424%)**다. Batch8/vLLM은 throughput **0.985882 (-1.412%)**, median TPOT **1.131935 (+13.193%)**, median E2E **1.036563 (+3.656%)**다. 일부 latency tail이 더 짧더라도 median TPOT와 throughput 목표를 대체하지 않는다. 전체 목표는 미달성이다.

## 종료·검증 범위

원래 C1/C2 × 세 비교 × 5쌍의 **30-pair 계획은 수정하지 않았다.** 모든 C1 반복인 **15쌍·30개 독립 서버·30,000개 retained 요청**을 마친 뒤 정확한 controller에 SIGINT를 보내 정상 정리·복원 경로로 중단했다. Nonstream 5개와 stream 5개 warmup을 각 서버에서 별도로 검증해 총 응답은 **30,300개**다. C2 서버 launch는 **0개**, C2 성능 결과도 없다. 전체 campaign에는 `KeyboardInterrupt` finalization만 있고 성공 `completion.json`은 없으므로 **의도적 미완료**다.

V3는 원본 raw 응답·해시·source/model/reference·pair 순서·모든 프로세스 정리·Blender 복원을 검증한 뒤, 완결된 C1 항목 세 개만 기술 통계로 보고한다. `cleanup_gate.validated=true`, errors는 비어 있고 V2의 전체 campaign 상태는 여전히 `incomplete`다. 아래 표를 전체 캠페인 완료나 폭넓은 serving 성공으로 해석하지 않는다.

조건은 동일 RTX 4090 / SmolLM2-135M BF16 / P128·O32 / GPU greedy / offered C1 / vLLM token budget 128이다. GUI는 유지하고 허용된 Blender 세 개만 일시 중지했다. Private 580.173.02 runtime, foreign CUDA compute 없음, 시작 온도 ≤48°C, idle GPU memory ≤512 MiB 조건이며 canonical headless 조건으로 표시하지 않는다. API baseline source는 `a179617070526068b66ba5627ba82a7151da8c64`, Batch8은 `8329c1aeec6e013f581128888c536e15f8bf7300`이다.

## 집계 읽는 법

절대값은 **다섯 프로세스 통계의 중앙값 [최소, 최대]**다. 각 프로세스에서 1,000개 retained 응답으로 median/P95/P99를 계산했다. 따라서 표의 P99는 다섯 개 process P99의 분포이며, 5,000개 응답을 합친 P99가 아니다. Paired ratio 열은 각 쌍에서 오른쪽/왼쪽을 계산한 **다섯 비율의 중앙값 [최소, 최대]**다. 절대값 중앙값끼리 나눈 결과로 바꾸지 않았다. Throughput 비율은 클수록, latency 비율은 작을수록 좋다. 동일 lane도 비교 항목마다 별도 프로세스이므로 항목 사이에서 합치지 않는다.

모든 throughput 분자는 프로세스당 output token **32,000개**다. Common wall은 첫 요청 시작부터 마지막 terminal delivery까지이며, 별도의 phase wall은 worker drain·EOF 검증까지 포함한다. 표는 common-wall rate를 사용한다. TTFT/TPOT는 실제 HTTP token-ID 프레임의 **클라이언트 전달 시각**이고 GPU commit/engine 시간은 아니다. TPOT는 각 응답에서 첫 ID부터 마지막 ID까지의 시간을 31개 간격으로 나눈 값이다. E2E는 terminal delivery까지 포함한다. 이번 retained 데이터에서는 첫 token frame이 비어 있지 않아 first-text 통계가 TTFT와 같지만 두 개념을 일반적으로 동일시하지 않는다.

## baseline-vllm

비율 방향: **Batch7 API baseline / vLLM**. 각 lane은 별도 5개 프로세스 × 1,000개 retained 응답이다.

| 지표 | vLLM | Batch7 API baseline | Paired ratio |
| --- | ---: | ---: | ---: |
| Throughput (tok/s) | 885.684 [881.386, 887.312] | 918.839 [917.786, 919.109] | 1.037739 [1.034345, 1.042369] |
| Token TTFT median (ms) | 7.366680 [7.349728, 7.462474] | 5.169137 [5.158699, 5.180565] | 0.701692 [0.694215, 0.703207] |
| Token TTFT p95 (ms) | 9.828049 [9.541521, 11.505292] | 5.286585 [5.282710, 5.314617] | 0.539041 [0.459492, 0.556999] |
| Token TTFT p99 (ms) | 16.384699 [16.116374, 16.662644] | 5.366364 [5.361346, 5.435628] | 0.327217 [0.322060, 0.337274] |
| Token TPOT median (ms) | 0.900157 [0.897316, 0.900464] | 0.948936 [0.948818, 0.949218] | 1.054239 [1.054023, 1.057501] |
| Token TPOT p95 (ms) | 0.990659 [0.980698, 0.998679] | 0.958355 [0.954058, 0.962875] | 0.965885 [0.956855, 0.981827] |
| Token TPOT p99 (ms) | 1.110615 [1.098495, 1.174159] | 0.966747 [0.963650, 0.971831] | 0.869203 [0.825222, 0.884693] |
| E2E median (ms) | 35.357063 [35.252223, 35.463351] | 34.634668 [34.617073, 34.641804] | 0.979405 [0.976712, 0.981983] |
| E2E p95 (ms) | 40.752830 [39.910965, 41.059586] | 34.957050 [34.901266, 35.053382] | 0.856413 [0.851943, 0.878290] |
| E2E p99 (ms) | 48.225265 [46.078313, 49.956549] | 35.269133 [35.234459, 35.491010] | 0.730622 [0.705402, 0.770232] |

## candidate-baseline

비율 방향: **Batch8 / Batch7 API baseline**. 각 lane은 별도 5개 프로세스 × 1,000개 retained 응답이다.

| 지표 | Batch7 API baseline | Batch8 | Paired ratio |
| --- | ---: | ---: | ---: |
| Throughput (tok/s) | 919.131 [918.475, 919.336] | 865.719 [865.471, 866.707] | 0.942007 [0.941888, 0.942752] |
| Token TTFT median (ms) | 5.165011 [5.150026, 5.170550] | 5.165701 [5.155036, 5.171065] | 0.999462 [0.998599, 1.003422] |
| Token TTFT p95 (ms) | 5.273814 [5.267340, 5.286960] | 5.272488 [5.253142, 5.279650] | 0.999386 [0.996934, 1.000356] |
| Token TTFT p99 (ms) | 5.377274 [5.362491, 5.402086] | 5.380490 [5.337833, 5.387491] | 1.000757 [0.988106, 1.003356] |
| Token TPOT median (ms) | 0.949140 [0.948847, 0.949204] | 1.018475 [1.018129, 1.018567] | 1.073050 [1.072708, 1.073478] |
| Token TPOT p95 (ms) | 0.957192 [0.952272, 0.960739] | 1.023242 [1.020085, 1.023808] | 1.069077 [1.065057, 1.073275] |
| Token TPOT p99 (ms) | 0.967069 [0.960952, 0.968609] | 1.034281 [1.022511, 1.035345] | 1.069500 [1.060604, 1.072419] |
| E2E median (ms) | 34.627021 [34.615259, 34.637509] | 36.777864 [36.756430, 36.781440] | 1.062003 [1.061774, 1.062475] |
| E2E p95 (ms) | 34.942591 [34.855606, 34.982694] | 36.984180 [36.873818, 37.055686] | 1.058861 [1.056729, 1.060264] |
| E2E p99 (ms) | 35.243297 [35.108909, 35.349789] | 37.378018 [37.012606, 37.443469] | 1.059635 [1.050203, 1.064711] |

## candidate-vllm

비율 방향: **Batch8 / vLLM**. 각 lane은 별도 5개 프로세스 × 1,000개 retained 응답이다.

| 지표 | vLLM | Batch8 | Paired ratio |
| --- | ---: | ---: | ---: |
| Throughput (tok/s) | 878.353 [871.063, 882.457] | 865.805 [865.670, 865.952] | 0.985882 [0.981130, 0.993920] |
| Token TTFT median (ms) | 7.456696 [7.374497, 7.600096] | 5.165586 [5.156032, 5.173191] | 0.693044 [0.679182, 0.701498] |
| Token TTFT p95 (ms) | 10.943427 [10.666388, 12.951874] | 5.274543 [5.253102, 5.282911] | 0.482747 [0.407466, 0.494501] |
| Token TTFT p99 (ms) | 16.795247 [16.708200, 17.301019] | 5.374115 [5.365771, 5.395473] | 0.319501 [0.310214, 0.322582] |
| Token TPOT median (ms) | 0.899813 [0.899617, 0.900301] | 1.018495 [1.018183, 1.018590] | 1.131935 [1.131121, 1.132187] |
| Token TPOT p95 (ms) | 1.006954 [1.003961, 1.042444] | 1.022622 [1.020948, 1.024094] | 1.016919 [0.979700, 1.017022] |
| Token TPOT p99 (ms) | 1.131647 [1.107004, 1.184280] | 1.034390 [1.028907, 1.038800] | 0.909213 [0.873394, 0.938388] |
| E2E median (ms) | 35.476852 [35.383534, 35.567696] | 36.773988 [36.768598, 36.779920] | 1.036563 [1.033965, 1.039144] |
| E2E p95 (ms) | 42.090206 [40.791081, 43.963441] | 36.964879 [36.934661, 37.024761] | 0.878230 [0.840639, 0.907668] |
| E2E p99 (ms) | 48.646495 [47.188940, 50.279658] | 37.433338 [37.251244, 37.510984] | 0.765754 [0.746047, 0.793989] |

## Grouped-frame 관측과 tail 한계

같은 SSE frame의 여러 ID에는 동일한 도착 시각을 부여하므로 내부 ITL은 0이다. 서로 다른 frame도 같은 socket read에서 완료되면 같은 시각을 가질 수 있다. 이는 동시 GPU 생성이나 0-cost decode의 증거가 아니다. 아래는 각 항목·lane의 5,000개 retained 요청에서 합한 관측 횟수이며, 요청 percentile을 합친 값이 아니다. 전체 frame 크기 histogram은 분석 JSON에 보존된다.

| 비교 | Lane | Multi-token frames | 그 frame의 ID 수 | 최대 IDs/frame | Frame 내부 zero ITL | Frame 사이 zero ITL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline-vllm | vLLM | 80 | 201 | 5 | 121 | 0 |
| baseline-vllm | Batch7 API baseline | 0 | 0 | 1 | 0 | 72 |
| candidate-baseline | Batch7 API baseline | 0 | 0 | 1 | 0 | 63 |
| candidate-baseline | Batch8 | 0 | 0 | 1 | 0 | 43 |
| candidate-vllm | vLLM | 73 | 190 | 7 | 117 | 0 |
| candidate-vllm | Batch8 | 0 | 0 | 1 | 0 | 56 |

각 프로세스는 ITL 31,000개를 기록했다. 이번 retained 응답의 blank generated frame/token은 모두 0개다. 별도 API correctness 검사에서 다룬 blank-token case를 이번 workload의 관측으로 옮기지 않았다. 모든 warmup과 retained 응답은 strict exact prompt/output IDs·text·finish·usage 검증을 통과했다.

표의 P95/P99는 1,000개 요청 표본의 초기 tail 관측이다. 예를 들어 baseline/vLLM에서 baseline의 median TPOT는 느리지만 P95/P99는 짧다. 이 차이를 median TPOT 동등성이나 높은 concurrency 안정성으로 바꾸지 않는다. Offered C1만 완료했으므로 C2 이상 throughput·queueing·P95/P99 안정성은 미검증이다.

## 보존된 증빙과 다음 진단

- [V3 분석](token-round14-completed-cell-analysis.json), SHA256 `b191caaaa9b6515f7f208cff6eed0f1824968693d9f7d93d8d856af5ab7701a4`.
- [Finalization](raw/token-serving-round14/finalization.json), SHA256 `5678b0eda15fc057cf94d8999620c0ed5320936ba5f134a38c1447702ef7dd3e`.
- [중단 결과](raw/token-round14-c1-stop-outcome.json): C1 15개 pair 해시, pidfd SIGINT, C2 launch 0개.
- [Blender 복원](raw/blender-round14/verified.json), SHA256 `5a822ce22a4cbfad1feb768f844bc7c730031690e18e915310d37bf37a0f55c9`. PID **2694511 / 2694607 / 2694748**, birth ticks **84045877 / 84045926 / 84045977**, 포트 **9876 / 9911 / 9887**. 세 프로세스의 명령·GUI 환경·private GL readiness와 관측된 vendor mapping을 검증했다. 이 receipt에서 CUDA는 첫 번째 프로세스에 로드됐고 나머지는 아직 로드되지 않았다. 이 복구 기록은 이후 Round15 진단의 일시 중지 이전 시점이다.

Baseline의 두 enqueue를 하나의 구간으로 잰 결과와 Batch8 fused 구간을 AB/BA로 비교하는 별도 Round15 진단이 시작됐다. [진단 계약](PAIRED_ROPE_ATTENTION_PROFILE.md)은 off-before/after, 이벤트 교란, 512 replay 중 retained 321–512 경계를 명시한다. 이 문서를 작성한 시점에는 새 paired operator timing이나 원인 확정 결과가 없다. 과거 30-request HTTP/engine 표와 새 1,000-request HTTP token-delivery 결과는 서로 다른 계약으로 보존한다.
