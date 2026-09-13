# Staged pair drain — actual host/GPU overlap, no stable serving gain

단계별 drain을 연결하여 첫 결과의 검증·token/stop 처리가 두 번째 graph의 실행과 실제로 겹치도록 했다. 같은 serving 비교에서 직전 paired 대비 throughput 변화는 **+0.58%**, 두 순서는 **+1.65% / −0.49%**다. 안정적인 성능 개선은 입증하지 못했다. vLLM 대비 **−7.39%**이며 default는 single을 유지한다. 이 날짜의 vLLM 결과를 이전 screen과 이어 붙여 개선율을 계산하지 않는다.

## 하나의 optimization batch

1. Runtime `wait_decode_window_first/second`로 drain을 분리한다. 첫 event 완료 후 첫 record를 검증하고, GPU token으로 host successor expectation을 확정한다. 첫 read 이후에도 retained owner와 successor를 보유하며 issue·commit을 거부한다. 두 번째 record의 검증까지 끝나야 pair commit 확인이 가능하다. 상태/검증 오류는 기존 poison/close 경로를 따른다.
2. Scheduler adapter에 host-only predecessor callback을 추가한다. Authority와 session borrow는 두 drain 동안 유지한다. Callback 반환값이 오류여도 successor drain을 수행한 다음 caller에 돌려준다. 기존 execute 함수는 compatibility wrapper로 유지한다.
3. Server는 callback 안에서 첫 token을 처리하고 stop suppression을 준비한다. 두 번째 GPU 결과 검증 후에 기존 pair settlement·publication을 수행한다. 첫 token을 먼저 외부 공개하지 않는다. Callback 실행 동안 scheduler와 graph를 local owner로 유지하고 정상/오류 반환 전에 server 필드로 복원한다. Pair 길이, GPU 연산, KV 예약 및 수치 계약은 바꾸지 않는다.

Runtime 경로는 Rust→C ABI→CUDA다. Python은 offline client·분석 도구다.

## Serving 비교

RTX 4090, SmolLM2-135M BF16, C32, 동일 frozen natural workload 및 checkpoint/tokenizer. 입력 16/128/398, 출력 32/64/128. Lane마다 fresh server, warmup192 + retained768. 순서 single→직전 paired→신규 paired→vLLM과 그 역순이다. GUI 유지·Blender 종료, host phase timing 및 Nsight를 끄고 측정했다. Throughput/P50은 두 반복의 중앙값, P95/P99는 더 나쁜 반복이다.

| 경로 | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| single | 10,463.73 | 10.626 | 2.850 | 395.538 | 492.190 |
| 직전 paired | 10,413.84 | 10.883 | 2.825 | 389.040 | 474.183 |
| staged paired | 10,473.76 | 10.841 | 2.804 | 387.946 | 494.428 |
| vLLM 0.27.1 | 11,309.35 | 20.820 | 2.316 | 370.623 | 520.728 |

[전체 P50/P95/P99](comparison.json), [반복별 결과](completion.json), [정확한 환경·hash](evidence/staged-drain-c32-v1/preparation.json). New paired는 single과 사실상 동률이다. TPOT P50은 직전보다 약 0.77% 낮지만 E2E P99는 증가했다. 이 짧은 shared-host screen만으로 tail 변화의 인과관계나 통계적 유의성을 확정하지 않는다. KV capacity와 engine별 BF16 수치 경로가 같다는 주장은 하지 않는다.

## 실제 중첩 증거

별도의 Nsight graph-node capture에서 single·paired 각 96 HTTP 요청이 reference exact, exit0, 잔여 소유 process0으로 완료됐다. Peak owned-tree RSS는 각각 1,496,252,416 / 1,521,807,360 bytes다. 8 GiB 감시는 sampled watchdog이며 cgroup hard limit이 아니다.

[Host overlap 분석](evidence/staged-drain-trace-v1/host-overlap.json)은 launch 개수 기준 중간80%에서 양쪽 이웃이 있는 83개 pair를 선택한다. 두 event wait 사이 host 구간 중앙값은 **81.771µs**이고, successor의 kernel/copy/memset interval union과 겹친 부분은 중앙값 **61.640µs**, 합 **4.618ms**다. 83개 모두 양의 overlap이 있다. Wait는 같은 CPU thread, pair의 두 launch와 다음 launch 사이에서 정확히 두 개를 요구하고 각 GPU envelope 완료를 확인한다. Host 구간에는 결과 read/validation·token/stop 처리 외에 다른 host overhead도 포함된다. Callback만의 실행 시간이나 unprofiled 절감량은 아니다.

[경계 분석](evidence/staged-drain-trace-v1/paired-analysis.json): within-pair gap 중앙값6.112µs, after-pair600.565µs다. 직전 trace after-pair716.528µs보다 작지만 두 trace의 batch 구성·profiler/client pacing이 완전히 같지 않으므로 그 차이를 순수 개선량으로 사용하지 않는다. 이번에 겹친 작업은 작은 일부이며 다음 작업 준비와 결과 처리 경계의 대부분은 여전히 남는다. 이는 serving 이득이 작고 반복별 부호도 달라진 결과와 일관되지만 단독 인과 증명은 아니다.

## 검증과 증거 범위

- CUDA release build 통과(11.33s). 실제 server 변경을 빌드했다. 이후 추가한 runtime `cfg(test)` 모듈·test-only fixture visibility 변경은 별도 CUDA-feature library test build에서 통과했다. 측정 binary는 이 test-only 추가 전 빌드이며 production 코드 변경은 동일하다.
- 새 runtime 상태 전이 mock 테스트3개 통과: 첫 read 후 successor wait 미실행/issue·commit·중복 read 거부, 정상 drain 후 commit, predecessor/successor 오염 시 poison·retention. 이 테스트를 GPU completion 자체의 증거로 대신하지 않는다.
- HTTP smoke: single/신규 paired 각 warmup96 + retained96 reference exact. Stop-string 각96개 token/text/finish/usage 동일. 연결 취소 각32개 이후 retained reference exact.
- Unprofiled8개 lane retained6,144개에서 HTTP/SSE 오류0. Riley6개 lane warmup+retained5,760개 reference exact. vLLM retained reference exact는1,137/1,536으로 cross-engine bitwise 일치나 모델 품질 qualification을 주장하지 않는다. 모든 server exit0. 측정/trace 후 GPU compute process가 없는 것을 확인했다.
- Host timing 집계 테스트2개, overlap 분석 테스트3개 통과. 기존 gap/paired trace 분석 테스트8개도 통과했다. 새로운 native kernel이나 memory-access 변경이 없어 whole-model sanitizer를 재실행하지 않았다.
- [소스 SHA](source-hashes.json)4개 파일은 원격 readback과 일치한다. 이전 binary SHA는 `b24a67903c86afd10b8bf7b4be4b1d05343fbd421733d76ae8c5d3eeaa151241`, 후보는 `9390800555e7f7abfb65fb4b207ad619bc868364dd13d34ccc374c223830317a`다. Smoke·serving·trace 후보 hash가 일치한다.
- [Raw manifest](evidence/raw-manifest.json)에 persistent remote 원본 경로·SHA·크기를 보존한다. 큰 request JSON은 frames를 제외한 token/text/usage/checks/timestamp를 gzip으로 보관했고 원본 full frames 및 Nsight/SQLite는 `ai-assistant:/data/riley-serving-260913-recovery`에 남아 있다. 압축하지 않은 보관 원본54개는 SHA를 대조했다.

## 결정과 다음 영역

Staged drain은 opt-in paired의 실제 overlap 경계로 유지한다. Default single 유지, PR02 전체 완료·성능 승격·vLLM 동등/우위는 미달성이다. C16/더 높은 concurrency/장시간 안정성/Hopper/Blackwell/multi-GPU는 이번 batch에서 검증하지 않았다.

다음은 token 처리 부분의 추가 미세 튜닝이 아니라 **다음 descriptor·scheduler 준비를 active GPU 실행 중에 수행할 수 있는 예약 구조**다. 현재 pair authority와 단일 in-flight scheduler의 의존성을 먼저 나누어야 한다. Stop/cancel/새 admission에 영향을 받지 않는 준비와 commit에 의존하는 확정을 분리하고, generation/page/slot 수명을 유지한 제출 경계를 함께 구현·검증한다. Pair 길이만 늘리거나 미완료 결과를 공개하여 처리량을 높이는 방향은 이번 근거로 선택하지 않는다.
