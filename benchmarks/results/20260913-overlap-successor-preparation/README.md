# Successor descriptor preparation during predecessor GPU execution

C32 serving에서 직전 paired 대비 throughput **+4.27%**, TPOT P50 약 **−4.2%**를 관측했다. 두 순서의 throughput 변화는 **+3.83% / +4.70%**다. 같은 screen의 vLLM 대비 throughput은 **−3.38%**이며 최종 목표는 미달성이다. Default single을 유지한다.

## Optimization batch

1. Scheduler authority의 first/second wire 준비를 분리한다. 기존 두-step KV 예약과 immutable authority borrow를 유지하면서 첫 expectation만 먼저 만든다.
2. Runtime이 predecessor를 먼저 제출하고 retained predecessor ticket 상태를 보유한다. 이 상태에서는 일반 단일-iteration query/wait/commit, 새 issue, NotDispatched abandon을 허용하지 않는다.
3. 두 번째 descriptor·future-token reference의 준비/검증/encoding을 첫 제출 이후 수행한 뒤 successor를 같은 stream에 제출한다. 준비/검증/submit 실패는 이미 제출한 predecessor를 무시하지 않고 poison/retention 후 close로 처리한다. 기존 staged drain, first-stop suppression, 전체 pair settlement 이후 publication을 유지한다.

CPU 처리·encoding을 GPU 실행 중으로 옮긴 변경이며, scheduler가 현재 pair보다 더 먼 iteration을 예약하는 구현은 아니다. 수치·kernel·pair 길이·KV 예약량을 바꾸지 않았다. Runtime은 Rust→C ABI→CUDA이며 Python은 외부 측정/분석 도구다.

## C32 serving

RTX4090, SmolLM2-135M BF16, 동일 frozen natural workload/checkpoint/tokenizer. 입력16/128/398, 출력32/64/128, active32, waiting64. Lane마다 fresh server와 warmup192+retained768. Single→직전 paired→후보→vLLM 및 역순. GUI 유지·Blender 종료, Nsight와 host phase timing을 끄고 측정했다. Throughput/P50은 두 반복의 중앙값, P95/P99는 더 나쁜 반복이다.

| 경로 | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| single | 10,479.90 | 10.338 | 2.858 | 389.494 | 472.501 |
| 직전 staged paired | 10,356.97 | 10.961 | 2.819 | 397.821 | 497.830 |
| 준비 overlap paired | 10,798.86 | 11.058 | 2.701 | 378.598 | 483.806 |
| vLLM 0.27.1 | 11,176.24 | 20.989 | 2.403 | 371.697 | 468.225 |

[반복별 결과](completion.json), [전체 P50/P95/P99](comparison.json), [환경과 hash](evidence/overlap-prepare-c32-v1/preparation.json). Candidate는 single보다 throughput 약3.04% 높지만 TTFT P50 및 E2E P99는 더 높다. vLLM의 이번 수치를 다른 screen과 이어 붙여 개선율을 계산하지 않는다. Shared-host exploratory 측정이며 CPU isolation/clock lock/장시간 soak/SLO qualification이 아니다. KV capacity와 BF16 연산 경로의 완전한 동일성이나 cross-engine bitwise 일치를 주장하지 않는다.

## Client C64 / active32 추가 screen

Client concurrency64, 양쪽 engine active capacity32. Workload·warmup192·retained768·두 순서는 C32와 동일하다. 첫 시도는 controller가 concurrency와 active capacity를 같이64로 변경해 readiness 전에 Riley가 unsupported capacity를 거부했다. [실패 로그](evidence/overlap-prepare-c64-v1/controller.log)를 보존한다. `--active-capacity`를 추가하고 `--concurrency 64 --active-capacity 32`로 새 v2를 실행했다. 실제 모든 lane의 argv 및 preparation metadata에서 active32를 확인했다. 실패를 runtime 성능이나 hardware skip으로 계산하지 않는다.

| 경로 | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| single | 10,589.12 | 225.559 | 2.881 | 637.697 | 695.640 |
| 직전 staged paired | 10,744.33 | 222.972 | 2.822 | 629.010 | 720.480 |
| 준비 overlap paired | 10,820.91 | 219.466 | 2.755 | 610.711 | 709.741 |
| vLLM 0.27.1 | 11,752.71 | 203.161 | 2.521 | 589.446 | 702.067 |

[전체 C64 P50/P95/P99](comparison-c64.json), [실행 조건](evidence/overlap-prepare-c64-v2/preparation.json). Candidate/직전 처리량 변화는+0.71%(각 순서+2.32%/−0.87%)로 일정하지 않다. vLLM 대비−7.93%이며 TTFT/TPOT도 더 높다. 따라서 C32 이득을 높은 부하 전반의 개선으로 일반화하지 않는다. C64 retained6,144개 protocol 오류0, Riley warmup+retained5,760개 exact, vLLM retained exact는1134/1,536이다. 모든 최종 lane exit0, 측정 후 GPU compute process없음. [추가 raw manifest](evidence/c64-raw-manifest.json)의 원본 복사39개를 SHA로 대조했다.

## 실제 preparation overlap

별도 Nsight capture에서 두 lane 각96 HTTP 요청이 reference exact이며 정상 종료했다. [Host/GPU 분석](evidence/overlap-prepare-trace-v1/host-overlap.json)은 launch 개수 기준 중간80%에서 양쪽 이웃이 있는88개 pair를 선택했다. 첫 cudaGraphLaunch 반환부터 두 번째 launch 시작까지 host 구간 중앙값은274.632µs이며, predecessor device activity union과 겹친 부분은271.239µs(합23.066ms)다. 이 구간에는 descriptor 준비 외에 native host overhead도 포함되며 pure preparation 비용으로 단정하지 않는다.

첫/두 번째 event wait 사이에는 기존 token/stop 처리 구간이 있고 device overlap 중앙값64.003µs다. [Graph 경계](evidence/overlap-prepare-trace-v1/paired-analysis.json)의 after-pair gap 중앙값은380.299µs로 남아 있다. Trace에는 profiler/client pacing이 있고 batch 구성도 달라질 수 있으므로 이 값을 unprofiled 개선량이나 제거 가능한 scheduler 비용으로 바꾸지 않는다.

## 검증과 증거

- CUDA release build11.36s, 새 상태를 포함한 runtime mock 테스트6개, scheduler decode-window CPU 테스트6개 통과. 최초 mock fixture의 aggregate stage 오류는 실제 실패 로그를 보존하고 Decode로 수정한 뒤 재검증했다. 실패를 skip으로 바꾸지 않았다.
- 실제 HTTP smoke에서 single/후보 각 warmup96+retained96 reference exact, stop-string 각96개 token/text/finish/usage 동일, disconnect 각32개 후 retained reference exact.
- C32 retained6,144개 HTTP/SSE 오류0. Riley warmup+retained5,760개 exact. vLLM retained의 Riley reference exact는1,133/1,536이다. 모든 server exit0. 품질 qualification이나 전체 하드웨어 검증 완료를 주장하지 않는다.
- Host timing 집계 테스트3개, host/GPU overlap 분석 테스트4개 통과. Native memory-access/kernel 변경이 없어 whole-model sanitizer를 재실행하지 않았다.
- [원격 소스 SHA](evidence/source-hashes.json)는 로컬 코드와 대조했다. 측정 후보 binary는 `a5036ff769a960d48dbd96fc971b6a5a3899c32c267c86da72137df76c0f0eaf`, 직전은 `9390800555e7f7abfb65fb4b207ad619bc868364dd13d34ccc374c223830317a`다. Smoke/serving/trace candidate hash가 일치한다.
- [Raw manifest](evidence/raw-manifest.json)에 persistent remote 원본 SHA·크기·경로가 있다. Frames를 제외한 request token/text/usage/checks/timestamp는 gzip으로 보관한다. Full frames 및 Nsight/SQLite는 `ai-assistant:/data/riley-serving-260913-recovery`에 남아 있다.

## 남은 범위

Candidate는 opt-in으로 유지한다. 다음 pair의 scheduler/descriptor 준비, admission·cancel·stop과의 의존성 분리, C64를 넘어서는 workload/arrival-rate 검증과 장시간 안정성, Hopper/Blackwell/multi-GPU 검증이 남아 있다. 4090 지원 범위의 실제 실패를 hardware skip으로 바꾸지 않는다. PR02 전체 완료나 성능 기본값 승격은 아직 아니다.
