# Opt-in mixed execution wall-time feedback policy

PR04의 첫 runtime scheduling 후보를 구현했다. 고정 chunk를 바꾸는 대신 성공한 mixed 실행의 wall-time을 기준으로 prefill token 예산을 조절한다. **하드 SLO 보장이나 POD attention 구현이 아니다.** 기존 기본값과 graph/kernel은 유지한다. **C32에서 직전 paired 대비 throughput−8.85% 및 모든 주요 latency 회귀로 후보를 승격하지 않는다.**

## Optimization batch

1. Scheduler에 allocation 없는8개 workload class controller를 추가했다. 최대 active prompt+generated 길이를 ≤128/256/512/>512로 나누고 decode 존재 여부를 결합한다. Shape의 최대 context만 사용하는 coarse class이며 calibration histogram의 상세 비용 predictor를 구현한 것은 아니다. Cold class는 configured iteration budget을 유지한다.
2. 실제 prefill tokens가 현재 제한−64 이상인 성공 샘플8개를 모아 평균 execute wall을 계산한다. 목표의120% 초과면64 감소,80% 미만이면64 증가, 나머지는 유지한다. Controller limit은128..configured budget이고 실제 prefill은 decode를 먼저 배정한 나머지 token budget 및 기존 per-request chunk cap으로 다시 제한된다. Unsaturated/0-time 샘플은 학습하지 않는다. 목표보다 늦은 실행이 생길 수 있다.
3. Runtime authority 준비·전송·GPU 대기·download/validation 구간을 host `Instant`로 측정한다. Scheduler 결과 검증과 성공 settlement 이후에만 관측을 반영한다. 잘못된 결과·abort·실패 settlement와 pure/paired decode는 학습하지 않는다. 변경된 plan도 기존 KV reservation 및 descriptor 검증을 통과한다. Admission·기존 ready-time 순서·pair settlement/publication은 유지한다.
4. `--mixed-time-budget-us 1000..100000` loopback V7 opt-in을 추가했다. 신규 타이머와 workload class 계산은 policy disabled 시 실행하지 않는다. Shutdown에 class별 최종 limit/조정 수를 남긴다. 동일 FFN paired를 직전 기준으로 고정한 vLLM 포함 serving 비교를 지원한다.

Runtime은 Rust→C ABI→CUDA이며 Python은 외부 benchmark 도구다. 새로운 CUDA kernel/workspace나 하드웨어별 연산을 추가하지 않았다. Multi-GPU/Hopper/Blackwell runtime의 검증 완료를 주장하지 않는다.

## 검증

- Scheduler CPU48개 통과. 새 controller 테스트는 cold/unsaturated 클래스 격리·8샘플 문턱·deadband·하한/상한·회복·context 경계를 검증한다. 실제 scheduler test는 예산128에서도 decode 전부와 prefill 진행, 잘못된 result 및 NotDispatched abort의 학습 금지, 완료 후 KV 회수를 확인한다.
- CLI34개, runtime config7개 통과. CUDA release build11.42초 통과. [로그](evidence/).
- Smoke는 single/후보 각각 warmup96+retained96 reference exact, stop96개 token/text/finish/usage 동일, disconnect32개 뒤 retained reference exact다. 후보가 실제로 class limit을448/320으로 조정하고248개 pair를 실행했다. 단순히 flag parsing만 검증한 결과가 아니다.
- 진단/calibration 자료는 [직전 chunk screen](../20260914-mixed-chunk-calibration/README.md)에 있다. 4ms target은 그 mixed 실행 평균 약4–5ms를 바탕으로 미리 정한 첫 실험값이다. Shared-host wall time은 pure GPU 비용이 아니며 고정4ms 하드 latency 상한을 뜻하지 않는다.

## 비교 조건

RTX4090, SmolLM2-135M BF16, frozen natural16/128/398 input·32/64/128 output, client C32/active32, graph capacity 및 total token budget512. 각 lane warmup192+retained768. Single→직전 FFN paired→시간 정책 FFN paired→vLLM 및 역순. GUI 유지·Blender 종료, profiler/phase diagnostics 없음. Candidate는 필요한 wall stopwatch만 켠다. 직전 binary에도 같은 prefill FFN 옵션을 켜서 이전 FFN 이득을 정책 효과로 계산하지 않는다.

Throughput과 P50은 두 반복 중앙값, P95/P99는 더 나쁜 반복이다. CPU isolation·clock lock·open-loop SLO·장기 soak가 없는 exploratory closed-loop screen이다.

## C32 결과와 판정

| 경로 | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| single | 10,444.10 | 10.677 | 2.857 | 395.785 | 505.058 |
| 직전 FFN paired | 10,907.84 | 10.630 | 2.664 | 377.657 | 468.548 |
| 시간 정책 FFN paired | 9,943.00 | 11.890 | 2.997 | 418.640 | 515.484 |
| vLLM 0.27.1 | 11,501.83 | 20.351 | 2.328 | 353.177 | 443.488 |

[전체 percentile 및 변화율](comparison-c32.json), [반복별 결과](evidence/mixed-time-c32-v1/completion.json). 후보/직전 throughput 변화는−8.85%, 두 순서−7.73%/−9.95%다. vLLM 대비−13.55%이며 vLLM 초과 목표는 미달성이다. 후보/직전 TTFT·TPOT·E2E P95/P99도 모두 나빠 승격하지 않는다. 기본 옵션은 바뀌지 않았고 정책을 생략하면 이전 선택 규칙을 사용한다.

Retained6,144개 HTTP/SSE 오류0, Riley warmup+retained5,760개 reference exact, 모든 lane 정상 종료다. vLLM retained의 Riley reference exact는 1,150/1,536이다. Candidate와 prior argv에서 prefill FFN·paired 옵션이 같고 time budget은 candidate에만 있음을 대조했다. [원본 SHA manifest](evidence/raw-manifest.json), [소스 대조](evidence/source-hashes.json). 원격 full frames는 `ai-assistant:/data/riley-serving-260913-recovery/mixed-time-c32-v1`에 보존한다. 로컬 gzip은 frames만 제외했다. Build 이후 추가된 config unit test는 production 동작을 바꾸지 않는다.

두 후보 반복 모두 context≤512+decode class는 최종256/조정6회, context>512+decode class는256/조정4회였다. Feedback은 실제로 적용됐다. 작은 batch의 per-iteration 시간을 목표로 삼으면 반복 횟수 증가, fixed per-iteration overhead 및 처리량 손실을 충분히 반영하지 못한다. 직전 fixed-chunk 회귀와 일관되지만 추가 profile 없이 이번 회귀 전체를 특정 한 원인으로 단정하지 않는다.

현재 coarse class/반응형 controller는 실험 옵션으로 비활성 유지한다. 4ms 목표를 사후 반복 튜닝하거나 C32 실패를 hardware skip으로 바꾸지 않는다. 기본 후보부터 회귀했으므로 이번 후보의 C64/장기 soak 승격 검증은 진행하지 않았다. 이는 미실행이며 통과가 아니다. PR04의 POD resource-sharing 및 detailed cost/predictor·starvation SLO/admission policy 전체 구현은 남아 있다. 다음 배치는 host feedback의 미세 threshold 조정 대신 실제 prefill/decode attention 자원 공유와 지원·수치 계약을 다루며, 그 구현 이후 frozen workload와 긴 prompt/open-loop 평가를 함께 한다.
