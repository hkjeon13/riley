# V55 serving 결과 — packed V 계열 종료

V55의 full K16 fast path는 V54의 decode 손실을 대부분 회복했지만, V52 대비 실제 serving throughput의 실질적 개선은 없었다. Packed V 계열 V54/V55를 채택하지 않고 token-major V로 복귀했다. 최종 목표는 아직 미달이다.

동일 RTX4090/SmolLM2-135M BF16, admission C16/C32, token budget512, fixed chunk128/natural chunk512 조건에서 V51/V52/V54/V55/vLLM0.27.1을 두 역순으로 비교했다. 40 lanes 각각 warmup96/retained384, 총 15,360 요청 실패0, Riley reference12,288개 모두 일치. 엔진별 생성 token 총량은 fixed7,680/natural28,672로 동일했다. vLLM의 일부 출력은 Riley reference와 달라 전체 cross-engine byte 일치는 주장하지 않는다. 두 번의 screen이며 장시간 안정성이나 통계적 우위 검증이 아니다.

| Case | V51 tok/s | V52 tok/s | V54 tok/s | V55 tok/s | vLLM tok/s | V55 TPOT ms | vLLM TPOT ms | V55 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 6226.756 | 6303.550 | 6275.119 | 6249.496 | 5123.142 | 2.298 | 2.276 | 86.530 | 107.434 |
| c16-natural | 7788.615 | 7796.246 | 7322.373 | 7695.035 | 7946.543 | 1.957 | 1.781 | 268.015 | 270.953 |
| c32-fixed | 8082.325 | 8267.771 | 8297.388 | 8285.851 | 7301.662 | 3.412 | 2.902 | 143.890 | 164.384 |
| c32-natural | 10319.962 | 10394.343 | 10104.390 | 10379.325 | 11973.606 | 2.893 | 2.280 | 423.267 | 361.195 |

V55 vs V52 throughput은 위 순서로 −0.86%, −1.30%, +0.22%, −0.14%다. V54 대비 natural throughput은 C16 +5.09%, C32 +2.72%로 회복했다. 그러나 vLLM 대비 TPOT는 모든 조건에서 0.96–26.86% 높고, C32 natural throughput은 13.31% 낮으며 P99는 17.19% 높았다. TTFT 이득만으로 목표 달성을 선언하지 않는다.

## Profiling과 결정

V55 C32 natural decode independent_values 평균10.876µs는 V54 15.267µs보다 낮지만 V52 9.730µs보다 높다. Mixed attention은 V55 44.641µs, V54 45.015µs, V52 52.307µs다. KV writer는 V52/V54에서 mixed9.016/9.127µs, decode1.529/1.561µs였으므로 큰 쓰기 비용 증가가 관찰된 것은 아니다. Prefill/mixed 집합과 active rows/context 구성이 변하므로 단계별 평균만으로 serving 향상을 계산하지 않는다.

V55 trace288개 요청은 기준 출력이 모두 일치했다. Middle80% replay window에서 pure prefill과 mixed를 `prefill_or_mixed`로 합쳐 분석했다. Profiler runtime API 시간은 production latency가 아니다.

V55 gate/up/SwiGLU는 C32 natural decode 평균8.312µs로 남은 주요 비용이다. V56 primitive는 입력 재사용, 중간 shared memory/CTA barrier 제거, load pipeline과 active-row 계산 범위 조정을 함께 비교한다. 아직 V56 application 통합이나 성능 채택은 없다.

Packed V 변경을 revert한 commit은 `badd99adb9ed6263d78a3601917553fd13b1ff78`이다. Tree `938eb454f1406f7d84de72f2425c1077b24c2522`가 V52 commit `06302d8d8396b8f2f4996fec8595bbfa1dcd7450`의 tree와 정확히 같음을 확인했다. V51 일반 기준과 V52 frozen 기준, V54/V55 실험 binary 및 raw evidence는 그대로 보존했다. 이는 V52의 모든 serving 조건에서 우위를 선언한 것이 아니다.

## 검증 및 artifact

V55 commit `a33cdc8488122eebefad090cc493dec370193e49`, binary SHA256 `897c6ec9118feecb4121aef82e9aeb117a50a09a18f9b54ac7076ab10bab0a81`. Model16 regressions, model memcheck3, HTTP37, CPU/GPU fallback22개 응답 일치. Primitive2,688 cases 및 memcheck/racecheck 오류0.

- `raw/mask-v55`: 51파일, archive SHA256 `052569307142c58242c018f492778530a2d759c33eec3656ce207f4ebecdefd2`.
- `raw/integration-v55`: 31파일, archive SHA256 `b297b4354250f63cf6606d0158918ab7e5762c23e64d562615ca15ba33005584`.
- `raw/serving-round61`: 332파일, archive SHA256 `799cc11bd5cf993b5fd1f3495541034bee7536e671b1f9e35e04f3c13d197033`.
- `raw/profile-v55`: 35파일, archive SHA256 `90a319b66f06f38a1fc46cc425133ff6eae2edff0a00f51e29e72b31d496adc3`.

각 파일 size/SHA256를 local에서 검증했다. Round61과 profiling controller는 정상 종료했다. Blender는 종료 상태 유지, 기존 PID/port 부재도 exporter가 재확인했다.
