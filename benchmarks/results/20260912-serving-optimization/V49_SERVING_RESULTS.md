# V49 matched serving 결과 — Round56

V49 mixed prefill/decode는 V48 대비 네 workload 모두 처리량과 TPOT를 개선했다. vLLM 대비 TPOT와 natural 처리량 차이가 남아 목표는 미달성이다. V49는 다음 개선 비교 후보로 유지하며 기본 profile로 승격하지 않는다.

동일 RTX 4090, SmolLM2-135M BF16 checkpoint, admission C16/C32, iteration budget512, fixed chunk128/natural chunk512 조건이다. V48/V49 모두 width32 compact GPU greedy다. 각 조건은 두 역순 실행이며 각 lane warmup96, retained384 요청이다. 총24 lane,9,216 요청 실패0, Riley6,144 응답 기준 일치. 모든 엔진의 lane당 출력 토큰 수는 fixed7,680, natural28,672로 같다. vLLM은 일부 출력이 Riley 기준과 달라 cross-engine exact correctness를 주장하지 않는다.

| 조건 | V48 tok/s | V49 tok/s | vLLM tok/s | V49 TTFT ms | V49 TPOT ms | vLLM TPOT ms | V49 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 5351.026 | 5590.627 | 4907.470 | 7.190 | 2.579 | 2.323 | 95.395 | 128.623 |
| c16-natural | 6947.214 | 7116.116 | 7870.572 | 8.099 | 2.109 | 1.780 | 291.106 | 287.545 |
| c32-fixed | 6503.886 | 7157.490 | 7183.684 | 9.375 | 4.027 | 2.947 | 161.992 | 172.137 |
| c32-natural | 9038.088 | 9568.017 | 11609.212 | 11.731 | 3.129 | 2.370 | 456.934 | 374.768 |

표는 두 실행 통계의 중앙값이다. 장기 안정성이나 P99 우월성의 최종 증명이 아니다. V48 대비 처리량 +2.43~10.05%, TPOT −2.93~11.43%지만 TTFT는 C16 natural +5.37%, C32 fixed +10.37%, C32 natural +19.61% 악화했다. vLLM 대비 C16 fixed 처리량 +13.92%, C32 fixed −0.36%, natural −9.59~17.58%; TPOT는 모든 조건에서 +11.02~36.62% 느리다.

V49 commit `c63a395cdbaea050b8281aa58bfb0f9ac681bb14`, binary SHA256 `3e822d5171511b95a8be77bbd104b3dc7b4ef52395a5e0f39a449fe4ea3a60e9`. Benchmark 원본202개 파일은 manifest SHA256으로 로컬 검증했다. Archive SHA256 `8bea5c2e2f195a9a85615ade7d1dd154fc5bbb3b0be33cdc5302549f95f7befc`.

[분석](raw/serving-round56/serving-round56-analysis.json), [원본 manifest](raw/serving-round56-manifest.json).

별도 Nsight natural C16/C32 및 fixed C32 trace는288 응답 모두 기준 일치로 완료했다. Natural C32 pure decode의 values 평균10.918µs, scores7.349µs가 각 layer마다 실행된다. Non-decode graph의 mapped attention은55.113µs다. Trace는 packet payload를 보존하지 않아 pure prefill과 mixed iteration을 구별할 수 없으므로 함께 분석한다. CPU CUDA API 시간은 profiler overhead를 포함하여 serving latency로 해석하지 않는다.

다음 후보는 decode attention의 GQA 그룹 단위 QK 계산과 공유 V 읽기/가중합이다. 실제 profile에서 확인한 중복 비용을 대상으로 두 개선을 함께 검증하되, 수치 순서와 full-logit 기준 일치를 유지하고 serving 재측정 전에는 채택하지 않는다. Fixed workload에서는 prefill/mixed projection 비용도 크므로 이 후보만으로 전체 격차 해소를 가정하지 않는다.

Blender는 사용자 지시에 따라 종료 상태를 유지한다. 복구 작업은 실행하지 않는다.


Profile 원본35개 파일 SHA256 검증 완료. Archive `c9a2a4c73272244fefccaed1ea59b4ba57ea4bc7599f7bc04f077deec9615d1c`. [Manifest](raw/profile-v49-manifest.json).

| Trace | prefill/mixed graph span 비중 | pure decode graph span 비중 |
|---|---:|---:|
| native-trace-v49 C16 | 30.62% | 69.38% |
| native-trace-v49 C32 | 41.84% | 58.16% |
| fixed-trace-v49 C32 | 79.16% | 20.84% |

중간80% replay window의 실제 graph span 합으로 계산했다.
