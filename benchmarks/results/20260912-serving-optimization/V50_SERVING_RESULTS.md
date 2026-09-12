# V50 serving 결과 — Round57

V50은 V49 대비 C16 처리량3.45~3.94%를 개선했지만 C32 개선은0.24~1.19%다. C32 fixed P99는2.62% 악화했다. 두 역순의 짧은 screen이므로 작은 C32 차이를 유의한 개선이나 안정성 입증으로 간주하지 않는다. vLLM 대비 TPOT 목표는 모든 조건에서 미달성이다.

24lane ×384retained=9,216 요청 실패0, Riley6,144응답 기준 일치. lane별 출력 토큰 수 fixed7,680/natural28,672는 세 엔진 모두 같다. 일부 vLLM 출력은 Riley 기준과 달라 cross-engine exact correctness는 주장하지 않는다. 동일 RTX4090/SmolLM2-135M BF16, C16/C32 admission, tokenbudget512, fixedchunk128/natural512, 두 Riley 모두V7 compact GPU-greedy다. 각lane96warmup, 두 역순 실행이다.

| 조건 | V49 tok/s | V50 tok/s | vLLM tok/s | V50 TTFT ms | V50 TPOT ms | vLLM TPOT ms | V50 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 5701.884 | 5898.811 | 5179.829 | 6.766 | 2.446 | 2.256 | 92.093 | 109.760 |
| c16-natural | 7247.970 | 7533.620 | 7851.917 | 7.882 | 1.993 | 1.793 | 272.306 | 276.363 |
| c32-fixed | 7073.438 | 7090.713 | 7319.121 | 8.878 | 4.179 | 2.730 | 162.615 | 151.692 |
| c32-natural | 9675.419 | 9790.383 | 11997.057 | 11.750 | 3.070 | 2.274 | 449.072 | 355.267 |

V50/vLLM 처리량은 C16fixed+13.88%,C16natural−4.05%,C32fixed−3.12%,C32natural−18.39%다. TPOT는 각각8.44%,11.13%,53.05%,35.03% 느리다. TTFT는 vLLM보다 낮지만 이것만으로 목표를 충족하지 않는다. V50을 다음 병목 분석 후보로 유지하며 기본profile 승격은 하지 않는다.

이번 변경은 QK에서3query head를 서로 다른 MMA행에 묶고, values에서3head를 같은CTA의 독립warp로 배치한 batch다. 값 공유MMA variant는 낮은 rows/긴context에서 손해가 있어 채택하지 않았다. Primitive 시간감소4.86~38.51%가 실제 serving 개선률을 대신하지 않는다.

Commit `44ccb8d0535e80e2f4b7fd09b84b770848ad4678`, frozen binary SHA256 `c618e2db269a1aa0e624f7d84036ee97b8ff8b45f82c6d58ebb6894e5b8a97c8`. [분석](raw/serving-round57/serving-round57-analysis.json), [원본 manifest](raw/serving-round57-manifest.json), [구현 및 correctness](GQA_ATTENTION_V50.md). 원본202파일 SHA256 검증 완료; archive `c05c72bb86c404b29eeca331578be553ec2c4e0746167acb141f9eb6a658fdfd`.

별도 V50 natural C16/C32 및 fixed C32 Nsight trace를 수집 중이다. CPU CUDA API 시간은 profiler overhead를 포함한다. Pure prefill/mixed graph는 trace에서 구분할 수 없어 함께 분석한다. 다음 batch는 새로운 profile에서 확인한 비용을 기준으로 결정한다. Blender는 종료 상태를 유지한다.


## V50 profile 완료

별도 trace288개 streaming 응답 모두 기준 일치. 35파일 SHA256 검증 완료. [Manifest](raw/profile-v50-manifest.json), archive `eca58fcd3baea2f1b184f25806496f89e7ca505495b0b5c4653bddfdbd472ab2`.

| Trace | prefill/mixed 비중 | decode 비중 | decode graph 중앙값 µs |
|---|---:|---:|---:|
| native-trace-v50 C16 | 34.77% | 65.23% | 1440.141 |
| native-trace-v50 C32 | 45.41% | 54.59% | 1635.582 |
| fixed-trace-v50 C32 | 77.33% | 22.67% | 1353.996 |

중간80% replay window의 실제 graph span 합을 사용했다. V49/V50 C32natural pure decode median은1717.073→1635.583µs이며 values평균10.918→9.714µs, scores7.349→5.575µs다. Trace는 진단용으로 실행되었으며 profiler GPU/CPU 시간이 production latency를 대신하지 않는다. 동일 workload라도 scheduler가 만든 group 구성은 다를 수 있다.

다음 batch는 prefill/mixed projection을 대상으로 한다. C32fixed graph span의77.33%가 이 경로다. Gate/up kernel 두 번은 각각 평균17.509µs, down은19.147µs이며30layer에 반복된다. (1) gate/up projection과 SwiGLU를 fusion해 입력 읽기 및 중간 BF16 buffer traffic과 launch를 줄이고, (2) 두M16 row tile에서 동일 weight fragment를 재사용하는 projection을 비교한다. 각 K16 MMA 및 BF16 chunk-round 순서를 유지한다. 기존 V27의 packed weights/unroll4는 이미 적용되어 있어 재구현하지 않는다. 커널/실제 모델/serving 검증이 뒤따르며, 고동시성·장기안정성 목표는 여전히 미검증이다.
