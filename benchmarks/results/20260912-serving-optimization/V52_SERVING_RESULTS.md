# V52 serving 결과 — Round59

V52는 V51 대비 C32 fixed 처리량+4.13%,TPOT−5.29%,P99−7.98%를보였지만 C16fixed 처리량−2.52%,TPOT+3.17%다. Natural변화는작다. 두역순 screen이라작은차이의통계적확정은아니며전반적승격을하지않는다. V51은일반비교기준으로보존하고V52는후속attention실험후보로유지한다. 전체vLLM목표는미달성이다.

24lane×384retained=9,216요청실패0,Riley6,144응답기준일치. lane별출력토큰fixed7,680/natural28,672는모든엔진동일. 동일RTX4090/SmolLM2-135M BF16,admissionC16/C32,두Riley V7/GPU-greedy/budget512/fixedchunk128/natural512,각lane96warmup,2역순이다. 일부vLLM출력은Riley기준과달라cross-engine exact correctness를주장하지않는다.

| 조건 | V51 tok/s | V52 tok/s | vLLM tok/s | V52 TTFT ms | V52 TPOT ms | vLLM TPOT ms | V52 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 6531.431 | 6366.639 | 5114.779 | 6.174 | 2.266 | 2.212 | 83.481 | 120.712 |
| c16-natural | 7792.420 | 7744.402 | 7767.045 | 7.124 | 1.942 | 1.806 | 269.144 | 287.264 |
| c32-fixed | 7833.896 | 8157.152 | 7076.588 | 7.937 | 3.552 | 2.921 | 138.315 | 162.618 |
| c32-natural | 10246.643 | 10314.134 | 11575.512 | 10.484 | 2.924 | 2.368 | 411.015 | 365.958 |

V52/vLLM C16fixed처리량+24.48%,TPOT+2.47%; C32fixed처리량+15.27%,TPOT+21.59%. Natural처리량은C16−0.29%,C32−10.90%;TPOT는각+7.49%,+23.49%다. C32natural P99도12.31%높다. 특정throughput목표만으로combinedgoal을완료하지않는다.

Commit `06302d8d8396b8f2f4996fec8595bbfa1dcd7450`, binarySHA256 `d24783e7aab9e5ffbf7489e720a54122b967c5274131b385c3c0f405b3fbe07d`. [분석](raw/serving-round59/serving-round59-analysis.json), [manifest](raw/serving-round59-manifest.json), [구현및correctness](ATTENTION_STATE_V52.md). 202파일SHA256검증완료; archive `88e9447cda1662b3c75ded289df868aadea4d4ffdcf19491cc09d134f82356b3`.

## V52 별도 trace

288streaming응답기준일치. 35파일SHA256검증완료, [manifest](raw/profile-v52-manifest.json), archive `970ce776f7b07380115a5686c092ec811a33ba47b2b0cb09169ad7258a9a09c7`.

| Trace | prefill/mixed 비중 | decode 비중 | prefill/mixed median µs |
|---|---:|---:|---:|
| native-trace-v52 C16 | 30.12% | 69.88% | 4083.633 |
| native-trace-v52 C32 | 40.22% | 59.78% | 4250.483 |
| fixed-trace-v52 C32 | 75.55% | 24.45% | 3467.529 |

중간80%replay window의실제graph span합이다. C32fixed mapped attention평균22.512→19.256µs, C32natural55.261→52.307µs다. C16natural prefill/mixed graph median은4075.868→4083.633µs로감소하지않았다. Profiler CPU API/GPU시간은production latency를대신하지않는다. 그래프구성은scheduler에따라달라질수있다.

Sharedmemory상한개선만으로serving효과를설명할수없다. 작은query의현재warp는8개출력열그룹을직렬계산한다. 다음V53prototype은(1)V가중합출력열분할,(2)분할작업을같은CTA의독립warp에배치하여CTA dispatch를줄이는방식을비교한다. 중복QK/softmax비용이늘수있어실제측정전에는개선을가정하지않는다. V51/V52기준과vLLM모두보존하며Blender는종료상태를유지한다.
