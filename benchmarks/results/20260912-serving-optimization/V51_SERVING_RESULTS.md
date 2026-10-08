# V51 serving 결과 — Round58

V51 fusion은 V50 대비 네 workload 모두 throughput·TTFT·TPOT·P95·P99를 개선했다. 다음 optimization 기준으로 V51을 유지한다. C16fixed에서는 vLLM 대비 처리량+33.81%,TPOT−5.18%지만 다른조건의TPOT와natural throughput격차가남아 전체목표는미달성이다. 두역순 screen을장기안정성이나통계적으로확정된우월성으로확대해석하지않는다.

동일 RTX4090/SmolLM2-135M BF16, C16/C32 admission, 두Riley V7/GPU-greedy/tokenbudget512/fixedchunk128/natural512. 24lane×384retained=9,216요청 실패0, Riley6,144응답 기준일치. lane별출력토큰fixed7,680/natural28,672는세엔진모두동일하다. 각lane96warmup,두역순반복. 일부vLLM출력은Riley기준과달라cross-engine exact correctness를주장하지않는다.

| 조건 | V50 tok/s | V51 tok/s | vLLM tok/s | V51 TTFT ms | V51 TPOT ms | vLLM TPOT ms | V51 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 5767.946 | 6437.586 | 4810.935 | 6.054 | 2.236 | 2.358 | 82.872 | 126.091 |
| c16-natural | 7430.723 | 7697.441 | 7781.162 | 7.225 | 1.951 | 1.799 | 271.746 | 302.199 |
| c32-fixed | 7020.458 | 7783.475 | 7235.816 | 7.972 | 3.769 | 2.857 | 149.313 | 175.116 |
| c32-natural | 9691.256 | 10199.769 | 11375.320 | 10.701 | 2.953 | 2.362 | 423.632 | 396.442 |

V51/V50 처리량+3.59~11.61%, TTFT−9.91~13.34%,TPOT−3.38~10.34%. vLLM 대비C32fixed처리량+7.57%이나TPOT+31.91%; C16natural처리량−1.08%,TPOT+8.47%; C32natural처리량−10.33%,TPOT+25.01%다. C32natural P99도vLLM보다6.86%높다. 높은concurrency범위확대와지속부하검증은추후수행해야한다.

Commit `a0536a5a3d7aebf721b7a41d9bd39897c60ddde5`, binary SHA256 `03262d945a3c2f31589d5525318b175c43e38001a45be7dcc9cceea22288018d`. [분석](raw/serving-round58/serving-round58-analysis.json), [manifest](raw/serving-round58-manifest.json), [구현및correctness](PREFILL_FUSION_V51.md). 원본202파일 SHA256검증완료; archive `c8ef8932d244d9db71830d41c86862921356ead9bf2667b5b106fab7583b538d`.

V51별도trace와mapped attention occupancy진단을실행중이다. Trace를통해실제fusion구간감소와다음병목을확인한다. Occupancy는synthetic P398+31decode metadata를사용하는진단이며serving결과를대신하지않는다. Blender는종료상태유지.


## V51 trace 완료

NaturalC16/C32 및fixedC32 별도288응답 모두 기준 일치. 35파일 SHA256검증 완료. [Manifest](raw/profile-v51-manifest.json), archive `6bc6a38ece79df9cbe2824a294e2df58e2c48680154edf53afa7d023cb6e037e`.

| Trace | prefill/mixed 비중 | decode 비중 | prefill/mixed median µs | decode median µs |
|---|---:|---:|---:|---:|
| native-trace-v51 C16 | 30.42% | 69.58% | 4075.868 | 1432.277 |
| native-trace-v51 C32 | 41.05% | 58.95% | 4340.767 | 1625.623 |
| fixed-trace-v51 C32 | 77.86% | 22.14% | 3625.508 | 1382.933 |

중간80% replay window의 실제 graph span 합이다. C32natural fusion은20.164µs,이전gate/up 두kernel은각16.524µs였다. Mixed attention은55.261µs로가장큰단일kernel이며C32fixed에서도22.512µs다. CPU CUDA API는profiler overhead를포함하며production latency로간주하지않는다.

후속synthetic P398+31decode occupancy진단은mapped kernel127registers,static shared memory20,480bytes,SM당최대4activeblocks/warps를확인했다. Nsight Compute는ERR_NVGPUCTRPERM으로하드웨어counter를읽지못했다. CUDA occupancy API의정적자원상한이며실제achieved occupancy나stall원인은아직측정하지못했다. 다음V52batch는8query tile에맞춘state크기축소와score/exponential 공유메모리재사용을비교한다.
