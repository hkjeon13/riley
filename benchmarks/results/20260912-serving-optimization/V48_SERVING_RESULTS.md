# V48 matched serving results

Round55 completes24 lanes: C16/C32 × fixed/natural × V46/V48/vLLM × two reversed orders,96 warmup and384 retained requests per lane. Same checkpoint, RTX4090, external workload, admission capacity and512-token budget. V46 uses16-row decode; V48 uses packed prefill and32-row decode. Both use GPU greedy with full-result fallback. Fixed chunks are128 tokens; natural chunks512. V48 source `4195aa5dd08e5ebbc604c89a09ae8831fc339e0d`.

All9,216 retained requests succeed. All6,144 Riley reference observations match. vLLM has some different generated outputs, so this does not establish cross-engine exact output/quality equivalence. Long-run stability and broader concurrency remain unqualified.

## Measured medians

Each cell is the median of two retained-lane measurements. TPOT and TTFT are token-transport metrics; tail values are end-to-end request latency.

| Case | Engine | Output tok/s | TTFT ms | TPOT ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|---:|---:|
| c16-fixed | V46 | 4398.780 | 6.281 | 3.390 | 117.817 | 129.117 |
| c16-fixed | V48 | 5328.479 | 7.637 | 2.712 | 96.988 | 101.987 |
| c16-fixed | vLLM | 4753.252 | 18.338 | 2.333 | 111.722 | 141.030 |
| c16-natural | V46 | 6946.972 | 6.889 | 2.184 | 291.405 | 319.264 |
| c16-natural | V48 | 7003.779 | 7.821 | 2.154 | 291.430 | 301.564 |
| c16-natural | vLLM | 7796.871 | 13.068 | 1.800 | 265.415 | 294.337 |
| c32-fixed | V46 | 4446.324 | 7.544 | 7.120 | 223.246 | 251.590 |
| c32-fixed | V48 | 6204.612 | 7.396 | 4.925 | 166.715 | 189.816 |
| c32-fixed | vLLM | 7238.355 | 30.187 | 2.884 | 144.000 | 161.302 |
| c32-natural | V46 | 6736.535 | 6.845 | 4.649 | 606.668 | 691.739 |
| c32-natural | V48 | 9164.328 | 9.806 | 3.321 | 447.965 | 481.733 |
| c32-natural | vLLM | 11524.737 | 23.245 | 2.316 | 360.042 | 378.538 |

## Interpretation and next gate

V48 versus V46: throughput improves21.14% at C16 fixed,0.82% at C16 natural,39.54% at C32 fixed and36.04% at C32 natural. TPOT improves in all four. TTFT increases21.58%,13.53% and43.26% in C16 fixed/C16 natural/C32 natural, respectively; C32 fixed TTFT decreases1.96%. Do not describe this as a universal latency win.

V48 versus vLLM: C16 fixed throughput is12.10% higher, but TPOT is16.23% slower. In the other three cases throughput is10.17–20.48% lower and TPOT19.65–70.77% slower. TTFT is lower in all four cases, while P95/P99 beat vLLM only on C16 fixed. The full performance goal is not achieved.

V48 is retained as the next optimization comparison candidate because packing materially improves C32 and removes the severe V47 fixed TTFT regression. Keep frozen V46 available as a latency reference; no general production/default promotion or long-run stability claim is made. The next batch must compare directly to V48 and vLLM, with V46 retained for context. Investigate remaining decode cost and the prefill/decode scheduling boundary using fresh V48 traces before choosing implementation.

All202 evidence files are SHA256 verified under `raw/serving-round55-manifest.json`; archive SHA256 `4ed066946870e8e445b4b20592ad6bba61937ca4e91011319508f6c797f02be0`. Blender remains stopped without restoration. V48 profiling completed after the benchmark controller was terminal.

## Fresh V48 profile evidence

Three96-request diagnostic lanes (natural C16/C32, fixed C32) complete with all288 token/text references matching. In the middle80% replay window, summed GPU graph spans give the following stage shares; the shares are not computed by multiplying medians.

| Workload | Prefill submissions, whole trace | Decode submissions, whole trace | Median prefill us | Median decode us | Prefill share | Decode share |
|---|---:|---:|---:|---:|---:|---:|
| C16 natural |71|518|3405.384|1536.114|26.87%|73.13%|
| C32 natural |58|298|3795.420|1727.636|32.46%|67.54%|
| C32 fixed |41|84|3399.912|1521.874|57.64%|42.36%|

Natural decode attention values/scores are major kernel costs. Fixed serving spends more graph time in prefill, while the scheduler still alternates separate prefill/decode iterations. The next batch will test mixed execution with owner-safe token packing and compact attention tile mapping; see `MIXED_PREFILL_DECODE_V49.md`. This direction is an inference from profiling and source, not a proven causal speedup.

Runtime API timing includes Nsight overhead and must not be substituted for serving latency. All35 profile files are SHA256 verified under `raw/profile-v48-manifest.json`; archive SHA256 `ec16ed433d8d04cb0e752d36514075c82fe47906ea61acc33ee97e19f4b00adb`. Benchmark and profile controllers are terminal. No further GPU job is running from this round.
