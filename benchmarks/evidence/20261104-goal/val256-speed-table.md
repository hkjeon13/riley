KV chunk 256 default for two-KV-head models (lp15) against vLLM 0.27.1, order rvvr (2026-11-08; GPU 7 for the Coder models and 0.5B, GPU 6 for the rest). 512 (lp12, `val512/speed`) in brackets.
Throughput Riley/vLLM (higher is better); TPOT and TTFT p50 Riley/vLLM (lower is better).

| model | reason-c8 tok/s | reason-c32 tok/s | reason-c8 TPOT | reason-c32 TPOT | rag-c1 TTFT | errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.028 [0.995] | 1.005 [0.998] | 0.983 [1.015] | 1.003 [1.012] | 1.248 [1.497] | 0 |
| Qwen2.5-0.5B-Instruct | 1.203 [1.180] | 1.133 [1.085] | 0.909 [0.964] | 0.883 [0.921] | 1.221 [1.622] | 0 |
| Qwen2.5-1.5B-Instruct | 0.967 [0.997] | 1.048 [1.026] | 0.981 [1.037] | 0.945 [0.983] | 1.260 [1.442] | 0 |
| Qwen2.5-3B-Instruct | 1.021 [0.965] | 1.046 [1.020] | 0.981 [1.023] | 0.976 [1.003] | 1.257 [1.369] | 0 |
| Qwen2.5-Coder-1.5B-Instruct | 1.081 [0.989] | 1.092 [1.036] | 0.985 [1.041] | 0.964 [0.990] | 1.263 [1.465] | 0 |
| Qwen2.5-Coder-3B-Instruct | 1.038 [0.940] | 1.043 [1.067] | 0.986 [1.022] | 0.973 [0.996] | 1.261 [1.358] | 0 |
