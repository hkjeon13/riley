Prompt rows 512, generated rows 256 on two-KV-head models (lp16) against vLLM 0.27.1, order rvvr (2026-11-08; GPU 6 for the Coder models, GPU 7 for 0.5B, earlier runs on either). 512 (lp12, `val512/speed`) in brackets.
Throughput Riley/vLLM (higher is better); TPOT and TTFT p50 Riley/vLLM (lower is better).

| model | reason-c8 tok/s | reason-c32 tok/s | reason-c8 TPOT | reason-c32 TPOT | rag-c1 TTFT | errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.029 [0.995] | 1.008 [0.998] | 0.982 [1.015] | 1.002 [1.012] | 1.236 [1.497] | 0 |
| Qwen2.5-0.5B-Instruct | 1.214 [1.180] | 1.171 [1.085] | 0.906 [0.964] | 0.888 [0.921] | 1.243 [1.622] | 0 |
| Qwen2.5-1.5B-Instruct | 0.972 [0.997] | 1.055 [1.026] | 0.977 [1.037] | 0.948 [0.983] | 1.245 [1.442] | 0 |
| Qwen2.5-3B-Instruct | 1.026 [0.965] | 1.042 [1.020] | 0.979 [1.023] | 0.975 [1.003] | 1.232 [1.369] | 0 |
| Qwen2.5-Coder-1.5B-Instruct | 1.087 [0.989] | 1.084 [1.036] | 0.986 [1.041] | 0.968 [0.990] | 1.256 [1.465] | 0 |
| Qwen2.5-Coder-3B-Instruct | 1.039 [0.940] | 1.047 [1.067] | 0.986 [1.022] | 0.970 [0.996] | 1.234 [1.358] | 0 |
