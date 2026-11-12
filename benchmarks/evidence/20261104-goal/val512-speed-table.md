KV chunk 512 (lp12) against vLLM 0.27.1, GPU 6, order rvvr (2026-11-05). Throughput Riley/vLLM (higher is better); TPOT p50 Riley/vLLM (lower is better).
14B rows in brackets: re-run on GPU 7 with the vLLM-matched KV share (`kv2600/`: 2,600 or 2,576 blocks instead of 1,859).

| model | reason-c8 tok/s | reason-c32 tok/s | reason-c8 TPOT | reason-c32 TPOT | errors |
|---|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Llama-8B | 0.977 | 1.044 | 1.027 | 0.961 | 0 |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.995 | 0.998 | 1.015 | 1.012 | 0 |
| DeepSeek-R1-Distill-Qwen-14B | 0.987 [0.987] | 0.803 [1.031] | 1.015 [1.014] | 0.953 [0.973] | 0 |
| DeepSeek-R1-Distill-Qwen-7B | 1.015 | 1.030 | 0.989 | 0.975 | 0 |
| Llama-3.2-1B-Instruct | 1.066 | 1.003 | 0.951 | 1.000 | 0 |
| Llama-3.2-3B-Instruct | 1.024 | 1.003 | 0.974 | 1.014 | 0 |
| Meta-Llama-3.1-8B-Instruct | 0.971 | 1.043 | 1.029 | 0.958 | 0 |
| Qwen2.5-0.5B-Instruct | 1.180 | 1.085 | 0.964 | 0.921 | 0 |
| Qwen2.5-1.5B-Instruct | 0.997 | 1.026 | 1.037 | 0.983 | 0 |
| Qwen2.5-14B-Instruct | 0.993 [0.994] | 0.830 [1.013] | 1.020 [1.018] | 0.960 [0.977] | 0 |
| Qwen2.5-3B-Instruct | 0.965 | 1.020 | 1.023 | 1.003 | 0 |
| Qwen2.5-7B-Instruct | 1.002 | 1.019 | 0.996 | 0.966 | 0 |
| Qwen2.5-Coder-1.5B-Instruct | 0.989 | 1.036 | 1.041 | 0.990 | 0 |
| Qwen2.5-Coder-14B-Instruct | 0.982 [0.982] | 0.867 [1.037] | 1.018 [1.018] | 0.959 [0.977] | 0 |
| Qwen2.5-Coder-3B-Instruct | 0.940 | 1.067 | 1.022 | 0.996 | 0 |
| Qwen2.5-Coder-7B-Instruct | 1.086 | 1.000 | 0.998 | 0.967 | 0 |
| Qwen3-0.6B | 1.079 | 1.040 | 0.944 | 0.976 | 0 |
| Qwen3-1.7B | 1.019 | 1.008 | 0.991 | 0.999 | 0 |
| Qwen3-14B | 0.987 | 1.048 | 1.015 | 0.958 | 0 |
| Qwen3-4B | 1.004 | 0.978 | 1.001 | 1.027 | 0 |
| Qwen3-8B | 0.975 | 0.987 | 1.029 | 1.018 | 0 |
| SmolLM2-1.7B-Instruct | 0.996 | 0.971 | 0.998 | 1.063 | 0 |
| SmolLM2-135M-Instruct | 1.088 | 1.063 | 0.951 | 0.963 | 0 |
| SmolLM2-360M-Instruct | 1.039 | 1.102 | 0.971 | 0.914 | 0 |
