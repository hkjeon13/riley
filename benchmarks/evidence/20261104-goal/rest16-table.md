| 모델 | c1 TPOT | lp-c8 느림(vLLM/Riley 처리량) | lp-c32 느림 | 오류 |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | 1.000 | 1.009 | 0.940 | 0 |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.882 | 0.923 | 0.876 | 0 |
| DeepSeek-R1-Distill-Qwen-14B | 0.982 | 1.012 | 0.956 | 0 |
| DeepSeek-R1-Distill-Qwen-7B | 0.957 | 0.976 | 0.937 | 0 |
| Llama-3.2-1B-Instruct | 0.929 | 0.874 | 0.774 | 0 |
| Llama-3.2-3B-Instruct | 0.965 | 0.963 | 0.970 | 0 |
| Meta-Llama-3.1-8B-Instruct | 1.001 | 1.008 | 0.932 | 0 |
| Qwen2.5-0.5B-Instruct | 0.969 | 0.850 | 0.642 | 0 |
| Qwen2.5-1.5B-Instruct | 0.890 | 0.942 | 0.893 | 0 |
| Qwen2.5-14B-Instruct | 0.984 | 1.008 | 0.945 | 0 |
| Qwen2.5-3B-Instruct | 0.930 | 0.970 | 0.920 | 0 |
| Qwen2.5-7B-Instruct | 0.957 | 0.981 | 0.932 | 0 |
| Qwen2.5-Coder-1.5B-Instruct | 0.883 | 0.940 | 0.887 | 0 |
| Qwen2.5-Coder-14B-Instruct | 0.983 | 1.008 | 0.945 | 0 |
| Qwen2.5-Coder-3B-Instruct | 0.928 | 0.947 | 0.922 | 0 |
| Qwen2.5-Coder-7B-Instruct | 0.957 | 0.982 | 0.937 | 0 |
| Qwen3-0.6B | 1.015 | 0.849 | 0.755 | 0 |
| Qwen3-1.7B | 0.948 | 0.944 | 0.926 | 0 |
| Qwen3-14B | 1.003 | 1.006 | 0.939 | 0 |
| Qwen3-4B | 0.953 | 0.977 | 0.987 | 0 |
| Qwen3-8B | 0.990 | 1.018 | 0.982 | 0 |
| SmolLM2-1.7B-Instruct | 0.972 | 0.951 | 0.906 | 0 |
| SmolLM2-135M-Instruct | 0.991 | 0.678 | 0.462 | 0 |
| SmolLM2-360M-Instruct | 0.920 | 0.718 | 0.589 | 0 |

7B shared-c8: TTFT p95 31.0 vs 46.9 ms = 0.660x, 처리량 627 vs 595, 오류 0

7B shared-c32: TTFT p95 38.9 vs 74.8 ms = 0.520x, 처리량 2356 vs 2113, 오류 0
32B TP2 closed-c1: TPOT 0.981x, 처리량 1.023x, TTFT 0.867x, 오류 0
32B TP2 closed-c8: TPOT 1.008x, 처리량 1.023x, TTFT 0.573x, 오류 0
32B TP2 closed-c16: TPOT 1.021x, 처리량 1.052x, TTFT 0.350x, 오류 0
32B TP2 closed-c32: TPOT 1.000x, 처리량 1.066x, TTFT 0.359x, 오류 0
