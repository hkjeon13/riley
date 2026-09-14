# vLLM 0.27.1 / 0.29.0 and frozen best Riley baseline

RTX 4090 / SmolLM2-135M BF16 / C32 / max-active 32, context 1024, prefill chunk and batch budget 512, prefix caching enabled, 754,974,720-byte KV payload per engine. Both vLLM versions use identical model/tokenizer paths; all lanes use the same generated prompt corpora and model weights (Riley uses its existing converted checkpoint). Riley uses frozen split-FFN + rolling decode (SHA256 `658c00b7b99b57d54a12dd345dd44a3ba705fdffa6749687eed9a8e1a04306cc`). Both vLLM versions use their own default runner/backend selection; inherited VLLM overrides are removed and batch invariance is disabled consistently.

Two reversed orders per workload, 64 warmup + 512 retained per lane. Values below are medians of two run-level estimates, not pooled percentiles. This is a screening baseline, not a broad capacity or stability qualification.

| Workload | Engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | Riley best | 10201.627 | 14.377 | 2.713 | 105.455 | 133.656 |
| shared | vLLM 0.27.1 | 11879.503 | 29.989 | 1.737 | 99.207 | 102.270 |
| shared | vLLM 0.29.0 | 11882.298 | 30.495 | 1.751 | 98.908 | 105.717 |
| unique | Riley best | 4114.030 | 65.774 | 5.862 | 291.277 | 319.148 |
| unique | vLLM 0.27.1 | 5122.371 | 39.985 | 4.966 | 216.503 | 326.380 |
| unique | vLLM 0.29.0 | 4964.412 | 40.888 | 5.033 | 240.040 | 316.956 |

Riley throughput versus 0.29.0 is **−14.14% shared / −17.13% unique**. Shared TTFT is lower but TPOT is higher; unique TTFT and TPOT are both higher. The goal is not met. vLLM 0.29.0 versus 0.27.1 changes +0.02% shared / −3.08% unique in this screen; two runs with no confidence interval do not establish a general version regression. A larger repeated concurrency/workload matrix remains required.

Environment: Python 3.13.15 and Torch 2.13.0 in both. 0.27.1 uses Transformers 5.15.1 and FlashInfer 0.6.16.post3; 0.29.0 uses Transformers 5.17.0 and FlashInfer 0.6.18. This compares supported resolved stacks, not a vLLM-code-only ablation. The 0.29.0 environment is isolated; 0.27.1 is unchanged. Wheel installation and pip dependency checks completed successfully; the pinned resolved package list is retained. [Official 0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0) was verified as latest on 2026-09-14.

Verification: latest-version preflight completes 32 valid responses. Independent verifier reconstructs 6,912 warmup/retained responses; all 2,304 Riley responses match frozen references. Riley stop/cancel/recovery checks pass (32 each). All 12 measured server exits and preflight exit are zero; lifecycle exit is zero and all three Blender processes are restored. vLLM token agreement is distinct from protocol validity and is not required to equal Riley numerical output. No GPU work or builds ran concurrently with measurement.

The first offline archive attempt erroneously tried to hash the local-only validator on the remote source tree; it failed after constructing the archive, without modifying or rerunning the benchmark. The corrected export hashes the remote controller/client and binds the local validator separately in the source snapshot.

Reproduce the independent result checks:

```sh
python3 benchmarks/analysis/verify_vllm_version_serving.py
```

The updated baseline supports continuing substantial attention/execution and scheduling work. It does not justify another short-prefill tile micro-variant: that candidate already failed serving promotion. Before the next batch, isolate ready-work host gaps and decode versus mixed/prefill device cost on the best ordinary path; use the 0.29.0 comparison lane for subsequent milestone screens. Multi-model, high-concurrency, multi-GPU, Hopper and Blackwell execution remain unqualified.
