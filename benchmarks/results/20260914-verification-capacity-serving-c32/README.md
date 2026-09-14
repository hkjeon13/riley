# Bounded verification graph: final batch result

Stage 2 retains the exact short-query reuse from stage 1 and limits only the wide verification graph launch capacity to min(configured capacity, 256). Stage-3 contracts allow at most 32 owners × 8 inputs. Ordinary graph capacity and backing allocations are unchanged. Catalog: `target-verification-head.v6.short-query-cap256`.

RTX 4090 / SmolLM2-135M BF16 / C32, context 1024, chunk 512, 720MiB KV per engine, prefix caching; two orders, 64 warmup + 512 retained requests per lane. Metrics are medians of two run-level estimates, not pooled percentiles. Comparator is **vLLM 0.27.1**, not a latest-release qualification.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 10341.880 | 14.165 | 2.697 | 100.870 | 130.543 |
| shared | control | 6946.296 | 14.549 | 4.186 | 169.796 | 180.864 |
| shared | speculative | 7018.767 | 14.768 | 4.094 | 167.544 | 179.305 |
| shared | vllm | 12218.461 | 30.302 | 1.665 | 90.421 | 92.985 |
| unique | prior | 4129.224 | 69.673 | 5.661 | 290.118 | 317.174 |
| unique | control | 3840.647 | 63.215 | 6.443 | 322.619 | 379.711 |
| unique | speculative | 3835.742 | 59.117 | 6.532 | 327.857 | 376.312 |
| unique | vllm | 4947.210 | 41.324 | 5.120 | 247.063 | 325.305 |

Prior is frozen split-FFN + rolling decode; control is stage-1 short-query reuse; speculative is stage-2 reuse + bounded graph. The extra graph bound changes throughput **+1.04% shared / −0.13% unique**. This is a small screen, without confidence intervals; unique has no established gain. Stage 1 independently showed +8.39% shared / −0.05% unique versus old speculative. Do not combine different-run ratios as a paired measurement.

The final speculative lane remains below best prior (−32.13% shared / −7.11% unique) and vLLM (−42.56% / −22.47%). **Performance goal not met; speculative remains opt-in and disabled by default.** No broad latency/stability qualification is claimed from C32 alone.

Correctness: all 9,216 warmup/retained responses reconstructed; all 6,912 Riley responses match frozen references; 96 stop, 96 cancellation, 96 recovery checks pass. Model normal and memcheck runs each retain all 1,024 exact tokens with 629 accepted drafts. Native stage-1 arithmetic is unchanged: 480 cases × two replays exact, small memcheck/racecheck clean. SM89/90a/100a compile; only SM89 executes. Hopper/Blackwell and multi-GPU execution remain untested. Blender restoration succeeds after measurement. Runtime remains Rust → C ABI → CUDA.

`manifest.json` binds archive members and sources; the six remote production source hashes match this checkout. Build binary SHA256: `c480e4917edaf3bf82a0392a7861b7119bc184140b3285e6c7670cea6f5c88e2`. Intermediate sources and Nsight numeric evidence are preserved separately in `../20260914-verification-attention-serving-c32/`; those profiles are stage 1, not stage 2. Raw profiler traces remain remote-private.

```sh
python3 benchmarks/analysis/verify_verification_capacity_serving.py
```

Next candidate: ordinary cached-prefix short prefill while preserving split-FFN and rolling decode. Full-page cache hits leave 1–16 prompt tokens, and ordinary attention still uses one-query work below 32 rows. Extending ordered reuse to 2–16 queries needs independent exact native tests (including 9–16), model gates and matched serving. This is a proposed direction, not an implemented or measured improvement. Rollback: unset `RILEY_SPECULATIVE_DECODE`; reverting this batch restores prior verification graph/kernel selection.
