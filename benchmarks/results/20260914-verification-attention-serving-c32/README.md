# Short-query verification attention: intermediate batch result

The measured bottleneck was per-query attention for all 1–8-query verification owners. Stage 1 reuses ordered eight-query tile arithmetic for 2–8 queries, preserves the single-query path for one input, and dispatches one CTA per owner/head. Ordinary prefill/decode captures remain unchanged. This is an intermediate source snapshot before the later 256-input graph bound.

RTX 4090 / SmolLM2-135M BF16 / C32, context 1024, chunk 512, 720MiB KV per engine, prefix caching. Two orders, 64 warmup + 512 retained requests per lane; median of run-level metrics (not pooled percentiles). vLLM version is 0.27.1.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 10399.410 | 14.026 | 2.681 | 100.945 | 133.352 |
| shared | control | 6414.008 | 15.153 | 4.519 | 182.205 | 196.956 |
| shared | speculative | 6952.101 | 13.754 | 4.160 | 168.472 | 179.018 |
| shared | vllm | 12044.537 | 25.455 | 1.838 | 96.887 | 104.808 |
| unique | prior | 4123.385 | 68.117 | 5.661 | 290.430 | 318.328 |
| unique | control | 3801.740 | 59.807 | 6.562 | 333.098 | 390.551 |
| unique | speculative | 3799.785 | 63.067 | 6.504 | 321.995 | 377.129 |
| unique | vllm | 4875.427 | 41.632 | 5.194 | 245.292 | 323.183 |

Prior is frozen split-FFN + rolling decode; control is the previous speculative binary; speculative is the new query-reuse binary. Throughput versus control improves **8.39% shared**, while unique changes **−0.05%**. Both remain below prior and vLLM. No default promotion or broad qualification.

Correctness: 480 native cases × two graph replays have exact full-buffer agreement (owners 1/8/32, query counts 1–8, contexts 8–4096, finite and NaN V). Small memcheck/racecheck pass. SM89/90a/100a compile; only SM89 executes. C32 warmed-prefix model gate and full-model memcheck: all 1,024 tokens exact, 629 accepted drafts, 39 serial / 33 speculative calls. Serving reconstructs all 9,216 warmup/retained responses; all 6,912 Riley responses match frozen references, with 96 stop, 96 cancellation and 96 recovery checks. Blender restored after each GPU lifecycle.

Four diagnostic Nsight traces (512 warmup/retained requests total) retain exact output. In the middle 80% of graph launches, observed mean verification graph span is 4.671→3.861ms shared and 4.641→3.910ms unique; mean attention kernel time per graph is 2.531→1.723ms and 2.620→1.885ms. These include warmup and varying shape mixes, not per-token or serving speedups. Raw traces remain remote-private.

An isolated CUDA Graph event experiment separates dispatch from reuse. At 32 owners × 8 queries, context 512, shared physical KV pages, two-order median kernel times are 130.81µs (legacy grid 1024×9), 122.70µs (same math, grid 256×9), 44.38µs (query reuse, owner grid 32×9). This supports retaining reuse; it is not a serving workload. All 144 timing records are retained.

The next part of this same optimization batch caps the entire verification transformer graph at 256 inputs, its proven maximum (32 owners × 8 inputs). This removes impossible padding work in projection, FFN and normalization without changing ordinary prefill capacity. See the separate capacity-stage result.

```sh
python3 benchmarks/analysis/verify_verification_attention_serving.py
```

`source-snapshot.tar.gz` binds intermediate production sources so this evidence remains verifiable after the capacity-stage edit.
