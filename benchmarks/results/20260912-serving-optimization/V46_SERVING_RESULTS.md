# V46 compact GPU results: serving comparison

V46 is accepted as the next baseline for the measured C16/C32 sixteen-row GPU-greedy configurations. V44 remains the CPU/full-logit reference. The overall vLLM goal is not achieved, and lower concurrency plus long-duration tail/stability qualification remain open.

Same pinned SmolLM2-135M BF16 checkpoint/tokenizer, RTX4090, external workloads, admission capacity, KV budget and token budget. V44 uses CPU normative validation; V46 explicitly uses compact GPU greedy. Two reversed orders per condition, 96 warmup and 384 retained requests per lane. All 9,216 retained requests completed with zero failures; all 6,144 Riley responses match immutable reference tokens/text. vLLM references are not all identical; no cross-engine exact-output or quality-equivalence claim.

| Workload | V44 tok/s | V46 tok/s | vLLM tok/s | V46 vs V44 | V46 vs vLLM |
|---|---:|---:|---:|---:|---:|
| c16-fixed | 3816.7 | 4436.7 | 5202.2 | +16.24% | -14.71% |
| c16-natural | 5886.5 | 7005.6 | 7869.6 | +19.01% | -10.98% |
| c32-fixed | 3850.2 | 4432.6 | 6830.4 | +15.13% | -35.10% |
| c32-natural | 5749.9 | 6757.8 | 11931.5 | +17.53% | -43.36% |

| Workload | TTFT vs V44 | TPOT vs V44 | P95 vs V44 | P99 vs V44 |
|---|---:|---:|---:|---:|
| c16-fixed | -11.46% | -14.17% | -15.06% | -11.74% |
| c16-natural | -7.93% | -16.23% | -16.51% | -15.78% |
| c32-fixed | -14.96% | -13.17% | -12.43% | -13.17% |
| c32-natural | -10.96% | -14.91% | -14.61% | -13.71% |

TTFT beats vLLM in these conditions, but TPOT and tail latency remain worse. In C32 natural, TPOT is 4.637ms vs vLLM 2.301ms and P99 is 697.97ms vs 346.53ms. This batch closes a measured result-path bottleneck; it does not establish vLLM parity.

Fresh node-level Nsight diagnostic: 192/192 streaming reference checks pass. Median compact decode D2H is 0.864us at C16/C32 (previous V44 trace approximately85–88us); compact D2H is exactly2048bytes. Prefill GPU graph spans are2805.555/2819.283us; decode1431.193/1432.922us. Profiler API timings are not serving CPU cost; middle80 graph medians and kernel means have different populations and must not be added.

Remaining C32 decode kernel ranking: values attention mean9.914us per layer, gate/up/SwiGLU6.988us, attention scores5.034us, down-projection parts4.626us. C16→C32 throughput remains flat or declines while only16 GPU rows are dispatched. Next investigate the coupled decode computation/batching limit, using measured retained width and kernel costs. V45 grouped/paired value reuse already regressed long contexts and must not be reintroduced without new evidence.

The next optimization batch should first establish whether a wider/tiled decode execution can amortize projection/launch costs and improve attention occupancy, then combine related kernel and dispatch changes. Require exact full-logit and compact-token checks, live-row/mask safety, repeated shape transitions, and comparison against both this V46 baseline and vLLM. Do not weaken finite checking, RNG semantics, or complete-result identity to improve timing.

Source commit4668b77b2e8c5311fdd5abfb2b0b665775b7ef66; frozen binary SHA256e64ea083114f8a56608e68d96be2aa2aa74ad09ddc3b50f19c32695e6cc5fabc. Compact integration correctness and fixed failures are detailed in `COMPACT_GREEDY_NEXT.md`.

Evidence: 223 files under `raw/serving-round53`, verified against `raw/serving-round53-manifest.json`; archive SHA256120f2d49026d7120f5674c998c503341b03c6514a984a1849d5f06485a1aabf9. The 75-file compact integration manifest also still verifies. Blender PIDs/ports were checked stopped again; no restore action was taken.
