# Short prefill serving: candidate rejected

RTX 4090 / SmolLM2-135M BF16 / C32, prefix caching, context 1024, chunk 512, equal 720MiB KV payload. Frozen best split-FFN + rolling decode is prior; split is the short-prefill candidate with the same policies; vLLM is **0.27.1**, not the latest release. Two reversed orders, 64 warmup + 512 retained requests per lane. Metrics are median run-level estimates, not pooled percentiles.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 10202.173 | 14.434 | 2.729 | 103.785 | 132.000 |
| shared | split | 9973.339 | 14.506 | 2.791 | 110.162 | 139.482 |
| shared | vllm | 11712.218 | 31.296 | 1.726 | 100.555 | 107.790 |
| unique | prior | 4116.785 | 70.477 | 5.752 | 290.455 | 317.328 |
| unique | split | 4168.104 | 65.294 | 5.695 | 287.053 | 313.644 |
| unique | vllm | 4715.813 | 42.492 | 5.321 | 269.639 | 340.321 |

Candidate throughput changes **−2.24% shared / +1.25% unique** against prior. Shared regresses in both orders, with worse P95/P99. Unique improvement is small and does not compensate. vLLM unique throughput varies 4,330.2→5,101.5 tok/s across orders, so this small screen is not a stable capacity estimate or a latest-version qualification. **Reject promotion and revert production integration.**

All 6,912 warmup/retained responses are reconstructed from retained SSE data; 4,608 Riley responses match frozen references. Stop/cancel/recovery checks pass (64 each). Native 768 cases × two replays are bit-exact, small memcheck/racecheck clean, full-model logits and full-model memcheck pass. SM89/90a/100a compile, only SM89 executes. Blender restoration succeeds. Numerical correctness does not imply a performance gain.

The measured result refutes promotion of this short-query mapping on the tested ordinary path. Possible register/resource or shape-mix costs are hypotheses, not profiler-confirmed causes. Do not attribute the regression to one without evidence, or keep tuning this mapping based only on isolated arithmetic savings. Refresh the latest vLLM baseline before choosing the next substantial execution/scheduling batch.

`source-snapshot.tar.gz` contains tested production and controller sources. `candidate.patch` applies to parent `844379560cf36f253b5e8ad46a61a6d835dd2440` in a separate checkout for reproduction; production source is reverted after the screen. The native probe requires that candidate header. Frozen remote candidate binary SHA256 is `f015d8ae28ae7ff86fbf03dac4f54bc958aa4493091970e74d8324206376748d`. Source hashes were checked against remote sources before rollback. The verifier reads immutable snapshots rather than current production files.

```sh
python3 benchmarks/analysis/verify_short_prefill_serving.py
```
