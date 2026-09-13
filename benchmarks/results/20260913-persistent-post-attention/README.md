# Persistent post-attention phase DAG

This diagnostic extends the previous FFN-only cooperative kernel to include attention output projection and its residual/RMSNorm. Attention score/value computation itself remains outside the persistent kernel. Production dispatch/defaults are unchanged.

## Changes and ownership

Five stages execute in one finite cooperative grid: output projection → residual/norm → gate/up/SwiGLU → down projection → residual/norm. Static warp workers process projection/GEMM tiles and complete256-thread CTAs process norm rows. Four grid barriers preserve producer/consumer ordering. In particular, all first norm readers finish before the same FP32 partial buffer is reused by down projection. Shared norm sums are also synchronized before CTA reuse.

The path keeps the V7 projection split sizes, MMA recurrences, BF16 rounding and normalization order. It reuses the existing body functions. Global intermediate allocations remain; this is not on-chip fusion of the whole layer. Cooperative capability, device identity and occupancy limit are checked before launch; an invalid/oversized plan is rejected. The internal helper relies on caller-validated extents and retained owners. Metadata is immutable during a launch and invalid row counts return uniformly.

`prepare_persistent_post_attention_model.py` creates an isolated source tree, retains mixed/prefill execution, connects this kernel to all30 pure-decode layers, binds a distinct graph identity and exposes only `--ffn-backend persistent-post-attention-diagnostic-v1` in the diagnostic copy. Runtime stays Rust → C/C++ → CUDA; Python only prepares builds and offline tests. Source reproduction hashes match all six modified diagnostic files.

## Numerical and hardware evidence

Native graph replay covers24 changing-input cases (rows0/1/15/16/17/31/32/33, three repetitions). Gate, down parts, normalized intermediate, final residual and normalized output match the original operator sequence bitwise. Inactive normalized/residual destinations retain sentinels. Invalid pointer/device/oversized-plan checks pass. Native memcheck and racecheck report0 errors and0 warnings. The sanitizer harness removes only the repeated timing section; its exact source and binary hash are retained.

The executed kernel uses48 CTAs,66 registers/thread,32 bytes shared memory and0 local memory. SM89 runtime passes. SM90a/SM100a compile passes, while runtime tests on those devices are skipped for missing hardware. Multi-GPU support is not implemented by this change.

Full-model independent generation is equal on1,024/1,024 tokens across32 requests, with repeated-prompt invariance. The fixed natural corpus produces12,582,912 bitwise-equal BF16 logits: both SHA256 `492a46578581d131ab67c8c1cdb2d70f36a6539c868530e36269f1484f19f939`, equal to the prior accepted V7 dump. Whole-model memcheck reports0 errors and outputs equal ordinary runs; teardown reports zero allocations. Full-model racecheck and broad quality/soak tests were not run. Test harness names retain `ffn_pipeline_*` because the isolated build replaces that selected branch. The original cp.async FFN pipeline is not combined with this candidate.

Native timing uses separate five-operator and one-persistent-node graphs,50 warmups/500 repeats in two orders. Mean graph-window time at32 rows is20.541→19.727µs, about4% lower; rows1/16 are about12%/10% lower. These reuse the same input/weights and do not represent30 layers, cold weight traffic or serving. The previous FFN-only native timing covered fewer operations, so its raw duration cannot be subtracted from this window.

## Reproduction

Use pinned nvcc13.0.88 with `-std=c++17 -O3 -lineinfo -arch=sm_89` on `benchmarks/analysis/persistent_post_attention_probe.cu`. SM90a/SM100a object checks use `-c` and their architecture flags. Native and model commands/logs/hashes are in sibling directories. Raw artifacts remain under `/tmp/riley-opt-260912/persistent-post-attention-v1` and `persistent-post-attention-model-v1`. Installed dependencies are unchanged. Blender remains stopped.

## Matched serving comparison

RTX4090, BF16 SmolLM2-135M, vLLM0.27.1, C32 natural16/128/398-token prompts and32/64/128-token output requests. Budget/chunk512, context1024, synchronous metadata, GPU greedy and required graphs. The new V7 and post-attention lanes use the same frozen binary; the prior FFN-only lane uses its frozen binary from the previous batch. Mixed/prefill execution remains V7. No cp.async FFN or FlashInfer candidate is combined into this experiment. Exact launch arguments and identities are archived.

Orders: V7 → prior FFN-only → post-attention → vLLM, then reverse. Each process has192 excluded warmups and768 retained requests:7,680 requests total,1,536 retained and114,688 output tokens per engine. All three Riley lanes exactly match all1,536 retained references. vLLM matches1,140/1,536, reported separately from its0 transport failures. All eight runs complete with0 transport failures. Median and nearest-rank pooled P95/P99 are used; throughput reconciles output tokens with summed retained wall time.

| Metric | Existing V7 | Prior FFN-only | Post-attention candidate | vLLM |
|---|---:|---:|---:|---:|
| Output tokens/s ↑ | 10,482.0 | 10,610.7 | 10,711.1 | 12,029.1 |
| Requests/s ↑ | 140.384 | 142.108 | 143.452 | 161.104 |
| TTFT P50 (ms) ↓ | 10.419 | 10.211 | 10.381 | 18.350 |
| TTFT P95 (ms) ↓ | 17.762 | 17.322 | 17.196 | 35.888 |
| TTFT P99 (ms) ↓ | 64.004 | 67.278 | 60.626 | 62.004 |
| TPOT P50 (ms) ↓ | 2.909 | 2.894 | 2.849 | 2.353 |
| TPOT P95 (ms) ↓ | 3.023 | 2.985 | 2.963 | 2.663 |
| TPOT P99 (ms) ↓ | 3.098 | 3.044 | 3.012 | 2.851 |
| E2E P50 (ms) ↓ | 193.186 | 191.413 | 189.055 | 167.307 |
| E2E P95 (ms) ↓ | 390.822 | 386.334 | 382.940 | 341.201 |
| E2E P99 (ms) ↓ | 399.478 | 391.725 | 391.468 | 362.254 |

Throughput changes for the candidate: **+2.19% versus V7, +0.95% versus FFN-only, -10.96% versus vLLM**. Both orderings preserve the throughput ranking. Two repetitions do not establish statistical significance for the small increment over FFN-only.

| Per-run tokens/s | V7 | FFN-only | Post-attention | vLLM |
|---|---:|---:|---:|---:|
| Run0 | 10,541.9 | 10,578.6 | 10,688.7 | 12,155.0 |
| Run1 | 10,422.8 | 10,643.0 | 10,733.6 | 11,905.8 |

The candidate improves the measured V7 tails, but its TPOT and E2E remain worse than vLLM. TTFT remains lower. The overall serving goal is unmet. This is one short C32 workload screen, not high-concurrency/long-soak qualification. Peak GPU memory was not measured, clocks were not locked and GUI stayed active. Do not compare throughput percentages across old and new runs as though vLLM and environmental conditions were identical; the fresh four-lane comparison is authoritative for this batch.

## Decision and next area

Keep this as an isolated diagnostic implementation; do not change defaults. Extending only the projection boundary yielded a small additional gain. Further tiny boundary/CTA changes are not the next optimization batch. The next substantial work is attention task scheduling plus score/value ownership within the finite layer DAG. The existing V7 attention has separate score and value passes and allocates per-row/head score scratch; its four-K16 MMA and reverse128-token softmax recurrence provide the reference arithmetic. Reusing that arithmetic while assigning ragged context work to resident workers needs a workload-balanced plan and explicit scratch/barrier boundaries. This is a proposed next implementation, not proof of its benefit. Full-layer attention, QKV/RoPE integration, persistent mixed/prefill, async tickets and high-concurrency stability remain incomplete in PR07/PR19.

`serving/compact` retains request timestamps/token IDs; raw frames remain in `/tmp/riley-opt-260912/persistent-post-attention-serving-v1`. Final verification confirms unchanged binaries and no GPU compute process. Candidate binary SHA256: `dd7a6f729e90192404fe4d9213a98a6bb01c995fd944ce1a61d99608d1cdbdf1`; prior FFN-only: `5e9b7215ff06773af4368a6210fb38775c53c66ccd55d8493982ac55101e190c`. Blender was not restored.
