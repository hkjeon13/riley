# GQA shared K/V staging — native gate

The candidate groups the three query heads of one KV head into a96-thread CTA. Each warp retains an independent8-query softmax/accumulator. A single128-token K/V tile is copied into shared memory with16-byte cp.async loads and reused by all three warps. Paged addresses, BF16 storage/rounding, reverse128-token softmax traversal and K16 MMA order are preserved. This is one-buffer staging, not copy/compute overlap across tiles. Nonfinite V causes a uniform CTA fallback to the original per-query path.

On RTX4090 the candidate reduces native graph time by2.30–16.90% across eight capacity512 shapes, including2/4/8 prefill owners. Grouping without shared staging regresses in every shape. These are synthetic kernel graphs, not model or serving throughput; there is no new vLLM claim or default promotion.

## Matched native timing

Median of two reversed-order samples;50 captured replays after10 warmups per sample. Prefill is the total token count, divided evenly among the listed prefill owners. Decode entries are separate owners. No GPU profiler/build overlaps timing. Native graph envelope includes decode rows on the mapped attention path, unlike the full model's separate pure-decode graph.

| Prefill | Decode | Prefix | Context | Prefill owners | Original µs | Compact µs | Grouped µs | Staged µs | Change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 31 | 384 | 1024 | 1 | 81.95 | 80.81 | 108.25 | 80.07 | -2.30% |
| 398 | 31 | 0 | 1024 | 1 | 94.40 | 89.37 | 124.01 | 78.45 | -16.90% |
| 480 | 31 | 16 | 1024 | 1 | 96.36 | 95.99 | 118.56 | 86.18 | -10.56% |
| 512 | 0 | 0 | 512 | 1 | 52.64 | 51.59 | 70.72 | 44.80 | -14.89% |
| 512 | 0 | 0 | 512 | 2 | 28.33 | 27.22 | 38.36 | 24.27 | -14.35% |
| 512 | 0 | 0 | 512 | 8 | 11.36 | 10.53 | 13.39 | 9.73 | -14.34% |
| 480 | 24 | 16 | 1024 | 4 | 88.12 | 88.01 | 103.96 | 78.21 | -11.24% |
| 256 | 24 | 384 | 1024 | 8 | 83.21 | 85.94 | 113.99 | 77.87 | -6.42% |

## Correctness and resource evidence

- Final gate:19 shapes ×3 input repetitions ×3 candidate/control modes =171 audit and171 captured replay comparisons. All BF16 bytes, untouched tails and task coverage agree with existing V7. Includes short/decode-only/empty, ragged31/32/33/47/49, context4096, scrambled pages, multi-prefill owners and injected nonfinite V.
- Bounded memcheck and racecheck each execute45 comparisons, with0 errors/hazards. This is not full-model sanitizer validation.
- Staging release:152 registers/thread,45,056 shared bytes,0 local bytes/spills. Occupancy API reports2 CTAs/SM =6 warps versus14 warps for the original one-warp kernel. Grouping-only has127 registers and12,288 shared bytes. Lower theoretical occupancy did not prevent a measured gain; no hardware-counter HBM traffic claim is made.
- SM90a and SM100a compilation passes. Actual Hopper/Blackwell/multi-GPU execution is unavailable and untested.
- Test controller restores all three Blender processes and validates actual scene-query responses, even on gate failure. Public3d/3dsol/3dfable viewers are separate CPU static servers and remain outside GPU lifecycle.

## Reproduction and next gate

`python3 benchmarks/analysis/export_gqa_staged_native.py benchmarks/results/20260914-gqa-staged-native` checks current source hashes, final raw comparison/sanitizer logs and lifecycle receipts, then recomputes [receipt.json](receipt.json). Final remote evidence is `/data/riley-serving-260913-recovery/gqa-staged-native-v2`; `initial/` retains the earlier single-prefill gate and `final/` is authoritative for the current probe. Initial hardware logs used the earlier probe version and are not the final source binding.

`gqa_staged_native_gate.sh` reproduces kernel builds/tests once the caller has exclusive GPU ownership; the caller must separately record process/lifecycle exit receipts and restore Blender in a finally handler. The executed campaign used remote `run.py` for this lifecycle. Binary/objects remain remote; repository logs contain no private profiler databases or environment dumps.

Next: additional Q/K nonfinite and boundary fault coverage, explicit retained C ABI recorder and source/catalog identity, pure-decode dispatch preservation, model logits/greedy and shared-cache/cancel checks. Then compare actual serving against the frozen rolling-capable Riley and vLLM at C8/C32/C64. Do not substitute this native improvement for that milestone. No production source currently references this header.
