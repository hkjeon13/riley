# Query reuse mixed attention — native candidate gate

The candidate pairs adjacent 8-query tasks into one 16-query CTA, reuses K/V across those queries, and optionally removes the shared BF16 probability mirror. Native graph timing improves by 9–11% on two mixed long-prefill shapes at the serving capacity of 512 tokens. A 128-token cached suffix is effectively flat and pure 512-token prefill improves only 1.9%. This is a shape-dependent native result, not a serving speedup or default promotion.

## Implementation and contract

`kernels/optional/query_reuse_mixed_attention.cuh` preserves the existing V7 tile-map ABI. Even local task indices execute a paired query tile; odd indices return. Counts below 32 retain single-query handling; ragged final tiles keep causal masks. Audit coverage marks both original tasks to detect omission/duplication. Metadata remains caller-validated and immutable during execution. No KV ownership, request ordering, wire format or Rust/Python runtime changes occur.

Modes: 0 existing8; 1 previous compact8 control; 2 query16 with BF16 probability mirror; 3 compact-query16. The existing ordered softmax/PV recurrence, BF16 rounding, nonfinite fallback and page addressing are retained through the existing template bodies. This candidate does not implement split-K/LeanAttention or POD overlap.

Mode 2 uses 127 registers and 12,288 shared bytes; mode 3 uses 128 registers and 8,192 shared bytes, with zero local bytes. Mode 3 has an estimated 11 active one-warp CTAs/SM versus 14 for the existing kernel. Increased query reuse competes with occupancy, so doubling query width is not automatically faster. The harness's reported 122,900-byte `Plan` is unused legacy audit instrumentation, not candidate runtime workspace; the candidate allocates no new global scratch or planning kernel.

## Validation

- CUDA13 SM89: 17 shapes × 3 input repetitions × 3 candidates = 153 audit comparisons plus 153 captured replay comparisons, all output BF16 bytes exact including untouched tail guards; every original task covered once.
- Includes empty/decode-only, 31/32/33/47/49-query boundaries, prefill up to 1,024, nonzero prefix offsets, context 4,096, scrambled physical page maps and a nonfinite V case.
- Separate timing probes and four capacity512 shapes also require exact audit/replay output before timing.
- Bounded memcheck and racecheck: 27 cases each, no errors/hazards. This is not a full-model sanitizer run.
- SM90a and SM100a object compilation passes; runtime tests on Hopper/Blackwell and multi-GPU remain unavailable, not passed.
- Final gate terminates and leaves no GPU compute process. Production release binary is unchanged.

## Capacity512 native graph timings

Microseconds per graph: median of two reversed-order samples, each 50 replays after 10 warmups. Same inputs, grid capacity, streams and graph replay mode. No profiling or build overlaps the timed probes.

| Prefill / decode / prefix offset / decode context | Existing8 | Compact8 | Query16 | Compact16 | Compact16 change |
|---|---:|---:|---:|---:|---:|
| 128 / 31 / 384 / 1024 | 82.21 | 80.88 | 83.43 | 82.58 | +0.45% |
| 398 / 31 / 0 / 1024 | 95.31 | 90.47 | 85.21 | 84.56 | −11.28% |
| 480 / 31 / 16 / 1024 | 96.48 | 96.23 | 86.70 | 87.60 | −9.20% |
| 512 / 0 / 0 / 512 | 52.46 | 51.15 | 57.53 | 51.47 | −1.90% |

The broader capacity1024 probe reaches larger native gains on long contexts; those are different shapes/capacity and must not replace this serving-shape result. The probe uses synthetic tensors and a single prefill owner, so real multi-prefill distributions and full-model numerics still require validation.

## Next integration and rollback

Qualify compact-query16 through an explicit retained mixed-model execution option with source/catalog identity and ordinary/future decode compatibility. Preserve the existing execution selection as rollback. Then run full-model logits/free-generation, cache/shared-page correctness, cancellation/stop and the matched shared/unique serving comparison against frozen current Riley and vLLM. Do not assume the native percentage carries through 30 layers or removes the measured serving gap. No new vLLM table is appropriate yet because serving execution is unchanged.

[receipt.json](receipt.json) derives timing from all checked logs and hashes the artifacts. Reproduce validation with `python3 benchmarks/analysis/export_query_reuse_native.py benchmarks/results/20260914-query-reuse-native`. `query_reuse_native_gate.sh` reproduces build, exact checks, timing, bounded sanitizers and hardware compilation in the prepared remote root. Final remote gate: `/data/riley-serving-260913-recovery/query-reuse-native-v2`.
