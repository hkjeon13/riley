# Prefill projection pipeline native gate

Isolated Q/K/V/output projection batch: tiled immutable weight packing, shared A reuse and two K64 copy/MMA buffers. Reuses the established prefill FFN copy helpers but preserves projection-specific K192/K128 intermediate BF16 rounding and increasing K16 MMA recurrence. No Rust/server/model recorder selects this header yet.

RTX4090 SM89, CUDA13.0.88. 78 conditions (three projection geometries, 13 live-row values, two refreshed-input repeats) compare all 30 layer outputs individually: 2,340 bitwise comparisons plus 78 captured 30-layer replays. Live rows include zero, ragged tails and invalid capacity1025; inactive outputs remain sentinel in per-layer audits. Packing is checked independently against row-major weights at every index. Bounded memcheck/racecheck each pass 12 conditions with no errors/hazards. SM90a and SM100a compile pass; runtime is untested because those devices are unavailable. A probe macro-variable collision caused the first compile failure and was fixed before any GPU execution; the original log is retained.

Times are microseconds per layer, median of two reversed-order estimates. Each timed graph rotates 30 distinct layer weight matrices with the same input (not an actual chained model); buffers may be cache resident. Capacity is1024 with device live-row predication. Packing/setup and model integration overhead are excluded. Native improvement is not serving improvement.

| Projection N / interval | Live rows | Original us | Pipeline us | Reduction |
|---|---:|---:|---:|---:|
| 576 / 192 | 32 | 6.188 | 4.265 | 31.07% |
| 576 / 192 | 128 | 6.520 | 4.470 | 31.44% |
| 576 / 192 | 398 | 14.231 | 8.497 | 40.29% |
| 576 / 192 | 512 | 15.322 | 8.502 | 44.51% |
| 192 / 192 | 32 | 4.872 | 3.930 | 19.32% |
| 192 / 192 | 128 | 4.948 | 4.010 | 18.96% |
| 192 / 192 | 398 | 6.127 | 4.667 | 23.83% |
| 192 / 192 | 512 | 6.339 | 4.745 | 25.15% |
| 576 / 128 | 32 | 5.677 | 3.929 | 30.78% |
| 576 / 128 | 128 | 5.877 | 4.135 | 29.63% |
| 576 / 128 | 398 | 13.164 | 8.010 | 39.15% |
| 576 / 128 | 512 | 14.453 | 8.482 | 41.31% |

Q and output use8,704B shared; K/V use6,656B shared. Resource records in [comparison.json](comparison.json) include registers and local bytes. Native storage is explicit; model packing will add53,084,160 bytes (50.625MiB) for all four projections across30 layers if original weights remain for decode. Peak preparation memory/time and graph ownership are pending model integration.

GPU work stopped only the three recorded Blender processes. All three were restored and scene queries succeeded; public static viewers remained up. No profiler or build overlapped timing. Recompute numeric evidence and validate source/log identities with `python3 benchmarks/analysis/export_prefill_projection_native.py benchmarks/results/20260914-prefill-projection-native`. `sources.json` matches the measured remote sources.

Build: prepared CUDA13 `nvcc -std=c++17 -O3 -arch=sm_89 -Xptxas=-v benchmarks/analysis/prefill_projection_pipeline_probe.cu -o probe`. Run `probe` for audit/timing and `compute-sanitizer --tool memcheck --error-exitcode 3 ./probe bounded` / `--tool racecheck` for bounded gates. The caller must obtain exclusive GPU access and restore Blender in a finally block; no global process kill is appropriate.

Proceed to [PR22 model integration](../../../deploy/260913/22-prefill-projection-pipeline.md), retaining prior rolling Riley and unchanged numeric/serving gates. Arbitrary nonfinite inputs, complete model logits/greedy, stop/cancel/recovery and matched serving have not yet been tested for this candidate. No default promotion.
