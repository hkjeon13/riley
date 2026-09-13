# Adaptive 16/32-row decode projections — native gate

Implemented an isolated candidate for QKV, attention-output projection and FFN down. When the device packet reports ≤16 active rows, it computes one MMA row tile; otherwise it uses the existing two-tile routine. K recurrence, partial BF16 rounding, output indices and 32-row scratch stride remain unchanged. No host readback, Q/KV repacking or extra workspace is introduced. **The serving/model recorder does not select it yet.**

The [C8 serving profile](../20260914-decode-capacity-serving/README.md) identifies these operations as 29.04% of decode kernel time. `kernels/optional/decode_adaptive_rows.cuh` is intentionally separate from the default implementation pending full-model/serving validation.

## Checks and native measurements

- RTX4090: 102 graph comparisons (3 input seeds × active0..33), all QKV/output/down partial bits equal to baseline; inactive and trailing guards preserved. The same captured graphs observe changing row metadata.
- Bounded memcheck and racecheck: 34 cases each, zero errors/hazards. These are small primitive processes, not whole-model sanitizer runs.
- CUDA13.0.88 SM89 build/run passed. SM90a and SM100a compile-only gates passed; runtime skipped because those GPUs are absent.
- Candidate QKV: 59 registers, no local memory. Compiler details and source/binary/object hashes are retained in the logs and receipt.

Timing uses one graph containing the three projection groups for 30 distinct weight sets (106,168,320 bytes). Each lane has 20 warmups and100 replays, four alternating orders. Inputs are synthetic and the operations are not connected into a forward pass. Values below are median CUDA-event times; this is **not full-model or serving performance**, and weight size is not a measured HBM traffic counter.

| Active rows | Existing µs | Adaptive µs | Time change |
|---|---:|---:|---:|
| 1 | 330.563 | 247.439 | −25.15% |
| 8 | 336.169 | 265.436 | −21.04% |
| 16 | 370.570 | 317.471 | −14.33% |
| 17 | 377.774 | 376.591 | −0.31% |
| 24 | 394.445 | 393.436 | −0.26% |
| 32 | 445.496 | 443.638 | −0.42% |

Tiny changes above16 rows are not evidence of an improvement; that branch executes the existing arithmetic. No claim is made that shared-register occupancy is reduced merely because one branch computes less work.

## Reproduce and next gate

```sh
nvcc -std=c++17 -O3 -arch=sm_89 -Xptxas=-v \
  benchmarks/analysis/decode_adaptive_rows_probe.cu -o /tmp/decode-adaptive-rows-probe
/tmp/decode-adaptive-rows-probe
timeout 90 compute-sanitizer --tool memcheck --error-exitcode 9 /tmp/decode-adaptive-rows-probe sanitizer
timeout 90 compute-sanitizer --tool racecheck --error-exitcode 9 /tmp/decode-adaptive-rows-probe sanitizer
```

Compile the same source with `-arch=sm_90a -c` and `-arch=sm_100a -c` for hardware-specific build checks. Kernel runtime remains native CUDA; Python is only an external analysis tool.

Next: integrate ordinary and future paired decode, include candidate source in model/profile identity, run full-model free-generation and logit equivalence, then perform matched serving measurements at low and high concurrency. The native result supports that experiment; it does not complete PR06 or the vLLM goal. Rollback leaves the current model recorder unchanged or removes this unselected optional candidate.
