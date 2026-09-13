# Attention task cost decomposition and rejected softmax-sharing prototypes

The previous joined layer regressed serving. This follow-up isolates native attention costs and evaluates two related execution alternatives. Both preserve tested outputs, but neither improves C16/C32 native performance. **Do not integrate or promote them as serving backends.** No new full-model or vLLM measurement was performed for these prototypes; the existing serving results remain authoritative.

## Cost decomposition

`attention_task_census.cu` compares original score/value kernels, separately launched task score/value kernels, both pairs, and the combined persistent kernel. The task pair's score and output buffers are reconciled with the original path. It starts with the15-case persistent-attention correctness probe, then measures seven graph layouts across rows1/16/32 and two context distributions, in forward/reverse order with30 warmups and300 retained graph repeats. The task-phase kernels are separately compiled variants; their timings are not instrumentation of phases within the combined kernel. Graph-window durations also include launch gaps, so separately measured phases need not add exactly to pair duration.

Natural-shaped context lengths are16/128/398. The long ragged distribution spans1/7/8/9/15/16/17/127/128/129/511/512/513/4095/4096. Repeated input/weights stay hot; this is not a model or HTTP serving workload.

| C32 graph window, µs (run0/run1) | Natural-shaped | Long ragged |
|---|---:|---:|
| Original score | 4.119 /4.116 | 11.093 /11.107 |
| Original value | 9.083 /9.079 | 46.285 /46.267 |
| Original pair | 12.947 /12.964 | 57.040 /57.068 |
| Task score | 7.127 /7.121 | 14.732 /14.722 |
| Task value | 19.197 /19.193 | 130.168 /130.126 |
| Task pair | 26.006 /26.006 | 144.664 /144.640 |
| Persistent attention | 26.361 /26.347 | 153.262 /153.265 |

The large regression is present in the separately launched task value kernel, so a global grid barrier is not its sole cause. Source inspection shows that a256-thread task CTA groups warps processing different request lengths. With128 CTAs, its1,024-warp stride is divisible by32 rows, so each warp repeatedly receives the same row in C32. This identifies a plausible scheduling problem, not a measured causal attribution of every microsecond. Original value arithmetic also repeats softmax for eight output-column groups.

## Alternative1: same-head CTA with shared softmax

`shared_head_values.cuh` assigns one request/head per256-thread CTA. Warp0 computes the original reverse128-token softmax recurrence and denominator, while all eight warps consume the same BF16 probabilities for distinct output columns. Two CTA barriers per tile protect producer/consumer reuse; final inverse is shared. Original score execution is unchanged. This removes repeated softmax work but changes parallelism and introduces CTA synchronization.

All15 changed-query/metadata graph replays match V7 output bitwise, including invalid context/inactive sentinels. Score storage also remains equal. Native memcheck/racecheck report0 errors/0 warnings. Resources:40 registers/thread,776 bytes shared,0 local memory.

C32 **value-only** graph windows are9.086/9.076µs original versus10.240/10.240µs shared-head on natural-shaped contexts; long ragged is46.281/46.288 versus48.524/48.524µs. C16 also regresses. The small C1 gain does not justify a general serving candidate, and no shape router was added based on these isolated samples.

## Alternative2: normalize once, retain original value parallelism

`planned_softmax_values.cuh` separates normalization from value MMA while retaining the original96-thread value launch geometry. It preserves FP32 denominator order, BF16 probability rounding, reverse-tile rescale and MMA order. Normalization destructively packs each tile's128 BF16 probabilities into the lower half of its original512-byte FP32 score tile. The upper half of the last valid tile holds32 FP32 rescale factors and one final inverse. That tile is consumed first, before its storage is reused. Warp barriers ensure all original score reads finish before packing and all exponential readers finish before reuse.

The existing32×9×4096 FP32 score allocation is large enough; no additional GPU allocation is introduced. This is a new scratch encoding, so score-buffer bitwise equality is intentionally not its contract. The unchanged score kernel must run before every normalization: normalization is not idempotent and cannot be replayed alone on the packed representation. Existing serving graphs do not select this protocol.

All15 primitive output comparisons and inactive/invalid sentinels pass exactly. Memcheck/racecheck report0 errors/0 warnings. The consumer reports48 registers/thread,0 shared and0 local memory. Both new headers compile for SM90a/SM100a via the combined compilation probe; runtime is skipped for missing hardware. The tested runtime is SM89.

Timing compares complete attention pipelines, including original score generation and the additional normalization kernel, with no artificial buffer copy. C32 natural-shaped attention is about12.971µs original versus15.063/15.070µs planned. Long ragged is59.849/59.860 versus79.558/79.565µs. Do not compare these whole-pipeline numbers with the value-only table above. This version is also rejected at the native performance screen.

## Counter limitation and decision

Nsight Compute basic metrics were attempted read-only, but returned `ERR_NVGPUCTRPERM`; passwordless administrative access is unavailable. No driver settings or counter permissions were changed. No hardware bandwidth/cache/occupancy-stall measurements were obtained. The logs do not establish a bandwidth bottleneck, and a sanitizer pass is not counter evidence.

Stop the task-remapping/softmax-sharing candidate family here. The tests show that simply removing arithmetic duplication or kernel boundaries is insufficient under these launch/storage choices. Preserve the candidate code as diagnostic evidence, keep V7 unchanged, and return to PR03's uncompleted native precision/backend alternative. The earlier [same-input FP16 experiment](../20260913-prefill-precision/README.md) reduced synthetic error relative to FI BF16; full-model validation remains missing. The next batch should verify pinned-library mixed-dtype behavior, define explicit BF16→FP16 range/subnormal handling and ownership/conversion costs, bind a separate profile, and run the existing full-model and serving gates. It must not reinterpret BF16 bits or turn the synthetic error result into a serving claim.

The overall throughput/latency/stability objective is still unfulfilled. These are diagnostic investigations within the optimization effort, not another serving-performance milestone.

## Reproduction

Three committed probes are in `benchmarks/analysis/`: `attention_task_census.cu`, `shared_head_values_probe.cu`, and `planned_softmax_values_probe.cu`. Compile with pinned nvcc13.0.88, `-std=c++17 -O3 -lineinfo -arch=sm_89`. For the latter two, `RILEY_SKIP_TIMING=1` omits timing loops during Compute Sanitizer12.8.1 memcheck/racecheck (`--error-exitcode 99`). Cross-architecture compile uses the planned probe with `-include kernels/optional/shared_head_values.cuh`, `-c`, and SM90a/SM100a.

All raw logs and source/binary/object hashes are archived here; remote artifacts remain under `/tmp/riley-opt-260912/attention-task-census-v1`, `shared-head-values-v1` and `planned-softmax-values-v1`. Graph timing clocks were not locked, no confidence interval is claimed, and no production backend or default changed. Blender remains stopped.
