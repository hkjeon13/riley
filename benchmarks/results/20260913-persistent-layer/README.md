# Joined attention-through-FFN: rejected serving candidate

The finite cooperative attention→projection→FFN implementation preserves measured outputs, but **regresses serving throughput and latency**. Keep it diagnostic; do not promote it. The attention integration optimization batch has completed its first full-model and serving assessment, with a negative performance decision. PR07/PR19 and the overall serving goal remain incomplete.

## Implementation and memory ordering

`persistent_layer.cuh` combines the verified attention task bodies with the earlier post-attention stages. Its ordered phases are score → value → output projection → residual/RMSNorm → gate/up/SwiGLU → down → residual/RMSNorm. One128-CTA resident cooperative grid executes all phases on the tested4090. Six grid barriers protect stage dependencies; the new value→projection barrier ensures that no score reader remains when that same FP32 scratch storage is overwritten with projection partials. The later norm readers likewise finish before down reuses the partial storage. Per-warp probabilities/exponentials and per-CTA norm sums have explicit reuse barriers.

Ragged score-prefix planning, row-interleaved value tasks, original four-K16 score recurrence and reverse128-token softmax order are retained. Projection/FFN MMA and BF16 rounding are unchanged. Global intermediate storage is still allocated; this does not establish on-chip whole-layer reuse. QKV/RoPE remain outside the kernel.

The raw internal launch helper validates pointers, current device and occupancy-limited grid size; the existing native/Rust owner retains responsibility for page/parent extents and stream lifetimes. Metadata is immutable for each replay. Invalid active counts return uniformly. The isolated model generator binds an independent graph identity including attention task and projection sources, and connects all30 pure-decode layers through `--ffn-backend persistent-layer-diagnostic-v1`. Mixed/prefill retains existing V7. The default checkout has no new serving selector. Python remains build/offline tooling; inference is Rust → C/C++ → CUDA.

## Correctness and hardware evidence

The combined primitive compares all score/partial storage, activation, attention output, normalized intermediate, final residual and output against the original seven-operator sequence.24 changing-query graph replays cover rows0/1/15/16/17/31/32/33 with16/128/398-token contexts and noncontiguous KV pages. Outputs compare bitwise; inactive destinations retain sentinels. The separate attention prerequisite additionally covered maximum4096 contexts and invalid context lengths; those are not new full-layer test cases here.

Native memcheck and racecheck report0 errors/0 warnings with `RILEY_SKIP_TIMING=1`. The timed and sanitizer runs use the same binary. Combined kernel resource attributes:60 registers/thread,6,308 bytes shared,0 local memory/thread,128 CTAs. SM89 runtime passed; SM90a/SM100a compile passed while runtime is skipped for missing devices. Multi-GPU scheduling is not implemented by this work.

Independent full-model generation matches1,024/1,024 tokens across32 requests. The natural fixture matches12,582,912 BF16 logits bitwise; baseline and candidate hashes are both `492a46578581d131ab67c8c1cdb2d70f36a6539c868530e36269f1484f19f939`, matching the prior accepted V7 dump. Whole-model memcheck reports0 errors and identical ordinary/sanitized logits; teardown reports zero allocations. Full-model racecheck, broad model-quality and long-soak checks were not performed. Test harnesses retain their `ffn_pipeline_*` names because this isolated source replaces that explicitly selected implementation.

## Matched serving result

RTX4090, BF16 SmolLM2-135M, vLLM0.27.1, C32 natural16/128/398-token prompts and32/64/128 output tokens. Budget/chunk512, context1024, required graphs, GPU greedy and synchronous metadata. Same new binary for V7/joined candidate; prior post-attention uses its frozen binary. Installed dependencies are unchanged. GUI stays active, Blender stays stopped and clocks are not locked.

Two orders: V7 → prior post-attention → joined → vLLM, then reverse. Each process has192 excluded warmups and768 retained requests:7,680 total,1,536 retained and114,688 output tokens per engine. Every Riley lane matches all1,536 retained references. vLLM matches1,122/1,536, reported independently of transport success. All eight runs complete with0 transport failures. Throughput reconciles total retained output tokens and wall time; P50 is median and P95/P99 are nearest rank over pooled requests.

| Metric | Existing V7 | Prior post-attention | Joined attention→FFN | vLLM |
|---|---:|---:|---:|---:|
| Output tokens/s ↑ | 10,486.0 | 10,785.2 | 8,597.8 | 12,265.3 |
| Requests/s ↑ | 140.437 | 144.445 | 115.150 | 164.268 |
| TTFT P50 (ms) ↓ | 10.382 | 10.393 | 10.422 | 16.865 |
| TTFT P95 (ms) ↓ | 15.108 | 17.638 | 17.335 | 35.482 |
| TTFT P99 (ms) ↓ | 61.762 | 66.724 | 62.853 | 67.198 |
| TPOT P50 (ms) ↓ | 2.907 | 2.834 | 3.577 | 2.316 |
| TPOT P95 (ms) ↓ | 3.021 | 2.943 | 3.655 | 2.609 |
| TPOT P99 (ms) ↓ | 3.068 | 2.981 | 3.693 | 2.960 |
| E2E P50 (ms) ↓ | 193.091 | 188.268 | 235.198 | 163.430 |
| E2E P95 (ms) ↓ | 391.122 | 380.925 | 472.545 | 334.367 |
| E2E P99 (ms) ↓ | 396.904 | 388.974 | 480.821 | 360.196 |

Joined throughput changes: **-18.01% versus V7, -20.28% versus prior post-attention, -29.90% versus vLLM**. Both orderings show the regression. TPOT and E2E also regress. The lower TTFT relative to vLLM does not satisfy the combined serving objective.

| Per-run tokens/s | V7 | Prior post-attention | Joined | vLLM |
|---|---:|---:|---:|---:|
| Run0 | 10,493.4 | 10,743.8 | 8,616.3 | 12,032.0 |
| Run1 | 10,478.5 | 10,827.0 | 8,579.4 | 12,507.9 |

This is a short C32 screen, not high-concurrency stability or peak-memory qualification. Two repetitions provide no general confidence interval. Use this fresh matched comparison rather than subtracting historical vLLM percentages from different sessions.

## Boundary experiment and next decision

After serving completed, a separate native boundary probe compared three graph layouts on the same fixtures: original operators; the joined kernel; and persistent attention followed by a separate persistent post-attention kernel. Each candidate's timed final output is rechecked against a fresh operator graph.50 warmups/500 repetitions in forward/reverse order use repeatedly reused input and weight buffers, so these are native graph windows, not serving results.

| Rows | Operators µs, run0/run1 | Joined µs | Separate persistent µs |
|---|---:|---:|---:|
| 1 | 19.227 /19.231 | 19.555 /19.565 | 17.197 /17.197 |
| 16 | 27.286 /26.094 | 37.544 /36.289 | 30.972 /30.964 |
| 32 | 31.314 /31.322 | 49.399 /49.412 | 42.858 /42.842 |

Splitting the persistent kernels recovers some of the loss, but C16/C32 remain slower than the original operators. Thus removing only the joined boundary does not resolve the regression. This experiment also changes per-phase grid size and compiled resource allocation (128-CTA attention versus48-CTA post-attention), so the difference cannot be attributed solely to barrier time. No profiler hardware-counter or per-phase GPU-cycle decomposition was collected; the precise cause is not established. Zero local memory is evidence against attributing this result to local spills without further profiling.

Stop enlarging this monolithic cooperative kernel. Preserve the numerically validated task interfaces, and investigate the attention task-planning and work-assignment cost before any further whole-layer fusion. Concrete questions include repeated prefix/index decoding in score workers, ragged value-task balance and repeated per-column softmax work; these are code-derived candidates, not confirmed bottlenecks. The next batch must change a coherent attention execution strategy and compare its full-model/serving result, rather than iterate unrelated CTA constants. The existing accepted path remains the fallback; neither joined nor separate persistent attention is promoted by this report.

QKV/RoPE persistence, mixed/prefill integration, async tickets, and long-running/high-concurrency qualification remain outstanding. A failed design trial is not completion of the serving goal.

## Reproduction and artifacts

`model/prepare-v1.py` reproduces the measured v1 source; all seven modified files match its source receipt. The current `prepare_persistent_layer_model.py` contains the fallback correction described below. Native probes use nvcc13.0.88, `-std=c++17 -O3 -lineinfo -arch=sm_89`; cross-architecture builds use `-c` with SM90a/SM100a. Model commands, memory logs and hashes are in `model/`; combined and boundary native evidence is in `native/`.

Raw HTTP frames remain in `/tmp/riley-opt-260912/persistent-layer-serving-v1`, with reconciled compact timestamps/token IDs and raw hashes under `serving/`. Candidate binary SHA256: `7b7c973cad8bd6ef07454f737fb88d2bc497a1c11501318e580669ea99234470`; prior post-attention: `dd7a6f729e90192404fe4d9213a98a6bb01c995fd944ce1a61d99608d1cdbdf1`. Binary verification confirms both unchanged; final native state confirms no GPU compute process after the boundary probe. Blender was not restored.

## Diagnostic generator fallback correction

Final review found that the v1 generator started the new branch immediately after the optional FlashInfer attention block. The measured V7 and joined profiles have no FlashInfer workspace, so their recorded execution is unaffected, but selecting the unmeasured FlashInfer mode in that diagnostic copy would skip post-attention work. The current generator instead wraps the complete original attention-plus-post-attention sequence. Structural verification confirms the entire original fallback block is preserved verbatim. Production checkout code was never modified by either generator.

A separate v2 source/build passes independent generation, natural full-logit equality, whole-model memcheck0 errors, ordinary/sanitized output equality, and server release compilation. Its source receipt is `isolated-source-v2-verification.json`; logs are in `model-v2/`. The initial v2 build accidentally reused v1 CMake cache and failed source-directory validation; it was corrected to a separate v2 Cargo target, with the failure log retained. No numerical gate was waived.

The serving table and boundary probe above remain measurements of frozen v1, whose exact generator is archived. V2 was not remeasured in serving and is not assigned that binary identity or a performance qualification. The attention-through-FFN kernel itself is unchanged; the candidate family remains rejected on the measured regression.
