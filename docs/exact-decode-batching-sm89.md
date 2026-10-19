# Exact decode batching on RTX4090

2026-10-17. This is a Qwen2.5-3B screening result, not completion of the multi-family vLLM parity goal. The latest adopted local source is V27, following V24/V25/V26; its adoption and all932 matching source hashes are recorded in `v27-source-adoption.json`. Earlier sections retain the chronology and evidence of their individual candidates.

## Structure and changes

The installed stock vLLM 0.27.1 Qwen2 model uses `QKVParallelLinear`, `MergedColumnParallelLinear` for Gate/Up, and the selected FlashAttention 2 backend. Its cache update and variable-length attention kernels avoid Riley's sequence of per-request matrix products and intermediate materializations. The inspected source excerpt is `artifacts/multimodel-parity-20261017/vllm0271-qwen2-merged-projection-attention-reference.txt`.

Riley's owned-HF route must preserve the same-device M=1 arithmetic. Plain cross-request GEMM or a different fused attention implementation can change BF16 results. Current qualified changes preserve the FP32 FMA/reduction order and BF16 rounding boundaries:

- QKV: existing qualified singleton bias batching preserves actual request computations. This work does not claim a new generic fused QKV GEMM or qualification for every model.
- Gate/Up, Down and LM Head: weight reuse across requests, with the exact lane-strided FMA and reduction. V17 extends the already-qualified RTX4090 dispatch to 2–7 rows as well as 8–32.
- Attention: adopted V15 combines KV gathering over request/head/depth-token axes for chunks of 2–8 requests and combines scale/softmax while preserving both original BF16 rounds. QK and PV remain the original cuBLAS arithmetic at each actual KV length. This is not FlashAttention equivalence.
- V18b: coalesced 16-byte weight loads stage LM Head and small Down weights in shared memory, with a padded stride. Gate/Up and large Down retain the earlier kernels because their tested staging variants did not improve performance. The actual weight pointer must be aligned before using vector loads.

The projection dispatch remains opt-in and gated to the qualified SM89 / 128 SM / runtime12080 / cuBLAS120804 environment and qualified shapes. A100 recipes and results remain separate.

## Latest matched screen: V17 to V18b

Single RTX4090, Qwen2.5-3B-Instruct BF16, same frozen checkpoint, CPU sampling, iteration command batching, `lt-mlp-v2`, `segments-v1`, context1024, token budget512, KV blocks64×concurrency, identical CPU affinity and original EOS. Two fresh processes per cell, fixed workload, warm24 and measured48 requests per process. Stock vLLM is unmodified; its installed Torch stack uses CUDA13.0 while Riley uses CUDA12.8.1/cuBLAS120804.

Throughput is output tokens/s. TPOT is the response-mean SSE estimate; each table value is the median of two process values, not a pooled token-level percentile.

| Concurrent requests | Before V17 | After V18b | Change | Stock vLLM | vLLM / after | TPOT p95 before → after, ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 118.44 | 118.39 | -0.04% | 130.24 | 1.10x | 8.18 → 8.18 |
| 8 | 222.23 | 255.16 | +14.82% | 706.80 | 2.77x | 33.00 → 28.61 |
| 16 | 314.52 | 329.72 | +4.83% | 1095.47 | 3.32x | 43.89 → 42.45 |
| 32 | 368.97 | 383.81 | +4.02% | 1503.45 | 3.92x | 68.32 → 68.45 |

C1 has no demonstrated gain. C32 p95 TPOT slightly worsens despite higher aggregate throughput. These two-repeat screens are not statistical parity proof. There were additional Riley graph captures during several measured windows (including 216 before / 144 after in the second C32 run), so this is the observed serving screen, not a fully warmed steady-state kernel speedup. Longer, stable-capture paired measurements remain required.

Other CPU/Java/build workloads were present on the shared host, with I/O/memory pressure observed during the final stock vLLM startup. Startup loading/compilation is excluded from measured throughput; CPU affinity does not prove isolation from those other jobs. The raw observation is preserved. No foreign process was stopped. A100 GPU2/3 remained owned by wckim; occupancy and owners were checked after each finite arm.

## Numerical and runtime evidence

- 481 CPU tests passed; 40 ignored GPU tests are not counted as passes.
- 109 actual projection fixtures and 4,469 component conditions: byte-exact, including changed-input Graph replay.
- 1,147 full-model step executions / 10,171 physical full-vocabulary BF16 rows: byte-exact against same-device eager references, including small counts, larger batches and shrinking/reordered batches.
- 1,092 persisted Riley HTTP values (token IDs/text, usage/finish fields and recovery responses) match the frozen reference; both original and optimized values are checked. All 24 arms passed saved GPU PID ownership checks.
- Safe-Rust weight alignment remains 256 bytes: 30 valid cases and 30 expected invalid-argument rejections per binary. An earlier test incorrectly expected offset2 to be valid; its failure is retained and the validator was not weakened.
- Four selected new GPU kernels match the qualified standalone staged device code; the six original weight reuse kernels remain unchanged. Separate runtime profiling observed 518 executions of the new staged kernels and zero diagnostic trace kernels; `owned-v18b-runtime-proof.json` and `source-adoption-v18b.json` record runtime and adoption independently.

Raw artifacts: `artifacts/multimodel-parity-20261017/owned-v18b-audit-v1.json`, `owned-v18b-screen-result.json`, `owned-v18b-full-vocabulary-summary.json`, `owned-v18b-full-vocabulary-gate.json`, `owned-v18b-weight-alignment-gate-v2.json`, `v18b-staged-component-device-code-equality.json`, `v17-v18b-device-code-equality.json`. Immutable remote snapshots, per-arm requests/logs/library maps and failed attempts are retained under `/data/riley-multimodel-parity-20261017`.

Stock vLLM is a throughput reference under its own numerics: full generated-token-sequence agreement with the frozen HF reference in the measured requests is C1: 64/96, C8: 68/96, C16: 64/96, C32: 68/96. Both Riley versions match 96/96 at each concurrency. This is exact sequence agreement, not a semantic quality score. Details: `owned-v18b-generation-agreement.json`.

## Remaining work

V18b leaves a large vLLM gap. The next staged candidate groups independent singleton RMSNorm requests into one grid while keeping each request's original 512-thread reduction. It is not part of V18b and must pass component, full-model and runtime/performance gates before source adoption. Qwen3/Mistral model qualification and broader model support, five fresh-process pairs on fixed and natural workloads, and all p50/p95/p99 latency gates remain open. No commit, push or deployment is part of this work.

## V19 RMSNorm candidate: complete Qwen2.5 screen, not adopted

481 CPU tests, 162 component conditions, 5,345 Qwen2.5 physical vocabulary rows and 997 Qwen3-1.7B physical rows passed. The Qwen3 gate covers one 32-step teacher prefix; all 32 selected same-device HF rows in each run are also exact. The 24-arm V18b/V19/stock screen completed with warm64/keep48, two fresh processes per cell, CPU sampling and fixed workload. All 1,092 persisted Riley HTTP values remain exact.

| Concurrency | V18b before | V19 candidate | Change | Stock vLLM | TPOT p95 before → candidate, ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 118.01 | 118.05 | +0.03% | 130.15 | 8.22 → 8.22 |
| 8 | 256.18 | 263.32 | +2.79% | 702.97 | 28.21 → 27.59 |
| 16 | 332.36 | 346.78 | +4.34% | 1091.16 | 42.61 → 40.70 |
| 32 | 385.75 | 408.52 | +5.90% | 1499.20 | 68.36 → 63.64 |

The separate 960-token C32 runtime profile reduces RMSNorm launches from 70,080 to 7,300, including 3,577 actual singleton-batch launches with grid>1/block512. Its prefill reductions retain the original block geometry. No diagnostic trace kernels ran. Raw proof: `owned-norm-v19-runtime-proof.json`.

Warm64 still leaves new Graph captures in some measured windows. The optional `measure-owned-bucket-warm-v1.py --warm-buckets` runner is prepared for finite full-reference warm waves at every eligible power-of-two count; it has not been executed and must pass a pilot before steady-state claims. All full-value numerical assertions are unchanged. Existing raw screens remain provisional.

Qwen3-4B and Mistral-7B same-device whole-model qualification is pending under the finite NVMe resume coordinator recorded in `nvme-wide-v19-handoff.json`. The first Mistral C1 reference hit its 1,200-second bound during HDD weight loading, before inference; its failed receipt is retained. A single CPU staging process copies the pinned Mistral and Qwen3-4B checkpoints to the independent NVMe volume and verifies source manifests plus destination readback hashes. Derived HF fixtures change only the model path. Loading time is separate from inference throughput. A generated CPU reporting-code defect was replaced with the standalone `verify-wide-norm-nvme-v3.py`; four CPU checks confirm exact outputs pass and changed bits, missing steps, and timed-out references fail. The GPU commands and numerical assertions remain unchanged. No throughput claim exists for these wider models. Local adopted source remains V18b pending the broader V19 model gates.

## V20a compact KV candidate: arithmetic feasibility passed, not adopted

The installed stock vLLM 0.27.1 FlashAttention backend stores KV with `num_kv_heads` and passes that cache to its variable-length attention path. The inspected cache layout and call sites are pinned in `vllm0271-compact-kv-reference-v20a.txt`. Riley's qualified Q16/KV2/D128 chunk path currently expands each KV head to eight query heads before QK/PV. V20a instead gathers two KV heads and constructs device pointer tables for the original sixteen-head per-request cuBLAS calls. Actual T, leading dimensions, batch count16, FP32 compute and BF16 rounding boundaries remain unchanged; scale/softmax is unchanged. This preserves the arithmetic constraints instead of assuming that FlashAttention itself is equivalent.

Preliminary GPU arithmetic gates on the same RTX4090 / CUDA12.8 / cuBLAS120804 passed with zero changed BF16 elements:

- All T1..512 with four finite synthetic input patterns: 2,048 eager conditions and 45 changed-input Graph conditions.
- Five actual activation boundary inputs (T1,31,128,257,512), with 32 rotated controls each: 160 eager conditions and 15 changed-input Graph conditions. Historical probabilities are only PV inputs; both QK and PV reference outputs are freshly computed using the original strided batch16 operations. Placeholder fixture outputs are never used as an oracle.

These are QK/PV component gates, not proof of complete attention or model equivalence. Both finite children exited0, ownership receipts were retained, and A100 GPU2/3 owners were rechecked after each experiment. CPU model copying overlapped these numerical-only gates; no throughput or kernel latency was measured.

The isolated V20a source is V18b plus one CUDA file; it does not include V19 RMSNorm. It writes one eighth as many KV payload elements and places six pointer tables in the unused tail of the existing K scratch. Scratch allocation sizes remain unchanged, and pointer writes add traffic. It keeps the original path for scratch alignment/capacity fallback. No additional buffers or hot-path CUDA allocations are introduced by the candidate. Full native compilation, server/private probe builds and 481 CPU tests passed; 40 GPU tests remained ignored. All 932 snapshot source hashes were audited.

The finite full paged-attention gate passed all 14,848 conditions over T1..512, request counts2..8, mixed lengths, paired lengths and reordered shared paged pools. Context/probabilities/scaled traces were byte-exact, native allocation counters remained stable, and Graph replay matched with zero then restored nonzero inputs. Receipt: `compact-kv-full-chunk-v20a-controller-result.json`; report SHA256 `317100c100aefb56fe316aa6462845a52b6567e4f1a28d0bc2ce5b86fc4f253f`. Complete model byte gates and matched serving performance remain required before adoption. Current handles are in `nvme-wide-v19-handoff.json`; worktree inference source remains V18b.


## Compact KV measured screen and combined V22 candidate

The ABBA component screen completed four fresh processes (V18b, V20a, V20a, V18b). Each process checks 45 cells and four CUDA-event windows per cell. The incremental comparison uses each binary's **batched chunk** time (`after_cuda_event_us`); the legacy per-request M1 time is not the incremental baseline. These are component timings, not HTTP throughput or a stock-vLLM comparison.

| Uniform KV length, 8-request chunk | Current V18b, us | Compact V20a, us | Time change |
|---:|---:|---:|---:|
| 1 | 37.82 | 38.28 | +1.22% |
| 31 | 41.84 | 40.06 | -4.26% |
| 128 | 51.27 | 44.20 | -13.79% |
| 257 | 73.22 | 56.36 | -23.02% |
| 512 | 94.31 | 63.80 | -32.35% |

39/45 cells were faster. All six slower cells had fewer than128 total KV tokens in the chunk; the worst observed slowdown was about2.3%. The combined V22 candidate therefore retains the original path below128 total KV tokens (below1MiB expanded K+V payload). The existing alignment/capacity fallbacks remain. This conservative guard also forgoes two small/noisy gains. V22 requires fresh component and full-model qualification; it is not adopted. Raw: `compact-kv-micro-v2-controller-result.json`, `compact-kv-small-work-dispatch-v1.json`, `candidate-combined-v22.json`.

## Qwen3-4B command-batch blocker and V21 fix

After correcting the private harness to match both server wide-query branches (merged QKV and fused QKV/RoPE), Qwen3-4B exposed an actual runtime blocker: `riley_cuda_hf_pack_qkv_execute` rejected command batches as diagnostic-only. The original V18b synchronous C1 path completed and matched all104 fresh same-device HF vocabulary rows. Its208 BF16 output files are retained as the before reference; this was an unsupported execution mode, not a numerical mismatch.

V21 replaces three device2D copies with one uint16 bit-copy kernel and uses the existing command-batch resource ledger. Dimensions, dtype, capacities, alias/context checks and completion/error handling stay in place. The primitive gate passed42 cases: rows1..8192, three offset combinations (including2-byte offsets), all65,536 BF16 encodings, repeated calls in one command batch, three changed-input Graph phases with three replays each, unchanged guard bytes and stable native allocation counters. 481 CPU tests passed;40 GPU tests remained ignored.

V21 Qwen3-4B C1 iteration completion, C8 eager, C8 Graph and shrinking/reordered C32 runs produced2,674 physical full-vocabulary rows, all exact to the original synchronous C1 reference. HF104 rows also match. C8 Graph logs show72 first captures and72 first replay markers; these markers prove actual segment Graph execution but are not total replay counts. Raw: `wide-pack-v21-controller-result.json`, `wide-pack-v21-physical-Graph-summary.json`, `wide-qkv-pack-v21-v2-result.json`. This is teacher-forced numerical qualification, not a complete own-generation or serving-performance result.

V22 combines V19 RMSNorm, V21 pack support and the guarded compact KV path in an isolated snapshot. The adopted worktree source remains V18b while combined model/runtime/HTTP/performance gates proceed. No deployment, commit or push has occurred.


## Mistral large-matrix exact reuse candidate V23

V22's guarded compact KV implementation passed the full 14,848-condition paged-attention gate, including byte-exact context/probability/scaled traces, changed-input Graph replay, and stable native allocation counters. It remains an isolated candidate until combined model and serving gates pass.

Two FP32 shared-activation experiments passed their 123 numerical conditions and Graph checks but failed the performance gate against the adopted V18b hybrid. Both were rejected; their raw results are retained in `float-input-v1-disposition.json`. No equality tolerance was changed.

Mistral-7B's 112MiB Gate/Up/Down matrices exceed the measured 72MiB RTX4090 L2, while the current path issues independent M=1 operations. V23 reuses each weight load across independent decode requests while retaining the original per-lane FMA sequence, half-tree addition order, and BF16 rounding. This follows vLLM's shared-weight batching principle; it does not assume that stock batched GEMM or FlashAttention is arithmetically interchangeable with the reference.

Actual 128-row activation inputs were captured for all225 Q/K/V/O/Gate/Up/Down/LM projections. Weights are read from SHA256-verified ranges of the pinned NVMe safetensors, avoiding a duplicate full-weight fixture. Every projection is compared against fresh same-device cuBLAS GemmEx M=1 outputs. All9,225 count/offset conditions passed, plus zero and changed-nonzero Graph replays for every projection. The qualified lane topology differs by shape: Q/K/V/O use16 lanes; Gate/Up/Down/LM use32. Other tested lane choices produced differences and remain rejected with raw mismatch files preserved.

Preliminary one-process warm-Graph component medians on actual layer0 inputs (four windows; **not model throughput or stock-vLLM comparison**):

| Projection, 32 independent rows | Original M=1 loop, us | Exact reuse, us | Speedup |
|---|---:|---:|---:|
| Q | 520.25 | 131.41 | 3.96x |
| K | 240.13 | 88.32 | 2.72x |
| Gate | 3972.57 | 412.67 | 9.63x |
| Down | 4008.57 | 432.72 | 9.26x |
| LM Head | 9012.21 | 916.98 | 9.83x |

R2 was slower for every tested geometry. The initial V23 candidate therefore enters only for8..32 independently grouped singleton rows, with16-byte aligned activation pointers and the existing SM89/128SM/CUDA12080/cuBLAS120804 qualification. Smaller tails, prefill row grouping, and unqualified environments retain their original dispatch. The 8-row pilot was faster for each tested geometry. Full-model C8 eager/Graph and reordered shrinking C32 gates, actual runtime kernel proof, and matched before/after/stock serving measurements are still required. Worktree inference source remains V18b; no adoption is claimed by the component results.

Receipts: `mistral-ranged-fixture-v1-controller-result.json`, `mistral-lane-pilot-v1-controller-result.json`, `mistral-attn-linear-pilot-v1-controller-result.json`, `mistral-all-linear-qual-v2-controller-result.json`. Immutable candidate CUDA SHA256: `52bc1922766b726ce62ba96030dd3f64c886582b2116a353de4a240bca27e20b`.


## Adopted V24: reachable QKV batches and staged-weight tails

V24 is now adopted in the local worktree after numerical, HTTP, runtime-use and performance audits. Seven inference source files changed from V18b. Unrelated dirty files were preserved; the original contents of the seven files are saved under `artifacts/multimodel-parity-20261017/v24-pre-adoption-local-source`. No deployment, commit or push occurred. The CPU regression suite passed481 tests;40 ordinary GPU unit tests remained ignored, with the finite GPU qualification below executed separately.

The V23 trace exposed a real wiring omission: the no-bias branch of `execute_projection_with_row_groups` forced decode mode0, so the new Q/K/V kernels were not reachable. V24 forwards the requested mode and enables singleton grouping only for the cold-qualified SM89 owned plans with K4096 and N4096/1024. Bias paths, multi-token prefill extents, unqualified hardware and other geometries retain their existing dispatch. Actual per-node tracing now proves6,400 K/V kernels at grid128 and6,400 Q/O kernels at grid512, alongside staged Gate/Up/Down/LM kernels:22,504 new kernel executions and22,176 Graph-node executions in the finite teacher run. All208 output files also match the original C1 reference.

BF16 weight bits are copied with coalesced16-byte loads into per-block shared memory. FMA order, lane reduction and BF16 rounding remain unchanged. This adds no persistent weight copy or global CUDA buffer. Measured crossover gates are shape-specific: Gate/Up/Down/LM start at2 rows and use staging through16; larger row tiles keep scalar weights. Q/O start at3 and stage through8; K/V start at4 and stage through32. Input and weight vector alignment and the original SM89/128SM/CUDA12080/cuBLAS120804 gate remain required. C1 keeps its original native operations.

All225 actual projections passed9,225 conditions and changed-input Graph replays at8/16/32 rows. The first expanded verifier reported188 zero-input Graph failures while every eager and new-nonzero result matched. Its full-input restore used the default stream while zeroing/replay used a nonblocking stream, allowing the restore to overtake zeroing. The repaired verifier orders the restore on the execution stream, completes cold uploads, and reads back/asserts the actual zero input before replay. With **identical shader bodies**, all225 cases pass all three Graph sizes. Failed raw results remain intact; no arithmetic or equality assertion was weakened. See `mistral-v24-verifier-stream-repair-and-runtime-audit.json`.

Mistral-7B, Qwen3-4B and Qwen2.5-3B all pass full-vocabulary before/after byte comparisons in C8 eager, C8 Graph and reordered/shrinking C32 runs. Mistral/Qwen4B also match all104 selected HF rows each. Qwen2.5's historical16-row HF fixture has a pre-existing cross-runtime difference; the fresh original native C1 differs from that fixture too. V24 exactly preserves the native before reference. This is not a claim of universal HF-logit identity.

Matched RTX4090 Mistral-7B C8 fixed-workload pilot, BF16, equal weights and512 KV blocks,22GiB process cap, **24 measured requests /480 output tokens per arm**, one fresh process per engine. The same full-reference warm waves1/2/4/8 and16 additional warm requests are used for every arm. Both Riley measurement windows have zero new Graph captures. These are preliminary serving measurements, not five-repeat or multi-model parity results.

| Metric | Original V18b | Adopted V24 | Stock vLLM0.27.1 |
|---|---:|---:|---:|
| Output tokens/s | 62.57 | 202.74 | 303.84 |
| TTFT p95, ms | 308.31 | 229.13 | 123.37 |
| Response p95, ms | 3942.64 | 1129.30 | 673.88 |

Throughput is3.24x the original, response p95 is71.36% lower and TTFT p95 is25.68% lower. V24 reaches66.73% of stock throughput in this pilot. Both native arms passed87 fully logged response comparisons plus16 asserted warm responses, including exact tokens/text/prompt/finish/usage, disconnect/retry recovery and stable native allocation counters. Every arm stayed below22GiB, with GPU process ownership verified. Stock's24 measured fixed requests have identical HF tokens/480-token totals; its mixed correctness diagnostic is reported separately.

The initial V23 serving monitor conservatively rejected its own stock EngineCore because that child lacked the custom initial-exec environment marker. The retained server log, GPU PID samples and loaded-library map independently identify the same worker; the amended V23 report is separate from its raw failure. V24's monitor additionally verifies parent ancestry and keeps a PID/start-time ledger through teardown; its three arms completed without amendment.

Remaining work: Attention is outside the existing pre/post layer Graph segments, and the current compact Q16/KV2 path does not cover Mistral's Q32/KV8 geometry. A finite measurement-window serving trace is queued to quantify this remaining CPU/kernel overhead before extending attention grouping. Five-fresh-process fixed/natural comparisons and other concurrency/model performance remain required for the broad goal.

Evidence: `v24-source-adoption.json`, `mistral-v24-model-controller-result.json`, `mistral-v24-serving-pilot-audit-v1.json`, `mistral-v24-runtime-profile-audit-v1.json`. `v24-diff-check-audit.json` records the sole pre-existing trailing blank whitespace retained from V18b; there are no new whitespace findings.


## Adopted V25: Qwen3-4B projection batching

V25 adds Qwen3-4B's actual2560-wide projection geometries to the qualified shared-weight decode path. Q/K/V/O, Gate/Up and the tied LM head retain their L16 FMA topology; Down retains L32. Each output still uses the original lane-strided FMA sequence, half-tree reduction and BF16 rounding. Small-row cutoffs follow the measured crossover: Gate/Up and Q/K/V at3 rows, Down at6, output at4, LM at2. Shapes, actual16-byte alignment, singleton grouping and SM89/128SM/runtime12080/cuBLAS120804 remain required. A row-first R16 grid improves Mistral K/V and Qwen4B Down/K/V/O above16 rows. Other Mistral paths remain unchanged.

All478 actual projections across Mistral and Qwen4B pass19,598 count/offset conditions, plus zero and changed-nonzero Graph replay at8/16/32. Three-model full-vocabulary comparisons pass in eager C8, Graph C8 and shrinking/reordered C32. Qwen2.5 preserves its original same-device native bytes; its historical cross-runtime HF discrepancy remains separately reported. Runtime node traces prove24,948 Qwen4B staged Graph-node executions and1,536 Mistral interleaved Graph-node executions; the208 Qwen4B and64 shrinking Mistral output files match both qualified outputs and original eager C1 bytes. CPU tests:481 passed,40 ordinary GPU tests ignored; finite GPU gates executed separately.

Matched RTX4090 fixed-workload serving screens, one fresh process per arm, before=V24 and after=V25, BF16, budget512, context1024,22GiB process cap, KV blocks64×concurrency, identical CPU affinity and greedy sampling. Every arm uses full-reference warm waves through its concurrency bucket and16 warm requests. Both Riley measurement windows have zero new Graph captures. These are **single-process pilots**, not five-repeat parity results.

| Model / concurrency | Before V24, tokens/s | After V25, tokens/s | Change | Stock vLLM0.27.1, tokens/s | After / stock |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B /8 |108.56|282.31|+160.05%|506.96|55.69%|
| Mistral-7B /32 |248.28|253.25|+2.00%|689.25|36.74%|

| Model / concurrency | TTFT p95 before → after, ms | Response p95 before → after, ms | Stock response p95, ms |
|---|---:|---:|---:|
| Qwen3-4B /8 |118.52 →105.73|2266.49 →763.87|399.75|
| Mistral-7B /32 |1125.18 →1119.53|3496.33 →3412.50|1087.18|

The Qwen4B improvement is material in this screen. Mistral's2% throughput change is small and has not established a repeatable whole-model gain. All logged native token/text/prompt/finish/usage and recovery values match the frozen V24 eager reference, and all GPU process identities and22GiB caps pass the audit. Stock produces the same480/960 output-token totals, but its complete token sequence matches23/24 Qwen4B and46/48 Mistral measured requests; stock agreement is reported separately from native before/after exactness.

Six later component candidates (two within-CTA row partitions and scalar-store FP32 shared activations on both model families) pass numerical checks but lack material gains or regress, so none is adopted. The initial FP32 candidate had a compile-only shared-symbol type collision; its failed log is preserved, and the separately named repaired candidate passes without changing arithmetic. Disposition: `row-partition-float128-disposition-v1.json`.

V26 separately stages a Q32/KV8 attention chunk implementation. It combines gather/scale-softmax work across independent requests while keeping per-request QK/PV batchCount32 and actualT. It is only a candidate while full component/model/runtime/performance gates are pending; local inference source remains V25. Stock parity, broader workloads and latency tails remain open. No deployment, commit or push occurred.

Evidence: `v25-source-adoption.json`, `v25-runtime-and-serving-audit-v1.json`, `v25-full-components-controller-result.json`, `mistral-v25-model-controller-result.json`, `reuse-v25-serving-controller-result.json`.


## Adopted V26: Q32/KV8 exact Attention chunks

V26 extends the Q16/KV2 chunk structure to Q32/KV8/D128, used by Mistral-7B and Qwen3-4B. Independent singleton requests (chunks2..8,T1..512) share a paged KV gather and scale/softmax launch. Compact staging writes each real KV head once instead of repeating it four times. Device pointer tables occupy the unused tail of the existing K scratch; each request still calls QK/PV with its original batchCount32 and actualT. Original scaled-logit BF16 and probability BF16 rounding boundaries remain. There is no online FlashAttention replacement. Other shapes, prefill and longer contexts retain their existing path. Optional wide chunk scratch adds at most64.25MiB at the tested512-token bound; the actual per-arm cap remains22GiB.

The14,848-condition full paged Attention gate passes allT1..512, chunks2..8, mixed/paired lengths, alignment boundaries, reversed slots and noncontiguous physical page IDs. Context/probabilities BF16 and scaled-logit F32 are byte-exact; zero and restored-nonzero Graph replay and stable native allocation counters pass. This component seed expands actual Q16/KV2 boundary inputs deterministically; actual wide-model activations are independently covered by the complete model gates. The raw zero-counter JSON omits geometry and coincidentally has the same digest as the earlier Q16 report; `wide-attention-v26-component-provenance-v1.json` separately binds Q32/KV8 geometry to the compiled source, executed binary and process receipt.

Mistral-7B, Qwen3-4B and Qwen2.5-3B pass full-vocabulary byte comparison against the original same-device C1 references in C8 eager, C8 Graph and shrinking/reordered C32. Mistral/Qwen4B also match all104 selected fresh HF rows each. The historical Qwen2.5 cross-runtime HF difference remains pre-existing. CPU regression:481 passed,40 ordinary GPU unit tests ignored. Separate runtime traces prove3,600 Qwen4B and1,824 Mistral wide-gather calls. All208 and64 respective output files match their qualified outputs and original C1 bytes. Attention itself remains **outside** the existing pre/post Graph segments; the traces separately prove those segment Graph nodes still execute.

Matched V25→V26 RTX4090 fixed screens, one fresh process per arm, identical budgets and warm policy as V25 above.24/48 measured requests emit480/960 tokens. Both native windows have zero additional Graph captures; all logged response/recovery values remain exact and GPU ownership/resource audits pass.

| Model / concurrency | Before V25, tokens/s | After V26, tokens/s | Change | Stock vLLM0.27.1, tokens/s | After / stock |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B /8 |283.90|301.45|+6.18%|513.26|58.73%|
| Mistral-7B /32 |250.68|263.85|+5.25%|692.70|38.09%|

| Model / concurrency | TTFT p95 before → after, ms | Response p95 before → after, ms | Stock response p95, ms |
|---|---:|---:|---:|
| Qwen3-4B /8 |106.03 →108.31|749.15 →704.98|393.81|
| Mistral-7B /32 |1117.79 →1094.92|3425.39 →3227.62|1085.21|

TTFT p95 slightly worsens in the single Qwen4B pilot, and Mistral TPOT p95 changes119.08→119.51ms. Aggregate throughput and response p95 gains do not establish all-tail or statistical parity. Stock measured sequences match23/24 and46/48 reference sequences; stock is a throughput reference under its own numerics. These are not five-fresh-process or natural-workload results, and the broad vLLM parity goal remains open.

Five source files are adopted with pre-adoption backups; all932 local source hashes match the compiled snapshot. Evidence: `v26-source-adoption.json`, `v26-runtime-and-serving-audit-v1.json`, `v26-build-and-CPU-audit.json`, `wide-attention-v26-full-controller-v2-result.json`, `mistral-v26-model-controller-v2-result.json`. No deployment, commit or push occurred. A subsequent packed-input vector-transpose prototype is component-only while further qualification proceeds; it is not part of V26.


## Adopted V27: packed shared inputs with vectorized staging

V27 reduces shared-load instructions by consuming four BF16 rows with each64-bit shared load. Aligned128-bit global input loads, a4×4 register transpose within four-lane groups and bit permutations retain aligned128-bit shared stores. This avoids the scalar-staging overhead that made the first packed-input prototypes slower. An explicit inline PTX block preserves the original per-output FP32 round-to-nearest FMA sequence; the original half-tree addition and final BF16 rounding remain. A memory clobber preserves the shared-memory load's compiler-visible effects. No persistent weight copy, global GPU buffer or new hot allocation is added.

Every measured row count2..32 has its own component comparison. Minimum counts retain the faster existing path below the measured crossover: Mistral Gate/Up, Down and LM start at9, Q/O at3; Qwen4B Gate/Up at3, Down at4, LM at9, Q at2 and O at8. K/V keep the previous kernel. The same SM89/128SM/runtime12080/cuBLAS120804, singleton grouping and actual vector-alignment guards remain. The measured minimum tail speedup is8.29% or greater across every selected geometry/count, with most large tiles around1.3–1.6x; these component gains are not model throughput.

All478 actual projections pass19,598 conditions, plus zero and changed-nonzero Graph replay at8/16/32. Three-model C8 eager, C8 Graph and shrinking/reordered C32 full-vocabulary comparisons remain byte-exact to the original same-device eager outputs. Mistral/Qwen4B HF104 rows each also match; the historical Qwen2.5 HF difference remains pre-existing.481 CPU tests pass,40 ordinary GPU tests remain ignored. Private runtime traces prove every new template and include a separate Qwen4B C32 trace for the >=9-row LM path that a C8 trace cannot exercise.

Matched RTX4090 V26→V27 fixed-workload pilots: BF16,22GiB process cap, budget512, context1024, KV blocks64×concurrency, CPU sampling and unchanged Graph/batch flags. One fresh process per arm, full-reference power-of-two warm waves plus16 warm requests,24/48 measured requests. Both native measurement windows have zero new Graph captures.

| Model / concurrency | Before V26, tokens/s | After V27, tokens/s | Change | Stock vLLM0.27.1, tokens/s | After / stock |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B /8 |298.26|303.06|+1.61%|514.20|58.94%|
| Mistral-7B /32 |263.16|307.87|+16.99%|695.86|44.24%|

| Model / concurrency | Response p95 before → after, ms | Stock response p95, ms |
|---|---:|---:|
| Qwen3-4B /8 |716.80 →695.57|392.15|
| Mistral-7B /32 |3272.90 →2744.82|1093.99|

The Mistral gain is material in this pilot. Qwen4B C8's1.6% gain is modest despite faster isolated kernels; no statistical significance or all-workload improvement is claimed. The stock measured token sequences match22/24 Qwen4B and46/48 Mistral references, with equal480/960-token totals. Native values are exact in every logged response, full-reference warm wave and recovery request. All GPU owners and resource caps pass. Stock parity, five fresh-process fixed/natural comparisons and full latency-tail acceptance remain open.

Separate **actual HTTP measurement-window traces** (excluded from the throughput table) prove10,584 Qwen4B and6,662 Mistral new packed-kernel calls, including5,580 and640 Graph-node executions. These are actual server calls rather than teacher-only evidence. Qwen4B spends425.00ms in the packed L16/R8 template and119.19ms in L32/R8 over the profiled workload; Mistral spends846.42ms in L32/R32 and831.21ms in a CUTLASS GEMM family. These are sums of GPU kernel durations, not wall time. CUDA API durations overlap GPU execution and are not added to them. The traces confirm that faster individual projection kernels do not eliminate the broader GEMM and dispatch work.

The source inspection identifies another Graph coverage limit: owned segments currently require actual input/output row counts to equal the prepared power-of-two dense shape and reject mixed prefill. Thus zero new captures does not mean all operations replay a Graph. A bounded cache keyed by exact live row count is a possible follow-up; generic padded replay cannot simply reuse a Graph first captured for a smaller row count because the recorded native GEMM shapes would be stale. No such cache change is part of V27. Exact singleton runs inside mixed batches are already coalesced in `execute_row_groups`; they are not a missing V27 feature.

The later within-CTA two-row-group variant passes all451 component conditions and changed-input Graph replay but regresses against the V27 packed baseline; it is rejected. Raw data and the earlier rejected scalar-store/ASM candidates are preserved. V27 changes only one CUDA file; all932 local source hashes match its compiled snapshot, with the pre-adoption CUDA source retained. No deployment, commit or push occurred.

Evidence: `v27-source-adoption.json`, `v27-adopted-performance-summary.json`, `v27-runtime-serving-audit-controller-result.json`, `v27-HTTP-runtime-audit-v1.json`, `v27-candidate-plan.json`, `packed4-row-partition-g2-v1-disposition.json`, `v27-build-and-CPU-audit.json`.


## V28–V30 evaluated and deferred: Graph coverage and smaller row CTAs

The local implementation remains V27. None of V28, V29 or V30 is adopted. These finite candidates pass their numerical and runtime gates but do not establish a material server improvement. Their immutable source, binaries, raw logs, and failed controller attempts remain available under `artifacts/multimodel-parity-20261017`.

V28 adds fixed-owner Graph states for each exact live count 1..32, retaining legacy dense shapes above32 and rejecting unsupported partial layouts. Partial reuse requires synchronous metadata, identity slots, no prefill and mode2. Each live count gets its own captured GEMM shape; there is no generic bucket padding. V29 changes only the Qwen3-4B Down N2560/K9728 and output N2560/K4096 row counts17..32 to packed16-row CTAs, with adjacent row CTAs sharing weight access through L2. Per-output FMA/reduction/rounding is unchanged; register usage falls94→74. Component gains are10.6–25.8% for these selected counts. Mistral and other Q4 geometries are rejected by measured crossover and retain V27. V30 combines the two candidates.

V29 passes all478 actual projections and19,598 numerical conditions, including zero and changed-nonzero Graph input. V30 uses the exact same CUDA source hash and separately passes all1..32 live Graph states for Mistral and Qwen3-4B. Three-model full-vocabulary native comparisons are byte-exact. The two wide models also match all104 fresh HF rows each; the historical Qwen2.5 cross-runtime HF difference remains unchanged. V30 CPU regression passes484 tests;40 ordinary GPU tests are ignored, with the dedicated finite GPU gates above executed separately.

Matched Qwen3-4B/C32 RTX4090 screens, BF16/22GiB/budget512/context1024/KV64×C, one fresh process per arm,48 measured requests/960 output tokens. Compare before/after/stock within each row: V29 uses power-of-two warm waves, V30 warms every live count1..32. Both retain full frozen native response checks and zero new Graph captures during measurement.

| Candidate | V27 before, tokens/s | Candidate, tokens/s | Change | Stock vLLM0.27.1, tokens/s | Candidate / stock |
|---|---:|---:|---:|---:|---:|
| V29 | 471.74 | 479.68 | +1.68% | 1164.62 | 41.19% |
| V30 | 474.10 | 479.02 | +1.04% | 1160.95 | 41.26% |

V30 E2E p95 changes1767.72→1739.25ms, while TTFT p95 changes560.86→572.38ms; stock is633.85/254.77ms respectively. These single-process screens are not five-repeat statistical results or broad parity evidence. The V30-screen stock outputs have46/48 measured token agreements with HF; native outputs match48/48 and every logged full native oracle value.

Separate instrumented HTTP traces prove the new kernel executes2016 times in both candidates, including72 Graph-node executions in V29 and936 in V30. Graph coverage increases but total throughput barely moves. Profiling TPS is excluded from the table; CUDA API time overlaps GPU work. V30 native peak GPU memory rises13792→14080MiB, below22GiB. Admission, process ownership, disconnect/retry recovery and allocation checks pass. After finite cleanup the RTX4090 has253MiB/0% and no compute processes; A1002/3 remain foreign occupied and untouched.

The outcome is to retain V27 and investigate the remaining per-request QK/PV Attention calls. The staged next plan preserves QK BF16, the existing scale-and-BF16 boundary, native softmax reduction/exp and BF16 probabilities, and native PV arithmetic before any fusion. It is a plan only; custom Attention exactness and performance are not yet claimed.

Evidence: `v28-v30-final-evaluation.json`, `v29-full-components-controller-result.json`, `live-graph-v30-model-controller-v2-result.json`, `live-graph-v30-runtime-serving-audit-controller-result.json`, `v30-build-and-CPU-audit.json`, and `next-exact-attention-native-order-plan.json`. The first V30 controller retained a version literal28 and was rejected by exclusive-create before a GPU child launched; its failure is preserved. A separate V29 terminal-receipt filename mismatch was corrected with a byte-identical receipt alias, without rerunning the completed GPU experiment.


## Adopted V31: native-order QK/PV across requests

V31 replaces the per-request QK and PV calls for two fully identified native cuBLAS kernel families with CUDA kernels spanning requests and heads. The family selection is based on native kernel identity, launch geometry and register usage traced for all T1..512, rather than on approximate output agreement. QK uses16 strided FMA partials followed by their ascending serial sum; PV uses contiguous groups of4, stride128 and the descending32-way reduction. BF16 QK, the existing scaled-score BF16 boundary, softmax and BF16 probability, and final BF16 PV are retained. The full native families cover210 QK lengths and252 PV lengths. Other lengths retain native calls.

The existing SM89/128SM/runtime12080/cuBLAS120804, Q32/KV8/D128, request2..8, T1..512 and actual scratch alignment gates apply. Compact and expanded scratch layouts work without new global buffers. The Q16 path and projection kernels are unchanged. Attention remains outside the pre/post CUDA Graph segments in actual serving; fixed-T component Graph replay is validated separately.

All14,848 full Attention conditions match context/probability BF16 and scaled scores F32 byte for byte, including zero and restored Graph inputs. Cancellation and wide-exponent controls independently validate the family arithmetic. Three models match their same-device original native full-vocabulary results; Mistral7B and Qwen3-4B also match104 fresh HF rows each. The historical Qwen2.5 other-runtime HF difference remains unchanged. The pinned build passes481 CPU tests;40 ordinary GPU tests are ignored, while the dedicated gates above run on the RTX4090.

Matched finite serving pilots use the same RTX4090,22GiB cap,BF16,budget512,context1024,KV64×C,prefix off and fixed20 output tokens/request. Each arm uses one fresh process,24 measured requests at C8 or48 at C32. These are screening measurements, not five-repeat statistical parity claims.

| Model | Concurrent requests | V27 before tokens/s | V31 tokens/s | Change | Stock vLLM0.27.1 tokens/s | V31/stock |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 8 | 302.26 | 307.99 | +1.90% | 510.60 | 60.32% |
| Mistral7B | 32 | 309.37 | 316.65 | +2.36% | 697.98 | 45.37% |
| Qwen3-4B | 32 | 473.75 | 487.99 | +3.01% | 1178.57 | 41.41% |

E2E p95 changes713.09→683.51ms,2747.95→2700.48ms and1770.09→1732.77ms respectively. Peak native process GPU memory is unchanged at10310/19062/13792MiB. Full frozen native responses, process ownership, admission and disconnect/retry recovery pass; no Graph captures occur during measurement. Separate HTTP traces observe2628/4128/4680 new QK calls and the same numbers of PV calls, with zero new Attention Graph nodes. Profiled timings are excluded from the performance table.

The source adoption changes only `kernels/src/cublas_gemm_probe.cu`; all932 pinned source hashes match V31. No deployment, commit or push is performed. The stock gap remains substantial. Projection kernels and prefill dominate the remaining trace, so this change does not establish broad vLLM parity. Evidence: `v31-adoption-receipt.json`, `v31-comparison-table.json`, `v31-build-and-CPU-audit.json`, `wide-attention-v31-full-controller-v1-result.json`, `native-attention-v31-model-controller-result.json`, `v31-serving-audit-v1.json`, `v31-HTTP-runtime-audit-v1.json`.

Two additional FP32 vector-staging projection prototypes (KT128/256) pass902 actual-input numerical conditions and changed-input Graph checks, but every tested R32 geometry is slower than V27. They are rejected and their raw results retained (`float-vector-v1-disposition.json`).


## Adopted V32: paired output columns in exact projection batches

V32 shares one packed BF16 input read and conversion across two independent output columns. Each column retains its native serial FP32 FMA chain, lane reduction and BF16 rounding. Physical C4 blocks produce8 output columns. No weight repacking, new global buffer or hot allocation is introduced. Shape/count crossovers retain existing kernels where this variant is slower: Mistral Gate/Up and LM at17..32 rows, Down at9..24; Qwen3-4B Gate/Up at the actual outer dispatch minimum3..8 and Down at17..32. The R2 Q4 Gate micro is exact and faster, but the server retains its original R2 path. Q/K/V and output retain their prior V27 exact batching in V32. The V31 Attention suffix is byte-unchanged.

The exact production projection definitions pass all478 projections across every Mistral/Qwen4B layer,19,598 row/offset conditions and zero/changed-nonzero Graph replays at8/16/32 rows. Mistral7B, Qwen3-4B and Qwen2.5-3B native full-vocabulary values match the original C1 reference in eager C8, Graph C8 and shrinking C32. The two wide models match208 fresh HF rows combined. Historical Qwen2.5 other-runtime HF differences are preserved. The pinned build passes481 CPU regressions, with40 ordinary GPU tests ignored and the dedicated finite GPU gates executed separately. All932 local pinned source hashes match V32 after adopting one CUDA file.

Matched incremental V31→V32 screens use one fresh process per arm and the V31 fresh C1 eager oracle (45 requests/model, identical to the original V27 full responses). Hardware and limits are as in V31.

| Model | Concurrency | V31 tokens/s | V32 tokens/s | Change | Stock vLLM0.27.1 tokens/s | V32/stock |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 8 | 308.04 | 317.64 | +3.12% | 504.33 | 62.98% |
| Mistral7B | 32 | 314.09 | 327.87 | +4.39% | 687.72 | 47.68% |
| Qwen3-4B | 32 | 483.25 | 495.18 | +2.47% | 1171.64 | 42.26% |

E2E p95 improves692.92→674.50ms,2705.31→2608.80ms and1729.10→1679.09ms. TTFT p95 changes100.94→106.88ms,1024.45→1002.71ms and575.33→555.05ms; the C8 TTFT regression prevents any claim that all latency measures improve. These single-process screens do not establish statistical or broad vLLM parity. Peak GPU memory is unchanged10310/19062/13792MiB. Full native responses,45 correctness queries/arm, disconnect/retry recovery and allocation accounting pass, with zero measurement-time Graph captures.

Separate actual HTTP profiles observe4824/2365/1872 paired-column calls and2448/96/216 Graph-node calls. Attention QK/PV remains active and outside the existing Graph segments. Profiled TPS is excluded from the performance table. Source and binary provenance, admission, PID ownership and resource limits are audited.

The first serving attempt used the old V27 oracle with the new V31 before binary and was rejected by the runner SHA binding before any server launched. That failed receipt and empty output directory are preserved. The recovery creates actual fresh V31 oracles and new v2 result directories; it does not edit the old oracle SHA or weaken the runner. Evidence: `v32-attempt1-reference-binding-diagnosis.json`, `v32-adoption-receipt.json`, `v32-incremental-comparison-table.json`, `v32-build-and-CPU-audit.json`, `v32-full-components-controller-result.json`, `column-pair-v32-model-controller-result.json`, `v32-serving-audit-v2.json`, `v32-HTTP-runtime-audit-v2.json`.

Further private probes are not adopted. The native-shape prefill strided batch agrees in120/154 conditions but differs in34 initial BF16 comparisons, mainly small-N Key and some M32 shapes; failing raw output pairs are retained. A controlled candidate-only FP32 reduction flag probe is queued after all performance measurements, leaving the separate-GEMM reference untouched. A Q4 R16/C8 row-first paired-column micro passes246 conditions and changed-input Graph, with18–38% gains for selected Q/O/Gate/LM counts17..32 over V27. Its staged V33 candidate is unbuilt and unqualified as a model/server; Down and K/V remain on V32/prior routes. Micro agreement and speed are not whole-model evidence.


## V27→V32 combined comparison

This final screen directly compares the original V27 binary against adopted V32 (Attention plus paired-column projections) and stock vLLM. It is independent of the incremental V31→V32 screen above. Same RTX4090/limits, one fresh process per arm;24 requests/480 output tokens at C8 and48/960 at C32. Full frozen native responses and recovery/resource gates pass. No captures occur during measurement.

| Model | C | Before V27 tokens/s | After V32 tokens/s | Gain | Stock vLLM tokens/s | V32/stock |
|---|---:|---:|---:|---:|---:|---:|
| qwen4b | 8 | 302.46 | 318.58 | +5.33% | 512.50 | 62.16% |
| mistral | 32 | 311.82 | 330.00 | +5.83% | 686.05 | 48.10% |
| qwen4b | 32 | 474.36 | 501.51 | +5.72% | 1167.06 | 42.97% |

E2E p95 changes708.41→657.93ms,2762.72→2597.80ms and1764.26→1688.55ms. TTFT p95 changes103.53→105.48ms,1025.04→997.72ms and561.56→567.35ms. Q4 TTFT therefore remains a small regression in this single screen; broad latency improvement is not established. Native GPU peaks remain10310/19062/13792MiB. Stock HF measured-token agreements are22/24,48/48 and47/48; its performance is a throughput reference, not proof of identical numerical outputs.

Evidence: `v32-final-comparison.json`, `v32-combined-serving-audit-v1.json`, `v32-combined-serving-controller-result.json`. All source, binary and raw negative controls are retained; no deployment, commit or push. The vLLM parity goal remains incomplete.

The candidate-only FP32 reduction flag probe fixes none of the34 prefill mismatches. A separate diagnostic third result changes the original per-request precision flag without changing the oracle: exactly the34 failing conditions change under this control, while120 initially matching conditions retain their outputs. Some diagnostic outputs match the batched output, others retain residual differences. This narrows the issue to native precision/kernel selection sensitivity; full native kernel/reduction tracing remains required. No precision setting, reference result or equality criterion is changed in adopted V32. Evidence: `prefill-native-precision-diagnosis-v1.json`.


## Adopted V33: row-first paired columns for large Qwen3-4B batches

V33 generalizes the paired-column kernel with a compile-time grid-order flag. Existing C4 routes use false. The new L16/R16/C8/row-first route applies only to Qwen3-4B Q, Gate/Up, output and tied LM Head for17..32 rows. Down retains the faster V32 C4 route; K/V retain their prior route. Each output retains the native lane-strided FMA sequence, half-tree sum and BF16 rounding. Actual original weights are reused, with no extra global CUDA buffer. The existing SM89/128SM/runtime12080/cuBLAS120804 and alignment guards remain.

All478 actual projections pass19,598 conditions, including zero and changed-nonzero Graph replays at8/16/32 rows. Three models match original native full-vocabulary logits, and Mistral/Qwen4B match208 fresh HF rows combined. Historical Qwen2.5 other-runtime HF differences are unchanged. CPU regressions pass481 tests;40 ordinary GPU tests remain ignored with dedicated finite GPU gates run separately. All932 local pinned source hashes match V33.

Matched incremental comparison: RTX4090, BF16,22GiB cap, budget512,context1024,KV16/64×C,prefix off,fixed20outputtokens,one fresh process per arm. C8 uses24measured requests;C32 uses48. Frozen full native responses,45 correctness requests/arm, admission/PID ownership, disconnect/retry recovery and allocation checks pass. Zero Graph captures occur during measurement.

| Qwen3-4B concurrency | V32 tokens/s | V33 tokens/s | Change | Stock vLLM0.27.1 tokens/s | V33/stock |
|---:|---:|---:|---:|---:|---:|
| 8 | 319.30 | 322.11 | +0.88% | 513.07 | 62.78% |
| 32 | 498.09 | 528.41 | +6.09% | 1156.11 | 45.71% |

C8 is an unchanged-branch control: the new row route executes0 times. Its0.88% throughput difference is not attributed to the new route. C32 improves6.09%; E2E p95 changes1691.42→1579.58ms, while TTFT p95 changes552.48→556.87ms (+0.8%). Peak native process GPU memory is unchanged10310/13792MiB. This is a single-process screen, not five-repeat statistical or broad-model parity evidence.

Actual instrumented HTTP traces observe4061 new row-first paired-column calls at C32, block16×8/registers95, and0 such calls at C8. New row-first Graph-node calls are0 in both HTTP profiles: measured C32 work still hits the mixed-prefill Graph exclusion. Component and whole-model Graph numerical tests pass independently; the HTTP trace is not evidence of new row-first Graph reuse. Profiling TPS is excluded from the table.

The initial postprocessing collector looked for a v1 receipt while the producer wrote its inherited v2 suffix. All six GPU measurement arms completed. The original failed parent result and actual terminal receipt are preserved; recovery reads actual filenames and runs only the unstarted profiles/audits. No measurements are rerun and no receipt is edited. See `v33-serving-result-path-diagnosis.json`.

Evidence: `v33-adoption-receipt.json`, `v33-comparison-table.json`, `v33-build-and-CPU-audit.json`, `v33-full-components-controller-result.json`, `column-pair-v33-model-controller-result.json`, `v33-serving-audit-v1.json`, `v33-HTTP-runtime-audit-v1.json`. No deployment, commit or push.

The unchanged original per-request prefill GEMM can be captured: an11-projection/two-model native Graph screen passes all242 conditions, including unchanged, zero and rotated inputs. Large matrix wall-time gains are mostly1–5%; small K/V reach24%. These component measurements exclude the runtime wrapper costs and do not establish server Graph speed. A full mixed-prefill parameter audit is retained in `mixed-prefill-Graph-parameter-audit-v1.json`; no mixed-prefill runtime cache is adopted.

A separate full-layer Mistral Gate/Up native strided prefill screen passes all4928 conditions (64 actual matrices, per-request M2..32 with B2/4/8 where M×B≤128, original DEFAULT_MATH oracle, same/zero/rotated Graph checks). This narrows a promising shared-weight prefill route. Its native kernel identity map and measured crossover are prerequisites for any bounded runtime bridge and full-model/HTTP adoption. Passing controls do not qualify arbitrary M/B/hardware.


## Adopted V38: wide Attention chunks32 and one exact softmax launch

V38 widens qualified Q32/KV8/D128 independent decode chunks from8 to32 requests. The legacy H16 cap8 and wide context cap512 remain unchanged. Internal PV groups stay bounded at8; arbitrary logical row indices and leftover native fallbacks are preserved. Existing V33 projection routes are unchanged. Private V34b prefill batching is not adopted: large-M controls failed equality and the qualified small-M bridge had negligible serving benefit.

The integrated QK/PV families preserve actual native FMA/MMA order. Batched softmax copies the original PyTorch-derived body, replacing only virtual thread/block indexing. It retains WARP_BATCH, the width-specific XOR tree, BF16 score/scale/probability boundaries and all FP32 expressions. One kernel processes all requests in a chunk. No new global GPU buffer or hot allocation is introduced; wider cold scratch is reflected in the actual ledger. The SM89/128SM/runtime12080/cuBLAS120804 gate remains.

V38 passes484 CPU tests (40 ordinary GPU tests ignored),343 actual-CAPI/Nsight smoke conditions, and25,088 full paged Attention cases including all T1..512, wide request tails/cohorts, FP32 trace and BF16 context/probability bytes, zero/restored Graph replay and hot allocation0. Three whole models match their original C1 full-vocabulary native references. Mistral/Qwen3 match208 fresh HF rows; historical Qwen2.5 other-runtime HF differences are unchanged.

The following same-RTX4090 unprofiled pilot uses one fresh process per arm, identical settings/output counts,22GiB cap,budget512,context1024,KV16/64×C,prefix off.

| Model/workload | C | V33 tok/s | V38 tok/s | Change | Stock vLLM0.27.1 tok/s |
|---|---:|---:|---:|---:|---:|
| mistral fixed | 8 | 214.01 | 216.35 | +1.09% | 311.96 |
| mistral fixed | 32 | 326.18 | 336.99 | +3.31% | 692.96 |
| mistral natural | 8 | 251.87 | 256.42 | +1.81% | 348.60 |
| mistral natural | 32 | 400.74 | 416.17 | +3.85% | 869.76 |
| qwen4b natural | 32 | 579.02 | 622.52 | +7.51% | 1399.62 |

Three selected conditions additionally pass two fresh Riley-only before/after pairs. Including the pilot, median paired throughput gains are1.13% (MistralC8fixed),3.54% (MistralC32natural),7.51% (Qwen4BC32natural); all three pairs have positive throughput gains. Median paired p50/p95/p99 TTFT/TPOT/E2E changes are negative for each selected condition. This does not establish five matched stock repeats or improvement in every individual pilot latency metric. Stock vLLM remains faster.

Actual serving profiles observe request grid32 and one softmax launch per chunk. Qwen4B QK launches change16,848→6,480 and CUDA runtime kernel launches383,998→243,195 against the private V35 profile. The new PV family executes0 times for this Qwen workload, so no PV benefit is claimed. Peak profiled GPU memory is19,270MiB for MistralC32 and14,002MiB for Qwen4BC32, below22GiB. Ownership/admission, native response bytes, recovery and ledger checks pass. No compute process remains after teardown.

Existing pre/post Graphs remain active with zero measured captures. Actual new Attention/softmax Graph-node calls are0: explicit component Graph qualification is separate from runtime Graph reuse. No full-Attention Graph claim is made.

All932 pinned source hashes match the V38 qualified build before this documentation-only addition;5 runtime files changed and927 remained identical. Evidence and the complete table/latency caveats are in `artifacts/multimodel-parity-20261017/v38-final-report.md`, `v38-adoption-receipt.json`, `v38-runtime.patch`, `v38-confirm-audit-v2.json`, and `v38-final-resource-and-Graph-receipt.json`. No deployment, commit, or push.
