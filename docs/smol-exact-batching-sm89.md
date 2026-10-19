# SmolLM2 exact decode batching on RTX4090

## V58: exact batched Attention for every decode length T2..1024

V58 is adopted over V54. Existing exact QKV/Gate/Up projection batching and segmented CUDA Graphs are retained. The remaining Attention gaps came from the native PV floating-point reduction topology: the missing recipes use consecutive, dynamically sized chunks of ceil(T/L) keys per lane, followed by a descending half-tree reduction. The earlier fixed-chunk search did not cover those partitions. Native cancellation-triplet queries reconstructed four representative addition trees, with 49,152 independent held-out queries. This diagnosis is followed by direct BF16 comparisons; tree reconstruction alone is not an arithmetic qualification.

The new PV kernel preserves ascending FP32 FMA within each native lane partition, the native half-tree, and BF16 round-to-nearest-even boundaries. Strong standalone comparisons pass for all 243 newly supported lengths at R32, three HF activation snapshots used as arbitrary BF16 controls, and five states including wide-range and cancellation inputs. The recipe table now supports all 1,023 lengths T2..1024; T1 and singleton active rows retain the original route. The backend keeps four CUDA launches, compact KV storage, GPU row/position/page metadata, existing alias/extent/owner checks and zero hot GPU allocation. It remains gated to the qualified SM89/128-SM, CUDA runtime12080, cuBLAS120804 environment and its existing cold architecture/workspace checks.

| Concurrency | Workload | V54 tok/s | V58 tok/s | Gain | Stock vLLM tok/s | V58 / stock | Native fresh pairs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | fixed | 654.36 | 660.57 | +0.95% | 759.88 | 86.9% | 1 |
| 8 | fixed | 1585.79 | 1761.89 | +11.10% | 3172.87 | 55.5% | 3 |
| 16 | fixed | 1873.55 | 2134.32 | +13.92% | 4445.26 | 48.0% | 1 |
| 32 | fixed | 2049.34 | 2366.57 | +15.48% | 6292.01 | 37.6% | 3 |
| 8 | natural | 2030.28 | 2090.00 | +2.94% | 3963.73 | 52.7% | 1 |
| 32 | natural | 2720.31 | 2838.76 | +4.35% | 9879.11 | 28.7% | 3 |

These are unprofiled HTTP results on the same RTX4090, SmolLM2-135M BF16, matched requests and output counts, context1024, token budget512, prefix cache off, neutral greedy with original EOS, and a22GiB process GPU-memory cap. Fixed requests generate20 tokens; natural requests use32/64/128 budgets. C8 fixed, C32 fixed and C32 natural report medians of three fresh native process pairs; other cells and all stock cells have one fresh process each. C1 is an unchanged-route control. Native uses CUDA12.8.1/cuBLAS120804; installed stock vLLM0.27.1 uses Torch2.13+cu130. Stock outputs sometimes differ from HF; its performance is not a numerical-equivalence claim. The stock performance target across model families remains unmet.

Qualification:

- CPU484 pass/40 ordinary GPU tests ignored; separate admitted GPU qualification actually executed. Actual CAPI comparisons pass19,344 conditions, including every accepted length, changed inputs, all selected row counts, noncontiguous source rows, ragged contexts, real paged CSR mapping, output guards and native leases.1,824 raw CUDA Graph cases pass changed-position transitions between SIMT, WMMA and the new contiguous PV families.
- Smol104 fresh HF full-vocabulary rows match C1, C8 eager, C8 segmented Graph and all live counts1..32 with changed contexts and chunked prefill. Mistral7B and Qwen3-4B retain104 current HF rows and before/after logits. Qwen2.5-3B before/after logits remain exact; its historical cross-runtime HF discrepancy remains explicit.
-2,624 unprofiled native API values are checked across correctness, measurement, warm and retry calls for IDs, text, prompt IDs, finish and usage. Resource recovery, process ownership/start ticks, memory cap and zero measurement-time graph captures pass. The before oracle is actually recorded from V54; no reference hash or equality threshold is rebound.
- Actual paired graph-level HTTP traces confirm all four Attention stages and compiled resources. Ordinary GPU kernel sums decline866.46→829.59ms in the profiled workload; those sums exclude graph-internal nodes and are separate from serving throughput. Prior NCU counter-permission denial and node-level Nsight stall remain recorded.

Attention itself is still outside runtime segmented Graph capture. Raw Graph replay tests use fully allocated CSR pages and do not establish runtime correctness when the active CSR page count grows. A future full-forward graph must refresh dynamic bounds safely and qualify growing page lists. No full-Attention runtime Graph claim is made here.

The broader-model V59 component screens find dynamically balanced PV matches for218 H16/D128 lengths and166 H32/D128 lengths. They are research results, not adopted wide-model Attention support or serving speedups. V60 separately tests the existing GPU-greedy sampling option using the same V58 binary on both sides; those one-pair pilots are not included in this V54→V58 table and do not change the default sampling policy.

Evidence is retained in artifacts/multimodel-parity-20261017: v58-final-evaluation.json, v58-adoption-receipt.json, v58-runtime.patch, candidate-v58-manifest.json, candidate-v58-binary.sha256, v58-CAPI-controller-v1-result.json, v58-Smol-wholemodel-controller-result.json, v58-existing-model-regression-result.json, v58-serving-audit-v2.json, v58-confirm-audit-v3.json, v58-Smol-profile-graph-v3-audit.json, v58-runtime-activation-geometry.json, v56-controller-result.json and v57-controller-result.json. All932 private build files and the candidate executable hash were reverified before adoption; unrelated dirty source and failed candidates are preserved. A100 GPU2/3 are idle on the latest admission checks; this table remains a separate architecture-specific SM89 lane. No deployment, commit, push or soak was performed.

## V54: native WMMA QK extends exact Attention coverage

V54 is adopted after the V49 baseline. The native all-length trace identifies six QK and nine PV kernel families. The43 previously uncovered QK lengths within130..242 (even, excluding multiples of8) select native CUTLASS WMMA, rather than the SIMT FMA order screened earlier. A separately executed WMMA prototype matches native results for all43 lengths with R32, three actual HF activation snapshots used as arbitrary BF16 controls, and five states including wide-range and cancellation inputs.37 of those lengths already have qualified exact PV recipes. The runtime recipe table therefore extends from743 to780 lengths;244 lengths keep the original fallback. R1 remains unchanged.

The combined QK kernel selects the original SIMT recipes or a warp-local native-order WMMA path from GPU positions. Its grid remains8×9×R with128 threads. Launch bounds retain a40-register compiled kernel; warp-local shared A/B/accumulator storage adds8192 bytes. The cold WMMA architecture qualifier is required for new recipes, in addition to the existing SM89/128SM/runtime12080/cuBLAS120804/workspace/MPS gates. The backend retains four launches, compact three-head KV and all existing ownership, byte extent and alias checks. Attention remains outside the runtime segmented Graph.

| Concurrency | Workload | V49 tok/s | V54 tok/s | Gain | Stock vLLM tok/s | Native fresh pairs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fixed | 659.01 | 657.42 | -0.24% | 760.55 | 1 |
| 8 | fixed | 1543.17 | 1578.41 | +2.28% | 2816.59 | 3 |
| 16 | fixed | 1795.29 | 1834.23 | +2.17% | 4400.16 | 1 |
| 32 | fixed | 1938.51 | 2045.75 | +5.53% | 6543.76 | 3 |
| 8 | natural | 1946.46 | 2033.91 | +4.49% | 4079.81 | 1 |
| 32 | natural | 2589.22 | 2734.10 | +5.60% | 9807.18 | 3 |

Same RTX4090, SmolLM2-135M BF16, matched input/output counts, context1024, token budget512, neutral greedy/original EOS, prefix off,22GiB GPU-process cap. C8 fixed, C32 fixed and C32 natural use three independent native process-pair medians; other native cells and stock have one fresh process each. C1 is an unchanged-route control. Native uses CUDA12.8.1/cuBLAS120804; stock vLLM0.27.1 uses installed Torch2.13+cu130. Full loaded-library records remain with each arm. Stock response IDs/text sometimes differ from HF and are not claimed numerically equivalent to the native arms. vLLM remains faster; broad-model parity is not achieved.

Qualification and provenance:

- CPU484 pass/40 ordinary GPU tests ignored; separate admitted GPU tests actually ran. Real CAPI10,368 conditions compare every BF16 probability and guarded output against original per-request execution, covering all780 accepted lengths, selected all-row counts, noncontiguous source rows, ragged contexts, real CSR/pages and three input states.1,152 raw Graph cases include both SIMT→WMMA and WMMA→SIMT transitions after GPU position updates. Guards, native resource leases and zero hot allocation pass.
- Smol104 full-vocabulary fresh HF rows match C1/C8 eager/C8 segmented Graph and live counts1..32 with changed contexts and chunked prefill. Mistral7B and Qwen3-4B retain104 current HF rows and before/after logits; Qwen2.5-3B retains before/after logits while its historical HF cross-runtime difference stays explicit.
- The V49 before oracle was actually recorded in45 requests and its15 full values match V41 and current HF. No hash was rebound. Every native measured/correctness/warm/retry value, recovery allocation, owner PID/start tick,22GiB cap and zero measurement-time capture check passes. All nine repeated native pair gains are positive.
- Native captured cubins identify the formerly mismatching QK family with8 HMMA instructions and matching64-register native profile. The compiled V54 QK has both HMMA and FFMA instructions and its40-register/8192-byte resources match the actual HTTP trace. A direct Nsight Compute tensor-counter attempt was denied by driver counter permissions; static ISA is not mislabeled as a dynamic instruction count. `v54-compiled-QK-ISA-receipt.json` and `v54-ncu-controller-result.json` retain these distinct evidence levels.
- The geometry collector initially retained its V41-before assumption that no Smol batch kernels should exist. V49 already has those kernels. The corrected collector validates both arms and the new register cap against unchanged trace files; no GPU, value or tolerance gate was rerun or weakened. Graph-level missing graphNodeId values remain NULL.

Two rejected follow-ons are preserved. V50 fused direct paged QK/softmax/PV passes9,198 conditions/864 Graphs and whole-model tests but slows C8 fixed by5.4% and C32 natural by11.2%. Its profile increases the four-stage Smol kernel sum from approximately125ms to307ms in one fused kernel. V51 cooperative128-token shared-memory tiles also pass the numerical/model gates but still slow those two cells by2.4% and6.7%. Neither is adopted. A separate120-condition identical-head-input diagnostic finds no head-dependent PV output changes, so a head-alignment explanation is not established.

Evidence: `v54-final-evaluation.json`, `v54-adoption-receipt.json`, `v54-runtime.patch`, `candidate-v54-manifest.json`, `candidate-v54-binary.sha256`, `v54-CAPI-controller-v1-result.json`, `v54-full-evidence-result.json`, `v54-serving-audit-v2.json`, `v54-confirm-audit-v3.json`, `v54-Smol-profile-graph-v3-audit.json`, `v54-runtime-activation-geometry.json`, `v52-native-geometry-allT.json`, `v52-native-function-SASS-lookup.json`, `v53-controller-result.json`, `v50-native-pilot-audit.json`, `v51-native-pilot-audit.json` and `v55-controller-result.json` in `artifacts/multimodel-parity-20261017`. No deployment, commit, push or soak was performed; hardware lanes and all failed/rejected raw evidence remain separate.

## V49: exact Attention batching with singleton fallback

V49 adds real paged-KV Attention batching for H9/KV3/D64 on the qualified RTX4090 SM89/128SM, runtime12080 and cuBLAS120804 path. It builds on the existing exact QKV/Gate/Up singleton projection batching and segmented Graph mode. Independent singleton rows with supported lengths are collected across the active row map. Two to32 selected rows run compact KV gather, exact QK, original scale/BF16-RN/softmax arithmetic, and exact PV in four CUDA launches. KV scratch stores three KV heads, rather than repeating nine query heads. GPU slots, CSR offsets, pages and positions bind the real cache and source rows; output writes directly to each original row. No hot GPU allocation is added.

743 context lengths within1..1024 have accepted QK/PV recipes. The281 remaining lengths, longer contexts, multi-token groups, other geometries, unqualified hardware/runtime, workspace overrides and MPS overrides retain the original path. This is measured recipe coverage, not a universal floating-point proof. V45 adversarial controls, V46 stages and the new real CAPI/model gates constrain acceptance. The goal of broad-model vLLM parity remains open.

V47/V48 passed numerical gates but regressed C1 by approximately3%. They were not adopted. V48 reduces empty CTA scheduling with smaller fixed grids and token loops while preserving every per-element operation. V49 keeps those grids and only selects the new runtime path for at least two eligible rows; C1 and singleton tails use the original V41 attention. No equality tolerance, gold output, or failed result was changed.

| Concurrency | Workload | V41 tok/s | V49 tok/s | Gain | Stock vLLM tok/s | V49 / stock | Native fresh pairs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | fixed | 645.74 | 653.57 | +1.21% | 751.06 | 87.0% | 1 |
| 8 | fixed | 1351.55 | 1528.91 | +13.12% | 2810.19 | 54.4% | 3 |
| 16 | fixed | 1497.07 | 1786.67 | +19.34% | 4226.72 | 42.3% | 1 |
| 32 | fixed | 1570.12 | 1937.17 | +23.38% | 6670.55 | 29.0% | 3 |
| 8 | natural | 1446.83 | 1943.77 | +34.35% | 4041.97 | 48.1% | 1 |
| 32 | natural | 1644.37 | 2596.06 | +57.88% | 9249.56 | 28.1% | 3 |

These are unprofiled, same-RTX4090, same input/output counts, context1024, token budget512, original EOS, prefix cache off, neutral greedy, and22GiB process-cap measurements. Fixed requests generate20 tokens; natural workloads use the same32/64/128 budgets. C8 fixed, C32 fixed and C32 natural report three fresh native process-pair medians; other native rows and all stock rows have one fresh process. C1 is an unchanged-route control. Stock vLLM0.27.1 uses its normal Graph path, has separate HF agreement counters, and is not numerically equivalent to the native arms in every request. Native uses CUDA12.8.1/cuBLAS120804; the stock environment uses Torch2.13.0+cu130 and Transformers5.15.1. Effective loaded-library records are retained per arm.

Validation against the pinned V49 build:

- CPU484 passed,40 ordinary GPU tests ignored; separate admitted GPU qualification actually ran.
- Real CAPI9,198 conditions: all743 accepted lengths, three HF activation snapshots used as arbitrary controls, original/zero/rotated inputs, selected noncontiguous source rows, paged CSR/cache mapping and heterogeneous contexts. BF16 probabilities, guarded output and untouched rows all match original per-request execution. Native buffer leases and zero hot allocation pass.864 captured Graph cases replay new inputs and changed GPU positions/recipe families. These raw Graph tests do not imply runtime Attention capture.
- Smol104 fresh HF full-vocabulary rows match C1, C8 eager, C8 segmented Graph and live counts1..32 with chunked prefill. Mistral7B and Qwen3-4B104 current HF rows and before/after logits remain exact; Qwen2.5-3B before/after logits remain exact. Its historical cross-runtime HF discrepancy remains explicit.
- Every retained native correctness, measured, bucket-warm and retry value is checked for IDs/text/prompt IDs/finish/usage. Ownership/start ticks,22GiB cap, recovery allocation and zero measurement-time captures pass. See the final JSON for the total checked response count.
- Actual paired graph-level HTTP traces prove the new batch kernels and smaller grids. Existing segmented Graphs remain around Attention; Attention itself is still outside runtime capture. Profile kernel sums exclude graph-internal nodes and are not serving throughput. The unresolved historical node-level Nsight stall remains preserved separately.

Installed vLLM source uses QKVParallelLinear, MergedColumnParallelLinear for Gate/Up, and batched paged Attention. Riley retains the original projection and attention reduction/rounding when selecting exact batching. Remaining limits include unsupported Attention lengths, materialized probability/KV scratch, host work outside segmented Graphs and unchanged projection arithmetic; the table does not establish broad-model parity.

Reproduction: `v49-final-evaluation.json`, `v49-adoption-receipt.json`, `v49-runtime.patch`, `candidate-v49-manifest.json`, `candidate-v49-binary.sha256`, `v49-CAPI-controller-v1-result.json`, `v49-Smol-wholemodel-controller-result.json`, `v49-existing-model-regression-result.json`, `v49-serving-audit-v2.json`, `v49-confirm-audit-v3.json`, `v49-Smol-profile-graph-v3-audit.json` and `v49-runtime-activation-geometry.json` in `artifacts/multimodel-parity-20261017`. The geometry collector initially treated SQL SUM over missing graphNodeId values as zero. The corrected collector retains NULL and identifies capture exclusion from the source guard; see `v49-geometry-collector-correction.json`. No GPU/value gate was changed or rerun for that metadata correction. Earlier evidence is retained. A1002/3 were idle on the renewed admission checks; these results remain a separate SM89 lane because the arithmetic recipes are architecture-specific. No deployment, commit or push was performed.

## V41: exact singleton Attention scale/softmax fusion

V41 removes the standalone scale/mask launch for qualified singleton H9/KV3/D64 attention up to T1024. QK/PV retain their original cuBLAS parameters, and the softmax keeps the original warp arithmetic. The original scale0.125 and BF16 round-to-nearest-even boundary are applied at the input load. M1 has no masked future keys. Multi-token prefill, other shapes and unqualified hardware keep the original path. The existing cold SM89/128SM/runtime12080/cuBLAS120804 qualifier controls selection.

| Concurrency | Workload | V40 tok/s | V41 tok/s | Gain | Stock vLLM tok/s | Native fresh pairs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fixed | 633.69 | 650.70 | +2.69% | 761.90 | 1 |
| 8 | fixed | 1276.44 | 1354.36 | +6.10% | 2845.35 | 3 |
| 16 | fixed | 1399.06 | 1508.03 | +7.79% | 4526.86 | 1 |
| 32 | fixed | 1470.04 | 1578.83 | +7.40% | 6347.75 | 3 |
| 8 | natural | 1358.01 | 1448.09 | +6.63% | 3852.61 | 1 |
| 32 | natural | 1531.03 | 1645.88 | +7.50% | 9747.47 | 3 |

C8 fixed, C32 fixed and C32 natural use medians from three independent native process pairs. Other native rows and each stock row use one fresh process. In the repeated rows all paired throughput changes are positive; the medians of TTFT/TPOT/E2E p50/p95/p99 all improve. Stock vLLM remains faster. Stock response IDs/text sometimes differ from HF; this table compares matched prompts/output-count work, and does not assert stock numerical equivalence.

- 9,279 admitted GPU conditions compare every BF16 probability and guarded context output against unchanged V40: all T1..1024, three real HF activation snapshots used as arbitrary BF16 controls, original/zero/rotated inputs, new-input Graph replay, T1025/2048 and prefill/H8 fallback controls. Original reference bytes and source-bound executable hashes are retained. The initial harness incorrectly expected raw Graph replay to re-acquire input leases; the existing V40 baseline exposed this assumption before any arithmetic comparison. The corrected harness checks native CAPI leases directly and serializes Graph input updates; no value comparison or tolerance is weakened.
- Smol104 full-vocabulary HF rows, C1/C8 eager/C8 Graph and all live counts1..32 pass byte-exactly. Mistral7B and Qwen3-4B retain104 HF rows each and native equivalence; Qwen2.5-3B retains native before/after equivalence while its historical cross-runtime HF discrepancy remains separate. CPU484 pass/40 ordinary GPU tests ignored.
- The previous frozen serving values were not rebound by changing a hash. V40 C1 actually recorded45 fresh requests and all15 full values match the frozen V38 values, binding the new comparison to the real V40 binary.
- Actual paired C32 natural-workload graph-level Nsight traces show standalone scale/mask launches108960→2880, with106080 fused BF16-input softmax launches in the after arm and4680 CUDA GraphLaunch APIs in each arm. Ordinary kernel sums1680.11→1560.87ms exclude graph-node internals in this trace mode; do not compare those totals to earlier node-level traces or use profile throughput as server performance. Attention remains outside the runtime segmented Graph.
- One node-level Nsight run timed out during warmup, then failed graceful shutdown. Its exact experiment PID, start tick, owner and executable hash were verified before stopping only that owned process. Raw logs and diagnostics remain under `v41-profile-stall-v1` remotely. Both versions subsequently passed paired graph-level tracing. The node-level stall root cause is not proven; this report does not claim it is fixed.

The private V42 head-batch screen is complete and rejected for runtime adoption. It ran46,080 conditions for each of H9/D64, H16/D128 and H32/D128. Respective failed-condition counts were9,101,17,198 and21,956. These are counts of conditions with at least one changed BF16 value, not percentages of changed output elements. All raw failures are retained; geometry screens do not establish HF-model acceptance.

Two counterexamples were replayed with bit-identical original and packed Q/K/V/probability inputs, native kernel traces and exact rational dot products. H9/D64 QK at T8/R8 changes the native GEMVX rank5/block4x8/batch9 kernel to rank6/block8x8/batch72. A value changes bba2→bba3 exactly at the real-dot BF16 midpoint -325/65536. H16/D128 PV at T5/R32 changes GEMVX block32x2/batch16 to GEMV2N block128x1/batch512. A value changes b87d→b87e; the real dot is3.638e-12 above the midpoint and the original value is nearer. Input-packing byte comparisons pass. Native cuBLAS dispatch and accumulation topology, rather than a packing error, explain why increasing only the head count is insufficient for exactness.

Official Lt pinning was explored in V43: the examined candidates covered11/24 selected H9/D64 shapes and19/24 H16/D128 shapes. This is not a universal solution or an exhaustive rejection of every possible Lt algorithm. No V43 runtime change was adopted.

V44/V45 then tested explicit FMA partitions and reduction trees. The stronger V45 screen includes three actual HF seed snapshots, independent wide-range BF16 inputs, and cancellation triplets that reveal reassociation errors. Across all T1..1024 at R32, candidate arithmetic matched native QK for980 lengths and PV for780, with743 jointly covered lengths. The remaining281 lengths are not claimed qualified. Four profiled native functions were also found in retained original runtime cubins, with static/profile register counts168matching; ISA provenance alone is not complete arithmetic proof.

The optimized V46 QK/PV-only prototype passed15,330 conditions across those743 lengths, selected lengths at all counts1..32, and five input states. Eighteen standalone Graph cases, each with five changed-input states and three replays, also pass. Median CPU wall time for the pair of native QK/PV operations at R32 changes as follows:

| T | Original QK/PV us | Prototype QK/PV us | Factor |
|---:|---:|---:|---:|
|128|145.84|9.87|14.78x|
|512|176.47|20.64|8.55x|
|1024|217.36|39.34|5.53x|

These are component times, including host library/launch/queue cost, excluding KV gather, softmax, model and server work. CUDA event spans can also include host queue gaps. They are not Riley serving throughput or a vLLM parity result. The V46 prototype is not integrated or adopted. The next concrete work is private V47 integration with real paged KV, original softmax arithmetic, mixed-request mapping, safe CAPI ownership, unchanged fallback for ineligible cases, and then full-HF/serving gates. See `v47-exact-D64-backend-plan.json`, `v46-stage-probe-controller-result.json`, `v46-stage-micro-summary.json`, and `v42-native-headbatch-mismatch-diagnosis.json`. The complete multi-model/vLLM goal remains open.

Latest receipts: `v41-final-evaluation.json`, `v41-Smol-qualification-v2-controller-result.json`, `v41-existing-model-regression-result.json`, `v41-serving-audit-v2.json`, `v41-confirm-audit-v3.json`, `v41-Smol-profile-graph-v3-audit.json`, `v41-fused-Attention-profile-proof-v1.json`, `v41-before-serving-reference-binding.json`, `candidate-v41-manifest.json`, `candidate-v41-binary.sha256` and `v41-runtime.patch`.

## Historical V40 result

V40 extends the existing owned HF path with exact strided singleton projection batching and fixed-geometry RMSNorm. It is opt-in through `owned-hf-eager-bf16-v1`, `lt-mlp-v2`, and the existing segmented Graph mode. No production deployment, commit, or push is part of this qualification.

## Runtime changes

- Q/K/V forwarding now reaches the qualified K576, N576/192 batch recipe. Output, Gate, Up, Down and LM-head use the V39 recipe, now included in V40. This batches independent M1 operations; it does not replace them with an unqualified dense M>1 GEMM or concatenate QKV/Gate-Up weights.
- The six Lt geometries retain the native M1 heuristic (algorithm13, tile0, stages0, splitK1, reduction0, zero workspace), row counts2..32, and actual 16-byte pointer alignment. Selection is cold-gated to SM89, 128 SMs, CUDA runtime12080 and cuBLAS120804, with workspace/MPS overrides absent. Other projection cases retain the original route.
- Smol singleton RMSNorm groups up to32 independent rows in one launch, with one 128x1 CTA per row. It reuses the unchanged original C1 kernel body and reduction order. Multi-token request groups keep the original prefill geometry; neither BF16 rounding boundary changes. RMSNorm execution is qualified here on the same RTX4090.
- Existing Graph buckets1/2/4/8/16/32 are retained. Non-power-of-two live counts use the existing eager fallback. Attention remains outside the segmented Graph.

## Unprofiled serving comparison

SmolLM2-135M, one RTX4090, context1024, neutral greedy sampling, original EOS, prefix cache off, matched input/output counts and 22GiB GPU-process cap. Fixed requests generate20 tokens; natural requests use the existing32/64/128 budgets. Native comparisons use the same owned HF profile and frozen V38 eager values. Stock vLLM0.27.1 uses its normal Graph path; it is not modified.

| Concurrency | Workload | V38 tok/s | V40 tok/s | Gain | Stock vLLM tok/s | Native fresh pairs |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fixed | 632.76 | 633.06 | +0.05% | 763.94 | 1 |
| 8 | fixed | 831.92 | 1268.14 | +52.44% | 2770.87 | 3 |
| 16 | fixed | 837.62 | 1399.39 | +67.07% | 4421.22 | 1 |
| 32 | fixed | 822.91 | 1466.93 | +78.26% | 6642.83 | 3 |
| 8 | natural | 889.19 | 1353.89 | +52.26% | 4019.08 | 1 |
| 32 | natural | 868.91 | 1527.43 | +75.79% | 9778.68 | 3 |

The selected three rows report native throughput medians from three independent process pairs. Other rows have one pair. Stock has one fresh process per row; this is not a five-restart parity result. C1 is an unchanged-route control and its small variation is not an optimization gain. In the three repeated rows, every paired throughput improves and the medians of TTFT, TPOT and E2E p50/p95/p99 all improve. vLLM remains faster.

## Numerical and resource evidence

- CPU:484 passed,40 ordinary GPU tests ignored. Separate admitted GPU harnesses were run against the pinned V40 libraries.
- Standalone Lt qualification inherited unchanged V39 arithmetic:241 actual projection matrices,9881 conditions, changed-input Graph replay. V39 actual Rust/CAPI4338 conditions are retained; V40 rechecks108 conditions across all six shapes, plus full-model gates.
- RMSNorm:61 actual scale tensors, all counts1..33 and64, mixed prefill/singletons and two base offsets:4392 conditions, each compared byte-for-byte to original per-request calls for original, zero and rotated inputs, including Graph replay, guards, resource leases and zero hot allocations. Projection activation snapshots serve as arbitrary finite BF16 norm inputs; true intermediate behavior is checked in the whole-model gate.
- Smol:104 full-vocabulary HF reference rows and V38 C1 logits match exactly, with C1/C8 eager/C8 Graph and live counts1..32. Existing six Graph buckets capture/replay360 segments; non-power-of-two execution is not claimed to be Graph-backed.
- Mistral7B and Qwen3-4B:104 current same-device HF rows each and before/after logits remain exact. Qwen2.5-3B before/after logits remain exact; its historical HF reference discrepancy is pre-existing and is not relabeled as a pass.
- Strict request-value checks include token IDs, text, finish reason and usage, plus warmup/retry checks. GPU ownership/start ticks, 22GiB cap, measurement capture count0 and recovery allocation checks are retained. Additional repeat1 processes deliberately omit the runner's separate repeat0 correctness/recovery phase; all their measured and warm values are still compared to the frozen reference. The initial CPU collector assumption failure is preserved and corrected without a GPU rerun.

## Actual HTTP kernel activation and remaining bottleneck

The C32 natural-workload Nsight trace proves native projection GEMV gridZ32 for both Q-width576 (gridX144) and K/V-width192 (gridX48), including Graph nodes. The fixed RMSNorm gridX32/block128x1 is also observed. Relative to V39, full-trace RMSNorm launches fall from218624 to14802 and kernel time from432.76ms to41.12ms. Total kernel time falls from2811.50ms to1929.70ms. These traces include warmup, correctness and recovery; use the unprofiled table for throughput.

KV gather remains approximately107510 launches/284.96ms and scale/mask approximately108950/132.68ms. QK/PV and softmax still execute per request. Installed vLLM source instead shows concatenated QKV/Gate-Up linear modules and FlashAttention with batched token metadata and paged KV. Its reduction/rounding behavior cannot be substituted without the same exactness gates.

V41 has now qualified and adopted this D64 singleton fusion; see the latest section above. Broader model/architecture coverage, attention batching and stock parity remain open. The classic Smol `native-preset` C8 discrepancy is also outside this owned-profile change and remains recorded.

## Reproduction receipts

Evidence directory: `artifacts/multimodel-parity-20261017`. Entry points: `v40-final-evaluation.json`, `v40-Smol-qualification-controller-result.json`, `v40-existing-model-regression-result.json`, `v40-serving-audit-v1.json`, `v40-confirm-audit-v2.json`, `v40-Smol-profile-audit-v1.json`, `v40-runtime-activation-geometry-v1.json`, `candidate-v40-manifest.json`, `candidate-v40-binary.sha256`, and `v40-runtime.patch`. Original failed collectors and raw measurements are preserved. A100 GPU2/3 remain a separate hardware lane, occupied by foreign work; all experiments here are RTX4090 results.
