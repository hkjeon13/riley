# Serving optimization campaign — 2026-09-12

Status: active; a narrow c1 HTTP throughput win is established; the overall goal remains open. User goal authorizes implementation and remote measurement; supersedes the earlier plan-only status for this campaign.

Target: same model/hardware/workload throughput >= vLLM (goal +15%), TTFT/TPOT <= vLLM (goal -10%), plus high-concurrency P95/P99 correctness and stability. The initial SmolLM2 c1/p128/o32 cell is diagnostic and cannot close the overall goal.

## Current accepted baseline: batch7

Round14 completed all three prescribed C1 cells: **15 pairs, 30 fresh server processes and 30,000 strict retained responses**, plus 300 warmup responses. The controller then received SIGINT through its held pidfd and completed owned-process cleanup and verified Blender restoration. The original **30-pair C1/C2 plan remains unchanged and intentionally incomplete**. No C2 server launched. V3 reports the three complete C1 cells separately; it does not promote whole-campaign completion or acceptance. See [ROUND14_RESULTS.md](ROUND14_RESULTS.md) for absolute values, five-pair ranges, tails and grouped-frame interpretation.

**Batch8 is rejected; Batch7 API baseline remains accepted.** All five candidate/baseline pairs regress in throughput and median token TPOT. Paired medians show **-5.799% throughput, +7.305% TPOT and +6.200% E2E**, with effectively unchanged TTFT. Correct output alone did not establish an improvement.

| Round14 ratio direction | Throughput | Median token TTFT | Median token TPOT | Median E2E |
| --- | ---: | ---: | ---: | ---: |
| Baseline / vLLM | 1.037739 | 0.701692 | 1.054239 | 0.979405 |
| Batch8 / baseline | 0.942007 | 0.999462 | 1.073050 | 1.062003 |
| Batch8 / vLLM | 0.985882 | 0.693044 | 1.131935 | 1.036563 |

These are medians of five paired right/left ratios. The baseline/vLLM cell has process-statistic medians of **918.839 / 885.684 tok/s**, token TTFT **5.169137 / 7.366680 ms**, TPOT **0.948936 / 0.900157 ms** and E2E **34.634668 / 35.357063 ms**. The baseline's median token TPOT is still **5.424% slower** by paired ratio, despite better TTFT and some shorter tails. This is HTTP client token delivery, distinct from older engine timing. The +15% throughput target, TPOT parity and high-concurrency stability remain unachieved.

Round15 completed its separate paired combined-region diagnosis: eight fresh processes, 128 strict HTTP responses and 4,096 native replays. The directly timed 30-layer RoPE/KV + attention region took **360,448 / 359,936 ns** in the baseline and **422,400 / 422,912 ns** in Batch8 for AB/BA, respectively (**+17.188% / +17.496%**). Operator-off whole decode brackets were **0.943216 / 0.943296 ms** for baseline and **1.003232 / 1.009632 ms** for Batch8. Operator events increase whole decode by roughly **12–14%**; these are diagnostic intervals, not serving latency or proof of a particular SASS cause. [ROUND15_PROFILE_RESULTS.md](ROUND15_PROFILE_RESULTS.md) records boundaries, copied-log verification and restoration. Round14 remains the serving evidence for rejection.

Separate synthetic GPU row probes passed: attention **34,560 cases + 3 retained graph transitions**, precise operations **3,456 cases + 3 transitions**, with all **432,017 / 107,179 allocations** freed and zero live bytes or cleanup errors. Anchored projection V2 admitted five M1 anchors but rejected all ten M2/M4 children as unsupported; all **1,089 cases were skipped**, so arithmetic equality is unproven. Standalone candidates are in preparation. These component probes do not qualify scheduler/graph/server integration, full-model multi-request execution or serving performance.

A separate fixed-history HF eager/cache-off/B1 diagnostic captured complete FP32/BF16 logit arrays. Output index 8 has a BF16 tie between 1443 and 2341; index 29 selects 638 uniquely in both dtypes. This is independent HF evidence, not actual vLLM graph logits or proof that higher-concurrency outputs are wrong. The strict serving gate remains unchanged; [COMMON_PREFIX_LOGITS_RESULTS.md](COMMON_PREFIX_LOGITS_RESULTS.md) records the scope and need for direct vLLM graph capture.

Batch2 through batch7 remain accepted improvements; batch1 remains rejected as a standalone change. The following table preserves the **earlier Batch7 five-pair, 30-request HTTP/engine campaign**, separate from Round14's token API workload:

| Batch7 metric | Riley | vLLM in the same campaign |
| --- | ---: | ---: |
| HTTP output tok/s | 916.998713 | 894.053067 |
| HTTP E2E ms | 34.805809 | 35.423666 |
| HTTP first-text ms | 5.174485 | 7.437874 |
| Engine output tok/service-s | 939.925432 | 826.963420 |
| Engine TTFT ms | 4.624290 | 7.185962 |
| Engine TPOT ms | 0.948323 | 0.938211 |
| Engine E2E ms | 34.025157 | 36.401517 |

Batch7 versus batch6: HTTP throughput **+13.929%**, HTTP E2E **-12.037%**, engine TPOT **-14.548%**, TTFT **+0.044%** (effectively unchanged). All five paired HTTP throughput ratios are **1.012–1.068**, median **1.0315**. Ratio of campaign throughput medians is **1.0257**; this is a different aggregation. Engine paired TPOT ratios are **0.947–1.016**, median **1.0122**, so TPOT parity is not established.

These are five fresh AB/BA pairs with 30 retained requests per process. HTTP throughput uses a common retained wall interval; engine throughput uses summed service durations. vLLM engine rates span 748.8–861.8 tok/service-s because slower tail requests affect the sum; the rate is not the reciprocal of median E2E. In these historical campaigns, HTTP first text is not token TTFT and HTTP TPOT was not measured. Round14 subsequently measured separate token-ID delivery timestamps. Thirty-request process P95/P99 estimates cannot establish high-concurrency stability. See `batch7-comparison.json` and exact raw summaries.

## Preserved historical baseline

Remote clean source `/tmp/riley-g04-vllm-profile-source-260911` at `59f02242a2993b698d5f1c6b970b96bad66fb0a4` and `/tmp/riley-g04-vllm-profile-target` binaries are preserved. Existing local dirty model-loader and unrelated work are preserved.

Before this campaign, measured engine TTFT was 435.240 ms, TPOT 2.305 ms and E2E 506.709 ms versus vLLM 7.005, 0.909 and 35.549 ms. This historical result is distinct from the fresh campaign baseline below; it did not itself attribute time to GPU operators.

## Campaign sequence

1. Reusable paired runner and fresh equal-condition baseline; separate diagnostic graph GPU/host timing.
2. GPU execution batch: separate prefill/decode graphs in one resource owner, remove overwritten prefill GEMMs, remove decode no-op prefill kernels, bind fresh stage selection and graph implementation identity. Select only after the profile supports the execute bottleneck.
3. Verify baseline/candidate internal outputs, reference tokens, stage failures, cancellation/reuse, resource cleanup, and compare actual HTTP serving to baseline and vLLM.
4. Use new cost evidence to choose batched prefill/decode and serving concurrency improvements.

GPU condition: user authorized temporary stop and exact restoration of the three existing Blender processes during measurement, preserving GUI. Restore is required after measurement. GUI-retained 512 MiB idle condition remains distinct from canonical 256 MiB; all lanes require no foreign CUDA compute and <=48 C start.

## First batch implementation and qualification

Fresh HTTP baseline: five AB/BA process pairs, each lane five nonstream+stream warmups followed by 30 streamed requests. Median of process medians: Riley E2E **512.918 ms**, first text **441.425 ms**, observed output throughput **62.376 tok/s**; vLLM **35.438 ms**, **7.293 ms**, **883.731 tok/s**. This is the same GUI condition and a new consistent cooldown/warmup protocol, not a reuse of the old raw environment label.

Initial isolated CUDA event diagnosis: 612 nodes (607 kernel, 5 memcpy), 159 replays/request. Median graph span is 3.374 ms/prefill token and 2.321 ms/decode token. Host staging ~1.5 us and launch ~5 us are small; host wait overlaps GPU execution and is **not additive**. The first diagnostic prototype's event destruction was later moved before CUDA context restoration; preserve the prototype result separately and use the corrected script for subsequent profiles. Per-kernel time is not measured.

The candidate has a single resource ledger retaining a prefill and a decode DAG. It removes 210 overwritten canonical GEMMs per prompt replay and 210 prefill-only launches per decode replay. Fresh validated position chooses the native stage; the owned Rust API rejects a scheduler stage mismatch before execution and invalidates stale output. Arithmetic profile remains `vllm-smol-p128-v1`; graph implementation identity is `F102`.

Remote candidate source commit: `321581f97af9fa3aa6c386c82d96f8efd8a0f2e9`, a separate scratch snapshot. Local main was not committed/pushed. Source model-loader files remain unchanged.

Correctness qualification passed:

- 809 replays/source across six requests: original prompt, partial prefill, partial output, two diverse prompts and repeated original prompt.
- **85,458,536 raw bytes equal** (all logits/output/status records plus both initialized KV caches). Byte-for-byte comparison, not only token comparison. SHA256 `7088898a2a270cc702fc1e23fbaa47b5a7c598786c4221b1e0fb9de6c7851662`.
- Fixed Hello prompt still produces the exact 32-token vLLM reference; diverse prompt checks establish parity with the old Riley candidate, not independent new vLLM qualification.
- Additional 491-replay native test passes stage mismatch rejection, cancellation/reuse, explicit close and Drop; zero device/pinned allocations after close.
- Actual release HTTP CPU/GPU greedy streaming/nonstreaming, invalid shapes, disconnect during prefill/output and successful reuse pass.
- CPU runtime 260 tests, HTTP runner 14 tests and diagnostic runner 6 tests passed. CUDA/native release builds passed.

First batch HTTP comparison is complete and **rejects the standalone optimization**. Candidate Riley E2E 543.051 ms (+5.87%), first text 476.328 ms (+7.91%), output throughput 58.937 tok/s (-5.51%). Candidate vLLM control: 35.388 ms E2E, 7.349 ms first text, 880.670 tok/s. Each lane has 150 retained requests in five alternating process pairs. For that historical measurement, first text is not token TTFT and HTTP TPOT was unmeasured; c1 tails do not establish high-concurrency stability. Engine comparison also rejects it: E2E 507.587→533.352 ms (+5.08%), TTFT 435.859→466.682 ms (+7.07%), TPOT 2.314→2.152 ms (-7.00%), service-bound output throughput 63.040→59.995 tok/s (-4.83%). Each campaign has five pairs with 150 retained requests per lane. Second-capture-failure cleanup was reviewed statically; a dedicated injected failure test is still absent.

Corrected CUDA event profiles show prefill span increasing from 3.391 to 3.679 ms, decode decreasing from 2.323 to 2.159 ms. Node count decreases 612 to 402 per graph, yet the prefill GEMV overwrite kernels slow. First-layer Q canonical+overwrite baseline 4.096+9.216 us versus candidate overwrite 12.288 us; gate 6.144+6.144 versus 10.240; down 6.144+17.472 versus 27.648. Instrumentation changes timing and is excluded from serving performance binaries. Cache warming and occupancy remain hypotheses without hardware counters.

## Second batch: complete P128 prefill

Implemented and qualified on CUDA: parallel rows for existing exact arithmetic kernels, bounded 2.25 MiB scratch under the same graph resource ledger, versioned P128 metadata and strict stage validation, and scheduler/server/engine integration. The expected replay count is 32 per P128/O32 request (one prefill and 31 decode), confirmed by retained-owner tests. M1 numerical profile remains an exact parity oracle. Qualification passed: three prompts, 96 full initialized KV snapshots across prefill/every decode, full logits/status equality, six uninterrupted owner requests, 42 malformed stage/shape rejections, zero live CUDA allocations after cleanup, two legacy graph regressions, ten profile CLI tests, and release HTTP CPU/GPU sampling stream/nonstream/disconnect/reuse. Source snapshot `1bfb23053fb15a3528eb6413abe6e1bbdb0db004`. HTTP comparison complete: Riley median E2E **90.418 ms**, first text **23.651 ms**, observed output throughput **353.408 tok/s** versus vLLM **35.639 ms**, **7.612 ms**, **882.019 tok/s**. Compared with preserved baseline: throughput **5.666x**, E2E **-82.372%**, first text **-94.642%**. Five paired processes/lane, 150 retained requests/lane, exact validated output text. This is an accepted c1/P128/O32 improvement over Riley baseline, but remains below vLLM and does not establish high-concurrency serving success. Engine results: Riley TTFT **13.732 ms**, TPOT **2.154 ms**, E2E **80.499 ms**, service throughput **397.545 tok/s**; vLLM **7.099 ms**, **0.932 ms**, **36.031 ms**, **879.027 tok/s**. CUDA events: full P128 prefill graph span median **14.008 ms**, per-decode **2.176 ms**, one prefill +31 decode replays/request. Each graph has 397 kernel nodes; prefill adds a 1152-byte D2D final-row copy. Instrumented timings are diagnostic and excluded from performance binaries. The roughly 9.9 ms HTTP/engine difference motivated the completed batch3 below.

## Third and fourth batches: measured and accepted for c1

Batch3 freezes only the HTTP service change over batch2: Unix listener readiness with persistent shutdown wakeup, blocking idle worker receive, and explicit blocking mode for accepted sockets before setting timeouts. Linux server tests passed (56 passed, one diagnostic ignored), and the new release binary passed both sampling backends, full text/SSE, unsupported shapes, disconnect/reuse and graceful shutdown. GPU component evidence is explicitly reused from batch2 by unchanged source identity; it is not relabelled as a new GPU test run. Source `83cda03e183e29f48a82f7e7be3282c2e83ee186`; qualification SHA256 `43817c8330f4d42fe8ee489adacd1f192bc16f7043f685211068f057c206aca0`.

Batch3 completed five HTTP and five engine pairs. HTTP throughput is **394.541 tok/s**, E2E **80.974 ms**, first text **14.269 ms**: throughput **+11.64%**, E2E **-10.45%** versus batch2. Engine E2E **80.434 ms**, TTFT **13.716 ms**, TPOT **2.152 ms** remain close to batch2 (E2E **-0.08%**), separating the serving-path effect from GPU execution. The HTTP change is accepted; the CPU loopback measurements remain mechanism diagnostics rather than substitutes for these serving measurements.

Batch4 adds three decode changes over batch3: packed QKV GEMM, packed gate/up GEMM, and fused RoPE/paged KV publication. It retains 62 extra device buffers and two qualified M1 plans, adding 139,353,984 device bytes (including outputs). Packing occurs only during preparation and verifies uploaded bytes by readback. The graph identity includes actual packed content and complete selected plan metadata; the graph implementation is F104, while the numerical contract remains `vllm-smol-p128-v1`. P128 prefill uses the previous arithmetic path. Source `53677519082135c42a5d044044c5b31cad4a146a`; server SHA256 `8f47897a1c2651d89d88986ab5c9dbf9a8e08fe20796244bd127e3562b1301c6`.

The independent packed-projection feasibility experiment passed 1,350 full-output comparisons across 30 checkpoint layers and nine synthetic vectors; that experiment alone was not model qualification. The integrated batch4 subsequently passed all three prompts, full logits/status, 96 initialized KV snapshots including every decode position, six retained requests with cancellation/reuse and 42 malformed cases, zero live allocations, both legacy graph tests, ten profile CLI tests, and release HTTP correctness under both samplers. Those checks ran on the new source with Blender restored; they are correctness evidence, not timing evidence. Five-pair HTTP and engine comparisons supported adoption: HTTP **420.459 tok/s**, E2E **76.013 ms**, first text **14.319 ms**, engine **424.338 tok/service-s**, TTFT **13.730 ms**, TPOT **1.990 ms**, E2E **75.433 ms**. Batch4 improved HTTP throughput **6.57%** and engine TPOT **7.54%** versus batch3; it remains the preserved predecessor to batch5.

Separate batch4 whole-graph diagnostics measured a **13.717 ms** P128 prefill span and **2.011 ms** per-decode span, compared with batch2's **14.008 ms** and **2.176 ms**. Decode kernel count changed from 397 to 277. One prefill plus 31 decode replays still places most GPU execution in decode. These instrumented spans support investigating decode but do not identify which remaining operator dominates; they are excluded from serving-performance binaries.

## Fifth batch: measured attention improvement

The 15-case operator screen and full-30-layer attention follow-up completed. Attention sums to **1.366016 ms**, with **0.125–0.132 ms** whole-graph event perturbation disclosed; exact output and retained intervals passed. This evidence motivated batch5's packed-decode attention changes: one warp per query head, two output halves per head and reuse of unrounded FP32 exponentials in the MODE6 denominator. Original M1/P128 attention remains unchanged. See `OPERATOR_PROFILE.md` for the historical measurement scope.

Batch5 source `a791d63081dcdb319c8e45c79ec9301166ee199a` uses **F105 / g09-decode-attention-v1**, retaining the g04 numerical gate. Fresh integrated GPU checks passed full logits/status, 96 complete initialized KV snapshots, three prompts, six retained requests, 42 invalid cases, cancellation/reuse, zero live allocations, both legacy GPU paths and profile CLI checks. Fresh release HTTP passed both samplers, exact text/SSE, invalid-shape rejection, disconnect/reuse and graceful close. Qualification SHA256: `c3d7973f99d9464885e44e01bd9a68e71c469831a70bd3d64243d117f5d74e04` (`raw/batch5-qualification.json`). Its completed HTTP/engine performance comparisons supported adoption; the correctness receipt itself remains a pre-measurement artifact.

Batch5 HTTP measured **654.542363 tok/s**, E2E **48.725000 ms**, first text **14.285465 ms**, versus vLLM **894.322296 tok/s**, **35.144337 ms**, **7.225100 ms**. Engine measured **664.682203 tok/service-s**, TTFT **13.7251985 ms**, TPOT **1.110364 ms**, E2E **48.1432265 ms**, versus vLLM **867.947559**, **7.059853 ms**, **0.920084 ms**, **35.981384 ms**. Relative to batch4, HTTP throughput improved **55.673%** and engine TPOT **44.201%**; TTFT was effectively unchanged. These remain historical batch5 results.

Batch5 adds **no GPU device allocation or graph kernel nodes over batch4**; decode retains **277 kernels** and the earlier **139,353,984-byte** packed weight/output cost. Its separate whole-graph diagnostic summary measures prefill **14.275 ms** and decode **1.123952 ms**. That existing summary includes all six requests (6 prefill / 186 decode replays), without warmup exclusion, and is not serving evidence. It does not establish prefill regression: the uninstrumented engine TTFT is effectively unchanged.

## Sixth batch: measured M16 prefill improvement

Full 30-layer prefill profiling completed seven fresh cases with three warmups and three retained requests each. Gate/up measured 5.982 ms, down 3.236 ms, QKV 2.107 ms, O 1.275 ms and attention 1.180 ms, with disclosed event perturbation. Batch6 implements true M16 rows, five exact shape/rounding specializations and 1/2/4-warp launch geometry, selected only for packed P128 prefill. Original row/M1 kernels and decode paths remain unchanged. It introduces no GPU parents or allocations and retains 397 prefill / 277 decode kernel nodes and the earlier 139,353,984-byte packed-parent cost.

Clean source `67a4197b007d4ec6c9f7ad473bdf1c4f2a636111` uses **F106 / g10-prefill-m16-v1** with the unchanged g04 gate. All 630 real-GPU projection cases passed full-byte equality, finite output, guards, input/weight/position/oracle immutability and cleanup of 3,150 allocations. Fresh full-model logits/status/KV, retained cancellation/reuse/rejection, legacy GPU, profile CLI and both HTTP sampler checks also passed. Qualification SHA256 is `de72a7dadbda9a73a2e869c6981853c2e2946a6c84a4cac5e95cc73f753619d7`. The first CMake parser failure was preserved before native compilation or GPU execution; the passing probe uses the corrected, separately pinned driver. See `PREFILL_PROFILE.md` for exact proof hashes.

Five fresh HTTP and five engine pairs supported batch6 acceptance: HTTP 804.884386 versus vLLM 882.943510 tok/s, engine TTFT 4.6222465 versus 7.108036 ms and TPOT 1.109772 versus 0.936472 ms. Batch6 is now the preserved predecessor to batch7. Separate whole-graph diagnostics measured **4.789680 ms** prefill and **1.143168 ms** decode. The existing summary includes six requests and all 6 prefill / 186 decode replays, without warmup exclusion; it is not serving evidence.

The subsequent isolated residual-attention diagnostic completed after serving measurement. Across 93 retained decode replays, the full 30-layer attention sum was **0.482304 ms**. Whole decode was **1.148064 ms** off-before, **1.227936 ms** with attention events and **1.107456 ms** off-after; 60 extra event nodes add about **0.080–0.120 ms** of perturbation. No intervals were missing or failed. Attention remains material, motivating batch7's independent QK, two-warp PV and warp-0 softmax publication work. Batch7 subsequently passed correctness and serving comparisons as described below.

## Seventh batch: measured two-warp attention improvement

Batch7 partitions independent QK token groups across two warps, splits complete PV output blocks, and publishes exact lane-specific FP32 alpha/inverse from warp zero. The complete accepted attention source remains an unchanged oracle prefix. Packed decode alone selects the new wrapper; P128 and all device allocations remain unchanged. Source `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`, F107 / `g11-two-warp-attention-v1`, retains the g04 numerical gate.

The real GPU probe passed all 8,640 synthetic cases (30 layer seeds × 32 positions × 3 patterns × 3 KV mappings), full 576-word output comparisons against both prior oracles, mapping invariance, guards and immutable inputs, with 60,480 allocations freed and zero cleanup errors. Fresh full-model logits/status/KV, retained reuse/cancellation/invalid cases, legacy GPU, profile CLI and both HTTP samplers also passed. Qualification SHA256 is `4a12927e05a1596e548cda536469b79c273941f0586e25bf27c5760a38f149dd`. Exact proof identities and profile rationale are in `ATTENTION_BATCH7.md`.

Five fresh HTTP and five engine pairs support acceptance with the metrics at the top. Separate whole-graph diagnostics measured 4.779152 ms prefill and 0.961136 ms decode; the historical summarizer includes all six requests (6 prefill / 186 decode), including warmups, and is not serving evidence. Kernel counts remain 397 prefill / 277 decode; no GPU allocation was added. Subsequent residual profiling, concurrency screening and Round14 token-delivery measurements are preserved separately below and in the current snapshot.

## Concurrency screening and driver incident

The new closed-loop HTTP runner passed 17 CPU/loopback tests. Round12 prepared seven offered-concurrency/token-budget settings, then completed C1 with256 retained requests per lane and the Riley C2 lane with256. C1 rates were922.821 Riley and875.929 vLLM tok/s. Riley C2 remained at933.788tok/s while median E2E increased to68.452ms. The vLLM C2/budget128 lane failed its first two nonstream warmups against the unchanged c1 text/length gate, so the campaign stopped and the remaining five settings never ran. There is no concurrent vLLM performance ratio. See `CONCURRENCY_SCREEN.md` and the preserved raw failure/accounting.

A later ten-request response diagnostic captured actual payloads and IDs: all returned HTTP200 and32 output tokens; the same capacity-two vLLM server matched the old reference at offered C1 but produced different continuations at C2. ID-bearing C2 responses first differ at zero-based index8 or29. This establishes observed concurrency-dependent continuation variation, not a numerical root cause. The diagnostic has no timing claim, and its results do not retroactively qualify the failed screen. The subsequent seven-setting output matrix completed 123 requests, described in `VLLM_OUTPUT_MATRIX.md`: C2/budget256 matched all nine reference continuations, while C4/C8 still varied. Optional HTTP token observations subsequently passed a fresh CUDA build, actual model GPU/publication tests and 92 default/opt-in HTTP checks under both samplers; Batch8 passed its own fresh 92-check HTTP proof. Round14 then measured the qualified token API with the separate grouped-token V2 client. Old text-only receipts remain unchanged.

The batch7 residual operator screen separately completed all12 fresh processes and72 validated requests, with36 retained. Attention measured0.314368ms, gate/up0.244736ms and RoPE/KV0.107520ms across30 layers. Disabled operator-timing whole decode brackets were0.969760/0.970976ms. `DECODE_BATCH7_PROFILE.md` records full coverage, event perturbation and exact source/binary/trace identities.

At06:43KST on2026-09-12, host unattended upgrades changed NVIDIA user libraries to580.178.04 while the loaded kernel module remained580.173.02. Default NVML then failed before a diagnostic server could launch. Signed Ubuntu20260901 archive metadata supplied exact580.173.02 compute packages; they were extracted under `/tmp` and used only through a recorded child-process environment. GPU identity/version and owned vLLM library mappings verified the private old libraries. No host package installation, global library replacement or reboot was performed by these helpers. Matching GL packages were also extracted. Fresh offscreen EGL/GLX contexts passed under the pinned child environment with all Blender sessions unchanged; a subsequent full Blender restoration passed in round13; see the current session receipt below. `DRIVER_RUNTIME.md` records the read-only host audit, successful contexts and preserved failed attempts. See `raw/driver580173-runtime-20260901/` and `raw/vllm-c2-output-diagnostic-runtime/`.

## Measurement session restoration and preserved evidence

Round15 restored successors **2730220 / 2730294 / 2730392** on ports **9876 / 9911 / 9887**, with verified commands, GUI environment, private GL readiness and every observed vendor mapping pinned (`raw/blender-round15/verified.json`). The actual process recheck at **2026-09-12 12:29:26 KST** is saved in `raw/round15-live-restoration-recheck.json`. CUDA was loaded in the first process only; lazy loading in the other two is not a failure. Round15 finalization has no failure and all eight diagnostic servers were cleaned up. Earlier restoration identities remain in their own receipts.

Round13 stopped at retained request 124 because the V1 measurement client rejected a valid two-token SSE frame after 123 successful retained responses and ten warmups. No pair completed; raw data stays under `raw/token-serving-round13/`. `TOKEN_BENCHMARK_CONTRACT_V2.md` defines the separate grouped-token delivery revision used for Round14. All earlier measurement, profiling and restoration receipts remain preserved.

Existing authorization covers temporary stops of only verified successors of these three processes for the next diagnostic session, preserving GUI and restoring their recorded files, commands and ports afterward. The GUI-retained 512 MiB idle / no foreign CUDA compute / <=48 C start envelope remains distinct from canonical 256 MiB qualification. Legacy vLLM raw environment labels remain unchanged and uncanonical; actual conditions are bound by envelopes and live preflight.

Sixteen historical completed HTTP/engine campaigns retain **80 pairs, 160 process runs and 4,800 retained requests** under `raw/`, plus source/binary/proof hashes, test logs, diagnostic traces and restoration receipts. Round14 separately adds **15 completed C1 pairs, 30 process runs and 30,000 retained requests**; its original 30-pair campaign remains incomplete and is not added to the completed-campaign count. Large parity binary payloads remain remote with exact comparison hashes/manifests retained locally. The starting hashes of 24 model-loader files remain preserved; no local commit/push was made for this campaign.
