# Batch8 RoPE/KV/attention fusion candidate

Status at2026-09-12 11:53KST: standalone, full-model GPU and actual HTTP correctness passed. Round14 serving measurement is active; all five C1 candidate/baseline pairs regress, with throughput ratios0.9419–0.9428 and median TPOT ratios1.0727–1.0735. Finalization, original-record analysis, direct vLLM comparisons and C2 remain unfinished. Batch7 remains the accepted serving baseline. No timing is collected by the synthetic probe.

The completed batch7 operator screen measured 30-layer RoPE/KV at 0.107520 ms and attention at 0.314368 ms. These separate instrumented-family medians motivate fusion but are not additive predicted serving savings.

The coherent batch has three changes: compute rounded current Q/K in CTA-local storage, bypass global current-token KV loads inside attention, and publish the existing Q scratch and current KV exactly once. It combines two launches per layer into one, reducing expected decode kernels from 277 to 247. It preserves grid9x2 /64 threads and all prior allocation costs, adding 256 bytes of shared storage per CTA and one barrier. Rotation is repeated across CTAs, so the net performance effect still requires serving measurement.

The exact first RoPE output is `fma(a,c,-mul(b,s))`; the second is `fma(a,s,mul(b,c))`. Explicit non-FTZ PTX operations and packed RN BF16 table conversion preserve the precise TU under the attention TU fast-math flags. The original experiment failed on signed-zero differences; both corrected v2 variants passed 436 real GPU cases. These arithmetic experiments remain distinct from fusion qualification.

The candidate preserves the complete 13,564-byte accepted numerical source prefix and the precise TU unchanged. Source is `8329c1aeec6e013f581128888c536e15f8bf7300`, direct child of optional-token HTTP source `a179617070526068b66ba5627ba82a7151da8c64`. Only four kernel/runtime files differ from that parent; F108 identifies the implementation and the g04 numerical gate remains unchanged.

## Actual standalone GPU evidence

The probe links the full precise RoPE TU with its exact production CMake options and the full candidate attention TU with its separate fast-math options. It uses 8,640 deterministic synthetic cases:30 layer seeds, positions128–159,3 finite BF16 patterns,3 physical block maps. Tables include signed zeros, subnormal values and BF16 rounding boundaries; they are synthetic, not checkpoint trigonometric tables or activations. Current K/V slots start as NaNs in both independent paths.

All cases passed byte-for-byte comparison of attention1152 bytes, rotary Q1152 bytes and complete physical K/V pools98,304 bytes each. Every inactive KV word remained unchanged, current V matched raw input, every guard and input was unchanged, and the oracle was re-read after the candidate to prove immutability. Q/attention were invariant across physical maps. All103,680 allocations were freed with zero live bytes/errors.

GPU UUID was `GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`, SM89, runtime13.0 and the private580.173.02 driver matching the loaded kernel. The three Blender sessions stayed unchanged. These checks establish correctness of the exercised synthetic cases; real model logits/KV and cancellation/reuse passed separately below. Serving wire behavior and actual throughput/latency remain separate gates.

## Immutable evidence

- `raw/batch8-build.json`: `629418e19f129436d9ee2d2a753a61317c5b8527e3f4fe438b9f9aa59a8392c6`
- `raw/batch8-fusion-probe/receipt.json`: `d18ee79cf87321ba7207bc2f130eb1c335daf92d83d3bedd45098c7a46e568ec`
- `raw/batch8-fusion-probe/compile.json`: `4adfa1561e00a8f2064f0064e4a6c4f6b04ed5cbb6f30882349fa0916c11ad5c`
- `raw/batch8-fusion-probe/execution-context.json`: `edc6a6c70b4ec298cd79622485b447136b2b8accead5b777d9650236c45b0e9f`

## Fresh full-model qualification

All four actual GPU tests passed on this exact build: three prompts with full logits/status and 96 entire initialized KV snapshots, retained reuse/cancellation/rejected-output invalidation with zero inference allocations, and both legacy reuse checks. The release CUDA server library passed 72 tests (2 intentionally ignored), and the profile suite passed 10. The qualifier revalidated the complete source lineage, unchanged four HTTP files, model/runtime/UUID, all raw test logs and the standalone fusion proof. This is not a serving performance result.

- `raw/batch8-model-tests.json`: `7f22786bf2393b5f695bf29f3c5cb3fe583fcfaee18c5fd4359a6dda523e9201`
- `raw/batch8-qualification.json`: `a2460fce858f02b96ef065ec352299278d1f90f1d59278838cb9c2f50f403f35`

## Fresh HTTP proof

The batch8 child V2 helper completed92 actual checks on this exact8329 source and4c087 binary: both CPU/GPU-greedy samplers, default/opt-in streaming/nonstreaming, O32/O1 and the real three-token stop prefix, offeredC1/2/4/8 with active1, deliberate disconnects and exact subsequent reuse. The proof preserves the g04 profile; C02 was not enabled and no per-request scheduler audit or native C02 shutdown receipt is claimed. It revalidates the model/fusion qualification, unchanged four HTTP source files and publication unit tests.

`raw/batch8-http-token-observation-correctness/completion.json` SHA256 `95966a1c73374fecbe9a68b5f6911aea818f75760d6fa15a290f8fb8874b2ebd`. The first child preflight rejected a runtime-library symlink alias although its bytes matched; V2 retains the pinned qualifier validations and removes the redundant incompatible path-format comparison. The original helper and failure log remain intact.

## Prepared post-measurement diagnostics

The next diagnostic compares one combined RoPE-through-attention event interval on the baseline with the fused call on Batch8, using fresh AB/BA processes and timing-off brackets. Adding the older separate-family medians would not answer this comparison. Static source inspection identifies testable costs: repeated rotation across CTAs, the extra shared-memory barrier, and current-token selection in every QK/PV loop. In particular, historical tiles need no current-token selection, while the tail tile does. These are hypotheses only; no SASS load attribution, new arithmetic variant, or further performance gain is claimed before the paired profile runs.

The local batch8 decode profiler wraps the fused RoPE/attention call once, with its arguments/order unchanged, and preserves the existing synchronization and event cleanup. It exposes nine decode families plus timing-off brackets. Prefill projection instrumentation is rejected because the inherited hook does not cover the active M16 path. Ten focused CPU tests passed; this is not a CUDA build or an actual profile.

`profile_decode_operators_batch8.py` SHA256 is `139577ccb86d14611b91b005bc4216389021fcf48c1ce70508301267cfb674d6`; its expected instrumented graph source is `604b99c668729667cd39ef3d42f3fcd5c0dfd8088cc8f3277ae4d6f85b27f372`. The separate builder `build_decode_profile_batch8.py` is `c21db7598264a10735b9f38fd3dba5c69233f63fa94ebf2a00c5bafe7380b67a`. It requires the pinned Round14 plan, matching preparation, verified Blender restoration and an exit receipt for every launched process before cloning/building. It verifies those receipts again after building. Five focused gate tests passed and independent review cleared the lifecycle checks. This builder has not run remotely; no diagnostic GPU work overlaps Round14 serving measurement.
