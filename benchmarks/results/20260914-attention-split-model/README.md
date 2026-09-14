# Context-split model integration: numerical screen passes, strict generation fails

The optional Rust → C ABI → CUDA profile now uses FP32 context splitting for both pure decode and decode rows inside mixed iterations. **Do not promote it: 168/1,024 independent generated tokens differ from the matched exact baseline.** No serving comparison was run and the overall vLLM goal remains unmet.

## Implementation

The graph recorder retains a separate 2,613,252-byte allocation containing partials, mixed-row Q/output buffers and metadata. Parent context, retention, byte extent and mutable/weight alias checks precede capture. The profile identity hashes both headers and labels fixed 32-way splitting explicitly. Existing factories select no split workspace and keep their previous behavior. The experimental factory uses the same projection/prepared FFN/adaptive decode configuration as its baseline; only decode attention changes. Prefill attention remains existing arithmetic.

The mixed adapter gathers decode queries and page maps, runs the same QK producer and FP32 partial/merge as pure decode, then scatters decode results. Existing mixed attention skips those decode rows and continues prefill work. Kernels run on the same stream and scratch is owned through graph teardown. The fixed split count intentionally exercises candidate arithmetic even at short context; this is a numerical control, not a measured production dispatch policy. No server flag or default promotion was added. Buffered/paired execution and long-context model/serving coverage remain unverified.

## Model results

RTX 4090, pinned SmolLM2-135M BF16 checkpoint; eight frozen natural passages, 32 prompt plus 32 target tokens each. Rust produces all logits; a separate offline HF eager FP32 evaluator with TF32 disabled calculates the reference. Full model/tokenizer/config hashes and evaluator package versions are in the archive.

| Metric / gate | Matched existing profile | Context-split v2 | Outcome |
| --- | ---: | ---: | --- |
| Target NLL, nats/token | 2.9592894316 | 2.9579495937 | Relative gate passes |
| KL(FP32 reference || engine) | 0.0007591519 | 0.0007398091 | Relative gate passes |
| FP32 argmax agreements | 244/256 | 241/256 | Descriptive, not a waiver |
| Strict free generation | Reference | 12/32 requests, 168/1,024 tokens differ | **FAIL** |
| Repeated-prompt batch invariance | Pass | Pass | Pass |
| Natural and generation full-model memcheck | Included | Included | 0 memory errors |
| vLLM serving performance | Not measured | Not measured | No qualification |

The raw BF16 logits are included (25,165,824 bytes per lane). The candidate differs in 10,173,377 BF16 logit elements in the Rust observation log. Lower NLL/KL on this small screen does not establish general quality, and numerical differences are not by themselves evidence of worse language quality. The existing strict acceptance requirement is unchanged and remains failed.

## Failures and correction

V1 changed pure decode only. Its generation run failed candidate batch invariance before printing its final summary, so its controller stopped and restored Blender. V1 logs, logits, source snapshot and build receipt are retained. V2 connects mixed decode to the same calculation; repeated-prompt invariance then passes. V2 strict generation still fails. These separate failures are not recast as successful tests.

V2 natural test exits 0; generation exits 101 for the preserved strict assertion. Natural memcheck exits 0; generation under memcheck exits 101 for the same assertion while reporting **0 memory errors**. Its token arrays exactly match the unsanitized generation. The original V2 metrics command exited 2 because the standalone evaluation script was missing remotely; the completion stage copies the evaluator and evaluates the existing logits successfully. Collection success is not model acceptance.

The first standalone mixed-mapping probe failed compilation because its fixture omitted `math_constants.h`; the failed compiler log is preserved. After adding that fixture include, SM89/SM90a/SM100a probe builds pass. Three replays change ragged decode counts 17/32 to no decode and back, preserving prefill sentinels and exact constant-value outputs. Its bounded memcheck and racecheck both report zero errors/hazards. Only SM89 executes; the other probe architectures have explicit hardware-absence runtime skips. The full model build/runtime evidence is SM89 only.

## Verification and reproduction

The [verifier](../../analysis/verify_context_split_model.py) reconstructs two bounded archive parts, checks all 46 regular members against hashes, current integration-source hashes, logit hashes/counts, fixed input tokens, metric arithmetic, execution statuses and sanitizer summaries. It independently recomputes generation differences and repeated-prompt invariance from retained token arrays, and confirms unsanitized/memcheck equality. See [verification.json](verification.json) and [part manifest](evidence/manifest.json). The original archive SHA256 is `a84612c8ab0228cd93b783648dd5ec6fb009606d5413fe658ae3108fa8539b1f`.

The archive includes both source snapshots and the three lifecycle controllers. V1 binary hashes were not captured before the test executables were rebuilt at the same paths; do not treat the final binary hashes as V1 provenance. Final V2 binaries and model files have separate digests. Raw logits and all failures are retained, not replaced by summaries.

Local runtime unit tests passed 355, with one existing ignored test, before the mixed-mapping correction. A final local runtime check and the final CUDA model build passed afterward; the earlier unit suite is not reported as a repeated final-tree run. `git diff --check` passed. All three Blender scene RPCs succeeded on restoration, and viewer loopback ports 31840/31970/32010 returned HTTP 200. This is not a new public-browser visual verification.

## Decision

Retain the profile as an explicitly unaccepted numerical experiment and reference control. Do not infer a serving gain from its earlier synthetic long-context P×V timings, and do not relax generation thresholds to admit it. Model integration uncovered and corrected a real mixed/pure execution inconsistency; the remaining strict numerical divergence blocks promotion. Further performance work must preserve the accepted target behavior or satisfy a separately approved numerical policy; this result does not establish such a policy.
