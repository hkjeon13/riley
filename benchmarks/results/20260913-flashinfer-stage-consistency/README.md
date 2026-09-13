# FlashInfer decode arithmetic across mixed and pure stages

2026-09-13; base `91e1e077c9220b3d4c21b2ee7596b90f9a2f7b85`.
**The previously observed 16 token mismatches and scheduling-history counterexample are resolved on the existing fixture. This is not general numerical acceptance or serving performance proof.**

## Controlled evidence and intervention

Before this change, pure decode used FlashInfer and mixed-stage decode used the existing Riley attention. The 32-request fixture used identical prompts and teacher-forced reference continuations but produced 16 reference-token mismatches and different tokens across requests at the same generated index.

A diagnostic scheduler policy separated prefill and decode while retaining the old experimental model implementation. Both the exact baseline and FlashInfer produced 4096/4096 reference argmax matches, with zero cross-request token or full-logit differences in 3968 comparisons. FlashInfer logits still differed from the baseline (max absolute error 0.78125, RMSE 0.11587948). Separating stages is a control experiment, not a proposed serving scheduler optimization.

The implementation now applies the same unsplit FlashInfer arithmetic to decode rows inside mixed batches. Under the original mixed scheduler it produces:

| Case | Mixed / pure decode iterations | Reference argmax | Cross-request token differences | Cross-request full-logit differences |
| --- | --- | --- | --- | --- |
| Compact/full alternating, 32 requests | 13 / 127 | 4096/4096 | 0/3968 | 0/1236 |
| Full logits, 32 requests | 13 / 127 | 4096/4096 | 0/3968 | 0/3968 |
| Partial fixture | 1 / 127 | 224/224 | Not applicable: different prompts | Not applicable |

Full32 inspects 201,326,592 BF16 logit values; 32,472,224 are bitwise equal to the baseline. Max absolute error is 0.78125 and RMSE is 0.1158794790782448. These aggregate values also match the pre-fix separated-stage control. No tolerance was relaxed. Fixture tests now assert exact greedy agreement and cross-request token/logit invariance; `numerical_profile_accepted=false` remains explicit because independent prompts and broader quality are not yet qualified. Teacher forcing is retained and is not a free-running generation evaluation.

Together, separating stages and then using consistent decode arithmetic with the original mixed scheduler support the diagnosis of stage-dependent arithmetic accumulation. The intervention does not claim that all possible numerical or scheduling issues are solved.

## Implementation and cost

- The pinned FlashInfer CUDA/C++ kernels are instantiated and statically linked into Riley. Actual execution is Rust → C ABI → CUDA, with no Python/PyTorch serving calls. Python is used only for build-time dependency checks and separate probes.
- Sparse `request_indices` map decode queries directly to packed token offsets. Sparse page indptr/length metadata maps the same offsets to existing HND KV. Neither Q nor KV is repacked.
- FlashInfer emits 32 compact output rows. A small CUDA scatter writes only valid decode rows back into packed attention output; the existing mixed attention kernel skips those decode rows and continues to compute prefill rows. Prefill arithmetic is unchanged.
- Metadata is prepared once per iteration and retained across 30 layers. Workspace increases from 33,456 to **78,256 bytes**, including aligned output storage. There is one additional scatter launch per mixed-model layer; this is a cost to measure in serving, not a claimed speedup. Prefill-only captures also contain masked FlashInfer/scatter launches. Pure decode has no scatter.
- Native recorder ownership/extent/alias checks remain, with the new exact workspace size. Catalog identity is explicitly `consistent-decode.v2`; existing exact factories do not select it.

## Failures found during integration

Two intermediate failures are preserved rather than relabeled as successes:

1. A CUDA misaligned-address failure: the temporary BF16 output field began at byte 41,388. It now has `alignas(16)` and a static offset assertion; it begins at 41,392.
2. After alignment, severe numerical corruption exposed a missing final indptr bound. FlashInfer protective KV loads read `indptr[paged_kv.batch_size]`, including for sparse mapped requests. Metadata now always initializes `indptr[1024]` to the total referenced page count. A paired mixed-versus-pure primitive comparison alone did not reveal this shared metadata defect; the full-model reference comparison did. Do not treat that earlier primitive comparison as independent correctness proof.

The failed logs correspond to intermediate source states, not to the final source manifest. Their exact intermediate binaries are not claimed to be reproduced by the final manifest.

## Final verification

Full32 Compute Sanitizer memcheck completed with **0 errors**, while the same strict 4096-output greedy and 3968 cross-request token/full-logit assertions passed. The existing exact factory also passed its 224-output/129-iteration partial GPU regression with the new native build. [outcomes.json](outcomes.json) distinguishes these fixture passes from unaccepted general quality and unmeasured serving performance. [sources.json](sources.json) verifies changed final local/remote sources; the final test binary hash is retained separately.

## Scope and next gate

The full32 and partial fixtures exercise scheduler completion, pending close/abort and zero retained CUDA allocations. Source/compiler/binary identity and final memory verification are recorded with this batch's evidence. The pre-fix control binary has a separate hash; its harness had the stage/cross-request instrumentation before the final strict regression assertions were added.

Hopper/Blackwell and multi-GPU runtime remain unavailable; this batch's runtime evidence is RTX 4090 only. Existing standalone AOT build evidence does not validate these new mapped kernels on other devices.

Next: validate independent prompts and free-running generation, preserve explicit experimental arithmetic, then connect the server profile and measure matched frozen baseline/vLLM workloads. The engine-wide throughput/latency goal remains active and unproven.
