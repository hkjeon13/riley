# Multi-position target head: exact GPU prerequisite verified

A separate experimental Rust → C ABI → CUDA path now returns up to eight causal positions per owner, with at most four prefill owners, using the existing single 32-row LM-head GEMM. **Speculative scheduler settlement and serving acceleration remain unimplemented.** This is a target-verification building block, not a vLLM performance result.

## Implementation and boundary

The ordinary last-query output remains in each owner's original head slot. Other query positions occupy the remaining slots, and unused selected rows are zeroed. The model runs once for the packed inputs, then gathers final hidden states and runs the existing head once; it does not rerun the transformer for each requested position. An independent 3,145,728-byte pinned buffer receives the full head logits. Normal response bytes and validation remain unchanged.

The recorder verifies pinned-parent context, ledger retention, exact extent and non-aliasing with ordinary staging. The profile hashes the selection source and explicitly excludes compact, buffered, FP32 context-split and alternate FFN profiles. Buffered graph preparation is rejected for this profile. Large/unsupported ordinary prefills retain normal selection, but their auxiliary read is rejected. This initial shape supports at most seven draft tokens plus the pending input; the separate host policy's eight-draft limit does not override this GPU limit.

`read_verification_logits` requires normal completion validation, a still-retained iteration, pure-prefill inputs with one to four owners and one to eight input tokens per owner. It scans all returned active BF16 logits for nonfinite values. Its immutable result preserves iteration, replay, owner generation, catalog digest, sequence tag, row cookie and absolute causal position. Reads before completion or after scheduler settlement are rejected. Reading these diagnostics does not authorize speculative token publication or KV commit.

## GPU evidence

| Check | Result |
| --- | --- |
| Native selector | 8 cases × 2 replays: single/max/ragged inputs, unused padding, status zeroing and unsupported-shape preservation |
| Native memcheck / racecheck | 0 errors / 0 hazards, 0 warnings |
| Actual target comparison | 72 BF16 logit rows, 7,077,888 bytes per lane, exactly identical |
| Actual model memcheck | 0 errors; allocation-zero assertions passed |
| Completion identity and read boundary | Final fixture checks iteration/generation/replay/digest/cookies and rejects reads before completion and after settlement |
| vLLM serving comparison | Not run |

The accepted projection/FFN target profile processes eight frozen natural cases. The serial lane produces nine causal outputs per request. The multi-position lane appends eight forced tokens to each 32-token prompt and uses eight-token prefill chunks, recording positions 31–39 exactly once. Two model invocations expose multiple queried endpoints at once (32 positions across four owners in each final chunk). Both lanes execute ten total model invocations in this fixture because the multi lane also splits initial prompt preparation into small chunks. Consequently this fixture neither measures nor implies a serving speedup.

The first run passed. A final Rust-only result-identity strengthening added explicit returned identity fields and assertions; the final model build and GPU/memcheck run also passed. Both runs are preserved. Selector code did not change between them. SM89 executes the final model; selector-only SM90a and SM100a builds pass with runtime skips for absent hardware. Full-model Hopper/Blackwell/multi-GPU execution is not claimed.

## Artifacts and verification

[evidence.tar.gz](evidence.tar.gz) contains both complete logit pairs, source/model/binary hashes, input cases, native/model build logs, lifecycle controllers, completion and sanitizer logs. [verification.json](verification.json) checks all 39 regular archive members, current source hashes, exact raw-logit equality, five successful final execution statuses, sanitizer summaries and restoration. Reproduce that audit with `python3 benchmarks/analysis/verify_verification_head.py`. Archive SHA256: `f4b2d653fea30d179d77e4e08ed7653df8f287b2eefcad3ab6712011be220d9b`.

The source verifier intentionally checks the current matching integration source; later changes require reviewing the historical snapshot instead of silently accepting drift. Both model binaries were hashed at build time even though their target paths are reused by Cargo. The v1 source snapshot is retained; final v2 source is the published integration revision. `git diff --check` passed.

All three Blender scene RPCs succeeded after final restoration, and viewer loopback ports 31840/31970/32010 returned HTTP 200. This is not a new public-browser visual verification.

## Next batch

Replace diagnostic full-logit transfer with GPU argmax results bound to the same positions and completion identity. Connect a private verification append to the existing greedy decision/KV settlement and scheduler multi-token accounting, including mismatch, EOS, length, cancellation and rollback. Preserve the exact target profile verified here; the rejected FP32 attention profile is not used. Then compare acceptance, total draft/verify/rollback cost and actual serving against the current Riley baseline and vLLM. Larger target/draft support remains outstanding.
