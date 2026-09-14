# Greedy speculation foundation and target-prefix equivalence

Implemented bounded Rust proposal/acceptance/KV settlement primitives and tested an existing target numerical prerequisite on RTX 4090. **Fused target verification, scheduler/serving integration and a performance gain are not established.** This is not completion of PR13 or the vLLM serving objective.

## Implemented batch

`riley_runtime::speculative` supports at most eight draft tokens. It validates K+1 target argmax rows, accepts only the matching prefix, emits the first target replacement or all-accepted bonus, and stops on EOS/remaining length/cancellation. Token arrays are fixed-size. No target distribution or sampling tolerance is substituted for greedy equality.

The target verifier must consume `[pending_token, draft...]`. If N outputs are emitted, N input positions are retained: the old pending token plus the accepted inputs before the final emitted output. That final output becomes the next pending token. This also handles EOS/length truncation without committing its speculative continuation. `settle_completed_greedy` binds start/extent to the existing reservation, reuses `commit_prefix` and `discard_completed_append`, releases rejected suffix pages and preserves logical bounds within a retained tail page. GPU quiescence and append-only writes are caller obligations, not conditions this host helper proves. Unknown writes require poisoning; errors retain authoritative sequence state for recovery.

Prompt lookup is an initial proposal source requiring no second checkpoint. It searches at most 4096 history tokens with n-gram width at most eight, chooses the longest/newest completed match and returns at most eight known tokens. Empty proposals naturally require one ordinary target prediction. This is not a trained draft model or EAGLE implementation, and a supported larger target/draft pair remains outstanding.

Both [vLLM's n-gram proposal documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/) and [TensorRT-LLM's speculative decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) describe history-derived proposals. The implementation here is independent Rust code, with a shared greedy acceptance boundary intended for later draft providers. Those sources do not establish Riley speedup.

## Verification

| Check | Evidence | Scope |
| --- | --- | --- |
| CPU policy tests | 5 passed | Accept none/partial/all, bonus, EOS, length, cancellation, invalid rows and bounded lookup |
| KV settlement | 252 combinations | Seven starting offsets including page boundaries, nine acceptance lengths, four output limits; exact logical/page counts and subsequent append/cleanup |
| Autoregressive policy check | 11,250 combinations | All 625 length-four drafts over five tokens, six limits and three EOS settings against an independent deterministic serial oracle |
| Real target logits | 72 rows, 7,077,888 bytes per lane, exactly equal | Eight frozen natural cases, nine causal endpoints each |
| Full-model memcheck | 0 errors | Same target-prefix test; allocation-zero assertions passed |
| Serving / vLLM | Not run | No throughput/TTFT/TPOT claim |

The GPU prerequisite uses the accepted projection/FFN target profile and identical forced histories. One run advances eight requests sequentially for nine outputs; another expands them into 72 independent prefix requests, scheduled with at most 32 active owners, each producing one output. The second run repeats prefill work. It is **not** one fused verification pass and does not test tentative KV reuse across speculative branches. Equality on this fixed fixture is evidence for further integration, not proof for all contexts, model sizes, sampling policies or batch shapes.

[Raw evidence](evidence.tar.gz) includes both complete BF16 dumps, input cases, source/binary/model hashes, build/execution receipts, sanitizer output and the exact GPU lifecycle controller. [Verification](verification.json) independently checks all 15 regular archive members and exact raw-logit equality. Re-run `python3 benchmarks/analysis/verify_speculative_foundation.py` from the checkout. CPU output is in [policy-cpu.log](policy-cpu.log). `git diff --check` passed.

SM89 CUDA build/runtime passed. No hardware-specific Hopper/Blackwell/multi-GPU path was added by this host-policy batch; execution on those devices is not claimed. Blender's three scene RPCs succeeded after restoration and the three viewer loopback ports returned HTTP 200. No public-browser visual check was performed.

## Next integration boundary

Add one target verification invocation exposing K+1 predictions per owner, retain a private append reservation through target completion, then apply the existing greedy decision and KV settlement to scheduler token accounting. Verify proposed-history and sequential-target logits before accepting output; the preceding rejected FP32 attention experiment remains excluded. Handle cancellation only after device work is quiescent. Until implemented, requests requiring stochastic sampling, stop-string integration or unsupported processors stay on ordinary decode.

Measure draft time, verifier time, accepted length, rollback cost and actual serving results together. Prompt lookup can have poor acceptance, especially on unique text; no-hit/low-benefit and high-concurrency cases require measured fallback. Larger target and draft-model support remain part of PR13 rather than being replaced by this initial proposal source.
