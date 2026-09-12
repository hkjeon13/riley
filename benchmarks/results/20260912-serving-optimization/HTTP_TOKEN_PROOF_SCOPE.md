# HTTP token proof scope correction

The first optional-token helper assumed that stop `I'm` consumed the entire first three decoded tokens. Actual frozen IDs28/339/5248 decode as `,`, `, I`, `, I'm`; the pre-launch check rejected that fixture. V2 corrected the stop to the complete prefix `, I'm`, preserving three blank committed events as the intended test.

V2 then exposed a second harness assumption: the existing CLI deliberately rejects C02 artifacts with `vllm-smol-p128-v1` (`crates/riley-server/src/main.rs:506–510`). The owned server exited2 before serving or GPU inference, and process cleanup was verified. Its failure directory and logs remain preserved. This is a qualification-profile boundary, not permission to bypass the existing numerical gate.

The next helper uses the g04 numerical profile's own evidence: the exact frozen source/binary, fresh real-model logits/status/generated-token/KV and reuse tests, committed-event source inspection and unit tests, plus actual default and opt-in HTTP observations against the same model reference. It will not claim a C02 artifact or a separately recorded scheduler-audit match.

The source inspection establishes this direct publication path:

- `engine.rs:3398`: successful `Scheduler::complete_iteration` precedes external publication; commit failure returns an error.
- `engine.rs:3433`: events iterate the scheduler's committed token events, require a matching staged sample/request, and reject differing token IDs.
- `engine.rs:3447`: generated index and first-prompt metadata are checked; the event's ID and index come directly from the committed token.
- `openai.rs:642`: delivery enforces contiguous indices, one initial prompt, bounded IDs and final usage counts. Blank text does not remove a token ID.
- The real CUDA server library run passed72 tests, including `committed_empty_token_is_delivered_and_channel_overflow_cancels`, `committed_tokens_and_usage_survive_concurrent_http_delivery`, and `nonstream_disconnect_probe_cadence_counts_invisible_tokens`.

These static/unit/runtime observations have separate scopes. They provide a coherent token-publication check without asserting a per-request C02 audit that this profile does not produce. HTTP token TTFT/TPOT remain client delivery metrics; they do not measure scheduler commit time.

V3 actual GPU/HTTP execution completed92 checks across both samplers. Receipt `raw/http-token-observation-correctness-v3/completion.json` SHA256 `287f35e37440c1b586bdf2aa2d4f9bc4cc8ead7113881eef72f32192c0dcb301`; both raw streaming stop cases delivered exactly3 blank events with IDs28/339/5248. Source `a179617070526068b66ba5627ba82a7151da8c64` / binary `18cbd5f8a8ad8583ecfcd2d38815a484831c05eb9f081b1db67be6194ffebd8c`. This qualifies the API baseline, not the batch8 candidate or serving performance.
