# Optional HTTP token observations

Date: 2026-09-12. Local source and saved-response review only. No production files, frozen runners, raw artifacts, GPU workloads or remote state were changed. This document proposes one coherent implementation batch; it is not a performance or correctness receipt.

**Add opt-in raw token IDs to the existing completion endpoint and timestamp their arrival in the HTTP client.** This can measure token-aligned HTTP delivery TTFT/TPOT while retaining the current production queue, graph, detokenizer and connection path. Non-streaming token IDs provide correctness evidence, but cannot recover individual token arrival times. Server-internal commit timing remains a different metric.

## Evidence and scope

The accepted implementation remains batch7, source `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`. The local `engine.rs` reviewed here matches its file pin in `raw/batch7-http-plan.json`. Other line references describe the current local source; they are navigation pointers, not a new source/binary qualification.

The saved ten-request diagnostic in `raw/vllm-c2-output-diagnostic-runtime/` is explicitly correctness-only. Readback establishes:

- The same vLLM capacity-two server returns the exact c1 reference under offered C1.
- Its two C2 non-streaming token-ID responses contain 32 tokens and first differ from that reference at zero-based indices 29 and 8.
- Both C2 streaming responses contain 32 generated-token frames, each with one token ID; prompt IDs appear once. Their first differences are at indices 8 and 29. These requests did not request `include_usage`, and the saved streams contain no usage frame.
- The diagnostic ran with the separately recorded driver-library override and Blender compute present. It is not a matched timing run. Its JSON does not record per-frame arrival times, so timings cannot be reconstructed afterward.

vLLM's version-pinned completion protocol defines optional `return_token_ids`: per-choice generated IDs plus prompt IDs, with prompt IDs only in the first streamed chunk. `stream_options` is limited to streaming requests. Its protocol also allows grouping tokens through `stream_interval`; an HTTP measurement must verify actual frame cardinality. [vLLM v0.27.1 completion protocol](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/entrypoints/openai/completion/protocol.py)

With `include_usage`, vLLM emits a final usage-only chunk with `choices: []` before `[DONE]`. Its generation chunks retain token IDs even when visible text is empty. Riley should match these observable shapes for the supported single-prompt, single-choice subset. [vLLM v0.27.1 completion serving](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/entrypoints/openai/completion/serving.py)

## Current source seams

| Area | Exact current path/symbol | Gap |
| --- | --- | --- |
| Request normalization | `crates/riley-server/src/openai.rs:62`, `CompletionRequest`; `:119`, `normalize_completion_request`; `:198`, `validate_supported_options` | Token/usage options currently enter `unsupported_fields` and are rejected. The supported prompt remains one text string, `n=1`, `echo=false`. |
| Domain | `crates/riley-server/src/domain.rs:60`, `GenerationRequest`; `:323`, `GenerationEvent` | Requests only select streaming; events carry visible text or terminal usage, with no raw token identity. |
| Tokenized prompt | `crates/riley-server/src/engine.rs:3119`, `CudaBackend::admit` | The actual prompt IDs are already available before scheduler submission and retained in generation state. |
| Sample versus commit | `engine.rs:2748–2779`, token acceptance/staging; `:3357–3413`, scheduler commit/publication | Pending tokens exist before authoritative commit. Publish IDs from `updates.token_events()` only after successful `complete_iteration`, checking the pending token and request mapping. |
| Empty output | `engine.rs:1745`, `publish_events` | Empty `TokenDelta` text is suppressed. That is unsuitable for a committed raw-token event. |
| Cancellation | `engine.rs:2571`, `process_completion` | Cancellation can flush withheld text without generating a new token. That flush must not invent a token ID. |
| Response DTOs | `openai.rs:595`, `CompletionChoice`; `:655`, `CompletionChunkChoice`; `:668`, `CompletionChunk` | No token-ID fields; every SSE chunk currently requires a choice, with no usage-only form. |
| SSE state machine | `openai.rs:794`, `SseStreamEncoder`; `:816`, `encode_event` | Existing ordering is delta → finish → DONE; `Finished.usage` is ignored. |
| HTTP routing | `crates/riley-server/src/service.rs:1777`, `handle_completion` | Request is moved into submission; response options need to be retained before that move. |
| Stream/collection | `service.rs:1937`, `stream_completion`; `:2137`, `collect_completion` | Every non-`TokenDelta` event is treated as terminal. Collection only bounds and accumulates text. |
| Existing observations | `service.rs:1891`, `RequestTracker` | Its historical first-token field is triggered by delivered text events and starts after submission returns. It is not a substitute for client request-to-token timing. |

The C02 generation audit already records committed token IDs, but only as a terminal audit payload (`engine.rs:620`) without token times. Requiring its durable sink for streaming would add unrelated I/O and still deliver the IDs too late. Reuse the commit-validation principle, not the terminal file-writing path.

## Four related changes in one batch

### 1. Explicit, backward-compatible delivery options

Extend `CompletionRequest` with optional `return_token_ids` and a typed `stream_options` containing `include_usage`. Missing, null and false retain the default behavior. Reject wrong JSON types, unsupported nested fields and non-streaming requests with substantive stream options. Continuous usage, token-ID prompts, prompt echo, multiple choices and logprobs remain separate unsupported capabilities; do not silently accept them while adding these two fields.

Normalize transport-independent flags, such as `include_token_ids`, into the domain request and retain response options in `handle_completion` before handing the request to the engine. `include_usage` controls HTTP encoding only and must not affect execution/sampling. All existing request constructors and test fixtures need explicit defaults; adding fields to a public Rust struct requires source updates even though the HTTP change is opt-in.

Preserve existing `CompletionResponse::new` and `SseStreamEncoder::new` as default constructors, with separate option-aware construction/helpers. New optional JSON response fields should be omitted when disabled so ordinary responses retain their current field set and SSE ordering.

### 2. A bounded, committed token event

Add a distinct domain event, for example `CommittedToken`, containing `token_id: u32`, `generated_index`, existing text delta, and optional first-event prompt IDs. Keeping the existing text-only `TokenDelta` supports ordinary consumers and text-only cancellation flushes. All enum consumers must explicitly classify the new variant as nonterminal; do not rely on `!matches!(TokenDelta)` after this addition.

For opt-in requests, reserve the bounded prompt-ID transfer buffer at admission and move it into the first committed token event exactly once. Source it from the tokenizer result actually submitted to the scheduler. Retain a small next-generation-index counter, validate the scheduler's index and token ID against pending publication, and emit one event per committed token. No new GPU readback, graph event, kernel, token resampling or token-wise detokenization is needed.

For ordinary requests, preserve the existing text event path. For opted-in requests, `publish_events` must deliver a committed token even when its text is empty; channel capacity and nonblocking send/slow-consumer cancellation rules still apply. Do not drop a raw token to keep a slow request alive. Stop/EOS/special tokens may count as generated even when they produce no visible bytes. Emit exactly the authoritative committed sequence, without masking or adding tokens to satisfy transport expectations.

Publication stays after commit. A sampling/dispatch/commit failure must expose no speculative ID. Check all mappings before successful publication and preserve conservative failure handling and KV reuse rules. Do not let an unsupported backend fabricate IDs from text: its opt-in capability must be implemented or rejected explicitly before a successful response starts.

### 3. Response assembly, SSE usage and cancellation as one contract

For non-streaming opt-in responses, collect IDs in order alongside existing text, then expose `choices[0].token_ids` and `choices[0].prompt_token_ids`. Validate generated indices, a single prompt-ID payload, input count, final ID count versus `usage.completion_tokens`, and maximum requested tokens. Preserve the complete text produced by the existing detokenizer. Text cannot generally be reconstructed by independently decoding each ID because UTF-8 and stop handling span tokens.

For streaming, encode one generated-token choice per committed event, including when `text == ""`; include prompt IDs once on the first such choice. A text-only flush carries no new token IDs. A separate empty finish choice is compatible with Riley's current event structure: it must not duplicate the last ID. The client's parser must also accept vLLM's observed last-token-plus-finish chunk.

Add an option-aware success sequence: token/text events → finish → optional usage-only chunk (`choices: []`) → exactly one DONE. Validate accumulated raw token count against terminal usage before writing a successful terminal response. Usage-only and tokenless finish chunks are never token timing observations. Failures/cancellation emit the existing error sequence; they must not emit fabricated complete usage or a success finish. A disconnected client may never receive final usage, which must remain an incomplete request in benchmark accounting.

Keep `SubmittedRequest` armed until successful terminal frames and DONE are written; write/flush failure, timeout, shutdown or premature return retains cancellation. Account raw-ID storage and serialized overhead with checked token/byte bounds in addition to the existing non-streaming text limit. Reserve at most the requested output count, not an unbounded vector. The initial prompt-ID frame is larger, so opt-in performance must be measured with the same option on both lanes.

Do not relabel historical `RequestTracker.time_to_first_token` data as raw-token timing. Either leave that text-observation metric unchanged or add a separately named metric. The new benchmark uses client-side monotonic timestamps and does not depend on this service statistic.

### 4. A new token-aware HTTP measurement contract

Keep the frozen text-only concurrency runner and all old receipts unchanged. Add a new runner mode/schema or a separate runner whose manifest requires `return_token_ids: true`, streaming `include_usage: true`, and fixed P128/O32. Non-streaming warmups request raw IDs too. Reuse the validated closed-loop workers, observed-C warmup gates, common wall interval, total request deadline and owned-process cleanup.

For each SSE frame, capture a client monotonic observation immediately when its complete data payload becomes available, before expensive validation or hashing. Validate response ID/model/choice, first prompt IDs, ordered token deltas, final token count, terminal reason, usage and DONE. Retain raw frames or a lossless parsed representation plus timestamps. Exact output checks apply to warmups and retained requests under the manifest's explicitly chosen numerical reference policy.

```text
HTTP token TTFT = first generated-token frame arrival − request start
HTTP token TPOT = (last generated-token frame arrival − first) / 31
HTTP token ITL  = differences between 32 one-token frame observations
HTTP E2E       = successful terminal/DONE receipt − request start
throughput     = successful generated tokens / common retained wall interval
```

Count token IDs even when visible text is empty. Ignore tokenless prefill/finish/usage frames for token timing, and reject duplicate prompts, missing IDs or ambiguous multi-ID token chunks for the strict one-token timing profile. A final chunk containing one token and a finish is valid. If a future setting deliberately groups tokens, record grouped observations and mark per-token ITL/TTFT ambiguity explicitly; do not distribute one timestamp across several tokens and call it measured per-token latency. TCP may coalesce several SSE frames into one read, so these metrics describe client-observed delivery, including buffering, rather than internal CUDA/commit intervals.

Non-streaming timing remains E2E only. Streaming raw IDs and usage enable exact token counts and observations; their existence does not make all internal token times available. First visible text remains a separate useful user-facing metric.

## Numerical correctness is a separate gate

The saved C2 vLLM output difference occurs with and without token-ID output and also within one server configuration when offered concurrency changes. It cannot be fixed by adding Riley response fields or by labeling the different text as an encoding failure. vLLM documents batch invariance as a separate beta capability enabled through `VLLM_BATCH_INVARIANT=1`. Enabling it is a different execution condition and needs fresh compatibility, numerical and performance qualification; this review does not establish SmolLM2 behavior under that mode. [vLLM batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)

Keep three independent results in new evidence:

1. **Riley numerical correctness:** the accepted logits/KV/profile proof and fresh opt-in versus opt-out token parity on the new binary.
2. **Publication correctness:** raw IDs equal that request's authoritative committed output, text/usage/finish are consistent, and no speculative/missing/duplicate tokens cross the HTTP boundary.
3. **Cross-engine equality:** whether both lanes produce the exact fixed c1 reference at the selected concurrent setting. Record observed differences explicitly. Default vLLM batching need not reproduce the c1 reference; its documented capability does not itself prove these particular changed outputs numerically correct.

The old strict screen correctly stopped when its predeclared text reference failed. Preserve that failure. A default-vLLM concurrency comparison requires a new, declared numerical acceptance contract and supporting checks, rather than editing old expected text, accepting any 32 IDs, or marking token equality true. Any batch-invariant comparison needs a new environment/configuration identity as well. Neither option should be chosen implicitly by the transport implementation.

## Focused qualification and regression tests

| Test group | Required cases |
| --- | --- |
| Request/response compatibility | Absent/null/false options preserve old JSON field sets and frame order; true options expose IDs/usage; wrong types/nested unknowns/non-streaming usage options reject; existing unsupported prompt/echo/n/logprobs checks remain. |
| Token versus text | Empty text for a committed special/EOS token, multi-token UTF-8 completion, stop-string withholding, output length one and 32, and text-only cancellation flush. IDs advance only on committed token events, never on a flush or usage chunk. |
| Event ordering | Missing/duplicate/out-of-order generated index, wrong prompt IDs, wrong terminal count, token after terminal, duplicate finish/usage/DONE, last-token-plus-finish parser support, and failure before the first token. |
| Concurrency/lifecycle | C2/4/8 opt-in and mixed opt-in/opt-out clients; request-ID isolation; queued and active cancellation; slow consumer/channel full; disconnect after an empty-text token, before first token, and after finish but before usage/DONE; graceful shutdown and immediate KV reuse after proven completion. |
| Sampling/backend parity | CPU and GPU sampling paths publish exactly their committed IDs, with unchanged seeds/stop behavior; unsupported mock/custom backends reject rather than claiming IDs; C02 audit remains independently valid when enabled. |
| Measurement validator | One-token SSE arrival timestamps, complete prompt/output/usage checks for both warmup transports, no usage chunk counted as a token, preserved failures and unresolved counts, raw reference mismatch reported, multi-ID ambiguity rejected, source/runner/request/schema changes rejected. |
| Fresh GPU/HTTP proof later | Accepted graph/kernel source unchanged; new server binary qualified for actual opt-in and ordinary requests, all reference tokens/full existing GPU tests, cancellation/reuse, then paired option-on timing plus option-off C1 regression. This is not established by the current read-only audit. |

Begin with fresh C1 and C2 correctness before C4/C8. Keep short one-pair runs labeled screening; use repeated fresh paired processes with at least 1,000 requests per process for baseline decisions and substantially more samples plus uncertainty/duration reporting for P99. Token telemetry adds bytes and may change publication costs, so measure that effect rather than assuming it is free. No performance gain, high-concurrency stability or vLLM win is claimed here.
