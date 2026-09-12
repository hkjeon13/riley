# Next batch: validate greedy results on GPU and reduce result transfer

V45 grouped/paired value attention prototypes are rejected before serving integration: both pass exact-reference and sanitizer checks, but regress the relevant longer contexts. V44 remains the accepted source and frozen binary.

Fresh instrumented V44 diagnostic:1536/1536 requests match references. Middle80 C16/C32 V4 decode totals1604/1654us, encode26/42us, native1469/1484us (includes replay, sync and readback), CPU validation109/125us. Native traces independently show approximately85–88us D2H per decode. These scopes overlap: do not sum native time and D2H, or claim all potential savings without measuring replacement GPU work. Native execution still dominates.

Existing `kernels/include/riley_cuda.h` already defines deterministic `RileyCudaBf16ArgmaxResult { token_id, status }`: scan every BF16 logit, choose the lowest token ID tied at the finite maximum, and reject the entire row if any NaN/Inf is present. `kernels/src/batch_primitives.cu` implements the row kernel and a single-row capture helper. Reuse this contract instead of weakening validation or inventing approximate sampling.

Proposed related changes, subject to measured benefit:

1. Measure the existing GPU full-row finite/argmax operation at vocab49152 and active1..16, including lowest-ID ties, signed zeros, every BF16 exponent class and invalid values. Compare exact selected token/status against the CPU validator. Measure GPU work plus small D2H against current full-row transfer and CPU validation.
2. Add a retained graph output mode carrying compact greedy decisions with complete per-row iteration/request/slot identity and completion status. Preserve input layout, parent leases, fresh H2D staging, inactive-row validation, failure visibility and close/abort ownership. Every active logit must still be scanned on GPU; unused rows must not affect active results.
3. Connect the compact mode only for requests whose sampling semantics are exactly supported. Keep full logits for non-greedy sampling, requested logits/logprobs, and reference verification; no silent loss of supported behavior. Test alternating request shapes, partial active rows, disconnects, invalid records and failure cleanup. Preserve a full-logit diagnostic path to cross-check compact outputs against the same model calculation.

Only after primitive cost and correctness are established should the protocol/session change be integrated. Then run owned model tests, concurrent streaming/nonstreaming HTTP and the complete matched serving matrix against V44 and vLLM. Do not infer a serving win from eliminating transfer bytes alone. The primitive stage below preceded the now completed V46 integration.

## V46 primitive screen completed

The existing unmodified BF16 argmax kernel passes306 correctness cases under both memcheck and racecheck: all65536 BF16 patterns in both column orders, sign-opposite pairs, signed-zero ties, tie indices crossing warp/block boundaries, NaN/Inf at varied positions, and vocab257/49152 with1/4/8/16 rows. Output guards remain intact. Its argmax plus compact D2H costs about41us at16 rows, versus about91us for full D2H alone.

A two-stage variant partitions vocabulary into2048-element chunks, preserves the same finite/lowest-ID comparator and invalid flag, then reduces partial records. It passes the same306 cases and both sanitizers. At16 rows, compact GPU validation plus D2H is14.11us versus91.80us for full D2H alone. At8 rows it is13.98us versus30.56us; at1 row it is13.92us versus4.45us. CPU full-logit validation is intentionally excluded from these copy-baseline numbers. Four reversed-order pairs,30 warmup/100 graph replays each, fixed pinned buffers, no profiler; this is not serving proof.

Integration findings: existing scheduler output already supports `GreedyTokens`, but retained variable-graph execution currently always asks for `ResultMode::FullLogits`, allocates/copies full rows and calls the CPU validator. Server reuse of validated argmax still applies temperature, repetition penalty, addressable vocabulary, token masks and history checks. `masked_finish_token_ids` can change with the minimum-token gate. Compact integration must decide supported sampling semantics before dispatch or retain a no-replay full-logit download fallback. Never replay model execution merely to recover logits after a compact result, as that can mutate KV twice. Respect the user's configured sampling backend and audit records rather than reporting CPU execution for a GPU sampling path. Native result identity, row publication, inactive records and close-before-KV-release remain mandatory. That was the pre-integration finding; V46 integration is recorded below.


## V46 integrated candidate and serving screen in progress

Remote isolated commit `4668b77b2e8c5311fdd5abfb2b0b665775b7ef66`; frozen binary SHA256 `e64ea083114f8a56608e68d96be2aa2aa74ad09ddc3b50f19c32695e6cc5fabc`. V44 remains the accepted serving baseline pending measurements. Local dirty application source is untouched; the source change is exported in `raw/compact-v46.patch`.

The batch combines two-stage exact finite argmax, 128-byte identity/status records per retained row, compact prefill/decode graph captures, and reuse of the server token workspace. Compact records use a distinct magic and exact extent; CPU validation binds all request/iteration/progress/output fields, rejects unsuccessful status and checks zero inactive rows. GPU validation scans all published logits. It is not an independent CPU re-scan of logits. Existing full graphs remain available, selected before dispatch when sampling is ineligible. No model replay is used for fallback. Scratch is reused only after model/head completion on the retained stream. All parent leases and close-before-KV-release rules remain.

Only explicit `variable-smol-v4 --sampling-backend gpu-greedy` enables compact capture. CPU configuration retains full output. Temperature, repetition penalty, vocabulary and finish-token mask eligibility continue through the existing typed selection/audit path. The CLI and engine both reject GPU greedy with eight-row variable graphs.

Validation:
- Wire regression: 13 tests pass, including two compact tests for both wire capacities, identity/status corruption, wrong mode, inactive rows and partial prefill publication.
- Compact GPU record probe: 170 cases pass memcheck and racecheck.
- Loaded model: seven owned GPU tests pass. Five existing full-logit tests check 5,344 rows exactly. Two alternating compact/full tests check 2,560 output tokens against immutable full-logit references, scheduler commit identity, 32 owners, partial prefill and allocation-zero closure. Final tests also verify workspace allocation reuse. The log's legacy `logits_checked` label means output checks for these two mixed-mode tests.
- Full-model compact tests under memcheck: two pass, zero errors. This ran before the later scheduler-only workspace integration; kernel/session code is unchanged.
- Final binary HTTP: 37 checked completions match reference tokens, including 32 concurrent streaming/nonstreaming requests, invalid bound rejection and disconnect/recovery; clean shutdown.
- Final binary fallback: 22 responses per backend, six sequential and 16 overlapping, temperatures 0/0.7 and top_p0.9; exact request ID, token, text and finish parity between CPU and GPU configuration. The earlier unordered concurrent stochastic comparison failed because arrival order changed request IDs used in RNG derivation (`generation.rs` combines seed and request ID). Ordered SSE-header admission fixes that comparison without changing product RNG semantics.
- Existing eligibility test (mask/penalty/temperature/vocabulary) and two workspace tests pass. Mask/penalty are not exposed in this HTTP request schema; no HTTP coverage is claimed for them.

Two integration failures were found and fixed before freeze: the CLI's old CPU-only guard, then failure to hand off the existing server greedy workspace. Their logs are retained. Evidence export has 75 SHA-verified files; archive SHA256 `3bef88e3b4bdc82bb2c9808fa1493ac2144f1f68b3eae2028b686fcf954309be` (`raw/compact-v46-manifest.json`).

Round53 is running: V44 CPU / V46 compact GPU greedy / vLLM, C16/C32, fixed and natural workloads, two reversed orders, 96 warmup and 384 retained requests per lane (24 lanes). Both Riley variants use 16-row execution, matched admission/KV/budget and external workload. No throughput or latency improvement is claimed until this run completes. Blender remains stopped; no restore helper is invoked.


## Round53 complete

All 24 lanes completed: 9,216 retained requests, zero failures; all 6,144 Riley reference checks pass. V46 improves V44 throughput15.13–19.01%, TTFT7.93–14.96%, TPOT13.17–16.23%, P95 12.43–16.51%, P99 11.74–15.78%. It remains10.98–43.36% below vLLM throughput, with worse TPOT and tails. V46 is the next baseline for the measured C16/C32 GPU-greedy conditions. See `V46_SERVING_RESULTS.md` for the complete comparison, new profile and remaining bottleneck. The fresh192-request profiler run confirms2048-byte result transfers and0.864us median decode D2H. All223 new evidence files are SHA-verified under `raw/serving-round53`; Blender remains stopped.
