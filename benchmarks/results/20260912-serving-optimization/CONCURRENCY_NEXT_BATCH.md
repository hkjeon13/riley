# Concurrency: next optimization batch readiness

Date: 2026-09-12. This is a source audit and implementation-readiness record, not a performance result, approval request, or completion of the serving-performance goal. No production code was changed and no GPU or remote workload was run for this audit.

Source pointers below describe the local working tree inspected on this date. Its Git HEAD was `3bc72c99dc62e6f93bfdb69b061f82b6c107d494`, but relevant files were modified or untracked. That commit alone does **not** identify the audited implementation or the separately frozen remote benchmark binary. Paths and symbols are the navigation authority; line numbers may move with subsequent work.

## Supported requests versus qualified evidence

- Runtime admission accepts exactly **128 prompt tokens and 1–32 requested output tokens** for `vllm-smol-p128-v1`: `OwnedLlamaDecodeExecutor::validate_request_shape`, `crates/riley-runtime/src/llama/graph_decode_full.rs:1490`. Backend shape validation occurs before scheduler submission at `crates/riley-server/src/engine.rs:3130`.
- The current native profile evidence contract is **concurrency 1, prompt 128, output 32**, enforced for `VLLM_REFERENCE` at `crates/riley-server/src/bin/riley-profile.rs:406`. This does not establish O1–31 performance or any higher-concurrency result.
- The existing serving runner rejects workload concurrency other than one at `benchmarks/scripts/run_serving_optimization.py:78`; the engine runner checks `--concurrency 1` at `benchmarks/scripts/run_engine_optimization.py:70`. Higher-concurrency evidence needs an explicit runner/schema/qualification extension. Closed c1 receipts must retain their original scope.
- The CLI requires graph execution and keeps numerical qualification separate from C02: `crates/riley-server/src/main.rs:503`.

## Current restrictions and ownership

| Layer | Source pointer | Current contract and implication |
| --- | --- | --- |
| CLI | `crates/riley-server/src/main.rs:917` | Profile requires `max_active_sequences=1`, fixed-max shape, and matching batch/prefill budgets of either 1 or 128. |
| Backend preparation | `crates/riley-server/src/engine.rs:2173` | Batched P128 separately requires active=1, budget=128, chunk=128. At line 2208 scheduler and executor token budgets must match. |
| Capacity construction | `crates/riley-server/src/main.rs:1000` | Scheduler active capacity currently also determines executor metadata rows and output slots. Those concepts must be separated for multi-request plans over an M1 graph. |
| Graph eligibility | `crates/riley-runtime/src/llama/graph_decode_full.rs:239` | Requires exact profile geometry, forward sequence length=1, metadata rows=1, qualified norm/attention settings and selected GEMM properties. |
| Graph owner | `crates/riley-runtime/src/llama/graph_decode_full.rs:1341` | One thread-confined retained owner holds the original executor/KV pool, one stream, metadata/result/staging, P128 scratch, optional packed parents, and a `GraphRegistry<1>`. Eager access is unavailable while graph leases remain active. |
| Runtime replay | `crates/riley-runtime/src/llama/graph_decode_full.rs:1558` | Exactly one request row: all P128 positions 0–127, or one decode token at position 128–159. The payload holds one token, scalar end position, block table and local output slot zero. |
| Native validation | `kernels/src/graph_resources.cu:297` | Validates one request's block IDs/valid lengths, local row mapping and output slot zero; the P128 extension permits only 1 or 128 token rows. |
| Output adapter | `crates/riley-scheduler/src/execution.rs:1962` | Validates the plan against one-row graph bounds, invokes the graph once, and obtains at most one greedy token. It is not presently a multi-output graph adapter. |
| Host output capacities | `crates/riley-server/src/engine.rs:2459` | Greedy tokens, sample staging and pending-token capacity derive from executor output slots, currently one. |

**P128 rows are positions within one request, not independent request rows.** In `kernels/src/graph_numerics.cu:25`, attention uses one block table and computes each row's causal count from `n - rows + row + 1`, where `n = *end_position + 1`. RoPE/KV row variants likewise derive positions from one scalar end position. Setting this row count to the number of active requests would produce incorrect positions and KV access.

The packed projection plans are qualified for M1. Replacing M1 with M=N, even with the same weights and BF16 outputs, can change arithmetic and algorithm selection. Such a change needs actual per-request logits/KV parity qualification; the existing packed projection probe cannot establish it.

## Request queues and continuous selection

The current CLI constructs `ServerConfig` using default worker settings (`crates/riley-server/src/main.rs:1154`). Those defaults are **eight HTTP workers and a connection queue of 128** (`crates/riley-server/src/service.rs:390`). Each worker processes a connection synchronously through response completion (`service.rs:1186`). Thus the current CLI path can have at most eight connections being processed concurrently; additional clients can wait before reaching engine admission. Eight is the current CLI configuration, not a hard library limit: `ServerConfig.worker_threads` is configurable. A C16/C32 experiment must expose/set that capacity or explicitly report the resulting connection-level queue.

The engine submission path has a bounded command channel and admission acknowledgement (`crates/riley-server/src/engine.rs:1250`). One engine worker exclusively owns the backend, scheduler and CUDA execution; it drains commands, executes one backend step, and publishes results (`engine.rs:1532`). There is one scheduler iteration in flight, not overlapping GPU iterations. An accepted backend request may still be in the scheduler's waiting state, so engine accepted-request counts are not scheduler active-sequence counts.

Scheduler admission already supports multiple active sequences, a bounded FCFS waiting queue, individual sequence reservations and promised KV capacity (`crates/riley-scheduler/src/scheduler.rs:743`, `:1457`, `:2352`). The current CLI sets a 100 ms prefill aging threshold, wait overload policy and 30 s admission timeout (`crates/riley-server/src/main.rs:987`). With a 160-token maximum request and 16-token KV blocks, each admitted request promises ten blocks; N active requests need at least 10N blocks/promise capacity. The CLI derives this default at `main.rs:971`.

Selection is already continuous across active requests (`Scheduler::select_candidates`, `crates/riley-scheduler/src/scheduler.rs:1504`):

1. Collect prefill/decode candidates and order each by ready time and request ID.
2. Usually select decode tokens first, then fill the remaining token budget with prefill chunks.
3. Allow an aged prefill to precede decode, while reserving one token of budget for decode and avoiding consecutive aging overrides.

**Concrete 127-token blocker:** with token budget 128 and one selected decoder, the normal path gives a pending P128 request only 127 tokens (`scheduler.rs:1572`, `:1590`). The aged-prefill path also selects 127 because it reserves one token for decode (`:1558`). Both violate the complete-P128 graph contract and are rejected. Lifting active=1 guards alone is therefore insufficient. A larger budget alone also does not guarantee atomic P128 selection when the remaining budget becomes smaller than 128.

## Proposed batch: four related architecture changes

This is the smallest coherent bridge to multiple active requests while retaining the current qualified M1/P128 computation. Its first implementation can serialize selected requests on the existing graph and shared KV pool; it is not a claim of GPU request batching or a throughput improvement.

1. **Separate scheduler-plan capacity from per-replay graph capacity.** Allow N active scheduler sequences and N aggregate output slots, while keeping one qualified graph row and the existing immutable weights/plans. Allocate aggregate engine workspaces for N and KV capacity for at least 10N blocks. Keep distinct validation bounds for the whole scheduler plan and each local graph replay. For an initial N≤128 and homogeneous plans, token budget 128 can remain unchanged; this does not require an arbitrary executor budget increase.
2. **Select complete P128 prompts with decode fairness.** Add a profile-aware scheduling rule that selects either one complete P128 prefill or up to N decode rows. A prompt must fit in full or be deferred. Preserve bounded aging and decode progress instead of leaving P128 starvation possible. Homogeneous plans avoid mixed-stage packet semantics during the first implementation.
3. **Aggregate qualified replays into one atomic scheduler result.** Validate every selected row, reservation, position, token and output mapping before the first dispatch. Rebase each graph replay to local row/output slot zero, preserve its own sequence block table, and copy each returned token/logit row into its original dense scheduler output slot. Reuse graph scratch only after completion. Publish the whole iteration after all selected replays succeed. Any failure after an earlier replay dispatched must not be labeled `NotDispatched`.
4. **Wire and qualify the serving capacity contract.** Update CLI/backend guards, capacity reporting, HTTP worker configuration and concurrency benchmark validation together. Exercise retained ownership, admission, output routing, cancellation and block reuse under multiple active requests. Preserve c1 regression evidence, and identify the new implementation/workload qualification explicitly rather than broadening old receipts.

The bridge still performs per-request graph completion/transfers. Its benefit, cost, and fairness effects are unknown until serving measurements. If profiling then shows transfer/completion or launch work limits throughput, the next GPU-oriented batch can evaluate several M1 lanes behind one completion boundary or true request-batched decode. Either option needs distinct per-request positions, block tables, outputs and inactive-lane rules. M=N arithmetic must be requalified.

## CPU/GPU ownership and failure requirements

- Keep one owner of shared weights, plans and the KV allocator. A full executor per request duplicates resources and splits authority; it is not an equivalent extension of the current retained owner.
- Each selected scheduler sequence already gets its own reservation and immutable copied block table (`crates/riley-scheduler/src/scheduler.rs:1736`). These identities must survive local replay-slot rebasing.
- `complete_iteration` requires the executor stream to have completed and no kernel to retain the plan, tables or reserved KV ranges (`scheduler.rs:895`). Validate all outputs before committing any scheduler state.
- Rollback is only valid before dispatch. After partial dispatch, use the existing conservative completion/poisoning contract; unknown completion must prevent KV reuse (`scheduler.rs:1068`, `crates/riley-scheduler/src/execution.rs:1965`). Async multi-stream execution would require a separate completion protocol rather than relaxing this rule.
- Native validation currently detects duplicate blocks only within one request's table. Aggregate validation must retain the scheduler's cross-request ownership guarantees; padding/inactive lanes must never write KV or publish tokens.
- Bounded per-request output channels cancel slow/disconnected consumers rather than blocking the engine (`crates/riley-server/src/engine.rs:1745`). Retain that behavior while testing cancellation with multiple reserved sequences.

## Required benchmark and correctness matrix

| Dimension | Required coverage |
| --- | --- |
| Client concurrency | C=1 regression, then 2/4/8/16/32. Record offered concurrency, scheduler max-active and observed active sequences separately. Start with bounded N=2/4/8 if needed. |
| HTTP capacity | Record/set worker and connection-queue capacities. C>8 on the unchanged CLI otherwise measures an additional queue ahead of the engine. Keep overload and timeout semantics explicit. |
| Arrival pattern | Closed-loop steady load, staggered arrivals forcing prefill while other requests decode, and bursts below/near/above saturation. |
| Request shape | P128/O32 first, with both repeated and distinct 128-token prompts; identical prefix caching, EOS/output-count and sampling conditions. O1–31 is a separate correctness/shape matrix until benchmarked. |
| Comparators | Accepted baseline, new implementation and vLLM on the same model weights/tokenizer, hardware/runtime, workload and GPU/display condition. Preserve source/binary/argv/environment identities and fresh paired-process runs. Label old active=1 serving under C>1 as queued c1 execution. |
| Throughput | Successful output tokens divided by observed wall time and successful requests/s. Include failed/rejected/cancelled counts; do not count attempted work as successful throughput. |
| Latency | Engine token-aligned TTFT/TPOT where instrumented; HTTP first-text and E2E separately; per-request inter-token latency only with a valid token mapping. Report P50/P95/P99 and process-to-process variation. SSE event counts do not define token counts. |
| Queues/resources | Connection/engine/scheduler waiting, active counts, prefill/decode occupancy, CPU time, allocation/copy/synchronization spans, GPU execution/utilization and peak memory. |
| Duration/tails | Use a long enough steady interval and enough completed requests for tail analysis. Thirty measured requests cannot establish reliable high-concurrency P99. For example, 10,000 requests yield roughly 100 observations in the upper 1%, without itself guaranteeing statistical precision. |
| Exact correctness | Per-request reference tokens/logits and KV across distinct prompts/tables, staggered decode positions, output-slot permutations, request finishes, inactive padding, queued/active cancellation, slow consumers and immediate block reuse after proven completion. |
| Failure containment | Invalid metadata before any dispatch; later-lane failures after earlier dispatch; no stale output, partial scheduler publication or KV reuse under unknown completion. |

Use the campaign's paired fresh-process ordering and matched warmup/cooldown conditions for comparisons, but extend the request count/duration for concurrency tails. Do not pool all requests across processes and present that as the only variation estimate.

## Measurement unknowns and decision rule

This audit establishes source restrictions and a reproducible scheduling incompatibility. It does not establish that scheduler selection, command draining, HTTP workers, graph synchronization, KV allocation or GPU occupancy is the current measured bottleneck. No high-concurrency result is inferred from the separate batch4 correctness or c1 performance campaign.

The next batch is useful only after exact correctness holds and matched serving measurements show its effects. Retain c1 behavior, compare throughput and latency at each concurrency, inspect P95/P99 and failure rates, then use collected queue/execution spans to select the next optimization area. Multiple active requests alone do not meet the final vLLM performance goal.
