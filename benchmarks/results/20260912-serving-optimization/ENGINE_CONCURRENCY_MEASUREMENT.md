# Token timing under queued concurrency

Date: 2026-09-12. This is an implementation contract from local source review. No engine-concurrency harness has been implemented or run for this document. Round12 HTTP screening is separate evidence; its first-text events cannot supply token TTFT or TPOT.

The smallest useful extension is a **new engine measurement entry point with C logical clients and one retained Riley executor**. Keep Riley's active capacity at one and measure requests while they wait in its real scheduler queue. Refill a logical client immediately after its request's committed completion. Drive vLLM with the same arrival policy, explicit active capacity at least C, and the token budget selected by the HTTP screen. This requires a benchmark API extension, but no kernel, graph geometry, KV ownership, or production scheduler-policy change.

The source reference is accepted batch7, `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`. The locally inspected `benchmark.rs`, `riley-profile.rs`, and `engine.rs` hashes match the corresponding immutable file pins in `raw/batch7-http-plan.json`. Other source pointers below describe the local files inspected on this date; this review does not identify a new qualified binary. Relevant files are dirty relative to local Git HEAD, so HEAD alone is not this audit's source identity.

## Existing contracts that prevent an option-only change

| Source seam | Current behavior | Required treatment |
| --- | --- | --- |
| `crates/riley-server/src/bin/riley-profile.rs:406`, `Options::validate` | `VLLM_REFERENCE` requires c1/P128/O32. Warmups and measured trials are bounded to 5–100 and 30–100. | Preserve this closed v1 contract. Give queued measurement its own schema and explicit offered-concurrency/count fields. |
| `riley-profile.rs:879`, `read_prompt_seeds`; `:986`, `materialize_token_rows` | Selects C distinct text seeds, tokenizes with special tokens, then repeats/truncates token IDs to P. | The repeated fixed-input experiment must instead load the exact 128 IDs from the pinned reference. Do not manufacture C distinct prompts or silently repeat a text corpus. |
| `riley-profile.rs:1009`, `benchmark_config` | Graph lanes reject C != 1; C also determines scheduler active capacity, metadata rows/output slots, and physical KV blocks. | Separate offered clients from prepared execution capacity in the new entry point. |
| `crates/riley-server/src/benchmark.rs:1042`, `NativeBenchmarkExecutor::prepare_trial` | Requires an idle scheduler and caps trial requests at `max_active_sequences`. | Add a separate prepared queued-run object; retain the old fixed trial API. |
| `benchmark.rs:1516`, `validate_composition` | Requires `RejectImmediately`; P128 requires active=1, budget=128, chunk=128. | A new explicit queued constructor may accept `Wait` while retaining all graph/capacity checks. Do not relax the canonical constructor globally. |
| `benchmark.rs:1129`, `run_trial_inner` | Submits all requests together, requires every submission to be `Admitted`, then drains the complete batch. | New driver accepts `Admitted` or `Waiting`, starts only C requests, and refills after each terminal result instead of awaiting a whole C-request batch. |
| `benchmark.rs:1267`, commit observation; `:450`, `observe_updates` | A host timestamp is captured after authoritative `Scheduler::complete_iteration`; token IDs and generation indices are checked against committed updates. | Reuse this observation boundary and validation. Keep raw IDs and all 32 timestamp offsets in the new output. |
| `riley-profile.rs:1225`, `RequestEvidence` | Raw token times are reduced to per-request hashes and TTFT/TPOT/E2E. | New raw output must retain times, IDs, request/slot mapping, and terminal status so queue and refill behavior can be checked independently. |
| `benchmarks/scripts/run_engine_optimization.py:36` | Five pairs, five warmups, thirty c1 trials; closed c1 validators and service-time throughput summaries. | Leave it frozen. Add a new orchestrator and validator; parent c1 evidence is reference-only. |

Calling the old profile binary C times concurrently would create C model/executor owners and a different GPU-memory/workload condition. Running old c1 processes sequentially would omit real queue timing and retained-owner behavior. Neither is the proposed experiment.

## Proposed Riley API and driver

Keep existing `prepare`, `prepare_trial`, and `run_trial` behavior. Add an explicit queued constructor and prepared-run API in `crates/riley-server/src/benchmark.rs`, with a separate CLI such as `src/bin/riley-profile-concurrency.rs` and its `bench,cuda` binary entry in `Cargo.toml`. Names below are proposed, not existing APIs:

```text
NativeBenchmarkExecutor::prepare_queued(model, config)
NativeBenchmarkExecutor::prepare_queued_run(exact_inputs, output_tokens,
                                          offered_concurrency, request_count)
NativeBenchmarkExecutor::run_queued(prepared_run)
```

Share the existing execute/sample/commit implementation between the old and new drivers. Avoid a second copy of graph dispatch and abort handling. The new CLI takes `--offered-concurrency`, `--retained-requests`, `--warmups-per-client`, exact request/reference paths and provenance; it must not reinterpret old `--concurrency` or `--measured-iterations` fields.

Use the current serving capacities explicitly: scheduler active=1, waiting=64, waiting prompt-token capacity=64×160, `Wait`, 30-second admission timeout, 100 ms prefill aging, maximum sequence length=160, budget/chunk=128, and ten physical/promised KV blocks. Metadata rows/output slots remain one, and the retained graph, packed parents, numerical profile, GPU greedy policy, and stream remain unchanged. `main.rs:971–1008` constructs these serving settings. `Scheduler::submit` (`crates/riley-scheduler/src/scheduler.rs:747`) already accepts bounded waiting requests; `admit_waiting` (`:1457`) admits them FCFS when the active slot and KV promise are available. Waiting requests do not require ten additional physical blocks each.

The complete phase runs on the executor's owning thread:

1. Cold-validate every fixed input/reference, reserve bounded result storage, and prepare C logical slots. Record phase start before the first arrival. For each initial slot, capture arrival immediately before `Scheduler::submit`, bind the unique scheduler request ID, and accept only `Admitted`/`Waiting`.
2. Plan and execute the next ordinary P128/M1 iteration, sample, and call `complete_iteration` exactly as the current benchmark does. Record a token timestamp only after a successful authoritative commit. Require one committed token at most per request per iteration, exact generation indices 0–31, and no event after a terminal record.
3. Process all token and terminal updates from that iteration before admitting replacements. On each successful length completion, free its logical slot and submit its next request immediately, up to the declared total. A replacement joins behind already waiting requests. The old completion backlog must be consumed before submit; do not create a retry loop that conceals admission errors.
4. Continue until the total is submitted, then drain all remaining slots. Record phase finish immediately after the final terminal observation. Require the scheduler to be idle, with no waiting/active requests, pending completions, in-flight iteration, allocated KV blocks or promised blocks. Hash and serialize retained data after this timed boundary.

Use an O(C) live request-ID-to-result-index mapping. The current trial lookup scans every request (`PreparedNativeBenchmarkTrial::request_mut`, `benchmark.rs:448`); copying that lookup over N=10,000 or 100,000 records would add avoidable per-token benchmark work. Reserve raw IDs/timestamps and aggregate counters before timing. Full per-iteration traces should be optional and bounded; the mandatory data is 32 token observations per request, not a GPU event or log write on every operator. Record host storage capacity and dropped/unresolved records; never silently truncate.

The logical offered C is an in-process closed-loop client population, not C GPU streams or threads. Assert observed submitted-but-not-terminal peak C, active scheduler peak <=1, and waiting peak <=C−1 in successful phases. Warmup uses the same loop and each logical slot completes five requests; it must actually reach C outstanding requests. Drain the entire warmup phase before starting retained traffic, retaining the same executor and caches.

## vLLM counterpart

The reusable seam is `benchmarks/lanes/vllm/riley_vllm_benchmark/adapter.py:809`, `VllmBackend.generate_batch`, not its fixed-repeat matrix runner. It already uses `TokensPrompt`, exact greedy fixed length (`min_tokens=max_tokens=32`, `ignore_eos=True`, `detokenize=False`), `RequestOutputKind.DELTA`, host monotonic observations, and internal-ID-aware aborts.

Create a new concurrency entry/driver that reuses the version/model/API guards and constructor, then replaces fixed-batch admission with C-slot refill. Add C requests, call `engine.step`, inspect the complete returned output set, record finished slots, and add replacements before the next step. Keep unique external IDs and returned internal IDs for the whole process, with explicit logical slot and request sequence mappings. Use an O(1) mapping instead of the existing `request_ids.index` scan over the full phase.

- Feed the same pinned 128 raw input IDs directly. Compare returned prompt IDs whenever supplied. Every warmup and retained request must generate exactly the same 32 ordered reference IDs; a shape-dependent vLLM output change fails that setting. Do not rewrite the reference to make it pass.
- Preserve the current strict one-token DELTA rule (`adapter.py:879–899`). Zero-token or multiple-token completion outputs make per-token observations ambiguous and fail the phase. An engine step with no request outputs is different and can be valid. Do not synthesize token timestamps by dividing a multi-token interval.
- Timestamp immediately before `add_request`, and once immediately after each `step` returns. Several different requests may legitimately share that latter observation. Process all outputs before refill, so host mapping/serialization order does not assign different timestamps to outputs from one step.
- Match the HTTP-selected explicit vLLM active capacity >=C and token budget, BF16, context160, seed0, memory utilization0.3, prefix caching off, non-eager/chunked-prefill/graph configuration, model/runtime and environment. Verify actual initialized configuration; do not infer settings from the legacy label.
- The frozen `benchmarks/results/20260911-g04-qualification-followup/vllm_lane.py:16` explicitly asserts `max_num_seqs == 1`. A new entry is required. `_llm_options` (`adapter.py:1115`) otherwise defaults to a different context configuration and lacks the campaign's explicit token budget, so merely invoking the generic adapter is also insufficient.
- Do not reuse `run_benchmark` warmup handling (`adapter.py:1636`), which discards the returned measurements, or its fixed five-warmup/thirty-batch matrix. Validate and retain evidence for all new warmups.

vLLM V1's wall-clock arrival and monotonic token timestamps must not be subtracted. The existing `_engine_timing_durations` (`adapter.py:649`) correctly combines `first_token_latency` with monotonic first-to-last token duration for a sanity check. Primary paired timings should use each process's own consistent host monotonic clock. Report their boundary as **engine host token observation**, not CUDA completion time; engine-reported durations may be supplementary.

## Raw evidence and metric definitions

Use new schemas, for example `riley.engine-concurrency-plan.v1` and `riley.engine-concurrency-result.v1`. Bind the accepted parent plan/qualification only as source and numerical references. Pin the newly built harness source/binary, old graph/kernel inputs, new/shared runners, interpreter, model/tokenizer files, exact request/reference, explicit environment, actual capacities/configuration, and plan bytes before/after every process. A new harness binary cannot inherit the old binary's checksum or old performance receipt.

Each raw request needs `phase`, `request_index`, `logical_client_id`, unique engine/scheduler ID, input/output IDs or their exact reference binding plus independently verified output hash, generated count, status/reason, `arrival_ns`, **32 ordered `token_observed_ns`**, and `terminal_observed_ns`. Prefer retaining the 32 output IDs directly for this small public fixture. Record phase start/end and attempted/successful/failed/rejected/cancelled/unresolved/not-started counts separately; every warmup needs the same correctness gate as a retained request.

For request i, with arrival a, token times t0…t31 and terminal z:

```text
TTFT = t0 − a                         (includes scheduler waiting)
TPOT = (t31 − t0) / 31
ITL[j] = t[j] − t[j−1]               (31 observed intervals)
E2E = z − a
wall throughput = 32 × successful requests / (phase_end − phase_start)
```

Require `phase_start <= a <= t0 <= … <= t31 <= z <= phase_end`, exactly 32 IDs, and finite nonnegative durations. Report TTFT/TPOT/E2E P50/P95/P99 by process, per-request TPOT separately from any pooled ITL distribution, successful wall throughput, and process/pair variation. **Never use summed request E2E as the throughput denominator under concurrency** because queue time overlaps across requests.

Queue admission is optional additional evidence: existing request snapshots expose state but no public admission timestamp. For waiting requests, observing the state transition around `plan_iteration` can give an explicit admission-observation bound; do not call that an exact internal timestamp without a new admission event. Arrival-to-first-scheduled-plan is a useful separately named scheduling-delay measure and must not be subtracted from headline TTFT. No queue timestamp is needed to make the token TTFT above valid.

These measurements exclude HTTP parsing, socket/worker queues, text detokenization and output transport. `InferenceEngine::submit` (`engine.rs:1223`) and `GenerationHandle` currently expose text events, not raw token events. Empty text deltas are suppressed at `engine.rs:1767`; the C02 terminal audit contains tokens but no commit times (`engine.rs:620`). Wrapping these interfaces alone cannot recover token-aligned timing. A later production token-observation side channel would need separate lifecycle, overhead and request-ID qualification; it is a distinct option if actual server-layer token latency is required.

## Qualification, counts and controls

Start with C=1/2/4/8 and the viable vLLM budget/capacity from HTTP screening. Keep P128/O32 and the exact single reference prompt. A fresh one-pair, 256-request/process run can qualify operation and expose gross effects; label it screening. Before choosing a concurrency baseline, use five alternating fresh AB/BA pairs and at least 1,000 retained requests per process. For a tail study, use at least 10,000 per process and report observed duration, process variation and quantile uncertainty; sample count alone does not establish high-concurrency stability. Warmup is five completions per logical client in its own validated phase. There is no HTTP transport warmup in this engine lane.

Every run retains the same GUI-specific no-foreign-compute, <=512 MiB pre-launch GPU memory and <=48 °C cooldown gates as the campaign, with a new workload scope. No GPU/CPU diagnostic collector or build may overlap timed phases. Parent GPU conditions can be checked by the shared preflight, but its c1-labelled receipt must remain a referenced condition receipt, not be relabelled as concurrency proof. Raw/model hashes and JSON writes occur outside timing. Record any background observability sampler explicitly and keep it identical across compared lanes or move it to a separate diagnostic run.

A total per-request deadline must cover queued time too. A long study needs progress/request deadlines, not a fixed 120-second whole-phase limit. On error, stop new submissions, retain partial records, abort known live IDs, and explicitly close the owned executor/process. A synchronous GPU call may not return to its host clock check; the outer runner needs a bounded owned-session watchdog/progress channel. Never terminate an existing service by name, port or GPU PID. Preserve the existing native graph-quiescence, abort and explicit-close checks (`benchmark.rs:1355–1503`) and the vLLM internal-ID abort/core shutdown path (`adapter.py:917–931`).

## Focused implementation tests

1. **Pure host driver:** synthetic step/commit events prove C initial submissions, waiting-state acceptance, immediate per-slot refill, unique IDs, bounded outstanding counts, warmup separation, out-of-order completion handling, and drain after the final submission. C1 fixed trials remain unchanged and reject queued settings.
2. **Timing and output validators:** all 32 exact tokens, including empty/special-token text cases; monotonic boundaries; duplicate/missing/reordered tokens and completions; wrong reference, unknown ID, zero/multiple vLLM DELTA tokens, failed warmups, wrong active/budget/source and old c1 schema rejection. Confirm queue time enters TTFT and common wall, while TPOT uses 31 intervals.
3. **Failure/lifecycle:** admission rejection/timeout, engine exception, incomplete final step, worker/core exit, clock failure, deadline while queued/executing, and cancellation during drain. Partial records and unresolved counts must survive; cleanup cannot depend on a stuck measurement loop returning normally.
4. **Fresh GPU qualification later:** unchanged accepted7 kernels/graph with the new harness; C1 and C2/4/8 exact outputs for every warmup/retained request, repeated graph/KV reuse, peak active=1, idle between phases, and explicit zero-allocation cleanup. Keep full logits/KV numerical proof tied to the accepted parent, then add fresh harness request/ownership proof. A new numerical change would require new full parity qualification.
5. **Paired measurement later:** verify C1's new boundary and instrumentation cost against a fresh old-profile control without pooling its old receipts. Then run the matched offered-C matrix. Keep engine and HTTP results separate; use their difference to investigate transport/queue effects only after boundaries and workload are verified.

This work would establish token-aligned engine evidence under queued client load. It would not itself implement multiple active Riley requests, prove that queuing is the throughput bottleneck, or satisfy the final serving-performance goal.
