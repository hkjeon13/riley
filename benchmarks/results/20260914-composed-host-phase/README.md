# Non-profiler host phases — retained expectation pipeline selected

The non-Nsight diagnostic confirms expensive host work in successor preparation (about1 ms/call) and retained expectation encode/result validation (about0.16–0.21 ms/call). Scheduler plan, sampling and commit are much smaller. Select a structural retained-expectation/metadata batch rather than more scheduler threshold or query-tile tuning.

Same current release `1fc55a483be57e8822bd45d24f2dbfa606cee2569c031272582a907aa223646e`, SmolLM2-135M BF16, RTX4090, C32, active32, batch/chunk512, context1024, KV2048 pages/cache512 pages. Prefill-FFN/paired/adaptive enabled; query reuse explicitly disabled. This measures the existing composed control. Only `RILEY_SERVING_PHASE_TIMING=1` is added. No Nsight or overlapping builds. GUI retained and Blender down.

Each workload executes64 warmup+256 retained requests; all640 requests are reference-exact with32 output tokens. Both lanes exit0 and leave no compute process. Phase timers cover warmup and retained together. No new performance comparison or default promotion is claimed.

## Mean host timer duration per call

| Runtime timer | Shared prefix | Unique prompts |
|---|---:|---:|
| Retain/encode | 212.02 µs | 160.15 µs |
| Buffered submit | 25.83 µs | 26.29 µs |
| Buffered wait | 2,071.42 µs | 4,197.80 µs |
| Read/validate | 198.02 µs | 164.27 µs |
| Future preparation | 1,065.60 µs | 967.83 µs |

For shared-prefix paired steps, mean scheduler planning is86.70 µs, sampling17.14 µs and commit/publication25.91 µs per pair. For unique paired steps they are66.62/17.82/20.63 µs. The runtime timers are nested inside server execute timers; do not add them to server wall time. Waiting includes device execution and future preparation partly overlaps predecessor GPU work. A1 ms preparation reduction is not necessarily1 ms off request latency.

## Code evidence and next batch contract

`VariableSession::retain_submission` encodes a fully validated expectation. The same immutable retained expectation is then validated again by result APIs. Future preparation validates the predecessor again, clones the successor to adjust structural replay state, validates it, and encodes it. The closure inside the measured future timer also builds the successor expectation. `variable_wire::validate` reconstructs page/owner/row maps; these timings do not isolate its exact share from cloning, construction and encoding.

Implement one batch of related changes:

1. Introduce an owned, immutable validated expectation capability that retains checked geometry and page-owner authority. No mutable expectation/ledger access escapes. Keep public APIs for arbitrary expectations fully validating.
2. Reuse that capability for predecessor identity/packet encoding and completion validation. Continue checking every result identity/status/token, inactive byte, owner generation and replay/iteration binding; skip only reconstruction of facts proven by the unchanged owned expectation.
3. Prepare successors using the validated predecessor and explicit append/replay/token-source checks. Reuse bounded metadata buffers where ownership permits it. Tentative replay and GPU token substitution must not mark a request committed.
4. Preserve event/drain, cancellation, shared/off-batch-reader, page-generation and poisoned-session barriers. Test tampered raw expectations/results, invalid successor sources, stale cookies, cache-only owners, cancellation and cleanup before matched serving evaluation.

This is not authorization to trust packets, drop live scheduler authority, or reuse a certificate across pool reset/owner mutation. A borrowed/owned proof must prevent mutation, not rely on a boolean flag. The current successor expectation is intentionally updated with validated GPU tokens before completion checking; any replacement must model that transition explicitly.

Measure again with the same phase timers and a same-day baseline/new-Riley/vLLM serving comparison. Do not promote from a synthetic serialization benchmark. Existing exactness/alternate-quality failures and broader hardware/soak qualification remain unchanged.

## Evidence

[receipt.json](receipt.json) parses numeric phase logs and checks all responses. [evidence.tar.gz](evidence.tar.gz) contains launch configuration, frozen references, raw rows, engine logs and completion/exit records; no profiler data is included. Reproduce with `python3 benchmarks/analysis/export_composed_host_phase.py benchmarks/results/20260914-composed-host-phase`. Remote completed run: `/data/riley-serving-260913-recovery/composed-host-phase-v1`.
