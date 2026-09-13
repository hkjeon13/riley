# Rolling decode reservation foundation — integration in progress

Implements `SequenceState::extend_reservation` for PR20. It preserves pending physical pages and committed length, adds only required new pages, updates table bounds and rotates the detached/retained reservation nonce. Existing `commit_prefix` can now be followed by another extension without draining and retiring the pending suffix. Old GPU work must use immutable table snapshots and remain ordered ahead of the new suffix; this host API is not a GPU fence.

Tests exercise six starting lengths with48 consecutive prefix-commit/extension transitions each, page boundaries, stale token rejection, target/capacity errors, nonce exhaustion, allocator failure after one successful new page allocation, retry, poison cleanup, and cache/off-batch shared-prefix readers. Existing physical handles never move. Committed prefix pages survive suffix discard and poison. Failed extension leaves the prior table and detached authority usable; allocation lifetime counters may record attempted allocations.

This is a component of the same rolling scheduler/ticket/streaming optimization batch, not its completion or a new serving milestone. The production server does not call this API yet. Scheduler partial settlement, terminal-event handling while a successor is live, GPU ring ownership, rolling dispatch and matched serving remain to be implemented. No kernel or runtime Python dependency is introduced. No throughput gain or GPU execution qualification follows from these host tests.

Validation logs and source hashes are stored here. CUDA-feature tests still test host KV ownership; they are not device model tests. The prior serving release is unchanged remotely. PR20 hardware/runtime gates remain pending; no executed failure is classified as a hardware skip.

Rollback: remove the currently uncalled extension method and its tests; existing ordinary and pair APIs are unchanged. Follow-up runtime integration must provide drain-aware policy rollback before promotion.

## Executed checks

- Local runtime library: 355 passed, 1 existing diagnostic ignored.
- Local scheduler suite: 157 passed, including doctests.
- Linux CUDA-feature runtime KV host tests: 37 passed.
- `git diff --check`: passed.

The full runtime log includes long-running descriptor corruption tests, which completed successfully; they were not cancelled or skipped.
