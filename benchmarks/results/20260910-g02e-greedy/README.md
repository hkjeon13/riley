# G02E GPU greedy cold bridge — 2026-09-10

Actual gathered logits and the executor's greedy result allocation are bound
to a borrowed standalone argmax graph. GPU token selection and completion-
ordered result download/status decoding are verified. **D2H is outside the
graph, after graph close. G02F retained output/completion graph is not complete.**

## Implementation

`BorrowedArgmaxGraph` reuses the existing native deterministic argmax capture
primitive and preflight. It retains exclusive logits/results/stream borrows
through graph destruction. Replays wait for known completion; a CUDA failure
makes the graph terminal. Explicit close/Drop release native ownership before
the buffers become available again. No kernel or ABI changes were made.

The M=1 completed-iteration audit borrows actual `gathered_logits` and
`greedy_results`, performs 16 replays, closes the graph, then downloads the
8-byte token/status record through the existing pinned I/O staging owner.
It restores and reads back the original result allocation and checks the
entire logits allocation unchanged. Existing `decode_greedy_tokens` validates
the record before publication. Invalid records or transaction failures poison
the executor; valid non-finite status stays a typed, non-poisoning result.
Uninitialized and unsupported output geometry are rejected before capture.

The model test compares GPU selection with independent CPU argmax over the
baseline executor's logits, then continues decoding. The cold probe does not
replace normal output dispatch, capture the output-row gather, retain a result
read view while replay is in flight, or claim full graph admission.

## Verification

Remote RTX 4090. Final GPU commands exit 0; 8 tests passed:

- Actual SmolLM2 and canonical model cases: 2 tests. Each has four iterations
  and 16 greedy replays per iteration (64 per model). CPU baseline tokens
  match: SmolLM2 `[808,2775,288,536]`, canonical `[2,2,2,2]`. Full logits/KV,
  prior operator audits, continuation and allocation accounting also pass.
  All allocations close to zero.
- Borrowed fixture: 1 test. Three rows exercise finite ties, signed-zero ties,
  NaN, positive infinity and negative infinity, followed by finite recovery.
  Tests 16 replays per case, explicit close and Drop, short-result rejection,
  input preservation and untouched result tail; allocation counts stay stable.
- Existing BF16 argmax eager tests: 3.
- Existing owned argmax graph lifecycle/parity tests: 2.

CPU: CUDA library 84 / graph contracts 32; runtime library 258 / architecture
15 / inventory 1 passed. Format/diff checks pass. Normal Clippy exits 0 with
no diagnostics in the two new modules; inherited repository warnings remain.

Source SHA256s are checked against remote scratch. Binary/checkpoint/GPU
identities and unchanged model-loader source are recorded. No commit, push or
deployment. These are correctness runs on a shared host, not timing evidence.

## Remaining

Next G02F: bind the actual output row mapping, pinned result destination and
completion-scoped read into a retained output graph, with stale/invalid mapping
and status/lifetime checks. Packed metadata offsets and pinned destination
capacity must be handled explicitly; do not substitute a synthetic row map and
claim actual-owner integration. Then remaining H2D/aggregate/full decode graph,
qualification and matched SmolLM2 vLLM comparison. Global inventory remains
7 Supported / 7 Unknown, aggregate Unknown.
