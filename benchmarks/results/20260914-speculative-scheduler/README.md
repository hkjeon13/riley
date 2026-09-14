# Host speculative scheduling and completed-prefix settlement

This batch adds an experimental scheduler path for up to four decode owners,
with up to seven prompt-lookup draft tokens per owner. Plans retain immutable
input tokens and exact private append block tables. Scoped execution authority
holds the scheduler borrow. Ordinary execution/completion rejects these plans.

After callers establish device quiescence and append-only writes, completion
validates all target rows, including unused rows after mismatch/cancel, before
any KV or output publication. Each request commits only the verified prefix,
discards the reserved suffix, emits ordered multi-token events, and retains the
last output as the next pending decode input. Cancellation emits no tokens;
EOS and length settle once. A commit failure contains the whole unpublished
batch. Verified input work is recorded as decode work in iteration metrics;
this metric is not the number of output tokens emitted.

The current prototype requires greedy argmax target rows and no shared prefix
cache. Waiting admission or prefill, absent lookup matches, or fewer than two
remaining output slots fall back to ordinary planning. This is not enabled in
serving. GPU descriptor/graph adaptation is **not connected**: current device
progress validation permits only prompt prefill or single-token decode. A real
verification stage must preserve actual prompt/generated progress rather than
masquerading generated inputs as a longer prompt.

## Verification

- 40 four-owner rounds cover five page-boundary prompt lengths and all eight
  mismatch/bonus positions, with different acceptance lengths per owner.
- Invalid final target rows, foreign plans, replayed completion and ordinary
  completion paths are rejected without publishing progress.
- Next ordinary decode authorizes successfully and consumes the exact final
  emitted token after partial KV settlement.
- Seven output budgets cover cancellation, EOS and length; both undispatched
  and quiesced-unknown aborts reclaim reservations.
- A forced second commit failure publishes zero tokens and reclaims all owners.
- One remaining output slot falls back to ordinary decode.

Run `cargo test -p riley-scheduler`. The retained log contains CPU unit,
integration and documentation test outcomes. No GPU was used or Blender stopped
for this batch. No serving performance or vLLM improvement is claimed.

Next: explicit verification progress and native validation, adapter binding to
retained argmax completion identities, strict serial GPU equivalence, then a
matched serving comparison before default policy decisions.
