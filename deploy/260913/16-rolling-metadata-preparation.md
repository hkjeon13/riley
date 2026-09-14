# Rolling metadata preparation batch

## Evidence

`benchmarks/results/20260914-rolling-host-profile/README.md` fills the missing rolling host timing scope. Future authority/check/encode preparation consumes 59.3ms shared / 48.2ms unique over 83/85 calls; the timer stops before GPU submission. Continuation wall time is larger and includes submission/synchronization. The latest C32 baseline remains 14.14% / 17.13% below vLLM 0.29.0 throughput.

## Scope

First split the existing preparation counter into authority construction and future structural validation/encoding. Use that evidence to implement one linked metadata/ownership batch:

1. Reuse an immutable validated ownership representation between retained predecessor and successor, with explicit owner generation and append-page changes.
2. Encode a structurally checked successor without cloning its complete expectation merely to change a replay validation field. Preserve actual committed replay independently from tentative predecessor replay.
3. Reuse bounded packet/reference buffers where lifetime and ticket ownership permit; retain immutable checked values and prevent mutation while GPU transfers are pending.

Do not skip ownership/replay validation or reuse stale proofs. Keep cache-only/off-batch owners, future page extensions, cancellation, stop suppression and commit-before-publication contracts. Avoid changing GPU arithmetic or defaults in this batch.

## Validation

Corrupt owner, replay, cookie, slot, page mapping, alias/writable ownership and progress cases must retain rejection. Compare encoded packet/sidecar bytes to the existing path at C1/8/32 and page boundaries. Run relevant scheduler/runtime tests, exact model/serving references and stop/cancel/drain. Compare a milestone serving screen against frozen best Riley and vLLM 0.29.0 in reversed orders, then broaden only on a confirmed gain. Restore Blender after GPU work.

## Status and rollback

Opt-in rolling instrumentation is implemented and validated against 512 exact serving responses. Optimization implementation is pending finer attribution. Rollback the eventual batch to the frozen prior code/identity; keep measurement receipts and rejected variants. Do not treat this plan or diagnostic as a serving performance improvement.
