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

## Candidate batch in verification

Implemented bounded direct page-owner/shared/alias lookup tables (pool limit
remains 4,096), dense output-slot bit checks, and tentative structural replay
validation against an immutable successor without cloning the whole ledger.
All ownership, COW, alias-position, cookie, slot, progress and committed replay
checks remain. No proof is cached across mutable iterations.

The old tree validator is retained only under cfg(test). A deterministic
9,600-case differential check covers capacities 8/16/32 plus shared-prefix
mutations, including identical error values/order. Six future-wire tests,
72 scheduler tests, 21 existing wire tests, and nine CUDA-feature runtime
ticket tests pass. Final server build passes. New opt-in subcounters separate
future authority construction from structural validation/encoding for the
next profile; they do not change runtime policy.

Frozen candidate SHA256:
`a525729d037b519e9c796b7574f960820fb6cbeb1e0d60e4a8a504c4cd616403`.
Matched C32 serving and the follow-up profile are complete. Throughput improves
5.48% shared / 6.31% unique over frozen best; TTFT/TPOT and E2E P95/P99 improve
in the run-median summary. Candidate remains 10.54% / 9.75% below vLLM 0.29.0.
All 6,912 serving and 512 profiler responses validate, including exact Riley
references and stop/cancel/recovery. Retain this measured C32 implementation;
broader concurrency and hardware qualification remain pending.
Evidence: `benchmarks/results/20260914-dense-wire-serving-c32/README.md`.

Future preparation drops in the diagnostic profile, with candidate check/encode
still larger than authority construction. Call/shape mixes differ, so do not
turn those sums into a serving speedup claim. Unique CPU preparation is now a
smaller target; next establish C8/16/64 behavior before another execution batch.
