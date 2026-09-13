# Retained expectation pipeline batch

The batch removes repeated immutable expectation validation from completion/predecessor preparation and replaces quadratic predecessor-owner coverage scans with a sorted index. Host costs fall substantially, but matched serving gains remain small. Keep the correctness-preserving implementation; do not claim the vLLM goal achieved.

## Changes and safety boundary

1. `OwnedCheckedExpectation` owns the fully validated expectation. Its field is private and only immutable borrowing/Deref is exposed; it has no DerefMut. A test-only consuming conversion explicitly discards the proof. This is not a trusted boolean, device-completion proof or live scheduler authorization.
2. Retained packet encoding and ordinary/predecessor result validation use a checked borrow from that owner. Public APIs for arbitrary expectations still validate first. All GPU status/identity/token/extent/inactive-byte checks remain; full-logit results still scan nonfinite/argmax values.
3. Future preparation reuses the checked predecessor and sorts successor owner pairs once for binary-search coverage, replacing a nested `previous × successor` linear scan. Every predecessor pair, including cache-only/off-batch owners, must remain present. Successor structural validation and replay/append/token-source checks remain.

The successor is still updated from validated GPU predecessor tokens and then fully validated at completion. This change does not introduce reusable successor certificates or new metadata buffer reuse. Given the measured overlap and small serving gain, further allocation-only refinement is deferred rather than claimed complete. Scheduler live authority, graph holds, generation/cookie, poisoned-session, event/drain and cancellation behavior are unchanged. Native CUDA kernels and wire bytes are unchanged; Rust never calls Python in serving.

## Validation

| Check | Result |
|---|---|
| Local runtime lib | 351 passed; 1 existing timing diagnostic ignored |
| Full local scheduler suites | 155 passed |
| CUDA scheduler lib | 55 passed |
| Staged-session lifecycle/mock GPU completion | 7 passed |
| Actual GPU model gates | 2 executed, 2 passed, 0 ignored |
| Full-model logits comparisons | Existing cache1,474,560 bytes and query-reuse/cache2,359,296 bytes exact |
| CUDA/server feature check and release build | Passed |

New tests verify rejection of missing owner authority, ordering-independent owner coverage, off-batch owner removal rejection, identical request bytes, and identical compact-result accept/reject decisions after flipping every result byte. Staged-session tests exercise corrupt predecessor/successor, poison/retention, preparation ordering and shared capability mismatch. GPU tests close with zero retained allocations. This is not new multi-GPU/Hopper/Blackwell execution qualification.

## Non-Nsight host means

Prior/new runs each include64 warmups+256 retained requests per shared/unique workload, all reference-exact. Same composed flags, query reuse disabled. Timer mixes differ slightly as iteration counts change, and timers are nested/overlap GPU work.

| Timer | Shared prior → new | Unique prior → new |
|---|---:|---:|
| Retain/encode | 212.02 → 216.78 µs | 160.15 → 163.32 µs |
| Read/validate | 198.02 → 81.67 µs | 164.27 → 26.74 µs |
| Future preparation | 1,065.60 → 566.77 µs | 967.83 → 541.20 µs |
| Buffered wait | 2,071.42 → 2,270.21 µs | 4,197.80 → 4,290.11 µs |

CPU work decreased while explicit GPU wait increased. Together with the small serving change, this supports the inference that much of the removed work already overlapped GPU execution. Do not add nested timers or convert the CPU percentage to a throughput prediction.

## Evidence and rollback

[receipt.json](receipt.json) records checks/source/log hashes. [phase-comparison.json](phase-comparison.json) is recomputed from the prior archive and [new host-phase archive](host-phase.tar.gz) by `export_retained_authority_phase.py`. [Serving results](../20260914-retained-authority-serving/README.md) include matched vLLM, stop/cancel/recovery and raw response evidence.

Prior release preserved at `/data/riley-serving-260913-recovery/riley-before-retained-authority-v1`, SHA256 `1fc55a483be57e8822bd45d24f2dbfa606cee2569c031272582a907aa223646e`. New release SHA256 `a5dda5e9d8260c3797d61cd13477f756c35bbec5716c4d50ec3685f990df6a1e`. The remote selected-file mirror matches `evidence/changed-source-sha256.txt`; no clean remote git checkout is implied. Rollback uses the saved binary or reverts the three runtime source changes. Query reuse and other experimental numerical options remain unchanged and disabled for this comparison.
