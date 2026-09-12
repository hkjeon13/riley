# Multi-sequence descriptor v1 CPU prototype

This isolated, dependency-free Rust crate implements the packet and admission
component of the planned genuine N2/N4 graph owner. It does not change production
crates, execute CUDA, or establish a performance result.

The byte contract is the frozen
[`MULTISEQUENCE_DESCRIPTOR_CONTRACT.md`](../../results/20260912-serving-optimization/MULTISEQUENCE_DESCRIPTOR_CONTRACT.md),
SHA-256 `5bed6d70342dec4f30c85ba67dd41cf560e6b7f4b0de5dedfe59424cecc9b641`.
The independent [golden fixtures](fixtures/README.md) pin the request and result
bytes separately from the Rust encoder.

## Run locally

From the repository root:

```sh
cargo test --manifest-path benchmarks/prototypes/multisequence_descriptor_v1/Cargo.toml --offline
cargo fmt --manifest-path benchmarks/prototypes/multisequence_descriptor_v1/Cargo.toml --check
```

The manifest declares its own workspace and has no dependencies. Build output is
confined to this directory's ignored `target/`. No environment variables, model
checkpoint, network access, CUDA library, or production test target are needed.

## Boundaries and usage

The adapter supplies `SubmissionExpectation` from an exclusive, current owner
and scheduler snapshot. Its fields have distinct authority:

| Input | Required external source |
| --- | --- |
| `OwnerExpectation` | Actual owner generation, last admitted replay, prepared graph catalog and its digest, cold capacity, physical pool size |
| `ReservationExpectation` | Registered reservation cookie, live sequence identity, exact input IDs, committed prefix, reserved target table, requested output bound and scheduler output slot |
| `block_ownership` | Independent live/reserved KV ledger, including off-batch resident requests |
| `CompletionEvidence` | Actual stream/graph quiescence established by the future runtime adapter |
| Settlement booleans | Completed scheduler commit or failure containment performed by that adapter |

Packet data never creates any of these authorities. In particular, physical IDs
within pool bounds are rejected unless the independent ledger assigns them to
the submitted sequence. The prototype checks structural consistency of the
supplied snapshot; it cannot verify that an application supplied a truthful one.

The low-level codec functions can be exercised independently:

- `encode_request` emits exactly 1,280 little-endian bytes after validating the
  expectations themselves.
- `decode_request` checks every field, unused entry and reserved byte against
  those expectations. Its returned rows are decoded data, not live reservations.
- `validate_bulk_result` accepts only a quiesced, complete, exactly sized result.
  It validates every row before returning any row, then orders the results by
  dense scheduler output slot. Full logits retain their original BF16 bytes;
  nonfinite active logits and nonzero inactive logits are rejected.

`CodecOwner` adds replay and readiness state across submissions:

1. Construct an owner and call `admit` with the encoded packet and the complete
   independent expectation. Invalid preflight does not advance its replay.
2. Submit and wait through the real runtime adapter, then call `accept_result`.
   Only the original admitted expectation is used; callers cannot replace it
   with a later reservation snapshot.
3. Read `result()` only after complete validation. The caller may now perform
   whole-iteration scheduler completion, including its deferred cancellation
   routing. Call `settle_success(true)` only after that commit succeeds.
4. On unknown completion, `settlement_advice()` is `RetainAll`. A later external
   quiescence assertion permits containment, never another publication attempt.
   On a quiesced failure, contain affected requests before `settle_failure(true)`.
   Failed owners remain poisoned; a fresh owner requires the real runtime's
   teardown/recovery policy.

A second admission while a transaction is unsettled invalidates output
readiness, preserves its reservation snapshot, and poisons the owner. Repeated
result attempts also invalidate the prior getter. The prototype does not revoke
data that a caller already copied; only the real scheduler controls publication.
`Transaction` exposes the same per-submission readiness model without the
cross-submission replay guard; integration should use `CodecOwner` or provide
equivalent external owner state.

## Supported exact layouts

| Mode | Active rows | Selected bucket | Request bytes | Result bytes |
| --- | --- | --- | --- | --- |
| P128, greedy | 1 request / 128 input tokens | 1 | 1,280 | 640 |
| P128, full logits | 1 request / 128 input tokens | 1 | 1,280 | 98,944 |
| Decode, greedy | 1 / 2 / 3 / 4 | 1 / 2 / 4 / 4 | 1,280 | 640 |
| Decode, full logits | 1 / 2 / 3 / 4 | 1 / 2 / 4 / 4 | 1,280 | 98,944 / 197,248 / 393,856 / 393,856 |

All selected `(stage, bucket, result mode)` entries must exist in the explicitly
prepared catalog. A maximum-N allocation never implies an exact-M plan exists.
The fixed P128 compatibility view at byte 640 is 600 bytes; decode leaves that
entire region zero. N3 uses a zero inactive fourth request/result row and, in
full-logit mode, a zero fourth logits row. Output slots can differ from execution
row order, but must form a dense permutation of the active rows.

Decode's numeric KV ceiling permits position 159, but the public P128/O1..32
contract exhausts output index 32 there. Admission therefore rejects that row;
the final admitted O32 decode position is 158. Each request's independent
requested output bound is checked as well.

## Verification coverage

The tests include both independently reconstructed golden files; all 1,280
single-byte corruptions for N3 decode and P128 request packets; exact sizes and
modes across active counts 1–4; heterogeneous positions and noncontiguous block
tables; off-batch ownership conflicts; cookie/tag/slot/physical-ID duplication;
wrong valid tails; integer overflow; stale replay/iteration/generation/catalog
identity; inactive-row and logits poisoning; malformed final-row atomicity;
signed-zero/subnormal BF16 preservation and nonfinite rejection; unknown
completion containment; repeated result reads; busy-owner replacement; and
4→3→1→2 owner reuse with fresh replay IDs after explicit settlement.

## Limits before production integration

- This is an allocating CPU prototype. It uses standard collections and copies
  full logits; no hot-path allocation or latency claim is made.
- Owner construction records its cold expectation; structural admission checks
  run before any packet is accepted. Catalog entries and digests are supplied
  assertions, not actual captured graphs or signed qualifications.
- There are no GPU handles, leases, device buffers, graph nodes, result-stamp
  kernels, D2H copies, scheduler mutations, KV commits, cancellation queues or
  resource reclamation here. Dropping a prototype value cannot release real
  resources because it owns none.
- Quiescence and settlement inputs are explicit adapter assertions. They do not
  independently prove hardware completion or atomic scheduler mutation. A
  production adapter must retain real reservations/leases across these states,
  validate all results before scheduler completion, and route cancellation only
  after the iteration is resolved.
- Finite logits and a valid argmax status do not prove model arithmetic. The
  prototype does not recompute argmax or compare outputs against a checkpoint.
  Genuine M2/M4 GEMM, row-stride, fused attention, complete-logit/KV and serving
  qualification remain separate gates. No fallback to serial M1 is implemented.

The next integration must bind this codec to the actual exclusive graph owner,
registered reservation ledger, graph catalog and scheduler completion path;
test malformed/stale results and cancellation with those real resources; and
then run the independent GPU arithmetic, full-model and serving gates.
