# G03 native aggregate resource ledger — 2026-09-11

## Implemented boundary

Added an opaque native `RileyCudaGraphResources` reservation with an additive
C ABI, private Rust FFI handle and safe borrowed-parent wrapper. It owns one
stream, one context-child reference, and a deduplicated ledger of device,
pinned-host and selected GEMM-plan leases. Repeated native handles reuse the
same lease. Registration is owner-thread confined and permits no raw stream
escape, capture, kernel launch or result publication.

The ledger admits up to 4096 occurrences and 1024 unique resources including
the stream. It validates all parent contexts and selected no-split plan topology
before acquisition. Existing GEMM algorithm/policy objects are retained intact.
A busy resource or capacity failure rolls back prior acquisitions in reverse
order. Close removes an entry only after known release; on an unexpected release
failure its pointer and remaining leases stay retained. No CUDA work is submitted,
so this boundary has no pending GPU completion to infer. Wrong-thread close is
rejected; null-owner close is idempotent. Existing graph guards remain intact.

The safe wrapper holds exclusive Rust borrows until native close/Drop. Runtime
cold validation borrows the actual uploaded physical weights, activation scratch,
KV pools, packed metadata, output buffers, five GEMM plans and two pinned parents.
For SmolLM2 this records **297 device parents, 5 plans, 2 pinned parents**, plus
the stream. It releases them before normal operator audits and continuation.

## Verified

- Native GPU lifecycle tests: **2 passed**. Cover duplicate registrations,
  busy buffer and busy plan rollback after prior acquisitions, resource/context
  close rejection while reserved, capture rejection while the stream is reserved,
  foreign contexts/plans, capacity boundaries, null table, active-capture rejection,
  wrong-thread close, 16 close/Drop cycles, and successful final context close.
- Runtime C07 GPU tests: **14 passed**. Five model cases perform **20 actual
  reservation/close calls** across four consecutive iterations. SmolLM2 L30,
  canonical L2 and synthetic L3 retain the previous full-logit and initialized-KV
  hashes and continuation tokens. Existing operator audits, status rejection,
  allocation stability and zero-allocation cleanup also pass. See `model-parity.json`.
- CPU: CUDA library **84**, graph contracts **32**, runtime library **259**,
  architecture **15**, inventory **1** passed (**391** total).
- C11 ABI function signatures compile in the native build. Cargo now watches
  the new native source in addition to listing it in CMake.
- Normal Clippy passed for CPU and CUDA libraries. Existing warnings remain;
  no diagnostics name the new resource modules. Format and diff checks pass.
- All 56 overlay source hashes match remote scratch. Binary/checkpoint/GPU
  identities are recorded. Six unrelated modified model-loader files are unchanged.
  No commit, push or deployment occurred.

## Build and execution notes

The initial SFTP transfer stalled; direct SSH transfer succeeded. The first build
used C++ allocating new/delete and failed the existing C-runtime-only link contract.
It was changed to the repository's malloc/placement-new/free convention. A second
link attempt exposed a missing Cargo rerun input for the new CUDA file, which was
added before the successful rebuild. Original failure logs are preserved.

The host showed about 79% I/O pressure; the test binary initially waited in
`folio_wait_bit_common`. An overlapping model run was stopped, then restarted
only after the ledger tests finished. Both final GPU commands exit 0. No timing
or speedup inference is made from these development runs. Unexpected corrupted
lease-counter release is fail-closed in code, but was not fault-injected here.

## Still required for a full graph

This is an implemented **resource reservation**, not a native aggregate capture
or executable graph. The native recorder must next consume this ownership model,
validate operation-specific spans/alias rules and dependencies, and record the
complete H2D → embedding → layer chain → final norm/LM head → output/status D2H.
Resource registration alone does not validate any operator DAG or grant capture
admission. Fresh input per replay and completion-gated status publication remain
unimplemented for the aggregate.

One retained full M=1 graph must then pass changing tokens/positions, independent
logits/token/KV comparisons, failure/cleanup and allocation checks. G02H/G03,
bucket qualification and matched SmolLM2 vLLM measurement remain incomplete;
global inventory stays 7 Supported / 7 Unknown with aggregate Unknown.
