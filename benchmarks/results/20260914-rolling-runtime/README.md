# Rolling runtime promotion — real GPU correctness gate

`VariableSession::promote_decode_window_successor` now promotes the live successor after scheduler prefix settlement/extension. It retains the successor's ticket, checked expectation, existing row cookies and native model owner; advances only the committed predecessor replay; and issues fresh cookies/replay for the new successor. It does not wait for or resubmit the promoted graph. Existing successor preparation binds the new future input to the live ticket and poisons the owner on any late failure. Quiescence is required before scheduler retirement.

The existing two-slot native transfer ring is reused without CUDA kernel changes. The predecessor slot has already been read/consumed; the successor slot remains live and is required by native future submission. GPU stream ordering places future-token resolution before the next model graph overwrites the shared result. Actual consecutive ring reuse is exercised below; a host fake alone is not treated as device lifetime evidence.

The scheduler exposes prepared window descriptors and `VariableOwnerGeometry` for retained integration. Geometry is caller-provided preparation data, not a dispatch certificate: runtime submission still binds generation/catalog/profile/cookies/replay to its own retained native owner.

## Executed validation

- CUDA-feature runtime state tests:9 passed, including six consecutive promotions with no successor wait/resubmit during promotion, wrong-phase/identity rejection, and injected late preparation failure retaining the live ticket.
- Local scheduler suite:164 passed. Linux CUDA-feature scheduler host library:64 passed.
- Actual RTX4090/SmolLM2-135M GPU test:1 executed, passed, four modes. Four ragged requests exercise full model generation and page progression. Serial mode generates24 tokens per request. Rolling mode performs21 promotions and matches all96 serial tokens. Cancellation mode performs19 promotions, returns lengths[5,24,24,24], and every output matches its serial prefix. All modes end with zero CUDA allocations.
- Failure mode submits a pair, rolls three times, then injects an invalid successor cookie after promotion. Runtime rejects it and prevents another issue. Session close establishes quiescence before scheduler `DeviceQuiescedMutationUnknown` abort, which completes all4 requests with serial-exact committed prefixes and reclaims all CUDA allocations.
- CUDA/server feature check passes. Final compute process inventory is empty. GUI retained; Blender remains down.

The first two build attempts exposed private owner-type visibility and were corrected before GPU execution. They were build errors, not hardware skips or numerical failures. The final expanded GPU run is `/data/riley-serving-260913-recovery/rolling-decode-model-v4`; prior logs remain remote. Host state log: `rolling-runtime-host.log`. Reproduce with `bash benchmarks/analysis/rolling_decode_model_gate.sh PREPARED_ROOT NEW_EVIDENCE_DIR` using the configured CUDA13.0 toolchain and checkpoint.

## Remaining batch work

This is real-model greedy-token equality and lifecycle evidence, not full-logit equivalence or serving performance. The HTTP server still uses fixed pairs; no new throughput/TTFT/TPOT result is claimed. PR20 remains incomplete until server tick state, cookie/replay ownership, stop/cancel/admission and actual streaming are connected and measured against current Riley/vLLM. The four-request model test does not establish C32/high-concurrency or long-soak stability, shared-cache composition, multi-GPU or Hopper/Blackwell runtime qualification.

Server integration must persist the rolling state across worker ticks and return each committed token event promptly. Do not copy this test's inner control loop into a server call that accumulates all outputs until the sequence ends. A published-prefix drain fallback must retain its first result until the second step is drained and settled without duplicate output. Preserve the existing pair fallback and opt-in configuration until matched serving demonstrates the benefit.

[receipt.json](receipt.json) records source hashes; `model/` and the adjacent host logs retain the executed evidence. No Python runs in Riley request processing. Rollback removes the currently uncalled promotion API/descriptor exposure or leaves the server on its existing pair policy; future integration must drain active rolling tickets before switching policies.
