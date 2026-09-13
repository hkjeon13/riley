# Rolling decode scheduler — runtime integration pending

Adds predecessor-only settlement and rolling pair extension on top of the retained KV reservation primitive. `complete_decode_window_prefix` validates the complete predecessor result and preflights publication storage before mutation. It publishes only nonterminal predecessor tokens, leaves all successor pages pending, and retains the exact result. Stop/cancel/output-limit cases defer to full-pair drain. Reservations remain held on errors; a not-dispatched rollback is forbidden once device progress has been published.

`roll_decode_window` advances the logical pair by one iteration, extending each append reservation and returning the old successor as the already-dispatched first plan plus a new second plan. Its first input comes from the validated prior token. Callers must **not redispatch that first plan**. Only the new successor should be enqueued, ordered behind the running step. Waiting admission, cancellation and output/context limits request drain rather than another extension.

If extension fails after changing some rows, the scheduler retains every row and rejects normal pair settlement; quiesced abort owns cleanup. Each successfully extended row updates its authoritative target immediately, so validation and abort still match the actual reservation. Final pair settlement accepts the retained predecessor result exactly and skips its already-published token. Cancellation after prefix publication suppresses the successor and retires pages only after caller-established GPU quiescence. Per-step metrics avoid double-counting an early-published predecessor, and the outstanding gauge falls to one until another successor is reserved.

## Validation scope

Seven new host regressions cover24 rolling advances across page boundaries with scheduler and runtime wire validation, single token publication per step, partial allocation followed by OOM and quiesced cleanup, admission/length fallback, cancel after prefix publication, stale/invalid results, not-dispatched abort rejection, terminal prefix drain, second-step length completion and failed-executor cleanup. Existing ordinary/pair tests remain in the full suite. Synthetic output tokens are host fixtures, not real-model GPU output evidence.

GPU kernels, native transfer ring and server dispatch are not changed in this step. The existing serving release is unchanged. PR20 is still in progress: runtime cookie/replay promotion, result/future-input lifetime, rolling GPU dispatch, token-level streaming and matched prior/new/vLLM serving are required before completion or performance claims. No new serving benchmark is run for this component alone.

Next runtime integration should reuse the existing two native transfer slots only after verifying both host-result consumption and predecessor-token device consumption. Native `submit_buffered_transfer` requires the immediately preceding live ticket. The current Rust `DecodeWindowState` and commit confirmation still assume a completed fixed pair; replacing that transition must retain successor identity and poison/drain handling. A host descriptor pass alone does not prove these device lifetimes.

Rollback: the new APIs are currently uncalled by the server; remove them and their state fields while retaining the existing pair-only path. Once integrated, policy rollback requires draining live tickets first.

Executed: local scheduler suite164 passed, including64 library tests and2 doctests; Linux CUDA-feature scheduler library64 passed. Both logs are retained here. CUDA-feature tests in this step exercise host scheduler contracts, not GPU inference.
