# G03 transfer graph — authorized remote verification

The user explicitly approved sending the five source files listed in the prior
pending-transfer receipt to `ai-assistant:/tmp/riley-g01-native-260910` and running
CUDA validation. Only those five files were transferred. Remote pre-transfer
hashes matched the previous resource-ledger receipt; post-transfer hashes match
the approved local sources. No model files, credentials or checkpoints were sent.

## Result

The native CUDA library and CUDA-enabled Rust test binary compiled successfully,
including the C11 ABI checks. **Three GPU tests passed**, with the final direct
binary invocation returning **exit code 0**:

- Retained transfer graph: eight graph owners, 512 successful fresh-input
  H2D → D2D → D2H replays. Exact bytes match each changing input. Unregistered
  parents, aliases, wrong sizes, repeated construction and reads before completion
  are rejected. A rejected replay invalidates the preceding output; a later valid
  replay recovers. Explicit close and Drop both release parents; context close passes.
- Resource context/capacity/capture rejection regression.
- Duplicate registration, busy-resource rollback and close/Drop regression.

The first cargo command built successfully and logged all three passing tests,
but returned timeout status 124 after substantial host I/O delay. That log is
retained as `gpu.log`; it is not relabeled exit 0. The same freshly built binary
was then run directly with a separate timeout, yielding the clean three-test
result in `gpu-final.log` and exit code 0. Device and binary identities are recorded.
No performance conclusion is drawn from these runs.

Prior CPU evidence remains in `../20260911-g03-transfer-replay`: 84 library tests,
32 graph contracts, ABI syntax, format/diff and normal Clippy checks. These were
not rerun because no source changes were needed after the approved transfer.
Six unrelated modified model-loader files still match their prior hashes.
No commit, push or deployment occurred.

## Remaining scope

This validates a **three-node transfer graph and its resource lifetime**, not a
complete model decode graph. Embedding, all decoder-layer operations, final norm,
LM head and output/status handling still need to be recorded in one retained
model graph, then verified against independent eager logits/token/KV results.
G02H/G03 full-model qualification, bucket dispatch and matched SmolLM2 vLLM
performance comparison remain incomplete. The prior source-transfer approval
block is resolved for these five files and this validation.
