# G03 retained transfer graph — implementation pending GPU verification

## Implemented locally

The existing native resource owner can now build an explicit three-node graph:
H2D → device-to-device copy → D2H. The nodes use the same retained ledger and
have explicit dependency edges. All four parents must be registered, distinct
within their memory kind, nonempty, equal sized and from the same context.
Repeated graph construction is rejected. No stream-capture guard is bypassed;
this uses explicit CUDA graph construction, not `cudaStreamBeginCapture`.

Replay requires a fresh exact-sized host slice on every call. It stages into the
leased pinned input, launches the retained executable and synchronizes before
making output readable. Every attempted replay invalidates prior output,
including a wrong-length attempt. Read is rejected before completion or after a
failed replay. The safe Rust API uses retained parent indices and borrowed byte
slices; it exposes no raw CUDA stream or allocation pointer.

Close destroys the executable and graph before releasing any parent. Failed
launches still attempt stream synchronization. Unknown completion or ambiguous
graph-destroy consumption keeps the owner and all leases retained, preventing
speculative release or a second destroy attempt. Such failure paths have not yet
been fault-injected. This remains a transfer lifecycle probe: it does not record
model operators, qualify full decode replay, or change production dispatch.

## Verification actually completed

- CPU CUDA library: 84 passed.
- CPU graph contracts: 32 passed.
- C11 public ABI syntax/signature check: passed.
- Formatting, diff whitespace checks and normal CPU Clippy: passed. Existing
  warnings remain; none name the new resource module.
- Six unrelated model-loader files retain their original hashes.

These CPU checks do not compile CUDA-enabled Rust branches or native CUDA code.
The added ignored GPU test covers membership, alias and size rejection, result
read before completion, rejected-replay stale-result invalidation, retained
parent close rejection, 512 successful fresh-input replays over eight graph
owners, explicit close/Drop and final context cleanup. It has **not run**.
The two existing native ledger tests should run alongside it.

## Remote verification blocked by automatic approval review

Read-only SSH reached the existing host. Automatic approval review rejected
sending the source overlay to `ai-assistant:/tmp/riley-g01-native-260910`, citing
unverified destination ownership and insufficient explicit approval to transfer
internal source code. No remote source write or GPU run occurred in this turn.
No workaround was used. Local implementation and checks were completed.

The concrete pending transfer is narrowed to these five source files, replacing
only their counterparts in the existing temporary test directory:

- `kernels/include/riley_cuda.h`
- `kernels/src/graph_resources.cu`
- `kernels/tests/abi_layout.c`
- `crates/riley-cuda/src/ffi.rs`
- `crates/riley-cuda/src/graph_resources.rs`

`pending-transfer.json` records previous/current hashes and the destination.
It excludes model-loader changes, credentials, environment files and checkpoints.
User approval is needed to transfer these five files and run the scoped CUDA
build plus `aggregate_` native library tests there. Until then, the implementation
is unverified on GPU and should not be considered ready for graph integration.

The full embedding/layer/LM-head/output-status DAG, its replay status contract,
G02H/G03 qualification and matched SmolLM2 vLLM comparison remain incomplete.
No performance claim, commit, push or deployment was made.
