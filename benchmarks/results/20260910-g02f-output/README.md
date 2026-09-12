# G02F actual output graph — 2026-09-10

**Actual packed row gather → GPU argmax → D2H now executes inside one graph.**
The host result read is allowed only after the matching replay has completed.
This replaces the G02E diagnostic's outside-graph D2H for this output audit.
It remains a cold standalone output graph, not the full decode aggregate or a
production dispatch/performance qualification result.

## Implementation

The additive native `riley_cuda_graph_capture_begin_output_parent_d2h` accepts
an aligned byte offset into the actual packed index parent and a result prefix
inside an existing pinned host parent. Input, index parent, gathered output,
result output and pinned parent retain their existing exclusive leases.
The offset and contract flag survive capture/graph/exec transitions, participate
in validation, and reset during release. Bounds checks use subtraction after
validating offset <= capacity. The gather kernel reads the specified index
span; D2H writes only the fixed result prefix. The legacy entrypoint continues
to require an exact-sized pinned allocation and zero-offset indices.

`BorrowedOutputGraph` captures the three existing operations, retains exclusive
Rust borrows, waits for native launch completion, and copies the fixed result
records through the existing completion-gated native read. It returns a byte
snapshot, not a mutable pinned view. A failed launch/completion/read makes the
wrapper terminal. Its graph can replay repeatedly while all actual owners
remain leased; explicit close or Drop releases them.

The executor audit obtains the actual output index offset from
`PackedIterationLayout`, checks device bytes against the packed host row map,
and binds `forward.logits`, the packed device slab, `gathered_logits`,
`greedy_results`, and the original `forward.io_staging` pinned parent.
No synthetic row map, separate pinned result allocation or result-download
node outside the graph is used. The post-completion read is a native CPU copy
from that pinned parent. Diagnostic snapshots/restoration still use ordinary
copies outside capture.

After 16 replays the audit verifies the pinned prefix/result records and
untouched pinned tail, complete input/slab preservation, gathered logits and
device result parity. It restores and reads back device output scratch, then
restores the pre-capture pinned snapshot. Existing status decoding validates
records before publishing the token. Transaction/mapping failures poison the
executor; valid non-finite status is a typed, non-poisoning result.

## Verification

Final GPU commands exit 0: **10 tests passed**.

| Gate | Passed |
|---|---:|
| Actual SmolLM2 and canonical model cases | 2 |
| New parent-offset output fixture | 1 |
| Native bounds/abort/completion-read gate | 1 |
| Legacy row-gather/argmax/D2H | 3 |
| Existing embedding/status/D2H | 3 |

Each model runs four iterations and 16 output-graph replays per iteration,
64 per model. Tokens equal independent CPU argmax of baseline logits:
SmolLM2 `[808,2775,288,536]`, canonical `[2,2,2,2]`. Full logits/KV,
continuation, prior operator audits and stable allocation accounting pass.
A deliberately corrupted actual device output row map is rejected with the
specific stale-map error and poisons that executor without allocation growth.
The separate existing stale-RoPE check is preserved. All allocations close to zero.

The new fixture uses an 8-byte index offset in a 32-byte parent and a 16-byte
result prefix in a 64-byte pinned parent. It verifies nonidentity permutations,
device index OOB status, finite recovery, 16 replays per case, close/Drop,
input/index preservation, pinned tail preservation and no allocation growth.
The native test bypasses Rust preflight to reject unaligned/out-of-range/overflow
offsets, preserve legacy pinned-size rejection, verify parent close is blocked
while captured, recover through abort, and reject result reads before any
completed launch. Replay numerical correctness is covered by the initialized
fixture and actual model tests, not the native lifecycle-only test.

CPU: CUDA 84 / graph contracts 32; runtime 258 / architecture 15 / inventory 1.
Format/diff and C ABI signature syntax checks pass. Normal Clippy exits 0 with
no diagnostics in the two new modules; inherited repository warnings remain.

Source hashes match remote scratch. Binary/checkpoint/GPU identities and
unchanged unrelated model-loader source are recorded. No commit/push/deployment.
Host I/O pressure makes these correctness runs unsuitable for speedup claims.

## Remaining

Next: actual input/metadata H2D chain and retained aggregate binding, then full
M=1 decode graph and bucket dispatch. G02F output subgraph evidence does not
promote the global inventory (7 Supported / 7 Unknown, aggregate Unknown).
Qualification and matched SmolLM2 vLLM performance comparison remain pending.
