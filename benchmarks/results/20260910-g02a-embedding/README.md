# G02A borrowed Embedding audit — 2026-09-10

The actual embedding table/token/output bridge and cold executor audit are
implemented and GPU-verified. Retained C07 aggregate admission and full decode
graph integration remain pending. This is correctness, not speedup evidence.

## Implementation

BorrowedEmbeddingGraph retains the actual BF16 table, token parent, output,
error scratch, exact 32-byte pinned report and stream through graph destruction.
It reuses the existing C05-20 validation/gather/status-D2H primitive; no native
kernel or ABI was changed. Geometry multiplication, capacity, report size,
resource identity and leases are checked before capture.

Replay waits for known completion and then reads the native-validated status.
Success requires a consistent report. The earliest invalid token is returned as
a reusable TokenOutOfRange value with no output writes; CUDA/report failures
instead make the owner terminal. Report storage cannot be borrowed independently
while the graph holds it.

The actual M=1 executor audit derives the embedding physical weight and BF16
shape from its plan, validates that packed token IDs occupy the expected byte-zero
prefix, and checks device token bytes against the packed request. It uses the
actual uploaded table, hidden_current and embedding_error_scratch. A caller-owned
cold 32-byte report is supplied separately; it is not a new allocation in normal
production dispatch.

After 16 replays, output is compared with the exact selected BF16 table row as an
independent gather oracle. The entire table and token parent are verified
unchanged; output tails are preserved. Original output and error scratch are
restored and read back before returning. Transaction failure poisons the
executor. The existing completed-iteration requirement protects uninitialized
scratch.

This audit runs between completed iterations. It does not integrate embedding
into a retained full decode graph or promote global capabilities. The global
inventory remains 7 primitive Supported / 7 Unknown, aggregate Unknown.

## Verification

**6 GPU tests passed**, all commands exited 0:

| Gate | Passed |
|---|---:|
| New borrowed embedding status/preservation fixture | 1 |
| Existing C05-20 embedding/status/eager regressions | 3 |
| Actual SmolLM2 + canonical continuation | 2 |
| CUDA CPU library / graph contracts | 83 / 32 |
| Runtime CPU / architecture / inventory | 258 / 15 / 1 |

The new fixture exercises a seven-row H64 table and three tokens within a larger
metadata parent. Success, invalid tokens [1,9,10], then successful recapture are
verified, with 32 replays each, explicit close and Drop. Invalid-token status
identifies position 1/token 9 and preserves the entire output. Table/token/tail
bytes and allocation counts remain unchanged. Short report rejection is
recoverable, and all allocations close to zero.

Both model tests run one-token prefill plus three decode steps with independent
baseline/candidate executors. The prior norm/RoPE/KV/pointwise audits also run.
Embedding output, full table/token preservation, restored scratch, full logits,
initialized KV and continuation agree. SmolLM2 remains [808,2775,288,536]; the
canonical H64/L2 fixture remains [2,2,2,2]. Resource counts return to zero.

Format/diff checks passed. Normal Clippy exited 0 with no diagnostics in the new
embedding modules; baseline repository warnings remain. Model and native tests
completed normally without failed-test waivers.

## Reproduce

Apply source-overlay.tar.gz over base_revision in manifest.json. Source hashes
were matched against the remote tree. Binary/checkpoint/GPU identities and raw
logs accompany this receipt. The model loader validates payloads on every model
load. Six unrelated local model-loader edits remain byte-for-byte unchanged.

```sh
cargo test -p riley-cuda --features cuda --test graph_embedding_borrowed_gpu --test graph_c05_20_gpu -- --include-ignored --nocapture --test-threads=1
cargo test -p riley-runtime --features cuda --lib c07_layer_norm_model -- --ignored --nocapture --test-threads=1
```

Remote ssh ai-assistant; scratch /tmp/riley-g01-native-260910; target
/tmp/riley-g01-native-260910-target. CUDA toolkit /data/cuda-12.8.1/bin;
CMAKE=/data/cmake-3.31.12/bin/cmake. Set RILEY_REAL_CHECKPOINT and
RILEY_CANONICAL_CHECKPOINT from runtime-identities.json.

Remaining: GEMM/head/output capabilities, selected metadata/H2D chain, retained
aggregate binding, full decode graph and runtime selection. SmolLM2-first matched
vLLM performance comparison follows correctness and qualification. No deployment,
default promotion or performance improvement is claimed here.
