# G02P5/P6/P7 borrowed pointwise audit — 2026-09-10

Standalone SiLU, gated multiply and both residual buffer mappings are connected
to existing native graph primitives and verified on GPU. This completes their
**cold resource audit**, not in-flight layer binding or full decode capture.

## Implementation

BorrowedPointwiseGraph retains exclusive actual input/output/stream owners through
native graph destruction. Binary operations additionally retain their second
input. Primitive selection is immutable, arity is exact, nonzero BF16 spans are
checked for overflow/capacity, and aliases/context/busy owners are rejected before
capture. Existing native primitives and kernel arithmetic are unchanged.

The executor derives sizes from its prepared workspace plan and audits:

| Site | Actual sources | Actual output |
|---|---|---|
| SiLU | gate_raw | gate_activated |
| Gated multiply | gate_activated, up_raw | gated_product |
| Attention residual | hidden_current, hidden_projection | hidden_rotary |
| MLP residual | hidden_rotary, hidden_current | hidden_projection |

Each transaction performs 16 graph replays and eager execution on the same
resources, compares complete output bytes, checks finite active output and
unchanged input/output tails, then restores and reads back the original output.
CUDA/transaction failure poisons the executor; errors retain the failing
operation site. Completed healthy M=1 execution and separate residual norm are
required. A fused residual configuration is rejected before mutation.

These buffers are reused between layers. The audit samples them between completed
iterations and does not claim their historical layer activations or validate the
future full-graph swap schedule. No diagnostic receipt promotes aggregate
inventory, hot dispatch or default configuration. Global inventory stays
7 primitive Supported / 7 Unknown, aggregate Unknown.

## Verification

**43 GPU tests passed**, all successful commands exited 0:

- New borrowed primitive matrix: 1 test covering all three operations, element
  counts 1/257/2048, explicit close and Drop, 32 replays per capture, complete
  input preservation, output tails, invalid sizes/arity and recovery. All
  allocations return to zero.
- Actual SmolLM2 + canonical executor tests: 2 tests, four sites after each of
  four iterations per model, 16 replays per site. Full logits, initialized KV,
  next-token continuation and allocation counters match independent executors.
  The previous norm/RoPE/KV audits also run. SmolLM2 continuation remains
  [808,2775,288,536]; canonical [2,2,2,2]. A healthy completed executor with fused
  residual configuration rejects the audit, then succeeds after restoring the
  separate configuration. Both executors close all resources to zero.
- Existing graph lifecycle suite: 40 tests, including original standalone
  activation/multiply/residual ownership and replay coverage.

CPU: CUDA library 82, graph contracts 32, runtime library 258, architecture 15,
inventory 1 passed. Format/diff checks passed. Final normal Clippy exited 0 with
no pointwise module diagnostics; baseline repository warnings remain. Initial
lint output is retained along with the corrected final output. Earlier focused
runs are not added to the final GPU count.

## Reproduce and limits

Use source-overlay.tar.gz over base_revision in manifest.json. Source SHA256s
were compared to remote scratch. Binary/checkpoint/GPU identities and logs are
included. Six unrelated local model-loader files remain byte-for-byte unchanged.
The loader validates checkpoint payloads at every model load.

```sh
cargo test -p riley-cuda --features cuda --test graph_pointwise_gpu -- --ignored --nocapture --test-threads=1
cargo test -p riley-runtime --features cuda --lib c07_layer_norm_model -- --ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_gpu -- --include-ignored --test-threads=1
```

Remote ssh ai-assistant; scratch /tmp/riley-g01-native-260910; target
/tmp/riley-g01-native-260910-target. PATH includes /data/cuda-12.8.1/bin and
CMAKE=/data/cmake-3.31.12/bin/cmake. Set RILEY_REAL_CHECKPOINT and
RILEY_CANONICAL_CHECKPOINT using runtime-identities.json.

Remaining work includes selected metadata/H2D ownership, Embedding, GEMM,
head/output capabilities, retained aggregate binding and full decode graph.
SmolLM2-first matched vLLM benchmarking remains pending after those correctness
and qualification gates. This receipt contains no measured speedup or deployment.
