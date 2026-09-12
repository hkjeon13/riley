# G03 SwiGLU operator chain — local implementation, GPU validation pending

## Implemented

The retained resource owner now records a six-node staged graph: gate H2D,
up H2D, SiLU, gated multiply, activated D2H, product D2H. The kernel nodes reuse
the exact BF16 eager kernels from `primitives.cu`, preserving the intermediate
BF16 rounding boundary. Multiplication depends on both the SiLU and up-copy
nodes. Output copies depend on their respective producers.

Four distinct registered device parents must be equal-sized, nonempty BF16
arrays in the owner context. A single registered pinned parent holds disjoint
[gate, up, activated, product] regions; any tail remains untouched. The native
owner retains each parent once, including the shared SiLU/multiply intermediate.
Existing fresh-source staging, completion-gated reads and fail-closed graph/lease
cleanup are reused. This does not use or relax stream-capture guards.

The runtime diagnostic uses actual gate/up/activated/product allocations and
existing I/O staging. It computes a reference with the two eager primitives,
alternates zero and actual inputs over 32 retained-graph replays, compares both
outputs exactly, then restores input, scratch outputs and pinned contents.
This inspects the final live scratch snapshot after a completed iteration;
it is not an in-flight all-layer activation trace or a whole MLP graph.

## Tests prepared, not executed on GPU

- Safe-wrapper fixture: sizes 1, 257 and 1536 BF16 elements, separate eager
  reference stream/buffers, two graph owners per size, 32 changed payloads each
  (192 successful replays), invalid indices/aliases, stale-result rejection,
  pinned-tail preservation, explicit close/Drop and allocation cleanup.
- Native boundary fixture: absent device parent, odd-sized BF16 parent,
  undersized pinned parent, valid zero replay and cleanup.
- Actual-model C07 cases now invoke the chain audit before existing operator
  checks and logits/KV/continuation comparisons. Five model cases would perform
  640 replays across 20 calls. These counts describe intended tests, not results.

## Verification completed locally

CPU tests: CUDA library 84 + graph contracts 32 + runtime library 259 +
architecture 15 + inventory 1 = **391 passed**. C11 public ABI syntax check,
format/diff checks and normal CPU Clippy pass. Inherited warnings remain;
no diagnostic names the new graph resource module after the documentation fix.
These checks do not compile CUDA-enabled Rust branches or native CUDA code.
Six unrelated modified model-loader files retain their original hashes.

## Exact pending approval

Automatic approval review rejected source transfer because the new payload
contains nine files, beyond the previously approved five-file payload. No remote
file write, native CUDA build or GPU execution occurred for this change.
There was no workaround. Pending destination is the same temporary test tree:
`ai-assistant:/tmp/riley-g01-native-260910`.

| File | Change |
|---|---|
| `kernels/include/riley_cuda.h` | Add SwiGLU graph recording ABI |
| `kernels/src/graph_resources.cu` | Record six nodes and offset completed output |
| `kernels/tests/abi_layout.c` | Pin new function signature |
| `kernels/src/ffi_internal.hpp` | Declare internal eager-kernel graph builder |
| `kernels/src/primitives.cu` | Add graph nodes using existing eager kernels |
| `crates/riley-cuda/src/ffi.rs` | Private FFI and native boundary fixture |
| `crates/riley-cuda/src/graph_resources.rs` | Safe recording API and eager parity fixture |
| `crates/riley-runtime/src/llama/graph_decode_pointwise_audit.rs` | Actual-model chain audit and restoration |
| `crates/riley-runtime/src/llama/graph_decode_final_norm_model_gpu.rs` | Invoke the audit in model tests |

`pending-transfer.json` contains exact previous/current hashes. The archive
contains only these source files; it excludes credentials, environment files,
checkpoints and unrelated model-loader changes. Approval is requested to transfer
these files and run the scoped CUDA build, SwiGLU and aggregate lifecycle tests,
then the existing actual-model C07 cases with the new audit.

This is incomplete until native compilation and GPU parity pass. Full model
recording still needs embedding, normalization, selected GEMMs, RoPE, KV write,
attention, residuals, final norm/head and output/status handling in one retained
DAG. G02H/G03, bucket qualification and vLLM comparison remain incomplete.
No commit, push or deployment occurred.
