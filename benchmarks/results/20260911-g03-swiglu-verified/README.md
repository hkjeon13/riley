# G03 SwiGLU chain — authorized remote validation

The user explicitly approved the nine-file source transfer and CUDA/model tests
from the preceding local implementation report. Only those nine files were sent
to `ai-assistant:/tmp/riley-g01-native-260910`. Remote files matched the expected
previous hashes before transfer, then matched all approved local hashes afterward.
No additional source edits were needed during remote validation.

## Results

CUDA native, C11 ABI checks and CUDA-enabled Rust builds succeeded. **19 GPU
tests passed**, with all three test commands returning **exit code 0**:

- **2 SwiGLU tests:** one safe-wrapper fixture covers BF16 element counts 1,
  257 and 1536, two graph owners per size and 32 changed input pairs per owner
  (**192 successful replays**). Both activated and product bytes exactly match
  separate eager primitive execution. Invalid indices/aliases, short payloads,
  stale-result reads, pinned-tail preservation, explicit close/Drop and stable
  allocation counts pass. The native boundary fixture additionally checks an
  absent parent, odd BF16 size, undersized pinned buffer and valid zero replay.
- **3 regressions:** retained transfer graph and both resource-ledger lifecycle
  tests pass with the updated shared owner.
- **14 C07 tests:** five actual-model cases invoke the new chain audit over four
  iterations each, with 32 alternating zero/original replays per audit call
  (**640 model replays**). SmolLM2 L30 and canonical L2/L3 full-logit and initialized
  KV hashes exactly match the preceding resource-ledger run. Continuation remains
  SmolLM2 `[808,2775,288,536]`, L2 `[2,2,2,2]`, L3 `[5,5,5,0]`.
  Scratch/pinned restoration, allocation stability, prior operator audits and
  final cleanup pass. `model-parity.json` records the exact hash comparison.

The graph records input copies, the existing eager BF16 SiLU kernel, the existing
eager BF16 multiply kernel and two completed-result copies. The shared
intermediate remains one ledger-held allocation. The model audit uses actual
retained scratch and I/O staging; it does not allocate substitute CUDA parents.

The prior 391 CPU tests, ABI syntax, format/diff and CPU Clippy results are in
`../20260911-g03-swiglu-chain`; they were not rerun because source was unchanged.
Existing compiler warnings remain. Binary/checkpoint/GPU identities and the
approved source hashes are retained here. Six unrelated model-loader files remain
unchanged. No commit, push or deployment occurred.

## Scope still pending

This establishes a **SiLU → gated multiply subgraph**, tested on a completed
iteration's final live scratch snapshot. It does not prove an in-flight activation
trace for every layer, a full MLP projection graph, or full decode replay.
Embedding, norm/GEMMs, RoPE, KV/attention, residuals, final norm/head and output
status still need to share one retained model DAG with the appropriate input
freshness, completion and error-publication contract.

Full G02H/G03 qualification, bucket selection and matched SmolLM2 vLLM benchmarking
remain incomplete. No speedup claim is made. The previous nine-file transfer
approval block is resolved for this implementation and validation.
