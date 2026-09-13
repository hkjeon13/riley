# Query reuse — retained model integration

The compact 16-query candidate now has a native retained V7 recorder, Rust ownership path, catalog identity and explicit server selection. Full-model teacher-forced BF16 logits match the existing baseline in all tested cases. Serving performance is evaluated separately; no default is changed.

## Implementation

- `record_v7_query_reuse` is an additive C ABI and Rust wrapper over the existing validated parent/resource recorder. It requires prefill-FFN and adaptive pure decode. Ordinary/paired decode retain existing adaptive kernels; the mixed/prefill graph uses query reuse in all 30 layers.
- `into_owned_variable_query_reuse_session` retains model buffers, compact/full completion, optional buffered completion and shared-prefix ownership. Query reuse adds no persistent allocation; native shared scratch remains kernel-local.
- The catalog hashes the query-reuse profile marker and both candidate/compact source bodies. Shared-prefix descriptors derive their identity from that catalog. Build dependency tracking includes the added headers.
- `RILEY_MIXED_QUERY_REUSE=1` explicitly selects the candidate; unset/0 selects existing execution. Other values fail. It requires V7 plus `--ffn-backend prefill-pipeline-experimental-v1 --decode-projection adaptive-rows-experimental-v1`. Paired decode and prefix caching can be combined. FlashInfer/FA3/full-decode-FFN combinations reject. Startup emits `RILEY_QUERY_REUSE prepared=true` only after successful graph recording.
- Runtime remains Rust → C ABI → CUDA. Python is offline measurement tooling only.

## Executed checks

| Check | Evidence |
|---|---|
| Local runtime lib | 349 passed, 1 existing timing diagnostic ignored |
| Full local scheduler suites | 155 passed |
| CUDA scheduler lib | 55 passed |
| CUDA full-model tests | 2 executed, 2 passed, 0 ignored |
| New query-reuse logits | 2,359,296 BF16 bytes exact |
| Existing automatic-cache regression | 1,474,560 BF16 bytes exact |
| CUDA/server feature check and release build | Passed |
| CUDA allocation cleanup | Every model run asserts zero remaining allocations |

New model fixtures include 398-token prompts with different suffixes, a 47-token ragged tile, 512-token prefill and four repeated 128-token prefixes. Three teacher-forced outputs per request isolate execution/KV differences from sampling. Both candidate cache-off and cache-on paths execute. The cache comparison observes three hits/336 reused tokens in both baseline and candidate. This does not replace broader-model quality or buffered serving tests.

The previous release is preserved as `/data/riley-serving-260913-recovery/riley-before-query-reuse-v1`, SHA256 `f1ccef367d9329a13e8c19b499ec22bf5817a5f82692fbaf38c1c3b6021688ea`. New release SHA256: `1fc55a483be57e8822bd45d24f2dbfa606cee2569c031272582a907aa223646e`. The remote tree is a selected-file source mirror; `evidence/changed-source-sha256.txt` verifies every changed implementation/test file against the local source. Build and hardware execution evidence is not a clean remote git-checkout claim.

[receipt.json](receipt.json) hashes logs and records the executed checks. [Serving evaluation](../20260914-query-reuse-serving/README.md) contains free-generation, stop/cancel/recovery and the matched vLLM comparison. Hopper/Blackwell runtime, multi-GPU, broader models and sustained-load qualification remain incomplete. Rollback unsets `RILEY_MIXED_QUERY_REUSE` or uses the preserved release.
