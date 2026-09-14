# FFN packed-row census — C32 diagnostic

FFN dispatch uses the **total packed token rows** (`plan.total_tokens()` → metadata `shape[2]`), including decode rows in mixed iterations. It does not use an individual request’s prefill length. The existing model already packs request tokens before FFN. V1 recorded only per-request fragmentation and cannot establish FFN branch frequency; its evidence and receipt are preserved under `request-only-v1/`.

V2 adds an exact packed-row histogram under `RILEY_SERVING_PHASE_TIMING=1`. Histograms are bounded to 1–1024 with explicit overflow counts. Successful ordinary prefill/mixed steps are recorded after commit/publication. No diagnostic map work is performed when phase timing is disabled. Existing graph, kernel, scheduler policy and default backend selection are unchanged.

## Verified observations

RTX 4090, SmolLM2-135M BF16, C32/active32, chunk512, prefix512, 720 MiB KV payload. Each lane includes 64 warmup + 256 retained requests. These are short diagnostic runs including warmup, not a new serving performance comparison.

| Lane | Prefill/mixed steps | Packed rows ≥192 | Fraction |
|---|---:|---:|---:|
| control-shared | 96 | 19 | 19.79% |
| adaptive-shared | 111 | 38 | 34.23% |
| adaptive-unique | 397 | 390 | 98.24% |
| control-unique | 397 | 391 | 98.49% |

For adaptive lanes, ≥192 selects M32; control always uses the existing M16 backend, so its counts show eligibility only. Counts are iterations, not kernel-time shares. Every measured prefill/mixed iteration executes the model layers; no GPU time is inferred from frequency.

Unique almost always reaches the large-row branch, consistent with the separately measured unique serving improvement. Shared frequently uses the small-row branch, where prior native evidence found unified dispatch resource overhead. This is a plausible explanation, not causal proof: shared runs have different batch counts/distributions and only one order was sampled. The measurements do not establish that threshold tuning or resource separation will improve serving.

## Validation and provenance

`python3 benchmarks/analysis/export_ffn_row_census.py benchmarks/results/20260914-ffn-row-census`

The exporter verifies all 24 regular JSON/log archive members, source/controller/client/fixture/binary identities, four clean exits, restored Blender receipt, GC policy and 1,280 full SSE responses against frozen reference token sequences. Per-request counts reconcile to request occurrences in the existing batch histogram. Packed counts independently reconcile exactly to its prefill/mixed counts in every 16-row bucket. All overflow counters are zero. The final local receipt binds the compressed archive SHA256.

CUDA release build exited 0; measured binary SHA256 is `716efc2fa85e75dfde265da5dc89ff009a778e16d3892f55b947d6288005ff40`. The archive carries measured source hashes; the exporter deliberately rejects a different current engine source. Blender scene RPC on 9876/9911/9887 subsequently returned success, and local viewer HTTP endpoints 31840/31970/32010 returned 200. This is not a fresh public-browser visual check.

The initial local archive read occurred before SCP completed and failed with EOF. After the same transfer completed successfully, the full verifier passed; no benchmark was restarted and no partial archive was accepted.

## Next optimization batch

Evaluate graph/backend resource separation as a single bounded batch: (1) retain original M16 kernels for small packed batches, (2) isolate M32 gate/up and down resource allocation for large batches, and (3) share weights/scratch while selecting a compatible graph from host-known total rows without a new GPU synchronization. First inspect graph capture identity and ownership so duplicated graphs do not silently double persistent buffers or break rolling decode. Fixed arithmetic and the 192 boundary remain unchanged.

Require graph replay across 191/192/193 and small/large alternation, actual-model logits and serving generation parity, bounded native sanitizer, then the same frozen-prior/control/candidate/vLLM C32 comparison in both orders. Account for graph memory, initialization and steady-state overhead. If graph selection cannot preserve contracts or the serving result fails, retain the current default and move to the planned attention work instead of sweeping thresholds. Hopper/Blackwell and multi-GPU runtime remain unverified; unavailable hardware tests are explicit skips.

The latest qualified scope of evidence remains the [full C32 comparison](../20260914-ffn-adaptive-serving-c32/README.md): candidate versus vLLM throughput −8.79% shared / −14.91% unique. The overall goal is not achieved and no default promotion is made.
