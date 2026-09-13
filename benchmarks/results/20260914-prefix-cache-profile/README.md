# Composed serving profile — attention resource batch selected

The corrected trace identifies prefill/mixed execution as the main device cost on unique prompts: 83.8% of selected graph spans. The mapped attention kernel alone accounts for 31.1% of summed kernel time. Prioritize a substantial mixed-attention resource/work-reuse batch, not another scheduler threshold or small projection variant.

## Scope and verification

Same frozen release binary and shared/unique fixtures as the [matched serving comparison](../20260914-prefix-cache-composition/README.md): RTX 4090, SmolLM2-135M BF16, C32/active32, prefill-FFN + paired-decode + adaptive projection + 512-page prefix cache. Rust → C ABI → CUDA is unchanged. This is an Nsight diagnostic, not a new throughput comparison.

Each workload executes 64 warmups plus 64 checked requests. All 256 responses match the frozen reference in prompt/output tokens, text and finish reason. Both profiled processes exit 0, leave no owned processes, and peak at approximately 1.5 GB profiler-tree RSS, below the 8 GiB limit. The final GPU compute inventory is empty. Blender stays down and GUI remains active.

The first trace wrote warmup responses to disk between phases, creating roughly 60 ms idle gaps. The corrected controller defers that write until requests finish. The remaining phase transition, thread-pool setup, client pacing, validation and profiler overhead are still included. The analyzer selects the middle 80% of graph count, including parts of warmup; it does not claim a clean steady-state retained-only window. Do not classify every inter-graph gap as engine CPU overhead or potential serving speedup.

## Corrected measurements

| Diagnostic | Shared prefix | Unique prompts |
|---|---:|---:|
| Selected graph launches | 120 | 215 |
| Sum of graph spans | 284.14 ms | 860.63 ms |
| Prefill/mixed graph spans | 138.95 ms (48.9%) | 720.95 ms (83.8%) |
| Mapped attention kernel time | 54.75 ms (20.4% of kernel sum) | 257.95 ms (31.1%) |
| Inter-graph gaps, including external overhead | 148.59 ms | 169.16 ms |
| Within-pair gap median | 5.26 µs | 5.25 µs |
| After-pair gap median | 1,029.05 µs | 872.11 µs |

The near-zero post-GPU-to-CPU-launch time within pairs confirms successful pre-submission of future decode. After-pair latency remains a separate diagnostic target. For unique prompts, prefill FFN gate/up and down contribute 114.83 ms and 84.92 ms, respectively; mapped attention is larger than either. Summed kernel durations and graph-span union are different denominators and are not interchanged.

## Next implementation batch

The current `mapped_attention` maps 8 queries per warp (single-query fallback below 32 rows), retains ordered BF16 probability rounding and PV recurrence, and uses an existing 16-query body only as an unselected template capability. It also repeats K/V work for each query-head CTA. These are code observations; the trace does not measure HBM bandwidth, occupancy or register stalls.

Next group phase-specific query work mapping, K/V data reuse and resource-bounded native dispatch into one candidate batch. First compare the larger query tile/resource envelope and staged K/V reuse against the existing exact mapped oracle on the observed long-prompt shapes, including prefix offsets and mixed decode rows. Inspect compiled register/shared-memory usage and bounded device timings. Preserve the single-query/nonfinite/causal/page semantics, arithmetic order, cache ownership, future-token catalog identity and fallback. Do not silently substitute a changed numerical mode. If exact equivalence cannot be retained, keep the alternate numerical candidate explicitly separate and require model-quality evaluation before serving promotion.

The previous POD and compact-POD native candidates regressed; do not merely repeat their CTA-role constants. Connect a new candidate to retained ordinary/mixed graph recording only after its correctness/resource gate, then compare standard and composed serving against the same-day vLLM. PR04 and PR06 remain incomplete; this profile is not evidence that attention split, POD overlap, multi-GPU or hardware qualification has been delivered.

## Reproducible, curated evidence

[receipt.json](receipt.json) records checked requests, source/curated SQLite hashes and rederived accounting. [evidence.tar.gz](evidence.tar.gz) contains freshly constructed numeric CUDA activity databases, public CUDA/kernel names, checked response records, launch hashes, logs and exit receipts. It excludes original Nsight reports, original SQLite pages, environment tables and process metadata. Originals remain in a private remote directory.

Run `python3 benchmarks/analysis/export_prefix_cache_profile.py benchmarks/results/20260914-prefix-cache-profile` to validate both archives and recompute graph/kernel accounting. The sanitizer initially rejected the two public anonymous-namespace RoPE table kernels; those exact prefixes were admitted. An offline Python 3.10 incompatibility in the hash helper was corrected with streaming SHA256. Neither correction changes GPU measurements. Final sanitized databases and the local rederivation pass.
