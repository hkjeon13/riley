# PR10 local CUDA COW transfer gate

Status: **local D2D/event correctness and lifetime passed; model/serving integration incomplete.** No throughput, TTFT or TPOT improvement is claimed by this gate.

## Implemented path

`CopyOnWrite::submit_cuda` moves the host transaction into `CudaPendingCow`, which owns a `CudaPendingKvCopy`. The path is Rust → C ABI → CUDA `cudaMemcpyAsync` with DeviceToDevice direction. One native event covers all initialized K/V head slices of a page. No Python runtime participates; the Python exporter only reads completed offline logs and SQLite.

The native token holds both device buffers, the stream, context child and capture-lifecycle admission. A partial submission keeps those holds and reports the error through completion. A successful event query/synchronize plus successful context restoration is required before the host transaction can publish or discard. Copy failures are sticky across retries. Cancellation requests discard but does not release in-flight pages. Unknown completion retains both source and staging ownership.

This adapter accepts **idle local K/V buffers**. It does not bypass captured graph ownership. Captured serving pools are retained by `RileyCudaGraphResources`; an owner-authorized transfer seam, shared read-only batch validation, scheduler cache policy and full-model identity binding still need integration.

## Executed evidence

RTX 4090, CUDA 13.0.88 isolated toolchain, GUI retained, Blender down. The test-only `cuda-test-fault-injection` feature was explicitly enabled. It must not be used for serving binaries.

| Gate | Result |
|---|---|
| COW normal publication | Lengths 1, 15, 17, 31, 33; exact full K/V buffers and unused-tail guards |
| Cancel/reset and orphan drain | Both passed; new reservation preserved and pages reclaimed only after completion |
| Partial native submission | First K slice copied, remaining slices untouched; no publication; retry preserves failure |
| Ambiguous completion/restoration | Isolated child retained 2 host pages and all native copy-use holds; cleanup by process exit |
| GPU tests | 4 passed, 0 failed, 0 ignored; ambiguity child also executed successfully |
| Memcheck | 0 errors, 0 leaked bytes in the 8 normal/cancel/orphan/partial cases |
| Existing H2D/D2H regression | 7 passed, 0 failed, 0 ignored |
| Local CPU validation | CUDA memory contracts 4 passed; paged-KV contracts 31 passed |

Each of the 8 non-ambiguity cases verifies both whole 12,288-byte buffers, totaling **196,608 bytes**. Source and destination addresses are checked against an independent dense-layout oracle in the host tests and GPU readback expectations. Every such case finishes with host pages and CUDA device/pinned allocations at zero, and context close succeeds. The deliberate ambiguity case is excluded only from the zero-leak sanitizer run, not from GPU execution; retaining resources is its expected correctness result.

Nsight SQLite independently records **85 D2D operations / 6,736 bytes**, 8 event creates, 8 records and 8 destroys, with 5 event queries and 3 event synchronizations. H2D/D2H are test setup/readback, 16 operations / 196,608 bytes each. These counts validate the actual CUDA path; they are not serving performance measurements.

Final test binary SHA256: `bc5d382abf83cb245ec56847feecd05cbf53130a7ed8ce84768701085968da0a`.

[Receipt](receipt.json) checks all expected cases, native event/copy counts, evidence hashes and 14 source hashes against the local checkout. [Evidence](evidence/) includes final build/test logs, Memcheck, curated Nsight SQLite, existing memory regressions and final empty compute-process inventory. `initial-gate.log` preserves the earlier successful 3-test run before adding the ambiguity case; final evidence is the 4-test v4 run.

Nsight originals can contain environment credentials and are **not published**. The runner profiles with a minimal environment, retains original reports in a private remote directory outside the export directory, and constructs a fresh SQLite containing only numeric CUDA activity and public CUDA API names. Environment/process metadata and unused SQLite pages are not copied. The first sanitization matcher rejected public `cuInit`/`cuGetProcAddress_v2` driver names; it was corrected before the final complete gate. This export change does not alter the CUDA implementation or its test cases.

Remote production binaries were not rebuilt or replaced by this test work: `target/release/riley` remains `d85368b3bc3af73593f28cac2ea7bcb01f09c45f2888e6a5f4fb6be5c9101bab`, and `riley-before-adaptive` remains `3abb1af2281811909ca58ed306b163f676bd7fa0f57b9f4bfb32ec7b29ce72ff`. Their prior serving results remain the latest available comparison.

## Reproduction and remaining qualification

Run `bash benchmarks/analysis/kv_prefix_gpu_gate.sh /data/riley-serving-260913-recovery <new-absolute-evidence-directory>` on the prepared remote host. Then use `benchmarks/analysis/export_kv_prefix_evidence.py <repo> <evidence> <new-receipt.json>` against the matching sources. The runner uses bounded jobs/timeouts and does not restart Blender or alter the serving binary.

Remaining: captured-model owner integration, shared-prefix batch/native validation, scheduler admission/cache lookup/eviction, model token/logit correctness, and matched cache-hit/cache-miss serving comparison against vLLM. Multi-GPU/peer and Hopper/Blackwell runtime validation are not established by this single-GPU test. PR10 is not complete or promoted.
