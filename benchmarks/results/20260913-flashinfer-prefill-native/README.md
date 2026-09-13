# Native FlashInfer paged-prefill batch — 2026-09-13

Status: **native adapter and graph functional checks complete; model/runtime/serving integration pending.** No serving speedup or model-quality acceptance is claimed. Existing decode-only FlashInfer quality failures remain unchanged.

## Implemented

`flashinfer_prefill.cu` instantiates the pinned FlashInfer0.6.16.post3 `BatchPrefillWithPagedKVCacheDispatched` kernel directly, with BF16 Q/K/V/O, Q9/KV3/head64, page16 HND, causal masking and unsplit KV. It is library use through native CUDA/C++; it is not a rewritten attention algorithm. The small adapter handles metadata and raw ABI boundaries.

A bounded GPU planner consumes V7 packed-request shapes and page lists. It validates contiguous query offsets, query/KV lengths, physical page bounds and tile capacity before publishing any valid work slots. The workspace is33,996 bytes. Q/K/V are not repacked or copied. Query/head packing uses3 query heads per KV head,128 packed Q positions per CTA and64 fixed graph work slots. At32 requests and1,024 total query rows, the maximum is55 work tiles. Graph replay updates metadata at fixed addresses. The current primitive processes every supplied request, including Q-length1; selective prefill routing for a mixed model is not connected yet.

Raw-pointer requirements still belong to the future Rust/native recorder: same context/stream, retained allocation parents, full extents/alignment and non-aliasing output/status/workspace. No serving registration or default selection was added.

## Functional evidence

The native C++ probe uses deterministic finite BF16 inputs and an independent host FP64 dot/softmax/value calculation. The synthetic absolute-error criterion0.01 was written before execution; it is not a full-model numerical gate.

| Query lengths | KV lengths | Compared BF16 output values | Max absolute error vs FP64 |
| --- | --- | ---: | ---: |
| 1 /17 /33 | 16 /129 /511 | 29,376 | 0.000324839 |
| 16 /32 | 4096 /128 | 27,648 | 0.000165180 |
| 127 /128 /129 | 128 /256 /398 | 221,184 | 0.000548303 |
| 31 requests of1, one of993 | 31 of16, one of1024 | 589,824 | 0.000510752 |

These cases cover noncontiguous pages, causal prefix/continuation boundaries, growing/shrinking replay shapes, context4096, all32 request slots and the55-tile maximum. Inactive output rows retain sentinels. A malformed page in the last request sets status4 and leaves **every** output untouched, including the otherwise-valid prefix. A subsequent valid maximum-sized replay succeeds.

The expanded probe compares868,032 values. Original and synchronized-overlay outputs are bitwise equal across all four valid cases (1,736,064 bytes per output dump). Their hashes, probe/source hashes and resolved overlay library path are retained in `patched/verification.json`.

## Racecheck finding and bounded fix

The original-header build passed functional checks and memcheck, but racecheck reported24 warning groups on the initial probe and **32 warning groups on the expanded probe**. These are not passes. The tool's process exit success with warnings does not clear the gate. Both original logs are retained.

A line-info build located the shared writes at `attention/prefill.cuh:1983/1984/1986/1988` and subsequent reads in `permuted_smem.cuh:185`. The `write_o_reg_gmem` output path writes lane-owned words and then performs cross-lane128-bit shared reads without an explicit intervening warp barrier. The bounded patch adds `__syncwarp()` between those stages. This supports the diagnosis for the tested specialization; it is not an assertion that every FlashInfer backend is affected.

`prepare_flashinfer_prefill_overlay.py` first verifies the complete original header-tree lock, copies headers into a new build directory and applies exactly one known-context insertion. It preserves the installed FlashInfer headers and original lock. `patched/header.patch` shows the exact delta; `patched/overlay.json` records before/after hashes. The patch is a disclosed local dependency modification, not an upstream release update. No message or issue was sent upstream.

With the same expanded probe and the patched native library:

- Racecheck: **0 hazards,0 errors,0 warnings**.
- Memcheck: **0 errors**.
- All functional checks pass and every saved BF16 output is bitwise unchanged.
- Original installed header tree still verifies as `2a7d8ab8f81f6cb7fd259270b63bd71e152bff77a105b3c0ba875108b5567e0c`.

Patched AOT build passes for SM89/SM90a/SM100a with nvcc13.0.88. Runtime checks above are RTX4090/SM89 only; Hopper/Blackwell runtime is **SKIP: hardware unavailable**. No performance measurement is inferred from successful compilation or sanitizer results.

## Reproduction and integration requirements

Create a fresh verified overlay using `prepare_flashinfer_prefill_overlay.py ORIGINAL_FLASHINFER_DATA NEW_OVERLAY`. Build `flashinfer_prefill.cu` with `build_flashinfer_adapter.py --flashinfer-data NEW_OVERLAY --lineinfo` and the desired explicit AOT architectures. The exact compiler/include commands are in build JSON. Link `flashinfer_prefill_probe.cu` against `adapter.so`; use the overlay directory first in LD_LIBRARY_PATH to select the patched library. The linkage receipt confirms that selection. Python here is build/offline orchestration only; the tested process is native C++/CUDA.

Raw binaries/dumps remain at `/tmp/riley-opt-260912/flashinfer-prefill-v1` and `flashinfer-prefill-sync-v1` on `ai-assistant`. Original and patched logs/manifests are archived here. Final verification reports no GPU compute processes. Blender and system drivers were not changed.

Before model integration, require the verified synchronization overlay for the prefill object; do not silently compile it against the unpatched include path. Keep the base-header lock and patch provenance distinct. Then (1) route only prefill rows to this kernel while preserving the selected decode arithmetic across pure/mixed stages, (2) bind both workspace lifetimes and graph identity in the Rust/native owner, (3) run existing full-model/free-generation/natural-quality gates, and (4) measure real serving against frozen Riley and fresh vLLM. Running tensor-core prefill on mixed decode rows accidentally would recreate the earlier stage-consistency problem. This adapter is not yet a POD resource-co-scheduling implementation.
