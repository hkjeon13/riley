# Prefill-only routing prerequisite

This implementation step adds `riley_flashinfer_prefill_only_prepare` to the experimental native adapter. It is a prerequisite for model integration, not a completed optimization batch or a serving performance result.

The V7 wire encoder defines each request's stage at byte 72 (`shape[18]`): 0 is prefill and 1 is decode. Query length alone is not sufficient: a one-token prefill must still use prefill arithmetic. The new planner retains every original Q/KV indptr and packed output offset but publishes work tiles only for stage 0. There is no Q/K/V copy or decode output scatter. Workspace remains 33,996 bytes. It validates the entire packet, including skipped decode requests, before publishing any work; invalid stage or decode query length other than one yields shape error bit 2.

The original all-query entry remains available for its existing primitive experiment. It is not the entry to use for mixed model integration.

## Evidence

Same RTX 4090 / nvcc 13.0.88 / pinned FlashInfer and verified warp-barrier overlay as the previous build. Entire native archive and archive-linked probe rebuild successfully. Native C++/CUDA graph execution only; no runtime Python.

The `--prefill-only` probe exercises ten replays of the same graph and workspace, including ragged shapes, 32 requests/1,024 packed queries, noncontiguous pages, all-decode no-work, a one-token prefill between decode requests, invalid suffix page, invalid stage, invalid multi-query decode, and recovery after errors.

- Memcheck: 0 errors.
- Racecheck: 0 hazards, 0 errors, 0 warnings.
- 849,600 prefill BF16 values are bitwise identical to the full adapter at identical packed offsets; 18,432 decode values in those matched cases remain sentinel-filled. Additional single-token prefill cases pass the unchanged FP64 synthetic oracle tolerance of 0.01 (observed maximum absolute error 0.000123301).
- All-query entry regression: all 868,032 outputs remain bitwise identical to the previously verified patched adapter.
- Racecheck and memcheck output dumps match bitwise. `verification.json` records source/output hashes and counts; `verify.py` performs the comparison. Remote output dumps remain under `/tmp/riley-opt-260912/prefill-routing-v1`.

Only SM89 runtime/build was rerun for this change. No new Hopper/Blackwell runtime claim is made. This is not full-model quality evidence.

## Next integration

Connect retained Rust/native recorder ownership and a separate experimental graph identity. Mixed execution must pair this planner/run with the selected decode backend, suppressing the old prefill computation while preserving decode arithmetic in both pure and mixed stages. Do not treat the earlier function named `enqueue_compiled_v7_flashinfer_prefill_model` as this integration: that function currently uses FlashInfer for mixed **decode** rows and the old prefill kernel.

Whole-model stage consistency, independent free generation and natural-language quality gates must precede the next matched serving comparison. The existing failed FlashInfer decode quality gate remains failed; these primitive tests do not override it.
