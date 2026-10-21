# Native-order SM80 decode Gate/Up and Down v8

Two-file layer on the immutable v7 source snapshot, not clean HEAD. No default mode changes: `native-prefill-sm80-v5` remains the opt-in selector; startup identifies both v8 routes. Existing v7 prefill and v6 paged Attention remain intact.

- Down actual R5..32: four warps,16-row/64-column tile, shared leading dimension72, four asynchronous K64 stages. Preserve split-K2 with two K32-interleaved FP32 accumulators per split, FP32 sliced sum→BF16 partial→FP32 split sum→BF16 output. Use existing private workspace and its managed use/replay ledger. R1..4 remain on native/Lt because the candidate is slower there.
- Separate Gate and Up actual R17..32: four warps,16-row/64-column tile, shared leading80, two K64 stages, two K32-interleaved FP32 accumulators, original FP32 epilogue and BF16 output. One launch per projection handles both row tiles. R1..16 retain native because the candidate is slower there.
- Cold mode10 preparation validates native SM80 kernel images and sets a private eligibility flag on Gate/Down plans. Existing SM80/108SM/CUDA12.4/cuBLAS120405, unmodified workspace/MPS, context/span/alias/alignment gates remain. No new device buffer or runtime algorithm search. Existing bounded MLP Graph mode9 retains its prior kernels; compatibility is checked separately.

Finite component proof: actual L0 weights/seed, R1..32×6 states×3 projections,576 comparisons,69,746,688 BF16 values, zero differences against native M1. Three alternating CUDA-event repeats compare against actual v7 dispatch; these are operator diagnostics, not server throughput.

Initial Down research with default-stream pageable H2D and a nonblocking compute stream was invalidated. Accepted qualification explicitly queues every upload on the compute stream. Invalid logs and ordered reruns are retained, and accuracy tolerances were not changed.

Remote root: `/mnt/efs/nlp/psyche/riley-decode-v8-20261020` on `aws-a100`. GPU6/7 original holder PIDs/contexts are never stopped; each restores20GiB after campaigns. Model/build/scratch/evidence stay on EFS; only the four previously approved private C02 fixtures use `/tmp` and are removed.

Server comparisons and final audit are in [the report](../../../docs/qwen-decode-mlp-v8-and-vllm-20261020.md).

Qualified server binary SHA256: `5b43383a92d067f492d77114f0cbcfcbff2a3f87ea767e0dcf6a3d01d83e2bcb`. Fresh three-repeat throughput medians: C16 522.831→566.788 tok/s (+8.41%); C32 643.552→718.490 (+11.64%). C16 TTFT p95 109.152→109.186ms is not an improvement; C32 359.498→345.379ms. Stock medians1121.457/1676.630 tok/s retain a substantial gap. CPU521 passed/40 ignored; exact full-vocabulary and raw SSE audit passed. Evidence: [independent audit](../../evidence/20261020-decode-v8/independent-audit.json), [source/binary binding](../../evidence/20261020-decode-v8/source-binary-binding.json).
