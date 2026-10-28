#pragma once

// Batch-invariant D64/D128 attention through a pinned FlashInfer paged-prefill
// kernel (third_party/flashinfer). One launch serves prompt and decode rows:
// a GPU planner turns Riley's device row metadata into FlashInfer's request,
// page and work lists, then the kernel with a fixed 64-row query tile and no
// KV split folds each row's whole context and writes BF16 outputs. Every
// row's reduction order depends only on its own context, so the output does
// not depend on the batch. A separately selected numerical mode.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace riley_fi {

// Riley's packed batch as the planner reads it (device pointers).
struct RowBatch {
  const uint32_t* sequence_block_offsets;
  const uint32_t* block_ids;
  const uint16_t* valid_tokens;
  const uint32_t* row_sequence_slots;
  const uint32_t* row_positions;
  uint64_t sequence_count;
  uint64_t block_count;
  uint64_t active_row_count;
  uint64_t physical_block_count;
  // Non-null in a dynamic step capture: `[0]` is the step's active row count.
  const uint32_t* dynamic_counts;
};

// Bytes of planner workspace for `output_rows` rows, `page_capacity` pages
// per row and `group_heads` query heads per KV head.
uint64_t workspace_bytes(uint64_t output_rows, uint64_t page_capacity,
                         uint32_t group_heads);

// Plans (when `plan`; otherwise the workspace and `output` must already hold
// the plan and padding of the same batch) and runs the attention for every
// row below the active count. The plan writes rows from the active count to
// `output_rows` as zeros and the rows of a request whose pages do not
// resolve as NaN. `query` and
// `output` are `[rows, query_heads, head_size]` BF16; the pools are Riley's
// paged layout `[block, kv_head, 16, head_size]`. `head_size` is 64 or 128.
cudaError_t paged_attention(const __nv_bfloat16* query,
                            const __nv_bfloat16* key_pool,
                            const __nv_bfloat16* value_pool,
                            __nv_bfloat16* output, const RowBatch& batch,
                            uint64_t output_rows, uint64_t page_capacity,
                            uint32_t query_heads, uint32_t key_value_heads,
                            uint32_t head_size, float scale, void* workspace,
                            uint64_t workspace_size, bool plan,
                            cudaStream_t stream);

}  // namespace riley_fi
