#ifndef RUSTINFER_CUDA_H_
#define RUSTINFER_CUDA_H_

#include <stdint.h>

#define RUSTINFER_CUDA_ABI_VERSION 1
#define RUSTINFER_CUDA_ERROR_MESSAGE_CAPACITY 256
#define RUSTINFER_CUDA_DEVICE_NAME_CAPACITY 256

// ABI v1 is a 64-bit ABI. New entry points below are additive: existing v1
// callers remain link- and layout-compatible.
#define RUSTINFER_CUDA_ABI_POINTER_WIDTH 64u

typedef int32_t RustInferCudaStatus;

#define RUSTINFER_CUDA_STATUS_SUCCESS ((RustInferCudaStatus)0)
#define RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT ((RustInferCudaStatus)1)
#define RUSTINFER_CUDA_STATUS_INVALID_DEVICE ((RustInferCudaStatus)2)
#define RUSTINFER_CUDA_STATUS_OUT_OF_RANGE ((RustInferCudaStatus)3)
#define RUSTINFER_CUDA_STATUS_NOT_READY ((RustInferCudaStatus)4)
#define RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY ((RustInferCudaStatus)5)
#define RUSTINFER_CUDA_STATUS_DRIVER_ERROR ((RustInferCudaStatus)6)
#define RUSTINFER_CUDA_STATUS_RUNTIME_ERROR ((RustInferCudaStatus)7)
#define RUSTINFER_CUDA_STATUS_INVALID_STATE ((RustInferCudaStatus)8)
#define RUSTINFER_CUDA_STATUS_INTERNAL_ERROR ((RustInferCudaStatus)9)
#define RUSTINFER_CUDA_STATUS_CUBLASLT_ERROR ((RustInferCudaStatus)10)
#define RUSTINFER_CUDA_STATUS_NOT_SUPPORTED ((RustInferCudaStatus)11)

#define RUSTINFER_CUDA_ERROR_DOMAIN_NONE 0u
#define RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION 1u
#define RUSTINFER_CUDA_ERROR_DOMAIN_DRIVER 2u
#define RUSTINFER_CUDA_ERROR_DOMAIN_RUNTIME 3u
#define RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL 4u
#define RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT 5u

#define RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE 1u
#define RUSTINFER_CUDA_ERROR_STAGE_VALIDATION 2u
#define RUSTINFER_CUDA_ERROR_STAGE_CREATE 3u
#define RUSTINFER_CUDA_ERROR_STAGE_LAUNCH 4u
#define RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE 5u
#define RUSTINFER_CUDA_ERROR_STAGE_QUERY 6u
#define RUSTINFER_CUDA_ERROR_STAGE_RECORD 7u
#define RUSTINFER_CUDA_ERROR_STAGE_COPY 8u
#define RUSTINFER_CUDA_ERROR_STAGE_CLOSE 9u
#define RUSTINFER_CUDA_ERROR_STAGE_PREPARE 10u

typedef struct RustInferCudaErrorInfo {
  uint32_t struct_size;
  int32_t native_code;
  uint32_t domain;
  uint32_t stage;
  char message[RUSTINFER_CUDA_ERROR_MESSAGE_CAPACITY];
} RustInferCudaErrorInfo;

typedef struct RustInferCudaDeviceProperties {
  uint32_t struct_size;
  int32_t ordinal;
  uint64_t total_memory_bytes;
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  uint32_t multiprocessor_count;
  uint32_t warp_size;
  uint32_t max_threads_per_block;
  int32_t driver_version;
  int32_t runtime_version;
  uint32_t reserved[5];
  char name[RUSTINFER_CUDA_DEVICE_NAME_CAPACITY];
} RustInferCudaDeviceProperties;

typedef struct RustInferCudaAllocationStats {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t device_live_bytes;
  uint64_t device_live_allocations;
  uint64_t pinned_host_live_bytes;
  uint64_t pinned_host_live_allocations;
} RustInferCudaAllocationStats;

#if defined(RUSTINFER_CUDA_ENABLE_TEST_FAULT_INJECTION)
// Destructive test-only ABI. These declarations and their symbols are absent
// from ordinary archives. Enabling this definition in production is unsupported.
#define RUSTINFER_CUDA_TEST_MEMORY_FAULT_DEVICE_CREATE_ROLLBACK_AMBIGUOUS 1u
#define RUSTINFER_CUDA_TEST_MEMORY_FAULT_PINNED_CREATE_ROLLBACK_AMBIGUOUS 2u
#define RUSTINFER_CUDA_TEST_MEMORY_FAULT_DEVICE_CLOSE_AMBIGUOUS 3u
#define RUSTINFER_CUDA_TEST_MEMORY_FAULT_PINNED_CLOSE_AMBIGUOUS 4u
#define RUSTINFER_CUDA_TEST_MEMORY_FAULT_COPY_DEFERRED_SUBMISSION_ERROR 5u
#define RUSTINFER_CUDA_TEST_MEMORY_FAULT_COPY_COMPLETION_RESTORE_AMBIGUOUS 6u

typedef struct RustInferCudaTestMemoryFaultStats {
  uint32_t struct_size;
  uint32_t armed_fault;
  uint64_t faults_fired;
  uint64_t device_free_attempts;
  uint64_t pinned_free_attempts;
  uint64_t copy_use_release_attempts;
  uint64_t reserved[3];
} RustInferCudaTestMemoryFaultStats;
#endif

typedef struct RustInferCudaContext RustInferCudaContext;
typedef struct RustInferCudaStream RustInferCudaStream;
typedef struct RustInferCudaEvent RustInferCudaEvent;
typedef struct RustInferCudaSmokeBuffer RustInferCudaSmokeBuffer;
typedef struct RustInferCudaDeviceBuffer RustInferCudaDeviceBuffer;
typedef struct RustInferCudaPinnedHostBuffer RustInferCudaPinnedHostBuffer;
typedef struct RustInferCudaCopy RustInferCudaCopy;
typedef struct RustInferCudaGemmPlan RustInferCudaGemmPlan;
typedef struct RustInferCudaFixed37GemmPlan RustInferCudaFixed37GemmPlan;

// Raw C callers must externally synchronize opaque-handle lifetime: no call
// may begin with a handle while another thread can close that same handle.
// Native active-use guards reject close/reuse after an operation has entered,
// but cannot make a stale raw pointer safe if close races a new call. The safe
// Rust boundary enforces this rule with ownership and exclusive borrows.

typedef int32_t RustInferCudaDType;

#define RUSTINFER_CUDA_DTYPE_INVALID ((RustInferCudaDType)0)
#define RUSTINFER_CUDA_DTYPE_F32 ((RustInferCudaDType)1)
#define RUSTINFER_CUDA_DTYPE_BF16 ((RustInferCudaDType)2)
#define RUSTINFER_CUDA_DTYPE_U32 ((RustInferCudaDType)3)
#define RUSTINFER_CUDA_DTYPE_U8 ((RustInferCudaDType)4)
#define RUSTINFER_CUDA_DTYPE_U16 ((RustInferCudaDType)5)

// A borrowed, typed subspan of an opaque device allocation. byte_len is the
// caller-declared accessible capacity from byte_offset, not the allocation's
// total size. All known reserved fields must be zero. Primitive calls validate
// both this capacity and the underlying allocation before pointer arithmetic.
typedef struct RustInferCudaBufferSpan {
  uint32_t struct_size;
  RustInferCudaDType dtype;
  RustInferCudaDeviceBuffer* buffer;
  uint64_t byte_offset;
  uint64_t byte_len;
  uint64_t reserved[2];
} RustInferCudaBufferSpan;

#define RUSTINFER_CUDA_EMBEDDING_ERROR_NONE 0u
#define RUSTINFER_CUDA_EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE 1u

// Embedding execution uses a caller-owned device scratch span of exactly this
// record shape and copies the completed record to out_report before returning.
// For code NONE, token_position and token_id are zero. For OOB, they identify
// the lowest invalid token position and its id deterministically.
typedef struct RustInferCudaEmbeddingErrorReport {
  uint32_t struct_size;
  uint32_t code;
  uint64_t token_position;
  uint64_t token_id;
  uint64_t reserved;
} RustInferCudaEmbeddingErrorReport;

// Every parameter record below is caller-owned for the synchronous call. Set
// struct_size to sizeof(the record) and every known reserved field to zero;
// larger forward-compatible records are accepted only for additive ABI tails.

typedef struct RustInferCudaEmbeddingParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan table;
  RustInferCudaBufferSpan token_ids;
  RustInferCudaBufferSpan output;
  RustInferCudaBufferSpan device_error_scratch;
  RustInferCudaEmbeddingErrorReport* out_report;
  uint64_t token_count;
  uint64_t vocabulary_size;
  uint64_t hidden_size;
  uint64_t reserved[3];
} RustInferCudaEmbeddingParams;

typedef struct RustInferCudaRmsNormParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan weight;
  RustInferCudaBufferSpan output;
  uint64_t row_count;
  uint64_t hidden_size;
  float epsilon;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaRmsNormParams;

typedef struct RustInferCudaResidualAddParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan left;
  RustInferCudaBufferSpan right;
  RustInferCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RustInferCudaResidualAddParams;

// Fuses the exact BF16/F32 residual storage boundary with the immediately
// following RMSNorm. residual_output stores round(left + right) exactly as the
// standalone residual primitive would. RMSNorm reduces those stored values in
// FP32 and preserves its existing normalized-to-storage boundary before the
// learned weight multiply. The two outputs and weight must not overlap; exact
// residual-output alias with either residual input remains supported.
typedef struct RustInferCudaResidualRmsNormParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan left;
  RustInferCudaBufferSpan right;
  RustInferCudaBufferSpan weight;
  RustInferCudaBufferSpan residual_output;
  RustInferCudaBufferSpan normalized_output;
  uint64_t row_count;
  uint64_t hidden_size;
  float epsilon;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaResidualRmsNormParams;

// Full-vector log-softmax used by the fixed-contiguous-37-balanced-v1
// calibration profile. Input is BF16 [element_count], output is F32
// [element_count], and the two spans must not overlap. Any NaN input is
// propagated as canonical quiet NaN to every output. A +Inf maximum or an
// all--Inf vector likewise produces all quiet NaNs, matching the literal
// stable-log-softmax expression. With a finite maximum, individual -Inf
// inputs produce -Inf outputs. Max reduction uses CUDA fmaxf signed-zero
// semantics (+0 wins a +/-0 pair).
typedef struct RustInferCudaFixed37LogSoftmaxParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan logits;
  RustInferCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RustInferCudaFixed37LogSoftmaxParams;

// Adds one BF16 [column_count] bias vector to every row of a contiguous BF16
// [row_count, column_count] matrix in place. Each pair is expanded to F32,
// added once, then rounded to BF16 with round-to-nearest-even. column_count
// must be non-zero; row_count may be zero.
typedef struct RustInferCudaRowBiasAddInPlaceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan matrix;
  RustInferCudaBufferSpan bias;
  uint64_t row_count;
  uint64_t column_count;
  uint64_t reserved[4];
} RustInferCudaRowBiasAddInPlaceParams;

typedef struct RustInferCudaSiluParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RustInferCudaSiluParams;

// activated_gate is already SiLU-activated. This operation is deliberately a
// plain multiply; SiLU+multiply fusion is outside ABI v1's PR 06 path.
typedef struct RustInferCudaGatedMultiplyParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan activated_gate;
  RustInferCudaBufferSpan up;
  RustInferCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RustInferCudaGatedMultiplyParams;

// Standard non-interleaved Llama RoPE rotates the two contiguous halves of
// rotary_dimension. cos and sin are F32 tables with logical shape
// [table_position_count, rotary_dimension / 2]. The input/output logical shape
// is [token_count, head_count, head_size].
typedef struct RustInferCudaRopeParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan cos;
  RustInferCudaBufferSpan sin;
  RustInferCudaBufferSpan output;
  uint64_t token_count;
  uint64_t head_count;
  uint64_t head_size;
  uint64_t rotary_dimension;
  uint64_t table_position_count;
  uint64_t position_offset;
  uint64_t reserved[5];
} RustInferCudaRopeParams;

// Row-indexed non-interleaved Llama RoPE. positions is U32 [active_row_count]
// and selects an independent cos/sin table row for every dense input row.
// input/output are [active_row_count,head_count,head_size]. The safe Rust
// boundary validates its mirrored host positions before submission; the
// native kernel also bounds-checks every device position and writes a NaN
// rotary row instead of reading outside the tables when raw C metadata is
// malformed. The non-rotary tail remains a bit-preserving copy.
typedef struct RustInferCudaIndexedRopeParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan cos;
  RustInferCudaBufferSpan sin;
  RustInferCudaBufferSpan positions;
  RustInferCudaBufferSpan output;
  uint64_t active_row_count;
  uint64_t head_count;
  uint64_t head_size;
  uint64_t rotary_dimension;
  uint64_t table_position_count;
  uint64_t reserved[4];
} RustInferCudaIndexedRopeParams;

// Only BF16<->F32 conversions are accepted by this operation.
typedef struct RustInferCudaCastParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RustInferCudaCastParams;

// Allocation-free gather from a contiguous row-major input matrix. row_indices
// is U32 [output_row_count], input is [input_row_count,column_count], and
// output is [output_row_count,column_count]. Input/output must have one
// matching F32 or BF16 dtype and may not overlap. The safe Rust boundary
// validates mirrored indices; malformed raw C device indices produce a NaN
// output row without reading beyond input.
typedef struct RustInferCudaRowGatherParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan row_indices;
  RustInferCudaBufferSpan output;
  uint64_t input_row_count;
  uint64_t output_row_count;
  uint64_t column_count;
  uint64_t reserved[4];
} RustInferCudaRowGatherParams;

// Correctness-first materialized GQA attention. Query is BF16
// [token_count, query_head_count, head_size], key is BF16
// [token_count, key_value_head_count, head_size], and output is BF16
// [query_head_count, token_count, token_count]. Each query head maps to
// q_head / (query_head_count / key_value_head_count).
typedef struct RustInferCudaQkGqaParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key;
  RustInferCudaBufferSpan output;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RustInferCudaQkGqaParams;

// In-place BF16 scaling followed by an additive causal mask on materialized
// [query_head_count, token_count, token_count] scores. The scaled value is
// rounded to BF16 before the BF16 mask is added. Strictly future positions use
// the finite BF16 minimum bit pattern 0xff7f; allowed positions add +0.
typedef struct RustInferCudaScaleCausalMaskParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan scores;
  uint64_t token_count;
  uint64_t query_head_count;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaScaleCausalMaskParams;

// Stable causal softmax in place over the last dimension of BF16 materialized
// [query_head_count, token_count, token_count] scores. Max and sum reductions
// are F32; each resulting probability is rounded to BF16.
typedef struct RustInferCudaCausalSoftmaxParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan scores;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t reserved[5];
} RustInferCudaCausalSoftmaxParams;

// Materialized BF16 probabilities are [query_head_count, token_count,
// token_count], value is BF16 [token_count, key_value_head_count, head_size],
// and output is BF16 [token_count, query_head_count, head_size]. Accumulation
// is F32 and uses the same GQA head mapping as RustInferCudaQkGqaParams.
typedef struct RustInferCudaAvGqaParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan probabilities;
  RustInferCudaBufferSpan value;
  RustInferCudaBufferSpan output;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RustInferCudaAvGqaParams;

#define RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL 1u
#define RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL_LOCAL 2u

// Allocation-free online-softmax GQA prefill over dense contiguous BF16
// tensors. Query and output use [batch_count, token_count, query_head_count,
// head_size]; key and value use [batch_count, token_count,
// key_value_head_count, head_size]. The initial implementation supports
// head_size=64 and maps each query head to
// q_head / (query_head_count / key_value_head_count).
//
// CAUSAL requires local_window_size=0. CAUSAL_LOCAL admits the current token
// plus at most local_window_size-1 preceding tokens; a zero local window masks
// every key and produces an all-zero BF16 row from the empty online state. The
// implementation keeps the online maximum, denominator, and value numerator
// in F32 and writes BF16 output. It requires no workspace and never
// materializes a [S,S] score matrix in HBM.
typedef struct RustInferCudaPrefillAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key;
  RustInferCudaBufferSpan value;
  RustInferCudaBufferSpan output;
  uint64_t batch_count;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t mask_kind;
  uint64_t local_window_size;
  uint64_t reserved[4];
} RustInferCudaPrefillAttentionParams;

// Copies paired BF16 K/V rows from dense single-request source layout
// [source_token_count, key_value_head_count, head_size] into the contiguous
// cache layout [key_value_head_count, maximum_token_count, head_size]. The
// destination token interval starts at destination_token_start. Source rows
// are bit-preserving; no arithmetic conversion is performed. Every dimension,
// including source_token_count, must be non-zero.
typedef struct RustInferCudaKvCacheWriteParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan key_source;
  RustInferCudaBufferSpan value_source;
  RustInferCudaBufferSpan key_cache;
  RustInferCudaBufferSpan value_cache;
  uint64_t source_token_count;
  uint64_t destination_token_start;
  uint64_t maximum_token_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RustInferCudaKvCacheWriteParams;

// Correctness-first single-query decode over contiguous BF16 caches. Query
// and output are [query_head_count, head_size]; key_cache and value_cache are
// [key_value_head_count, maximum_token_count, head_size]. The first
// logical_token_count cache positions participate. score_workspace is BF16
// [query_head_count, logical_token_count] and is materialized through four
// stages: QK, scale, stable softmax, and AV. QK and scaling each round to BF16;
// softmax probabilities and the final output also round to BF16. NaN scores
// propagate; positive-infinity maxima share equal probability and an all
// negative-infinity row produces zero output, matching the online-state rules.
typedef struct RustInferCudaDecodeAttentionReferenceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key_cache;
  RustInferCudaBufferSpan value_cache;
  RustInferCudaBufferSpan score_workspace;
  RustInferCudaBufferSpan output;
  uint64_t maximum_token_count;
  uint64_t logical_token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaDecodeAttentionReferenceParams;

#define RUSTINFER_CUDA_DECODE_REDUCTION_ASCENDING 1u
#define RUSTINFER_CUDA_DECODE_REDUCTION_DESCENDING 2u
#define RUSTINFER_CUDA_DECODE_PARTIAL_STATE_VERSION 1u

// Partitioned online-softmax decode. The optimized producer supports
// head_size=64 and writes F32 partial_states with packed logical layout
// [partial_state_capacity, query_head_count, head_size + 2]. Within each
// packed row element 0 is max_score (m), element 1 is exp_sum (l), and elements
// [2, head_size+2) are the unnormalized weighted_value_sum (n). Only the first
// ceil(logical_token_count / tokens_per_partition) partitions are written;
// capacity tail bytes remain untouched. The selected ordered reducer merges
// those states and normalizes exactly once into BF16 output
// [query_head_count, head_size].
typedef struct RustInferCudaDecodeAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key_cache;
  RustInferCudaBufferSpan value_cache;
  RustInferCudaBufferSpan partial_states;
  RustInferCudaBufferSpan output;
  uint64_t maximum_token_count;
  uint64_t logical_token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t tokens_per_partition;
  uint64_t partial_state_capacity;
  float scale;
  uint32_t reduction_order;
  uint64_t reserved[4];
} RustInferCudaDecodeAttentionParams;

// Standalone ordered merge for the packed F32 partial-state ABI above. This
// reducer supports every positive head_size, so PR 10 paged producers can
// reuse it. It reads the first partial_state_count states from a buffer sized
// for partial_state_capacity, merges without normalizing intermediate states,
// then writes BF16 [query_head_count, head_size]. An empty state is encoded as
// (m=-inf, l=0, n=0); well-formed producer states use l>0 otherwise. A zero
// partial_state_count is valid and writes an all-zero output.
typedef struct RustInferCudaDecodePartialStateReduceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan partial_states;
  RustInferCudaBufferSpan output;
  uint64_t partial_state_count;
  uint64_t partial_state_capacity;
  uint64_t query_head_count;
  uint64_t head_size;
  uint32_t reduction_order;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaDecodePartialStateReduceParams;

#define RUSTINFER_CUDA_PAGED_KV_BLOCK_TABLE_VERSION 1u
#define RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE 16u
#define RUSTINFER_CUDA_PAGED_KV_METADATA_NONE 0u

// Exact paged-cache address-translation descriptor. block_ids is U32
// [block_count] in logical-block order; entries name physical blocks in
// [0, physical_block_count). valid_tokens is U16 [block_count]. Every block
// except the last contains 16 valid tokens and the last contains
// ((logical_token_count - 1) % 16) + 1. The safe Rust boundary validates the
// mirrored host arrays (including distinct/in-range physical IDs) before this
// device descriptor is submitted. metadata_kind/version must both be zero in
// v1: the exact path has no optional sidecar dependency, while the reserved
// tail leaves an additive extension point.
typedef struct RustInferCudaPagedKvBlockTableV1 {
  uint32_t struct_size;
  uint32_t format_version;
  RustInferCudaBufferSpan block_ids;
  RustInferCudaBufferSpan valid_tokens;
  uint64_t logical_token_count;
  uint64_t block_count;
  uint64_t physical_block_count;
  uint32_t block_size;
  uint32_t metadata_kind;
  uint32_t metadata_version;
  uint32_t reserved0;
  uint64_t reserved[3];
} RustInferCudaPagedKvBlockTableV1;

// Bit-preserving scatter from dense BF16 [T,KVH,D] sources into separate BF16
// pools [physical_block_count,KVH,16,D]. destination_token_start is logical;
// the table performs address translation and may contain shuffled physical
// block IDs. The table describes the post-write logical length.
typedef struct RustInferCudaPagedKvCacheWriteParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan key_source;
  RustInferCudaBufferSpan value_source;
  RustInferCudaBufferSpan key_pool;
  RustInferCudaBufferSpan value_pool;
  RustInferCudaPagedKvBlockTableV1 block_table;
  uint64_t source_token_count;
  uint64_t destination_token_start;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RustInferCudaPagedKvCacheWriteParams;

// Four-stage staged-BF16 correctness reference over paged BF16 pools. Scores
// are materialized as BF16 [QH,logical_token_count]. Only logical table order
// affects attention order; physical block numbering is opaque.
typedef struct RustInferCudaPagedDecodeAttentionReferenceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key_pool;
  RustInferCudaBufferSpan value_pool;
  RustInferCudaBufferSpan score_workspace;
  RustInferCudaBufferSpan output;
  RustInferCudaPagedKvBlockTableV1 block_table;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaPagedDecodeAttentionReferenceParams;

// Exact D64 paged online producer. One packed F32 DecodePartialState is
// produced for each logical 16-token block, then the unchanged PR 09 reducer
// merges block slots in logical order and normalizes once. Capacity is the
// preallocated number of logical block slots.
typedef struct RustInferCudaPagedDecodeAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key_pool;
  RustInferCudaBufferSpan value_pool;
  RustInferCudaBufferSpan partial_states;
  RustInferCudaBufferSpan output;
  RustInferCudaPagedKvBlockTableV1 block_table;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t partial_state_capacity;
  float scale;
  uint32_t reduction_order;
  uint64_t reserved[4];
} RustInferCudaPagedDecodeAttentionParams;

#define RUSTINFER_CUDA_PACKED_BATCH_VERSION 1u

// Packed multi-sequence address-translation descriptor. The device arrays are
// CSR sequence_block_offsets U32 [sequence_count+1], block_ids U32
// [block_count], valid_tokens U16 [block_count], row_sequence_slots U32
// [active_row_count], and row_positions U32 [active_row_count]. Offsets index
// block_ids/valid_tokens, whose entries remain in logical-block order per
// sequence. block_ids name blocks in [0,physical_block_count), block_size is
// exactly 16, and each row position is the logical token written/queried by
// that active row. Every non-last block of a sequence has 16 valid tokens;
// its last block has [1,16], thereby defining the sequence's post-write
// logical length, which must include every row position assigned to it. The
// safe Rust boundary validates mirrored arrays including CSR monotonicity,
// non-empty sequence ranges, physical-ID uniqueness/range, canonical
// valid-token counts, row slots, row positions, and uniqueness of each
// (sequence,row-position) pair. Native kernels independently bounds-guard all
// device-derived indices so malformed raw C metadata cannot access outside a
// declared span or pool.
typedef struct RustInferCudaPackedBatchV1 {
  uint32_t struct_size;
  uint32_t format_version;
  RustInferCudaBufferSpan sequence_block_offsets;
  RustInferCudaBufferSpan block_ids;
  RustInferCudaBufferSpan valid_tokens;
  RustInferCudaBufferSpan row_sequence_slots;
  RustInferCudaBufferSpan row_positions;
  uint64_t sequence_count;
  uint64_t block_count;
  uint64_t active_row_count;
  uint64_t physical_block_count;
  uint32_t block_size;
  uint32_t reserved0;
  uint64_t reserved[4];
} RustInferCudaPackedBatchV1;

// Bit-preserving BF16 scatter from dense active rows [T,KVH,D] into shared
// pools [physical_block_count,KVH,16,D]. Each row uses its packed sequence slot
// and logical position. Invalid raw C metadata makes only that row a no-op.
typedef struct RustInferCudaRaggedPagedKvCacheWriteParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan key_source;
  RustInferCudaBufferSpan value_source;
  RustInferCudaBufferSpan key_pool;
  RustInferCudaBufferSpan value_pool;
  RustInferCudaPackedBatchV1 batch;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RustInferCudaRaggedPagedKvCacheWriteParams;

// Allocation-free ragged causal paged attention. Query is BF16 [T,QH,64],
// output is BF16 [output_row_count,QH,64], pools are BF16
// [physical_block_count,KVH,16,64], and each active row attends logical tokens
// [0,row_positions[row]] in its own CSR sequence. output_row_count must be at
// least T; every padding row [T,output_row_count) is overwritten with zero so
// a prepared fixed-M projection never observes uninitialized bytes. QH must be
// divisible by KVH. Dot products, online softmax state, and value accumulation
// are F32; the final output rounds once to BF16. Invalid raw C metadata produces
// a NaN output row rather than an out-of-bounds access.
typedef struct RustInferCudaRaggedPagedAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan query;
  RustInferCudaBufferSpan key_pool;
  RustInferCudaBufferSpan value_pool;
  RustInferCudaBufferSpan output;
  RustInferCudaPackedBatchV1 batch;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t output_row_count;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RustInferCudaRaggedPagedAttentionParams;

#define RUSTINFER_CUDA_GEMM_TRANSPOSE_N 0u
#define RUSTINFER_CUDA_GEMM_TRANSPOSE_T 1u
#define RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR 1u
#define RUSTINFER_CUDA_GEMM_EPILOGUE_NONE 0u
#define RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED 1u
#define RUSTINFER_CUDA_GEMM_BACKEND_CUBLASLT 1u
#define RUSTINFER_CUDA_GEMM_BACKEND_FIXED37 2u

#define RUSTINFER_CUDA_FIXED37_REDUCTION_VERSION 1u
#define RUSTINFER_CUDA_FIXED37_CHUNK_ELEMENTS 37u
#define RUSTINFER_CUDA_FIXED37_MAX_CHUNK_COUNT 4096u

// PR 06 deliberately exposes one exact dense GEMM contract. The logical
// operation is row-major Y[M,N] = X[M,K] * W[N,K]^T with BF16 X/W/Y and F32
// accumulation. input_transpose must be N, weight_transpose must be T, all
// layouts must be ROW_MAJOR, epilogue must be NONE, and deterministic must be
// DETERMINISTIC_REQUIRED. max_workspace_bytes is a preparation-time cap; the
// selected exact requirement is returned by gemm_plan_info. flags, reserved0,
// and every reserved element must be zero.
typedef struct RustInferCudaGemmConfig {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  RustInferCudaDType input_dtype;
  RustInferCudaDType weight_dtype;
  RustInferCudaDType accumulator_dtype;
  RustInferCudaDType output_dtype;
  uint32_t input_transpose;
  uint32_t weight_transpose;
  uint32_t input_layout;
  uint32_t weight_layout;
  uint32_t output_layout;
  uint32_t epilogue;
  uint32_t deterministic;
  uint32_t reserved0;
  uint64_t max_workspace_bytes;
  uint64_t reserved[3];
} RustInferCudaGemmConfig;

// Immutable metadata for the algorithm prepared into an opaque GEMM plan.
// IDs are cuBLASLt algorithm configuration values and are meaningful together
// with compute capability and the recorded CUDA Runtime/cuBLASLt versions.
typedef struct RustInferCudaGemmAlgorithmInfo {
  uint32_t struct_size;
  uint32_t backend;
  int32_t algorithm_id;
  uint32_t tile_id;
  uint32_t stages_id;
  uint32_t split_k;
  uint32_t reduction_scheme;
  uint32_t cta_swizzling;
  uint32_t custom_option;
  uint32_t deterministic;
  uint64_t workspace_bytes;
  uint64_t numerical_implementation_flags;
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  int32_t runtime_version;
  int32_t cublaslt_version;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t reserved[2];
} RustInferCudaGemmAlgorithmInfo;

// Immutable metadata for the custom fixed-contiguous-37-balanced-v1 GEMM
// plan. The plan uses no caller workspace. dynamic_shared_memory_bytes is the
// exact per-block scratch used for two alternating arrays of chunk partials.
typedef struct RustInferCudaFixed37GemmPlanInfo {
  uint32_t struct_size;
  uint32_t backend;
  uint32_t reduction_version;
  uint32_t chunk_elements;
  RustInferCudaDType accumulator_dtype;
  RustInferCudaDType output_dtype;
  uint32_t threads_per_block;
  uint32_t deterministic;
  uint64_t dynamic_shared_memory_bytes;
  uint64_t workspace_bytes;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t reserved[3];
} RustInferCudaFixed37GemmPlanInfo;

#ifdef __cplusplus
#define RUSTINFER_CUDA_NOEXCEPT noexcept
extern "C" {
#else
#define RUSTINFER_CUDA_NOEXCEPT
#endif

// Compile-time ABI metadata. These functions do not initialize a device.
uint32_t rustinfer_cuda_abi_version(void) RUSTINFER_CUDA_NOEXCEPT;
const char* rustinfer_cuda_build_info(void) RUSTINFER_CUDA_NOEXCEPT;

RustInferCudaStatus rustinfer_cuda_device_count(
    uint32_t* out_count,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_device_properties(
    int32_t ordinal,
    RustInferCudaDeviceProperties* out_properties,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Context is a retained lease on the target device's CUDA primary context.
RustInferCudaStatus rustinfer_cuda_context_create(
    int32_t ordinal,
    RustInferCudaContext** out_context,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_context_synchronize(
    RustInferCudaContext* context,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_context_memory_info(
    RustInferCudaContext* context,
    uint64_t* out_free_bytes,
    uint64_t* out_total_bytes,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_context_allocation_stats(
    RustInferCudaContext* context,
    RustInferCudaAllocationStats* out_stats,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Once primary-context release is attempted, *context is null even if a
// deferred asynchronous error is returned. Validation, poison, live-child, or
// non-zero allocation-accounting failures before the attempt leave it intact.
RustInferCudaStatus rustinfer_cuda_context_close(
    RustInferCudaContext** context,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Streams are explicitly created as non-blocking, non-default streams.
RustInferCudaStatus rustinfer_cuda_stream_create(
    RustInferCudaContext* context,
    RustInferCudaStream** out_stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// A command batch retains the stream and every unique device-buffer/GEMM-plan
// used by supported operations on the owning thread. Those operations enqueue
// without per-operation synchronization; end performs the single completion
// synchronization. Failed or ambiguous completion deliberately retains all
// leases. Nested, cross-thread, query/synchronize/wait/close use is rejected.
RustInferCudaStatus rustinfer_cuda_stream_command_batch_begin(
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_stream_command_batch_end(
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_stream_query(
    RustInferCudaStream* stream,
    uint8_t* out_complete,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_stream_synchronize(
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_stream_wait_event(
    RustInferCudaStream* stream,
    RustInferCudaEvent* event,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Once native destruction is attempted, *stream is null even if a deferred
// asynchronous error is returned; callers must inspect both status and handle.
RustInferCudaStatus rustinfer_cuda_stream_close(
    RustInferCudaStream** stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Events are timing-enabled so elapsed time remains available.
RustInferCudaStatus rustinfer_cuda_event_create(
    RustInferCudaContext* context,
    RustInferCudaEvent** out_event,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_event_record(
    RustInferCudaEvent* event,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_event_query(
    RustInferCudaEvent* event,
    uint8_t* out_complete,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_event_synchronize(
    RustInferCudaEvent* event,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_event_elapsed_ms(
    RustInferCudaEvent* start,
    RustInferCudaEvent* end,
    float* out_elapsed_ms,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Uses the same single-attempt ownership rule as stream_close.
RustInferCudaStatus rustinfer_cuda_event_close(
    RustInferCudaEvent** event,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// General byte-addressed allocations. A zero-byte allocation still returns an
// owned opaque handle and contributes one live allocation with zero live bytes.
RustInferCudaStatus rustinfer_cuda_device_buffer_create(
    RustInferCudaContext* context,
    uint64_t byte_len,
    RustInferCudaDeviceBuffer** out_buffer,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Active copy/primitive uses make close fail with INVALID_STATE before cudaFree.
// Once cudaFree is attempted, the handle follows the single-shot close rule.
// An ambiguous failed free stays logically accounted and keeps a context-child
// lease so allocation stats/context teardown remain fail closed.
RustInferCudaStatus rustinfer_cuda_device_buffer_close(
    RustInferCudaDeviceBuffer** buffer,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_create(
    RustInferCudaContext* context,
    uint64_t byte_len,
    RustInferCudaPinnedHostBuffer** out_buffer,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Synchronous CPU access is rejected while an async use is active.
RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_write(
    RustInferCudaPinnedHostBuffer* buffer,
    uint64_t destination_offset,
    const uint8_t* source,
    uint64_t source_len,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_read(
    RustInferCudaPinnedHostBuffer* buffer,
    uint64_t source_offset,
    uint8_t* destination,
    uint64_t destination_len,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Active copy uses make close fail before cudaFreeHost. A free attempt is
// otherwise single-shot even when CUDA reports a deferred earlier error;
// ambiguous failure remains logically live/accounted.
RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_close(
    RustInferCudaPinnedHostBuffer** buffer,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Non-zero copies return one owning pending token. Submission errors observed
// after cudaMemcpyAsync is attempted are stored in that token and surfaced by
// query/synchronize, preserving all buffer lifetimes until completion. A
// zero-byte copy is a successful no-op and returns *out_copy == NULL.
RustInferCudaStatus rustinfer_cuda_copy_h2d_async(
    RustInferCudaDeviceBuffer* destination,
    uint64_t destination_offset,
    RustInferCudaPinnedHostBuffer* source,
    uint64_t source_offset,
    uint64_t byte_len,
    RustInferCudaStream* stream,
    RustInferCudaCopy** out_copy,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_copy_d2h_async(
    RustInferCudaPinnedHostBuffer* destination,
    uint64_t destination_offset,
    RustInferCudaDeviceBuffer* source,
    uint64_t source_offset,
    uint64_t byte_len,
    RustInferCudaStream* stream,
    RustInferCudaCopy** out_copy,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// out_complete is 1 when native buffer-use counters have been released, even
// if the returned status reports a deferred submission error.
RustInferCudaStatus rustinfer_cuda_copy_query(
    RustInferCudaCopy* copy,
    uint8_t* out_complete,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_copy_synchronize(
    RustInferCudaCopy* copy,
    uint8_t* out_complete,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// An incomplete token is synchronized before close. It is consumed only after
// completion is confirmed; otherwise the handle and active-use guards remain.
RustInferCudaStatus rustinfer_cuda_copy_close(
    RustInferCudaCopy** copy,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

#if defined(RUSTINFER_CUDA_ENABLE_TEST_FAULT_INJECTION)
// One process-local injector session is bound to one context. Reset clears all
// counters; arm is one-shot. Ambiguous cases intentionally poison/leak their
// subprocess, so callers must isolate every case in a fresh process.
RustInferCudaStatus rustinfer_cuda_test_memory_fault_reset(
    RustInferCudaContext* context,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_test_memory_fault_arm(
    RustInferCudaContext* context,
    uint32_t fault,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_test_memory_fault_stats(
    RustInferCudaContext* context,
    RustInferCudaTestMemoryFaultStats* out_stats,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
#endif

// Core primitive calls are synchronously complete even though their device
// work is enqueued on the explicit stream: each call validates and exclusively
// borrows all opaque handles, launches without host/device allocation, then
// synchronizes that same stream before returning. A synchronization or CUDA
// context-restoration failure leaves the handles permanently busy/poisoned so
// close or reuse fails closed instead of risking use-after-free.
//
// Declared spans may have excess trailing capacity, but every touched byte is
// checked against both span.byte_len and the opaque allocation. Exact
// input/output alias is accepted only where documented below; all partial
// write/input overlap is rejected. Zero logical elements are a validated
// allocation-free no-op and do not launch or synchronize. Arithmetic follows
// CUDA IEEE NaN/Inf propagation and BF16 round-to-nearest behavior; values are
// never silently sanitized.

// table/output accept F32 or BF16 and must match; token_ids is U32 and scratch
// is U8. output may not alias any input or scratch. If any token is OOB, output
// remains untouched, the lowest bad token is reported, and OUT_OF_RANGE is
// returned only after the stream is confirmed complete. A zero-token call
// clears only out_report; its device scratch is intentionally untouched.
RustInferCudaStatus rustinfer_cuda_embedding_execute(
    const RustInferCudaEmbeddingParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// RMSNorm accumulates sum(x*x), mean, reciprocal square root, and scaling in
// F32 for both accepted storage dtypes. Exact input/output alias is supported;
// weight/output overlap and partial input/output overlap are rejected.
RustInferCudaStatus rustinfer_cuda_rms_norm_execute(
    const RustInferCudaRmsNormParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Alternate RMSNorm entry point for fixed-contiguous-37-balanced-v1. The
// storage, alias, exceptional-value, and synchronous-completion contract is
// identical to rustinfer_cuda_rms_norm_execute; only the sum-of-squares
// reduction order changes.
RustInferCudaStatus rustinfer_cuda_fixed37_rms_norm_execute(
    const RustInferCudaRmsNormParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Exact output alias with either input is supported. Partial overlap is not.
RustInferCudaStatus rustinfer_cuda_residual_add_execute(
    const RustInferCudaResidualAddParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Exact fused equivalent of residual_add followed by RMSNorm. This is an
// additive ABI-v1 entry point; the standalone functions remain the fallback.
RustInferCudaStatus rustinfer_cuda_residual_rms_norm_execute(
    const RustInferCudaResidualRmsNormParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Fixed-contiguous-37-balanced-v1 sibling of the exact fused residual plus
// RMSNorm primitive. Residual storage rounding remains unchanged.
RustInferCudaStatus rustinfer_cuda_fixed37_residual_rms_norm_execute(
    const RustInferCudaResidualRmsNormParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Computes F32 log-softmax from BF16 logits with fixed 37-element max and sum
// chunks followed by the reviewed adjacent balanced tree. element_count must
// be non-zero and no canonical implementation is selected as a fallback.
RustInferCudaStatus rustinfer_cuda_fixed37_log_softmax_execute(
    const RustInferCudaFixed37LogSoftmaxParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// BF16-only row-wise bias addition. matrix is the sole output and therefore
// is always exact-in-place. bias may not overlap any touched matrix byte.
RustInferCudaStatus rustinfer_cuda_row_bias_add_in_place_execute(
    const RustInferCudaRowBiasAddInPlaceParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Exact input/output alias is supported. silu(x) is evaluated as
// x / (1 + exp(-x)); exceptional values follow CUDA arithmetic.
RustInferCudaStatus rustinfer_cuda_silu_execute(
    const RustInferCudaSiluParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Computes activated_gate * up. Exact output alias with either input is
// supported; partial overlap is rejected.
RustInferCudaStatus rustinfer_cuda_gated_multiply_execute(
    const RustInferCudaGatedMultiplyParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// cos/sin must be F32. input/output accept one matching F32 or BF16 dtype.
// Exact input/output alias is supported because each rotary pair is owned by a
// single CUDA thread; table/output overlap and partial alias are rejected.
RustInferCudaStatus rustinfer_cuda_rope_execute(
    const RustInferCudaRopeParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Indexed RoPE has the same dtype, aliasing, synchronous-stream, and
// allocation-free guarantees as rustinfer_cuda_rope_execute.
RustInferCudaStatus rustinfer_cuda_indexed_rope_execute(
    const RustInferCudaIndexedRopeParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// BF16<->F32 only. F32 NaNs narrow to CUDA's canonical BF16 NaN 0x7fff;
// BF16-to-F32 expansion preserves the source BF16 bits. Any input/output
// overlap is rejected.
RustInferCudaStatus rustinfer_cuda_cast_execute(
    const RustInferCudaCastParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

RustInferCudaStatus rustinfer_cuda_row_gather_execute(
    const RustInferCudaRowGatherParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// These four allocation-free calls expose the deliberately materialized PR 07
// reference-attention boundary. All spans must be BF16. Unlike the general PR
// 06 primitives, every attention dimension must be non-zero. QK and AV use F32
// accumulators and round their BF16 outputs once per completed dot product.
RustInferCudaStatus rustinfer_cuda_qk_gqa_execute(
    const RustInferCudaQkGqaParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_scale_causal_mask_in_place_execute(
    const RustInferCudaScaleCausalMaskParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_causal_softmax_in_place_execute(
    const RustInferCudaCausalSoftmaxParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_av_gqa_execute(
    const RustInferCudaAvGqaParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Fixed-contiguous-37 materialized attention siblings. Every reduction axis
// is split into ascending chunks of 37 elements. Each chunk is accumulated by
// an ascending F32 left fold, then adjacent partials are merged by a balanced
// binary tree with odd carry. QK and AV round once to BF16 after their complete
// dot product. The existing scale/causal-mask primitive stages the canonical
// finite BF16-minimum mask. Softmax reduces the complete logical S axis,
// including masked entries, and rounds each probability to BF16 before AV.
// A row containing NaN, with a +Inf maximum, or containing only -Inf becomes
// a complete canonical BF16 qNaN row (bits 0x7fff). With a finite maximum,
// -Inf entries produce zero probability; because AV consumes the rounded BF16
// probabilities, zero probability multiplied by an infinite value produces the
// same canonical BF16 qNaN result.
RustInferCudaStatus rustinfer_cuda_fixed37_qk_gqa_execute(
    const RustInferCudaQkGqaParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_fixed37_causal_softmax_in_place_execute(
    const RustInferCudaCausalSoftmaxParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_fixed37_av_gqa_execute(
    const RustInferCudaAvGqaParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Fixed37 no-HBM two-pass prefill. The current implementation requires D=64
// and S<=8192, uses no caller workspace, and has the same dense BSHD and mask
// contract as RustInferCudaPrefillAttentionParams except that CAUSAL_LOCAL with
// local_window_size=0 returns NOT_SUPPORTED. Its two score passes reproduce the
// materialized raw-BF16 -> scaled-BF16 -> finite-min-mask-BF16 staging and the
// same fixed37 maximum, denominator, BF16-probability, and AV reduction order.
RustInferCudaStatus rustinfer_cuda_fixed37_prefill_attention_execute(
    const RustInferCudaPrefillAttentionParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Executes the dense online-softmax prefill contract above synchronously on
// stream. Unsupported head dimensions return NOT_SUPPORTED before launching.
RustInferCudaStatus rustinfer_cuda_prefill_attention_execute(
    const RustInferCudaPrefillAttentionParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// These single-request cache/decode calls are allocation-free, exclusively
// borrow every distinct opaque buffer and the explicit stream, and synchronize
// that stream before returning. Writable spans may not overlap any other
// touched span. Cache reads use only the logical prefix but cache spans must
// declare the complete maximum-token strided capacity.
RustInferCudaStatus rustinfer_cuda_kv_cache_write_execute(
    const RustInferCudaKvCacheWriteParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_decode_attention_reference_execute(
    const RustInferCudaDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Fixed-contiguous-37 materialized decode reuses the reference descriptor and
// BF16 [QH,T] workspace. Logical D and T are each limited to 151552 elements.
// QK, softmax maximum/denominator, and AV use ascending 37-element F32 left
// folds followed by adjacent balanced-tree merges with odd carry. QK rounds to
// raw BF16, scaling rounds again to BF16, softmax probabilities round to BF16,
// and AV consumes those rounded probabilities. A row containing NaN, with a
// +Inf maximum, or containing only -Inf becomes a complete canonical BF16 qNaN
// row (bits 0x7fff); finite-max -Inf entries have zero probability and 0*Inf in
// AV becomes canonical BF16 qNaN.
RustInferCudaStatus rustinfer_cuda_fixed37_decode_attention_reference_execute(
    const RustInferCudaDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_decode_attention_execute(
    const RustInferCudaDecodeAttentionParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_decode_partial_state_reduce_execute(
    const RustInferCudaDecodePartialStateReduceParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_paged_kv_cache_write_execute(
    const RustInferCudaPagedKvCacheWriteParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_paged_decode_attention_reference_execute(
    const RustInferCudaPagedDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Paged fixed37 materialized decode has the same numerical contract as the
// contiguous fixed37 symbol. Page16 performs address translation only: every T
// reduction chunk remains anchored at logical token zero, independent of page
// or physical-block boundaries and numbering.
RustInferCudaStatus
rustinfer_cuda_fixed37_paged_decode_attention_reference_execute(
    const RustInferCudaPagedDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_paged_decode_attention_execute(
    const RustInferCudaPagedDecodeAttentionParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Packed-batch calls are allocation-free, exclusively borrow every distinct
// opaque buffer and the explicit stream, and synchronize that stream before
// returning. Writable spans may not overlap any other touched span.
RustInferCudaStatus rustinfer_cuda_ragged_paged_kv_cache_write_execute(
    const RustInferCudaRaggedPagedKvCacheWriteParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_ragged_paged_attention_execute(
    const RustInferCudaRaggedPagedAttentionParams* params,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Plan creation performs all cuBLASLt descriptor construction and heuristic
// selection. A successful plan owns one context-child lease and is immutable.
RustInferCudaStatus rustinfer_cuda_gemm_plan_create(
    RustInferCudaContext* context,
    const RustInferCudaGemmConfig* config,
    RustInferCudaGemmPlan** out_plan,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_gemm_plan_info(
    RustInferCudaGemmPlan* plan,
    RustInferCudaGemmAlgorithmInfo* out_info,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Executes the prepared logical row-major operation without byte reordering:
// cuBLASLt sees column-major TN(W, X, Y). Every span must have exactly the
// prepared byte length (workspace uses the selected requirement), a
// 256-byte-aligned byte_offset, and a handle owned by the plan's context. Any
// overlap among X/W/Y/workspace is rejected. The call exclusively borrows the
// plan, buffers, and explicit stream, synchronizes that same stream, and only
// releases the guards after completion and context restoration are confirmed.
// No allocation, heuristic query, or descriptor creation occurs here.
RustInferCudaStatus rustinfer_cuda_gemm_plan_execute(
    RustInferCudaGemmPlan* plan,
    const RustInferCudaBufferSpan* input,
    const RustInferCudaBufferSpan* weight,
    const RustInferCudaBufferSpan* output,
    const RustInferCudaBufferSpan* workspace,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Active or permanently guarded plans cannot close. Native descriptor
// destruction and context restoration must both complete before *plan is
// consumed and its context-child lease is released.
RustInferCudaStatus rustinfer_cuda_gemm_plan_close(
    RustInferCudaGemmPlan** plan,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Prepares the custom fixed-contiguous-37-balanced-v1 implementation for the
// same logical BF16/F32 GEMM contract as RustInferCudaGemmConfig. The custom
// plan never selects or falls back to cuBLASLt and requires no caller
// workspace.
RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_create(
    RustInferCudaContext* context,
    const RustInferCudaGemmConfig* config,
    RustInferCudaFixed37GemmPlan** out_plan,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_info(
    RustInferCudaFixed37GemmPlan* plan,
    RustInferCudaFixed37GemmPlanInfo* out_info,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_execute(
    RustInferCudaFixed37GemmPlan* plan,
    const RustInferCudaBufferSpan* input,
    const RustInferCudaBufferSpan* weight,
    const RustInferCudaBufferSpan* output,
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_close(
    RustInferCudaFixed37GemmPlan** plan,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

// Diagnostic-only storage keeps generic tensor allocation outside PR 03.
RustInferCudaStatus rustinfer_cuda_smoke_buffer_create(
    RustInferCudaContext* context,
    uint64_t element_count,
    RustInferCudaSmokeBuffer** out_buffer,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_smoke_fill_launch(
    RustInferCudaSmokeBuffer* buffer,
    RustInferCudaStream* stream,
    float value,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
RustInferCudaStatus rustinfer_cuda_smoke_copy_to_host(
    RustInferCudaSmokeBuffer* buffer,
    RustInferCudaStream* stream,
    float* host_output,
    uint64_t host_element_capacity,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Once cudaFree is attempted, *buffer is null even if CUDA reports a deferred
// asynchronous error; this prevents a retry from double-freeing the storage.
RustInferCudaStatus rustinfer_cuda_smoke_buffer_close(
    RustInferCudaSmokeBuffer** buffer,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;
// Intentionally returns a launch-stage error without poisoning the context.
RustInferCudaStatus rustinfer_cuda_smoke_invalid_launch(
    RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) RUSTINFER_CUDA_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

#undef RUSTINFER_CUDA_NOEXCEPT

#endif  // RUSTINFER_CUDA_H_
