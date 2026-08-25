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

typedef struct RustInferCudaContext RustInferCudaContext;
typedef struct RustInferCudaStream RustInferCudaStream;
typedef struct RustInferCudaEvent RustInferCudaEvent;
typedef struct RustInferCudaSmokeBuffer RustInferCudaSmokeBuffer;
typedef struct RustInferCudaDeviceBuffer RustInferCudaDeviceBuffer;
typedef struct RustInferCudaPinnedHostBuffer RustInferCudaPinnedHostBuffer;
typedef struct RustInferCudaCopy RustInferCudaCopy;
typedef struct RustInferCudaGemmPlan RustInferCudaGemmPlan;

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

// Only BF16<->F32 conversions are accepted by this operation.
typedef struct RustInferCudaCastParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RustInferCudaBufferSpan input;
  RustInferCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RustInferCudaCastParams;

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

#define RUSTINFER_CUDA_GEMM_TRANSPOSE_N 0u
#define RUSTINFER_CUDA_GEMM_TRANSPOSE_T 1u
#define RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR 1u
#define RUSTINFER_CUDA_GEMM_EPILOGUE_NONE 0u
#define RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED 1u
#define RUSTINFER_CUDA_GEMM_BACKEND_CUBLASLT 1u

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

// Exact output alias with either input is supported. Partial overlap is not.
RustInferCudaStatus rustinfer_cuda_residual_add_execute(
    const RustInferCudaResidualAddParams* params,
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

// BF16<->F32 only. F32 NaNs narrow to CUDA's canonical BF16 NaN 0x7fff;
// BF16-to-F32 expansion preserves the source BF16 bits. Any input/output
// overlap is rejected.
RustInferCudaStatus rustinfer_cuda_cast_execute(
    const RustInferCudaCastParams* params,
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
