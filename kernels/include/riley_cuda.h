#ifndef RILEY_CUDA_H_
#define RILEY_CUDA_H_

#include <stdint.h>

#define RILEY_CUDA_ABI_VERSION 1
#define RILEY_CUDA_ERROR_MESSAGE_CAPACITY 256
#define RILEY_CUDA_DEVICE_NAME_CAPACITY 256
#define RILEY_CUDA_NVIDIA_DRIVER_VERSION_CAPACITY 80
#define RILEY_CUDA_NVIDIA_ENVIRONMENT_MAX_DEVICES 32u
#define RILEY_CUDA_NVIDIA_CLOCK_NOT_AVAILABLE UINT32_MAX

// Riley ABI v1 is a 64-bit ABI and starts a namespace reset from the
// pre-release rustinfer_cuda_* symbols. Callers of that development ABI must
// rebuild; additions within the Riley v1 namespace remain link- and
// layout-compatible.
#define RILEY_CUDA_ABI_POINTER_WIDTH 64u

typedef int32_t RileyCudaStatus;

#define RILEY_CUDA_STATUS_SUCCESS ((RileyCudaStatus)0)
#define RILEY_CUDA_STATUS_INVALID_ARGUMENT ((RileyCudaStatus)1)
#define RILEY_CUDA_STATUS_INVALID_DEVICE ((RileyCudaStatus)2)
#define RILEY_CUDA_STATUS_OUT_OF_RANGE ((RileyCudaStatus)3)
#define RILEY_CUDA_STATUS_NOT_READY ((RileyCudaStatus)4)
#define RILEY_CUDA_STATUS_OUT_OF_MEMORY ((RileyCudaStatus)5)
#define RILEY_CUDA_STATUS_DRIVER_ERROR ((RileyCudaStatus)6)
#define RILEY_CUDA_STATUS_RUNTIME_ERROR ((RileyCudaStatus)7)
#define RILEY_CUDA_STATUS_INVALID_STATE ((RileyCudaStatus)8)
#define RILEY_CUDA_STATUS_INTERNAL_ERROR ((RileyCudaStatus)9)
#define RILEY_CUDA_STATUS_CUBLASLT_ERROR ((RileyCudaStatus)10)
#define RILEY_CUDA_STATUS_NOT_SUPPORTED ((RileyCudaStatus)11)

#define RILEY_CUDA_ERROR_DOMAIN_NONE 0u
#define RILEY_CUDA_ERROR_DOMAIN_VALIDATION 1u
#define RILEY_CUDA_ERROR_DOMAIN_DRIVER 2u
#define RILEY_CUDA_ERROR_DOMAIN_RUNTIME 3u
#define RILEY_CUDA_ERROR_DOMAIN_INTERNAL 4u
#define RILEY_CUDA_ERROR_DOMAIN_CUBLASLT 5u
#define RILEY_CUDA_ERROR_DOMAIN_NVML 6u

#define RILEY_CUDA_ERROR_STAGE_INITIALIZE 1u
#define RILEY_CUDA_ERROR_STAGE_VALIDATION 2u
#define RILEY_CUDA_ERROR_STAGE_CREATE 3u
#define RILEY_CUDA_ERROR_STAGE_LAUNCH 4u
#define RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE 5u
#define RILEY_CUDA_ERROR_STAGE_QUERY 6u
#define RILEY_CUDA_ERROR_STAGE_RECORD 7u
#define RILEY_CUDA_ERROR_STAGE_COPY 8u
#define RILEY_CUDA_ERROR_STAGE_CLOSE 9u
#define RILEY_CUDA_ERROR_STAGE_PREPARE 10u

typedef struct RileyCudaErrorInfo {
  uint32_t struct_size;
  int32_t native_code;
  uint32_t domain;
  uint32_t stage;
  char message[RILEY_CUDA_ERROR_MESSAGE_CAPACITY];
} RileyCudaErrorInfo;

typedef struct RileyCudaDeviceProperties {
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
  char name[RILEY_CUDA_DEVICE_NAME_CAPACITY];
} RileyCudaDeviceProperties;

#define RILEY_CUDA_NVIDIA_PERSISTENCE_DISABLED 0u
#define RILEY_CUDA_NVIDIA_PERSISTENCE_ENABLED 1u

// One process-local NVML device snapshot. used_memory_bytes is allocated
// device memory and excludes the v2 system-reserved driver/firmware amount,
// matching the user-visible nvidia-smi memory.used value. Application clocks use
// RILEY_CUDA_NVIDIA_CLOCK_NOT_AVAILABLE only when NVML reports that the
// query is unsupported; every other NVML error fails the complete probe.
typedef struct RileyCudaNvidiaDeviceSnapshot {
  uint32_t struct_size;
  uint32_t index;
  uint64_t total_memory_bytes;
  uint64_t used_memory_bytes;
  uint32_t temperature_c;
  uint32_t persistence_mode;
  uint32_t power_limit_milliwatts;
  uint32_t application_graphics_clock_mhz;
  uint32_t application_memory_clock_mhz;
  uint32_t compute_process_count;
  uint64_t reserved[2];
  char name[RILEY_CUDA_DEVICE_NAME_CAPACITY];
} RileyCudaNvidiaDeviceSnapshot;

// Fixed-capacity, caller-owned output keeps NVML ownership and allocations
// entirely inside one synchronous C call. Hosts with more than the documented
// maximum fail closed instead of returning a truncated environment.
typedef struct RileyCudaNvidiaEnvironmentSnapshot {
  uint32_t struct_size;
  int32_t cuda_driver_api_version;
  uint32_t device_count;
  uint32_t compute_process_count;
  uint64_t reserved[2];
  char driver_version[RILEY_CUDA_NVIDIA_DRIVER_VERSION_CAPACITY];
  RileyCudaNvidiaDeviceSnapshot
      devices[RILEY_CUDA_NVIDIA_ENVIRONMENT_MAX_DEVICES];
} RileyCudaNvidiaEnvironmentSnapshot;

typedef struct RileyCudaAllocationStats {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t device_live_bytes;
  uint64_t device_live_allocations;
  uint64_t pinned_host_live_bytes;
  uint64_t pinned_host_live_allocations;
} RileyCudaAllocationStats;

#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
// Destructive test-only ABI. These declarations and their symbols are absent
// from ordinary archives. Enabling this definition in production is unsupported.
#define RILEY_CUDA_TEST_MEMORY_FAULT_DEVICE_CREATE_ROLLBACK_AMBIGUOUS 1u
#define RILEY_CUDA_TEST_MEMORY_FAULT_PINNED_CREATE_ROLLBACK_AMBIGUOUS 2u
#define RILEY_CUDA_TEST_MEMORY_FAULT_DEVICE_CLOSE_AMBIGUOUS 3u
#define RILEY_CUDA_TEST_MEMORY_FAULT_PINNED_CLOSE_AMBIGUOUS 4u
#define RILEY_CUDA_TEST_MEMORY_FAULT_COPY_DEFERRED_SUBMISSION_ERROR 5u
#define RILEY_CUDA_TEST_MEMORY_FAULT_COPY_COMPLETION_RESTORE_AMBIGUOUS 6u

typedef struct RileyCudaTestMemoryFaultStats {
  uint32_t struct_size;
  uint32_t armed_fault;
  uint64_t faults_fired;
  uint64_t device_free_attempts;
  uint64_t pinned_free_attempts;
  uint64_t copy_use_release_attempts;
  uint64_t reserved[3];
} RileyCudaTestMemoryFaultStats;
#endif

typedef struct RileyCudaContext RileyCudaContext;
typedef struct RileyCudaStream RileyCudaStream;
typedef struct RileyCudaEvent RileyCudaEvent;
typedef struct RileyCudaSmokeBuffer RileyCudaSmokeBuffer;
typedef struct RileyCudaDeviceBuffer RileyCudaDeviceBuffer;
typedef struct RileyCudaPinnedHostBuffer RileyCudaPinnedHostBuffer;
typedef struct RileyCudaCopy RileyCudaCopy;
typedef struct RileyCudaGemmPlan RileyCudaGemmPlan;
typedef struct RileyCudaFixed37GemmPlan RileyCudaFixed37GemmPlan;
typedef struct RileyCudaHfPrefillAttentionPlan
    RileyCudaHfPrefillAttentionPlan;
typedef struct RileyCudaGraphCapture RileyCudaGraphCapture;
typedef struct RileyCudaGraph RileyCudaGraph;
typedef struct RileyCudaGraphExec RileyCudaGraphExec;
typedef struct RileyCudaGraphLaunch RileyCudaGraphLaunch;

// CUDA's raw capture-mode numeric values are deliberately not part of the
// Riley ABI. The first graph ABI slice admits only thread-local capture; more
// permissive modes require their own ownership and thread-safety review.
typedef uint32_t RileyCudaGraphCaptureMode;

#define RILEY_CUDA_GRAPH_CAPTURE_MODE_INVALID \
  ((RileyCudaGraphCaptureMode)0)
#define RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL \
  ((RileyCudaGraphCaptureMode)1)

// Per-operation graph-capture admission result. A zero-initialized or absent
// query is unknown and must be denied; this enum does not claim that a whole
// stream, context, or library is graph-capture capable.
typedef uint32_t RileyCudaGraphCaptureCapability;

#define RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN \
  ((RileyCudaGraphCaptureCapability)0)
#define RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNSUPPORTED \
  ((RileyCudaGraphCaptureCapability)1)
#define RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_SUPPORTED \
  ((RileyCudaGraphCaptureCapability)2)

// Exact C05 capture operation whose capability is being queried. The zero
// value and unrecognized future values intentionally produce `UNKNOWN` rather
// than inferring admission from a related operation or a whole CUDA context.
typedef uint32_t RileyCudaGraphCaptureOperationKind;

#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_UNKNOWN \
  ((RileyCudaGraphCaptureOperationKind)0)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_FILL_F32 \
  ((RileyCudaGraphCaptureOperationKind)1)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_H2D \
  ((RileyCudaGraphCaptureOperationKind)2)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_SILU_BF16 \
  ((RileyCudaGraphCaptureOperationKind)3)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_GATED_MULTIPLY_BF16 \
  ((RileyCudaGraphCaptureOperationKind)4)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_RESIDUAL_ADD_BF16 \
  ((RileyCudaGraphCaptureOperationKind)5)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_CANONICAL_RMS_NORM_BF16 \
  ((RileyCudaGraphCaptureOperationKind)6)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ARGMAX \
  ((RileyCudaGraphCaptureOperationKind)7)
#define RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ROW_GATHER \
  ((RileyCudaGraphCaptureOperationKind)8)

// Detailed graph lifecycle phase recorded separately from the established
// RileyCudaErrorInfo stage. Unknown future values must never be interpreted as
// a successful or reusable graph state by a caller.
typedef uint32_t RileyCudaGraphStage;

#define RILEY_CUDA_GRAPH_STAGE_NONE ((RileyCudaGraphStage)0)
#define RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN ((RileyCudaGraphStage)1)
#define RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE ((RileyCudaGraphStage)2)
#define RILEY_CUDA_GRAPH_STAGE_CAPTURE_END ((RileyCudaGraphStage)3)
#define RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT ((RileyCudaGraphStage)4)
#define RILEY_CUDA_GRAPH_STAGE_INSTANTIATE ((RileyCudaGraphStage)5)
#define RILEY_CUDA_GRAPH_STAGE_UPDATE ((RileyCudaGraphStage)6)
#define RILEY_CUDA_GRAPH_STAGE_LAUNCH ((RileyCudaGraphStage)7)
#define RILEY_CUDA_GRAPH_STAGE_COMPLETION ((RileyCudaGraphStage)8)
#define RILEY_CUDA_GRAPH_STAGE_CLOSE ((RileyCudaGraphStage)9)
// Copies ordinary host bytes into a graph-retained pinned source immediately
// before one fixed-address H2D graph replay. This is intentionally distinct
// from capture enqueue and CUDA launch: the payload copy is synchronous CPU
// work and must not claim submission or completion evidence.
#define RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE ((RileyCudaGraphStage)10)

// Caller-owned companion metadata for future graph entry points. This is a
// separate fixed-size record rather than a tail extension of RileyCudaErrorInfo
// so existing v1 error buffers remain valid. A non-null output must set
// struct_size to at least sizeof(RileyCudaGraphErrorInfo) and zero every
// reserved field. Native graph calls will initialize this record on both
// success and failure while preserving a larger caller-provided struct_size.
//
// submission_started becomes one only after CUDA launch submission is
// attempted. completion_known becomes one only after a completion boundary is
// observed unambiguously. resource_release_known becomes one only after every
// transient lease cleanup outcome is known. poisoned means the relevant graph,
// graph exec, or stream must not be reused. Zero IDs mean no capture or exec
// was assigned to the operation.
typedef struct RileyCudaGraphErrorInfo {
  uint32_t struct_size;
  RileyCudaGraphStage graph_stage;
  uint64_t capture_id;
  uint64_t exec_id;
  uint8_t submission_started;
  uint8_t completion_known;
  uint8_t resource_release_known;
  uint8_t poisoned;
  uint32_t reserved0;
  uint64_t reserved[3];
} RileyCudaGraphErrorInfo;

// Raw C callers must externally synchronize opaque-handle lifetime: no call
// may begin with a handle while another thread can close that same handle.
// Native active-use guards reject close/reuse after an operation has entered,
// but cannot make a stale raw pointer safe if close races a new call. The safe
// Rust boundary enforces this rule with ownership and exclusive borrows.

typedef int32_t RileyCudaDType;

#define RILEY_CUDA_DTYPE_INVALID ((RileyCudaDType)0)
#define RILEY_CUDA_DTYPE_F32 ((RileyCudaDType)1)
#define RILEY_CUDA_DTYPE_BF16 ((RileyCudaDType)2)
#define RILEY_CUDA_DTYPE_U32 ((RileyCudaDType)3)
#define RILEY_CUDA_DTYPE_U8 ((RileyCudaDType)4)
#define RILEY_CUDA_DTYPE_U16 ((RileyCudaDType)5)

// A borrowed, typed subspan of an opaque device allocation. byte_len is the
// caller-declared accessible capacity from byte_offset, not the allocation's
// total size. All known reserved fields must be zero. Primitive calls validate
// both this capacity and the underlying allocation before pointer arithmetic.
typedef struct RileyCudaBufferSpan {
  uint32_t struct_size;
  RileyCudaDType dtype;
  RileyCudaDeviceBuffer* buffer;
  uint64_t byte_offset;
  uint64_t byte_len;
  uint64_t reserved[2];
} RileyCudaBufferSpan;

#define RILEY_CUDA_EMBEDDING_ERROR_NONE 0u
#define RILEY_CUDA_EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE 1u

// Embedding execution uses a caller-owned device scratch span of exactly this
// record shape and copies the completed record to out_report before returning.
// For code NONE, token_position and token_id are zero. For OOB, they identify
// the lowest invalid token position and its id deterministically.
typedef struct RileyCudaEmbeddingErrorReport {
  uint32_t struct_size;
  uint32_t code;
  uint64_t token_position;
  uint64_t token_id;
  uint64_t reserved;
} RileyCudaEmbeddingErrorReport;

// Every parameter record below is caller-owned for the synchronous call. Set
// struct_size to sizeof(the record) and every known reserved field to zero;
// larger forward-compatible records are accepted only for additive ABI tails.

typedef struct RileyCudaEmbeddingParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan table;
  RileyCudaBufferSpan token_ids;
  RileyCudaBufferSpan output;
  RileyCudaBufferSpan device_error_scratch;
  RileyCudaEmbeddingErrorReport* out_report;
  uint64_t token_count;
  uint64_t vocabulary_size;
  uint64_t hidden_size;
  uint64_t reserved[3];
} RileyCudaEmbeddingParams;

typedef struct RileyCudaRmsNormParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan input;
  RileyCudaBufferSpan weight;
  RileyCudaBufferSpan output;
  uint64_t row_count;
  uint64_t hidden_size;
  float epsilon;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaRmsNormParams;

typedef struct RileyCudaResidualAddParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan left;
  RileyCudaBufferSpan right;
  RileyCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RileyCudaResidualAddParams;

// Fuses the exact BF16/F32 residual storage boundary with the immediately
// following RMSNorm. residual_output stores round(left + right) exactly as the
// standalone residual primitive would. RMSNorm reduces those stored values in
// FP32 and preserves its existing normalized-to-storage boundary before the
// learned weight multiply. The two outputs and weight must not overlap; exact
// residual-output alias with either residual input remains supported.
typedef struct RileyCudaResidualRmsNormParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan left;
  RileyCudaBufferSpan right;
  RileyCudaBufferSpan weight;
  RileyCudaBufferSpan residual_output;
  RileyCudaBufferSpan normalized_output;
  uint64_t row_count;
  uint64_t hidden_size;
  float epsilon;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaResidualRmsNormParams;

// Full-vector log-softmax used by the fixed-contiguous-37-balanced-v1
// calibration profile. Input is BF16 [element_count], output is F32
// [element_count], and the two spans must not overlap. Any NaN input is
// propagated as canonical quiet NaN to every output. A +Inf maximum or an
// all--Inf vector likewise produces all quiet NaNs, matching the literal
// stable-log-softmax expression. With a finite maximum, individual -Inf
// inputs produce -Inf outputs. Max reduction uses CUDA fmaxf signed-zero
// semantics (+0 wins a +/-0 pair).
typedef struct RileyCudaFixed37LogSoftmaxParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan logits;
  RileyCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RileyCudaFixed37LogSoftmaxParams;

// Adds one BF16 [column_count] bias vector to every row of a contiguous BF16
// [row_count, column_count] matrix in place. Each pair is expanded to F32,
// added once, then rounded to BF16 with round-to-nearest-even. column_count
// must be non-zero; row_count may be zero.
typedef struct RileyCudaRowBiasAddInPlaceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan matrix;
  RileyCudaBufferSpan bias;
  uint64_t row_count;
  uint64_t column_count;
  uint64_t reserved[4];
} RileyCudaRowBiasAddInPlaceParams;

typedef struct RileyCudaSiluParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan input;
  RileyCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RileyCudaSiluParams;

// activated_gate is already SiLU-activated. This operation is deliberately a
// plain multiply; SiLU+multiply fusion is outside ABI v1's PR 06 path.
typedef struct RileyCudaGatedMultiplyParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan activated_gate;
  RileyCudaBufferSpan up;
  RileyCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RileyCudaGatedMultiplyParams;

// Cold RoPE-table preparation. angles_cos starts as an F32 row-major angle
// table and is replaced in place with its cosine. sin receives the matching
// F32 sine table. Both spans have element_count logical elements.
typedef struct RileyCudaRopeTableParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan angles_cos;
  RileyCudaBufferSpan sin;
  uint64_t element_count;
  uint64_t reserved[5];
} RileyCudaRopeTableParams;

// Standard non-interleaved Llama RoPE rotates the two contiguous halves of
// rotary_dimension. cos and sin are F32 tables with logical shape
// [table_position_count, rotary_dimension / 2]. The input/output logical shape
// is [token_count, head_count, head_size].
typedef struct RileyCudaRopeParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan input;
  RileyCudaBufferSpan cos;
  RileyCudaBufferSpan sin;
  RileyCudaBufferSpan output;
  uint64_t token_count;
  uint64_t head_count;
  uint64_t head_size;
  uint64_t rotary_dimension;
  uint64_t table_position_count;
  uint64_t position_offset;
  uint64_t reserved[5];
} RileyCudaRopeParams;

// Row-indexed non-interleaved Llama RoPE. positions is U32 [active_row_count]
// and selects an independent cos/sin table row for every dense input row.
// input/output are [active_row_count,head_count,head_size]. The safe Rust
// boundary validates its mirrored host positions before submission; the
// native kernel also bounds-checks every device position and writes a NaN
// rotary row instead of reading outside the tables when raw C metadata is
// malformed. The non-rotary tail remains a bit-preserving copy.
typedef struct RileyCudaIndexedRopeParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan input;
  RileyCudaBufferSpan cos;
  RileyCudaBufferSpan sin;
  RileyCudaBufferSpan positions;
  RileyCudaBufferSpan output;
  uint64_t active_row_count;
  uint64_t head_count;
  uint64_t head_size;
  uint64_t rotary_dimension;
  uint64_t table_position_count;
  uint64_t reserved[4];
} RileyCudaIndexedRopeParams;

// Only BF16<->F32 conversions are accepted by this operation.
typedef struct RileyCudaCastParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan input;
  RileyCudaBufferSpan output;
  uint64_t element_count;
  uint64_t reserved[5];
} RileyCudaCastParams;

// Allocation-free gather from a contiguous row-major input matrix. row_indices
// is U32 [output_row_count], input is [input_row_count,column_count], and
// output is [output_row_count,column_count]. Input/output must have one
// matching F32 or BF16 dtype and may not overlap. The safe Rust boundary
// validates mirrored indices; malformed raw C device indices produce a NaN
// output row without reading beyond input.
typedef struct RileyCudaRowGatherParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan input;
  RileyCudaBufferSpan row_indices;
  RileyCudaBufferSpan output;
  uint64_t input_row_count;
  uint64_t output_row_count;
  uint64_t column_count;
  uint64_t reserved[4];
} RileyCudaRowGatherParams;

#define RILEY_CUDA_BF16_ARGMAX_STATUS_SUCCESS 0u
#define RILEY_CUDA_BF16_ARGMAX_STATUS_NON_FINITE 1u
#define RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID UINT32_MAX

// One deterministic greedy result per BF16 logits row. A successful row holds
// the lowest token id whose finite logit equals the row maximum. If any value
// in the row is NaN or infinity, token_id is INVALID_TOKEN_ID and status is
// NON_FINITE. The record is deliberately two U32 words so callers can use an
// ordinary U32 device span without a representation conversion.
typedef struct RileyCudaBf16ArgmaxResult {
  uint32_t token_id;
  uint32_t status;
} RileyCudaBf16ArgmaxResult;

// Allocation-free deterministic greedy selection over contiguous BF16
// [row_count, vocabulary_size] logits. results is U32 storage for exactly
// row_count RileyCudaBf16ArgmaxResult records. No RNG state is accepted,
// consumed, or produced.
typedef struct RileyCudaBf16ArgmaxParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan logits;
  RileyCudaBufferSpan results;
  uint64_t row_count;
  uint64_t vocabulary_size;
  uint64_t reserved[4];
} RileyCudaBf16ArgmaxParams;

// Correctness-first materialized GQA attention. Query is BF16
// [token_count, query_head_count, head_size], key is BF16
// [token_count, key_value_head_count, head_size], and output is BF16
// [query_head_count, token_count, token_count]. Each query head maps to
// q_head / (query_head_count / key_value_head_count).
typedef struct RileyCudaQkGqaParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key;
  RileyCudaBufferSpan output;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RileyCudaQkGqaParams;

// In-place BF16 scaling followed by an additive causal mask on materialized
// [query_head_count, token_count, token_count] scores. The scaled value is
// rounded to BF16 before the BF16 mask is added. Strictly future positions use
// the finite BF16 minimum bit pattern 0xff7f; allowed positions add +0.
typedef struct RileyCudaScaleCausalMaskParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan scores;
  uint64_t token_count;
  uint64_t query_head_count;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaScaleCausalMaskParams;

// Stable causal softmax in place over the last dimension of BF16 materialized
// [query_head_count, token_count, token_count] scores. Max and sum reductions
// are F32; each resulting probability is rounded to BF16.
typedef struct RileyCudaCausalSoftmaxParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan scores;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t reserved[5];
} RileyCudaCausalSoftmaxParams;

// Materialized BF16 probabilities are [query_head_count, token_count,
// token_count], value is BF16 [token_count, key_value_head_count, head_size],
// and output is BF16 [token_count, query_head_count, head_size]. Accumulation
// is F32 and uses the same GQA head mapping as RileyCudaQkGqaParams.
typedef struct RileyCudaAvGqaParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan probabilities;
  RileyCudaBufferSpan value;
  RileyCudaBufferSpan output;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RileyCudaAvGqaParams;

#define RILEY_CUDA_ATTENTION_MASK_CAUSAL 1u
#define RILEY_CUDA_ATTENTION_MASK_CAUSAL_LOCAL 2u

// Allocation-free online-softmax GQA prefill over dense contiguous BF16
// tensors. Query and output use [batch_count, token_count, query_head_count,
// head_size]; key and value use [batch_count, token_count,
// key_value_head_count, head_size]. The current implementation supports
// head_size=64 and maps each query head to
// q_head / (query_head_count / key_value_head_count).
//
// CAUSAL requires local_window_size=0. It performs three no-HBM score passes:
// serial D-order QK and staged-BF16 scale/mask, logical-key-order maximum and
// denominator, then staged-BF16 probability plus unconditional
// logical-key-order F32 AV. This is byte-exact with the native materialized
// reference, including future masked NaN/+Inf and zero-probability AV behavior.
//
// CAUSAL_LOCAL admits the current token plus at most local_window_size-1
// preceding tokens; a zero local window masks every key and produces an
// all-zero BF16 row from the empty online state. Its first score pass retains
// the online F32 maximum/denominator and its second pass stages normalized
// probabilities to BF16 before logical-key-order AV. NaN scores poison their
// row, +Inf maxima receive equal staged weight, an all-negative-infinity or
// fully masked row remains all-zero, and zero probabilities skip AV. Both mask
// paths require no caller workspace and never materialize a complete [S,S]
// score or probability matrix in HBM.
typedef struct RileyCudaPrefillAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key;
  RileyCudaBufferSpan value;
  RileyCudaBufferSpan output;
  uint64_t batch_count;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t mask_kind;
  uint64_t local_window_size;
  uint64_t reserved[4];
} RileyCudaPrefillAttentionParams;

// Prepared full-causal BF16 prefill matching the Hugging Face eager sequence:
// repeat_kv, strided-batched cuBLASLt QK, staged BF16 scale/mask/softmax, then
// repeat_kv and strided-batched cuBLASLt AV. Inputs and output retain the dense
// BSHD layout. The caller workspace stores one [QH,S,S] BF16 score matrix
// followed by one [QH,S,D] BF16 repeated-K/V scratch and is reused for every
// batch item. max_cublas_workspace_bytes is a cold heuristic cap; the current
// exact contract rejects an algorithm requiring non-zero cuBLASLt workspace.
typedef struct RileyCudaHfPrefillAttentionConfig {
  uint32_t struct_size;
  uint32_t reserved0;
  uint64_t batch_count;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t deterministic;
  uint64_t max_cublas_workspace_bytes;
  uint64_t reserved[4];
} RileyCudaHfPrefillAttentionConfig;

#define RILEY_CUDA_ATTENTION_BACKEND_HF_CUBLASLT 3u

// Immutable algorithm, environment, and memory provenance for both prepared
// matmuls. layout_copy_bytes counts repeat-K plus repeat-V bytes per batch.
typedef struct RileyCudaHfPrefillAttentionPlanInfo {
  uint32_t struct_size;
  uint32_t backend;
  int32_t qk_algorithm_id;
  uint32_t qk_tile_id;
  uint32_t qk_stages_id;
  uint32_t qk_split_k;
  uint32_t qk_reduction_scheme;
  uint32_t qk_cta_swizzling;
  uint32_t qk_custom_option;
  uint32_t qk_reserved0;
  uint64_t qk_workspace_bytes;
  uint64_t qk_numerical_implementation_flags;
  int32_t av_algorithm_id;
  uint32_t av_tile_id;
  uint32_t av_stages_id;
  uint32_t av_split_k;
  uint32_t av_reduction_scheme;
  uint32_t av_cta_swizzling;
  uint32_t av_custom_option;
  uint32_t av_reserved0;
  uint64_t av_workspace_bytes;
  uint64_t av_numerical_implementation_flags;
  uint32_t deterministic;
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  int32_t runtime_version;
  int32_t cublaslt_version;
  uint32_t reserved0;
  uint64_t workspace_bytes;
  uint64_t score_bytes;
  uint64_t repeated_key_value_bytes;
  uint64_t layout_copy_bytes;
  uint64_t batch_count;
  uint64_t token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[2];
} RileyCudaHfPrefillAttentionPlanInfo;

// Copies paired BF16 K/V rows from dense single-request source layout
// [source_token_count, key_value_head_count, head_size] into the contiguous
// cache layout [key_value_head_count, maximum_token_count, head_size]. The
// destination token interval starts at destination_token_start. Source rows
// are bit-preserving; no arithmetic conversion is performed. Every dimension,
// including source_token_count, must be non-zero.
typedef struct RileyCudaKvCacheWriteParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan key_source;
  RileyCudaBufferSpan value_source;
  RileyCudaBufferSpan key_cache;
  RileyCudaBufferSpan value_cache;
  uint64_t source_token_count;
  uint64_t destination_token_start;
  uint64_t maximum_token_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RileyCudaKvCacheWriteParams;

// Correctness-first single-query decode over contiguous BF16 caches. Query
// and output are [query_head_count, head_size]; key_cache and value_cache are
// [key_value_head_count, maximum_token_count, head_size]. The first
// logical_token_count cache positions participate. score_workspace is BF16
// [query_head_count, logical_token_count] and is materialized through four
// stages: QK, scale, stable softmax, and AV. QK and scaling each round to BF16;
// softmax probabilities and the final output also round to BF16. NaN scores
// propagate; positive-infinity maxima share equal probability and an all
// negative-infinity row produces zero output, matching the online-state rules.
typedef struct RileyCudaDecodeAttentionReferenceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key_cache;
  RileyCudaBufferSpan value_cache;
  RileyCudaBufferSpan score_workspace;
  RileyCudaBufferSpan output;
  uint64_t maximum_token_count;
  uint64_t logical_token_count;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaDecodeAttentionReferenceParams;

#define RILEY_CUDA_DECODE_REDUCTION_ASCENDING 1u
#define RILEY_CUDA_DECODE_REDUCTION_DESCENDING 2u
#define RILEY_CUDA_DECODE_PARTIAL_STATE_VERSION 1u

// Partitioned online-softmax decode. The ordinary head_size=64 producer writes
// F32 partial_states with packed logical layout [partial_state_capacity,
// query_head_count, head_size + 2]. Within each packed row element 0 is
// max_score (m), element 1 is exp_sum (l), and elements [2, head_size+2) are
// the unnormalized weighted_value_sum (n). Only the first
// ceil(logical_token_count / tokens_per_partition) partitions are written;
// capacity tail bytes remain untouched. The selected ordered reducer merges
// those states and normalizes exactly once into BF16 output
// [query_head_count, head_size]. The reviewed 9QH/3KVH/D64 v2 hybrid instead
// uses the aligned workspace prefix as BF16 [9,T] scores/probabilities for
// logical T<=32 (at most 576 bytes), then writes the HF-eager-exact output
// directly. T>=33 keeps the packed F32 layout and reducer above unchanged.
// The separate materialized-reference params and v1 entry point are unchanged.
typedef struct RileyCudaDecodeAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key_cache;
  RileyCudaBufferSpan value_cache;
  RileyCudaBufferSpan partial_states;
  RileyCudaBufferSpan output;
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
} RileyCudaDecodeAttentionParams;

// Standalone ordered merge for the packed F32 partial-state ABI above. This
// reducer supports every positive head_size, so PR 10 paged producers can
// reuse it. It reads the first partial_state_count states from a buffer sized
// for partial_state_capacity, merges without normalizing intermediate states,
// then writes BF16 [query_head_count, head_size]. An empty state is encoded as
// (m=-inf, l=0, n=0); well-formed producer states use l>0 otherwise. A zero
// partial_state_count is valid and writes an all-zero output.
typedef struct RileyCudaDecodePartialStateReduceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan partial_states;
  RileyCudaBufferSpan output;
  uint64_t partial_state_count;
  uint64_t partial_state_capacity;
  uint64_t query_head_count;
  uint64_t head_size;
  uint32_t reduction_order;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaDecodePartialStateReduceParams;

#define RILEY_CUDA_PAGED_KV_BLOCK_TABLE_VERSION 1u
#define RILEY_CUDA_PAGED_KV_BLOCK_SIZE 16u
#define RILEY_CUDA_PAGED_KV_METADATA_NONE 0u

// Exact paged-cache address-translation descriptor. block_ids is U32
// [block_count] in logical-block order; entries name physical blocks in
// [0, physical_block_count). valid_tokens is U16 [block_count]. Every block
// except the last contains 16 valid tokens and the last contains
// ((logical_token_count - 1) % 16) + 1. The safe Rust boundary validates the
// mirrored host arrays (including distinct/in-range physical IDs) before this
// device descriptor is submitted. metadata_kind/version must both be zero in
// v1: the exact path has no optional sidecar dependency, while the reserved
// tail leaves an additive extension point.
typedef struct RileyCudaPagedKvBlockTableV1 {
  uint32_t struct_size;
  uint32_t format_version;
  RileyCudaBufferSpan block_ids;
  RileyCudaBufferSpan valid_tokens;
  uint64_t logical_token_count;
  uint64_t block_count;
  uint64_t physical_block_count;
  uint32_t block_size;
  uint32_t metadata_kind;
  uint32_t metadata_version;
  uint32_t reserved0;
  uint64_t reserved[3];
} RileyCudaPagedKvBlockTableV1;

// Bit-preserving scatter from dense BF16 [T,KVH,D] sources into separate BF16
// pools [physical_block_count,KVH,16,D]. destination_token_start is logical;
// the table performs address translation and may contain shuffled physical
// block IDs. The table describes the post-write logical length.
typedef struct RileyCudaPagedKvCacheWriteParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan key_source;
  RileyCudaBufferSpan value_source;
  RileyCudaBufferSpan key_pool;
  RileyCudaBufferSpan value_pool;
  RileyCudaPagedKvBlockTableV1 block_table;
  uint64_t source_token_count;
  uint64_t destination_token_start;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RileyCudaPagedKvCacheWriteParams;

// Four-stage staged-BF16 correctness reference over paged BF16 pools. Scores
// are materialized as BF16 [QH,logical_token_count]. Only logical table order
// affects attention order; physical block numbering is opaque.
typedef struct RileyCudaPagedDecodeAttentionReferenceParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key_pool;
  RileyCudaBufferSpan value_pool;
  RileyCudaBufferSpan score_workspace;
  RileyCudaBufferSpan output;
  RileyCudaPagedKvBlockTableV1 block_table;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaPagedDecodeAttentionReferenceParams;

// Exact D64 paged online producer. Ordinarily one packed F32
// DecodePartialState is produced for each logical 16-token block, then the
// unchanged PR 09 reducer merges block slots in logical order and normalizes
// once. Capacity is the preallocated number of logical block slots. The
// reviewed 9QH/3KVH/D64 v2 hybrid instead uses the aligned workspace prefix as
// BF16 [9,T] scores/probabilities for logical T<=32 (at most 576 bytes) and
// writes the HF-eager-exact output directly. T>=33 preserves the existing
// one-F32-state-per-logical-block layout. The paged materialized-reference v1
// contract remains independent and unchanged.
typedef struct RileyCudaPagedDecodeAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key_pool;
  RileyCudaBufferSpan value_pool;
  RileyCudaBufferSpan partial_states;
  RileyCudaBufferSpan output;
  RileyCudaPagedKvBlockTableV1 block_table;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t partial_state_capacity;
  float scale;
  uint32_t reduction_order;
  uint64_t reserved[4];
} RileyCudaPagedDecodeAttentionParams;

#define RILEY_CUDA_PACKED_BATCH_VERSION 1u

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
typedef struct RileyCudaPackedBatchV1 {
  uint32_t struct_size;
  uint32_t format_version;
  RileyCudaBufferSpan sequence_block_offsets;
  RileyCudaBufferSpan block_ids;
  RileyCudaBufferSpan valid_tokens;
  RileyCudaBufferSpan row_sequence_slots;
  RileyCudaBufferSpan row_positions;
  uint64_t sequence_count;
  uint64_t block_count;
  uint64_t active_row_count;
  uint64_t physical_block_count;
  uint32_t block_size;
  uint32_t reserved0;
  uint64_t reserved[4];
} RileyCudaPackedBatchV1;

// Bit-preserving BF16 scatter from dense active rows [T,KVH,D] into shared
// pools [physical_block_count,KVH,16,D]. Each row uses its packed sequence slot
// and logical position. Invalid raw C metadata makes only that row a no-op.
typedef struct RileyCudaRaggedPagedKvCacheWriteParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan key_source;
  RileyCudaBufferSpan value_source;
  RileyCudaBufferSpan key_pool;
  RileyCudaBufferSpan value_pool;
  RileyCudaPackedBatchV1 batch;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t reserved[4];
} RileyCudaRaggedPagedKvCacheWriteParams;

// Allocation-free ragged causal paged attention. Query is BF16 [T,QH,64],
// output is BF16 [output_row_count,QH,64], pools are BF16
// [physical_block_count,KVH,16,64], and each active row attends logical tokens
// [0,row_positions[row]] in its own CSR sequence. output_row_count must be at
// least T; every padding row [T,output_row_count) is overwritten with zero so
// a prepared fixed-M projection never observes uninitialized bytes. QH must be
// divisible by KVH. Dot products, online softmax state, and value accumulation
// are F32; the final output rounds once to BF16. Invalid raw C metadata produces
// a NaN output row rather than an out-of-bounds access.
typedef struct RileyCudaRaggedPagedAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key_pool;
  RileyCudaBufferSpan value_pool;
  RileyCudaBufferSpan output;
  RileyCudaPackedBatchV1 batch;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t output_row_count;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaRaggedPagedAttentionParams;

// No-HBM fixed37 two-pass form of ragged paged attention. Tensor and packed
// batch layouts match RileyCudaRaggedPagedAttentionParams, while
// head_size is exactly 64 and every active row has logical T=row_position+1
// in [1,maximum_logical_token_count], with maximum_logical_token_count in
// [1,8192]. The maximum sizes one exact dynamic-shared-memory allocation;
// scores and probabilities are never materialized in global memory. Invalid
// device metadata, including a row T above the declared maximum, produces a
// full qNaN row/head without an out-of-bounds access. Padding output rows are
// overwritten with storage-exact zero.
typedef struct RileyCudaFixed37RaggedPagedAttentionParams {
  uint32_t struct_size;
  uint32_t reserved0;
  RileyCudaBufferSpan query;
  RileyCudaBufferSpan key_pool;
  RileyCudaBufferSpan value_pool;
  RileyCudaBufferSpan output;
  RileyCudaPackedBatchV1 batch;
  uint64_t query_head_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint64_t output_row_count;
  uint64_t maximum_logical_token_count;
  float scale;
  uint32_t reserved1;
  uint64_t reserved[4];
} RileyCudaFixed37RaggedPagedAttentionParams;

#define RILEY_CUDA_GEMM_TRANSPOSE_N 0u
#define RILEY_CUDA_GEMM_TRANSPOSE_T 1u
#define RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR 1u
#define RILEY_CUDA_GEMM_EPILOGUE_NONE 0u
#define RILEY_CUDA_GEMM_DETERMINISTIC_REQUIRED 1u
#define RILEY_CUDA_GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K 1u
#define RILEY_CUDA_GEMM_FLAG_ALLOW_INPLACE_SPLIT_K 2u
#define RILEY_CUDA_GEMM_BACKEND_CUBLASLT 1u
#define RILEY_CUDA_GEMM_BACKEND_FIXED37 2u

#define RILEY_CUDA_FIXED37_REDUCTION_VERSION 1u
#define RILEY_CUDA_FIXED37_CHUNK_ELEMENTS 37u
#define RILEY_CUDA_FIXED37_MAX_CHUNK_COUNT 4096u

// PR 06 deliberately exposes one exact dense GEMM contract. The logical
// operation is row-major Y[M,N] = X[M,K] * W[N,K]^T with BF16 X/W/Y and F32
// accumulation. input_transpose must be N, weight_transpose must be T, all
// layouts must be ROW_MAJOR, epilogue must be NONE, and deterministic must be
// DETERMINISTIC_REQUIRED. max_workspace_bytes is a preparation-time cap; the
// selected exact requirement is returned by gemm_plan_info. flags is either
// zero for strict split-K=1/NONE selection, or a bitwise combination of
// GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K and GEMM_FLAG_ALLOW_INPLACE_SPLIT_K for
// the reviewed deterministic split-K extensions. OUTPUT_TYPE stores partials
// in output type for a separate reduction; INPLACE uses output-type storage
// plus workspace counters that guarantee sequentiality. Unknown flags,
// reserved0, and every reserved element must be zero.
typedef struct RileyCudaGemmConfig {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  RileyCudaDType input_dtype;
  RileyCudaDType weight_dtype;
  RileyCudaDType accumulator_dtype;
  RileyCudaDType output_dtype;
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
} RileyCudaGemmConfig;

// Immutable metadata for the algorithm prepared into an opaque GEMM plan.
// IDs are cuBLASLt algorithm configuration values and are meaningful together
// with compute capability and the recorded CUDA Runtime/cuBLASLt versions.
typedef struct RileyCudaGemmAlgorithmInfo {
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
} RileyCudaGemmAlgorithmInfo;

// Immutable metadata for the custom fixed-contiguous-37-balanced-v1 GEMM
// plan. The plan uses no caller workspace. dynamic_shared_memory_bytes is the
// exact per-block scratch used for two alternating arrays of chunk partials.
typedef struct RileyCudaFixed37GemmPlanInfo {
  uint32_t struct_size;
  uint32_t backend;
  uint32_t reduction_version;
  uint32_t chunk_elements;
  RileyCudaDType accumulator_dtype;
  RileyCudaDType output_dtype;
  uint32_t threads_per_block;
  uint32_t deterministic;
  uint64_t dynamic_shared_memory_bytes;
  uint64_t workspace_bytes;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t reserved[3];
} RileyCudaFixed37GemmPlanInfo;

#ifdef __cplusplus
#define RILEY_CUDA_NOEXCEPT noexcept
extern "C" {
#else
#define RILEY_CUDA_NOEXCEPT
#endif

// Compile-time ABI metadata. These functions do not initialize a device.
uint32_t riley_cuda_abi_version(void) RILEY_CUDA_NOEXCEPT;
const char* riley_cuda_build_info(void) RILEY_CUDA_NOEXCEPT;

// Returns the graph-capture admission result for one exact C05 operation.
// This is a pure ABI capability query: it initializes no CUDA runtime or
// context, allocates nothing, and does not inspect a stream or resource. An
// unrecognized operation kind succeeds with `UNKNOWN`, which callers must
// deny; `out_capability` itself is required and initialized to `UNKNOWN`
// before validation.
RileyCudaStatus riley_cuda_graph_capture_query_capability(
    RileyCudaGraphCaptureOperationKind operation,
    RileyCudaGraphCaptureCapability* out_capability,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

RileyCudaStatus riley_cuda_device_count(
    uint32_t* out_count,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_device_properties(
    int32_t ordinal,
    RileyCudaDeviceProperties* out_properties,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Performs one in-process NVML snapshot. It does not create a CUDA context or
// launch CUDA work. On any error the compatible output record is zeroed except
// for struct_size, and a successful NVML initialization is always shut down.
RileyCudaStatus riley_cuda_nvidia_environment_probe(
    RileyCudaNvidiaEnvironmentSnapshot* out_snapshot,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Context is a retained lease on the target device's CUDA primary context.
RileyCudaStatus riley_cuda_context_create(
    int32_t ordinal,
    RileyCudaContext** out_context,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_context_synchronize(
    RileyCudaContext* context,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_context_memory_info(
    RileyCudaContext* context,
    uint64_t* out_free_bytes,
    uint64_t* out_total_bytes,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_context_allocation_stats(
    RileyCudaContext* context,
    RileyCudaAllocationStats* out_stats,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Once primary-context release is attempted, *context is null even if a
// deferred asynchronous error is returned. Validation, poison, live-child, or
// non-zero allocation-accounting failures before the attempt leave it intact.
RileyCudaStatus riley_cuda_context_close(
    RileyCudaContext** context,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Additive safe-wrapper-only ownership transfer. When the calling host thread
// owns a live ThreadLocal graph capture, moves *context into that capture's
// post-end cleanup queue and nulls it. It never changes the retry semantics of
// riley_cuda_context_close for raw C callers.
RileyCudaStatus riley_cuda_context_defer_to_active_capture(
    RileyCudaContext** context,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Streams are explicitly created as non-blocking, non-default streams.
RileyCudaStatus riley_cuda_stream_create(
    RileyCudaContext* context,
    RileyCudaStream** out_stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// A command batch retains the stream and every unique device-buffer/GEMM-plan
// used by supported operations on the owning thread. Those operations enqueue
// without per-operation synchronization; end performs the single completion
// synchronization. Failed or ambiguous completion deliberately retains all
// leases. Nested, cross-thread, query/synchronize/wait/close use is rejected.
RileyCudaStatus riley_cuda_stream_command_batch_begin(
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_stream_command_batch_end(
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins one thread-local CUDA Graph capture and returns a native capture owner
// only when the stream lease, CUDA begin call, and current-context restoration
// are all known to have succeeded. out_capture is required and is initialized
// to null before validation. A non-null capture may also accompany a failing
// begin when CUDA may have entered capture while reporting a prior asynchronous
// error; that owner must be passed to riley_cuda_graph_capture_abort exactly
// once before the stream can be reused. out_graph_error is optional companion
// metadata for the attempted begin.
RileyCudaStatus riley_cuda_graph_capture_begin(
    RileyCudaStream* stream,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Ends and discards an active or invalidated capture without exposing a graph.
// The capture must be ended from the thread that began this thread-local
// capture. Once CUDA end has been attempted, *capture is null even when an
// asynchronous error leaves recovery ambiguous; native then retains the
// owner/stream lease fail-closed rather than permitting a retry. A validation
// failure before the CUDA end attempt leaves *capture unchanged.
RileyCudaStatus riley_cuda_graph_capture_abort(
    RileyCudaGraphCapture** capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins the only C05-5 capture-admitted operation set: repeated fixed-shape
// f32 fills of one caller-owned, preallocated device buffer. The exact buffer,
// element count, final graph storage, and native resource leases are prepared
// before cudaStreamBeginCapture. A successful graph/exec retains the captured
// stream and buffer until graph/exec close; ordinary stream/buffer close and
// use therefore remain busy while the graph exists.
RileyCudaStatus riley_cuda_graph_capture_begin_fill_f32(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* buffer,
    uint64_t element_count,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues one fill node into a capture created by
// riley_cuda_graph_capture_begin_fill_f32. It has no allocation, pointer, or
// shape input: the begin call fixed all graph-visible resource addresses and
// geometry. At least one successful enqueue is required before capture_end.
RileyCudaStatus riley_cuda_graph_capture_enqueue_fill_f32(
    RileyCudaGraphCapture* capture,
    float value,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-7 capture containing exactly one fixed-address, whole-slab H2D
// memcpy node. `source` and `destination` must be live allocations in the
// stream's context, have the same non-zero byte length, and remain permanently
// leased until the resulting graph or exec is closed. No offsets, ranges, or
// dynamic pointers are accepted.
RileyCudaStatus riley_cuda_graph_capture_begin_h2d(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* destination,
    RileyCudaPinnedHostBuffer* source,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole fixed-address H2D node for a capture created by
// riley_cuda_graph_capture_begin_h2d. The captured source and destination
// pointers and exact byte length are immutable; payload staging occurs only
// through riley_cuda_graph_exec_stage_h2d_source after instantiation.
RileyCudaStatus riley_cuda_graph_capture_enqueue_h2d(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-8 capture containing exactly one fixed-address, out-of-place
// BF16 SiLU node. `input` and `output` must be distinct live device
// allocations in the stream's context; `element_count` describes contiguous
// elements from allocation offset zero. Both device allocations remain leased
// until the resulting graph or exec is closed. This intentionally does not
// accept spans, offsets, dynamic dtype, in-place aliasing, or fresh replay
// input.
RileyCudaStatus riley_cuda_graph_capture_begin_silu_bf16(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* output,
    uint64_t element_count,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole BF16 SiLU node for a capture created by
// riley_cuda_graph_capture_begin_silu_bf16. The captured input, output, and
// element count are immutable for the lifetime of the graph.
RileyCudaStatus riley_cuda_graph_capture_enqueue_silu_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-10 capture containing exactly one fixed-address, out-of-place
// BF16 activated-gate multiply node. `activated_gate`, `up`, and `output`
// must be three distinct live device allocations in the stream's context;
// `element_count` describes contiguous elements from allocation offset zero.
// Every device allocation remains leased until the resulting graph or exec is
// closed. This deliberately does not accept spans, offsets, dtype selection,
// in-place aliasing, fresh replay input, or SiLU fusion.
RileyCudaStatus riley_cuda_graph_capture_begin_gated_multiply_bf16(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* activated_gate,
    RileyCudaDeviceBuffer* up,
    RileyCudaDeviceBuffer* output,
    uint64_t element_count,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole BF16 activated-gate multiply node for a capture created
// by riley_cuda_graph_capture_begin_gated_multiply_bf16. The three captured
// allocations and exact element count are immutable for the graph lifetime.
RileyCudaStatus riley_cuda_graph_capture_enqueue_gated_multiply_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-11 capture containing exactly one fixed-address, out-of-place
// BF16 residual-add node. `left`, `right`, and `output` must be three distinct
// live device allocations in the stream's context; `element_count` describes
// contiguous elements from allocation offset zero. Every device allocation
// remains leased until the resulting graph or exec is closed. This deliberately
// does not accept spans, offsets, dtype selection, in-place aliasing, fresh
// replay input, or fused normalization.
RileyCudaStatus riley_cuda_graph_capture_begin_residual_add_bf16(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* left,
    RileyCudaDeviceBuffer* right,
    RileyCudaDeviceBuffer* output,
    uint64_t element_count,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole BF16 residual-add node for a capture created by
// riley_cuda_graph_capture_begin_residual_add_bf16. The three captured
// allocations and exact element count are immutable for the graph lifetime.
RileyCudaStatus riley_cuda_graph_capture_enqueue_residual_add_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-12 capture containing exactly one fixed-address, out-of-place
// canonical BF16 RMSNorm node. `input`, `weight`, and `output` must be three
// distinct live device allocations in the stream's context. `row_count`,
// `hidden_size`, and positive finite `epsilon` are immutable capture-time
// geometry. The operation follows only riley_cuda_rms_norm_execute's generic
// BF16 reduction and storage-rounding contract; it does not cover the
// profile-specific SmolLM2 or Fixed37 variants. No spans, offsets, in-place
// aliasing, dynamic profile selection, fresh replay input, or fusion is
// admitted.
RileyCudaStatus riley_cuda_graph_capture_begin_canonical_rms_norm_bf16(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* weight,
    RileyCudaDeviceBuffer* output,
    uint64_t row_count,
    uint64_t hidden_size,
    float epsilon,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole canonical BF16 RMSNorm node for a capture created by
// riley_cuda_graph_capture_begin_canonical_rms_norm_bf16. All three captured
// allocations and exact geometry are immutable for the graph lifetime.
RileyCudaStatus riley_cuda_graph_capture_enqueue_canonical_rms_norm_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-13 capture containing exactly one fixed-address deterministic
// BF16 argmax node. `logits` is contiguous BF16 `[row_count,
// vocabulary_size]`; `results` is U32 `[row_count, 2]` containing the exact
// token/status records defined by RileyCudaBf16ArgmaxResult. The two live
// allocations must be distinct and remain fixed for the graph lifetime.
// `row_count` must be nonzero and `vocabulary_size` must be in
// 1..=UINT32_MAX. This narrow graph contract preserves the eager primitive's
// lower-token-id finite tie rule and non-finite row status, but admits no
// spans, offsets, fresh logits, row gather, host result handling, sampling,
// executor wiring, or C07 capability evidence.
RileyCudaStatus riley_cuda_graph_capture_begin_bf16_argmax(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* logits,
    RileyCudaDeviceBuffer* results,
    uint64_t row_count,
    uint64_t vocabulary_size,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole deterministic BF16 argmax node for a capture created by
// riley_cuda_graph_capture_begin_bf16_argmax. The captured logits, result
// records, and exact shape remain immutable for the graph lifetime.
RileyCudaStatus riley_cuda_graph_capture_enqueue_bf16_argmax(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Begins a C05-14 capture containing exactly one fixed-address, out-of-place
// BF16 row-gather node. `input` is contiguous BF16 `[input_row_count,
// column_count]`, `row_indices` is contiguous U32 `[output_row_count]`, and
// `output` is contiguous BF16 `[output_row_count, column_count]`. All three
// live allocations must be distinct, share the stream context, and remain
// fixed for the graph lifetime. Every dimension must be nonzero. Raw device
// indices outside input_row_count retain riley_cuda_row_gather_execute's
// per-element BF16 NaN output behavior. This narrow contract accepts no host
// mirror, spans, offsets, H2D/D2H, fresh inputs, node updates, argmax,
// sampling, executor wiring, or C07 capability evidence.
RileyCudaStatus riley_cuda_graph_capture_begin_bf16_row_gather(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* output,
    uint64_t input_row_count,
    uint64_t output_row_count,
    uint64_t column_count,
    RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Enqueues the sole BF16 row-gather node for a capture created by
// riley_cuda_graph_capture_begin_bf16_row_gather. The three captured
// allocations and exact geometry remain immutable for the graph lifetime.
RileyCudaStatus riley_cuda_graph_capture_enqueue_bf16_row_gather(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Ends one prepared fixed-operation capture and transfers its context-child,
// exact stream, device-destination, and (for H2D) pinned-source leases to the
// returned graph. Once CUDA end has been attempted, *capture is null even if
// recovery is ambiguous. A validation failure before that attempt leaves it
// intact.
RileyCudaStatus riley_cuda_graph_capture_end(
    RileyCudaGraphCapture** capture,
    RileyCudaGraph** out_graph,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Instantiation consumes the captured graph on every CUDA instantiate attempt.
// A non-null exec is returned only after native state and context restoration
// are known; ambiguous attempted instantiation is retained fail-closed.
RileyCudaStatus riley_cuda_graph_instantiate(
    RileyCudaGraph** graph,
    RileyCudaGraphExec** out_exec,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Launch is restricted to the exact stream retained by the graph capture.
// A non-null completion owner can accompany a failing launch attempt so the
// caller can settle or deliberately retain the in-flight native lease.
RileyCudaStatus riley_cuda_graph_exec_launch(
    RileyCudaGraphExec* exec,
    RileyCudaStream* stream,
    RileyCudaGraphLaunch** out_launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Stages one exact whole-slab payload into the pinned source retained by an
// H2D graph exec. This special graph-owner operation is the only permitted
// mutable access while the pinned allocation's permanent graph lease is held.
// It performs no CUDA call and is consumed by the next graph-launch attempt;
// a graph H2D exec cannot replay stale staged bytes.
RileyCudaStatus riley_cuda_graph_exec_stage_h2d_source(
    RileyCudaGraphExec* exec,
    RileyCudaPinnedHostBuffer* source,
    const uint8_t* bytes,
    uint64_t byte_len,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Performs the one completion boundary for an in-flight graph launch. After a
// CUDA completion attempt, *launch is null even on an ambiguous error; native
// retains the graph exec and its leases fail-closed in that case.
RileyCudaStatus riley_cuda_graph_launch_complete(
    RileyCudaGraphLaunch** launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// These one-shot closes consume their raw owner once CUDA destruction is
// attempted. Close is rejected before CUDA entry while an exec launch is
// in-flight; a destroy ambiguity retains all leases fail-closed.
RileyCudaStatus riley_cuda_graph_close(
    RileyCudaGraph** graph,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_graph_exec_close(
    RileyCudaGraphExec** exec,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_stream_query(
    RileyCudaStream* stream,
    uint8_t* out_complete,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_stream_synchronize(
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_stream_wait_event(
    RileyCudaStream* stream,
    RileyCudaEvent* event,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Once native destruction is attempted, *stream is null even if a deferred
// asynchronous error is returned; callers must inspect both status and handle.
RileyCudaStatus riley_cuda_stream_close(
    RileyCudaStream** stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// See riley_cuda_context_defer_to_active_capture. The captured stream itself
// has an active use lease and is therefore not transferable through this API.
RileyCudaStatus riley_cuda_stream_defer_to_active_capture(
    RileyCudaStream** stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Events are timing-enabled so elapsed time remains available.
RileyCudaStatus riley_cuda_event_create(
    RileyCudaContext* context,
    RileyCudaEvent** out_event,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_event_record(
    RileyCudaEvent* event,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_event_query(
    RileyCudaEvent* event,
    uint8_t* out_complete,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_event_synchronize(
    RileyCudaEvent* event,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_event_elapsed_ms(
    RileyCudaEvent* start,
    RileyCudaEvent* end,
    float* out_elapsed_ms,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Uses the same single-attempt ownership rule as stream_close.
RileyCudaStatus riley_cuda_event_close(
    RileyCudaEvent** event,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_event_defer_to_active_capture(
    RileyCudaEvent** event,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// General byte-addressed allocations. A zero-byte allocation still returns an
// owned opaque handle and contributes one live allocation with zero live bytes.
RileyCudaStatus riley_cuda_device_buffer_create(
    RileyCudaContext* context,
    uint64_t byte_len,
    RileyCudaDeviceBuffer** out_buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Active copy/primitive uses make close fail with INVALID_STATE before cudaFree.
// Once cudaFree is attempted, the handle follows the single-shot close rule.
// An ambiguous failed free stays logically accounted and keeps a context-child
// lease so allocation stats/context teardown remain fail closed.
RileyCudaStatus riley_cuda_device_buffer_close(
    RileyCudaDeviceBuffer** buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_device_buffer_defer_to_active_capture(
    RileyCudaDeviceBuffer** buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

RileyCudaStatus riley_cuda_pinned_host_buffer_create(
    RileyCudaContext* context,
    uint64_t byte_len,
    RileyCudaPinnedHostBuffer** out_buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Synchronous CPU access is rejected while an async use is active.
RileyCudaStatus riley_cuda_pinned_host_buffer_write(
    RileyCudaPinnedHostBuffer* buffer,
    uint64_t destination_offset,
    const uint8_t* source,
    uint64_t source_len,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_pinned_host_buffer_read(
    RileyCudaPinnedHostBuffer* buffer,
    uint64_t source_offset,
    uint8_t* destination,
    uint64_t destination_len,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Active copy uses make close fail before cudaFreeHost. A free attempt is
// otherwise single-shot even when CUDA reports a deferred earlier error;
// ambiguous failure remains logically live/accounted.
RileyCudaStatus riley_cuda_pinned_host_buffer_close(
    RileyCudaPinnedHostBuffer** buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_pinned_host_buffer_defer_to_active_capture(
    RileyCudaPinnedHostBuffer** buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Non-zero copies return one owning pending token. Submission errors observed
// after cudaMemcpyAsync is attempted are stored in that token and surfaced by
// query/synchronize, preserving all buffer lifetimes until completion. A
// zero-byte copy is a successful no-op and returns *out_copy == NULL.
RileyCudaStatus riley_cuda_copy_h2d_async(
    RileyCudaDeviceBuffer* destination,
    uint64_t destination_offset,
    RileyCudaPinnedHostBuffer* source,
    uint64_t source_offset,
    uint64_t byte_len,
    RileyCudaStream* stream,
    RileyCudaCopy** out_copy,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_copy_d2h_async(
    RileyCudaPinnedHostBuffer* destination,
    uint64_t destination_offset,
    RileyCudaDeviceBuffer* source,
    uint64_t source_offset,
    uint64_t byte_len,
    RileyCudaStream* stream,
    RileyCudaCopy** out_copy,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Enqueues one pinned-host-to-device copy in the command batch currently owned
// by this thread. No standalone RileyCudaCopy token is created and this call
// does not synchronize. The command batch retains both buffers until
// riley_cuda_stream_command_batch_end confirms stream completion. Any
// submission or context-restoration ambiguity therefore remains fail closed
// behind the batch resource ledger.
RileyCudaStatus riley_cuda_command_batch_copy_h2d_async(
    RileyCudaDeviceBuffer* destination,
    uint64_t destination_offset,
    RileyCudaPinnedHostBuffer* source,
    uint64_t source_offset,
    uint64_t byte_len,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// out_complete is 1 when native buffer-use counters have been released, even
// if the returned status reports a deferred submission error.
RileyCudaStatus riley_cuda_copy_query(
    RileyCudaCopy* copy,
    uint8_t* out_complete,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_copy_synchronize(
    RileyCudaCopy* copy,
    uint8_t* out_complete,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// An incomplete token is synchronized before close. It is consumed only after
// completion is confirmed; otherwise the handle and active-use guards remain.
RileyCudaStatus riley_cuda_copy_close(
    RileyCudaCopy** copy,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
// One process-local injector session is bound to one context. Reset clears all
// counters; arm is one-shot. Ambiguous cases intentionally poison/leak their
// subprocess, so callers must isolate every case in a fresh process.
RileyCudaStatus riley_cuda_test_memory_fault_reset(
    RileyCudaContext* context,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_test_memory_fault_arm(
    RileyCudaContext* context,
    uint32_t fault,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_test_memory_fault_stats(
    RileyCudaContext* context,
    RileyCudaTestMemoryFaultStats* out_stats,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
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
RileyCudaStatus riley_cuda_embedding_execute(
    const RileyCudaEmbeddingParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// RMSNorm accumulates sum(x*x), mean, reciprocal square root, and scaling in
// F32 for both accepted storage dtypes. Exact input/output alias is supported;
// weight/output overlap and partial input/output overlap are rejected.
RileyCudaStatus riley_cuda_rms_norm_execute(
    const RileyCudaRmsNormParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Additive byte-exact sibling for the reviewed Hugging Face SmolLM2 BF16
// contract. It accepts only hidden_size=576, row_count<=8192, and epsilon=1e-5
// exactly; unsupported descriptors fail closed instead of falling back.
RileyCudaStatus riley_cuda_hugging_face_smollm2_rms_norm_execute(
    const RileyCudaRmsNormParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Alternate RMSNorm entry point for fixed-contiguous-37-balanced-v1. The
// storage, alias, exceptional-value, and synchronous-completion contract is
// identical to riley_cuda_rms_norm_execute; only the sum-of-squares
// reduction order changes.
RileyCudaStatus riley_cuda_fixed37_rms_norm_execute(
    const RileyCudaRmsNormParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Exact output alias with either input is supported. Partial overlap is not.
RileyCudaStatus riley_cuda_residual_add_execute(
    const RileyCudaResidualAddParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Exact fused equivalent of residual_add followed by RMSNorm. This is an
// additive ABI-v1 entry point; the standalone functions remain the fallback.
RileyCudaStatus riley_cuda_residual_rms_norm_execute(
    const RileyCudaResidualRmsNormParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Fused sibling of the reviewed SmolLM2 path. The residual sum is materialized
// in BF16 before its square enters the Hugging Face reduction topology.
RileyCudaStatus
riley_cuda_hugging_face_smollm2_residual_rms_norm_execute(
    const RileyCudaResidualRmsNormParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Fixed-contiguous-37-balanced-v1 sibling of the exact fused residual plus
// RMSNorm primitive. Residual storage rounding remains unchanged.
RileyCudaStatus riley_cuda_fixed37_residual_rms_norm_execute(
    const RileyCudaResidualRmsNormParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Computes F32 log-softmax from BF16 logits with fixed 37-element max and sum
// chunks followed by the reviewed adjacent balanced tree. element_count must
// be non-zero and no canonical implementation is selected as a fallback.
RileyCudaStatus riley_cuda_fixed37_log_softmax_execute(
    const RileyCudaFixed37LogSoftmaxParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// BF16-only row-wise bias addition. matrix is the sole output and therefore
// is always exact-in-place. bias may not overlap any touched matrix byte.
RileyCudaStatus riley_cuda_row_bias_add_in_place_execute(
    const RileyCudaRowBiasAddInPlaceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Exact input/output alias is supported. silu(x) is evaluated as
// x / (1 + exp(-x)); exceptional values follow CUDA arithmetic.
RileyCudaStatus riley_cuda_silu_execute(
    const RileyCudaSiluParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Computes activated_gate * up. Exact output alias with either input is
// supported; partial overlap is rejected.
RileyCudaStatus riley_cuda_gated_multiply_execute(
    const RileyCudaGatedMultiplyParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Computes CUDA sinf/cosf for an F32 angle table during cold preparation.
// angles_cos is exact-in-place; it must not overlap sin. The operation is
// allocation-free and follows the stream's ordinary completion/command-batch
// contract.
RileyCudaStatus riley_cuda_rope_table_execute(
    const RileyCudaRopeTableParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// cos/sin must be F32. input/output accept one matching F32 or BF16 dtype.
// Exact input/output alias is supported because each rotary pair is owned by a
// single CUDA thread; table/output overlap and partial alias are rejected.
RileyCudaStatus riley_cuda_rope_execute(
    const RileyCudaRopeParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Indexed RoPE has the same dtype, aliasing, synchronous-stream, and
// allocation-free guarantees as riley_cuda_rope_execute.
RileyCudaStatus riley_cuda_indexed_rope_execute(
    const RileyCudaIndexedRopeParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// BF16<->F32 only. F32 NaNs narrow to CUDA's canonical BF16 NaN 0x7fff;
// BF16-to-F32 expansion preserves the source BF16 bits. Any input/output
// overlap is rejected.
RileyCudaStatus riley_cuda_cast_execute(
    const RileyCudaCastParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

RileyCudaStatus riley_cuda_row_gather_execute(
    const RileyCudaRowGatherParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// BF16-only deterministic row argmax. logits and results may not overlap.
// Non-finite logits are reported per row and do not fail the enclosing CUDA
// operation. vocabulary_size must be in 1..=UINT32_MAX.
RileyCudaStatus riley_cuda_bf16_argmax_execute(
    const RileyCudaBf16ArgmaxParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// These four allocation-free calls expose the deliberately materialized PR 07
// reference-attention boundary. All spans must be BF16. Unlike the general PR
// 06 primitives, every attention dimension must be non-zero. QK and AV use F32
// accumulators and round their BF16 outputs once per completed dot product.
RileyCudaStatus riley_cuda_qk_gqa_execute(
    const RileyCudaQkGqaParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_scale_causal_mask_in_place_execute(
    const RileyCudaScaleCausalMaskParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_causal_softmax_in_place_execute(
    const RileyCudaCausalSoftmaxParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_av_gqa_execute(
    const RileyCudaAvGqaParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

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
RileyCudaStatus riley_cuda_fixed37_qk_gqa_execute(
    const RileyCudaQkGqaParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_fixed37_causal_softmax_in_place_execute(
    const RileyCudaCausalSoftmaxParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_fixed37_av_gqa_execute(
    const RileyCudaAvGqaParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Fixed37 no-HBM two-pass prefill. The current implementation requires D=64
// and S<=8192, uses no caller workspace, and has the same dense BSHD and mask
// contract as RileyCudaPrefillAttentionParams except that CAUSAL_LOCAL with
// local_window_size=0 returns NOT_SUPPORTED. Its two score passes reproduce the
// materialized raw-BF16 -> scaled-BF16 -> finite-min-mask-BF16 staging and the
// same fixed37 maximum, denominator, BF16-probability, and AV reduction order.
RileyCudaStatus riley_cuda_fixed37_prefill_attention_execute(
    const RileyCudaPrefillAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Executes the dense online-softmax prefill contract above synchronously on
// stream. Unsupported head dimensions return NOT_SUPPORTED before launching.
RileyCudaStatus riley_cuda_prefill_attention_execute(
    const RileyCudaPrefillAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Additive prepared HF-eager full-causal backend. Creation performs all
// descriptor construction and first-heuristic selection. Execute performs no
// allocation or descriptor query, exclusively borrows all spans, and writes
// direct BSHD output. Close consumes the plan only after descriptor teardown
// and context restoration are confirmed.
RileyCudaStatus riley_cuda_hf_prefill_attention_plan_create(
    RileyCudaContext* context,
    const RileyCudaHfPrefillAttentionConfig* config,
    RileyCudaHfPrefillAttentionPlan** out_plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_hf_prefill_attention_plan_info(
    RileyCudaHfPrefillAttentionPlan* plan,
    RileyCudaHfPrefillAttentionPlanInfo* out_info,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_hf_prefill_attention_plan_execute(
    RileyCudaHfPrefillAttentionPlan* plan,
    const RileyCudaBufferSpan* query,
    const RileyCudaBufferSpan* key,
    const RileyCudaBufferSpan* value,
    const RileyCudaBufferSpan* output,
    const RileyCudaBufferSpan* workspace,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_hf_prefill_attention_plan_close(
    RileyCudaHfPrefillAttentionPlan** plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_hf_prefill_attention_plan_defer_to_active_capture(
    RileyCudaHfPrefillAttentionPlan** plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// These single-request cache/decode calls are allocation-free, exclusively
// borrow every distinct opaque buffer and the explicit stream, and synchronize
// that stream before returning. Writable spans may not overlap any other
// touched span. Cache reads use only the logical prefix but cache spans must
// declare the complete maximum-token strided capacity.
RileyCudaStatus riley_cuda_kv_cache_write_execute(
    const RileyCudaKvCacheWriteParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_decode_attention_reference_execute(
    const RileyCudaDecodeAttentionReferenceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Fixed-contiguous-37 materialized decode reuses the reference descriptor and
// BF16 [QH,T] workspace. Logical D and T are each limited to 151552 elements.
// QK, softmax maximum/denominator, and AV use ascending 37-element F32 left
// folds followed by adjacent balanced-tree merges with odd carry. QK rounds to
// raw BF16, scaling rounds again to BF16, softmax probabilities round to BF16,
// and AV consumes those rounded probabilities. A row containing NaN, with a
// +Inf maximum, or containing only -Inf becomes a complete canonical BF16 qNaN
// row (bits 0x7fff); finite-max -Inf entries have zero probability and 0*Inf in
// AV becomes canonical BF16 qNaN.
RileyCudaStatus riley_cuda_fixed37_decode_attention_reference_execute(
    const RileyCudaDecodeAttentionReferenceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Fixed37 no-HBM two-score-pass decode for D=64 and logical T<=8192. The
// reference descriptor is reused additively, but score_workspace is ignored
// completely and may be a zero/null placeholder. Shared memory holds F32
// exp[T] plus two fixed37 partial arrays; its exact launch size is
// 4*T + 8*max(max(ceil(T/37),2),ceil(D/37)). Numerical staging, reduction
// order, and special-value behavior match the fixed37 materialized sibling.
RileyCudaStatus riley_cuda_fixed37_decode_attention_two_pass_execute(
    const RileyCudaDecodeAttentionReferenceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_decode_attention_execute(
    const RileyCudaDecodeAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_decode_partial_state_reduce_execute(
    const RileyCudaDecodePartialStateReduceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_paged_kv_cache_write_execute(
    const RileyCudaPagedKvCacheWriteParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_paged_decode_attention_reference_execute(
    const RileyCudaPagedDecodeAttentionReferenceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Paged fixed37 materialized decode has the same numerical contract as the
// contiguous fixed37 symbol. Page16 performs address translation only: every T
// reduction chunk remains anchored at logical token zero, independent of page
// or physical-block boundaries and numbering.
RileyCudaStatus
riley_cuda_fixed37_paged_decode_attention_reference_execute(
    const RileyCudaPagedDecodeAttentionReferenceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Paged fixed37 no-HBM two-pass decode has the same D64/T8192 and ignored
// score_workspace contract as its contiguous sibling. Page16 is address
// translation only; fixed37 token chunks remain anchored at logical token 0.
RileyCudaStatus
riley_cuda_fixed37_paged_decode_attention_two_pass_execute(
    const RileyCudaPagedDecodeAttentionReferenceParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_paged_decode_attention_execute(
    const RileyCudaPagedDecodeAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Packed-batch calls are allocation-free and exclusively borrow every distinct
// opaque buffer and the explicit stream. Outside a stream command batch they
// synchronize before returning; inside one, all registered resources remain
// retained until command-batch finish. Writable spans may not overlap any
// other touched span.
RileyCudaStatus riley_cuda_ragged_paged_kv_cache_write_execute(
    const RileyCudaRaggedPagedKvCacheWriteParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_ragged_paged_attention_execute(
    const RileyCudaRaggedPagedAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Executes the same canonical D64 GQA contract while grouping independent
// query-head warps into each CTA. This additive entry point leaves the legacy
// one-warp launch available for rollout control and paired profiling.
RileyCudaStatus riley_cuda_ragged_paged_attention_grouped_heads_execute(
    const RileyCudaRaggedPagedAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// D64/T8192 no-HBM fixed37 execution follows the packed-call lifetime above:
// an active command batch retains all nine real buffers until batch finish.
RileyCudaStatus
riley_cuda_fixed37_ragged_paged_attention_two_pass_execute(
    const RileyCudaFixed37RaggedPagedAttentionParams* params,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Plan creation performs all cuBLASLt descriptor construction and heuristic
// selection. A successful plan owns one context-child lease and is immutable.
RileyCudaStatus riley_cuda_gemm_plan_create(
    RileyCudaContext* context,
    const RileyCudaGemmConfig* config,
    RileyCudaGemmPlan** out_plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Creates an exact-M child plan by copying the opaque algorithm selected by
// anchor_plan, rather than querying a shape-specific heuristic. The child and
// anchor must belong to the same context and have identical GEMM contracts
// except for M and the child workspace cap. Native validates the copied
// algorithm against the child descriptors with cublasLtMatmulAlgoCheck,
// records the child-specific workspace requirement, and returns NOT_SUPPORTED
// without heuristic fallback when that reduction topology cannot execute at
// the requested M. The child retains no pointer to anchor_plan after this
// synchronous call returns.
RileyCudaStatus riley_cuda_gemm_plan_create_anchored(
    RileyCudaContext* context,
    const RileyCudaGemmConfig* config,
    RileyCudaGemmPlan* anchor_plan,
    RileyCudaGemmPlan** out_plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_gemm_plan_info(
    RileyCudaGemmPlan* plan,
    RileyCudaGemmAlgorithmInfo* out_info,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Executes the prepared logical row-major operation without byte reordering:
// cuBLASLt sees column-major TN(W, X, Y). Every span must have exactly the
// prepared byte length (workspace uses the selected requirement), a
// 256-byte-aligned byte_offset, and a handle owned by the plan's context. Any
// overlap among X/W/Y/workspace is rejected. The call exclusively borrows the
// plan, buffers, and explicit stream, synchronizes that same stream, and only
// releases the guards after completion and context restoration are confirmed.
// No allocation, heuristic query, or descriptor creation occurs here.
RileyCudaStatus riley_cuda_gemm_plan_execute(
    RileyCudaGemmPlan* plan,
    const RileyCudaBufferSpan* input,
    const RileyCudaBufferSpan* weight,
    const RileyCudaBufferSpan* output,
    const RileyCudaBufferSpan* workspace,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Active or permanently guarded plans cannot close. Native descriptor
// destruction and context restoration must both complete before *plan is
// consumed and its context-child lease is released.
RileyCudaStatus riley_cuda_gemm_plan_close(
    RileyCudaGemmPlan** plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_gemm_plan_defer_to_active_capture(
    RileyCudaGemmPlan** plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Prepares the custom fixed-contiguous-37-balanced-v1 implementation for the
// same logical BF16/F32 GEMM contract as RileyCudaGemmConfig. The custom
// plan never selects or falls back to cuBLASLt and requires no caller
// workspace.
RileyCudaStatus riley_cuda_fixed37_gemm_plan_create(
    RileyCudaContext* context,
    const RileyCudaGemmConfig* config,
    RileyCudaFixed37GemmPlan** out_plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_fixed37_gemm_plan_info(
    RileyCudaFixed37GemmPlan* plan,
    RileyCudaFixed37GemmPlanInfo* out_info,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_fixed37_gemm_plan_execute(
    RileyCudaFixed37GemmPlan* plan,
    const RileyCudaBufferSpan* input,
    const RileyCudaBufferSpan* weight,
    const RileyCudaBufferSpan* output,
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_fixed37_gemm_plan_close(
    RileyCudaFixed37GemmPlan** plan,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

// Diagnostic-only storage keeps generic tensor allocation outside PR 03.
RileyCudaStatus riley_cuda_smoke_buffer_create(
    RileyCudaContext* context,
    uint64_t element_count,
    RileyCudaSmokeBuffer** out_buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_smoke_fill_launch(
    RileyCudaSmokeBuffer* buffer,
    RileyCudaStream* stream,
    float value,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
RileyCudaStatus riley_cuda_smoke_copy_to_host(
    RileyCudaSmokeBuffer* buffer,
    RileyCudaStream* stream,
    float* host_output,
    uint64_t host_element_capacity,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Once cudaFree is attempted, *buffer is null even if CUDA reports a deferred
// asynchronous error; this prevents a retry from double-freeing the storage.
RileyCudaStatus riley_cuda_smoke_buffer_close(
    RileyCudaSmokeBuffer** buffer,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;
// Intentionally returns a launch-stage error without poisoning the context.
RileyCudaStatus riley_cuda_smoke_invalid_launch(
    RileyCudaStream* stream,
    RileyCudaErrorInfo* error) RILEY_CUDA_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

#undef RILEY_CUDA_NOEXCEPT

#endif  // RILEY_CUDA_H_
