#ifndef RUSTINFER_CUDA_H_
#define RUSTINFER_CUDA_H_

#include <stdint.h>

#define RUSTINFER_CUDA_ABI_VERSION 1
#define RUSTINFER_CUDA_ERROR_MESSAGE_CAPACITY 256
#define RUSTINFER_CUDA_DEVICE_NAME_CAPACITY 256

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

#define RUSTINFER_CUDA_ERROR_DOMAIN_NONE 0u
#define RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION 1u
#define RUSTINFER_CUDA_ERROR_DOMAIN_DRIVER 2u
#define RUSTINFER_CUDA_ERROR_DOMAIN_RUNTIME 3u
#define RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL 4u

#define RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE 1u
#define RUSTINFER_CUDA_ERROR_STAGE_VALIDATION 2u
#define RUSTINFER_CUDA_ERROR_STAGE_CREATE 3u
#define RUSTINFER_CUDA_ERROR_STAGE_LAUNCH 4u
#define RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE 5u
#define RUSTINFER_CUDA_ERROR_STAGE_QUERY 6u
#define RUSTINFER_CUDA_ERROR_STAGE_RECORD 7u
#define RUSTINFER_CUDA_ERROR_STAGE_COPY 8u
#define RUSTINFER_CUDA_ERROR_STAGE_CLOSE 9u

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
// Active copy tokens make close fail with INVALID_STATE before cudaFree.
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
// Synchronous CPU access is rejected while an async copy token is active.
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
// Active copy tokens make close fail before cudaFreeHost. A free attempt is
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
