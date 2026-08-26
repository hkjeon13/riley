#ifndef RUSTINFER_CUDA_FFI_INTERNAL_HPP_
#define RUSTINFER_CUDA_FFI_INTERNAL_HPP_

#include "rustinfer_cuda.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

struct RustInferCudaContext {
  RustInferCudaContext(CUdevice selected_device, CUcontext primary_context,
                       int32_t device_ordinal) noexcept
      : device(selected_device),
        context(primary_context),
        ordinal(device_ordinal),
        live_children(0),
        restoration_failed(false),
        device_live_bytes(0),
        device_live_allocations(0),
        pinned_host_live_bytes(0),
        pinned_host_live_allocations(0) {}

  CUdevice device;
  CUcontext context;
  int32_t ordinal;
  std::atomic<uint32_t> live_children;
  std::atomic<bool> restoration_failed;
  std::atomic_flag allocation_stats_lock = ATOMIC_FLAG_INIT;
  std::atomic<uint64_t> device_live_bytes;
  std::atomic<uint64_t> device_live_allocations;
  std::atomic<uint64_t> pinned_host_live_bytes;
  std::atomic<uint64_t> pinned_host_live_allocations;
};

struct RustInferCudaStream {
  // SmolLM2 owns roughly 332 physical weight buffers before activation,
  // cache, metadata, and GEMM-plan handles are counted. Keep a conservative
  // cold 8 KiB pointer ledger per stream; overflow remains fail-closed.
  static constexpr size_t kCommandBatchUseCapacity = 1024;

  RustInferCudaStream(RustInferCudaContext* owning_context,
                      cudaStream_t native_stream) noexcept
      : owner(owning_context),
        stream(native_stream),
        active_uses(0),
        command_batch_owner(nullptr),
        command_batch_use_count(0),
        command_batch_uses{} {}

  RustInferCudaContext* owner;
  cudaStream_t stream;
  // One exclusive asynchronous-use lease covers copies and synchronously
  // completing primitives. A stuck value is an intentional fail-closed leak.
  std::atomic<uint32_t> active_uses;
  // A command batch owns the stream from begin through successful end. Only
  // the owner thread may touch this cold-preallocated ledger. Each entry is
  // the active-use counter of one unique buffer or GEMM plan retained by work
  // enqueued on this stream. A non-null owner after failure is intentional:
  // ambiguous CUDA completion must keep every opaque resource alive.
  std::atomic<const void*> command_batch_owner;
  size_t command_batch_use_count;
  std::atomic<uint32_t>*
      command_batch_uses[kCommandBatchUseCapacity];
};

struct RustInferCudaEvent {
  RustInferCudaContext* owner;
  cudaEvent_t event;
};

struct RustInferCudaSmokeBuffer {
  RustInferCudaContext* owner;
  float* device_data;
  uint64_t element_count;
  bool in_flight;
  cudaStream_t launch_stream;
};

struct RustInferCudaDeviceBuffer {
  RustInferCudaDeviceBuffer(RustInferCudaContext* owning_context,
                            void* allocation, uint64_t allocation_bytes) noexcept
      : owner(owning_context),
        device_data(allocation),
        byte_len(allocation_bytes),
        active_uses(0) {}

  RustInferCudaContext* owner;
  void* device_data;
  uint64_t byte_len;
  std::atomic<uint32_t> active_uses;
};

struct RustInferCudaPinnedHostBuffer {
  RustInferCudaPinnedHostBuffer(RustInferCudaContext* owning_context,
                                void* allocation,
                                uint64_t allocation_bytes) noexcept
      : owner(owning_context),
        host_data(allocation),
        byte_len(allocation_bytes),
        active_uses(0) {}

  RustInferCudaContext* owner;
  void* host_data;
  uint64_t byte_len;
  std::atomic<uint32_t> active_uses;
};

struct RustInferCudaCopy {
  RustInferCudaCopy(RustInferCudaContext* owning_context,
                    RustInferCudaStream* copy_stream,
                    RustInferCudaDeviceBuffer* device_buffer,
                    RustInferCudaPinnedHostBuffer* host_buffer) noexcept
      : owner(owning_context),
        stream(copy_stream),
        device(device_buffer),
        host(host_buffer),
        deferred_status(RUSTINFER_CUDA_STATUS_SUCCESS),
        deferred_error{},
        completed(false) {
    deferred_error.struct_size = sizeof(deferred_error);
  }

  RustInferCudaContext* owner;
  RustInferCudaStream* stream;
  RustInferCudaDeviceBuffer* device;
  RustInferCudaPinnedHostBuffer* host;
  RustInferCudaStatus deferred_status;
  RustInferCudaErrorInfo deferred_error;
  bool completed;
};

namespace rustinfer_cuda_internal {

class AllocationStatsGuard final {
 public:
  explicit AllocationStatsGuard(RustInferCudaContext* context) noexcept
      : lock_(context->allocation_stats_lock) {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  AllocationStatsGuard(const AllocationStatsGuard&) = delete;
  AllocationStatsGuard& operator=(const AllocationStatsGuard&) = delete;
  ~AllocationStatsGuard() noexcept { lock_.clear(std::memory_order_release); }

 private:
  std::atomic_flag& lock_;
};

static_assert(sizeof(RustInferCudaErrorInfo) == 272,
              "RustInferCudaErrorInfo ABI size changed");
static_assert(offsetof(RustInferCudaErrorInfo, message) == 16,
              "RustInferCudaErrorInfo ABI layout changed");
static_assert(sizeof(RustInferCudaDeviceProperties) == 320,
              "RustInferCudaDeviceProperties ABI size changed");
static_assert(offsetof(RustInferCudaDeviceProperties, name) == 64,
              "RustInferCudaDeviceProperties ABI layout changed");
static_assert(sizeof(RustInferCudaAllocationStats) == 40,
              "RustInferCudaAllocationStats ABI size changed");
static_assert(offsetof(RustInferCudaAllocationStats, device_live_bytes) == 8,
              "RustInferCudaAllocationStats ABI layout changed");
static_assert(
    offsetof(RustInferCudaAllocationStats, pinned_host_live_allocations) == 32,
    "RustInferCudaAllocationStats ABI tail layout changed");
static_assert(sizeof(void*) * 8 == RUSTINFER_CUDA_ABI_POINTER_WIDTH,
              "rustinfer CUDA ABI requires 64-bit pointers");
static_assert(sizeof(RustInferCudaDType) == 4,
              "RustInferCudaDType ABI width changed");
static_assert(RUSTINFER_CUDA_DTYPE_F32 == 1 &&
                  RUSTINFER_CUDA_DTYPE_BF16 == 2 &&
                  RUSTINFER_CUDA_DTYPE_U32 == 3 &&
                  RUSTINFER_CUDA_DTYPE_U8 == 4 &&
                  RUSTINFER_CUDA_DTYPE_U16 == 5,
              "RustInferCudaDType ABI discriminants changed");
static_assert(sizeof(RustInferCudaBufferSpan) == 48,
              "RustInferCudaBufferSpan ABI size changed");
static_assert(offsetof(RustInferCudaBufferSpan, buffer) == 8,
              "RustInferCudaBufferSpan ABI handle offset changed");
static_assert(offsetof(RustInferCudaBufferSpan, reserved) == 32,
              "RustInferCudaBufferSpan ABI tail changed");
static_assert(sizeof(RustInferCudaEmbeddingErrorReport) == 32,
              "RustInferCudaEmbeddingErrorReport ABI size changed");
static_assert(offsetof(RustInferCudaEmbeddingErrorReport, token_position) == 8,
              "embedding error report ABI layout changed");
static_assert(sizeof(RustInferCudaEmbeddingParams) == 256,
              "RustInferCudaEmbeddingParams ABI size changed");
static_assert(offsetof(RustInferCudaEmbeddingParams, table) == 8,
              "embedding params ABI first span changed");
static_assert(offsetof(RustInferCudaEmbeddingParams, out_report) == 200,
              "embedding params ABI report offset changed");
static_assert(offsetof(RustInferCudaEmbeddingParams, reserved) == 232,
              "embedding params ABI tail changed");
static_assert(sizeof(RustInferCudaRmsNormParams) == 208,
              "RustInferCudaRmsNormParams ABI size changed");
static_assert(offsetof(RustInferCudaRmsNormParams, epsilon) == 168,
              "RMSNorm params ABI epsilon offset changed");
static_assert(sizeof(RustInferCudaFixed37LogSoftmaxParams) == 152,
              "RustInferCudaFixed37LogSoftmaxParams ABI size changed");
static_assert(offsetof(RustInferCudaFixed37LogSoftmaxParams, logits) == 8,
              "fixed37 log-softmax input layout changed");
static_assert(offsetof(RustInferCudaFixed37LogSoftmaxParams, output) == 56,
              "fixed37 log-softmax output layout changed");
static_assert(
    offsetof(RustInferCudaFixed37LogSoftmaxParams, element_count) == 104,
    "fixed37 log-softmax dimension layout changed");
static_assert(sizeof(RustInferCudaResidualAddParams) == 200,
              "RustInferCudaResidualAddParams ABI size changed");
static_assert(sizeof(RustInferCudaRowBiasAddInPlaceParams) == 152,
              "RustInferCudaRowBiasAddInPlaceParams ABI size changed");
static_assert(offsetof(RustInferCudaRowBiasAddInPlaceParams, matrix) == 8,
              "row-bias params ABI matrix offset changed");
static_assert(offsetof(RustInferCudaRowBiasAddInPlaceParams, row_count) == 104,
              "row-bias params ABI dimensions changed");
static_assert(offsetof(RustInferCudaRowBiasAddInPlaceParams, reserved) == 120,
              "row-bias params ABI tail changed");
static_assert(sizeof(RustInferCudaSiluParams) == 152,
              "RustInferCudaSiluParams ABI size changed");
static_assert(sizeof(RustInferCudaGatedMultiplyParams) == 200,
              "RustInferCudaGatedMultiplyParams ABI size changed");
static_assert(sizeof(RustInferCudaRopeParams) == 288,
              "RustInferCudaRopeParams ABI size changed");
static_assert(offsetof(RustInferCudaRopeParams, token_count) == 200,
              "RoPE params ABI dimension offset changed");
static_assert(sizeof(RustInferCudaCastParams) == 152,
              "RustInferCudaCastParams ABI size changed");
static_assert(sizeof(RustInferCudaQkGqaParams) == 216,
              "RustInferCudaQkGqaParams ABI size changed");
static_assert(offsetof(RustInferCudaQkGqaParams, token_count) == 152,
              "RustInferCudaQkGqaParams ABI layout changed");
static_assert(sizeof(RustInferCudaScaleCausalMaskParams) == 112,
              "RustInferCudaScaleCausalMaskParams ABI size changed");
static_assert(offsetof(RustInferCudaScaleCausalMaskParams, scale) == 72,
              "RustInferCudaScaleCausalMaskParams ABI layout changed");
static_assert(sizeof(RustInferCudaCausalSoftmaxParams) == 112,
              "RustInferCudaCausalSoftmaxParams ABI size changed");
static_assert(offsetof(RustInferCudaCausalSoftmaxParams, reserved) == 72,
              "RustInferCudaCausalSoftmaxParams ABI layout changed");
static_assert(sizeof(RustInferCudaAvGqaParams) == 216,
              "RustInferCudaAvGqaParams ABI size changed");
static_assert(offsetof(RustInferCudaAvGqaParams, token_count) == 152,
              "RustInferCudaAvGqaParams ABI layout changed");
static_assert(sizeof(RustInferCudaKvCacheWriteParams) == 272,
              "RustInferCudaKvCacheWriteParams ABI size changed");
static_assert(offsetof(RustInferCudaKvCacheWriteParams, key_source) == 8,
              "KV cache write source ABI layout changed");
static_assert(offsetof(RustInferCudaKvCacheWriteParams, key_cache) == 104,
              "KV cache write destination ABI layout changed");
static_assert(
    offsetof(RustInferCudaKvCacheWriteParams, source_token_count) == 200,
    "KV cache write dimension ABI layout changed");
static_assert(offsetof(RustInferCudaKvCacheWriteParams, reserved) == 240,
              "KV cache write ABI tail changed");
static_assert(sizeof(RustInferCudaDecodeAttentionReferenceParams) == 328,
              "RustInferCudaDecodeAttentionReferenceParams ABI size changed");
static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, query) == 8,
    "decode reference query ABI layout changed");
static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, output) == 200,
    "decode reference output ABI layout changed");
static_assert(offsetof(RustInferCudaDecodeAttentionReferenceParams,
                       maximum_token_count) == 248,
              "decode reference dimension ABI layout changed");
static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, scale) == 288,
    "decode reference scale ABI layout changed");
static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, reserved) == 296,
    "decode reference ABI tail changed");
static_assert(RUSTINFER_CUDA_DECODE_PARTIAL_STATE_VERSION == 1 &&
                  RUSTINFER_CUDA_DECODE_REDUCTION_ASCENDING == 1 &&
                  RUSTINFER_CUDA_DECODE_REDUCTION_DESCENDING == 2,
              "decode partial-state ABI constants changed");
static_assert(sizeof(RustInferCudaDecodeAttentionParams) == 344,
              "RustInferCudaDecodeAttentionParams ABI size changed");
static_assert(offsetof(RustInferCudaDecodeAttentionParams, query) == 8,
              "decode attention query ABI layout changed");
static_assert(offsetof(RustInferCudaDecodeAttentionParams, output) == 200,
              "decode attention output ABI layout changed");
static_assert(
    offsetof(RustInferCudaDecodeAttentionParams, maximum_token_count) == 248,
    "decode attention dimension ABI layout changed");
static_assert(offsetof(RustInferCudaDecodeAttentionParams, scale) == 304,
              "decode attention scale ABI layout changed");
static_assert(offsetof(RustInferCudaDecodeAttentionParams, reserved) == 312,
              "decode attention ABI tail changed");
static_assert(sizeof(RustInferCudaDecodePartialStateReduceParams) == 176,
              "RustInferCudaDecodePartialStateReduceParams ABI size changed");
static_assert(
    offsetof(RustInferCudaDecodePartialStateReduceParams, partial_states) == 8,
    "decode reducer partial-state ABI layout changed");
static_assert(
    offsetof(RustInferCudaDecodePartialStateReduceParams,
             partial_state_count) == 104,
    "decode reducer dimension ABI layout changed");
static_assert(offsetof(RustInferCudaDecodePartialStateReduceParams,
                       reduction_order) == 136,
              "decode reducer order ABI layout changed");
static_assert(
    offsetof(RustInferCudaDecodePartialStateReduceParams, reserved) == 144,
    "decode reducer ABI tail changed");
static_assert(sizeof(RustInferCudaGemmConfig) == 112,
              "RustInferCudaGemmConfig ABI size changed");
static_assert(RUSTINFER_CUDA_GEMM_TRANSPOSE_N == 0 &&
                  RUSTINFER_CUDA_GEMM_TRANSPOSE_T == 1 &&
                  RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR == 1 &&
                  RUSTINFER_CUDA_GEMM_EPILOGUE_NONE == 0 &&
                  RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED == 1 &&
                  RUSTINFER_CUDA_GEMM_BACKEND_CUBLASLT == 1 &&
                  RUSTINFER_CUDA_GEMM_BACKEND_FIXED37 == 2 &&
                  RUSTINFER_CUDA_FIXED37_REDUCTION_VERSION == 1 &&
                  RUSTINFER_CUDA_FIXED37_CHUNK_ELEMENTS == 37 &&
                  RUSTINFER_CUDA_FIXED37_MAX_CHUNK_COUNT == 4096,
              "GEMM ABI discriminants changed");
static_assert(offsetof(RustInferCudaGemmConfig, m) == 8,
              "GEMM config dimension layout changed");
static_assert(offsetof(RustInferCudaGemmConfig, input_dtype) == 32,
              "GEMM config dtype layout changed");
static_assert(offsetof(RustInferCudaGemmConfig, max_workspace_bytes) == 80,
              "GEMM config workspace layout changed");
static_assert(sizeof(RustInferCudaGemmAlgorithmInfo) == 112,
              "RustInferCudaGemmAlgorithmInfo ABI size changed");
static_assert(offsetof(RustInferCudaGemmAlgorithmInfo, workspace_bytes) == 40,
              "GEMM algorithm workspace layout changed");
static_assert(
    offsetof(RustInferCudaGemmAlgorithmInfo,
             numerical_implementation_flags) == 48,
    "GEMM algorithm numerical metadata layout changed");
static_assert(offsetof(RustInferCudaGemmAlgorithmInfo, m) == 72,
              "GEMM algorithm dimension layout changed");
static_assert(sizeof(RustInferCudaFixed37GemmPlanInfo) == 96,
              "RustInferCudaFixed37GemmPlanInfo ABI size changed");
static_assert(
    offsetof(RustInferCudaFixed37GemmPlanInfo,
             dynamic_shared_memory_bytes) == 32,
    "fixed37 GEMM plan shared-memory layout changed");
static_assert(offsetof(RustInferCudaFixed37GemmPlanInfo, m) == 48,
              "fixed37 GEMM plan dimension layout changed");
static_assert(offsetof(RustInferCudaFixed37GemmPlanInfo, reserved) == 72,
              "fixed37 GEMM plan tail layout changed");

inline void clear_error(RustInferCudaErrorInfo* error) noexcept {
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  std::memset(error, 0, sizeof(*error));
  error->struct_size = sizeof(*error);
}

inline RustInferCudaStatus set_error(RustInferCudaErrorInfo* error,
                                     RustInferCudaStatus status,
                                     int32_t native_code, uint32_t domain,
                                     uint32_t stage, const char* operation,
                                     const char* detail) noexcept {
  if (error != nullptr && error->struct_size >= sizeof(*error)) {
    const uint32_t struct_size = error->struct_size;
    std::memset(error, 0, sizeof(*error));
    error->struct_size = struct_size;
    error->native_code = native_code;
    error->domain = domain;
    error->stage = stage;
    std::snprintf(error->message, sizeof(error->message), "%s: %s", operation,
                  detail == nullptr ? "unknown error" : detail);
  }
  return status;
}

inline RustInferCudaStatus validation_error(RustInferCudaErrorInfo* error,
                                            RustInferCudaStatus status,
                                            uint32_t stage,
                                            const char* operation,
                                            const char* detail) noexcept {
  return set_error(error, status, 0, RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION,
                   stage, operation, detail);
}

inline RustInferCudaStatus internal_error(RustInferCudaErrorInfo* error,
                                          uint32_t stage,
                                          const char* operation,
                                          const char* detail) noexcept {
  return set_error(error, RUSTINFER_CUDA_STATUS_INTERNAL_ERROR, 0,
                   RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL, stage, operation,
                   detail);
}

inline RustInferCudaStatus driver_error(CUresult result,
                                        RustInferCudaErrorInfo* error,
                                        uint32_t stage,
                                        const char* operation) noexcept {
  if (result == CUDA_SUCCESS) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  const char* detail = nullptr;
  (void)cuGetErrorString(result, &detail);
  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_DRIVER_ERROR;
  if (result == CUDA_ERROR_INVALID_DEVICE) {
    status = RUSTINFER_CUDA_STATUS_INVALID_DEVICE;
  } else if (result == CUDA_ERROR_INVALID_VALUE) {
    status = RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    status = RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY;
  } else if (result == CUDA_ERROR_NOT_READY) {
    status = RUSTINFER_CUDA_STATUS_NOT_READY;
  }
  return set_error(error, status, static_cast<int32_t>(result),
                   RUSTINFER_CUDA_ERROR_DOMAIN_DRIVER, stage, operation,
                   detail);
}

inline RustInferCudaStatus runtime_error(cudaError_t result,
                                         RustInferCudaErrorInfo* error,
                                         uint32_t stage,
                                         const char* operation) noexcept {
  if (result == cudaSuccess) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_RUNTIME_ERROR;
  if (result == cudaErrorInvalidDevice) {
    status = RUSTINFER_CUDA_STATUS_INVALID_DEVICE;
  } else if (result == cudaErrorInvalidValue ||
             result == cudaErrorInvalidConfiguration) {
    status = RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == cudaErrorMemoryAllocation) {
    status = RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY;
  } else if (result == cudaErrorNotReady) {
    status = RUSTINFER_CUDA_STATUS_NOT_READY;
  }
  return set_error(error, status, static_cast<int32_t>(result),
                   RUSTINFER_CUDA_ERROR_DOMAIN_RUNTIME, stage, operation,
                   cudaGetErrorString(result));
}

class CurrentContext final {
 public:
  explicit CurrentContext(RustInferCudaContext* context) noexcept
      : context_(context), previous_(nullptr), active_(false) {}
  CurrentContext(const CurrentContext&) = delete;
  CurrentContext& operator=(const CurrentContext&) = delete;

  ~CurrentContext() noexcept {
    if (active_) {
      CUcontext popped = nullptr;
      if (cuCtxPopCurrent(&popped) == CUDA_SUCCESS &&
          popped == context_->context) {
        active_ = false;
      } else {
        poison_context();
        reconcile_after_uncertain_pop();
      }
    }
  }

  bool active() const noexcept { return active_; }

  RustInferCudaStatus enter(RustInferCudaErrorInfo* error, uint32_t stage,
                            const char* operation) noexcept {
    if (context_ == nullptr) {
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              stage, operation, "context is null");
    }
    if (context_->restoration_failed.load(std::memory_order_acquire)) {
      return validation_error(
          error, RUSTINFER_CUDA_STATUS_INVALID_STATE, stage, operation,
          "a prior CUDA context-stack restoration failed");
    }

    const CUresult snapshot_result = cuCtxGetCurrent(&previous_);
    if (snapshot_result != CUDA_SUCCESS) {
      poison_context();
      return driver_error(snapshot_result, error, stage, operation);
    }
    if (previous_ == context_->context) {
      // The caller already made this primary context current. Borrow it for
      // this call and do not disturb the caller's stack on leave.
      return RUSTINFER_CUDA_STATUS_SUCCESS;
    }

    const CUresult result = cuCtxPushCurrent(context_->context);
    if (result != CUDA_SUCCESS) {
      // Driver context APIs may surface a prior asynchronous error after doing
      // their own side effect. Re-observe current state before deciding whether
      // a pop is owed. If observation itself is ambiguous, poison the lease and
      // leave it retained rather than pop or release uncertain ownership.
      CUcontext observed = nullptr;
      if (cuCtxGetCurrent(&observed) != CUDA_SUCCESS) {
        poison_context();
      } else if (observed == context_->context) {
        active_ = true;
      } else if (observed != previous_) {
        poison_context();
      }
      return driver_error(result, error, stage, operation);
    }
    active_ = true;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  RustInferCudaStatus leave(RustInferCudaStatus operation_status,
                            RustInferCudaErrorInfo* error, uint32_t stage,
                            const char* operation) noexcept {
    if (!active_) {
      return operation_status;
    }
    CUcontext popped = nullptr;
    const CUresult result = cuCtxPopCurrent(&popped);
    if (result != CUDA_SUCCESS) {
      poison_context();
      reconcile_after_uncertain_pop();
      // NOT_READY is a successful non-blocking observation at the safe Rust
      // boundary, so a context-stack restoration failure must take precedence
      // over it instead of being collapsed to Ok(false). Preserve only genuine
      // operation failures that already carry the more relevant diagnostic.
      if (operation_status != RUSTINFER_CUDA_STATUS_SUCCESS &&
          operation_status != RUSTINFER_CUDA_STATUS_NOT_READY) {
        return operation_status;
      }
      return driver_error(result, error, stage, operation);
    }
    if (popped != context_->context) {
      poison_context();
      reconcile_after_uncertain_pop();
      if (operation_status != RUSTINFER_CUDA_STATUS_SUCCESS &&
          operation_status != RUSTINFER_CUDA_STATUS_NOT_READY) {
        return operation_status;
      }
      return internal_error(error, stage, operation,
                            "CUDA context stack returned a different context");
    }
    active_ = false;
    if (operation_status != RUSTINFER_CUDA_STATUS_SUCCESS) {
      return operation_status;
    }
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

 private:
  void poison_context() noexcept {
    if (context_ != nullptr) {
      context_->restoration_failed.store(true, std::memory_order_release);
    }
  }

  void reconcile_after_uncertain_pop() noexcept {
    CUcontext observed = nullptr;
    if (cuCtxGetCurrent(&observed) != CUDA_SUCCESS) {
      // State cannot be identified. Do not risk popping a caller-owned context;
      // the poison bit prevents release of this primary-context lease.
      active_ = false;
    } else if (observed == previous_) {
      // Pop had its side effect and only reported a deferred earlier error.
      active_ = false;
    } else if (observed == context_->context) {
      // The target remains current, so an explicit caller-side retry is safe.
      active_ = true;
    } else {
      active_ = false;
    }
  }

  RustInferCudaContext* context_;
  CUcontext previous_;
  bool active_;
};

inline bool same_context(const RustInferCudaContext* left,
                         const RustInferCudaContext* right) noexcept {
  return left != nullptr && left == right;
}

inline bool try_acquire_exclusive_use(std::atomic<uint32_t>& active) noexcept {
  uint32_t expected = 0;
  return active.compare_exchange_strong(expected, 1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
}

inline bool release_exclusive_use(std::atomic<uint32_t>& active) noexcept {
  uint32_t expected = 1;
  return active.compare_exchange_strong(expected, 0,
                                        std::memory_order_release,
                                        std::memory_order_relaxed);
}

// The address of this thread-local byte is a process-unique, allocation-free
// ownership token. Unlike std::thread::id it can be published atomically and
// compared by native entry points without racing a non-atomic object.
inline const void* command_batch_thread_token() noexcept {
  static thread_local const uint8_t token = 0;
  return &token;
}

inline bool command_batch_is_active(
    const RustInferCudaStream* stream) noexcept {
  return stream != nullptr &&
         stream->command_batch_owner.load(std::memory_order_acquire) !=
             nullptr;
}

inline bool command_batch_is_owned_by_current_thread(
    const RustInferCudaStream* stream) noexcept {
  return stream != nullptr &&
         stream->command_batch_owner.load(std::memory_order_acquire) ==
             command_batch_thread_token();
}

// Registers a resource lease in the active command batch. Repeated use of the
// same counter is reentrant only for the owning stream/thread and consumes no
// additional ledger entry. Capacity is checked before acquisition, so the hot
// path never allocates and a full ledger cannot strand an unrecorded lease.
inline RustInferCudaStatus command_batch_register_use(
    RustInferCudaStream* stream, std::atomic<uint32_t>* active,
    RustInferCudaErrorInfo* error, const char* operation,
    const char* busy_detail) noexcept {
  if (stream == nullptr || active == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "command-batch stream or resource is null");
  }
  if (!command_batch_is_owned_by_current_thread(stream)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "an active stream command batch is owned by another thread");
  }
  for (size_t index = 0; index < stream->command_batch_use_count; ++index) {
    if (stream->command_batch_uses[index] == active) {
      return RUSTINFER_CUDA_STATUS_SUCCESS;
    }
  }
  if (stream->command_batch_use_count ==
      RustInferCudaStream::kCommandBatchUseCapacity) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "stream command-batch resource ledger capacity was exceeded");
  }
  if (!try_acquire_exclusive_use(*active)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            busy_detail);
  }
  stream->command_batch_uses[stream->command_batch_use_count++] = active;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

inline bool retain_child(RustInferCudaContext* context) noexcept {
  if (context == nullptr) {
    return false;
  }
  uint32_t current = context->live_children.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<uint32_t>::max()) {
    if (context->live_children.compare_exchange_weak(
            current, current + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

inline bool release_child(RustInferCudaContext* context) noexcept {
  if (context == nullptr) {
    return false;
  }
  uint32_t current = context->live_children.load(std::memory_order_relaxed);
  while (current != 0) {
    if (context->live_children.compare_exchange_weak(
            current, current - 1, std::memory_order_release,
            std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

}  // namespace rustinfer_cuda_internal

#endif  // RUSTINFER_CUDA_FFI_INTERNAL_HPP_
