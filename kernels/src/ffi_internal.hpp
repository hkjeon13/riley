#ifndef RILEY_CUDA_FFI_INTERNAL_HPP_
#define RILEY_CUDA_FFI_INTERNAL_HPP_

#include "riley_cuda.h"

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

struct RileyCudaContext {
  RileyCudaContext(CUdevice selected_device, CUcontext primary_context,
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

struct RileyCudaStream {
  // SmolLM2 owns roughly 332 physical weight buffers before activation,
  // cache, metadata, and GEMM-plan handles are counted. Keep a conservative
  // cold 8 KiB pointer ledger per stream; overflow remains fail-closed.
  static constexpr size_t kCommandBatchUseCapacity = 1024;

  RileyCudaStream(RileyCudaContext* owning_context,
                      cudaStream_t native_stream) noexcept
      : owner(owning_context),
        stream(native_stream),
        active_uses(0),
        command_batch_owner(nullptr),
        command_batch_use_count(0),
        command_batch_uses{} {}

  RileyCudaContext* owner;
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

struct RileyCudaEvent {
  RileyCudaContext* owner;
  cudaEvent_t event;
};

struct RileyCudaSmokeBuffer {
  RileyCudaContext* owner;
  float* device_data;
  uint64_t element_count;
  bool in_flight;
  cudaStream_t launch_stream;
};

struct RileyCudaDeviceBuffer {
  RileyCudaDeviceBuffer(RileyCudaContext* owning_context,
                            void* allocation, uint64_t allocation_bytes) noexcept
      : owner(owning_context),
        device_data(allocation),
        byte_len(allocation_bytes),
        active_uses(0) {}

  RileyCudaContext* owner;
  void* device_data;
  uint64_t byte_len;
  std::atomic<uint32_t> active_uses;
};

struct RileyCudaPinnedHostBuffer {
  RileyCudaPinnedHostBuffer(RileyCudaContext* owning_context,
                                void* allocation,
                                uint64_t allocation_bytes) noexcept
      : owner(owning_context),
        host_data(allocation),
        byte_len(allocation_bytes),
        active_uses(0) {}

  RileyCudaContext* owner;
  void* host_data;
  uint64_t byte_len;
  std::atomic<uint32_t> active_uses;
};

struct RileyCudaCopy {
  RileyCudaCopy(RileyCudaContext* owning_context,
                    RileyCudaStream* copy_stream,
                    RileyCudaDeviceBuffer* device_buffer,
                    RileyCudaPinnedHostBuffer* host_buffer) noexcept
      : owner(owning_context),
        stream(copy_stream),
        device(device_buffer),
        host(host_buffer),
        deferred_status(RILEY_CUDA_STATUS_SUCCESS),
        deferred_error{},
        completed(false) {
    deferred_error.struct_size = sizeof(deferred_error);
  }

  RileyCudaContext* owner;
  RileyCudaStream* stream;
  RileyCudaDeviceBuffer* device;
  RileyCudaPinnedHostBuffer* host;
  RileyCudaStatus deferred_status;
  RileyCudaErrorInfo deferred_error;
  bool completed;
};

namespace riley_cuda_internal {

class AllocationStatsGuard final {
 public:
  explicit AllocationStatsGuard(RileyCudaContext* context) noexcept
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

static_assert(sizeof(RileyCudaErrorInfo) == 272,
              "RileyCudaErrorInfo ABI size changed");
static_assert(offsetof(RileyCudaErrorInfo, message) == 16,
              "RileyCudaErrorInfo ABI layout changed");
static_assert(sizeof(RileyCudaDeviceProperties) == 320,
              "RileyCudaDeviceProperties ABI size changed");
static_assert(offsetof(RileyCudaDeviceProperties, name) == 64,
              "RileyCudaDeviceProperties ABI layout changed");
static_assert(sizeof(RileyCudaNvidiaDeviceSnapshot) == 320,
              "RileyCudaNvidiaDeviceSnapshot ABI size changed");
static_assert(offsetof(RileyCudaNvidiaDeviceSnapshot, name) == 64,
              "RileyCudaNvidiaDeviceSnapshot ABI layout changed");
static_assert(sizeof(RileyCudaNvidiaEnvironmentSnapshot) == 10352,
              "RileyCudaNvidiaEnvironmentSnapshot ABI size changed");
static_assert(
    offsetof(RileyCudaNvidiaEnvironmentSnapshot, driver_version) == 32,
    "RileyCudaNvidiaEnvironmentSnapshot driver layout changed");
static_assert(offsetof(RileyCudaNvidiaEnvironmentSnapshot, devices) == 112,
              "RileyCudaNvidiaEnvironmentSnapshot device layout changed");
static_assert(RILEY_CUDA_NVIDIA_PERSISTENCE_DISABLED == 0 &&
                  RILEY_CUDA_NVIDIA_PERSISTENCE_ENABLED == 1 &&
                  RILEY_CUDA_ERROR_DOMAIN_NVML == 6,
              "NVML ABI discriminants changed");
static_assert(sizeof(RileyCudaAllocationStats) == 40,
              "RileyCudaAllocationStats ABI size changed");
static_assert(offsetof(RileyCudaAllocationStats, device_live_bytes) == 8,
              "RileyCudaAllocationStats ABI layout changed");
static_assert(
    offsetof(RileyCudaAllocationStats, pinned_host_live_allocations) == 32,
    "RileyCudaAllocationStats ABI tail layout changed");
static_assert(sizeof(void*) * 8 == RILEY_CUDA_ABI_POINTER_WIDTH,
              "riley CUDA ABI requires 64-bit pointers");
static_assert(sizeof(RileyCudaDType) == 4,
              "RileyCudaDType ABI width changed");
static_assert(RILEY_CUDA_DTYPE_F32 == 1 &&
                  RILEY_CUDA_DTYPE_BF16 == 2 &&
                  RILEY_CUDA_DTYPE_U32 == 3 &&
                  RILEY_CUDA_DTYPE_U8 == 4 &&
                  RILEY_CUDA_DTYPE_U16 == 5,
              "RileyCudaDType ABI discriminants changed");
static_assert(sizeof(RileyCudaBufferSpan) == 48,
              "RileyCudaBufferSpan ABI size changed");
static_assert(offsetof(RileyCudaBufferSpan, buffer) == 8,
              "RileyCudaBufferSpan ABI handle offset changed");
static_assert(offsetof(RileyCudaBufferSpan, reserved) == 32,
              "RileyCudaBufferSpan ABI tail changed");
static_assert(sizeof(RileyCudaQkGqaParams) == 216,
              "QK GQA params ABI size changed");
static_assert(sizeof(RileyCudaCausalSoftmaxParams) == 112,
              "causal softmax params ABI size changed");
static_assert(sizeof(RileyCudaAvGqaParams) == 216,
              "AV GQA params ABI size changed");
static_assert(sizeof(RileyCudaPrefillAttentionParams) == 288,
              "prefill attention params ABI size changed");
static_assert(sizeof(RileyCudaHfPrefillAttentionConfig) == 96,
              "HF prefill attention config ABI size changed");
static_assert(offsetof(RileyCudaHfPrefillAttentionConfig, batch_count) == 8,
              "HF prefill attention config dimension layout changed");
static_assert(offsetof(RileyCudaHfPrefillAttentionConfig,
                       max_cublas_workspace_bytes) == 56,
              "HF prefill attention config workspace layout changed");
static_assert(sizeof(RileyCudaHfPrefillAttentionPlanInfo) == 216,
              "HF prefill attention plan info ABI size changed");
static_assert(offsetof(RileyCudaHfPrefillAttentionPlanInfo,
                       qk_workspace_bytes) == 40,
              "HF prefill attention plan QK workspace layout changed");
static_assert(offsetof(RileyCudaHfPrefillAttentionPlanInfo,
                       av_workspace_bytes) == 88,
              "HF prefill attention plan AV workspace layout changed");
static_assert(offsetof(RileyCudaHfPrefillAttentionPlanInfo,
                       workspace_bytes) == 128,
              "HF prefill attention plan memory layout changed");
static_assert(offsetof(RileyCudaHfPrefillAttentionPlanInfo,
                       batch_count) == 160,
              "HF prefill attention plan dimension layout changed");
static_assert(sizeof(RileyCudaEmbeddingErrorReport) == 32,
              "RileyCudaEmbeddingErrorReport ABI size changed");
static_assert(offsetof(RileyCudaEmbeddingErrorReport, token_position) == 8,
              "embedding error report ABI layout changed");
static_assert(sizeof(RileyCudaEmbeddingParams) == 256,
              "RileyCudaEmbeddingParams ABI size changed");
static_assert(offsetof(RileyCudaEmbeddingParams, table) == 8,
              "embedding params ABI first span changed");
static_assert(offsetof(RileyCudaEmbeddingParams, out_report) == 200,
              "embedding params ABI report offset changed");
static_assert(offsetof(RileyCudaEmbeddingParams, reserved) == 232,
              "embedding params ABI tail changed");
static_assert(sizeof(RileyCudaBf16ArgmaxResult) == 8,
              "RileyCudaBf16ArgmaxResult ABI size changed");
static_assert(offsetof(RileyCudaBf16ArgmaxResult, status) == 4,
              "BF16 argmax result ABI layout changed");
static_assert(sizeof(RileyCudaBf16ArgmaxParams) == 152,
              "RileyCudaBf16ArgmaxParams ABI size changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, logits) == 8,
              "BF16 argmax input ABI layout changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, results) == 56,
              "BF16 argmax output ABI layout changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, row_count) == 104,
              "BF16 argmax dimension ABI layout changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, reserved) == 120,
              "BF16 argmax tail ABI layout changed");
static_assert(sizeof(RileyCudaRmsNormParams) == 208,
              "RileyCudaRmsNormParams ABI size changed");
static_assert(offsetof(RileyCudaRmsNormParams, epsilon) == 168,
              "RMSNorm params ABI epsilon offset changed");
static_assert(sizeof(RileyCudaFixed37LogSoftmaxParams) == 152,
              "RileyCudaFixed37LogSoftmaxParams ABI size changed");
static_assert(offsetof(RileyCudaFixed37LogSoftmaxParams, logits) == 8,
              "fixed37 log-softmax input layout changed");
static_assert(offsetof(RileyCudaFixed37LogSoftmaxParams, output) == 56,
              "fixed37 log-softmax output layout changed");
static_assert(
    offsetof(RileyCudaFixed37LogSoftmaxParams, element_count) == 104,
    "fixed37 log-softmax dimension layout changed");
static_assert(sizeof(RileyCudaResidualAddParams) == 200,
              "RileyCudaResidualAddParams ABI size changed");
static_assert(sizeof(RileyCudaRowBiasAddInPlaceParams) == 152,
              "RileyCudaRowBiasAddInPlaceParams ABI size changed");
static_assert(offsetof(RileyCudaRowBiasAddInPlaceParams, matrix) == 8,
              "row-bias params ABI matrix offset changed");
static_assert(offsetof(RileyCudaRowBiasAddInPlaceParams, row_count) == 104,
              "row-bias params ABI dimensions changed");
static_assert(offsetof(RileyCudaRowBiasAddInPlaceParams, reserved) == 120,
              "row-bias params ABI tail changed");
static_assert(sizeof(RileyCudaSiluParams) == 152,
              "RileyCudaSiluParams ABI size changed");
static_assert(sizeof(RileyCudaGatedMultiplyParams) == 200,
              "RileyCudaGatedMultiplyParams ABI size changed");
static_assert(sizeof(RileyCudaRopeTableParams) == 152,
              "RileyCudaRopeTableParams ABI size changed");
static_assert(offsetof(RileyCudaRopeTableParams, element_count) == 104,
              "RoPE table params ABI dimension offset changed");
static_assert(sizeof(RileyCudaRopeParams) == 288,
              "RileyCudaRopeParams ABI size changed");
static_assert(offsetof(RileyCudaRopeParams, token_count) == 200,
              "RoPE params ABI dimension offset changed");
static_assert(sizeof(RileyCudaCastParams) == 152,
              "RileyCudaCastParams ABI size changed");
static_assert(sizeof(RileyCudaQkGqaParams) == 216,
              "RileyCudaQkGqaParams ABI size changed");
static_assert(offsetof(RileyCudaQkGqaParams, token_count) == 152,
              "RileyCudaQkGqaParams ABI layout changed");
static_assert(sizeof(RileyCudaScaleCausalMaskParams) == 112,
              "RileyCudaScaleCausalMaskParams ABI size changed");
static_assert(offsetof(RileyCudaScaleCausalMaskParams, scale) == 72,
              "RileyCudaScaleCausalMaskParams ABI layout changed");
static_assert(sizeof(RileyCudaCausalSoftmaxParams) == 112,
              "RileyCudaCausalSoftmaxParams ABI size changed");
static_assert(offsetof(RileyCudaCausalSoftmaxParams, reserved) == 72,
              "RileyCudaCausalSoftmaxParams ABI layout changed");
static_assert(sizeof(RileyCudaAvGqaParams) == 216,
              "RileyCudaAvGqaParams ABI size changed");
static_assert(offsetof(RileyCudaAvGqaParams, token_count) == 152,
              "RileyCudaAvGqaParams ABI layout changed");
static_assert(sizeof(RileyCudaKvCacheWriteParams) == 272,
              "RileyCudaKvCacheWriteParams ABI size changed");
static_assert(offsetof(RileyCudaKvCacheWriteParams, key_source) == 8,
              "KV cache write source ABI layout changed");
static_assert(offsetof(RileyCudaKvCacheWriteParams, key_cache) == 104,
              "KV cache write destination ABI layout changed");
static_assert(
    offsetof(RileyCudaKvCacheWriteParams, source_token_count) == 200,
    "KV cache write dimension ABI layout changed");
static_assert(offsetof(RileyCudaKvCacheWriteParams, reserved) == 240,
              "KV cache write ABI tail changed");
static_assert(sizeof(RileyCudaDecodeAttentionReferenceParams) == 328,
              "RileyCudaDecodeAttentionReferenceParams ABI size changed");
static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, query) == 8,
    "decode reference query ABI layout changed");
static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, output) == 200,
    "decode reference output ABI layout changed");
static_assert(offsetof(RileyCudaDecodeAttentionReferenceParams,
                       maximum_token_count) == 248,
              "decode reference dimension ABI layout changed");
static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, scale) == 288,
    "decode reference scale ABI layout changed");
static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, reserved) == 296,
    "decode reference ABI tail changed");
static_assert(RILEY_CUDA_DECODE_PARTIAL_STATE_VERSION == 1 &&
                  RILEY_CUDA_DECODE_REDUCTION_ASCENDING == 1 &&
                  RILEY_CUDA_DECODE_REDUCTION_DESCENDING == 2,
              "decode partial-state ABI constants changed");
static_assert(sizeof(RileyCudaDecodeAttentionParams) == 344,
              "RileyCudaDecodeAttentionParams ABI size changed");
static_assert(offsetof(RileyCudaDecodeAttentionParams, query) == 8,
              "decode attention query ABI layout changed");
static_assert(offsetof(RileyCudaDecodeAttentionParams, output) == 200,
              "decode attention output ABI layout changed");
static_assert(
    offsetof(RileyCudaDecodeAttentionParams, maximum_token_count) == 248,
    "decode attention dimension ABI layout changed");
static_assert(offsetof(RileyCudaDecodeAttentionParams, scale) == 304,
              "decode attention scale ABI layout changed");
static_assert(offsetof(RileyCudaDecodeAttentionParams, reserved) == 312,
              "decode attention ABI tail changed");
static_assert(sizeof(RileyCudaDecodePartialStateReduceParams) == 176,
              "RileyCudaDecodePartialStateReduceParams ABI size changed");
static_assert(
    offsetof(RileyCudaDecodePartialStateReduceParams, partial_states) == 8,
    "decode reducer partial-state ABI layout changed");
static_assert(
    offsetof(RileyCudaDecodePartialStateReduceParams,
             partial_state_count) == 104,
    "decode reducer dimension ABI layout changed");
static_assert(offsetof(RileyCudaDecodePartialStateReduceParams,
                       reduction_order) == 136,
              "decode reducer order ABI layout changed");
static_assert(
    offsetof(RileyCudaDecodePartialStateReduceParams, reserved) == 144,
    "decode reducer ABI tail changed");
static_assert(sizeof(RileyCudaGemmConfig) == 112,
              "RileyCudaGemmConfig ABI size changed");
static_assert(RILEY_CUDA_GEMM_TRANSPOSE_N == 0 &&
                  RILEY_CUDA_GEMM_TRANSPOSE_T == 1 &&
                  RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR == 1 &&
                  RILEY_CUDA_GEMM_EPILOGUE_NONE == 0 &&
                  RILEY_CUDA_GEMM_DETERMINISTIC_REQUIRED == 1 &&
                  RILEY_CUDA_GEMM_BACKEND_CUBLASLT == 1 &&
                  RILEY_CUDA_GEMM_BACKEND_FIXED37 == 2 &&
                  RILEY_CUDA_FIXED37_REDUCTION_VERSION == 1 &&
                  RILEY_CUDA_FIXED37_CHUNK_ELEMENTS == 37 &&
                  RILEY_CUDA_FIXED37_MAX_CHUNK_COUNT == 4096,
              "GEMM ABI discriminants changed");
static_assert(offsetof(RileyCudaGemmConfig, m) == 8,
              "GEMM config dimension layout changed");
static_assert(offsetof(RileyCudaGemmConfig, input_dtype) == 32,
              "GEMM config dtype layout changed");
static_assert(offsetof(RileyCudaGemmConfig, max_workspace_bytes) == 80,
              "GEMM config workspace layout changed");
static_assert(sizeof(RileyCudaGemmAlgorithmInfo) == 112,
              "RileyCudaGemmAlgorithmInfo ABI size changed");
static_assert(offsetof(RileyCudaGemmAlgorithmInfo, workspace_bytes) == 40,
              "GEMM algorithm workspace layout changed");
static_assert(
    offsetof(RileyCudaGemmAlgorithmInfo,
             numerical_implementation_flags) == 48,
    "GEMM algorithm numerical metadata layout changed");
static_assert(offsetof(RileyCudaGemmAlgorithmInfo, m) == 72,
              "GEMM algorithm dimension layout changed");
static_assert(sizeof(RileyCudaFixed37GemmPlanInfo) == 96,
              "RileyCudaFixed37GemmPlanInfo ABI size changed");
static_assert(
    offsetof(RileyCudaFixed37GemmPlanInfo,
             dynamic_shared_memory_bytes) == 32,
    "fixed37 GEMM plan shared-memory layout changed");
static_assert(offsetof(RileyCudaFixed37GemmPlanInfo, m) == 48,
              "fixed37 GEMM plan dimension layout changed");
static_assert(offsetof(RileyCudaFixed37GemmPlanInfo, reserved) == 72,
              "fixed37 GEMM plan tail layout changed");

inline void clear_error(RileyCudaErrorInfo* error) noexcept {
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  std::memset(error, 0, sizeof(*error));
  error->struct_size = sizeof(*error);
}

inline RileyCudaStatus set_error(RileyCudaErrorInfo* error,
                                     RileyCudaStatus status,
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

inline RileyCudaStatus validation_error(RileyCudaErrorInfo* error,
                                            RileyCudaStatus status,
                                            uint32_t stage,
                                            const char* operation,
                                            const char* detail) noexcept {
  return set_error(error, status, 0, RILEY_CUDA_ERROR_DOMAIN_VALIDATION,
                   stage, operation, detail);
}

inline RileyCudaStatus internal_error(RileyCudaErrorInfo* error,
                                          uint32_t stage,
                                          const char* operation,
                                          const char* detail) noexcept {
  return set_error(error, RILEY_CUDA_STATUS_INTERNAL_ERROR, 0,
                   RILEY_CUDA_ERROR_DOMAIN_INTERNAL, stage, operation,
                   detail);
}

inline RileyCudaStatus driver_error(CUresult result,
                                        RileyCudaErrorInfo* error,
                                        uint32_t stage,
                                        const char* operation) noexcept {
  if (result == CUDA_SUCCESS) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  const char* detail = nullptr;
  (void)cuGetErrorString(result, &detail);
  RileyCudaStatus status = RILEY_CUDA_STATUS_DRIVER_ERROR;
  if (result == CUDA_ERROR_INVALID_DEVICE) {
    status = RILEY_CUDA_STATUS_INVALID_DEVICE;
  } else if (result == CUDA_ERROR_INVALID_VALUE) {
    status = RILEY_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    status = RILEY_CUDA_STATUS_OUT_OF_MEMORY;
  } else if (result == CUDA_ERROR_NOT_READY) {
    status = RILEY_CUDA_STATUS_NOT_READY;
  }
  return set_error(error, status, static_cast<int32_t>(result),
                   RILEY_CUDA_ERROR_DOMAIN_DRIVER, stage, operation,
                   detail);
}

inline RileyCudaStatus runtime_error(cudaError_t result,
                                         RileyCudaErrorInfo* error,
                                         uint32_t stage,
                                         const char* operation) noexcept {
  if (result == cudaSuccess) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaStatus status = RILEY_CUDA_STATUS_RUNTIME_ERROR;
  if (result == cudaErrorInvalidDevice) {
    status = RILEY_CUDA_STATUS_INVALID_DEVICE;
  } else if (result == cudaErrorInvalidValue ||
             result == cudaErrorInvalidConfiguration) {
    status = RILEY_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == cudaErrorMemoryAllocation) {
    status = RILEY_CUDA_STATUS_OUT_OF_MEMORY;
  } else if (result == cudaErrorNotReady) {
    status = RILEY_CUDA_STATUS_NOT_READY;
  }
  return set_error(error, status, static_cast<int32_t>(result),
                   RILEY_CUDA_ERROR_DOMAIN_RUNTIME, stage, operation,
                   cudaGetErrorString(result));
}

class CurrentContext final {
 public:
  explicit CurrentContext(RileyCudaContext* context) noexcept
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

  RileyCudaStatus enter(RileyCudaErrorInfo* error, uint32_t stage,
                            const char* operation) noexcept {
    if (context_ == nullptr) {
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                              stage, operation, "context is null");
    }
    if (context_->restoration_failed.load(std::memory_order_acquire)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE, stage, operation,
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
      return RILEY_CUDA_STATUS_SUCCESS;
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
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  RileyCudaStatus leave(RileyCudaStatus operation_status,
                            RileyCudaErrorInfo* error, uint32_t stage,
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
      if (operation_status != RILEY_CUDA_STATUS_SUCCESS &&
          operation_status != RILEY_CUDA_STATUS_NOT_READY) {
        return operation_status;
      }
      return driver_error(result, error, stage, operation);
    }
    if (popped != context_->context) {
      poison_context();
      reconcile_after_uncertain_pop();
      if (operation_status != RILEY_CUDA_STATUS_SUCCESS &&
          operation_status != RILEY_CUDA_STATUS_NOT_READY) {
        return operation_status;
      }
      return internal_error(error, stage, operation,
                            "CUDA context stack returned a different context");
    }
    active_ = false;
    if (operation_status != RILEY_CUDA_STATUS_SUCCESS) {
      return operation_status;
    }
    return RILEY_CUDA_STATUS_SUCCESS;
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

  RileyCudaContext* context_;
  CUcontext previous_;
  bool active_;
};

inline bool same_context(const RileyCudaContext* left,
                         const RileyCudaContext* right) noexcept {
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
    const RileyCudaStream* stream) noexcept {
  return stream != nullptr &&
         stream->command_batch_owner.load(std::memory_order_acquire) !=
             nullptr;
}

inline bool command_batch_is_owned_by_current_thread(
    const RileyCudaStream* stream) noexcept {
  return stream != nullptr &&
         stream->command_batch_owner.load(std::memory_order_acquire) ==
             command_batch_thread_token();
}

// Registers a resource lease in the active command batch. Repeated use of the
// same counter is reentrant only for the owning stream/thread and consumes no
// additional ledger entry. Capacity is checked before acquisition, so the hot
// path never allocates and a full ledger cannot strand an unrecorded lease.
inline RileyCudaStatus command_batch_register_use(
    RileyCudaStream* stream, std::atomic<uint32_t>* active,
    RileyCudaErrorInfo* error, const char* operation,
    const char* busy_detail) noexcept {
  if (stream == nullptr || active == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "command-batch stream or resource is null");
  }
  if (!command_batch_is_owned_by_current_thread(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
        "an active stream command batch is owned by another thread");
  }
  for (size_t index = 0; index < stream->command_batch_use_count; ++index) {
    if (stream->command_batch_uses[index] == active) {
      return RILEY_CUDA_STATUS_SUCCESS;
    }
  }
  if (stream->command_batch_use_count ==
      RileyCudaStream::kCommandBatchUseCapacity) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
        "stream command-batch resource ledger capacity was exceeded");
  }
  if (!try_acquire_exclusive_use(*active)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            busy_detail);
  }
  stream->command_batch_uses[stream->command_batch_use_count++] = active;
  return RILEY_CUDA_STATUS_SUCCESS;
}

inline bool retain_child(RileyCudaContext* context) noexcept {
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

inline bool release_child(RileyCudaContext* context) noexcept {
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

}  // namespace riley_cuda_internal

#endif  // RILEY_CUDA_FFI_INTERNAL_HPP_
