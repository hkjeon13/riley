#include "ffi_internal.hpp"

#include <climits>

namespace {

using rustinfer_cuda_internal::AllocationStatsGuard;
using rustinfer_cuda_internal::CurrentContext;
using rustinfer_cuda_internal::clear_error;
using rustinfer_cuda_internal::driver_error;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::retain_child;
using rustinfer_cuda_internal::release_child;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::set_error;
using rustinfer_cuda_internal::validation_error;

RustInferCudaStatus device_attribute(CUdevice device,
                                     CUdevice_attribute attribute,
                                     uint32_t* output,
                                     RustInferCudaErrorInfo* error,
                                     const char* operation) noexcept {
  int value = 0;
  const CUresult result = cuDeviceGetAttribute(&value, attribute, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        operation);
  }
  if (value < 0) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                          operation, "CUDA returned a negative device attribute");
  }
  *output = static_cast<uint32_t>(value);
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

void destroy_stream_after_failed_create(RustInferCudaContext* context,
                                        cudaStream_t stream) noexcept {
  CurrentContext cleanup(context);
  RustInferCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  if (cleanup.enter(&ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                    "cleanup stream after create") ==
      RUSTINFER_CUDA_STATUS_SUCCESS) {
    (void)cudaStreamDestroy(stream);
    (void)cleanup.leave(RUSTINFER_CUDA_STATUS_SUCCESS, &ignored,
                        RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                        "cleanup stream after create");
  }
}

void destroy_event_after_failed_create(RustInferCudaContext* context,
                                       cudaEvent_t event) noexcept {
  CurrentContext cleanup(context);
  RustInferCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  if (cleanup.enter(&ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                    "cleanup event after create") ==
      RUSTINFER_CUDA_STATUS_SUCCESS) {
    (void)cudaEventDestroy(event);
    (void)cleanup.leave(RUSTINFER_CUDA_STATUS_SUCCESS, &ignored,
                        RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                        "cleanup event after create");
  }
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_device_count(
    uint32_t* out_count, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_count == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "device count", "out_count is null");
  }
  *out_count = 0;
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "initialize CUDA driver");
  }
  int count = 0;
  result = cuDeviceGetCount(&count);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "enumerate CUDA devices");
  }
  if (count < 0) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                          "enumerate CUDA devices",
                          "CUDA returned a negative device count");
  }
  *out_count = static_cast<uint32_t>(count);
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_device_properties(
    int32_t ordinal, RustInferCudaDeviceProperties* out_properties,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_properties == nullptr ||
      out_properties->struct_size < sizeof(*out_properties)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, "query device properties",
        "out_properties is null or has an incompatible struct_size");
  }
  std::memset(out_properties, 0, sizeof(*out_properties));
  out_properties->struct_size = sizeof(*out_properties);
  out_properties->ordinal = ordinal;
  if (ordinal < 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_DEVICE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "query device properties",
                            "device ordinal must be non-negative");
  }

  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "initialize CUDA driver");
  }
  CUdevice device = 0;
  result = cuDeviceGet(&device, ordinal);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "select CUDA device");
  }
  result = cuDeviceGetName(out_properties->name,
                           RUSTINFER_CUDA_DEVICE_NAME_CAPACITY, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "query CUDA device name");
  }
  size_t total_memory = 0;
  result = cuDeviceTotalMem(&total_memory, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "query CUDA device memory");
  }
  out_properties->total_memory_bytes = static_cast<uint64_t>(total_memory);

  RustInferCudaStatus status = device_attribute(
      device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      &out_properties->compute_capability_major, error,
      "query compute capability major");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device,
                            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                            &out_properties->compute_capability_minor, error,
                            "query compute capability minor");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                            &out_properties->multiprocessor_count, error,
                            "query multiprocessor count");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                            &out_properties->warp_size, error,
                            "query warp size");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            &out_properties->max_threads_per_block, error,
                            "query maximum threads per block");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  result = cuDriverGetVersion(&out_properties->driver_version);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "query CUDA driver version");
  }
  const cudaError_t runtime_result =
      cudaRuntimeGetVersion(&out_properties->runtime_version);
  if (runtime_result != cudaSuccess) {
    return runtime_error(runtime_result, error,
                         RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                         "query CUDA Runtime version");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_context_create(
    int32_t ordinal, RustInferCudaContext** out_context,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA context", "out_context is null");
  }
  *out_context = nullptr;
  if (ordinal < 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_DEVICE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA context",
                            "device ordinal must be non-negative");
  }
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                        "initialize CUDA driver");
  }
  CUdevice device = 0;
  result = cuDeviceGet(&device, ordinal);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                        "select CUDA context device");
  }
  CUcontext primary = nullptr;
  result = cuDevicePrimaryCtxRetain(&primary, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                        "retain CUDA primary context");
  }
  void* context_storage = std::calloc(1, sizeof(RustInferCudaContext));
  if (context_storage == nullptr) {
    (void)cuDevicePrimaryCtxRelease(device);
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create CUDA context",
                     "host allocation failed");
  }
  auto* context =
      new (context_storage) RustInferCudaContext(device, primary, ordinal);

  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  bool context_stack_restored = false;
  {
    CurrentContext scope(context);
    status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                         "initialize CUDA context");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      status = runtime_error(cudaFree(nullptr), error,
                             RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                             "initialize CUDA Runtime in primary context");
      status = scope.leave(status, error,
                           RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                           "initialize CUDA context");
    }
    if (scope.active()) {
      // A failed pop must not be followed by releasing storage still needed by
      // a current primary context. Retry once while preserving the first error.
      status = scope.leave(status, error,
                           RUSTINFER_CUDA_ERROR_STAGE_INITIALIZE,
                           "restore CUDA context after initialization");
    }
    context_stack_restored = !scope.active();
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    if (!context_stack_restored ||
        context->restoration_failed.load(std::memory_order_acquire)) {
      // The driver rejected repeated restoration attempts. Keep the retained
      // context and wrapper alive rather than release ambiguous current-context
      // ownership. This catastrophic path intentionally leaks.
      return status;
    }
    (void)cuDevicePrimaryCtxRelease(device);
    context->~RustInferCudaContext();
    std::free(context);
    return status;
  }
  *out_context = context;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_context_synchronize(
    RustInferCudaContext* context, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  CurrentContext scope(context);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE, "synchronize CUDA context");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaDeviceSynchronize(), error,
                           RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA context");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                     "synchronize CUDA context");
}

extern "C" RustInferCudaStatus rustinfer_cuda_context_memory_info(
    RustInferCudaContext* context, uint64_t* out_free_bytes,
    uint64_t* out_total_bytes, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_free_bytes == nullptr || out_total_bytes == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA memory info",
                            "memory output pointer is null");
  }
  *out_free_bytes = 0;
  *out_total_bytes = 0;
  CurrentContext scope(context);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_QUERY, "query CUDA memory info");
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                           RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA memory info");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                       "query CUDA memory info");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    *out_free_bytes = static_cast<uint64_t>(free_bytes);
    *out_total_bytes = static_cast<uint64_t>(total_bytes);
  }
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_context_allocation_stats(
    RustInferCudaContext* context, RustInferCudaAllocationStats* out_stats,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr || out_stats == nullptr ||
      out_stats->struct_size < sizeof(*out_stats)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, "query CUDA allocation stats",
        "context or out_stats is null, or struct_size is incompatible");
  }
  std::memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = sizeof(*out_stats);
  const AllocationStatsGuard guard(context);
  out_stats->device_live_bytes =
      context->device_live_bytes.load(std::memory_order_relaxed);
  out_stats->device_live_allocations =
      context->device_live_allocations.load(std::memory_order_relaxed);
  out_stats->pinned_host_live_bytes =
      context->pinned_host_live_bytes.load(std::memory_order_relaxed);
  out_stats->pinned_host_live_allocations =
      context->pinned_host_live_allocations.load(std::memory_order_relaxed);
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_context_close(
    RustInferCudaContext** context, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA context", "context pointer is null");
  }
  if (*context == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if ((*context)->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_CLOSE, "close CUDA context",
        "a prior CUDA context-stack restoration failed; refusing to release the primary-context lease");
  }
  const uint32_t live_children =
      (*context)->live_children.load(std::memory_order_acquire);
  if (live_children != 0) {
    char detail[128]{};
    std::snprintf(detail, sizeof(detail),
                  "context still owns %u live stream/event/buffer/copy resources",
                  live_children);
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA context", detail);
  }
  bool has_live_allocation_accounting = false;
  {
    const AllocationStatsGuard guard(*context);
    has_live_allocation_accounting =
        (*context)->device_live_bytes.load(std::memory_order_relaxed) != 0 ||
        (*context)->device_live_allocations.load(std::memory_order_relaxed) != 0 ||
        (*context)->pinned_host_live_bytes.load(std::memory_order_relaxed) != 0 ||
        (*context)->pinned_host_live_allocations.load(
            std::memory_order_relaxed) != 0;
  }
  if (has_live_allocation_accounting) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_CLOSE, "close CUDA context",
        "context allocation accounting is non-zero; refusing to release the "
        "primary-context lease");
  }
  const CUresult result = cuDevicePrimaryCtxRelease((*context)->device);
  const RustInferCudaStatus status =
      driver_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                   "release CUDA primary context");
  // Driver release may report an earlier asynchronous error after decrementing
  // the primary-context refcount. Consume the wrapper after the single release
  // attempt; a genuine failure becomes a safe lease leak, never a double
  // release of another module's shared primary-context ownership.
  (*context)->~RustInferCudaContext();
  std::free(*context);
  *context = nullptr;
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_stream_create(
    RustInferCudaContext* context, RustInferCudaStream** out_stream,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA stream", "out_stream is null");
  }
  *out_stream = nullptr;
  CurrentContext scope(context);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create CUDA stream");
  cudaStream_t native = nullptr;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), error,
        RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create non-default CUDA stream");
  }
  void* stream_storage = std::calloc(1, sizeof(RustInferCudaStream));
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && stream_storage == nullptr) {
    status = set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                       RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                       RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create CUDA stream",
                       "host allocation failed");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                       "create CUDA stream");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    if (native != nullptr) {
      destroy_stream_after_failed_create(context, native);
    }
    std::free(stream_storage);
    return status;
  }
  auto* stream = new (stream_storage) RustInferCudaStream{context, native};
  if (!retain_child(context)) {
    destroy_stream_after_failed_create(context, native);
    stream->~RustInferCudaStream();
    std::free(stream);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA stream",
                          "context child-resource counter overflow");
  }
  *out_stream = stream;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_stream_query(
    RustInferCudaStream* stream, uint8_t* out_complete,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr || out_complete == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA stream",
                            "stream or out_complete is null");
  }
  *out_complete = 0;
  CurrentContext scope(stream->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_QUERY, "query CUDA stream");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaStreamQuery(stream->stream);
    if (result == cudaSuccess) {
      *out_complete = 1;
    }
    status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA stream");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                     "query CUDA stream");
}

extern "C" RustInferCudaStatus rustinfer_cuda_stream_synchronize(
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA stream", "stream is null");
  }
  CurrentContext scope(stream->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize CUDA stream");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaStreamSynchronize(stream->stream), error,
                           RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA stream");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                     "synchronize CUDA stream");
}

extern "C" RustInferCudaStatus rustinfer_cuda_stream_wait_event(
    RustInferCudaStream* stream, RustInferCudaEvent* event,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr || event == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "wait for CUDA event",
                            "stream or event is null");
  }
  if (!same_context(stream->owner, event->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "wait for CUDA event",
                            "stream and event belong to different contexts");
  }
  CurrentContext scope(stream->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_RECORD, "wait for CUDA event");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaStreamWaitEvent(stream->stream, event->event, 0),
                           error, RUSTINFER_CUDA_ERROR_STAGE_RECORD,
                           "wait for CUDA event");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_RECORD,
                     "wait for CUDA event");
}

extern "C" RustInferCudaStatus rustinfer_cuda_stream_close(
    RustInferCudaStream** stream, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA stream", "stream pointer is null");
  }
  if (*stream == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if ((*stream)->active_copies.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA stream",
                            "stream still owns an active copy token");
  }
  CurrentContext scope((*stream)->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE, "close CUDA stream");
  bool destroy_attempted = false;
  cudaError_t destroy_result = cudaErrorUnknown;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    destroy_attempted = true;
    destroy_result = cudaStreamDestroy((*stream)->stream);
    status = runtime_error(destroy_result, error,
                           RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                           "close CUDA stream");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                       "close CUDA stream");
  if (destroy_attempted) {
    // Runtime destroy calls may report a prior asynchronous error after the
    // resource has already been consumed. Null the opaque owner after the
    // single destroy attempt; retrying could double-destroy. A genuine destroy
    // failure is therefore fail-closed as a native-resource leak.
    const bool released = release_child((*stream)->owner);
    (*stream)->~RustInferCudaStream();
    std::free(*stream);
    *stream = nullptr;
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS && !released) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA stream",
                            "context child-resource counter underflow");
    }
  }
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_event_create(
    RustInferCudaContext* context, RustInferCudaEvent** out_event,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_event == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA event", "out_event is null");
  }
  *out_event = nullptr;
  CurrentContext scope(context);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create CUDA event");
  cudaEvent_t native = nullptr;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaEventCreateWithFlags(&native, cudaEventDefault),
                           error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                           "create timing-enabled CUDA event");
  }
  void* event_storage = std::calloc(1, sizeof(RustInferCudaEvent));
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && event_storage == nullptr) {
    status = set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                       RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                       RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create CUDA event",
                       "host allocation failed");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                       "create CUDA event");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    if (native != nullptr) {
      destroy_event_after_failed_create(context, native);
    }
    std::free(event_storage);
    return status;
  }
  auto* event = new (event_storage) RustInferCudaEvent{context, native};
  if (!retain_child(context)) {
    destroy_event_after_failed_create(context, native);
    event->~RustInferCudaEvent();
    std::free(event);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA event",
                          "context child-resource counter overflow");
  }
  *out_event = event;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_event_record(
    RustInferCudaEvent* event, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr || stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "record CUDA event", "event or stream is null");
  }
  if (!same_context(event->owner, stream->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "record CUDA event",
                            "event and stream belong to different contexts");
  }
  CurrentContext scope(event->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_RECORD, "record CUDA event");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaEventRecord(event->event, stream->stream), error,
                           RUSTINFER_CUDA_ERROR_STAGE_RECORD,
                           "record CUDA event");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_RECORD,
                     "record CUDA event");
}

extern "C" RustInferCudaStatus rustinfer_cuda_event_query(
    RustInferCudaEvent* event, uint8_t* out_complete,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr || out_complete == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA event",
                            "event or out_complete is null");
  }
  *out_complete = 0;
  CurrentContext scope(event->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_QUERY, "query CUDA event");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaEventQuery(event->event);
    if (result == cudaSuccess) {
      *out_complete = 1;
    }
    status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA event");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                     "query CUDA event");
}

extern "C" RustInferCudaStatus rustinfer_cuda_event_synchronize(
    RustInferCudaEvent* event, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA event", "event is null");
  }
  CurrentContext scope(event->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize CUDA event");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaEventSynchronize(event->event), error,
                           RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA event");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                     "synchronize CUDA event");
}

extern "C" RustInferCudaStatus rustinfer_cuda_event_elapsed_ms(
    RustInferCudaEvent* start, RustInferCudaEvent* end, float* out_elapsed_ms,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (start == nullptr || end == nullptr || out_elapsed_ms == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "measure CUDA event elapsed time",
                            "event or output pointer is null");
  }
  *out_elapsed_ms = 0.0F;
  if (!same_context(start->owner, end->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "measure CUDA event elapsed time",
                            "events belong to different contexts");
  }
  CurrentContext scope(start->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
      "measure CUDA event elapsed time");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaEventElapsedTime(out_elapsed_ms, start->event, end->event), error,
        RUSTINFER_CUDA_ERROR_STAGE_QUERY, "measure CUDA event elapsed time");
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                     "measure CUDA event elapsed time");
}

extern "C" RustInferCudaStatus rustinfer_cuda_event_close(
    RustInferCudaEvent** event, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA event", "event pointer is null");
  }
  if (*event == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  CurrentContext scope((*event)->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE, "close CUDA event");
  bool destroy_attempted = false;
  cudaError_t destroy_result = cudaErrorUnknown;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    destroy_attempted = true;
    destroy_result = cudaEventDestroy((*event)->event);
    status = runtime_error(destroy_result, error,
                           RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                           "close CUDA event");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                       "close CUDA event");
  if (destroy_attempted) {
    // See stream_close: a non-success may be a deferred error even when the
    // event was consumed, so ownership is single-shot after destroy begins.
    const bool released = release_child((*event)->owner);
    (*event)->~RustInferCudaEvent();
    std::free(*event);
    *event = nullptr;
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS && !released) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA event",
                            "context child-resource counter underflow");
    }
  }
  return status;
}
