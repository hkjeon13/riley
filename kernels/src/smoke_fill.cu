#include "ffi_internal.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

using rustinfer_cuda_internal::CurrentContext;
using rustinfer_cuda_internal::clear_error;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::release_child;
using rustinfer_cuda_internal::retain_child;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::set_error;
using rustinfer_cuda_internal::validation_error;

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint64_t kMaximumGridX = static_cast<uint64_t>(INT_MAX);

__global__ void smoke_fill_f32(float* output, uint64_t element_count,
                               float value) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = value;
  }
}

RustInferCudaStatus prior_launch_error(RustInferCudaErrorInfo* error,
                                       const char* operation) noexcept {
  const cudaError_t prior = cudaGetLastError();
  if (prior == cudaSuccess) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  return runtime_error(prior, error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                       operation);
}

void free_buffer_after_failed_create(RustInferCudaContext* context,
                                     float* device_data) noexcept {
  if (device_data == nullptr) {
    return;
  }
  CurrentContext cleanup(context);
  RustInferCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  if (cleanup.enter(&ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                    "cleanup smoke buffer after create") ==
      RUSTINFER_CUDA_STATUS_SUCCESS) {
    (void)cudaFree(device_data);
    (void)cleanup.leave(RUSTINFER_CUDA_STATUS_SUCCESS, &ignored,
                        RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                        "cleanup smoke buffer after create");
  }
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_smoke_buffer_create(
    RustInferCudaContext* context, uint64_t element_count,
    RustInferCudaSmokeBuffer** out_buffer,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create smoke buffer",
                            "out_buffer is null");
  }
  *out_buffer = nullptr;
  if (context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create smoke buffer", "context is null");
  }
  if (element_count > SIZE_MAX / sizeof(float)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create smoke buffer",
                            "element_count overflows host size_t byte length");
  }
  if (!retain_child(context)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create smoke buffer",
                          "context child-resource counter overflow");
  }

  void* buffer_storage = std::calloc(1, sizeof(RustInferCudaSmokeBuffer));
  if (buffer_storage == nullptr) {
    (void)release_child(context);
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create smoke buffer",
                     "host allocation failed");
  }
  auto* buffer = new (buffer_storage) RustInferCudaSmokeBuffer{
      context, nullptr, element_count, false, nullptr};

  CurrentContext scope(context);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CREATE, "create smoke buffer");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && element_count != 0) {
    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);
    void* allocation = nullptr;
    const cudaError_t allocation_result = cudaMalloc(&allocation, bytes);
    buffer->device_data = static_cast<float*>(allocation);
    status = runtime_error(allocation_result, error,
                           RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                           "allocate smoke device buffer");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                       "create smoke buffer");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    free_buffer_after_failed_create(context, buffer->device_data);
    buffer->~RustInferCudaSmokeBuffer();
    std::free(buffer);
    (void)release_child(context);
    return status;
  }
  *out_buffer = buffer;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_smoke_fill_launch(
    RustInferCudaSmokeBuffer* buffer, RustInferCudaStream* stream, float value,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr || stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "launch smoke fill",
                            "buffer or stream is null");
  }
  if (!same_context(buffer->owner, stream->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "launch smoke fill",
                            "buffer and stream belong to different contexts");
  }
  if (buffer->in_flight) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "launch smoke fill",
                            "buffer already has an in-flight operation");
  }
  if (buffer->element_count == 0) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  const uint64_t block_count =
      (buffer->element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (block_count == 0 || block_count > kMaximumGridX) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "launch smoke fill",
                            "element_count exceeds the supported grid range");
  }

  CurrentContext scope(buffer->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, "launch smoke fill");
  bool launch_enqueued = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_error(error, "observe prior CUDA launch error");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    smoke_fill_f32<<<static_cast<uint32_t>(block_count), kThreadsPerBlock, 0,
                     stream->stream>>>(buffer->device_data,
                                      buffer->element_count, value);
    status = runtime_error(cudaGetLastError(), error,
                           RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                           "launch smoke fill");
    launch_enqueued = status == RUSTINFER_CUDA_STATUS_SUCCESS;
    if (launch_enqueued) {
      // Commit native ownership before context restoration: a failed pop must
      // never make an enqueued kernel look idle to close/drop paths.
      buffer->in_flight = true;
      buffer->launch_stream = stream->stream;
    }
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                       "launch smoke fill");
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_smoke_copy_to_host(
    RustInferCudaSmokeBuffer* buffer, RustInferCudaStream* stream,
    float* host_output, uint64_t host_element_capacity,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr || stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "copy smoke buffer to host",
                            "buffer or stream is null");
  }
  if (!same_context(buffer->owner, stream->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "copy smoke buffer to host",
                            "buffer and stream belong to different contexts");
  }
  if (host_element_capacity < buffer->element_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "copy smoke buffer to host",
                            "host output is smaller than element_count");
  }
  if (buffer->element_count != 0 && host_output == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "copy smoke buffer to host",
                            "host output is null");
  }
  if (buffer->element_count == 0) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if (!buffer->in_flight) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "copy smoke buffer to host",
                            "smoke fill has not been launched");
  }
  if (buffer->launch_stream != stream->stream) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "copy smoke buffer to host",
                            "copy must use the stream that launched the fill");
  }

  CurrentContext scope(buffer->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize smoke fill before host copy");
  uint32_t leave_stage = RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t synchronize_result = cudaStreamSynchronize(stream->stream);
    status = runtime_error(synchronize_result, error,
                           RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "complete smoke fill before host copy");
    if (synchronize_result == cudaSuccess) {
      // Completion is true even if restoring the caller's context later fails.
      buffer->in_flight = false;
      buffer->launch_stream = nullptr;
    }
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    // The originating non-default stream is complete, so a synchronous copy
    // cannot retain the caller-owned host pointer beyond this ABI call. This
    // avoids pageable-host lifetime ambiguity if an async enqueue reports a
    // deferred earlier CUDA error after taking a side effect.
    leave_stage = RUSTINFER_CUDA_ERROR_STAGE_COPY;
    const size_t bytes =
        static_cast<size_t>(buffer->element_count) * sizeof(float);
    status = runtime_error(
        cudaMemcpy(host_output, buffer->device_data, bytes,
                   cudaMemcpyDeviceToHost),
        error, RUSTINFER_CUDA_ERROR_STAGE_COPY,
        "copy completed smoke buffer to host");
  }
  status = scope.leave(status, error, leave_stage,
                       "copy smoke buffer to host");
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_smoke_buffer_close(
    RustInferCudaSmokeBuffer** buffer,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close smoke buffer", "buffer pointer is null");
  }
  if (*buffer == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  CurrentContext scope((*buffer)->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE, "close smoke buffer");
  bool resource_consumed = false;
  bool free_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && (*buffer)->in_flight) {
    status = runtime_error(cudaDeviceSynchronize(), error,
                           RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize in-flight smoke buffer before close");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      (*buffer)->in_flight = false;
      (*buffer)->launch_stream = nullptr;
    }
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (*buffer)->device_data != nullptr) {
    free_attempted = true;
    const cudaError_t free_result = cudaFree((*buffer)->device_data);
    status = runtime_error(free_result, error,
                           RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                           "free smoke device buffer");
    // cudaFree may return a deferred asynchronous error after consuming the
    // allocation. Discard ownership after the single attempt; retaining it
    // would permit a retry to double-free. A genuine free failure leaks safely.
    (*buffer)->device_data = nullptr;
  }
  if ((*buffer)->device_data == nullptr && !(*buffer)->in_flight &&
      (status == RUSTINFER_CUDA_STATUS_SUCCESS || free_attempted)) {
    resource_consumed = true;
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                       "close smoke buffer");
  if (resource_consumed) {
    RustInferCudaContext* owner = (*buffer)->owner;
    (*buffer)->~RustInferCudaSmokeBuffer();
    std::free(*buffer);
    *buffer = nullptr;
    if (!release_child(owner) && status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close smoke buffer",
                            "context child-resource counter underflow");
    }
  }
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_smoke_invalid_launch(
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "launch intentionally invalid smoke kernel",
                            "stream is null");
  }
  CurrentContext scope(stream->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
      "launch intentionally invalid smoke kernel");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_error(error, "observe prior CUDA launch error");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    smoke_fill_f32<<<0, kThreadsPerBlock, 0, stream->stream>>>(nullptr, 0, 0.0F);
    const cudaError_t launch_result = cudaGetLastError();
    if (launch_result == cudaSuccess) {
      status = internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                              "launch intentionally invalid smoke kernel",
                              "CUDA accepted a zero-sized launch grid");
    } else {
      status = runtime_error(launch_result, error,
                             RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                             "launch intentionally invalid smoke kernel");
    }
  }
  return scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                     "launch intentionally invalid smoke kernel");
}
