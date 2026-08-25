#include "ffi_internal.hpp"

#include <cstddef>
#include <cstdint>

namespace {

using rustinfer_cuda_internal::AllocationStatsGuard;
using rustinfer_cuda_internal::CurrentContext;
using rustinfer_cuda_internal::clear_error;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::release_child;
using rustinfer_cuda_internal::retain_child;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::set_error;
using rustinfer_cuda_internal::validation_error;

bool valid_range(uint64_t total, uint64_t offset, uint64_t length) noexcept {
  return offset <= total && length <= total - offset;
}

void publish_error(RustInferCudaErrorInfo* destination,
                   const RustInferCudaErrorInfo& source) noexcept {
  if (destination == nullptr || destination->struct_size < sizeof(*destination)) {
    return;
  }
  const uint32_t struct_size = destination->struct_size;
  std::memcpy(destination, &source, sizeof(source));
  destination->struct_size = struct_size;
}

bool account_allocation(RustInferCudaContext* context,
                        std::atomic<uint64_t>& live_bytes,
                        std::atomic<uint64_t>& live_allocations,
                        uint64_t byte_len) noexcept {
  const AllocationStatsGuard guard(context);
  const uint64_t current_bytes = live_bytes.load(std::memory_order_relaxed);
  const uint64_t current_allocations =
      live_allocations.load(std::memory_order_relaxed);
  if (current_bytes > UINT64_MAX - byte_len ||
      current_allocations == UINT64_MAX) {
    return false;
  }
  live_bytes.store(current_bytes + byte_len, std::memory_order_relaxed);
  live_allocations.store(current_allocations + 1, std::memory_order_relaxed);
  return true;
}

bool release_allocation(RustInferCudaContext* context,
                        std::atomic<uint64_t>& live_bytes,
                        std::atomic<uint64_t>& live_allocations,
                        uint64_t byte_len) noexcept {
  const AllocationStatsGuard guard(context);
  const uint64_t current_bytes = live_bytes.load(std::memory_order_relaxed);
  const uint64_t current_allocations =
      live_allocations.load(std::memory_order_relaxed);
  if (current_bytes < byte_len || current_allocations == 0) {
    return false;
  }
  live_bytes.store(current_bytes - byte_len, std::memory_order_relaxed);
  live_allocations.store(current_allocations - 1, std::memory_order_relaxed);
  return true;
}

bool try_acquire_copy(std::atomic<uint32_t>& active) noexcept {
  uint32_t expected = 0;
  return active.compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                        std::memory_order_relaxed);
}

bool release_copy(std::atomic<uint32_t>& active) noexcept {
  uint32_t expected = 1;
  return active.compare_exchange_strong(expected, 0, std::memory_order_acq_rel,
                                        std::memory_order_relaxed);
}

bool free_device_after_failed_create(RustInferCudaContext* context,
                                     void* allocation) noexcept {
  if (allocation == nullptr) {
    return true;
  }
  CurrentContext cleanup(context);
  RustInferCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  RustInferCudaStatus status = cleanup.enter(
      &ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
      "cleanup device allocation after create");
  bool free_confirmed = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaFree(allocation);
    free_confirmed = result == cudaSuccess;
    status = runtime_error(result, &ignored,
                           RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                           "cleanup device allocation after create");
  }
  (void)cleanup.leave(status, &ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                      "cleanup device allocation after create");
  return free_confirmed;
}

bool free_pinned_after_failed_create(RustInferCudaContext* context,
                                     void* allocation) noexcept {
  if (allocation == nullptr) {
    return true;
  }
  CurrentContext cleanup(context);
  RustInferCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  RustInferCudaStatus status = cleanup.enter(
      &ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
      "cleanup pinned allocation after create");
  bool free_confirmed = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaFreeHost(allocation);
    free_confirmed = result == cudaSuccess;
    status = runtime_error(result, &ignored,
                           RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                           "cleanup pinned allocation after create");
  }
  (void)cleanup.leave(status, &ignored, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                      "cleanup pinned allocation after create");
  return free_confirmed;
}

void preserve_unresolved_allocation(
    RustInferCudaContext* context, std::atomic<uint64_t>& live_bytes,
    std::atomic<uint64_t>& live_allocations, uint64_t byte_len) noexcept {
  // No caller-visible handle exists on a failed create, so an allocation whose
  // rollback cannot be confirmed must remain permanently accounted. Either
  // the stats or the child lease (normally both) independently prevents a
  // false-success primary-context release.
  (void)account_allocation(context, live_bytes, live_allocations, byte_len);
  (void)retain_child(context);
}

bool release_copy_uses(RustInferCudaCopy* copy) noexcept {
  if (copy->completed) {
    return true;
  }
  if (copy->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      copy->device->active_uses.load(std::memory_order_acquire) != 1 ||
      copy->host->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  // Raw C callers must serialize mutation of opaque handles. Under that ABI
  // rule this precheck makes the three-resource state transition all-or-none.
  copy->stream->active_uses.store(0, std::memory_order_release);
  copy->device->active_uses.store(0, std::memory_order_release);
  copy->host->active_uses.store(0, std::memory_order_release);
  copy->completed = true;
  return true;
}

RustInferCudaStatus deferred_status(RustInferCudaCopy* copy,
                                    RustInferCudaErrorInfo* error) noexcept {
  if (copy->deferred_status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    publish_error(error, copy->deferred_error);
  }
  return copy->deferred_status;
}

RustInferCudaStatus submit_copy(RustInferCudaDeviceBuffer* device,
                                uint64_t device_offset,
                                RustInferCudaPinnedHostBuffer* host,
                                uint64_t host_offset, uint64_t byte_len,
                                RustInferCudaStream* stream,
                                cudaMemcpyKind direction,
                                RustInferCudaCopy** out_copy,
                                RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_copy == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy", "out_copy is null");
  }
  *out_copy = nullptr;
  if (device == nullptr || host == nullptr || stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "device, host, or stream handle is null");
  }
  if (!same_context(device->owner, host->owner) ||
      !same_context(device->owner, stream->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "copy resources belong to different context owners");
  }
  if (device->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, "submit CUDA copy",
        "CUDA context is poisoned by a prior restoration failure");
  }
  if (!valid_range(device->byte_len, device_offset, byte_len) ||
      !valid_range(host->byte_len, host_offset, byte_len)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "copy offset and length exceed a buffer range");
  }
  if (byte_len == 0) {
    if (device->active_uses.load(std::memory_order_acquire) != 0 ||
        host->active_uses.load(std::memory_order_acquire) != 0 ||
        stream->active_uses.load(std::memory_order_acquire) != 0) {
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              "submit CUDA copy",
                              "copy resource already has an active copy token");
    }
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  void* copy_storage = std::calloc(1, sizeof(RustInferCudaCopy));
  if (copy_storage == nullptr) {
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE, "submit CUDA copy",
                     "host copy-token allocation failed");
  }
  if (!try_acquire_copy(device->active_uses)) {
    std::free(copy_storage);
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "device buffer already has an active copy token");
  }
  if (!try_acquire_copy(host->active_uses)) {
    (void)release_copy(device->active_uses);
    std::free(copy_storage);
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "pinned host buffer already has an active copy token");
  }
  if (!try_acquire_copy(stream->active_uses)) {
    (void)release_copy(host->active_uses);
    (void)release_copy(device->active_uses);
    std::free(copy_storage);
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "stream already has an active copy token");
  }
  if (!retain_child(device->owner)) {
    (void)release_copy(stream->active_uses);
    (void)release_copy(host->active_uses);
    (void)release_copy(device->active_uses);
    std::free(copy_storage);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "submit CUDA copy",
                          "context child-resource counter overflow");
  }

  auto* copy = new (copy_storage)
      RustInferCudaCopy(device->owner, stream, device, host);
  RustInferCudaErrorInfo operation_error{};
  operation_error.struct_size = sizeof(operation_error);
  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  bool copy_attempted = false;
  {
    CurrentContext scope(device->owner);
    status = scope.enter(&operation_error, RUSTINFER_CUDA_ERROR_STAGE_COPY,
                         "submit CUDA copy");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      auto* device_bytes = static_cast<uint8_t*>(device->device_data);
      auto* host_bytes = static_cast<uint8_t*>(host->host_data);
      const size_t native_device_offset = static_cast<size_t>(device_offset);
      const size_t native_host_offset = static_cast<size_t>(host_offset);
      void* destination =
          direction == cudaMemcpyHostToDevice
              ? static_cast<void*>(device_bytes + native_device_offset)
              : static_cast<void*>(host_bytes + native_host_offset);
      const void* source = direction == cudaMemcpyHostToDevice
                               ? static_cast<const void*>(host_bytes +
                                                          native_host_offset)
                               : static_cast<const void*>(device_bytes +
                                                          native_device_offset);
      copy_attempted = true;
      status = runtime_error(
          cudaMemcpyAsync(destination, source, static_cast<size_t>(byte_len),
                          direction, stream->stream),
          &operation_error, RUSTINFER_CUDA_ERROR_STAGE_COPY,
          direction == cudaMemcpyHostToDevice ? "enqueue pinned host-to-device copy"
                                               : "enqueue device-to-pinned-host copy");
    }
    status = scope.leave(status, &operation_error,
                         RUSTINFER_CUDA_ERROR_STAGE_COPY,
                         "submit CUDA copy");
  }

  if (!copy_attempted) {
    copy->~RustInferCudaCopy();
    std::free(copy);
    (void)release_child(device->owner);
    (void)release_copy(stream->active_uses);
    (void)release_copy(host->active_uses);
    (void)release_copy(device->active_uses);
    publish_error(error, operation_error);
    return status;
  }

  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    copy->deferred_status = status;
    copy->deferred_error = operation_error;
  }
  clear_error(error);
  *out_copy = copy;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_device_buffer_create(
    RustInferCudaContext* context, uint64_t byte_len,
    RustInferCudaDeviceBuffer** out_buffer,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA device buffer", "out_buffer is null");
  }
  *out_buffer = nullptr;
  if (context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA device buffer", "context is null");
  }
  if (byte_len > SIZE_MAX) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA device buffer",
                            "byte_len does not fit host size_t");
  }
  void* storage = std::calloc(1, sizeof(RustInferCudaDeviceBuffer));
  if (storage == nullptr) {
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                     "create CUDA device buffer", "host allocation failed");
  }

  void* allocation = nullptr;
  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  if (byte_len != 0) {
    CurrentContext scope(context);
    status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                         "create CUDA device buffer");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      const cudaError_t result =
          cudaMalloc(&allocation, static_cast<size_t>(byte_len));
      status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                             "allocate CUDA device buffer");
    }
    status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                         "create CUDA device buffer");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    if (!free_device_after_failed_create(context, allocation)) {
      preserve_unresolved_allocation(
          context, context->device_live_bytes,
          context->device_live_allocations, byte_len);
    }
    std::free(storage);
    return status;
  }
  if (!account_allocation(context, context->device_live_bytes,
                          context->device_live_allocations, byte_len)) {
    if (!free_device_after_failed_create(context, allocation)) {
      preserve_unresolved_allocation(
          context, context->device_live_bytes,
          context->device_live_allocations, byte_len);
    }
    std::free(storage);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA device buffer",
                          "device allocation accounting overflow");
  }
  if (!retain_child(context)) {
    if (free_device_after_failed_create(context, allocation)) {
      (void)release_allocation(context, context->device_live_bytes,
                               context->device_live_allocations, byte_len);
    }
    std::free(storage);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA device buffer",
                          "context child-resource counter overflow");
  }
  auto* buffer =
      new (storage) RustInferCudaDeviceBuffer(context, allocation, byte_len);
  *out_buffer = buffer;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_device_buffer_close(
    RustInferCudaDeviceBuffer** buffer,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA device buffer", "buffer pointer is null");
  }
  if (*buffer == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if ((*buffer)->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA device buffer",
                            "device buffer still has an active asynchronous use");
  }

  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  bool free_attempted = false;
  bool free_confirmed = (*buffer)->device_data == nullptr;
  if ((*buffer)->device_data != nullptr) {
    CurrentContext scope((*buffer)->owner);
    status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                         "close CUDA device buffer");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      free_attempted = true;
      const cudaError_t result = cudaFree((*buffer)->device_data);
      free_confirmed = result == cudaSuccess;
      status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                             "free CUDA device buffer");
      // cudaFree may surface an earlier asynchronous error after performing
      // its destructive side effect. Never retry an attempted free.
      (*buffer)->device_data = nullptr;
    }
    status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                         "close CUDA device buffer");
  }
  const bool consume = free_confirmed || free_attempted;
  if (consume) {
    RustInferCudaContext* owner = (*buffer)->owner;
    const uint64_t byte_len = (*buffer)->byte_len;
    bool accounted = true;
    bool released = true;
    if (free_confirmed) {
      accounted = release_allocation(owner, owner->device_live_bytes,
                                     owner->device_live_allocations, byte_len);
      released = release_child(owner);
    }
    (*buffer)->~RustInferCudaDeviceBuffer();
    std::free(*buffer);
    *buffer = nullptr;
    // An ambiguous failed free intentionally leaves allocation accounting and
    // the native context-child lease live. That fail-closed leak prevents the
    // primary context from being released around possibly live device memory.
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS && (!accounted || !released)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA device buffer",
                            "allocation or context accounting underflow");
    }
  }
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_create(
    RustInferCudaContext* context, uint64_t byte_len,
    RustInferCudaPinnedHostBuffer** out_buffer,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create pinned host buffer", "out_buffer is null");
  }
  *out_buffer = nullptr;
  if (context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create pinned host buffer", "context is null");
  }
  if (byte_len > SIZE_MAX) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create pinned host buffer",
                            "byte_len does not fit host size_t");
  }
  void* storage = std::calloc(1, sizeof(RustInferCudaPinnedHostBuffer));
  if (storage == nullptr) {
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                     "create pinned host buffer", "host allocation failed");
  }

  void* allocation = nullptr;
  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  if (byte_len != 0) {
    CurrentContext scope(context);
    status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                         "create pinned host buffer");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      const cudaError_t result = cudaHostAlloc(
          &allocation, static_cast<size_t>(byte_len), cudaHostAllocDefault);
      status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                             "allocate pinned host buffer");
    }
    status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                         "create pinned host buffer");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    if (!free_pinned_after_failed_create(context, allocation)) {
      preserve_unresolved_allocation(
          context, context->pinned_host_live_bytes,
          context->pinned_host_live_allocations, byte_len);
    }
    std::free(storage);
    return status;
  }
  if (!account_allocation(context, context->pinned_host_live_bytes,
                          context->pinned_host_live_allocations, byte_len)) {
    if (!free_pinned_after_failed_create(context, allocation)) {
      preserve_unresolved_allocation(
          context, context->pinned_host_live_bytes,
          context->pinned_host_live_allocations, byte_len);
    }
    std::free(storage);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create pinned host buffer",
                          "pinned allocation accounting overflow");
  }
  if (!retain_child(context)) {
    if (free_pinned_after_failed_create(context, allocation)) {
      (void)release_allocation(context, context->pinned_host_live_bytes,
                               context->pinned_host_live_allocations, byte_len);
    }
    std::free(storage);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create pinned host buffer",
                          "context child-resource counter overflow");
  }
  auto* buffer =
      new (storage) RustInferCudaPinnedHostBuffer(context, allocation, byte_len);
  *out_buffer = buffer;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_write(
    RustInferCudaPinnedHostBuffer* buffer, uint64_t destination_offset,
    const uint8_t* source, uint64_t source_len,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr || (source_len != 0 && source == nullptr)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "write pinned host buffer",
                            "buffer or non-empty source is null");
  }
  if (!valid_range(buffer->byte_len, destination_offset, source_len)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "write pinned host buffer",
                            "write exceeds pinned host buffer range");
  }
  if (buffer->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "write pinned host buffer",
                            "pinned host buffer has an active copy token");
  }
  if (source_len != 0) {
    auto* destination = static_cast<uint8_t*>(buffer->host_data);
    std::memcpy(destination + destination_offset, source,
                static_cast<size_t>(source_len));
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_read(
    RustInferCudaPinnedHostBuffer* buffer, uint64_t source_offset,
    uint8_t* destination, uint64_t destination_len,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr || (destination_len != 0 && destination == nullptr)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "read pinned host buffer",
                            "buffer or non-empty destination is null");
  }
  if (!valid_range(buffer->byte_len, source_offset, destination_len)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "read pinned host buffer",
                            "read exceeds pinned host buffer range");
  }
  if (buffer->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "read pinned host buffer",
                            "pinned host buffer has an active copy token");
  }
  if (destination_len != 0) {
    const auto* source = static_cast<const uint8_t*>(buffer->host_data);
    std::memcpy(destination, source + source_offset,
                static_cast<size_t>(destination_len));
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_pinned_host_buffer_close(
    RustInferCudaPinnedHostBuffer** buffer,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close pinned host buffer", "buffer pointer is null");
  }
  if (*buffer == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if ((*buffer)->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close pinned host buffer",
                            "pinned host buffer still has an active copy token");
  }

  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  bool free_attempted = false;
  bool free_confirmed = (*buffer)->host_data == nullptr;
  if ((*buffer)->host_data != nullptr) {
    CurrentContext scope((*buffer)->owner);
    status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                         "close pinned host buffer");
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      free_attempted = true;
      const cudaError_t result = cudaFreeHost((*buffer)->host_data);
      free_confirmed = result == cudaSuccess;
      status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                             "free pinned host buffer");
      // cudaFreeHost is likewise single-shot when its reported error may be
      // deferred from earlier work.
      (*buffer)->host_data = nullptr;
    }
    status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                         "close pinned host buffer");
  }
  const bool consume = free_confirmed || free_attempted;
  if (consume) {
    RustInferCudaContext* owner = (*buffer)->owner;
    const uint64_t byte_len = (*buffer)->byte_len;
    bool accounted = true;
    bool released = true;
    if (free_confirmed) {
      accounted = release_allocation(owner, owner->pinned_host_live_bytes,
                                     owner->pinned_host_live_allocations,
                                     byte_len);
      released = release_child(owner);
    }
    (*buffer)->~RustInferCudaPinnedHostBuffer();
    std::free(*buffer);
    *buffer = nullptr;
    // Preserve non-zero logical accounting and the context lease if the
    // destructive result was ambiguous; reporting zero would be unsafe.
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS && (!accounted || !released)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close pinned host buffer",
                            "allocation or context accounting underflow");
    }
  }
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_copy_h2d_async(
    RustInferCudaDeviceBuffer* destination, uint64_t destination_offset,
    RustInferCudaPinnedHostBuffer* source, uint64_t source_offset,
    uint64_t byte_len, RustInferCudaStream* stream,
    RustInferCudaCopy** out_copy, RustInferCudaErrorInfo* error) noexcept {
  return submit_copy(destination, destination_offset, source, source_offset,
                     byte_len, stream, cudaMemcpyHostToDevice, out_copy, error);
}

extern "C" RustInferCudaStatus rustinfer_cuda_copy_d2h_async(
    RustInferCudaPinnedHostBuffer* destination, uint64_t destination_offset,
    RustInferCudaDeviceBuffer* source, uint64_t source_offset,
    uint64_t byte_len, RustInferCudaStream* stream,
    RustInferCudaCopy** out_copy, RustInferCudaErrorInfo* error) noexcept {
  return submit_copy(source, source_offset, destination, destination_offset,
                     byte_len, stream, cudaMemcpyDeviceToHost, out_copy, error);
}

extern "C" RustInferCudaStatus rustinfer_cuda_copy_query(
    RustInferCudaCopy* copy, uint8_t* out_complete,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (copy == nullptr || out_complete == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA copy",
                            "copy or out_complete is null");
  }
  *out_complete = copy->completed ? 1 : 0;
  if (copy->completed) {
    return deferred_status(copy, error);
  }

  CurrentContext scope(copy->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_QUERY, "query CUDA copy");
  bool completion_observed = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaStreamQuery(copy->stream->stream);
    completion_observed = result == cudaSuccess;
    status = runtime_error(result, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA copy stream");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                       "query CUDA copy");
  if (!completion_observed || status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const bool released = release_copy_uses(copy);
  if (!released) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                          "query CUDA copy",
                          "copy active-use counter underflow");
  }
  *out_complete = 1;
  return deferred_status(copy, error);
}

extern "C" RustInferCudaStatus rustinfer_cuda_copy_synchronize(
    RustInferCudaCopy* copy, uint8_t* out_complete,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (copy == nullptr || out_complete == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA copy",
                            "copy or out_complete is null");
  }
  *out_complete = copy->completed ? 1 : 0;
  if (copy->completed) {
    return deferred_status(copy, error);
  }

  CurrentContext scope(copy->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize CUDA copy");
  bool completion_observed = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaStreamSynchronize(copy->stream->stream);
    completion_observed = result == cudaSuccess;
    status = runtime_error(result, error,
                           RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA copy stream");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                       "synchronize CUDA copy");
  if (!completion_observed || status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const bool released = release_copy_uses(copy);
  if (!released) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          "synchronize CUDA copy",
                          "copy active-use counter underflow");
  }
  *out_complete = 1;
  return deferred_status(copy, error);
}

extern "C" RustInferCudaStatus rustinfer_cuda_copy_close(
    RustInferCudaCopy** copy, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (copy == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA copy", "copy pointer is null");
  }
  if (*copy == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  uint8_t complete = (*copy)->completed ? 1 : 0;
  RustInferCudaStatus status = complete != 0
                                   ? deferred_status(*copy, error)
                                   : RUSTINFER_CUDA_STATUS_SUCCESS;
  if (complete == 0) {
    status = rustinfer_cuda_copy_synchronize(*copy, &complete, error);
  }
  if (complete != 0) {
    RustInferCudaContext* owner = (*copy)->owner;
    (*copy)->~RustInferCudaCopy();
    std::free(*copy);
    *copy = nullptr;
    if (!release_child(owner) && status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA copy",
                            "context child-resource counter underflow");
    }
  }
  return status;
}
