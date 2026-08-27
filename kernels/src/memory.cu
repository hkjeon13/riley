#include "ffi_internal.hpp"

#include <cstddef>
#include <cstdint>

namespace {

using riley_cuda_internal::AllocationStatsGuard;
using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_error;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::release_child;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::same_context;
using riley_cuda_internal::set_error;
using riley_cuda_internal::validation_error;

#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
struct MemoryFaultInjector final {
  std::atomic<RileyCudaContext*> owner{nullptr};
  std::atomic<uint32_t> armed{0};
  std::atomic<uint64_t> faults_fired{0};
  std::atomic<uint64_t> device_free_attempts{0};
  std::atomic<uint64_t> pinned_free_attempts{0};
  std::atomic<uint64_t> copy_use_release_attempts{0};
};

static_assert(sizeof(RileyCudaTestMemoryFaultStats) == 64,
              "test memory fault stats ABI drift");

MemoryFaultInjector g_memory_faults;

bool injector_owns(RileyCudaContext* context) noexcept {
  return g_memory_faults.owner.load(std::memory_order_acquire) == context;
}

bool fault_is_armed(RileyCudaContext* context, uint32_t fault) noexcept {
  return injector_owns(context) &&
         g_memory_faults.armed.load(std::memory_order_acquire) == fault;
}

bool consume_fault(RileyCudaContext* context, uint32_t fault) noexcept {
  if (!injector_owns(context)) {
    return false;
  }
  uint32_t expected = fault;
  if (!g_memory_faults.armed.compare_exchange_strong(
          expected, 0, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return false;
  }
  g_memory_faults.faults_fired.fetch_add(1, std::memory_order_relaxed);
  return true;
}

RileyCudaStatus injected_runtime_error(RileyCudaErrorInfo* error,
                                           uint32_t stage,
                                           const char* operation) noexcept {
  return set_error(error, RILEY_CUDA_STATUS_RUNTIME_ERROR,
                   static_cast<int32_t>(cudaErrorUnknown),
                   RILEY_CUDA_ERROR_DOMAIN_RUNTIME, stage, operation,
                   "test-injected ambiguous CUDA result");
}
#endif

bool valid_range(uint64_t total, uint64_t offset, uint64_t length) noexcept {
  return offset <= total && length <= total - offset;
}

void publish_error(RileyCudaErrorInfo* destination,
                   const RileyCudaErrorInfo& source) noexcept {
  if (destination == nullptr || destination->struct_size < sizeof(*destination)) {
    return;
  }
  const uint32_t struct_size = destination->struct_size;
  std::memcpy(destination, &source, sizeof(source));
  destination->struct_size = struct_size;
}

bool account_allocation(RileyCudaContext* context,
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

bool release_allocation(RileyCudaContext* context,
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

bool free_device_after_failed_create(RileyCudaContext* context,
                                     void* allocation) noexcept {
  if (allocation == nullptr) {
    return true;
  }
  CurrentContext cleanup(context);
  RileyCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  RileyCudaStatus status = cleanup.enter(
      &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
      "cleanup device allocation after create");
  bool free_confirmed = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
    if (injector_owns(context)) {
      g_memory_faults.device_free_attempts.fetch_add(1,
                                                      std::memory_order_relaxed);
    }
#endif
    const cudaError_t result = cudaFree(allocation);
    free_confirmed = result == cudaSuccess;
    status = runtime_error(result, &ignored,
                           RILEY_CUDA_ERROR_STAGE_CLOSE,
                           "cleanup device allocation after create");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
    if (consume_fault(
            context,
            RILEY_CUDA_TEST_MEMORY_FAULT_DEVICE_CREATE_ROLLBACK_AMBIGUOUS)) {
      free_confirmed = false;
      status = injected_runtime_error(
          &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
          "cleanup device allocation after injected create failure");
    }
#endif
  }
  (void)cleanup.leave(status, &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
                      "cleanup device allocation after create");
  return free_confirmed;
}

bool free_pinned_after_failed_create(RileyCudaContext* context,
                                     void* allocation) noexcept {
  if (allocation == nullptr) {
    return true;
  }
  CurrentContext cleanup(context);
  RileyCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  RileyCudaStatus status = cleanup.enter(
      &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
      "cleanup pinned allocation after create");
  bool free_confirmed = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
    if (injector_owns(context)) {
      g_memory_faults.pinned_free_attempts.fetch_add(1,
                                                      std::memory_order_relaxed);
    }
#endif
    const cudaError_t result = cudaFreeHost(allocation);
    free_confirmed = result == cudaSuccess;
    status = runtime_error(result, &ignored,
                           RILEY_CUDA_ERROR_STAGE_CLOSE,
                           "cleanup pinned allocation after create");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
    if (consume_fault(
            context,
            RILEY_CUDA_TEST_MEMORY_FAULT_PINNED_CREATE_ROLLBACK_AMBIGUOUS)) {
      free_confirmed = false;
      status = injected_runtime_error(
          &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
          "cleanup pinned allocation after injected create failure");
    }
#endif
  }
  (void)cleanup.leave(status, &ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
                      "cleanup pinned allocation after create");
  return free_confirmed;
}

void preserve_unresolved_allocation(
    RileyCudaContext* context, std::atomic<uint64_t>& live_bytes,
    std::atomic<uint64_t>& live_allocations, uint64_t byte_len) noexcept {
  // No caller-visible handle exists on a failed create, so an allocation whose
  // rollback cannot be confirmed must remain permanently accounted. Either
  // the stats or the child lease (normally both) independently prevents a
  // false-success primary-context release.
  (void)account_allocation(context, live_bytes, live_allocations, byte_len);
  (void)retain_child(context);
}

bool release_copy_uses(RileyCudaCopy* copy) noexcept {
  if (copy->completed) {
    return true;
  }
  if (copy->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      copy->device->active_uses.load(std::memory_order_acquire) != 1 ||
      copy->host->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
  if (injector_owns(copy->owner)) {
    g_memory_faults.copy_use_release_attempts.fetch_add(
        1, std::memory_order_relaxed);
  }
#endif
  // Raw C callers must serialize mutation of opaque handles. Under that ABI
  // rule this precheck makes the three-resource state transition all-or-none.
  copy->stream->active_uses.store(0, std::memory_order_release);
  copy->device->active_uses.store(0, std::memory_order_release);
  copy->host->active_uses.store(0, std::memory_order_release);
  copy->completed = true;
  return true;
}

RileyCudaStatus deferred_status(RileyCudaCopy* copy,
                                    RileyCudaErrorInfo* error) noexcept {
  if (copy->deferred_status != RILEY_CUDA_STATUS_SUCCESS) {
    publish_error(error, copy->deferred_error);
  }
  return copy->deferred_status;
}

RileyCudaStatus submit_copy(RileyCudaDeviceBuffer* device,
                                uint64_t device_offset,
                                RileyCudaPinnedHostBuffer* host,
                                uint64_t host_offset, uint64_t byte_len,
                                RileyCudaStream* stream,
                                cudaMemcpyKind direction,
                                RileyCudaCopy** out_copy,
                                RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_copy == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy", "out_copy is null");
  }
  *out_copy = nullptr;
  if (device == nullptr || host == nullptr || stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "device, host, or stream handle is null");
  }
  if (!same_context(device->owner, host->owner) ||
      !same_context(device->owner, stream->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "copy resources belong to different context owners");
  }
  if (device->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "submit CUDA copy",
        "CUDA context is poisoned by a prior restoration failure");
  }
  if (!valid_range(device->byte_len, device_offset, byte_len) ||
      !valid_range(host->byte_len, host_offset, byte_len)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "copy offset and length exceed a buffer range");
  }
  if (byte_len == 0) {
    if (device->active_uses.load(std::memory_order_acquire) != 0 ||
        host->active_uses.load(std::memory_order_acquire) != 0 ||
        stream->active_uses.load(std::memory_order_acquire) != 0) {
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION,
                              "submit CUDA copy",
                              "copy resource already has an active copy token");
    }
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  void* copy_storage = std::calloc(1, sizeof(RileyCudaCopy));
  if (copy_storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, "submit CUDA copy",
                     "host copy-token allocation failed");
  }
  if (!try_acquire_copy(device->active_uses)) {
    std::free(copy_storage);
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "device buffer already has an active copy token");
  }
  if (!try_acquire_copy(host->active_uses)) {
    (void)release_copy(device->active_uses);
    std::free(copy_storage);
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "pinned host buffer already has an active copy token");
  }
  if (!try_acquire_copy(stream->active_uses)) {
    (void)release_copy(host->active_uses);
    (void)release_copy(device->active_uses);
    std::free(copy_storage);
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "submit CUDA copy",
                            "stream already has an active copy token");
  }
  if (!retain_child(device->owner)) {
    (void)release_copy(stream->active_uses);
    (void)release_copy(host->active_uses);
    (void)release_copy(device->active_uses);
    std::free(copy_storage);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "submit CUDA copy",
                          "context child-resource counter overflow");
  }

  auto* copy = new (copy_storage)
      RileyCudaCopy(device->owner, stream, device, host);
  RileyCudaErrorInfo operation_error{};
  operation_error.struct_size = sizeof(operation_error);
  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  bool copy_attempted = false;
  {
    CurrentContext scope(device->owner);
    status = scope.enter(&operation_error, RILEY_CUDA_ERROR_STAGE_COPY,
                         "submit CUDA copy");
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
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
          &operation_error, RILEY_CUDA_ERROR_STAGE_COPY,
          direction == cudaMemcpyHostToDevice ? "enqueue pinned host-to-device copy"
                                               : "enqueue device-to-pinned-host copy");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (status == RILEY_CUDA_STATUS_SUCCESS &&
          consume_fault(
              device->owner,
              RILEY_CUDA_TEST_MEMORY_FAULT_COPY_DEFERRED_SUBMISSION_ERROR)) {
        status = injected_runtime_error(
            &operation_error, RILEY_CUDA_ERROR_STAGE_COPY,
            "enqueue test-injected deferred CUDA copy error");
      }
#endif
    }
    status = scope.leave(status, &operation_error,
                         RILEY_CUDA_ERROR_STAGE_COPY,
                         "submit CUDA copy");
  }

  if (!copy_attempted) {
    copy->~RileyCudaCopy();
    std::free(copy);
    (void)release_child(device->owner);
    (void)release_copy(stream->active_uses);
    (void)release_copy(host->active_uses);
    (void)release_copy(device->active_uses);
    publish_error(error, operation_error);
    return status;
  }

  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    copy->deferred_status = status;
    copy->deferred_error = operation_error;
  }
  clear_error(error);
  *out_copy = copy;
  return RILEY_CUDA_STATUS_SUCCESS;
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_device_buffer_create(
    RileyCudaContext* context, uint64_t byte_len,
    RileyCudaDeviceBuffer** out_buffer,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_buffer == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA device buffer", "out_buffer is null");
  }
  *out_buffer = nullptr;
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA device buffer", "context is null");
  }
  if (byte_len > SIZE_MAX) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA device buffer",
                            "byte_len does not fit host size_t");
  }
  void* storage = std::calloc(1, sizeof(RileyCudaDeviceBuffer));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     "create CUDA device buffer", "host allocation failed");
  }

  void* allocation = nullptr;
  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  if (byte_len != 0) {
    CurrentContext scope(context);
    status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                         "create CUDA device buffer");
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      const cudaError_t result =
          cudaMalloc(&allocation, static_cast<size_t>(byte_len));
      status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                             "allocate CUDA device buffer");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (status == RILEY_CUDA_STATUS_SUCCESS &&
          fault_is_armed(
              context,
              RILEY_CUDA_TEST_MEMORY_FAULT_DEVICE_CREATE_ROLLBACK_AMBIGUOUS)) {
        status = injected_runtime_error(
            error, RILEY_CUDA_ERROR_STAGE_CREATE,
            "create test-injected CUDA device buffer failure");
      }
#endif
    }
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                         "create CUDA device buffer");
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
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
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA device buffer",
                          "device allocation accounting overflow");
  }
  if (!retain_child(context)) {
    if (free_device_after_failed_create(context, allocation)) {
      (void)release_allocation(context, context->device_live_bytes,
                               context->device_live_allocations, byte_len);
    }
    std::free(storage);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA device buffer",
                          "context child-resource counter overflow");
  }
  auto* buffer =
      new (storage) RileyCudaDeviceBuffer(context, allocation, byte_len);
  *out_buffer = buffer;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_device_buffer_close(
    RileyCudaDeviceBuffer** buffer,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA device buffer", "buffer pointer is null");
  }
  if (*buffer == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if ((*buffer)->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA device buffer",
                            "device buffer still has an active asynchronous use");
  }

  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  bool free_attempted = false;
  bool free_confirmed = (*buffer)->device_data == nullptr;
  if ((*buffer)->device_data != nullptr) {
    CurrentContext scope((*buffer)->owner);
    status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                         "close CUDA device buffer");
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      free_attempted = true;
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (injector_owns((*buffer)->owner)) {
        g_memory_faults.device_free_attempts.fetch_add(
            1, std::memory_order_relaxed);
      }
#endif
      const cudaError_t result = cudaFree((*buffer)->device_data);
      free_confirmed = result == cudaSuccess;
      status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                             "free CUDA device buffer");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (consume_fault(
              (*buffer)->owner,
              RILEY_CUDA_TEST_MEMORY_FAULT_DEVICE_CLOSE_AMBIGUOUS)) {
        free_confirmed = false;
        status = injected_runtime_error(
            error, RILEY_CUDA_ERROR_STAGE_CLOSE,
            "close test-injected CUDA device buffer");
      }
#endif
      // cudaFree may surface an earlier asynchronous error after performing
      // its destructive side effect. Never retry an attempted free.
      (*buffer)->device_data = nullptr;
    }
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                         "close CUDA device buffer");
  }
  const bool consume = free_confirmed || free_attempted;
  if (consume) {
    RileyCudaContext* owner = (*buffer)->owner;
    const uint64_t byte_len = (*buffer)->byte_len;
    bool accounted = true;
    bool released = true;
    if (free_confirmed) {
      accounted = release_allocation(owner, owner->device_live_bytes,
                                     owner->device_live_allocations, byte_len);
      released = release_child(owner);
    }
    (*buffer)->~RileyCudaDeviceBuffer();
    std::free(*buffer);
    *buffer = nullptr;
    // An ambiguous failed free intentionally leaves allocation accounting and
    // the native context-child lease live. That fail-closed leak prevents the
    // primary context from being released around possibly live device memory.
    if (status == RILEY_CUDA_STATUS_SUCCESS && (!accounted || !released)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA device buffer",
                            "allocation or context accounting underflow");
    }
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_pinned_host_buffer_create(
    RileyCudaContext* context, uint64_t byte_len,
    RileyCudaPinnedHostBuffer** out_buffer,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_buffer == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create pinned host buffer", "out_buffer is null");
  }
  *out_buffer = nullptr;
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create pinned host buffer", "context is null");
  }
  if (byte_len > SIZE_MAX) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create pinned host buffer",
                            "byte_len does not fit host size_t");
  }
  void* storage = std::calloc(1, sizeof(RileyCudaPinnedHostBuffer));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     "create pinned host buffer", "host allocation failed");
  }

  void* allocation = nullptr;
  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  if (byte_len != 0) {
    CurrentContext scope(context);
    status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                         "create pinned host buffer");
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      const cudaError_t result = cudaHostAlloc(
          &allocation, static_cast<size_t>(byte_len), cudaHostAllocDefault);
      status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                             "allocate pinned host buffer");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (status == RILEY_CUDA_STATUS_SUCCESS &&
          fault_is_armed(
              context,
              RILEY_CUDA_TEST_MEMORY_FAULT_PINNED_CREATE_ROLLBACK_AMBIGUOUS)) {
        status = injected_runtime_error(
            error, RILEY_CUDA_ERROR_STAGE_CREATE,
            "create test-injected CUDA pinned host buffer failure");
      }
#endif
    }
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                         "create pinned host buffer");
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
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
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "create pinned host buffer",
                          "pinned allocation accounting overflow");
  }
  if (!retain_child(context)) {
    if (free_pinned_after_failed_create(context, allocation)) {
      (void)release_allocation(context, context->pinned_host_live_bytes,
                               context->pinned_host_live_allocations, byte_len);
    }
    std::free(storage);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "create pinned host buffer",
                          "context child-resource counter overflow");
  }
  auto* buffer =
      new (storage) RileyCudaPinnedHostBuffer(context, allocation, byte_len);
  *out_buffer = buffer;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_pinned_host_buffer_write(
    RileyCudaPinnedHostBuffer* buffer, uint64_t destination_offset,
    const uint8_t* source, uint64_t source_len,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr || (source_len != 0 && source == nullptr)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "write pinned host buffer",
                            "buffer or non-empty source is null");
  }
  if (!valid_range(buffer->byte_len, destination_offset, source_len)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "write pinned host buffer",
                            "write exceeds pinned host buffer range");
  }
  if (buffer->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "write pinned host buffer",
                            "pinned host buffer has an active copy token");
  }
  if (source_len != 0) {
    auto* destination = static_cast<uint8_t*>(buffer->host_data);
    std::memcpy(destination + destination_offset, source,
                static_cast<size_t>(source_len));
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_pinned_host_buffer_read(
    RileyCudaPinnedHostBuffer* buffer, uint64_t source_offset,
    uint8_t* destination, uint64_t destination_len,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr || (destination_len != 0 && destination == nullptr)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "read pinned host buffer",
                            "buffer or non-empty destination is null");
  }
  if (!valid_range(buffer->byte_len, source_offset, destination_len)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "read pinned host buffer",
                            "read exceeds pinned host buffer range");
  }
  if (buffer->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "read pinned host buffer",
                            "pinned host buffer has an active copy token");
  }
  if (destination_len != 0) {
    const auto* source = static_cast<const uint8_t*>(buffer->host_data);
    std::memcpy(destination, source + source_offset,
                static_cast<size_t>(destination_len));
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_pinned_host_buffer_close(
    RileyCudaPinnedHostBuffer** buffer,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (buffer == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close pinned host buffer", "buffer pointer is null");
  }
  if (*buffer == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if ((*buffer)->active_uses.load(std::memory_order_acquire) != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close pinned host buffer",
                            "pinned host buffer still has an active copy token");
  }

  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  bool free_attempted = false;
  bool free_confirmed = (*buffer)->host_data == nullptr;
  if ((*buffer)->host_data != nullptr) {
    CurrentContext scope((*buffer)->owner);
    status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                         "close pinned host buffer");
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      free_attempted = true;
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (injector_owns((*buffer)->owner)) {
        g_memory_faults.pinned_free_attempts.fetch_add(
            1, std::memory_order_relaxed);
      }
#endif
      const cudaError_t result = cudaFreeHost((*buffer)->host_data);
      free_confirmed = result == cudaSuccess;
      status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                             "free pinned host buffer");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
      if (consume_fault(
              (*buffer)->owner,
              RILEY_CUDA_TEST_MEMORY_FAULT_PINNED_CLOSE_AMBIGUOUS)) {
        free_confirmed = false;
        status = injected_runtime_error(
            error, RILEY_CUDA_ERROR_STAGE_CLOSE,
            "close test-injected CUDA pinned host buffer");
      }
#endif
      // cudaFreeHost is likewise single-shot when its reported error may be
      // deferred from earlier work.
      (*buffer)->host_data = nullptr;
    }
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                         "close pinned host buffer");
  }
  const bool consume = free_confirmed || free_attempted;
  if (consume) {
    RileyCudaContext* owner = (*buffer)->owner;
    const uint64_t byte_len = (*buffer)->byte_len;
    bool accounted = true;
    bool released = true;
    if (free_confirmed) {
      accounted = release_allocation(owner, owner->pinned_host_live_bytes,
                                     owner->pinned_host_live_allocations,
                                     byte_len);
      released = release_child(owner);
    }
    (*buffer)->~RileyCudaPinnedHostBuffer();
    std::free(*buffer);
    *buffer = nullptr;
    // Preserve non-zero logical accounting and the context lease if the
    // destructive result was ambiguous; reporting zero would be unsafe.
    if (status == RILEY_CUDA_STATUS_SUCCESS && (!accounted || !released)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close pinned host buffer",
                            "allocation or context accounting underflow");
    }
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_copy_h2d_async(
    RileyCudaDeviceBuffer* destination, uint64_t destination_offset,
    RileyCudaPinnedHostBuffer* source, uint64_t source_offset,
    uint64_t byte_len, RileyCudaStream* stream,
    RileyCudaCopy** out_copy, RileyCudaErrorInfo* error) noexcept {
  return submit_copy(destination, destination_offset, source, source_offset,
                     byte_len, stream, cudaMemcpyHostToDevice, out_copy, error);
}

extern "C" RileyCudaStatus riley_cuda_copy_d2h_async(
    RileyCudaPinnedHostBuffer* destination, uint64_t destination_offset,
    RileyCudaDeviceBuffer* source, uint64_t source_offset,
    uint64_t byte_len, RileyCudaStream* stream,
    RileyCudaCopy** out_copy, RileyCudaErrorInfo* error) noexcept {
  return submit_copy(source, source_offset, destination, destination_offset,
                     byte_len, stream, cudaMemcpyDeviceToHost, out_copy, error);
}

extern "C" RileyCudaStatus riley_cuda_copy_query(
    RileyCudaCopy* copy, uint8_t* out_complete,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (copy == nullptr || out_complete == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA copy",
                            "copy or out_complete is null");
  }
  *out_complete = copy->completed ? 1 : 0;
  if (copy->completed) {
    return deferred_status(copy, error);
  }

  CurrentContext scope(copy->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_QUERY, "query CUDA copy");
  bool completion_observed = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaStreamQuery(copy->stream->stream);
    completion_observed = result == cudaSuccess;
    status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA copy stream");
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                       "query CUDA copy");
  if (!completion_observed || status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const bool released = release_copy_uses(copy);
  if (!released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query CUDA copy",
                          "copy active-use counter underflow");
  }
  *out_complete = 1;
  return deferred_status(copy, error);
}

extern "C" RileyCudaStatus riley_cuda_copy_synchronize(
    RileyCudaCopy* copy, uint8_t* out_complete,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (copy == nullptr || out_complete == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA copy",
                            "copy or out_complete is null");
  }
  *out_complete = copy->completed ? 1 : 0;
  if (copy->completed) {
    return deferred_status(copy, error);
  }

  CurrentContext scope(copy->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize CUDA copy");
  bool completion_observed = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaStreamSynchronize(copy->stream->stream);
    completion_observed = result == cudaSuccess;
    status = runtime_error(result, error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA copy stream");
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
    if (status == RILEY_CUDA_STATUS_SUCCESS &&
        consume_fault(
            copy->owner,
            RILEY_CUDA_TEST_MEMORY_FAULT_COPY_COMPLETION_RESTORE_AMBIGUOUS)) {
      // Model an unconfirmable current-context restoration after the device work
      // has completed. Completion is deliberately not published: resources and
      // the context remain poisoned/busy rather than risking early reuse.
      copy->owner->restoration_failed.store(true, std::memory_order_release);
      completion_observed = false;
      status = injected_runtime_error(
          error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
          "synchronize test-injected CUDA copy with ambiguous restoration");
    }
#endif
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                       "synchronize CUDA copy");
  if (!completion_observed || status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const bool released = release_copy_uses(copy);
  if (!released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          "synchronize CUDA copy",
                          "copy active-use counter underflow");
  }
  *out_complete = 1;
  return deferred_status(copy, error);
}

extern "C" RileyCudaStatus riley_cuda_copy_close(
    RileyCudaCopy** copy, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (copy == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA copy", "copy pointer is null");
  }
  if (*copy == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  uint8_t complete = (*copy)->completed ? 1 : 0;
  RileyCudaStatus status = complete != 0
                                   ? deferred_status(*copy, error)
                                   : RILEY_CUDA_STATUS_SUCCESS;
  if (complete == 0) {
    status = riley_cuda_copy_synchronize(*copy, &complete, error);
  }
  if (complete != 0) {
    RileyCudaContext* owner = (*copy)->owner;
    (*copy)->~RileyCudaCopy();
    std::free(*copy);
    *copy = nullptr;
    if (!release_child(owner) && status == RILEY_CUDA_STATUS_SUCCESS) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            "close CUDA copy",
                            "context child-resource counter underflow");
    }
  }
  return status;
}

#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
extern "C" RileyCudaStatus riley_cuda_test_memory_fault_reset(
    RileyCudaContext* context, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "reset CUDA memory fault injector",
                            "context is null");
  }
  g_memory_faults.armed.store(0, std::memory_order_release);
  g_memory_faults.faults_fired.store(0, std::memory_order_relaxed);
  g_memory_faults.device_free_attempts.store(0, std::memory_order_relaxed);
  g_memory_faults.pinned_free_attempts.store(0, std::memory_order_relaxed);
  g_memory_faults.copy_use_release_attempts.store(0,
                                                   std::memory_order_relaxed);
  g_memory_faults.owner.store(context, std::memory_order_release);
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_test_memory_fault_arm(
    RileyCudaContext* context, uint32_t fault,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr || !injector_owns(context)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "arm CUDA memory fault injector",
                            "injector is not reset for this context");
  }
  if (fault <
          RILEY_CUDA_TEST_MEMORY_FAULT_DEVICE_CREATE_ROLLBACK_AMBIGUOUS ||
      fault >
          RILEY_CUDA_TEST_MEMORY_FAULT_COPY_COMPLETION_RESTORE_AMBIGUOUS) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "arm CUDA memory fault injector",
                            "unknown fault identifier");
  }
  uint32_t expected = 0;
  if (!g_memory_faults.armed.compare_exchange_strong(
          expected, fault, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "arm CUDA memory fault injector",
                            "another one-shot fault is still armed");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_test_memory_fault_stats(
    RileyCudaContext* context, RileyCudaTestMemoryFaultStats* out_stats,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr || !injector_owns(context) || out_stats == nullptr ||
      out_stats->struct_size < sizeof(*out_stats)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA memory fault injector",
                            "context/session/output is invalid");
  }
  const uint32_t struct_size = out_stats->struct_size;
  std::memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = struct_size;
  out_stats->armed_fault =
      g_memory_faults.armed.load(std::memory_order_acquire);
  out_stats->faults_fired =
      g_memory_faults.faults_fired.load(std::memory_order_relaxed);
  out_stats->device_free_attempts =
      g_memory_faults.device_free_attempts.load(std::memory_order_relaxed);
  out_stats->pinned_free_attempts =
      g_memory_faults.pinned_free_attempts.load(std::memory_order_relaxed);
  out_stats->copy_use_release_attempts =
      g_memory_faults.copy_use_release_attempts.load(std::memory_order_relaxed);
  return RILEY_CUDA_STATUS_SUCCESS;
}
#endif
