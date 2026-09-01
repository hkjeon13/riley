#include "ffi_internal.hpp"

#include <climits>
#if defined(RILEY_CUDA_ENABLE_NVML_PROBE)
#include <nvml.h>
#endif

namespace {

using riley_cuda_internal::AllocationStatsGuard;
using riley_cuda_internal::CaptureDeferredCloseEnqueueResult;
using riley_cuda_internal::CaptureDomainControlLease;
using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::capture_domain_for_device;
using riley_cuda_internal::clear_error;
using riley_cuda_internal::command_batch_thread_token;
using riley_cuda_internal::driver_error;
using riley_cuda_internal::enqueue_capture_deferred_close;
using riley_cuda_internal::initialize_capture_deferred_close_node;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::release_exclusive_use;
using riley_cuda_internal::release_thread_command_batch;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::release_child;
using riley_cuda_internal::same_context;
using riley_cuda_internal::set_error;
using riley_cuda_internal::thread_has_active_graph_capture;
using riley_cuda_internal::thread_graph_capture_is_owner;
using riley_cuda_internal::try_acquire_exclusive_use;
using riley_cuda_internal::try_publish_thread_command_batch;
using riley_cuda_internal::validation_error;

#if defined(RILEY_CUDA_ENABLE_NVML_PROBE)
RileyCudaStatus nvml_error(nvmlReturn_t result,
                               RileyCudaErrorInfo* error, uint32_t stage,
                               const char* operation) noexcept {
  if (result == NVML_SUCCESS) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaStatus status = RILEY_CUDA_STATUS_RUNTIME_ERROR;
  if (result == NVML_ERROR_INVALID_ARGUMENT) {
    status = RILEY_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
    status = RILEY_CUDA_STATUS_OUT_OF_RANGE;
  } else if (result == NVML_ERROR_NOT_SUPPORTED) {
    status = RILEY_CUDA_STATUS_NOT_SUPPORTED;
  } else if (result == NVML_ERROR_DRIVER_NOT_LOADED ||
             result == NVML_ERROR_LIBRARY_NOT_FOUND ||
             result == NVML_ERROR_FUNCTION_NOT_FOUND ||
             result == NVML_ERROR_GPU_IS_LOST) {
    status = RILEY_CUDA_STATUS_DRIVER_ERROR;
  }
  return set_error(error, status, static_cast<int32_t>(result),
                   RILEY_CUDA_ERROR_DOMAIN_NVML, stage, operation,
                   nvmlErrorString(result));
}

class NvmlSession final {
 public:
  NvmlSession() noexcept : active_(false) {}
  NvmlSession(const NvmlSession&) = delete;
  NvmlSession& operator=(const NvmlSession&) = delete;

  ~NvmlSession() noexcept {
    if (active_) {
      (void)nvmlShutdown();
    }
  }

  RileyCudaStatus initialize(RileyCudaErrorInfo* error) noexcept {
    const nvmlReturn_t result = nvmlInit_v2();
    if (result == NVML_SUCCESS) {
      active_ = true;
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                      "initialize NVML");
  }

  RileyCudaStatus shutdown(RileyCudaStatus primary_status,
                               RileyCudaErrorInfo* error) noexcept {
    if (!active_) {
      return primary_status;
    }
    const nvmlReturn_t result = nvmlShutdown();
    active_ = false;
    if (primary_status != RILEY_CUDA_STATUS_SUCCESS) {
      return primary_status;
    }
    return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                      "shutdown NVML after environment probe");
  }

 private:
  bool active_;
};
#endif

void clear_nvidia_environment_snapshot(
    RileyCudaNvidiaEnvironmentSnapshot* snapshot) noexcept {
  if (snapshot == nullptr || snapshot->struct_size < sizeof(*snapshot)) {
    return;
  }
  std::memset(snapshot, 0, sizeof(*snapshot));
  snapshot->struct_size = sizeof(*snapshot);
}

#if defined(RILEY_CUDA_ENABLE_NVML_PROBE)
RileyCudaStatus optional_application_clock(
    nvmlDevice_t device, nvmlClockType_t clock_type, uint32_t* output,
    RileyCudaErrorInfo* error, const char* operation) noexcept {
  unsigned int clock_mhz = 0;
  const nvmlReturn_t result =
      nvmlDeviceGetApplicationsClock(device, clock_type, &clock_mhz);
  if (result == NVML_ERROR_NOT_SUPPORTED) {
    *output = RILEY_CUDA_NVIDIA_CLOCK_NOT_AVAILABLE;
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  const RileyCudaStatus status =
      nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY, operation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    *output = static_cast<uint32_t>(clock_mhz);
  }
  return status;
}
#endif

RileyCudaStatus device_attribute(CUdevice device,
                                     CUdevice_attribute attribute,
                                     uint32_t* output,
                                     RileyCudaErrorInfo* error,
                                     const char* operation) noexcept {
  int value = 0;
  const CUresult result = cuDeviceGetAttribute(&value, attribute, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        operation);
  }
  if (value < 0) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                          operation, "CUDA returned a negative device attribute");
  }
  *output = static_cast<uint32_t>(value);
  return RILEY_CUDA_STATUS_SUCCESS;
}

void destroy_stream_after_failed_create(RileyCudaContext* context,
                                        cudaStream_t stream) noexcept {
  CurrentContext cleanup(context);
  RileyCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  if (cleanup.enter(&ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
                    "cleanup stream after create") ==
      RILEY_CUDA_STATUS_SUCCESS) {
    (void)cudaStreamDestroy(stream);
    (void)cleanup.leave(RILEY_CUDA_STATUS_SUCCESS, &ignored,
                        RILEY_CUDA_ERROR_STAGE_CLOSE,
                        "cleanup stream after create");
  }
}

void destroy_event_after_failed_create(RileyCudaContext* context,
                                       cudaEvent_t event) noexcept {
  CurrentContext cleanup(context);
  RileyCudaErrorInfo ignored{};
  ignored.struct_size = sizeof(ignored);
  if (cleanup.enter(&ignored, RILEY_CUDA_ERROR_STAGE_CLOSE,
                    "cleanup event after create") ==
      RILEY_CUDA_STATUS_SUCCESS) {
    (void)cudaEventDestroy(event);
    (void)cleanup.leave(RILEY_CUDA_STATUS_SUCCESS, &ignored,
                        RILEY_CUDA_ERROR_STAGE_CLOSE,
                        "cleanup event after create");
  }
}

RileyCudaStatus validate_context_close_preconditions(
    RileyCudaContext* context, RileyCudaErrorInfo* error,
    const char* operation) noexcept {
  if (context->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, operation,
        "a prior CUDA context-stack restoration failed; refusing to release the primary-context lease");
  }
  const uint32_t live_children =
      context->live_children.load(std::memory_order_acquire);
  if (live_children != 0) {
    char detail[128]{};
    std::snprintf(detail, sizeof(detail),
                  "context still owns %u live stream/event/buffer/copy resources",
                  live_children);
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, operation, detail);
  }
  bool has_live_allocation_accounting = false;
  {
    const AllocationStatsGuard guard(context);
    has_live_allocation_accounting =
        context->device_live_bytes.load(std::memory_order_relaxed) != 0 ||
        context->device_live_allocations.load(std::memory_order_relaxed) != 0 ||
        context->pinned_host_live_bytes.load(std::memory_order_relaxed) != 0 ||
        context->pinned_host_live_allocations.load(std::memory_order_relaxed) != 0;
  }
  if (has_live_allocation_accounting) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, operation,
        "context allocation accounting is non-zero; refusing to release the primary-context lease");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus context_close_impl(
    RileyCudaContext** context, RileyCudaErrorInfo* error,
    const RileyCudaGraphCapture* capture_owner) noexcept {
  constexpr const char* kOperation = "close CUDA context";
  clear_error(error);
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "context pointer is null");
  }
  if (*context == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (capture_owner == nullptr) {
    if (thread_has_active_graph_capture()) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
          "this host thread has an active thread-local CUDA Graph capture");
    }
  } else if (!thread_graph_capture_is_owner(capture_owner) ||
             !capture_owner->capture_terminated) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
        "deferred context close requires this exact physically terminated CUDA Graph capture");
  }
  const CaptureDomainControlLease capture_control(
      (*context)->capture_domain, capture_owner);
  if (!capture_control.active()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
        "the CUDA primary context has an active graph capture or broad control operation");
  }
  RileyCudaStatus status =
      validate_context_close_preconditions(*context, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const CUresult result = cuDevicePrimaryCtxRelease((*context)->device);
  status = driver_error(result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                        "release CUDA primary context");
  // Driver release may report an earlier asynchronous error after decrementing
  // the primary-context refcount. Consume the wrapper after the single release
  // attempt; a genuine failure becomes a safe lease leak, never a double
  // release of another module's shared primary-context ownership.
  (*context)->~RileyCudaContext();
  std::free(*context);
  *context = nullptr;
  return status;
}

RileyCudaDeferredCloseResult deferred_context_close(
    RileyCudaDeferredCloseNode* node,
    const RileyCudaGraphCapture* capture_owner,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "drain deferred CUDA context close";
  if (node == nullptr || node->payload == nullptr || node->owner == nullptr) {
    return {validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                             "deferred context node is incomplete"),
            false};
  }
  auto* value = static_cast<RileyCudaContext*>(node->payload);
  if (value != node->owner) {
    return {validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                             "deferred context node owner does not match its payload"),
            false};
  }
  RileyCudaContext* raw = value;
  const RileyCudaStatus status =
      context_close_impl(&raw, error, capture_owner);
  return {status, raw == nullptr};
}

RileyCudaStatus stream_close_impl(
    RileyCudaStream** stream, RileyCudaErrorInfo* error,
    const RileyCudaGraphCapture* capture_owner) noexcept {
  constexpr const char* kOperation = "close CUDA stream";
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "stream pointer is null");
  }
  if (*stream == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if ((*stream)->active_uses.load(std::memory_order_acquire) != 0 ||
      (*stream)->command_batch_owner.load(std::memory_order_acquire) !=
          nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "stream still has an active asynchronous use");
  }
  CurrentContext scope((*stream)->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kOperation, capture_owner);
  bool destroy_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    destroy_attempted = true;
    status = runtime_error(cudaStreamDestroy((*stream)->stream), error,
                           RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kOperation);
  if (destroy_attempted) {
    const bool released = release_child((*stream)->owner);
    (*stream)->~RileyCudaStream();
    std::free(*stream);
    *stream = nullptr;
    if (status == RILEY_CUDA_STATUS_SUCCESS && !released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "context child-resource counter underflow");
    }
  }
  return status;
}

RileyCudaDeferredCloseResult deferred_stream_close(
    RileyCudaDeferredCloseNode* node,
    const RileyCudaGraphCapture* capture_owner,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "drain deferred CUDA stream close";
  if (node == nullptr || node->payload == nullptr || node->owner == nullptr) {
    return {validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                             "deferred stream node is incomplete"),
            false};
  }
  auto* value = static_cast<RileyCudaStream*>(node->payload);
  if (value->owner != node->owner) {
    return {validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                             "deferred stream node owner does not match its payload"),
            false};
  }
  RileyCudaStream* raw = value;
  const RileyCudaStatus status = stream_close_impl(&raw, error, capture_owner);
  return {status, raw == nullptr};
}

RileyCudaStatus event_close_impl(
    RileyCudaEvent** event, RileyCudaErrorInfo* error,
    const RileyCudaGraphCapture* capture_owner) noexcept {
  constexpr const char* kOperation = "close CUDA event";
  clear_error(error);
  if (event == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "event pointer is null");
  }
  if (*event == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  CurrentContext scope((*event)->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kOperation, capture_owner);
  bool destroy_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    destroy_attempted = true;
    status = runtime_error(cudaEventDestroy((*event)->event), error,
                           RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kOperation);
  if (destroy_attempted) {
    const bool released = release_child((*event)->owner);
    (*event)->~RileyCudaEvent();
    std::free(*event);
    *event = nullptr;
    if (status == RILEY_CUDA_STATUS_SUCCESS && !released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "context child-resource counter underflow");
    }
  }
  return status;
}

RileyCudaDeferredCloseResult deferred_event_close(
    RileyCudaDeferredCloseNode* node,
    const RileyCudaGraphCapture* capture_owner,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "drain deferred CUDA event close";
  if (node == nullptr || node->payload == nullptr || node->owner == nullptr) {
    return {validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                             "deferred event node is incomplete"),
            false};
  }
  auto* value = static_cast<RileyCudaEvent*>(node->payload);
  if (value->owner != node->owner) {
    return {validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                             "deferred event node owner does not match its payload"),
            false};
  }
  RileyCudaEvent* raw = value;
  const RileyCudaStatus status = event_close_impl(&raw, error, capture_owner);
  return {status, raw == nullptr};
}

}  // namespace

#if defined(RILEY_CUDA_ENABLE_NVML_PROBE)
extern "C" RileyCudaStatus riley_cuda_nvidia_environment_probe(
    RileyCudaNvidiaEnvironmentSnapshot* out_snapshot,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_snapshot == nullptr ||
      out_snapshot->struct_size < sizeof(*out_snapshot)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "probe NVIDIA environment",
        "out_snapshot is null or has an incompatible struct_size");
  }
  clear_nvidia_environment_snapshot(out_snapshot);

  NvmlSession session;
  RileyCudaStatus status = session.initialize(error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = [&]() noexcept -> RileyCudaStatus {
    nvmlReturn_t result = nvmlSystemGetDriverVersion(
        out_snapshot->driver_version,
        RILEY_CUDA_NVIDIA_DRIVER_VERSION_CAPACITY);
    if (result != NVML_SUCCESS) {
      return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                        "query NVIDIA driver version");
    }
    if (out_snapshot->driver_version[0] == '\0') {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                            "query NVIDIA driver version",
                            "NVML returned an empty driver version");
    }

    result = nvmlSystemGetCudaDriverVersion_v2(
        &out_snapshot->cuda_driver_api_version);
    if (result != NVML_SUCCESS) {
      return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                        "query CUDA Driver API version through NVML");
    }
    if (out_snapshot->cuda_driver_api_version <= 0) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                            "query CUDA Driver API version through NVML",
                            "NVML returned a non-positive CUDA version");
    }

    unsigned int device_count = 0;
    result = nvmlDeviceGetCount_v2(&device_count);
    if (result != NVML_SUCCESS) {
      return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                        "enumerate NVIDIA devices through NVML");
    }
    if (device_count > RILEY_CUDA_NVIDIA_ENVIRONMENT_MAX_DEVICES) {
      return validation_error(
          error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
          RILEY_CUDA_ERROR_STAGE_QUERY, "probe NVIDIA environment",
          "NVML device count exceeds the fixed environment snapshot capacity");
    }
    out_snapshot->device_count = static_cast<uint32_t>(device_count);

    uint32_t aggregate_process_count = 0;
    for (unsigned int ordinal = 0; ordinal < device_count; ++ordinal) {
      nvmlDevice_t device = nullptr;
      result = nvmlDeviceGetHandleByIndex_v2(ordinal, &device);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "select NVIDIA device through NVML");
      }

      RileyCudaNvidiaDeviceSnapshot* output =
          &out_snapshot->devices[ordinal];
      output->struct_size = sizeof(*output);
      output->application_graphics_clock_mhz =
          RILEY_CUDA_NVIDIA_CLOCK_NOT_AVAILABLE;
      output->application_memory_clock_mhz =
          RILEY_CUDA_NVIDIA_CLOCK_NOT_AVAILABLE;

      unsigned int index = 0;
      result = nvmlDeviceGetIndex(device, &index);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query NVIDIA device index");
      }
      if (index != ordinal) {
        return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                              "query NVIDIA device index",
                              "NVML device index disagrees with enumeration order");
      }
      output->index = static_cast<uint32_t>(index);

      result = nvmlDeviceGetName(device, output->name,
                                 RILEY_CUDA_DEVICE_NAME_CAPACITY);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query NVIDIA device name");
      }
      if (output->name[0] == '\0') {
        return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                              "query NVIDIA device name",
                              "NVML returned an empty device name");
      }

      nvmlMemory_v2_t memory{};
      memory.version = nvmlMemory_v2;
      result = nvmlDeviceGetMemoryInfo_v2(device, &memory);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query NVIDIA device memory");
      }
      output->total_memory_bytes = static_cast<uint64_t>(memory.total);
      output->used_memory_bytes = static_cast<uint64_t>(memory.used);
      if (output->total_memory_bytes == 0 || memory.reserved > memory.total ||
          memory.used > memory.total - memory.reserved ||
          output->used_memory_bytes > output->total_memory_bytes) {
        return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                              "query NVIDIA device memory",
                              "NVML returned inconsistent device memory");
      }

      unsigned int temperature_c = 0;
      result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU,
                                        &temperature_c);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query NVIDIA device temperature");
      }
      output->temperature_c = static_cast<uint32_t>(temperature_c);

      nvmlEnableState_t persistence_mode = NVML_FEATURE_DISABLED;
      result = nvmlDeviceGetPersistenceMode(device, &persistence_mode);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query NVIDIA persistence mode");
      }
      if (persistence_mode == NVML_FEATURE_DISABLED) {
        output->persistence_mode =
            RILEY_CUDA_NVIDIA_PERSISTENCE_DISABLED;
      } else if (persistence_mode == NVML_FEATURE_ENABLED) {
        output->persistence_mode = RILEY_CUDA_NVIDIA_PERSISTENCE_ENABLED;
      } else {
        return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                              "query NVIDIA persistence mode",
                              "NVML returned an unknown persistence mode");
      }

      unsigned int power_limit_milliwatts = 0;
      result =
          nvmlDeviceGetPowerManagementLimit(device, &power_limit_milliwatts);
      if (result != NVML_SUCCESS) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "query NVIDIA power limit");
      }
      output->power_limit_milliwatts =
          static_cast<uint32_t>(power_limit_milliwatts);

      RileyCudaStatus clock_status = optional_application_clock(
          device, NVML_CLOCK_GRAPHICS,
          &output->application_graphics_clock_mhz, error,
          "query NVIDIA application graphics clock");
      if (clock_status != RILEY_CUDA_STATUS_SUCCESS) {
        return clock_status;
      }
      clock_status = optional_application_clock(
          device, NVML_CLOCK_MEM, &output->application_memory_clock_mhz, error,
          "query NVIDIA application memory clock");
      if (clock_status != RILEY_CUDA_STATUS_SUCCESS) {
        return clock_status;
      }

      unsigned int process_count = 0;
      result = nvmlDeviceGetComputeRunningProcesses_v3(device, &process_count,
                                                       nullptr);
      if (result != NVML_SUCCESS && result != NVML_ERROR_INSUFFICIENT_SIZE) {
        return nvml_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          "count NVIDIA compute processes");
      }
      output->compute_process_count = static_cast<uint32_t>(process_count);
      if (UINT32_MAX - aggregate_process_count <
          output->compute_process_count) {
        return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                              "count NVIDIA compute processes",
                              "aggregate compute-process count overflowed");
      }
      aggregate_process_count += output->compute_process_count;
    }
    out_snapshot->compute_process_count = aggregate_process_count;
    return RILEY_CUDA_STATUS_SUCCESS;
  }();

  status = session.shutdown(status, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    clear_nvidia_environment_snapshot(out_snapshot);
  }
  return status;
}
#else
extern "C" RileyCudaStatus riley_cuda_nvidia_environment_probe(
    RileyCudaNvidiaEnvironmentSnapshot* out_snapshot,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_snapshot == nullptr ||
      out_snapshot->struct_size < sizeof(*out_snapshot)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "probe NVIDIA environment",
        "out_snapshot is null or has an incompatible struct_size");
  }
  clear_nvidia_environment_snapshot(out_snapshot);
  return set_error(
      error, RILEY_CUDA_STATUS_NOT_SUPPORTED, 0,
      RILEY_CUDA_ERROR_DOMAIN_NVML,
      RILEY_CUDA_ERROR_STAGE_INITIALIZE, "probe NVIDIA environment",
      "native archive was built without NVML probe support");
}
#endif

extern "C" RileyCudaStatus riley_cuda_device_count(
    uint32_t* out_count, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_count == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "device count", "out_count is null");
  }
  *out_count = 0;
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "device count",
        "this host thread has an active thread-local CUDA Graph capture");
  }
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "initialize CUDA driver");
  }
  int count = 0;
  result = cuDeviceGetCount(&count);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "enumerate CUDA devices");
  }
  if (count < 0) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                          "enumerate CUDA devices",
                          "CUDA returned a negative device count");
  }
  *out_count = static_cast<uint32_t>(count);
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_device_properties(
    int32_t ordinal, RileyCudaDeviceProperties* out_properties,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_properties == nullptr ||
      out_properties->struct_size < sizeof(*out_properties)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "query device properties",
        "out_properties is null or has an incompatible struct_size");
  }
  std::memset(out_properties, 0, sizeof(*out_properties));
  out_properties->struct_size = sizeof(*out_properties);
  out_properties->ordinal = ordinal;
  if (ordinal < 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_DEVICE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query device properties",
                            "device ordinal must be non-negative");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "query device properties",
        "this host thread has an active thread-local CUDA Graph capture");
  }

  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "initialize CUDA driver");
  }
  CUdevice device = 0;
  result = cuDeviceGet(&device, ordinal);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "select CUDA device");
  }
  result = cuDeviceGetName(out_properties->name,
                           RILEY_CUDA_DEVICE_NAME_CAPACITY, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "query CUDA device name");
  }
  size_t total_memory = 0;
  result = cuDeviceTotalMem(&total_memory, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "query CUDA device memory");
  }
  out_properties->total_memory_bytes = static_cast<uint64_t>(total_memory);

  RileyCudaStatus status = device_attribute(
      device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      &out_properties->compute_capability_major, error,
      "query compute capability major");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device,
                            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                            &out_properties->compute_capability_minor, error,
                            "query compute capability minor");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                            &out_properties->multiprocessor_count, error,
                            "query multiprocessor count");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                            &out_properties->warp_size, error,
                            "query warp size");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            &out_properties->max_threads_per_block, error,
                            "query maximum threads per block");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  result = cuDriverGetVersion(&out_properties->driver_version);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "query CUDA driver version");
  }
  const cudaError_t runtime_result =
      cudaRuntimeGetVersion(&out_properties->runtime_version);
  if (runtime_result != cudaSuccess) {
    return runtime_error(runtime_result, error,
                         RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                         "query CUDA Runtime version");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_context_create(
    int32_t ordinal, RileyCudaContext** out_context,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA context", "out_context is null");
  }
  *out_context = nullptr;
  if (ordinal < 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_DEVICE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA context",
                            "device ordinal must be non-negative");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "create CUDA context",
        "this host thread has an active thread-local CUDA Graph capture");
  }
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                        "initialize CUDA driver");
  }
  CUdevice device = 0;
  result = cuDeviceGet(&device, ordinal);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                        "select CUDA context device");
  }
  RileyCudaCaptureDomain* const capture_domain =
      capture_domain_for_device(device);
  if (capture_domain == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, "create CUDA context",
                     "host allocation failed for primary-context capture domain");
  }
  const CaptureDomainControlLease capture_control(capture_domain);
  if (!capture_control.active()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "create CUDA context",
        "the CUDA primary context has an active graph capture or broad control operation");
  }
  CUcontext primary = nullptr;
  result = cuDevicePrimaryCtxRetain(&primary, device);
  if (result != CUDA_SUCCESS) {
    return driver_error(result, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                        "retain CUDA primary context");
  }
  void* context_storage = std::calloc(1, sizeof(RileyCudaContext));
  if (context_storage == nullptr) {
    (void)cuDevicePrimaryCtxRelease(device);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, "create CUDA context",
                     "host allocation failed");
  }
  auto* context =
      new (context_storage) RileyCudaContext(device, primary, ordinal,
                                             capture_domain);

  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  bool context_stack_restored = false;
  {
    CurrentContext scope(context);
    status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                         "initialize CUDA context");
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = runtime_error(cudaFree(nullptr), error,
                             RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                             "initialize CUDA Runtime in primary context");
      status = scope.leave(status, error,
                           RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                           "initialize CUDA context");
    }
    if (scope.active()) {
      // A failed pop must not be followed by releasing storage still needed by
      // a current primary context. Retry once while preserving the first error.
      status = scope.leave(status, error,
                           RILEY_CUDA_ERROR_STAGE_INITIALIZE,
                           "restore CUDA context after initialization");
    }
    context_stack_restored = !scope.active();
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    if (!context_stack_restored ||
        context->restoration_failed.load(std::memory_order_acquire)) {
      // The driver rejected repeated restoration attempts. Keep the retained
      // context and wrapper alive rather than release ambiguous current-context
      // ownership. This catastrophic path intentionally leaks.
      return status;
    }
    (void)cuDevicePrimaryCtxRelease(device);
    context->~RileyCudaContext();
    std::free(context);
    return status;
  }
  *out_context = context;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_context_synchronize(
    RileyCudaContext* context, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA context", "context is null");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "synchronize CUDA context",
        "this host thread has an active thread-local CUDA Graph capture");
  }
  const CaptureDomainControlLease capture_control(context->capture_domain);
  if (!capture_control.active()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "synchronize CUDA context",
        "the CUDA primary context has an active graph capture or broad control operation");
  }
  CurrentContext scope(context);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, "synchronize CUDA context");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaDeviceSynchronize(), error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA context");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                     "synchronize CUDA context");
}

extern "C" RileyCudaStatus riley_cuda_context_memory_info(
    RileyCudaContext* context, uint64_t* out_free_bytes,
    uint64_t* out_total_bytes, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_free_bytes == nullptr || out_total_bytes == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA memory info",
                            "memory output pointer is null");
  }
  *out_free_bytes = 0;
  *out_total_bytes = 0;
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA memory info", "context is null");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "query CUDA memory info",
        "this host thread has an active thread-local CUDA Graph capture");
  }
  const CaptureDomainControlLease capture_control(context->capture_domain);
  if (!capture_control.active()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "query CUDA memory info",
        "the CUDA primary context has an active graph capture or broad control operation");
  }
  CurrentContext scope(context);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_QUERY, "query CUDA memory info");
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaMemGetInfo(&free_bytes, &total_bytes), error,
                           RILEY_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA memory info");
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                       "query CUDA memory info");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    *out_free_bytes = static_cast<uint64_t>(free_bytes);
    *out_total_bytes = static_cast<uint64_t>(total_bytes);
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_context_allocation_stats(
    RileyCudaContext* context, RileyCudaAllocationStats* out_stats,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (context == nullptr || out_stats == nullptr ||
      out_stats->struct_size < sizeof(*out_stats)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "query CUDA allocation stats",
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
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_context_close(
    RileyCudaContext** context, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  return context_close_impl(context, error, nullptr);
}

extern "C" RileyCudaStatus riley_cuda_context_defer_to_active_capture(
    RileyCudaContext** context, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "defer CUDA context close to active capture";
  clear_error(error);
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "context pointer is null");
  }
  if (*context == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  const RileyCudaStatus ready =
      validate_context_close_preconditions(*context, error, kOperation);
  if (ready != RILEY_CUDA_STATUS_SUCCESS) {
    return ready;
  }
  if (!initialize_capture_deferred_close_node(
          &(*context)->deferred_close, *context, *context,
          deferred_context_close)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                          "could not initialize the embedded deferred context-close node");
  }
  const CaptureDeferredCloseEnqueueResult enqueue_result =
      enqueue_capture_deferred_close(&(*context)->deferred_close);
  if (enqueue_result == CaptureDeferredCloseEnqueueResult::kQueued) {
    *context = nullptr;
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (enqueue_result == CaptureDeferredCloseEnqueueResult::kNotCapturing) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "no active thread-local CUDA Graph capture owns this host thread");
  }
  return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                        "the active capture rejected the deferred context-close node");
}

extern "C" RileyCudaStatus riley_cuda_stream_create(
    RileyCudaContext* context, RileyCudaStream** out_stream,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA stream", "out_stream is null");
  }
  *out_stream = nullptr;
  CurrentContext scope(context);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_CREATE, "create CUDA stream");
  cudaStream_t native = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), error,
        RILEY_CUDA_ERROR_STAGE_CREATE, "create non-default CUDA stream");
  }
  void* stream_storage = std::calloc(1, sizeof(RileyCudaStream));
  if (status == RILEY_CUDA_STATUS_SUCCESS && stream_storage == nullptr) {
    status = set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                       RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                       RILEY_CUDA_ERROR_STAGE_CREATE, "create CUDA stream",
                       "host allocation failed");
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                       "create CUDA stream");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    if (native != nullptr) {
      destroy_stream_after_failed_create(context, native);
    }
    std::free(stream_storage);
    return status;
  }
  auto* stream = new (stream_storage) RileyCudaStream{context, native};
  if (!retain_child(context)) {
    destroy_stream_after_failed_create(context, native);
    stream->~RileyCudaStream();
    std::free(stream);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA stream",
                          "context child-resource counter overflow");
  }
  *out_stream = stream;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_stream_command_batch_begin(
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "begin CUDA stream command batch";
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream is null");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "this host thread has an active thread-local CUDA Graph capture");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (stream->command_batch_owner.load(std::memory_order_acquire) != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream already has an active command batch");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream has an active asynchronous use");
  }
  if (stream->command_batch_use_count != 0) {
    (void)release_exclusive_use(stream->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation,
                          "inactive command batch retained ledger entries");
  }
  if (!try_publish_thread_command_batch()) {
    (void)release_exclusive_use(stream->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE, kOperation,
                          "thread-local command-batch counter overflow");
  }

  const void* expected = nullptr;
  if (!stream->command_batch_owner.compare_exchange_strong(
          expected, command_batch_thread_token(), std::memory_order_release,
          std::memory_order_acquire)) {
    const bool thread_released = release_thread_command_batch();
    const bool stream_released = release_exclusive_use(stream->active_uses);
    if (!thread_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "failed to release a rejected command-batch owner");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream already has an active command batch");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_stream_command_batch_end(
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "end CUDA stream command batch";
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream is null");
  }
  const void* owner =
      stream->command_batch_owner.load(std::memory_order_acquire);
  if (owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream has no active command batch");
  }
  if (owner != command_batch_thread_token()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream command batch is owned by another thread");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kOperation);
  bool completion_confirmed = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t synchronize_result =
        cudaStreamSynchronize(stream->stream);
    completion_confirmed = synchronize_result == cudaSuccess;
    status = runtime_error(synchronize_result, error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                       kOperation);
  const bool restoration_confirmed =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (!completion_confirmed || !restoration_confirmed) {
    // Completion ambiguity intentionally keeps owner, stream, buffer, and plan
    // leases live. All later query/sync/close and non-owner work fail closed.
    return status;
  }

  // Validate every counter before changing any of them. Only the owner thread
  // can mutate the ledger and the stream lease excludes all other users.
  if (stream->active_uses.load(std::memory_order_acquire) != 1) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kOperation,
                          "command-batch stream lease was corrupted");
  }
  for (size_t index = 0; index < stream->command_batch_use_count; ++index) {
    const auto* active = stream->command_batch_uses[index];
    if (active == nullptr || active->load(std::memory_order_acquire) != 1) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            kOperation,
                            "command-batch resource lease was corrupted");
    }
  }
  while (stream->command_batch_use_count != 0) {
    --stream->command_batch_use_count;
    std::atomic<uint32_t>* active =
        stream->command_batch_uses[stream->command_batch_use_count];
    stream->command_batch_uses[stream->command_batch_use_count] = nullptr;
    if (!release_exclusive_use(*active)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            kOperation,
                            "command-batch resource release was corrupted");
    }
  }
  // Publish inactivity before dropping the stream lease. During this final
  // window ordinary operations observe the still-busy stream and roll back;
  // after the release this function never dereferences the stream again.
  stream->command_batch_owner.store(nullptr, std::memory_order_release);
  if (!release_exclusive_use(stream->active_uses)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kOperation,
                          "command-batch stream release was corrupted");
  }
  if (!release_thread_command_batch()) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kOperation,
                          "thread-local command-batch counter was corrupted");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_stream_query(
    RileyCudaStream* stream, uint8_t* out_complete,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr || out_complete == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA stream",
                            "stream or out_complete is null");
  }
  *out_complete = 0;
  if (stream->active_uses.load(std::memory_order_acquire) != 0 ||
      stream->command_batch_owner.load(std::memory_order_acquire) != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_QUERY,
                            "query CUDA stream",
                            "stream has an active asynchronous use");
  }
  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_QUERY, "query CUDA stream");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaStreamQuery(stream->stream);
    if (result == cudaSuccess) {
      *out_complete = 1;
    }
    status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA stream");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                     "query CUDA stream");
}

extern "C" RileyCudaStatus riley_cuda_stream_synchronize(
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA stream", "stream is null");
  }
  if (stream->active_uses.load(std::memory_order_acquire) != 0 ||
      stream->command_batch_owner.load(std::memory_order_acquire) != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            "synchronize CUDA stream",
                            "stream has an active asynchronous use");
  }
  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize CUDA stream");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaStreamSynchronize(stream->stream), error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA stream");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                     "synchronize CUDA stream");
}

extern "C" RileyCudaStatus riley_cuda_stream_wait_event(
    RileyCudaStream* stream, RileyCudaEvent* event,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (stream == nullptr || event == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "wait for CUDA event",
                            "stream or event is null");
  }
  if (!same_context(stream->owner, event->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "wait for CUDA event",
                            "stream and event belong to different contexts");
  }
  if (stream->active_uses.load(std::memory_order_acquire) != 0 ||
      stream->command_batch_owner.load(std::memory_order_acquire) != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_RECORD,
                            "wait for CUDA event",
                            "stream has an active asynchronous use");
  }
  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_RECORD, "wait for CUDA event");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaStreamWaitEvent(stream->stream, event->event, 0),
                           error, RILEY_CUDA_ERROR_STAGE_RECORD,
                           "wait for CUDA event");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_RECORD,
                     "wait for CUDA event");
}

extern "C" RileyCudaStatus riley_cuda_stream_close(
    RileyCudaStream** stream, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  return stream_close_impl(stream, error, nullptr);
}

extern "C" RileyCudaStatus riley_cuda_stream_defer_to_active_capture(
    RileyCudaStream** stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "defer CUDA stream close to active capture";
  clear_error(error);
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kOperation, "stream pointer is null");
  }
  if (*stream == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if ((*stream)->active_uses.load(std::memory_order_acquire) != 0 ||
      (*stream)->command_batch_owner.load(std::memory_order_acquire) !=
          nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kOperation,
                            "stream still has an active asynchronous use");
  }
  if (!initialize_capture_deferred_close_node(
          &(*stream)->deferred_close, (*stream)->owner, *stream,
          deferred_stream_close)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                          "could not initialize the embedded deferred stream-close node");
  }
  const CaptureDeferredCloseEnqueueResult enqueue_result =
      enqueue_capture_deferred_close(&(*stream)->deferred_close);
  if (enqueue_result == CaptureDeferredCloseEnqueueResult::kQueued) {
    *stream = nullptr;
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (enqueue_result == CaptureDeferredCloseEnqueueResult::kNotCapturing) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "no active thread-local CUDA Graph capture owns this host thread");
  }
  return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                        "the active capture rejected the deferred stream-close node");
}

extern "C" RileyCudaStatus riley_cuda_event_create(
    RileyCudaContext* context, RileyCudaEvent** out_event,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_event == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "create CUDA event", "out_event is null");
  }
  *out_event = nullptr;
  CurrentContext scope(context);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_CREATE, "create CUDA event");
  cudaEvent_t native = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaEventCreateWithFlags(&native, cudaEventDefault),
                           error, RILEY_CUDA_ERROR_STAGE_CREATE,
                           "create timing-enabled CUDA event");
  }
  void* event_storage = std::calloc(1, sizeof(RileyCudaEvent));
  if (status == RILEY_CUDA_STATUS_SUCCESS && event_storage == nullptr) {
    status = set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                       RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                       RILEY_CUDA_ERROR_STAGE_CREATE, "create CUDA event",
                       "host allocation failed");
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                       "create CUDA event");
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    if (native != nullptr) {
      destroy_event_after_failed_create(context, native);
    }
    std::free(event_storage);
    return status;
  }
  auto* event = new (event_storage) RileyCudaEvent{context, native};
  if (!retain_child(context)) {
    destroy_event_after_failed_create(context, native);
    event->~RileyCudaEvent();
    std::free(event);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          "create CUDA event",
                          "context child-resource counter overflow");
  }
  *out_event = event;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_event_record(
    RileyCudaEvent* event, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr || stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "record CUDA event", "event or stream is null");
  }
  if (!same_context(event->owner, stream->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "record CUDA event",
                            "event and stream belong to different contexts");
  }
  if (stream->active_uses.load(std::memory_order_acquire) != 0 ||
      stream->command_batch_owner.load(std::memory_order_acquire) != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_RECORD,
                            "record CUDA event",
                            "stream has an active asynchronous use");
  }
  CurrentContext scope(event->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_RECORD, "record CUDA event");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaEventRecord(event->event, stream->stream), error,
                           RILEY_CUDA_ERROR_STAGE_RECORD,
                           "record CUDA event");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_RECORD,
                     "record CUDA event");
}

extern "C" RileyCudaStatus riley_cuda_event_query(
    RileyCudaEvent* event, uint8_t* out_complete,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr || out_complete == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "query CUDA event",
                            "event or out_complete is null");
  }
  *out_complete = 0;
  CurrentContext scope(event->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_QUERY, "query CUDA event");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t result = cudaEventQuery(event->event);
    if (result == cudaSuccess) {
      *out_complete = 1;
    }
    status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                           "query CUDA event");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                     "query CUDA event");
}

extern "C" RileyCudaStatus riley_cuda_event_synchronize(
    RileyCudaEvent* event, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (event == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "synchronize CUDA event", "event is null");
  }
  CurrentContext scope(event->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
      "synchronize CUDA event");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaEventSynchronize(event->event), error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           "synchronize CUDA event");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                     "synchronize CUDA event");
}

extern "C" RileyCudaStatus riley_cuda_event_elapsed_ms(
    RileyCudaEvent* start, RileyCudaEvent* end, float* out_elapsed_ms,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (start == nullptr || end == nullptr || out_elapsed_ms == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "measure CUDA event elapsed time",
                            "event or output pointer is null");
  }
  *out_elapsed_ms = 0.0F;
  if (!same_context(start->owner, end->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "measure CUDA event elapsed time",
                            "events belong to different contexts");
  }
  CurrentContext scope(start->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_QUERY,
      "measure CUDA event elapsed time");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaEventElapsedTime(out_elapsed_ms, start->event, end->event), error,
        RILEY_CUDA_ERROR_STAGE_QUERY, "measure CUDA event elapsed time");
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_QUERY,
                     "measure CUDA event elapsed time");
}

extern "C" RileyCudaStatus riley_cuda_event_close(
    RileyCudaEvent** event, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  return event_close_impl(event, error, nullptr);
}

extern "C" RileyCudaStatus riley_cuda_event_defer_to_active_capture(
    RileyCudaEvent** event, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "defer CUDA event close to active capture";
  clear_error(error);
  if (event == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kOperation, "event pointer is null");
  }
  if (*event == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (!initialize_capture_deferred_close_node(
          &(*event)->deferred_close, (*event)->owner, *event,
          deferred_event_close)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                          "could not initialize the embedded deferred event-close node");
  }
  const CaptureDeferredCloseEnqueueResult enqueue_result =
      enqueue_capture_deferred_close(&(*event)->deferred_close);
  if (enqueue_result == CaptureDeferredCloseEnqueueResult::kQueued) {
    *event = nullptr;
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (enqueue_result == CaptureDeferredCloseEnqueueResult::kNotCapturing) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "no active thread-local CUDA Graph capture owns this host thread");
  }
  return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kOperation,
                        "the active capture rejected the deferred event-close node");
}
