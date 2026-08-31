#include "ffi_internal.hpp"

#include <cstddef>
#include <cstring>

namespace {

bool graph_error_is_compatible(const RileyCudaGraphErrorInfo* error) noexcept {
  return error == nullptr || error->struct_size >= sizeof(*error);
}

bool graph_error_reserved_is_zero(
    const RileyCudaGraphErrorInfo* error) noexcept {
  if (error == nullptr) {
    return true;
  }
  if (error->reserved0 != 0) {
    return false;
  }
  for (size_t index = 0; index < 3; ++index) {
    if (error->reserved[index] != 0) {
      return false;
    }
  }
  return true;
}

void clear_graph_error(RileyCudaGraphErrorInfo* error,
                       RileyCudaGraphStage stage) noexcept {
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  const uint32_t struct_size = error->struct_size;
  std::memset(error, 0, sizeof(*error));
  error->struct_size = struct_size;
  error->graph_stage = stage;
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin(
    RileyCudaStream* stream, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;
  using riley_cuda_internal::set_error;
  using riley_cuda_internal::validation_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "begin CUDA Graph capture", "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, "begin CUDA Graph capture",
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "begin CUDA Graph capture", "stream is null");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            "begin CUDA Graph capture",
                            "only thread-local capture mode is admitted");
  }

  // This C05-1 vertical slice deliberately does not begin CUDA capture, touch
  // the stream, acquire resource leases, or allocate a graph owner. Keeping
  // the deterministic native result makes the linked ABI fail closed until a
  // later slice can provide those ownership guarantees.
  return set_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED, 0,
                   RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                   RILEY_CUDA_ERROR_STAGE_PREPARE, "begin CUDA Graph capture",
                   "native CUDA Graph capture is not linked into this build");
}
