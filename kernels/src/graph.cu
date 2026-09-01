#include "ffi_internal.hpp"

#include <cstddef>
#include <cstring>

namespace {

using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_thread_graph_capture_owner;
using riley_cuda_internal::command_batch_is_active;
using riley_cuda_internal::drain_capture_deferred_closes;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::native_thread_token;
using riley_cuda_internal::next_graph_capture_id;
using riley_cuda_internal::release_child;
using riley_cuda_internal::release_capture_domain_capture;
using riley_cuda_internal::release_exclusive_use;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::set_error;
using riley_cuda_internal::thread_graph_capture_is_owner;
using riley_cuda_internal::thread_has_active_command_batch;
using riley_cuda_internal::thread_has_active_graph_capture;
using riley_cuda_internal::try_publish_thread_graph_capture;
using riley_cuda_internal::try_acquire_exclusive_use;
using riley_cuda_internal::try_begin_capture_domain;
using riley_cuda_internal::validation_error;

constexpr const char* kBeginOperation = "begin CUDA Graph capture";
constexpr const char* kAbortOperation = "abort CUDA Graph capture";

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

void record_graph_outcome(RileyCudaGraphErrorInfo* error,
                          RileyCudaGraphStage stage, uint64_t capture_id,
                          bool resource_release_known,
                          bool poisoned) noexcept {
  clear_graph_error(error, stage);
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  error->capture_id = capture_id;
  error->resource_release_known = resource_release_known ? 1 : 0;
  error->poisoned = poisoned ? 1 : 0;
}

// This wrapper is released only after every native side effect is known. Keep
// the thread-local gate published until the child and stream leases have both
// released; a failed release leaves the owner published and fail-closed.
bool release_capture_owner(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->owner == nullptr ||
      capture->stream == nullptr || capture->capture_domain == nullptr ||
      capture->deferred_close_head != nullptr ||
      capture->deferred_close_tail != nullptr) {
    return false;
  }
  RileyCudaContext* const owner = capture->owner;
  RileyCudaStream* const stream = capture->stream;
  if (!release_child(owner)) {
    return false;
  }
  if (!release_exclusive_use(stream->active_uses)) {
    return false;
  }
  if (!release_capture_domain_capture(capture->capture_domain)) {
    return false;
  }
  if (!clear_thread_graph_capture_owner(capture)) {
    return false;
  }
  capture->~RileyCudaGraphCapture();
  std::free(capture);
  return true;
}

// cudaStreamBeginCapture is documented to surface a prior asynchronous CUDA
// error. If it does, use a direct capture-state observation before deciding
// whether the returned owner must be retained for recovery. An observation
// failure itself is ambiguous and therefore treated as an active capture.
bool capture_may_be_active_after_failed_begin(RileyCudaStream* stream) noexcept {
  cudaStreamCaptureStatus state = cudaStreamCaptureStatusActive;
  const cudaError_t result = cudaStreamIsCapturing(stream->stream, &state);
  return result != cudaSuccess || state != cudaStreamCaptureStatusNone;
}

bool capture_end_is_known(RileyCudaStream* stream) noexcept {
  cudaStreamCaptureStatus state = cudaStreamCaptureStatusActive;
  return cudaStreamIsCapturing(stream->stream, &state) == cudaSuccess &&
         state == cudaStreamCaptureStatusNone;
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin(
    RileyCudaStream* stream, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation, "stream or its owner is null");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation,
                            "only thread-local capture mode is admitted");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "this host thread has an active CUDA stream command batch");
  }
  if (command_batch_is_active(stream)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation,
                            "stream already has an active command batch");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginOperation,
                          "context child-resource counter overflow");
  }
  void* storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginOperation,
                     "host allocation failed");
  }
  auto* capture = new (storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginOperation,
                            "failed to release a blocked capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginOperation,
                            "failed to release a rejected capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);

  if (capture_may_be_active) {
    // A non-success status can still accompany an entered capture if CUDA
    // surfaced a deferred asynchronous error. Returning the owner permits the
    // safe Rust boundary to run the same one-shot abort/recovery path before it
    // reports that error. A restoration failure marks that owner poisoned but
    // is likewise retained rather than silently abandoning active capture.
    *out_capture = capture;
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN,
                         capture_id, false,
                         status != RILEY_CUDA_STATUS_SUCCESS ||
                             !restoration_known);
    return status;
  }

  if (!release_capture_owner(capture)) {
    // The capture never began, but an ownership-counter corruption must still
    // strand the stream lease rather than permit unsound reuse.
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginOperation,
                          "failed to release an unstarted capture owner");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN,
                       0, true, !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_abort(
    RileyCudaGraphCapture** capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kAbortOperation, "capture pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kAbortOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  if (*capture == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT,
                         0, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  RileyCudaGraphCapture* const owner = *capture;
  const uint64_t capture_id = owner->capture_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT,
                       capture_id, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kAbortOperation,
                          "capture owner has a null context or stream");
  }
  if (owner->owner_thread != native_thread_token()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kAbortOperation,
                            "thread-local capture must end on its begin thread");
  }
  if (!thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kAbortOperation,
        "the supplied capture owner is not active on this host thread");
  }
  if (!owner->capture_started) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kAbortOperation,
                          "capture owner was not marked active");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kAbortOperation,
                          "capture stream lease was corrupted");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kAbortOperation, owner);
  bool end_attempted = false;
  bool termination_known = false;
  bool graph_release_known = false;
  cudaGraph_t returned_graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    end_attempted = true;
    const cudaError_t end_result =
        cudaStreamEndCapture(owner->stream->stream, &returned_graph);
    if (end_result == cudaSuccess) {
      termination_known = true;
    } else {
      status = runtime_error(end_result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                             kAbortOperation);
      termination_known = capture_end_is_known(owner->stream);
    }
    if (termination_known) {
      graph_release_known = true;
      if (returned_graph != nullptr) {
        const cudaError_t destroy_result = cudaGraphDestroy(returned_graph);
        if (destroy_result != cudaSuccess) {
          // cudaGraphDestroy may report a deferred error after consuming the
          // resource. Preserve the opaque graph only in the intentionally
          // leaked owner and never issue a second destroy attempt.
          owner->unreleased_graph = returned_graph;
          graph_release_known = false;
          if (status == RILEY_CUDA_STATUS_SUCCESS) {
            status = runtime_error(destroy_result, error,
                                   RILEY_CUDA_ERROR_STAGE_CLOSE,
                                   kAbortOperation);
          }
        }
      }
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kAbortOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);

  if (!end_attempted) {
    // No CUDA end attempt occurred, so a raw caller may retry after correcting
    // the precondition. The safe Rust wrapper deliberately abandons this
    // handle instead of retrying from Drop.
    record_graph_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, capture_id,
                         false, !restoration_known);
    return status;
  }

  // End capture is a one-shot CUDA lifecycle transition. Consume the caller's
  // raw handle before reporting any result, even when capture termination or
  // graph destruction cannot be proven. The retained owner/lease below is an
  // intentional fail-closed leak, never a retryable dangling pointer.
  *capture = nullptr;

  // The ThreadLocal gate remains published after cudaStreamEndCapture,
  // cudaGraphDestroy, and capture-context restoration are all known. Drain
  // only in that state, before releasing the graph child, stream lease, domain
  // admission, or TLS owner. Each callback receives `owner` as the exact
  // CurrentContext bypass and can close a resource belonging to another Riley
  // context. A failed drain intentionally strands its remaining FIFO plus all
  // capture leases; reissuing CUDA closes after an ambiguous error is unsafe.
  bool released = false;
  if (termination_known && graph_release_known && restoration_known) {
    // CUDA has physically left capture and the transient graph is gone, but
    // the exact TLS owner intentionally remains published until deferred safe
    // resource cleanup succeeds. This narrowly permits a childless foreign
    // context lease release through the matching capture-domain control gate.
    owner->capture_terminated = true;
    RileyCudaErrorInfo deferred_close_error{};
    deferred_close_error.struct_size = sizeof(deferred_close_error);
    const RileyCudaStatus deferred_close_status =
        drain_capture_deferred_closes(owner, &deferred_close_error);
    if (deferred_close_status == RILEY_CUDA_STATUS_SUCCESS) {
      released = release_capture_owner(owner);
      if (!released && status == RILEY_CUDA_STATUS_SUCCESS) {
        status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                kAbortOperation,
                                "failed to release recovered capture owner");
      }
    } else if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = deferred_close_status;
      if (error != nullptr && error->struct_size >= sizeof(*error)) {
        *error = deferred_close_error;
      }
    }
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT,
                       capture_id, released, !released);
  return status;
}
