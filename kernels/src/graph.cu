#include "ffi_internal.hpp"

#include <cstddef>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <new>

namespace {

using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_thread_graph_capture_owner;
using riley_cuda_internal::command_batch_is_active;
using riley_cuda_internal::drain_capture_deferred_closes;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::native_thread_token;
using riley_cuda_internal::next_graph_capture_id;
using riley_cuda_internal::next_graph_exec_id;
using riley_cuda_internal::release_child;
using riley_cuda_internal::release_capture_domain_capture;
using riley_cuda_internal::release_exclusive_use;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::same_context;
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
constexpr const char* kBeginFillOperation = "begin CUDA Graph fill capture";
constexpr const char* kEnqueueFillOperation = "enqueue CUDA Graph fill";
constexpr const char* kEndOperation = "end CUDA Graph capture";
constexpr const char* kInstantiateOperation = "instantiate CUDA Graph";
constexpr const char* kLaunchOperation = "launch CUDA Graph exec";
constexpr const char* kCompleteOperation = "complete CUDA Graph launch";
constexpr const char* kCloseGraphOperation = "close CUDA Graph";
constexpr const char* kCloseExecOperation = "close CUDA Graph exec";
constexpr uint32_t kGraphFillThreads = 256;
constexpr uint64_t kMaximumGraphFillGridX = static_cast<uint64_t>(INT_MAX);

__global__ void graph_fill_f32(float* output, uint64_t element_count,
                               float value) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = value;
  }
}

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
                          uint64_t exec_id, bool submission_started,
                          bool completion_known,
                          bool resource_release_known,
                          bool poisoned) noexcept {
  clear_graph_error(error, stage);
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  error->capture_id = capture_id;
  error->exec_id = exec_id;
  error->submission_started = submission_started ? 1 : 0;
  error->completion_known = completion_known ? 1 : 0;
  error->resource_release_known = resource_release_known ? 1 : 0;
  error->poisoned = poisoned ? 1 : 0;
}

void record_capture_outcome(RileyCudaGraphErrorInfo* error,
                            RileyCudaGraphStage stage, uint64_t capture_id,
                            bool resource_release_known,
                            bool poisoned) noexcept {
  record_graph_outcome(error, stage, capture_id, 0, false, false,
                       resource_release_known, poisoned);
}

// This wrapper is released only after every native side effect is known. Keep
// the thread-local gate published until the child and stream leases have both
// released; a failed release leaves the owner published and fail-closed.
bool release_capture_owner(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->owner == nullptr ||
      capture->stream == nullptr || capture->capture_domain == nullptr ||
      capture->prepared_graph != nullptr || capture->fill_buffer != nullptr ||
      capture->fill_lease_held || capture->unreleased_graph != nullptr ||
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

// C05-5 reserves one existing device buffer before capture begins. It is not
// an asynchronous operation token: the exact address stays leased at one
// through captured-graph and graph-exec ownership, then returns to zero only
// after a known abort or graph close. Keep this tiny helper separate from the
// generic capture-owner release so C05-4 owners retain their original layout.
bool release_capture_fill_lease(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr) {
    return false;
  }
  if (!capture->fill_lease_held) {
    return capture->fill_buffer == nullptr;
  }
  if (capture->fill_buffer == nullptr ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_lease_held = false;
  return true;
}

bool destroy_prepared_graph_storage(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->prepared_graph == nullptr) {
    return capture != nullptr;
  }
  RileyCudaGraph* const graph = capture->prepared_graph;
  if (graph->owner != capture->owner || graph->stream != capture->stream ||
      graph->graph != nullptr || graph->owns_capture_leases) {
    return false;
  }
  graph->~RileyCudaGraph();
  std::free(graph);
  capture->prepared_graph = nullptr;
  return true;
}

// Move only the capture-domain/TLS ownership off the capture wrapper. The
// context-child, stream, and buffer leases deliberately remain acquired and
// become the returned graph's permanent resource guard.
bool transfer_capture_owner_to_graph(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->owner == nullptr ||
      capture->stream == nullptr || capture->capture_domain == nullptr ||
      capture->prepared_graph == nullptr || capture->fill_buffer == nullptr ||
      !capture->fill_lease_held || capture->deferred_close_head != nullptr ||
      capture->deferred_close_tail != nullptr || capture->unreleased_graph != nullptr) {
    return false;
  }
  RileyCudaGraph* const graph = capture->prepared_graph;
  if (graph->owner != capture->owner || graph->stream != capture->stream ||
      graph->fill_buffer != capture->fill_buffer || graph->graph == nullptr ||
      graph->owns_capture_leases ||
      capture->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  if (!release_capture_domain_capture(capture->capture_domain) ||
      !clear_thread_graph_capture_owner(capture)) {
    return false;
  }
  graph->owns_capture_leases = true;
  capture->prepared_graph = nullptr;
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_lease_held = false;
  capture->~RileyCudaGraphCapture();
  std::free(capture);
  return true;
}

bool release_graph_leases(RileyCudaContext* owner, RileyCudaStream* stream,
                          RileyCudaDeviceBuffer* buffer) noexcept {
  if (owner == nullptr || stream == nullptr || buffer == nullptr ||
      !same_context(owner, stream->owner) ||
      !same_context(owner, buffer->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  // Validate every counter first; each release is then a deterministic 1->0
  // transition. Any impossible underflow is retained fail-closed by callers.
  return release_exclusive_use(buffer->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
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

// Riley cannot safely adopt an already-active capture that was initiated by a
// foreign CUDA caller. Observe the exact stream while its owning context is
// current before creating/publishing a Riley capture owner. An observation
// error is deliberately a denial rather than a guess: begin may otherwise
// return a local abort owner for a foreign graph and destroy foreign work.
RileyCudaStatus require_stream_capture_idle(RileyCudaStream* stream,
                                            RileyCudaErrorInfo* error,
                                            const char* operation) noexcept {
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "stream or its owner is null while observing capture state");
  }
  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        operation);
  cudaStreamCaptureStatus state = cudaStreamCaptureStatusActive;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaStreamIsCapturing(stream->stream, &state), error,
                           RILEY_CUDA_ERROR_STAGE_PREPARE, operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS &&
        state != cudaStreamCaptureStatusNone) {
      status = validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
          "stream already has an active or invalidated foreign CUDA capture");
    }
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                     operation);
}

// C05-5's fixed-buffer entry point deliberately has its own admission path.
// Its capture wrapper and future graph wrapper are both allocated before
// cudaStreamBeginCapture, and both exact resource leases are established
// before any CUDA entry. This keeps capture enqueue allocation-free and means
// the graph can later retain the same device address safely.
RileyCudaStatus capture_begin_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* fill_buffer,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
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
                            kBeginFillOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || fill_buffer == nullptr || stream->owner == nullptr ||
      fill_buffer->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "stream, fill buffer, or their owner is null");
  }
  if (!same_context(stream->owner, fill_buffer->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "capture stream and fill buffer belong to different context owners");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "only thread-local capture mode is admitted");
  }
  if (element_count == 0 ||
      element_count > fill_buffer->byte_len / sizeof(float)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "fixed f32 fill element count exceeds the preallocated buffer");
  }
  if (fill_buffer->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "fixed f32 fill buffer has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "a stream command batch blocks fixed-fill graph capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginFillOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(fill_buffer->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "fixed f32 fill buffer has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool buffer_released =
        release_exclusive_use(fill_buffer->active_uses);
    if (!buffer_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginFillOperation,
                            "failed to release a rejected fixed-fill buffer lease");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginFillOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginFillOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginFillOperation,
                     "host allocation failed for fixed-fill capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginFillOperation,
                     "host allocation failed for captured graph owner");
  }
  capture->prepared_graph =
      new (graph_storage) RileyCudaGraph(stream->owner, stream, fill_buffer,
                                         capture_id);
  capture->fill_buffer = fill_buffer;
  capture->fill_element_count = element_count;
  capture->fill_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool buffer_released = release_capture_fill_lease(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !buffer_released || !child_released ||
        !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginFillOperation,
                            "failed to release a blocked fixed-fill capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool buffer_released = release_capture_fill_lease(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !buffer_released ||
        !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginFillOperation,
                            "failed to release a rejected fixed-fill capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginFillOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginFillOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginFillOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);

  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool buffer_released = release_capture_fill_lease(capture);
  const bool capture_released =
      graph_released && buffer_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginFillOperation,
                          "failed to release an unstarted fixed-fill capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
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
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
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
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  if (!release_capture_owner(capture)) {
    // The capture never began, but an ownership-counter corruption must still
    // strand the stream lease rather than permit unsound reuse.
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginOperation,
                          "failed to release an unstarted capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
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
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, 0, true,
                           false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  RileyCudaGraphCapture* const owner = *capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, capture_id,
                         false, false);
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
    record_capture_outcome(out_graph_error,
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
      const bool fill_released = release_capture_fill_lease(owner);
      const bool prepared_graph_released =
          fill_released && destroy_prepared_graph_storage(owner);
      released = prepared_graph_released && release_capture_owner(owner);
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
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, capture_id,
                         released, !released);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_fill_f32(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* buffer,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_impl(stream, buffer, element_count, mode, out_capture,
                            out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_fill_f32(
    RileyCudaGraphCapture* capture, float value,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueFillOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation, "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      !owner->fill_lease_held || owner->capture_terminated ||
      owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "capture owner is not a live fixed-fill capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueFillOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started || owner->fill_element_count == 0 ||
      owner->fill_element_count > owner->fill_buffer->byte_len / sizeof(float) ||
      owner->fill_buffer->device_data == nullptr ||
      owner->fill_enqueue_count == std::numeric_limits<uint32_t>::max()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "fixed-fill capture owner has invalid immutable geometry");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "fixed-fill capture resource lease is unavailable");
  }
  const uint64_t grid_x =
      ((owner->fill_element_count - 1) / kGraphFillThreads) + 1;
  if (grid_x > kMaximumGraphFillGridX) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "fixed f32 fill grid exceeds CUDA's x-dimension limit");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kEnqueueFillOperation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_fill_f32<<<static_cast<unsigned int>(grid_x), kGraphFillThreads, 0,
                     owner->stream->stream>>>(
        static_cast<float*>(owner->fill_buffer->device_data),
        owner->fill_element_count, value);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueFillOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      ++owner->fill_enqueue_count;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueFillOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_end(
    RileyCudaGraphCapture** capture, RileyCudaGraph** out_graph,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_graph != nullptr) {
    *out_graph = nullptr;
  }
  if (capture == nullptr || out_graph == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture pointer or out_graph is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END);
  if (*capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = *capture;
  const uint64_t capture_id = owner->capture_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                       capture_id, 0, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      !owner->fill_lease_held || owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture owner is not a live fixed-fill capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "thread-local capture must end on its begin thread");
  }
  if (!owner->capture_started || owner->capture_terminated ||
      owner->fill_enqueue_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "fixed-fill capture end requires at least one live enqueue");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kEndOperation,
                          "fixed-fill capture resource lease was corrupted");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kEndOperation, owner);
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
                             kEndOperation);
      termination_known = capture_end_is_known(owner->stream);
    }
    if (termination_known) {
      graph_release_known = true;
      // Successful end transfers the graph below. An error outcome instead
      // discards any returned graph exactly once so the capture owner can
      // still recover its Rust deferred-close ledger when that destruction is
      // fully known.
      if (status != RILEY_CUDA_STATUS_SUCCESS && returned_graph != nullptr) {
        const cudaError_t destroy_result = cudaGraphDestroy(returned_graph);
        if (destroy_result != cudaSuccess) {
          owner->unreleased_graph = returned_graph;
          graph_release_known = false;
        }
      }
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kEndOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!end_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                         capture_id, 0, false, false, false,
                         !restoration_known);
    return status;
  }

  // cudaStreamEndCapture is one-shot. Consume the raw input after the CUDA
  // attempt and retain any uncertain owner/lease state rather than permitting
  // a second end or destroy attempt through a stale handle.
  *capture = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && termination_known &&
      graph_release_known && restoration_known && returned_graph != nullptr) {
    owner->prepared_graph->graph = returned_graph;
    owner->capture_terminated = true;
    RileyCudaErrorInfo deferred_close_error{};
    deferred_close_error.struct_size = sizeof(deferred_close_error);
    const RileyCudaStatus deferred_close_status =
        drain_capture_deferred_closes(owner, &deferred_close_error);
    RileyCudaGraph* const graph = owner->prepared_graph;
    if (deferred_close_status == RILEY_CUDA_STATUS_SUCCESS &&
        transfer_capture_owner_to_graph(owner)) {
      *out_graph = graph;
      // owner has been freed. Do not dereference it beyond this point.
      record_graph_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_END, capture_id, 0,
                           false, false, true, false);
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    if (deferred_close_status != RILEY_CUDA_STATUS_SUCCESS &&
        error != nullptr && error->struct_size >= sizeof(*error)) {
      *error = deferred_close_error;
    }
    if (deferred_close_status != RILEY_CUDA_STATUS_SUCCESS) {
      status = deferred_close_status;
    } else {
      status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                              kEndOperation,
                              "failed to transfer the recovered capture owner to graph ownership");
    }
  } else if (termination_known && graph_release_known && restoration_known) {
    // A deferred end error can still leave capture physically terminated. In
    // that case discard the graph and run the same one-shot recovery as abort;
    // this lets the safe wrapper finish its TLS deferred-context ledger only
    // when every capture-local close has a known result.
    if (returned_graph == nullptr && status == RILEY_CUDA_STATUS_SUCCESS) {
      status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                              kEndOperation,
                              "cudaStreamEndCapture succeeded without a graph handle");
    }
    owner->capture_terminated = true;
    RileyCudaErrorInfo deferred_close_error{};
    deferred_close_error.struct_size = sizeof(deferred_close_error);
    const RileyCudaStatus deferred_close_status =
        drain_capture_deferred_closes(owner, &deferred_close_error);
    bool released = false;
    if (deferred_close_status == RILEY_CUDA_STATUS_SUCCESS) {
      const bool fill_released = release_capture_fill_lease(owner);
      const bool prepared_graph_released =
          fill_released && destroy_prepared_graph_storage(owner);
      released = prepared_graph_released && release_capture_owner(owner);
    } else if (error != nullptr && error->struct_size >= sizeof(*error)) {
      *error = deferred_close_error;
    }
    if (!released && status == RILEY_CUDA_STATUS_SUCCESS) {
      status = deferred_close_status == RILEY_CUDA_STATUS_SUCCESS
                   ? internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                    kEndOperation,
                                    "failed to release recovered capture after graph end")
                   : deferred_close_status;
    }
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                         capture_id, 0, false, false, released, !released);
    return status;
  }

  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                       capture_id, 0, false, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_instantiate(
    RileyCudaGraph** graph, RileyCudaGraphExec** out_exec,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_exec != nullptr) {
    *out_exec = nullptr;
  }
  if (graph == nullptr || out_exec == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "graph pointer or out_exec is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE);
  if (*graph == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation, "graph owner is null");
  }
  RileyCudaGraph* const owner = *graph;
  const uint64_t capture_id = owner->capture_id;
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      !owner->owns_capture_leases ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph resource lease is invalid");
  }
  if (owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph context is poisoned");
  }
  const uint64_t exec_id = next_graph_exec_id();
  if (exec_id == 0) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kInstantiateOperation,
                          "CUDA Graph exec ID space is exhausted");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                       capture_id, exec_id, false, false, false, false);
  void* storage = std::calloc(1, sizeof(RileyCudaGraphExec));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kInstantiateOperation,
                     "host allocation failed for CUDA Graph exec owner");
  }
  auto* exec = new (storage) RileyCudaGraphExec(
      owner->owner, owner->stream, owner->fill_buffer, capture_id, exec_id);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                                        kInstantiateOperation);
  bool instantiate_attempted = false;
  cudaGraphExec_t native_exec = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    instantiate_attempted = true;
    status = runtime_error(
        cudaGraphInstantiate(&native_exec, owner->graph, nullptr, nullptr, 0),
        error, RILEY_CUDA_ERROR_STAGE_CREATE, kInstantiateOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                       kInstantiateOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!instantiate_attempted) {
    exec->~RileyCudaGraphExec();
    std::free(exec);
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                         capture_id, exec_id, false, false, false,
                         !restoration_known);
    return status;
  }

  // Instantiation is one-shot from the safe ownership perspective. The graph
  // pointer is consumed after any CUDA instantiate attempt; an uncertain
  // native exec/graph pair is retained intentionally rather than retried.
  *graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known &&
      native_exec != nullptr) {
    exec->graph = owner->graph;
    exec->exec = native_exec;
    exec->owns_capture_leases = true;
    owner->graph = nullptr;
    owner->owns_capture_leases = false;
    owner->~RileyCudaGraph();
    std::free(owner);
    *out_exec = exec;
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                         capture_id, exec_id, false, false, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  // Preserve any opaque CUDA outputs in deliberately leaked host owners. No
  // close/retry is safe after a failed instantiate call because CUDA may have
  // consumed or partially initialized either native object before surfacing a
  // deferred error.
  exec->graph = owner->graph;
  exec->exec = native_exec;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                       capture_id, exec_id, false, false, false, true);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kInstantiateOperation,
                          "cudaGraphInstantiate succeeded without an exec handle");
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_exec_launch(
    RileyCudaGraphExec* exec, RileyCudaStream* stream,
    RileyCudaGraphLaunch** out_launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_launch != nullptr) {
    *out_launch = nullptr;
  }
  if (out_launch == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation, "out_launch is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH);
  if (exec == nullptr || stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation, "graph exec or stream is null");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                       capture_id, exec_id, false, false, false, false);
  if (exec->owner == nullptr || exec->stream == nullptr ||
      exec->fill_buffer == nullptr || exec->graph == nullptr ||
      exec->exec == nullptr || !exec->owns_capture_leases) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec has invalid retained capture resources");
  }
  if (stream != exec->stream) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec must launch on its exact captured stream");
  }
  if (!same_context(exec->owner, stream->owner) ||
      !same_context(exec->owner, exec->fill_buffer->owner) ||
      exec->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->launch_in_flight || exec->poisoned ||
      exec->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec is busy, poisoned, or lost its retained resource lease");
  }
  void* storage = std::calloc(1, sizeof(RileyCudaGraphLaunch));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kLaunchOperation,
                     "host allocation failed for CUDA Graph launch owner");
  }
  auto* launch = new (storage) RileyCudaGraphLaunch(exec, stream);

  CurrentContext scope(exec->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kLaunchOperation);
  bool launch_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    status = runtime_error(cudaGraphLaunch(exec->exec, stream->stream), error,
                           RILEY_CUDA_ERROR_STAGE_LAUNCH, kLaunchOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kLaunchOperation);
  const bool restoration_known =
      !exec->owner->restoration_failed.load(std::memory_order_acquire);
  if (!launch_attempted) {
    launch->~RileyCudaGraphLaunch();
    std::free(launch);
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                         capture_id, exec_id, false, false, false,
                         !restoration_known);
    return status;
  }

  // Once cudaGraphLaunch has been attempted, close/relaunch must remain
  // blocked even if CUDA reports a deferred failure. Give raw callers the
  // one completion owner, but fail closed if they decline to settle it.
  exec->launch_in_flight = true;
  *out_launch = launch;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                         capture_id, exec_id, true, false, false, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  exec->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                       capture_id, exec_id, true, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_launch_complete(
    RileyCudaGraphLaunch** launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (launch == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCompleteOperation, "launch pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCompleteOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION);
  if (*launch == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         0, 0, false, true, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaGraphLaunch* const owner = *launch;
  RileyCudaGraphExec* const exec = owner->exec;
  if (exec == nullptr || owner->stream == nullptr || exec->owner == nullptr ||
      exec->stream != owner->stream || !exec->launch_in_flight ||
      exec->exec == nullptr || !exec->owns_capture_leases) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCompleteOperation,
                            "graph launch owner is not a live completion boundary");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, true, false, false, false);

  CurrentContext scope(exec->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                                        kCompleteOperation);
  bool completion_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    completion_attempted = true;
    status = runtime_error(cudaStreamSynchronize(exec->stream->stream), error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           kCompleteOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                       kCompleteOperation);
  const bool restoration_known =
      !exec->owner->restoration_failed.load(std::memory_order_acquire);
  if (!completion_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         capture_id, exec_id, true, false, false,
                         !restoration_known);
    return status;
  }

  // Completion is also one-shot. Consume the raw owner even when a CUDA sync
  // reports an error; native resources remain poisoned and retained on an
  // ambiguous completion rather than accepting a second synchronize attempt.
  *launch = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known) {
    // A failed cudaGraphLaunch can surface a deferred error after submitting
    // work. This single successful synchronization proves the only in-flight
    // boundary has settled, so the launch-specific poison is recoverable and
    // the safe FFI may return the original launch error without stranding the
    // graph exec's permanent stream/buffer leases.
    exec->launch_in_flight = false;
    exec->poisoned = false;
    owner->~RileyCudaGraphLaunch();
    std::free(owner);
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         capture_id, exec_id, true, true, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  exec->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, true, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_close(
    RileyCudaGraph** graph, RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (graph == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation, "graph pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE);
  if (*graph == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE, 0, 0,
                         false, false, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaGraph* const owner = *graph;
  const uint64_t capture_id = owner->capture_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, 0, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      !owner->owns_capture_leases ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation,
                            "captured graph has invalid retained resource leases");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kCloseGraphOperation);
  bool destroy_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    destroy_attempted = true;
    status = runtime_error(cudaGraphDestroy(owner->graph), error,
                           RILEY_CUDA_ERROR_STAGE_CLOSE, kCloseGraphOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kCloseGraphOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!destroy_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                         capture_id, 0, false, false, false,
                         !restoration_known);
    return status;
  }

  *graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known) {
    owner->graph = nullptr;
    const bool released =
        release_graph_leases(owner->owner, owner->stream, owner->fill_buffer);
    if (released) {
      owner->owns_capture_leases = false;
      owner->~RileyCudaGraph();
      std::free(owner);
      record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                           capture_id, 0, false, false, true, false);
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kCloseGraphOperation,
                            "failed to release graph stream, buffer, or context lease");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, 0, false, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_exec_close(
    RileyCudaGraphExec** exec, RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (exec == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation, "graph exec pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE);
  if (*exec == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE, 0, 0,
                         false, false, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaGraphExec* const owner = *exec;
  const uint64_t capture_id = owner->capture_id;
  const uint64_t exec_id = owner->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, exec_id, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      owner->exec == nullptr || !owner->owns_capture_leases ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->launch_in_flight || owner->poisoned ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation,
                            "graph exec is busy, poisoned, or lost its retained resource lease");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kCloseExecOperation);
  bool exec_destroy_attempted = false;
  bool graph_destroy_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    exec_destroy_attempted = true;
    status = runtime_error(cudaGraphExecDestroy(owner->exec), error,
                           RILEY_CUDA_ERROR_STAGE_CLOSE, kCloseExecOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_destroy_attempted = true;
      status = runtime_error(cudaGraphDestroy(owner->graph), error,
                             RILEY_CUDA_ERROR_STAGE_CLOSE,
                             kCloseExecOperation);
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kCloseExecOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!exec_destroy_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                         capture_id, exec_id, false, false, false,
                         !restoration_known);
    return status;
  }

  // Both native destroy operations are one-shot. A failure after the first
  // call retains every graph lease permanently; retrying could double-destroy
  // either opaque CUDA object.
  *exec = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && graph_destroy_attempted &&
      restoration_known) {
    owner->exec = nullptr;
    owner->graph = nullptr;
    const bool released =
        release_graph_leases(owner->owner, owner->stream, owner->fill_buffer);
    if (released) {
      owner->owns_capture_leases = false;
      owner->~RileyCudaGraphExec();
      std::free(owner);
      record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                           capture_id, exec_id, false, false, true, false);
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kCloseExecOperation,
                            "failed to release graph exec stream, buffer, or context lease");
  }
  owner->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, exec_id, false, false, false, true);
  return status;
}
