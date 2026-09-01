#include "ffi_internal.hpp"

#include <cuda_bf16.h>

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
constexpr const char* kBeginH2DOperation = "begin CUDA Graph H2D capture";
constexpr const char* kEnqueueH2DOperation = "enqueue CUDA Graph H2D";
constexpr const char* kBeginSiluBf16Operation =
    "begin CUDA Graph BF16 SiLU capture";
constexpr const char* kEnqueueSiluBf16Operation =
    "enqueue CUDA Graph BF16 SiLU";
constexpr const char* kStageH2DOperation = "stage CUDA Graph H2D source";
constexpr const char* kEndOperation = "end CUDA Graph capture";
constexpr const char* kInstantiateOperation = "instantiate CUDA Graph";
constexpr const char* kLaunchOperation = "launch CUDA Graph exec";
constexpr const char* kCompleteOperation = "complete CUDA Graph launch";
constexpr const char* kCloseGraphOperation = "close CUDA Graph";
constexpr const char* kCloseExecOperation = "close CUDA Graph exec";
constexpr uint32_t kGraphFillThreads = 256;
constexpr uint64_t kMaximumGraphFillGridX = static_cast<uint64_t>(INT_MAX);
constexpr uint32_t kGraphSiluThreads = 256;
constexpr uint32_t kMaximumGraphSiluBlocks = 65535;

__global__ void graph_fill_f32(float* output, uint64_t element_count,
                               float value) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = value;
  }
}

// This is deliberately capture-local rather than riley_cuda_silu_execute:
// eager SiLU owns transient ExclusiveUses and synchronizes completion, neither
// of which is admissible while a graph capture owns permanent input/output
// leases. Keep the arithmetic and grid-stride topology equal to the eager
// BF16 primitive so graph parity includes its exact storage-rounding boundary.
__global__ void graph_silu_bf16(const __nv_bfloat16* input,
                                 __nv_bfloat16* output,
                                 uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const float value = __bfloat162float(input[index]);
    output[index] = __float2bfloat16_rn(value / (1.0F + expf(-value)));
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
      capture->prepared_graph != nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kNone ||
      capture->fill_buffer != nullptr || capture->fill_element_count != 0 ||
      capture->fill_enqueue_count != 0 || capture->fill_lease_held ||
      capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
      capture->h2d_enqueue_count != 0 || capture->h2d_source_lease_held ||
      capture->silu_input != nullptr || capture->silu_element_count != 0 ||
      capture->silu_enqueue_count != 0 || capture->silu_input_lease_held ||
      capture->unreleased_graph != nullptr ||
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
    return capture->fill_buffer == nullptr &&
           capture->operation != RileyCudaGraphCaptureOperation::kFillF32;
  }
  if (capture->fill_buffer == nullptr ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  if (capture->operation == RileyCudaGraphCaptureOperation::kFillF32) {
    capture->operation = RileyCudaGraphCaptureOperation::kNone;
  }
  return true;
}

// The H2D source is a captured raw host pointer. It must stay leased alongside
// the destination device allocation for the entire capture/graph/exec
// lifetime; otherwise a normal pinned-buffer close could create a graph UAF.
bool release_capture_h2d_leases(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kH2D ||
      capture->fill_buffer == nullptr || capture->h2d_source == nullptr ||
      !capture->fill_lease_held || !capture->h2d_source_lease_held ||
      capture->h2d_byte_len == 0 ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->h2d_source->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  // Both counters are verified before either deterministic 1->0 transition.
  if (!release_exclusive_use(capture->h2d_source->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->h2d_source = nullptr;
  capture->h2d_byte_len = 0;
  capture->h2d_enqueue_count = 0;
  capture->h2d_source_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-8 retains two distinct BF16 device allocations. Both are graph-visible
// raw addresses, so validate every immutable field and both 1->0 transitions
// before releasing either lease. A malformed raw ABI owner remains fail-closed.
bool release_capture_silu_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kSiluBf16 ||
      capture->fill_buffer == nullptr || capture->silu_input == nullptr ||
      capture->fill_buffer == capture->silu_input ||
      !capture->fill_lease_held || !capture->silu_input_lease_held ||
      capture->silu_element_count == 0 ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->silu_input->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  if (!release_exclusive_use(capture->silu_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->silu_input = nullptr;
  capture->silu_element_count = 0;
  capture->silu_enqueue_count = 0;
  capture->silu_input_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
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
  if (capture->operation == RileyCudaGraphCaptureOperation::kFillF32) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kFillF32 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kH2D) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kH2D ||
        graph->h2d_source != capture->h2d_source ||
        graph->h2d_byte_len != capture->h2d_byte_len ||
        graph->silu_input != nullptr || graph->silu_element_count != 0) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kSiluBf16) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kSiluBf16 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != capture->silu_input ||
        graph->silu_element_count != capture->silu_element_count) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kNone) {
    // C05-5's historical cleanup releases the fixed-buffer lease before it
    // frees this preallocated graph wrapper. That order is valid only for a
    // fill graph, which has no pinned source pointer to preserve.
    if (graph->operation != RileyCudaGraphCaptureOperation::kFillF32 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0) {
      return false;
    }
  } else {
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
      graph->fill_buffer != capture->fill_buffer ||
      graph->operation != capture->operation || graph->graph == nullptr ||
      graph->owns_capture_leases ||
      capture->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  if (capture->operation == RileyCudaGraphCaptureOperation::kFillF32) {
    if (graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
        capture->h2d_source_lease_held || graph->silu_input != nullptr ||
        graph->silu_element_count != 0 || capture->silu_input != nullptr ||
        capture->silu_element_count != 0 || capture->silu_input_lease_held) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kH2D) {
    if (capture->h2d_source == nullptr || !capture->h2d_source_lease_held ||
        capture->h2d_byte_len == 0 ||
        graph->h2d_source != capture->h2d_source ||
        graph->h2d_byte_len != capture->h2d_byte_len ||
        capture->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        capture->silu_input != nullptr || capture->silu_element_count != 0 ||
        capture->silu_input_lease_held) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kSiluBf16) {
    if (capture->silu_input == nullptr ||
        capture->silu_input == capture->fill_buffer ||
        !capture->silu_input_lease_held ||
        capture->silu_element_count == 0 ||
        graph->silu_input != capture->silu_input ||
        graph->silu_element_count != capture->silu_element_count ||
        capture->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
        capture->h2d_source_lease_held) {
      return false;
    }
  } else {
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
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->h2d_source = nullptr;
  capture->h2d_byte_len = 0;
  capture->h2d_enqueue_count = 0;
  capture->h2d_source_lease_held = false;
  capture->silu_input = nullptr;
  capture->silu_element_count = 0;
  capture->silu_enqueue_count = 0;
  capture->silu_input_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
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

// H2D graph ownership adds a pinned source lease to C05-5's existing stream
// and device-buffer set. Verify every resource before releasing any of them;
// an impossible raw-ABI corruption stays fail-closed rather than exposing a
// captured pointer whose lifetime is no longer known.
bool release_graph_h2d_leases(RileyCudaContext* owner, RileyCudaStream* stream,
                              RileyCudaDeviceBuffer* destination,
                              RileyCudaPinnedHostBuffer* source) noexcept {
  if (owner == nullptr || stream == nullptr || destination == nullptr ||
      source == nullptr || !same_context(owner, stream->owner) ||
      !same_context(owner, destination->owner) ||
      !same_context(owner, source->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      destination->active_uses.load(std::memory_order_acquire) != 1 ||
      source->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(source->active_uses) &&
         release_exclusive_use(destination->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_silu_bf16_leases(RileyCudaContext* owner,
                                    RileyCudaStream* stream,
                                    RileyCudaDeviceBuffer* input,
                                    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      output == nullptr || input == output || !same_context(owner, stream->owner) ||
      !same_context(owner, input->owner) || !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(input->active_uses) &&
         release_exclusive_use(output->active_uses) &&
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
  capture->operation = RileyCudaGraphCaptureOperation::kFillF32;
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
                                         capture_id,
                                         RileyCudaGraphCaptureOperation::kFillF32);
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

// C05-7 is deliberately a sibling admission path rather than an extension of
// the fixed-fill geometry. It captures exactly one whole-allocation H2D node
// and acquires all three permanent resource leases before cudaStreamBeginCapture
// can make their raw pointers observable to CUDA.
RileyCudaStatus capture_begin_h2d_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* destination,
    RileyCudaPinnedHostBuffer* source, RileyCudaGraphCaptureMode mode,
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
                            kBeginH2DOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || destination == nullptr || source == nullptr ||
      stream->owner == nullptr || destination->owner == nullptr ||
      source->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "stream, H2D source, destination, or their owner is null");
  }
  if (!same_context(stream->owner, destination->owner) ||
      !same_context(stream->owner, source->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "capture stream, H2D source, and destination must share one context owner");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "only thread-local capture mode is admitted");
  }
  if (source->byte_len == 0 || source->byte_len != destination->byte_len) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D requires equal nonzero whole source and destination slabs");
  }
  if (source->host_data == nullptr || destination->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D source or destination has no live allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "a stream command batch blocks fixed-address graph H2D capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginH2DOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(source->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D source has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(destination->active_uses)) {
    const bool source_released = release_exclusive_use(source->active_uses);
    if (!source_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release a rejected graph H2D source lease");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D destination has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool destination_released =
        release_exclusive_use(destination->active_uses);
    const bool source_released = release_exclusive_use(source->active_uses);
    if (!destination_released || !source_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release rejected graph H2D resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginH2DOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginH2DOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginH2DOperation,
                     "host allocation failed for graph H2D capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kH2D;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginH2DOperation,
                     "host allocation failed for captured graph H2D owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, destination, capture_id,
      RileyCudaGraphCaptureOperation::kH2D, source, source->byte_len);
  capture->fill_buffer = destination;
  capture->fill_lease_held = true;
  capture->h2d_source = source;
  capture->h2d_byte_len = source->byte_len;
  capture->h2d_source_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_h2d_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release a blocked graph H2D capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_h2d_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release a rejected graph H2D capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginH2DOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginH2DOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginH2DOperation);
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
  const bool leases_released = release_capture_h2d_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginH2DOperation,
                          "failed to release an unstarted graph H2D capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-8 is intentionally a separate sibling admission path. The two device
// pointers and fixed BF16 geometry are prepared before capture begins, and both
// device leases remain held through graph/exec close. It does not generalize
// the eager subspan/aliasing SiLU ABI.
RileyCudaStatus capture_begin_silu_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* output, uint64_t element_count,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
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
                            kBeginSiluBf16Operation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || output == nullptr ||
      stream->owner == nullptr || input->owner == nullptr ||
      output->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "stream, BF16 SiLU input, output, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "capture stream, BF16 SiLU input, and output must share one context owner");
  }
  if (input == output) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU requires distinct input and output allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  if (element_count == 0 ||
      element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      element_count > output->byte_len / sizeof(__nv_bfloat16)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "fixed BF16 SiLU element count exceeds an input or output allocation");
  }
  if (input->device_data == nullptr || output->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU input or output has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "a stream command batch blocks fixed-address BF16 SiLU graph capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginSiluBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(input->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(output->active_uses)) {
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!input_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginSiluBf16Operation,
                            "failed to release a rejected graph BF16 SiLU input lease");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU output has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool output_released = release_exclusive_use(output->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!output_released || !input_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginSiluBf16Operation,
                            "failed to release rejected graph BF16 SiLU resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginSiluBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginSiluBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginSiluBf16Operation,
                     "host allocation failed for graph BF16 SiLU capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kSiluBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginSiluBf16Operation,
                     "host allocation failed for captured graph BF16 SiLU owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kSiluBf16, nullptr, 0, input,
      element_count);
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->silu_input = input;
  capture->silu_element_count = element_count;
  capture->silu_input_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_silu_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginSiluBf16Operation,
          "failed to release a blocked graph BF16 SiLU capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_silu_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginSiluBf16Operation,
          "failed to release a rejected graph BF16 SiLU capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginSiluBf16Operation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginSiluBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginSiluBf16Operation);
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
  const bool leases_released = release_capture_silu_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginSiluBf16Operation,
                          "failed to release an unstarted graph BF16 SiLU capture owner");
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
      const bool is_h2d =
          owner->operation == RileyCudaGraphCaptureOperation::kH2D;
      const bool is_silu_bf16 =
          owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
      const bool is_fill_or_generic =
          owner->operation == RileyCudaGraphCaptureOperation::kFillF32 ||
          owner->operation == RileyCudaGraphCaptureOperation::kNone;
      const bool release_graph_first = is_h2d || is_silu_bf16;
      const bool prepared_graph_released =
          release_graph_first ? destroy_prepared_graph_storage(owner) : true;
      const bool operation_released =
          prepared_graph_released &&
          (is_h2d ? release_capture_h2d_leases(owner)
                  : is_silu_bf16 ? release_capture_silu_bf16_leases(owner)
                                 : is_fill_or_generic
                                       ? release_capture_fill_lease(owner)
                                       : false);
      const bool cleanup_graph_released =
          release_graph_first
              ? prepared_graph_released
              : operation_released && destroy_prepared_graph_storage(owner);
      released = cleanup_graph_released && operation_released &&
                 release_capture_owner(owner);
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

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_h2d(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* destination,
    RileyCudaPinnedHostBuffer* source, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_h2d_impl(stream, destination, source, mode, out_capture,
                                out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_silu_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* output, uint64_t element_count,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_silu_bf16_impl(stream, input, output, element_count,
                                      mode, out_capture, out_graph_error,
                                      error);
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
      owner->operation != RileyCudaGraphCaptureOperation::kFillF32 ||
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

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_h2d(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueH2DOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation, "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->h2d_source == nullptr ||
      owner->operation != RileyCudaGraphCaptureOperation::kH2D ||
      !owner->fill_lease_held || !owner->h2d_source_lease_held ||
      owner->capture_terminated || owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation,
                            "capture owner is not a live graph H2D capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueH2DOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started || owner->h2d_byte_len == 0 ||
      owner->h2d_byte_len != owner->fill_buffer->byte_len ||
      owner->h2d_byte_len != owner->h2d_source->byte_len ||
      owner->fill_buffer->device_data == nullptr ||
      owner->h2d_source->host_data == nullptr || owner->h2d_enqueue_count != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation,
                            "graph H2D capture has invalid immutable geometry or already enqueued its sole node");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation,
                            "graph H2D capture resource lease is unavailable");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kEnqueueH2DOperation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaMemcpyAsync(owner->fill_buffer->device_data, owner->h2d_source->host_data,
                        owner->h2d_byte_len, cudaMemcpyHostToDevice,
                        owner->stream->stream),
        error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueH2DOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->h2d_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueH2DOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_silu_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueSiluBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueSiluBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->silu_input == nullptr ||
      owner->operation != RileyCudaGraphCaptureOperation::kSiluBf16 ||
      !owner->fill_lease_held || !owner->silu_input_lease_held ||
      owner->capture_terminated || owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueSiluBf16Operation,
                            "capture owner is not a live graph BF16 SiLU capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueSiluBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started || owner->silu_input == owner->fill_buffer ||
      owner->silu_element_count == 0 ||
      owner->silu_element_count >
          owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
      owner->silu_element_count >
          owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
      owner->silu_input->device_data == nullptr ||
      owner->fill_buffer->device_data == nullptr ||
      owner->silu_enqueue_count != 0 || owner->h2d_source != nullptr ||
      owner->h2d_byte_len != 0 || owner->h2d_source_lease_held ||
      owner->fill_element_count != 0 || owner->fill_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueSiluBf16Operation,
        "graph BF16 SiLU capture has invalid immutable geometry or already enqueued its sole node");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueSiluBf16Operation,
                            "graph BF16 SiLU capture resource lease is unavailable");
  }
  const uint64_t needed_blocks =
      ((owner->silu_element_count - 1) / kGraphSiluThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      needed_blocks < kMaximumGraphSiluBlocks ? needed_blocks
                                               : kMaximumGraphSiluBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kEnqueueSiluBf16Operation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_silu_bf16<<<grid_x, kGraphSiluThreads, 0, owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(owner->silu_input->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->silu_element_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueSiluBf16Operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->silu_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueSiluBf16Operation);
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
                            "capture owner is not a live fixed-operation capture");
  }
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  if ((!is_fill && !is_h2d && !is_silu_bf16) ||
      (is_fill && (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
                   owner->h2d_source_lease_held || owner->silu_input != nullptr ||
                   owner->silu_element_count != 0 ||
                   owner->silu_enqueue_count != 0 ||
                   owner->silu_input_lease_held)) ||
      (is_h2d && (owner->h2d_source == nullptr ||
                  !owner->h2d_source_lease_held || owner->h2d_byte_len == 0 ||
                  owner->h2d_source->host_data == nullptr ||
                  owner->h2d_source->byte_len != owner->h2d_byte_len ||
                  owner->fill_buffer->byte_len != owner->h2d_byte_len ||
                  owner->silu_input != nullptr || owner->silu_element_count != 0 ||
                  owner->silu_enqueue_count != 0 ||
                  owner->silu_input_lease_held)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_enqueue_count != 0 || owner->h2d_source_lease_held ||
        owner->silu_input == nullptr ||
        owner->silu_input == owner->fill_buffer ||
        !owner->silu_input_lease_held || owner->silu_element_count == 0 ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->fill_element_count != 0 || owner->fill_enqueue_count != 0))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture owner has invalid fixed-operation geometry");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "thread-local capture must end on its begin thread");
  }
  if (!owner->capture_started || owner->capture_terminated ||
      (is_fill && owner->fill_enqueue_count == 0) ||
      (is_h2d && owner->h2d_enqueue_count != 1) ||
      (is_silu_bf16 && owner->silu_enqueue_count != 1)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture end requires its admitted operation enqueue contract");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      (is_h2d && owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1) ||
      (is_silu_bf16 &&
       owner->silu_input->active_uses.load(std::memory_order_acquire) != 1)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kEndOperation,
                          "fixed graph capture resource lease was corrupted");
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
      const bool is_h2d =
          owner->operation == RileyCudaGraphCaptureOperation::kH2D;
      const bool is_silu_bf16 =
          owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
      const bool is_fill =
          owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
      const bool release_graph_first = is_h2d || is_silu_bf16;
      const bool prepared_graph_released =
          release_graph_first ? destroy_prepared_graph_storage(owner) : true;
      const bool operation_released =
          prepared_graph_released &&
          (is_h2d ? release_capture_h2d_leases(owner)
                  : is_silu_bf16 ? release_capture_silu_bf16_leases(owner)
                                 : is_fill ? release_capture_fill_lease(owner)
                                           : false);
      const bool cleanup_graph_released =
          release_graph_first
              ? prepared_graph_released
              : operation_released && destroy_prepared_graph_storage(owner);
      released = cleanup_graph_released && operation_released &&
                 release_capture_owner(owner);
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
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      !owner->owns_capture_leases || (!is_fill && !is_h2d && !is_silu_bf16) ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph resource lease is invalid");
  }
  if ((is_fill &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0)) ||
      (is_h2d &&
       (owner->h2d_byte_len == 0 ||
        owner->h2d_source == nullptr ||
        !same_context(owner->owner, owner->h2d_source->owner) ||
        owner->h2d_source->host_data == nullptr ||
        owner->h2d_source->byte_len != owner->h2d_byte_len ||
        owner->fill_buffer->byte_len != owner->h2d_byte_len ||
        owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input == nullptr || owner->silu_input == owner->fill_buffer ||
        owner->silu_element_count == 0 ||
        !same_context(owner->owner, owner->silu_input->owner) ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_input->active_uses.load(std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph has invalid fixed-operation resource state");
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
      owner->owner, owner->stream, owner->fill_buffer, capture_id, exec_id,
      owner->operation, owner->h2d_source, owner->h2d_byte_len,
      owner->silu_input, owner->silu_element_count);

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

extern "C" RileyCudaStatus riley_cuda_graph_exec_stage_h2d_source(
    RileyCudaGraphExec* exec, RileyCudaPinnedHostBuffer* source,
    const uint8_t* bytes, uint64_t byte_len,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kStageH2DOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE);
  if (exec == nullptr || source == nullptr || bytes == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kStageH2DOperation,
                            "graph H2D exec, retained source, or payload is null");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE,
                       capture_id, exec_id, false, false, false, false);
  if (exec->owner == nullptr || exec->stream == nullptr ||
      exec->fill_buffer == nullptr || exec->h2d_source == nullptr ||
      exec->operation != RileyCudaGraphCaptureOperation::kH2D ||
      exec->h2d_source != source || exec->graph == nullptr ||
      exec->exec == nullptr || !exec->owns_capture_leases ||
      exec->h2d_byte_len == 0 || byte_len != exec->h2d_byte_len ||
      source->host_data == nullptr || source->byte_len != exec->h2d_byte_len ||
      exec->fill_buffer->byte_len != exec->h2d_byte_len ||
      !same_context(exec->owner, exec->stream->owner) ||
      !same_context(exec->owner, exec->fill_buffer->owner) ||
      !same_context(exec->owner, source->owner) ||
      exec->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      source->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->silu_input != nullptr || exec->silu_element_count != 0 ||
      exec->launch_in_flight || exec->h2d_input_staged || exec->poisoned ||
      exec->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kStageH2DOperation,
                            "graph H2D exec is busy, poisoned, or lost its exact retained resource lease");
  }
  // The graph node retains this exact pinned allocation address. This private
  // stage is deliberately the sole mutable path while its normal active-use
  // guard remains held; no CUDA call, node update, or pointer mutation occurs.
  std::memmove(source->host_data, bytes, static_cast<size_t>(byte_len));
  exec->h2d_input_staged = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE,
                       capture_id, exec_id, false, false, false, false);
  return RILEY_CUDA_STATUS_SUCCESS;
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
  const bool is_fill =
      exec->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = exec->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      exec->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                       capture_id, exec_id, false, false, false, false);
  if (exec->owner == nullptr || exec->stream == nullptr ||
      exec->fill_buffer == nullptr || exec->graph == nullptr ||
      exec->exec == nullptr || !exec->owns_capture_leases ||
      (!is_fill && !is_h2d && !is_silu_bf16)) {
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
  if ((is_fill &&
       (exec->h2d_source != nullptr || exec->h2d_byte_len != 0 ||
        exec->h2d_input_staged || exec->silu_input != nullptr ||
        exec->silu_element_count != 0)) ||
      (is_h2d &&
       (exec->h2d_byte_len == 0 ||
        exec->h2d_source == nullptr ||
        !same_context(exec->owner, exec->h2d_source->owner) ||
        exec->h2d_source->host_data == nullptr ||
        exec->h2d_source->byte_len != exec->h2d_byte_len ||
        exec->fill_buffer->byte_len != exec->h2d_byte_len ||
        exec->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        !exec->h2d_input_staged || exec->silu_input != nullptr ||
        exec->silu_element_count != 0)) ||
      (is_silu_bf16 &&
       (exec->h2d_source != nullptr || exec->h2d_byte_len != 0 ||
        exec->h2d_input_staged || exec->silu_input == nullptr ||
        exec->silu_input == exec->fill_buffer ||
        exec->silu_element_count == 0 ||
        !same_context(exec->owner, exec->silu_input->owner) ||
        exec->silu_input->device_data == nullptr ||
        exec->fill_buffer->device_data == nullptr ||
        exec->silu_element_count >
            exec->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        exec->silu_element_count >
            exec->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        exec->silu_input->active_uses.load(std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec has invalid fixed-operation replay state");
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
    if (is_h2d) {
      // A launch attempt consumes its stage even if CUDA subsequently reports
      // a deferred error. Completion never restores this bit: every replay
      // must explicitly stage a new exact payload.
      exec->h2d_input_staged = false;
    }
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
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, 0, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      !owner->owns_capture_leases || (!is_fill && !is_h2d && !is_silu_bf16) ||
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
  if ((is_fill &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0)) ||
      (is_h2d &&
       (owner->h2d_byte_len == 0 ||
        owner->h2d_source == nullptr ||
        !same_context(owner->owner, owner->h2d_source->owner) ||
        owner->h2d_source->host_data == nullptr ||
        owner->h2d_source->byte_len != owner->h2d_byte_len ||
        owner->fill_buffer->byte_len != owner->h2d_byte_len ||
        owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input == nullptr || owner->silu_input == owner->fill_buffer ||
        owner->silu_element_count == 0 ||
        !same_context(owner->owner, owner->silu_input->owner) ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_input->active_uses.load(std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation,
                            "captured graph has invalid fixed-operation resource state");
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
        is_h2d ? release_graph_h2d_leases(owner->owner, owner->stream,
                                           owner->fill_buffer,
                                           owner->h2d_source)
               : is_silu_bf16
                     ? release_graph_silu_bf16_leases(owner->owner,
                                                       owner->stream,
                                                       owner->silu_input,
                                                       owner->fill_buffer)
                     : release_graph_leases(owner->owner, owner->stream,
                                            owner->fill_buffer);
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
                            "failed to release graph stream, retained buffers, or context lease");
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
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, exec_id, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      owner->exec == nullptr || !owner->owns_capture_leases ||
      (!is_fill && !is_h2d && !is_silu_bf16) ||
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
  if ((is_fill &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_input_staged || owner->silu_input != nullptr ||
        owner->silu_element_count != 0)) ||
      (is_h2d &&
       (owner->h2d_byte_len == 0 ||
        owner->h2d_source == nullptr ||
        !same_context(owner->owner, owner->h2d_source->owner) ||
        owner->h2d_source->host_data == nullptr ||
        owner->h2d_source->byte_len != owner->h2d_byte_len ||
        owner->fill_buffer->byte_len != owner->h2d_byte_len ||
        owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_input_staged || owner->silu_input == nullptr ||
        owner->silu_input == owner->fill_buffer ||
        owner->silu_element_count == 0 ||
        !same_context(owner->owner, owner->silu_input->owner) ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_input->active_uses.load(std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation,
                            "graph exec has invalid fixed-operation resource state");
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
        is_h2d ? release_graph_h2d_leases(owner->owner, owner->stream,
                                           owner->fill_buffer,
                                           owner->h2d_source)
               : is_silu_bf16
                     ? release_graph_silu_bf16_leases(owner->owner,
                                                       owner->stream,
                                                       owner->silu_input,
                                                       owner->fill_buffer)
                     : release_graph_leases(owner->owner, owner->stream,
                                            owner->fill_buffer);
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
                            "failed to release graph exec stream, retained buffers, or context lease");
  }
  owner->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, exec_id, false, false, false, true);
  return status;
}
