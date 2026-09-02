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

// CUDA's primary context is shared by every RileyCudaContext for one device.
// Keep a small process-lifetime domain per device so a capture begun through
// one wrapper can safely gate context-wide controls reached through another.
// Domains are intentionally never reclaimed: a CUDA primary context can be
// retained by code outside this ABI, so retiring shared host metadata would
// otherwise create a use-after-free race. The set is bounded by the devices a
// process can observe.
struct RileyCudaCaptureDomain {
  explicit RileyCudaCaptureDomain(CUdevice selected_device) noexcept
      : device(selected_device),
        active_captures(0),
        broad_control_uses(0),
        pending_smoke_fills(0),
        pending_copies(0),
        next(nullptr) {}

  CUdevice device;
  std::atomic_flag admission_lock = ATOMIC_FLAG_INIT;
  std::atomic<uint32_t> active_captures;
  std::atomic<uint32_t> broad_control_uses;
  // A diagnostic smoke buffer owns allocation, completion, and free work that
  // cannot be abandoned into a live capture. This count remains published from
  // before allocation through successful native-buffer consumption.
  std::atomic<uint32_t> pending_smoke_fills;
  // A pending asynchronous copy owns Rust-side buffer-borrow state that must
  // settle through CUDA synchronization. Keep it visible from before enqueue
  // through native-copy consumption so capture cannot strand that owner.
  std::atomic<uint32_t> pending_copies;
  RileyCudaCaptureDomain* next;
};

struct RileyCudaContext;
struct RileyCudaGraphCapture;
struct RileyCudaGraph;
struct RileyCudaGraphExec;
struct RileyCudaGraphLaunch;
struct RileyCudaDeviceBuffer;
struct RileyCudaPinnedHostBuffer;
struct RileyCudaDeferredCloseNode;

// `consumed` records native close ownership independently of status. CUDA may
// report a deferred error after consuming an object, so a consumed node must be
// removed from the FIFO even if `status` is non-success; retaining it would be
// a dangling pointer. A non-consumed failed node stays queued fail-closed.
struct RileyCudaDeferredCloseResult {
  RileyCudaStatus status;
  bool consumed;
};

// A deferred callback owns the payload and may destroy the containing node.
// The exact active capture owner is supplied so callbacks may enter the
// resource's own (including foreign) context through CurrentContext.
using RileyCudaDeferredCloseCallback = RileyCudaDeferredCloseResult (*)(
    RileyCudaDeferredCloseNode* node,
    const RileyCudaGraphCapture* capture_owner,
    RileyCudaErrorInfo* error) noexcept;

// This node is embedded in a resource that has a CUDA-bearing close path. No
// allocation occurs when a safe close transfers that resource to an active
// capture. `owner` is the resource's actual context, not necessarily the
// capture owner's context; that distinction is required for foreign wrappers
// sharing the same primary-device capture domain.
struct RileyCudaDeferredCloseNode {
  RileyCudaDeferredCloseNode() noexcept
      : next(nullptr),
        owner(nullptr),
        payload(nullptr),
        callback(nullptr),
        queued(false) {}

  RileyCudaDeferredCloseNode* next;
  RileyCudaContext* owner;
  void* payload;
  RileyCudaDeferredCloseCallback callback;
  bool queued;
};

struct RileyCudaContext {
  RileyCudaContext(CUdevice selected_device, CUcontext primary_context,
                   int32_t device_ordinal,
                   RileyCudaCaptureDomain* selected_capture_domain) noexcept
      : device(selected_device),
        context(primary_context),
        ordinal(device_ordinal),
        capture_domain(selected_capture_domain),
        deferred_close(),
        live_children(0),
        restoration_failed(false),
        device_live_bytes(0),
        device_live_allocations(0),
        pinned_host_live_bytes(0),
        pinned_host_live_allocations(0) {}

  CUdevice device;
  CUcontext context;
  int32_t ordinal;
  RileyCudaCaptureDomain* capture_domain;
  RileyCudaDeferredCloseNode deferred_close;
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
        deferred_close(),
        command_batch_owner(nullptr),
        command_batch_use_count(0),
        command_batch_uses{} {}

  RileyCudaContext* owner;
  cudaStream_t stream;
  // One exclusive asynchronous-use lease covers copies and synchronously
  // completing primitives. A stuck value is an intentional fail-closed leak.
  std::atomic<uint32_t> active_uses;
  RileyCudaDeferredCloseNode deferred_close;
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

// An active ThreadLocal stream capture owns this wrapper until a single
// cudaStreamEndCapture attempt has a fully known recovery result. Keeping the
// stream's active-use lease in parallel prevents unrelated entry points from
// dereferencing the stream while CUDA is capturing it.
enum class RileyCudaGraphCaptureOperation : uint8_t {
  kNone = 0,
  kFillF32 = 1,
  kH2D = 2,
  kSiluBf16 = 3,
  kGatedMultiplyBf16 = 4,
  kResidualAddBf16 = 5,
  kCanonicalRmsNormBf16 = 6,
  kBf16Argmax = 7,
  kBf16RowGather = 8,
  kBf16RowGatherArgmax = 9,
  kBf16RowGatherArgmaxD2H = 10,
  kIndexedRopeBf16 = 11,
  kRaggedPagedKvCacheWriteBf16 = 12,
};

// C05-18's raw device metadata has no host-side lifetime. The primary key
// pool stays in the legacy fill_buffer slot so its lease follows the existing
// graph lifecycle; these eight booleans cover the remaining fixed allocations.
struct RileyCudaRaggedPagedKvCacheWriteBf16State {
  RileyCudaRaggedPagedKvCacheWriteBf16State() noexcept
      : key_source(nullptr),
        value_source(nullptr),
        key_pool(nullptr),
        value_pool(nullptr),
        sequence_block_offsets(nullptr),
        block_ids(nullptr),
        valid_tokens(nullptr),
        row_sequence_slots(nullptr),
        row_positions(nullptr),
        sequence_count(0),
        block_count(0),
        active_row_count(0),
        physical_block_count(0),
        key_value_head_count(0),
        head_size(0),
        enqueue_count(0),
        key_source_lease_held(false),
        value_source_lease_held(false),
        value_pool_lease_held(false),
        sequence_block_offsets_lease_held(false),
        block_ids_lease_held(false),
        valid_tokens_lease_held(false),
        row_sequence_slots_lease_held(false),
        row_positions_lease_held(false) {}

  RileyCudaDeviceBuffer* key_source;
  RileyCudaDeviceBuffer* value_source;
  RileyCudaDeviceBuffer* key_pool;
  RileyCudaDeviceBuffer* value_pool;
  RileyCudaDeviceBuffer* sequence_block_offsets;
  RileyCudaDeviceBuffer* block_ids;
  RileyCudaDeviceBuffer* valid_tokens;
  RileyCudaDeviceBuffer* row_sequence_slots;
  RileyCudaDeviceBuffer* row_positions;
  uint64_t sequence_count;
  uint64_t block_count;
  uint64_t active_row_count;
  uint64_t physical_block_count;
  uint64_t key_value_head_count;
  uint64_t head_size;
  uint32_t enqueue_count;
  bool key_source_lease_held;
  bool value_source_lease_held;
  bool value_pool_lease_held;
  bool sequence_block_offsets_lease_held;
  bool block_ids_lease_held;
  bool valid_tokens_lease_held;
  bool row_sequence_slots_lease_held;
  bool row_positions_lease_held;
};

struct RileyCudaGraphCapture {
  RileyCudaGraphCapture(RileyCudaContext* owning_context,
                        RileyCudaStream* captured_stream,
                        RileyCudaCaptureDomain* owning_capture_domain,
                        const void* capture_thread,
                        uint64_t identifier) noexcept
      : owner(owning_context),
        stream(captured_stream),
        capture_domain(owning_capture_domain),
        owner_thread(capture_thread),
        capture_id(identifier),
        capture_started(false),
        capture_terminated(false),
        prepared_graph(nullptr),
        operation(RileyCudaGraphCaptureOperation::kNone),
        fill_buffer(nullptr),
        fill_element_count(0),
        fill_enqueue_count(0),
        fill_lease_held(false),
        h2d_source(nullptr),
        h2d_byte_len(0),
        h2d_enqueue_count(0),
        h2d_source_lease_held(false),
        silu_input(nullptr),
        silu_element_count(0),
        silu_enqueue_count(0),
        silu_input_lease_held(false),
        gated_multiply_activated_gate(nullptr),
        gated_multiply_up(nullptr),
        gated_multiply_element_count(0),
        gated_multiply_enqueue_count(0),
        gated_multiply_activated_gate_lease_held(false),
        gated_multiply_up_lease_held(false),
        residual_add_left(nullptr),
        residual_add_right(nullptr),
        residual_add_element_count(0),
        residual_add_enqueue_count(0),
        residual_add_left_lease_held(false),
        residual_add_right_lease_held(false),
        canonical_rms_norm_input(nullptr),
        canonical_rms_norm_weight(nullptr),
        canonical_rms_norm_row_count(0),
        canonical_rms_norm_hidden_size(0),
        canonical_rms_norm_epsilon(0.0F),
        canonical_rms_norm_enqueue_count(0),
        canonical_rms_norm_input_lease_held(false),
        canonical_rms_norm_weight_lease_held(false),
        bf16_argmax_logits(nullptr),
        bf16_argmax_row_count(0),
        bf16_argmax_vocabulary_size(0),
        bf16_argmax_enqueue_count(0),
        bf16_argmax_logits_lease_held(false),
        bf16_row_gather_input(nullptr),
        bf16_row_gather_indices(nullptr),
        bf16_row_gather_input_row_count(0),
        bf16_row_gather_output_row_count(0),
        bf16_row_gather_column_count(0),
        bf16_row_gather_enqueue_count(0),
        bf16_row_gather_input_lease_held(false),
        bf16_row_gather_indices_lease_held(false),
        bf16_row_gather_argmax_input(nullptr),
        bf16_row_gather_argmax_indices(nullptr),
        bf16_row_gather_argmax_gathered_logits(nullptr),
        bf16_row_gather_argmax_input_row_count(0),
        bf16_row_gather_argmax_output_row_count(0),
        bf16_row_gather_argmax_vocabulary_size(0),
        bf16_row_gather_argmax_enqueue_count(0),
        bf16_row_gather_argmax_input_lease_held(false),
        bf16_row_gather_argmax_indices_lease_held(false),
        bf16_row_gather_argmax_gathered_logits_lease_held(false),
        bf16_row_gather_argmax_d2h_input(nullptr),
        bf16_row_gather_argmax_d2h_indices(nullptr),
        bf16_row_gather_argmax_d2h_gathered_logits(nullptr),
        bf16_row_gather_argmax_d2h_pinned_results(nullptr),
        bf16_row_gather_argmax_d2h_input_row_count(0),
        bf16_row_gather_argmax_d2h_output_row_count(0),
        bf16_row_gather_argmax_d2h_vocabulary_size(0),
        bf16_row_gather_argmax_d2h_result_byte_len(0),
        bf16_row_gather_argmax_d2h_enqueue_count(0),
        bf16_row_gather_argmax_d2h_input_lease_held(false),
        bf16_row_gather_argmax_d2h_indices_lease_held(false),
        bf16_row_gather_argmax_d2h_gathered_logits_lease_held(false),
        bf16_row_gather_argmax_d2h_pinned_results_lease_held(false),
        indexed_rope_bf16_input(nullptr),
        indexed_rope_bf16_cos(nullptr),
        indexed_rope_bf16_sin(nullptr),
        indexed_rope_bf16_positions(nullptr),
        indexed_rope_bf16_active_row_count(0),
        indexed_rope_bf16_head_count(0),
        indexed_rope_bf16_head_size(0),
        indexed_rope_bf16_rotary_dimension(0),
        indexed_rope_bf16_table_position_count(0),
        indexed_rope_bf16_enqueue_count(0),
        indexed_rope_bf16_input_lease_held(false),
        indexed_rope_bf16_cos_lease_held(false),
        indexed_rope_bf16_sin_lease_held(false),
        indexed_rope_bf16_positions_lease_held(false),
        deferred_close_head(nullptr),
        deferred_close_tail(nullptr),
        unreleased_graph(nullptr) {}

  RileyCudaContext* owner;
  RileyCudaStream* stream;
  RileyCudaCaptureDomain* capture_domain;
  const void* owner_thread;
  uint64_t capture_id;
  bool capture_started;
  // Set only after end capture, returned-graph destruction, and the capture
  // context restoration are all known. Deferred context release may use this
  // narrow post-physical-capture state while the TLS owner stays published.
  bool capture_terminated;
  // C05-5's sole capture whitelist is a fixed-address f32 fill. Its graph
  // wrapper is allocated before cudaStreamBeginCapture, so capture enqueue
  // itself cannot allocate host bookkeeping. The buffer's active-use lease is
  // acquired before begin and transfers to the graph/exec after successful
  // end; abort releases it with the capture's other leases.
  RileyCudaGraph* prepared_graph;
  // The current C05 operation family. `fill_buffer` continues to name the
  // retained primary device allocation for ABI continuity: it is the fill
  // target, H2D destination, or BF16 SiLU output. H2D retains a fixed pinned
  // source, while the BF16 primitives retain their fixed device inputs. Gated
  // multiply keeps both its activated-gate and up inputs; residual add keeps
  // its left and right inputs; canonical RMSNorm keeps its input and weight;
  // deterministic BF16 argmax keeps its logits while fill_buffer is results;
  // BF16 row gather keeps input and U32 indices while fill_buffer is output;
  // C05-15 keeps input, U32 indices, and gathered BF16 logits while
  // fill_buffer is deterministic argmax results.
  // C05-16 retains the same four device allocations plus an exact pinned
  // result destination for its captured D2H node. C05-17 retains BF16 input
  // and output (through fill_buffer), F32 cosine/sine tables, and U32 device
  // positions; its temporary host position mirror is never retained.
  RileyCudaGraphCaptureOperation operation;
  RileyCudaDeviceBuffer* fill_buffer;
  uint64_t fill_element_count;
  uint32_t fill_enqueue_count;
  bool fill_lease_held;
  RileyCudaPinnedHostBuffer* h2d_source;
  uint64_t h2d_byte_len;
  uint32_t h2d_enqueue_count;
  bool h2d_source_lease_held;
  RileyCudaDeviceBuffer* silu_input;
  uint64_t silu_element_count;
  uint32_t silu_enqueue_count;
  bool silu_input_lease_held;
  RileyCudaDeviceBuffer* gated_multiply_activated_gate;
  RileyCudaDeviceBuffer* gated_multiply_up;
  uint64_t gated_multiply_element_count;
  uint32_t gated_multiply_enqueue_count;
  bool gated_multiply_activated_gate_lease_held;
  bool gated_multiply_up_lease_held;
  RileyCudaDeviceBuffer* residual_add_left;
  RileyCudaDeviceBuffer* residual_add_right;
  uint64_t residual_add_element_count;
  uint32_t residual_add_enqueue_count;
  bool residual_add_left_lease_held;
  bool residual_add_right_lease_held;
  RileyCudaDeviceBuffer* canonical_rms_norm_input;
  RileyCudaDeviceBuffer* canonical_rms_norm_weight;
  uint64_t canonical_rms_norm_row_count;
  uint64_t canonical_rms_norm_hidden_size;
  float canonical_rms_norm_epsilon;
  uint32_t canonical_rms_norm_enqueue_count;
  bool canonical_rms_norm_input_lease_held;
  bool canonical_rms_norm_weight_lease_held;
  RileyCudaDeviceBuffer* bf16_argmax_logits;
  uint64_t bf16_argmax_row_count;
  uint64_t bf16_argmax_vocabulary_size;
  uint32_t bf16_argmax_enqueue_count;
  bool bf16_argmax_logits_lease_held;
  RileyCudaDeviceBuffer* bf16_row_gather_input;
  RileyCudaDeviceBuffer* bf16_row_gather_indices;
  uint64_t bf16_row_gather_input_row_count;
  uint64_t bf16_row_gather_output_row_count;
  uint64_t bf16_row_gather_column_count;
  uint32_t bf16_row_gather_enqueue_count;
  bool bf16_row_gather_input_lease_held;
  bool bf16_row_gather_indices_lease_held;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_input;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_indices;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_gathered_logits;
  uint64_t bf16_row_gather_argmax_input_row_count;
  uint64_t bf16_row_gather_argmax_output_row_count;
  uint64_t bf16_row_gather_argmax_vocabulary_size;
  uint32_t bf16_row_gather_argmax_enqueue_count;
  bool bf16_row_gather_argmax_input_lease_held;
  bool bf16_row_gather_argmax_indices_lease_held;
  bool bf16_row_gather_argmax_gathered_logits_lease_held;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_input;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_indices;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_gathered_logits;
  RileyCudaPinnedHostBuffer* bf16_row_gather_argmax_d2h_pinned_results;
  uint64_t bf16_row_gather_argmax_d2h_input_row_count;
  uint64_t bf16_row_gather_argmax_d2h_output_row_count;
  uint64_t bf16_row_gather_argmax_d2h_vocabulary_size;
  uint64_t bf16_row_gather_argmax_d2h_result_byte_len;
  uint32_t bf16_row_gather_argmax_d2h_enqueue_count;
  bool bf16_row_gather_argmax_d2h_input_lease_held;
  bool bf16_row_gather_argmax_d2h_indices_lease_held;
  bool bf16_row_gather_argmax_d2h_gathered_logits_lease_held;
  bool bf16_row_gather_argmax_d2h_pinned_results_lease_held;
  RileyCudaDeviceBuffer* indexed_rope_bf16_input;
  RileyCudaDeviceBuffer* indexed_rope_bf16_cos;
  RileyCudaDeviceBuffer* indexed_rope_bf16_sin;
  RileyCudaDeviceBuffer* indexed_rope_bf16_positions;
  uint64_t indexed_rope_bf16_active_row_count;
  uint64_t indexed_rope_bf16_head_count;
  uint64_t indexed_rope_bf16_head_size;
  uint64_t indexed_rope_bf16_rotary_dimension;
  uint64_t indexed_rope_bf16_table_position_count;
  uint32_t indexed_rope_bf16_enqueue_count;
  bool indexed_rope_bf16_input_lease_held;
  bool indexed_rope_bf16_cos_lease_held;
  bool indexed_rope_bf16_sin_lease_held;
  bool indexed_rope_bf16_positions_lease_held;
  RileyCudaRaggedPagedKvCacheWriteBf16State ragged_paged_kv_write_bf16;
  // Capture-thread-only FIFO. A successful callback can free its node, so the
  // drain saves `next` before invoking it and never touches that node again.
  RileyCudaDeferredCloseNode* deferred_close_head;
  RileyCudaDeferredCloseNode* deferred_close_tail;
  // Non-null only after an end/destroy ambiguity. The wrapper is intentionally
  // leaked together with its context child lease; retrying cudaGraphDestroy
  // could double-destroy a graph consumed before a deferred CUDA error.
  cudaGraph_t unreleased_graph;
};

struct RileyCudaEvent {
  RileyCudaContext* owner;
  cudaEvent_t event;
  RileyCudaDeferredCloseNode deferred_close;
};

struct RileyCudaSmokeBuffer {
  RileyCudaContext* owner;
  float* device_data;
  uint64_t element_count;
  bool in_flight;
  // Every successful create, including zero-element diagnostic fills, reserves
  // the primary-context capture domain. It is released only when native buffer
  // consumption is known, so capture cannot strand a recoverable Rust Drop
  // path.
  bool capture_admission_held;
  cudaStream_t launch_stream;
};

struct RileyCudaDeviceBuffer {
  RileyCudaDeviceBuffer(RileyCudaContext* owning_context,
                            void* allocation, uint64_t allocation_bytes) noexcept
      : owner(owning_context),
        device_data(allocation),
        byte_len(allocation_bytes),
        active_uses(0),
        deferred_close() {}

  RileyCudaContext* owner;
  void* device_data;
  uint64_t byte_len;
  std::atomic<uint32_t> active_uses;
  RileyCudaDeferredCloseNode deferred_close;
};

// A captured graph owns the existing capture context-child lease, the exact
// captured stream's active-use lease, and the fixed fill buffer's active-use
// lease. Those leases remain at one for the whole graph/exec lifetime. This
// intentionally makes ordinary stream/buffer operations busy while preserving
// a stable raw stream and device address for graph replay.
struct RileyCudaGraph {
  RileyCudaGraph(RileyCudaContext* owning_context,
                 RileyCudaStream* captured_stream,
                 RileyCudaDeviceBuffer* captured_fill_buffer,
                 uint64_t capture_identifier,
                 RileyCudaGraphCaptureOperation captured_operation,
                 RileyCudaPinnedHostBuffer* captured_h2d_source = nullptr,
                 uint64_t captured_h2d_byte_len = 0,
                 RileyCudaDeviceBuffer* captured_silu_input = nullptr,
                 uint64_t captured_silu_element_count = 0,
                 RileyCudaDeviceBuffer* captured_gated_multiply_activated_gate = nullptr,
                 RileyCudaDeviceBuffer* captured_gated_multiply_up = nullptr,
                 uint64_t captured_gated_multiply_element_count = 0,
                 RileyCudaDeviceBuffer* captured_residual_add_left = nullptr,
                 RileyCudaDeviceBuffer* captured_residual_add_right = nullptr,
                 uint64_t captured_residual_add_element_count = 0,
                 RileyCudaDeviceBuffer* captured_canonical_rms_norm_input = nullptr,
                 RileyCudaDeviceBuffer* captured_canonical_rms_norm_weight = nullptr,
                 uint64_t captured_canonical_rms_norm_row_count = 0,
                 uint64_t captured_canonical_rms_norm_hidden_size = 0,
                 float captured_canonical_rms_norm_epsilon = 0.0F,
                 RileyCudaDeviceBuffer* captured_bf16_argmax_logits = nullptr,
                 uint64_t captured_bf16_argmax_row_count = 0,
                 uint64_t captured_bf16_argmax_vocabulary_size = 0,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_input = nullptr,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_indices = nullptr,
                 uint64_t captured_bf16_row_gather_input_row_count = 0,
                 uint64_t captured_bf16_row_gather_output_row_count = 0,
                 uint64_t captured_bf16_row_gather_column_count = 0,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_input = nullptr,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_indices = nullptr,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_gathered_logits = nullptr,
                 uint64_t captured_bf16_row_gather_argmax_input_row_count = 0,
                 uint64_t captured_bf16_row_gather_argmax_output_row_count = 0,
                 uint64_t captured_bf16_row_gather_argmax_vocabulary_size = 0,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_d2h_input = nullptr,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_d2h_indices = nullptr,
                 RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_d2h_gathered_logits = nullptr,
                 RileyCudaPinnedHostBuffer* captured_bf16_row_gather_argmax_d2h_pinned_results = nullptr,
                 uint64_t captured_bf16_row_gather_argmax_d2h_input_row_count = 0,
                 uint64_t captured_bf16_row_gather_argmax_d2h_output_row_count = 0,
                 uint64_t captured_bf16_row_gather_argmax_d2h_vocabulary_size = 0,
                 uint64_t captured_bf16_row_gather_argmax_d2h_result_byte_len = 0,
                 RileyCudaDeviceBuffer* captured_indexed_rope_bf16_input = nullptr,
                 RileyCudaDeviceBuffer* captured_indexed_rope_bf16_cos = nullptr,
                 RileyCudaDeviceBuffer* captured_indexed_rope_bf16_sin = nullptr,
                 RileyCudaDeviceBuffer* captured_indexed_rope_bf16_positions = nullptr,
                 uint64_t captured_indexed_rope_bf16_active_row_count = 0,
                 uint64_t captured_indexed_rope_bf16_head_count = 0,
                 uint64_t captured_indexed_rope_bf16_head_size = 0,
                 uint64_t captured_indexed_rope_bf16_rotary_dimension = 0,
                 uint64_t captured_indexed_rope_bf16_table_position_count = 0) noexcept
      : owner(owning_context),
        stream(captured_stream),
        fill_buffer(captured_fill_buffer),
        operation(captured_operation),
        h2d_source(captured_h2d_source),
        h2d_byte_len(captured_h2d_byte_len),
        silu_input(captured_silu_input),
        silu_element_count(captured_silu_element_count),
        gated_multiply_activated_gate(captured_gated_multiply_activated_gate),
        gated_multiply_up(captured_gated_multiply_up),
        gated_multiply_element_count(captured_gated_multiply_element_count),
        residual_add_left(captured_residual_add_left),
        residual_add_right(captured_residual_add_right),
        residual_add_element_count(captured_residual_add_element_count),
        canonical_rms_norm_input(captured_canonical_rms_norm_input),
        canonical_rms_norm_weight(captured_canonical_rms_norm_weight),
        canonical_rms_norm_row_count(captured_canonical_rms_norm_row_count),
        canonical_rms_norm_hidden_size(captured_canonical_rms_norm_hidden_size),
        canonical_rms_norm_epsilon(captured_canonical_rms_norm_epsilon),
        bf16_argmax_logits(captured_bf16_argmax_logits),
        bf16_argmax_row_count(captured_bf16_argmax_row_count),
        bf16_argmax_vocabulary_size(captured_bf16_argmax_vocabulary_size),
        bf16_row_gather_input(captured_bf16_row_gather_input),
        bf16_row_gather_indices(captured_bf16_row_gather_indices),
        bf16_row_gather_input_row_count(
            captured_bf16_row_gather_input_row_count),
        bf16_row_gather_output_row_count(
            captured_bf16_row_gather_output_row_count),
        bf16_row_gather_column_count(captured_bf16_row_gather_column_count),
        bf16_row_gather_argmax_input(captured_bf16_row_gather_argmax_input),
        bf16_row_gather_argmax_indices(captured_bf16_row_gather_argmax_indices),
        bf16_row_gather_argmax_gathered_logits(
            captured_bf16_row_gather_argmax_gathered_logits),
        bf16_row_gather_argmax_input_row_count(
            captured_bf16_row_gather_argmax_input_row_count),
        bf16_row_gather_argmax_output_row_count(
            captured_bf16_row_gather_argmax_output_row_count),
        bf16_row_gather_argmax_vocabulary_size(
            captured_bf16_row_gather_argmax_vocabulary_size),
        bf16_row_gather_argmax_d2h_input(
            captured_bf16_row_gather_argmax_d2h_input),
        bf16_row_gather_argmax_d2h_indices(
            captured_bf16_row_gather_argmax_d2h_indices),
        bf16_row_gather_argmax_d2h_gathered_logits(
            captured_bf16_row_gather_argmax_d2h_gathered_logits),
        bf16_row_gather_argmax_d2h_pinned_results(
            captured_bf16_row_gather_argmax_d2h_pinned_results),
        bf16_row_gather_argmax_d2h_input_row_count(
            captured_bf16_row_gather_argmax_d2h_input_row_count),
        bf16_row_gather_argmax_d2h_output_row_count(
            captured_bf16_row_gather_argmax_d2h_output_row_count),
        bf16_row_gather_argmax_d2h_vocabulary_size(
            captured_bf16_row_gather_argmax_d2h_vocabulary_size),
        bf16_row_gather_argmax_d2h_result_byte_len(
            captured_bf16_row_gather_argmax_d2h_result_byte_len),
        indexed_rope_bf16_input(captured_indexed_rope_bf16_input),
        indexed_rope_bf16_cos(captured_indexed_rope_bf16_cos),
        indexed_rope_bf16_sin(captured_indexed_rope_bf16_sin),
        indexed_rope_bf16_positions(captured_indexed_rope_bf16_positions),
        indexed_rope_bf16_active_row_count(
            captured_indexed_rope_bf16_active_row_count),
        indexed_rope_bf16_head_count(captured_indexed_rope_bf16_head_count),
        indexed_rope_bf16_head_size(captured_indexed_rope_bf16_head_size),
        indexed_rope_bf16_rotary_dimension(
            captured_indexed_rope_bf16_rotary_dimension),
        indexed_rope_bf16_table_position_count(
            captured_indexed_rope_bf16_table_position_count),
        capture_id(capture_identifier),
        graph(nullptr),
        owns_capture_leases(false) {}

  RileyCudaContext* owner;
  RileyCudaStream* stream;
  RileyCudaDeviceBuffer* fill_buffer;
  RileyCudaGraphCaptureOperation operation;
  RileyCudaPinnedHostBuffer* h2d_source;
  uint64_t h2d_byte_len;
  RileyCudaDeviceBuffer* silu_input;
  uint64_t silu_element_count;
  RileyCudaDeviceBuffer* gated_multiply_activated_gate;
  RileyCudaDeviceBuffer* gated_multiply_up;
  uint64_t gated_multiply_element_count;
  RileyCudaDeviceBuffer* residual_add_left;
  RileyCudaDeviceBuffer* residual_add_right;
  uint64_t residual_add_element_count;
  RileyCudaDeviceBuffer* canonical_rms_norm_input;
  RileyCudaDeviceBuffer* canonical_rms_norm_weight;
  uint64_t canonical_rms_norm_row_count;
  uint64_t canonical_rms_norm_hidden_size;
  float canonical_rms_norm_epsilon;
  RileyCudaDeviceBuffer* bf16_argmax_logits;
  uint64_t bf16_argmax_row_count;
  uint64_t bf16_argmax_vocabulary_size;
  RileyCudaDeviceBuffer* bf16_row_gather_input;
  RileyCudaDeviceBuffer* bf16_row_gather_indices;
  uint64_t bf16_row_gather_input_row_count;
  uint64_t bf16_row_gather_output_row_count;
  uint64_t bf16_row_gather_column_count;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_input;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_indices;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_gathered_logits;
  uint64_t bf16_row_gather_argmax_input_row_count;
  uint64_t bf16_row_gather_argmax_output_row_count;
  uint64_t bf16_row_gather_argmax_vocabulary_size;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_input;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_indices;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_gathered_logits;
  RileyCudaPinnedHostBuffer* bf16_row_gather_argmax_d2h_pinned_results;
  uint64_t bf16_row_gather_argmax_d2h_input_row_count;
  uint64_t bf16_row_gather_argmax_d2h_output_row_count;
  uint64_t bf16_row_gather_argmax_d2h_vocabulary_size;
  uint64_t bf16_row_gather_argmax_d2h_result_byte_len;
  RileyCudaDeviceBuffer* indexed_rope_bf16_input;
  RileyCudaDeviceBuffer* indexed_rope_bf16_cos;
  RileyCudaDeviceBuffer* indexed_rope_bf16_sin;
  RileyCudaDeviceBuffer* indexed_rope_bf16_positions;
  uint64_t indexed_rope_bf16_active_row_count;
  uint64_t indexed_rope_bf16_head_count;
  uint64_t indexed_rope_bf16_head_size;
  uint64_t indexed_rope_bf16_rotary_dimension;
  uint64_t indexed_rope_bf16_table_position_count;
  RileyCudaRaggedPagedKvCacheWriteBf16State ragged_paged_kv_write_bf16;
  uint64_t capture_id;
  cudaGraph_t graph;
  bool owns_capture_leases;
};

// Instantiation consumes RileyCudaGraph and transfers its immutable graph,
// exact stream, fixed buffer, and capture leases into this exec. C05-5 admits
// at most one in-flight replay; normal stream work remains blocked by the
// permanent graph lease rather than a second active-use count.
struct RileyCudaGraphExec {
  RileyCudaGraphExec(RileyCudaContext* owning_context,
                     RileyCudaStream* captured_stream,
                     RileyCudaDeviceBuffer* captured_fill_buffer,
                     uint64_t capture_identifier,
                     uint64_t executable_identifier,
                     RileyCudaGraphCaptureOperation captured_operation,
                     RileyCudaPinnedHostBuffer* captured_h2d_source = nullptr,
                     uint64_t captured_h2d_byte_len = 0,
                     RileyCudaDeviceBuffer* captured_silu_input = nullptr,
                     uint64_t captured_silu_element_count = 0,
                     RileyCudaDeviceBuffer* captured_gated_multiply_activated_gate = nullptr,
                     RileyCudaDeviceBuffer* captured_gated_multiply_up = nullptr,
                     uint64_t captured_gated_multiply_element_count = 0,
                     RileyCudaDeviceBuffer* captured_residual_add_left = nullptr,
                     RileyCudaDeviceBuffer* captured_residual_add_right = nullptr,
                     uint64_t captured_residual_add_element_count = 0,
                     RileyCudaDeviceBuffer* captured_canonical_rms_norm_input = nullptr,
                     RileyCudaDeviceBuffer* captured_canonical_rms_norm_weight = nullptr,
                     uint64_t captured_canonical_rms_norm_row_count = 0,
                     uint64_t captured_canonical_rms_norm_hidden_size = 0,
                     float captured_canonical_rms_norm_epsilon = 0.0F,
                     RileyCudaDeviceBuffer* captured_bf16_argmax_logits = nullptr,
                     uint64_t captured_bf16_argmax_row_count = 0,
                     uint64_t captured_bf16_argmax_vocabulary_size = 0,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_input = nullptr,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_indices = nullptr,
                     uint64_t captured_bf16_row_gather_input_row_count = 0,
                     uint64_t captured_bf16_row_gather_output_row_count = 0,
                     uint64_t captured_bf16_row_gather_column_count = 0,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_input = nullptr,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_indices = nullptr,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_gathered_logits = nullptr,
                     uint64_t captured_bf16_row_gather_argmax_input_row_count = 0,
                     uint64_t captured_bf16_row_gather_argmax_output_row_count = 0,
                     uint64_t captured_bf16_row_gather_argmax_vocabulary_size = 0,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_d2h_input = nullptr,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_d2h_indices = nullptr,
                     RileyCudaDeviceBuffer* captured_bf16_row_gather_argmax_d2h_gathered_logits = nullptr,
                     RileyCudaPinnedHostBuffer* captured_bf16_row_gather_argmax_d2h_pinned_results = nullptr,
                     uint64_t captured_bf16_row_gather_argmax_d2h_input_row_count = 0,
                     uint64_t captured_bf16_row_gather_argmax_d2h_output_row_count = 0,
                     uint64_t captured_bf16_row_gather_argmax_d2h_vocabulary_size = 0,
                     uint64_t captured_bf16_row_gather_argmax_d2h_result_byte_len = 0,
                     RileyCudaDeviceBuffer* captured_indexed_rope_bf16_input = nullptr,
                     RileyCudaDeviceBuffer* captured_indexed_rope_bf16_cos = nullptr,
                     RileyCudaDeviceBuffer* captured_indexed_rope_bf16_sin = nullptr,
                     RileyCudaDeviceBuffer* captured_indexed_rope_bf16_positions = nullptr,
                     uint64_t captured_indexed_rope_bf16_active_row_count = 0,
                     uint64_t captured_indexed_rope_bf16_head_count = 0,
                     uint64_t captured_indexed_rope_bf16_head_size = 0,
                     uint64_t captured_indexed_rope_bf16_rotary_dimension = 0,
                     uint64_t captured_indexed_rope_bf16_table_position_count = 0) noexcept
      : owner(owning_context),
        stream(captured_stream),
        fill_buffer(captured_fill_buffer),
        operation(captured_operation),
        h2d_source(captured_h2d_source),
        h2d_byte_len(captured_h2d_byte_len),
        silu_input(captured_silu_input),
        silu_element_count(captured_silu_element_count),
        gated_multiply_activated_gate(captured_gated_multiply_activated_gate),
        gated_multiply_up(captured_gated_multiply_up),
        gated_multiply_element_count(captured_gated_multiply_element_count),
        residual_add_left(captured_residual_add_left),
        residual_add_right(captured_residual_add_right),
        residual_add_element_count(captured_residual_add_element_count),
        canonical_rms_norm_input(captured_canonical_rms_norm_input),
        canonical_rms_norm_weight(captured_canonical_rms_norm_weight),
        canonical_rms_norm_row_count(captured_canonical_rms_norm_row_count),
        canonical_rms_norm_hidden_size(captured_canonical_rms_norm_hidden_size),
        canonical_rms_norm_epsilon(captured_canonical_rms_norm_epsilon),
        bf16_argmax_logits(captured_bf16_argmax_logits),
        bf16_argmax_row_count(captured_bf16_argmax_row_count),
        bf16_argmax_vocabulary_size(captured_bf16_argmax_vocabulary_size),
        bf16_row_gather_input(captured_bf16_row_gather_input),
        bf16_row_gather_indices(captured_bf16_row_gather_indices),
        bf16_row_gather_input_row_count(
            captured_bf16_row_gather_input_row_count),
        bf16_row_gather_output_row_count(
            captured_bf16_row_gather_output_row_count),
        bf16_row_gather_column_count(captured_bf16_row_gather_column_count),
        bf16_row_gather_argmax_input(captured_bf16_row_gather_argmax_input),
        bf16_row_gather_argmax_indices(captured_bf16_row_gather_argmax_indices),
        bf16_row_gather_argmax_gathered_logits(
            captured_bf16_row_gather_argmax_gathered_logits),
        bf16_row_gather_argmax_input_row_count(
            captured_bf16_row_gather_argmax_input_row_count),
        bf16_row_gather_argmax_output_row_count(
            captured_bf16_row_gather_argmax_output_row_count),
        bf16_row_gather_argmax_vocabulary_size(
            captured_bf16_row_gather_argmax_vocabulary_size),
        bf16_row_gather_argmax_d2h_input(
            captured_bf16_row_gather_argmax_d2h_input),
        bf16_row_gather_argmax_d2h_indices(
            captured_bf16_row_gather_argmax_d2h_indices),
        bf16_row_gather_argmax_d2h_gathered_logits(
            captured_bf16_row_gather_argmax_d2h_gathered_logits),
        bf16_row_gather_argmax_d2h_pinned_results(
            captured_bf16_row_gather_argmax_d2h_pinned_results),
        bf16_row_gather_argmax_d2h_input_row_count(
            captured_bf16_row_gather_argmax_d2h_input_row_count),
        bf16_row_gather_argmax_d2h_output_row_count(
            captured_bf16_row_gather_argmax_d2h_output_row_count),
        bf16_row_gather_argmax_d2h_vocabulary_size(
            captured_bf16_row_gather_argmax_d2h_vocabulary_size),
        bf16_row_gather_argmax_d2h_result_byte_len(
            captured_bf16_row_gather_argmax_d2h_result_byte_len),
        indexed_rope_bf16_input(captured_indexed_rope_bf16_input),
        indexed_rope_bf16_cos(captured_indexed_rope_bf16_cos),
        indexed_rope_bf16_sin(captured_indexed_rope_bf16_sin),
        indexed_rope_bf16_positions(captured_indexed_rope_bf16_positions),
        indexed_rope_bf16_active_row_count(
            captured_indexed_rope_bf16_active_row_count),
        indexed_rope_bf16_head_count(captured_indexed_rope_bf16_head_count),
        indexed_rope_bf16_head_size(captured_indexed_rope_bf16_head_size),
        indexed_rope_bf16_rotary_dimension(
            captured_indexed_rope_bf16_rotary_dimension),
        indexed_rope_bf16_table_position_count(
            captured_indexed_rope_bf16_table_position_count),
        capture_id(capture_identifier),
        exec_id(executable_identifier),
        graph(nullptr),
        exec(nullptr),
        owns_capture_leases(false),
        launch_in_flight(false),
        bf16_row_gather_argmax_d2h_completion_visible(false),
        h2d_input_staged(false),
        poisoned(false) {}

  RileyCudaContext* owner;
  RileyCudaStream* stream;
  RileyCudaDeviceBuffer* fill_buffer;
  RileyCudaGraphCaptureOperation operation;
  RileyCudaPinnedHostBuffer* h2d_source;
  uint64_t h2d_byte_len;
  RileyCudaDeviceBuffer* silu_input;
  uint64_t silu_element_count;
  RileyCudaDeviceBuffer* gated_multiply_activated_gate;
  RileyCudaDeviceBuffer* gated_multiply_up;
  uint64_t gated_multiply_element_count;
  RileyCudaDeviceBuffer* residual_add_left;
  RileyCudaDeviceBuffer* residual_add_right;
  uint64_t residual_add_element_count;
  RileyCudaDeviceBuffer* canonical_rms_norm_input;
  RileyCudaDeviceBuffer* canonical_rms_norm_weight;
  uint64_t canonical_rms_norm_row_count;
  uint64_t canonical_rms_norm_hidden_size;
  float canonical_rms_norm_epsilon;
  RileyCudaDeviceBuffer* bf16_argmax_logits;
  uint64_t bf16_argmax_row_count;
  uint64_t bf16_argmax_vocabulary_size;
  RileyCudaDeviceBuffer* bf16_row_gather_input;
  RileyCudaDeviceBuffer* bf16_row_gather_indices;
  uint64_t bf16_row_gather_input_row_count;
  uint64_t bf16_row_gather_output_row_count;
  uint64_t bf16_row_gather_column_count;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_input;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_indices;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_gathered_logits;
  uint64_t bf16_row_gather_argmax_input_row_count;
  uint64_t bf16_row_gather_argmax_output_row_count;
  uint64_t bf16_row_gather_argmax_vocabulary_size;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_input;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_indices;
  RileyCudaDeviceBuffer* bf16_row_gather_argmax_d2h_gathered_logits;
  RileyCudaPinnedHostBuffer* bf16_row_gather_argmax_d2h_pinned_results;
  uint64_t bf16_row_gather_argmax_d2h_input_row_count;
  uint64_t bf16_row_gather_argmax_d2h_output_row_count;
  uint64_t bf16_row_gather_argmax_d2h_vocabulary_size;
  uint64_t bf16_row_gather_argmax_d2h_result_byte_len;
  RileyCudaDeviceBuffer* indexed_rope_bf16_input;
  RileyCudaDeviceBuffer* indexed_rope_bf16_cos;
  RileyCudaDeviceBuffer* indexed_rope_bf16_sin;
  RileyCudaDeviceBuffer* indexed_rope_bf16_positions;
  uint64_t indexed_rope_bf16_active_row_count;
  uint64_t indexed_rope_bf16_head_count;
  uint64_t indexed_rope_bf16_head_size;
  uint64_t indexed_rope_bf16_rotary_dimension;
  uint64_t indexed_rope_bf16_table_position_count;
  RileyCudaRaggedPagedKvCacheWriteBf16State ragged_paged_kv_write_bf16;
  uint64_t capture_id;
  uint64_t exec_id;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  bool owns_capture_leases;
  bool launch_in_flight;
  // C05-16 exposes raw D2H result bytes only through its exact completion
  // receipt after a successful launch completion. A later launch clears this
  // fact before CUDA submission; permanent pinned ownership remains held.
  bool bf16_row_gather_argmax_d2h_completion_visible;
  // H2D execs require a fresh exact payload stage before every replay. A
  // launch attempt consumes this flag even when CUDA reports a deferred error.
  bool h2d_input_staged;
  bool poisoned;
};

// One launch owner exists only between cudaGraphLaunch and a single explicit
// completion attempt. An ambiguous completion intentionally retains this
// owner, the graph exec, and all graph leases fail-closed.
struct RileyCudaGraphLaunch {
  RileyCudaGraphLaunch(RileyCudaGraphExec* owning_exec,
                       RileyCudaStream* launch_stream) noexcept
      : exec(owning_exec), stream(launch_stream) {}

  RileyCudaGraphExec* exec;
  RileyCudaStream* stream;
};

struct RileyCudaPinnedHostBuffer {
  RileyCudaPinnedHostBuffer(RileyCudaContext* owning_context,
                                void* allocation,
                                uint64_t allocation_bytes) noexcept
      : owner(owning_context),
        host_data(allocation),
        byte_len(allocation_bytes),
        active_uses(0),
        deferred_close() {}

  RileyCudaContext* owner;
  void* host_data;
  uint64_t byte_len;
  std::atomic<uint32_t> active_uses;
  RileyCudaDeferredCloseNode deferred_close;
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
        completed(false),
        capture_admission_held(true) {
    deferred_error.struct_size = sizeof(deferred_error);
  }

  RileyCudaContext* owner;
  RileyCudaStream* stream;
  RileyCudaDeviceBuffer* device;
  RileyCudaPinnedHostBuffer* host;
  RileyCudaStatus deferred_status;
  RileyCudaErrorInfo deferred_error;
  bool completed;
  bool capture_admission_held;
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

class CaptureDomainRegistryGuard final {
 public:
  CaptureDomainRegistryGuard() noexcept : lock_(capture_domain_registry_lock()) {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  CaptureDomainRegistryGuard(const CaptureDomainRegistryGuard&) = delete;
  CaptureDomainRegistryGuard& operator=(const CaptureDomainRegistryGuard&) = delete;
  ~CaptureDomainRegistryGuard() noexcept {
    lock_.clear(std::memory_order_release);
  }

 private:
  static std::atomic_flag& capture_domain_registry_lock() noexcept {
    static std::atomic_flag lock = ATOMIC_FLAG_INIT;
    return lock;
  }

  std::atomic_flag& lock_;
};

inline RileyCudaCaptureDomain*& capture_domain_registry_head() noexcept {
  static RileyCudaCaptureDomain* head = nullptr;
  return head;
}

inline RileyCudaCaptureDomain* capture_domain_for_device(
    CUdevice device) noexcept {
  const CaptureDomainRegistryGuard guard;
  for (RileyCudaCaptureDomain* domain = capture_domain_registry_head();
       domain != nullptr; domain = domain->next) {
    if (domain->device == device) {
      return domain;
    }
  }

  void* storage = std::calloc(1, sizeof(RileyCudaCaptureDomain));
  if (storage == nullptr) {
    return nullptr;
  }
  auto* domain = new (storage) RileyCudaCaptureDomain(device);
  domain->next = capture_domain_registry_head();
  capture_domain_registry_head() = domain;
  return domain;
}

class CaptureDomainAdmissionGuard final {
 public:
  explicit CaptureDomainAdmissionGuard(RileyCudaCaptureDomain* domain) noexcept
      : lock_(domain->admission_lock) {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  CaptureDomainAdmissionGuard(const CaptureDomainAdmissionGuard&) = delete;
  CaptureDomainAdmissionGuard& operator=(const CaptureDomainAdmissionGuard&) =
      delete;
  ~CaptureDomainAdmissionGuard() noexcept {
    lock_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag& lock_;
};

// ThreadLocal CUDA capture is scoped to a host thread, but a pending safe-Rust
// copy/fill token can be moved to that thread and later need CUDA to settle.
// Make capture-vs-pending-token admission process-global so a token on device
// A cannot be stranded by a capture on device B. This gate intentionally does
// not serialize independent captures: it only excludes pending lifecycle work.
inline std::atomic_flag& capture_lifecycle_admission_lock() noexcept {
  static std::atomic_flag lock = ATOMIC_FLAG_INIT;
  return lock;
}

inline std::atomic<uint32_t>& capture_lifecycle_active_captures() noexcept {
  static std::atomic<uint32_t> active_captures{0};
  return active_captures;
}

inline std::atomic<uint32_t>& capture_lifecycle_pending_lifecycles() noexcept {
  static std::atomic<uint32_t> pending_lifecycles{0};
  return pending_lifecycles;
}

class CaptureLifecycleAdmissionGuard final {
 public:
  CaptureLifecycleAdmissionGuard() noexcept
      : lock_(capture_lifecycle_admission_lock()) {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  CaptureLifecycleAdmissionGuard(const CaptureLifecycleAdmissionGuard&) =
      delete;
  CaptureLifecycleAdmissionGuard& operator=(
      const CaptureLifecycleAdmissionGuard&) = delete;
  ~CaptureLifecycleAdmissionGuard() noexcept {
    lock_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag& lock_;
};

static_assert(sizeof(RileyCudaErrorInfo) == 272,
              "RileyCudaErrorInfo ABI size changed");
static_assert(offsetof(RileyCudaErrorInfo, struct_size) == 0,
              "RileyCudaErrorInfo struct-size offset changed");
static_assert(offsetof(RileyCudaErrorInfo, native_code) == 4,
              "RileyCudaErrorInfo native-code offset changed");
static_assert(offsetof(RileyCudaErrorInfo, domain) == 8,
              "RileyCudaErrorInfo domain offset changed");
static_assert(offsetof(RileyCudaErrorInfo, stage) == 12,
              "RileyCudaErrorInfo stage offset changed");
static_assert(offsetof(RileyCudaErrorInfo, message) == 16,
              "RileyCudaErrorInfo ABI layout changed");
static_assert(sizeof(RileyCudaGraphCaptureMode) == 4,
              "RileyCudaGraphCaptureMode ABI width changed");
static_assert(RILEY_CUDA_GRAPH_CAPTURE_MODE_INVALID == 0 &&
                  RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL == 1,
              "RileyCudaGraphCaptureMode ABI discriminants changed");
static_assert(sizeof(RileyCudaGraphCaptureCapability) == 4,
              "RileyCudaGraphCaptureCapability ABI width changed");
static_assert(RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN == 0 &&
                  RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNSUPPORTED == 1 &&
                  RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_SUPPORTED == 2,
              "RileyCudaGraphCaptureCapability ABI discriminants changed");
static_assert(sizeof(RileyCudaGraphStage) == 4,
              "RileyCudaGraphStage ABI width changed");
static_assert(RILEY_CUDA_GRAPH_STAGE_NONE == 0 &&
                  RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN == 1 &&
                  RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE == 2 &&
                  RILEY_CUDA_GRAPH_STAGE_CAPTURE_END == 3 &&
                  RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT == 4 &&
                  RILEY_CUDA_GRAPH_STAGE_INSTANTIATE == 5 &&
                   RILEY_CUDA_GRAPH_STAGE_UPDATE == 6 &&
                   RILEY_CUDA_GRAPH_STAGE_LAUNCH == 7 &&
                   RILEY_CUDA_GRAPH_STAGE_COMPLETION == 8 &&
                   RILEY_CUDA_GRAPH_STAGE_CLOSE == 9 &&
                   RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE == 10,
              "RileyCudaGraphStage ABI discriminants changed");
static_assert(sizeof(RileyCudaGraphErrorInfo) == 56,
              "RileyCudaGraphErrorInfo ABI size changed");
static_assert(alignof(RileyCudaGraphErrorInfo) == 8,
              "RileyCudaGraphErrorInfo ABI alignment changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, struct_size) == 0,
              "RileyCudaGraphErrorInfo struct-size offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, graph_stage) == 4,
              "RileyCudaGraphErrorInfo stage offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, capture_id) == 8,
              "RileyCudaGraphErrorInfo capture-id offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, exec_id) == 16,
              "RileyCudaGraphErrorInfo exec-id offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, submission_started) == 24,
              "RileyCudaGraphErrorInfo submission flag offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, completion_known) == 25,
              "RileyCudaGraphErrorInfo completion flag offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, resource_release_known) == 26,
              "RileyCudaGraphErrorInfo release flag offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, poisoned) == 27,
              "RileyCudaGraphErrorInfo poisoned flag offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, reserved0) == 28,
              "RileyCudaGraphErrorInfo reserved0 offset changed");
static_assert(offsetof(RileyCudaGraphErrorInfo, reserved) == 32,
              "RileyCudaGraphErrorInfo reserved tail offset changed");
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

// CUDA's ThreadLocal capture mode forbids potentially unsafe CUDA calls made
// by the same host thread. Keep the exact native owner in TLS so every normal
// context entry can reject before touching the CUDA driver. The owner remains
// published until abort has proved end/destroy/context restoration and every
// local lease release; uncertainty intentionally strands this gate.
inline RileyCudaGraphCapture*& thread_graph_capture_owner() noexcept {
  static thread_local RileyCudaGraphCapture* owner = nullptr;
  return owner;
}

inline bool thread_has_active_graph_capture() noexcept {
  return thread_graph_capture_owner() != nullptr;
}

inline bool try_publish_thread_graph_capture(
    RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || thread_has_active_graph_capture()) {
    return false;
  }
  thread_graph_capture_owner() = capture;
  return true;
}

inline bool thread_graph_capture_is_owner(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr && thread_graph_capture_owner() == capture;
}

inline bool clear_thread_graph_capture_owner(
    const RileyCudaGraphCapture* capture) noexcept {
  if (!thread_graph_capture_is_owner(capture)) {
    return false;
  }
  thread_graph_capture_owner() = nullptr;
  return true;
}

// Resource-specific constructors configure an embedded node before returning
// the resource to safe Rust. Existing raw `*_close` entry points deliberately
// do not call this path: their retry-on-InvalidState ABI remains unchanged.
inline bool initialize_capture_deferred_close_node(
    RileyCudaDeferredCloseNode* node, RileyCudaContext* owner, void* payload,
    RileyCudaDeferredCloseCallback callback) noexcept {
  if (node == nullptr || owner == nullptr || payload == nullptr ||
      callback == nullptr || node->queued) {
    return false;
  }
  node->next = nullptr;
  node->owner = owner;
  node->payload = payload;
  node->callback = callback;
  return true;
}

enum class CaptureDeferredCloseEnqueueResult : uint8_t {
  kNotCapturing,
  kQueued,
  kInvalidNode,
};

// This is a capture-thread-only, allocation-free handoff. A resource-specific
// additive `*_defer_to_active_capture` entry point must consume its raw handle
// only after this returns kQueued. The callback receives this exact owner and
// can therefore use CurrentContext with a foreign resource context safely.
inline CaptureDeferredCloseEnqueueResult enqueue_capture_deferred_close(
    RileyCudaDeferredCloseNode* node) noexcept {
  RileyCudaGraphCapture* const capture = thread_graph_capture_owner();
  if (capture == nullptr) {
    return CaptureDeferredCloseEnqueueResult::kNotCapturing;
  }
  if (!capture->capture_started || capture->owner == nullptr ||
      capture->stream == nullptr || node == nullptr || node->owner == nullptr ||
      node->payload == nullptr || node->callback == nullptr || node->queued ||
      node->next != nullptr ||
      ((capture->deferred_close_head == nullptr) !=
       (capture->deferred_close_tail == nullptr))) {
    return CaptureDeferredCloseEnqueueResult::kInvalidNode;
  }

  node->queued = true;
  if (capture->deferred_close_tail == nullptr) {
    capture->deferred_close_head = node;
  } else {
    capture->deferred_close_tail->next = node;
  }
  capture->deferred_close_tail = node;
  return CaptureDeferredCloseEnqueueResult::kQueued;
}

// A consumed callback may destroy the resource and its embedded node. Save the
// next link first and never touch that node after invocation. A consumed error
// pops the current node but retains the unvisited FIFO suffix. A non-consumed
// error retains the current node plus that suffix. That fail-closed owner is
// intentionally never retried because CUDA close side effects may be ambiguous.
inline RileyCudaStatus drain_capture_deferred_closes(
    RileyCudaGraphCapture* capture, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kDrainOperation = "drain deferred CUDA capture closes";
  if (capture == nullptr || !thread_graph_capture_is_owner(capture)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, kDrainOperation,
        "the supplied CUDA Graph capture owner is not active on this host thread");
  }
  if ((capture->deferred_close_head == nullptr) !=
      (capture->deferred_close_tail == nullptr)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kDrainOperation,
                          "deferred-close FIFO head/tail state is corrupt");
  }

  while (capture->deferred_close_head != nullptr) {
    RileyCudaDeferredCloseNode* const node = capture->deferred_close_head;
    RileyCudaDeferredCloseNode* const next = node->next;
    if (!node->queued || node->owner == nullptr || node->payload == nullptr ||
        node->callback == nullptr ||
        (node == capture->deferred_close_tail && next != nullptr) ||
        (node != capture->deferred_close_tail && next == nullptr)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kDrainOperation,
                            "deferred-close FIFO node state is corrupt");
    }

    const RileyCudaDeferredCloseResult result =
        node->callback(node, capture, error);
    if (result.consumed) {
      capture->deferred_close_head = next;
      if (next == nullptr) {
        capture->deferred_close_tail = nullptr;
      }
      if (result.status != RILEY_CUDA_STATUS_SUCCESS) {
        return result.status;
      }
      continue;
    }
    if (result.status == RILEY_CUDA_STATUS_SUCCESS) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kDrainOperation,
                            "deferred-close callback succeeded without consuming its node");
    }
    return result.status;
  }
  return RILEY_CUDA_STATUS_SUCCESS;
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

  RileyCudaStatus enter(
      RileyCudaErrorInfo* error, uint32_t stage, const char* operation,
      const RileyCudaGraphCapture* capture_owner = nullptr) noexcept {
    if (context_ == nullptr) {
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                              stage, operation, "context is null");
    }
    if (context_->restoration_failed.load(std::memory_order_acquire)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE, stage, operation,
          "a prior CUDA context-stack restoration failed");
    }
    if (capture_owner != nullptr) {
      if (!thread_graph_capture_is_owner(capture_owner)) {
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_STATE, stage, operation,
            "the supplied CUDA Graph capture owner is not active on this host thread");
      }
    } else if (thread_has_active_graph_capture()) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE, stage, operation,
          "this host thread has an active thread-local CUDA Graph capture");
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

inline bool try_increment_capture_domain_counter(
    std::atomic<uint32_t>& counter) noexcept {
  uint32_t current = counter.load(std::memory_order_relaxed);
  while (current != std::numeric_limits<uint32_t>::max()) {
    if (counter.compare_exchange_weak(current, current + 1,
                                      std::memory_order_acq_rel,
                                      std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

inline bool release_capture_domain_counter(
    std::atomic<uint32_t>& counter) noexcept {
  uint32_t current = counter.load(std::memory_order_relaxed);
  while (current != 0) {
    if (counter.compare_exchange_weak(current, current - 1,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

// Acquire the process-global lock before the per-domain lock everywhere. The
// global token gate and the primary-context control gate are then both
// linearized with their corresponding counter mark before any CUDA entry.
inline bool try_begin_capture_domain(
    RileyCudaCaptureDomain* domain) noexcept {
  if (domain == nullptr) {
    return false;
  }
  const CaptureLifecycleAdmissionGuard lifecycle_admission;
  const CaptureDomainAdmissionGuard admission(domain);
  std::atomic<uint32_t>& global_active = capture_lifecycle_active_captures();
  std::atomic<uint32_t>& global_pending =
      capture_lifecycle_pending_lifecycles();
  if (global_pending.load(std::memory_order_acquire) != 0 ||
      domain->broad_control_uses.load(std::memory_order_acquire) != 0 ||
      domain->pending_smoke_fills.load(std::memory_order_acquire) != 0 ||
      domain->pending_copies.load(std::memory_order_acquire) != 0) {
    return false;
  }
  if (!try_increment_capture_domain_counter(global_active)) {
    return false;
  }
  if (try_increment_capture_domain_counter(domain->active_captures)) {
    return true;
  }
  (void)release_capture_domain_counter(global_active);
  return false;
}

inline bool release_capture_domain_capture(
    RileyCudaCaptureDomain* domain) noexcept {
  if (domain == nullptr) {
    return false;
  }
  const CaptureLifecycleAdmissionGuard lifecycle_admission;
  const CaptureDomainAdmissionGuard admission(domain);
  std::atomic<uint32_t>& global_active = capture_lifecycle_active_captures();
  if (global_active.load(std::memory_order_acquire) == 0 ||
      domain->active_captures.load(std::memory_order_acquire) == 0) {
    return false;
  }
  return release_capture_domain_counter(domain->active_captures) &&
         release_capture_domain_counter(global_active);
}

// Pending lifecycle work has to survive a later safe-Rust Drop, which may
// need CUDA synchronization. Pair one global token with the per-domain token
// before the first CUDA call. The global gate is deliberately broader than
// primary-context ownership: ThreadLocal capture blocks ordinary CUDA entries
// on its host thread even when the pending token belongs to another device.
inline bool try_begin_capture_domain_pending_lifecycle(
    RileyCudaCaptureDomain* domain,
    std::atomic<uint32_t>& pending_counter) noexcept {
  if (domain == nullptr) {
    return false;
  }
  const CaptureLifecycleAdmissionGuard lifecycle_admission;
  const CaptureDomainAdmissionGuard admission(domain);
  std::atomic<uint32_t>& global_active = capture_lifecycle_active_captures();
  std::atomic<uint32_t>& global_pending =
      capture_lifecycle_pending_lifecycles();
  if (global_active.load(std::memory_order_acquire) != 0 ||
      domain->active_captures.load(std::memory_order_acquire) != 0 ||
      domain->broad_control_uses.load(std::memory_order_acquire) != 0) {
    return false;
  }
  if (!try_increment_capture_domain_counter(global_pending)) {
    return false;
  }
  if (try_increment_capture_domain_counter(pending_counter)) {
    return true;
  }
  (void)release_capture_domain_counter(global_pending);
  return false;
}

inline bool release_capture_domain_pending_lifecycle(
    RileyCudaCaptureDomain* domain,
    std::atomic<uint32_t>& pending_counter) noexcept {
  if (domain == nullptr) {
    return false;
  }
  const CaptureLifecycleAdmissionGuard lifecycle_admission;
  const CaptureDomainAdmissionGuard admission(domain);
  std::atomic<uint32_t>& global_pending =
      capture_lifecycle_pending_lifecycles();
  if (global_pending.load(std::memory_order_acquire) == 0 ||
      pending_counter.load(std::memory_order_acquire) == 0) {
    return false;
  }
  return release_capture_domain_counter(pending_counter) &&
         release_capture_domain_counter(global_pending);
}

// A launched smoke fill can require synchronization and allocation release on
// Drop. Reserve it before enqueue, and keep that reservation until the native
// buffer is consumed, so capture and this recoverable lifecycle never overlap.
inline bool try_begin_capture_domain_smoke_fill(
    RileyCudaCaptureDomain* domain) noexcept {
  if (domain == nullptr) {
    return false;
  }
  return try_begin_capture_domain_pending_lifecycle(
      domain, domain->pending_smoke_fills);
}

inline bool release_capture_domain_smoke_fill(
    RileyCudaCaptureDomain* domain) noexcept {
  if (domain == nullptr) {
    return false;
  }
  return release_capture_domain_pending_lifecycle(
      domain, domain->pending_smoke_fills);
}

// Pending copies carry a Rust-side borrow token whose Drop must synchronize
// before releasing it. Reserve before cudaMemcpyAsync and retain until the
// native copy is consumed, so a capture on any device cannot make that Drop
// unrecoverable.
inline bool try_begin_capture_domain_pending_copy(
    RileyCudaCaptureDomain* domain) noexcept {
  if (domain == nullptr) {
    return false;
  }
  return try_begin_capture_domain_pending_lifecycle(
      domain, domain->pending_copies);
}

inline bool release_capture_domain_pending_copy(
    RileyCudaCaptureDomain* domain) noexcept {
  if (domain == nullptr) {
    return false;
  }
  return release_capture_domain_pending_lifecycle(
      domain, domain->pending_copies);
}

class CaptureDomainControlLease final {
 public:
  explicit CaptureDomainControlLease(
      RileyCudaCaptureDomain* domain,
      const RileyCudaGraphCapture* terminated_capture_owner = nullptr) noexcept
      : domain_(nullptr) {
    if (domain == nullptr) {
      return;
    }
    const CaptureDomainAdmissionGuard admission(domain);
    if (domain->active_captures.load(std::memory_order_acquire) != 0) {
      // The only permitted exception is drain-time release of a childless
      // foreign context after this exact capture is physically terminated. It
      // remains bounded to one capture in the same primary-context domain;
      // ordinary controls and concurrent capture still fail closed.
      if (terminated_capture_owner == nullptr ||
          !thread_graph_capture_is_owner(terminated_capture_owner) ||
          !terminated_capture_owner->capture_terminated ||
          terminated_capture_owner->capture_domain != domain ||
          domain->active_captures.load(std::memory_order_acquire) != 1) {
        return;
      }
    }
    if (try_increment_capture_domain_counter(domain->broad_control_uses)) {
      domain_ = domain;
    }
  }
  CaptureDomainControlLease(const CaptureDomainControlLease&) = delete;
  CaptureDomainControlLease& operator=(const CaptureDomainControlLease&) =
      delete;
  ~CaptureDomainControlLease() noexcept { (void)release(); }

  bool active() const noexcept { return domain_ != nullptr; }

  bool release() noexcept {
    if (domain_ == nullptr) {
      return true;
    }
    RileyCudaCaptureDomain* const domain = domain_;
    domain_ = nullptr;
    return release_capture_domain_counter(domain->broad_control_uses);
  }

 private:
  RileyCudaCaptureDomain* domain_;
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
inline const void* native_thread_token() noexcept {
  static thread_local const uint8_t token = 0;
  return &token;
}

// A command batch has a stream-local owner, but it also carries pending work
// that its owner thread must finish with CUDA synchronization. Capture cannot
// begin on any stream of that same host thread until every such batch closes.
// A count preserves the existing ability to batch multiple streams while
// avoiding a dangling TLS pointer if a stream is later destroyed.
inline uint32_t& thread_command_batch_count() noexcept {
  static thread_local uint32_t count = 0;
  return count;
}

inline bool thread_has_active_command_batch() noexcept {
  return thread_command_batch_count() != 0;
}

inline bool try_publish_thread_command_batch() noexcept {
  uint32_t& count = thread_command_batch_count();
  if (count == std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  ++count;
  return true;
}

inline bool release_thread_command_batch() noexcept {
  uint32_t& count = thread_command_batch_count();
  if (count == 0) {
    return false;
  }
  --count;
  return true;
}

inline const void* command_batch_thread_token() noexcept {
  return native_thread_token();
}

inline uint64_t next_graph_capture_id() noexcept {
  static std::atomic<uint64_t> next{1};
  // An all-zero externally visible identity is reserved for "no capture".
  // Once the finite ID space wraps, leave the counter at zero and fail all
  // later allocations rather than reuse an observable identity.
  uint64_t identifier = next.load(std::memory_order_relaxed);
  while (identifier != 0) {
    if (next.compare_exchange_weak(identifier, identifier + 1,
                                   std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return identifier;
    }
  }
  return 0;
}

inline uint64_t next_graph_exec_id() noexcept {
  static std::atomic<uint64_t> next{1};
  // Exec IDs are independently observable from capture IDs. Preserve zero as
  // the ABI's no-exec sentinel and fail closed instead of wrapping/reusing an
  // externally visible identifier.
  uint64_t identifier = next.load(std::memory_order_relaxed);
  while (identifier != 0) {
    if (next.compare_exchange_weak(identifier, identifier + 1,
                                   std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return identifier;
    }
  }
  return 0;
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
