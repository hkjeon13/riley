#include <vector>
#include "ffi_internal.hpp"
#include "prefill_shape_packet.hpp"
#include <cublasLt.h>
#include <new>
#include <cstdlib>
#include <cstring>
#include <cmath>

// The ledger remains the single authority for all retained parents. The
// optional transfer graph is a lifecycle probe, not a model decode graph.
struct RileyCudaGraphResources {
  // Cold additional decode entries share the parent's single resource ledger.
  struct CatalogEntry {
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    RileyCudaPinnedHostBuffer* staging=nullptr;
    uint64_t transfer=0;
    uint32_t bucket=0,physical=0,full=0;
  };
  CatalogEntry catalog[6]{};
  uint32_t last_catalog=0;
  bool catalog_sealed=false;
  enum class Kind { Counter, Plan };
  struct Entry {
    Kind kind;
    void* identity;
    std::atomic<uint32_t>* counter;
  };
  static constexpr size_t kCapacity = 1024;
  RileyCudaContext* owner = nullptr;
  const void* thread = nullptr;
  bool child_held = false;
  size_t count = 0;
  Entry entries[kCapacity]{};
  RileyCudaStream* stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  // The fixed-P128 arithmetic profile has two stage DAGs under this one ledger.
  // They share every parent and are only replayed sequentially on this stream.
  cudaGraph_t prefill_graph = nullptr;
  cudaGraphExec_t prefill_exec = nullptr;
  RileyCudaPinnedHostBuffer* input = nullptr;
  RileyCudaPinnedHostBuffer* output = nullptr;
  uint64_t transfer_bytes = 0;
  uint64_t output_byte_offset = 0;
  uint64_t rope_table_positions = 0;
  uint64_t rope_payload_position_offset = 0;
  bool kv_write_bound = false;
  uint64_t kv_logical_block = 0;
  uint64_t kv_valid_tokens = 0;
  bool bound_attention = false;
  uint32_t attention_physical_block = 0;
  uint64_t decode_capacity = 0, decode_physical = 0, decode_vocab = 0;
  uint32_t multi_bucket=0, multi_physical=0, multi_full=0;
  bool v3_shared=false;
  uint32_t v3_prefill_capacity=0,v3_prefill_physical=0,v3_prefill_context=0;
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
  uint32_t test_replay_fault = 0;
#endif
  bool terminal = false;
  bool completion_visible = false;
  bool vllm_smol_p128 = false;
  bool prefill128 = false;
  bool completion_unknown = false;
  RileyCudaGraphCapture* scoped_capture = nullptr;
};

namespace {
using namespace riley_cuda_internal;
constexpr const char* kReserve = "riley_cuda_graph_resources_reserve";
constexpr const char* kClose = "riley_cuda_graph_resources_close";

RileyCudaStatus reject(RileyCudaErrorInfo* error, const char* detail,
                       RileyCudaStatus status = RILEY_CUDA_STATUS_INVALID_STATE) noexcept {
  return validation_error(error, status, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kReserve, detail);
}

RileyCudaStatus acquire(RileyCudaGraphResources* resources,
                        RileyCudaGraphResources::Entry entry,
                        RileyCudaErrorInfo* error) noexcept {
  for (size_t i = 0; i < resources->count; ++i) {
    const auto& existing = resources->entries[i];
    if (existing.kind == entry.kind && existing.identity == entry.identity)
      return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (resources->count == RileyCudaGraphResources::kCapacity)
    return reject(error, "aggregate resource ledger capacity exceeded",
                  RILEY_CUDA_STATUS_OUT_OF_RANGE);
  RileyCudaStatus status = RILEY_CUDA_STATUS_SUCCESS;
  if (entry.kind == RileyCudaGraphResources::Kind::Plan) {
    status = acquire_canonical_gemm_bf16_graph_plan_lease(
        static_cast<RileyCudaGemmPlan*>(entry.identity), error, kReserve);
  } else if (!try_acquire_exclusive_use(*entry.counter)) {
    status = reject(error, "aggregate resource is busy");
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS)
    resources->entries[resources->count++] = entry;
  return status;
}

// Remove an entry only after known release. A failure retains that entry and
// all earlier entries, including the stream and context. No double release on
// retry, no speculative deletion after a broken counter invariant.
bool release_all(RileyCudaGraphResources* resources) noexcept {
  while (resources->count != 0) {
    const auto& entry = resources->entries[resources->count - 1];
    const bool released = entry.kind == RileyCudaGraphResources::Kind::Plan
        ? release_canonical_gemm_bf16_graph_plan_lease(
              static_cast<RileyCudaGemmPlan*>(entry.identity))
        : release_exclusive_use(*entry.counter);
    if (!released) return false;
    --resources->count;
  }
  if (resources->child_held) {
    if (!release_child(resources->owner)) return false;
    resources->child_held = false;
  }
  return true;
}
}  // namespace

extern "C" RileyCudaStatus riley_cuda_graph_resources_reserve(
    RileyCudaStream* stream,
    RileyCudaDeviceBuffer* const* devices, uint64_t device_count,
    RileyCudaPinnedHostBuffer* const* pinned, uint64_t pinned_count,
    RileyCudaGemmPlan* const* plans, uint64_t plan_count,
    RileyCudaGraphResources** out_resources, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_resources != nullptr) *out_resources = nullptr;
  if (out_resources == nullptr || stream == nullptr || stream->owner == nullptr ||
      (device_count != 0 && devices == nullptr) ||
      (pinned_count != 0 && pinned == nullptr) ||
      (plan_count != 0 && plans == nullptr))
    return reject(error, "null aggregate resource argument", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  // Bound each operand first, so the sum cannot overflow.
  if (device_count > 4096 || pinned_count > 4096 || plan_count > 4096 ||
      device_count + pinned_count + plan_count > 4096)
    return reject(error, "too many aggregate resource occurrences", RILEY_CUDA_STATUS_OUT_OF_RANGE);
  if (thread_has_active_graph_capture() || thread_has_active_command_batch() ||
      stream->owner->restoration_failed.load(std::memory_order_acquire))
    return reject(error, "aggregate reservation requires a healthy idle lifecycle");
  CaptureDomainControlLease admission(stream->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain is busy");
  // Validate the entire input table before acquiring any resource. Native
  // handle identity, not a device byte span or a role name, defines duplicates.
  for (uint64_t i = 0; i < device_count; ++i)
    if (devices[i] == nullptr || !same_context(devices[i]->owner, stream->owner))
      return reject(error, "device parent has a foreign or null context");
  for (uint64_t i = 0; i < pinned_count; ++i)
    if (pinned[i] == nullptr || !same_context(pinned[i]->owner, stream->owner))
      return reject(error, "pinned parent has a foreign or null context");
  for (uint64_t i = 0; i < plan_count; ++i)
    if (!aggregate_gemm_plan_matches_context(plans[i], stream->owner))
      return reject(error, "GEMM plan is foreign or lacks a selected no-split topology");
  void* storage = std::malloc(sizeof(RileyCudaGraphResources));
  if (storage == nullptr)
    return reject(error, "aggregate ledger allocation failed", RILEY_CUDA_STATUS_OUT_OF_MEMORY);
  auto* resources = new (storage) RileyCudaGraphResources{};
  resources->owner = stream->owner;
  resources->stream = stream;
  resources->thread = native_thread_token();
  resources->child_held = retain_child(resources->owner);
  if (!resources->child_held) {
    resources->~RileyCudaGraphResources();
    std::free(resources);
    return reject(error, "aggregate context child count exhausted");
  }
  using Kind = RileyCudaGraphResources::Kind;
  auto status = acquire(resources, {Kind::Counter, &stream->active_uses, &stream->active_uses}, error);
  for (uint64_t i = 0; status == RILEY_CUDA_STATUS_SUCCESS && i < device_count; ++i)
    status = acquire(resources, {Kind::Counter, &devices[i]->active_uses, &devices[i]->active_uses}, error);
  for (uint64_t i = 0; status == RILEY_CUDA_STATUS_SUCCESS && i < pinned_count; ++i)
    status = acquire(resources, {Kind::Counter, &pinned[i]->active_uses, &pinned[i]->active_uses}, error);
  for (uint64_t i = 0; status == RILEY_CUDA_STATUS_SUCCESS && i < plan_count; ++i)
    status = acquire(resources, {Kind::Plan, plans[i], nullptr}, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS && release_all(resources)) {
    resources->~RileyCudaGraphResources();
    std::free(resources);
  } else {
    *out_resources = resources;
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_close(
    RileyCudaGraphResources** resources, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (resources == nullptr)
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_CLOSE, kClose, "null aggregate owner address");
  if (*resources == nullptr) return RILEY_CUDA_STATUS_SUCCESS;
  if ((*resources)->thread != native_thread_token())
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, kClose, "aggregate owner belongs to another thread");
  if ((*resources)->completion_unknown)
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_CLOSE, kClose, "completion unknown; graph and parents retained");
  if ((*resources)->graph != nullptr || (*resources)->exec != nullptr ||
      (*resources)->prefill_graph != nullptr || (*resources)->prefill_exec != nullptr ||
      (*resources)->catalog[0].graph || (*resources)->catalog[0].exec ||
      (*resources)->catalog[1].graph || (*resources)->catalog[1].exec ||
      (*resources)->catalog[2].graph || (*resources)->catalog[2].exec ||
      (*resources)->catalog[3].graph || (*resources)->catalog[3].exec ||
      (*resources)->catalog[4].graph || (*resources)->catalog[4].exec ||
      (*resources)->catalog[5].graph || (*resources)->catalog[5].exec) {
    CaptureDomainControlLease admission((*resources)->owner->capture_domain);
    if (!admission.active())
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_CLOSE, kClose, "capture domain busy; graph retained");
    CurrentContext scope((*resources)->owner);
    auto status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
    if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
    if ((*resources)->exec != nullptr) {
      status = runtime_error(cudaGraphExecDestroy((*resources)->exec), error,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
      if (status == RILEY_CUDA_STATUS_SUCCESS) (*resources)->exec = nullptr;
      else (*resources)->completion_unknown = true; // Destroy consumption may be ambiguous; never retry.
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS && (*resources)->graph != nullptr) {
      status = runtime_error(cudaGraphDestroy((*resources)->graph), error,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
      if (status == RILEY_CUDA_STATUS_SUCCESS) (*resources)->graph = nullptr;
      else (*resources)->completion_unknown = true;
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS && (*resources)->prefill_exec != nullptr) {
      status = runtime_error(cudaGraphExecDestroy((*resources)->prefill_exec), error,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
      if (status == RILEY_CUDA_STATUS_SUCCESS) (*resources)->prefill_exec = nullptr;
      else (*resources)->completion_unknown = true;
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS && (*resources)->prefill_graph != nullptr) {
      status = runtime_error(cudaGraphDestroy((*resources)->prefill_graph), error,
                             RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
      if (status == RILEY_CUDA_STATUS_SUCCESS) (*resources)->prefill_graph = nullptr;
      else (*resources)->completion_unknown = true;
    }
    for(auto& entry:(*resources)->catalog){
      if(status==RILEY_CUDA_STATUS_SUCCESS&&entry.exec){
        status=runtime_error(cudaGraphExecDestroy(entry.exec),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);
        if(status==RILEY_CUDA_STATUS_SUCCESS)entry.exec=nullptr;else (*resources)->completion_unknown=true;
      }
      if(status==RILEY_CUDA_STATUS_SUCCESS&&entry.graph){
        status=runtime_error(cudaGraphDestroy(entry.graph),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);
        if(status==RILEY_CUDA_STATUS_SUCCESS)entry.graph=nullptr;else (*resources)->completion_unknown=true;
      }
    }
    status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
    if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  }
  if (!release_all(*resources))
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose,
                          "aggregate resource release is unknown; owner retained");
  (*resources)->~RileyCudaGraphResources();
  std::free(*resources);
  *resources = nullptr;
  return RILEY_CUDA_STATUS_SUCCESS;
}

namespace {
constexpr const char* kTransfer = "aggregate transfer graph";
RileyCudaStatus transfer_ready(RileyCudaGraphResources* r, RileyCudaErrorInfo* error) noexcept {
  if (r == nullptr || r->thread != native_thread_token() || r->terminal ||
      r->owner->restoration_failed.load(std::memory_order_acquire))
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kTransfer, "owner unavailable, foreign-thread or terminal");
  return RILEY_CUDA_STATUS_SUCCESS;
}
bool holds_counter(RileyCudaGraphResources* r, std::atomic<uint32_t>* counter) noexcept {
  for (size_t i = 0; i < r->count; ++i)
    if (r->entries[i].kind == RileyCudaGraphResources::Kind::Counter &&
        r->entries[i].counter == counter)
      return counter->load(std::memory_order_acquire) == 1;
  return false;
}
}  // namespace

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_transfer(
    RileyCudaGraphResources* r, RileyCudaPinnedHostBuffer* input,
    RileyCudaDeviceBuffer* first, RileyCudaDeviceBuffer* second,
    RileyCudaPinnedHostBuffer* output, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (input == nullptr || first == nullptr || second == nullptr || output == nullptr ||
      input == output || first == second || input->byte_len == 0 ||
      input->byte_len != first->byte_len || input->byte_len != second->byte_len ||
      input->byte_len != output->byte_len || input->byte_len > SIZE_MAX ||
      !same_context(input->owner, r->owner) || !same_context(first->owner, r->owner) ||
      !same_context(second->owner, r->owner) || !same_context(output->owner, r->owner) ||
      !holds_counter(r, &input->active_uses) || !holds_counter(r, &first->active_uses) ||
      !holds_counter(r, &second->active_uses) || !holds_counter(r, &output->active_uses) ||
      r->graph != nullptr || r->exec != nullptr)
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kTransfer, "transfer parents or graph state invalid");
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain busy");
  CurrentContext scope(r->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  // An error after graph construction starts is terminal, but close remains
  // available because no work has been submitted and handles stay owned here.
  r->terminal = true;
  cudaGraphNode_t nodes[3]{};
  auto result = cudaGraphCreate(&r->graph, 0);
  if (result == cudaSuccess)
    result = cudaGraphAddMemcpyNode1D(&nodes[0], r->graph, nullptr, 0,
        first->device_data, input->host_data, input->byte_len, cudaMemcpyHostToDevice);
  if (result == cudaSuccess)
    result = cudaGraphAddMemcpyNode1D(&nodes[1], r->graph, &nodes[0], 1,
        second->device_data, first->device_data, input->byte_len, cudaMemcpyDeviceToDevice);
  if (result == cudaSuccess)
    result = cudaGraphAddMemcpyNode1D(&nodes[2], r->graph, &nodes[1], 1,
        output->host_data, second->device_data, input->byte_len, cudaMemcpyDeviceToHost);
  if (result == cudaSuccess) result = cudaGraphInstantiate(&r->exec, r->graph, 0);
  status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    r->input = input;
    r->output = output;
    r->transfer_bytes = input->byte_len;
    r->terminal = false;
  }
  return status;
}

#include "graph_multisequence_packet.inc"

extern "C" RileyCudaStatus riley_cuda_graph_resources_replay_transfer(
    RileyCudaGraphResources* r, const uint8_t* source, uint64_t bytes,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  // Any attempted replay invalidates the previous result, even preflight errors.
  r->completion_visible = false;
  r->last_catalog = 0;
  if (r->exec == nullptr || source == nullptr || bytes != (r->v3_prefill_capacity?17536:r->transfer_bytes))
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kTransfer, "fresh source must cover exact transfer size");
  if(r->multi_bucket && !valid_multisequence_packet(source,bytes,r->multi_bucket,r->multi_physical,r->multi_full))
    return reject(error,"multi decode packet geometry invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  if(r->v3_prefill_capacity){
    if(!valid_prefill_shape_packet(source,bytes,r->v3_prefill_physical,8))return reject(error,"invalid V3 prefill packet",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    auto value=[&](size_t at){uint32_t v;std::memcpy(&v,source+at,4);return v;};
    if((!r->v3_shared&&value(20)!=1)||value(136)>r->v3_prefill_capacity||value(164)>r->v3_prefill_context)
      return reject(error,"V3 prefill graph shape mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(uint32_t row=0;row<value(20);++row)if(value(164+row*1664)>r->v3_prefill_context)return reject(error,"V3 row context exceeds retained tables",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  auto selected_exec = r->exec;
  if(r->v3_prefill_capacity){uint32_t stage;std::memcpy(&stage,source+16,4);if(stage==0){if(!r->prefill_exec)return reject(error,"V3 prefill capture missing");selected_exec=r->prefill_exec;}}
  if(r->decode_capacity!=0){
    auto u32=[&](uint64_t offset){uint32_t v;std::memcpy(&v,source+offset,4);return v;};
    const uint64_t capacity=r->decode_capacity,pos=u32(4),live=pos/16+1;
    const uint64_t slot=(16+6*capacity+3)&~uint64_t(3);
    if((r->vllm_smol_p128&&pos>=160)||u32(0)>=r->decode_vocab||live>capacity||u32(8)!=0||u32(12)!=live||u32(slot)!=0)
      return reject(error,"decode token or row mapping invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(uint64_t i=0;i<capacity;++i){
      uint16_t valid;std::memcpy(&valid,source+16+4*capacity+2*i,2);
      if(i<live){
        if(u32(16+4*i)>=r->decode_physical||valid!=(i+1<live?16:pos%16+1))
          return reject(error,"decode block mapping invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
        for(uint64_t j=0;j<i;++j) if(u32(16+4*i)==u32(16+4*j)) return reject(error,"decode duplicate physical block",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      }else if(valid!=0||u32(16+4*i)!=0) return reject(error,"decode padding invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
    uint32_t rows=1;
    if(r->prefill128){
      const uint64_t extension=slot+4;
      rows=u32(extension+4);
      if(u32(extension)!=0x50313238U || (rows!=1&&rows!=128) ||
          (rows==128&&(pos!=127||u32(0)!=u32(extension+8+127*4))) ||
          (rows==1&&pos<128))
        return reject(error,"P128 row extension invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(uint64_t i=0;i<128;++i){
        const uint32_t token=u32(extension+8+4*i);
        if((rows==128&&token>=r->decode_vocab)||(rows==1&&token!=0))
          return reject(error,"P128 appended tokens invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      }
    }
    // P128 is a fixed numerical contract. Select from this replay's validated
    // host metadata, never from a position value frozen during CUDA capture.
    if(r->vllm_smol_p128){
      if(r->prefill_exec==nullptr || r->prefill_graph==nullptr)
        return reject(error,"P128 stage graph is not prepared");
      selected_exec=(r->prefill128?rows==128:pos<128)?r->prefill_exec:r->exec;
    }
  }
  if (r->rope_table_positions != 0) {
    uint32_t position = 0;
    std::memcpy(&position, source + r->rope_payload_position_offset, sizeof(position));
    if (r->kv_write_bound && (position / 16 != r->kv_logical_block || position % 16 >= r->kv_valid_tokens))
      return reject(error, "KV replay position outside bound block", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    if (position >= r->rope_table_positions)
      return reject(error, "RoPE replay position outside table", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain busy");
  CurrentContext scope(r->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kTransfer);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  r->terminal = true;
  r->catalog_sealed=true;
  std::memmove(r->input->host_data, source, static_cast<size_t>(bytes));
  if (r->bound_attention) {
    auto* host = static_cast<uint8_t*>(r->input->host_data) + r->rope_payload_position_offset;
    const uint32_t offsets[2] = {0,1}; const uint32_t slot = 0;
    const uint16_t valid = static_cast<uint16_t>(r->kv_valid_tokens);
    std::memcpy(host+4,offsets,8); std::memcpy(host+12,&r->attention_physical_block,4);
    std::memcpy(host+16,&valid,2); std::memcpy(host+20,&slot,4);
  }
  r->completion_unknown = true;
  auto launched = cudaGraphLaunch(selected_exec, r->stream->stream);
  // Synchronize even on launch error; that is the only authority to release
  // possibly submitted work. A failed synchronization intentionally pins owners.
  auto completed = cudaStreamSynchronize(r->stream->stream);
#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
  // Test only: actual work is synchronized first. Override the observed result
  // to exercise fail-closed lifetime policy without faulting the shared GPU.
  if(r->test_replay_fault==1) launched=cudaErrorUnknown;
  if(r->test_replay_fault==2) completed=cudaErrorUnknown;
  r->test_replay_fault=0;
#endif
  if (completed == cudaSuccess) r->completion_unknown = false;
  status = runtime_error(launched != cudaSuccess ? launched : completed,
                         error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kTransfer);
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kTransfer);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    r->terminal = false;
    r->completion_visible = true;
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_read_transfer(
    RileyCudaGraphResources* r, uint8_t* destination, uint64_t bytes,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (r->last_catalog != 0 || !r->completion_visible || destination == nullptr || bytes != r->transfer_bytes)
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kTransfer, "no completed output of the requested size");
  std::memmove(destination, static_cast<uint8_t*>(r->output->host_data) + r->output_byte_offset, static_cast<size_t>(bytes));
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_swiglu(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* gate, RileyCudaDeviceBuffer* up,
    RileyCudaDeviceBuffer* activated, RileyCudaDeviceBuffer* product,
    RileyCudaPinnedHostBuffer* staging, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (gate == nullptr || staging == nullptr || gate->byte_len == 0 ||
      gate->byte_len % 2 != 0 || gate->byte_len > SIZE_MAX / 4 ||
      staging->byte_len < gate->byte_len * 4 ||
      !same_context(staging->owner, r->owner) || !holds_counter(r, &staging->active_uses) ||
      r->graph != nullptr || r->exec != nullptr)
    return reject(error, "SwiGLU graph geometry, staging or state invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  RileyCudaDeviceBuffer* parents[] = {gate, up, activated, product};
  for (size_t i = 0; i < 4; ++i) {
    if (parents[i] == nullptr || parents[i]->byte_len != gate->byte_len ||
        !same_context(parents[i]->owner, r->owner) || !holds_counter(r, &parents[i]->active_uses))
      return reject(error, "SwiGLU device parent invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t j = 0; j < i; ++j)
      if (parents[i] == parents[j]) return reject(error, "SwiGLU alias rejected", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain busy");
  CurrentContext scope(r->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  r->terminal = true;
  const auto bytes = gate->byte_len;
  auto* host = static_cast<uint8_t*>(staging->host_data);
  cudaGraphNode_t nodes[6]{};
  auto result = cudaGraphCreate(&r->graph, 0);
  if (result == cudaSuccess) result = cudaGraphAddMemcpyNode1D(&nodes[0], r->graph, nullptr, 0, gate->device_data, host, bytes, cudaMemcpyHostToDevice);
  if (result == cudaSuccess) result = cudaGraphAddMemcpyNode1D(&nodes[1], r->graph, nullptr, 0, up->device_data, host + bytes, bytes, cudaMemcpyHostToDevice);
  if (result == cudaSuccess) result = add_swiglu_graph_nodes(r->graph, nodes[0], nodes[1], gate->device_data, up->device_data, activated->device_data, product->device_data, bytes / 2, &nodes[2], &nodes[3]);
  if (result == cudaSuccess) result = cudaGraphAddMemcpyNode1D(&nodes[4], r->graph, &nodes[2], 1, host + 2 * bytes, activated->device_data, bytes, cudaMemcpyDeviceToHost);
  if (result == cudaSuccess) result = cudaGraphAddMemcpyNode1D(&nodes[5], r->graph, &nodes[3], 1, host + 3 * bytes, product->device_data, bytes, cudaMemcpyDeviceToHost);
  if (result == cudaSuccess) result = cudaGraphInstantiate(&r->exec, r->graph, 0);
  status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_PREPARE, "record SwiGLU graph");
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE, "record SwiGLU graph");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    r->input = staging;
    r->output = staging;
    r->transfer_bytes = bytes * 2;
    r->output_byte_offset = bytes * 2;
    r->terminal = false;
  }
  return status;
}

namespace {
bool holds_plan(RileyCudaGraphResources* r, RileyCudaGemmPlan* plan) noexcept {
  for (size_t i = 0; i < r->count; ++i)
    if (r->entries[i].kind == RileyCudaGraphResources::Kind::Plan && r->entries[i].identity == plan)
      return true;
  return false;
}
}  // namespace

// Internal synchronous callback only; callers preflight all parents before capture.
template <typename Enqueue>
static RileyCudaStatus record_reserved_sequence(RileyCudaGraphResources* r,
    RileyCudaPinnedHostBuffer* staging, uint64_t transfer_bytes,
    Enqueue enqueue, RileyCudaErrorInfo* error) noexcept {
  auto status = RILEY_CUDA_STATUS_SUCCESS;
  const auto id = next_graph_capture_id();
  if (id == 0) return reject(error, "capture identity exhausted");
  void* storage = std::malloc(sizeof(RileyCudaGraphCapture));
  if (storage == nullptr) return reject(error, "capture guard allocation failed", RILEY_CUDA_STATUS_OUT_OF_MEMORY);
  auto* capture = new (storage) RileyCudaGraphCapture(r->owner, r->stream, r->owner->capture_domain, native_thread_token(), id);
  if (!try_begin_capture_domain(capture->capture_domain)) {
    capture->~RileyCudaGraphCapture(); std::free(capture);
    return reject(error, "capture domain has pending work");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    (void)release_capture_domain_capture(capture->capture_domain);
    capture->~RileyCudaGraphCapture(); std::free(capture);
    return reject(error, "capture thread is busy");
  }
  // A single synchronous FFI call owns the capture. No Rust/user callback or
  // borrowed-resource destructor runs between publication and termination.
  r->scoped_capture = capture;
  r->terminal = true;
  r->completion_unknown = true;
  CurrentContext scope(r->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE, "record MLP graph", capture);
  bool begin_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    begin_attempted = true;
    const auto started = cudaStreamBeginCapture(r->stream->stream, cudaStreamCaptureModeThreadLocal);
    capture->capture_started = started == cudaSuccess;
    status = runtime_error(started, error, RILEY_CUDA_ERROR_STAGE_PREPARE, "record MLP graph");
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue();
  bool termination_known = !begin_attempted;
  if (begin_attempted) {
    cudaStreamCaptureStatus observed{};
    const auto queried = cudaStreamIsCapturing(r->stream->stream, &observed);
    if (queried == cudaSuccess && observed == cudaStreamCaptureStatusNone) {
      termination_known = true;
    } else {
      cudaGraph_t captured = nullptr;
      const auto ended = cudaStreamEndCapture(r->stream->stream, &captured);
      if (captured != nullptr) r->graph = captured;
      if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(ended, error, RILEY_CUDA_ERROR_STAGE_PREPARE, "end MLP capture");
      termination_known = cudaStreamIsCapturing(r->stream->stream, &observed) == cudaSuccess && observed == cudaStreamCaptureStatusNone;
    }
  }
  capture->capture_terminated = termination_known;
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE, "record MLP graph");
  // Unexpected deferred work or ambiguous capture end strands the published
  // guard, context child and all leases; no false close/replay authorization.
  if (!termination_known || capture->deferred_close_head != nullptr || capture->deferred_close_tail != nullptr)
    return reject(error, "MLP capture termination unknown; owner retained");
  if (!clear_thread_graph_capture_owner(capture) || !release_capture_domain_capture(capture->capture_domain))
    return reject(error, "MLP capture guard release unknown; owner retained");
  capture->~RileyCudaGraphCapture(); std::free(capture); r->scoped_capture = nullptr;
  r->completion_unknown = false;
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (r->graph == nullptr) return reject(error, "MLP capture returned no graph");
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "MLP instantiate domain busy");
  CurrentContext instantiate_scope(r->owner);
  status = instantiate_scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE, "instantiate MLP");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaGraphInstantiate(&r->exec, r->graph, 0), error, RILEY_CUDA_ERROR_STAGE_PREPARE, "instantiate MLP");
  status = instantiate_scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE, "instantiate MLP");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    r->input = staging; r->output = staging; r->transfer_bytes = transfer_bytes;
    r->output_byte_offset = transfer_bytes; r->terminal = false;
  }
  return status;
}

static RileyCudaStatus record_mlp_impl(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* intermediate, RileyCudaGemmPlan* down,
    RileyCudaPinnedHostBuffer* staging, RileyCudaDeviceBuffer* norm_weight,
    float epsilon, uint32_t norm_profile, RileyCudaDeviceBuffer* projection_weight,
    RileyCudaGemmPlan* projection_plan, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (d == nullptr || staging == nullptr || r->graph != nullptr || r->exec != nullptr ||
      thread_has_active_graph_capture() || thread_has_active_command_batch() ||
      !holds_plan(r, intermediate) || !holds_plan(r, down) ||
      !same_context(staging->owner, r->owner) || !holds_counter(r, &staging->active_uses))
    return reject(error, "MLP graph parents or lifecycle invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for (size_t i = 0; i < 12; ++i) {
    if (i == 11 && d[i] == nullptr) continue;
    if (d[i] == nullptr || !same_context(d[i]->owner, r->owner) || !holds_counter(r, &d[i]->active_uses))
      return reject(error, "MLP parent absent from ledger", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t j = 0; j < i; ++j)
      if (d[i] == d[j]) return reject(error, "MLP alias rejected", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  const uint64_t hidden_bytes = d[0]->byte_len;
  const uint64_t intermediate_bytes = d[2]->byte_len;
  if (hidden_bytes == 0 || hidden_bytes % 2 != 0 || hidden_bytes > SIZE_MAX / 4 ||
      intermediate_bytes == 0 || intermediate_bytes % 2 != 0 ||
      staging->byte_len < hidden_bytes * 4 || d[1]->byte_len != hidden_bytes ||
      d[6]->byte_len != hidden_bytes || d[7]->byte_len != hidden_bytes ||
      d[3]->byte_len != intermediate_bytes || d[4]->byte_len != intermediate_bytes ||
      d[5]->byte_len != intermediate_bytes)
    return reject(error, "MLP scratch geometry mismatch", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  if (norm_weight != nullptr) {
    if (!same_context(norm_weight->owner, r->owner) ||
        !holds_counter(r, &norm_weight->active_uses) ||
        norm_weight->byte_len != hidden_bytes || !std::isfinite(epsilon) ||
        epsilon <= 0.0F || norm_profile > 1 ||
        (norm_profile == 1 && (hidden_bytes != 1152 || epsilon != 1e-5F)))
      return reject(error, "norm MLP profile or weight invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t i = 0; i < 12; ++i)
      if (norm_weight == d[i]) return reject(error, "norm MLP weight alias", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  RileyCudaCanonicalGemmBf16GraphState gate_state{}, up_state{}, down_state{};
  status = bind_reserved_gemm_state(intermediate, r->stream, d[0], d[8], d[2], d[11], &gate_state, error);
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = bind_reserved_gemm_state(intermediate, r->stream, d[0], d[9], d[3], d[11], &up_state, error);
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = bind_reserved_gemm_state(down, r->stream, d[5], d[10], d[6], d[11], &down_state, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  RileyCudaCanonicalGemmBf16GraphState projection_state{};
  if (projection_weight != nullptr) {
    if (norm_weight == nullptr || projection_weight == norm_weight ||
        !holds_plan(r, projection_plan) ||
        !same_context(projection_weight->owner, r->owner) ||
        !holds_counter(r, &projection_weight->active_uses))
      return reject(error, "layer tail projection parent invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t i = 0; i < 12; ++i)
      if (projection_weight == d[i]) return reject(error, "layer tail projection alias", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    status = bind_reserved_gemm_state(projection_plan, r->stream, d[0], projection_weight, d[7], d[11], &projection_state, error);
    if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  }
  return record_reserved_sequence(r, staging, hidden_bytes * 2, [&]() noexcept {
    auto status = RILEY_CUDA_STATUS_SUCCESS;
  auto* host = static_cast<uint8_t*>(staging->host_data);
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(d[0]->device_data, host, hidden_bytes, cudaMemcpyHostToDevice, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "MLP input");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(d[1]->device_data, host + hidden_bytes, hidden_bytes, cudaMemcpyHostToDevice, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "MLP residual");
  if (status == RILEY_CUDA_STATUS_SUCCESS && projection_weight != nullptr)
    status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[7], projection_state, error, "layer tail output projection");
  // Each residual output element depends only on its matching input elements;
  // the eager elementwise kernel supports this explicitly ordered in-place use.
  if (status == RILEY_CUDA_STATUS_SUCCESS && projection_weight != nullptr)
    status = runtime_error(enqueue_mlp_residual(r->stream->stream, d[1]->device_data,
        d[7]->device_data, d[1]->device_data, hidden_bytes / 2), error,
        RILEY_CUDA_ERROR_STAGE_LAUNCH, "layer tail attention residual");
  if (status == RILEY_CUDA_STATUS_SUCCESS && norm_weight != nullptr)
    status = runtime_error(enqueue_mlp_norm(r->stream->stream, d[1]->device_data,
        norm_weight->device_data, d[0]->device_data, hidden_bytes / 2, epsilon,
        norm_profile), error, RILEY_CUDA_ERROR_STAGE_LAUNCH, "MLP post-attention norm");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[2], gate_state, error, "MLP gate");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[3], up_state, error, "MLP up");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(enqueue_mlp_pointwise(r->stream->stream, d[2]->device_data, d[3]->device_data, d[4]->device_data, d[5]->device_data, intermediate_bytes / 2), error, RILEY_CUDA_ERROR_STAGE_LAUNCH, "MLP pointwise");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[6], down_state, error, "MLP down");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(enqueue_mlp_residual(r->stream->stream, d[1]->device_data, d[6]->device_data, d[7]->device_data, hidden_bytes / 2), error, RILEY_CUDA_ERROR_STAGE_LAUNCH, "MLP residual add");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(host + 2 * hidden_bytes, d[6]->device_data, hidden_bytes, cudaMemcpyDeviceToHost, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "MLP down result");
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(host + 3 * hidden_bytes, d[7]->device_data, hidden_bytes, cudaMemcpyDeviceToHost, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "MLP final result");
    return status;
  }, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_mlp(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* intermediate, RileyCudaGemmPlan* down,
    RileyCudaPinnedHostBuffer* staging, RileyCudaErrorInfo* error) noexcept {
  return record_mlp_impl(r, d, intermediate, down, staging, nullptr, 0.0F, 0, nullptr, nullptr, error);
}
extern "C" RileyCudaStatus riley_cuda_graph_resources_record_norm_mlp(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* intermediate, RileyCudaGemmPlan* down,
    RileyCudaPinnedHostBuffer* staging, RileyCudaDeviceBuffer* norm_weight,
    float epsilon, uint32_t profile, RileyCudaErrorInfo* error) noexcept {
  if (norm_weight == nullptr) {
    clear_error(error);
    return reject(error, "norm MLP weight required", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  return record_mlp_impl(r, d, intermediate, down, staging, norm_weight, epsilon, profile, nullptr, nullptr, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_layer_tail(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* intermediate, RileyCudaGemmPlan* down,
    RileyCudaPinnedHostBuffer* staging, RileyCudaDeviceBuffer* norm_weight,
    float epsilon, uint32_t profile, RileyCudaDeviceBuffer* projection_weight,
    RileyCudaGemmPlan* projection_plan, RileyCudaErrorInfo* error) noexcept {
  if (norm_weight == nullptr || projection_weight == nullptr || projection_plan == nullptr) {
    clear_error(error);
    return reject(error, "layer tail parents required", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  return record_mlp_impl(r, d, intermediate, down, staging, norm_weight, epsilon,
      profile, projection_weight, projection_plan, error);
}

static RileyCudaStatus record_norm_qkv_impl(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* query, RileyCudaGemmPlan* kv,
    RileyCudaPinnedHostBuffer* staging, float epsilon, uint32_t profile,
    RileyCudaDeviceBuffer* const* rope, uint64_t position_offset,
    RileyCudaDeviceBuffer* const* caches, const uint64_t* geometry,
    RileyCudaDeviceBuffer* attention, const uint64_t* fields,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (d == nullptr || staging == nullptr || r->graph != nullptr || r->exec != nullptr ||
      thread_has_active_graph_capture() || thread_has_active_command_batch() ||
      !holds_plan(r, query) || !holds_plan(r, kv) ||
      !same_context(staging->owner, r->owner) || !holds_counter(r, &staging->active_uses))
    return reject(error, "norm QKV lifecycle or parent invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for (size_t i = 0; i < 10; ++i) {
    if (i == 9 && d[i] == nullptr) continue;
    if (d[i] == nullptr || !same_context(d[i]->owner, r->owner) || !holds_counter(r, &d[i]->active_uses))
      return reject(error, "norm QKV parent absent", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t j = 0; j < i; ++j)
      if (d[i] == d[j]) return reject(error, "norm QKV alias", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  const uint64_t h = d[0]->byte_len, k = d[3]->byte_len;
  if (h == 0 || h % 2 != 0 || h > SIZE_MAX / 6 || k == 0 || k % 2 != 0 || k > h ||
      d[1]->byte_len != h || d[2]->byte_len != h || d[4]->byte_len != k || d[5]->byte_len != h ||
      staging->byte_len < 2 * (h + 2 * k) || !std::isfinite(epsilon) || epsilon <= 0 || profile > 1 ||
      (profile == 1 && (h != 1152 || epsilon != 1e-5F)))
    return reject(error, "norm QKV geometry/profile invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  uint64_t table_positions = 0;
  if (rope != nullptr) {
    for (size_t i = 0; i < 5; ++i) {
      if (rope[i] == nullptr || !same_context(rope[i]->owner, r->owner) || !holds_counter(r, &rope[i]->active_uses))
        return reject(error, "QKV RoPE parent absent", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for (size_t j = 0; j < 10; ++j)
        if (rope[i] == d[j]) return reject(error, "QKV RoPE alias", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for (size_t j = 0; j < i; ++j)
        if (rope[i] == rope[j]) return reject(error, "QKV RoPE alias", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
    if (h % 128 != 0 || k % 128 != 0 || rope[0]->byte_len != h || rope[1]->byte_len != k ||
        rope[2]->byte_len == 0 || rope[2]->byte_len % 128 != 0 || rope[3]->byte_len != rope[2]->byte_len ||
        position_offset % 4 != 0 || position_offset > rope[4]->byte_len || rope[4]->byte_len - position_offset < 4)
      return reject(error, "QKV RoPE D64 geometry invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    table_positions = rope[2]->byte_len / 128;
  }
  uint64_t kv_layer_offset = 0;
  if (caches != nullptr) {
    if (rope == nullptr || geometry == nullptr || geometry[0] == 0 || geometry[1] >= geometry[0] ||
        geometry[2] == 0 || geometry[3] >= geometry[2] || geometry[4] > UINT32_MAX / 16 ||
        geometry[5] == 0 || geometry[5] > 16 || k > UINT64_MAX / 16 / geometry[2] ||
        geometry[0] > UINT64_MAX / (k * 16 * geometry[2]))
      return reject(error, "KV fixed block geometry invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    const uint64_t stride = k * 16 * geometry[2];
    for (size_t i = 0; i < 2; ++i) {
      if (caches[i] == nullptr || !same_context(caches[i]->owner,r->owner) ||
          !holds_counter(r,&caches[i]->active_uses) || caches[i]->byte_len != stride * geometry[0])
        return reject(error, "KV parent missing or wrong size", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for (size_t j = 0; j < 10; ++j) if (caches[i] == d[j]) return reject(error,"KV alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for (size_t j = 0; j < 5; ++j) if (caches[i] == rope[j]) return reject(error,"KV alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
    if (caches[0] == caches[1]) return reject(error,"KV parents alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    kv_layer_offset = stride * geometry[1];
  }
  if (attention != nullptr) {
    if (caches == nullptr || fields == nullptr || geometry[4] != 0 || geometry[3] > UINT32_MAX ||
        geometry[2] > UINT32_MAX || h % k != 0 || h / 128 > UINT32_MAX ||
        attention->byte_len != h || !same_context(attention->owner,r->owner) ||
        !holds_counter(r,&attention->active_uses))
      return reject(error,"attention fixed one-block geometry invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t i=0;i<10;++i) if (attention==d[i]) return reject(error,"attention alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for (size_t i=0;i<5;++i) if (attention==rope[i]) return reject(error,"attention alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    if (attention==caches[0] || attention==caches[1]) return reject(error,"attention alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    const uint64_t lengths[5]={8,4,2,4,4};
    const uint64_t offsets[5]={fields[0],fields[1],fields[2],fields[3],position_offset};
    for (size_t i=0;i<5;++i) {
      if (offsets[i] % (i==2?2:4) != 0 || offsets[i]>rope[4]->byte_len || lengths[i]>rope[4]->byte_len-offsets[i])
        return reject(error,"attention metadata range invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<i;++j) if(offsets[i]<offsets[j]+lengths[j] && offsets[j]<offsets[i]+lengths[i])
        return reject(error,"attention metadata overlap",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
  }
  RileyCudaCanonicalGemmBf16GraphState q{}, key{}, value{};
  status = bind_reserved_gemm_state(query, r->stream, d[1], d[6], d[2], d[9], &q, error);
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = bind_reserved_gemm_state(kv, r->stream, d[1], d[7], d[3], d[9], &key, error);
  if (status == RILEY_CUDA_STATUS_SUCCESS) status = bind_reserved_gemm_state(kv, r->stream, d[1], d[8], d[4], d[9], &value, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  const uint64_t result_bytes = h + 2 * k;
  status = record_reserved_sequence(r, staging, result_bytes, [&]() noexcept {
    auto* host = static_cast<uint8_t*>(staging->host_data);
    auto status = runtime_error(cudaMemcpyAsync(d[0]->device_data, host, h, cudaMemcpyHostToDevice, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "QKV input");
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(enqueue_mlp_norm(r->stream->stream, d[0]->device_data, d[5]->device_data, d[1]->device_data, h / 2, epsilon, profile), error, RILEY_CUDA_ERROR_STAGE_LAUNCH, "QKV norm");
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[2], q, error, "QKV query");
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[3], key, error, "QKV key");
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = enqueue_canonical_gemm_bf16_graph_matmul(r->owner, r->stream, d[4], value, error, "QKV value");
    if (status == RILEY_CUDA_STATUS_SUCCESS && rope != nullptr)
      status = runtime_error(cudaMemcpyAsync(static_cast<uint8_t*>(rope[4]->device_data) + position_offset, host + h, 4, cudaMemcpyHostToDevice, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "QKV RoPE position");
    if (status == RILEY_CUDA_STATUS_SUCCESS && rope != nullptr)
      status = runtime_error(enqueue_qkv_rope(r->stream->stream, d[2]->device_data, d[3]->device_data,
          rope[0]->device_data, rope[1]->device_data, rope[2]->device_data, rope[3]->device_data,
          static_cast<uint8_t*>(rope[4]->device_data) + position_offset, h / 128, k / 128, table_positions), error,
          RILEY_CUDA_ERROR_STAGE_LAUNCH, "QKV RoPE");
    if (status == RILEY_CUDA_STATUS_SUCCESS && caches != nullptr)
      status = runtime_error(enqueue_bound_kv_write(r->stream->stream, rope[1]->device_data, d[4]->device_data,
          static_cast<uint8_t*>(caches[0]->device_data)+kv_layer_offset,
          static_cast<uint8_t*>(caches[1]->device_data)+kv_layer_offset,
          static_cast<uint8_t*>(rope[4]->device_data)+position_offset, k / 128,
          geometry[3]), error, RILEY_CUDA_ERROR_STAGE_LAUNCH, "bound KV write");
    if (status == RILEY_CUDA_STATUS_SUCCESS && attention != nullptr) {
      const uint64_t lengths[4]={8,4,2,4}, source_offsets[4]={4,12,16,20};
      for (size_t i=0;i<4 && status==RILEY_CUDA_STATUS_SUCCESS;++i)
        status=runtime_error(cudaMemcpyAsync(static_cast<uint8_t*>(rope[4]->device_data)+fields[i],host+h+source_offsets[i],lengths[i],cudaMemcpyHostToDevice,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"attention fixed metadata");
      if(status==RILEY_CUDA_STATUS_SUCCESS)
        status=runtime_error(enqueue_bound_attention(r->stream->stream,rope[0]->device_data,
            static_cast<uint8_t*>(caches[0]->device_data)+kv_layer_offset,
            static_cast<uint8_t*>(caches[1]->device_data)+kv_layer_offset,attention->device_data,
            rope[4]->device_data,fields,position_offset,h/128,k/128,geometry[2]),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"bound attention");
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(host + result_bytes, (attention != nullptr ? attention : (rope != nullptr ? rope[0] : d[2]))->device_data, h, cudaMemcpyDeviceToHost, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "QKV query result");
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(host + result_bytes + h, (rope != nullptr ? rope[1] : d[3])->device_data, k, cudaMemcpyDeviceToHost, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "QKV key result");
    if (status == RILEY_CUDA_STATUS_SUCCESS) status = runtime_error(cudaMemcpyAsync(host + result_bytes + h + k, d[4]->device_data, k, cudaMemcpyDeviceToHost, r->stream->stream), error, RILEY_CUDA_ERROR_STAGE_COPY, "QKV value result");
    return status;
  }, error);
  if (status == RILEY_CUDA_STATUS_SUCCESS && rope != nullptr) {
    r->rope_table_positions = table_positions;
    r->rope_payload_position_offset = h;
    if (caches != nullptr) {
      r->kv_write_bound = true; r->kv_logical_block = geometry[4]; r->kv_valid_tokens = geometry[5];
      if(attention!=nullptr){r->bound_attention=true;r->attention_physical_block=static_cast<uint32_t>(geometry[3]);}
    }
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_norm_qkv(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* query, RileyCudaGemmPlan* kv, RileyCudaPinnedHostBuffer* staging,
    float epsilon, uint32_t profile, RileyCudaErrorInfo* error) noexcept {
  return record_norm_qkv_impl(r,d,query,kv,staging,epsilon,profile,nullptr,0,nullptr,nullptr,nullptr,nullptr,error);
}
extern "C" RileyCudaStatus riley_cuda_graph_resources_record_qkv_rope(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* query, RileyCudaGemmPlan* kv, RileyCudaPinnedHostBuffer* staging,
    float epsilon, uint32_t profile, RileyCudaDeviceBuffer* const* rope,
    uint64_t position_offset, RileyCudaErrorInfo* error) noexcept {
  if (rope == nullptr) { clear_error(error); return reject(error,"RoPE parents required",RILEY_CUDA_STATUS_INVALID_ARGUMENT); }
  return record_norm_qkv_impl(r,d,query,kv,staging,epsilon,profile,rope,position_offset,nullptr,nullptr,nullptr,nullptr,error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_qkv_kv(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d,
    RileyCudaGemmPlan* query, RileyCudaGemmPlan* kv, RileyCudaPinnedHostBuffer* staging,
    float epsilon, uint32_t profile, RileyCudaDeviceBuffer* const* rope,
    uint64_t position_offset, RileyCudaDeviceBuffer* const* caches,
    const uint64_t* geometry, RileyCudaErrorInfo* error) noexcept {
  if (rope == nullptr || caches == nullptr || geometry == nullptr) {clear_error(error);return reject(error,"KV binding required",RILEY_CUDA_STATUS_INVALID_ARGUMENT);}
  return record_norm_qkv_impl(r,d,query,kv,staging,epsilon,profile,rope,position_offset,caches,geometry,nullptr,nullptr,error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_attention_chain(
    RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,RileyCudaGemmPlan* q,
    RileyCudaGemmPlan* kv,RileyCudaPinnedHostBuffer* staging,float epsilon,uint32_t profile,
    RileyCudaDeviceBuffer* const* rope,uint64_t offset,RileyCudaDeviceBuffer* const* caches,
    const uint64_t* geometry,RileyCudaDeviceBuffer* attention,const uint64_t* fields,RileyCudaErrorInfo* error) noexcept {
  if(rope==nullptr||caches==nullptr||geometry==nullptr||attention==nullptr||fields==nullptr){clear_error(error);return reject(error,"attention binding required",RILEY_CUDA_STATUS_INVALID_ARGUMENT);}
  return record_norm_qkv_impl(r,d,q,kv,staging,epsilon,profile,rope,offset,caches,geometry,attention,fields,error);
}

// Full single-row model DAG. d: token, hidden, norm, projection, rotary-Q,
// context, raw-K, raw-V, rotary-K, gate, up, activated, product, logits,
// embedding-report, key-pool, value-pool, cos, sin, metadata, argmax, workspace.
// w: embedding, final-norm, head, then [input-norm,Q,K,V,O,post-norm,gate,up,down].
static RileyCudaStatus record_decode_impl(
    RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,
    RileyCudaDeviceBuffer* const* w,uint64_t weight_count,RileyCudaGemmPlan* const* plans,
    RileyCudaPinnedHostBuffer* staging,const uint64_t* geometry,const float* eps,
    uint32_t profile,uint32_t publish_logits,RileyCudaDeviceBuffer* const* prefill_buffers,
    RileyCudaDeviceBuffer* const* packed_buffers,RileyCudaGemmPlan* const* packed_plans,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status=transfer_ready(r,error); if(status!=RILEY_CUDA_STATUS_SUCCESS) return status;
  if(d==nullptr||w==nullptr||plans==nullptr||geometry==nullptr||eps==nullptr||staging==nullptr||
      r->graph!=nullptr||r->exec!=nullptr||r->prefill_graph!=nullptr||r->prefill_exec!=nullptr||
      thread_has_active_graph_capture()||thread_has_active_command_batch()||
      !same_context(staging->owner,r->owner)||!holds_counter(r,&staging->active_uses))
    return reject(error,"decode lifecycle invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  const uint64_t layers=geometry[0],physical=geometry[1],capacity=geometry[2],vocab=geometry[3];
  if(layers==0||layers>128||physical==0||physical>4096||capacity==0||capacity>physical||vocab==0||vocab>UINT32_MAX||
      weight_count!=3+9*layers||profile>2||publish_logits>1||(prefill_buffers!=nullptr&&profile!=2)||
      ((packed_buffers==nullptr)!=(packed_plans==nullptr))||
      (packed_buffers!=nullptr&&(profile!=2||prefill_buffers==nullptr||layers!=30)))
    return reject(error,"decode bucket unsupported",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  if(profile==2){
    int major=0,minor=0,runtime=0;
    if(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,r->owner->device)!=CUDA_SUCCESS||
       cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,r->owner->device)!=CUDA_SUCCESS||
       cudaRuntimeGetVersion(&runtime)!=cudaSuccess||major!=8||minor!=9||runtime!=13000||cublasLtGetVersion()!=130101)
      return reject(error,"vllm-smol-p128-v1 requires SM89 and CUDA runtime 13.0",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  for(size_t i=0;i<22;++i){
    if(i==21&&d[i]==nullptr) continue;
    if(d[i]==nullptr||!same_context(d[i]->owner,r->owner)||!holds_counter(r,&d[i]->active_uses))
      return reject(error,"decode device parent absent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(size_t j=0;j<i;++j) if(d[i]==d[j]) return reject(error,"decode scratch alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  for(uint64_t i=0;i<weight_count;++i){
    if(w[i]==nullptr||!same_context(w[i]->owner,r->owner)||!holds_counter(r,&w[i]->active_uses))
      return reject(error,"decode weight absent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(size_t j=0;j<22;++j) if(w[i]==d[j]) return reject(error,"decode weight scratch alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  for(size_t i=0;i<5;++i) if(!holds_plan(r,plans[i])) return reject(error,"decode plan absent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  const uint64_t h=d[1]->byte_len,k=d[6]->byte_len,inter=d[9]->byte_len;
  const uint64_t legacy_metadata=((16+6*capacity+3)&~uint64_t(3))+4;
  const uint64_t metadata=legacy_metadata+(prefill_buffers!=nullptr?520:0);
  const uint64_t logits_output=publish_logits?vocab*2:0;
  const uint64_t result=logits_output+sizeof(RileyCudaBf16ArgmaxResult)+sizeof(RileyCudaEmbeddingErrorReport);
  const uint64_t transfer=result>metadata?result:metadata;
  if(h==0||h%128!=0||h>UINT32_MAX||k==0||k%128!=0||h%k!=0||inter==0||inter%2!=0||inter>UINT32_MAX||
      d[0]->byte_len!=4||d[7]->byte_len!=k||d[8]->byte_len!=k||d[13]->byte_len!=vocab*2||
      d[14]->byte_len!=sizeof(RileyCudaEmbeddingErrorReport)||d[20]->byte_len!=sizeof(RileyCudaBf16ArgmaxResult)||
      d[15]->byte_len!=k*16*physical*layers||d[16]->byte_len!=d[15]->byte_len||
      d[17]->byte_len==0||d[17]->byte_len%128!=0||d[18]->byte_len!=d[17]->byte_len||
      d[19]->byte_len<metadata||staging->byte_len<2*transfer||w[0]->byte_len!=vocab*h||w[1]->byte_len!=h||
      (profile>=1&&h!=1152)||(profile==2&&(k!=384||inter!=3072||layers!=30||vocab!=49152)))
    return reject(error,"decode geometry mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t i=2;i<=5;++i) if(d[i]->byte_len!=h) return reject(error,"decode hidden geometry",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t i=10;i<=12;++i) if(d[i]->byte_len!=inter) return reject(error,"decode MLP geometry",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(uint64_t i=0;i<1+2*layers;++i) if(!std::isfinite(eps[i])||eps[i]<=0||(profile>=1&&eps[i]!=1e-5F))
    return reject(error,"decode norm profile",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(uint64_t l=0;l<layers;++l) if(w[3+9*l]->byte_len!=h||w[3+9*l+5]->byte_len!=h)
    return reject(error,"decode norm size",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  if(prefill_buffers!=nullptr){
    const uint64_t row_bytes[12]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};
    if(d[19]->byte_len!=metadata)
      return reject(error,"P128 metadata size mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(size_t i=0;i<12;++i){
      auto* parent=prefill_buffers[i];
      if(parent==nullptr||!same_context(parent->owner,r->owner)||
          !holds_counter(r,&parent->active_uses)||parent->byte_len!=128*row_bytes[i])
        return reject(error,"P128 scratch parent mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<i;++j) if(parent==prefill_buffers[j])
        return reject(error,"P128 scratch alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<22;++j) if(parent==d[j])
        return reject(error,"P128 decode scratch alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(uint64_t j=0;j<weight_count;++j) if(parent==w[j])
        return reject(error,"P128 weight alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
  }
  if(packed_buffers!=nullptr){
    for(size_t i=0;i<62;++i){
      auto* parent=packed_buffers[i];
      const uint64_t bytes=i==60?1920:i==61?6144:(i%2==0?1105920:3538944);
      if(parent==nullptr||!same_context(parent->owner,r->owner)||
          !holds_counter(r,&parent->active_uses)||parent->byte_len!=bytes)
        return reject(error,"packed decode parent mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<i;++j) if(parent==packed_buffers[j])
        return reject(error,"packed decode parent alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<22;++j) if(parent==d[j])
        return reject(error,"packed decode scratch alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(uint64_t j=0;j<weight_count;++j) if(parent==w[j])
        return reject(error,"packed decode weight alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<12;++j) if(parent==prefill_buffers[j])
        return reject(error,"packed decode prefill alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
    for(size_t i=0;i<2;++i){
      if(packed_plans[i]==nullptr||!holds_plan(r,packed_plans[i]))
        return reject(error,"packed decode plan absent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      if(i!=0&&packed_plans[i]==packed_plans[0])
        return reject(error,"packed decode plan alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      for(size_t j=0;j<5;++j) if(packed_plans[i]==plans[j])
        return reject(error,"packed decode original plan alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
      if(!reserved_packed_decode_gemm_plan_matches(packed_plans[i],r->owner,i==0?960:3072))
        return reject(error,"packed decode plan differs from qualified identity",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    }
  }
  using State=RileyCudaCanonicalGemmBf16GraphState;
  const uint64_t original_state_count=7*layers+1;
  const uint64_t state_count=original_state_count+(packed_buffers!=nullptr?2*layers:0);
  auto* states=static_cast<State*>(std::malloc(static_cast<size_t>(state_count)*sizeof(State)));
  if(states==nullptr) return reject(error,"decode binding allocation failed",RILEY_CUDA_STATUS_OUT_OF_MEMORY);
  for(uint64_t i=0;i<state_count;++i) new (&states[i]) State();
  auto* packed_states=states+original_state_count;
  auto release_states=[&]() noexcept { for(uint64_t i=0;i<state_count;++i) states[i].~State(); std::free(states); };
  for(uint64_t l=0;l<layers&&status==RILEY_CUDA_STATUS_SUCCESS;++l){
    auto* weights=w+3+9*l;
    const size_t pi[7]={0,1,1,0,2,2,3},inputs[7]={2,2,2,5,2,2,12},outputs[7]={3,6,7,3,9,10,5},wi[7]={1,2,3,4,6,7,8};
    for(size_t j=0;j<7&&status==RILEY_CUDA_STATUS_SUCCESS;++j)
      status=bind_reserved_gemm_state(plans[pi[j]],r->stream,d[inputs[j]],weights[wi[j]],d[outputs[j]],d[21],&states[l*7+j],error);
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS) status=bind_reserved_gemm_state(plans[4],r->stream,d[2],w[2],d[13],d[21],&states[layers*7],error);
  if(packed_buffers!=nullptr){
    for(uint64_t l=0;l<layers&&status==RILEY_CUDA_STATUS_SUCCESS;++l){
      status=bind_reserved_gemm_state(packed_plans[0],r->stream,d[2],packed_buffers[2*l],packed_buffers[60],nullptr,&packed_states[2*l],error);
      if(status==RILEY_CUDA_STATUS_SUCCESS)
        status=bind_reserved_gemm_state(packed_plans[1],r->stream,d[2],packed_buffers[2*l+1],packed_buffers[61],nullptr,&packed_states[2*l+1],error);
    }
  }
  if(status!=RILEY_CUDA_STATUS_SUCCESS){release_states();return status;}
  std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,static_cast<size_t>(transfer));
  auto record_stage = [&](bool prefill) noexcept {
    const bool batched=prefill&&prefill_buffers!=nullptr;
    const bool packed_decode=!prefill&&packed_buffers!=nullptr;
    const uint32_t rows=batched?128:1;
    auto buffer=[&](size_t index){
      return batched&&index>=1&&index<=12?prefill_buffers[index-1]->device_data:d[index]->device_data;
    };
    return record_reserved_sequence(r,staging,transfer,[&]() noexcept {
    auto* host=static_cast<uint8_t*>(staging->host_data);
    auto copy=[&](void* dst,const void* src,uint64_t n,cudaMemcpyKind kind){return runtime_error(cudaMemcpyAsync(dst,src,n,kind,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"decode transfer");};
    auto kernel=[&](cudaError_t e){return runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"decode kernel");};
    auto s=copy(d[19]->device_data,host,metadata,cudaMemcpyHostToDevice);
    if(s==RILEY_CUDA_STATUS_SUCCESS) s=copy(d[0]->device_data,host,4,cudaMemcpyHostToDevice);
    if(s==RILEY_CUDA_STATUS_SUCCESS){
      if(batched) s=kernel(enqueue_decode_embedding_rows(r->stream->stream,w[0]->device_data,
          static_cast<uint8_t*>(d[19]->device_data)+legacy_metadata+8,buffer(1),d[14]->device_data,h/2,vocab,rows));
      else s=kernel(enqueue_decode_embedding(r->stream->stream,w[0]->device_data,d[0]->device_data,d[1]->device_data,d[14]->device_data,h/2,vocab));
    }
    for(uint64_t l=0;l<layers&&s==RILEY_CUDA_STATUS_SUCCESS;++l){
      auto* weights=w+3+9*l;
      auto gemm=[&](size_t j,size_t out){
        if(profile==2&&prefill){
          const int ns[7]={576,192,192,576,1536,1536,576};const int ks[7]={576,576,576,576,576,576,1536};const int chunks[7]={192,192,192,128,0,0,320};
          const size_t inputs[7]={2,2,2,5,2,2,12};const size_t weight_ids[7]={1,2,3,4,6,7,8};
          if(batched&&packed_buffers!=nullptr)return kernel(enqueue_shape_prefill_gemm(r->stream->stream,buffer(inputs[j]),weights[weight_ids[j]]->device_data,buffer(out),ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4,rows));
          return kernel(enqueue_compiled_prefill_gemm_rows(r->stream->stream,buffer(inputs[j]),weights[weight_ids[j]]->device_data,buffer(out),ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4,rows));
        }
        return enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[out],states[l*7+j],error,"decode GEMM");
      };
      if(profile==2){if(l==0)s=kernel(enqueue_compiled_norm_rows(r->stream->stream,buffer(1),nullptr,weights[0]->device_data,nullptr,buffer(2),0,rows));}else s=kernel(enqueue_mlp_norm(r->stream->stream,d[1]->device_data,weights[0]->device_data,d[2]->device_data,h/2,eps[1+2*l],profile));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&packed_decode)
        s=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,packed_buffers[60],packed_states[2*l],error,"packed decode QKV GEMM");
      if(s==RILEY_CUDA_STATUS_SUCCESS&&!packed_decode) s=gemm(0,3);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&!packed_decode) s=gemm(1,6);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&!packed_decode) s=gemm(2,7);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&packed_decode){
        const auto* qkv=static_cast<const uint8_t*>(packed_buffers[60]->device_data);
        s=kernel(enqueue_compiled_packed_decode_rope_attention(r->stream->stream,qkv,qkv+1152,qkv+1536,d[4]->device_data,
            static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,
            static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,
            d[5]->device_data,d[17]->device_data,d[18]->device_data,d[19]->device_data));
      }
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&batched&&packed_buffers!=nullptr)
        s=kernel(enqueue_shape_prefill_rope_kv(r->stream->stream,buffer(3),buffer(6),buffer(7),buffer(4),
          static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,
          d[17]->device_data,d[18]->device_data,d[19]->device_data,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode&&!(batched&&packed_buffers!=nullptr))s=kernel(enqueue_compiled_rope_rows(r->stream->stream,buffer(3),buffer(6),buffer(4),buffer(8),d[17]->device_data,d[18]->device_data,static_cast<uint8_t*>(d[19]->device_data)+4,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_qkv_rope(r->stream->stream,d[3]->device_data,d[6]->device_data,d[4]->device_data,d[8]->device_data,d[17]->device_data,d[18]->device_data,static_cast<uint8_t*>(d[19]->device_data)+4,h/128,k/128,d[17]->byte_len/128));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode&&!(batched&&packed_buffers!=nullptr))s=kernel(enqueue_compiled_kv_write_rows(r->stream->stream,buffer(8),buffer(7),static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,d[19]->device_data,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_decode_kv_attention(r->stream->stream,d[4]->device_data,d[8]->device_data,d[7]->device_data,static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,d[5]->device_data,d[19]->device_data,capacity,h/128,k/128,physical));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode)s=kernel(enqueue_compiled_attention_rows(r->stream->stream,buffer(4),static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,buffer(5),d[19]->device_data,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(3,3);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_residual(r->stream->stream,d[1]->device_data,d[3]->device_data,d[4]->device_data,h/2));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_norm(r->stream->stream,d[4]->device_data,weights[5]->device_data,d[2]->device_data,h/2,eps[2+2*l],profile));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_norm_rows(r->stream->stream,buffer(3),buffer(1),weights[5]->device_data,buffer(11),buffer(2),1,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&packed_decode)
        s=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,packed_buffers[61],packed_states[2*l+1],error,"packed decode gate/up GEMM");
      if(s==RILEY_CUDA_STATUS_SUCCESS&&!packed_decode) s=gemm(4,9);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&!packed_decode) s=gemm(5,10);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2){
        const void* gate=packed_decode?packed_buffers[61]->device_data:buffer(9);
        const void* up=packed_decode?static_cast<uint8_t*>(packed_buffers[61]->device_data)+3072:buffer(10);
        s=kernel(enqueue_compiled_swiglu_rows(r->stream->stream,gate,up,buffer(12),rows));
      }
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_pointwise(r->stream->stream,d[9]->device_data,d[10]->device_data,d[11]->device_data,d[12]->device_data,inter/2));
      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(6,5);
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2)s=kernel(enqueue_compiled_norm_rows(r->stream->stream,buffer(5),buffer(11),(l+1<layers?w[3+9*(l+1)]:w[1])->device_data,buffer(1),buffer(2),2,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_residual(r->stream->stream,d[4]->device_data,d[5]->device_data,d[1]->device_data,h/2));
    }
    if(s==RILEY_CUDA_STATUS_SUCCESS&&profile!=2) s=kernel(enqueue_mlp_norm(r->stream->stream,d[1]->device_data,w[1]->device_data,d[2]->device_data,h/2,eps[0],profile));
    if(s==RILEY_CUDA_STATUS_SUCCESS&&batched)
      s=copy(d[2]->device_data,static_cast<uint8_t*>(buffer(2))+127*h,h,cudaMemcpyDeviceToDevice);
    if(s==RILEY_CUDA_STATUS_SUCCESS) s=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[13],states[layers*7],error,"decode head");
    if(s==RILEY_CUDA_STATUS_SUCCESS) s=kernel(enqueue_decode_argmax(r->stream->stream,d[13]->device_data,d[20]->device_data,vocab));
    if(s==RILEY_CUDA_STATUS_SUCCESS&&publish_logits) s=copy(host+transfer,d[13]->device_data,vocab*2,cudaMemcpyDeviceToHost);
    if(s==RILEY_CUDA_STATUS_SUCCESS) s=copy(host+transfer+logits_output,d[20]->device_data,sizeof(RileyCudaBf16ArgmaxResult),cudaMemcpyDeviceToHost);
    if(s==RILEY_CUDA_STATUS_SUCCESS) s=copy(host+transfer+logits_output+sizeof(RileyCudaBf16ArgmaxResult),d[14]->device_data,sizeof(RileyCudaEmbeddingErrorReport),cudaMemcpyDeviceToHost);
    return s;
    },error);
  };
  if(profile==2){
    status=record_stage(true);
    if(status==RILEY_CUDA_STATUS_SUCCESS){
      // Keep the first pair owned even if the second capture fails. close() must
      // destroy both pairs before releasing any shared parent or plan lease.
      r->prefill_graph=r->graph;r->prefill_exec=r->exec;
      r->graph=nullptr;r->exec=nullptr;
    }
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS) status=record_stage(false);
  if(status!=RILEY_CUDA_STATUS_SUCCESS) r->terminal=true;
  release_states();
  if(status==RILEY_CUDA_STATUS_SUCCESS){r->vllm_smol_p128=profile==2;r->prefill128=prefill_buffers!=nullptr;r->decode_capacity=capacity;r->decode_physical=physical;r->decode_vocab=vocab;r->rope_table_positions=d[17]->byte_len/128;r->rope_payload_position_offset=4;}
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_decode(
    RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,
    RileyCudaDeviceBuffer* const* w,uint64_t weight_count,RileyCudaGemmPlan* const* plans,
    RileyCudaPinnedHostBuffer* staging,const uint64_t* geometry,const float* eps,
    uint32_t profile,uint32_t publish_logits,RileyCudaErrorInfo* error) noexcept {
  return record_decode_impl(r,d,w,weight_count,plans,staging,geometry,eps,profile,publish_logits,nullptr,nullptr,nullptr,error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_decode_prefill128(
    RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,
    RileyCudaDeviceBuffer* const* w,uint64_t weight_count,RileyCudaGemmPlan* const* plans,
    RileyCudaPinnedHostBuffer* staging,const uint64_t* geometry,const float* eps,
    uint32_t profile,uint32_t publish_logits,RileyCudaDeviceBuffer* const* prefill,
    RileyCudaErrorInfo* error) noexcept {
  if(prefill==nullptr){clear_error(error);return reject(error,"P128 scratch parents required",RILEY_CUDA_STATUS_INVALID_ARGUMENT);}
  return record_decode_impl(r,d,w,weight_count,plans,staging,geometry,eps,profile,publish_logits,prefill,nullptr,nullptr,error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_decode_prefill128_packed(
    RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,
    RileyCudaDeviceBuffer* const* w,uint64_t weight_count,RileyCudaGemmPlan* const* plans,
    RileyCudaPinnedHostBuffer* staging,const uint64_t* geometry,const float* eps,
    uint32_t profile,uint32_t publish_logits,RileyCudaDeviceBuffer* const* prefill,
    RileyCudaDeviceBuffer* const* packed,RileyCudaGemmPlan* const* packed_plans,
    RileyCudaErrorInfo* error) noexcept {
  if(prefill==nullptr||packed==nullptr||packed_plans==nullptr){
    clear_error(error);return reject(error,"packed P128 parents and plans required",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  return record_decode_impl(r,d,w,weight_count,plans,staging,geometry,eps,profile,publish_logits,prefill,packed,packed_plans,error);
}

#if defined(RILEY_CUDA_ENABLE_TEST_FAULT_INJECTION)
extern "C" RileyCudaStatus riley_cuda_graph_resources_test_replay_fault(
    RileyCudaGraphResources* r,uint32_t fault,RileyCudaErrorInfo* error) noexcept {
  clear_error(error);auto status=transfer_ready(r,error);
  if(status!=RILEY_CUDA_STATUS_SUCCESS) return status;
  if(r->exec==nullptr||fault<1||fault>2) return reject(error,"invalid test replay fault",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  r->completion_visible=false;r->test_replay_fault=fault;
  return RILEY_CUDA_STATUS_SUCCESS;
}
#endif

#include "graph_multisequence_record.inc"

#include "graph_multisequence_catalog.inc"

// Retains all parents through the existing ledger. V3 result: status/publish at
//0/4, canonical argmax at8, device-echoed identities at16..128, logits at128.
extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(
 RileyCudaGraphResources* r,RileyCudaDeviceBuffer*const* d,RileyCudaDeviceBuffer*const* w,
 uint64_t weight_count,RileyCudaGemmPlan* head,RileyCudaPinnedHostBuffer* staging,
 uint32_t capacity,uint32_t physical,RileyCudaErrorInfo* error) noexcept {
 clear_error(error);auto status=transfer_ready(r,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 if(!d||!w||!head||!staging||(weight_count!=273&&weight_count!=363)||!capacity||capacity>1024||!physical||physical>4096||
    r->graph||r->exec||r->prefill_graph||r->prefill_exec||thread_has_active_graph_capture()||thread_has_active_command_batch()||
    !same_context(staging->owner,r->owner)||!holds_counter(r,&staging->active_uses)||!holds_plan(r,head)||staging->byte_len<196864)
   return reject(error,"V3 recorder lifecycle",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 for(size_t i=0;i<23;++i){if(i==22&&!d[i])continue;
  if(!d[i]||!same_context(d[i]->owner,r->owner)||!holds_counter(r,&d[i]->active_uses))return reject(error,"V3 device parent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<i;++j)if(d[i]==d[j])return reject(error,"V3 mutable alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 for(size_t i=0;i<weight_count;++i){if(!w[i]||!same_context(w[i]->owner,r->owner)||!holds_counter(r,&w[i]->active_uses))return reject(error,"V3 weight parent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<23;++j)if(w[i]==d[j])return reject(error,"V3 weight/mutable alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 const uint64_t widths[12]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};
 for(size_t i=0;i<12;++i)if(d[i]->byte_len!=(i==7?std::max<uint64_t>(capacity*widths[i],9*4096*4):capacity*widths[i]))return reject(error,"V3 scratch extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(d[12]->byte_len!=uint64_t(30)*physical*16*384||d[13]->byte_len!=d[12]->byte_len||!d[14]->byte_len||d[14]->byte_len%128||d[14]->byte_len>8192*128||d[15]->byte_len!=d[14]->byte_len||d[16]->byte_len!=17536||d[17]->byte_len!=1152||d[18]->byte_len!=128||d[19]->byte_len!=4||d[20]->byte_len!=98304||d[21]->byte_len!=sizeof(RileyCudaBf16ArgmaxResult))return reject(error,"V3 device extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(w[0]->byte_len!=56623104||w[1]->byte_len!=1152||w[2]->byte_len!=56623104)return reject(error,"V3 global weights",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 const uint64_t sizes[9]={1152,663552,221184,221184,663552,1152,1769472,1769472,1769472};
 for(size_t i=3;i<273;++i)if(w[i]->byte_len!=sizes[(i-3)%9])return reject(error,"V3 layer weights",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(weight_count==363)for(size_t i=273;i<363;++i){
  if(w[i]->byte_len!=1769472)return reject(error,"V3 tiled weight extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<i;++j)if(w[i]==w[j])return reject(error,"V3 tiled weight alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 RileyCudaCanonicalGemmBf16GraphState state{};
 status=bind_reserved_gemm_state(head,r->stream,d[17],w[2],d[20],d[22],&state,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 void* scratch[12];const void* weights[273];for(size_t i=0;i<12;++i)scratch[i]=d[i]->device_data;for(size_t i=0;i<273;++i)weights[i]=w[i]->device_data;
 constexpr uint64_t transfer=98432;
 std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,128);
 auto record_shape=[&](uint32_t row_capacity) noexcept {
  if(weight_count==363)for(size_t l=0;l<30;++l)for(size_t j=0;j<3;++j)weights[3+l*9+6+j]=w[273+l*3+j]->device_data;
  return record_reserved_sequence(r,staging,transfer,[&]() noexcept {
  auto* host=static_cast<uint8_t*>(staging->host_data);
  auto copy=[&](void* a,const void* b,uint64_t n,cudaMemcpyKind kind){return runtime_error(cudaMemcpyAsync(a,b,n,kind,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"V3 transfer");};
  auto result=copy(d[16]->device_data,host,17536,cudaMemcpyHostToDevice);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_prefill_model(r->stream->stream,scratch,weights,d[16]->device_data,d[12]->device_data,d[13]->device_data,d[14]->device_data,d[15]->device_data,d[17]->device_data,static_cast<uint32_t*>(d[18]->device_data),static_cast<uint32_t*>(d[19]->device_data),row_capacity,physical,weight_count==363),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 model");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[20],state,error,"V3 head");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_decode_argmax(r->stream->stream,d[20]->device_data,d[21]->device_data,49152),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 argmax");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_result_header(r->stream->stream,d[16]->device_data,d[18]->device_data,d[19]->device_data,d[21]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 completion header");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[18]->device_data,128,cudaMemcpyDeviceToHost);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer+128,d[20]->device_data,98304,cudaMemcpyDeviceToHost);
  return result;
 },error);};
 status=record_shape(capacity);
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;status=record_shape(1);}
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->v3_prefill_capacity=capacity;r->v3_prefill_physical=physical;r->v3_prefill_context=std::min<uint64_t>(4096,d[14]->byte_len/128);}
 return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_shared(
 RileyCudaGraphResources* r,RileyCudaDeviceBuffer*const* d,RileyCudaDeviceBuffer*const* w,
 uint64_t weight_count,RileyCudaGemmPlan* head,RileyCudaGemmPlan* shared_head,RileyCudaPinnedHostBuffer* staging,
 uint32_t capacity,uint32_t physical,RileyCudaErrorInfo* error) noexcept {
 clear_error(error);auto status=transfer_ready(r,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 if(!d||!w||!head||!shared_head||!holds_plan(r,shared_head)||capacity<8||!staging||(weight_count!=273&&weight_count!=363)||!capacity||capacity>1024||!physical||physical>4096||
    r->graph||r->exec||r->prefill_graph||r->prefill_exec||thread_has_active_graph_capture()||thread_has_active_command_batch()||
    !same_context(staging->owner,r->owner)||!holds_counter(r,&staging->active_uses)||!holds_plan(r,head)||staging->byte_len<1574912)
   return reject(error,"V3 recorder lifecycle",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 for(size_t i=0;i<26;++i){if(i==22&&!d[i])continue;
  if(!d[i]||!same_context(d[i]->owner,r->owner)||!holds_counter(r,&d[i]->active_uses))return reject(error,"V3 device parent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<i;++j)if(d[i]==d[j])return reject(error,"V3 mutable alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 for(size_t i=0;i<weight_count;++i){if(!w[i]||!same_context(w[i]->owner,r->owner)||!holds_counter(r,&w[i]->active_uses))return reject(error,"V3 weight parent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<26;++j)if(w[i]==d[j])return reject(error,"V3 weight/mutable alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 const uint64_t widths[12]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};
 for(size_t i=0;i<12;++i)if(d[i]->byte_len!=(i==7?std::max<uint64_t>(capacity*widths[i],8*9*4096*4):capacity*widths[i]))return reject(error,"V3 scratch extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(d[12]->byte_len!=uint64_t(30)*physical*16*384||d[13]->byte_len!=d[12]->byte_len||!d[14]->byte_len||d[14]->byte_len%128||d[14]->byte_len>8192*128||d[15]->byte_len!=d[14]->byte_len||d[16]->byte_len!=17536||d[17]->byte_len!=1152||d[18]->byte_len!=128||d[19]->byte_len!=4||d[20]->byte_len!=98304||d[21]->byte_len!=sizeof(RileyCudaBf16ArgmaxResult))return reject(error,"V3 device extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(w[0]->byte_len!=56623104||w[1]->byte_len!=1152||w[2]->byte_len!=56623104)return reject(error,"V3 global weights",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 const uint64_t sizes[9]={1152,663552,221184,221184,663552,1152,1769472,1769472,1769472};
 for(size_t i=3;i<273;++i)if(w[i]->byte_len!=sizes[(i-3)%9])return reject(error,"V3 layer weights",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(weight_count==363)for(size_t i=273;i<363;++i){
  if(w[i]->byte_len!=1769472)return reject(error,"V3 tiled weight extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<i;++j)if(w[i]==w[j])return reject(error,"V3 tiled weight alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 if(d[23]->byte_len!=8*1152||d[24]->byte_len!=8*98304||d[25]->byte_len!=787456)return reject(error,"V3 shared output extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 RileyCudaCanonicalGemmBf16GraphState shared_state{};
 status=bind_reserved_shared_row_gemm_state(shared_head,r->stream,d[23],w[2],d[24],8,49152,576,&shared_state,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 RileyCudaCanonicalGemmBf16GraphState state{};
 status=bind_reserved_gemm_state(head,r->stream,d[17],w[2],d[20],d[22],&state,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 void* scratch[12];const void* weights[363];for(size_t i=0;i<12;++i)scratch[i]=d[i]->device_data;for(size_t i=0;i<weight_count;++i)weights[i]=w[i]->device_data;
 const void* prefill_weights[273];for(size_t i=0;i<273;++i)prefill_weights[i]=weights[i];
 if(weight_count==363)for(size_t l=0;l<30;++l)for(size_t j=0;j<3;++j)prefill_weights[3+l*9+6+j]=weights[273+l*3+j];
 constexpr uint64_t transfer=787456;
 std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,128);
 auto record_shape=[&](uint32_t row_capacity) noexcept {
  return record_reserved_sequence(r,staging,transfer,[&]() noexcept {
  auto* host=static_cast<uint8_t*>(staging->host_data);
  auto copy=[&](void* a,const void* b,uint64_t n,cudaMemcpyKind kind){return runtime_error(cudaMemcpyAsync(a,b,n,kind,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"V3 transfer");};
  auto result=copy(d[16]->device_data,host,17536,cudaMemcpyHostToDevice);
  if(row_capacity==1){
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_shared_model(r->stream->stream,scratch,weights,d[16]->device_data,d[12]->device_data,d[13]->device_data,d[14]->device_data,d[15]->device_data,static_cast<uint32_t*>(d[18]->device_data),physical,std::min<uint64_t>(4096,d[14]->byte_len/128),weight_count==363),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 shared model");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(d[23]->device_data,d[1]->device_data,8*1152,cudaMemcpyDeviceToDevice);
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[24],shared_state,error,"V3 shared head");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_shared_result(r->stream->stream,d[16]->device_data,d[24]->device_data,d[18]->device_data,d[25]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 shared completion");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,transfer,cudaMemcpyDeviceToHost);
   return result;
  }
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(cudaMemsetAsync(d[25]->device_data,0,transfer,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"V3 clear inactive output");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_prefill_model(r->stream->stream,scratch,prefill_weights,d[16]->device_data,d[12]->device_data,d[13]->device_data,d[14]->device_data,d[15]->device_data,d[17]->device_data,static_cast<uint32_t*>(d[18]->device_data),static_cast<uint32_t*>(d[19]->device_data),row_capacity,physical,weight_count==363),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 model");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[20],state,error,"V3 head");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_decode_argmax(r->stream->stream,d[20]->device_data,d[21]->device_data,49152),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 argmax");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_result_header(r->stream->stream,d[16]->device_data,d[18]->device_data,d[19]->device_data,d[21]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 completion header");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(d[25]->device_data,d[18]->device_data,128,cudaMemcpyDeviceToDevice);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(static_cast<uint8_t*>(d[25]->device_data)+128,d[20]->device_data,98304,cudaMemcpyDeviceToDevice);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,transfer,cudaMemcpyDeviceToHost);
  return result;
 },error);};
 status=record_shape(capacity);
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;status=record_shape(1);}
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->v3_shared=true;r->v3_prefill_capacity=capacity;r->v3_prefill_physical=physical;r->v3_prefill_context=std::min<uint64_t>(4096,d[14]->byte_len/128);}
 return status;
}
