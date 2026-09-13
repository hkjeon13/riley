// Included after the aggregate ledger helpers. No separate context or device
// allocation: slot staging is supplied by and retained in the parent's ledger.
static RileyCudaStatus buffered_transfer_ready(RileyCudaGraphResources* r, RileyCudaErrorInfo* error) noexcept {
  if (!r || r->thread != native_thread_token() || r->terminal || !r->buffered_ready ||
      r->owner->restoration_failed.load(std::memory_order_acquire))
    return reject(error, "buffered owner unavailable");
  return RILEY_CUDA_STATUS_SUCCESS;
}

static cudaError_t clone_buffered_graph(RileyCudaGraphResources* r,
    RileyCudaGraphResources::BufferedSlot& slot, unsigned index, cudaGraph_t source) noexcept {
  if (!source) return cudaSuccess;
  if (slot.staging == r->input && r->input == r->output) {
    // Model staging already contains disjoint input/output regions. Slot zero
    // can reuse the original DAG and staging without a third pinned buffer.
    slot.borrows_original_graphs = true;
    cudaGraphExec_t originals[] = {r->exec,r->prefill_exec,r->compact_exec[0],r->compact_exec[1]};
    slot.exec[index] = originals[index];
    return cudaSuccess;
  }
  auto result = cudaGraphClone(&slot.graph[index], source);
  size_t count = 0;
  if (result == cudaSuccess) result = cudaGraphGetNodes(slot.graph[index], nullptr, &count);
  // malloc is fallible without exceptions crossing the C ABI.
  auto* nodes = static_cast<cudaGraphNode_t*>(std::calloc(count, sizeof(cudaGraphNode_t)));
  if (!nodes) return cudaErrorMemoryAllocation;
  if (result == cudaSuccess) result = cudaGraphGetNodes(slot.graph[index], nodes, &count);
  unsigned inputs = 0, outputs = 0;
  for (size_t i = 0; i < count && result == cudaSuccess; ++i) {
    cudaGraphNodeType type{};
    result = cudaGraphNodeGetType(nodes[i], &type);
    if (result != cudaSuccess) break;
    if (type == cudaGraphNodeTypeGraph) { result = cudaErrorNotSupported; break; }
    if (type != cudaGraphNodeTypeMemcpy) continue;
    cudaMemcpy3DParms params{};
    result = cudaGraphMemcpyNodeGetParams(nodes[i], &params);
    if (result != cudaSuccess) break;
    auto relocate = [&](void*& pointer, RileyCudaPinnedHostBuffer* parent, uint64_t base) {
      const auto address = reinterpret_cast<uintptr_t>(pointer);
      const auto start = reinterpret_cast<uintptr_t>(parent->host_data);
      if (address < start || address - start >= parent->byte_len) return false;
      pointer = static_cast<uint8_t*>(slot.staging->host_data) + base + (address - start);
      return true;
    };
    bool changed = false;
    if (params.kind == cudaMemcpyHostToDevice) {
      changed = relocate(params.srcPtr.ptr, r->input, 0); inputs += changed;
      if (!changed) { result = cudaErrorNotSupported; break; }
    } else if (params.kind == cudaMemcpyDeviceToHost) {
      changed = relocate(params.dstPtr.ptr, r->output, r->input == r->output ? 0 : r->buffered_output_offset);
      outputs += changed;
      if (!changed) { result = cudaErrorNotSupported; break; }
    } else if (params.kind != cudaMemcpyDeviceToDevice) {
      result = cudaErrorNotSupported; break;
    }
    if (changed) result = cudaGraphMemcpyNodeSetParams(nodes[i], &params);
  }
  std::free(nodes);
  if (result == cudaSuccess && (!inputs || !outputs)) result = cudaErrorNotSupported;
  if (result == cudaSuccess) result = cudaGraphInstantiate(&slot.exec[index], slot.graph[index], 0);
  return result;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_prepare_buffered_transfers(
    RileyCudaGraphResources* r, RileyCudaPinnedHostBuffer* first,
    RileyCudaPinnedHostBuffer* second, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (!r->graph || !r->exec || !r->input || !r->output || r->catalog_sealed || r->catalog[0].graph ||
      r->bound_attention || !first || !second || first == second ||
      ((first == r->input || first == r->output) && r->input != r->output) || second == r->input || second == r->output ||
      !same_context(first->owner, r->owner) || !same_context(second->owner, r->owner) ||
      !holds_counter(r, &first->active_uses) || !holds_counter(r, &second->active_uses))
    return reject(error, "buffered slots or graph state invalid", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  const bool shared = r->input == r->output;
  if (!shared && r->input->byte_len > UINT64_MAX - r->output->byte_len)
    return reject(error, "buffered staging extent overflow", RILEY_CUDA_STATUS_OUT_OF_RANGE);
  const auto bytes = shared ? r->input->byte_len : r->input->byte_len + r->output->byte_len;
  if (bytes > SIZE_MAX || first->byte_len < bytes || second->byte_len < bytes)
    return reject(error, "buffered staging too small", RILEY_CUDA_STATUS_OUT_OF_RANGE);
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain busy");
  CurrentContext scope(r->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  const uint64_t generation = next_graph_capture_id();
  if (!generation || generation > UINT32_MAX)
    return scope.leave(reject(error, "buffered owner generation exhausted"), error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  r->next_buffered_ticket = (generation << 32) | 1;
  r->terminal = true;
  r->buffered_output_offset = shared ? r->output_byte_offset : r->input->byte_len;
  r->buffered[0].staging = first; r->buffered[1].staging = second;
  cudaGraph_t graphs[] = {r->graph, r->prefill_graph, r->compact_graph[0], r->compact_graph[1]};
  auto result = cudaSuccess;
  for (auto& slot : r->buffered) {
    for (unsigned i = 0; i < 4 && result == cudaSuccess; ++i)
      result = clone_buffered_graph(r, slot, i, graphs[i]);
    if (result == cudaSuccess) result = cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming);
  }
  status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE, kTransfer);
  if (status == RILEY_CUDA_STATUS_SUCCESS) { r->buffered_ready = true; r->terminal = false; r->catalog_sealed = true; }
  return status;
}

static RileyCudaStatus launch_buffered_transfer(RileyCudaGraphResources* r,
    const uint8_t* source, uint64_t bytes, cudaGraphExec_t selected,
    uint64_t* ticket, RileyCudaErrorInfo* error) noexcept {
  if ((r->next_buffered_ticket & UINT32_MAX) == UINT32_MAX) return reject(error, "buffered ticket exhausted");
  auto& slot = r->buffered[(r->next_buffered_ticket - 1) % 2];
  if (slot.ticket) return reject(error, "buffered slot must be consumed before reuse");
  const unsigned index = selected == r->exec ? 0 : selected == r->prefill_exec ? 1 : selected == r->compact_exec[0] ? 2 : 3;
  if (!slot.exec[index]) return reject(error, "buffered stage unavailable");
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain busy");
  CurrentContext scope(r->owner);
  auto status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kTransfer);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  std::memmove(slot.staging->host_data, source, static_cast<size_t>(bytes));
  slot.ticket = r->next_buffered_ticket++;
  slot.result_bytes = r->last_compact ? r->variable_rows * 128 : r->transfer_bytes;
  r->completion_unknown = true;
  auto result = cudaGraphLaunch(slot.exec[index], r->stream->stream);
  if (result == cudaSuccess) result = cudaEventRecord(slot.event, r->stream->stream);
  if (result == cudaSuccess) slot.pending = true;
  else {
    r->terminal = true;
    const auto drained = cudaStreamSynchronize(r->stream->stream);
    for (auto& other : r->buffered) other.pending = false;
    r->completion_unknown = drained != cudaSuccess;
  }
  status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kTransfer);
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kTransfer);
  if (status == RILEY_CUDA_STATUS_SUCCESS) *ticket = slot.ticket;
  else r->terminal = true;
  return status;
}

static RileyCudaStatus finish_buffered_transfer(RileyCudaGraphResources* r, unsigned index,
    bool wait, uint32_t* ready, RileyCudaErrorInfo* error) noexcept {
  *ready = 0;
  auto& slot = r->buffered[index];
  if (!slot.pending) { *ready = 1; return RILEY_CUDA_STATUS_SUCCESS; }
  if (r->owner->restoration_failed.load(std::memory_order_acquire)) return reject(error, "context restoration failed");
  CaptureDomainControlLease admission(r->owner->capture_domain);
  if (!admission.active()) return reject(error, "capture domain busy");
  CurrentContext scope(r->owner);
  auto status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kTransfer);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  const auto result = wait ? cudaEventSynchronize(slot.event) : cudaEventQuery(slot.event);
  if (result == cudaErrorNotReady && !wait)
    return scope.leave(RILEY_CUDA_STATUS_SUCCESS, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kTransfer);
  if (result == cudaSuccess) {
    slot.pending = false;
    r->completion_unknown = r->buffered[0].pending || r->buffered[1].pending;
  } else {
    r->terminal = true;
    r->completion_unknown = true;
    // Do not retry an ambiguous completion during close or let another slot's
    // successful event erase this failure. All parents stay quarantined.
    for (auto& other : r->buffered) other.pending = false;
  }
  status = runtime_error(result, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kTransfer);
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kTransfer);
  if (status == RILEY_CUDA_STATUS_SUCCESS) *ready = 1;
  else r->terminal = true;
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_query_buffered_transfer(
    RileyCudaGraphResources* r, uint64_t ticket, uint32_t wait,
    uint32_t* ready, RileyCudaErrorInfo* error) noexcept {
  clear_error(error); if (ready) *ready = 0;
  auto status = buffered_transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (!ready || !ticket || wait > 1 || r->buffered[(ticket - 1) % 2].ticket != ticket)
    return reject(error, "invalid or consumed buffered ticket", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  return finish_buffered_transfer(r, (ticket - 1) % 2, wait != 0, ready, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_read_buffered_transfer(
    RileyCudaGraphResources* r, uint64_t ticket, uint8_t* destination,
    uint64_t bytes, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status = buffered_transfer_ready(r, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
  if (!ticket || !destination) return reject(error, "null buffered read argument", RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  auto& slot = r->buffered[(ticket - 1) % 2];
  if (slot.ticket != ticket || slot.pending || bytes != slot.result_bytes)
    return reject(error, "buffered result unavailable or wrong extent");
  std::memmove(destination, static_cast<uint8_t*>(slot.staging->host_data) + r->buffered_output_offset, static_cast<size_t>(bytes));
  slot.ticket = 0; slot.result_bytes = 0;
  return RILEY_CUDA_STATUS_SUCCESS;
}

// Caller has entered the owning context and established device quiescence.
static RileyCudaStatus destroy_buffered_transfers(RileyCudaGraphResources* r, RileyCudaErrorInfo* error) noexcept {
  auto status = RILEY_CUDA_STATUS_SUCCESS;
  for (auto& slot : r->buffered) {
    for (unsigned i = 0; i < 4; ++i) {
      if (status == RILEY_CUDA_STATUS_SUCCESS && slot.exec[i] && !slot.borrows_original_graphs) {
        status = runtime_error(cudaGraphExecDestroy(slot.exec[i]), error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
        if (status == RILEY_CUDA_STATUS_SUCCESS) slot.exec[i] = nullptr;
      }
      if (status == RILEY_CUDA_STATUS_SUCCESS && slot.graph[i]) {
        status = runtime_error(cudaGraphDestroy(slot.graph[i]), error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
        if (status == RILEY_CUDA_STATUS_SUCCESS) slot.graph[i] = nullptr;
      }
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS && slot.event) {
      status = runtime_error(cudaEventDestroy(slot.event), error, RILEY_CUDA_ERROR_STAGE_CLOSE, kClose);
      if (status == RILEY_CUDA_STATUS_SUCCESS) slot.event = nullptr;
    }
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) r->completion_unknown = true;
  return status;
}
