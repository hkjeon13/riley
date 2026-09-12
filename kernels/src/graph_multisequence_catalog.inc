namespace {
// These fields describe the selected DAG, never ownership of its parent leases.
// Lifecycle state (terminal/completion_unknown/scoped_capture) is deliberately
// not restored: an ambiguous failure remains terminal across every entry.
struct SavedGraphBinding {
  cudaGraph_t graph,prefill_graph;
  cudaGraphExec_t exec,prefill_exec;
  RileyCudaPinnedHostBuffer *input,*output;
  uint64_t transfer,offset,rope_positions,rope_offset,kv_logical,kv_valid,capacity,physical,vocab;
  uint32_t attention_physical,multi_bucket,multi_physical,multi_full;
  bool kv_bound,attention_bound,vllm,prefill128;
  explicit SavedGraphBinding(RileyCudaGraphResources* r):
    graph(r->graph),prefill_graph(r->prefill_graph),exec(r->exec),prefill_exec(r->prefill_exec),
    input(r->input),output(r->output),transfer(r->transfer_bytes),offset(r->output_byte_offset),
    rope_positions(r->rope_table_positions),rope_offset(r->rope_payload_position_offset),
    kv_logical(r->kv_logical_block),kv_valid(r->kv_valid_tokens),capacity(r->decode_capacity),
    physical(r->decode_physical),vocab(r->decode_vocab),attention_physical(r->attention_physical_block),
    multi_bucket(r->multi_bucket),multi_physical(r->multi_physical),multi_full(r->multi_full),
    kv_bound(r->kv_write_bound),attention_bound(r->bound_attention),vllm(r->vllm_smol_p128),prefill128(r->prefill128){}
  void restore(RileyCudaGraphResources* r)const noexcept{
    r->graph=graph;r->exec=exec;r->prefill_graph=prefill_graph;r->prefill_exec=prefill_exec;
    r->input=input;r->output=output;r->transfer_bytes=transfer;r->output_byte_offset=offset;
    r->rope_table_positions=rope_positions;r->rope_payload_position_offset=rope_offset;
    r->kv_logical_block=kv_logical;r->kv_valid_tokens=kv_valid;r->decode_capacity=capacity;
    r->decode_physical=physical;r->decode_vocab=vocab;r->attention_physical_block=attention_physical;
    r->multi_bucket=multi_bucket;r->multi_physical=multi_physical;r->multi_full=multi_full;
    r->kv_write_bound=kv_bound;r->bound_attention=attention_bound;r->vllm_smol_p128=vllm;r->prefill128=prefill128;
  }
};
void select_catalog_binding(RileyCudaGraphResources* r,const RileyCudaGraphResources::CatalogEntry& e)noexcept{
  r->graph=e.graph;r->exec=e.exec;r->prefill_graph=nullptr;r->prefill_exec=nullptr;
  r->input=e.staging;r->output=e.staging;r->transfer_bytes=e.transfer;r->output_byte_offset=e.transfer;
  r->rope_table_positions=0;r->rope_payload_position_offset=0;r->kv_logical_block=0;r->kv_valid_tokens=0;
  r->decode_capacity=0;r->decode_physical=0;r->decode_vocab=0;r->attention_physical_block=0;
  r->multi_bucket=e.bucket;r->multi_physical=e.physical;r->multi_full=e.full;
  r->kv_write_bound=false;r->bound_attention=false;r->vllm_smol_p128=false;r->prefill128=false;
}
}
extern "C" RileyCudaStatus riley_cuda_graph_resources_append_multisequence_decode(
    RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,uint64_t device_count,
    RileyCudaDeviceBuffer* const* w,uint64_t weight_count,RileyCudaGemmPlan* const* plans,uint64_t plan_count,
    RileyCudaPinnedHostBuffer* staging,uint32_t bucket,uint32_t physical,uint32_t full,RileyCudaErrorInfo* error)noexcept{
  clear_error(error);auto status=transfer_ready(r,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  // Cold-only: never invalidate an already published completion to grow a catalog.
  if((bucket!=2&&bucket!=4&&bucket!=8)||full>1||!r->exec||r->catalog_sealed||r->completion_visible||r->multi_bucket==bucket)
    return reject(error,"catalog append requires a distinct cold decode bucket",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  auto& entry=r->catalog[(full?0:3)+(bucket==2?0:bucket==4?1:2)];
  if(entry.graph||entry.exec)return reject(error,"catalog entry already recorded",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  SavedGraphBinding saved(r);select_catalog_binding(r,{});
  status=riley_cuda_graph_resources_record_multisequence_decode(r,d,device_count,w,weight_count,plans,plan_count,staging,bucket,physical,full,error);
  // Retain even a partially constructed graph for aggregate close. Restoring
  // the main binding must never discard native ownership after capture failure.
  entry={r->graph,r->exec,r->input,r->transfer_bytes,r->multi_bucket,r->multi_physical,r->multi_full};
  saved.restore(r);
  return status;
}
extern "C" RileyCudaStatus riley_cuda_graph_resources_replay_catalog(
    RileyCudaGraphResources* r,uint32_t index,const uint8_t* source,uint64_t bytes,RileyCudaErrorInfo* error)noexcept{
  clear_error(error);auto status=transfer_ready(r,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  r->completion_visible=false;
  if(index==0)return riley_cuda_graph_resources_replay_transfer(r,source,bytes,error);
  if(index>6||!r->catalog[index-1].exec)return reject(error,"catalog entry absent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  SavedGraphBinding saved(r);select_catalog_binding(r,r->catalog[index-1]);
  status=riley_cuda_graph_resources_replay_transfer(r,source,bytes,error);
  saved.restore(r);r->last_catalog=index;
  return status;
}
extern "C" RileyCudaStatus riley_cuda_graph_resources_read_catalog(
    RileyCudaGraphResources* r,uint32_t index,uint8_t* output,uint64_t bytes,RileyCudaErrorInfo* error)noexcept{
  clear_error(error);auto status=transfer_ready(r,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  if(index!=r->last_catalog||index>6||!r->completion_visible)return reject(error,"catalog completion is stale",RILEY_CUDA_STATUS_INVALID_STATE);
  if(index==0)return riley_cuda_graph_resources_read_transfer(r,output,bytes,error);
  SavedGraphBinding saved(r);select_catalog_binding(r,r->catalog[index-1]);r->last_catalog=0;
  status=riley_cuda_graph_resources_read_transfer(r,output,bytes,error);
  saved.restore(r);r->last_catalog=index;
  return status;
}
