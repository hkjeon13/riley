// Fixed SmolLM2 strided decode DAG. Included in the aggregate owner TU so all
// parent membership, capture rollback and graph destruction use one ledger.
static RileyCudaStatus record_multisequence_decode_impl(
    RileyCudaGraphResources* r, RileyCudaDeviceBuffer* const* d, uint64_t device_count,
    RileyCudaDeviceBuffer* const* w, uint64_t weight_count,
    RileyCudaGemmPlan* const* plans, uint64_t plan_count,
    RileyCudaPinnedHostBuffer* staging, uint32_t bucket, uint32_t physical,
    uint32_t full, bool shared_rows, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  auto status=transfer_ready(r,error);
  if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  if(!d||!w||!plans||!staging||device_count!=18||weight_count!=333||plan_count!=5||
      (bucket!=2&&bucket!=4&&bucket!=8)||physical<10||physical>4096||full>1||r->graph||r->exec||
      r->prefill_graph||r->prefill_exec||thread_has_active_graph_capture()||thread_has_active_command_batch()||
      !same_context(staging->owner,r->owner)||!holds_counter(r,&staging->active_uses))
    return reject(error,"multi decode catalog or parents invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  const uint64_t result=1152+(full?uint64_t(bucket)*98304:0),transfer=result>1792?result:1792;
  const uint64_t sizes[18]={1152ULL*bucket,1152ULL*bucket,1920ULL*bucket,
      1152ULL*bucket,1152ULL*bucket,1152ULL*bucket,2304ULL*bucket,6144ULL*bucket,
      3072ULL*bucket,1152ULL*bucket,98304ULL*bucket,1792,result,8ULL*bucket,
      384ULL*16*physical*30,384ULL*16*physical*30,160*128,160*128};
  if(staging->byte_len!=2*transfer)return reject(error,"multi staging extent invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t i=0;i<18;++i){
    if(!d[i]||!same_context(d[i]->owner,r->owner)||!holds_counter(r,&d[i]->active_uses)||(i<16?(i==12?(d[i]->byte_len!=sizes[i]&&d[i]->byte_len!=1152+uint64_t(bucket)*98304):d[i]->byte_len!=sizes[i]):(d[i]->byte_len<160*128||d[i]->byte_len%128!=0)))
      return reject(error,"multi device geometry or membership invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(size_t j=0;j<i;++j)if(d[i]==d[j])return reject(error,"multi device alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  if(d[16]->byte_len!=d[17]->byte_len)return reject(error,"multi RoPE table geometry differs",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t i=0;i<333;++i){
    if(!w[i]||!same_context(w[i]->owner,r->owner)||!holds_counter(r,&w[i]->active_uses))
      return reject(error,"multi weight membership invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    for(size_t j=0;j<18;++j)if(w[i]==d[j])return reject(error,"multi weight scratch alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
  if(w[0]->byte_len!=49152ULL*1152||w[1]->byte_len!=1152||w[2]->byte_len!=49152ULL*1152)
    return reject(error,"multi embedding/head geometry invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t i=0;i<5;++i)if(!plans[i]||!holds_plan(r,plans[i]))
    return reject(error,"multi plan membership invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  using State=RileyCudaCanonicalGemmBf16GraphState;
  State states[121]{};
  auto bind=[&](size_t index,size_t plan,size_t input,size_t weight,size_t output,uint64_t n,uint64_t k){
    if(shared_rows && (plan==0||plan==1||plan==4))
      return bind_reserved_shared_row_gemm_state(plans[plan],r->stream,d[input],w[weight],d[output],bucket,n,k,&states[index],error);
    return bind_reserved_strided_gemm_state(plans[plan],r->stream,d[input],w[weight],d[output],bucket,n,k,k,n,&states[index],error);
  };
  for(size_t l=0;l<30&&status==RILEY_CUDA_STATUS_SUCCESS;++l){
    if(w[3+9*l]->byte_len!=1152||w[8+9*l]->byte_len!=1152)
      return reject(error,"multi norm geometry invalid",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    status=bind(4*l,0,1,273+2*l,2,960,576);
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=bind(4*l+1,2,4,7+9*l,5,576,576);
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=bind(4*l+2,1,1,274+2*l,7,3072,576);
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=bind(4*l+3,3,8,11+9*l,9,576,1536);
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=bind(120,4,1,2,10,49152,576);
  if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  std::memset(static_cast<uint8_t*>(staging->host_data)+transfer,0,transfer);
  status=record_reserved_sequence(r,staging,transfer,[&]() noexcept {
    const auto stream=r->stream->stream;
    auto kernel=[&](cudaError_t e){return runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"multi decode kernel");};
    auto gemm=[&](size_t state,size_t output){return enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[output],states[state],error,"multi decode GEMM");};
    auto s=runtime_error(cudaMemcpyAsync(d[11]->device_data,staging->host_data,1792,cudaMemcpyHostToDevice,stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"multi packet upload");
    if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_multi_embedding(stream,w[0]->device_data,d[11]->device_data,d[0]->device_data,bucket));
    if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_compiled_norm_rows(stream,d[0]->device_data,nullptr,w[3]->device_data,nullptr,d[1]->device_data,0,bucket));
    for(size_t l=0;l<30&&s==RILEY_CUDA_STATUS_SUCCESS;++l){
      auto* keys=static_cast<uint8_t*>(d[14]->device_data)+l*384*16*physical;
      auto* values=static_cast<uint8_t*>(d[15]->device_data)+l*384*16*physical;
      s=gemm(4*l,2);
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_compiled_packed_multisequence_rope_attention(stream,d[2]->device_data,d[3]->device_data,keys,values,d[4]->device_data,d[16]->device_data,d[17]->device_data,d[11]->device_data,bucket));
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=gemm(4*l+1,5);
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_compiled_norm_rows(stream,d[5]->device_data,d[0]->device_data,w[8+9*l]->device_data,d[6]->device_data,d[1]->device_data,1,bucket));
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=gemm(4*l+2,7);
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(riley_multisequence_precise::enqueue_swiglu_rows(stream,d[7]->device_data,d[8]->device_data,d[11]->device_data,bucket));
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=gemm(4*l+3,9);
      if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_compiled_norm_rows(stream,d[9]->device_data,d[6]->device_data,w[l+1<30?3+9*(l+1):1]->device_data,d[0]->device_data,d[1]->device_data,2,bucket));
    }
    if(s==RILEY_CUDA_STATUS_SUCCESS)s=gemm(120,10);
    if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_multi_argmax(stream,d[10]->device_data,d[13]->device_data,bucket));
    if(s==RILEY_CUDA_STATUS_SUCCESS)s=kernel(enqueue_multi_completion(stream,d[11]->device_data,d[13]->device_data,d[10]->device_data,d[12]->device_data,bucket,full));
    if(s==RILEY_CUDA_STATUS_SUCCESS)s=runtime_error(cudaMemcpyAsync(static_cast<uint8_t*>(staging->host_data)+transfer,d[12]->device_data,result,cudaMemcpyDeviceToHost,stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"multi completion download");
    return s;
  },error);
  if(status==RILEY_CUDA_STATUS_SUCCESS){r->multi_bucket=bucket;r->multi_physical=physical;r->multi_full=full;}
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_resources_record_multisequence_decode(RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,uint64_t dc,RileyCudaDeviceBuffer* const* w,uint64_t wc,RileyCudaGemmPlan* const* p,uint64_t pc,RileyCudaPinnedHostBuffer* h,uint32_t b,uint32_t ph,uint32_t f,RileyCudaErrorInfo* e) noexcept { return record_multisequence_decode_impl(r,d,dc,w,wc,p,pc,h,b,ph,f,false,e); }
extern "C" RileyCudaStatus riley_cuda_graph_resources_record_shared_multisequence_decode(RileyCudaGraphResources* r,RileyCudaDeviceBuffer* const* d,uint64_t dc,RileyCudaDeviceBuffer* const* w,uint64_t wc,RileyCudaGemmPlan* const* p,uint64_t pc,RileyCudaPinnedHostBuffer* h,uint32_t b,uint32_t ph,uint32_t f,RileyCudaErrorInfo* e) noexcept { return record_multisequence_decode_impl(r,d,dc,w,wc,p,pc,h,b,ph,f,true,e); }
