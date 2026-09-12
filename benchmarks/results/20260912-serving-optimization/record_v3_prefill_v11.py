from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();s+='''
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v3_prefill_model(cudaStream_t,void*const*,const void*const*,const void*,void*,void*,const void*,const void*,void*,uint32_t*,uint32_t*,uint32_t,uint32_t) noexcept;
}
''';p.write_text(s)
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v3_prefill_model(cudaStream_t s,void*const* d,const void*const* w,const void* m,void* k,void* v,const void* c,const void* sn,void* selected,uint32_t* status,uint32_t* publish,uint32_t rows,uint32_t physical) noexcept {
 return enqueue_v3_prefill_model(s,d,w,m,k,v,c,sn,selected,status,publish,rows,physical);
}
}
''';p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text();s=s.replace('#include "ffi_internal.hpp"','#include "ffi_internal.hpp"\n#include "prefill_shape_packet.hpp"',1)
s=s.replace('  uint32_t multi_bucket=0, multi_physical=0, multi_full=0;','  uint32_t multi_bucket=0, multi_physical=0, multi_full=0;\n  uint32_t v3_prefill_capacity=0,v3_prefill_physical=0,v3_prefill_context=0;')
s=s.replace('bytes != r->transfer_bytes)\n    return validation_error', 'bytes != (r->v3_prefill_capacity?17536:r->transfer_bytes))\n    return validation_error',1)
needle='  auto selected_exec = r->exec;'
assert needle in s;s=s.replace(needle,'''  if(r->v3_prefill_capacity){
    if(!valid_prefill_shape_packet(source,bytes,r->v3_prefill_physical,8))return reject(error,"invalid V3 prefill packet",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
    auto value=[&](size_t at){uint32_t v;std::memcpy(&v,source+at,4);return v;};
    if(value(16)!=0||value(20)!=1||value(136)>r->v3_prefill_capacity||value(164)>r->v3_prefill_context)
      return reject(error,"V3 prefill graph shape mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  }
'''+needle,1)
s+='''
// Retains all parents through the existing ledger. V3 result: status/publish at
//0/4, canonical argmax at8, logits at128; unused prefix bytes stay zero.
extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(
 RileyCudaGraphResources* r,RileyCudaDeviceBuffer*const* d,RileyCudaDeviceBuffer*const* w,
 uint64_t weight_count,RileyCudaGemmPlan* head,RileyCudaPinnedHostBuffer* staging,
 uint32_t capacity,uint32_t physical,RileyCudaErrorInfo* error) noexcept {
 clear_error(error);auto status=transfer_ready(r,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 if(!d||!w||!head||!staging||weight_count!=273||!capacity||capacity>1024||!physical||physical>4096||
    r->graph||r->exec||r->prefill_graph||r->prefill_exec||thread_has_active_graph_capture()||thread_has_active_command_batch()||
    !same_context(staging->owner,r->owner)||!holds_counter(r,&staging->active_uses)||!holds_plan(r,head)||staging->byte_len<196864)
   return reject(error,"V3 recorder lifecycle",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 for(size_t i=0;i<23;++i){if(i==22&&!d[i])continue;
  if(!d[i]||!same_context(d[i]->owner,r->owner)||!holds_counter(r,&d[i]->active_uses))return reject(error,"V3 device parent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<i;++j)if(d[i]==d[j])return reject(error,"V3 mutable alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 for(size_t i=0;i<273;++i){if(!w[i]||!same_context(w[i]->owner,r->owner)||!holds_counter(r,&w[i]->active_uses))return reject(error,"V3 weight parent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
  for(size_t j=0;j<23;++j)if(w[i]==d[j])return reject(error,"V3 weight/mutable alias",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 }
 const uint64_t widths[12]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};
 for(size_t i=0;i<12;++i)if(d[i]->byte_len!=capacity*widths[i])return reject(error,"V3 scratch extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(d[12]->byte_len!=uint64_t(30)*physical*16*384||d[13]->byte_len!=d[12]->byte_len||!d[14]->byte_len||d[14]->byte_len%128||d[14]->byte_len>4096*128||d[15]->byte_len!=d[14]->byte_len||d[16]->byte_len!=17536||d[17]->byte_len!=1152||d[18]->byte_len!=4||d[19]->byte_len!=4||d[20]->byte_len!=98304||d[21]->byte_len!=sizeof(RileyCudaBf16ArgmaxResult))return reject(error,"V3 device extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 if(w[0]->byte_len!=56623104||w[1]->byte_len!=1152||w[2]->byte_len!=56623104)return reject(error,"V3 global weights",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 const uint64_t sizes[9]={1152,663552,221184,221184,663552,1152,1769472,1769472,1769472};
 for(size_t i=3;i<273;++i)if(w[i]->byte_len!=sizes[(i-3)%9])return reject(error,"V3 layer weights",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 GemmCanonicalExecutionState state{};
 status=bind_reserved_gemm_state(head,r->stream,d[17],w[2],d[20],d[22],&state,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 void* scratch[12];const void* weights[273];for(size_t i=0;i<12;++i)scratch[i]=d[i]->device_data;for(size_t i=0;i<273;++i)weights[i]=w[i]->device_data;
 constexpr uint64_t transfer=98432;
 status=record_reserved_sequence(r,staging,transfer,[&]() noexcept {
  auto* host=static_cast<uint8_t*>(staging->host_data);
  auto copy=[&](void* a,const void* b,uint64_t n,cudaMemcpyKind kind){return runtime_error(cudaMemcpyAsync(a,b,n,kind,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"V3 transfer");};
  auto result=copy(d[16]->device_data,host,17536,cudaMemcpyHostToDevice);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_prefill_model(r->stream->stream,scratch,weights,d[16]->device_data,d[12]->device_data,d[13]->device_data,d[14]->device_data,d[15]->device_data,d[17]->device_data,static_cast<uint32_t*>(d[18]->device_data),static_cast<uint32_t*>(d[19]->device_data),capacity,physical),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 model");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[20],state,error,"V3 head");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_decode_argmax(r->stream->stream,d[20]->device_data,d[21]->device_data,49152),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 argmax");
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[18]->device_data,4,cudaMemcpyDeviceToHost);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer+4,d[19]->device_data,4,cudaMemcpyDeviceToHost);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer+8,d[21]->device_data,sizeof(RileyCudaBf16ArgmaxResult),cudaMemcpyDeviceToHost);
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer+128,d[20]->device_data,98304,cudaMemcpyDeviceToHost);
  return result;
 },error);
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->v3_prefill_capacity=capacity;r->v3_prefill_physical=physical;r->v3_prefill_context=d[14]->byte_len/128;}
 return status;
}
'''
# Resolve the actual canonical execution state type from existing declarations.
import re
match=re.search(r'([A-Za-z0-9_]+)\s+states\[',s)
print('existing state',match.group(1) if match else 'missing')
if match:s=s.replace('GemmCanonicalExecutionState state{};',match.group(1)+' state{};')
p.write_text(s)
p=r/'kernels/include/riley_cuda.h';s=p.read_text();at=s.index('RileyCudaStatus riley_cuda_graph_resources_record_decode_prefill128_packed(');s=s[:at]+'''/* V3 borrowed prefill recorder: 23 device parents, 273 weights, canonical head. */
RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(
    RileyCudaGraphResources*, RileyCudaDeviceBuffer* const*, RileyCudaDeviceBuffer* const*,
    uint64_t, RileyCudaGemmPlan*, RileyCudaPinnedHostBuffer*, uint32_t, uint32_t,
    RileyCudaErrorInfo*) RILEY_CUDA_NOEXCEPT;

'''+s[at:];p.write_text(s)
