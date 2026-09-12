from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/graph_resources.cu';s=p.read_text();s=s.replace('uint32_t v3_prefill_capacity=0','bool v3_shared=false;\n  uint32_t v3_prefill_capacity=0')
s=s.replace('if(value(20)!=1||value(136)>r->v3_prefill_capacity||value(164)>r->v3_prefill_context)','if((!r->v3_shared&&value(20)!=1)||value(136)>r->v3_prefill_capacity||value(164)>r->v3_prefill_context)')
a='      return reject(error,"V3 prefill graph shape mismatch",RILEY_CUDA_STATUS_INVALID_ARGUMENT);';s=s.replace(a,a+'\n    for(uint32_t row=0;row<value(20);++row)if(value(164+row*1664)>r->v3_prefill_context)return reject(error,"V3 row context exceeds retained tables",RILEY_CUDA_STATUS_INVALID_ARGUMENT);')
a=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(');copy=s[a:];assert copy.strip().endswith('}')
copy=copy.replace('record_v3_prefill(','record_v3_shared(').replace('RileyCudaGemmPlan* head,RileyCudaPinnedHostBuffer* staging','RileyCudaGemmPlan* head,RileyCudaGemmPlan* shared_head,RileyCudaPinnedHostBuffer* staging')
copy=copy.replace('if(!d||!w||!head||!staging','if(!d||!w||!head||!shared_head||!holds_plan(r,shared_head)||capacity<8||!staging')
copy=copy.replace('staging->byte_len<196864','staging->byte_len<1574912').replace('i<23','i<26').replace('j<23','j<26').replace('9*4096*4','8*9*4096*4')
a=' RileyCudaCanonicalGemmBf16GraphState state{};'
b=''' if(d[23]->byte_len!=8*1152||d[24]->byte_len!=8*98304||d[25]->byte_len!=787456)return reject(error,"V3 shared output extent",RILEY_CUDA_STATUS_INVALID_ARGUMENT);
 RileyCudaCanonicalGemmBf16GraphState shared_state{};
 status=bind_reserved_shared_row_gemm_state(shared_head,r->stream,d[23],w[2],d[24],8,49152,576,&shared_state,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
'''+a
assert a in copy;copy=copy.replace(a,b).replace('constexpr uint64_t transfer=98432;','constexpr uint64_t transfer=787456;')
# Shared model uses original weights. Its head runs all8initialized hidden rows.
start=copy.index('  if(row_capacity==1&&weight_count==363)');end=copy.index('  return record_reserved_sequence',start);copy=copy[:start]+copy[end:]
a='  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_prefill_model'
pos=copy.index(a)
branch='''  if(row_capacity==1){
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_shared_model(r->stream->stream,scratch,weights,d[16]->device_data,d[12]->device_data,d[13]->device_data,d[14]->device_data,d[15]->device_data,static_cast<uint32_t*>(d[18]->device_data),physical,std::min<uint64_t>(4096,d[14]->byte_len/128)),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 shared model");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(d[23]->device_data,d[1]->device_data,8*1152,cudaMemcpyDeviceToDevice);
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[24],shared_state,error,"V3 shared head");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v3_shared_result(r->stream->stream,d[16]->device_data,d[24]->device_data,d[18]->device_data,d[25]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 shared completion");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,transfer,cudaMemcpyDeviceToHost);
   return result;
  }
  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(cudaMemsetAsync(d[25]->device_data,0,transfer,r->stream->stream),error,RILEY_CUDA_ERROR_STAGE_COPY,"V3 clear inactive output");
'''
copy=copy[:pos]+branch+copy[pos:]
copy=copy.replace('copy(host+transfer,d[18]->device_data,128,cudaMemcpyDeviceToHost)','copy(d[25]->device_data,d[18]->device_data,128,cudaMemcpyDeviceToDevice)').replace('copy(host+transfer+128,d[20]->device_data,98304,cudaMemcpyDeviceToHost)','copy(static_cast<uint8_t*>(d[25]->device_data)+128,d[20]->device_data,98304,cudaMemcpyDeviceToDevice)')
copy=copy.replace('  return result;\n },error);};','  if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,transfer,cudaMemcpyDeviceToHost);\n  return result;\n },error);};')
copy=copy.replace('{r->v3_prefill_capacity=capacity;','{r->v3_shared=true;r->v3_prefill_capacity=capacity;')
s+='\n'+copy;p.write_text(s)
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
#include "decode_shared_model.cuh"
#include "decode_shared_result.cuh"
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v3_shared_model(cudaStream_t s,void*const* d,const void*const* w,const void* m,void* k,void* v,const void* c,const void* sn,uint32_t* status,uint32_t physical,uint32_t context) noexcept {
 return riley_shared_model::enqueue(s,d,w,m,k,v,static_cast<const float*>(c),static_cast<const float*>(sn),status,physical,context);
}
cudaError_t enqueue_compiled_v3_shared_result(cudaStream_t s,const void* m,const void* logits,const void* status,void* result) noexcept {
 return riley_shared_result::enqueue(s,m,static_cast<const __nv_bfloat16*>(logits),static_cast<const uint32_t*>(status),result);
}
}
''';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();a='cudaError_t enqueue_compiled_v3_prefill_model(';pos=s.index(a);s=s[:pos]+'''cudaError_t enqueue_compiled_v3_shared_model(cudaStream_t,void*const*,const void*const*,const void*,void*,void*,const void*,const void*,uint32_t*,uint32_t,uint32_t) noexcept;
cudaError_t enqueue_compiled_v3_shared_result(cudaStream_t,const void*,const void*,const void*,void*) noexcept;
'''+s[pos:];p.write_text(s)
p=r/'kernels/include/riley_cuda.h';s=p.read_text();a=s.index('RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(');b=s.index(';',a)+1;decl=s[a:b].replace('record_v3_prefill','record_v3_shared').replace('RileyCudaGemmPlan* head,','RileyCudaGemmPlan* head, RileyCudaGemmPlan* shared_head,');s=s[:b]+'\n'+decl+s[b:];p.write_text(s)
p=r/'crates/riley-cuda/build.rs';s=p.read_text();a='        kernels_dir.join("src/decode_tiled.cuh"),';s=s.replace(a,a+'\n'+''.join('        kernels_dir.join("src/'+n+'"),\n' for n in ['decode_shared.cuh','decode_shared_attention.cuh','decode_shared_model.cuh','decode_shared_result.cuh']));p.write_text(s)
