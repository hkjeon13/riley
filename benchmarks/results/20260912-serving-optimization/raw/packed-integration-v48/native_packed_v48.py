from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');fixture=Path('/tmp/riley-opt-260912/packed-prefill-loaded-v48')
for n in ['packed_prefill_attention_v48.cuh','packed_prefill_rope_v48.cuh','packed_prefill_model_v48.cuh']:
 s=(fixture/n).read_text()
 if n=='packed_prefill_model_v48.cuh':s=s.replace('if(threadIdx.x==0)publish[owner]', 'if(publish&&threadIdx.x==0)publish[owner]').replace('||!publish||','||').replace('Caller must validate V3 packet','Caller must validate V6 packet')
 (r/'kernels/src'/n).write_text(s)
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
#include "packed_prefill_model_v48.cuh"
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v6_prefill_model(cudaStream_t s,void*const* d,const void*const* w,const void* m,void* k,void* v,const void* c,const void* sn,void* selected,uint32_t* status,uint32_t* publish,uint32_t rows,uint32_t physical,bool tiled) noexcept {
 return enqueue_packed_prefill_model<32>(s,d,w,m,k,v,c,sn,selected,status,publish,rows,physical,tiled);
}
cudaError_t enqueue_compiled_v6_shared_result(cudaStream_t s,const void* m,const void* logits,const void* status,void* result) noexcept {
 return riley_shared32_result::enqueue<0x36524d52>(s,m,static_cast<const __nv_bfloat16*>(logits),static_cast<const uint32_t*>(status),result);
}
}
''';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text()
for name in ['prefill_model','shared_result']:
 line=next(x for x in s.splitlines() if x.startswith('cudaError_t enqueue_compiled_v5_'+name+'('));s=s.replace(line,line+'\n'+line.replace('v5_','v6_'))
p.write_text(s)
p=r/'kernels/src/compact_shared_result.cuh';s=p.read_text().replace('template<uint32_t Rows>\n__global__ void finish','template<uint32_t Rows,bool Packed=false>\n__global__ void finish').replace('template<uint32_t Rows>\ninline cudaError_t enqueue','template<uint32_t Rows,bool Packed=false>\ninline cudaError_t enqueue').replace('result[31]=(Rows==8?', 'result[31]=(Packed?0x36524d52U:(Rows==8?').replace('0x35524d52U))^','0x35524d52U)))^').replace('finish<Rows><<<','finish<Rows,Packed><<<');p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('uint32_t variable_rows=8;', 'uint32_t variable_rows=8;\n  bool packed_prefill=false;')
s=s.replace('r->variable_rows==32?valid_v5_shape_packet','r->packed_prefill?valid_v6_shape_packet(source,bytes,r->v3_prefill_physical,32):r->variable_rows==32?valid_v5_shape_packet')
s=s.replace('value(136)>r->v3_prefill_capacity', '(r->packed_prefill&&value(16)==0?value(36):value(136))>r->v3_prefill_capacity')
s=s.replace('template<uint32_t Rows,bool Compact=false>\nstatic RileyCudaStatus record_variable_shared','template<uint32_t Rows,bool Compact=false,bool Packed=false>\nstatic RileyCudaStatus record_variable_shared')
s=s.replace(' constexpr uint64_t request_bytes=128+Rows*1664+4096;', ' static_assert(!Packed||Rows==32,"V6 requires32 rows");\n constexpr uint64_t request_bytes=128+Rows*1664+4096;')
s=s.replace('riley_compact_result::enqueue<Rows>', 'riley_compact_result::enqueue<Rows,Packed>')
s=s.replace('(Rows==8?enqueue_compiled_v3_shared_result:', '(Packed?enqueue_compiled_v6_shared_result:Rows==8?enqueue_compiled_v3_shared_result:')
marker='  if(!compact&&result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(cudaMemsetAsync(d[25]->device_data,0,transfer,r->stream->stream)'
a=s.index(marker)
block='''  if constexpr(Packed){
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v6_prefill_model(r->stream->stream,scratch,prefill_weights,d[16]->device_data,d[12]->device_data,d[13]->device_data,d[14]->device_data,d[15]->device_data,d[23]->device_data,static_cast<uint32_t*>(d[18]->device_data),nullptr,row_capacity,physical,weight_count==363),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V6 packed model");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=enqueue_canonical_gemm_bf16_graph_matmul(r->owner,r->stream,d[24],shared_state,error,"V6 packed head");
   if(compact){
    if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(riley_compact_result::enqueue<Rows,Packed>(r->stream->stream,d[16]->device_data,d[24]->device_data,static_cast<uint32_t*>(d[18]->device_data),d[7]->device_data,d[25]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V6 compact completion");
   }else if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_compiled_v6_shared_result(r->stream->stream,d[16]->device_data,d[24]->device_data,d[18]->device_data,d[25]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V6 full completion");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,compact?Rows*128:transfer,cudaMemcpyDeviceToHost);
   return result;
  }
'''
s=s[:a]+block+s[a:]
s=s.replace('r->compact_ready=Compact;r->v3_shared=true;', 'r->packed_prefill=Packed;r->compact_ready=Compact;r->v3_shared=true;')
# Duplicate existing V5 ABI functions with packed template enabled.
for suffix in ['','_greedy']:
 a=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v5_shared'+suffix+'(');b=s.index('\n}',a)+2;t=s[a:b].replace('v5_shared','v6_shared').replace('record_variable_shared<32>', 'record_variable_shared<32,false,true>').replace('record_variable_shared<32,true>', 'record_variable_shared<32,true,true>');s+='\n'+t+'\n'
p.write_text(s)
# Duplicate Rust wrapper methods by balanced braces, including braces in cfg blocks.
for path,prefix in [('crates/riley-cuda/src/graph_resources.rs','pub fn '),('crates/riley-cuda/src/ffi.rs','pub(super) fn ')]:
 p=r/path;s=p.read_text()
 for suffix in ['','_greedy']:
  marker=prefix+'record_v5_shared'+suffix+'(';a=s.index(marker);brace=s.index('{',a);depth=1;b=brace+1
  while depth:
   depth+=(s[b]=='{')-(s[b]=='}');b+=1
  t=s[a:b].replace('v5_shared','v6_shared').replace('V4 shared','V6 packed');s=s[:b]+'\n'+t+s[b:]
 if path.endswith('ffi.rs'):
  for suffix in ['','_greedy']:
   line=next(x for x in s.splitlines() if x.startswith('fn riley_cuda_graph_resources_record_v5_shared'+suffix+'('));s=s.replace(line,line+'\n'+line.replace('v5_shared','v6_shared'))
 p.write_text(s)
p=r/'kernels/include/riley_cuda.h';s=p.read_text()
for suffix in ['','_greedy']:
 a=s.index('RileyCudaStatus riley_cuda_graph_resources_record_v5_shared'+suffix+'(');b=s.index(';',a)+1;t=s[a:b].replace('v5_shared','v6_shared');s=s[:b]+'\n'+t+s[b:]
p.write_text(s)
p=r/'crates/riley-cuda/build.rs';s=p.read_text();marker='        kernels_dir.join("src/prefill_shape_model.cuh"),';s=s.replace(marker,marker+'\n'+''.join('        kernels_dir.join("src/'+n+'"),\n' for n in ['packed_prefill_attention_v48.cuh','packed_prefill_rope_v48.cuh','packed_prefill_model_v48.cuh']));p.write_text(s)
print('Packed native capture, versioned full/compact results and FFI implemented')
