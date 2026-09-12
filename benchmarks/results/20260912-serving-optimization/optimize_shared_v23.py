from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
def edit(name,a,b):
 p=r/name;s=p.read_text();assert a in s,(name,a);p.write_text(s.replace(a,b))
p=r/'kernels/src/decode_shared_attention.cuh';s=p.read_text().replace('if(token>=count || count>4096)return;','if(count<1 || count>4096)return;\n for(;token<count;token+=gridDim.x*8){').replace('\n}\n// Independent output','\n }\n}\n// Independent output').replace('dim3((context+7)/8,72)','dim3(min((context+7)/8,32u),72)');p.write_text(s)
edit('crates/riley-runtime/src/llama/variable_session.rs','capacity.max(8)},false)?','capacity.max(8)},true)?')
edit('crates/riley-runtime/src/llama/graph_decode_full.rs','if scratch.shared_head.is_none() {\n        for (index,buffer)','if !scratch.tiled.is_empty() {\n        for (index,buffer)')
p=r/'kernels/src/decode_shared_model.cuh';s=p.read_text().replace('uint32_t physical,uint32_t context){','uint32_t physical,uint32_t context,bool tiled=false){');s=s.replace(' for(int layer=0;layer<30;++layer){',' if(tiled)for(int i=273;i<363;++i)if(!weights[i])return cudaErrorInvalidValue;\n for(int layer=0;layer<30;++layer){')
for n,k,i,part,inp,out in [(1536,576,0,0,1,8),(1536,576,0,1,1,9),(576,1536,320,2,11,4)]:
 a=f'enqueue_shared_projection<{n},{k},{i},false>(stream,b({inp}),w(base+{6+part}),b({out}),static_cast<float*>(scratch[7]),active);'
 b=f'if(tiled)enqueue_shared_projection<{n},{k},{i},true>(stream,b({inp}),w(273+layer*3+{part}),b({out}),static_cast<float*>(scratch[7]),active);\n else '+a
 assert a in s;s=s.replace(a,b)
p.write_text(s)
edit('kernels/src/ffi_internal.hpp','uint32_t*,uint32_t,uint32_t) noexcept;\ncudaError_t enqueue_compiled_v3_shared_result','uint32_t*,uint32_t,uint32_t,bool) noexcept;\ncudaError_t enqueue_compiled_v3_shared_result')
edit('kernels/src/graph_numerics_precise.cu','uint32_t physical,uint32_t context) noexcept {\n return riley_shared_model::enqueue','uint32_t physical,uint32_t context,bool tiled) noexcept {\n return riley_shared_model::enqueue')
edit('kernels/src/graph_numerics_precise.cu','status,physical,context);\n}\ncudaError_t enqueue_compiled_v3_shared_result','status,physical,context,tiled);\n}\ncudaError_t enqueue_compiled_v3_shared_result')
p=r/'kernels/src/graph_resources.cu';s=p.read_text();start=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_shared(');a=s[:start];b=s[start:].replace('const void* weights[273]','const void* weights[363]').replace('i<273;++i)weights[i]=w[i]->device_data','i<weight_count;++i)weights[i]=w[i]->device_data').replace('physical,std::min<uint64_t>(4096,d[14]->byte_len/128)),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 shared model"','physical,std::min<uint64_t>(4096,d[14]->byte_len/128),weight_count==363),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"V3 shared model"');p.write_text(a+b)
