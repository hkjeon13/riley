from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
// V11 shape primitives share the established graph owner's parent validation.
#include "prefill_shape_projection.cuh"
#include "prefill_shape_rope_kv.cuh"
namespace riley_cuda_internal {
cudaError_t enqueue_shape_prefill_gemm(cudaStream_t stream,const void* x,const void* w,void* y,int n,int k,int interval,const void* position,uint32_t rows) noexcept {
 // Keep the previously measured P128 Q/K/V/O load path; gate/up/down use the
 // tail-safe shape primitive. Other shapes use that primitive for every role.
 if(rows==128&&(n==192||(n==576&&k==576)))return enqueue_compiled_prefill_m16_gemm(stream,x,w,y,n,k,interval,position);
 auto* a=static_cast<const __nv_bfloat16*>(x);auto* b=static_cast<const __nv_bfloat16*>(w);auto* c=static_cast<__nv_bfloat16*>(y);
 if(n==576&&k==576&&interval==192)return launch_prefill_shape<576,576,192,2>(stream,a,b,c,rows);
 if(n==192&&k==576&&interval==192)return launch_prefill_shape<192,576,192,1>(stream,a,b,c,rows);
 if(n==576&&k==576&&interval==128)return launch_prefill_shape<576,576,128,2>(stream,a,b,c,rows);
 if(n==1536&&k==576&&interval==0)return launch_prefill_shape<1536,576,0,4>(stream,a,b,c,rows);
 if(n==576&&k==1536&&interval==320)return launch_prefill_shape<576,1536,320,2>(stream,a,b,c,rows);
 return cudaErrorInvalidValue;
}
cudaError_t enqueue_shape_prefill_rope_kv(cudaStream_t stream,const void* q,const void* k,const void* v,void* qo,void* keys,void* values,const void* cos,const void* sin,const void* metadata,uint32_t rows) noexcept {
 // Only the validated full P128 stage is connected so far. Do not infer new
 // request support from the wider private primitive.
 if(rows!=128)return cudaErrorInvalidValue;
 return launch_prefill_shape_rope_kv(stream,static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),static_cast<__nv_bfloat16*>(qo),static_cast<__nv_bfloat16*>(keys),static_cast<__nv_bfloat16*>(values),static_cast<const float*>(cos),static_cast<const float*>(sin),reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+16),0,128,160);
}
}
''';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text();needle='cudaError_t enqueue_compiled_prefill_m16_gemm('
i=s.index(needle);s=s[:i]+'''cudaError_t enqueue_shape_prefill_gemm(cudaStream_t,const void*,const void*,void*,int,int,int,const void*,uint32_t) noexcept;
cudaError_t enqueue_shape_prefill_rope_kv(cudaStream_t,const void*,const void*,const void*,void*,void*,void*,const void*,const void*,const void*,uint32_t) noexcept;
'''+s[i:];p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text();s=s.replace('enqueue_compiled_prefill_m16_gemm(r->stream->stream,buffer(inputs[j]),weights[weight_ids[j]]->device_data,buffer(out),ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4)', 'enqueue_shape_prefill_gemm(r->stream->stream,buffer(inputs[j]),weights[weight_ids[j]]->device_data,buffer(out),ns[j],ks[j],chunks[j],static_cast<uint8_t*>(d[19]->device_data)+4,rows)')
needle='      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode)s=kernel(enqueue_compiled_rope_rows'
assert needle in s
s=s.replace(needle,'''      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&batched&&packed_buffers!=nullptr)
        s=kernel(enqueue_shape_prefill_rope_kv(r->stream->stream,buffer(3),buffer(6),buffer(7),buffer(4),
          static_cast<uint8_t*>(d[15]->device_data)+l*k*16*physical,static_cast<uint8_t*>(d[16]->device_data)+l*k*16*physical,
          d[17]->device_data,d[18]->device_data,d[19]->device_data,rows));
      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode&&!(batched&&packed_buffers!=nullptr))s=kernel(enqueue_compiled_rope_rows''')
s=s.replace('if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode)s=kernel(enqueue_compiled_kv_write_rows','if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2&&!packed_decode&&!(batched&&packed_buffers!=nullptr))s=kernel(enqueue_compiled_kv_write_rows');p.write_text(s)
p=r/'crates/riley-runtime/src/llama/graph_decode_multi_session.rs';s=p.read_text();needle='        hash.update(include_bytes!("../../../../kernels/src/graph_numerics_precise.cu"));';assert needle in s;s=s.replace(needle,needle+'\n'+''.join(f'        hash.update(include_bytes!("../../../../kernels/src/{name}"));\n' for name in ['prefill_shape_projection.cuh','prefill_shape_rope_kv.cuh','prefill_shape_attention.cuh']));p.write_text(s)
print('connected shape projection and fused RoPE/KV to retained P128 graph; admission unchanged')
