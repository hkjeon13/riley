from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/kernels/src')
rope=(r/'prefill_shape_rope_kv.cuh').read_text().split('__global__ void prefill_shape_rope_kv')[1].split('cudaError_t launch_prefill')[0]
rope='__global__ void shared_rope_kv'+rope
rope=rope.replace('const uint32_t* pages,uint32_t start,uint32_t rows,const uint32_t* shape=nullptr)', 'const uint32_t* pages,const uint32_t* shape,const uint32_t* active)')
a=' if(shape){uint32_t live=shape[2];if(!live||live>rows)return;rows=live;start=shape[4];}\n const uint32_t row=blockIdx.y;if(row>=rows)return;'
b=' const uint32_t rows=*active,row=blockIdx.y;if(!rows||rows>8||row>=rows)return;\n pages+=row*416;shape+=row*416;'
assert a in rope;rope=rope.replace(a,b).replace('pos=start+row','pos=shape[1]')
s='''#pragma once
#include "prefill_shape_model.cuh"
#include "decode_shared.cuh"
#include "decode_shared_attention.cuh"
// Internal model sequence: caller validates all row identities, context/page
// ownership and buffer extents before enqueue. Final hidden rows stay in b(1).
namespace riley_shared_model {
__global__ void embedding(const __nv_bfloat16* weights,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* active,uint32_t* status){
 uint32_t row=blockIdx.x,rows=*active;if(!rows||rows>8){if(threadIdx.x==0)atomicOr(status,2u);return;}if(row>=rows)return;
 uint32_t token=shape[row*416];if(token>=49152){if(threadIdx.x==0)atomicOr(status,1u);return;}
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)out[row*576+i]=weights[token*576+i];
}
'''+rope
s+='''inline cudaError_t enqueue(cudaStream_t stream,void*const* scratch,const void*const* weights,const void* metadata,void* keys,void* values,const float* cos,const float* sin,uint32_t* status,uint32_t physical,uint32_t context){
 if(!scratch||!weights||!metadata||!keys||!values||!cos||!sin||!status||!physical||physical>4096||!context||context>4096)return cudaErrorInvalidValue;
 for(int i=0;i<12;++i)if(!scratch[i])return cudaErrorInvalidValue;
 for(int i=0;i<273;++i)if(!weights[i])return cudaErrorInvalidValue;
 auto b=[&](int i){return static_cast<__nv_bfloat16*>(scratch[i]);};
 auto w=[&](int i){return static_cast<const __nv_bfloat16*>(weights[i]);};
 const auto* meta=static_cast<const uint32_t*>(metadata);auto* shape=meta+32;auto* pages=meta+64;auto* active=meta+5;auto* pointwise=meta+3;
 auto err=cudaMemsetAsync(status,0,4,stream);if(err!=cudaSuccess)return err;
 embedding<<<8,256,0,stream>>>(w(0),b(0),shape,active,status);
 for(int layer=0;layer<30;++layer){int base=3+9*layer;
 if(layer==0)riley_prefill_pointwise::norm_rows<<<8,256,0,stream>>>(b(0),nullptr,w(base),nullptr,b(1),0,pointwise,8);
'''
for N,K,I,src,wo,dst in [(576,576,192,1,1,2),(192,576,192,1,2,5),(192,576,192,1,3,6)]:s+=f' enqueue_shared_projection<{N},{K},{I},false>(stream,b({src}),w(base+{wo}),b({dst}),static_cast<float*>(scratch[7]),active);\n'
s+=''' auto* lk=static_cast<__nv_bfloat16*>(keys)+uint64_t(layer)*physical*16*192;
 auto* lv=static_cast<__nv_bfloat16*>(values)+uint64_t(layer)*physical*16*192;
 shared_rope_kv<<<dim3(2,8),256,0,stream>>>(b(2),b(5),b(6),b(3),lk,lv,cos,sin,pages,shape,active);
 riley_shared_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);
 enqueue_shared_projection<576,576,128,false>(stream,b(4),w(base+4),b(2),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<8,256,0,stream>>>(b(2),b(0),w(base+5),scratch[10],b(1),1,pointwise,8);
 enqueue_shared_projection<1536,576,0,false>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]),active);
 enqueue_shared_projection<1536,576,0,false>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::swiglu_rows<<<dim3(6,8),256,0,stream>>>(b(8),b(9),b(11),pointwise,8);
 enqueue_shared_projection<576,1536,320,false>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<8,256,0,stream>>>(b(4),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,pointwise,8);
 }
 return cudaGetLastError();
}
}
'''
(r/'decode_shared_model.cuh').write_text(s)
