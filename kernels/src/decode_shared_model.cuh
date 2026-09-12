#pragma once
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
__global__ void shared_rope_kv(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,
 const uint32_t* pages,const uint32_t* shape,const uint32_t* active){
 const uint32_t rows=*active,row=blockIdx.y;if(!rows||rows>8||row>=rows)return;
 pages+=row*416;shape+=row*416;
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;
 const uint32_t pos=shape[1];int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[pos*32+dim]));
 const __nv_bfloat16* src=head<9?q+row*576:k+row*192;
 int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 const __nv_bfloat16 first=__float2bfloat16_rn(a*c-b*s),second=__float2bfloat16_rn(b*c+a*s);
 if(head<9){qo[row*576+base+dim]=first;qo[row*576+base+dim+32]=second;}
 else{
  uint32_t destination=((pages[pos/16]*3+head-9)*16+pos%16)*64+dim;
  keys[destination]=first;keys[destination+32]=second;
  values[destination]=v[row*192+base+dim];values[destination+32]=v[row*192+base+dim+32];
 }
}
// Fixed-capacity GEMM also reads inactive rows; initialize those inputs.
__global__ void clear_inactive_hidden(__nv_bfloat16* hidden,const uint32_t* active){
 uint32_t rows=*active,row=blockIdx.x;if(rows<1||rows>8||row<rows)return;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)hidden[row*576+i]=__float2bfloat16_rn(0.F);
}
inline cudaError_t enqueue(cudaStream_t stream,void*const* scratch,const void*const* weights,const void* metadata,void* keys,void* values,const float* cos,const float* sin,uint32_t* status,uint32_t physical,uint32_t context,bool tiled=false){
 if(!scratch||!weights||!metadata||!keys||!values||!cos||!sin||!status||!physical||physical>4096||!context||context>4096)return cudaErrorInvalidValue;
 for(int i=0;i<12;++i)if(!scratch[i])return cudaErrorInvalidValue;
 for(int i=0;i<273;++i)if(!weights[i])return cudaErrorInvalidValue;
 auto b=[&](int i){return static_cast<__nv_bfloat16*>(scratch[i]);};
 auto w=[&](int i){return static_cast<const __nv_bfloat16*>(weights[i]);};
 const auto* meta=static_cast<const uint32_t*>(metadata);auto* shape=meta+32;auto* pages=meta+64;auto* active=meta+5;auto* pointwise=meta+3;
 auto err=cudaMemsetAsync(status,0,4,stream);if(err!=cudaSuccess)return err;
 embedding<<<8,256,0,stream>>>(w(0),b(0),shape,active,status);
 if(tiled)for(int i=273;i<363;++i)if(!weights[i])return cudaErrorInvalidValue;
 for(int layer=0;layer<30;++layer){int base=3+9*layer;
 if(layer==0)riley_prefill_pointwise::norm_rows<<<8,256,0,stream>>>(b(0),nullptr,w(base),nullptr,b(1),0,pointwise,8);
 enqueue_shared_projection<576,576,192,false>(stream,b(1),w(base+1),b(2),static_cast<float*>(scratch[7]),active);
 enqueue_shared_projection<192,576,192,false>(stream,b(1),w(base+2),b(5),static_cast<float*>(scratch[7]),active);
 enqueue_shared_projection<192,576,192,false>(stream,b(1),w(base+3),b(6),static_cast<float*>(scratch[7]),active);
 auto* lk=static_cast<__nv_bfloat16*>(keys)+uint64_t(layer)*physical*16*192;
 auto* lv=static_cast<__nv_bfloat16*>(values)+uint64_t(layer)*physical*16*192;
 shared_rope_kv<<<dim3(2,8),256,0,stream>>>(b(2),b(5),b(6),b(3),lk,lv,cos,sin,pages,shape,active);
 riley_shared_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);
 enqueue_shared_projection<576,576,128,false>(stream,b(4),w(base+4),b(2),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<8,256,0,stream>>>(b(2),b(0),w(base+5),scratch[10],b(1),1,pointwise,8);
 if(tiled)enqueue_shared_projection<1536,576,0,true>(stream,b(1),w(273+layer*3+0),b(8),static_cast<float*>(scratch[7]),active);
 else enqueue_shared_projection<1536,576,0,false>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]),active);
 if(tiled)enqueue_shared_projection<1536,576,0,true>(stream,b(1),w(273+layer*3+1),b(9),static_cast<float*>(scratch[7]),active);
 else enqueue_shared_projection<1536,576,0,false>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::swiglu_rows<<<dim3(6,8),256,0,stream>>>(b(8),b(9),b(11),pointwise,8);
 if(tiled)enqueue_shared_projection<576,1536,320,true>(stream,b(11),w(273+layer*3+2),b(4),static_cast<float*>(scratch[7]),active);
 else enqueue_shared_projection<576,1536,320,false>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<8,256,0,stream>>>(b(4),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,pointwise,8);
 }
 clear_inactive_hidden<<<8,256,0,stream>>>(b(1),active);
 return cudaGetLastError();
}
}
