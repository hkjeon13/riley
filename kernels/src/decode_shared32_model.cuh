#ifdef RILEY_CUDA_ENABLE_FA3
#include "../optional/fa3_api.h"
#endif
#include "../optional/ffn_pipeline.cuh"
#include "../optional/decode_adaptive_rows.cuh"
#pragma once
#include "../optional/context_split_fp32.cuh"
#include "prefill_shape_model.cuh"
#include "decode_shared32.cuh"
#include "decode_shared32_attention.cuh"
#include "decode_gqa_attention_v50.cuh"
#include "decode_gate_v56.cuh"
#include "decode_merge_norm_v56.cuh"
#ifdef RILEY_CUDA_ENABLE_FLASHINFER
#include "../optional/flashinfer_api.h"
#endif
// Internal model sequence: caller validates all row identities, context/page
// ownership and buffer extents before enqueue. Final hidden rows stay in b(1).
namespace riley_shared32_model {
__global__ void embedding(const __nv_bfloat16* weights,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* active,uint32_t* status){
 uint32_t row=blockIdx.x,rows=*active;if(!rows||rows>32){if(threadIdx.x==0)atomicOr(status,2u);return;}if(row>=rows)return;
 uint32_t token=shape[row*416];if(token>=49152){if(threadIdx.x==0)atomicOr(status,1u);return;}
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)out[row*576+i]=weights[token*576+i];
}
__global__ void shared_rope_kv(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,
 const uint32_t* pages,const uint32_t* shape,const uint32_t* active){
 const uint32_t rows=*active,row=blockIdx.y;if(!rows||rows>32||row>=rows)return;
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

__device__ __forceinline__ __nv_bfloat16 qkv_rounded(const float* p,int n,int row,int col){
 float value=0.;for(int chunk=0;chunk<3;++chunk)value+=p[chunk*32*n+row*n+col];return __float2bfloat16_rn(value);
}
__global__ void qkv_merge_rope(const float* parts,__nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,const uint32_t* pages,const uint32_t* shape,const uint32_t* active){
 uint32_t rows=*active,row=blockIdx.y;if(rows<1||rows>32||row>=rows)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=384)return;
 shape+=row*416;pages+=row*416;uint32_t pos=shape[1];int head=i/32,dim=i%32;
 const float* src=head<9?parts:parts+3*32*576;int n=head<9?576:192,base=(head<9?head:head-9)*64;
 float a=__bfloat162float(qkv_rounded(src,n,row,base+dim)),b=__bfloat162float(qkv_rounded(src,n,row,base+dim+32));
 float c=__bfloat162float(__float2bfloat16_rn(cos[pos*32+dim])),sn=__bfloat162float(__float2bfloat16_rn(sin[pos*32+dim]));
 auto first=__float2bfloat16_rn(a*c-b*sn),second=__float2bfloat16_rn(b*c+a*sn);
 if(head<9){qo[row*576+base+dim]=first;qo[row*576+base+dim+32]=second;}
 else {uint32_t dst=((pages[pos/16]*3+head-9)*16+pos%16)*64+dim;keys[dst]=first;keys[dst+32]=second;const float* v=parts+3*32*(576+192);values[dst]=qkv_rounded(v,192,row,base+dim);values[dst+32]=qkv_rounded(v,192,row,base+dim+32);}
}
// Fixed-capacity GEMM also reads inactive rows; initialize those inputs.
__global__ void clear_inactive_hidden(__nv_bfloat16* hidden,const uint32_t* active){
 uint32_t rows=*active,row=blockIdx.x;if(rows<1||rows>32||row<rows)return;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)hidden[row*576+i]=__float2bfloat16_rn(0.F);
}
template<bool Fa3=false,bool AdaptiveRows=false,bool ContextSplit=false>
inline cudaError_t enqueue(cudaStream_t stream,void*const* scratch,const void*const* weights,const void* metadata,void* keys,void* values,const float* cos,const float* sin,uint32_t* status,uint32_t physical,uint32_t context,bool tiled=false,bool grouped_attention=false,void* attention_workspace=nullptr,uint64_t attention_workspace_bytes=0,bool ffn_pipeline=false,void* split_workspace=nullptr){
 if constexpr(ContextSplit) {if(!split_workspace || !grouped_attention || !tiled)return cudaErrorInvalidValue;}
if constexpr(!Fa3) {
#ifndef RILEY_CUDA_ENABLE_FLASHINFER
 if(attention_workspace||attention_workspace_bytes)return cudaErrorNotSupported;
#else
 if((attention_workspace==nullptr)!=(attention_workspace_bytes==0))return cudaErrorInvalidValue;
 if(attention_workspace&&(!grouped_attention||attention_workspace_bytes<riley_flashinfer_decode_workspace_bytes()))return cudaErrorInvalidValue;
#endif
 }
 if(!scratch||!weights||!metadata||!keys||!values||!cos||!sin||!status||!physical||physical>4096||!context||context>4096)return cudaErrorInvalidValue;
 for(int i=0;i<12;++i)if(!scratch[i])return cudaErrorInvalidValue;
 for(int i=0;i<273;++i)if(!weights[i])return cudaErrorInvalidValue;
 auto b=[&](int i){return static_cast<__nv_bfloat16*>(scratch[i]);};
 auto w=[&](int i){return static_cast<const __nv_bfloat16*>(weights[i]);};
 const auto* meta=static_cast<const uint32_t*>(metadata);auto* shape=meta+32;auto* pages=meta+64;auto* active=meta+5;auto* pointwise=meta+3;
 auto err=cudaMemsetAsync(status,0,4,stream);if(err!=cudaSuccess)return err;
if constexpr(Fa3) {
#ifdef RILEY_CUDA_ENABLE_FA3
  err=static_cast<cudaError_t>(riley_fa3_model_metadata_prepare(stream,metadata,32*4+32*1664,attention_workspace,attention_workspace_bytes,physical,32,context,status));
  if(err!=cudaSuccess)return err;
  err=static_cast<cudaError_t>(riley_fa3_model_schedule(stream,attention_workspace,attention_workspace_bytes,32,context));
  if(err!=cudaSuccess)return err;
#else
  return cudaErrorNotSupported;
#endif
 } else {
#ifdef RILEY_CUDA_ENABLE_FLASHINFER
 if(attention_workspace){
  err=static_cast<cudaError_t>(riley_flashinfer_decode_prepare(stream,shape,active,attention_workspace,attention_workspace_bytes,physical,context,status));
  if(err!=cudaSuccess)return err;
 }
#endif
 }
 embedding<<<32,256,0,stream>>>(w(0),b(0),shape,active,status);
 if(tiled)for(int i=273;i<363;++i)if(!weights[i])return cudaErrorInvalidValue;
 for(int layer=0;layer<30;++layer){int base=3+9*layer;
 if(layer==0)riley_prefill_pointwise::norm_rows<<<32,256,0,stream>>>(b(0),nullptr,w(base),nullptr,b(1),0,pointwise,32);
 if constexpr(AdaptiveRows)riley_adaptive_decode::qkv_parts<<<dim3(120,3),32,0,stream>>>(b(1),w(base+1),w(base+2),w(base+3),static_cast<float*>(scratch[7]),active);
 else shared32_qkv_parts<<<dim3(120,3),32,0,stream>>>(b(1),w(base+1),w(base+2),w(base+3),static_cast<float*>(scratch[7]),active);
 auto* lk=static_cast<__nv_bfloat16*>(keys)+uint64_t(layer)*physical*16*192;
 auto* lv=static_cast<__nv_bfloat16*>(values)+uint64_t(layer)*physical*16*192;
 qkv_merge_rope<<<dim3(2,32),256,0,stream>>>(static_cast<float*>(scratch[7]),b(3),lk,lv,cos,sin,pages,shape,active);
if constexpr(Fa3) {
#ifdef RILEY_CUDA_ENABLE_FA3
  err=cudaMemsetAsync(b(4),0,32*1152,stream);if(err!=cudaSuccess)return err;
  err=static_cast<cudaError_t>(riley_fa3_model_attention(stream,b(3),lk,lv,b(4),attention_workspace,attention_workspace_bytes,physical,32,context,0));
  if(err!=cudaSuccess)return err;
#endif
 } else {
#ifdef RILEY_CUDA_ENABLE_FLASHINFER
 if(attention_workspace){
  err=static_cast<cudaError_t>(riley_flashinfer_decode_run(stream,b(3),lk,lv,b(4),attention_workspace,attention_workspace_bytes));
  if(err!=cudaSuccess)return err;
 }else
#endif
 if constexpr(ContextSplit) {
  riley_gqa50_attention::scores<<<dim3(min((context+7)/8,32u),96),32,0,stream>>>(b(3),lk,static_cast<float*>(scratch[7]),shape,pages,active);
  err=riley_split_fp32::enqueue(stream,static_cast<float*>(scratch[7]),lv,static_cast<riley_split_fp32::Partial*>(split_workspace),b(4),shape,pages,active,32,context);
  if(err!=cudaSuccess)return err;
 } else if(grouped_attention)riley_gqa50_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);
 else riley_shared32_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);
 }
 if(grouped_attention){
  if constexpr(AdaptiveRows)riley_adaptive_decode::projection_parts<576,576,128,false><<<dim3(72,5),32,0,stream>>>(b(4),w(base+4),static_cast<float*>(scratch[7]),active);
  else shared32_projection_parts<576,576,128,false><<<dim3(72,5),32,0,stream>>>(b(4),w(base+4),static_cast<float*>(scratch[7]),b(2),active);
  riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(static_cast<const float*>(scratch[7]),b(0),w(base+5),scratch[10],b(1),1,pointwise,32);
 }else{
 enqueue_shared32_projection<576,576,128,false>(stream,b(4),w(base+4),b(2),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<32,256,0,stream>>>(b(2),b(0),w(base+5),scratch[10],b(1),1,pointwise,32);
 }
 if(ffn_pipeline)riley_ffn_pipeline::gate_up<<<192,64,0,stream>>>(b(1),w(273+layer*3),w(274+layer*3),b(11),active);
 else if(grouped_attention&&tiled)riley_gate_v56::split_rows<4><<<192,64,0,stream>>>(b(1),w(273+layer*3),w(274+layer*3),b(11),active);
 else if(tiled)shared32_gate_up_swiglu<true><<<192,64,0,stream>>>(b(1),w(273+layer*3),w(274+layer*3),b(11),active);
 else shared32_gate_up_swiglu<false><<<192,64,0,stream>>>(b(1),w(base+6),w(base+7),b(11),active);
 if(grouped_attention){
  if constexpr(AdaptiveRows)riley_adaptive_decode::projection_parts<576,1536,320,true><<<dim3(72,5),32,0,stream>>>(b(11),w(273+layer*3+2),static_cast<float*>(scratch[7]),active);
  else if(ffn_pipeline)riley_ffn_pipeline::down_parts<<<dim3(72,5),32,0,stream>>>(b(11),w(273+layer*3+2),static_cast<float*>(scratch[7]),active);
  else if(tiled)shared32_projection_parts<576,1536,320,true><<<dim3(72,5),32,0,stream>>>(b(11),w(273+layer*3+2),static_cast<float*>(scratch[7]),b(4),active);
  else shared32_projection_parts<576,1536,320,false><<<dim3(72,5),32,0,stream>>>(b(11),w(base+8),static_cast<float*>(scratch[7]),b(4),active);
  riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(static_cast<const float*>(scratch[7]),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,pointwise,32);
 }else{
 if(tiled)enqueue_shared32_projection<576,1536,320,true>(stream,b(11),w(273+layer*3+2),b(4),static_cast<float*>(scratch[7]),active);
 else enqueue_shared32_projection<576,1536,320,false>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<32,256,0,stream>>>(b(4),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,pointwise,32);
 }
 }
 clear_inactive_hidden<<<32,256,0,stream>>>(b(1),active);
 return cudaGetLastError();
}
}
