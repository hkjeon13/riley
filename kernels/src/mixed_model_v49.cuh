#include "mixed_attention_v49.cuh"
#include "mixed_rope_v49.cuh"
#pragma once
#include "prefill_shape_projection.cuh"
#include "prefill_fused_gate_v51.cuh"
#include "prefill_shape_rope_kv.cuh"
#include "prefill_shape_attention.cuh"
#include "prefill_shape_pointwise.cuh"
#include "decode_tiled.cuh"
// Borrowed 30-layer SmolLM2 BF16 prefill sequence. Caller must validate V7 packet,
// all allocation extents/aliases and hold the resource ledger until completion.
// This enqueues work; it does not authorize scheduler settlement or serving output.
__global__ void mixed_select_hidden_v7(const __nv_bfloat16* rows,__nv_bfloat16* selected,const uint32_t* meta,uint32_t capacity,const uint32_t* status,uint32_t* publish){
 uint32_t owner=blockIdx.x,active=meta[5],total=meta[9];bool ready=false;uint32_t at=0;
 if(active>=1&&active<=32&&owner<active&&total>0&&total<=capacity){const auto* shape=meta+32+owner*416;uint32_t n=shape[2],offset=shape[16],row=shape[11];ready=*status==0&&n>0&&offset<=total&&n<=total-offset&&row<n;at=offset+row;}
 if(publish&&threadIdx.x==0)publish[owner]=ready?1:0;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)selected[owner*576+i]=ready?rows[at*576+i]:__float2bfloat16_rn(0.F);
}
template<uint32_t WireRows=8>
inline cudaError_t enqueue_mixed_model_v7(cudaStream_t stream,void*const* scratch,const void*const* weights,
 const void* metadata,void* keys,void* values,const void* cos,const void* sin,void* selected,
 uint32_t* status,uint32_t* publish,uint32_t capacity,uint32_t physical,bool tiled=false){
 static_assert(WireRows==8||WireRows==16||WireRows==32,"wire capacity");
 if(!scratch||!weights||!metadata||!keys||!values||!cos||!sin||!selected||!status||!capacity||capacity>1024||!physical||physical>4096)return cudaErrorInvalidValue;
 for(int i=0;i<12;++i)if(!scratch[i])return cudaErrorInvalidValue;
 for(int i=0;i<273;++i)if(!weights[i])return cudaErrorInvalidValue;
 auto b=[&](int i){return static_cast<__nv_bfloat16*>(scratch[i]);};
 auto w=[&](int i){return static_cast<const __nv_bfloat16*>(weights[i]);};
 const auto* meta=static_cast<const uint32_t*>(metadata);
 auto shape=meta+7; // shape[2] is total packed token count, not one owner count.
 auto pages=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+256);
 auto tokens=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+(128+WireRows*1664));
 auto err=cudaMemsetAsync(status,0,4,stream);if(err!=cudaSuccess)return err;
 riley_prefill_pointwise::embedding_rows<<<capacity,256,0,stream>>>(w(0),tokens,b(0),shape,capacity,49152,status,nullptr);
 for(int layer=0;layer<30;++layer){int base=3+9*layer;
  if(layer==0)riley_prefill_pointwise::norm_rows<<<capacity,256,0,stream>>>(b(0),nullptr,w(base),nullptr,b(1),0,shape,capacity);
  if(capacity==1)enqueue_decode_projection<576,576,192>(stream,b(1),w(base+1),b(2),static_cast<float*>(scratch[7]));
  else gemm_prefill_shape_vector<576,576,192,2><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(1),w(base+1),b(2),capacity,shape+2);
  if(capacity==1)enqueue_decode_projection<192,576,192>(stream,b(1),w(base+2),b(5),static_cast<float*>(scratch[7]));
  else gemm_prefill_shape_vector<192,576,192,1><<<dim3(24,(capacity+15)/16),32,0,stream>>>(b(1),w(base+2),b(5),capacity,shape+2);
  if(capacity==1)enqueue_decode_projection<192,576,192>(stream,b(1),w(base+3),b(6),static_cast<float*>(scratch[7]));
  else gemm_prefill_shape_vector<192,576,192,1><<<dim3(24,(capacity+15)/16),32,0,stream>>>(b(1),w(base+3),b(6),capacity,shape+2);
  auto* lk=static_cast<__nv_bfloat16*>(keys)+uint64_t(layer)*physical*16*192;
  auto* lv=static_cast<__nv_bfloat16*>(values)+uint64_t(layer)*physical*16*192;
  mixed_rope_kv_v7<<<dim3(2,capacity),256,0,stream>>>(b(2),b(5),b(6),b(3),lk,lv,static_cast<const float*>(cos),static_cast<const float*>(sin),meta,capacity);
  riley_mixed_attention::mapped_attention<true><<<dim3(capacity,9),32,0,stream>>>(b(3),lk,lv,b(4),capacity,meta);
  if(capacity==1)enqueue_decode_projection<576,576,128>(stream,b(4),w(base+4),b(2),static_cast<float*>(scratch[7]));
  else gemm_prefill_shape_vector<576,576,128,2><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(4),w(base+4),b(2),capacity,shape+2);
  riley_prefill_pointwise::norm_rows<<<capacity,256,0,stream>>>(b(2),b(0),w(base+5),scratch[10],b(1),1,shape,capacity);
  if(capacity>1&&tiled){
   riley_prefill51::gate_up<4><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+6),w(base+7),b(11),capacity,shape+2);
  }else{
  if(capacity==1&&tiled)enqueue_tile_projection<1536,576,0>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]));
  else if(capacity==1)enqueue_decode_projection<1536,576,0>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]));
  else if(tiled)gemm_prefill_shape_vector<1536,576,0,4,true><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+6),b(8),capacity,shape+2);
  else gemm_prefill_shape_vector<1536,576,0,4><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+6),b(8),capacity,shape+2);
  if(capacity==1&&tiled)enqueue_tile_projection<1536,576,0>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]));
  else if(capacity==1)enqueue_decode_projection<1536,576,0>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]));
  else if(tiled)gemm_prefill_shape_vector<1536,576,0,4,true><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+7),b(9),capacity,shape+2);
  else gemm_prefill_shape_vector<1536,576,0,4><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+7),b(9),capacity,shape+2);
  riley_prefill_pointwise::swiglu_rows<<<dim3(6,capacity),256,0,stream>>>(b(8),b(9),b(11),shape,capacity);
  }
  if(capacity==1&&tiled)enqueue_tile_projection<576,1536,320>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]));
  else if(capacity==1)enqueue_decode_projection<576,1536,320>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]));
  else if(tiled)gemm_prefill_shape_vector<576,1536,320,2,true><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(11),w(base+8),b(4),capacity,shape+2);
  else gemm_prefill_shape_vector<576,1536,320,2><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(11),w(base+8),b(4),capacity,shape+2);
  riley_prefill_pointwise::norm_rows<<<capacity,256,0,stream>>>(b(4),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,shape,capacity);
 }
 mixed_select_hidden_v7<<<32,256,0,stream>>>(b(1),static_cast<__nv_bfloat16*>(selected),meta,capacity,status,publish);
 return cudaGetLastError();
}
