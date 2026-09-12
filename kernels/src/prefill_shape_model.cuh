#pragma once
#include "prefill_shape_projection.cuh"
#include "prefill_shape_rope_kv.cuh"
#include "prefill_shape_attention.cuh"
#include "prefill_shape_pointwise.cuh"
#include "decode_tiled.cuh"
// Borrowed 30-layer SmolLM2 BF16 prefill sequence. Caller must validate V3 packet,
// all allocation extents/aliases and hold the resource ledger until completion.
// This enqueues work; it does not authorize scheduler settlement or serving output.
template<uint32_t WireRows=8>
inline cudaError_t enqueue_v3_prefill_model(cudaStream_t stream,void*const* scratch,const void*const* weights,
 const void* metadata,void* keys,void* values,const void* cos,const void* sin,void* selected,
 uint32_t* status,uint32_t* publish,uint32_t capacity,uint32_t physical,bool tiled=false){
 static_assert(WireRows==8||WireRows==16,"wire capacity");
 if(!scratch||!weights||!metadata||!keys||!values||!cos||!sin||!selected||!status||!publish||!capacity||capacity>1024||!physical||physical>4096)return cudaErrorInvalidValue;
 for(int i=0;i<12;++i)if(!scratch[i])return cudaErrorInvalidValue;
 for(int i=0;i<273;++i)if(!weights[i])return cudaErrorInvalidValue;
 auto b=[&](int i){return static_cast<__nv_bfloat16*>(scratch[i]);};
 auto w=[&](int i){return static_cast<const __nv_bfloat16*>(weights[i]);};
 auto shape=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+128);
 auto pages=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+256);
 auto tokens=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+(128+WireRows*1664));
 auto err=cudaMemsetAsync(status,0,4,stream);if(err!=cudaSuccess)return err;
 riley_prefill_pointwise::embedding_rows<<<capacity,256,0,stream>>>(w(0),tokens,b(0),shape,capacity,49152,status,reinterpret_cast<const uint32_t*>(metadata)+4);
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
  prefill_shape_rope_kv<<<dim3(2,capacity),256,0,stream>>>(b(2),b(5),b(6),b(3),lk,lv,static_cast<const float*>(cos),static_cast<const float*>(sin),pages,0,capacity,shape);
  if(capacity==1)riley_decode_shape::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,physical);
  else riley_prefill_shape::attention_shape<<<dim3(capacity,3),96,0,stream>>>(b(3),lk,lv,b(4),capacity,0,reinterpret_cast<const int*>(shape+1),pages,shape+2);
  if(capacity==1)enqueue_decode_projection<576,576,128>(stream,b(4),w(base+4),b(2),static_cast<float*>(scratch[7]));
  else gemm_prefill_shape_vector<576,576,128,2><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(4),w(base+4),b(2),capacity,shape+2);
  riley_prefill_pointwise::norm_rows<<<capacity,256,0,stream>>>(b(2),b(0),w(base+5),scratch[10],b(1),1,shape,capacity);
  if(capacity==1&&tiled)enqueue_tile_projection<1536,576,0>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]));
  else if(capacity==1)enqueue_decode_projection<1536,576,0>(stream,b(1),w(base+6),b(8),static_cast<float*>(scratch[7]));
  else if(tiled)gemm_prefill_shape_vector<1536,576,0,4,true><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+6),b(8),capacity,shape+2);
  else gemm_prefill_shape_vector<1536,576,0,4><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+6),b(8),capacity,shape+2);
  if(capacity==1&&tiled)enqueue_tile_projection<1536,576,0>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]));
  else if(capacity==1)enqueue_decode_projection<1536,576,0>(stream,b(1),w(base+7),b(9),static_cast<float*>(scratch[7]));
  else if(tiled)gemm_prefill_shape_vector<1536,576,0,4,true><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+7),b(9),capacity,shape+2);
  else gemm_prefill_shape_vector<1536,576,0,4><<<dim3(48,(capacity+15)/16),128,0,stream>>>(b(1),w(base+7),b(9),capacity,shape+2);
  riley_prefill_pointwise::swiglu_rows<<<dim3(6,capacity),256,0,stream>>>(b(8),b(9),b(11),shape,capacity);
  if(capacity==1&&tiled)enqueue_tile_projection<576,1536,320>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]));
  else if(capacity==1)enqueue_decode_projection<576,1536,320>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]));
  else if(tiled)gemm_prefill_shape_vector<576,1536,320,2,true><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(11),w(base+8),b(4),capacity,shape+2);
  else gemm_prefill_shape_vector<576,1536,320,2><<<dim3(36,(capacity+15)/16),64,0,stream>>>(b(11),w(base+8),b(4),capacity,shape+2);
  riley_prefill_pointwise::norm_rows<<<capacity,256,0,stream>>>(b(4),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,shape,capacity);
 }
 riley_prefill_pointwise::select_hidden<<<1,256,0,stream>>>(b(1),static_cast<__nv_bfloat16*>(selected),shape,capacity,status,publish);
 return cudaGetLastError();
}
