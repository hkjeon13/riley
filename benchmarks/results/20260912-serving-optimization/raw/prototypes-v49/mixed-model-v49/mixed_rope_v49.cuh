// Internal variable-prefill primitive, not a serving entry point.
// Caller retains all parents and validates a unique logical-to-physical page map,
// its full extent, and allocation sizes before enqueue; no host publication here.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__global__ void packed_prefill_rope_kv(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,const float* cos,const float* sin,
 const uint32_t* meta,uint32_t capacity){
 const uint32_t row=blockIdx.y,active=meta[5],total=meta[9];if(!active||active>32||!total||total>capacity||row>=total)return;
 uint32_t owner=0;for(;owner<active;++owner){const auto* s=meta+32+owner*416;if(row>=s[16]&&row-s[16]<s[2])break;}if(owner==active)return;
 const auto* shape=meta+32+owner*416;const auto* pages=shape+32;const uint32_t pos=shape[4]+row-shape[16];
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;
 int head=i/32,dim=i%32;
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