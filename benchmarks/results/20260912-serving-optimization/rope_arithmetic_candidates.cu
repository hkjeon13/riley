// Isolated numerical experiment. Compile with the frozen attention TU flags,
// including --use_fast_math. No production file includes this translation unit.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float rope_probe_widen(uint16_t bits){
 return __uint_as_float(uint32_t(bits)<<16);
}
__device__ __forceinline__ uint16_t rope_probe_round(float value){
 uint16_t bits;
 asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
 return bits;
}
__device__ __forceinline__ float rope_probe_mul(float a,float b){
 float value;
 asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a),"f"(b));
 return value;
}
__device__ __forceinline__ float rope_probe_fma(float a,float b,float c){
 float value;
 asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(value) : "f"(a),"f"(b),"f"(c));
 return value;
}
__device__ __forceinline__ float rope_probe_negate(float value){
 return __uint_as_float(__float_as_uint(value)^0x80000000u);
}

template<int Order>
__global__ void rope_probe_explicit(const uint16_t* q,const uint16_t* k,const uint16_t* v,
 uint16_t* qo,uint16_t* keys,uint16_t* values,const float* cos,const float* sin,const uint32_t* metadata){
 const uint32_t position=metadata[1];
 if(position<128||position>=160)return;
 const int i=threadIdx.x+blockIdx.x*blockDim.x;
 if(i>=384)return;
 const int head=i/32,dim=i%32,base=(head<9?head:head-9)*64;
 const uint16_t* src=head<9?q:k;
 const float a=rope_probe_widen(src[base+dim]),b=rope_probe_widen(src[base+dim+32]);
 const float c=rope_probe_widen(rope_probe_round(cos[position*32+dim]));
 const float s=rope_probe_widen(rope_probe_round(sin[position*32+dim]));
 float first,second;
 if constexpr(Order==0){
  // Round the sine product, then fuse the cosine product and final add.
  first=rope_probe_fma(a,c,rope_probe_negate(rope_probe_mul(b,s)));
  second=rope_probe_fma(b,c,rope_probe_mul(a,s));
 }else{
  // Round the cosine product, then fuse the sine product and final add.
  first=rope_probe_fma(rope_probe_negate(b),s,rope_probe_mul(a,c));
  second=rope_probe_fma(a,s,rope_probe_mul(b,c));
 }
 const uint16_t x=rope_probe_round(first),y=rope_probe_round(second);
 if(head<9){qo[base+dim]=x;qo[base+dim+32]=y;}
 else{
  const uint32_t physical=metadata[4+position/16];
  const uint32_t dst=((physical*3+head-9)*16+position%16)*64+dim;
  keys[dst]=x;keys[dst+32]=y;
  values[dst]=v[base+dim];values[dst+32]=v[base+dim+32];
 }
}

extern "C" cudaError_t riley_rope_probe_explicit(int order,cudaStream_t stream,
 const void* q,const void* k,const void* v,void* qo,void* keys,void* values,
 const void* cos,const void* sin,const void* metadata){
 if(order==0)rope_probe_explicit<0><<<2,256,0,stream>>>((const uint16_t*)q,(const uint16_t*)k,(const uint16_t*)v,
  (uint16_t*)qo,(uint16_t*)keys,(uint16_t*)values,(const float*)cos,(const float*)sin,(const uint32_t*)metadata);
 else if(order==1)rope_probe_explicit<1><<<2,256,0,stream>>>((const uint16_t*)q,(const uint16_t*)k,(const uint16_t*)v,
  (uint16_t*)qo,(uint16_t*)keys,(uint16_t*)values,(const float*)cos,(const float*)sin,(const uint32_t*)metadata);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
