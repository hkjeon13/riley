// V2 isolated numerical experiment. Compile with the frozen attention TU flags,
// including --use_fast_math. No production file includes this translation unit.
// Both variants use the mixed FMA pair seen in the frozen SM89 oracle SASS.
// Variant 0 retains v1 scalar table conversion. Variant 1 changes only that
// conversion to packed round-to-nearest-even; RZ in SASS is a zero register.
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
template<int Mode>
__device__ __forceinline__ float rope_probe_table(float value){
 if constexpr(Mode==0)return rope_probe_widen(rope_probe_round(value));
 else{
  uint32_t packed;
  // PTX places the second source in the low half and +0 in the high half.
  // No .ftz/.relu/saturation or signed-zero normalization is permitted.
  asm volatile("cvt.rn.bf16x2.f32 %0, 0f00000000, %1;" : "=r"(packed) : "f"(value));
  return __uint_as_float(packed<<16);
 }
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
__global__ void rope_probe_explicit_v2(const uint16_t* q,const uint16_t* k,const uint16_t* v,
 uint16_t* qo,uint16_t* keys,uint16_t* values,const float* cos,const float* sin,const uint32_t* metadata){
 const uint32_t position=metadata[1];
 if(position<128||position>=160)return;
 const int i=threadIdx.x+blockIdx.x*blockDim.x;
 if(i>=384)return;
 const int head=i/32,dim=i%32,base=(head<9?head:head-9)*64;
 const uint16_t* src=head<9?q:k;
 const float a=rope_probe_widen(src[base+dim]),b=rope_probe_widen(src[base+dim+32]);
 const float c=rope_probe_table<Order>(cos[position*32+dim]);
 const float s=rope_probe_table<Order>(sin[position*32+dim]);
 // Oracle SASS rounds b*s for the first result and b*c for the second.
 // Keep non-FTZ explicit operations even under the attention fast-math flags.
 const float first=rope_probe_fma(a,c,rope_probe_negate(rope_probe_mul(b,s)));
 const float second=rope_probe_fma(a,s,rope_probe_mul(b,c));
 const uint16_t x=rope_probe_round(first),y=rope_probe_round(second);
 if(head<9){qo[base+dim]=x;qo[base+dim+32]=y;}
 else{
  const uint32_t physical=metadata[4+position/16];
  const uint32_t dst=((physical*3+head-9)*16+position%16)*64+dim;
  keys[dst]=x;keys[dst+32]=y;
  values[dst]=v[base+dim];values[dst+32]=v[base+dim+32];
 }
}

extern "C" cudaError_t riley_rope_probe_explicit_v2(int order,cudaStream_t stream,
 const void* q,const void* k,const void* v,void* qo,void* keys,void* values,
 const void* cos,const void* sin,const void* metadata){
 if(order==0)rope_probe_explicit_v2<0><<<2,256,0,stream>>>((const uint16_t*)q,(const uint16_t*)k,(const uint16_t*)v,
  (uint16_t*)qo,(uint16_t*)keys,(uint16_t*)values,(const float*)cos,(const float*)sin,(const uint32_t*)metadata);
 else if(order==1)rope_probe_explicit_v2<1><<<2,256,0,stream>>>((const uint16_t*)q,(const uint16_t*)k,(const uint16_t*)v,
  (uint16_t*)qo,(uint16_t*)keys,(uint16_t*)values,(const float*)cos,(const float*)sin,(const uint32_t*)metadata);
 else return cudaErrorInvalidValue;
 return cudaGetLastError();
}
