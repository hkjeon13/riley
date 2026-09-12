#include "ffi_internal.hpp"
#include <cuda_bf16.h>
namespace riley_multisequence_precise {
__global__ void rope_rows(const __nv_bfloat16* packed,
    __nv_bfloat16* qo,__nv_bfloat16* keys,__nv_bfloat16* values,
    const float* cos,const float* sin,const uint32_t* packet){
 const uint32_t row=blockIdx.y,active_rows=packet[6];
 qo+=row*576;
 if(row>=active_rows){
  for(int i=threadIdx.x+blockIdx.x*blockDim.x;i<576;i+=blockDim.x*gridDim.x)qo[i]=__ushort_as_bfloat16(0);
  return;
 }
 const __nv_bfloat16* q=packed+row*960;const __nv_bfloat16* k=q+576;const __nv_bfloat16* v=q+768;
 const uint32_t* metadata=packet+32+row*32;
 const uint32_t* pos=metadata+1;
 if(*pos<128||*pos>=160)return;
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[*pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[*pos*32+dim]));
 const __nv_bfloat16* src=head<9?q:k;int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 const __nv_bfloat16 first=__float2bfloat16_rn(a*c-b*s),second=__float2bfloat16_rn(b*c+a*s);
 if(head<9){qo[base+dim]=first;qo[base+dim+32]=second;}
 else{
  const uint32_t physical=metadata[4+*pos/16];
  const uint32_t destination=((physical*3+head-9)*16+*pos%16)*64+dim;
  keys[destination]=first;keys[destination+32]=second;
  values[destination]=v[base+dim];values[destination+32]=v[base+dim+32];
 }
}
__global__ void swiglu_rows(const __nv_bfloat16* packed,__nv_bfloat16* out,const uint32_t* packet){
 const uint32_t row=blockIdx.y,active_rows=packet[6];
 out+=row*1536;
 if(row>=active_rows){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536)out[i]=__ushort_as_bfloat16(0);return;}
 const __nv_bfloat16* g=packed+row*3072;const __nv_bfloat16* u=g+1536;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536){float x=__bfloat162float(g[i]);out[i]=__float2bfloat16_rn((x/(1.F+expf(-x)))*__bfloat162float(u[i]));}
}
cudaError_t enqueue_rope_rows(cudaStream_t stream,const void* packed,void* qo,void* keys,void* values,const void* cos,const void* sin,const void* packet,uint32_t bucket) noexcept {
 if(!packed||!qo||!keys||!values||!cos||!sin||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 rope_rows<<<dim3(2,bucket),256,0,stream>>>((const __nv_bfloat16*)packed,(__nv_bfloat16*)qo,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(const float*)cos,(const float*)sin,(const uint32_t*)packet);
 return cudaGetLastError();
}
cudaError_t enqueue_swiglu_rows(cudaStream_t stream,const void* packed,void* out,const void* packet,uint32_t bucket) noexcept {
 if(!packed||!out||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 swiglu_rows<<<dim3(6,bucket),256,0,stream>>>((const __nv_bfloat16*)packed,(__nv_bfloat16*)out,(const uint32_t*)packet);
 return cudaGetLastError();
}
}
