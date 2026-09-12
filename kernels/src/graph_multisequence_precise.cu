#include "ffi_internal.hpp"
#include <cuda_bf16.h>
namespace riley_multisequence_precise {
__global__ void swiglu_rows(const __nv_bfloat16* packed,__nv_bfloat16* out,const uint32_t* packet){
 const uint32_t row=blockIdx.y,active_rows=packet[6];
 out+=row*1536;
 if(row>=active_rows){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536)out[i]=__ushort_as_bfloat16(0);return;}
 const __nv_bfloat16* g=packed+row*3072;const __nv_bfloat16* u=g+1536;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536){float x=__bfloat162float(g[i]);out[i]=__float2bfloat16_rn((x/(1.F+expf(-x)))*__bfloat162float(u[i]));}
}
cudaError_t enqueue_swiglu_rows(cudaStream_t stream,const void* packed,void* out,const void* packet,uint32_t bucket) noexcept {
 if(!packed||!out||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 swiglu_rows<<<dim3(6,bucket),256,0,stream>>>((const __nv_bfloat16*)packed,(__nv_bfloat16*)out,(const uint32_t*)packet);
 return cudaGetLastError();
}
}
