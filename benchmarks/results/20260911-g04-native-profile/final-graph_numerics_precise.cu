#include "ffi_internal.hpp"
#include <cuda_bf16.h>
__global__ void compiled_norm(const __nv_bfloat16* a,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode){
 __shared__ float sums[8];int tid=threadIdx.x,lane=tid%32;float x[4]={};
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576){x[j]=__bfloat162float(a[i]);
  if(mode==1)x[j]+=__bfloat162float(((const __nv_bfloat16*)b)[i]);if(mode==2)x[j]+=((const float*)b)[i];
  if(mode==1)((float*)residual)[i]=x[j];if(mode==2)((__nv_bfloat16*)residual)[i]=__float2bfloat16_rn(x[j]);}}
 float sum=x[1]*x[1];sum=fmaf(x[0],x[0],sum);sum=fmaf(x[2],x[2],sum);sum=fmaf(x[3],x[3],sum);
 for(int step=16;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);
 if(lane==0)sums[tid/32]=sum;__syncthreads();
 if(tid<32){sum=lane<8?sums[lane]:0.;for(int step=4;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);if(lane==0)sums[0]=sum;}__syncthreads();
 float mean=fmaf(sums[0],1.F/576.F,1e-5F),inv;
 asm("rsqrt.approx.ftz.f32 %0, %1;":"=f"(inv):"f"(mean));
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576)out[i]=__float2bfloat16_rn((x[j]*inv)*__bfloat162float(w[i]));}
}

__global__ void compiled_rope(const __nv_bfloat16* q,const __nv_bfloat16* k,__nv_bfloat16* qo,__nv_bfloat16* ko,const float* cos,const float* sin,const uint32_t* pos){
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=384)return;int head=i/32,dim=i%32;
 float c=__bfloat162float(__float2bfloat16_rn(cos[*pos*32+dim])),s=__bfloat162float(__float2bfloat16_rn(sin[*pos*32+dim]));
 const __nv_bfloat16* src=head<9?q:k;__nv_bfloat16* dst=head<9?qo:ko;int base=(head<9?head:head-9)*64;
 float a=__bfloat162float(src[base+dim]),b=__bfloat162float(src[base+dim+32]);
 dst[base+dim]=__float2bfloat16_rn(a*c-b*s);dst[base+dim+32]=__float2bfloat16_rn(b*c+a*s);
}
__global__ void compiled_swiglu(const __nv_bfloat16* g,const __nv_bfloat16* u,__nv_bfloat16* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536){float x=__bfloat162float(g[i]);out[i]=__float2bfloat16_rn((x/(1.F+expf(-x)))*__bfloat162float(u[i]));}
}
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_norm(cudaStream_t s,const void* a,const void* b,const void* w,void* residual,void* out,int mode) noexcept {
 compiled_norm<<<1,256,0,s>>>((const __nv_bfloat16*)a,b,(const __nv_bfloat16*)w,residual,(__nv_bfloat16*)out,mode);return cudaGetLastError();}
cudaError_t enqueue_compiled_rope(cudaStream_t s,const void* q,const void* k,void* qo,void* ko,const void* cos,const void* sin,const void* pos) noexcept {
 compiled_rope<<<2,256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(__nv_bfloat16*)qo,(__nv_bfloat16*)ko,(const float*)cos,(const float*)sin,(const uint32_t*)pos);return cudaGetLastError();}
cudaError_t enqueue_compiled_swiglu(cudaStream_t s,const void* g,const void* u,void* out) noexcept {
 compiled_swiglu<<<6,256,0,s>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,(__nv_bfloat16*)out);return cudaGetLastError();}
}
