#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
// Internal V3 pointwise operators; the retained owner validates parent extents.
namespace riley_merge_norm_v56 {
__device__ __forceinline__ void merge_norm_row(const float* parts,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode,const uint32_t* shape,uint32_t capacity,uint32_t row,float* sums){
 const uint32_t live=shape[2];if(capacity!=32||!live||live>capacity||row>=live)return;
 parts+=row*576;out+=row*576;
 if(mode==1){b=static_cast<const __nv_bfloat16*>(b)+row*576;residual=static_cast<float*>(residual)+row*576;}
 if(mode==2){b=static_cast<const float*>(b)+row*576;residual=static_cast<__nv_bfloat16*>(residual)+row*576;}

 int tid=threadIdx.x,lane=tid%32;float x[4]={};
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576){float value=0.F;
  #pragma unroll
  for(int chunk=0;chunk<5;++chunk)value+=parts[chunk*32*576+i];
  x[j]=__bfloat162float(__float2bfloat16_rn(value));
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
__global__ void merge_norm(const float* parts,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode,const uint32_t* shape,uint32_t capacity){
 __shared__ float sums[8];
 merge_norm_row(parts,b,w,residual,out,mode,shape,capacity,blockIdx.x,sums);
}

}
