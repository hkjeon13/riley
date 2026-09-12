#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
// Internal V3 pointwise operators; the retained owner validates parent extents.
namespace riley_prefill_pointwise {
__global__ void norm_rows(const __nv_bfloat16* a,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode,const uint32_t* shape,uint32_t capacity){
 const uint32_t row=blockIdx.x,live=shape[2];if(!live||live>capacity||row>=live)return;
 a+=row*576;out+=row*576;
 if(mode==1){b=static_cast<const __nv_bfloat16*>(b)+row*576;residual=static_cast<float*>(residual)+row*576;}
 if(mode==2){b=static_cast<const float*>(b)+row*576;residual=static_cast<__nv_bfloat16*>(residual)+row*576;}

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
__global__ void swiglu_rows(const __nv_bfloat16* g,const __nv_bfloat16* u,__nv_bfloat16* out,const uint32_t* shape,uint32_t capacity){
 const uint32_t row=blockIdx.y,live=shape[2];if(!live||live>capacity||row>=live)return;g+=row*1536;u+=row*1536;out+=row*1536;

 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<1536){float x=__bfloat162float(g[i]);out[i]=__float2bfloat16_rn((x/(1.F+expf(-x)))*__bfloat162float(u[i]));}
}

__global__ void embedding_rows(const __nv_bfloat16* weights,const uint32_t* tokens,__nv_bfloat16* out,const uint32_t* shape,uint32_t capacity,uint32_t vocabulary,uint32_t* status,const uint32_t* stage=nullptr){
 uint32_t row=blockIdx.x,live=shape[2];if(!live||live>capacity){if(threadIdx.x==0)atomicOr(status,2U);return;}if(row>=live)return;
 if(stage&&*stage==1&&live!=1){if(threadIdx.x==0)atomicOr(status,2U);return;}
 uint32_t token=stage&&*stage==1?shape[0]:tokens[row];if(token>=vocabulary){if(threadIdx.x==0)atomicOr(status,1U);return;}
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)out[row*576+i]=weights[token*576+i];
}
__global__ void select_hidden(const __nv_bfloat16* rows,__nv_bfloat16* selected,const uint32_t* shape,uint32_t capacity,const uint32_t* status,uint32_t* publish){
 uint32_t live=shape[2],row=shape[11];bool ready=*status==0&&live>0&&live<=capacity&&row<live;
 if(threadIdx.x==0)*publish=ready?1:0;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)selected[i]=ready?rows[row*576+i]:__float2bfloat16_rn(0.F);
}
} // namespace riley_prefill_pointwise
