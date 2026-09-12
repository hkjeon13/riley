#include "ffi_internal.hpp"
// Diagnostic SM89 BF16 tensor-core attention, fixed head dimension 64.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef MODE
#define MODE 6
#endif
__device__ float exponential(float score,float maximum){
#if MODE == 1 || MODE >= 4
 return exp2f(fmaf(score,1.4426950408889634F,-maximum*1.4426950408889634F));
#elif MODE == 2
 float x=(score-maximum)*1.4426950408889634F,y;
 asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
#else
 return exp2f((score-maximum)*1.4426950408889634F);
#endif
}
__device__ uint32_t pair(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b))<<16);}
__device__ void mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ int cache_index(int token,int head,int dim,const uint32_t* blocks){
 return blocks ? ((blocks[token/16]*3+head)*16+token%16)*64+dim : (token*3+head)*64+dim;
}
__global__ void attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const int* dynamic_n,const uint32_t* blocks){
 if(dynamic_n)n=*dynamic_n+1;
 int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 int row=blockIdx.x,kvh=blockIdx.y,qh=kvh*3+warp,count=n-rows+row+1;
 int qb=(row*9+qh)*64;
 if(n<rows || n>160){out[qb+lane]=__float2bfloat16_rn(CUDART_NAN_F);out[qb+lane+32]=__float2bfloat16_rn(CUDART_NAN_F);return;}
 __shared__ float scores[3][128];
 __shared__ __nv_bfloat16 probs[3][128];
 float maximum=-CUDART_INF_F,den=0.;float accum[8][4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  for(int token=begin;token<end;token+=8){
   float d[4]={};
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);
    uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);
    int kb=token+group<end?cache_index(token+group,kvh,depth,blocks):0;
    uint32_t b=token+group<end?pair(k[kb+2*t],k[kb+2*t+1]):0;
    uint32_t bb=token+group<end?pair(k[kb+2*t+8],k[kb+2*t+9]):0;
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[warp][token-begin+2*t+j]=d[j]*.125F;}
  }
  __syncwarp();
  float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[warp][i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  #if MODE >= 4
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
#else
  float alpha=exponential(maximum,mx);
#endif
  float local_den=0.;
  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[warp][i],mx):0.;
   local_den+=p;probs[warp][i]=__float2bfloat16_rn(p);
  }
  for(int offset=16;offset>0;offset>>=1)local_den+=__shfl_xor_sync(0xffffffff,local_den,offset);
#if MODE == 3 || MODE == 5 || MODE == 6
#if MODE == 6
  local_den=den*alpha;
#else
  local_den=0.;
#endif
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=exponential(scores[warp][i],mx);}
#if MODE != 6
  local_den+=__shfl_xor_sync(0xffffffff,local_den,2);
  local_den+=__shfl_xor_sync(0xffffffff,local_den,1);
#endif
#endif
#if MODE == 6
  den=local_den;
#else
  den=den*alpha+local_den;
#endif
  maximum=mx;
  __syncwarp();
  for(int block=0;block<8;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=pair(probs[warp][pi+2*t],probs[warp][pi+2*t+1]);
    uint32_t aa=pair(probs[warp][pi+2*t+8],probs[warp][pi+2*t+9]);
    int dim=block*8+group;
    auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:zero;};
    uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
  __syncwarp();
 }
#if MODE == 6
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
#endif
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<8;++block)for(int j=0;j<2;++j)
#if MODE >= 4
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
#else
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]/den);
#endif
}

__global__ void compiled_norm(const __nv_bfloat16* a,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode){
 __shared__ float sums[1024];int i=threadIdx.x;float x=0.;
 if(i<576){x=__bfloat162float(a[i]);if(mode==1)x+=__bfloat162float(((const __nv_bfloat16*)b)[i]);if(mode==2)x+=((const float*)b)[i];
 if(mode==1)((float*)residual)[i]=x;if(mode==2)((__nv_bfloat16*)residual)[i]=__float2bfloat16_rn(x);}
 sums[i]=x*x;__syncthreads();for(int step=512;step;step>>=1){if(i<step)sums[i]+=sums[i+step];__syncthreads();}
 if(i<576)out[i]=__float2bfloat16_rn((x*rsqrtf(sums[0]/576.F+1e-5F))*__bfloat162float(w[i]));
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
 compiled_norm<<<1,1024,0,s>>>((const __nv_bfloat16*)a,b,(const __nv_bfloat16*)w,residual,(__nv_bfloat16*)out,mode);return cudaGetLastError();}
cudaError_t enqueue_compiled_rope(cudaStream_t s,const void* q,const void* k,void* qo,void* ko,const void* cos,const void* sin,const void* pos) noexcept {
 compiled_rope<<<2,256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(__nv_bfloat16*)qo,(__nv_bfloat16*)ko,(const float*)cos,(const float*)sin,(const uint32_t*)pos);return cudaGetLastError();}
cudaError_t enqueue_compiled_swiglu(cudaStream_t s,const void* g,const void* u,void* out) noexcept {
 compiled_swiglu<<<6,256,0,s>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,(__nv_bfloat16*)out);return cudaGetLastError();}
cudaError_t enqueue_compiled_attention(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* metadata) noexcept {
 auto* m=(const uint8_t*)metadata;attention<<<dim3(1,3),96,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,1,160,(const int*)(m+4),(const uint32_t*)(m+16));return cudaGetLastError();}
}
