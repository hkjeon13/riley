#include "ffi_internal.hpp"
// Diagnostic SM89 BF16 tensor-core attention, fixed head dimension 64.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef MODE
#define MODE 6
#endif
#if MODE != 6
#error "This experiment requires accepted MODE6 arithmetic"
#endif
namespace riley_multisequence_attention {
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
__global__ void attention_rows_v1(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* packet){
 const uint32_t row=blockIdx.z;
 out+=row*576;
 const uint32_t active_rows=packet[6]; // fresh common header, byte24
 // Uniform inactive return precedes row metadata/Q/KV loads and every CTA barrier.
 if(row>=active_rows){
  if(threadIdx.x<32)out[blockIdx.x*64+blockIdx.y*32+threadIdx.x]=__ushort_as_bfloat16(0);
  return;
 }
 q+=row*576;
 const uint32_t* metadata=packet+32+row*32; // byte128 + row*128: C10 view
 const uint32_t position=metadata[1];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 const int qh=blockIdx.x,kvh=qh/3,first_block=blockIdx.y*4+warp*2,qb=qh*64;
 if(position<128||position>=160){
  if(warp==0)out[qb+blockIdx.y*32+lane]=__float2bfloat16_rn(CUDART_NAN_F);
  return;
 }
 const int count=static_cast<int>(position)+1;
 const uint32_t* blocks=metadata+4;
 __shared__ float scores[128];
 __shared__ __nv_bfloat16 probs[128];
 __shared__ float alpha_by_lane[32],inverse_by_lane[32];
 float maximum=-CUDART_INF_F,den=0.;float accum[2][4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  for(int token=begin+8*warp;token<end;token+=16){
   float d[4]={};
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);
    uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);
    int kb=token+group<end?cache_index(token+group,kvh,depth,blocks):0;
    uint32_t b=token+group<end?pair(k[kb+2*t],k[kb+2*t+1]):0;
    uint32_t bb=token+group<end?pair(k[kb+2*t+8],k[kb+2*t+9]):0;
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[token-begin+2*t+j]=d[j]*.125F;}
  }
  // Includes the idle second warp when the tail has at most eight tokens.
  __syncthreads();
  if(warp==0){
   float mx=maximum;
   for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[i]);
   for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
   float alpha=exp2f((maximum-mx)*1.4426950408889634F);
   alpha_by_lane[lane]=alpha;
   for(int i=lane;i<128;i+=32){
    float p=i<end-begin?exponential(scores[i],mx):0.;
    probs[i]=__float2bfloat16_rn(p);
    scores[i]=p;
   }
   __syncwarp();
   // Warp zero retains the accepted per-lane j/z addition order and keeps
   // the original FP32 probabilities, not their BF16 PV representation.
   float local_den=den*alpha;
   for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
     if(i<end-begin)local_den+=scores[i];}
   den=local_den;
   maximum=mx;
  }
  __syncthreads(); // Publish probabilities and the corresponding lane's alpha.
  const float alpha=alpha_by_lane[lane];
  for(int block=0;block<2;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=pair(probs[pi+2*t],probs[pi+2*t+1]);
    uint32_t aa=pair(probs[pi+2*t+8],probs[pi+2*t+9]);
    int dim=(first_block+block)*8+group;
    auto val=[&](int pos){return pos<end?v[cache_index(pos,kvh,dim,blocks)]:zero;};
    uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
  __syncthreads(); // Finish both PV readers before reusing scores/probabilities.
 }
 if(warp==0){
  den+=__shfl_xor_sync(0xffffffff,den,2);
  den+=__shfl_xor_sync(0xffffffff,den,1);
  inverse_by_lane[lane]=1.0F/den;
 }
 __syncthreads();
 const float inverse=inverse_by_lane[lane];
 if(group==0)for(int block=0;block<2;++block)for(int j=0;j<2;++j)
  out[qb+(first_block+block)*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}
cudaError_t enqueue_rows(cudaStream_t s,const void* q,const void* k,const void* v,void* out,const void* packet,uint32_t bucket) noexcept {
 if(!q||!k||!v||!out||!packet||(bucket!=1&&bucket!=2&&bucket!=4))return cudaErrorInvalidValue;
 attention_rows_v1<<<dim3(9,2,bucket),64,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,(const uint32_t*)packet);
 return cudaGetLastError();
}
} // namespace riley_multisequence_attention
