#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
// Private primitive: caller validates complete cache/map extents and retains parents.
// No serving admission change. Preserve MODE6 accumulation and causal tile order.
namespace riley_prefill_shape {
__device__ float exponential(float score,float maximum){
 return exp2f(fmaf(score,1.4426950408889634F,-maximum*1.4426950408889634F));
}
__device__ uint32_t pair(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b))<<16);}
__device__ void mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ int cache_index(int token,int head,int dim,const uint32_t* blocks){
 return blocks ? ((blocks[token/16]*3+head)*16+token%16)*64+dim : (token*3+head)*64+dim;
}
__global__ void attention_shape(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const int* dynamic_n,const uint32_t* blocks,const uint32_t* live_rows=nullptr){
 if(live_rows){uint32_t live=*live_rows;if(!live||live>static_cast<uint32_t>(rows))return;rows=live;}
 if(blockIdx.x>=static_cast<uint32_t>(rows))return;
 if(dynamic_n)n=*dynamic_n+1;
 int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;
 int row=blockIdx.x,kvh=blockIdx.y,qh=kvh*3+warp,count=n-rows+row+1;
 int qb=(row*9+qh)*64;
 if(n<rows || n>4096){out[qb+lane]=__float2bfloat16_rn(CUDART_NAN_F);out[qb+lane+32]=__float2bfloat16_rn(CUDART_NAN_F);return;}
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
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  float local_den=0.;
  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[warp][i],mx):0.;
   local_den+=p;probs[warp][i]=__float2bfloat16_rn(p);
  }
  for(int offset=16;offset>0;offset>>=1)local_den+=__shfl_xor_sync(0xffffffff,local_den,offset);
  local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=exponential(scores[warp][i],mx);}
  den=local_den;
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
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<8;++block)for(int j=0;j<2;++j)
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}

inline cudaError_t launch(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* pages,uint32_t start,uint32_t rows,uint32_t context){
 if(!q||!k||!v||!out||!pages||!rows||rows>1024||!context||context>4096||start>=context||rows>context-start)return cudaErrorInvalidValue;
 attention_shape<<<dim3(rows,3),96,0,stream>>>(q,k,v,out,rows,start+rows,nullptr,pages);
 return cudaGetLastError();
}
} // namespace riley_prefill_shape
