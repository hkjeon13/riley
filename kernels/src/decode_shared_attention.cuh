#pragma once
#include "decode_shape.cuh"
namespace riley_shared_attention {

// Parallel QK score tiles, preserving the exact four K16 MMA recurrence.
__global__ void scores(const __nv_bfloat16* q,const __nv_bfloat16* k,float* result,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/9;uint32_t active=*live_rows;if(active<1||active>8||row>=active)return;
 shape+=row*416;pages+=row*416;
 q+=row*576;result+=row*9*4096;
 int count=shape[1]+1,token=blockIdx.x*8,head=blockIdx.y%9;
 if(count<1 || count>4096)return;
 for(;token<count;token+=gridDim.x*8){
 int lane=threadIdx.x,g=lane/4,t=lane%4;float d[4]={};
 #pragma unroll
 for(int depth=0;depth<64;depth+=16){
  auto* qp=q+head*64+depth;
  uint32_t a=riley_prefill_shape::pair(qp[2*t],qp[2*t+1]);
  uint32_t aa=riley_prefill_shape::pair(qp[2*t+8],qp[2*t+9]);
  int kb=token+g<count?riley_prefill_shape::cache_index(token+g,head/3,depth,pages):0;
  uint32_t b=token+g<count?riley_prefill_shape::pair(k[kb+2*t],k[kb+2*t+1]):0;
  uint32_t bb=token+g<count?riley_prefill_shape::pair(k[kb+2*t+8],k[kb+2*t+9]):0;
  riley_prefill_shape::mma(d,a,a,aa,aa,b,bb);
 }
 if(g==0)for(int j=0;j<2;++j)if(token+2*t+j<count)result[head*4096+token+2*t+j]=d[j]*.125F;
 }
}
// Each 128-float score tile becomes 64 packed BF16 pairs plus its FP32
// online scale. Packing stays within that tile, so unread lower tiles survive.
__global__ void prepare_probabilities(float* scratch,const uint32_t* shape,const uint32_t* live_rows){
 int row=blockIdx.x/9,head=blockIdx.x%9,lane=threadIdx.x,t=lane%4;
 uint32_t active=*live_rows;if(active<1||active>8||row>=active)return;
 int count=shape[row*416+1]+1;if(count<1||count>4096)return;
 float* scores=scratch+(row*9+head)*4096;
 __shared__ float exponentials[128];float maximum=-CUDART_INF_F,den=0.;
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,n=min(128,count-begin);float mx=maximum;
  for(int i=lane;i<n;i+=32)mx=fmaxf(mx,scores[begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  for(int i=lane;i<128;i+=32)exponentials[i]=i<n?riley_prefill_shape::exponential(scores[begin+i],mx):0.;
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<n)local_den+=exponentials[i];}
  den=local_den;maximum=mx;
  for(int i=lane;i<64;i+=32)reinterpret_cast<uint32_t*>(scores+begin)[i]=riley_prefill_shape::pair(__float2bfloat16_rn(exponentials[2*i]),__float2bfloat16_rn(exponentials[2*i+1]));
  if(lane==0)scores[begin+64]=alpha;
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 if(lane<4)scores[65+lane]=den;
}
// Preserve all output CTAs, ordered K16 MMA steps and online rescaling.
__global__ void values(const float* scratch,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/9;uint32_t active=*live_rows;if(active<1||active>8||row>=active)return;
 shape+=row*416;pages+=row*416;out+=row*576;
 int count=shape[1]+1,head=blockIdx.y%9,block=blockIdx.x,lane=threadIdx.x,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 const float* packed=scratch+(row*9+head)*4096;float accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);float alpha=packed[begin+64];
  for(int j=0;j<4;++j)accum[j]*=alpha;
  const uint32_t* probs=reinterpret_cast<const uint32_t*>(packed+begin);
  for(int token=begin;token<end;token+=16){
   int pi=(token-begin)/2;uint32_t a=probs[pi+t],aa=probs[pi+t+4];
   int dim=block*8+g,value_base=((pages[token/16]*3+head/3)*16)*64+dim;
   auto val=[&](int pos){return pos<end?v[value_base+(pos-token)*64]:zero;};
   uint32_t b=riley_prefill_shape::pair(val(token+2*t),val(token+2*t+1));
   uint32_t bb=riley_prefill_shape::pair(val(token+2*t+8),val(token+2*t+9));
   riley_prefill_shape::mma(accum,a,a,aa,aa,b,bb);
  }
 }
 float inverse=1.0F/packed[65+t];
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}

inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 scores<<<dim3(min((context+7)/8,32u),72),32,0,stream>>>(q,k,scratch,shape,pages,live_rows);
 prepare_probabilities<<<72,32,0,stream>>>(scratch,shape,live_rows);
 values<<<dim3(8,72),32,0,stream>>>(scratch,v,out,shape,pages,live_rows);
}
}
