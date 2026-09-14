#pragma once
#include "decode_gqa_attention_v50.cuh"

// Experimental: explicit external workspace, no serving dispatch until qualified.
namespace riley_decode_softmax_reuse {
constexpr size_t probability_elements=32u*9u*4096u;
constexpr size_t alpha_elements=32u*9u*32u;
constexpr size_t inverse_elements=32u*9u*4u;

__global__ void prepare(const float* scores,__nv_bfloat16* probabilities,float* alphas,float* inverses,const uint32_t* shape,const uint32_t* live_rows){
 int row=blockIdx.x/9,head=blockIdx.x%9;
 uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 int count=shape[row*416+1]+1;if(count<1||count>4096)return;
 int lane=threadIdx.x,t=lane%4;
 scores+=(row*9+head)*4096;probabilities+=(row*9+head)*4096;
 alphas+=(row*9+head)*32;inverses+=(row*9+head)*4;
 __shared__ float exponentials[128];
 float maximum=-CUDART_INF_F,den=0.;
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  if(lane==0)alphas[tile]=alpha;
  for(int i=lane;i<128;i+=32){float value=i<end-begin?riley_prefill_shape::exponential(scores[begin+i],mx):0.;exponentials[i]=value;probabilities[begin+i]=__float2bfloat16_rn(value);}
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=exponentials[i];}
  den=local_den;maximum=mx;__syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.0F/den;if(lane<4)inverses[t]=inverse;
}

__global__ void values(const __nv_bfloat16* probabilities,const float* alphas,const float* inverses,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/3;uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 shape+=row*416;pages+=row*416;
 probabilities+=row*9*4096;alphas+=row*9*32;inverses+=row*9*4;out+=row*576;
 int count=shape[1]+1,warp=threadIdx.x/32,head=(blockIdx.y%3)*3+warp,block=blockIdx.x,lane=threadIdx.x%32,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 float accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  const auto* probs=probabilities+head*4096+begin;
  float alpha=alphas[head*32+tile];
  for(int j=0;j<4;++j)accum[j]*=alpha;
  #pragma unroll 1
  for(int token=begin;token<end;token+=64){
   uint32_t pa[4],paa[4],vb[4],vbb[4];
   #pragma unroll
   for(int part=0;part<4;++part){
    int at=token+part*16,pi=at-begin;bool live=at<end;
    pa[part]=live?riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]):0;
    paa[part]=live?riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]):0;
    int dim=block*8+g;int value_base=live?((pages[at/16]*3+head/3)*16)*64+dim:0;
    auto val=[&](int pos){return pos<end&&live?v[value_base+(pos-at)*64]:zero;};
    vb[part]=riley_prefill_shape::pair(val(at+2*t),val(at+2*t+1));
    vbb[part]=riley_prefill_shape::pair(val(at+2*t+8),val(at+2*t+9));
   }
   #pragma unroll
   for(int part=0;part<4;++part)if(token+part*16<end)riley_prefill_shape::mma(accum,pa[part],pa[part],paa[part],paa[part],vb[part],vbb[part]);
  }
  __syncwarp();
 }
 float inverse=inverses[head*4+t];
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}

inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scores,__nv_bfloat16* probabilities,float* alphas,float* inverses,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 riley_gqa50_attention::scores<<<dim3(min((context+7)/8,32u),96),32,0,stream>>>(q,k,scores,shape,pages,live_rows);
 prepare<<<32*9,32,0,stream>>>(scores,probabilities,alphas,inverses,shape,live_rows);
 values<<<dim3(8,96),96,0,stream>>>(probabilities,alphas,inverses,v,out,shape,pages,live_rows);
}
}
