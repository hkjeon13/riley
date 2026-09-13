#pragma once
#include "../src/decode_shape.cuh"
// One CTA per request/head: warp 0 produces softmax state; eight warps consume
// the same BF16 probabilities for distinct output columns. Original recurrence.
namespace riley_shared_head_values {
__global__ void values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y,head=blockIdx.x,block=threadIdx.x/32;uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 shape+=row*416;pages+=row*416;
 scores+=row*9*4096;out+=row*576;
 int count=shape[1]+1,lane=threadIdx.x%32,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 __shared__ __nv_bfloat16 probs[128];
 __shared__ float exponentials[128],rescale,inverse_shared;
 float maximum=-CUDART_INF_F,den=0.,accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  if(block==0){float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[head*4096+begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  // Preserve unrounded exponentials for the original lane-local sum order.
  for(int i=lane;i<128;i+=32){float value=i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0.;exponentials[i]=value;probs[i]=__float2bfloat16_rn(value);}
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=exponentials[i];}
  den=local_den;maximum=mx;if(lane==0)rescale=alpha;
  }
  __syncthreads();
  float alpha=rescale;
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
  __syncthreads();
 }
 if(block==0){den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);if(lane==0)inverse_shared=1.0F/den;}
 __syncthreads();float inverse=inverse_shared;
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}

}
