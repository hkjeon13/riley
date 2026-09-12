#pragma once
#include "decode_shared16_attention.cuh"
namespace riley_shared16_grouped {
// Three query heads use the same value matrix; MMA rows carry distinct heads.
__global__ void values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 const int row=blockIdx.y/3,kvh=blockIdx.y%3;
 const uint32_t active=*live_rows;
 if(active<1||active>16||row>=active)return;
 shape+=row*416;pages+=row*416;scores+=row*9*4096;out+=row*576;
 const int count=shape[1]+1,block=blockIdx.x,lane=threadIdx.x,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 const int norm_head=lane/8,norm_lane=lane%8;
 __shared__ __nv_bfloat16 probs[3][128];
 __shared__ float exponentials[3][128];
 float maximum=-CUDART_INF_F,den=0.F,accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.F);
 for(int tile=(count-1)/128;tile>=0;--tile){
  const int begin=tile*128,end=min(begin+128,count);
  // Eight lanes scan each head; four-lane groups below retain the original
  // denominator recurrence and own the corresponding MMA output row.
  float norm_max=__shfl_sync(0xffffffff,maximum,norm_head*4);
  if(norm_head<3)for(int i=norm_lane;i<end-begin;i+=8)norm_max=fmaxf(norm_max,scores[(kvh*3+norm_head)*4096+begin+i]);
  for(int offset=4;offset>0;offset>>=1)norm_max=fmaxf(norm_max,__shfl_xor_sync(0xffffffff,norm_max,offset,8));
  const float mx=__shfl_sync(0xffffffff,norm_max,(g<3?g:0)*8);
  const float alpha=g<3?exp2f((maximum-mx)*1.4426950408889634F):1.F;
  if(norm_head<3)for(int i=norm_lane;i<128;i+=8){
   const float p=i<end-begin?riley_prefill_shape::exponential(scores[(kvh*3+norm_head)*4096+begin+i],norm_max):0.F;
   exponentials[norm_head][i]=p;probs[norm_head][i]=__float2bfloat16_rn(p);
  }
  __syncwarp();
  float local_den=den*alpha;
  if(g<3)for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=exponentials[g][i];}
  den=local_den;if(g<3)maximum=mx;
  #pragma unroll
  for(int j=0;j<4;++j)accum[j]*=alpha;
  #pragma unroll 1
  for(int token=begin;token<end;token+=64){
   uint32_t pa[4],paa[4],vb[4],vbb[4];
   #pragma unroll
   for(int part=0;part<4;++part){
    const int at=token+part*16,pi=at-begin;const bool live=at<end;
    pa[part]=live&&g<3?riley_prefill_shape::pair(probs[g][pi+2*t],probs[g][pi+2*t+1]):0;
    paa[part]=live&&g<3?riley_prefill_shape::pair(probs[g][pi+2*t+8],probs[g][pi+2*t+9]):0;
    const int dim=block*8+g,base=live?((pages[at/16]*3+kvh)*16)*64+dim:0;
    auto val=[&](int pos){return live&&pos<end?v[base+(pos-at)*64]:zero;};
    vb[part]=riley_prefill_shape::pair(val(at+2*t),val(at+2*t+1));
    vbb[part]=riley_prefill_shape::pair(val(at+2*t+8),val(at+2*t+9));
   }
   #pragma unroll
   for(int part=0;part<4;++part)if(token+part*16<end)riley_prefill_shape::mma(accum,pa[part],pa[part],paa[part],paa[part],vb[part],vbb[part]);
  }
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2,4);den+=__shfl_xor_sync(0xffffffff,den,1,4);
 const float inverse=1.F/den;
 if(g<3)for(int j=0;j<2;++j)out[(kvh*3+g)*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}
inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 riley_shared16_attention::scores<<<dim3(min((context+7)/8,32u),144),32,0,stream>>>(q,k,scratch,shape,pages,live_rows);
 values<<<dim3(8,48),32,0,stream>>>(scratch,v,out,shape,pages,live_rows);
}
}
