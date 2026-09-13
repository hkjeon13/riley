#pragma once
#include "packed_value_v54.cuh"
#include "decode_shape.cuh"
namespace riley_gqa50_attention {

// Parallel QK score tiles, preserving the exact four K16 MMA recurrence.
__global__ void scores(const __nv_bfloat16* q,const __nv_bfloat16* k,float* result,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/3;uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 shape+=row*416;pages+=row*416;
 q+=row*576;result+=row*9*4096;
 int count=shape[1]+1,token=blockIdx.x*8,kvhead=blockIdx.y%3;
 if(count<1 || count>4096)return;
 int lane=threadIdx.x,g=lane/4,t=lane%4,head=kvhead*3+g;
 uint32_t query[4],query_hi[4];
 #pragma unroll
 for(int depth=0;depth<4;++depth){auto* qp=q+(g<3?head:0)*64+depth*16;query[depth]=g<3?riley_prefill_shape::pair(qp[2*t],qp[2*t+1]):0;query_hi[depth]=g<3?riley_prefill_shape::pair(qp[2*t+8],qp[2*t+9]):0;}
 for(;token<count;token+=gridDim.x*8){
 float d[4]={};uint32_t key[4],key_hi[4];
 bool valid=token+g<count;int kb=valid?riley_prefill_shape::cache_index(token+g,kvhead,0,pages):0;
 #pragma unroll
 for(int depth=0;depth<4;++depth){key[depth]=valid?riley_prefill_shape::pair(k[kb+depth*16+2*t],k[kb+depth*16+2*t+1]):0;key_hi[depth]=valid?riley_prefill_shape::pair(k[kb+depth*16+2*t+8],k[kb+depth*16+2*t+9]):0;}
 #pragma unroll
 for(int depth=0;depth<4;++depth)riley_prefill_shape::mma(d,query[depth],0,query_hi[depth],0,key[depth],key_hi[depth]);
 if(g<3)for(int j=0;j<2;++j)if(token+2*t+j<count)result[head*4096+token+2*t+j]=d[j]*.125F;
 }
}

__global__ void independent_values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/3;uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 shape+=row*416;pages+=row*416;
 scores+=row*9*4096;out+=row*576;
 int count=shape[1]+1,warp=threadIdx.x/32,head=(blockIdx.y%3)*3+warp,block=blockIdx.x,lane=threadIdx.x%32,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 __shared__ __nv_bfloat16 all_probs[3][128];auto* probs=all_probs[warp];
 __shared__ float all_exponentials[3][128];auto* exponentials=all_exponentials[warp];float maximum=-CUDART_INF_F,den=0.,accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[head*4096+begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  // Preserve unrounded exponentials for the original lane-local sum order.
  for(int i=lane;i<128;i+=32){float value=i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0.;exponentials[i]=value;probs[i]=__float2bfloat16_rn(value);}
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=exponentials[i];}
  den=local_den;maximum=mx;__syncwarp();
  for(int j=0;j<4;++j)accum[j]*=alpha;
  #pragma unroll 1
  for(int token=begin;token<end;token+=64){
   uint32_t pa[4],paa[4],vb[4],vbb[4];
   #pragma unroll
   for(int part=0;part<4;++part){
    int at=token+part*16,pi=at-begin;bool live=at<end;
    pa[part]=live?riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]):0;
    paa[part]=live?riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]):0;
    // Each lane reads an aligned adjacent-token pair from the packed V tile.
    int vi=live?(pages[at/16]*3+head/3)*1024+block*128:0;
    vb[part]=live?riley_packed_value_v54::masked_pair(v+vi+lane*2,at+2*t,end):0;
    vbb[part]=live?riley_packed_value_v54::masked_pair(v+vi+64+lane*2,at+2*t+8,end):0;
   }
   #pragma unroll
   for(int part=0;part<4;++part)if(token+part*16<end)riley_prefill_shape::mma(accum,pa[part],pa[part],paa[part],paa[part],vb[part],vbb[part]);
  }
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);float inverse=1.0F/den;
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}


inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 scores<<<dim3(min((context+7)/8,32u),96),32,0,stream>>>(q,k,scratch,shape,pages,live_rows);
 independent_values<<<dim3(8,96),96,0,stream>>>(scratch,v,out,shape,pages,live_rows);
}
}
