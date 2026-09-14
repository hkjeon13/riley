#pragma once
#include "prefill_ffn_pipeline.cuh"

// Isolated projection mainloop; launch requires N % (8*Warps) == 0.
namespace riley_prefill_projection {
// Prepare once per immutable weight owner. Layout is [N/8,K/16,128].
template<int N,int K>
__global__ void pack(const __nv_bfloat16* input,__nv_bfloat16* output) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*K)return;
 unsigned tile=i/128,offset=i%128;
 unsigned n=(tile/(K/16))*8+(offset%64)/8;
 unsigned k=(tile%(K/16))*16+(offset/64)*8+offset%8;
 output[i]=input[n*K+k];
}
template<int N,int K,int Interval,int Warps>
__global__ void project(const __nv_bfloat16* x,const __nv_bfloat16* weight,
 __nv_bfloat16* out,unsigned capacity,const unsigned* live_rows) {
 static_assert(K%64==0 && N%(8*Warps)==0 && Interval%16==0);
 unsigned rows=*live_rows;if(!rows||rows>capacity||blockIdx.y*16>=rows)return;
 using namespace riley_prefill_ffn_pipeline;
 __shared__ Stage<Warps,false> stages[2];
 int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 int first=blockIdx.y*16+g,base=(blockIdx.x*Warps+warp)*8;
 float d[4]={},total[4]={};
 load<K,Warps,false>(stages[0],x,weight,weight,rows,0);wait();__syncthreads();
 for(int depth=0,index=0;depth<K;depth+=64,index^=1){
  if(depth+64<K)load<K,Warps,false>(stages[index^1],x,weight,weight,rows,depth+64);
  const auto& s=stages[index];
  #pragma unroll
  for(int step=0;step<4;++step){
   const auto* a=s.x+g*72+step*16+2*t;
   const auto* b=s.weights+warp*512+step*128+lane*2;
   riley_prefill51::mma(d,pair(a),pair(a+8*72),pair(a+8),pair(a+8*72+8),pair(b),pair(b+64));
   if constexpr(Interval>0)if((depth+step*16+16)%Interval==0||depth+step*16+16==K)
    for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.F;}
  }
  wait();__syncthreads();
 }
 for(int j=0;j<4;++j){int row=first+(j>=2?8:0),col=base+2*t+j%2;
  if(row<rows)out[row*N+col]=__float2bfloat16_rn(Interval>0?total[j]:d[j]);}
}
}
