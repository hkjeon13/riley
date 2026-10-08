#pragma once
#include "decode_shape.cuh"
// One-row projection: independent original BF16 rounding chunks run in parallel.
// Each warp writes eight output columns. The merge retains the original order.
template<int N,int K,int Interval>
__global__ void tile_projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out){
 const int lane=threadIdx.x,g=lane/4,t=lane%4,base=blockIdx.x*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=blockIdx.y*Chunk,end=min(begin+Chunk,K);float d[4]={};
 #pragma unroll
 for(int depth=0;depth<Chunk;depth+=16){
  if(begin+depth>=end)break;
  int k=begin+depth;
  uint32_t a=*reinterpret_cast<const uint32_t*>(x+k+2*t),aa=*reinterpret_cast<const uint32_t*>(x+k+2*t+8);
  const auto* tile=w+((base/8)*(K/16)+k/16)*128;
  uint32_t b=*reinterpret_cast<const uint32_t*>(tile+lane*2),bb=*reinterpret_cast<const uint32_t*>(tile+64+lane*2);
  riley_prefill_shape::mma(d,a,a,aa,aa,b,bb);
 }
 if(g==0)for(int j=0;j<2;++j){
  auto v=__float2bfloat16_rn(d[j]);
  if constexpr(Interval>0)parts[blockIdx.y*N+base+2*t+j]=__bfloat162float(v);
  else out[base+2*t+j]=v;
 }
}
template<int N,int K,int Interval>
__global__ void tile_projection_merge(const float* parts,__nv_bfloat16* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;float value=0.;
 #pragma unroll
 for(int chunk=0;chunk<(K+Interval-1)/Interval;++chunk)value+=parts[chunk*N+i];
 out[i]=__float2bfloat16_rn(value);
}
template<int N,int K,int Interval>
inline void enqueue_tile_projection(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,float* parts){
 constexpr int chunk=Interval>0?Interval:K;
 constexpr int chunks=(K+chunk-1)/chunk;
 tile_projection_parts<N,K,Interval><<<dim3(N/8,chunks),32,0,stream>>>(x,w,parts,out);
 if constexpr(Interval>0)tile_projection_merge<N,K,Interval><<<(N+255)/256,256,0,stream>>>(parts,out);
}
