#pragma once
#include "decode_shape.cuh"
// One-row projection: independent original BF16 rounding chunks run in parallel.
// Each warp writes eight output columns. The merge retains the original order.
template<int N,int K,int Interval,int Warps,bool Packed>
__global__ void probe_projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out){
 const int lane=threadIdx.x%32,g=lane/4,t=lane%4,base=(blockIdx.x*Warps+threadIdx.x/32)*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=blockIdx.y*Chunk,end=min(begin+Chunk,K);float d[4]={};
 #pragma unroll
 for(int depth=0;depth<Chunk;depth+=16){
  if(begin+depth>=end)break;
  int k=begin+depth;
  uint32_t a=*reinterpret_cast<const uint32_t*>(x+k+2*t),aa=*reinterpret_cast<const uint32_t*>(x+k+2*t+8);
  uint32_t b,bb;
  if constexpr(Packed){
   uint2 pair=*reinterpret_cast<const uint2*>(w+(base+g)*K+k+4*t);
   uint32_t low0=__shfl_sync(0xffffffff,pair.x,g*4+t/2),low1=__shfl_sync(0xffffffff,pair.y,g*4+t/2);
   uint32_t high0=__shfl_sync(0xffffffff,pair.x,g*4+t/2+2),high1=__shfl_sync(0xffffffff,pair.y,g*4+t/2+2);
   b=(t%2)?low1:low0;bb=(t%2)?high1:high0;
  }else{b=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t);bb=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t+8);}
  riley_prefill_shape::mma(d,a,a,aa,aa,b,bb);
 }
 if(g==0)for(int j=0;j<2;++j){
  auto v=__float2bfloat16_rn(d[j]);
  if constexpr(Interval>0)parts[blockIdx.y*N+base+2*t+j]=__bfloat162float(v);
  else out[base+2*t+j]=v;
 }
}
template<int N,int K,int Interval,int Warps,bool Packed>
__global__ void probe_projection_merge(const float* parts,__nv_bfloat16* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;float value=0.;
 #pragma unroll
 for(int chunk=0;chunk<(K+Interval-1)/Interval;++chunk)value+=parts[chunk*N+i];
 out[i]=__float2bfloat16_rn(value);
}
template<int N,int K,int Interval,int Warps,bool Packed>
inline void enqueue_probe_projection(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,float* parts){
 constexpr int chunk=Interval>0?Interval:K;
 constexpr int chunks=(K+chunk-1)/chunk;
 probe_projection_parts<N,K,Interval,Warps,Packed><<<dim3(N/(8*Warps),chunks),32*Warps,0,stream>>>(x,w,parts,out);
 if constexpr(Interval>0)probe_projection_merge<N,K,Interval,Warps,Packed><<<(N+255)/256,256,0,stream>>>(parts,out);
}
