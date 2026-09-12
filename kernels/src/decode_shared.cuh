#pragma once
#include "decode_tiled.cuh"
// One MMA shares weights across up to eight independent decode rows.
// Live rows are supplied by validated metadata; inactive rows never load/store.
template<int N,int K,int Interval,bool Tiled>
__global__ void shared_projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out,const uint32_t* live_rows){
 const uint32_t rows=*live_rows;if(rows<1||rows>8)return;
 const int lane=threadIdx.x,g=lane/4,t=lane%4,base=blockIdx.x*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=blockIdx.y*Chunk,end=min(begin+Chunk,K);float d[4]={};
 #pragma unroll
 for(int depth=0;depth<Chunk;depth+=16){
  if(begin+depth>=end)break;
  int k=begin+depth;
  uint32_t a=g<rows?*reinterpret_cast<const uint32_t*>(x+g*K+k+2*t):0,aa=g<rows?*reinterpret_cast<const uint32_t*>(x+g*K+k+2*t+8):0;
  const auto* tile=w+((base/8)*(K/16)+k/16)*128;
  uint32_t b,bb;
  if constexpr(Tiled){b=*reinterpret_cast<const uint32_t*>(tile+lane*2);bb=*reinterpret_cast<const uint32_t*>(tile+64+lane*2);}
  else {b=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t);bb=*reinterpret_cast<const uint32_t*>(w+(base+g)*K+k+2*t+8);}
  riley_prefill_shape::mma(d,a,0,aa,0,b,bb);
 }
 if(g<rows)for(int j=0;j<2;++j){
  auto v=__float2bfloat16_rn(d[j]);
  if constexpr(Interval>0)parts[(blockIdx.y*8+g)*N+base+2*t+j]=__bfloat162float(v);
  else out[g*N+base+2*t+j]=v;
 }
}

template<int N,int K,int Interval>
__global__ void shared_projection_merge(const float* parts,__nv_bfloat16* out,const uint32_t* live_rows){
 uint32_t rows=*live_rows;if(rows<1||rows>8)return;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*N)return;float value=0.;
 #pragma unroll
 for(int chunk=0;chunk<(K+Interval-1)/Interval;++chunk)value+=parts[chunk*8*N+i];
 out[i]=__float2bfloat16_rn(value);
}
template<int N,int K,int Interval,bool Tiled>
inline void enqueue_shared_projection(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* out,float* parts,const uint32_t* live_rows){
 constexpr int chunk=Interval>0?Interval:K;
 shared_projection_parts<N,K,Interval,Tiled><<<dim3(N/8,(K+chunk-1)/chunk),32,0,stream>>>(x,w,parts,out,live_rows);
 if constexpr(Interval>0)shared_projection_merge<N,K,Interval><<<(8*N+255)/256,256,0,stream>>>(parts,out,live_rows);
}
