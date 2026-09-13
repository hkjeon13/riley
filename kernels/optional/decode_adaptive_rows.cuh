#pragma once
#include "../src/decode_shared32.cuh"
// Experimental native candidate. Preserve K recurrence, BF16 partial rounding
// and the existing 32-row scratch stride. One 16-row MMA tile is sufficient
// when the device packet has at most 16 live rows; no host readback is needed.
namespace riley_adaptive_decode {
// Derived from shared32_projection_compute; intentionally isolated until full-model
// parity and serving measurements qualify the adaptive variant.
template<int N,int K,int Interval,bool Tiled,bool Compact=false,int Tiles=1>
__device__ __forceinline__ void projection_compute(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,__nv_bfloat16* out,const uint32_t* live_rows,int column,int chunk_index){
 const uint32_t rows=*live_rows;if(rows<1||rows>32)return;
 const int lane=threadIdx.x%32,g=lane/4,t=lane%4,base=column*8;
 constexpr int Chunk=Interval>0?Interval:K;
 int begin=chunk_index*Chunk,end=min(begin+Chunk,K);float d[Tiles][4]={};
 const auto* xp=x+g*K+begin+2*t;
 const auto* wp=Tiled?w+column*(K/16)*128+(begin/16)*128+lane*2:w+(base+g)*K+begin+2*t;
 #pragma unroll 1
 for(int depth=0;depth<Chunk;depth+=64){
  uint32_t a[Tiles][4],aa[Tiles][4],a1[Tiles][4],a3[Tiles][4],b[4],bb[4];
  #pragma unroll
  for(int step=0;step<4;++step){
   int offset=depth+step*16;bool valid=begin+offset<end;
   #pragma unroll
   for(int tile=0;tile<Tiles;++tile){
    const auto* p=xp+tile*16*K;int row=g+tile*16;
    a[tile][step]=valid&&row<rows?*reinterpret_cast<const uint32_t*>(p+offset):0;
    aa[tile][step]=valid&&row<rows?*reinterpret_cast<const uint32_t*>(p+offset+8):0;
    a1[tile][step]=valid&&row+8<rows?*reinterpret_cast<const uint32_t*>(p+8*K+offset):0;
    a3[tile][step]=valid&&row+8<rows?*reinterpret_cast<const uint32_t*>(p+8*K+offset+8):0;
   }
   if constexpr(Tiled){b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128):0;bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128+64):0;}
   else{b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset):0;bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset+8):0;}
  }
  #pragma unroll
  for(int step=0;step<4;++step)if(begin+depth+step*16<end){
   #pragma unroll
   for(int tile=0;tile<Tiles;++tile)riley_prefill_shape::mma(d[tile],a[tile][step],a1[tile][step],aa[tile][step],a3[tile][step],b[step],bb[step]);
  }
 }
 #pragma unroll
 for(int tile=0;tile<Tiles;++tile)for(int half=0;half<2;++half){int row=tile*16+g+half*8;if(row>=rows)continue;
  for(int j=0;j<2;++j){auto value=__float2bfloat16_rn(d[tile][half*2+j]);
   if constexpr(Interval>0)parts[(chunk_index*32+row)*N+base+2*t+j]=__bfloat162float(value);
   else if constexpr(Compact)out[row*8+2*t+j]=value;
   else out[row*N+base+2*t+j]=value;
  }
 }
}

template<int N,int K,int Interval,bool Tiled>
__device__ __forceinline__ void dispatch(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,const uint32_t* live,int col,int chunk) {
 if(*live<=16)projection_compute<N,K,Interval,Tiled,false,1>(x,w,parts,nullptr,live,col,chunk);
 else shared32_projection_compute<N,K,Interval,Tiled>(x,w,parts,nullptr,live,col,chunk);
}
template<int N,int K,int Interval,bool Tiled>
__global__ void projection_parts(const __nv_bfloat16* x,const __nv_bfloat16* w,float* parts,const uint32_t* live) {
 dispatch<N,K,Interval,Tiled>(x,w,parts,live,blockIdx.x,blockIdx.y);
}
__global__ void qkv_parts(const __nv_bfloat16* x,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,float* parts,const uint32_t* live) {
 int col=blockIdx.x;
 if(col<72)dispatch<576,576,192,false>(x,q,parts,live,col,blockIdx.y);
 else dispatch<192,576,192,false>(x,col<96?k:v,parts+3*32*576+(col<96?0:3*32*192),live,col<96?col-72:col-96,blockIdx.y);
}
}
