#pragma once
#include "prefill_shape_projection.cuh"
namespace riley_prefill51 {
__device__ __forceinline__ void mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b,uint32_t bb){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
}
template<int N,int K,int Interval,int Warps,bool Tiled,int Tiles>
__global__ void projection(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,uint32_t capacity,const uint32_t* live_rows){
 uint32_t rows=*live_rows;if(!rows||rows>capacity||blockIdx.y*16*Tiles>=rows)return;
 static_assert(Tiles==1||Tiles==2||Tiles==4,"bounded row reuse");
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int first=blockIdx.y*16*Tiles+g,base=(blockIdx.x*Warps+warp)*8;
 float d[Tiles][4]={},total[Tiles][4]={};
 #pragma unroll 4
 for(int depth=0;depth<K;depth+=16){
  uint32_t b,bb;
  if constexpr(Tiled){const auto* tile=w+((base/8)*(K/16)+depth/16)*128;b=*reinterpret_cast<const uint32_t*>(tile+lane*2);bb=*reinterpret_cast<const uint32_t*>(tile+64+lane*2);}
  else{b=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t));bb=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t+8));}
  #pragma unroll
  for(int tile=0;tile<Tiles;++tile){
   int row=first+tile*16,next=row+8;
   uint32_t a0=row<rows?*reinterpret_cast<const uint32_t*>(x+row*K+depth+2*t):0;
   uint32_t a1=next<rows?*reinterpret_cast<const uint32_t*>(x+next*K+depth+2*t):0;
   uint32_t a2=row<rows?*reinterpret_cast<const uint32_t*>(x+row*K+depth+2*t+8):0;
   uint32_t a3=next<rows?*reinterpret_cast<const uint32_t*>(x+next*K+depth+2*t+8):0;
   mma(d[tile],a0,a1,a2,a3,b,bb);
   if constexpr(Interval>0)if((depth+16)%Interval==0||depth+16==K)for(int j=0;j<4;++j){total[tile][j]+=__bfloat162float(__float2bfloat16_rn(d[tile][j]));d[tile][j]=0.;}
  }
 }
 #pragma unroll
 for(int tile=0;tile<Tiles;++tile){
  if constexpr(Interval>0)for(int j=0;j<4;++j)d[tile][j]=total[tile][j];
  int row=first+tile*16,next=row+8;
  if(row<rows){y[row*N+base+2*t]=__float2bfloat16_rn(d[tile][0]);y[row*N+base+2*t+1]=__float2bfloat16_rn(d[tile][1]);}
  if(next<rows){y[next*N+base+2*t]=__float2bfloat16_rn(d[tile][2]);y[next*N+base+2*t+1]=__float2bfloat16_rn(d[tile][3]);}
 }
}
template<int Warps,int Tiles>
__global__ void gate_up(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,uint32_t capacity,const uint32_t* live_rows){
 uint32_t rows=*live_rows;if(!rows||rows>capacity||blockIdx.y*16*Tiles>=rows)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int first=blockIdx.y*16*Tiles+g,base=(blockIdx.x*Warps+warp)*8;
 float gd[Tiles][4]={},ud[Tiles][4]={};
 #pragma unroll 4
 for(int depth=0;depth<576;depth+=16){
  int at=((base/8)*36+depth/16)*128+lane*2;
  uint32_t gb=*reinterpret_cast<const uint32_t*>(gate+at),gbb=*reinterpret_cast<const uint32_t*>(gate+at+64);
  uint32_t ub=*reinterpret_cast<const uint32_t*>(up+at),ubb=*reinterpret_cast<const uint32_t*>(up+at+64);
  #pragma unroll
  for(int tile=0;tile<Tiles;++tile){
   int row=first+tile*16,next=row+8;
   uint32_t a0=row<rows?*reinterpret_cast<const uint32_t*>(x+row*576+depth+2*t):0;
   uint32_t a1=next<rows?*reinterpret_cast<const uint32_t*>(x+next*576+depth+2*t):0;
   uint32_t a2=row<rows?*reinterpret_cast<const uint32_t*>(x+row*576+depth+2*t+8):0;
   uint32_t a3=next<rows?*reinterpret_cast<const uint32_t*>(x+next*576+depth+2*t+8):0;
   mma(gd[tile],a0,a1,a2,a3,gb,gbb);mma(ud[tile],a0,a1,a2,a3,ub,ubb);
  }
 }
 #pragma unroll
 for(int tile=0;tile<Tiles;++tile)for(int j=0;j<4;++j){
  int row=first+tile*16+(j>=2?8:0),col=base+2*t+j%2;
  if(row<rows){float gval=__bfloat162float(__float2bfloat16_rn(gd[tile][j]));float uval=__bfloat162float(__float2bfloat16_rn(ud[tile][j]));out[row*1536+col]=__float2bfloat16_rn((gval/(1.F+expf(-gval)))*uval);}
 }
}
}
