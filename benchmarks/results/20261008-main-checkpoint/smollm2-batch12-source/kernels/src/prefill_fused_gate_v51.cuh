#pragma once
#include "prefill_shape_projection.cuh"
namespace riley_prefill51 {
__device__ __forceinline__ void mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b,uint32_t bb){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
}
template<int Warps>
__global__ void gate_up(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,uint32_t capacity,const uint32_t* live_rows){
 constexpr int Tiles=1;
 uint32_t rows=*live_rows;if(!rows||rows>capacity||blockIdx.y*16*Tiles>=rows)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int first=blockIdx.y*16*Tiles+g,base=(blockIdx.x*Warps+warp)*8;
 float gd[Tiles][4]={},ud[Tiles][4]={};
 // Two K16 steps per stage keep the next fragments live across current MMA.
 // Each gate/up accumulator still consumes K16 steps in the original order.
 struct Stage {uint32_t a0[2],a1[2],a2[2],a3[2],gb[2],gbb[2],ub[2],ubb[2];};
 Stage current{},next{};
 auto load_stage=[&](int depth,Stage& stage){
  #pragma unroll
  for(int step=0;step<2;++step){
   int offset=depth+step*16,at=((base/8)*36+offset/16)*128+lane*2;
   stage.gb[step]=*reinterpret_cast<const uint32_t*>(gate+at);stage.gbb[step]=*reinterpret_cast<const uint32_t*>(gate+at+64);
   stage.ub[step]=*reinterpret_cast<const uint32_t*>(up+at);stage.ubb[step]=*reinterpret_cast<const uint32_t*>(up+at+64);
   stage.a0[step]=first<rows?*reinterpret_cast<const uint32_t*>(x+first*576+offset+2*t):0;
   stage.a1[step]=first+8<rows?*reinterpret_cast<const uint32_t*>(x+(first+8)*576+offset+2*t):0;
   stage.a2[step]=first<rows?*reinterpret_cast<const uint32_t*>(x+first*576+offset+2*t+8):0;
   stage.a3[step]=first+8<rows?*reinterpret_cast<const uint32_t*>(x+(first+8)*576+offset+2*t+8):0;
  }
 };
 load_stage(0,current);
 #pragma unroll 1
 for(int depth=0;depth<576;depth+=32){
  if(depth+32<576)load_stage(depth+32,next);
  #pragma unroll
  for(int step=0;step<2;++step){
   mma(gd[0],current.a0[step],current.a1[step],current.a2[step],current.a3[step],current.gb[step],current.gbb[step]);
   mma(ud[0],current.a0[step],current.a1[step],current.a2[step],current.a3[step],current.ub[step],current.ubb[step]);
  }
  if(depth+32<576)current=next;
 }
 #pragma unroll
 for(int tile=0;tile<Tiles;++tile)for(int j=0;j<4;++j){
  int row=first+tile*16+(j>=2?8:0),col=base+2*t+j%2;
  if(row<rows){float gval=__bfloat162float(__float2bfloat16_rn(gd[tile][j]));float uval=__bfloat162float(__float2bfloat16_rn(ud[tile][j]));out[row*1536+col]=__float2bfloat16_rn((gval/(1.F+expf(-gval)))*uval);}
 }
}
}
