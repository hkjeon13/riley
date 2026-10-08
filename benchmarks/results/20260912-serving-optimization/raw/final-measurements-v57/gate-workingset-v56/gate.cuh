#pragma once
#include "decode_shared32.cuh"
namespace riley_gate_v56 {
template<int Steps,int Tiles>
__device__ __forceinline__ void compute(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,uint32_t rows){
 const int lane=threadIdx.x%32,g=lane/4,t=lane%4,column=blockIdx.x;
 float dg[Tiles][4]={},du[Tiles][4]={};
 #pragma unroll 1
 for(int depth=0;depth<576;depth+=Steps*16){
  uint32_t a[Tiles][Steps],aa[Tiles][Steps],ah[Tiles][Steps],aah[Tiles][Steps];
  uint32_t bg[Steps],bgh[Steps],bu[Steps],buh[Steps];
  #pragma unroll
  for(int step=0;step<Steps;++step){
   int at=depth+step*16;
   #pragma unroll
   for(int tile=0;tile<Tiles;++tile){int row=g+tile*16;auto* p=x+row*576+at+2*t;
    a[tile][step]=row<rows?*reinterpret_cast<const uint32_t*>(p):0;
    aa[tile][step]=row<rows?*reinterpret_cast<const uint32_t*>(p+8):0;
    ah[tile][step]=row+8<rows?*reinterpret_cast<const uint32_t*>(p+8*576):0;
    aah[tile][step]=row+8<rows?*reinterpret_cast<const uint32_t*>(p+8*576+8):0;
   }
   int wi=column*36*128+(at/16)*128+lane*2;
   bg[step]=*reinterpret_cast<const uint32_t*>(gate+wi);bgh[step]=*reinterpret_cast<const uint32_t*>(gate+wi+64);
   bu[step]=*reinterpret_cast<const uint32_t*>(up+wi);buh[step]=*reinterpret_cast<const uint32_t*>(up+wi+64);
  }
  #pragma unroll
  for(int step=0;step<Steps;++step){
   #pragma unroll
   for(int tile=0;tile<Tiles;++tile){
    riley_prefill_shape::mma(dg[tile],a[tile][step],ah[tile][step],aa[tile][step],aah[tile][step],bg[step],bgh[step]);
    riley_prefill_shape::mma(du[tile],a[tile][step],ah[tile][step],aa[tile][step],aah[tile][step],bu[step],buh[step]);
   }
  }
 }
 #pragma unroll
 for(int tile=0;tile<Tiles;++tile)for(int half=0;half<2;++half){int row=tile*16+g+half*8;if(row>=rows)continue;
  for(int j=0;j<2;++j){float gv=__bfloat162float(__float2bfloat16_rn(dg[tile][half*2+j]));float uv=__bfloat162float(__float2bfloat16_rn(du[tile][half*2+j]));
   out[row*1536+column*8+2*t+j]=__float2bfloat16_rn((gv/(1.F+expf(-gv)))*uv);
  }
 }
}
template<int Steps,bool Adaptive>
__global__ void gate_up(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* active){
 uint32_t rows=*active;if(!rows||rows>32)return;
 if constexpr(Adaptive){if(rows<=16){compute<Steps,1>(x,gate,up,out,rows);return;}}
 compute<Steps,2>(x,gate,up,out,rows);
}
}

namespace riley_gate_v56 {
template<int Steps>
__global__ void split_rows(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* active){
 uint32_t rows=*active;int warp=threadIdx.x/32;if(!rows||rows>32||warp*16>=rows)return;
 compute<Steps,1>(x+warp*16*576,gate,up,out+warp*16*1536,min(rows-uint32_t(warp*16),16u));
}
__global__ void hybrid(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,__nv_bfloat16* out,const uint32_t* active){
 uint32_t rows=*active;if(!rows||rows>32)return;
 if(rows<=16){if(threadIdx.x<32)compute<2,1>(x,gate,up,out,rows);return;}
 int warp=threadIdx.x/32;__shared__ __nv_bfloat16 partials[2][256];
 shared32_projection_compute<1536,576,0,true,true>(x,warp?up:gate,nullptr,partials[warp],active,blockIdx.x,0);
 __syncthreads();
 for(int i=threadIdx.x;i<256;i+=blockDim.x){int row=i/8,col=i%8;if(row<rows){float g=__bfloat162float(partials[0][i]);out[row*1536+blockIdx.x*8+col]=__float2bfloat16_rn((g/(1.F+expf(-g)))*__bfloat162float(partials[1][i]));}}
}
}
