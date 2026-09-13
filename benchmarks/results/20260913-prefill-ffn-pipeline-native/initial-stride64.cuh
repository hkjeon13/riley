#pragma once
#include "../src/prefill_fused_gate_v51.cuh"

// Experimental M16 prefill FFN: preserve K16 MMA and BF16 recurrence.
namespace riley_prefill_ffn_pipeline {
template<int Warps, bool Gate> struct alignas(16) Stage {
  __nv_bfloat16 x[16*64];
  __nv_bfloat16 weights[Warps*512*(Gate?2:1)];
};
static_assert(sizeof(Stage<4,true>)*2==20480);
static_assert(sizeof(Stage<2,false>)*2==8192);
__device__ __forceinline__ void copy16(void* dst,const void* src,bool valid) {
  unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(src),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void wait(){asm volatile("cp.async.wait_group 0;" ::: "memory");}
__device__ __forceinline__ unsigned pair(const __nv_bfloat16* p){return *reinterpret_cast<const unsigned*>(p);}
template<int K,int Warps,bool Gate>
__device__ __forceinline__ void load(Stage<Warps,Gate>& stage,const __nv_bfloat16* x,
 const __nv_bfloat16* weights,const __nv_bfloat16* up,unsigned rows,int depth) {
  for(int v=threadIdx.x;v<128;v+=blockDim.x){
    int row=blockIdx.y*16+v/8,col=(v%8)*8;
    bool valid=row<rows;
    copy16(stage.x+v*8,valid?x+row*K+depth+col:x,valid);
  }
  for(int v=threadIdx.x;v<Warps*64*(Gate?2:1);v+=blockDim.x){
    int matrix=v/(Warps*64),warp=(v/64)%Warps,offset=(v%64)*8;
    const auto* base=matrix?up:weights;
    const auto* src=base+(blockIdx.x*Warps+warp)*(K/16)*128+(depth/16)*128+offset;
    copy16(stage.weights+v*8,src,true);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__global__ void gate_up(const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* up,
 __nv_bfloat16* out,unsigned capacity,const unsigned* live_rows){
  unsigned rows=*live_rows;if(!rows||rows>capacity||blockIdx.y*16>=rows)return;
  __shared__ Stage<4,true> stages[2];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
  int first=blockIdx.y*16+g,base=(blockIdx.x*4+warp)*8;
  float gd[4]={},ud[4]={};
  load<576,4,true>(stages[0],x,gate,up,rows,0);wait();__syncthreads();
  for(int depth=0,index=0;depth<576;depth+=64,index^=1){
    if(depth+64<576)load<576,4,true>(stages[index^1],x,gate,up,rows,depth+64);
    const auto& s=stages[index];
    #pragma unroll
    for(int step=0;step<4;++step){
      const auto* a=s.x+g*64+step*16+2*t;
      const auto* b=s.weights+warp*512+step*128+lane*2;
      unsigned a0=pair(a),a1=pair(a+8*64),a2=pair(a+8),a3=pair(a+8*64+8);
      riley_prefill51::mma(gd,a0,a1,a2,a3,pair(b),pair(b+64));
      riley_prefill51::mma(ud,a0,a1,a2,a3,pair(b+2048),pair(b+2112));
    }
    wait();__syncthreads();
  }
  for(int j=0;j<4;++j){int row=first+(j>=2?8:0),col=base+2*t+j%2;
    if(row<rows){float gv=__bfloat162float(__float2bfloat16_rn(gd[j])),uv=__bfloat162float(__float2bfloat16_rn(ud[j]));
      out[row*1536+col]=__float2bfloat16_rn((gv/(1.F+expf(-gv)))*uv);}
  }
}
__global__ void down(const __nv_bfloat16* x,const __nv_bfloat16* weight,__nv_bfloat16* out,
 unsigned capacity,const unsigned* live_rows){
  unsigned rows=*live_rows;if(!rows||rows>capacity||blockIdx.y*16>=rows)return;
  __shared__ Stage<2,false> stages[2];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
  int first=blockIdx.y*16+g,base=(blockIdx.x*2+warp)*8;
  float d[4]={},total[4]={};
  load<1536,2,false>(stages[0],x,weight,weight,rows,0);wait();__syncthreads();
  for(int depth=0,index=0;depth<1536;depth+=64,index^=1){
    if(depth+64<1536)load<1536,2,false>(stages[index^1],x,weight,weight,rows,depth+64);
    const auto& s=stages[index];
    #pragma unroll
    for(int step=0;step<4;++step){
      const auto* a=s.x+g*64+step*16+2*t;
      const auto* b=s.weights+warp*512+step*128+lane*2;
      riley_prefill51::mma(d,pair(a),pair(a+8*64),pair(a+8),pair(a+8*64+8),pair(b),pair(b+64));
      if((depth+step*16+16)%320==0||depth+step*16+16==1536)
        for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.F;}
    }
    wait();__syncthreads();
  }
  for(int j=0;j<4;++j){int row=first+(j>=2?8:0),col=base+2*t+j%2;
    if(row<rows)out[row*576+col]=__float2bfloat16_rn(total[j]);}
}
}
