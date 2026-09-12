// Internal primitive for a future variable-prefill owner. Not wired into serving.
// Rows are validated by the launcher; inactive tail rows do not read or write memory.
// Each active row preserves the fixed kernel MMA and intermediate BF16 recurrence.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
template<int N,int K,int Interval,int Warps>
__global__ void gemm_prefill_shape_vector(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,uint32_t rows,const uint32_t* live_rows=nullptr){
 if(live_rows){uint32_t live=*live_rows;if(!live||live>rows)return;rows=live;}
 static_assert(N%(8*Warps)==0&&K%16==0,"fixed M16 projection geometry");

 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int row=blockIdx.y*16+g,next_row=row+8,base=(blockIdx.x*Warps+warp)*8;
 float d[4]={},total[4]={};
 // Keep the depth recurrence compact; MMA and BF16 chunk-round order are unchanged.
 #pragma unroll 1
 for(int depth=0;depth<K;depth+=16){
  uint32_t a0=row<rows?*reinterpret_cast<const uint32_t*>(x+(row*K+depth+2*t)):0;
  uint32_t a1=next_row<rows?*reinterpret_cast<const uint32_t*>(x+(next_row*K+depth+2*t)):0;
  uint32_t a2=row<rows?*reinterpret_cast<const uint32_t*>(x+(row*K+depth+2*t+8)):0;
  uint32_t a3=next_row<rows?*reinterpret_cast<const uint32_t*>(x+(next_row*K+depth+2*t+8)):0;
  uint32_t b=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t)),bb=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t+8));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
  if constexpr(Interval>0){
   if((depth+16)%Interval==0||depth+16==K){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
  }
 }
 if constexpr(Interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 if(row<rows)y[row*N+base+2*t]=__float2bfloat16_rn(d[0]);
 if(row<rows)y[row*N+base+2*t+1]=__float2bfloat16_rn(d[1]);
 if(next_row<rows)y[next_row*N+base+2*t]=__float2bfloat16_rn(d[2]);
 if(next_row<rows)y[next_row*N+base+2*t+1]=__float2bfloat16_rn(d[3]);
}

template<int N,int K,int Interval,int Warps>
cudaError_t launch_prefill_shape(cudaStream_t stream,const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,uint32_t rows){
 if(rows==0||rows>1024||x==nullptr||w==nullptr||y==nullptr)return cudaErrorInvalidValue;
 gemm_prefill_shape_vector<N,K,Interval,Warps><<<dim3(N/(8*Warps),(rows+15)/16),32*Warps,0,stream>>>(x,w,y,rows);
 return cudaGetLastError();
}
