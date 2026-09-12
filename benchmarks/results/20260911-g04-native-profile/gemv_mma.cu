#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ uint32_t pack(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a))|(uint32_t(__bfloat16_as_ushort(b))<<16);}
__global__ void gemv(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,int n,int k){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4,base=(blockIdx.x*4+warp)*8;
 if(base>=n)return;float d[4]={};
 for(int depth=0;depth<k;depth+=16){
  uint32_t a=pack(x[depth+2*t],x[depth+2*t+1]),aa=pack(x[depth+2*t+8],x[depth+2*t+9]);
  uint32_t b=pack(w[(base+g)*k+depth+2*t],w[(base+g)*k+depth+2*t+1]),bb=pack(w[(base+g)*k+depth+2*t+8],w[(base+g)*k+depth+2*t+9]);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a),"r"(a),"r"(aa),"r"(aa),"r"(b),"r"(bb));
 }
 if(g==0){y[base+2*t]=__float2bfloat16_rn(d[0]);y[base+2*t+1]=__float2bfloat16_rn(d[1]);}
}
extern "C" void run(cudaStream_t s,const void* x,const void* w,void* y,int n,int k){gemv<<<(n+31)/32,128,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(__nv_bfloat16*)y,n,k);}
