#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
__device__ uint32_t pack(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a))|(uint32_t(__bfloat16_as_ushort(b))<<16);}
template<int N,int K,int Interval,int Warps>
__global__ void gemm_prefill_m16(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,const uint32_t* position){
 static_assert(N%(8*Warps)==0&&K%16==0,"fixed M16 projection geometry");
 if(*position!=127)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int row=blockIdx.y*16+g,next_row=row+8,base=(blockIdx.x*Warps+warp)*8;
 float d[4]={},total[4]={};
 for(int depth=0;depth<K;depth+=16){
  uint32_t a0=pack(x[row*K+depth+2*t],x[row*K+depth+2*t+1]);
  uint32_t a1=pack(x[next_row*K+depth+2*t],x[next_row*K+depth+2*t+1]);
  uint32_t a2=pack(x[row*K+depth+2*t+8],x[row*K+depth+2*t+9]);
  uint32_t a3=pack(x[next_row*K+depth+2*t+8],x[next_row*K+depth+2*t+9]);
  uint32_t b=pack(w[(base+g)*K+depth+2*t],w[(base+g)*K+depth+2*t+1]),bb=pack(w[(base+g)*K+depth+2*t+8],w[(base+g)*K+depth+2*t+9]);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
  if constexpr(Interval>0){
   if((depth+16)%Interval==0||depth+16==K){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
  }
 }
 if constexpr(Interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 y[row*N+base+2*t]=__float2bfloat16_rn(d[0]);
 y[row*N+base+2*t+1]=__float2bfloat16_rn(d[1]);
 y[next_row*N+base+2*t]=__float2bfloat16_rn(d[2]);
 y[next_row*N+base+2*t+1]=__float2bfloat16_rn(d[3]);
}

// CUDA allocations and fixed even BF16 offsets guarantee four-byte alignment.
// Packed loads replace paired scalar loads; no floating-point operations change.
template<int N,int K,int Interval,int Warps>
__global__ void gemm_prefill_m16_vector(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,const uint32_t* position){
 static_assert(N%(8*Warps)==0&&K%16==0,"fixed M16 projection geometry");
 if(*position!=127)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int row=blockIdx.y*16+g,next_row=row+8,base=(blockIdx.x*Warps+warp)*8;
 float d[4]={},total[4]={};
 // Keep the depth recurrence compact; MMA and BF16 chunk-round order are unchanged.
 #pragma unroll 1
 for(int depth=0;depth<K;depth+=16){
  uint32_t a0=*reinterpret_cast<const uint32_t*>(x+(row*K+depth+2*t));
  uint32_t a1=*reinterpret_cast<const uint32_t*>(x+(next_row*K+depth+2*t));
  uint32_t a2=*reinterpret_cast<const uint32_t*>(x+(row*K+depth+2*t+8));
  uint32_t a3=*reinterpret_cast<const uint32_t*>(x+(next_row*K+depth+2*t+8));
  uint32_t b=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t)),bb=*reinterpret_cast<const uint32_t*>(w+((base+g)*K+depth+2*t+8));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
  if constexpr(Interval>0){
   if((depth+16)%Interval==0||depth+16==K){for(int j=0;j<4;++j){total[j]+=__bfloat162float(__float2bfloat16_rn(d[j]));d[j]=0.;}}
  }
 }
 if constexpr(Interval>0)for(int j=0;j<4;++j)d[j]=total[j];
 y[row*N+base+2*t]=__float2bfloat16_rn(d[0]);
 y[row*N+base+2*t+1]=__float2bfloat16_rn(d[1]);
 y[next_row*N+base+2*t]=__float2bfloat16_rn(d[2]);
 y[next_row*N+base+2*t+1]=__float2bfloat16_rn(d[3]);
}

#include "prefill_shape_projection.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<int N,int K,int I,int W>void run(int rows,int pattern){
 int padded=((rows+127)/128)*128;
 std::vector<unsigned short>x(padded*K),w(N*K),a(padded*N),b(rows*N+64,0x55aa);unsigned int rng=17+pattern;
 for(auto* v:{&x,&w})for(auto& q:*v){rng=rng*1664525u+1013904223u;float f=float(int(rng%4097)-2048)/2048.f;
 if(pattern==1)f=ldexpf(f,int((rng>>16)%17)-8);
 q=__bfloat16_as_ushort(__float2bfloat16_rn(f));}
 __nv_bfloat16 *dx,*dw,*da,*db;uint32_t *dp;
 // Input has exactly live rows: sanitizer can catch accidental padded reads.
 ck(cudaMalloc(&dx,rows*K*2));ck(cudaMalloc(&dw,w.size()*2));ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&db,b.size()*2));ck(cudaMalloc(&dp,4));
 ck(cudaMemcpy(dx,x.data(),rows*K*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));uint32_t pos=127;ck(cudaMemcpy(dp,&pos,4,cudaMemcpyHostToDevice));
 __nv_bfloat16* ref;ck(cudaMalloc(&ref,x.size()*2));ck(cudaMemcpy(ref,x.data(),x.size()*2,cudaMemcpyHostToDevice));
 for(int offset=0;offset<padded;offset+=128)gemm_prefill_m16<N,K,I,W><<<dim3(N/(8*W),8),32*W>>>(ref+offset*K,dw,da+offset*N,dp);
 ck(launch_prefill_shape<N,K,I,W>(0,dx,dw,db,rows));ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
 for(int i=0;i<rows*N;++i)if(a[i]!=b[i]){fprintf(stderr,"mismatch rows%d N%d K%d pattern%d index%d\n",rows,N,K,pattern,i);exit(3);}
 for(int i=rows*N;i<int(b.size());++i)if(b[i]!=0x55aa){fprintf(stderr,"tail overwrite\n");exit(4);}
 if(launch_prefill_shape<N,K,I,W>(0,dx,dw,db,0)!=cudaErrorInvalidValue||launch_prefill_shape<N,K,I,W>(0,dx,dw,db,1025)!=cudaErrorInvalidValue)exit(5);
 for(void* p:{(void*)dx,(void*)dw,(void*)da,(void*)db,(void*)dp,(void*)ref})ck(cudaFree(p));
 printf("{\"rows\":%d,\"n\":%d,\"k\":%d,\"pattern\":%d,\"exact_elements\":%d,\"tail_guard\":true}\n",rows,N,K,pattern,rows*N);
}
int main(){for(int p:{0,1})for(int m:{1,7,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,398,511,512,1023,1024}){
run<576,576,192,2>(m,p);run<192,576,192,1>(m,p);run<576,576,128,2>(m,p);run<1536,576,0,4>(m,p);run<576,1536,320,2>(m,p);
}}
