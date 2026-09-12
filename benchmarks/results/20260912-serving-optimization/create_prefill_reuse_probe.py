from pathlib import Path
s=Path('/tmp/riley-multisequence-integration-20260912/kernels/src/graph_numerics_precise.cu').read_text();a=s.index('template<int N,int K,int Interval,int Warps>\n__global__ void gemm_prefill_m16');b=s.index('\nnamespace riley_cuda_internal {',a)
base=s[a:b]
head='''#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <algorithm>
__device__ uint32_t pack(__nv_bfloat16 a,__nv_bfloat16 b){return uint32_t(__bfloat16_as_ushort(a))|(uint32_t(__bfloat16_as_ushort(b))<<16);}
'''
new='''template<int N,int K,int Interval,int Warps,int Tiles,int Unroll>
__global__ void reuse(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,const uint32_t* position){
 if(*position!=127)return;
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,g=lane/4,t=lane%4;
 const int base=(blockIdx.x*Warps+warp)*8;
 float d[Tiles][4]={},total[Tiles][4]={};
 #pragma unroll Unroll
 for(int depth=0;depth<K;depth+=16){
  uint32_t b=*(const uint32_t*)(w+(base+g)*K+depth+2*t),bb=*(const uint32_t*)(w+(base+g)*K+depth+2*t+8);
  #pragma unroll
  for(int tile=0;tile<Tiles;++tile){
   int row=(blockIdx.y*Tiles+tile)*16+g,next=row+8;
   uint32_t a0=*(const uint32_t*)(x+row*K+depth+2*t),a1=*(const uint32_t*)(x+next*K+depth+2*t);
   uint32_t a2=*(const uint32_t*)(x+row*K+depth+2*t+8),a3=*(const uint32_t*)(x+next*K+depth+2*t+8);
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[tile][0]),"+f"(d[tile][1]),"+f"(d[tile][2]),"+f"(d[tile][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b),"r"(bb));
   if constexpr(Interval>0)if((depth+16)%Interval==0||depth+16==K){
    #pragma unroll
    for(int j=0;j<4;++j){total[tile][j]+=__bfloat162float(__float2bfloat16_rn(d[tile][j]));d[tile][j]=0.;}
   }
  }
 }
 #pragma unroll
 for(int tile=0;tile<Tiles;++tile){
  int row=(blockIdx.y*Tiles+tile)*16+g,next=row+8;
  if constexpr(Interval>0)for(int j=0;j<4;++j)d[tile][j]=total[tile][j];
  y[row*N+base+2*t]=__float2bfloat16_rn(d[tile][0]);y[row*N+base+2*t+1]=__float2bfloat16_rn(d[tile][1]);
  y[next*N+base+2*t]=__float2bfloat16_rn(d[tile][2]);y[next*N+base+2*t+1]=__float2bfloat16_rn(d[tile][3]);
 }
}
'''
tail='''void check(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"%s\\n",cudaGetErrorString(s));exit(2);}}
template<int N,int K,int I,int W,int T,int U> void run(){
 std::vector<uint16_t> x(128*K),w(N*K),a(128*N),b(128*N);uint32_t state=12345;
 for(auto* v:{&x,&w})for(auto& q:*v){state=1664525*state+1013904223;q=__bfloat16_as_ushort(__float2bfloat16_rn((int(state%4097)-2048)/1024.f));}
 __nv_bfloat16 *dx,*dw,*da,*db;uint32_t* dp;check(cudaMalloc(&dx,x.size()*2));check(cudaMalloc(&dw,w.size()*2));check(cudaMalloc(&da,a.size()*2));check(cudaMalloc(&db,b.size()*2));check(cudaMalloc(&dp,4));uint32_t pos=127;
 check(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dp,&pos,4,cudaMemcpyHostToDevice));
 auto old=[&]{gemm_prefill_m16<N,K,I,W><<<dim3(N/(8*W),8),32*W>>>(dx,dw,da,dp);};
 auto cand=[&]{reuse<N,K,I,W,T,U><<<dim3(N/(8*W),8/T),32*W>>>(dx,dw,db,dp);};
 old();cand();check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
 size_t diff=0;for(size_t i=0;i<a.size();++i)diff+=a[i]!=b[i];if(diff){fprintf(stderr,"mismatch %d %d %d %d %zu\\n",N,K,T,U,diff);exit(3);}
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));float times[4];
 for(int round=0;round<4;++round){bool candidate=round==1||round==2;for(int i=0;i<20;++i){if(candidate)cand();else old();}
 check(cudaEventRecord(start));for(int i=0;i<200;++i){if(candidate)cand();else old();}check(cudaEventRecord(end));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&times[round],start,end));}
 printf("{\\"n\\":%d,\\"k\\":%d,\\"interval\\":%d,\\"tiles\\":%d,\\"unroll\\":%d,\\"exact_elements\\":%zu,\\"old_us\\":%.4f,\\"new_us\\":%.4f}\\n",N,K,I,T,U,a.size(),(times[0]+times[3])*2.5,(times[1]+times[2])*2.5);
 check(cudaEventDestroy(start));check(cudaEventDestroy(end));for(void* p:{(void*)dx,(void*)dw,(void*)da,(void*)db,(void*)dp})check(cudaFree(p));
}
template<int T,int U>void all(){run<576,576,192,2,T,U>();run<192,576,192,1,T,U>();run<576,576,128,2,T,U>();run<1536,576,0,4,T,U>();run<576,1536,320,2,T,U>();}
int main(){all<1,1>();all<2,1>();all<4,1>();all<2,4>();all<4,4>();}
'''
Path('/tmp/prefill_reuse_probe.cu').write_text(head+base+new+tail)
