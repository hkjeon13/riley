#include "prefill_projection_v51.cuh"
#include "prefill_shape_pointwise.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define OK(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(err));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
void pack(const __nv_bfloat16* w,__nv_bfloat16* tw,int N,int K){for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z)tw[((n/8)*(K/16)+k/16)*128+hi*64+lane*2+z]=w[(n+lane/4)*K+k+hi*8+2*(lane%4)+z];}
unsigned long long hash(const void* p,size_t n){auto*b=(const unsigned char*)p;unsigned long long h=1469598103934665603ull;for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ull;return h;}
template<int N,int K,int I,int W,bool Tiled,bool Fused=false>void check(bool timing){
 auto*x=alloc<__nv_bfloat16>(1024*K);auto*w=alloc<__nv_bfloat16>(N*K);auto*tw=alloc<__nv_bfloat16>(N*K);auto*u=alloc<__nv_bfloat16>(N*K);auto*tu=alloc<__nv_bfloat16>(N*K);
 auto*a=alloc<__nv_bfloat16>(1024*N+16);auto*b=alloc<__nv_bfloat16>(1024*N+16);auto*gout=alloc<__nv_bfloat16>(1024*N);auto*uout=alloc<__nv_bfloat16>(1024*N);auto*meta=alloc<uint32_t>(3);
 cudaStream_t stream;OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));int cases=0;int capacity=1024;
 auto launch=[&](int variant,__nv_bfloat16* out){
  if(variant==0){
   if constexpr(Fused){gemm_prefill_shape_vector<N,K,I,W,true><<<dim3(N/(8*W),(capacity+15)/16),32*W,0,stream>>>(x,tw,gout,capacity,meta+2);gemm_prefill_shape_vector<N,K,I,W,true><<<dim3(N/(8*W),(capacity+15)/16),32*W,0,stream>>>(x,tu,uout,capacity,meta+2);riley_prefill_pointwise::swiglu_rows<<<dim3(6,capacity),256,0,stream>>>(gout,uout,out,meta,capacity);}
   else gemm_prefill_shape_vector<N,K,I,W,Tiled><<<dim3(N/(8*W),(capacity+15)/16),32*W,0,stream>>>(x,Tiled?tw:w,out,capacity,meta+2);
  }else if constexpr(Fused){
   if(variant==1)riley_prefill51::gate_up<W,1><<<dim3(N/(8*W),(capacity+15)/16),32*W,0,stream>>>(x,tw,tu,out,capacity,meta+2);
   if(variant==2)riley_prefill51::gate_up<W,2><<<dim3(N/(8*W),(capacity+31)/32),32*W,0,stream>>>(x,tw,tu,out,capacity,meta+2);
   if(variant==3)riley_prefill51::gate_up<W,4><<<dim3(N/(8*W),(capacity+63)/64),32*W,0,stream>>>(x,tw,tu,out,capacity,meta+2);
  }else{
   if(variant==1)riley_prefill51::projection<N,K,I,W,Tiled,1><<<dim3(N/(8*W),(capacity+15)/16),32*W,0,stream>>>(x,Tiled?tw:w,out,capacity,meta+2);
   if(variant==2)riley_prefill51::projection<N,K,I,W,Tiled,2><<<dim3(N/(8*W),(capacity+31)/32),32*W,0,stream>>>(x,Tiled?tw:w,out,capacity,meta+2);
   if(variant==3)riley_prefill51::projection<N,K,I,W,Tiled,4><<<dim3(N/(8*W),(capacity+63)/64),32*W,0,stream>>>(x,Tiled?tw:w,out,capacity,meta+2);
  }
 };
 for(int seed:{1,17,99}){
  if(timing&&seed!=1)continue;
  for(int i=0;i<1024*K;++i)x[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
  for(int i=0;i<N*K;++i){w[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/256);u[i]=__float2bfloat16_rn(float((i*19+seed)%127-63)/128);}pack(w,tw,N,K);pack(u,tu,N,K);
  auto hx=hash(x,1024*K*2),hw=hash(w,N*K*2),htw=hash(tw,N*K*2),hu=hash(tu,N*K*2);
  for(int cap:{512,1024})for(int rows:{0,1,7,15,16,17,31,32,33,73,128,256,398,512,1024,1025}){
   if(timing&&rows!=16&&rows!=32&&rows!=128&&rows!=256&&rows!=398&&rows!=512&&rows!=1024)continue;
   if(timing&&cap!=(rows<=512?512:1024))continue;capacity=cap;
   meta[0]=meta[1]=0;meta[2]=rows;
   for(int i=0;i<1024*N+16;++i)a[i]=__ushort_as_bfloat16(0x4123);launch(0,a);OK(cudaStreamSynchronize(stream));
   for(int variant=1;variant<=3;++variant){for(int i=0;i<1024*N+16;++i)b[i]=__ushort_as_bfloat16(0x4123);launch(variant,b);OK(cudaStreamSynchronize(stream));
    if(memcmp(a,b,(1024*N+16)*2)){for(int i=0;i<1024*N+16;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"mismatch N%d K%d I%d fused%d seed%d rows%d variant%d i%d expected%04x got%04x\n",N,K,I,Fused,seed,rows,variant,i,__bfloat16_as_ushort(a[i]),__bfloat16_as_ushort(b[i]));exit(3);}}
    ++cases;
   }
   if(timing){cudaGraphExec_t execs[4];for(int v=0;v<4;++v){cudaGraph_t graph;OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(v,b);OK(cudaStreamEndCapture(stream,&graph));OK(cudaGraphInstantiate(&execs[v],graph,0,0,0));OK(cudaGraphDestroy(graph));}
    cudaEvent_t start,end;OK(cudaEventCreate(&start));OK(cudaEventCreate(&end));
    for(int pair=0;pair<4;++pair)for(int order=0;order<4;++order){int v=pair%2?3-order:order;for(int i=0;i<10;++i)OK(cudaGraphLaunch(execs[v],stream));OK(cudaStreamSynchronize(stream));OK(cudaEventRecord(start,stream));for(int i=0;i<100;++i)OK(cudaGraphLaunch(execs[v],stream));OK(cudaEventRecord(end,stream));OK(cudaEventSynchronize(end));float ms;OK(cudaEventElapsedTime(&ms,start,end));printf("{\"N\":%d,\"K\":%d,\"I\":%d,\"fused\":%d,\"capacity\":%d,\"rows\":%d,\"variant\":%d,\"pair\":%d,\"us\":%.6f}\n",N,K,I,Fused,capacity,rows,v,pair,ms*10);}
    for(auto e:execs)OK(cudaGraphExecDestroy(e));OK(cudaEventDestroy(start));OK(cudaEventDestroy(end));
   }
  }
  if(hx!=hash(x,1024*K*2)||hw!=hash(w,N*K*2)||htw!=hash(tw,N*K*2)||hu!=hash(tu,N*K*2)){fprintf(stderr,"input changed\n");exit(4);}
 }
 fprintf(stderr,"PASS N%d K%d I%d fused%d cases%d exact=true inputs_unchanged=true\n",N,K,I,Fused,cases);
 for(void*p:{(void*)x,(void*)w,(void*)tw,(void*)u,(void*)tu,(void*)a,(void*)b,(void*)gout,(void*)uout,(void*)meta})OK(cudaFree(p));OK(cudaStreamDestroy(stream));
}
int main(int argc,char**){bool timing=argc>1;check<576,576,192,2,false>(timing);check<192,576,192,1,false>(timing);check<576,576,128,2,false>(timing);check<1536,576,0,4,true>(timing);check<576,1536,320,2,true>(timing);check<1536,576,0,4,true,true>(timing);}
