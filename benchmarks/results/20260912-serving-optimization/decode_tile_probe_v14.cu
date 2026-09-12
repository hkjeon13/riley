#include "prefill_shape_projection.cuh"
#include "decode_tile_probe_v14.cuh"
#include <cstdio>
#include <cstdlib>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
void compare(const __nv_bfloat16*a,const __nv_bfloat16*b,int n){for(int i=0;i<n;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"mismatch %d %g %g\n",i,__bfloat162float(a[i]),__bfloat162float(b[i]));exit(3);}}

template<int N,int K,int I>void run(int matrices){
 auto*x=alloc<__nv_bfloat16>(K);auto*w=alloc<__nv_bfloat16>(size_t(matrices)*N*K);auto*tiled=alloc<__nv_bfloat16>(size_t(matrices)*N*K);auto*a=alloc<__nv_bfloat16>(N);auto*b=alloc<__nv_bfloat16>(N);auto*p=alloc<float>(9*4096);
 for(int i=0;i<K;++i)x[i]=__float2bfloat16_rn(float((i*17+1)%127-63)/128);
 for(int m=0;m<matrices;++m){
  auto*wm=w+size_t(m)*N*K;auto*tm=tiled+size_t(m)*N*K;
  for(int i=0;i<N*K;++i)wm[i]=__float2bfloat16_rn(float((i*13+m*7+1)%251-125)/256);
  for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)for(int z=0;z<2;++z){
   size_t base=((n/8)*(K/16)+k/16)*128;
   tm[base+lane*2+z]=wm[(n+lane/4)*K+k+2*(lane%4)+z];
   tm[base+64+lane*2+z]=wm[(n+lane/4)*K+k+8+2*(lane%4)+z];
  }
  enqueue_decode_projection<N,K,I>(0,x,wm,a,p);enqueue_tile_projection<N,K,I>(0,x,tm,b,p);OK(cudaDeviceSynchronize());compare(a,b,N);
 }
 cudaStream_t stream;OK(cudaStreamCreate(&stream));
 for(int reverse=0;reverse<2;++reverse)for(int lane=0;lane<2;++lane){
  bool packed=(lane^reverse)!=0;cudaGraph_t graph;cudaGraphExec_t exec;
  OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  for(int m=0;m<matrices;++m){if(packed)enqueue_tile_projection<N,K,I>(stream,x,tiled+size_t(m)*N*K,b,p);else enqueue_decode_projection<N,K,I>(stream,x,w+size_t(m)*N*K,b,p);}
  OK(cudaStreamEndCapture(stream,&graph));OK(cudaGraphInstantiate(&exec,graph,0));
  for(int j=0;j<5;++j)OK(cudaGraphLaunch(exec,stream));
  cudaEvent_t start,end;OK(cudaEventCreate(&start));OK(cudaEventCreate(&end));OK(cudaEventRecord(start,stream));
  for(int j=0;j<30;++j)OK(cudaGraphLaunch(exec,stream));
  OK(cudaEventRecord(end,stream));OK(cudaEventSynchronize(end));float ms;OK(cudaEventElapsedTime(&ms,start,end));
  printf("N=%d K=%d I=%d matrices=%d order=%d tiled=%d us=%f exact=true\n",N,K,I,matrices,reverse,packed,ms*1000/(30*matrices));fflush(stdout);
  OK(cudaEventDestroy(start));OK(cudaEventDestroy(end));OK(cudaGraphExecDestroy(exec));OK(cudaGraphDestroy(graph));
 }
 OK(cudaStreamDestroy(stream));for(void*z:{(void*)x,(void*)w,(void*)tiled,(void*)a,(void*)b,(void*)p})OK(cudaFree(z));
}
int main(){for(int n:{1,60}){run<1536,576,0>(n);run<576,1536,320>(n);run<192,576,192>(n);run<576,576,192>(n);run<576,576,128>(n);}}
