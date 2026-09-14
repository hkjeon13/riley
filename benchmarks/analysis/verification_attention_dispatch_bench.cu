#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../kernels/src/mixed_attention_v49.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 constexpr unsigned cap=1024,Q=cap*576,KV=4096*192,M=32+32*416+2048;
 auto*q=alloc<__nv_bfloat16>(Q),*k=alloc<__nv_bfloat16>(KV),*v=alloc<__nv_bfloat16>(KV),*out=alloc<__nv_bfloat16>(Q);auto*m=alloc<unsigned>(M);
 std::vector<__nv_bfloat16> data(KV);unsigned long long seed=9140502;
 auto fill=[&](auto*p,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;data[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/8192.F);}CK(cudaMemcpy(p,data.data(),n*2,cudaMemcpyHostToDevice));};fill(q,Q);fill(k,KV);fill(v,KV);
 std::vector<unsigned> meta(M);cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
 for(unsigned owners:{8u,32u})for(unsigned count:{1u,2u,4u,8u})for(unsigned context:{128u,512u,1024u}){
  std::fill(meta.begin(),meta.end(),0);meta[4]=3;meta[5]=owners;meta[9]=owners*count;meta[24]=owners*count;
  for(unsigned owner=0;owner<owners;++owner){auto*s=meta.data()+32+owner*416;s[1]=context-1;s[2]=count;s[16]=owner*count;s[18]=3;for(unsigned p=0;p<(context+15)/16;++p)s[32+p]=p;for(unsigned local=0;local<count;++local)meta[32+32*416+1024+owner*count+local]=(owner<<16)|local;}
  CK(cudaMemcpy(m,meta.data(),M*4,cudaMemcpyHostToDevice));cudaGraph_t graph[3];cudaGraphExec_t exec[3];
  for(unsigned mode=0;mode<3;++mode){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
   if(mode==2)riley_verification_attention::mapped<<<dim3(32,9),32,0,stream>>>(q,k,v,out,cap,m);
   else riley_mixed_attention::mapped_attention<<<dim3(mode==0?1024:256,9),32,0,stream>>>(q,k,v,out,cap,m);
   CK(cudaStreamEndCapture(stream,&graph[mode]));CK(cudaGraphInstantiate(&exec[mode],graph[mode],nullptr,nullptr,0));
  }
  for(unsigned order=0;order<2;++order)for(unsigned at=0;at<3;++at){unsigned mode=order?2-at:at;
   for(unsigned i=0;i<10;++i)CK(cudaGraphLaunch(exec[mode],stream));CK(cudaEventRecord(start,stream));
   for(unsigned i=0;i<100;++i)CK(cudaGraphLaunch(exec[mode],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));
   std::printf("{\"owners\":%u,\"queries\":%u,\"context\":%u,\"mode\":%u,\"order\":%u,\"mean_us\":%.6f}\n",owners,count,context,mode,order,ms*10);
  }
  for(unsigned mode=0;mode<3;++mode){CK(cudaGraphExecDestroy(exec[mode]));CK(cudaGraphDestroy(graph[mode]));}
 }
 CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));CK(cudaStreamDestroy(stream));for(void*p:{(void*)q,(void*)k,(void*)v,(void*)out,(void*)m})CK(cudaFree(p));
}
