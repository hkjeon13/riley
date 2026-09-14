#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../kernels/src/mixed_attention_v49.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 bool small=argc>1;constexpr unsigned cap=256,Q=cap*576,KV=4096*192,M=32+32*416+2048;
 auto*q=alloc<__nv_bfloat16>(Q),*k=alloc<__nv_bfloat16>(KV),*v=alloc<__nv_bfloat16>(KV),*a=alloc<__nv_bfloat16>(Q),*b=alloc<__nv_bfloat16>(Q);auto*m=alloc<unsigned>(M);
 std::vector<__nv_bfloat16> data(KV);unsigned long long seed=9140501;
 auto fill=[&](auto*p,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;data[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/8192.F);}CK(cudaMemcpy(p,data.data(),n*2,cudaMemcpyHostToDevice));};
 fill(q,Q);fill(k,KV);fill(v,KV);
 std::vector<unsigned> meta(M);std::vector<unsigned short> ha(Q),hb(Q);cudaStream_t stream;CK(cudaStreamCreate(&stream));unsigned cases=0;
 for(unsigned owners:small?std::vector<unsigned>{2}:std::vector<unsigned>{1,8,32})for(unsigned count:small?std::vector<unsigned>{1,3,8}:std::vector<unsigned>{1,2,3,4,5,6,7,8})for(unsigned context:small?std::vector<unsigned>{129}:std::vector<unsigned>{8,16,31,32,127,128,129,512,1024,4096}){
  std::fill(meta.begin(),meta.end(),0);meta[4]=3;meta[5]=owners;meta[9]=owners*count;meta[24]=owners*count;
  for(unsigned owner=0;owner<owners;++owner){auto*s=meta.data()+32+owner*416;s[1]=context-1;s[2]=count;s[16]=owner*count;s[18]=3;for(unsigned p=0;p<(context+15)/16;++p)s[32+p]=p;for(unsigned local=0;local<count;++local)meta[32+32*416+1024+owner*count+local]=(owner<<16)|local;}
  CK(cudaMemcpy(m,meta.data(),M*4,cudaMemcpyHostToDevice));
  for(unsigned exceptional=0;exceptional<2;++exceptional){
   unsigned short saved;CK(cudaMemcpy(&saved,v,2,cudaMemcpyDeviceToHost));if(exceptional){unsigned short nan=0x7fc1;CK(cudaMemcpy(v,&nan,2,cudaMemcpyHostToDevice));}
   CK(cudaMemset(a,0x55,Q*2));CK(cudaMemset(b,0x55,Q*2));
   riley_mixed_attention::mapped_attention<<<dim3(cap,9),32,0,stream>>>(q,k,v,a,cap,m);
   cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
   riley_verification_attention::mapped<<<dim3(32,9),32,0,stream>>>(q,k,v,b,cap,m);
   CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
   for(unsigned replay=0;replay<2;++replay){CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(ha.data(),a,Q*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hb.data(),b,Q*2,cudaMemcpyDeviceToHost));size_t diff=0;for(unsigned i=0;i<Q;++i)diff+=ha[i]!=hb[i];if(diff){std::printf("FAIL owners=%u count=%u context=%u exceptional=%u diff=%zu\n",owners,count,context,exceptional,diff);return 1;}}
   CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaMemcpy(v,&saved,2,cudaMemcpyHostToDevice));++cases;
  }
 }
 CK(cudaStreamDestroy(stream));for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)m})CK(cudaFree(p));std::printf("PASS cases=%u replays_each=2 full_buffer_exact=true nonfinite_v=true\n",cases);
}
