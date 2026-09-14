#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../kernels/optional/prefill_projection_pipeline.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%d %s: %s\n",__LINE__,#call,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<int N,int Interval,int Warps>void run(bool bounded){
 constexpr int K=576,C=1024,L=30;size_t W=N*K,O=C*N,X=C*K;
 auto*x=alloc<__nv_bfloat16>(X);auto*w=alloc<__nv_bfloat16>(W*L);auto*p=alloc<__nv_bfloat16>(W*L);
 auto*a=alloc<__nv_bfloat16>(O);auto*b=alloc<__nv_bfloat16>(O);auto*live=alloc<unsigned>(1);
 unsigned long long seed=9132201;std::vector<__nv_bfloat16> data(W*L);
 auto fill=[&](auto* dest,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;data[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/131072.F);}CHECK(cudaMemcpy(dest,data.data(),n*2,cudaMemcpyHostToDevice));};
 fill(w,W*L);fill(x,X);cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
 for(int l=0;l<L;++l)riley_prefill_projection::pack<N,K><<<(W+255)/256,256,0,stream>>>(w+l*W,p+l*W);
 CHECK(cudaStreamSynchronize(stream));
 // Independently check packed layout, including every layer.
 std::vector<unsigned short> packed(W*L);CHECK(cudaMemcpy(packed.data(),p,W*L*2,cudaMemcpyDeviceToHost));
 CHECK(cudaMemcpy(data.data(),w,W*L*2,cudaMemcpyDeviceToHost));
 for(int l=0;l<L;++l)for(int n=0;n<N;++n)for(int k=0;k<K;++k){size_t i=l*W+((n/8)*(K/16)+k/16)*128+(k%16/8)*64+(n%8)*8+k%8;
  if(packed[i]!=reinterpret_cast<unsigned short*>(data.data())[l*W+n*K+k])std::exit(3);}
 cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
 for(int lane=0;lane<2;++lane){CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  for(int l=0;l<L;++l)if(lane)riley_prefill_projection::project<N,K,Interval,Warps><<<dim3(N/(8*Warps),C/16),32*Warps,0,stream>>>(x,p+l*W,b,C,live);
  else gemm_prefill_shape_vector<N,K,Interval,Warps><<<dim3(N/(8*Warps),C/16),32*Warps,0,stream>>>(x,w+l*W,a,C,live);
  CHECK(cudaStreamEndCapture(stream,&graphs[lane]));CHECK(cudaGraphInstantiate(&execs[lane],graphs[lane],nullptr,nullptr,0));}
 std::vector<unsigned short> ha(O),hb(O);
 auto cases=bounded?std::vector<unsigned>{0,17,398,1025}:std::vector<unsigned>{0,1,8,16,17,31,32,64,128,398,512,1024,1025};
 for(int repeat=0;repeat<(bounded?1:2);++repeat)for(unsigned rows:cases){fill(x,X);CHECK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(a,0x55,O*2));CHECK(cudaMemset(b,0x55,O*2));
  // Audit all layer outputs individually, not just the final overwritten graph output.
  for(int l=0;l<L;++l){
   gemm_prefill_shape_vector<N,K,Interval,Warps><<<dim3(N/(8*Warps),C/16),32*Warps,0,stream>>>(x,w+l*W,a,C,live);
   riley_prefill_projection::project<N,K,Interval,Warps><<<dim3(N/(8*Warps),C/16),32*Warps,0,stream>>>(x,p+l*W,b,C,live);
   CHECK(cudaStreamSynchronize(stream));CHECK(cudaMemcpy(ha.data(),a,O*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,O*2,cudaMemcpyDeviceToHost));
   for(size_t i=0;i<O;++i)if(ha[i]!=hb[i]||((rows>C||i>=rows*N)&&(ha[i]!=0x5555||hb[i]!=0x5555))){std::fprintf(stderr,"mismatch N%d I%d rows%u layer%d index%zu\n",N,Interval,rows,l,i);std::exit(1);}
  }
  for(auto executable:execs)CHECK(cudaGraphLaunch(executable,stream));CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(ha.data(),a,O*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,O*2,cudaMemcpyDeviceToHost));if(ha!=hb)std::exit(4);
  std::printf("{\"N\":%d,\"interval\":%d,\"rows\":%u,\"repeat\":%d,\"layers_exact\":30,\"graph_exact\":true}\n",N,Interval,rows,repeat);std::fflush(stdout);
 }
 if(!bounded){cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
  for(unsigned rows:{32u,128u,398u,512u}){CHECK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));
   for(int order=0;order<2;++order)for(int index=0;index<2;++index){int lane=order?1-index:index;
    for(int n=0;n<5;++n)CHECK(cudaGraphLaunch(execs[lane],stream));CHECK(cudaEventRecord(start,stream));
    for(int n=0;n<50;++n)CHECK(cudaGraphLaunch(execs[lane],stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));
    std::printf("{\"N\":%d,\"interval\":%d,\"timing_rows\":%u,\"order\":%d,\"candidate\":%d,\"layer_us\":%.6f}\n",N,Interval,rows,order,lane,ms*1000.F/(50*L));}}
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));}
 cudaFuncAttributes attrs;CHECK(cudaFuncGetAttributes(&attrs,riley_prefill_projection::project<N,K,Interval,Warps>));std::printf("{\"N\":%d,\"interval\":%d,\"shared\":%zu,\"registers\":%d,\"local\":%zu}\n",N,Interval,attrs.sharedSizeBytes,attrs.numRegs,attrs.localSizeBytes);
 for(int i=0;i<2;++i){CHECK(cudaGraphExecDestroy(execs[i]));CHECK(cudaGraphDestroy(graphs[i]));}CHECK(cudaStreamDestroy(stream));for(void*q:{(void*)x,(void*)w,(void*)p,(void*)a,(void*)b,(void*)live})CHECK(cudaFree(q));
}
int main(int argc,char**){run<576,192,2>(argc>1);run<192,192,1>(argc>1);run<576,128,2>(argc>1);}
